"""
Battery Charge/Discharge Planning System for Growatt WIT Inverter

Uses Nord Pool price forecasts to schedule optimal battery charge/hold/discharge
periods. Implements adaptive re-optimization based on actual SOC and PV production.

Author: AppDaemon Battery Optimizer
"""

import appdaemon.plugins.hass.hassapi as hass
import datetime
import functools
import math
import time
from typing import Dict, List, Optional, Tuple

# Import from the battery_optimizer_lib package
from battery_optimizer_lib import (
    # Config
    BatteryOptimizerConfig,
    # Models
    BatteryMode,
    PricePoint,
    ScheduleEntry,
    BatteryLearningEngine,
    LoadProfile,
    NordPoolPriceService,
    DirectControl,
    # DP Optimizer
    DPOptimizer,
    DPOptimizerConfig,
    # Timezone utilities
    normalize_tz_pair,
    datetimes_match_slot,
    instant_key,
    canonical_slot_key,
    dt_ge,
    ensure_local_tz,
    align_to_slot,
    next_slot_time,
    prev_slot_time,
    slot_offset,
    next_interval_time,
    lookup_by_time,
    # HA helpers
    SensorReader,
    # Cost tracker
    BatteryCostTracker,
    BatteryCostConfig,
    # Charge rate utilities
    compute_charge_rates_per_slot,
    # Shared slot SOC transition model
    SocProjectionParams,
    project_slot_soc,
    # SOC deviation detection
    SocDeviationDetector,
    SocDeviationConfig,
    # Schedule formatting
    ScheduleFormatter,
    ScheduleFormatterConfig,
    # Prediction tracker
    LoadPredictionTracker,
    # PV forecast service
    PvForecastService,
    PvForecastServiceConfig,
    # PV forecast bias
    PvBiasTracker,
    PvBiasConfig,
    # Shared thermal model + ambient temperature
    TemperatureProjector,
    AmbientTemperatureService,
    AmbientServiceConfig,
)
from battery_optimizer_lib.direct_control import ApplyOutcome
from battery_optimizer_lib.models import ScheduleModeCounts, count_schedule_modes
from battery_optimizer_lib.pv_profile import PvProfile
from battery_optimizer_lib.slot_outcome_tracker import SlotOutcomeTracker


def _timed_callback(func):
    """Measure a callback's wall time and warn when it hogs the thread.

    AppDaemon runs an app's callbacks on a shared worker thread and prints
    "Excessive time spent in callback ... (limit=10.0s)" when one overruns —
    without naming what this app was doing. Since `set_wit_mode` is a SYNCHRONOUS
    service call, one slow inverter write stalls every other callback of this
    app. Measuring here names the callback and lets us advise `total_threads`.

    functools.wraps + *args/**kwargs is mandatory: AppDaemon inspects and calls
    these with positional args (`execute_scheduled_mode(kwargs, force=True)`),
    and the signature must survive untouched.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return func(self, *args, **kwargs)
        finally:
            try:
                self._record_callback_duration(
                    func.__name__, time.monotonic() - started
                )
            except Exception:
                pass
    return wrapper



class BatteryOptimizer(hass.Hass):
    """
    AppDaemon app for optimizing battery charge/discharge based on Nord Pool prices.

    Features:
    - Fetches prices from Nord Pool sensor (today + tomorrow after 13:00 CET)
    - Calculates optimal charge/discharge schedule
    - Adapts schedule every slot based on actual SOC
    - Solar-aware adjustments when PV is producing
    - Safety checks for min/max SOC
    - Manual override support
    """

    def initialize(self):
        """Initialize the battery optimizer"""
        self.log("Initializing Battery Optimizer")

        # Load configuration
        self._load_config()

        # Sensor reader for HA state access
        self._sensors = SensorReader(self.get_state, self.log)

        # Internal state
        self.current_mode: BatteryMode = BatteryMode.HOLD
        self.schedule: Dict[datetime.datetime, ScheduleEntry] = {}
        self.last_optimization: Optional[datetime.datetime] = None
        self.expected_soc_schedule: Dict[datetime.datetime, float] = {}
        self.expected_temp_schedule: Dict[datetime.datetime, float] = {}
        # Wall-clock instant the expected SOC trajectory starts from. The first
        # entry of expected_soc_schedule describes THIS instant, not the slot
        # boundary, whenever the trajectory was built mid-slot.
        self._expected_soc_anchor: Optional[datetime.datetime] = None
        self._last_nonzero_load_w: Optional[float] = None
        self._previous_schedule_from_sensor: Optional[Dict[datetime.datetime, BatteryMode]] = None
        # Sliding PV forecast bias multiplier applied to the remaining horizon.
        self._pv_bias_factor: float = 1.0

        # Inverter-control health counters (exposed on a diagnostics sensor).
        self._apply_failure_count: int = 0
        self._consecutive_apply_failures: int = 0
        self._apply_success_count: int = 0
        # A command that timed out client-side was NEVER confirmed: it must not
        # count as a success, and a run of them is exactly the hung-modbus case.
        self._apply_unconfirmed_count: int = 0
        self._consecutive_apply_unconfirmed: int = 0
        # Neutral outcomes — nothing was transmitted, so they say nothing about
        # inverter health either way.
        self._apply_duplicate_count: int = 0
        self._apply_dry_run_count: int = 0
        self._callback_overrun_count: int = 0
        self._slowest_callback: Optional[Tuple[str, float]] = None
        self._threads_hint_logged: bool = False
        self._last_terminal_warning_time: Optional[datetime.datetime] = None

        # Decision context tracking (for transparency logging and sensor exposure)
        self._last_recalc_trigger: str = "startup"  # "startup", "daily_13:15", "soc_deviation", "manual", "battery_depleted"
        self._last_recalc_time: Optional[datetime.datetime] = None
        self._last_depletion_recalc_time: Optional[datetime.datetime] = None
        self._last_soc_deviation: Optional[float] = None  # Deviation that triggered recalculation
        self._last_min_charge_slots: int = 0  # Min charge slots from last calculation
        # Mode census of the last schedule as it will execute (post cloud-safe
        # conversion), not the DP's pre-conversion counts.
        self._last_schedule_counts: ScheduleModeCounts = ScheduleModeCounts()
        self._last_charge_slots: List[Dict] = []  # Selected charge slots with prices
        self._last_projected_costs: Dict[datetime.datetime, float] = {}  # Projected battery cost evolution
        self._last_dp_soc_trajectory: Dict[datetime.datetime, Tuple[float, float]] = {}  # DP's SOC trajectory (start, end) per slot
        self._last_dp_temp_trajectory: Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]] = {}  # DP's temp trajectory

        # Self-learning engine for adaptive optimization
        self.learning_engine = BatteryLearningEngine(
            battery_capacity_kwh=self.config.battery_capacity,
            nominal_charge_rate_kw=self.config.charge_rate,
            nominal_efficiency=self.config.efficiency,
            min_soc=self.config.default_min_soc,
            max_soc=self.config.default_max_soc,
            log_func=self.log,
        )
        self._init_learning_engine()

        # Ambient temperature: weather forecast -> outdoor sensor -> diurnal
        # profile around the learned battery minimum. Never a single constant
        # for the whole horizon.
        self._ambient_service = AmbientTemperatureService(
            config=AmbientServiceConfig.from_main_config(self.config),
            get_state_func=self.get_state,
            call_service_func=self.call_service,
            get_datetime_func=self.datetime,
            get_timezone_func=self._get_local_timezone,
            log_func=self.log,
            min_temp_provider=self._estimated_ambient_min_temp,
        )

        # One thermal model for the DP trajectory, the expected-SOC trajectory,
        # the schedule log and the charge-rate pre-computation.
        self._temp_projector = TemperatureProjector(
            learning_engine=self.learning_engine,
            ambient_provider=self._ambient_service,
            log_func=self.log,
            default_cooling_rate=self.config.thermal_default_cooling_rate_per_min,
            default_heating_c_per_kwh=self.config.thermal_default_heating_c_per_kwh,
        )

        # Load profile for probabilistic scheduling
        self.load_profile = LoadProfile(
            slot_minutes=self.config.slot_minutes,
            default_load_w=self.config.base_consumption,
            max_samples=self.config.load_profile_max_samples,
            min_samples=self.config.load_profile_min_samples,
            log_func=self.log,
        )
        self._init_load_profile()

        # Prediction accuracy tracker
        self.prediction_tracker = LoadPredictionTracker(
            slot_minutes=self.config.slot_minutes,
            log_func=self.log,
        )
        self._init_prediction_tracker()

        # PV production profile for self-consumption planning
        self.pv_profile = PvProfile(
            slot_minutes=self.config.slot_minutes,
            default_pv_w=0.0,
            max_samples=self.config.pv_profile_max_samples,
            min_samples=self.config.pv_profile_min_samples,
            log_func=self.log,
        )
        self._init_pv_profile()

        # PV forecast service (Solcast / Forecast.Solar)
        self._pv_forecast_service = PvForecastService(
            config=PvForecastServiceConfig.from_main_config(self.config),
            get_state_func=self.get_state,
            get_datetime_func=self.datetime,
            get_timezone_func=self._get_local_timezone,
            log_func=self.log,
        )

        # Sliding PV forecast bias tracker (slot-energy sampling + bias factor)
        self._pv_bias = PvBiasTracker(
            config=PvBiasConfig.from_main_config(self.config),
            align_to_slot_func=self._align_to_slot,
            log_func=self.log,
        )

        # Slot outcome tracker for prediction monitoring
        self._outcome_tracker = SlotOutcomeTracker(
            slot_minutes=self.config.slot_minutes,
            log_func=self.log,
        )

        # Direct inverter control via set_wit_mode service
        self._direct_control = DirectControl(self, self.config)

        # Nord Pool price service for fetching electricity prices
        self._price_service = NordPoolPriceService(
            nordpool_config_entry=self.config.nordpool_config_entry,
            nordpool_area=self.config.nordpool_area,
            nordpool_sensor=self.config.nordpool_sensor,
            ha_url=self.config.ha_url,
            ha_token=self.config.ha_token,
            tomorrow_prices_hour=self.config.tomorrow_prices_hour,
            slot_minutes=self.config.slot_minutes,
            get_state_func=self.get_state,
            call_service_func=self.call_service,
            get_datetime_func=self.datetime,
            get_date_func=self.date,
            get_timezone_func=self._get_local_timezone,
            log_func=self.log,
        )

        # Schedule formatter for logging and sensor updates
        self._schedule_formatter = ScheduleFormatter(
            config=ScheduleFormatterConfig(
                slot_minutes=self.config.slot_minutes,
                slot_hours=self.config.slot_hours,
                battery_capacity=self.config.battery_capacity,
                charge_rate=self.config.charge_rate,
                discharge_rate=self.config.discharge_rate,
                export_discharge_rate=self.config.export_discharge_rate,
                efficiency=self.config.efficiency,
                battery_wear_cost=self.config.battery_wear_cost,
                decision_log_level=self.config.decision_log_level,
                inverter_efficiency=self.config.inverter_efficiency,
            ),
            log_func=self.log,
            learning_engine=self.learning_engine,
            temp_projector=self._temp_projector,
        )

        # Battery cost tracker (must be after learning engine and price service)
        self._cost_tracker = BatteryCostTracker(
            config=BatteryCostConfig.from_main_config(self.config),
            get_state_func=self.get_state,
            call_service_func=self.call_service,
            get_datetime_func=self.datetime,
            get_timezone_func=self._get_local_timezone,
            align_to_slot_func=self._align_to_slot,
            get_min_soc_func=lambda: self.min_soc,
            get_max_soc_func=lambda: self.max_soc,
            get_current_soc_func=self._get_current_soc,
            get_battery_temp_func=self._get_battery_temp,
            learning_engine=self.learning_engine,
            get_cached_prices_func=lambda: self._price_service.cached_prices,
            save_learning_data_func=self._save_learning_data,
            update_learning_sensor_func=self._update_learning_sensor,
            log_func=self.log,
            get_ambient_temp_func=lambda: self._ambient_service.predict_c(self.datetime()),
        )
        self._init_battery_cost()

        # Restore previous schedule from sensor (for continuity on restart)
        self._restore_previous_schedule_from_sensor()

        # Full re-optimization after Nord Pool publishes tomorrow's prices
        # Uses configured hour (default 14 for EET = 13 CET) plus 15 minutes buffer
        optimize_hour = self.config.tomorrow_prices_hour
        self.run_daily(self.full_optimize, datetime.time(optimize_hour, 15))

        # Startup optimization is triggered from _init_battery_cost() after battery cost is loaded
        # (either immediately if HA available, or after homeassistant_start event)

        # Adaptive re-evaluation (can be more frequent than schedule slots)
        self.run_every(
            self.adaptive_optimize,
            self._next_interval_time(self.config.adaptive_recalc_minutes),
            self.config.adaptive_recalc_minutes * 60
        )

        # Schedule execution every slot (hourly if slot_minutes=60)
        self.run_every(self.execute_scheduled_mode, self._next_slot_time(), self.config.slot_minutes * 60)

        # Sample PV power frequently so a slot's shortfall is judged on its
        # ENERGY (mean of many samples), not one boundary reading.
        self.run_every(
            self._sample_pv,
            self.datetime() + datetime.timedelta(seconds=10),
            self.config.pv_sample_seconds,
        )

        # Sample battery temperature on a TIMER. Before this, observations only
        # arrived from SOC-change and mode-transition events, so the rolling
        # window could span a few hours instead of the assumed two days and
        # could never contain a diurnal minimum.
        self.run_every(
            self._record_ambient_observation,
            self.datetime() + datetime.timedelta(seconds=20),
            self.config.slot_minutes * 60,
        )

        # Record load observations (can be more frequent than schedule slots)
        self.run_every(
            self.record_load_observation,
            self._next_interval_time(self.config.load_observation_minutes),
            self.config.load_observation_minutes * 60
        )

        # Listen for optimizer enable/disable
        if self.config.enabled_entity:
            self.listen_state(self._on_enabled_change, self.config.enabled_entity)

        # Listen for manual override changes
        if self.config.override_entity:
            self.listen_state(self.on_override_change, self.config.override_entity)
        if self.config.manual_mode_entity:
            self.listen_state(self.on_manual_mode_change, self.config.manual_mode_entity)

        # Listen to SOC changes for instant response (replaces polling-based checks)
        self.listen_state(self._on_soc_change, self.config.soc_sensor)

        # Listen to inverter energy sensors (primary trigger when available)
        if self.config.use_inverter_energy_sensors:
            self.listen_state(self._on_energy_sensor_change, self.config.battery_charge_sensor)
            self.listen_state(self._on_energy_sensor_change, self.config.battery_discharge_sensor)
            self.log(f"Listening to energy sensors: {self.config.battery_charge_sensor}, {self.config.battery_discharge_sensor}")

        # Run initial SOC check on startup (listener only fires on changes)
        startup_soc = self._get_current_soc()
        if startup_soc is not None:
            self._check_soc_boundaries(startup_soc)
            # Initialize tracking state if not already set
            if self._last_soc is None:
                self._process_soc_change_event(startup_soc)

        # Create sensor for exposing schedule
        self._update_schedule_sensor()
        self._update_control_health_sensor()

        self.log("Battery Optimizer initialized successfully")

    def _should_warn_degenerate_terminal(self) -> bool:
        """Rate-limit the legacy terminal-value warning to once every 6 hours.

        The schedule is rebuilt every 15 minutes; without this the degenerate
        configuration would produce ~96 identical WARNINGs a day and be ignored
        exactly like the 70 identical INFO lines it replaces.
        """
        if self.config.terminal_energy_value_eur_kwh != 0.0:
            return False
        now = self.datetime()
        last = getattr(self, "_last_terminal_warning_time", None)
        if last is not None:
            last, now_cmp = normalize_tz_pair(last, now)
            if (now_cmp - last).total_seconds() < 6 * 3600:
                return False
        self._last_terminal_warning_time = now
        return True

    def _record_callback_duration(self, name: str, seconds: float) -> None:
        """Warn about a callback that blocked the AppDaemon worker thread."""
        limit = getattr(self.config, "callback_warn_seconds", 10.0)
        if self._slowest_callback is None or seconds > self._slowest_callback[1]:
            self._slowest_callback = (name, seconds)
        if seconds <= limit:
            return

        self._callback_overrun_count += 1
        self.log(
            f"Callback {name} took {seconds:.1f}s (> {limit:.0f}s) — AppDaemon "
            f"serializes this app's callbacks, so everything else waited",
            level="WARNING",
        )
        if self._callback_overrun_count >= 3 and not self._threads_hint_logged:
            self._threads_hint_logged = True
            self.log(
                "Repeated slow callbacks: give this app more AppDaemon threads "
                "(appdaemon.yaml -> appdaemon: total_threads: 4, or an app-level "
                "pin_thread). set_wit_mode is a blocking service call; with a "
                "single thread it stalls schedule execution, SOC listeners and "
                "PV sampling alike.",
                level="WARNING",
            )

    def _estimated_ambient_min_temp(self) -> Optional[float]:
        """Learning engine's rolling battery minimum, or None if it has none."""
        engine = getattr(self, "learning_engine", None)
        if engine is None or not engine.has_ambient_observations():
            return None
        return engine.get_estimated_ambient_min_temp()

    @_timed_callback
    def _record_ambient_observation(self, kwargs=None):
        """Timer callback: keep the ambient observation window time-uniform."""
        temp = self._get_battery_temp()
        if temp is not None:
            self.learning_engine.record_temperature_observation(temp)
        self._ambient_service.refresh()

    def _load_config(self):
        """Load configuration from apps.yaml into typed config object."""
        self.config = BatteryOptimizerConfig.from_args(self.args, log_func=self.log)
        self.config.log_summary(
            self.log, warn_func=lambda msg: self.log(msg, level="WARNING")
        )

    # =========================================================================
    # Price Fetching (delegates to NordPoolPriceService)
    # =========================================================================

    def get_prices(self) -> List[PricePoint]:
        """Fetch prices from Nord Pool. Delegates to NordPoolPriceService."""
        return self._price_service.get_prices()

    # =========================================================================
    # Optimization Algorithm
    # =========================================================================

    def calculate_min_charge_slots_for_horizon(
        self,
        current_soc: float,
        prices: List[PricePoint]
    ) -> int:
        """
        Calculate minimum charge slots needed to avoid hitting min_soc during the planning horizon.

        This simulates expected load over the horizon and determines if/when we'd run out
        of battery. Only requires charging if we'd drop below min_soc.

        Returns 0 if current battery has enough energy to survive the horizon.
        """
        if not prices:
            return 0

        # Simulate SOC through the horizon with all-discharge/hold strategy
        simulated_soc = current_soc
        total_load_kwh = 0.0

        for price_point in sorted(prices, key=lambda p: p.time):
            # Predict load for this slot
            load_kw = self._predict_load_kw(price_point.time)
            load_kwh = min(load_kw, self.config.discharge_rate) * self.config.slot_hours
            total_load_kwh += load_kwh

        # Energy available above min_soc
        usable_energy_kwh = (current_soc - self.min_soc) / 100 * self.config.battery_capacity

        # Energy deficit (how much we'd be short)
        energy_deficit_kwh = total_load_kwh - usable_energy_kwh

        if energy_deficit_kwh <= 0:
            # We have enough battery to survive the horizon
            if self.config.decision_log_level >= 1:
                self.log(
                    f"Charge calculation: SOC {current_soc:.1f}% | "
                    f"Usable energy: {usable_energy_kwh:.2f} kWh (above {self.min_soc}% min) | "
                    f"Expected load: {total_load_kwh:.2f} kWh over {len(prices)} slots | "
                    f"Surplus: {-energy_deficit_kwh:.2f} kWh | "
                    f"Result: 0 charge slots needed"
                )
            return 0

        # We need to charge to avoid hitting min_soc
        # Account for charging efficiency
        grid_energy_needed = energy_deficit_kwh / self.config.efficiency

        # Slots at charge rate
        energy_per_slot = self.config.charge_rate * self.config.efficiency * self.config.slot_hours  # Energy INTO battery per slot
        if energy_per_slot <= 0:
            return 0

        charge_slots_raw = energy_deficit_kwh / energy_per_slot
        charge_slots = math.ceil(charge_slots_raw)

        if self.config.decision_log_level >= 1:
            self.log(
                f"Charge calculation: SOC {current_soc:.1f}% | "
                f"Usable energy: {usable_energy_kwh:.2f} kWh (above {self.min_soc}% min) | "
                f"Expected load: {total_load_kwh:.2f} kWh over {len(prices)} slots | "
                f"Deficit: {energy_deficit_kwh:.2f} kWh | "
                f"Slots @ {energy_per_slot:.2f} kWh/slot | "
                f"Result: {charge_slots_raw:.2f} -> {charge_slots} charge slots needed"
            )

        return charge_slots

    def find_optimal_schedule(self, prices: List[PricePoint], charge_hours_needed: int,
                               current_soc: float = None) -> Dict[datetime.datetime, ScheduleEntry]:
        """
        Generate optimal charge/hold/discharge schedule based on prices.

        Statistical optimization with probabilistic load forecasting:
        - Uses quantile-based load predictions per slot
        - Optimizes expected profit via dynamic programming
        - Ensures SOC constraints across the full horizon
        """
        if not prices:
            return {}

        # Refresh PV and ambient forecasts (no-ops if the caches are fresh)
        self._pv_forecast_service.refresh()
        ambient_service = getattr(self, "_ambient_service", None)
        if ambient_service is not None:
            ambient_service.refresh()

        now = self.datetime()
        current_slot = self._align_to_slot(now)
        # Ensure consistent timezone awareness for arithmetic
        now, current_slot = normalize_tz_pair(now, current_slot)

        future_prices = [p for p in prices if dt_ge(p.time, current_slot)]
        if not future_prices:
            return {}

        # Ensure current slot is included (Nord Pool may exclude current hour as "past")
        future_prices = self._ensure_current_slot_price(prices, future_prices, current_slot)

        # Calculate min_charge_slots for informational purposes
        slots_sorted_by_time = sorted(future_prices, key=lambda p: instant_key(p.time))
        n_slots = len(slots_sorted_by_time)
        min_charge_slots = max(0, int(charge_hours_needed))
        if min_charge_slots > n_slots:
            min_charge_slots = n_slots

        # Prepare inputs for optimizer
        current_soc_for_calc = current_soc if current_soc is not None else 50.0
        minutes_into_slot = max(0.0, (now - current_slot).total_seconds() / 60.0)
        current_temp = self._get_battery_temp()

        # Rate-limited legacy-terminal-value warning. Resolved defensively:
        # several test doubles borrow this method without the full app state.
        _warn_gate = getattr(self, "_should_warn_degenerate_terminal", None)
        warn_degenerate_terminal = bool(_warn_gate()) if callable(_warn_gate) else False

        # Create optimizer with fresh config (min_soc/max_soc are dynamic properties)
        optimizer = DPOptimizer(
            config=DPOptimizerConfig.from_main_config(
                self.config, min_soc=self.min_soc, max_soc=self.max_soc,
            ),
            load_predictor=self._predict_load_kw,
            charge_rate_predictor=self.learning_engine.get_charge_rate_for_soc,
            temp_after_charge_predictor=self.learning_engine.predict_temp_after_duration,
            temp_after_idle_predictor=self.learning_engine.predict_temp_after_idle,
            log_fn=self.log,
            decision_log_level=self.config.decision_log_level,
            pv_predictor=self._predict_pv_kw,
            temp_projector=getattr(self, "_temp_projector", None),
            warn_degenerate_terminal=warn_degenerate_terminal,
        )

        # Run optimization
        result = optimizer.optimize(
            prices=future_prices,
            current_slot=current_slot,
            current_soc=current_soc_for_calc,
            current_temp=current_temp,
            minutes_into_slot=minutes_into_slot,
        )

        schedule = result.schedule
        # Reported trajectories. Rebuilt below if the schedule is modified after
        # the DP ran — they must describe the plan that will actually execute.
        soc_trajectory = result.soc_trajectory
        temp_trajectory = result.temp_trajectory

        # Cloud-safe conversion: HOLD → DISCHARGE(to load) during PV hours.
        # discharge_to_load charges from PV surplus (confirmed on Growatt WIT),
        # so it behaves identically to HOLD while PV covers the load. But when
        # clouds kill PV, the battery covers the load instead of the grid —
        # cheaper whenever the import price exceeds battery wear. The forced
        # forecast refresh on PV shortfall complements this: the hedge bridges
        # the gap until re-optimization, without waiting for it.
        cloud_safe_count = 0
        prices_by_slot_map = {
            canonical_slot_key(p.time): p.price for p in slots_sorted_by_time
        }
        for slot_time, entry in schedule.items():
            if entry.mode == BatteryMode.HOLD:
                pv_kw = self._predict_pv_kw(slot_time)
                if pv_kw > 0:
                    price = prices_by_slot_map.get(slot_time, 0.0)
                    buy_price = (
                        (price + self.config.grid_fee)
                        * self.config.import_price_multiplier
                    )
                    if buy_price > self.config.battery_wear_cost:
                        entry.mode = BatteryMode.DISCHARGE
                        entry.export_rate = 0
                        entry.reason += " [cloud-safe]"
                        cloud_safe_count += 1
        if cloud_safe_count > 0:
            self.log(f"Cloud-safe: converted {cloud_safe_count} HOLD→DISCHARGE(to load) "
                     f"slots during PV hours (buy_price > wear_cost)")
            # The DP built its trajectories for the PRE-conversion HOLD plan:
            # flat SOC and a cooling pack, while the plan that executes drains
            # the battery on every cloudy minute and heats it. Those same
            # trajectories are what the schedule log prefers (schedule_formatter
            # falls back to the expected-SOC map only when they are absent), so
            # leaving them stale means the log used to diagnose SOC deviations
            # describes a plan nobody runs. Rebuild through the shared model.
            soc_trajectory, temp_trajectory = self.project_schedule_trajectory(
                schedule,
                current_soc_for_calc,
                starting_temp=current_temp,
                current_slot=current_slot,
                minutes_into_slot=minutes_into_slot,
            )
            if current_temp is None:
                # Parity with DPOptimizer._build_temp_trajectory: no starting
                # temperature means no temperature trajectory at all.
                temp_trajectory = {}

        # Mode census of the plan that will actually execute. Derived here, at
        # the same point as the trajectory rebuild above, so the summary line
        # can never describe the pre-conversion schedule again.
        schedule_counts = count_schedule_modes(schedule)
        self._last_schedule_counts = schedule_counts

        # Project landed costs when the plan can add stored energy. Besides
        # explicit CHARGE, HOLD and discharge-to-load can accept PV surplus.
        has_projected_charge = any(
            entry.mode == BatteryMode.CHARGE for entry in schedule.values()
        )
        has_projected_pv_gain = any(
            entry.mode in (BatteryMode.HOLD, BatteryMode.DISCHARGE)
            and not (entry.export_rate is not None and entry.export_rate > 0)
            and self._predict_pv_kw(slot_time) > self._predict_load_kw(slot_time)
            for slot_time, entry in schedule.items()
        )
        if schedule and (has_projected_charge or has_projected_pv_gain):
            local_tz = self._get_local_timezone()
            prices_by_slot = {
                canonical_slot_key(p.time): p.price for p in slots_sorted_by_time
            }
            # Re-compute charge rates per slot for cost projection
            slot_fractions = self._compute_slot_fractions(
                slots_sorted_by_time,
                current_slot,
                minutes_into_slot,
                local_tz=local_tz,
            )
            charge_rates_per_slot = self._compute_charge_rates_per_slot(
                slots_sorted_by_time, slot_fractions, current_soc_for_calc, current_temp
            )
            slot_charge_rates_by_slot = {
                canonical_slot_key(p.time): charge_rates_per_slot[i]
                for i, p in enumerate(slots_sorted_by_time)
            }
            slot_fractions_by_slot = {
                canonical_slot_key(p.time): slot_fractions[i]
                for i, p in enumerate(slots_sorted_by_time)
            }
            projected_costs, _ = self._cost_tracker.project_costs(
                schedule,
                current_soc_for_calc,
                self.battery_avg_cost,
                prices_by_slot,
                predict_load_func=self._predict_load_kw,
                predict_pv_func=self._predict_pv_kw,
                charge_rates_by_slot=slot_charge_rates_by_slot,
                slot_fractions_by_slot=slot_fractions_by_slot,
                # Same CHARGE/thermal model as project_schedule_trajectory, so
                # the projected-cost column cannot disagree with the SOC and
                # temperature ones.
                starting_temp=current_temp,
                learning_engine=getattr(self, "learning_engine", None),
                # getattr, like project_schedule_trajectory above: the test
                # doubles for this method construct neither attribute.
                temp_projector=getattr(self, "_temp_projector", None),
            )
            self._last_projected_costs = projected_costs
        else:
            self._last_projected_costs = {}

        # Counted from the FINAL schedule, not from result.*_count: the DP
        # counted the pre-conversion plan, so on any run with cloud_safe_count
        # > 0 this line contradicted the schedule log's own "Total:" census a
        # few lines below it.
        parts = schedule_counts.summary_parts()
        self.log(f"Schedule generated: {', '.join(parts)} slots "
                 f"(slot={self.config.slot_minutes}min, "
                 f"load_quantile={self.config.load_quantile:.2f}, min_charge_slots={min_charge_slots})")

        # Store min_charge_slots for sensor exposure
        self._last_min_charge_slots = min_charge_slots

        # Log decision context for transparency
        if self.config.decision_log_level >= 1:
            load_kw = [self._predict_load_kw(p.time) for p in slots_sorted_by_time]
            self._last_charge_slots = self._schedule_formatter.log_decision_context(
                prices_sorted=slots_sorted_by_time,
                schedule=schedule,
                load_kw=load_kw,
                current_soc=current_soc_for_calc,
                min_charge_slots=min_charge_slots,
                battery_avg_cost=self.battery_avg_cost,
                min_soc=self.min_soc,
            )

        # Store trajectories for use in _log_schedule (rebuilt above if the
        # cloud-safe conversion changed the schedule).
        self._last_dp_soc_trajectory = soc_trajectory
        self._last_dp_temp_trajectory = temp_trajectory

        return schedule

    def _ensure_current_slot_price(
        self,
        all_prices: List[PricePoint],
        future_prices: List[PricePoint],
        current_slot: datetime.datetime,
    ) -> List[PricePoint]:
        """
        Ensure current slot is included in future_prices.

        Nord Pool may exclude current hour as "past", so we try to
        synthesize the price from yesterday's data or previous prices.
        """
        local_tz = self._get_local_timezone()

        def prices_contains_slot(prices_list, slot):
            return any(datetimes_match_slot(p.time, slot, local_tz)
                       for p in prices_list)

        if prices_contains_slot(future_prices, current_slot):
            return future_prices

        # Current slot missing - try yesterday's Nord Pool data (timezone shift around midnight)
        tz = local_tz
        yesterday = current_slot.date() - datetime.timedelta(days=1)
        yesterday_prices = self._price_service.get_prices_for_date(yesterday, tz)

        def find_same_clock_price(prices_list, slot):
            """Find yesterday's equivalent local clock slot, ignoring its date."""
            local_slot = ensure_local_tz(slot, tz)
            candidates = []
            for point in prices_list:
                local_point = ensure_local_tz(point.time, tz)
                if (local_point.hour, local_point.minute) == (local_slot.hour, local_slot.minute):
                    candidates.append((local_point, point))
            if not candidates:
                return None
            # On an autumn-fold day prefer the corresponding occurrence.
            for local_point, point in candidates:
                if local_point.fold == local_slot.fold:
                    return point
            return candidates[0][1]

        slot_price_point = find_same_clock_price(yesterday_prices, current_slot)
        if slot_price_point is not None:
            synth_price = slot_price_point.price
            self.log(
                f"Added missing current slot {current_slot} using yesterday's price "
                f"{synth_price:.4f} EUR/kWh from {slot_price_point.time}"
            )
        else:
            # If still missing, synthesize using most recent past price if available
            prev_price_point = None
            for p in all_prices:
                if dt_ge(current_slot, p.time, tz):
                    if (prev_price_point is None or
                            dt_ge(p.time, prev_price_point.time, tz)):
                        prev_price_point = p

            if prev_price_point is not None:
                synth_price = prev_price_point.price
                self.log(
                    f"Added missing current slot {current_slot} using previous price "
                    f"{synth_price:.4f} EUR/kWh from {prev_price_point.time}"
                )
            else:
                # Fallback: use first available future price
                synth_price = min(future_prices, key=lambda p: instant_key(p.time)).price
                self.log(f"Added missing current slot {current_slot} using next price {synth_price:.4f} EUR/kWh")

        # Normalize timezone to match existing prices (avoid mixing aware/naive)
        if future_prices:
            sample_hour = future_prices[0].time
            if sample_hour.tzinfo is not None and current_slot.tzinfo is None:
                # Prices are aware, current_slot is naive - add timezone
                tz = self._get_local_timezone()
                current_slot = ensure_local_tz(current_slot, tz)
            elif sample_hour.tzinfo is None and current_slot.tzinfo is not None:
                # Prices are naive, current_slot is aware - strip timezone
                current_slot = current_slot.replace(tzinfo=None)

        current_slot_price = PricePoint(time=current_slot, price=synth_price)
        return future_prices + [current_slot_price]

    def _compute_slot_fractions(
        self,
        slots_sorted_by_time: List[PricePoint],
        current_slot: datetime.datetime,
        minutes_into_slot: float,
        local_tz=None,
    ) -> List[float]:
        """Compute fraction of each slot that is usable (partial first slot)."""
        n_slots = len(slots_sorted_by_time)
        first_fraction = min(1.0, max(0.0, (self.config.slot_minutes - minutes_into_slot) / max(1, self.config.slot_minutes)))
        slot_fractions = [1.0] * n_slots
        if local_tz is None:
            local_tz = self._get_local_timezone()

        for i, p in enumerate(slots_sorted_by_time):
            if datetimes_match_slot(p.time, current_slot, local_tz):
                slot_fractions[i] = first_fraction
                break

        return slot_fractions

    def _compute_charge_rates_per_slot(
        self,
        slots_sorted_by_time: List[PricePoint],
        slot_fractions: List[float],
        current_soc: float,
        current_temp: Optional[float],
    ) -> List[float]:
        """Pre-compute temperature-aware charge rates for each slot.

        This is the only path where the temperature forecast changes DP
        decisions, so it uses the shared projector (time-varying ambient,
        bounded) rather than the unbounded linear warming projection.
        """
        projector = getattr(self, "_temp_projector", None)
        return compute_charge_rates_per_slot(
            slots_sorted_by_time=slots_sorted_by_time,
            slot_fractions=slot_fractions,
            slot_minutes=self.config.slot_minutes,
            current_soc=current_soc,
            current_temp=current_temp,
            get_charge_rate_for_soc=self.learning_engine.get_charge_rate_for_soc,
            predict_temp_after_duration=self.learning_engine.predict_temp_after_duration,
            project_temp=projector.project if projector is not None else None,
            battery_capacity=self.config.battery_capacity,
            efficiency=self.config.efficiency,
            max_soc=self.max_soc,
        )

    def calculate_expected_soc_schedule(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        starting_soc: float,
        starting_temp: Optional[float] = None,
        current_slot: Optional[datetime.datetime] = None,
        minutes_into_slot: float = 0.0,
    ) -> Tuple[Dict[datetime.datetime, float], Dict[datetime.datetime, float]]:
        """
        Calculate expected SOC and temperature at each slot based on schedule.
        Used for adaptive optimization to detect deviations.

        The slot transition itself is delegated to ``project_slot_soc`` so that
        this trajectory, the deviation detector and the DP share one physics
        model (see battery_optimizer_lib/soc_projection.py).

        Args:
            schedule: Schedule entries keyed by slot datetime
            starting_soc: Initial SOC percentage
            starting_temp: Initial battery temperature in Celsius (optional)
            current_slot: Slot the projection starts in. When given together
                with ``minutes_into_slot``, only the remaining part of that slot
                is projected — exactly like ``DPOptimizer`` does with
                ``slot_fractions`` (see ``_compute_slot_fractions``). Without it
                every slot is treated as a full slot (legacy behaviour).
            minutes_into_slot: Minutes already elapsed in ``current_slot``.

        Returns:
            Tuple of (soc_trajectory, temp_trajectory) dicts keyed by slot datetime

        Notes:
            - The recorded value for a slot is the SOC/temperature at the START
              of the projected interval. For a partial first slot that is the
              projection instant, not the slot boundary.
            - Discharge drains at predicted net load (PV serves load first) and
              stores PV surplus; export slots drain at the export rate
            - Charge adds energy using temperature-aware rates when temp available
            - Temperature evolves in EVERY slot through the shared thermal model
              (``thermal_model.TemperatureProjector``): relaxation toward a
              time-varying ambient plus ``k2*|P_bat|``. Discharging heats the
              pack too — it is not thermally idle.
        """
        soc_trajectory, temp_trajectory = self.project_schedule_trajectory(
            schedule,
            starting_soc,
            starting_temp=starting_temp,
            current_slot=current_slot,
            minutes_into_slot=minutes_into_slot,
        )
        expected_soc = {slot: pair[0] for slot, pair in soc_trajectory.items()}
        expected_temp = {
            slot: pair[0]
            for slot, pair in temp_trajectory.items()
            if pair[0] is not None
        }
        return expected_soc, expected_temp

    def project_schedule_trajectory(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        starting_soc: float,
        starting_temp: Optional[float] = None,
        current_slot: Optional[datetime.datetime] = None,
        minutes_into_slot: float = 0.0,
    ) -> Tuple[
        Dict[datetime.datetime, Tuple[float, float]],
        Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]],
    ]:
        """Walk a schedule through the shared slot-SOC/thermal models.

        Returns ``({slot: (soc_start, soc_end)}, {slot: (temp_start, temp_end)})``
        — the same shape ``DPOptimizerResult`` uses, so a schedule that was
        modified after the DP ran (the cloud-safe HOLD -> DISCHARGE conversion)
        can have its reported trajectories rebuilt for the plan that will
        actually execute.

        The transition itself is ``soc_projection.project_slot_soc`` — never a
        local re-implementation (CLAUDE.md "One slot-SOC model").
        """
        # Built inline (not via a helper) so that any object providing the same
        # config/min_soc/max_soc surface can reuse this method directly.
        params = SocProjectionParams(
            battery_capacity=self.config.battery_capacity,
            efficiency=self.config.efficiency,
            charge_rate=self.config.charge_rate,
            discharge_rate=self.config.discharge_rate,
            export_discharge_rate=self.config.export_discharge_rate,
            inverter_efficiency=self.config.inverter_efficiency,
            min_soc=self.min_soc,
            max_soc=self.max_soc,
            slot_minutes=self.config.slot_minutes,
        )
        # Same partial-slot formula as _compute_slot_fractions / DPOptimizer.
        first_fraction = min(
            1.0,
            max(0.0, (self.config.slot_minutes - minutes_into_slot) / max(1, self.config.slot_minutes)),
        )
        local_tz = None
        if current_slot is not None:
            get_tz = getattr(self, "_get_local_timezone", None)
            if get_tz is not None:
                local_tz = get_tz()
        partial_applied = False
        temp_projector = getattr(self, "_temp_projector", None)

        soc_trajectory: Dict[datetime.datetime, Tuple[float, float]] = {}
        temp_trajectory: Dict[
            datetime.datetime, Tuple[Optional[float], Optional[float]]
        ] = {}
        current_soc = starting_soc
        current_temp = starting_temp

        for hour in sorted(schedule.keys()):
            entry = schedule[hour]

            fraction = 1.0
            if (current_slot is not None and not partial_applied
                    and datetimes_match_slot(hour, current_slot, local_tz)):
                fraction = first_fraction
                partial_applied = True

            transition = project_slot_soc(
                soc_start=current_soc,
                mode=entry.mode,
                params=params,
                load_kw=self._predict_load_kw(hour),
                pv_kw=self._predict_pv_kw(hour),
                fraction=fraction,
                export_rate=entry.export_rate,
                temp_start=current_temp,
                learning_engine=self.learning_engine,
                temp_projector=temp_projector,
                slot_time=hour,
            )
            soc_trajectory[hour] = (current_soc, transition.soc_end)
            temp_trajectory[hour] = (current_temp, transition.temp_end)
            current_soc = transition.soc_end
            current_temp = transition.temp_end

        return soc_trajectory, temp_trajectory

    # =========================================================================
    # Schedule Execution
    # =========================================================================

    @_timed_callback
    def full_optimize(self, kwargs=None):
        """
        Perform full optimization - called daily at 13:15 and at startup.
        """
        self.log("Starting full optimization")

        # Determine trigger type: startup (no previous recalc) or daily schedule
        now = self.datetime()
        is_startup = self._last_recalc_time is None
        if is_startup:
            self._last_recalc_trigger = "startup"
        else:
            # Check if this is around the scheduled daily time (within 30 min of tomorrow_prices_hour:15)
            scheduled_hour = self.config.tomorrow_prices_hour
            if now.hour == scheduled_hour and 0 <= now.minute <= 45:
                self._last_recalc_trigger = "daily_scheduled"
            else:
                self._last_recalc_trigger = "manual"
        self._last_recalc_time = now
        self._last_soc_deviation = None  # Clear deviation since this isn't SOC-triggered

        if not self._is_enabled():
            self.log("Optimizer disabled, skipping")
            return

        # Get current SOC
        current_soc = self._get_current_soc()
        if current_soc is None:
            self.log("Cannot get current SOC, skipping optimization", level="WARNING")
            return

        # Fetch prices
        prices = self.get_prices()
        if not prices:
            self.log("No price data available, skipping optimization", level="WARNING")
            return

        # Filter to future prices only (avoid past slots inflating min-charge calculation)
        current_slot = self._align_to_slot(now)
        future_prices = [p for p in prices if dt_ge(p.time, current_slot)]
        if not future_prices:
            self.log("No future prices available, skipping optimization", level="WARNING")
            return

        # Freeze the PV bias factor for the whole generation pass
        self._refresh_pv_bias_factor()

        # Calculate minimum charge slots needed to survive the planning horizon
        charge_hours_needed = self.calculate_min_charge_slots_for_horizon(current_soc, future_prices)
        self.log(f"Current SOC: {current_soc}%, min charge slots needed: {charge_hours_needed}")

        # Generate schedule
        self.schedule = self.find_optimal_schedule(future_prices, charge_hours_needed, current_soc)

        # On restart, preserve CHARGE/DISCHARGE intent for current slot if it was active before
        self._preserve_mode_on_restart(current_slot)

        # Calculate expected SOC and temperature trajectory. The current slot is
        # already partially elapsed, so only its remaining fraction is projected
        # (matching the DP's slot_fractions) — otherwise the next slot boundary
        # would compare actual SOC against a full extra slot of charging.
        current_temp = self._get_battery_temp()
        # self.datetime() is naive local; _align_to_slot() returns tz-aware via
        # ensure_local_tz(). Subtracting them directly raises TypeError.
        now_cmp, slot_cmp = normalize_tz_pair(now, current_slot)
        minutes_into_slot = max(0.0, (now_cmp - slot_cmp).total_seconds() / 60.0)
        self.expected_soc_schedule, self.expected_temp_schedule = self.calculate_expected_soc_schedule(
            self.schedule,
            current_soc,
            starting_temp=current_temp,
            current_slot=current_slot,
            minutes_into_slot=minutes_into_slot,
        )
        self._expected_soc_anchor = now

        # Log the generated schedule (with expected SOC and temperature)
        self._schedule_formatter.log_schedule(
            schedule=self.schedule,
            expected_soc=self.expected_soc_schedule,
            expected_temp=self.expected_temp_schedule,
            dp_soc_trajectory=self._last_dp_soc_trajectory,
            dp_temp_trajectory=self._last_dp_temp_trajectory,
            projected_costs=self._last_projected_costs,
            local_tz=self._get_local_timezone(),
            predict_load_kw=self._predict_load_kw,
            predict_pv_kw=self._predict_pv_kw,
            min_soc=self.min_soc,
            max_soc=self.max_soc,
        )

        self.last_optimization = self.datetime()

        # Apply current slot's mode immediately
        self.execute_scheduled_mode(None)

        # Update sensor
        self._update_schedule_sensor()

        # If PV forecast was unavailable at startup, HA sensors may not have been
        # ready yet (Solcast HACS integration loads attributes asynchronously).
        # Schedule a re-optimization after 30s when sensors are more likely populated.
        if (is_startup
                and not self._pv_forecast_service.has_forecast):
            self.log(
                "PV forecast unavailable at startup — scheduling re-optimization "
                "in 30s to pick up Solcast data",
                level="WARNING",
            )
            self.run_in(self.full_optimize, 30)

        self.log("Full optimization complete")

    @_timed_callback
    def adaptive_optimize(self, kwargs=None):
        """
        Adaptive re-evaluation on a configurable interval.
        Handles PV override and schedule change logging.
        SOC deviation detection is now event-driven via _on_soc_change.
        """
        if not self._is_enabled() or self._is_override_active():
            return

        current_soc = self._get_current_soc()
        if current_soc is None:
            return

        # Safety: if significant PV is producing during a grid-charge slot and
        # the DP didn't anticipate it (e.g. fresh PV profile), pause charging to
        # let the inverter use solar instead of paying for grid.
        pv_power = self._get_pv_power()
        if pv_power > self.pv_threshold and self.current_mode == BatteryMode.CHARGE:
            self.log(f"Solar override: PV={pv_power}W > threshold={self.pv_threshold}W, "
                     f"switching from charge to hold (PV covers load, surplus exports)")
            entry = ScheduleEntry(
                time=self._align_to_slot(self.datetime()),
                mode=BatteryMode.HOLD,
                reason="solar_override",
            )
            self._handle_mode_transition(BatteryMode.HOLD)
            self._apply_mode_tracked(entry)
            return

        # Reactive PV check on the just-COMPLETED slot.  The comparison uses the
        # slot's mean measured power (slot energy / slot hours) built from many
        # samples, never a single instantaneous reading taken at the boundary,
        # and requires `pv_reactive_consecutive_slots` consecutive shortfalls
        # before paying for a full recalculation.
        self._check_pv_shortfall(current_soc)

    def _check_pv_shortfall(self, current_soc: float) -> bool:
        """Evaluate the completed slot for a measured PV shortfall.

        Returns True if a recalculation was triggered.
        """
        now = self.datetime()
        now_slot = self._align_to_slot(now)

        # Snapshot the RAW forecast before refresh_for_shortfall can cap it.
        self._pv_bias.ensure_slot_forecast(now_slot, self._predict_pv_kw_raw(now_slot))

        # Idempotent — the sampling timer may have closed these already.
        if self._pv_bias.close_slots_before(now_slot):
            self._refresh_pv_bias_factor()

        # DST-safe: wall-clock subtraction on an aware datetime is a 1h15m step
        # across the Europe/Riga autumn fold (and -45min in spring), which made
        # get_closed() miss the slot that just closed. prev_slot_time moves by
        # one slot as a UTC instant.
        prev_slot = prev_slot_time(
            now_slot, self.config.slot_minutes, self._get_local_timezone()
        )
        completed = self._pv_bias.get_closed(prev_slot)
        if completed is None or completed.samples < self.config.pv_reactive_min_samples:
            return False
        if completed.forecast_kw * 1000.0 <= self.config.pv_reactive_min_forecast_w:
            return False
        if completed.ratio >= self.config.pv_reactive_threshold:
            return False

        streak = self._pv_bias.shortfall_streak
        detail = (
            f"PV below forecast at {prev_slot.strftime('%H:%M')}: "
            f"actual={completed.actual_kw * 1000:.0f}W vs "
            f"forecast={completed.forecast_kw * 1000:.0f}W "
            f"({completed.ratio * 100:.0f}%, n={completed.samples})"
        )
        if streak < self.config.pv_reactive_consecutive_slots:
            self.log(
                f"{detail}, streak {streak}/"
                f"{self.config.pv_reactive_consecutive_slots} — no recalc "
                f"(bias={self._pv_bias_factor:.2f})"
            )
            return False

        self.log(
            f"{detail}, streak {streak} — recalculating "
            f"(bias={self._pv_bias_factor:.2f})"
        )
        # Bypass the normal forecast cache and cap the current slot at observed
        # production. The forced provider read is separately rate-limited, so
        # persistent clouds cannot hammer the API or immediately reintroduce the
        # same optimistic cached value.
        self._pv_forecast_service.refresh_for_shortfall(
            now_slot, completed.actual_kw
        )
        # Restart the streak: without this the counter only ever grows while the
        # clouds last, so after the first trigger EVERY following slot would
        # recalculate. The guard is "N consecutive shortfalls since the last
        # recalculation", not "since the last sunny slot".
        self._pv_bias.reset_shortfall_streak()
        self._last_recalc_trigger = "pv_shortfall"
        self._last_recalc_time = now
        self._recalculate_remaining_schedule(current_soc)
        return True

    def _get_pv_power_optional(self) -> Optional[float]:
        """Current PV power (W), or None when the sensor is unavailable.

        Unlike `_get_pv_power()` this does NOT collapse an unavailable sensor
        into 0 W — a missing reading must not be recorded as "no production".
        """
        return self._sensors.get_float(self.config.pv_power_sensor)

    @_timed_callback
    def _sample_pv(self, kwargs=None):
        """Accumulate PV power samples and close completed slots."""
        try:
            now = self.datetime()
            now_slot = self._align_to_slot(now)

            closed = self._pv_bias.close_slots_before(now_slot)
            for entry in closed:
                if (self.config.decision_log_level >= 1
                        and entry.forecast_kw * 1000.0
                        >= self.config.pv_reactive_min_forecast_w):
                    self.log(
                        f"PV slot {entry.slot.strftime('%m-%d %H:%M')}: "
                        f"actual={entry.actual_kw * 1000:.0f}W vs "
                        f"forecast={entry.forecast_kw * 1000:.0f}W "
                        f"(ratio {entry.ratio:.2f}, n={entry.samples})"
                    )
            if closed:
                self._refresh_pv_bias_factor()

            # Snapshot the RAW forecast for the open slot (first write wins).
            self._pv_bias.ensure_slot_forecast(
                now_slot, self._predict_pv_kw_raw(now_slot)
            )

            pv_w = self._get_pv_power_optional()
            if pv_w is not None and pv_w >= 0:
                self._pv_bias.add_sample(now, pv_w / 1000.0)
        except Exception as e:
            self.log(f"PV sampling failed: {e}", level="WARNING")

    def _refresh_pv_bias_factor(self) -> float:
        """Recompute the sliding PV bias factor and log material changes."""
        now = self.datetime()
        new_factor = self._pv_bias.get_factor(now)
        if abs(new_factor - self._pv_bias_factor) >= 0.02:
            self.log(
                f"PV bias factor {self._pv_bias_factor:.2f} -> {new_factor:.2f} "
                f"({self._pv_bias.describe(now)})"
            )
        self._pv_bias_factor = new_factor
        return new_factor

    def _recalculate_remaining_schedule(self, current_soc: float, extra_charge_slots: int = 0):
        """
        Recalculate schedule for remaining hours based on current SOC.

        Args:
            current_soc: Current battery state of charge (%)
            extra_charge_slots: Additional charge slots to add beyond minimum required
                               (used for catch-up charging when behind schedule)
        """
        local_tz = self._get_local_timezone()
        now = ensure_local_tz(self.datetime(), local_tz)
        now_slot = self._align_to_slot(now)
        prices = self.get_prices()

        # Filter to future prices only
        future_prices = [p for p in prices if dt_ge(p.time, now_slot, local_tz)]

        if not future_prices:
            self.log("No future prices available for recalculation")
            return

        self.log(f"Recalculating with {len(future_prices)} future price points "
                 f"({future_prices[0].time.strftime('%m-%d %H:%M')} to {future_prices[-1].time.strftime('%m-%d %H:%M')})")

        # Freeze the PV bias factor for the whole generation pass
        self._refresh_pv_bias_factor()

        # Recalculate minimum charge slots needed to survive the remaining horizon
        charge_hours_needed = self.calculate_min_charge_slots_for_horizon(current_soc, future_prices)

        # Add extra slots for catch-up charging (when behind schedule and need to reach max_soc)
        if extra_charge_slots > 0:
            self.log(f"Boosting min_charge_slots by {extra_charge_slots} to catch up to target SOC "
                     f"(base={charge_hours_needed}, total={charge_hours_needed + extra_charge_slots})")
            charge_hours_needed += extra_charge_slots

        # Generate new schedule for remaining time
        new_schedule = self.find_optimal_schedule(future_prices, charge_hours_needed, current_soc)

        # Remove all future entries and replace with new schedule
        # This prevents stale entries from persisting if price list shrinks
        hours_to_remove = [h for h in self.schedule.keys() if dt_ge(h, now_slot, local_tz)]
        for hour in hours_to_remove:
            del self.schedule[hour]

        # Add new schedule entries
        for hour, entry in new_schedule.items():
            self.schedule[hour] = entry

        # Recalculate expected SOC
        current_temp = self._get_battery_temp()
        future_schedule = {k: v for k, v in self.schedule.items() if dt_ge(k, now_slot, local_tz)}
        # See full_optimize(): naive self.datetime() vs tz-aware slot boundary.
        now_cmp, slot_cmp = normalize_tz_pair(now, now_slot)
        minutes_into_slot = max(0.0, (now_cmp - slot_cmp).total_seconds() / 60.0)
        self.expected_soc_schedule, self.expected_temp_schedule = self.calculate_expected_soc_schedule(
            future_schedule,
            current_soc,
            starting_temp=current_temp,
            current_slot=now_slot,
            minutes_into_slot=minutes_into_slot,
        )
        self._expected_soc_anchor = now

        # Log recalculated schedule (current/future only)
        if self.config.decision_log_level >= 1:
            self._schedule_formatter.log_schedule(
                schedule=future_schedule,
                expected_soc=self.expected_soc_schedule,
                expected_temp=self.expected_temp_schedule,
                dp_soc_trajectory=self._last_dp_soc_trajectory,
                dp_temp_trajectory=self._last_dp_temp_trajectory,
                projected_costs=self._last_projected_costs,
                local_tz=local_tz,
                predict_load_kw=self._predict_load_kw,
                predict_pv_kw=self._predict_pv_kw,
                min_soc=self.min_soc,
                max_soc=self.max_soc,
            )

        # Apply updated mode immediately
        self.execute_scheduled_mode(None)

        self._update_schedule_sensor()

    @_timed_callback
    def execute_scheduled_mode(self, kwargs, force: bool = False):
        """
        Execute the scheduled mode for the current slot.
        Called at the start of each slot. Sends a direct mode command to the inverter.

        Args:
            kwargs: AppDaemon callback kwargs
            force: If True, skip override check (used when manual mode set to "Auto")
        """
        if not self._is_enabled():
            return

        if not force and self._is_override_active():
            self.log("Manual override active, skipping scheduled execution")
            return

        # Get current slot in local timezone for schedule lookup
        now = self.datetime()
        local_tz = self._get_local_timezone()
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        elif local_tz is not None:
            now = now.replace(tzinfo=local_tz)
        current_slot = self._align_to_slot(now)

        # Pre-execution SOC check: if actual SOC is significantly behind plan,
        # recalculate the schedule before blindly applying the planned mode.
        # This prevents e.g. starting Max Export when the preceding charge slot
        # didn't reach its target SOC.
        current_soc = self._get_current_soc()
        if current_soc is not None and self.expected_soc_schedule:
            # Compare against the plan AT THIS INSTANT, not at the slot
            # boundary. Mid-slot entry points (override toggle, manual "Auto",
            # a forced re-execution) land part-way into a slot, where the
            # start-of-slot value is several points above reality during a
            # DISCHARGE — that difference is elapsed time, not a shortfall, and
            # it used to trip soc_shortfall_recalc_threshold on its own. The
            # interpolation goes through the shared soc_projection model via
            # SocDeviationDetector, never a local formula.
            expected_soc = self._build_soc_deviation_detector().expected_soc_at(
                current_soc=current_soc,
                schedule=self.schedule,
                expected_soc_schedule=self.expected_soc_schedule,
                now=now,
                current_slot=current_slot,
                local_tz=local_tz,
                current_temp=self._get_battery_temp(),
                predict_load_kw=self._predict_load_kw,
                predict_pv_kw=self._predict_pv_kw,
                expected_soc_anchor=getattr(self, "_expected_soc_anchor", None),
            )
            if expected_soc is not None:
                soc_shortfall = expected_soc - current_soc
                if soc_shortfall > self.config.soc_shortfall_recalc_threshold:
                    self.log(f"SOC behind plan: {current_soc:.1f}% vs expected {expected_soc:.1f}% "
                             f"(shortfall: {soc_shortfall:.1f}%), recalculating before executing slot")
                    self._recalculate_remaining_schedule(current_soc)
                    # _recalculate_remaining_schedule calls execute_scheduled_mode
                    # internally with the updated schedule, so we return here.
                    return

        entry = self.schedule.get(current_slot)

        # If not found, try matching by time components with different tz representations
        if entry is None and self.schedule:
            for schedule_hour, schedule_entry in self.schedule.items():
                compare_schedule = schedule_hour
                if schedule_hour.tzinfo is not None and local_tz is not None:
                    compare_schedule = schedule_hour.astimezone(local_tz)
                compare_current = current_slot
                if current_slot.tzinfo is not None and local_tz is not None:
                    compare_current = current_slot.astimezone(local_tz)
                if (compare_schedule.date() == compare_current.date() and
                    compare_schedule.hour == compare_current.hour and
                    compare_schedule.minute == compare_current.minute):
                    entry = schedule_entry
                    current_slot = schedule_hour
                    break

        if entry is None:
            entry = ScheduleEntry(
                time=current_slot,
                mode=BatteryMode.HOLD,
                reason="no_schedule",
            )

        self.log(f"Executing scheduled mode for {current_slot}: {entry.mode.name} ({entry.reason})")

        # Finalize previous slot outcome and record predictions for this slot
        current_soc = self._get_current_soc()
        self._outcome_tracker.record_slot_end(
            actual_soc=current_soc,
            actual_pv_w=self._get_pv_power(),
            actual_mode=self._get_inverter_mode(),
        )
        # DST-safe slot step + tz-normalized lookup: plain `+ timedelta` and a
        # bare dict .get() both miss around a DST transition.
        next_slot = slot_offset(current_slot, self.config.slot_minutes, 1, local_tz)
        predicted_soc_end = lookup_by_time(
            self.expected_soc_schedule, next_slot, local_tz
        )
        if predicted_soc_end is None:
            predicted_soc_end = lookup_by_time(
                self.expected_soc_schedule, current_slot, local_tz
            )
        if predicted_soc_end is None:
            predicted_soc_end = current_soc if current_soc is not None else 50.0
        self._outcome_tracker.record_slot_start(
            slot_time=current_slot,
            mode=entry.mode.name,
            predicted_soc_end=predicted_soc_end,
            predicted_load_kw=self._predict_load_kw(current_slot),
            predicted_pv_kw=self._predict_pv_kw(current_slot),
        )

        # At max SOC with PV surplus, DISCHARGE remote-control clips PV
        # export — use HOLD instead so surplus PV exports to grid.
        # When PV < load, keep DISCHARGE so battery covers the deficit.
        if (entry.mode == BatteryMode.DISCHARGE
                and current_soc is not None
                and current_soc >= self.max_soc):
            pv_w = self._get_pv_power()
            load_w = self._get_load_power() or 0.0
            if pv_w > load_w:
                self.log(f"Overriding DISCHARGE→HOLD at max SOC ({current_soc}%) "
                         f"— PV {pv_w:.0f}W > load {load_w:.0f}W, allowing export")
                entry = ScheduleEntry(
                    time=entry.time,
                    mode=BatteryMode.HOLD,
                    reason="safety_max_soc_pv_export",
                )

        # Track mode transition for cost tracking / learning
        self._handle_mode_transition(entry.mode)

        # Send command to inverter
        self._apply_mode_tracked(entry)

    def _apply_mode_tracked(self, entry: ScheduleEntry) -> bool:
        """Send one mode command and account for its outcome.

        EVERY ``DirectControl.apply_mode`` call must go through here. Five of
        the six call sites used to discard the boolean result, so:

        * a successful safety HOLD never reset ``_consecutive_apply_failures``
          — the counter kept climbing across unrelated slots and eventually
          raised the "inverter is NOT following the schedule" ERROR while the
          inverter was in fact obeying every command;
        * a FAILED safety apply (min-SOC HOLD, max-SOC HOLD, solar override,
          manual mode) was completely silent — the battery kept discharging
          below min_soc with nothing in the log.

        It accounts for the OUTCOME, not for ``apply_mode``'s boolean. That
        boolean is True for three cases the inverter never acknowledged — a dry
        run, a duplicate that was never transmitted, and a client-side timeout —
        and counting them as successes reset ``_consecutive_apply_failures`` on
        every call. With growatt_modbus hung but not raising, every command hit
        its ``hass_timeout``, the health sensor reported climbing
        ``apply_successes``, and the "inverter is NOT following the schedule"
        ERROR could never fire. Now:

        * SENT                -> success, resets both streaks;
        * FAILED              -> failure, escalates after 3 in a row;
        * UNCONFIRMED_TIMEOUT -> NOT a success (leaves the failure streak
          alone) and escalates on its own after 3 in a row;
        * SKIPPED_DUPLICATE / DRY_RUN -> neutral, no streak is touched.

        Args:
            entry: Schedule entry to send to the inverter.

        Returns:
            False only for a CONFIRMED failure — an unconfirmed timeout still
            returns True because DirectControl recorded it as sent and
            verify-after-set is what resolves it.
        """
        outcome = self._direct_control.apply_mode_with_outcome(entry)

        if outcome is ApplyOutcome.FAILED:
            self._apply_failure_count += 1
            self._consecutive_apply_failures += 1
            self.log(
                f"Failed to apply mode {entry.mode.name} ({entry.reason}) — "
                f"will retry next slot",
                level="WARNING",
            )
            if self._consecutive_apply_failures >= 3:
                # Three slots in a row means the inverter has been running on
                # its panel-configured base mode for ~45 minutes.
                self.log(
                    f"{self._consecutive_apply_failures} consecutive apply_mode "
                    f"failures — the inverter is NOT following the schedule. "
                    f"Check the growatt_modbus connection and "
                    f"sensor.battery_inverter_control_health.",
                    level="ERROR",
                )
        elif outcome is ApplyOutcome.UNCONFIRMED_TIMEOUT:
            self._apply_unconfirmed_count += 1
            self._consecutive_apply_unconfirmed += 1
            if self._consecutive_apply_unconfirmed >= 3:
                self.log(
                    f"{self._consecutive_apply_unconfirmed} consecutive "
                    f"unconfirmed apply_mode timeouts — the inverter is NOT "
                    f"following the schedule (no command has been acknowledged). "
                    f"Check the growatt_modbus connection and "
                    f"sensor.battery_inverter_control_health.",
                    level="ERROR",
                )
        elif outcome is ApplyOutcome.SKIPPED_DUPLICATE:
            # Nothing was transmitted — neither evidence of health nor of
            # failure, so no streak may be reset here.
            self._apply_duplicate_count += 1
        elif outcome is ApplyOutcome.DRY_RUN:
            # Dry-run must not look like perfect health on the sensor.
            self._apply_dry_run_count += 1
        else:
            self._apply_success_count += 1
            self._consecutive_apply_failures = 0
            self._consecutive_apply_unconfirmed = 0

        self._update_control_health_sensor()
        return outcome is not ApplyOutcome.FAILED

    def _update_control_health_sensor(self) -> None:
        """Publish inverter-control diagnostics as its own HA sensor.

        Created with set_state (like the markdown sensor), so it is not declared
        in homeassistant/packages and disappears until the app republishes it
        after an HA restart. Use it for alerting on trends, not as history.
        """
        try:
            diagnostics = self._direct_control.get_diagnostics()
            self.set_state(
                "sensor.battery_inverter_control_health",
                # MUST be a string: HA states are strings, and an int 0 is
                # falsy — it was dropped from the POST body, so HA rejected the
                # whole call with "[400] Bad Request" on every mode apply.
                state=str(diagnostics.get("persistent_mismatch_count", 0)),
                attributes={
                    **diagnostics,
                    "apply_failures": self._apply_failure_count,
                    "consecutive_apply_failures": self._consecutive_apply_failures,
                    "apply_successes": self._apply_success_count,
                    # Added, never renamed: an unconfirmed timeout used to be
                    # indistinguishable from a confirmed send on this sensor.
                    "apply_unconfirmed": self._apply_unconfirmed_count,
                    "consecutive_apply_unconfirmed":
                        self._consecutive_apply_unconfirmed,
                    "apply_duplicates_skipped": self._apply_duplicate_count,
                    "apply_dry_runs": self._apply_dry_run_count,
                    "callback_overruns": self._callback_overrun_count,
                    "slowest_callback": (
                        f"{self._slowest_callback[0]} "
                        f"{self._slowest_callback[1]:.1f}s"
                        if self._slowest_callback else None
                    ),
                    "friendly_name": "Inverter Control Health",
                },
            )
        except Exception as e:
            self.log(f"Error updating control health sensor: {e}", level="WARNING")


    def _check_soc_boundaries(self, current_soc: float) -> bool:
        """
        Check SOC boundaries and enforce safety limits.

        Safety limits are enforced regardless of manual override status -
        protecting the battery from over-discharge or over-charge is more
        important than honoring a manual mode selection.

        Args:
            current_soc: Current battery state of charge (%)

        Returns:
            True if mode was changed due to boundary violation, False otherwise
        """
        now = self.datetime()

        # Stop discharge if SOC too low
        if current_soc <= self.min_soc and self.current_mode == BatteryMode.DISCHARGE:
            self.log(f"Safety: HOLD (battery depleted at {current_soc}%)")
            entry = ScheduleEntry(
                time=self._align_to_slot(now),
                mode=BatteryMode.HOLD,
                reason="safety_min_soc",
            )
            self._handle_mode_transition(BatteryMode.HOLD)
            self._apply_mode_tracked(entry)
            # Schedule re-optimization to find charging opportunities
            if (self._last_depletion_recalc_time is None or
                    (now - self._last_depletion_recalc_time).total_seconds() > 1800):
                self._last_depletion_recalc_time = now
                self.log("Scheduling re-optimization in 120s after battery depletion")
                self.run_in(self._on_depletion_recalc, 120)
            return True

        # Stop charge if SOC full
        if current_soc >= self.max_soc and self.current_mode == BatteryMode.CHARGE:
            self.log(f"Safety: Stopping charge, SOC at maximum ({current_soc}%)")
            entry = ScheduleEntry(
                time=self._align_to_slot(now),
                mode=BatteryMode.HOLD,
                reason="safety_max_soc",
            )
            self._handle_mode_transition(BatteryMode.HOLD)
            self._apply_mode_tracked(entry)
            return True

        # Switch DISCHARGE → HOLD at max SOC when PV covers load, so surplus
        # PV exports to grid. When PV < load, keep DISCHARGE so battery
        # covers the deficit (as the schedule intended).
        if current_soc >= self.max_soc and self.current_mode == BatteryMode.DISCHARGE:
            pv_w = self._get_pv_power()
            load_w = self._get_load_power() or 0.0
            if pv_w > load_w:
                self.log(f"Safety: HOLD at max SOC ({current_soc}%) — "
                         f"PV {pv_w:.0f}W > load {load_w:.0f}W, allowing export")
                entry = ScheduleEntry(
                    time=self._align_to_slot(now),
                    mode=BatteryMode.HOLD,
                    reason="safety_max_soc_pv_export",
                )
                self._handle_mode_transition(BatteryMode.HOLD)
                self._apply_mode_tracked(entry)
                return True

        return False

    def _on_depletion_recalc(self, kwargs=None):
        """Re-optimize after battery depletion to find charging opportunities."""
        current_soc = self._get_current_soc()
        if current_soc is None:
            return
        self.log(f"Re-optimizing after battery depletion (SOC={current_soc}%)")
        self._last_recalc_trigger = "battery_depleted"
        self._last_recalc_time = self.datetime()
        self._recalculate_remaining_schedule(current_soc)

    def _get_cheapest_upcoming_prices(self, remaining_hours: List[datetime.datetime], count: int) -> List[float]:
        """
        Get the N cheapest prices from remaining hours that are currently HOLD.

        Used when charging is behind schedule to identify cheap HOLD slots that
        could be converted to CHARGE for catch-up charging.

        Args:
            remaining_hours: List of future hour timestamps to consider
            count: Number of cheapest prices to return

        Returns:
            List of up to 'count' cheapest prices from HOLD slots, sorted ascending
        """
        if count <= 0:
            return []

        prices = self.get_prices()
        price_map = {p.time: p.price for p in prices}

        # Get prices for remaining hours that are HOLD in current schedule
        hold_prices = []
        local_tz = self._get_local_timezone()

        for hour in remaining_hours:
            entry = self.schedule.get(hour)

            # Also try timezone-aware matching if direct lookup fails
            if entry is None and self.schedule:
                for schedule_hour, schedule_entry in self.schedule.items():
                    compare_schedule = schedule_hour
                    compare_hour = hour
                    if local_tz is not None:
                        if schedule_hour.tzinfo is not None:
                            compare_schedule = schedule_hour.astimezone(local_tz)
                        if hour.tzinfo is not None:
                            compare_hour = hour.astimezone(local_tz)
                    if (compare_schedule.date() == compare_hour.date() and
                        compare_schedule.hour == compare_hour.hour and
                        compare_schedule.minute == compare_hour.minute):
                        entry = schedule_entry
                        break

            if entry and entry.mode == BatteryMode.HOLD:
                # Find the price for this hour
                price = price_map.get(hour)

                # Also try timezone-aware matching for price lookup
                if price is None:
                    for price_hour, price_val in price_map.items():
                        compare_price = price_hour
                        compare_hour = hour
                        if local_tz is not None:
                            if price_hour.tzinfo is not None:
                                compare_price = price_hour.astimezone(local_tz)
                            if hour.tzinfo is not None:
                                compare_hour = hour.astimezone(local_tz)
                        if (compare_price.date() == compare_hour.date() and
                            compare_price.hour == compare_hour.hour and
                            compare_price.minute == compare_hour.minute):
                            price = price_val
                            break

                if price is not None:
                    hold_prices.append(price)

        # Return the cheapest N
        hold_prices.sort()
        return hold_prices[:count]

    def _build_soc_deviation_detector(self) -> SocDeviationDetector:
        """Build a detector from the CURRENT dynamic config.

        min_soc/max_soc are HA-backed properties, so the detector is rebuilt per
        use rather than cached. Shared by the periodic deviation check and by
        the pre-execution "SOC behind plan" test in `execute_scheduled_mode`,
        which must interpolate the same way (see `expected_soc_at`).
        """
        config = SocDeviationConfig(
            slot_minutes=self.config.slot_minutes,
            charge_rate=self.config.charge_rate,
            discharge_rate=self.config.discharge_rate,
            efficiency=self.config.efficiency,
            battery_capacity=self.config.battery_capacity,
            min_soc=self.min_soc,
            max_soc=self.max_soc,
            soc_deviation_threshold=self.config.soc_deviation_threshold,
            grid_fee=self.config.grid_fee,
            import_price_multiplier=self.config.import_price_multiplier,
            inverter_efficiency=self.config.inverter_efficiency,
            export_discharge_rate=self.config.export_discharge_rate,
            decision_log_level=self.config.decision_log_level,
        )
        return SocDeviationDetector(
            config=config,
            learning_engine=self.learning_engine,
            log_func=self.log,
        )

    def _check_soc_deviation(self, current_soc: float) -> bool:
        """
        Check if SOC deviates significantly from expected and trigger recalculation.

        Delegates to SocDeviationDetector for the actual deviation analysis.

        Args:
            current_soc: Current battery state of charge (%)

        Returns:
            True if recalculation was triggered, False otherwise
        """
        if not self._is_enabled() or self._is_override_active():
            return False

        # Prepare timing context
        now = self.datetime()
        local_tz = self._get_local_timezone()
        now = ensure_local_tz(now, local_tz)
        current_slot = self._align_to_slot(now)
        current_temp = self._get_battery_temp()

        detector = self._build_soc_deviation_detector()

        # Run deviation check
        result = detector.check_deviation(
            current_soc=current_soc,
            schedule=self.schedule,
            expected_soc_schedule=self.expected_soc_schedule,
            now=now,
            current_slot=current_slot,
            local_tz=local_tz,
            current_temp=current_temp,
            predict_load_kw=self._predict_load_kw,
            predict_pv_kw=self._predict_pv_kw,
            expected_soc_anchor=getattr(self, "_expected_soc_anchor", None),
            get_cheapest_upcoming_prices=self._get_cheapest_upcoming_prices,
            get_discharge_threshold=self._get_discharge_threshold,
        )

        # Output log messages from detector
        for msg in result.log_messages:
            self.log(msg)

        # If no recalculation needed, we're done
        if not result.should_recalculate:
            return False

        # Store trigger context for sensor exposure
        self._last_recalc_trigger = "soc_deviation"
        self._last_recalc_time = self.datetime()
        self._last_soc_deviation = result.deviation

        # Trigger recalculation with extra charge slots if needed
        self._recalculate_remaining_schedule(current_soc, extra_charge_slots=result.extra_charge_slots)
        return True

    # =========================================================================
    # VPP Control
    # =========================================================================

    def _handle_mode_transition(self, new_mode: BatteryMode):
        """
        Handle mode transition for learning engine tracking.
        Delegates to BatteryCostTracker for tracking state updates.
        """
        old_mode = self.current_mode
        current_soc = self._get_current_soc()

        # Get energy sensor readings if available
        charge_kwh = None
        discharge_kwh = None
        if self._cost_tracker.is_energy_sensor_available:
            try:
                charge_state = self.get_state(self.config.battery_charge_sensor)
                discharge_state = self.get_state(self.config.battery_discharge_sensor)
                if charge_state not in ("unknown", "unavailable", None) and \
                   discharge_state not in ("unknown", "unavailable", None):
                    charge_kwh = float(charge_state)
                    discharge_kwh = float(discharge_state)
            except (ValueError, TypeError):
                pass

        # Delegate to cost tracker for tracking state updates
        self._cost_tracker.on_mode_transition(
            old_mode=old_mode,
            new_mode=new_mode,
            current_soc=current_soc,
            charge_kwh=charge_kwh,
            discharge_kwh=discharge_kwh
        )

        self.current_mode = new_mode
        self._update_schedule_sensor()


    # =========================================================================
    # Enable/Disable and Manual Override Handling
    # =========================================================================

    def _on_enabled_change(self, entity, attribute, old, new, kwargs):
        """Handle optimizer enable/disable toggle."""
        if new == "off":
            self.log("Optimizer disabled — releasing inverter overrides")
            self._direct_control.release_control()
        elif new == "on" and old == "off":
            self.log("Optimizer re-enabled — resuming scheduled operation")
            self.execute_scheduled_mode(None)

    def on_override_change(self, entity, attribute, old, new, kwargs):
        """Handle manual override toggle"""
        if new == "on":
            self.log("Manual override activated")
            # Read and apply manual mode
            manual_mode = self.get_state(self.config.manual_mode_entity)
            self._apply_manual_mode(manual_mode)
        else:
            self.log("Manual override deactivated, resuming schedule")
            self.execute_scheduled_mode(None)

    def on_manual_mode_change(self, entity, attribute, old, new, kwargs):
        """Handle manual mode selection change"""
        if self._is_override_active():
            self._apply_manual_mode(new)

    def _apply_manual_mode(self, mode_str: str):
        """Apply the manual mode selection"""
        mode_map = {
            "Charge": BatteryMode.CHARGE,
            "Hold": BatteryMode.HOLD,
            "Discharge": BatteryMode.DISCHARGE,
            "Auto": None
        }

        mode = mode_map.get(mode_str)
        if mode is not None:
            self.log(f"Applying manual mode: {mode.name}")
            now = self.datetime()
            entry = ScheduleEntry(
                time=self._align_to_slot(now),
                mode=mode,
                reason=f"manual_{mode.name.lower()}",
            )
            self._handle_mode_transition(mode)
            self._apply_mode_tracked(entry)
        elif mode_str == "Auto":
            # Turn off override to fully resume automatic scheduling
            self.log("Manual mode set to Auto, turning off override and resuming schedule")
            try:
                self.call_service("input_boolean/turn_off",
                    entity_id=self.config.override_entity
                )
            except Exception as e:
                self.log(f"Could not turn off override: {e}", level="WARNING")
            # Execute scheduled mode immediately
            self.execute_scheduled_mode(None)

    # =========================================================================
    # SOC State Change Handler
    # =========================================================================

    @_timed_callback
    def _on_soc_change(self, entity, attribute, old, new, kwargs):
        """
        Handle SOC state changes - instant response to battery level changes.

        This event-driven handler replaces the previous polling-based approach for:
        1. Safety checks (boundary enforcement)
        2. Cost tracking and learning
        3. SOC deviation detection for schedule recalculation

        Called automatically when the SOC sensor value changes in Home Assistant.
        """
        # Skip invalid states
        if new in ("unknown", "unavailable", None):
            return
        if old in ("unknown", "unavailable", None):
            old = None

        try:
            current_soc = float(new)
        except (ValueError, TypeError):
            return

        # 1. Safety check - immediate boundary enforcement
        self._check_soc_boundaries(current_soc)

        # 2. Cost tracking & learning (always process to keep state accurate)
        self._process_soc_change_event(current_soc)

        # 3. Deviation detection for adaptive optimization
        self._check_soc_deviation(current_soc)

    # =========================================================================
    # Battery Cost Tracking
    # =========================================================================

    def _init_battery_cost(self):
        """Initialize battery cost tracking via BatteryCostTracker."""
        # Initialize the cost tracker
        self._cost_tracker.initialize()

        # Check if HA is ready
        ha_state = self.get_state("sun.sun")
        if not ha_state or ha_state in ("unknown", "unavailable"):
            self.log("HA not ready, waiting for homeassistant_start event")
            self.listen_event(self._on_ha_start, "homeassistant_start")
            return

        # HA is ready - load cost and start
        self._cost_tracker.load_from_ha()
        self._schedule_startup_optimization()

    @_timed_callback
    def _on_energy_sensor_change(self, entity, attribute, old, new, kwargs):
        """
        Handle changes to inverter energy sensors.
        Delegates to BatteryCostTracker for cost tracking and learning.
        """
        self._cost_tracker.on_energy_sensor_change(entity, old, new)

    def _schedule_startup_optimization(self):
        """Schedule the startup optimization"""
        self.log("Scheduling startup optimization")
        self.run_in(self.full_optimize, 1)

    def _on_ha_start(self, event_name, data, kwargs):
        """HA started - load state and begin optimization"""
        self._cost_tracker.load_from_ha()
        self._schedule_startup_optimization()

    def _init_learning_engine(self):
        """Initialize learning engine from persistent storage (file-based)"""
        # Track timing for learning observations
        self._charge_start_soc: Optional[float] = None
        self._charge_start_time: Optional[datetime.datetime] = None
        self._discharge_start_soc: Optional[float] = None
        self._discharge_start_time: Optional[datetime.datetime] = None

        if self.config.learning_data_file:
            try:
                with open(self.config.learning_data_file, "r", encoding="utf-8") as fh:
                    data = fh.read()
                if data and self.learning_engine.load_from_json(data):
                    summary = self.learning_engine.get_learning_summary()
                    self.log(f"Loaded learning data from file: {summary['total_observations']} observations")
                    return
            except FileNotFoundError:
                pass
            except Exception as e:
                self.log(f"Could not load learning data file: {e}", level="WARNING")

        self.log("Starting with fresh learning data")

    def _init_load_profile(self):
        """Initialize load profile from persistent storage"""
        # Prefer file-based persistence if configured
        if self.config.load_profile_file:
            try:
                with open(self.config.load_profile_file, "r", encoding="utf-8") as fh:
                    data = fh.read()
                if data and self.load_profile.load_from_json(data):
                    self.log(f"Loaded load profile from file: {self.load_profile.stats.observation_count} observations")
                    self._update_load_profile_sensors()
                    return
            except FileNotFoundError:
                pass
            except Exception as e:
                self.log(f"Could not load load profile file: {e}", level="WARNING")

        # Fallback to HA entity if configured
        if self.config.load_profile_entity:
            try:
                state = self.get_state(self.config.load_profile_entity)
                if state and state not in ("unknown", "unavailable", ""):
                    if self.load_profile.load_from_json(state):
                        self.log(f"Loaded load profile: {self.load_profile.stats.observation_count} observations")
                        self._update_load_profile_sensors()
                        return
            except Exception as e:
                self.log(f"Could not load load profile data: {e}", level="WARNING")

        self.log("Starting with fresh load profile")

    def _init_prediction_tracker(self):
        """Initialize prediction tracker from persistent storage."""
        if self.config.prediction_tracker_file:
            try:
                with open(self.config.prediction_tracker_file, "r", encoding="utf-8") as fh:
                    data = fh.read()
                if data and self.prediction_tracker.load_from_json(data):
                    self.log(
                        f"Loaded prediction tracker: "
                        f"{self.prediction_tracker.stats.total_comparisons} comparisons"
                    )
                    return
            except FileNotFoundError:
                pass
            except Exception as e:
                self.log(f"Could not load prediction tracker file: {e}", level="WARNING")

        self.log("Starting with fresh prediction tracker")

    def _init_pv_profile(self):
        """Initialize PV profile from persistent storage."""
        if self.config.pv_profile_file:
            try:
                with open(self.config.pv_profile_file, "r", encoding="utf-8") as fh:
                    data = fh.read()
                if data and self.pv_profile.load_from_json(data):
                    self.log(
                        f"Loaded PV profile: "
                        f"{self.pv_profile.stats.observation_count} observations"
                    )
                    return
            except FileNotFoundError:
                pass
            except Exception as e:
                self.log(f"Could not load PV profile file: {e}", level="WARNING")

        self.log("Starting with fresh PV profile")

    def _save_pv_profile(self):
        """Save PV profile to persistent storage."""
        if self.config.pv_profile_file:
            try:
                with open(self.config.pv_profile_file, "w", encoding="utf-8") as fh:
                    fh.write(self.pv_profile.to_json())
            except Exception as e:
                self.log(f"Could not save PV profile: {e}", level="WARNING")

    def _restore_previous_schedule_from_sensor(self):
        """
        Restore the previous schedule from sensor.battery_optimizer on startup.
        This enables continuity when restarting mid-hour during a charge or discharge slot.
        """
        try:
            attrs = self.get_state("sensor.battery_optimizer", attribute="all")
            if not attrs or "attributes" not in attrs:
                self.log("No previous schedule found in sensor")
                return

            schedule_data = attrs.get("attributes", {}).get("schedule", [])
            if not schedule_data:
                self.log("No schedule data in sensor attributes")
                return

            restored = {}
            for entry in schedule_data:
                try:
                    hour_str = entry.get("time")
                    mode_str = entry.get("mode")
                    if hour_str and mode_str:
                        hour = datetime.datetime.fromisoformat(hour_str)
                        mode = BatteryMode[mode_str]
                        restored[hour] = mode
                except (ValueError, KeyError) as e:
                    self.log(f"Could not parse schedule entry {entry}: {e}", level="DEBUG")
                    continue

            if restored:
                self._previous_schedule_from_sensor = restored
                self.log(f"Restored previous schedule from sensor: {len(restored)} entries")
            else:
                self.log("No valid entries found in previous schedule")

        except Exception as e:
            self.log(f"Could not restore previous schedule: {e}", level="WARNING")

    def _preserve_mode_on_restart(self, current_slot: datetime.datetime):
        """
        If restarting mid-hour during a CHARGE or DISCHARGE slot, preserve that mode.

        The DP algorithm sees only remaining time in the current slot and may decide HOLD
        is optimal when only minutes remain. But if this was meant to be a charging or
        discharging hour, we should continue to maintain the original intent.

        This prevents wasteful scenarios like:
        - Stopping mid-charge and losing the slot's charging opportunity
        - Holding during expensive grid hours when we should be discharging cheap battery energy
        """
        if self._previous_schedule_from_sensor is None:
            return  # Not a restart with previous schedule

        # Find the previous mode for the current slot (handle timezone variations)
        previous_mode = None
        slot_naive = current_slot.replace(tzinfo=None) if current_slot.tzinfo else current_slot

        for prev_hour, prev_mode in self._previous_schedule_from_sensor.items():
            prev_naive = prev_hour.replace(tzinfo=None) if prev_hour.tzinfo else prev_hour
            if prev_naive == slot_naive:
                previous_mode = prev_mode
                break
            # Also match by date and hour if exact match fails
            if (prev_naive.date() == slot_naive.date() and
                prev_naive.hour == slot_naive.hour and
                prev_naive.minute == slot_naive.minute):
                previous_mode = prev_mode
                break

        if previous_mode not in (BatteryMode.CHARGE, BatteryMode.DISCHARGE):
            return  # Previous slot wasn't charging or discharging, nothing to preserve

        # Find current slot in new schedule
        current_entry = None
        current_key = None
        for sched_hour, entry in self.schedule.items():
            sched_naive = sched_hour.replace(tzinfo=None) if sched_hour.tzinfo else sched_hour
            if sched_naive == slot_naive:
                current_entry = entry
                current_key = sched_hour
                break
            if (sched_naive.date() == slot_naive.date() and
                sched_naive.hour == slot_naive.hour and
                sched_naive.minute == slot_naive.minute):
                current_entry = entry
                current_key = sched_hour
                break

        if current_entry is None:
            return  # Current slot not in schedule

        if current_entry.mode == BatteryMode.HOLD:
            # Override to previous mode to maintain continuity
            mode_name = previous_mode.name
            self.log(
                f"Preserving {mode_name} mode for current slot {current_slot.strftime('%H:%M')} "
                f"(was {mode_name.lower()}ing before restart, algorithm chose HOLD due to partial slot)"
            )
            self.schedule[current_key] = ScheduleEntry(
                time=current_entry.time,
                mode=previous_mode,
                reason=f"continuing_{mode_name.lower()}_from_restart"
            )

        # Clear the previous schedule after first use (only needed for startup)
        self._previous_schedule_from_sensor = None

    def _save_load_profile(self):
        """Persist load profile to Home Assistant entity"""
        json_data = self.load_profile.to_json()

        # Prefer file-based persistence if configured
        if self.config.load_profile_file:
            try:
                with open(self.config.load_profile_file, "w", encoding="utf-8") as fh:
                    fh.write(json_data)
            except Exception as e:
                self.log(f"Could not save load profile file: {e}", level="DEBUG")

        # Optional HA entity persistence
        if self.config.load_profile_entity:
            try:
                self.call_service("input_text/set_value",
                    entity_id=self.config.load_profile_entity,
                    value=json_data
                )
            except Exception as e:
                self.log(f"Could not save load profile to {self.config.load_profile_entity}: {e}", level="DEBUG")

        self._update_load_profile_sensors()

    def _update_load_profile_sensors(self):
        """Update load profile status sensors in Home Assistant."""
        try:
            count = self.load_profile.stats.observation_count
            last_obs = self.load_profile.stats.last_observation or ""
            if self.config.load_profile_count_entity:
                self.set_state(
                    self.config.load_profile_count_entity,
                    state=str(count),
                    attributes={
                        "friendly_name": "Load Profile Observation Count",
                        "unit_of_measurement": "samples"
                    }
                )
            if self.config.load_profile_last_obs_entity:
                self.set_state(
                    self.config.load_profile_last_obs_entity,
                    state=last_obs,
                    attributes={
                        "friendly_name": "Load Profile Last Observation"
                    }
                )
        except Exception as e:
            self.log(f"Could not update load profile sensors: {e}", level="DEBUG")

    def _save_prediction_tracker(self):
        """Persist prediction tracker to file."""
        if not self.config.prediction_tracker_file:
            return
        try:
            json_data = self.prediction_tracker.to_json()
            with open(self.config.prediction_tracker_file, "w", encoding="utf-8") as fh:
                fh.write(json_data)
        except Exception as e:
            self.log(f"Could not save prediction tracker file: {e}", level="DEBUG")
        self._update_prediction_accuracy_sensor()

    def _update_prediction_accuracy_sensor(self):
        """Update prediction accuracy sensor in Home Assistant."""
        try:
            metrics = self.prediction_tracker.get_risk_metrics()
            bias = metrics["overall_bias"]
            state_str = f"{bias:.2f}x" if bias != 1.0 else "1.00x"
            attrs = {
                "friendly_name": "Battery Prediction Accuracy",
                "icon": "mdi:chart-bell-curve",
                "overall_bias": metrics["overall_bias"],
                "underestimate_pct": metrics["underestimate_pct"],
                "p90_ratio": metrics["p90_ratio"],
                "worst_slot": metrics["worst_slot"],
                "worst_slot_ratio": metrics["worst_slot_ratio"],
                "confidence": metrics["confidence"],
                "total_comparisons": self.prediction_tracker.stats.total_comparisons,
            }
            # Add schedule risk if we have a current schedule
            if self.schedule:
                risk = self.prediction_tracker.get_schedule_risk_assessment(self.schedule)
                attrs["discharge_risk"] = risk["overall_risk"]
                attrs["discharge_slot_risks"] = risk["discharge_slot_risks"]
            self.set_state("sensor.battery_prediction_accuracy", state=state_str, attributes=attrs)
        except Exception as e:
            self.log(f"Could not update prediction accuracy sensor: {e}", level="DEBUG")

    def record_load_observation(self, kwargs=None):
        """Record current house load into the statistical load profile."""
        if not self.config.load_power_sensor:
            return
        load_w = self._get_load_power()
        if load_w is None:
            return
        now = self._align_to_slot(self.datetime())

        # Record actual for just-completed slot comparison
        self.prediction_tracker.record_actual(now, load_w / 1000.0)

        # Record in load profile
        self.load_profile.record(now, load_w)

        # Record prediction for the next slot (to compare at next observation).
        # DST-safe step (see timezone_utils.slot_offset).
        next_slot = slot_offset(
            now, self.config.slot_minutes, 1, self._get_local_timezone()
        )
        predicted_kw = self._predict_load_kw(next_slot)
        self.prediction_tracker.record_prediction(next_slot, predicted_kw)

        self._save_load_profile()
        self._save_prediction_tracker()

        # Record PV observation (including 0 during daytime for cloudy day accuracy)
        pv_w = self._get_pv_power()
        hour = now.hour
        is_daytime = 6 <= hour <= 21
        if pv_w > 0 or is_daytime:
            self.pv_profile.record(now, pv_w)
            self._save_pv_profile()

    def _save_learning_data(self):
        """Persist learning data to file"""
        if not self.config.learning_data_file:
            return
        try:
            json_data = self.learning_engine.save_to_json()
            with open(self.config.learning_data_file, "w", encoding="utf-8") as fh:
                fh.write(json_data)
        except Exception as e:
            self.log(f"Could not save learning data file: {e}", level="ERROR")

    def _update_learning_sensor(self):
        """Update learning stats sensor for dashboard display"""
        try:
            summary = self.learning_engine.get_learning_summary()

            # Learning confidence: scale total observations against the engine's
            # per-bucket cap of 50 samples, capped at 100%.
            total_observations = summary["total_observations"]
            confidence_pct = round(min(100.0, total_observations / 50.0 * 100.0), 1)

            # Currently learned charge rate at present SOC and battery temperature
            # (same rate-query API the optimizer uses). Fall back to 0 without SOC.
            current_soc = self._get_current_soc()
            if current_soc is not None:
                learned_charge_rate_kw = round(
                    self.learning_engine.get_charge_rate_for_soc(
                        current_soc, self._get_battery_temp()
                    ),
                    2,
                )
            else:
                learned_charge_rate_kw = 0.0

            self.set_state(
                "sensor.battery_learning_stats",
                state=str(summary["total_observations"]),
                attributes={
                    "friendly_name": "Battery Learning Stats",
                    "unit_of_measurement": "observations",
                    "learned_efficiency": summary["learned_efficiency"],
                    "total_energy_charged_kwh": summary["total_energy_charged_kwh"],
                    "total_energy_discharged_kwh": summary["total_energy_discharged_kwh"],
                    "overall_efficiency": summary["overall_efficiency"],
                    "total_profit_eur": summary["total_profit_eur"],
                    "total_observations": summary["total_observations"],
                    "confidence_pct": confidence_pct,
                    "learned_charge_rate_kw": learned_charge_rate_kw,
                    "soc_charge_rates": summary["soc_charge_rates"],
                    "temp_aware_rates": summary.get("temp_aware_rates", {}),
                    "icon": "mdi:brain",
                }
            )
        except Exception as e:
            self.log(f"Could not update learning sensor: {e}", level="DEBUG")

    def _process_soc_change_event(self, current_soc: float):
        """
        Process SOC change for battery cost tracking and learning.
        Delegates to BatteryCostTracker.
        """
        self._cost_tracker.process_soc_change(current_soc)

    def _get_discharge_threshold(self) -> float:
        """Calculate discharge threshold based on actual battery cost."""
        return self._cost_tracker.get_discharge_threshold()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_current_soc(self) -> Optional[float]:
        """Get current battery SOC."""
        return self._sensors.get_soc(self.config.soc_sensor)

    def _get_pv_power(self) -> float:
        """Get current PV power production."""
        return self._sensors.get_power(self.config.pv_power_sensor, default=0.0)

    def _get_inverter_mode(self) -> Optional[str]:
        """Get current inverter mode from mode sensor."""
        if not self.config.inverter_mode_sensor:
            return None
        try:
            state = self.get_state(self.config.inverter_mode_sensor)
            if state and state not in ("unknown", "unavailable"):
                return str(state)
        except Exception:
            pass
        return None

    def _get_battery_temp(self) -> Optional[float]:
        """Get current battery temperature in Celsius."""
        return self._sensors.get_temperature(self.config.battery_temp_sensor)

    def _get_load_power(self) -> Optional[float]:
        """Get current household load in Watts (from configured sensor)."""
        if not self.config.load_power_sensor:
            return None
        load_w = self._sensors.get_float(self.config.load_power_sensor)
        if load_w is None:
            return None
        if load_w <= 0:
            # Use last known value or floor when sensor reports zero
            if self._last_nonzero_load_w is not None:
                return max(self._last_nonzero_load_w, self.config.load_zero_floor_w)
            return self.config.load_zero_floor_w
        self._last_nonzero_load_w = load_w
        return load_w

    def _predict_load_kw(self, dt: datetime.datetime) -> float:
        """Predict expected load (kW) for a slot using load profile with correction."""
        correction = self.prediction_tracker.get_correction_factor(dt)
        if self.load_profile:
            return self.load_profile.predict_kw(dt, self.config.load_quantile, correction)
        return (self.config.base_consumption / 1000.0) * correction

    def _predict_pv_kw(self, dt: datetime.datetime) -> float:
        """Predict PV production (kW) for a slot, corrected for forecast bias.

        The sliding bias factor (measured/forecast median over the last window)
        is applied to the CURRENT AND EVERY REMAINING slot, so a systematically
        optimistic provider forecast is corrected across the whole horizon
        rather than only in the single slot where the shortfall was observed.

        Beyond the current local day the factor is attenuated by
        ``PvBiasTracker.factor_for_slot``: today's cloud cover is weather, not a
        calibration error of tomorrow's forecast, and the daily 13:15 run would
        otherwise plan all of tomorrow on a 0.2x-clamped PV forecast.

        Past slots are left raw: they are only rendered for logging and must
        keep showing what was actually forecast at the time.

        A slot whose provider value was already capped at OBSERVED production by
        ``PvForecastService.refresh_for_shortfall`` is also left raw: that value
        carries the shortfall itself, and the bias factor is the median of the
        very same shortfall ratios. Applying both discounted the current slot
        twice (4.0 kW forecast, two measured 0.8 kW slots -> ~0.16 kW planned
        instead of 0.8), which fed the DP a phantom collapse right where it had
        the best measurement it will ever get.
        """
        raw = self._predict_pv_kw_raw(dt)
        if raw <= 0.0 or self._pv_bias_factor == 1.0:
            return raw
        now = self.datetime()
        slot = self._align_to_slot(dt)
        current_slot = self._align_to_slot(now)
        if not dt_ge(slot, current_slot, self._get_local_timezone()):
            return raw
        if self._pv_forecast_service.is_observation_capped(slot):
            return raw
        factor = self._pv_bias.factor_for_slot(self._pv_bias_factor, now, slot)
        if factor == 1.0:
            return raw
        return max(0.0, raw * factor)

    def _predict_pv_kw_raw(self, dt: datetime.datetime) -> float:
        """Predict PV production (kW) for a slot — provider value, no bias.

        Three-tier fallback:
        1. PV forecast service (Solcast / Forecast.Solar) — per-slot forecast
        2. Legacy pv_forecast_sensor — single value, current slot only
        3. PV profile — statistical history
        """
        # Tier 1: PV forecast service (Solcast / Forecast.Solar)
        # If the forecast service has data for this specific slot, trust it
        # (including 0.0 — "no sun" is an authoritative forecast)
        if self._pv_forecast_service.has_slot(dt):
            return self._pv_forecast_service.predict_kw(dt)

        # Tier 2: Legacy single-sensor forecast (current slot only)
        if self.config.pv_forecast_sensor:
            current_slot = self._align_to_slot(self.datetime())
            slot_dt = self._align_to_slot(dt)
            if slot_dt == current_slot:
                try:
                    state = self.get_state(self.config.pv_forecast_sensor)
                    if state and state not in ("unknown", "unavailable"):
                        val = float(state)
                        if val >= 0:
                            if self.config.pv_forecast_unit.lower() == "kw":
                                return val
                            return val / 1000.0
                except (ValueError, TypeError):
                    pass

        # Tier 3: Statistical PV profile
        return self.pv_profile.predict_kw(dt, self.config.pv_quantile)

    def _align_to_slot(self, dt: datetime.datetime) -> datetime.datetime:
        """Floor datetime to the start of the current time slot."""
        return align_to_slot(dt, self.config.slot_minutes, self._get_local_timezone())

    def _next_slot_time(self) -> datetime.datetime:
        """Get the next slot boundary time."""
        return next_slot_time(self.datetime(), self.config.slot_minutes, self._get_local_timezone())

    def _next_interval_time(self, interval_minutes: int) -> datetime.datetime:
        """Get the next boundary time for a given interval."""
        return next_interval_time(self.datetime(), interval_minutes, self._get_local_timezone())

    def _is_enabled(self) -> bool:
        """Check if optimizer is enabled."""
        return self._sensors.is_on(self.config.enabled_entity, default=True)

    def _is_override_active(self) -> bool:
        """Check if manual override is active."""
        return self._sensors.is_on(self.config.override_entity, default=False)

    def _get_local_timezone(self):
        """
        Get the local timezone reliably.
        Tries AppDaemon's timezone first, falls back to system local timezone.
        """
        # Try AppDaemon's timezone first
        now = self.datetime()
        if not isinstance(now, datetime.datetime):
            try:
                if hasattr(now, "done") and now.done():
                    result = now.result()
                    if isinstance(result, datetime.datetime):
                        now = result
            except Exception:
                pass
        if not isinstance(now, datetime.datetime):
            try:
                now = datetime.datetime.now().astimezone()
            except Exception:
                now = datetime.datetime.now()

        tz = now.tzinfo
        if tz is not None:
            return tz

        # Fallback: get system local timezone
        # datetime.now().astimezone() returns local time with timezone info
        try:
            return datetime.datetime.now().astimezone().tzinfo
        except Exception:
            return None

    @property
    def min_soc(self) -> float:
        """Get min SOC from HA entity or default"""
        try:
            state = self.get_state(self.config.min_soc_entity)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return self.config.default_min_soc

    @property
    def max_soc(self) -> float:
        """Get max SOC from HA entity or default"""
        try:
            state = self.get_state(self.config.max_soc_entity)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return self.config.default_max_soc

    @property
    def pv_threshold(self) -> float:
        """Get PV threshold from HA entity or default"""
        try:
            state = self.get_state(self.config.pv_threshold_entity)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return self.config.default_pv_threshold

    @property
    def battery_avg_cost(self) -> float:
        """Get the weighted average cost of energy in the battery (EUR/kWh)."""
        return self._cost_tracker.avg_cost

    @property
    def _energy_sensor_available(self) -> bool:
        """Check if inverter energy sensors are available."""
        return self._cost_tracker.is_energy_sensor_available

    @property
    def _last_soc(self) -> Optional[float]:
        """Get last recorded SOC from cost tracker."""
        return self._cost_tracker.last_soc

    def _get_load_profile_stats(self) -> List[Dict]:
        """
        Compute load profile statistics per hour for visualization.

        Returns a list of 24 dicts (one per hour) with:
        - hour: 0-23
        - avg: average consumption in W
        - min: minimum observed in W
        - max: maximum observed in W
        - p25: 25th percentile in W
        - p75: 75th percentile in W
        - samples: number of samples
        """
        from battery_optimizer_lib.load_profile import _quantile

        stats = []
        slots_per_hour = max(1, 60 // self.config.slot_minutes)

        for hour in range(24):
            # Collect samples from all slots in this hour
            all_samples = []
            for slot_offset in range(slots_per_hour):
                slot_idx = str(hour * slots_per_hour + slot_offset)
                samples = self.load_profile.stats.samples_by_slot.get(slot_idx, [])
                all_samples.extend(samples)

            if all_samples:
                stats.append({
                    "hour": hour,
                    "avg": round(sum(all_samples) / len(all_samples), 0),
                    "min": round(min(all_samples), 0),
                    "max": round(max(all_samples), 0),
                    "p25": round(_quantile(all_samples, 0.25), 0),
                    "p75": round(_quantile(all_samples, 0.75), 0),
                    "samples": len(all_samples),
                })
            else:
                # No data for this hour - use default
                stats.append({
                    "hour": hour,
                    "avg": round(self.load_profile.default_load_w, 0),
                    "min": None,
                    "max": None,
                    "p25": None,
                    "p75": None,
                    "samples": 0,
                })

        return stats

    def _update_schedule_sensor(self):
        """Update the schedule sensor in Home Assistant"""
        try:
            # Format schedule for sensor using formatter
            schedule_data = self._schedule_formatter.format_schedule_list(self.schedule)

            # Find next charge/discharge times using formatter
            now = self.datetime()
            local_tz = self._get_local_timezone()
            if now.tzinfo is not None and local_tz is not None:
                now = now.astimezone(local_tz)
            next_charge, next_discharge = self._schedule_formatter.find_next_events(
                self.schedule, now, local_tz
            )

            # Get temperature-aware rate information
            current_temp = self._get_battery_temp()
            current_soc = self._get_current_soc() or 50.0
            current_predicted_rate = self.learning_engine.get_charge_rate_for_soc(
                current_soc, current_temp
            )

            # Get temperature-aware rates summary from learning engine
            learning_summary = self.learning_engine.get_learning_summary()
            temp_aware_rates = learning_summary.get("temp_aware_rates", {})

            # Generate markdown schedule for dashboard display
            schedule_md = self._schedule_formatter.format_schedule_markdown(
                schedule=self.schedule,
                now=now,
                local_tz=local_tz,
                align_to_slot_func=self._align_to_slot,
                dp_soc_trajectory=self._last_dp_soc_trajectory,
                predict_load_kw=self._predict_load_kw,
                predict_pv_kw=self._predict_pv_kw,
            )

            # Set sensor state
            self.set_state("sensor.battery_optimizer",
                state=self.current_mode.name,
                attributes={
                    "schedule": schedule_data,
                    "current_mode": self.current_mode.name,
                    "next_charge": next_charge,
                    "next_discharge": next_discharge,
                    "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None,
                    "prices_cached": len(self._price_service.cached_prices),
                    "battery_avg_cost": round(self.battery_avg_cost, 4),
                    "discharge_threshold": round(self._get_discharge_threshold(), 4),
                    # Decision transparency attributes
                    "last_recalc_trigger": self._last_recalc_trigger,
                    "last_recalc_time": self._last_recalc_time.isoformat() if self._last_recalc_time else None,
                    "last_soc_deviation": round(self._last_soc_deviation, 1) if self._last_soc_deviation is not None else None,
                    "min_charge_slots_required": self._last_min_charge_slots,
                    "charge_slots": self._last_charge_slots,
                    # Temperature-aware charge rate attributes
                    "current_battery_temp": round(current_temp, 1) if current_temp is not None else None,
                    "current_predicted_rate": round(current_predicted_rate, 2),
                    "temp_aware_rates": temp_aware_rates,
                    # Energy measurement source
                    "energy_measurement_source": "inverter" if self._energy_sensor_available else "soc",
                    # Load profile statistics for visualization
                    "load_profile_stats": self._get_load_profile_stats(),
                    "load_profile_observations": self.load_profile.stats.observation_count,
                    "pv_profile_observations": self.pv_profile.stats.observation_count,
                    # Sliding PV forecast bias
                    "pv_bias_factor": self._pv_bias_factor,
                    "pv_bias_samples": self._pv_bias.ratio_count(self.datetime()),
                    "pv_bias_enabled": self.config.pv_bias_enabled,
                    "slot_minutes": self.config.slot_minutes,
                    # Slot outcome monitoring
                    "slot_outcomes_recent": self._outcome_tracker.get_recent_outcomes(10),
                    "prediction_accuracy": self._outcome_tracker.get_accuracy_stats(),
                    # Inverter control health (verify-after-set counters)
                    "inverter_control_health": self._direct_control.get_diagnostics(),
                    "friendly_name": "Battery Optimizer"
                }
            )

            # Separate markdown sensor — lightweight, used by dashboard template
            self.set_state("sensor.battery_optimizer_schedule_markdown",
                state=self.current_mode.name,
                attributes={
                    "md": schedule_md,
                    "friendly_name": "Battery Schedule",
                }
            )
        except Exception as e:
            self.log(f"Error updating schedule sensor: {e}", level="WARNING")
