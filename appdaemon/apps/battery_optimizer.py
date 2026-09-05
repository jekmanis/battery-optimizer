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
    # Price coverage health + bounded recovery backoff
    HorizonHealth,
    PriceHorizonConfig,
    PriceHorizonMonitor,
    is_coverage_reason,
    DirectControl,
    # App-wide callback lock (AppDaemon multi-thread dispatch)
    CallbackLock,
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
from battery_optimizer_lib.plan_validation import (
    DEFAULT_SOC_TOLERANCE,
    PlanReplay,
    replay_plan,
)
from battery_optimizer_lib.models import ScheduleModeCounts, count_schedule_modes
from battery_optimizer_lib.pv_profile import PvProfile
from battery_optimizer_lib.slot_outcome_tracker import SlotOutcomeTracker

# Bumped with behaviour changes. Logged at initialize together with the
# filesystem paths the orchestrator and the library were imported from, and
# published on sensor.battery_optimizer, so a deploy can be PROVEN to be
# running: on 2026-09-02 the add-on silently imported the previous commit out
# of a backup directory inside apps/ while SHA256 verification of apps/ passed.
APP_VERSION = "2026-09-02c"


def _code_paths() -> Tuple[str, str]:
    """(orchestrator file, library package file) actually imported."""
    import battery_optimizer_lib as _lib
    return __file__, getattr(_lib, "__file__", "?")


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

    It is ALSO the app's thread-safety chokepoint. With `total_threads` > 1 and
    `pin_app: false`, AppDaemon round-robins this app's callbacks across worker
    threads, so a schedule rebuild and a slot execution can run concurrently.
    Every registered callback carries this decorator, so running its body under
    the single app-wide `CallbackLock` restores the single-threaded semantics
    the app was written against — with exactly one deliberate escape hatch, the
    blocking `set_wit_mode` write in `_apply_mode_tracked`.

    `time.monotonic()` is sampled OUTSIDE the acquire on purpose: waiting for
    the app lock is time this callback spent hogging its AppDaemon thread and
    must show up in the overrun accounting.

    The duration is RECORDED under the same lock, from a nested try/finally
    inside the `with lock:` block (the lock is re-entrant, so nesting is free).
    `_record_callback_duration` mutates `_callback_overrun_count`,
    `_slowest_callback` and `_threads_hint_logged` — plain attributes read and
    then written (check-then-set) from every worker thread — so recording after
    the lock was released would lose overruns and could emit the one-shot
    `total_threads` hint more than once. `record_external_callback_duration`
    takes the app lock for exactly the same reason.

    `getattr(self, "_lock", None)` keeps the decorator usable by test doubles
    that never ran `initialize()`; in the app the lock always exists, because
    it is the first statement of `initialize`.
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        started = time.monotonic()
        lock = getattr(self, "_lock", None)

        def _record():
            try:
                self._record_callback_duration(
                    func.__name__, time.monotonic() - started
                )
            except Exception:
                pass

        if lock is None:
            try:
                return func(self, *args, **kwargs)
            finally:
                _record()
        with lock:
            try:
                return func(self, *args, **kwargs)
            finally:
                _record()
    return wrapper


# --- Cloud-safe hedge -------------------------------------------------------
#
# Rewriting a DP action after the fact is only sound where the DP cannot tell
# the two actions apart. Everything below exists to establish that, per slot.
# Module-level and pure so it can be reasoned about (and tested) without an
# AppDaemon instance.

_HEDGE_VALUE_EPS = 1e-9      # EUR/kWh
_HEDGE_ENERGY_EPS = 1e-6     # kWh
_HEDGE_POWER_EPS = 1e-9      # kW


def _cloud_safe_candidates(schedule, predict_load_kw, predict_pv_kw):
    """HOLD slots where the forecast has PV covering the whole house load.

    This is the first equivalence condition and the one that killed the old
    ``pv > 0`` test: with ``0 < pv < load`` the net load is positive, so
    ``soc_projection.project_slot_soc`` drains the pack under DISCHARGE and
    leaves it flat under HOLD. The two actions are then not interchangeable at
    any price — converting spends energy the DP assigned to a later slot.

    ``pv <= 0`` is excluded because there is no PV forecast to be wrong about:
    the hedge would be a plain discharge, which is the DP's decision to make.
    """
    candidates = []
    for slot_time, entry in schedule.items():
        if entry.mode != BatteryMode.HOLD:
            continue
        pv_kw = predict_pv_kw(slot_time)
        if pv_kw <= 0:
            continue
        if pv_kw + _HEDGE_POWER_EPS < predict_load_kw(slot_time):
            continue
        candidates.append(slot_time)
    return candidates


def _best_later_discharge_value(schedule):
    """Per slot: the best EUR/kWh any LATER slot of this plan discharges at.

    This is the opportunity cost of one stored DC kWh at that point in the
    horizon, taken from the plan's own numbers: ``marginal_value_eur_kwh`` is
    what the DP scored each DISCHARGE slot with (avoided import or export
    revenue, net of wear, per battery DC kWh). The horizon-end terminal value
    is NOT that quantity — with the common ``terminal_energy_value_eur_kwh: 0``
    it is zero while the plan may still be reserving the kWh for a 1.00
    EUR/kWh evening slot, which is exactly the "silently spend energy assigned
    to later slots" failure.

    A missing value counts as 0, and the running maximum starts at 0, so the
    result is never negative.

    Conservative in one direction: it ignores whether the pack would have been
    recharged before that expensive slot, so the hedge is refused on some slots
    where spending the kWh would in fact have cost nothing. Under-hedging is
    the accepted direction of error — the hedge is an optional insurance, the
    energy the plan is counting on is not.
    """
    best_after = {}
    running = 0.0
    for slot_time in sorted(schedule.keys(), key=instant_key, reverse=True):
        best_after[slot_time] = running
        entry = schedule[slot_time]
        if entry.mode == BatteryMode.DISCHARGE:
            value = getattr(entry, "marginal_value_eur_kwh", None) or 0.0
            if value > running:
                running = value
    return best_after


def _hold_sells_nothing(*, slot_time, pre_hedge_replay) -> bool:
    """Does the PRE-HEDGE plan sell any of this slot's PV surplus?

    The second equivalence condition, and the one that is invisible from the
    energy model alone: ``discharge_to_load`` pins the export limiter to 0 %
    (``direct_control.expected_registers`` — "Zero export is CRITICAL here"),
    while ``hold`` leaves it open. Any surplus the DP priced as export revenue
    (``hold_excess_pv_kwh`` in ``DPOptimizer._run_dp``) would therefore be
    curtailed instead of sold, which changes the objective the schedule was
    chosen by.

    The number comes from ``plan_validation.replay_plan`` over the pre-hedge
    schedule, whose charge-rate lookup is pinned to
    ``DPOptimizerResult.planning_temp_by_slot`` — the temperatures the plan was
    actually priced at. The previous test inferred absorption from
    ``project_schedule_trajectory``'s SOC span, which looks the rate up at the
    PROJECTOR's own evolving temperature: whenever the rate refinement falls
    back to its conservative idle profile those differ by the whole of the
    plan's warming, and a re-projection then "absorbs" surplus the DP had booked
    as revenue. An unknown slot answers False — the hedge is optional, the
    export revenue is not.
    """
    if not pre_hedge_replay:
        return False
    slot_replay = pre_hedge_replay.by_slot.get(slot_time)
    if slot_replay is None:
        return False
    return slot_replay.grid_export_ac_kwh <= _HEDGE_ENERGY_EPS


def _cloud_safe_hedge(
    schedule,
    *,
    candidates,
    config,
    prices_by_slot,
    predict_load_kw,
    predict_pv_kw,
    slot_fractions_by_slot,
    pre_hedge_replay,
    terminal_rate,
):
    """Convert the forecast-equivalent HOLD slots to DISCHARGE(to load).

    Returns the list of converted slot times.

    A converted slot is one where, under the forecast the schedule was built
    from, ``discharge_to_load`` and ``hold`` have the SAME modeled energy flow
    (PV covers the load; the surplus charges the pack in both) and the SAME
    export behaviour (nothing was going to be sold anyway). What it buys is the
    cloud case: if PV collapses, the battery — not the grid — picks up the
    load, without waiting for the shortfall re-optimization.

    Forecast equivalence is NOT equivalence under every cloud event: if PV does
    collapse, the hedge spends a kWh the plan may have assigned to a later,
    more expensive slot. So the avoided import must beat wear AND the value of
    keeping the kWh, where "keeping" is priced at the better of the horizon-end
    terminal rate and the best later DISCHARGE slot of this very plan
    (``_best_later_discharge_value``). The reactive PV-shortfall replan stays in
    place to bound what is left. It is a hedge, not an improvement on the DP's
    economics.
    """
    inv_eff = config.inverter_efficiency if config.inverter_efficiency > 0 else 1.0
    terminal_floor = max(0.0, terminal_rate or 0.0)
    best_later_value = _best_later_discharge_value(schedule)
    converted = []

    for slot_time in candidates:
        price = prices_by_slot.get(slot_time)
        if price is None:
            # No price means no way to establish the hedge is worth taking.
            continue
        buy_price = (price + config.grid_fee) * config.import_price_multiplier
        # Value of serving the load from the pack instead of the grid IF the
        # forecast is wrong, per battery DC kWh. It must beat wear AND what the
        # plan says that kWh is worth kept — at the horizon end, and in the
        # best later slot the plan already means to spend it in.
        hedge_value = buy_price * inv_eff - config.battery_wear_cost
        keep_value = max(terminal_floor, best_later_value.get(slot_time, 0.0))
        if hedge_value <= keep_value + _HEDGE_VALUE_EPS:
            continue

        sell_price = max(
            0.0, price * config.export_rate_multiplier - config.grid_export_fee
        )
        if sell_price > 0 and not _hold_sells_nothing(
            slot_time=slot_time,
            pre_hedge_replay=pre_hedge_replay,
        ):
            continue

        entry = schedule[slot_time]
        entry.mode = BatteryMode.DISCHARGE
        entry.export_rate = 0
        entry.reason += " [cloud-safe]"
        # The DP valued this slot as kept energy and, because the conversion is
        # restricted to slots whose modeled flow is unchanged, that number is
        # still the right one. Say so explicitly: a DISCHARGE row reporting a
        # bare "kept" basis reads like a stale label.
        entry.value_basis = "kept (cloud-safe)"
        converted.append(slot_time)

    return converted


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
        # FIRST statement, before anything else can be dispatched: the one
        # app-wide callback lock. `_timed_callback` looks it up with getattr,
        # so any callback AppDaemon fires while `initialize` is still running
        # (the startup SOC check below, the `run_in(full_optimize, 1)` armed
        # from `_init_battery_cost`) is already serialized against this frame.
        self._lock = CallbackLock(log_func=self.log)

        # The rest of construction runs under the lock: on a multi-threaded
        # AppDaemon the startup safety check and the deferred first optimize
        # can otherwise overlap the tail of this method and read half-built
        # state.
        with self._lock:
            self.log("Initializing Battery Optimizer")
            orchestrator_path, lib_path = _code_paths()
            self.log(
                f"Battery Optimizer version {APP_VERSION}: orchestrator={orchestrator_path} "
                f"lib={lib_path}"
            )

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

            # Wall-clock instant a grid_charge command stops being in force at the
            # inverter. A set_wit_mode override runs for slot_minutes +
            # direct_control_buffer_minutes, so it can still be charging after the
            # app has already moved on to the next slot's mode — which is how a
            # pre-dawn grid charge got booked as PV (see _grid_charge_active).
            self._grid_charge_active_until: Optional[datetime.datetime] = None
            # Slot most recently applied, and when — used to drop the redundant
            # timer execution that follows a recalculation by a few seconds.
            self._last_executed_slot: Optional[datetime.datetime] = None
            self._last_executed_monotonic: Optional[float] = None

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
                # Feeds the shared plausibility bound max(charge, discharge) * 2:
                # without it a 5.9 kW discharge installation was judged against the
                # 4.5 kW charge rate.
                nominal_discharge_rate_kw=self.config.discharge_rate,
                # Export slots run at the export discharge rate, which is
                # normally the LARGEST of the three configured powers; without
                # it a genuine max_export sample is judged against the smaller
                # load-discharge rate and thrown away.
                nominal_export_rate_kw=self.config.effective_export_discharge_rate,
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
            self._direct_control = DirectControl(
                self, self.config, verify_enabled=self.config.verify_enabled
            )

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

            # One owner for price-coverage health and recovery scheduling.
            # Built AFTER the price service and BEFORE the first thing that can
            # fetch prices, because `get_prices()` merges through it.
            self._init_price_recovery_state()

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
                # Reached only from guarded (@_timed_callback) listeners, so
                # this save also runs under the app lock - see
                # _save_learning_data.
                save_learning_data_func=self._save_learning_data,
                update_learning_sensor_func=self._update_learning_sensor,
                log_func=self.log,
                get_ambient_temp_func=lambda: self._ambient_service.predict_c(self.datetime()),
                # Attribution guards: measured charging is only PV when the sun is
                # actually producing, and never while a grid_charge command is
                # still in force at the inverter.
                get_pv_power_w_func=self._get_pv_power_optional,
                grid_charge_active_func=self._grid_charge_active,
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

            # Initialize SOC tracking state on startup (the listener only fires
            # on changes).
            startup_soc = self._get_current_soc()
            if startup_soc is not None and self._last_soc is None:
                self._process_soc_change_event(startup_soc)

            # Create sensor for exposing schedule
            self._update_schedule_sensor()
            self._update_control_health_sensor()

            # Initial safety boundary check LAST, after every sensor is
            # published. It is the only statement in `initialize` that can reach
            # `_apply_mode_tracked` -> `CallbackLock.unlocked()`, and
            # `initialize` runs at depth 1 — so that call WOULD genuinely drop
            # the app lock mid-initialize, with the listeners and timers
            # registered above already live. Today the path is unreachable
            # (`current_mode` is HOLD at startup; both boundary branches require
            # CHARGE or DISCHARGE), but "unreachable" should not be the only
            # thing keeping it safe. Running it here means a released lock
            # exposes a fully built app. Keep listener/timer registration above
            # this call, and keep this call last.
            if startup_soc is not None:
                self._check_soc_boundaries(startup_soc)

            self.log("Battery Optimizer initialized successfully")

    @_timed_callback
    def terminate(self):
        """AppDaemon teardown: make every pending timer of this instance inert.

        AppDaemon cancels an app's timers on reload, but the price retry is the
        one callback that would otherwise re-plan and re-apply a mode on behalf
        of an app that is going away. `_terminated` is the belt to
        `_cancel_price_retry`'s braces: a callback already queued by the
        scheduler finds both the flag set and its generation token cleared.
        """
        self._terminated = True
        self._cancel_price_retry()
        self.log("Battery Optimizer terminated")

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

    def record_external_callback_duration(self, name: str, seconds: float) -> None:
        """Public hook so collaborators can report their own callback duration.

        ``DirectControl`` schedules its verify/re-check callbacks through
        ``run_in`` on THIS app, so an overrun there ("Excessive time spent in
        callback DirectControl._verify_mode ... 15.8s") is invisible to
        ``_timed_callback``, which only wraps methods of this class. Routing
        those durations here keeps one overrun counter, one slowest-callback
        record and one ``total_threads`` hint for the whole app.

        It is called from another AppDaemon worker thread (DirectControl's
        verify/re-check callbacks), so it must take the app lock: the counters
        and the one-shot ``total_threads`` hint below are plain attributes with
        check-then-set semantics. DirectControl calls this from the ``finally``
        of ``_verify_mode`` AFTER releasing both of its own locks, which is what
        keeps the lock order (app lock -> _io_lock -> _state_lock) intact.
        """
        with self._lock:
            self._record_callback_duration(name, seconds)

    def _grid_charge_active(self) -> bool:
        """True while a grid_charge command is still in force at the inverter.

        The cost tracker uses this so a charge measured moments after the app
        has moved on to HOLD/DISCHARGE is still attributed to the grid: a
        set_wit_mode override runs for ``slot_minutes +
        direct_control_buffer_minutes``, and the charge counter lags it further.
        """
        until = getattr(self, "_grid_charge_active_until", None)
        if until is None:
            return False
        now, until_cmp = normalize_tz_pair(self.datetime(), until)
        return now < until_cmp

    def _shrink_grid_charge_window(self) -> None:
        """Clip an open grid-charge window to a short grace period from now.

        Used whenever the grid charge stops being what the inverter executes:
        another mode superseded it, or `release_control()` handed the inverter
        back. The energy counters lag the command, so the window is trimmed to
        `cost_grid_charge_grace_seconds` rather than closed outright — and never
        extended past where the original command was going to end anyway.
        """
        # getattr: _apply_mode_tracked is exercised by test doubles that build
        # only the health counters, and cost attribution must never be the
        # reason an inverter command is not sent.
        active_until = getattr(self, "_grid_charge_active_until", None)
        if active_until is None:
            return
        grace = self.datetime() + datetime.timedelta(
            seconds=self.config.cost_grid_charge_grace_seconds
        )
        grace, current = normalize_tz_pair(grace, active_until)
        self._grid_charge_active_until = min(grace, current)

    def _note_applied_mode(
        self, entry: ScheduleEntry, outcome: ApplyOutcome
    ) -> None:
        """Track how long the command just sent stays in force at the inverter.

        The OUTCOME decides, not the entry. A window is a claim about what the
        inverter is executing right now, so only a command that actually went
        out on the wire may open or extend one:

        * ``SENT`` / ``UNCONFIRMED_TIMEOUT`` — transmitted (the timeout is
          client-side; the request was already on the wire), so a CHARGE stamps
          a fresh `slot_minutes + direct_control_buffer_minutes` window.
        * ``SKIPPED_DUPLICATE`` — nothing was transmitted. The *original* send
          opened the window and its expiry is the true one; re-stamping here
          would slide the window forward every slot the same CHARGE repeats and
          keep attributing PV charging to the grid long after the command ended.
        * ``DRY_RUN`` — `device_id: ""`, no inverter at all. Opening a window
          would make the cost tracker book PV as grid on an installation that
          never sent a command.

        Superseding is judged the other way round: once the app has moved to a
        non-CHARGE mode the grid charge is over regardless of how that command
        fared, so the window is trimmed for every outcome that reaches here.
        """
        if entry.mode == BatteryMode.CHARGE:
            if outcome not in (
                ApplyOutcome.SENT,
                ApplyOutcome.UNCONFIRMED_TIMEOUT,
            ):
                return
            duration = (
                self.config.slot_minutes + self.config.direct_control_buffer_minutes
            )
            self._grid_charge_active_until = self.datetime() + datetime.timedelta(
                minutes=duration
            )
            return

        self._shrink_grid_charge_window()

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
                "Repeated slow callbacks: give this app more AppDaemon threads. "
                "In appdaemon.yaml set `appdaemon: total_threads: 4` AND in "
                "apps.yaml set `pin_app: false` on this app - total_threads "
                "alone leaves pin_app at its default true (AppDaemon 4.5.13), "
                "so every callback still lands on thread-0 with an 'Invalid "
                "thread ID for pinned thread' warning. set_wit_mode is a "
                "blocking service call; on one thread it stalls schedule "
                "execution, SOC listeners and PV sampling alike.",
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
        """Fetch prices from Nord Pool, keeping still-valid known intervals.

        `NordPoolPriceService` already falls back to its cache when EVERY source
        fails, but it replaces the cache wholesale on any non-empty reply. A
        service that answers with today only - which happens around the daily
        publication, and whenever the tomorrow request errors on its own - used
        to shorten a horizon that already contained tomorrow.

        `PriceHorizonMonitor.merge_with_retained` fills only the FUTURE
        intervals the fresh reply does not contain, and a fresh value always
        wins over a retained one. Nothing is invented, extrapolated, or carried
        across the retention age limit.
        """
        monitor = getattr(self, "_price_horizon", None)
        try:
            prices = self._price_service.get_prices()
        except Exception as e:
            # A raising provider (REST layer, HA restart, malformed response)
            # must degrade to "no fresh data", not abort the callback that was
            # about to notice the missing horizon and schedule a retry.
            self.log(f"Price fetch failed: {e}", level="WARNING")
            prices = []
        if monitor is None:
            return prices
        return monitor.merge_with_retained(prices, self.datetime())

    # =========================================================================
    # Price coverage health and bounded recovery
    #
    # Defect: `full_optimize` returned on missing prices without scheduling a
    # retry, `adaptive_optimize` was not a price-refresh pass, and
    # `execute_scheduled_mode` applied HOLD/no_schedule forever. A transient
    # empty reply, or a reply without tomorrow after the publication window,
    # left the optimizer on an old or absent plan until the next daily run.
    #
    # ONE owner: `PriceHorizonMonitor` decides *whether* coverage is usable and
    # *how long* to wait; this section owns the AppDaemon timer, the generation
    # guard, and the "at most one pending retry" rule. Recovery never invents a
    # price and never forces a mode: it re-fetches, and on success rebuilds
    # through the normal execution path so the enabled/override checks and the
    # command tracking still apply.
    # =========================================================================

    def _init_price_recovery_state(self) -> None:
        """Create the horizon monitor and the retry bookkeeping."""
        self._price_horizon = PriceHorizonMonitor(
            config=PriceHorizonConfig.from_main_config(self.config),
            get_timezone_func=self._get_local_timezone,
            # Local midnight needs REAL DST rules, not the fixed offset
            # `_get_local_timezone()` degrades to when `self.datetime()` is
            # naive - see `_get_region_timezone`.
            get_zone_func=self._get_region_timezone,
            log_func=self.log,
        )
        # At most ONE pending retry per app instance. `_price_retry_token` is
        # the token of the retry that is allowed to run; a callback arriving
        # with any other token belongs to a superseded generation and is inert.
        self._price_retry_token: Optional[int] = None
        self._price_retry_seq: int = 0
        self._price_retry_handle = None
        self._terminated: bool = False

    def _price_retry_pending(self) -> bool:
        return getattr(self, "_price_retry_token", None) is not None

    def _arm_price_retry(self, delay_seconds: float, reason: str) -> None:
        """Schedule the single pending price retry, if one is not already due."""
        if getattr(self, "_terminated", False):
            return
        if not self.config.price_retry_enabled:
            return
        if self._price_retry_pending():
            # Already one in flight: a retry storm is exactly what the bounded
            # backoff exists to prevent, and every failing path (full_optimize,
            # execute_scheduled_mode, adaptive_optimize) can fire in the same
            # minute.
            return
        self._price_retry_seq = getattr(self, "_price_retry_seq", 0) + 1
        token = self._price_retry_seq
        self._price_retry_token = token
        try:
            self._price_retry_handle = self.run_in(
                self._price_recovery_retry,
                delay_seconds,
                price_retry_token=token,
            )
        except Exception as e:
            # A timer we could not register must not look pending forever.
            self._price_retry_token = None
            self._price_retry_handle = None
            self.log(f"Could not schedule price retry: {e}", level="WARNING")
            return
        self.log(
            f"Price recovery scheduled in {delay_seconds:.0f}s ({reason}, "
            f"attempt {self._price_horizon.attempts})",
            level="DEBUG",
        )

    def _cancel_price_retry(self) -> None:
        """Invalidate the pending retry (generation guard + best-effort cancel).

        Clearing the token is what actually makes the callback inert: AppDaemon
        may still fire an already-queued timer, and a timer registered before a
        disable/terminate must never replace a newer valid plan.
        """
        self._price_retry_token = None
        handle = getattr(self, "_price_retry_handle", None)
        self._price_retry_handle = None
        if handle is None:
            return
        try:
            self.cancel_timer(handle)
        except Exception:
            pass

    def _review_price_horizon(
        self,
        prices: Optional[List[PricePoint]],
        now: Optional[datetime.datetime] = None,
        context: str = "",
    ) -> "HorizonHealth":
        """Judge coverage and either reset the backoff or arm the next retry."""
        now = now if now is not None else self.datetime()
        monitor = self._price_horizon
        health = monitor.evaluate(prices, now)
        if health.ok:
            pending_reason = monitor.last_failure_reason
            if self._price_retry_pending() and not is_coverage_reason(pending_reason):
                # Usable prices say nothing about an empty schedule or an
                # unreadable SOC. Disarming that retry here left the app on
                # HOLD/no_schedule until the next adaptive pass - which is
                # exactly what happened on re-enable.
                monitor.note_coverage_ok(health, now)
                return health
            was_failing = monitor.attempts > 0
            monitor.record_success(health, now)
            self._cancel_price_retry()
            if was_failing:
                monitor.clear_log_gate("horizon_incomplete")
                self.log(
                    f"Price horizon recovered ({context or 'check'}): coverage to "
                    f"{health.horizon_end}"
                )
            return health

        self._note_price_horizon_failure(health.reason, now, health, context)
        return health

    def _note_price_horizon_failure(
        self,
        reason: str,
        now: Optional[datetime.datetime] = None,
        health: Optional["HorizonHealth"] = None,
        context: str = "",
    ) -> None:
        """Record a coverage failure and arm the bounded retry.

        Also reached from `execute_scheduled_mode`'s HOLD/no_schedule branch:
        an empty current slot is a coverage failure even when nothing fetched.
        """
        if not self._is_enabled():
            # A disabled optimizer plans nothing; retrying would only produce
            # background work and log noise.
            return
        if self._price_retry_pending():
            # One armed retry already covers this failure. Counting it again
            # would inflate the backoff (and the log) purely because several
            # paths - full_optimize, the slot execution and the adaptive pass -
            # can notice the same missing horizon within the same minute.
            return
        now = now if now is not None else self.datetime()
        monitor = self._price_horizon
        delay = monitor.record_failure(reason, now, health)
        if monitor.should_log("horizon_incomplete", now):
            detail = ""
            if health is not None and health.horizon_end is not None:
                detail = (
                    f" (coverage to {health.horizon_end}, required "
                    f"{health.required_end})"
                )
            self.log(
                f"Price horizon unusable: {reason}{detail}"
                f"{' during ' + context if context else ''} - retrying in "
                f"{delay:.0f}s (attempt {monitor.attempts})",
                level="WARNING",
            )
        self._arm_price_retry(delay, reason)

    @_timed_callback
    def _price_recovery_retry(self, kwargs=None) -> None:
        """Timer callback: re-fetch prices and rebuild the plan if they arrived.

        Inert unless it is the generation that is currently allowed to run, so
        a timer queued before a disable, a terminate, or a successful recovery
        cannot resurrect a superseded plan.
        """
        token = (kwargs or {}).get("price_retry_token")
        if getattr(self, "_terminated", False):
            return
        if token is None or token != getattr(self, "_price_retry_token", None):
            self.log("Ignoring stale price recovery callback", level="DEBUG")
            return
        # Consume this generation before doing any work: whatever happens next
        # either succeeds (nothing pending) or arms the next attempt itself.
        self._price_retry_token = None
        self._price_retry_handle = None

        if not self._is_enabled():
            self.log("Optimizer disabled, abandoning price recovery", level="DEBUG")
            return

        now = self.datetime()

        # SOC FIRST, before the horizon review. Nothing can be planned without
        # it, and reviewing first meant a healthy price snapshot called
        # `record_success` (attempts -> 0) a moment before the SOC check failed
        # and recorded attempt 1 again: an unreadable battery pinned the backoff
        # at its first step and re-fetched prices under the app lock every 30 s
        # for as long as the sensor stayed away.
        current_soc = self._get_current_soc()
        if current_soc is None:
            self.log(
                "Price recovery: SOC unavailable, cannot rebuild yet",
                level="DEBUG",
            )
            self._note_price_horizon_failure("soc_unavailable", now)
            return

        prices = self.get_prices()
        health = self._review_price_horizon(prices, now, context="price_recovery")
        if not health.ok:
            return

        # Rebuild from the CURRENT SOC and time. `_recalculate_remaining_schedule`
        # projects only the remaining fraction of the active slot and finishes
        # with `execute_scheduled_mode(None)`, which is the normal execution
        # path: it re-checks enabled/override and tracks the command. During a
        # manual override it therefore refreshes the plan without sending
        # anything to the inverter.
        self.log("Price data recovered - rebuilding the remaining schedule")
        self._recalculate_remaining_schedule(current_soc)
        self._last_recalc_trigger = "price_recovery"
        self._last_recalc_time = now

    def _check_price_horizon_health(self, current_soc: Optional[float]) -> bool:
        """Periodic horizon check for `adaptive_optimize`.

        Deliberately evaluates the LAST KNOWN price snapshot rather than
        fetching: a fetch is a blocking REST call on the shared AppDaemon
        thread, and this runs every `adaptive_recalc_minutes`. When the snapshot
        is unusable the bounded retry does the fetching.

        Returns True ONLY when it rebuilt the schedule, i.e. when a second pass
        over the same callback would be duplicated work. A coverage failure is
        not such an action: `tomorrow_missing` is the routine state between
        `tomorrow_prices_hour` and tomorrow's publication, which sits squarely
        inside the PV day, so returning True there suppressed the reactive
        PV-shortfall replan for hours.
        """
        now = self.datetime()
        monitor = self._price_horizon
        health = monitor.evaluate(monitor.retained_prices, now)
        if not health.ok:
            # Note it, arm the bounded retry - and fall through, because the
            # rest of the adaptive pass works on the plan we already have.
            self._note_price_horizon_failure(
                health.reason, now, health, context="adaptive"
            )
            return False

        # Coverage is fine, so an absent entry for the current slot means the
        # PLAN ran out (or was never built), not that prices are missing.
        current_slot = self._align_to_slot(now)
        if lookup_by_time(self.schedule, current_slot, self._get_local_timezone()) is not None:
            return False
        if current_soc is None:
            return False
        if monitor.should_log("schedule_exhausted", now):
            self.log(
                f"No schedule entry for {current_slot} but prices cover the "
                f"horizon - rebuilding"
            )
        self._recalculate_remaining_schedule(current_soc)
        self._last_recalc_trigger = "horizon_extension"
        self._last_recalc_time = now
        return True

    def _price_horizon_diagnostics(self) -> Dict:
        """Coverage/retry payload published on `sensor.battery_optimizer`."""
        monitor = getattr(self, "_price_horizon", None)
        if monitor is None:
            return {}
        return monitor.diagnostics(retry_pending=self._price_retry_pending())

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

        # Cloud-safe hedge: HOLD → DISCHARGE(to load) during PV hours.
        # discharge_to_load charges from PV surplus (confirmed on Growatt WIT),
        # so it behaves identically to HOLD while PV covers the load. But when
        # clouds kill PV, the battery covers the load instead of the grid. The
        # forced forecast refresh on PV shortfall complements this: the hedge
        # bridges the gap until re-optimization, without waiting for it.
        #
        # It is restricted to slots the DP's own model cannot tell apart from
        # HOLD (see _cloud_safe_hedge). The old test — any PV at all, and an
        # import price above wear — was not an equivalence test and not an
        # economic one: with two slots priced 0.10 and 1.00, PV at half the
        # load and one slot of usable charge, it emptied the pack in the cheap
        # slot and left the expensive one to the grid, with exact forecasts and
        # no cloud anywhere. The DP had already chosen the rest of the horizon
        # on the assumption that HOLD preserved that energy.
        local_tz = self._get_local_timezone()
        slot_fractions = self._compute_slot_fractions(
            slots_sorted_by_time, current_slot, minutes_into_slot, local_tz=local_tz,
        )
        slot_fractions_by_slot = {
            canonical_slot_key(p.time): slot_fractions[i]
            for i, p in enumerate(slots_sorted_by_time)
        }
        prices_by_slot_map = {
            canonical_slot_key(p.time): p.price for p in slots_sorted_by_time
        }
        planning_temp_by_slot = getattr(result, "planning_temp_by_slot", None) or {}
        converted_slots = []
        hedge_candidates = _cloud_safe_candidates(
            schedule, self._predict_load_kw, self._predict_pv_kw
        )
        if hedge_candidates:
            # The export-equivalence test needs the export the DP actually
            # PRICED, at the temperatures it priced it at — not a re-projection
            # at whatever temperature the projector happens to reach. Same
            # replay, same rate lookup, as _validate_final_plan.
            pre_hedge_replay = self._replay_schedule(
                schedule=schedule,
                starting_soc=current_soc_for_calc,
                starting_temp=current_temp,
                current_slot=current_slot,
                minutes_into_slot=minutes_into_slot,
                prices_sorted=slots_sorted_by_time,
                planning_temp_by_slot=planning_temp_by_slot,
            )
            converted_slots = _cloud_safe_hedge(
                schedule,
                candidates=hedge_candidates,
                config=self.config,
                prices_by_slot=prices_by_slot_map,
                predict_load_kw=self._predict_load_kw,
                predict_pv_kw=self._predict_pv_kw,
                slot_fractions_by_slot=slot_fractions_by_slot,
                pre_hedge_replay=pre_hedge_replay,
                terminal_rate=result.terminal_value_eur_kwh,
            )
        cloud_safe_count = len(converted_slots)
        if cloud_safe_count > 0:
            self.log(
                f"Cloud-safe: converted {cloud_safe_count} of "
                f"{len(hedge_candidates)} eligible HOLD→DISCHARGE(to load) "
                f"slots (forecast PV covers the load, no export at risk, "
                f"avoided import beats wear and the value of keeping it)"
            )
            # Rebuild the reported trajectories from the FINAL schedule. Under
            # the restriction above they equal the HOLD plan's by construction,
            # but the published trajectory must describe the plan that will
            # execute by derivation, not by argument: schedule_formatter
            # prefers these over the expected-SOC map, and this is the very log
            # used to diagnose SOC deviations.
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
            # local_tz / slot_fractions / prices were computed for the hedge
            # above; recomputing them here would be a second source of truth
            # for the same partial-first-slot fraction.
            #
            # No per-slot rate array is passed: `project_costs` walks the
            # schedule through `project_slot_soc`, which asks the learning
            # engine for the rate at the SOC and temperature each slot actually
            # reaches. A time-indexed array never reached the column at all --
            # `_effective_charge_rate` always preferred the engine — while
            # building it cost a 132-slot projection plus a lookup per slot on
            # the planning path.
            projected_costs, _ = self._cost_tracker.project_costs(
                schedule,
                current_soc_for_calc,
                self.battery_avg_cost,
                prices_by_slot_map,
                predict_load_func=self._predict_load_kw,
                predict_pv_func=self._predict_pv_kw,
                slot_fractions_by_slot=slot_fractions_by_slot,
                # Same CHARGE/thermal model as project_schedule_trajectory, so
                # the projected-cost column cannot disagree with the SOC and
                # temperature ones.
                starting_temp=current_temp,
                planning_temp_by_slot=planning_temp_by_slot,
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
        # The temperatures the plan was BUILT with, kept so every later
        # re-projection of this schedule (the expected-SOC trajectory the
        # deviation detector runs on, the projected-cost column, the hedge's
        # export test) looks charge rates up at the same place the DP did.
        # One plan, one trajectory.
        self._last_planning_temp_by_slot = getattr(
            result, "planning_temp_by_slot", None
        ) or {}
        self._last_dp_soc_trajectory = soc_trajectory
        self._last_dp_temp_trajectory = temp_trajectory

        # Replay the FINAL action sequence — after any postprocessing — through
        # the shared physical model and check it against the trajectory about to
        # be published. Deliberately the last thing this method does, so it sees
        # the plan that executes and not an intermediate generation.
        #
        # getattr, like the terminal-value warning gate above: several test
        # doubles borrow this method without the full app surface.
        _validate = getattr(self, "_validate_final_plan", None)
        if callable(_validate):
            _replay = _validate(
                schedule=schedule,
                soc_trajectory=soc_trajectory,
                starting_soc=current_soc_for_calc,
                starting_temp=current_temp,
                current_slot=current_slot,
                minutes_into_slot=minutes_into_slot,
                prices_sorted=slots_sorted_by_time,
                planning_temp_by_slot=planning_temp_by_slot,
            )
            if _replay is not None and getattr(_replay, "corrected", False):
                # A material disagreement was resolved: publish what the shared
                # physical model says the plan does, not what the planner said.
                self._last_dp_soc_trajectory = _replay.soc_trajectory()
                self._last_dp_temp_trajectory = (
                    _replay.temp_trajectory() if current_temp is not None else {}
                )

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

    def calculate_expected_soc_schedule(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        starting_soc: float,
        starting_temp: Optional[float] = None,
        current_slot: Optional[datetime.datetime] = None,
        minutes_into_slot: float = 0.0,
        planning_temp_by_slot: Optional[
            Dict[datetime.datetime, Optional[float]]
        ] = None,
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
            planning_temp_by_slot: The temperature the PLAN was built with per
                slot (``DPOptimizerResult.planning_temp_by_slot``). Charge rates
                are looked up at those, so this trajectory and the DP's describe
                one plan. Without it the two diverged by 7.5 SOC points after
                three slots on the brief's Task 4 case — the schedule log
                printing one and this, the deviation detector's input, the
                other.

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
            planning_temp_by_slot=planning_temp_by_slot,
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
        planning_temp_by_slot: Optional[
            Dict[datetime.datetime, Optional[float]]
        ] = None,
    ) -> Tuple[
        Dict[datetime.datetime, Tuple[float, float]],
        Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]],
    ]:
        """Walk a schedule through the shared slot-SOC/thermal models.

        ``planning_temp_by_slot`` pins the CHARGE-RATE lookup to the temperature
        the plan was built with for each slot, so this trajectory describes the
        same plan the DP chose. The pack's own temperature still evolves from
        where it really is. Omitting it re-plans at whatever rate the projected
        temperature implies, which is a different plan — on the brief's Task 4
        case, where the rate refinement falls back to a conservative idle
        profile, that was a 7.5 SOC-point divergence after three slots.

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
                # Rates at the temperature the plan was BUILT with, when the
                # caller knows it. One plan, one trajectory.
                rate_lookup_temp=(
                    (planning_temp_by_slot or {}).get(hour)
                    if planning_temp_by_slot
                    else None
                ),
                learning_engine=self.learning_engine,
                temp_projector=temp_projector,
                slot_time=hour,
            )
            soc_trajectory[hour] = (current_soc, transition.soc_end)
            temp_trajectory[hour] = (current_temp, transition.temp_end)
            current_soc = transition.soc_end
            current_temp = transition.temp_end

        return soc_trajectory, temp_trajectory

    def _replay_schedule(
        self,
        *,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        starting_soc: float,
        starting_temp: Optional[float],
        current_slot: Optional[datetime.datetime],
        minutes_into_slot: float,
        prices_sorted: Optional[List[PricePoint]] = None,
        planning_temp_by_slot: Optional[
            Dict[datetime.datetime, Optional[float]]
        ] = None,
        planned_soc_by_slot: Optional[Dict[datetime.datetime, float]] = None,
        soc_tolerance: Optional[float] = None,
    ) -> Optional["PlanReplay"]:
        """Walk a schedule through the shared physical model.

        ONE construction of the replay, used by the cloud-safe hedge's export
        test and by the final-plan validation, so the two can never disagree
        about what the plan does. In particular the charge-rate lookup is pinned
        to the temperature the plan was PRICED at
        (``DPOptimizerResult.planning_temp_by_slot``): looking it up at the
        projector's own temperature answers a different question, and answering
        it made the hedge curtail export revenue the DP had booked.

        Returns None rather than raising: neither caller may break planning.
        """
        if not schedule:
            return None
        try:
            dp_config = DPOptimizerConfig.from_main_config(
                self.config, min_soc=self.min_soc, max_soc=self.max_soc
            )
            prices_by_slot = None
            if prices_sorted:
                prices_by_slot = {
                    canonical_slot_key(p.time): p.price for p in prices_sorted
                }
            return replay_plan(
                schedule=schedule,
                config=dp_config,
                starting_soc=starting_soc,
                predict_load_kw=self._predict_load_kw,
                predict_pv_kw=self._predict_pv_kw,
                charge_rate_for=(
                    lambda slot, soc, temp: self.learning_engine.get_charge_rate_for_soc(
                        soc,
                        (planning_temp_by_slot or {}).get(slot, temp),
                    )
                ),
                current_slot=current_slot,
                minutes_into_slot=minutes_into_slot,
                starting_temp=starting_temp,
                temp_projector=getattr(self, "_temp_projector", None),
                prices_by_slot=prices_by_slot,
                planned_soc_by_slot=planned_soc_by_slot,
                soc_tolerance=(
                    soc_tolerance
                    if soc_tolerance is not None
                    else DEFAULT_SOC_TOLERANCE
                ),
                slot_matcher=lambda slot, target: datetimes_match_slot(
                    slot, target, self._get_local_timezone()
                ),
            )
        except Exception as err:  # pragma: no cover - never break planning
            self.log(f"Plan replay skipped: {err}", level="DEBUG")
            return None

    def _validate_final_plan(
        self,
        *,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        soc_trajectory: Dict[datetime.datetime, Tuple[float, float]],
        starting_soc: float,
        starting_temp: Optional[float],
        current_slot: Optional[datetime.datetime],
        minutes_into_slot: float,
        prices_sorted: Optional[List[PricePoint]] = None,
        planning_temp_by_slot: Optional[Dict[datetime.datetime, Optional[float]]] = None,
    ) -> Optional["PlanReplay"]:
        """Replay the published plan and report physical disagreement.

        A schedule is chosen on a discrete SOC grid and then published: the log,
        the expected-SOC trajectory, the projected cost column and the deviation
        detector all describe it, and they all inherit the planner's arithmetic.
        Clamping the reported SOC into range hides an infeasible path entirely.
        So the FINAL action sequence is walked once more through
        ``slot_energy.simulate_slot`` in continuous energy, and the result is
        compared with the trajectory about to be published.

        Tolerance: the two evaluate the same closed-form transition on the same
        fixed forecasts and the same per-slot rates, so they should differ only
        by floating-point accumulation — bounded by roughly
        ``n_slots * 2.2e-16 * capacity``, i.e. ~1e-12 SOC % over a 132-slot
        horizon. A tenth of one DP grid step is therefore orders of magnitude
        above float noise while still being far too small to be quantization:
        anything above it is a model disagreement, not rounding. The tolerance
        is derived from the representation, not fitted to an observed error.

        A material disagreement is RESOLVED, not merely mentioned. Publishing
        numbers the shared model contradicts leaves the schedule log, the
        deviation detector and the projected-cost column all describing a plan
        that cannot happen — and the deviation detector then chases the
        difference. So the replay's own trajectory replaces the planner's
        (``PlanReplay.corrected``), and the correction is verified by replaying
        the corrected trajectory once more. Only if THAT still disagrees — which
        cannot happen for the shared model's own output, so it means something
        upstream is not deterministic — is an ERROR raised.

        The actions themselves are left alone. Re-solving the DP here would
        re-enter planning from inside its own tail and would have to re-apply
        the cloud-safe hedge; the value fields are per-DC-kWh rates and do not
        depend on the SOC path, so replacing the trajectory is a complete and
        consistent correction of what gets published.
        """
        if not schedule:
            return None
        planned_end_soc = {
            slot: pair[1] for slot, pair in (soc_trajectory or {}).items()
        }
        tolerance = max(1e-6, 0.1 * self.config.soc_step_percent)
        replay = self._replay_schedule(
            schedule=schedule,
            starting_soc=starting_soc,
            starting_temp=starting_temp,
            current_slot=current_slot,
            minutes_into_slot=minutes_into_slot,
            prices_sorted=prices_sorted,
            planning_temp_by_slot=planning_temp_by_slot,
            planned_soc_by_slot=planned_end_soc or None,
            soc_tolerance=tolerance,
        )
        if replay is None:
            return None

        self._last_plan_replay = replay
        if replay.conservation_violations:
            self.log(
                "Plan replay: the published plan discharges more than it holds — "
                + replay.conservation_violations[0]
                + f" ({len(replay.conservation_violations)} slot(s))",
                level="ERROR",
            )
        if replay.trajectory_disagreements:
            # RESOLVE it: the shared model's own trajectory replaces the
            # planner's, and the correction is verified by replaying it.
            corrected_soc = replay.soc_trajectory()
            recheck = self._replay_schedule(
                schedule=schedule,
                starting_soc=starting_soc,
                starting_temp=starting_temp,
                current_slot=current_slot,
                minutes_into_slot=minutes_into_slot,
                prices_sorted=prices_sorted,
                planning_temp_by_slot=planning_temp_by_slot,
                planned_soc_by_slot={
                    slot: pair[1] for slot, pair in corrected_soc.items()
                },
                soc_tolerance=tolerance,
            )
            still_wrong = recheck is None or recheck.trajectory_disagreements
            replay.corrected = not still_wrong
            self.log(
                "Plan replay: published SOC trajectory disagreed with the shared "
                f"physical model by more than {tolerance:.4f}% — "
                + replay.trajectory_disagreements[0]
                + f" ({len(replay.trajectory_disagreements)} slot(s)); "
                + (
                    "publishing the replayed trajectory instead"
                    if replay.corrected
                    else "the replayed trajectory does not reproduce itself either"
                ),
                level="WARNING" if replay.corrected else "ERROR",
            )
        if (
            replay.total_unmet_battery_ac_kwh > 1e-6
            and getattr(self.config, "decision_log_level", 0) >= 1
        ):
            self.log(
                "Plan replay: "
                f"{replay.total_unmet_battery_ac_kwh:.3f} kWh of planned battery "
                "service runs the pack dry and is covered by the grid "
                f"(battery serves {replay.total_battery_ac_served_kwh:.2f} kWh, "
                f"imports {replay.total_grid_import_ac_kwh:.2f} kWh)"
            )
        return replay

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

        # Fetch prices, then judge COVERAGE - not just "did the call return
        # something". A reply with today only, after the tomorrow publication
        # window, is an incomplete horizon and arms the bounded retry here
        # rather than waiting for the next daily run.
        prices = self.get_prices()
        self._review_price_horizon(prices, now, context="full_optimize")
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
            planning_temp_by_slot=getattr(self, "_last_planning_temp_by_slot", None),
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

        # Apply current slot's mode immediately. This is a NESTED callback
        # (depth 2 under the app lock), so `_apply_mode_tracked`'s unlocked
        # region will NOT release: the inverter write runs with this rebuild's
        # lock held. That is deliberate - releasing here would expose the
        # just-rebuilt schedule to another worker thread. Do NOT defer this via
        # run_in: run_in passes a kwargs dict, which flips the
        # `kwargs is not None` branch of the execute dedupe and makes the
        # deferred apply skippable.
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

        # Horizon health on the periodic pass. Without it, an absent or
        # exhausted schedule could only recover through an unrelated SOC/PV
        # event or the next daily optimization. This reads the last known price
        # snapshot - it never adds a blocking fetch to the periodic path; the
        # bounded retry does the fetching when the snapshot is unusable.
        if self._check_price_horizon_health(current_soc):
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
            planning_temp_by_slot=getattr(self, "_last_planning_temp_by_slot", None),
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

        # Apply updated mode immediately. Nested (depth 2): the unlocked region
        # in `_apply_mode_tracked` correctly keeps the lock, because this frame
        # has just replaced the future half of `self.schedule`.
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

        # Drop the redundant TIMER execution that trails a recalculation.
        # `_recalculate_remaining_schedule` applies the current slot itself, and
        # the quarter-hour timer then re-applied the identical entry seconds
        # later (production 07:30:06 -> 07:30:12, 08:30:06 -> 08:30:15).
        # DirectControl suppressed the duplicate, but the call still costs a
        # blocking set_wit_mode round trip on the single AppDaemon thread, plus
        # a second record_slot_start/record_slot_end pair for one slot.
        # Only the AppDaemon timer is deduped: it is the only caller that
        # passes a kwargs dict. Every internal call (recalculation, override
        # resume, manual "Auto", enable) passes None and always executes, so a
        # genuine mode change is never suppressed.
        if kwargs is not None and not force and self._recently_executed(current_slot):
            self.log(
                f"Skipping timer execution for {current_slot}: already applied "
                f"{self._seconds_since_execution():.0f}s ago",
                level="DEBUG",
            )
            return

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
            # HOLD stays the safe answer while there is no plan - recovery must
            # never invent a cheap price and force charging. But an empty
            # current slot IS a coverage failure, so ask for prices again on a
            # bounded backoff instead of holding until the next daily run.
            entry = ScheduleEntry(
                time=current_slot,
                mode=BatteryMode.HOLD,
                reason="no_schedule",
            )
            self._note_price_horizon_failure(
                "no_schedule", now, context="execute_scheduled_mode"
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

        # A DISCHARGE entry must never be (re-)applied at min SOC. The plan can
        # legitimately schedule DISCHARGE there (a cloud-safe slot where PV is
        # expected to cover the load), but after a depletion the safety HOLD
        # from _check_soc_boundaries is the state that must stand: on
        # 2026-09-02 the depletion re-optimization re-executed the old 06:45
        # DISCHARGE entry at 06:54:05 with SOC at 10.0 %, overriding the safety
        # HOLD sent three minutes earlier without re-checking depletion. The
        # SOC listener could not correct it either — at min SOC the SOC stops
        # changing, so no further boundary check fires. HOLD accepts PV surplus
        # exactly like discharge_to_load does, so nothing is lost.
        if (entry.mode == BatteryMode.DISCHARGE
                and current_soc is not None
                and current_soc <= self.min_soc):
            self.log(
                f"Overriding DISCHARGE->HOLD at min SOC ({current_soc}%) "
                f"- battery depleted, nothing to discharge"
            )
            entry = ScheduleEntry(
                time=entry.time,
                mode=BatteryMode.HOLD,
                reason="safety_min_soc",
            )

        # Track mode transition for cost tracking / learning
        self._handle_mode_transition(entry.mode)

        # Send command to inverter
        applied = self._apply_mode_tracked(entry)

        # Only a command the inverter may be acting on suppresses the timer
        # re-execution; a CONFIRMED failure must stay retryable.
        if applied:
            self._last_executed_slot = current_slot
            self._last_executed_monotonic = time.monotonic()

    def _seconds_since_execution(self) -> float:
        """Seconds since the last successfully applied slot (inf if none)."""
        if self._last_executed_monotonic is None:
            return float("inf")
        return time.monotonic() - self._last_executed_monotonic

    def _recently_executed(self, current_slot: datetime.datetime) -> bool:
        """True when *current_slot* was already applied within the dedupe window.

        Uses ``time.monotonic()`` rather than ``self.datetime()``: this is a
        "how long ago did we do that" question and must not be affected by a
        DST step or a clock correction.
        """
        window = getattr(self.config, "execute_dedupe_seconds", 0)
        if window <= 0 or self._last_executed_slot is None:
            return False
        if not datetimes_match_slot(
            self._last_executed_slot, current_slot, self._get_local_timezone()
        ):
            return False
        return self._seconds_since_execution() < window

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
        # THE ONE UNLOCKED REGION IN THIS APP. `set_wit_mode` is a synchronous
        # modbus write that legitimately takes up to ~15 s; holding the app lock
        # across it would serialize every other callback behind one inverter
        # write, which is exactly the single-thread stall multi-threading is
        # meant to fix. Only this one expression runs unlocked: `entry` is a
        # local, `current_mode` was already committed by
        # `_handle_mode_transition` before we got here, DirectControl guards its
        # own state, and everything below re-acquires the lock before touching
        # app state.
        #
        # At depth >= 2 (a nested call from `full_optimize` /
        # `_recalculate_remaining_schedule` -> `execute_scheduled_mode`)
        # `unlocked()` deliberately does NOT release: the outer frame is
        # mid-rebuild of `self.schedule`, so the write runs under the lock -
        # today's behaviour, and correct.
        with self._lock.unlocked():
            outcome = self._direct_control.apply_mode_with_outcome(entry)

        if outcome is not ApplyOutcome.FAILED:
            # The outcome, not the entry, decides what happens to the
            # grid-charge window: a duplicate must not re-extend it and a dry
            # run must not open one. See _note_applied_mode.
            self._note_applied_mode(entry, outcome)

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
            # Schedule re-optimization to find charging opportunities — but
            # only when the depletion was NOT the plan. An "(until depleted)"
            # EXPORT/DISCHARGE slot ends AT min_soc by design: on 2026-09-02 the
            # DP planned "EXPORT (until depleted) -> 10.0%" at 06:45, the
            # battery hit 10.0 % as planned at 06:51, and the safety net then
            # paid for a full 17 s re-optimization of a state the schedule had
            # asked for. The safety HOLD still fires (the inverter's discharge
            # cutoff is the real protection); only the recalculation is skipped.
            if self._depletion_was_planned(now):
                self.log(
                    "Depletion at min SOC was planned for this slot "
                    "(scheduled end SOC is min_soc) - applying safety HOLD "
                    "without re-optimizing",
                    level="DEBUG",
                )
                return True
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

    def _depletion_was_planned(self, now: datetime.datetime) -> bool:
        """True when the executing slot was already scheduled to end at min SOC.

        The expected-SOC trajectory records the SOC at the START of each slot,
        so the planned END of the current slot is the value stored for the NEXT
        slot. When that is min_soc (within
        ``planned_depletion_margin_percent``), reaching min_soc is the plan
        executing correctly, not a deviation worth replanning for.

        A low planned END is not sufficient on its own. The trajectory can
        already be sitting just above min_soc while the plan expects the slot to
        *rise* — a cloud-safe DISCHARGE slot whose PV surplus charges the pack,
        e.g. planned start 10.5 %, planned end 11.0 % with a 1.0 % margin. When
        the PV then fails to appear and the battery really does hit min_soc, the
        forecast the schedule rests on has been invalidated and the
        re-optimization must run. So the plan must also have expected a
        non-rising slot: ``planned_end <= planned_start``.

        The comparison is ``<=``, not ``<``: a second consecutive
        "(until depleted)" slot is flat at min_soc (10 % -> 10 %), which is
        planned depletion and must NOT trigger a recalculation.

        The caller (`_check_soc_boundaries`) already requires the running mode
        to be DISCHARGE, and the cloud-safe conversion rewrites `entry.mode` to
        DISCHARGE, so re-checking the schedule entry's mode here would add
        nothing.

        Returns False whenever the answer is unknown (no trajectory, no entry
        for the current slot, or the horizon ends here), preserving the previous
        "always re-optimize" behaviour for genuinely unexplained depletion.
        """
        if not self.expected_soc_schedule:
            return False
        local_tz = self._get_local_timezone()
        current_slot = self._align_to_slot(now)
        next_slot = slot_offset(current_slot, self.config.slot_minutes, 1, local_tz)
        planned_end = lookup_by_time(self.expected_soc_schedule, next_slot, local_tz)
        if planned_end is None:
            return False
        planned_start = lookup_by_time(
            self.expected_soc_schedule, current_slot, local_tz
        )
        if planned_start is None:
            return False
        if planned_end > planned_start:
            return False
        return planned_end <= self.min_soc + self.config.planned_depletion_margin_percent

    @_timed_callback
    def _on_depletion_recalc(self, kwargs=None):
        """Re-optimize after battery depletion to find charging opportunities.

        Decorated: this is a ``run_in`` callback like any other, and its 17.0 s
        overrun on 2026-09-02 was invisible to the app's own instrumentation —
        only AppDaemon's generic "Excessive time spent in callback" line
        recorded it.
        """
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

    @_timed_callback
    def _on_enabled_change(self, entity, attribute, old, new, kwargs):
        """Handle optimizer enable/disable toggle.

        Disabling releases the inverter overrides, and closes the grid-charge
        cost-attribution window only if that release was actually accepted.
        """
        if new == "off":
            self.log("Optimizer disabled — releasing inverter overrides")
            # Second (and last) blocking inverter call in this file; same rule
            # as `_apply_mode_tracked` - drop the app lock around it only.
            with self._lock.unlocked():
                released = self._direct_control.release_control()
            # Handing the inverter back to passthrough ends any grid charge just
            # as a superseding mode does. Without this the window kept running
            # to its full slot+buffer expiry with the optimizer disabled, so a
            # sunny disabled-app charge was booked at the grid price.
            #
            # Only on a release the inverter accepted, though. `release_control`
            # returns False on a CONFIRMED service failure (it has already
            # logged the ERROR), which means the grid_charge override is still
            # executing at the inverter until its normal expiry. Shrinking the
            # window there would misattribute the charging that follows as PV /
            # uncommanded grid energy, so the window keeps its original expiry.
            if released:
                self._shrink_grid_charge_window()
            # A disabled optimizer must not keep waking up to fetch prices, and
            # a retry armed before the toggle must not plan on its behalf.
            self._cancel_price_retry()
        elif new == "on" and old == "off":
            self.log("Optimizer re-enabled — resuming scheduled operation")
            # Start recovery from a clean backoff: the horizon may have gone
            # stale while disabled, and the previous attempt count says nothing
            # about the price service now.
            self._price_horizon.reset_backoff()
            # Review BEFORE executing. The other order armed a `no_schedule`
            # retry in `execute_scheduled_mode` and then cancelled it with a
            # healthy price verdict, so a re-enabled app with an empty schedule
            # sat on HOLD until the next adaptive pass.
            self._review_price_horizon(
                self._price_horizon.retained_prices, context="enable"
            )
            self.execute_scheduled_mode(None)

    @_timed_callback
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

    @_timed_callback
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

    @_timed_callback
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
        # Runs under the app lock (every caller is a @_timed_callback frame):
        # `open(..., "w")` truncates, so two concurrent saves would leave a
        # half-written file. Never call this from inside an unlocked region.
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
        # Runs under the app lock (every caller is a @_timed_callback frame):
        # `open(..., "w")` truncates, so two concurrent saves would leave a
        # half-written file. Never call this from inside an unlocked region.
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
        # Runs under the app lock (every caller is a @_timed_callback frame):
        # `open(..., "w")` truncates, so two concurrent saves would leave a
        # half-written file. Never call this from inside an unlocked region.
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

    @_timed_callback
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
        # Runs under the app lock (every caller is a @_timed_callback frame):
        # `open(..., "w")` truncates, so two concurrent saves would leave a
        # half-written file. Never call this from inside an unlocked region.
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

    def _get_region_timezone(self):
        """A timezone with real DST rules, for local-midnight arithmetic.

        `_get_local_timezone()` returns `self.datetime().tzinfo` when AppDaemon
        hands over an aware datetime and otherwise falls back to
        `datetime.now().astimezone().tzinfo` - a FIXED `datetime.timezone`
        carrying today's offset. That is perfectly good for ordering instants
        and aligning slots, and wrong for "the local midnight after next" on the
        two DST days: `combine(2024-04-01, 00:00, +02:00)` is an hour later than
        Europe/Riga's actual midnight that day, so a complete price horizon
        reads as `tomorrow_missing` all afternoon.

        AppDaemon's own `get_timezone()` reports the configured zone, normally
        as a name. Anything unusable falls back to `_get_local_timezone()`; the
        monitor then says once that the boundary has no DST rules.
        """
        zone = None
        getter = getattr(self, "get_timezone", None)
        if callable(getter):
            try:
                zone = getter()
            except Exception as e:
                self.log(f"get_timezone() failed: {e}", level="DEBUG")
                zone = None
        if isinstance(zone, datetime.tzinfo):
            return zone
        if isinstance(zone, str) and zone.strip():
            try:
                from zoneinfo import ZoneInfo
                return ZoneInfo(zone.strip())
            except Exception as e:
                self.log(
                    f"Cannot resolve timezone '{zone}': {e} - falling back to the "
                    f"local offset for horizon boundaries",
                    level="WARNING",
                )
        return self._get_local_timezone()

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
                    # Coverage health: last usable horizon end, why it is not
                    # usable, and whether a bounded retry is pending. A nonempty
                    # `prices_cached` alone never proved a usable horizon.
                    "price_horizon": self._price_horizon_diagnostics(),
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
                    # Deploy proof: which code is actually running (see APP_VERSION)
                    "app_version": APP_VERSION,
                    "code_paths": dict(zip(("orchestrator", "lib"), _code_paths())),
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
