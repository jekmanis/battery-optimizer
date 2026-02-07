"""
Battery Charge/Discharge Planning System for Growatt WIT Inverter

Uses Nord Pool price forecasts to schedule optimal battery charge/hold/discharge
periods. Implements adaptive re-optimization based on actual SOC and PV production.

Author: AppDaemon Battery Optimizer
"""

import appdaemon.plugins.hass.hassapi as hass
import datetime
import math
import traceback
import json
from typing import Dict, List, Optional, Tuple

# Import from the battery_optimizer_lib package
from battery_optimizer_lib import (
    # Config
    BatteryOptimizerConfig,
    # Models
    BatteryMode,
    PricePoint,
    ScheduleEntry,
    TouPeriod,
    LearningStats,
    LoadProfileStats,
    BatteryLearningEngine,
    LoadProfile,
    NordPoolPriceService,
    TouSyncManager,
    # DP Optimizer
    DPOptimizer,
    DPOptimizerConfig,
    DPOptimizerResult,
    # Timezone utilities
    normalize_tz_pair,
    dt_ge,
    ensure_local_tz,
    align_to_slot,
    next_slot_time,
    next_interval_time,
    # HA helpers
    SensorReader,
    # Cost tracker
    BatteryCostTracker,
    BatteryCostConfig,
    # Charge rate utilities
    compute_charge_rates_per_slot,
    # SOC deviation detection
    SocDeviationDetector,
    SocDeviationConfig,
    # Schedule formatting
    ScheduleFormatter,
    ScheduleFormatterConfig,
)


# Note: Data models (BatteryMode, PricePoint, ScheduleEntry, TouPeriod, LearningStats,
# LoadProfileStats) and helper classes (BatteryLearningEngine, LoadProfile, NordPoolPriceService,
# TouSyncManager) have been moved to the battery_optimizer/ package for better organization.
# See battery_optimizer/__init__.py for the full list of exports.




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
        self._last_nonzero_load_w: Optional[float] = None
        self._previous_schedule_from_sensor: Optional[Dict[datetime.datetime, BatteryMode]] = None

        # Decision context tracking (for transparency logging and sensor exposure)
        self._last_recalc_trigger: str = "startup"  # "startup", "daily_13:15", "soc_deviation", "manual"
        self._last_recalc_time: Optional[datetime.datetime] = None
        self._last_soc_deviation: Optional[float] = None  # Deviation that triggered recalculation
        self._last_min_charge_slots: int = 0  # Min charge slots from last calculation
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

        # Load profile for probabilistic scheduling
        self.load_profile = LoadProfile(
            slot_minutes=self.config.slot_minutes,
            default_load_w=self.config.base_consumption,
            max_samples=self.config.load_profile_max_samples,
            min_samples=self.config.load_profile_min_samples,
            log_func=self.log,
        )
        self._init_load_profile()

        # TOU sync manager for inverter schedule sync
        self._tou_sync_manager = TouSyncManager(
            device_id=self.config.device_id,
            slot_minutes=self.config.slot_minutes,
            ha_url=self.config.ha_url,
            ha_token=self.config.ha_token,
            call_service_func=self.call_service,
            get_datetime_func=self.datetime,
            get_timezone_func=self._get_local_timezone,
            sleep_func=self.sleep,
            create_task_func=self.create_task,
            log_func=self.log,
            get_schedule_func=lambda: self.schedule,  # Fresh schedule at execution time
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

        # Schedule formatter for logging and sensor updates
        self._schedule_formatter = ScheduleFormatter(
            config=ScheduleFormatterConfig(
                slot_minutes=self.config.slot_minutes,
                slot_hours=self.config.slot_hours,
                battery_capacity=self.config.battery_capacity,
                charge_rate=self.config.charge_rate,
                discharge_rate=self.config.discharge_rate,
                efficiency=self.config.efficiency,
                battery_wear_cost=self.config.battery_wear_cost,
                decision_log_level=self.config.decision_log_level,
            ),
            log_func=self.log,
            learning_engine=self.learning_engine,
        )

        # Battery cost tracker (must be after learning engine and price service)
        self._cost_tracker = BatteryCostTracker(
            config=BatteryCostConfig(
                battery_cost_entity=self.config.battery_cost_entity,
                battery_charge_sensor=self.config.battery_charge_sensor,
                battery_discharge_sensor=self.config.battery_discharge_sensor,
                use_inverter_energy_sensors=self.config.use_inverter_energy_sensors,
                battery_capacity=self.config.battery_capacity,
                efficiency=self.config.efficiency,
                slot_minutes=self.config.slot_minutes,
                charge_rate=self.config.charge_rate,
                discharge_rate=self.config.discharge_rate,
                grid_fee=self.config.grid_fee,
                battery_wear_cost=self.config.battery_wear_cost,
            ),
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

        # Record load observations (can be more frequent than schedule slots)
        self.run_every(
            self.record_load_observation,
            self._next_interval_time(self.config.load_observation_minutes),
            self.config.load_observation_minutes * 60
        )

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

        self.log("Battery Optimizer initialized successfully")

    def _load_config(self):
        """Load configuration from apps.yaml into typed config object."""
        self.config = BatteryOptimizerConfig.from_args(self.args, log_func=self.log)
        self.config.log_summary(self.log)

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
        _ = charge_hours_needed  # kept for compatibility with previous interface
        if not prices:
            return {}

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
        slots_sorted_by_time = sorted(future_prices, key=lambda p: p.time)
        n_slots = len(slots_sorted_by_time)
        min_charge_slots = max(0, int(charge_hours_needed))
        if min_charge_slots > n_slots:
            min_charge_slots = n_slots

        # Prepare inputs for optimizer
        current_soc_for_calc = current_soc if current_soc is not None else 50.0
        minutes_into_slot = max(0.0, (now - current_slot).total_seconds() / 60.0)
        current_temp = self._get_battery_temp()

        # Create optimizer with fresh config (min_soc/max_soc are dynamic properties)
        optimizer = DPOptimizer(
            config=DPOptimizerConfig(
                battery_capacity=self.config.battery_capacity,
                min_soc=self.min_soc,
                max_soc=self.max_soc,
                efficiency=self.config.efficiency,
                discharge_rate=self.config.discharge_rate,
                slot_minutes=self.config.slot_minutes,
                soc_step_percent=self.config.soc_step_percent,
                grid_fee=self.config.grid_fee,
                battery_wear_cost=self.config.battery_wear_cost,
            ),
            load_predictor=self._predict_load_kw,
            charge_rate_predictor=self.learning_engine.get_charge_rate_for_soc,
            temp_after_charge_predictor=self.learning_engine.predict_temp_after_duration,
            temp_after_idle_predictor=self.learning_engine.predict_temp_after_idle,
            log_fn=self.log,
            decision_log_level=self.config.decision_log_level,
        )

        # Run optimization
        result = optimizer.optimize(
            prices=future_prices,
            current_slot=current_slot,
            current_soc=current_soc_for_calc,
            current_temp=current_temp,
            minutes_into_slot=minutes_into_slot,
            min_charge_slots_hint=min_charge_slots,
        )

        schedule = result.schedule

        # Project costs for sensor exposure
        has_charge_slots = any(e.mode == BatteryMode.CHARGE for e in schedule.values())
        if has_charge_slots:
            prices_by_slot = {p.time: p.price for p in slots_sorted_by_time}
            # Re-compute charge rates per slot for cost projection
            slot_fractions = self._compute_slot_fractions(slots_sorted_by_time, current_slot, minutes_into_slot)
            charge_rates_per_slot = self._compute_charge_rates_per_slot(
                slots_sorted_by_time, slot_fractions, current_soc_for_calc, current_temp
            )
            slot_charge_rates_by_slot = {
                p.time: charge_rates_per_slot[i] for i, p in enumerate(slots_sorted_by_time)
            }
            slot_fractions_by_slot = {
                p.time: slot_fractions[i] for i, p in enumerate(slots_sorted_by_time)
            }
            projected_costs, _ = self._cost_tracker.project_costs(
                schedule,
                current_soc_for_calc,
                self.battery_avg_cost,
                prices_by_slot,
                predict_load_func=self._predict_load_kw,
                charge_rates_by_slot=slot_charge_rates_by_slot,
                slot_fractions_by_slot=slot_fractions_by_slot,
            )
            self._last_projected_costs = projected_costs
        else:
            self._last_projected_costs = {}

        self.log(f"Schedule generated: {result.charge_count} charge, {result.discharge_count} discharge, "
                 f"{result.hold_count} hold slots (slot={self.config.slot_minutes}min, load_quantile={self.config.load_quantile:.2f}, "
                 f"min_charge_slots={min_charge_slots})")

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

        # Store trajectories for use in _log_schedule
        self._last_dp_soc_trajectory = result.soc_trajectory
        self._last_dp_temp_trajectory = result.temp_trajectory

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
        def prices_contains_slot(prices_list, slot):
            slot_naive = slot.replace(tzinfo=None) if slot.tzinfo else slot
            for p in prices_list:
                p_naive = p.time.replace(tzinfo=None) if p.time.tzinfo else p.time
                if p_naive == slot_naive:
                    return True
            return False

        if prices_contains_slot(future_prices, current_slot):
            return future_prices

        # Current slot missing - try yesterday's Nord Pool data (timezone shift around midnight)
        tz = self._get_local_timezone()
        yesterday = current_slot.date() - datetime.timedelta(days=1)
        yesterday_prices = self._price_service.get_prices_for_date(yesterday, tz)

        def find_slot_price(prices_list, slot):
            slot_naive = slot.replace(tzinfo=None) if slot.tzinfo else slot
            for p in prices_list:
                p_naive = p.time.replace(tzinfo=None) if p.time.tzinfo else p.time
                if p_naive == slot_naive:
                    return p
            return None

        slot_price_point = find_slot_price(yesterday_prices, current_slot)
        if slot_price_point is not None:
            synth_price = slot_price_point.price
            self.log(
                f"Added missing current slot {current_slot} using yesterday's price "
                f"{synth_price:.4f} EUR/kWh from {slot_price_point.time}"
            )
        else:
            # If still missing, synthesize using most recent past price if available
            current_slot_naive = current_slot.replace(tzinfo=None) if current_slot.tzinfo else current_slot
            prev_price_point = None
            for p in all_prices:
                p_hour = p.time
                p_naive = p_hour.replace(tzinfo=None) if p_hour.tzinfo else p_hour
                if p_naive <= current_slot_naive:
                    if prev_price_point is None:
                        prev_price_point = p
                    else:
                        prev_naive = prev_price_point.time.replace(tzinfo=None) if prev_price_point.time.tzinfo else prev_price_point.time
                        if p_naive > prev_naive:
                            prev_price_point = p

            if prev_price_point is not None:
                synth_price = prev_price_point.price
                self.log(
                    f"Added missing current slot {current_slot} using previous price "
                    f"{synth_price:.4f} EUR/kWh from {prev_price_point.time}"
                )
            else:
                # Fallback: use first available future price
                synth_price = min(future_prices, key=lambda p: p.time).price
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
    ) -> List[float]:
        """Compute fraction of each slot that is usable (partial first slot)."""
        n_slots = len(slots_sorted_by_time)
        first_fraction = min(1.0, max(0.0, (self.config.slot_minutes - minutes_into_slot) / max(1, self.config.slot_minutes)))
        slot_fractions = [1.0] * n_slots

        for i, p in enumerate(slots_sorted_by_time):
            p_hour = p.time
            compare_current = current_slot
            if p_hour.tzinfo is not None and compare_current.tzinfo is None:
                p_hour = p_hour.replace(tzinfo=None)
            elif p_hour.tzinfo is None and compare_current.tzinfo is not None:
                compare_current = compare_current.replace(tzinfo=None)
            if p_hour == compare_current:
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
        """Pre-compute temperature-aware charge rates for each slot."""
        # Use learning_engine callbacks if available, otherwise use simple fallback
        if self.learning_engine:
            get_charge_rate = self.learning_engine.get_charge_rate_for_soc
            predict_temp = self.learning_engine.predict_temp_after_duration
        else:
            # Fallback when no learning engine is available
            get_charge_rate = lambda soc, temp: self.config.charge_rate
            predict_temp = lambda temp, duration: temp if temp is not None else 25.0

        return compute_charge_rates_per_slot(
            slots_sorted_by_time=slots_sorted_by_time,
            slot_fractions=slot_fractions,
            slot_minutes=self.config.slot_minutes,
            current_soc=current_soc,
            current_temp=current_temp if self.learning_engine else None,
            get_charge_rate_for_soc=get_charge_rate,
            predict_temp_after_duration=predict_temp,
        )

    def calculate_expected_soc_schedule(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        starting_soc: float,
        starting_temp: Optional[float] = None
    ) -> Tuple[Dict[datetime.datetime, float], Dict[datetime.datetime, float]]:
        """
        Calculate expected SOC and temperature at each slot based on schedule.
        Used for adaptive optimization to detect deviations.

        Args:
            schedule: Schedule entries keyed by slot datetime
            starting_soc: Initial SOC percentage
            starting_temp: Initial battery temperature in Celsius (optional)

        Returns:
            Tuple of (soc_trajectory, temp_trajectory) dicts keyed by slot datetime

        Notes:
            - Discharge drains at predicted load rate
            - Charge adds energy using temperature-aware rates when temp available
            - Temperature evolves during CHARGE slots based on learned warming rates
        """
        expected_soc = {}
        expected_temp = {}
        current_soc = starting_soc
        current_temp = starting_temp

        for hour in sorted(schedule.keys()):
            entry = schedule[hour]
            expected_soc[hour] = current_soc
            if current_temp is not None:
                expected_temp[hour] = current_temp

            if entry.mode == BatteryMode.CHARGE:
                # Use temperature-aware charging if temp available and learning engine exists
                if current_temp is not None and self.learning_engine:
                    # Use predict_charge_energy_with_warming for accurate cold->warm transitions
                    energy_added, end_temp = self.learning_engine.predict_charge_energy_with_warming(
                        current_soc,
                        current_temp,
                        self.config.slot_minutes,
                        temp_threshold=16.0
                    )
                    # Apply efficiency (predict_charge_energy_with_warming returns grid energy)
                    energy_to_battery = energy_added * self.config.efficiency
                    soc_increase = (energy_to_battery / self.config.battery_capacity) * 100
                    current_soc = min(self.max_soc, current_soc + soc_increase)
                    current_temp = end_temp
                else:
                    # Fallback: Use SOC-only learned charge rate
                    effective_charge_rate = self.config.charge_rate
                    if self.learning_engine:
                        learned_rate = self.learning_engine.get_charge_rate_for_soc(current_soc)
                        if learned_rate is not None and learned_rate > 0:
                            effective_charge_rate = learned_rate

                    # Charging: grid energy * efficiency goes into battery
                    energy_added = effective_charge_rate * self.config.efficiency * self.config.slot_hours
                    soc_increase = (energy_added / self.config.battery_capacity) * 100
                    current_soc = min(self.max_soc, current_soc + soc_increase)

            elif entry.mode == BatteryMode.DISCHARGE:
                # Discharging: battery drains at predicted load rate (limited by discharge rate)
                load_kw = self._predict_load_kw(hour)
                energy_removed = min(load_kw, self.config.discharge_rate) * self.config.slot_hours
                soc_decrease = (energy_removed / self.config.battery_capacity) * 100
                current_soc = max(self.min_soc, current_soc - soc_decrease)
                # Temperature cools toward ambient during discharge (no active warming)
                if current_temp is not None and self.learning_engine:
                    current_temp = self.learning_engine.predict_temp_after_idle(
                        current_temp, self.config.slot_minutes
                    )

            else:  # HOLD
                # In hold mode, grid covers base load
                # Battery has minimal standby drain, negligible for hourly planning
                # Temperature cools toward ambient during idle
                if current_temp is not None and self.learning_engine:
                    current_temp = self.learning_engine.predict_temp_after_idle(
                        current_temp, self.config.slot_minutes
                    )

        return expected_soc, expected_temp

    # =========================================================================
    # Schedule Execution
    # =========================================================================

    def full_optimize(self, kwargs=None):
        """
        Perform full optimization - called daily at 13:15 and at startup.
        """
        self.log("Starting full optimization")

        # Determine trigger type: startup (no previous recalc) or daily schedule
        now = self.datetime()
        if self._last_recalc_time is None:
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

        # Calculate minimum charge slots needed to survive the planning horizon
        charge_hours_needed = self.calculate_min_charge_slots_for_horizon(current_soc, future_prices)
        self.log(f"Current SOC: {current_soc}%, min charge slots needed: {charge_hours_needed}")

        # Generate schedule
        self.schedule = self.find_optimal_schedule(future_prices, charge_hours_needed, current_soc)

        # On restart, preserve CHARGE/DISCHARGE intent for current slot if it was active before
        self._preserve_mode_on_restart(current_slot)

        # Calculate expected SOC and temperature trajectory
        current_temp = self._get_battery_temp()
        self.expected_soc_schedule, self.expected_temp_schedule = self.calculate_expected_soc_schedule(
            self.schedule, current_soc, starting_temp=current_temp
        )

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
            min_soc=self.min_soc,
            max_soc=self.max_soc,
        )

        self.last_optimization = self.datetime()

        # Sync schedule to inverter TOU registers (if configured)
        if self.config.tou_sync_enabled and self.config.device_id:
            self._schedule_tou_sync(reason="full_optimize")

        # Apply current hour's mode
        self.execute_scheduled_mode(None)

        # Update sensor
        self._update_schedule_sensor()

        self.log("Full optimization complete")

    def adaptive_optimize(self, kwargs=None):
        """
        Adaptive re-evaluation on a configurable interval.
        Handles PV override, schedule change logging, and TOU sync.
        SOC deviation detection is now event-driven via _on_soc_change.
        """
        if not self._is_enabled() or self._is_override_active():
            return

        current_soc = self._get_current_soc()
        if current_soc is None:
            return

        # Snapshot schedule for change detection
        schedule_snapshot = {h: e.mode for h, e in self.schedule.items()} if self.schedule else {}

        pv_power = self._get_pv_power()

        # Check for solar override
        if pv_power > self.pv_threshold and self.current_mode == BatteryMode.CHARGE:
            self.log(f"Solar override: PV={pv_power}W, switching from charge to hold")
            if self.config.tou_sync_enabled and self.config.device_id:
                self._insert_hold_and_resync("solar_override")
            else:
                self.set_mode(BatteryMode.HOLD)
            return

        # Get current slot for schedule change logging
        now = self.datetime()
        local_tz = self._get_local_timezone()
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        elif local_tz is not None:
            now = now.replace(tzinfo=local_tz)
        current_slot = self._align_to_slot(now)

        # Check if schedule changed and log if so
        current_schedule_snapshot = {h: e.mode for h, e in self.schedule.items()} if self.schedule else {}
        if current_schedule_snapshot != schedule_snapshot:
            # Determine what changed
            new_hours = set(current_schedule_snapshot.keys()) - set(schedule_snapshot.keys())
            changed_modes = {h for h in current_schedule_snapshot if h in schedule_snapshot
                           and current_schedule_snapshot[h] != schedule_snapshot[h]}

            if new_hours or changed_modes:
                changes = []
                if new_hours:
                    changes.append(f"{len(new_hours)} new slots")
                if changed_modes:
                    changes.append(f"{len(changed_modes)} mode changes")
                self.log(f"Schedule updated ({', '.join(changes)}), SOC: {current_soc}%")
                # Log only current and future schedule entries (not past hours)
                future_schedule = {
                    h: e for h, e in self.schedule.items()
                    if dt_ge(h, current_slot, local_tz)
                }
                self._schedule_formatter.log_schedule(
                    schedule=future_schedule,
                    expected_soc=self.expected_soc_schedule,
                    expected_temp=self.expected_temp_schedule,
                    dp_soc_trajectory=self._last_dp_soc_trajectory,
                    dp_temp_trajectory=self._last_dp_temp_trajectory,
                    projected_costs=self._last_projected_costs,
                    local_tz=local_tz,
                    predict_load_kw=self._predict_load_kw,
                    min_soc=self.min_soc,
                    max_soc=self.max_soc,
                )

        # Check if TOU needs rolling update (every adaptive cycle)
        if self.config.tou_sync_enabled and self.config.device_id:
            self._check_and_sync_rolling_tou()

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
        self.expected_soc_schedule, self.expected_temp_schedule = self.calculate_expected_soc_schedule(
            future_schedule,
            current_soc,
            starting_temp=current_temp
        )

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
                min_soc=self.min_soc,
                max_soc=self.max_soc,
            )

        # Sync updated schedule to inverter TOU registers (if configured)
        if self.config.tou_sync_enabled and self.config.device_id:
            self._schedule_tou_sync(reason="recalculate")

        self._update_schedule_sensor()

    def execute_scheduled_mode(self, kwargs, force: bool = False):
        """
        Execute the scheduled mode for the current slot.
        Called at the start of each slot.

        Args:
            kwargs: AppDaemon callback kwargs
            force: If True, skip override check (used when manual mode set to "Auto")
        """
        if not self._is_enabled():
            return

        if not force and self._is_override_active():
            self.log("Manual override active, skipping scheduled execution")
            return

        # Get current hour in local timezone for schedule lookup
        now = self.datetime()
        local_tz = self._get_local_timezone()
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        elif local_tz is not None:
            now = now.replace(tzinfo=local_tz)
        current_slot = self._align_to_slot(now)

        entry = self.schedule.get(current_slot)

        # If not found, try matching by hour value with different timezone representations
        if entry is None and self.schedule:
            for schedule_hour, schedule_entry in self.schedule.items():
                # Convert both to local timezone for comparison
                compare_schedule = schedule_hour
                if schedule_hour.tzinfo is not None and local_tz is not None:
                    compare_schedule = schedule_hour.astimezone(local_tz)
                compare_current = current_slot
                if current_slot.tzinfo is not None and local_tz is not None:
                    compare_current = current_slot.astimezone(local_tz)
                # Compare the hour components
                if (compare_schedule.date() == compare_current.date() and
                    compare_schedule.hour == compare_current.hour and
                    compare_schedule.minute == compare_current.minute):
                    entry = schedule_entry
                    current_slot = schedule_hour
                    break

        if entry:
            self.log(f"Executing scheduled mode for {current_slot}: {entry.mode.name} ({entry.reason})")

            # When TOU sync is enabled, avoid hourly set_mode which clears TOU periods (30411=0)
            if self.config.tou_sync_enabled and self.config.device_id:
                # Still track mode transitions for learning engine baseline reset
                self._handle_mode_transition(entry.mode)
                self.log("TOU sync enabled; skipping hourly set_mode to preserve inverter TOU schedule")
                return
            self.set_mode(entry.mode)
        else:
            self.log(f"No schedule entry for {current_slot}, defaulting to HOLD")
            if self.config.tou_sync_enabled and self.config.device_id:
                self._handle_mode_transition(BatteryMode.HOLD)
                self.log("TOU sync enabled; skipping hourly set_mode to preserve inverter TOU schedule")
                return
            self.set_mode(BatteryMode.HOLD)

    def _insert_hold_and_resync(self, reason: str = "safety"):
        """
        Insert HOLD for the current slot and resync TOU schedule.

        This preserves the rest of the schedule while only modifying the current
        hour to HOLD. Much better than set_mode(HOLD) which destroys the entire
        TOU schedule and creates a 2-hour HOLD period that can conflict.

        Args:
            reason: Reason for the HOLD (used in log messages and schedule entry)
        """
        now = self.datetime()
        local_tz = self._get_local_timezone()
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        current_slot = self._align_to_slot(now)

        # Check if both schedule AND current mode are already HOLD
        # Only skip if truly nothing needs to change (both aligned to HOLD)
        if current_slot in self.schedule:
            old_entry = self.schedule[current_slot]
            if old_entry.mode == BatteryMode.HOLD and self.current_mode == BatteryMode.HOLD:
                return  # Both schedule and inverter already HOLD, nothing to do
            if old_entry.mode != BatteryMode.HOLD:
                # Schedule needs updating
                self.schedule[current_slot] = ScheduleEntry(
                    time=current_slot,
                    mode=BatteryMode.HOLD,
                    reason=f"{reason}_hold (was {old_entry.mode.name})"
                )
                self.log(f"Inserted HOLD at {current_slot} (was {old_entry.mode.name})")
            else:
                # Schedule is HOLD but current_mode isn't - log the enforcement
                self.log(f"Enforcing HOLD at {current_slot} (schedule was HOLD but mode was {self.current_mode.name})")
        else:
            self.schedule[current_slot] = ScheduleEntry(
                time=current_slot,
                mode=BatteryMode.HOLD,
                reason=f"{reason}_hold"
            )
            self.log(f"Inserted HOLD at {current_slot}")

        self._handle_mode_transition(BatteryMode.HOLD)
        self._update_schedule_sensor()

        # Resync TOU with updated schedule (uses existing async wrapper)
        if self.config.tou_sync_enabled and self.config.device_id:
            self._schedule_tou_sync(skip_fit_check=True, reason=f"{reason}_hold_resync")

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
        # Stop discharge if SOC too low
        if current_soc <= self.min_soc and self.current_mode == BatteryMode.DISCHARGE:
            self.log(f"Safety: HOLD (battery depleted at {current_soc}%)")
            if self.config.tou_sync_enabled and self.config.device_id:
                self._insert_hold_and_resync("battery_depleted")
            else:
                self.set_mode(BatteryMode.HOLD)
            return True

        # Stop charge if SOC full
        if current_soc >= self.max_soc and self.current_mode == BatteryMode.CHARGE:
            self.log(f"Safety: Stopping charge, SOC at maximum ({current_soc}%)")
            if self.config.tou_sync_enabled and self.config.device_id:
                self._insert_hold_and_resync("safety_max_soc")
            else:
                self.set_mode(BatteryMode.HOLD)
            return True

        return False

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

        # Create detector with current config (min_soc/max_soc are dynamic properties)
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
            decision_log_level=self.config.decision_log_level,
        )
        detector = SocDeviationDetector(
            config=config,
            learning_engine=self.learning_engine,
            log_func=self.log,
        )

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

    def set_mode(self, mode: BatteryMode, power_percent: int = 100):
        """
        Set the battery mode via VPP protocol registers.

        Delegates to TouSyncManager for the actual register writes.
        Updates mode tracking on success or in dry-run mode.

        Mode Mapping:
        - CHARGE: Remote control with positive power
        - DISCHARGE: Remote control with negative power
        - HOLD: TOU with +1% charge (firmware quirk for true standby)

        Dry-run mode: When device_id is empty, no register writes are performed
        but internal state is still updated for testing/simulation purposes.
        """
        # Delegate to TouSyncManager for register writes
        success = self._tou_sync_manager.set_mode(mode, power_percent)

        # Update mode tracking on success, or always in dry-run mode (no device_id)
        # for state consistency during testing/simulation
        if success or not self.config.device_id:
            self._handle_mode_transition(mode)

    # =========================================================================
    # TOU Schedule Sync (delegates to TouSyncManager)
    # =========================================================================

    def schedule_to_tou_periods(self, boundary_minute: int = None) -> List[TouPeriod]:
        """Convert current schedule to TOU periods for inverter programming."""
        return self._tou_sync_manager.schedule_to_tou_periods(self.schedule, boundary_minute)

    def _schedule_tou_sync(self, boundary_minute: int = None, skip_fit_check: bool = False,
                           allow_queue: bool = True, reason: str = ""):
        """Schedule a TOU sync, avoiding overlapping register writes."""
        self._tou_sync_manager.schedule_tou_sync(
            boundary_minute, skip_fit_check, allow_queue, reason
        )

    def _check_and_sync_rolling_tou(self):
        """Check if TOU schedule needs rolling update and sync if needed."""
        self._tou_sync_manager.check_and_sync_rolling_tou()

    async def sync_schedule_to_inverter(self, boundary_minute: int = None,
                                        skip_fit_check: bool = False) -> bool:
        """Sync the current schedule to the inverter's TOU registers."""
        return await self._tou_sync_manager.sync_schedule_to_inverter(
            self.schedule, boundary_minute, skip_fit_check
        )

    # =========================================================================
    # Manual Override Handling
    # =========================================================================

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
            self.set_mode(mode)
        elif mode_str == "Auto":
            # Turn off override to fully resume automatic scheduling
            # This ensures subsequent hourly executions work correctly
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

    def record_load_observation(self, kwargs=None):
        """Record current house load into the statistical load profile."""
        if not self.config.load_power_sensor:
            return
        load_w = self._get_load_power()
        if load_w is None:
            return
        now = self._align_to_slot(self.datetime())
        self.load_profile.record(now, load_w)
        self._save_load_profile()

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

    def _get_discharge_threshold_for_cost(self, avg_cost: float) -> float:
        """Calculate discharge threshold for a given battery average cost."""
        return self._cost_tracker.get_discharge_threshold_for_cost(avg_cost)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_current_soc(self) -> Optional[float]:
        """Get current battery SOC."""
        return self._sensors.get_soc(self.config.soc_sensor)

    def _get_pv_power(self) -> float:
        """Get current PV power production."""
        return self._sensors.get_power(self.config.pv_power_sensor, default=0.0)

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
        """Predict expected load (kW) for a slot using load profile."""
        if self.load_profile:
            predicted = self.load_profile.predict_kw(dt, self.config.load_quantile)
        else:
            predicted = self.config.base_consumption / 1000.0

        return predicted

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

    def _normalize_to_local(self, dt: datetime.datetime, local_tz) -> datetime.datetime:
        """Normalize a datetime to local timezone for comparison."""
        if dt is None:
            return dt
        return ensure_local_tz(dt, local_tz)

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
                    "slot_minutes": self.config.slot_minutes,
                    "friendly_name": "Battery Optimizer"
                }
            )
        except Exception as e:
            self.log(f"Error updating schedule sensor: {e}", level="WARNING")

    def get_schedule_summary(self) -> str:
        """Generate a human-readable schedule summary"""
        return self._schedule_formatter.format_summary(
            schedule=self.schedule,
            now=self.datetime(),
            local_tz=self._get_local_timezone(),
            align_to_slot_func=self._align_to_slot,
        )
