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

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Import from the battery_optimizer_lib package
from battery_optimizer_lib import (
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
    dt_gt,
    dt_lt,
    ensure_local_tz,
    align_to_slot,
    next_slot_time,
    next_interval_time,
    lookup_by_hour,
    duration_minutes,
    # HA helpers
    SensorReader,
    # Cost tracker
    BatteryCostTracker,
    BatteryCostConfig,
    # Charge rate utilities
    compute_charge_rates_per_slot,
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
            battery_capacity_kwh=self.battery_capacity,
            nominal_charge_rate_kw=self.charge_rate,
            nominal_efficiency=self.efficiency,
            min_soc=self._default_min_soc,
            max_soc=self._default_max_soc,
            log_func=self.log,
        )
        self._init_learning_engine()

        # Load profile for probabilistic scheduling
        self.load_profile = LoadProfile(
            slot_minutes=self.slot_minutes,
            default_load_w=self.base_consumption,
            max_samples=self.load_profile_max_samples,
            min_samples=self.load_profile_min_samples,
            log_func=self.log,
        )
        self._init_load_profile()

        # TOU sync manager for inverter schedule sync
        self._tou_sync_manager = TouSyncManager(
            device_id=self.device_id,
            slot_minutes=self.slot_minutes,
            ha_url=self.args.get("ha_url", ""),
            ha_token=self.args.get("ha_token", ""),
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
            nordpool_config_entry=self.nordpool_config_entry,
            nordpool_area=self.nordpool_area,
            nordpool_sensor=self.nordpool_sensor,
            ha_url=self.args.get("ha_url", ""),
            ha_token=self.args.get("ha_token", ""),
            tomorrow_prices_hour=self.tomorrow_prices_hour,
            slot_minutes=self.slot_minutes,
            get_state_func=self.get_state,
            call_service_func=self.call_service,
            get_datetime_func=self.datetime,
            get_date_func=self.date,
            get_timezone_func=self._get_local_timezone,
            log_func=self.log,
        )

        # Battery cost tracker (must be after learning engine and price service)
        self._cost_tracker = BatteryCostTracker(
            config=BatteryCostConfig(
                battery_cost_entity=self.args.get("battery_cost_entity", "input_number.battery_avg_cost"),
                battery_charge_sensor=self.battery_charge_sensor,
                battery_discharge_sensor=self.battery_discharge_sensor,
                use_inverter_energy_sensors=self.use_inverter_energy_sensors,
                battery_capacity=self.battery_capacity,
                efficiency=self.efficiency,
                slot_minutes=self.slot_minutes,
                charge_rate=self.charge_rate,
                discharge_rate=self.discharge_rate,
                grid_fee=self.grid_fee,
                battery_wear_cost=self.battery_wear_cost,
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
        optimize_hour = self.tomorrow_prices_hour
        self.run_daily(self.full_optimize, datetime.time(optimize_hour, 15))

        # Startup optimization is triggered from _init_battery_cost() after battery cost is loaded
        # (either immediately if HA available, or after homeassistant_start event)

        # Adaptive re-evaluation (can be more frequent than schedule slots)
        self.run_every(
            self.adaptive_optimize,
            self._next_interval_time(self.adaptive_recalc_minutes),
            self.adaptive_recalc_minutes * 60
        )

        # Schedule execution every slot (hourly if slot_minutes=60)
        self.run_every(self.execute_scheduled_mode, self._next_slot_time(), self.slot_minutes * 60)

        # Record load observations (can be more frequent than schedule slots)
        self.run_every(
            self.record_load_observation,
            self._next_interval_time(self.load_observation_minutes),
            self.load_observation_minutes * 60
        )

        # Listen for manual override changes
        if self.args.get("override_entity"):
            self.listen_state(self.on_override_change, self.args["override_entity"])
        if self.args.get("manual_mode_entity"):
            self.listen_state(self.on_manual_mode_change, self.args["manual_mode_entity"])

        # Listen to SOC changes for instant response (replaces polling-based checks)
        self.listen_state(self._on_soc_change, self.soc_sensor)

        # Listen to inverter energy sensors (primary trigger when available)
        if self.use_inverter_energy_sensors:
            self.listen_state(self._on_energy_sensor_change, self.battery_charge_sensor)
            self.listen_state(self._on_energy_sensor_change, self.battery_discharge_sensor)
            self.log(f"Listening to energy sensors: {self.battery_charge_sensor}, {self.battery_discharge_sensor}")

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
        """Load configuration from apps.yaml"""
        # Nord Pool configuration
        # For built-in HA integration: set nordpool_config_entry (from diagnostics or HA URL)
        # For HACS custom component: set nordpool_sensor
        self.nordpool_config_entry = self.args.get("nordpool_config_entry", "")
        self.nordpool_area = self.args.get("nordpool_area", "LV")
        self.nordpool_sensor = self.args.get("nordpool_sensor", "sensor.nord_pool_lv_current_price")

        # Other sensor entities
        self.soc_sensor = self.args.get("soc_sensor", "sensor.growatt_battery_soc")
        self.pv_power_sensor = self.args.get("pv_power_sensor", "sensor.growatt_pv_power")
        self.battery_temp_sensor = self.args.get("battery_temp_sensor", "")

        # Inverter energy sensors for precise energy measurement
        self.battery_charge_sensor = self.args.get("battery_charge_sensor", "sensor.growatt_battery_charge_today")
        self.battery_discharge_sensor = self.args.get("battery_discharge_sensor", "sensor.growatt_battery_discharge_today")
        self.use_inverter_energy_sensors = self.args.get("use_inverter_energy_sensors", True)

        # Nord Pool publishes tomorrow's prices at 13:00 CET
        # Default to 14 for EET (Latvia, Lithuania, Estonia) which is 13:00 CET
        # Adjust this based on your local timezone relative to CET
        self.tomorrow_prices_hour = int(self.args.get("tomorrow_prices_hour", 14))

        self.log(f"Nord Pool config: config_entry='{self.nordpool_config_entry}', area='{self.nordpool_area}', sensor='{self.nordpool_sensor}'")
        self.log(f"HA connection: ha_url='{self.args.get('ha_url', 'NOT SET')}', ha_token={'SET' if self.args.get('ha_token') else 'NOT SET'}")

        # Device control
        self.device_id = self.args.get("device_id", "")

        # TOU schedule sync (uses growatt_modbus integration)
        # When device_id is set and tou_sync_enabled is true, writes schedule to inverter TOU registers
        self.tou_sync_enabled = self.args.get("tou_sync_enabled", True)
        if self.tou_sync_enabled and self.device_id:
            self.log(f"TOU sync enabled via growatt_modbus (device: {self.device_id})")

        # Battery parameters (static - from apps.yaml)
        self.battery_capacity = float(self.args.get("battery_capacity_kwh", 14.3))
        self.charge_rate = float(self.args.get("charge_rate_kw", 4.5))
        self.discharge_rate = float(self.args.get("discharge_rate_kw", self.charge_rate))
        self.efficiency = float(self.args.get("efficiency", 0.85))
        self.base_consumption = float(self.args.get("base_consumption_w", 500))

        # Scheduling resolution (minutes per slot)
        self.slot_minutes = int(self.args.get("slot_minutes", 30))
        if self.slot_minutes <= 0 or 1440 % self.slot_minutes != 0:
            self.log(f"Invalid slot_minutes={self.slot_minutes}, falling back to 30", level="WARNING")
            self.slot_minutes = 30
        self.slot_hours = self.slot_minutes / 60.0

        # Recalculation and observation intervals (minutes)
        self.adaptive_recalc_minutes = int(self.args.get("adaptive_recalc_minutes", 30))
        if self.adaptive_recalc_minutes <= 0 or 1440 % self.adaptive_recalc_minutes != 0:
            self.log(f"Invalid adaptive_recalc_minutes={self.adaptive_recalc_minutes}, falling back to 30", level="WARNING")
            self.adaptive_recalc_minutes = 30
        self.load_observation_minutes = int(self.args.get("load_observation_minutes", self.adaptive_recalc_minutes))
        if self.load_observation_minutes <= 0 or 1440 % self.load_observation_minutes != 0:
            self.log(f"Invalid load_observation_minutes={self.load_observation_minutes}, falling back to 30", level="WARNING")
            self.load_observation_minutes = 30

        # DP resolution for SOC (percent step)
        self.soc_step_percent = float(self.args.get("soc_step_percent", 1.0))
        if self.soc_step_percent <= 0:
            self.soc_step_percent = 1.0

        # Load profile configuration
        self.load_power_sensor = self.args.get("load_power_sensor", "")
        self.load_quantile = float(self.args.get("load_quantile", 0.75))
        self.load_quantile = min(1.0, max(0.0, self.load_quantile))
        self.load_profile_entity = self.args.get("load_profile_entity", "input_text.battery_load_profile")
        self.load_profile_max_samples = int(self.args.get("load_profile_max_samples", 60))
        self.load_profile_min_samples = int(self.args.get("load_profile_min_samples", 6))
        self.load_zero_floor_w = float(self.args.get("load_zero_floor_w", 450))
        self.load_profile_file = self.args.get("load_profile_file", "/config/load_profile.json")
        self.load_profile_last_obs_entity = self.args.get(
            "load_profile_last_observation_entity",
            "sensor.load_profile_last_observation"
        )
        self.load_profile_count_entity = self.args.get(
            "load_profile_observation_count_entity",
            "sensor.load_profile_observation_count"
        )
        # Default values (can be overridden by HA input_numbers at runtime)
        self._default_min_soc = float(self.args.get("min_soc", 10))
        self._default_max_soc = float(self.args.get("max_soc", 100))
        self._default_pv_threshold = float(self.args.get("pv_threshold_w", 500))

        # Pricing
        self.grid_fee = float(self.args.get("grid_fee_eur_kwh", 0.05))
        self.battery_wear_cost = float(self.args.get("battery_wear_cost_eur_kwh", 0.0))
        self.export_rate_multiplier = float(self.args.get("export_rate_multiplier", 1.0))
        self.log(f"Loaded grid_fee: {self.grid_fee} EUR/kWh")

        # HA entities for dynamic config (optional - falls back to defaults)
        self.min_soc_entity = self.args.get("min_soc_entity", "input_number.battery_min_soc")
        self.max_soc_entity = self.args.get("max_soc_entity", "input_number.battery_max_soc")
        self.pv_threshold_entity = self.args.get("pv_threshold_entity", "input_number.battery_pv_threshold")
        self.soc_deviation_threshold = float(self.args.get("soc_deviation_threshold", 10))

        # Decision transparency logging (0=minimal, 1=summary, 2=verbose)
        self.decision_log_level = int(self.args.get("decision_log_level", 1))

        # Control entities
        self.enabled_entity = self.args.get("enabled_entity", "input_boolean.battery_optimizer_enabled")
        self.override_entity = self.args.get("override_entity", "input_boolean.battery_optimizer_override")
        self.manual_mode_entity = self.args.get("manual_mode_entity", "input_select.battery_manual_mode")

        self.log(f"Config loaded: capacity={self.battery_capacity}kWh, "
                 f"charge_rate={self.charge_rate}kW, discharge_rate={self.discharge_rate}kW, "
                 f"efficiency={self.efficiency}, slot={self.slot_minutes}min")

    # =========================================================================
    # Price Fetching (delegates to NordPoolPriceService)
    # =========================================================================

    def get_prices(self) -> List[PricePoint]:
        """Fetch prices from Nord Pool. Delegates to NordPoolPriceService."""
        return self._price_service.get_prices()

    # =========================================================================
    # Optimization Algorithm
    # =========================================================================

    def calculate_charge_hours(self, current_soc: float, target_soc: float = None) -> int:
        """
        Calculate how many slots of charging needed to reach target SOC.
        This is a simple calculation used for basic estimates.
        For actual scheduling, use calculate_min_charge_slots_for_horizon().
        """
        if target_soc is None:
            target_soc = self.max_soc

        if current_soc >= target_soc:
            return 0

        # Energy needed in kWh
        soc_gap = target_soc - current_soc
        energy_needed = soc_gap / 100 * self.battery_capacity

        # Account for charging efficiency
        grid_energy_needed = energy_needed / self.efficiency

        # Slots at charge rate
        energy_per_slot = self.charge_rate * self.slot_hours
        if energy_per_slot <= 0:
            return 0

        return math.ceil(grid_energy_needed / energy_per_slot)

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

        for price_point in sorted(prices, key=lambda p: p.hour):
            # Predict load for this slot
            load_kw = self._predict_load_kw(price_point.hour)
            load_kwh = min(load_kw, self.discharge_rate) * self.slot_hours
            total_load_kwh += load_kwh

        # Energy available above min_soc
        usable_energy_kwh = (current_soc - self.min_soc) / 100 * self.battery_capacity

        # Energy deficit (how much we'd be short)
        energy_deficit_kwh = total_load_kwh - usable_energy_kwh

        if energy_deficit_kwh <= 0:
            # We have enough battery to survive the horizon
            if self.decision_log_level >= 1:
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
        grid_energy_needed = energy_deficit_kwh / self.efficiency

        # Slots at charge rate
        energy_per_slot = self.charge_rate * self.efficiency * self.slot_hours  # Energy INTO battery per slot
        if energy_per_slot <= 0:
            return 0

        charge_slots_raw = energy_deficit_kwh / energy_per_slot
        charge_slots = math.ceil(charge_slots_raw)

        if self.decision_log_level >= 1:
            self.log(
                f"Charge calculation: SOC {current_soc:.1f}% | "
                f"Usable energy: {usable_energy_kwh:.2f} kWh (above {self.min_soc}% min) | "
                f"Expected load: {total_load_kwh:.2f} kWh over {len(prices)} slots | "
                f"Deficit: {energy_deficit_kwh:.2f} kWh | "
                f"Slots @ {energy_per_slot:.2f} kWh/slot | "
                f"Result: {charge_slots_raw:.2f} -> {charge_slots} charge slots needed"
            )

        return charge_slots

    def calculate_discharge_hours(self, current_soc: float, target_soc: float = None) -> int:
        """
        Calculate how many hours of discharge available until target SOC.

        During discharge mode:
        - Battery powers the house load (base_consumption)
        - Drain rate depends on actual consumption, not max inverter rate
        """
        if target_soc is None:
            target_soc = self.min_soc

        if current_soc <= target_soc:
            return 0

        # Energy available in kWh
        energy_available = (current_soc - target_soc) / 100 * self.battery_capacity

        # Battery drains at expected load rate, limited by discharge rate
        base_consumption_kw = self.base_consumption / 1000  # Convert W to kW
        if base_consumption_kw <= 0:
            return 0

        discharge_hours = energy_available / base_consumption_kw

        return int(discharge_hours)

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

        future_prices = [p for p in prices if dt_ge(p.hour, current_slot)]
        if not future_prices:
            return {}

        # Ensure current slot is included (Nord Pool may exclude current hour as "past")
        future_prices = self._ensure_current_slot_price(prices, future_prices, current_slot)

        # Calculate min_charge_slots for informational purposes
        hours_sorted_by_time = sorted(future_prices, key=lambda p: p.hour)
        n_slots = len(hours_sorted_by_time)
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
                battery_capacity=self.battery_capacity,
                min_soc=self.min_soc,
                max_soc=self.max_soc,
                efficiency=self.efficiency,
                discharge_rate=self.discharge_rate,
                slot_minutes=self.slot_minutes,
                soc_step_percent=self.soc_step_percent,
                grid_fee=self.grid_fee,
                battery_wear_cost=self.battery_wear_cost,
            ),
            load_predictor=self._predict_load_kw,
            charge_rate_predictor=self.learning_engine.get_charge_rate_for_soc,
            temp_after_charge_predictor=self.learning_engine.predict_temp_after_duration,
            temp_after_idle_predictor=self.learning_engine.predict_temp_after_idle,
            log_fn=self.log,
            decision_log_level=self.decision_log_level,
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
            prices_by_slot = {p.hour: p.price for p in hours_sorted_by_time}
            # Re-compute charge rates per slot for cost projection
            slot_fractions = self._compute_slot_fractions(hours_sorted_by_time, current_slot, minutes_into_slot)
            charge_rates_per_slot = self._compute_charge_rates_per_slot(
                hours_sorted_by_time, slot_fractions, current_soc_for_calc, current_temp
            )
            slot_charge_rates_by_slot = {
                p.hour: charge_rates_per_slot[i] for i, p in enumerate(hours_sorted_by_time)
            }
            slot_fractions_by_slot = {
                p.hour: slot_fractions[i] for i, p in enumerate(hours_sorted_by_time)
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
                 f"{result.hold_count} hold slots (slot={self.slot_minutes}min, load_quantile={self.load_quantile:.2f}, "
                 f"min_charge_slots={min_charge_slots})")

        # Store min_charge_slots for sensor exposure
        self._last_min_charge_slots = min_charge_slots

        # Log decision context for transparency
        if self.decision_log_level >= 1:
            load_kw = [self._predict_load_kw(p.hour) for p in hours_sorted_by_time]
            self._log_schedule_decision_context(
                hours_sorted_by_time, schedule, load_kw, current_soc_for_calc, min_charge_slots
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
                p_naive = p.hour.replace(tzinfo=None) if p.hour.tzinfo else p.hour
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
                p_naive = p.hour.replace(tzinfo=None) if p.hour.tzinfo else p.hour
                if p_naive == slot_naive:
                    return p
            return None

        slot_price_point = find_slot_price(yesterday_prices, current_slot)
        if slot_price_point is not None:
            synth_price = slot_price_point.price
            self.log(
                f"Added missing current slot {current_slot} using yesterday's price "
                f"{synth_price:.4f} EUR/kWh from {slot_price_point.hour}"
            )
        else:
            # If still missing, synthesize using most recent past price if available
            current_slot_naive = current_slot.replace(tzinfo=None) if current_slot.tzinfo else current_slot
            prev_price_point = None
            for p in all_prices:
                p_hour = p.hour
                p_naive = p_hour.replace(tzinfo=None) if p_hour.tzinfo else p_hour
                if p_naive <= current_slot_naive:
                    if prev_price_point is None:
                        prev_price_point = p
                    else:
                        prev_naive = prev_price_point.hour.replace(tzinfo=None) if prev_price_point.hour.tzinfo else prev_price_point.hour
                        if p_naive > prev_naive:
                            prev_price_point = p

            if prev_price_point is not None:
                synth_price = prev_price_point.price
                self.log(
                    f"Added missing current slot {current_slot} using previous price "
                    f"{synth_price:.4f} EUR/kWh from {prev_price_point.hour}"
                )
            else:
                # Fallback: use first available future price
                synth_price = min(future_prices, key=lambda p: p.hour).price
                self.log(f"Added missing current slot {current_slot} using next price {synth_price:.4f} EUR/kWh")

        current_slot_price = PricePoint(hour=current_slot, price=synth_price)
        return future_prices + [current_slot_price]

    def _compute_slot_fractions(
        self,
        hours_sorted_by_time: List[PricePoint],
        current_slot: datetime.datetime,
        minutes_into_slot: float,
    ) -> List[float]:
        """Compute fraction of each slot that is usable (partial first slot)."""
        n_slots = len(hours_sorted_by_time)
        first_fraction = min(1.0, max(0.0, (self.slot_minutes - minutes_into_slot) / max(1, self.slot_minutes)))
        slot_fractions = [1.0] * n_slots

        for i, p in enumerate(hours_sorted_by_time):
            p_hour = p.hour
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
        hours_sorted_by_time: List[PricePoint],
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
            get_charge_rate = lambda soc, temp: self.charge_rate
            predict_temp = lambda temp, duration: temp if temp is not None else 25.0

        return compute_charge_rates_per_slot(
            hours_sorted_by_time=hours_sorted_by_time,
            slot_fractions=slot_fractions,
            slot_minutes=self.slot_minutes,
            current_soc=current_soc,
            current_temp=current_temp if self.learning_engine else None,
            get_charge_rate_for_soc=get_charge_rate,
            predict_temp_after_duration=predict_temp,
        )

    def _log_schedule_decision_context(
        self,
        prices_sorted: List[PricePoint],
        schedule: Dict[datetime.datetime, ScheduleEntry],
        load_kw: List[float],
        current_soc: float,
        min_charge_slots: int
    ):
        """
        Log detailed decision context for transparency.
        Shows why specific charge/discharge slots were selected.
        """
        # Extract charge and discharge slots from schedule
        charge_slots = []
        discharge_slots = []
        for hour, entry in schedule.items():
            price_point = next((p for p in prices_sorted if p.hour == hour), None)
            price = price_point.price if price_point else 0.0
            if entry.mode == BatteryMode.CHARGE:
                charge_slots.append({"hour": hour, "price": price})
            elif entry.mode == BatteryMode.DISCHARGE:
                # Find corresponding load
                idx = next((i for i, p in enumerate(prices_sorted) if p.hour == hour), 0)
                load = load_kw[idx] if idx < len(load_kw) else 0.0
                discharge_slots.append({"hour": hour, "price": price, "load": load})

        # Sort all prices to rank candidates
        all_prices_sorted = sorted(prices_sorted, key=lambda p: p.price)
        price_rank = {p.hour: i + 1 for i, p in enumerate(all_prices_sorted)}

        # Store charge slots for sensor exposure
        self._last_charge_slots = [
            {"time": s["hour"].isoformat(), "price": round(s["price"], 4)}
            for s in sorted(charge_slots, key=lambda x: x["hour"])
        ]

        # Build decision context log
        if self.decision_log_level >= 1:
            self.log("=" * 70)
            self.log("DECISION CONTEXT")
            self.log("=" * 70)

            # Input state
            self.log(f"Input State:")
            self.log(f"  Current SOC: {current_soc:.1f}%")
            self.log(f"  Min SOC target: {self.min_soc:.1f}%")
            self.log(f"  Min charge slots (informational): {min_charge_slots}")
            self.log(f"  Battery avg cost: {self.battery_avg_cost:.4f} EUR/kWh")
            self.log(f"  Discharge wear cost: {self.battery_wear_cost:.4f} EUR/kWh")
            self.log(f"  Note: DP evaluates all options; discharge only costs wear (no double-counting)")

        # Verbose logging (level 2): show candidates and analysis
        if self.decision_log_level >= 2:
            def _fmt_dt(dt: datetime.datetime) -> str:
                return dt.strftime("%Y-%m-%d %H:%M")

            # Cheapest charge candidates
            self.log(f"\nCheapest 5 charge candidates:")
            for i, p in enumerate(all_prices_sorted[:5]):
                marker = " *" if any(s["hour"] == p.hour for s in charge_slots) else ""
                self.log(f"  {i+1}. {_fmt_dt(p.hour)} @ {p.price:.4f} EUR/kWh{marker}")

            # Selected charge slots with rankings
            if charge_slots:
                self.log(f"\nSelected charge slots ({len(charge_slots)}):")
                for slot in sorted(charge_slots, key=lambda s: s["hour"]):
                    rank = price_rank.get(slot["hour"], "?")
                    total_prices = len(prices_sorted)
                    self.log(f"  {_fmt_dt(slot['hour'])} @ {slot['price']:.4f} EUR/kWh (rank {rank}/{total_prices})")

            # Selected discharge slots
            if discharge_slots:
                self.log(f"\nSelected discharge slots ({len(discharge_slots)}):")
                for slot in sorted(discharge_slots, key=lambda s: s["hour"]):
                    self.log(f"  {_fmt_dt(slot['hour'])} @ {slot['price']:.4f} EUR/kWh (load~{slot['load']:.2f}kW)")

            # Arbitrage analysis
            if charge_slots and discharge_slots:
                avg_charge_price = sum(s["price"] for s in charge_slots) / len(charge_slots)
                avg_discharge_price = sum(s["price"] for s in discharge_slots) / len(discharge_slots)
                spread = avg_discharge_price - avg_charge_price
                effective_spread = spread - (avg_charge_price * (1 - self.efficiency))

                self.log(f"\nArbitrage Analysis:")
                self.log(f"  Avg charge price: {avg_charge_price:.4f} EUR/kWh")
                self.log(f"  Avg discharge price: {avg_discharge_price:.4f} EUR/kWh")
                self.log(f"  Spread: {spread:.4f} EUR/kWh (after {self.efficiency*100:.0f}% efficiency: {effective_spread:.4f} EUR/kWh)")

        if self.decision_log_level >= 1:
            self.log("=" * 70)

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
                        self.slot_minutes,
                        temp_threshold=16.0
                    )
                    # Apply efficiency (predict_charge_energy_with_warming returns grid energy)
                    energy_to_battery = energy_added * self.efficiency
                    soc_increase = (energy_to_battery / self.battery_capacity) * 100
                    current_soc = min(self.max_soc, current_soc + soc_increase)
                    current_temp = end_temp
                else:
                    # Fallback: Use SOC-only learned charge rate
                    effective_charge_rate = self.charge_rate
                    if self.learning_engine:
                        learned_rate = self.learning_engine.get_charge_rate_for_soc(current_soc)
                        if learned_rate is not None and learned_rate > 0:
                            effective_charge_rate = learned_rate

                    # Charging: grid energy * efficiency goes into battery
                    energy_added = effective_charge_rate * self.efficiency * self.slot_hours
                    soc_increase = (energy_added / self.battery_capacity) * 100
                    current_soc = min(self.max_soc, current_soc + soc_increase)

            elif entry.mode == BatteryMode.DISCHARGE:
                # Discharging: battery drains at predicted load rate (limited by discharge rate)
                load_kw = self._predict_load_kw(hour)
                energy_removed = min(load_kw, self.discharge_rate) * self.slot_hours
                soc_decrease = (energy_removed / self.battery_capacity) * 100
                current_soc = max(self.min_soc, current_soc - soc_decrease)
                # Temperature cools toward ambient during discharge (no active warming)
                if current_temp is not None and self.learning_engine:
                    current_temp = self.learning_engine.predict_temp_after_idle(
                        current_temp, self.slot_minutes
                    )

            else:  # HOLD
                # In hold mode, grid covers base load
                # Battery has minimal standby drain, negligible for hourly planning
                # Temperature cools toward ambient during idle
                if current_temp is not None and self.learning_engine:
                    current_temp = self.learning_engine.predict_temp_after_idle(
                        current_temp, self.slot_minutes
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
            scheduled_hour = self.tomorrow_prices_hour
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
        future_prices = [p for p in prices if dt_ge(p.hour, current_slot)]
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
        self._log_schedule(self.schedule, self.expected_soc_schedule, self.expected_temp_schedule)

        self.last_optimization = self.datetime()

        # Sync schedule to inverter TOU registers (if configured)
        if self.tou_sync_enabled and self.device_id:
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
            if self.tou_sync_enabled and self.device_id:
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
                self._log_schedule(future_schedule, self.expected_soc_schedule, self.expected_temp_schedule)

        # Check if TOU needs rolling update (every adaptive cycle)
        if self.tou_sync_enabled and self.device_id:
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
        future_prices = [p for p in prices if dt_ge(p.hour, now_slot, local_tz)]

        if not future_prices:
            self.log("No future prices available for recalculation")
            return

        self.log(f"Recalculating with {len(future_prices)} future price points "
                 f"({future_prices[0].hour.strftime('%m-%d %H:%M')} to {future_prices[-1].hour.strftime('%m-%d %H:%M')})")

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
        if self.decision_log_level >= 1:
            self._log_schedule(
                future_schedule,
                self.expected_soc_schedule,
                self.expected_temp_schedule
            )

        # Sync updated schedule to inverter TOU registers (if configured)
        if self.tou_sync_enabled and self.device_id:
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
            if self.tou_sync_enabled and self.device_id:
                # Still track mode transitions for learning engine baseline reset
                self._handle_mode_transition(entry.mode)
                self.log("TOU sync enabled; skipping hourly set_mode to preserve inverter TOU schedule")
                return
            self.set_mode(entry.mode)
        else:
            self.log(f"No schedule entry for {current_slot}, defaulting to HOLD")
            if self.tou_sync_enabled and self.device_id:
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
                    hour=current_slot,
                    mode=BatteryMode.HOLD,
                    reason=f"{reason}_hold (was {old_entry.mode.name})"
                )
                self.log(f"Inserted HOLD at {current_slot} (was {old_entry.mode.name})")
            else:
                # Schedule is HOLD but current_mode isn't - log the enforcement
                self.log(f"Enforcing HOLD at {current_slot} (schedule was HOLD but mode was {self.current_mode.name})")
        else:
            self.schedule[current_slot] = ScheduleEntry(
                hour=current_slot,
                mode=BatteryMode.HOLD,
                reason=f"{reason}_hold"
            )
            self.log(f"Inserted HOLD at {current_slot}")

        self._handle_mode_transition(BatteryMode.HOLD)
        self._update_schedule_sensor()

        # Resync TOU with updated schedule (uses existing async wrapper)
        if self.tou_sync_enabled and self.device_id:
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
            if self.tou_sync_enabled and self.device_id:
                self._insert_hold_and_resync("battery_depleted")
            else:
                self.set_mode(BatteryMode.HOLD)
            return True

        # Stop charge if SOC full
        if current_soc >= self.max_soc and self.current_mode == BatteryMode.CHARGE:
            self.log(f"Safety: Stopping charge, SOC at maximum ({current_soc}%)")
            if self.tou_sync_enabled and self.device_id:
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
        price_map = {p.hour: p.price for p in prices}

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

        Args:
            current_soc: Current battery state of charge (%)

        Returns:
            True if recalculation was triggered, False otherwise
        """
        if not self._is_enabled() or self._is_override_active():
            return False

        now = self.datetime()
        local_tz = self._get_local_timezone()
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        elif local_tz is not None:
            now = now.replace(tzinfo=local_tz)
        current_slot = self._align_to_slot(now)

        expected_soc = self.expected_soc_schedule.get(current_slot)

        # If not found, try matching by hour value with different timezone representations
        if expected_soc is None and self.expected_soc_schedule:
            for schedule_hour, soc_value in self.expected_soc_schedule.items():
                compare_schedule = schedule_hour
                if schedule_hour.tzinfo is not None and local_tz is not None:
                    compare_schedule = schedule_hour.astimezone(local_tz)
                compare_current = current_slot
                if current_slot.tzinfo is not None and local_tz is not None:
                    compare_current = current_slot.astimezone(local_tz)
                if (compare_schedule.date() == compare_current.date() and
                    compare_schedule.hour == compare_current.hour and
                    compare_schedule.minute == compare_current.minute):
                    expected_soc = soc_value
                    break

        if expected_soc is None:
            return False

        # Adjust expected SOC within the slot based on elapsed time
        entry = self.schedule.get(current_slot)
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
                    break

        minutes_into_slot = max(0.0, (now - current_slot).total_seconds() / 60.0)
        fraction = min(1.0, minutes_into_slot / max(1, self.slot_minutes))
        expected_soc_now = expected_soc

        # Get current battery temperature for temperature-aware rate lookups
        # (inverter charge rate varies significantly with temperature)
        current_temp = self._get_battery_temp()

        if entry and fraction > 0:
            if entry.mode == BatteryMode.CHARGE:
                # Use learned charge rate if available for more accurate mid-slot projection
                effective_charge_rate = self.charge_rate
                if self.learning_engine:
                    learned_rate = self.learning_engine.get_charge_rate_for_soc(current_soc, current_temp)
                    if learned_rate is not None and learned_rate > 0:
                        effective_charge_rate = learned_rate

                energy_added = effective_charge_rate * self.efficiency * self.slot_hours * fraction
                expected_soc_now = min(
                    self.max_soc,
                    expected_soc + (energy_added / self.battery_capacity) * 100
                )
            elif entry.mode == BatteryMode.DISCHARGE:
                load_kw = self._predict_load_kw(current_slot)
                energy_removed = min(load_kw, self.discharge_rate) * self.slot_hours * fraction
                expected_soc_now = max(
                    self.min_soc,
                    expected_soc - (energy_removed / self.battery_capacity) * 100
                )

        soc_delta = current_soc - expected_soc_now

        if abs(soc_delta) > self.soc_deviation_threshold:
            # During CHARGE: if behind schedule (negative delta), check if we'll still reach max_soc
            # with remaining scheduled charge hours. If yes, skip recalculation - we're just
            # charging slower than expected but will still reach the target.
            if entry and entry.mode == BatteryMode.CHARGE and soc_delta < 0:
                # Calculate remaining charge capacity from all scheduled charge hours
                # Account for temperature warming: battery heats up during charging,
                # which may cause the inverter to switch to a higher charge rate
                remaining_charge_energy = 0.0
                projected_temp = current_temp if current_temp is not None else 15.0  # Default assumption
                effective_rate_for_log = self.charge_rate

                # Determine the temperature threshold where charge rate increases
                # (typically around 16°C for many inverters)
                temp_threshold = 16.0

                # Helper for timezone-safe datetime comparison
                def compare_hours(h1, h2):
                    """Compare two datetimes, handling mixed timezone-aware/naive."""
                    cmp_h1, cmp_h2 = h1, h2
                    if local_tz is not None:
                        if h1.tzinfo is not None:
                            cmp_h1 = h1.astimezone(local_tz)
                        if h2.tzinfo is not None:
                            cmp_h2 = h2.astimezone(local_tz)
                    # Handle mixed aware/naive by stripping tzinfo
                    if cmp_h1.tzinfo is not None and cmp_h2.tzinfo is None:
                        cmp_h1 = cmp_h1.replace(tzinfo=None)
                    elif cmp_h1.tzinfo is None and cmp_h2.tzinfo is not None:
                        cmp_h2 = cmp_h2.replace(tzinfo=None)
                    return cmp_h1, cmp_h2

                for future_hour in sorted(self.schedule.keys()):
                    cmp_future, cmp_current = compare_hours(future_hour, current_slot)
                    if cmp_future >= cmp_current:
                        future_entry = self.schedule.get(future_hour)
                        if future_entry and future_entry.mode == BatteryMode.CHARGE:
                            # For current slot, only count remaining time
                            if cmp_future == cmp_current:
                                remaining_minutes = (1.0 - fraction) * self.slot_minutes
                            else:
                                remaining_minutes = self.slot_minutes

                            # Use warming-aware projection if learning engine has the data
                            if self.learning_engine and current_temp is not None:
                                energy, projected_temp = self.learning_engine.predict_charge_energy_with_warming(
                                    current_soc=current_soc,
                                    start_temp=projected_temp,
                                    duration_minutes=remaining_minutes,
                                    temp_threshold=temp_threshold
                                )
                                remaining_charge_energy += energy * self.efficiency
                                effective_rate_for_log = energy / (remaining_minutes / 60) if remaining_minutes > 0 else 0
                            else:
                                # Fallback: use simple rate-based calculation
                                effective_charge_rate = self.charge_rate
                                if self.learning_engine:
                                    learned_rate = self.learning_engine.get_charge_rate_for_soc(current_soc, projected_temp)
                                    if learned_rate is not None and learned_rate > 0:
                                        effective_charge_rate = learned_rate
                                remaining_charge_energy += effective_charge_rate * self.efficiency * (remaining_minutes / 60)
                                effective_rate_for_log = effective_charge_rate

                remaining_soc_gain = (remaining_charge_energy / self.battery_capacity) * 100
                projected_final_soc = current_soc + remaining_soc_gain

                # If we'll still reach max_soc (with 5% tolerance), don't recalculate
                if projected_final_soc >= self.max_soc - 5:
                    temp_info = f", temp={current_temp:.1f}C->~{projected_temp:.1f}C" if current_temp is not None else ""
                    self.log(
                        f"SOC behind by {abs(soc_delta):.1f}% during CHARGE (actual={current_soc:.1f}%, "
                        f"expected={expected_soc_now:.1f}%), but projected to reach {projected_final_soc:.1f}% "
                        f"with remaining charge hours (rate~{effective_rate_for_log:.2f}kW{temp_info}) - skipping recalculation"
                    )
                    return False

            # During DISCHARGE: if ahead of schedule (positive delta means draining slower),
            # this is actually favorable - we have more energy than expected. Only recalculate
            # if significantly ahead, as this might indicate load predictions are off.
            if entry and entry.mode == BatteryMode.DISCHARGE and soc_delta > 0:
                # Being ahead during discharge is good - we have more buffer than expected
                # Only recalculate if very significantly ahead (2x threshold) to update load predictions
                if soc_delta <= self.soc_deviation_threshold * 2:
                    self.log(
                        f"SOC ahead by {soc_delta:.1f}% during DISCHARGE (actual={current_soc:.1f}%, "
                        f"expected={expected_soc_now:.1f}%) - favorable deviation, skipping recalculation"
                    )
                    return False

            # Store trigger context for sensor exposure
            self._last_recalc_trigger = "soc_deviation"
            self._last_recalc_time = self.datetime()
            self._last_soc_deviation = soc_delta

            # Calculate extra charge slots when behind schedule during CHARGE mode
            extra_charge_slots = 0
            if entry and entry.mode == BatteryMode.CHARGE and soc_delta < 0:
                # We're behind schedule and won't reach max_soc with current schedule
                # Calculate extra charge slots needed
                soc_deficit = self.max_soc - projected_final_soc
                energy_deficit_kwh = (soc_deficit / 100) * self.battery_capacity

                # Use learned or configured charge rate for calculation
                effective_charge_rate = self.charge_rate
                if self.learning_engine:
                    learned_rate = self.learning_engine.get_charge_rate_for_soc(current_soc, current_temp)
                    if learned_rate is not None and learned_rate > 0:
                        effective_charge_rate = learned_rate

                energy_per_slot = effective_charge_rate * self.efficiency * self.slot_hours
                extra_slots_needed = math.ceil(energy_deficit_kwh / energy_per_slot) if energy_per_slot > 0 else 0

                if extra_slots_needed > 0:
                    # Get remaining hours for price lookup (with timezone-safe comparison)
                    def is_future_hour(h):
                        """Compare hours handling mixed timezone-aware/naive datetimes."""
                        compare_h = h
                        compare_slot = current_slot
                        if local_tz is not None:
                            if h.tzinfo is not None:
                                compare_h = h.astimezone(local_tz)
                            if current_slot.tzinfo is not None:
                                compare_slot = current_slot.astimezone(local_tz)
                        # Handle mixed timezone-aware/naive by comparing as naive
                        if compare_h.tzinfo is not None and compare_slot.tzinfo is None:
                            compare_h = compare_h.replace(tzinfo=None)
                        elif compare_h.tzinfo is None and compare_slot.tzinfo is not None:
                            compare_slot = compare_slot.replace(tzinfo=None)
                        return compare_h > compare_slot

                    remaining_hours = [h for h in sorted(self.schedule.keys()) if is_future_hour(h)]

                    # Only add extra slots if economically beneficial
                    upcoming_prices = self._get_cheapest_upcoming_prices(remaining_hours, extra_slots_needed)
                    if upcoming_prices:
                        avg_extra_charge_price = sum(upcoming_prices) / len(upcoming_prices)
                        discharge_threshold = self._get_discharge_threshold()

                        # Economic check: charging cost < what we'd pay from grid during discharge
                        # Charge price includes grid fee, threshold is already grid-aware
                        charge_cost = avg_extra_charge_price + self.grid_fee
                        if charge_cost < discharge_threshold:
                            self.log(
                                f"Charging behind schedule: projected {projected_final_soc:.1f}% vs target {self.max_soc}%, "
                                f"adding {extra_slots_needed} slot(s) at avg {avg_extra_charge_price:.4f} EUR/kWh "
                                f"(charge cost {charge_cost:.4f} < discharge threshold {discharge_threshold:.4f})"
                            )
                            extra_charge_slots = extra_slots_needed
                        else:
                            self.log(
                                f"Charging behind schedule but extra charging not economical: "
                                f"projected {projected_final_soc:.1f}% vs target {self.max_soc}%, "
                                f"avg price {avg_extra_charge_price:.4f} + fee {self.grid_fee:.4f} = {charge_cost:.4f} "
                                f">= threshold {discharge_threshold:.4f}"
                            )
                    else:
                        self.log(
                            f"Charging behind schedule: projected {projected_final_soc:.1f}% vs target {self.max_soc}%, "
                            f"but no HOLD slots available for extra charging"
                        )

            # Enhanced logging for decision transparency
            if self.decision_log_level >= 1:
                self.log("=" * 70)
                self.log("RECALCULATION TRIGGERED: SOC Deviation")
                self.log("=" * 70)
                self.log(f"  Expected SOC: {expected_soc_now:.1f}%")
                self.log(f"  Actual SOC: {current_soc:.1f}%")
                self.log(f"  Deviation: {soc_delta:+.1f}% (threshold: {self.soc_deviation_threshold}%)")
                if extra_charge_slots > 0:
                    self.log(f"  Extra charge slots requested: {extra_charge_slots}")
                self.log("=" * 70)
            else:
                self.log(f"SOC deviation detected: actual={current_soc}%, expected={expected_soc_now:.1f}%, delta={soc_delta}%")

            self._recalculate_remaining_schedule(current_soc, extra_charge_slots=extra_charge_slots)
            return True

        return False

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
                charge_state = self.get_state(self.battery_charge_sensor)
                discharge_state = self.get_state(self.battery_discharge_sensor)
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
        if success or not self.device_id:
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
            manual_mode = self.get_state(self.manual_mode_entity)
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
                    entity_id=self.override_entity
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
        """Initialize learning engine from persistent storage"""
        self.learning_data_file = self.args.get("learning_data_file", "")
        self.learning_data_entity = self.args.get("learning_data_entity", "")

        # Track timing for learning observations
        self._charge_start_soc: Optional[float] = None
        self._charge_start_time: Optional[datetime.datetime] = None
        self._discharge_start_soc: Optional[float] = None
        self._discharge_start_time: Optional[datetime.datetime] = None

        # Prefer file-based persistence if configured
        if self.learning_data_file:
            try:
                with open(self.learning_data_file, "r", encoding="utf-8") as fh:
                    data = fh.read()
                if data and self.learning_engine.load_from_json(data):
                    summary = self.learning_engine.get_learning_summary()
                    self.log(f"Loaded learning data from file: {summary['total_observations']} observations")
                    return
            except FileNotFoundError:
                pass
            except Exception as e:
                self.log(f"Could not load learning data file: {e}", level="WARNING")

        # Try to load learning data from HA
        if self.learning_data_entity:
            try:
                state = self.get_state(self.learning_data_entity)
                if state and state not in ("unknown", "unavailable", ""):
                    if self.learning_engine.load_from_json(state):
                        summary = self.learning_engine.get_learning_summary()
                        self.log(f"Loaded learning data: {summary['total_observations']} observations")
                        return
            except Exception as e:
                self.log(f"Could not load learning data: {e}", level="WARNING")

        self.log("Starting with fresh learning data")

    def _init_load_profile(self):
        """Initialize load profile from persistent storage"""
        # Prefer file-based persistence if configured
        if self.load_profile_file:
            try:
                with open(self.load_profile_file, "r", encoding="utf-8") as fh:
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
        if self.load_profile_entity:
            try:
                state = self.get_state(self.load_profile_entity)
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
                hour=current_entry.hour,
                mode=previous_mode,
                reason=f"continuing_{mode_name.lower()}_from_restart"
            )

        # Clear the previous schedule after first use (only needed for startup)
        self._previous_schedule_from_sensor = None

    def _save_load_profile(self):
        """Persist load profile to Home Assistant entity"""
        json_data = self.load_profile.to_json()

        # Prefer file-based persistence if configured
        if self.load_profile_file:
            try:
                with open(self.load_profile_file, "w", encoding="utf-8") as fh:
                    fh.write(json_data)
            except Exception as e:
                self.log(f"Could not save load profile file: {e}", level="DEBUG")

        # Optional HA entity persistence
        if self.load_profile_entity:
            try:
                self.call_service("input_text/set_value",
                    entity_id=self.load_profile_entity,
                    value=json_data
                )
            except Exception as e:
                self.log(f"Could not save load profile to {self.load_profile_entity}: {e}", level="DEBUG")

        self._update_load_profile_sensors()

    def _update_load_profile_sensors(self):
        """Update load profile status sensors in Home Assistant."""
        try:
            count = self.load_profile.stats.observation_count
            last_obs = self.load_profile.stats.last_observation or ""
            if self.load_profile_count_entity:
                self.set_state(
                    self.load_profile_count_entity,
                    state=str(count),
                    attributes={
                        "friendly_name": "Load Profile Observation Count",
                        "unit_of_measurement": "samples"
                    }
                )
            if self.load_profile_last_obs_entity:
                self.set_state(
                    self.load_profile_last_obs_entity,
                    state=last_obs,
                    attributes={
                        "friendly_name": "Load Profile Last Observation"
                    }
                )
        except Exception as e:
            self.log(f"Could not update load profile sensors: {e}", level="DEBUG")

    def record_load_observation(self, kwargs=None):
        """Record current house load into the statistical load profile."""
        if not self.load_power_sensor:
            return
        load_w = self._get_load_power()
        if load_w is None:
            return
        now = self._align_to_slot(self.datetime())
        self.load_profile.record(now, load_w)
        self._save_load_profile()

    def _save_learning_data(self):
        """Persist learning data to Home Assistant entity"""
        try:
            json_data = self.learning_engine.save_to_json()
            if self.learning_data_file:
                try:
                    with open(self.learning_data_file, "w", encoding="utf-8") as fh:
                        fh.write(json_data)
                except Exception as e:
                    self.log(f"Could not save learning data file: {e}", level="DEBUG")

            # Optional HA entity persistence (limited to 255 chars)
            if self.learning_data_entity:
                if len(json_data) <= 255:
                    self.call_service("input_text/set_value",
                        entity_id=self.learning_data_entity,
                        value=json_data
                    )
                else:
                    self.log("Learning data exceeds 255 chars; skipping input_text persistence", level="DEBUG")
        except Exception as e:
            self.log(f"Could not save learning data: {e}", level="DEBUG")

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
        return self._sensors.get_soc(self.soc_sensor)

    def _get_pv_power(self) -> float:
        """Get current PV power production."""
        return self._sensors.get_power(self.pv_power_sensor, default=0.0)

    def _get_battery_temp(self) -> Optional[float]:
        """Get current battery temperature in Celsius."""
        return self._sensors.get_temperature(self.battery_temp_sensor)

    def _get_load_power(self) -> Optional[float]:
        """Get current household load in Watts (from configured sensor)."""
        if not self.load_power_sensor:
            return None
        load_w = self._sensors.get_float(self.load_power_sensor)
        if load_w is None:
            return None
        if load_w <= 0:
            # Use last known value or floor when sensor reports zero
            if self._last_nonzero_load_w is not None:
                return max(self._last_nonzero_load_w, self.load_zero_floor_w)
            return self.load_zero_floor_w
        self._last_nonzero_load_w = load_w
        return load_w

    def _predict_load_kw(self, dt: datetime.datetime) -> float:
        """Predict expected load (kW) for a slot using load profile."""
        if self.load_profile:
            predicted = self.load_profile.predict_kw(dt, self.load_quantile)
        else:
            predicted = self.base_consumption / 1000.0

        return predicted

    def _align_to_slot(self, dt: datetime.datetime) -> datetime.datetime:
        """Floor datetime to the start of the current time slot."""
        return align_to_slot(dt, self.slot_minutes, self._get_local_timezone())

    def _get_expected_soc_for_hour(
        self,
        expected_soc: Dict[datetime.datetime, float],
        hour: datetime.datetime,
        local_tz
    ) -> Optional[float]:
        """Get expected SOC for a specific hour, handling timezone differences."""
        return lookup_by_hour(expected_soc, hour, local_tz)

    def _get_expected_temp_for_hour(
        self,
        expected_temp: Dict[datetime.datetime, float],
        hour: datetime.datetime,
        local_tz
    ) -> Optional[float]:
        """Get expected temperature for a specific hour, handling timezone differences."""
        return lookup_by_hour(expected_temp, hour, local_tz)

    def _get_dp_trajectory_for_hour(
        self,
        trajectory: Dict[datetime.datetime, Tuple[float, float]],
        hour: datetime.datetime,
        local_tz
    ) -> Optional[Tuple[float, float]]:
        """Get DP trajectory data (start, end) tuple for a specific hour, handling timezone differences."""
        return lookup_by_hour(trajectory, hour, local_tz)

    def _next_slot_time(self) -> datetime.datetime:
        """Get the next slot boundary time."""
        return next_slot_time(self.datetime(), self.slot_minutes, self._get_local_timezone())

    def _next_interval_time(self, interval_minutes: int) -> datetime.datetime:
        """Get the next boundary time for a given interval."""
        return next_interval_time(self.datetime(), interval_minutes, self._get_local_timezone())

    def _is_enabled(self) -> bool:
        """Check if optimizer is enabled."""
        return self._sensors.is_on(self.enabled_entity, default=True)

    def _is_override_active(self) -> bool:
        """Check if manual override is active."""
        return self._sensors.is_on(self.override_entity, default=False)

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
            state = self.get_state(self.min_soc_entity)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return self._default_min_soc

    @property
    def max_soc(self) -> float:
        """Get max SOC from HA entity or default"""
        try:
            state = self.get_state(self.max_soc_entity)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return self._default_max_soc

    @property
    def pv_threshold(self) -> float:
        """Get PV threshold from HA entity or default"""
        try:
            state = self.get_state(self.pv_threshold_entity)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return self._default_pv_threshold

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
        slots_per_hour = max(1, 60 // self.slot_minutes)

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
            # Format schedule for sensor
            schedule_data = []
            for hour in sorted(self.schedule.keys()):
                entry = self.schedule[hour]
                schedule_data.append({
                    "time": hour.isoformat(),
                    "mode": entry.mode.name,
                    "reason": entry.reason
                })

            # Find next charge/discharge times
            now = self.datetime()
            local_tz = self._get_local_timezone()
            # Convert now to local timezone
            if now.tzinfo is not None and local_tz is not None:
                now = now.astimezone(local_tz)
            next_charge = None
            next_discharge = None

            for hour in sorted(self.schedule.keys()):
                # Convert both to local timezone for comparison
                compare_hour = hour
                compare_now = now
                if local_tz is not None:
                    if hour.tzinfo is not None:
                        compare_hour = hour.astimezone(local_tz)
                    if compare_now.tzinfo is not None:
                        compare_now = compare_now.astimezone(local_tz)
                # Handle mixed timezone-aware/naive
                if compare_hour.tzinfo is not None and compare_now.tzinfo is None:
                    compare_hour = compare_hour.replace(tzinfo=None)
                elif compare_hour.tzinfo is None and compare_now.tzinfo is not None:
                    compare_now = compare_now.replace(tzinfo=None)
                if compare_hour < compare_now:
                    continue
                entry = self.schedule[hour]
                if entry.mode == BatteryMode.CHARGE and next_charge is None:
                    next_charge = hour.isoformat()
                if entry.mode == BatteryMode.DISCHARGE and next_discharge is None:
                    next_discharge = hour.isoformat()

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
                    "friendly_name": "Battery Optimizer"
                }
            )
        except Exception as e:
            self.log(f"Error updating schedule sensor: {e}", level="WARNING")

    # =========================================================================
    # Statistics and Logging
    # =========================================================================

    def _log_schedule(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        expected_soc: Optional[Dict[datetime.datetime, float]] = None,
        expected_temp: Optional[Dict[datetime.datetime, float]] = None
    ):
        """Log the full schedule in a readable format with optional expected SOC and temperature.

        Prefers the DP optimizer's actual SOC trajectory (_last_dp_soc_trajectory) when available,
        as this reflects the exact values the optimizer computed. Falls back to expected_soc
        (from calculate_expected_soc_schedule) for backwards compatibility.
        """
        if not schedule:
            self.log("No schedule to log")
            return

        self.log("=" * 60)
        self.log("GENERATED SCHEDULE")
        self.log("=" * 60)

        local_tz = self._get_local_timezone()
        sorted_hours = sorted(schedule.keys())

        # Prefer DP trajectory if available (more accurate to what optimizer computed)
        dp_soc_trajectory = getattr(self, '_last_dp_soc_trajectory', None)
        dp_temp_trajectory = getattr(self, '_last_dp_temp_trajectory', None)

        for i, hour in enumerate(sorted_hours):
            entry = schedule[hour]
            # Ensure time is displayed in local timezone
            display_hour = hour
            if hour.tzinfo is not None and local_tz is not None:
                display_hour = hour.astimezone(local_tz)
            time_str = display_hour.strftime("%Y-%m-%d %H:%M")
            mode_str = entry.mode.name.ljust(9)

            # Get SOC and temperature values
            soc_str = ""

            # Try DP trajectory first (exact values from optimizer)
            dp_soc_data = self._get_dp_trajectory_for_hour(dp_soc_trajectory, hour, local_tz) if dp_soc_trajectory else None
            dp_temp_data = self._get_dp_trajectory_for_hour(dp_temp_trajectory, hour, local_tz) if dp_temp_trajectory else None

            if dp_soc_data is not None:
                # Use DP's actual SOC trajectory
                start_soc, end_soc = dp_soc_data
                if dp_temp_data is not None:
                    start_temp, end_temp = dp_temp_data
                    if start_temp is not None and end_temp is not None:
                        soc_str = f" {start_soc:5.1f}%->{end_soc:5.1f}% ({start_temp:.0f}C->{end_temp:.0f}C)"
                    else:
                        soc_str = f" {start_soc:5.1f}%->{end_soc:5.1f}%"
                else:
                    soc_str = f" {start_soc:5.1f}%->{end_soc:5.1f}%"
            elif expected_soc:
                # Fallback to recalculated SOC (for backwards compatibility)
                start_soc = self._get_expected_soc_for_hour(expected_soc, hour, local_tz)
                start_temp = self._get_expected_temp_for_hour(expected_temp, hour, local_tz) if expected_temp else None

                if start_soc is not None:
                    # Calculate end-of-slot SOC based on the action
                    if entry.mode == BatteryMode.CHARGE:
                        # Use temperature-aware charging if temp available
                        if start_temp is not None and self.learning_engine:
                            energy_added, end_temp = self.learning_engine.predict_charge_energy_with_warming(
                                start_soc, start_temp, self.slot_minutes, temp_threshold=16.0
                            )
                            energy_to_battery = energy_added * self.efficiency
                            end_soc = min(self.max_soc, start_soc + (energy_to_battery / self.battery_capacity) * 100)
                            soc_str = f" {start_soc:5.1f}%->{end_soc:5.1f}% ({start_temp:.0f}C->{end_temp:.0f}C)"
                        else:
                            # Fallback: Use learned charge rate without temperature
                            effective_charge_rate = self.charge_rate
                            if self.learning_engine:
                                learned_rate = self.learning_engine.get_charge_rate_for_soc(start_soc)
                                if learned_rate is not None and learned_rate > 0:
                                    effective_charge_rate = learned_rate
                            energy_added = effective_charge_rate * self.efficiency * self.slot_hours
                            end_soc = min(self.max_soc, start_soc + (energy_added / self.battery_capacity) * 100)
                            soc_str = f" {start_soc:5.1f}%->{end_soc:5.1f}%"
                    elif entry.mode == BatteryMode.DISCHARGE:
                        load_kw = self._predict_load_kw(hour)
                        energy_removed = min(load_kw, self.discharge_rate) * self.slot_hours
                        end_soc = max(self.min_soc, start_soc - (energy_removed / self.battery_capacity) * 100)
                        # Show cooling during discharge if temp available
                        if start_temp is not None and self.learning_engine:
                            end_temp = self.learning_engine.predict_temp_after_idle(
                                start_temp, self.slot_minutes
                            )
                            soc_str = f" {start_soc:5.1f}%->{end_soc:5.1f}% ({start_temp:.0f}C->{end_temp:.0f}C)"
                        else:
                            soc_str = f" {start_soc:5.1f}%->{end_soc:5.1f}%"
                    else:  # HOLD
                        end_soc = start_soc
                        # Show cooling during hold if temp available
                        if start_temp is not None and self.learning_engine:
                            end_temp = self.learning_engine.predict_temp_after_idle(
                                start_temp, self.slot_minutes
                            )
                            soc_str = f" {start_soc:5.1f}%->{end_soc:5.1f}% ({start_temp:.0f}C->{end_temp:.0f}C)"
                        else:
                            soc_str = f" {start_soc:5.1f}%->{end_soc:5.1f}%"

            # For discharge, show projected battery cost as primary, grid price in parentheses
            reason_display = entry.reason
            if entry.mode == BatteryMode.DISCHARGE and self._last_projected_costs:
                proj_cost = self._last_projected_costs.get(hour)
                if proj_cost is not None:
                    # Parse grid price from reason: "X.XXXX EUR/kWh load~Y.YYkW"
                    parts = entry.reason.split(" EUR/kWh")
                    if len(parts) == 2:
                        grid_price = parts[0]
                        rest = parts[1]
                        reason_display = f"{proj_cost:.4f} EUR/kWh (grid {grid_price}){rest}"

            self.log(f"  {time_str}  {mode_str}  {reason_display}{soc_str}")

        self.log("=" * 60)

        # Summary counts
        charge_count = len([e for e in schedule.values() if e.mode == BatteryMode.CHARGE])
        discharge_count = len([e for e in schedule.values() if e.mode == BatteryMode.DISCHARGE])
        hold_count = len([e for e in schedule.values() if e.mode == BatteryMode.HOLD])
        self.log(f"Total: {charge_count} charge, {discharge_count} discharge, {hold_count} hold slots")

    def get_schedule_summary(self) -> str:
        """Generate a human-readable schedule summary"""
        if not self.schedule:
            return "No schedule available"

        now = self.datetime()
        local_tz = self._get_local_timezone()
        # Convert now to local timezone
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        now_slot = self._align_to_slot(now)

        def is_future_or_current(k):
            compare_k = k
            compare_now = now_slot
            if local_tz is not None:
                if k.tzinfo is not None:
                    compare_k = k.astimezone(local_tz)
                if compare_now.tzinfo is not None:
                    compare_now = compare_now.astimezone(local_tz)
            # Handle mixed timezone-aware/naive
            if compare_k.tzinfo is not None and compare_now.tzinfo is None:
                compare_k = compare_k.replace(tzinfo=None)
            elif compare_k.tzinfo is None and compare_now.tzinfo is not None:
                compare_now = compare_now.replace(tzinfo=None)
            return compare_k >= compare_now

        future_schedule = {k: v for k, v in self.schedule.items() if is_future_or_current(k)}

        charge_hours = [h for h, e in future_schedule.items() if e.mode == BatteryMode.CHARGE]
        discharge_hours = [h for h, e in future_schedule.items() if e.mode == BatteryMode.DISCHARGE]
        hold_hours = [h for h, e in future_schedule.items() if e.mode == BatteryMode.HOLD]

        summary = (f"Schedule: {len(charge_hours)} slots charge, "
                   f"{len(discharge_hours)} slots discharge, "
                   f"{len(hold_hours)} slots hold")

        local_tz = self._get_local_timezone()
        if charge_hours:
            next_charge = min(charge_hours)
            if next_charge.tzinfo is not None and local_tz is not None:
                next_charge = next_charge.astimezone(local_tz)
            summary += f"\nNext charge: {next_charge.strftime('%H:%M')}"
        if discharge_hours:
            next_discharge = min(discharge_hours)
            if next_discharge.tzinfo is not None and local_tz is not None:
                next_discharge = next_discharge.astimezone(local_tz)
            summary += f"\nNext discharge: {next_discharge.strftime('%H:%M')}"

        return summary
