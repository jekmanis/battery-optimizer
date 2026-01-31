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

        # Internal state
        self.current_mode: BatteryMode = BatteryMode.HOLD
        self.schedule: Dict[datetime.datetime, ScheduleEntry] = {}
        self.last_optimization: Optional[datetime.datetime] = None
        self.expected_soc_schedule: Dict[datetime.datetime, float] = {}
        self.expected_temp_schedule: Dict[datetime.datetime, float] = {}
        self._last_nonzero_load_w: Optional[float] = None
        self._previous_schedule_from_sensor: Optional[Dict[datetime.datetime, BatteryMode]] = None

        # Battery cost tracking (weighted average cost of energy in battery)
        self.battery_avg_cost: float = 0.0  # EUR/kWh
        self._init_battery_cost()

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
        if now.tzinfo is None and current_slot.tzinfo is not None:
            now = now.replace(tzinfo=current_slot.tzinfo)
        elif now.tzinfo is not None and current_slot.tzinfo is None:
            current_slot = current_slot.replace(tzinfo=now.tzinfo)

        def is_future_price(p):
            p_hour = p.hour
            compare_time = current_slot
            if p_hour.tzinfo is not None and compare_time.tzinfo is None:
                p_hour = p_hour.replace(tzinfo=None)
            elif p_hour.tzinfo is None and compare_time.tzinfo is not None:
                compare_time = compare_time.replace(tzinfo=None)
            return p_hour >= compare_time

        future_prices = [p for p in prices if is_future_price(p)]
        if not future_prices:
            return {}

        # Ensure current slot is included (Nord Pool may exclude current hour as "past")
        def prices_contains_slot(prices_list, slot):
            slot_naive = slot.replace(tzinfo=None) if slot.tzinfo else slot
            for p in prices_list:
                p_naive = p.hour.replace(tzinfo=None) if p.hour.tzinfo else p.hour
                if p_naive == slot_naive:
                    return True
            return False

        if not prices_contains_slot(future_prices, current_slot):
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
                for p in prices:
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
            future_prices.append(current_slot_price)

        schedule = {}
        hours_sorted_by_time = sorted(future_prices, key=lambda p: p.hour)
        n_slots = len(hours_sorted_by_time)
        min_charge_slots = max(0, int(charge_hours_needed))
        if min_charge_slots > n_slots:
            min_charge_slots = n_slots
        # Precompute which slots are "favorable" for charging (raw price basis)
        charge_price_threshold = self.battery_avg_cost * 1.05
        discharge_threshold = self._get_discharge_threshold()
        favorable_flags = [p.price <= charge_price_threshold for p in hours_sorted_by_time]
        favorable_remaining = [0] * n_slots
        remaining = 0
        for i in range(n_slots - 1, -1, -1):
            if favorable_flags[i]:
                remaining += 1
            favorable_remaining[i] = remaining

        # Energy bounds in kWh
        current_soc_for_calc = current_soc if current_soc is not None else 50.0
        min_energy = (self.min_soc / 100) * self.battery_capacity
        max_energy = (self.max_soc / 100) * self.battery_capacity
        start_energy = min(max_energy, max(min_energy, (current_soc_for_calc / 100) * self.battery_capacity))

        # DP resolution
        step_kwh = max(0.01, (self.soc_step_percent / 100) * self.battery_capacity)
        n_states = int(round((max_energy - min_energy) / step_kwh)) + 1
        energy_levels = [min_energy + i * step_kwh for i in range(n_states)]

        # Per-slot energy changes (adjust first slot if partial)
        minutes_into_slot = max(0.0, (now - current_slot).total_seconds() / 60.0)
        first_fraction = min(1.0, max(0.0, (self.slot_minutes - minutes_into_slot) / max(1, self.slot_minutes)))
        slot_fractions = [1.0] * n_slots
        current_slot_index = None
        for i, p in enumerate(hours_sorted_by_time):
            p_hour = p.hour
            compare_current = current_slot
            if p_hour.tzinfo is not None and compare_current.tzinfo is None:
                p_hour = p_hour.replace(tzinfo=None)
            elif p_hour.tzinfo is None and compare_current.tzinfo is not None:
                compare_current = compare_current.replace(tzinfo=None)
            if p_hour == compare_current:
                slot_fractions[i] = first_fraction
                current_slot_index = i
                break

        # Get temperature-aware charge rate if temperature sensor is available
        # This improves scheduling accuracy for batteries that charge slower when cold
        current_temp = self._get_battery_temp()
        learned_charge_rate = self.learning_engine.get_charge_rate_for_soc(
            current_soc_for_calc, current_temp
        )

        # Pre-compute temperature-aware charge rates for each slot.
        # Project temperature evolution assuming continuous charging to estimate warming trajectory.
        # This gives the DP optimizer accurate rate estimates for later slots.
        # Important: Use actual slot duration (accounting for partial first slot) for accurate warming.
        charge_rates_per_slot = []
        if current_temp is not None and self.learning_engine:
            projected_temp = current_temp
            for i, price_point in enumerate(hours_sorted_by_time):
                # Get rate for this slot's projected temperature
                rate = self.learning_engine.get_charge_rate_for_soc(
                    current_soc_for_calc, projected_temp
                )
                charge_rates_per_slot.append(rate)
                # Project temperature after this slot (assuming charging to estimate warming)
                # Use actual slot duration: partial first slot, full for rest
                slot_duration_minutes = self.slot_minutes * slot_fractions[i]
                projected_temp = self.learning_engine.predict_temp_after_duration(
                    projected_temp, slot_duration_minutes
                )
        else:
            # Fallback: use single learned rate for all slots (existing behavior)
            charge_rates_per_slot = [learned_charge_rate for _ in hours_sorted_by_time]

        # Use learned rate for base calculation (falls back to nominal if no data)
        base_charge_rate = learned_charge_rate
        base_charge_energy_kwh = base_charge_rate * self.efficiency * self.slot_hours
        base_charge_cost_kwh = base_charge_rate * self.slot_hours
        load_kw = [self._predict_load_kw(p.hour) for p in hours_sorted_by_time]
        discharge_energy_kwh = [
            min(lk, self.discharge_rate) * self.slot_hours * slot_fractions[i]
            for i, lk in enumerate(load_kw)
        ]

        # DP helpers
        neg_inf = -1e18
        max_charge_slots = n_slots
        # Tiny tie-breakers to prefer cheaper/later charge slots when values are effectively equal
        tie_val_eps = 1e-6
        tie_tie_eps = 1e-12
        tie_price_weight = 1e-5
        tie_time_weight = 1e-7

        def _compute_favorable(prices_list: List[PricePoint]):
            flags = [p.price <= charge_price_threshold for p in prices_list]
            remaining_list = [0] * len(prices_list)
            remaining_count = 0
            for i in range(len(prices_list) - 1, -1, -1):
                if flags[i]:
                    remaining_count += 1
                remaining_list[i] = remaining_count
            return flags, remaining_list

        favorable_flags, favorable_remaining = _compute_favorable(hours_sorted_by_time)

        def _run_dp(
            hours_list: List[PricePoint],
            load_kw_list: List[float],
            charge_rates_list: List[float],
            slot_fractions_list: List[float],
            favorable_flags_list: List[bool],
            favorable_remaining_list: List[int],
            start_energy_kwh: float,
            start_c: int,
            discharge_thresholds_list: Optional[List[float]] = None,
            start_idx_override: Optional[int] = None,
        ) -> Tuple[List[BatteryMode], List[bool], float, bool, List[int]]:
            n_list_slots = len(hours_list)
            if n_list_slots == 0:
                return [], [], 0.0, True, []

            dp = [[neg_inf] * n_states for _ in range(max_charge_slots + 1)]
            dp_tie = [[neg_inf] * n_states for _ in range(max_charge_slots + 1)]
            if start_idx_override is None:
                start_idx_local = int(round((start_energy_kwh - min_energy) / step_kwh))
            else:
                start_idx_local = start_idx_override
            start_idx_local = min(max(start_idx_local, 0), n_states - 1)
            start_c = min(max(start_c, 0), max_charge_slots)
            dp[start_c][start_idx_local] = 0.0
            dp_tie[start_c][start_idx_local] = 0.0

            prev_idx = [[[None] * n_states for _ in range(max_charge_slots + 1)] for _ in range(n_list_slots)]
            prev_c = [[[None] * n_states for _ in range(max_charge_slots + 1)] for _ in range(n_list_slots)]
            prev_action = [[[None] * n_states for _ in range(max_charge_slots + 1)] for _ in range(n_list_slots)]
            prev_partial = [[[False] * n_states for _ in range(max_charge_slots + 1)] for _ in range(n_list_slots)]

            def _should_update(curr_val: float, curr_tie: float, cand_val: float, cand_tie: float) -> bool:
                if cand_val > curr_val + tie_val_eps:
                    return True
                if abs(cand_val - curr_val) <= tie_val_eps and cand_tie > curr_tie + tie_tie_eps:
                    return True
                return False

            # Collect DP trace info for diagnostic logging
            dp_trace_slots = []  # Will store (slot_hour, traces) for interesting slots

            for t in range(n_list_slots):
                price = hours_list[t].price
                buy_price = price + self.grid_fee
                slot_discharge_threshold = (
                    discharge_thresholds_list[t]
                    if discharge_thresholds_list is not None
                    else discharge_threshold
                )
                discharge_allowed = buy_price >= (slot_discharge_threshold - 1e-6)
                fraction = slot_fractions_list[t]
                discharge_kwh = min(load_kw_list[t], self.discharge_rate) * self.slot_hours * fraction
                slot_charge_rate = charge_rates_list[t]
                charge_energy_kwh = slot_charge_rate * self.efficiency * self.slot_hours * fraction
                charge_cost_kwh = slot_charge_rate * self.slot_hours * fraction
                charge_count_increment = 1  # Always count - energy is tracked separately

                next_dp = [[neg_inf] * n_states for _ in range(max_charge_slots + 1)]
                next_dp_tie = [[neg_inf] * n_states for _ in range(max_charge_slots + 1)]
                next_prev_idx = [[None] * n_states for _ in range(max_charge_slots + 1)]
                next_prev_c = [[None] * n_states for _ in range(max_charge_slots + 1)]
                next_prev_action = [[None] * n_states for _ in range(max_charge_slots + 1)]
                next_prev_partial = [[False] * n_states for _ in range(max_charge_slots + 1)]

                # Trace info for this slot (for diagnostic logging)
                slot_trace = []
                trace_this_slot = (
                    self.decision_log_level >= 3 and
                    discharge_allowed and
                    fraction > 0.5  # Full slots only
                )

                # Deep trace: track paths from discharge-allowed slots
                # Tag states that came from a discharge so we can follow their evolution
                deep_trace_this_slot = self.decision_log_level >= 3 and t < 5

                for c in range(max_charge_slots + 1):
                    for idx, val in enumerate(dp[c]):
                        if val <= neg_inf / 2:
                            continue
                        curr_tie = dp_tie[c][idx]
                        curr_soc = self.min_soc + (idx * step_kwh / self.battery_capacity) * 100

                        # HOLD - costs grid price for load (we import from grid during HOLD)
                        hold_updated = False
                        hold_cost = buy_price * discharge_kwh  # discharge_kwh = load consumption
                        hold_val = val - hold_cost
                        if _should_update(next_dp[c][idx], next_dp_tie[c][idx], hold_val, curr_tie):
                            next_dp[c][idx] = hold_val
                            next_dp_tie[c][idx] = curr_tie
                            next_prev_idx[c][idx] = idx
                            next_prev_c[c][idx] = c
                            next_prev_action[c][idx] = BatteryMode.HOLD
                            hold_updated = True

                        # Track DISCHARGE attempt for tracing
                        discharge_attempted = False
                        discharge_updated = False
                        discharge_blocked_reason = None
                        discharge_next_idx = None
                        discharge_next_val = None

                        # CHARGE
                        price_is_favorable = favorable_flags_list[t]
                        must_use_unfavorable = (c + favorable_remaining_list[t]) < min_charge_slots
                        allow_charge = price_is_favorable or must_use_unfavorable

                        if allow_charge and charge_energy_kwh > 0 and c + charge_count_increment <= max_charge_slots:
                            new_energy = energy_levels[idx] + charge_energy_kwh
                            actual_charge_energy = charge_energy_kwh
                            actual_charge_cost = charge_cost_kwh

                            if new_energy > max_energy + 1e-6:
                                headroom = max_energy - energy_levels[idx]
                                if headroom >= step_kwh:
                                    actual_charge_energy = headroom
                                    actual_charge_cost = headroom / self.efficiency
                                    new_energy = max_energy
                                else:
                                    actual_charge_energy = 0

                            if actual_charge_energy > 0:
                                next_idx = int(round((new_energy - min_energy) / step_kwh))
                                next_idx = min(max(next_idx, 0), n_states - 1)
                                # Charge cost + load cost (must import both from grid)
                                next_val = val - (buy_price * actual_charge_cost) - (buy_price * discharge_kwh)
                                charge_tie_bias = (-price * tie_price_weight) + (t * tie_time_weight)
                                next_tie = curr_tie + charge_tie_bias
                                c_next = c + charge_count_increment
                                if _should_update(next_dp[c_next][next_idx], next_dp_tie[c_next][next_idx], next_val, next_tie):
                                    next_dp[c_next][next_idx] = next_val
                                    next_dp_tie[c_next][next_idx] = next_tie
                                    next_prev_idx[c_next][next_idx] = idx
                                    next_prev_c[c_next][next_idx] = c
                                    next_prev_action[c_next][next_idx] = BatteryMode.CHARGE

                        # DISCHARGE
                        if discharge_allowed and discharge_kwh > 0:
                            discharge_attempted = True
                            new_energy = energy_levels[idx] - discharge_kwh
                            is_partial = False
                            actual_discharge_kwh = discharge_kwh

                            if new_energy >= min_energy - 1e-6:
                                # Full discharge - battery can cover entire load
                                # Value = avoided grid import (no cost, vs HOLD which pays buy_price * load)
                                next_idx = int(round((new_energy - min_energy) / step_kwh))
                                next_idx = min(max(next_idx, 0), n_states - 1)
                                next_val = val  # No grid cost - battery covers load
                            elif energy_levels[idx] > min_energy + discharge_kwh * 0.5:
                                # Partial discharge - discharge until min_soc, grid covers remainder
                                # Only allow if we have at least half the load's worth of energy
                                is_partial = True
                                actual_discharge_kwh = energy_levels[idx] - min_energy
                                grid_import = discharge_kwh - actual_discharge_kwh
                                # Value = avoided import for battery portion, pay grid for remainder
                                next_val = val - (buy_price * grid_import)
                                next_idx = 0  # End up at min_energy
                                new_energy = min_energy
                            else:
                                discharge_blocked_reason = f"would_hit_min_soc ({new_energy:.2f} < {min_energy:.2f})"
                                next_val = None
                                next_idx = None

                            if next_val is not None:
                                discharge_next_idx = next_idx
                                discharge_next_val = next_val
                                if _should_update(next_dp[c][next_idx], next_dp_tie[c][next_idx], next_val, curr_tie):
                                    next_dp[c][next_idx] = next_val
                                    next_dp_tie[c][next_idx] = curr_tie
                                    next_prev_idx[c][next_idx] = idx
                                    next_prev_c[c][next_idx] = c
                                    next_prev_action[c][next_idx] = BatteryMode.DISCHARGE
                                    next_prev_partial[c][next_idx] = is_partial
                                    discharge_updated = True
                                else:
                                    # Discharge transition was blocked - another path was better
                                    discharge_blocked_reason = (
                                        f"existing_val={next_dp[c][next_idx]:.4f} >= discharge_val={next_val:.4f}"
                                    )

                        # Collect trace for this state if interesting
                        # Log all charge counts for high-SOC states (to debug post-charge HOLD issues)
                        trace_high_soc = curr_soc >= 95.0 and c >= min_charge_slots
                        if trace_this_slot and discharge_attempted and (c == 0 or trace_high_soc):
                            next_soc_discharge = self.min_soc + (discharge_next_idx * step_kwh / self.battery_capacity) * 100 if discharge_next_idx is not None else 0
                            slot_trace.append({
                                "charge_count": c,
                                "from_soc": curr_soc,
                                "from_idx": idx,
                                "from_val": val,
                                "hold_val": hold_val,
                                "hold_cost": hold_cost,
                                "hold_updated": hold_updated,
                                "discharge_attempted": discharge_attempted,
                                "discharge_updated": discharge_updated,
                                "discharge_blocked": discharge_blocked_reason,
                                "discharge_to_soc": next_soc_discharge,
                                "discharge_to_idx": discharge_next_idx,
                                "discharge_val": discharge_next_val,
                            })

                # Store trace for this slot
                if trace_this_slot and slot_trace:
                    dp_trace_slots.append((hours_list[t].hour, price, slot_trace))

                # Deep trace: log state evolution for first few slots
                if deep_trace_this_slot:
                    self.log(f"[DeepTrace] After slot {t} ({hours_list[t].hour.strftime('%H:%M')} @ {price:.4f}):")
                    # Show states with valid values, grouped by charge count
                    for c in range(min(3, max_charge_slots + 1)):
                        active_states = [
                            (i, next_dp[c][i], next_prev_action[c][i])
                            for i in range(n_states)
                            if next_dp[c][i] > neg_inf / 2
                        ]
                        if active_states:
                            # Show top 3 by value
                            active_states.sort(key=lambda x: x[1], reverse=True)
                            top_states = active_states[:3]
                            self.log(f"  c={c}: " + ", ".join(
                                f"idx={i} ({self.min_soc + i*step_kwh/self.battery_capacity*100:.1f}%) val={v:.4f} via {a.name if a else 'None'}"
                                for i, v, a in top_states
                            ))

                dp = next_dp
                dp_tie = next_dp_tie
                prev_idx[t] = next_prev_idx
                prev_c[t] = next_prev_c
                prev_action[t] = next_prev_action
                prev_partial[t] = next_prev_partial

            best_val = neg_inf
            best_tie = neg_inf
            best_idx = None
            best_c = None
            max_charge_achieved = 0
            for c in range(max_charge_slots + 1):
                for i in range(n_states):
                    if dp[c][i] > neg_inf / 2:
                        if c > max_charge_achieved:
                            max_charge_achieved = c
                        if c >= min_charge_slots and _should_update(best_val, best_tie, dp[c][i], dp_tie[c][i]):
                            best_val = dp[c][i]
                            best_tie = dp_tie[c][i]
                            best_idx = i
                            best_c = c

            meets_min = True
            if best_idx is None:
                meets_min = False
                for c in range(max_charge_slots + 1):
                    for i in range(n_states):
                        if _should_update(best_val, best_tie, dp[c][i], dp_tie[c][i]):
                            best_val = dp[c][i]
                            best_tie = dp_tie[c][i]
                            best_idx = i
                            best_c = c
                self.log(
                    f"Minimum charge slots not achievable (required {min_charge_slots}, achieved {max_charge_achieved})",
                    level="WARNING",
                )

            actions: List[BatteryMode] = []
            partial_flags: List[bool] = []
            idx_trajectory: List[int] = []  # Track idx (energy level) at end of each slot
            idx = best_idx if best_idx is not None else start_idx_local
            c = best_c if best_c is not None else start_c

            # Deep trace: log backtracking
            if self.decision_log_level >= 3:
                self.log(f"[DeepTrace] Backtracking from best final state: c={c}, idx={idx}, val={best_val:.4f}")

            backtrack_trace = []
            for t in range(n_list_slots - 1, -1, -1):
                action = prev_action[t][c][idx] or BatteryMode.HOLD
                is_partial = prev_partial[t][c][idx] if action == BatteryMode.DISCHARGE else False
                actions.append(action)
                partial_flags.append(is_partial)
                idx_trajectory.append(idx)  # idx is the energy level at END of slot t
                prev_i = prev_idx[t][c][idx]
                prev_c_val = prev_c[t][c][idx]

                # Record backtrack info for first few slots
                if t < 5 and self.decision_log_level >= 3:
                    soc_at_t = self.min_soc + (idx * step_kwh / self.battery_capacity) * 100
                    backtrack_trace.append(f"t={t} ({hours_list[t].hour.strftime('%H:%M')}): action={action.name}, c={c}->prev_c={prev_c_val}, idx={idx} ({soc_at_t:.1f}%)->prev_i={prev_i}")

                if prev_i is None or prev_c_val is None:
                    idx = idx
                    c = c
                else:
                    idx = prev_i
                    c = prev_c_val
            actions.reverse()
            partial_flags.reverse()
            idx_trajectory.reverse()  # Now idx_trajectory[t] = end-of-slot idx for slot t

            # Log backtrack trace in chronological order
            if backtrack_trace and self.decision_log_level >= 3:
                self.log("[DeepTrace] Backtrack path (first 5 slots):")
                for line in reversed(backtrack_trace):
                    self.log(f"  {line}")

            # Log DP trace for diagnostic slots
            if dp_trace_slots and self.decision_log_level >= 3:
                self.log("=" * 70)
                self.log("DP TRACE: Detailed state transitions for discharge-allowed slots")
                self.log("=" * 70)
                for slot_hour, slot_price, traces in dp_trace_slots:
                    # Find what action was chosen for this slot
                    slot_idx = next((i for i, h in enumerate(hours_list) if h.hour == slot_hour), -1)
                    chosen_action = actions[slot_idx] if 0 <= slot_idx < len(actions) else None
                    self.log(f"\n{slot_hour.strftime('%Y-%m-%d %H:%M')} @ {slot_price:.4f} EUR/kWh -> {chosen_action.name if chosen_action else '?'}")

                    # Show traces for states that were active (had valid values)
                    # Focus on the most relevant states (near the current SOC trajectory)
                    relevant_traces = [t for t in traces if t["from_val"] > -1e10]
                    if relevant_traces:
                        # Sort by SOC to show trajectory
                        relevant_traces.sort(key=lambda x: x["from_soc"], reverse=True)
                        for trace in relevant_traces[:5]:  # Show top 5 by SOC
                            status = ""
                            if trace["discharge_updated"]:
                                status = "[OK] DISCHARGE wins"
                            elif trace["discharge_blocked"]:
                                status = f"[X] blocked: {trace['discharge_blocked']}"
                            elif trace["hold_updated"]:
                                status = "-> HOLD set (no discharge attempted)"

                            c_info = f"c={trace['charge_count']}, " if trace.get('charge_count', 0) > 0 else ""
                            delta = (trace['discharge_val'] - trace['hold_val']) if trace['discharge_val'] is not None else None
                            delta_str = f"+{delta:.4f}" if delta is not None and delta >= 0 else (f"{delta:.4f}" if delta is not None else "N/A")
                            discharge_val_str = f"{trace['discharge_val']:.4f}" if trace['discharge_val'] is not None else "N/A"
                            self.log(
                                f"  SOC {trace['from_soc']:.1f}% ({c_info}idx={trace['from_idx']}): "
                                f"hold={trace['hold_val']:.4f} vs discharge={discharge_val_str} (delta={delta_str}) -> {status}"
                            )
                self.log("=" * 70)

            return actions, partial_flags, best_val, meets_min, idx_trajectory

        def _build_schedule(discharge_thresholds: Optional[List[float]] = None) -> Tuple[Dict[datetime.datetime, ScheduleEntry], List[int]]:
            schedule_local: Dict[datetime.datetime, ScheduleEntry] = {}
            partial_index = current_slot_index
            partial_fraction = (
                slot_fractions[partial_index]
                if partial_index is not None
                else 1.0
            )
            has_partial = partial_index is not None and partial_fraction < 0.999

            if has_partial:
                price_point = hours_sorted_by_time[partial_index]
                price = price_point.price
                buy_price = price + self.grid_fee
                slot_threshold = (
                    discharge_thresholds[partial_index]
                    if discharge_thresholds is not None
                    else discharge_threshold
                )
                discharge_allowed = buy_price >= (slot_threshold - 1e-6)
                fraction = slot_fractions[partial_index]
                slot_load_kw = load_kw[partial_index]
                discharge_kwh = min(slot_load_kw, self.discharge_rate) * self.slot_hours * fraction
                slot_charge_rate = charge_rates_per_slot[partial_index]
                charge_energy_kwh = slot_charge_rate * self.efficiency * self.slot_hours * fraction
                charge_cost_kwh = slot_charge_rate * self.slot_hours * fraction

                # Remaining slots for DP
                remaining_slice = slice(partial_index + 1, None)
                hours_remaining = hours_sorted_by_time[remaining_slice]
                load_remaining = load_kw[remaining_slice]
                charge_rates_remaining = charge_rates_per_slot[remaining_slice]
                slot_fractions_remaining = slot_fractions[remaining_slice]
                favorable_flags_remaining, favorable_remaining_list = _compute_favorable(hours_remaining)
                discharge_thresholds_remaining = (
                    discharge_thresholds[remaining_slice]
                    if discharge_thresholds is not None
                    else None
                )

                # Candidates: (action, new_energy, immediate_val, start_c, start_idx_override, is_partial)
                candidates = []

                # HOLD - must pay grid price for load (same as main DP)
                hold_cost = buy_price * discharge_kwh
                candidates.append(
                    (BatteryMode.HOLD, start_energy, -hold_cost, 0, None, False)
                )

                # CHARGE
                price_is_favorable = favorable_flags[partial_index]
                must_use_unfavorable = (0 + favorable_remaining[partial_index]) < min_charge_slots
                allow_charge = price_is_favorable or must_use_unfavorable
                partial_charge_increment = 1  # Always count - energy is tracked separately
                if allow_charge and charge_energy_kwh > 0 and partial_charge_increment <= max_charge_slots:
                    new_energy = start_energy + charge_energy_kwh
                    actual_charge_energy = charge_energy_kwh
                    actual_charge_cost = charge_cost_kwh
                    if new_energy > max_energy + 1e-6:
                        headroom = max_energy - start_energy
                        if headroom >= step_kwh:
                            actual_charge_energy = headroom
                            actual_charge_cost = headroom / self.efficiency
                            new_energy = max_energy
                        else:
                            actual_charge_energy = 0
                    if actual_charge_energy > 0:
                        idx_float = (new_energy - min_energy) / step_kwh
                        start_idx_override = int(math.floor(idx_float + 1e-9))
                        # Charge cost + load cost (must import both from grid)
                        charge_immediate_cost = -buy_price * actual_charge_cost - buy_price * discharge_kwh
                        candidates.append(
                            (
                                BatteryMode.CHARGE,
                                new_energy,
                                charge_immediate_cost,
                                partial_charge_increment,
                                start_idx_override,
                                False,
                            )
                        )

                # DISCHARGE
                if discharge_allowed and discharge_kwh > 0:
                    new_energy = start_energy - discharge_kwh
                    if new_energy >= min_energy - 1e-6:
                        # Full discharge - no grid cost (battery covers load)
                        idx_float = (new_energy - min_energy) / step_kwh
                        start_idx_override = int(math.ceil(idx_float - 1e-9))
                        candidates.append(
                            (
                                BatteryMode.DISCHARGE,
                                new_energy,
                                0,  # No cost - battery covers load
                                0,
                                start_idx_override,
                                False,
                            )
                        )
                    elif start_energy > min_energy + discharge_kwh * 0.5:
                        # Partial discharge - discharge until min_soc
                        actual_discharge_kwh = start_energy - min_energy
                        grid_import = discharge_kwh - actual_discharge_kwh
                        partial_value = -buy_price * grid_import  # Pay only for grid portion
                        candidates.append(
                            (
                                BatteryMode.DISCHARGE,
                                min_energy,
                                partial_value,
                                0,
                                0,  # End at min_energy = index 0
                                True,
                            )
                        )

                best_action = BatteryMode.HOLD
                best_is_partial = False
                best_actions_remaining: List[BatteryMode] = []
                best_partial_flags_remaining: List[bool] = []
                best_idx_trajectory_remaining: List[int] = []
                best_first_slot_end_idx: int = int(round((start_energy - min_energy) / step_kwh))  # Default: HOLD keeps same idx
                best_value = neg_inf
                best_feasible_action: Optional[BatteryMode] = None
                best_feasible_is_partial = False
                best_feasible_actions_remaining: List[BatteryMode] = []
                best_feasible_partial_flags_remaining: List[bool] = []
                best_feasible_idx_trajectory_remaining: List[int] = []
                best_feasible_first_slot_end_idx: int = int(round((start_energy - min_energy) / step_kwh))
                best_feasible_value = neg_inf

                # Log greedy lookahead candidates
                if self.decision_log_level >= 3:
                    self.log(f"[GreedyLookahead] Partial slot {price_point.hour.strftime('%H:%M')} @ {price:.4f}, discharge_allowed={discharge_allowed}")
                    self.log(f"  Candidates: {[(c[0].name, c[2]) for c in candidates]}")

                greedy_results = []
                for action, new_energy, immediate_val, start_c, start_idx_override, is_partial in candidates:
                    actions_remaining, partial_flags_remaining, future_val, meets_min, idx_traj_remaining = _run_dp(
                        hours_remaining,
                        load_remaining,
                        charge_rates_remaining,
                        slot_fractions_remaining,
                        favorable_flags_remaining,
                        favorable_remaining_list,
                        new_energy,
                        start_c,
                        discharge_thresholds_list=discharge_thresholds_remaining,
                        start_idx_override=start_idx_override,
                    )
                    total_val = immediate_val + future_val
                    greedy_results.append((action.name, immediate_val, future_val, total_val, meets_min))
                    # Calculate idx at end of first slot based on the action's new_energy
                    first_slot_end_idx = int(round((new_energy - min_energy) / step_kwh))
                    first_slot_end_idx = min(max(first_slot_end_idx, 0), n_states - 1)
                    if total_val > best_value:
                        best_value = total_val
                        best_action = action
                        best_is_partial = is_partial
                        best_actions_remaining = actions_remaining
                        best_partial_flags_remaining = partial_flags_remaining
                        best_idx_trajectory_remaining = idx_traj_remaining
                        best_first_slot_end_idx = first_slot_end_idx
                    if meets_min and total_val > best_feasible_value:
                        best_feasible_value = total_val
                        best_feasible_action = action
                        best_feasible_is_partial = is_partial
                        best_feasible_actions_remaining = actions_remaining
                        best_feasible_partial_flags_remaining = partial_flags_remaining
                        best_feasible_idx_trajectory_remaining = idx_traj_remaining
                        best_feasible_first_slot_end_idx = first_slot_end_idx

                # Log greedy lookahead results
                if self.decision_log_level >= 3:
                    for name, imm, fut, tot, meets in greedy_results:
                        self.log(f"  {name}: immediate={imm:.4f}, future={fut:.4f}, total={tot:.4f}, meets_min={meets}")
                    self.log(f"  -> Best feasible: {best_feasible_action.name if best_feasible_action else 'None'} (val={best_feasible_value:.4f})")

                    # Explain HOLD vs DISCHARGE decision if both were candidates
                    hold_result = next((r for r in greedy_results if r[0] == "HOLD"), None)
                    discharge_result = next((r for r in greedy_results if r[0] == "DISCHARGE"), None)
                    if hold_result and discharge_result:
                        hold_imm, hold_fut, hold_tot = hold_result[1], hold_result[2], hold_result[3]
                        disc_imm, disc_fut, disc_tot = discharge_result[1], discharge_result[2], discharge_result[3]
                        saved_by_discharge = disc_imm - hold_imm  # Should be positive (HOLD pays, DISCHARGE doesn't)
                        extra_charge_cost = disc_fut - hold_fut  # Should be negative (DISCHARGE path costs more to recharge)
                        net_benefit = disc_tot - hold_tot  # Positive = DISCHARGE better, Negative = HOLD better

                        if net_benefit > 0.001:
                            self.log(f"  [DECISION] DISCHARGE wins: saves {saved_by_discharge:.4f} now, extra charge cost {-extra_charge_cost:.4f}, net benefit {net_benefit:.4f}")
                        elif net_benefit < -0.001:
                            self.log(f"  [DECISION] HOLD wins: would save {saved_by_discharge:.4f} by discharging, but recharging costs {-extra_charge_cost:.4f} extra (>{saved_by_discharge:.4f})")
                            # Calculate effective round-trip cost
                            if discharge_kwh > 0.01:
                                effective_recharge_cost_per_kwh = -extra_charge_cost / (discharge_kwh / self.efficiency)
                                self.log(f"             Overnight recharge cost: ~{effective_recharge_cost_per_kwh:.4f}/kWh vs discharge value {buy_price:.4f}/kWh")
                        else:
                            self.log(f"  [DECISION] Tie (within 0.001): HOLD preferred by default")

                if best_feasible_action is not None:
                    actions = [best_feasible_action] + best_feasible_actions_remaining
                    partial_flags = [best_feasible_is_partial] + best_feasible_partial_flags_remaining
                    idx_trajectory = [best_feasible_first_slot_end_idx] + best_feasible_idx_trajectory_remaining
                else:
                    actions = [best_action] + best_actions_remaining
                    partial_flags = [best_is_partial] + best_partial_flags_remaining
                    idx_trajectory = [best_first_slot_end_idx] + best_idx_trajectory_remaining
            else:
                actions, partial_flags, _, _, idx_trajectory = _run_dp(
                    hours_sorted_by_time,
                    load_kw,
                    charge_rates_per_slot,
                    slot_fractions,
                    favorable_flags,
                    favorable_remaining,
                    start_energy,
                    0,
                    discharge_thresholds_list=discharge_thresholds,
                )

            for price_point, action, lk, is_partial in zip(hours_sorted_by_time, actions, load_kw, partial_flags):
                hour = price_point.hour
                price = price_point.price
                reason = f"{price:.4f} EUR/kWh load~{lk:.2f}kW"
                if is_partial:
                    reason += " (until depleted)"
                schedule_local[hour] = ScheduleEntry(hour=hour, mode=action, reason=reason)
            return schedule_local, idx_trajectory

        schedule, idx_trajectory = _build_schedule()

        # === Two-pass enhancement: re-run if projected costs differ significantly ===
        has_charge_slots = any(e.mode == BatteryMode.CHARGE for e in schedule.values())

        if has_charge_slots:
            prices_by_slot = {p.hour: p.price for p in hours_sorted_by_time}
            slot_charge_rates_by_slot = {
                p.hour: charge_rates_per_slot[i] for i, p in enumerate(hours_sorted_by_time)
            }
            slot_fractions_by_slot = {
                p.hour: slot_fractions[i] for i, p in enumerate(hours_sorted_by_time)
            }
            projected_costs, final_cost = self._project_battery_costs(
                schedule,
                current_soc_for_calc,
                self.battery_avg_cost,
                prices_by_slot,
                charge_rates_by_slot=slot_charge_rates_by_slot,
                slot_fractions_by_slot=slot_fractions_by_slot,
            )
            self._last_projected_costs = projected_costs

            # Check if projected costs differ enough to warrant second pass
            final_cost = final_cost if projected_costs else self.battery_avg_cost
            cost_change_pct = abs(final_cost - self.battery_avg_cost) / max(0.01, self.battery_avg_cost)

            if cost_change_pct >= 0.01:  # >1% change
                self.log(f"Two-pass DP: battery cost projected to change from {self.battery_avg_cost:.4f} to {final_cost:.4f} EUR/kWh ({cost_change_pct*100:.1f}%), re-running with dynamic thresholds")

                # Build per-slot discharge thresholds based on projected costs
                discharge_thresholds = [
                    self._get_discharge_threshold_for_cost(
                        projected_costs.get(hours_sorted_by_time[t].hour, self.battery_avg_cost)
                    )
                    for t in range(n_slots)
                ]

                # Re-run DP with per-slot thresholds
                schedule, idx_trajectory = _build_schedule(discharge_thresholds=discharge_thresholds)

                # Update projected costs for the final schedule
                slot_charge_rates_by_slot = {
                    p.hour: charge_rates_per_slot[i] for i, p in enumerate(hours_sorted_by_time)
                }
                slot_fractions_by_slot = {
                    p.hour: slot_fractions[i] for i, p in enumerate(hours_sorted_by_time)
                }
                projected_costs, _ = self._project_battery_costs(
                    schedule,
                    current_soc_for_calc,
                    self.battery_avg_cost,
                    prices_by_slot,
                    charge_rates_by_slot=slot_charge_rates_by_slot,
                    slot_fractions_by_slot=slot_fractions_by_slot,
                )
                self._last_projected_costs = projected_costs
        else:
            self._last_projected_costs = {}

        charge_count = len([s for s in schedule.values() if s.mode == BatteryMode.CHARGE])
        discharge_count = len([s for s in schedule.values() if s.mode == BatteryMode.DISCHARGE])
        hold_count = len([s for s in schedule.values() if s.mode == BatteryMode.HOLD])

        self.log(f"Schedule generated: {charge_count} charge, {discharge_count} discharge, {hold_count} hold slots "
                 f"(slot={self.slot_minutes}min, load_quantile={self.load_quantile:.2f}, "
                 f"min_charge_slots={min_charge_slots})")

        # Store min_charge_slots for sensor exposure
        self._last_min_charge_slots = min_charge_slots

        # Log decision context for transparency
        if self.decision_log_level >= 1:
            self._log_schedule_decision_context(
                hours_sorted_by_time, schedule, load_kw, current_soc_for_calc, min_charge_slots
            )

        # Convert idx_trajectory to SOC trajectory and store for logging
        # idx_trajectory[t] = energy index at END of slot t
        # SOC trajectory maps slot hour -> (start_soc, end_soc)
        # Temperature trajectory maps slot hour -> (start_temp, end_temp)
        dp_soc_trajectory: Dict[datetime.datetime, Tuple[float, float]] = {}
        dp_temp_trajectory: Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]] = {}

        if idx_trajectory and len(idx_trajectory) == len(hours_sorted_by_time):
            # Starting idx from current SOC
            start_idx = int(round((start_energy - min_energy) / step_kwh))
            start_idx = min(max(start_idx, 0), n_states - 1)

            for t, price_point in enumerate(hours_sorted_by_time):
                hour = price_point.hour
                # Start SOC for this slot
                if t == 0:
                    slot_start_idx = start_idx
                else:
                    slot_start_idx = idx_trajectory[t - 1]  # Previous slot's end = this slot's start
                slot_end_idx = idx_trajectory[t]

                # Convert idx to SOC percentage
                start_soc = self.min_soc + (slot_start_idx * step_kwh / self.battery_capacity) * 100
                end_soc = self.min_soc + (slot_end_idx * step_kwh / self.battery_capacity) * 100
                dp_soc_trajectory[hour] = (start_soc, end_soc)

            # Build temperature trajectory based on scheduled mode for each slot
            # CHARGE: battery warms up (predict_temp_after_duration)
            # HOLD/DISCHARGE: battery cools toward ambient (predict_temp_after_idle)
            if current_temp is not None and self.learning_engine:
                projected_temp = current_temp
                for t, price_point in enumerate(hours_sorted_by_time):
                    hour = price_point.hour
                    start_temp = projected_temp
                    slot_duration_minutes = self.slot_minutes * slot_fractions[t]

                    # Look up the mode for this slot
                    entry = schedule.get(hour)
                    if entry is not None and entry.mode == BatteryMode.CHARGE:
                        # Charging: battery warms up
                        projected_temp = self.learning_engine.predict_temp_after_duration(
                            projected_temp, slot_duration_minutes
                        )
                    else:
                        # HOLD or DISCHARGE: battery cools toward ambient
                        projected_temp = self.learning_engine.predict_temp_after_idle(
                            projected_temp, slot_duration_minutes
                        )
                    dp_temp_trajectory[hour] = (start_temp, projected_temp)

        # Store trajectories for use in _log_schedule
        self._last_dp_soc_trajectory = dp_soc_trajectory
        self._last_dp_temp_trajectory = dp_temp_trajectory

        return schedule

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
            charge_price_threshold = self.battery_avg_cost * 1.05  # Raw price threshold used in DP
            discharge_threshold = self._get_discharge_threshold()
            self.log(f"Input State:")
            self.log(f"  Current SOC: {current_soc:.1f}%")
            self.log(f"  Min SOC target: {self.min_soc:.1f}%")
            self.log(f"  Min charge slots required: {min_charge_slots} (to avoid hitting min SOC)")
            self.log(f"  Battery avg cost: {self.battery_avg_cost:.4f} EUR/kWh")
            self.log(f"  Charge price threshold: {charge_price_threshold:.4f} EUR/kWh (raw price, excl. grid fee)")
            self.log(f"  Discharge cost threshold: {discharge_threshold:.4f} EUR/kWh (avg/eff + grid fee + wear)")

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

        # Log projected cost evolution if two-pass DP was used
        if self.decision_log_level >= 1 and self._last_projected_costs:
            self.log(f"\nProjected cost evolution:")
            prev_cost = self.battery_avg_cost
            for slot, cost in sorted(self._last_projected_costs.items()):
                if abs(cost - prev_cost) > 0.001:
                    threshold = self._get_discharge_threshold_for_cost(cost)
                    self.log(f"  After {slot.strftime('%Y-%m-%d %H:%M')}: avg_cost={cost:.4f}, threshold={threshold:.4f}")
                    prev_cost = cost
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

        def is_future_price(p):
            p_hour = p.hour
            compare_time = current_slot
            if p_hour.tzinfo is not None and compare_time.tzinfo is None:
                p_hour = p_hour.replace(tzinfo=None)
            elif p_hour.tzinfo is None and compare_time.tzinfo is not None:
                compare_time = compare_time.replace(tzinfo=None)
            return p_hour >= compare_time

        future_prices = [p for p in prices if is_future_price(p)]
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
                def is_current_or_future(h):
                    compare_h = self._normalize_to_local(h, local_tz)
                    compare_now = self._normalize_to_local(current_slot, local_tz)
                    if compare_h is None or compare_now is None:
                        return False
                    if compare_h.tzinfo is not None and compare_now.tzinfo is None:
                        compare_h = compare_h.replace(tzinfo=None)
                    elif compare_h.tzinfo is None and compare_now.tzinfo is not None:
                        compare_now = compare_now.replace(tzinfo=None)
                    return compare_h >= compare_now

                future_schedule = {
                    h: e for h, e in self.schedule.items()
                    if is_current_or_future(h)
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
        now = self.datetime()
        local_tz = self._get_local_timezone()
        # Convert now to local timezone
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        now_slot = self._align_to_slot(now)
        prices = self.get_prices()
        compare_now_slot = self._align_to_slot(now)

        # Filter to future prices only using proper timezone conversion
        def is_future(p):
            p_hour = p.hour
            compare_now = compare_now_slot
            if local_tz is not None:
                if p_hour.tzinfo is not None:
                    p_hour = p_hour.astimezone(local_tz)
                if compare_now.tzinfo is not None:
                    compare_now = compare_now.astimezone(local_tz)
            # Handle mixed timezone-aware/naive by comparing as naive
            if p_hour.tzinfo is not None and compare_now.tzinfo is None:
                p_hour = p_hour.replace(tzinfo=None)
            elif p_hour.tzinfo is None and compare_now.tzinfo is not None:
                compare_now = compare_now.replace(tzinfo=None)
            return p_hour >= compare_now

        future_prices = [p for p in prices if is_future(p)]

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
        def is_future_hour(h):
            compare_h = h
            compare_now = compare_now_slot
            if local_tz is not None:
                if h.tzinfo is not None:
                    compare_h = h.astimezone(local_tz)
                if compare_now.tzinfo is not None:
                    compare_now = compare_now.astimezone(local_tz)
            # Handle mixed timezone-aware/naive
            if compare_h.tzinfo is not None and compare_now.tzinfo is None:
                compare_h = compare_h.replace(tzinfo=None)
            elif compare_h.tzinfo is None and compare_now.tzinfo is not None:
                compare_now = compare_now.replace(tzinfo=None)
            return compare_h >= compare_now

        hours_to_remove = [h for h in self.schedule.keys() if is_future_hour(h)]
        for hour in hours_to_remove:
            del self.schedule[hour]

        # Add new schedule entries
        for hour, entry in new_schedule.items():
            self.schedule[hour] = entry

        # Recalculate expected SOC
        now_hour = compare_now_slot

        def is_current_or_future(k):
            compare_k = k
            compare_now = now_hour
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

        current_temp = self._get_battery_temp()
        self.expected_soc_schedule, self.expected_temp_schedule = self.calculate_expected_soc_schedule(
            {k: v for k, v in self.schedule.items() if is_current_or_future(k)},
            current_soc,
            starting_temp=current_temp
        )

        # Log recalculated schedule (current/future only)
        if self.decision_log_level >= 1:
            self._log_schedule(
                {k: v for k, v in self.schedule.items() if is_current_or_future(k)},
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

    def safety_check(self, kwargs=None):
        """
        Safety check - ensures SOC stays within bounds.
        Now called via SOC state listener for instant response.
        Kept for backward compatibility.
        """
        current_soc = self._get_current_soc()
        if current_soc is None:
            return
        self._check_soc_boundaries(current_soc)

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

        When transitioning to CHARGE or DISCHARGE, resets the learning baseline
        so duration is measured from when the new mode actually starts, not from
        the last significant SOC change (which could be hours ago during HOLD).

        Also tracks idle periods (HOLD/DISCHARGE) for cooling rate learning.
        """
        old_mode = self.current_mode
        now = self.datetime()
        current_temp = self._get_battery_temp()

        # Record temperature observation for ambient estimation
        if current_temp is not None and self.learning_engine:
            self.learning_engine.record_temperature_observation(current_temp)

        # Transitioning TO CHARGE: record cooling if we were idle, then reset baseline
        if new_mode == BatteryMode.CHARGE and old_mode != BatteryMode.CHARGE:
            # Record cooling observation if we have idle tracking data
            if (self._idle_start_time is not None and
                self._idle_start_temp is not None and
                current_temp is not None and
                self.learning_engine):
                duration_minutes = (now - self._idle_start_time).total_seconds() / 60
                if duration_minutes >= 10:  # Only record if idle for at least 10 minutes
                    self.learning_engine.record_cooling(
                        temp_start=self._idle_start_temp,
                        temp_end=current_temp,
                        duration_minutes=duration_minutes
                    )
                    self._save_learning_data()

            # Clear idle tracking
            self._idle_start_time = None
            self._idle_start_temp = None

            # Reset learning baseline for charging
            current_soc = self._get_current_soc()
            if current_soc is not None:
                self._last_sig_soc = current_soc
                self._last_sig_soc_time = now
                self._last_sig_temp = current_temp
                temp_str = f", temp={current_temp:.1f}C" if current_temp is not None else ""
                self.log(f"Mode transition to CHARGE: reset learning baseline to {current_soc:.1f}%{temp_str}")

        # Transitioning TO HOLD or DISCHARGE: start idle tracking
        elif new_mode in (BatteryMode.HOLD, BatteryMode.DISCHARGE) and old_mode == BatteryMode.CHARGE:
            # Start tracking idle period for cooling rate learning
            self._idle_start_time = now
            self._idle_start_temp = current_temp
            temp_str = f", temp={current_temp:.1f}C" if current_temp is not None else ""
            self.log(f"Mode transition to {new_mode.name}: started idle tracking{temp_str}")

            # Also reset learning baseline for discharge tracking
            if new_mode == BatteryMode.DISCHARGE:
                current_soc = self._get_current_soc()
                if current_soc is not None:
                    self._last_sig_soc = current_soc
                    self._last_sig_soc_time = now
                    self._last_sig_temp = current_temp

        # Transitioning between HOLD and DISCHARGE: keep idle tracking, update discharge baseline
        elif new_mode == BatteryMode.DISCHARGE and old_mode == BatteryMode.HOLD:
            current_soc = self._get_current_soc()
            if current_soc is not None:
                self._last_sig_soc = current_soc
                self._last_sig_soc_time = now
                self._last_sig_temp = current_temp
                temp_str = f", temp={current_temp:.1f}C" if current_temp is not None else ""
                self.log(f"Mode transition to DISCHARGE: reset learning baseline to {current_soc:.1f}%{temp_str}")

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
        """Initialize battery cost from persistent storage or estimate"""
        # Entity for persisting battery cost
        self.battery_cost_entity = self.args.get("battery_cost_entity", "input_number.battery_avg_cost")

        # Track SOC for measuring actual charge/discharge
        self._last_soc: Optional[float] = self._get_current_soc()
        self._last_mode: BatteryMode = BatteryMode.HOLD
        # Track SOC sample timing and pricing slot separately
        self._last_soc_time: Optional[datetime.datetime] = self.datetime()
        # Track last significant SOC change (>=1%) for learning durations
        self._last_sig_soc: Optional[float] = self._last_soc
        self._last_sig_soc_time: Optional[datetime.datetime] = self._last_soc_time
        # Track battery temperature at start of charge session (for warming rate learning)
        self._last_sig_temp: Optional[float] = self._get_battery_temp()
        # Track idle period start (for cooling rate learning)
        self._idle_start_time: Optional[datetime.datetime] = None
        self._idle_start_temp: Optional[float] = None
        self._last_price_slot: Optional[datetime.datetime] = self._align_to_slot(self.datetime())

        # Try to load from persistent storage
        try:
            state = self.get_state(self.battery_cost_entity)
            if state and state not in ("unknown", "unavailable"):
                self.battery_avg_cost = float(state)
                self.log(f"Loaded battery avg cost from HA: {self.battery_avg_cost:.4f} EUR/kWh")
                self._schedule_startup_optimization()
                return
        except (ValueError, TypeError) as e:
            self.log(f"Could not load battery cost from {self.battery_cost_entity}: {e}", level="WARNING")

        # HA not ready yet - wait for homeassistant_start event to load cost and start optimizer
        self.log("HA entities not available yet, waiting for homeassistant_start event")
        self.listen_event(self._on_ha_start, "homeassistant_start")

    def _save_battery_cost(self):
        """Persist battery cost to Home Assistant entity"""
        try:
            self.call_service("input_number/set_value",
                entity_id=self.battery_cost_entity,
                value=round(self.battery_avg_cost, 4)
            )
        except Exception as e:
            self.log(f"Could not save battery cost to {self.battery_cost_entity}: {e}", level="DEBUG")

    def _schedule_startup_optimization(self):
        """Schedule the startup optimization"""
        self.log("Scheduling startup optimization")
        self.run_in(self.full_optimize, 1)

    def _on_ha_start(self, event_name, data, kwargs):
        """Load battery cost after HA start and trigger startup optimization"""
        try:
            state = self.get_state(self.battery_cost_entity)
            if state and state not in ("unknown", "unavailable"):
                self.battery_avg_cost = float(state)
                self.log(f"Loaded battery avg cost from HA: {self.battery_avg_cost:.4f} EUR/kWh")
            else:
                self.battery_avg_cost = 0.10  # Default fallback
                self.log(f"Battery cost entity unavailable, using default: {self.battery_avg_cost:.4f} EUR/kWh", level="WARNING")
        except (ValueError, TypeError) as e:
            self.battery_avg_cost = 0.10
            self.log(f"Could not load battery cost ({e}), using default: {self.battery_avg_cost:.4f} EUR/kWh", level="WARNING")

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
        Called by _on_soc_change when SOC state changes.

        Args:
            current_soc: Current battery state of charge (%)
        """
        now = self.datetime()
        current_slot = self._align_to_slot(now)

        # Record temperature observation for ambient estimation
        current_temp = self._get_battery_temp()
        if current_temp is not None and self.learning_engine:
            self.learning_engine.record_temperature_observation(current_temp)

        if self._last_soc is None:
            self._last_soc = current_soc
            self._last_soc_time = now
            self._last_sig_soc = current_soc
            self._last_sig_soc_time = now
            self._last_price_slot = current_slot
            return

        soc_change = current_soc - self._last_soc

        # Only process significant changes (> 1%)
        if abs(soc_change) < 1.0:
            self._last_soc = current_soc
            self._last_soc_time = now
            self._last_price_slot = current_slot  # Always update slot to prevent stale pricing
            return

        energy_change_kwh = abs(soc_change) / 100 * self.battery_capacity

        # Calculate time since last observation
        if self._last_sig_soc_time:
            # Ensure consistent timezone handling to avoid naive/aware mismatch
            last_time = self._last_sig_soc_time
            compare_now = now
            if compare_now.tzinfo is not None and last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=compare_now.tzinfo)
            elif compare_now.tzinfo is None and last_time.tzinfo is not None:
                compare_now = compare_now.replace(tzinfo=last_time.tzinfo)
            duration_minutes = (compare_now - last_time).total_seconds() / 60
        else:
            duration_minutes = 1.0  # Fallback

        if soc_change > 0:
            # Battery charged - get price for the charging period
            charge_price = self._get_price_for_hour(self._last_price_slot) if self._last_price_slot else None
            if charge_price is None:
                charge_price = self.battery_avg_cost  # Fallback to current avg

            # Calculate energy BEFORE this charge (at old SOC)
            old_energy = max(0, (self._last_soc - self.min_soc) / 100 * self.battery_capacity)

            # Weighted average: (old_energy * old_cost + new_energy * charge_price) / total
            old_total_cost = old_energy * self.battery_avg_cost
            new_total_cost = old_total_cost + (energy_change_kwh * charge_price)
            new_total_energy = old_energy + energy_change_kwh

            if new_total_energy > 0:
                self.battery_avg_cost = new_total_cost / new_total_energy

            self.log(f"Battery charged: +{soc_change:.1f}% (+{energy_change_kwh:.2f} kWh) at {charge_price:.4f} EUR/kWh, "
                     f"new avg cost: {self.battery_avg_cost:.4f} EUR/kWh")

            self._save_battery_cost()

            # Feed learning engine with charging observation (include battery temp if available)
            battery_temp_end = self._get_battery_temp()
            self.learning_engine.record_charging(
                soc_start=self._last_soc,
                soc_end=current_soc,
                duration_minutes=duration_minutes,
                charge_price=charge_price,
                battery_temp=battery_temp_end,
                battery_temp_start=self._last_sig_temp,
                battery_temp_end=battery_temp_end
            )
            self._save_learning_data()
            self._update_learning_sensor()

        elif soc_change < 0:
            # Battery discharged - cost per kWh stays same, just less energy
            discharge_price = self._get_price_for_hour(self._last_price_slot) if self._last_price_slot else 0.0

            self.log(f"Battery discharged: {soc_change:.1f}% ({energy_change_kwh:.2f} kWh)")

            # Feed learning engine with discharging observation
            self.learning_engine.record_discharging(
                soc_start=self._last_soc,
                soc_end=current_soc,
                duration_minutes=duration_minutes,
                price_eur_kwh=discharge_price or 0.0
            )
            self._save_learning_data()
            self._update_learning_sensor()

        self._last_soc = current_soc
        self._last_soc_time = now
        self._last_sig_soc = current_soc
        self._last_sig_soc_time = now
        self._last_sig_temp = self._get_battery_temp()  # Track temp at start of next session
        self._last_price_slot = current_slot

    def _update_battery_cost_from_soc_change(self, kwargs=None):
        """
        Update battery cost based on actual SOC change since last check.
        Kept for backward compatibility - now delegates to _process_soc_change_event.
        """
        current_soc = self._get_current_soc()
        if current_soc is None:
            return
        self._process_soc_change_event(current_soc)

    def _get_discharge_threshold(self) -> float:
        """Calculate discharge threshold based on actual battery cost"""
        # Threshold = (what we paid + grid fees) / efficiency + wear cost
        # Only discharge if we can "sell" above this price
        threshold = ((self.battery_avg_cost + self.grid_fee) / self.efficiency) + self.battery_wear_cost
        return threshold

    def _get_discharge_threshold_for_cost(self, avg_cost: float) -> float:
        """Calculate discharge threshold for a given battery average cost"""
        return ((avg_cost + self.grid_fee) / self.efficiency) + self.battery_wear_cost

    def _project_battery_costs(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        starting_soc: float,
        starting_cost: float,
        prices_by_slot: Dict[datetime.datetime, float],
        charge_rates_by_slot: Optional[Dict[datetime.datetime, float]] = None,
        slot_fractions_by_slot: Optional[Dict[datetime.datetime, float]] = None,
    ) -> Tuple[Dict[datetime.datetime, float], float]:
        """
        Project battery avg cost evolution through a schedule.
        Returns (dict mapping slot -> projected cost at START of that slot, final avg cost).
        """
        projected_costs = {}
        current_soc = starting_soc
        current_cost = starting_cost

        for hour in sorted(schedule.keys()):
            projected_costs[hour] = current_cost
            entry = schedule[hour]

            if entry.mode == BatteryMode.CHARGE:
                charge_price = prices_by_slot.get(hour, current_cost)
                old_energy = max(0, (current_soc - self.min_soc) / 100 * self.battery_capacity)
                slot_charge_rate = (
                    charge_rates_by_slot.get(hour, self.charge_rate)
                    if charge_rates_by_slot is not None
                    else self.charge_rate
                )
                fraction = (
                    slot_fractions_by_slot.get(hour, 1.0)
                    if slot_fractions_by_slot is not None
                    else 1.0
                )
                energy_added = slot_charge_rate * self.efficiency * self.slot_hours * fraction
                headroom_kwh = max(0.0, (self.max_soc - current_soc) / 100 * self.battery_capacity)
                energy_added = max(0.0, min(energy_added, headroom_kwh))

                if old_energy + energy_added > 0:
                    current_cost = (old_energy * current_cost + energy_added * charge_price) / (old_energy + energy_added)

                current_soc = min(self.max_soc, current_soc + (energy_added / self.battery_capacity) * 100)

            elif entry.mode == BatteryMode.DISCHARGE:
                load_kw = self._predict_load_kw(hour)
                fraction = (
                    slot_fractions_by_slot.get(hour, 1.0)
                    if slot_fractions_by_slot is not None
                    else 1.0
                )
                energy_removed = min(load_kw, self.discharge_rate) * self.slot_hours * fraction
                current_soc = max(self.min_soc, current_soc - (energy_removed / self.battery_capacity) * 100)

        return projected_costs, current_cost

    def _get_price_for_hour(self, hour: datetime.datetime) -> Optional[float]:
        """Get the electricity price for a specific slot from price service cache."""
        local_tz = self._get_local_timezone()
        for price_point in self._price_service.cached_prices:
            # Convert both to local timezone for comparison
            p_hour = price_point.hour
            compare_hour = hour
            if p_hour.tzinfo is not None and local_tz is not None:
                p_hour = p_hour.astimezone(local_tz)
            if compare_hour.tzinfo is not None and local_tz is not None:
                compare_hour = compare_hour.astimezone(local_tz)

            # Compare date and slot components
            if (p_hour.date() == compare_hour.date() and
                p_hour.hour == compare_hour.hour and
                p_hour.minute == compare_hour.minute):
                return price_point.price
        return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_current_soc(self) -> Optional[float]:
        """Get current battery SOC"""
        try:
            state = self.get_state(self.soc_sensor)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError) as e:
            self.log(f"Error reading SOC: {e}", level="WARNING")
        return None

    def _get_pv_power(self) -> float:
        """Get current PV power production"""
        try:
            state = self.get_state(self.pv_power_sensor)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return 0.0

    def _get_battery_temp(self) -> Optional[float]:
        """Get current battery temperature in Celsius.

        Returns None if:
        - No temp sensor configured
        - Sensor is unavailable/unknown
        - Sensor value cannot be parsed
        """
        if not self.battery_temp_sensor:
            return None
        try:
            state = self.get_state(self.battery_temp_sensor)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return None

    def _get_load_power(self) -> Optional[float]:
        """Get current household load in Watts (from configured sensor)."""
        if not self.load_power_sensor:
            return None
        try:
            state = self.get_state(self.load_power_sensor)
            if state and state not in ("unknown", "unavailable"):
                load_w = float(state)
                if load_w <= 0:
                    if self._last_nonzero_load_w is not None:
                        return max(self._last_nonzero_load_w, self.load_zero_floor_w)
                    return self.load_zero_floor_w
                self._last_nonzero_load_w = load_w
                return load_w
        except (ValueError, TypeError):
            pass
        return None

    def _predict_load_kw(self, dt: datetime.datetime) -> float:
        """Predict expected load (kW) for a slot using load profile."""
        if self.load_profile:
            predicted = self.load_profile.predict_kw(dt, self.load_quantile)
        else:
            predicted = self.base_consumption / 1000.0

        return predicted

    def _align_to_slot(self, dt: datetime.datetime) -> datetime.datetime:
        """Floor datetime to the start of the current time slot."""
        local_tz = self._get_local_timezone()
        if dt.tzinfo is not None and local_tz is not None:
            dt = dt.astimezone(local_tz)
        elif local_tz is not None and dt.tzinfo is None:
            dt = dt.replace(tzinfo=local_tz)
        minutes = dt.hour * 60 + dt.minute
        slot_start = (minutes // self.slot_minutes) * self.slot_minutes
        return dt.replace(
            hour=int(slot_start // 60),
            minute=int(slot_start % 60),
            second=0,
            microsecond=0
        )

    def _get_expected_soc_for_hour(
        self,
        expected_soc: Dict[datetime.datetime, float],
        hour: datetime.datetime,
        local_tz
    ) -> Optional[float]:
        """Get expected SOC for a specific hour, handling timezone differences."""
        # Direct lookup first
        if hour in expected_soc:
            return expected_soc[hour]

        # Try matching by local time components
        for sched_hour, soc_value in expected_soc.items():
            compare_sched = sched_hour
            compare_hour = hour
            if local_tz is not None:
                if sched_hour.tzinfo is not None:
                    compare_sched = sched_hour.astimezone(local_tz)
                if hour.tzinfo is not None:
                    compare_hour = hour.astimezone(local_tz)
            # Handle mixed timezone-aware/naive
            if compare_sched.tzinfo is not None and compare_hour.tzinfo is None:
                compare_sched = compare_sched.replace(tzinfo=None)
            elif compare_sched.tzinfo is None and compare_hour.tzinfo is not None:
                compare_hour = compare_hour.replace(tzinfo=None)

            if (compare_sched.date() == compare_hour.date() and
                compare_sched.hour == compare_hour.hour and
                compare_sched.minute == compare_hour.minute):
                return soc_value

        return None

    def _get_expected_temp_for_hour(
        self,
        expected_temp: Dict[datetime.datetime, float],
        hour: datetime.datetime,
        local_tz
    ) -> Optional[float]:
        """Get expected temperature for a specific hour, handling timezone differences."""
        # Direct lookup first
        if hour in expected_temp:
            return expected_temp[hour]

        # Try matching by local time components
        for sched_hour, temp_value in expected_temp.items():
            compare_sched = sched_hour
            compare_hour = hour
            if local_tz is not None:
                if sched_hour.tzinfo is not None:
                    compare_sched = sched_hour.astimezone(local_tz)
                if hour.tzinfo is not None:
                    compare_hour = hour.astimezone(local_tz)
            # Handle mixed timezone-aware/naive
            if compare_sched.tzinfo is not None and compare_hour.tzinfo is None:
                compare_sched = compare_sched.replace(tzinfo=None)
            elif compare_sched.tzinfo is None and compare_hour.tzinfo is not None:
                compare_hour = compare_hour.replace(tzinfo=None)

            if (compare_sched.date() == compare_hour.date() and
                compare_sched.hour == compare_hour.hour and
                compare_sched.minute == compare_hour.minute):
                return temp_value

        return None

    def _get_dp_trajectory_for_hour(
        self,
        trajectory: Dict[datetime.datetime, Tuple[float, float]],
        hour: datetime.datetime,
        local_tz
    ) -> Optional[Tuple[float, float]]:
        """Get DP trajectory data (start, end) tuple for a specific hour, handling timezone differences."""
        if not trajectory:
            return None

        # Direct lookup first
        if hour in trajectory:
            return trajectory[hour]

        # Try matching by local time components
        for sched_hour, traj_value in trajectory.items():
            compare_sched = sched_hour
            compare_hour = hour
            if local_tz is not None:
                if sched_hour.tzinfo is not None:
                    compare_sched = sched_hour.astimezone(local_tz)
                if hour.tzinfo is not None:
                    compare_hour = hour.astimezone(local_tz)
            # Handle mixed timezone-aware/naive
            if compare_sched.tzinfo is not None and compare_hour.tzinfo is None:
                compare_sched = compare_sched.replace(tzinfo=None)
            elif compare_sched.tzinfo is None and compare_hour.tzinfo is not None:
                compare_hour = compare_hour.replace(tzinfo=None)

            if (compare_sched.date() == compare_hour.date() and
                compare_sched.hour == compare_hour.hour and
                compare_sched.minute == compare_hour.minute):
                return traj_value

        return None

    def _next_slot_time(self) -> datetime.datetime:
        """Get the next slot boundary time."""
        now = self.datetime()
        local_tz = self._get_local_timezone()
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        elif local_tz is not None and now.tzinfo is None:
            now = now.replace(tzinfo=local_tz)
        minutes = now.hour * 60 + now.minute
        next_slot = ((minutes // self.slot_minutes) + 1) * self.slot_minutes
        if next_slot >= 1440:
            next_slot = 0
            now = now + datetime.timedelta(days=1)
        return now.replace(
            hour=int(next_slot // 60),
            minute=int(next_slot % 60),
            second=5,
            microsecond=0
        )

    def _next_interval_time(self, interval_minutes: int) -> datetime.datetime:
        """Get the next boundary time for a given interval."""
        interval_minutes = max(1, int(interval_minutes))
        now = self.datetime()
        local_tz = self._get_local_timezone()
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        elif local_tz is not None and now.tzinfo is None:
            now = now.replace(tzinfo=local_tz)
        minutes = now.hour * 60 + now.minute
        next_boundary = ((minutes // interval_minutes) + 1) * interval_minutes
        if next_boundary >= 1440:
            next_boundary = 0
            now = now + datetime.timedelta(days=1)
        return now.replace(
            hour=int(next_boundary // 60),
            minute=int(next_boundary % 60),
            second=5,
            microsecond=0
        )

    def _is_enabled(self) -> bool:
        """Check if optimizer is enabled"""
        try:
            state = self.get_state(self.enabled_entity)
            return state == "on"
        except:
            return True  # Default to enabled if entity doesn't exist

    def _is_override_active(self) -> bool:
        """Check if manual override is active"""
        try:
            state = self.get_state(self.override_entity)
            return state == "on"
        except:
            return False

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
        if local_tz is not None and dt.tzinfo is not None:
            return dt.astimezone(local_tz)
        return dt

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

            # For discharge, show battery avg cost as primary, grid price in parentheses
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
