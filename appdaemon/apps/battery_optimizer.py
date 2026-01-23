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
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =============================================================================
# Self-Learning Battery Performance Tracking
# =============================================================================

@dataclass
class LearningStats:
    """Aggregated learning statistics for battery performance."""
    # Charging rates by SOC range (kW observed at different SOC levels)
    charge_rates_by_soc: Dict[str, List[float]] = field(default_factory=dict)
    # Discharge rates by SOC range
    discharge_rates_by_soc: Dict[str, List[float]] = field(default_factory=dict)
    # Round-trip efficiency observations
    efficiency_history: List[float] = field(default_factory=list)
    # Prediction accuracy (predicted vs actual charge time)
    prediction_errors: List[float] = field(default_factory=list)
    # Totals
    total_energy_charged_kwh: float = 0.0
    total_energy_discharged_kwh: float = 0.0
    total_charge_cost_eur: float = 0.0
    total_discharge_revenue_eur: float = 0.0
    total_cycles: int = 0
    # Timestamps
    first_observation: Optional[str] = None
    last_observation: Optional[str] = None
    # Temperature-aware charge rates: {"25-50": {"5-10": [3.1, 3.2], "10-15": [4.2, 4.5]}}
    charge_rates_by_soc_temp: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'LearningStats':
        # Handle backward compatibility for older versions without charge_rates_by_soc_temp
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


class BatteryLearningEngine:
    """
    Self-learning engine that adapts predictions based on actual battery performance.

    Learns:
    - Actual charge rate (may differ from configured)
    - SOC-dependent charge rate curve (batteries charge slower when full)
    - Round-trip efficiency
    - Provides confidence intervals for predictions
    """

    def __init__(
        self,
        battery_capacity_kwh: float = 14.3,
        nominal_charge_rate_kw: float = 4.5,
        nominal_efficiency: float = 0.85,
        min_soc: float = 10.0,
        max_soc: float = 100.0,
        log_func=None,
        temp_ranges: Optional[List[int]] = None,
    ):
        self.battery_capacity = battery_capacity_kwh
        self.nominal_charge_rate = nominal_charge_rate_kw
        self.nominal_efficiency = nominal_efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.log = log_func or print

        # Temperature range boundaries for bucketing (default: <5, 5-10, 10-15, 15-20, >20)
        self.temp_ranges = temp_ranges if temp_ranges is not None else [5, 10, 15, 20]

        # Learning data
        self.stats = LearningStats()

        # Exponential moving average factor (0.1 = slow learning, stable)
        self.ema_alpha = 0.1

        # Learned parameters (start with nominals)
        self.learned_efficiency = nominal_efficiency

        # SOC-dependent charge rate multipliers
        # NOTE: Default assumes flat curve - learning will discover actual behavior
        # Some batteries (like user's LiPO) charge SLOWER at low SOC and FASTER at high SOC
        # (opposite of typical CC-CV), due to BMS protection or inverter behavior
        self.soc_charge_multipliers = {
            "0-25": 1.0,    # Will be learned
            "25-50": 1.0,   # Will be learned
            "50-75": 1.0,   # Will be learned
            "75-90": 1.0,   # Will be learned
            "90-100": 1.0,  # Will be learned
        }

    def _get_soc_range(self, soc: float) -> str:
        """Get the SOC range bucket for a given SOC."""
        if soc < 25:
            return "0-25"
        elif soc < 50:
            return "25-50"
        elif soc < 75:
            return "50-75"
        elif soc < 90:
            return "75-90"
        else:
            return "90-100"

    def _get_temp_range(self, temp: float) -> str:
        """Get the temperature range bucket for a given temperature (Celsius).

        Uses configured temp_ranges boundaries. For default [5, 10, 15, 20]:
        - "<5" for temps below 5°C
        - "5-10" for temps 5-10°C
        - "10-15" for temps 10-15°C
        - "15-20" for temps 15-20°C
        - ">20" for temps above 20°C
        """
        if not self.temp_ranges:
            return "all"

        for i, boundary in enumerate(self.temp_ranges):
            if temp < boundary:
                if i == 0:
                    return f"<{boundary}"
                else:
                    return f"{self.temp_ranges[i-1]}-{boundary}"

        # Above the highest boundary
        return f">{self.temp_ranges[-1]}"

    def record_charging(
        self,
        soc_start: float,
        soc_end: float,
        duration_minutes: float,
        energy_from_grid_kwh: Optional[float] = None,
        charge_price: float = 0.0,
        battery_temp: Optional[float] = None
    ):
        """
        Record a charging observation and update learned parameters.

        Args:
            soc_start: SOC at start of charging
            soc_end: SOC at end of charging
            duration_minutes: How long charging took
            energy_from_grid_kwh: Energy drawn from grid (if available from meter)
            charge_price: Price paid per kWh
            battery_temp: Battery temperature in Celsius (if available)
        """
        if duration_minutes <= 0 or soc_end <= soc_start:
            return

        # Calculate energy added to battery
        energy_added = (soc_end - soc_start) / 100 * self.battery_capacity

        # Calculate observed charge rate
        charge_rate = energy_added / (duration_minutes / 60)

        # Calculate efficiency if grid energy known
        if energy_from_grid_kwh and energy_from_grid_kwh > 0:
            observed_efficiency = energy_added / energy_from_grid_kwh
            if 0.5 < observed_efficiency < 1.0:
                self.learned_efficiency = (
                    self.ema_alpha * observed_efficiency +
                    (1 - self.ema_alpha) * self.learned_efficiency
                )
                self.stats.efficiency_history.append(observed_efficiency)

        # Update SOC-range specific charge rates (legacy, SOC-only)
        soc_range = self._get_soc_range((soc_start + soc_end) / 2)
        if soc_range not in self.stats.charge_rates_by_soc:
            self.stats.charge_rates_by_soc[soc_range] = []
        self.stats.charge_rates_by_soc[soc_range].append(charge_rate)

        # Keep last 50 observations per range
        if len(self.stats.charge_rates_by_soc[soc_range]) > 50:
            self.stats.charge_rates_by_soc[soc_range] = \
                self.stats.charge_rates_by_soc[soc_range][-50:]

        # Update temperature-aware charge rates (2D: SOC + temp)
        if battery_temp is not None:
            temp_range = self._get_temp_range(battery_temp)
            if soc_range not in self.stats.charge_rates_by_soc_temp:
                self.stats.charge_rates_by_soc_temp[soc_range] = {}
            if temp_range not in self.stats.charge_rates_by_soc_temp[soc_range]:
                self.stats.charge_rates_by_soc_temp[soc_range][temp_range] = []
            self.stats.charge_rates_by_soc_temp[soc_range][temp_range].append(charge_rate)

            # Keep last 50 observations per SOC+temp combination
            if len(self.stats.charge_rates_by_soc_temp[soc_range][temp_range]) > 50:
                self.stats.charge_rates_by_soc_temp[soc_range][temp_range] = \
                    self.stats.charge_rates_by_soc_temp[soc_range][temp_range][-50:]

        # Update totals
        self.stats.total_energy_charged_kwh += energy_added
        self.stats.total_charge_cost_eur += energy_added * charge_price

        # Update timestamps
        now = datetime.datetime.now().isoformat()
        if self.stats.first_observation is None:
            self.stats.first_observation = now
        self.stats.last_observation = now

        temp_str = f", temp={battery_temp:.1f}°C" if battery_temp is not None else ""
        # Get observation count for this bucket
        obs_count = len(self.stats.charge_rates_by_soc.get(soc_range, []))
        self.log(f"Learning: Recorded charge {soc_start:.1f}%->{soc_end:.1f}% "
                 f"in {duration_minutes:.0f}min, rate={charge_rate:.2f}kW{temp_str}, "
                 f"bucket={soc_range} ({obs_count} obs)")

    def record_discharging(
        self,
        soc_start: float,
        soc_end: float,
        duration_minutes: float,
        energy_delivered_kwh: Optional[float] = None,
        price_eur_kwh: float = 0.0
    ):
        """Record a discharging observation."""
        if duration_minutes <= 0 or soc_start <= soc_end:
            return

        if energy_delivered_kwh is None:
            energy_delivered_kwh = (soc_start - soc_end) / 100 * self.battery_capacity

        # Update totals
        self.stats.total_energy_discharged_kwh += energy_delivered_kwh
        self.stats.total_discharge_revenue_eur += energy_delivered_kwh * price_eur_kwh

    def get_charge_rate_for_soc(self, soc: float, battery_temp: Optional[float] = None) -> float:
        """
        Get predicted charge rate for a given SOC level and optional temperature.
        Uses learned data with fallback chain:
        1. Exact SOC+temp match (>=3 observations) -> median of last 10
        2. SOC match, aggregate all temps
        3. SOC-only legacy data
        4. Nominal rate

        Args:
            soc: Current state of charge (%)
            battery_temp: Current battery temperature in Celsius (optional)

        Returns:
            Predicted charge rate in kW
        """
        soc_range = self._get_soc_range(soc)

        # Fallback 1: Try temperature-aware lookup if temp is available
        if battery_temp is not None:
            temp_range = self._get_temp_range(battery_temp)

            # Check for exact SOC+temp match
            if soc_range in self.stats.charge_rates_by_soc_temp:
                temp_data = self.stats.charge_rates_by_soc_temp[soc_range]
                if temp_range in temp_data and len(temp_data[temp_range]) >= 3:
                    return statistics.median(temp_data[temp_range][-10:])

                # Fallback 2: Aggregate all temps for this SOC range
                all_rates = []
                for rates in temp_data.values():
                    all_rates.extend(rates[-10:])  # Last 10 from each temp bucket
                if len(all_rates) >= 3:
                    return statistics.median(all_rates)

        # Fallback 3: Use SOC-only legacy data
        if soc_range in self.stats.charge_rates_by_soc:
            observations = self.stats.charge_rates_by_soc[soc_range]
            if len(observations) >= 3:
                return statistics.median(observations[-10:])

        # Fallback 4: Use configured nominal charge rate
        multiplier = self.soc_charge_multipliers.get(soc_range, 1.0)
        return self.nominal_charge_rate * multiplier

    def get_confidence_for_soc(self, soc: float, battery_temp: Optional[float] = None) -> float:
        """
        Get prediction confidence for a given SOC level and optional temperature.

        Confidence reflects both data source quality and observation count:
        - SOC+temp exact match: 0.7 base + up to 0.3 based on count (max at 10 obs)
        - SOC+temp aggregated: 0.5 base + up to 0.2 based on count (max at 15 obs)
        - SOC-only data: 0.3 base + up to 0.2 based on count (max at 10 obs)
        - Nominal fallback: 0.0

        Returns:
            Confidence level 0.0 to 1.0
        """
        soc_range = self._get_soc_range(soc)

        # Check temperature-aware data first
        if battery_temp is not None:
            temp_range = self._get_temp_range(battery_temp)

            if soc_range in self.stats.charge_rates_by_soc_temp:
                temp_data = self.stats.charge_rates_by_soc_temp[soc_range]

                # Exact SOC+temp match
                if temp_range in temp_data and len(temp_data[temp_range]) >= 3:
                    count = len(temp_data[temp_range])
                    return 0.7 + min(0.3, (count - 3) / 7 * 0.3)

                # Aggregated temps for this SOC range
                all_rates = []
                for rates in temp_data.values():
                    all_rates.extend(rates)
                if len(all_rates) >= 3:
                    return 0.5 + min(0.2, (len(all_rates) - 3) / 12 * 0.2)

        # SOC-only data
        if soc_range in self.stats.charge_rates_by_soc:
            observations = self.stats.charge_rates_by_soc[soc_range]
            if len(observations) >= 3:
                count = len(observations)
                return 0.3 + min(0.2, (count - 3) / 7 * 0.2)

        # Nominal fallback - no confidence
        return 0.0

    def predict_charge_time(
        self,
        current_soc: float,
        target_soc: float,
        battery_temp: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Predict time to charge from current to target SOC.

        Args:
            current_soc: Current state of charge (%)
            target_soc: Target state of charge (%)
            battery_temp: Current battery temperature in Celsius (optional)

        Returns: (expected_hours, min_hours, max_hours)
        """
        if current_soc >= target_soc:
            return 0.0, 0.0, 0.0

        total_time = 0.0
        total_confidence = 0.0
        step_count = 0
        soc = current_soc
        step = 5.0

        while soc < target_soc:
            next_soc = min(soc + step, target_soc)
            energy_needed = (next_soc - soc) / 100 * self.battery_capacity
            charge_rate = self.get_charge_rate_for_soc(soc, battery_temp)
            grid_energy = energy_needed / self.learned_efficiency
            time_hours = grid_energy / max(0.1, charge_rate)
            total_time += time_hours
            total_confidence += self.get_confidence_for_soc(soc, battery_temp)
            step_count += 1
            soc = next_soc

        # Confidence interval based on average per-bucket confidence
        avg_confidence = total_confidence / max(1, step_count)
        uncertainty = 0.3 * (1 - avg_confidence)
        return total_time, total_time * (1 - uncertainty), total_time * (1 + uncertainty)

    def get_learning_summary(self) -> Dict:
        """Get summary of learned parameters."""
        if self.stats.total_energy_charged_kwh > 0:
            overall_efficiency = (
                self.stats.total_energy_discharged_kwh /
                self.stats.total_energy_charged_kwh
            )
        else:
            overall_efficiency = self.nominal_efficiency

        total_profit = (
            self.stats.total_discharge_revenue_eur -
            self.stats.total_charge_cost_eur
        )

        # Build temperature-aware rates summary with per-bucket confidence
        temp_aware_rates = {}
        for soc_range, temp_data in self.stats.charge_rates_by_soc_temp.items():
            temp_aware_rates[soc_range] = {}
            for temp_range, rates in temp_data.items():
                if rates:
                    count = len(rates)
                    # Confidence: 0.7 base + up to 0.3 based on count (max at 10 obs)
                    confidence = 0.7 + min(0.3, (count - 3) / 7 * 0.3) if count >= 3 else 0.0
                    temp_aware_rates[soc_range][temp_range] = {
                        "median_kw": round(statistics.median(rates[-10:]), 2),
                        "observations": count,
                        "confidence": round(confidence, 2)
                    }

        # Build SOC-only rates with confidence
        soc_charge_rates = {}
        for soc_range, rates in self.stats.charge_rates_by_soc.items():
            if rates:
                count = len(rates)
                # Confidence: 0.3 base + up to 0.2 based on count (max at 10 obs)
                confidence = 0.3 + min(0.2, (count - 3) / 7 * 0.2) if count >= 3 else 0.0
                soc_charge_rates[soc_range] = {
                    "median_kw": round(statistics.median(rates[-10:]), 2),
                    "observations": count,
                    "confidence": round(confidence, 2)
                }

        # Calculate total observations across all buckets
        total_observations = sum(len(v) for v in self.stats.charge_rates_by_soc.values())

        return {
            "learned_efficiency": round(self.learned_efficiency, 3),
            "total_energy_charged_kwh": round(self.stats.total_energy_charged_kwh, 1),
            "total_energy_discharged_kwh": round(self.stats.total_energy_discharged_kwh, 1),
            "overall_efficiency": round(overall_efficiency, 3),
            "total_profit_eur": round(total_profit, 2),
            "total_observations": total_observations,
            "soc_charge_rates": soc_charge_rates,
            "temp_aware_rates": temp_aware_rates,
        }

    def save_to_json(self) -> str:
        """Serialize learning state for persistence."""
        data = {
            "version": 5,  # v5 removes global confidence (use per-bucket confidence instead)
            "learned_efficiency": self.learned_efficiency,
            "stats": self.stats.to_dict(),
        }
        return json.dumps(data)

    def load_from_json(self, json_str: str) -> bool:
        """Load learning state from JSON. Returns True if successful."""
        try:
            data = json.loads(json_str)
            if data.get("version", 0) >= 1:
                # Note: learned_charge_rate removed in v4, global confidence removed in v5
                self.learned_efficiency = data.get("learned_efficiency", self.nominal_efficiency)
                if "stats" in data:
                    self.stats = LearningStats.from_dict(data["stats"])
                return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.log(f"Could not load learning data: {e}")
        return False


def _quantile(values: List[float], q: float) -> float:
    """Return the q-quantile (0..1) with linear interpolation."""
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_vals[lower]
    frac = pos - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


@dataclass
class LoadProfileStats:
    """Aggregated load observations per time slot."""
    samples_by_slot: Dict[str, List[float]] = field(default_factory=dict)  # slot -> W samples
    observation_count: int = 0
    last_observation: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'LoadProfileStats':
        return cls(**data)


class LoadProfile:
    """
    Simple statistical load profile by time-of-day slots.
    Stores recent samples per slot and returns a quantile-based forecast.
    """

    def __init__(
        self,
        slot_minutes: int,
        default_load_w: float,
        max_samples: int = 60,
        min_samples: int = 6,
        log_func=None,
    ):
        self.slot_minutes = max(1, int(slot_minutes))
        self.slots_per_day = int(1440 / self.slot_minutes)
        self.default_load_w = float(default_load_w)
        self.max_samples = max(1, int(max_samples))
        self.min_samples = max(1, int(min_samples))
        self.log = log_func or print
        self.stats = LoadProfileStats()

    def _slot_index(self, dt: datetime.datetime) -> int:
        minutes = dt.hour * 60 + dt.minute
        return int(minutes // self.slot_minutes)

    def record(self, dt: datetime.datetime, load_w: float):
        if load_w <= 0:
            return
        slot = str(self._slot_index(dt))
        samples = self.stats.samples_by_slot.get(slot, [])
        samples.append(float(load_w))
        if len(samples) > self.max_samples:
            samples = samples[-self.max_samples:]
        self.stats.samples_by_slot[slot] = samples
        self.stats.observation_count += 1
        self.stats.last_observation = dt.isoformat()

    def predict_kw(self, dt: datetime.datetime, quantile: float = 0.75) -> float:
        slot = str(self._slot_index(dt))
        samples = self.stats.samples_by_slot.get(slot, [])
        if not samples:
            return self.default_load_w / 1000.0
        q_value = _quantile(samples, quantile)
        confidence = min(1.0, len(samples) / self.min_samples)
        blended = (self.default_load_w * (1 - confidence)) + (q_value * confidence)
        return max(0.0, blended) / 1000.0

    def to_json(self) -> str:
        data = {
            "version": 1,
            "slot_minutes": self.slot_minutes,
            "stats": self.stats.to_dict(),
        }
        return json.dumps(data)

    def load_from_json(self, json_str: str) -> bool:
        try:
            data = json.loads(json_str)
            if data.get("version", 0) >= 1:
                if data.get("slot_minutes") != self.slot_minutes:
                    self.log("Load profile slot size changed, ignoring saved data")
                    return False
                if "stats" in data:
                    self.stats = LoadProfileStats.from_dict(data["stats"])
                return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.log(f"Could not load load profile data: {e}")
        return False

class BatteryMode(Enum):
    HOLD = 0
    CHARGE = 1
    DISCHARGE = 2


@dataclass
class PricePoint:
    """Represents a single time-slot price data point"""
    hour: datetime.datetime
    price: float

    def __lt__(self, other):
        return self.price < other.price


@dataclass
class ScheduleEntry:
    """Represents a scheduled battery mode for a specific time slot"""
    hour: datetime.datetime
    mode: BatteryMode
    reason: str


@dataclass
class TouPeriod:
    """Represents a TOU period for inverter scheduling"""
    start: int      # Minutes since midnight
    end: int        # Minutes since midnight
    power: int      # -100 to +100 (positive=charge, negative=discharge)


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
        self.cached_prices: List[PricePoint] = []
        self.cached_prices_date: Optional[datetime.date] = None  # Date when prices were cached
        self.last_optimization: Optional[datetime.datetime] = None
        self.expected_soc_schedule: Dict[datetime.datetime, float] = {}
        self._last_nonzero_load_w: Optional[float] = None

        # Battery cost tracking (weighted average cost of energy in battery)
        self.battery_avg_cost: float = 0.0  # EUR/kWh
        self._init_battery_cost()

        # Decision context tracking (for transparency logging and sensor exposure)
        self._last_recalc_trigger: str = "startup"  # "startup", "daily_13:15", "soc_deviation", "manual"
        self._last_recalc_time: Optional[datetime.datetime] = None
        self._last_soc_deviation: Optional[float] = None  # Deviation that triggered recalculation
        self._last_min_charge_slots: int = 0  # Min charge slots from last calculation
        self._last_charge_slots: List[Dict] = []  # Selected charge slots with prices

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

        # Full re-optimization after Nord Pool publishes tomorrow's prices
        # Uses configured hour (default 14 for EET = 13 CET) plus 15 minutes buffer
        optimize_hour = self.tomorrow_prices_hour
        self.run_daily(self.full_optimize, datetime.time(optimize_hour, 15))

        # Also run optimization at startup (with delay for HA services to be ready)
        startup_delay = int(self.args.get("startup_delay_seconds", 30))
        self.log(f"Scheduling startup optimization in {startup_delay}s")
        self.run_in(self.full_optimize, startup_delay)

        # Adaptive re-evaluation (can be more frequent than schedule slots)
        self.run_every(
            self.adaptive_optimize,
            self._next_interval_time(self.adaptive_recalc_minutes),
            self.adaptive_recalc_minutes * 60
        )

        # Safety check every 5 minutes
        self.run_every(self.safety_check, self.datetime(), 5 * 60)

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
    # Price Fetching and Analysis
    # =========================================================================

    def get_prices(self) -> List[PricePoint]:
        """
        Fetch prices from Nord Pool.
        Returns combined today + tomorrow prices when available.
        Supports both built-in HA Nord Pool integration (via service call) and HACS custom component (via attributes).
        """
        self.log(f"get_prices called. Config: nordpool_config_entry='{self.nordpool_config_entry}', area='{self.nordpool_area}', ha_url='{self.args.get('ha_url', 'NOT SET')}'")

        prices = []
        today = self.date()
        tomorrow = today + datetime.timedelta(days=1)
        tz = self._get_local_timezone()

        # Try built-in HA Nord Pool integration first (uses service call)
        if self.nordpool_config_entry:
            prices = self._get_prices_via_service(today, tomorrow, tz)
            if prices:
                prices = self._normalize_prices(prices)
                self.cached_prices = prices
                self.cached_prices_date = today
                self.log(f"Fetched {len(prices)} price points via service call")
                return prices

        # Fall back to HACS custom component (uses sensor attributes)
        prices = self._get_prices_via_sensor(today, tomorrow, tz)
        if prices:
            prices = self._normalize_prices(prices)
            self.cached_prices = prices
            self.cached_prices_date = today
            self.log(f"Fetched {len(prices)} price points via sensor attributes")
            return prices

        # Validate cached prices are not stale before using
        if self.cached_prices and self.cached_prices_date:
            # Cache is valid if it was fetched today or yesterday (may contain tomorrow's prices)
            cache_age_days = (today - self.cached_prices_date).days
            if cache_age_days <= 1:
                # Additional check: ensure we have prices for today
                has_today_prices = any(p.hour.date() == today for p in self.cached_prices)
                if has_today_prices:
                    self.log(f"Using cached prices (cached {cache_age_days} day(s) ago)", level="WARNING")
                    return self.cached_prices
                else:
                    self.log(f"Cached prices don't contain today's prices, clearing stale cache", level="WARNING")
                    self.cached_prices = []
                    self.cached_prices_date = None
            else:
                self.log(f"Cached prices are {cache_age_days} days old, clearing stale cache", level="WARNING")
                self.cached_prices = []
                self.cached_prices_date = None

        self.log("No price data available from any source", level="WARNING")
        return []

    def _normalize_prices(self, prices: List[PricePoint]) -> List[PricePoint]:
        """
        Normalize price points to the configured slot size.
        Aggregates higher-resolution data or expands lower-resolution data.
        """
        if not prices:
            return []

        sorted_prices = sorted(prices, key=lambda p: p.hour)
        deltas = []
        for i in range(1, len(sorted_prices)):
            delta_min = (sorted_prices[i].hour - sorted_prices[i - 1].hour).total_seconds() / 60
            if delta_min > 0:
                deltas.append(delta_min)
        min_delta = min(deltas) if deltas else self.slot_minutes

        # Already at desired resolution
        if abs(min_delta - self.slot_minutes) < 0.1:
            return sorted_prices

        # Aggregate finer data into slot buckets
        if min_delta < self.slot_minutes:
            buckets: Dict[datetime.datetime, List[float]] = {}
            for p in sorted_prices:
                bucket = self._align_to_slot(p.hour)
                buckets.setdefault(bucket, []).append(p.price)
            normalized = [
                PricePoint(hour=dt, price=sum(vals) / len(vals))
                for dt, vals in buckets.items()
            ]
            return sorted(normalized, key=lambda p: p.hour)

        # Expand coarser data into multiple slots
        factor = int(round(min_delta / self.slot_minutes))
        if factor <= 1 or abs(min_delta - factor * self.slot_minutes) > 0.1:
            # Fallback to bucketing if resolution is irregular
            buckets: Dict[datetime.datetime, List[float]] = {}
            for p in sorted_prices:
                bucket = self._align_to_slot(p.hour)
                buckets.setdefault(bucket, []).append(p.price)
            normalized = [
                PricePoint(hour=dt, price=sum(vals) / len(vals))
                for dt, vals in buckets.items()
            ]
            return sorted(normalized, key=lambda p: p.hour)

        expanded: List[PricePoint] = []
        for p in sorted_prices:
            for i in range(factor):
                expanded.append(PricePoint(
                    hour=p.hour + datetime.timedelta(minutes=i * self.slot_minutes),
                    price=p.price
                ))
        return sorted(expanded, key=lambda p: p.hour)

    def _get_prices_via_service(self, today, tomorrow, tz) -> List[PricePoint]:
        """
        Fetch prices using the built-in HA Nord Pool integration service.
        Uses nordpool.get_prices_for_date action.
        """
        prices = []

        if not self.nordpool_config_entry:
            self.log("nordpool_config_entry not configured, skipping service call", level="DEBUG")
            return prices

        self.log(f"Fetching prices via service for config_entry={self.nordpool_config_entry}, area={self.nordpool_area}")

        try:
            # Fetch today's prices
            self.log(f"Fetching today's prices ({today.isoformat()})...")
            today_data = self._call_nordpool_service(today.isoformat())
            self.log(f"Today data received: {today_data is not None}, type={type(today_data)}")
            if today_data:
                today_prices = self._parse_service_response(today_data, tz)
                self.log(f"Parsed {len(today_prices)} prices for today")
                prices.extend(today_prices)

            # Fetch tomorrow's prices (available after ~13:00 CET)
            # Use configured hour adjusted for local timezone (default 14 for EET = 13 CET)
            current_hour = self.datetime().hour
            self.log(f"Current hour: {current_hour}, will fetch tomorrow: {current_hour >= self.tomorrow_prices_hour}")
            if current_hour >= self.tomorrow_prices_hour:
                try:
                    self.log(f"Fetching tomorrow's prices ({tomorrow.isoformat()})...")
                    tomorrow_data = self._call_nordpool_service(tomorrow.isoformat())
                    if tomorrow_data:
                        tomorrow_prices = self._parse_service_response(tomorrow_data, tz)
                        self.log(f"Parsed {len(tomorrow_prices)} prices for tomorrow")
                        prices.extend(tomorrow_prices)
                except Exception as e:
                    self.log(f"Tomorrow's prices not yet available: {e}", level="DEBUG")

        except Exception as e:
            self.log(f"Error fetching prices via service: {e}", level="WARNING")
            self.log(traceback.format_exc(), level="WARNING")

        self.log(f"Service method returned {len(prices)} total prices")
        return prices

    def _get_prices_for_date(self, date_obj, tz) -> List[PricePoint]:
        """Fetch and normalize prices for a specific date via Nord Pool service."""
        if not self.nordpool_config_entry:
            return []

        try:
            data = self._call_nordpool_service(date_obj.isoformat())
            if not data:
                return []
            parsed = self._parse_service_response(data, tz)
            if not parsed:
                return []
            return self._normalize_prices(parsed)
        except Exception as e:
            self.log(f"Error fetching prices for {date_obj.isoformat()}: {e}", level="DEBUG")
            return []

    def _call_nordpool_service(self, date_str: str) -> Optional[Dict]:
        """
        Call the nordpool.get_prices_for_date service.
        Tries AppDaemon call_service first, falls back to REST API.
        """
        # Try REST API approach (more reliable for response actions)
        result = self._call_nordpool_rest_api(date_str)
        if result:
            return result

        # Fallback to AppDaemon call_service (may not return response data)
        try:
            result = self.call_service(
                "nordpool/get_prices_for_date",
                config_entry=self.nordpool_config_entry,
                date=date_str,
                areas=self.nordpool_area,  # Single string, not a list
                return_result=True
            )
            self.log(f"Nord Pool service response for {date_str}: {type(result)}")
            return result
        except Exception as e:
            self.log(f"Service call failed for {date_str}: {e}", level="WARNING")
            return None

    def _call_nordpool_rest_api(self, date_str: str) -> Optional[Dict]:
        """
        Call Nord Pool service via Home Assistant REST API.
        This is more reliable for response-returning actions.
        """
        if not REQUESTS_AVAILABLE:
            self.log("requests module not available for REST API", level="WARNING")
            return None

        try:
            # Get HA URL and token from apps.yaml
            ha_url = self.args.get("ha_url", "").rstrip("/")
            token = self.args.get("ha_token", "")

            self.log(f"REST API config: ha_url={ha_url[:30] if ha_url else 'MISSING'}..., token={'SET' if token else 'MISSING'}")

            if not ha_url or not token:
                self.log("ha_url and ha_token not configured in apps.yaml - needed for Nord Pool service calls", level="WARNING")
                return None

            # Add return_response for HA 2023.7+ response-returning actions
            url = f"{ha_url}/api/services/nordpool/get_prices_for_date?return_response"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            payload = {
                "config_entry": self.nordpool_config_entry,
                "date": date_str,
                "areas": self.nordpool_area  # Single string, not a list
            }

            self.log(f"Calling Nord Pool API: POST {url}")
            self.log(f"Payload: {json.dumps(payload)}")
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            self.log(f"REST API response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                self.log(f"REST API response for {date_str}: type={type(data)}, keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}")

                # Response is wrapped in service_response for HA 2023.7+ actions
                if isinstance(data, dict):
                    # Extract from service_response wrapper
                    if "service_response" in data:
                        self.log("Found service_response wrapper")
                        return data["service_response"]
                    # Check for response wrapper (older format)
                    if "response" in data:
                        return data["response"]
                    # Check for area data directly
                    if self.nordpool_area in data or self.nordpool_area.lower() in data:
                        return data

                    self.log(f"Response keys: {list(data.keys())[:10]}", level="DEBUG")
                    return data
                elif isinstance(data, list) and data:
                    self.log(f"Response is list with {len(data)} items", level="DEBUG")
                    return {self.nordpool_area: data}
                return data
            else:
                self.log(f"REST API returned status {response.status_code}: {response.text[:200]}", level="WARNING")
                return None

        except requests.exceptions.RequestException as e:
            self.log(f"REST API request failed: {e}", level="WARNING")
            return None
        except Exception as e:
            self.log(f"REST API call failed: {e}", level="WARNING")
            self.log(traceback.format_exc(), level="DEBUG")
            return None

    def _parse_service_response(self, data: Dict, tz) -> List[PricePoint]:
        """
        Parse the response from nordpool.get_prices_for_date service.
        Response format: {area: [{start, end, price}, ...]}
        Prices are in EUR/MWh, need to convert to EUR/kWh.
        """
        prices = []

        if not data:
            self.log("No data to parse", level="DEBUG")
            return prices

        self.log(f"Parsing response data type: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

        # Handle different response formats
        area_prices = []

        if isinstance(data, dict):
            # Try direct area lookup
            area_prices = data.get(self.nordpool_area, [])
            if not area_prices:
                area_prices = data.get(self.nordpool_area.lower(), [])

            # Try nested in 'prices' key
            if not area_prices and "prices" in data:
                area_prices = data["prices"]

            # Try the first key if it contains a list
            if not area_prices:
                for key, value in data.items():
                    if isinstance(value, list) and value:
                        self.log(f"Found price list under key '{key}' with {len(value)} entries")
                        area_prices = value
                        break

        elif isinstance(data, list):
            area_prices = data

        self.log(f"Parsing {len(area_prices)} price entries for area {self.nordpool_area}")

        for entry in area_prices:
            if not isinstance(entry, dict):
                continue

            start_str = entry.get("start", "")
            price_mwh = entry.get("price")

            if price_mwh is None:
                continue

            # Parse start time (UTC format: 2026-01-21T00:00:00+00:00)
            try:
                if isinstance(start_str, str):
                    # Handle various datetime formats
                    start_str = start_str.replace("Z", "+00:00")
                    start_dt = datetime.datetime.fromisoformat(start_str)
                else:
                    continue

                # Convert to local timezone
                if tz and start_dt.tzinfo:
                    start_dt = start_dt.astimezone(tz)
                elif tz:
                    start_dt = start_dt.replace(tzinfo=tz)

                # Convert EUR/MWh to EUR/kWh
                price_kwh = float(price_mwh) / 1000.0
                prices.append(PricePoint(hour=start_dt, price=price_kwh))

            except (ValueError, TypeError) as e:
                self.log(f"Error parsing price entry {entry}: {e}", level="DEBUG")
                continue

        return prices

    def _get_prices_via_sensor(self, today, tomorrow, tz) -> List[PricePoint]:
        """
        Fetch prices from HACS custom component sensor attributes.
        """
        try:
            state = self.get_state(self.nordpool_sensor, attribute="all")
            if not state:
                return []

            prices = []
            attrs = state.get("attributes", {})

            # HACS custom component uses raw_today/raw_tomorrow or today/tomorrow
            today_prices = attrs.get("raw_today") or attrs.get("today") or []
            tomorrow_prices = attrs.get("raw_tomorrow") or attrs.get("tomorrow") or []

            # Process today's prices
            if today_prices:
                if isinstance(today_prices, list) and today_prices:
                    if isinstance(today_prices[0], dict):
                        # raw format: list of {start, end, value}
                        for entry in today_prices:
                            price = entry.get("value")
                            start = entry.get("start")
                            if price is not None and start:
                                try:
                                    if isinstance(start, str):
                                        dt = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
                                    else:
                                        dt = start
                                    if tz and dt.tzinfo:
                                        dt = dt.astimezone(tz)
                                    elif tz:
                                        dt = dt.replace(tzinfo=tz)
                                    prices.append(PricePoint(hour=dt, price=float(price)))
                                except (ValueError, TypeError):
                                    pass
                    else:
                        # Simple list of prices (infer step size)
                        step_minutes = 60
                        if len(today_prices) > 0 and 1440 % len(today_prices) == 0:
                            step_minutes = int(1440 / len(today_prices))
                        for idx, price in enumerate(today_prices):
                            if price is not None:
                                minutes = idx * step_minutes
                                dt = datetime.datetime.combine(
                                    today,
                                    datetime.time(minutes // 60, minutes % 60)
                                )
                                if tz:
                                    dt = dt.replace(tzinfo=tz)
                                prices.append(PricePoint(hour=dt, price=float(price)))

            # Process tomorrow's prices
            if tomorrow_prices:
                if isinstance(tomorrow_prices, list) and tomorrow_prices:
                    if isinstance(tomorrow_prices[0], dict):
                        for entry in tomorrow_prices:
                            price = entry.get("value")
                            start = entry.get("start")
                            if price is not None and start:
                                try:
                                    if isinstance(start, str):
                                        dt = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
                                    else:
                                        dt = start
                                    if tz and dt.tzinfo:
                                        dt = dt.astimezone(tz)
                                    elif tz:
                                        dt = dt.replace(tzinfo=tz)
                                    prices.append(PricePoint(hour=dt, price=float(price)))
                                except (ValueError, TypeError):
                                    pass
                    else:
                        step_minutes = 60
                        if len(tomorrow_prices) > 0 and 1440 % len(tomorrow_prices) == 0:
                            step_minutes = int(1440 / len(tomorrow_prices))
                        for idx, price in enumerate(tomorrow_prices):
                            if price is not None:
                                minutes = idx * step_minutes
                                dt = datetime.datetime.combine(
                                    tomorrow,
                                    datetime.time(minutes // 60, minutes % 60)
                                )
                                if tz:
                                    dt = dt.replace(tzinfo=tz)
                                prices.append(PricePoint(hour=dt, price=float(price)))

            return prices

        except Exception as e:
            self.log(f"Error fetching prices via sensor: {e}", level="WARNING")
            return []

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
            yesterday_prices = self._get_prices_for_date(yesterday, tz)

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

        # Pre-compute charge rates for each slot considering temperature warm-up
        # Model: battery warms ~2°C per hour of consecutive charging, capped at 25°C
        charge_rates_per_slot = []
        estimated_temp = current_temp
        estimated_soc = current_soc_for_calc
        for i, p in enumerate(hours_sorted_by_time):
            slot_rate = self.learning_engine.get_charge_rate_for_soc(estimated_soc, estimated_temp)
            charge_rates_per_slot.append(slot_rate)
            # Rough estimate: if charging, SOC increases and temp increases
            # We use the base rate for estimation since we don't know actual charging yet
            estimated_soc = min(self.max_soc, estimated_soc + (slot_rate * self.efficiency * self.slot_hours / self.battery_capacity * 100))
            if estimated_temp is not None:
                # Model temperature warm-up during charging (conservative: assume charging)
                estimated_temp = min(25.0, estimated_temp + 2.0 * self.slot_hours)

        # Use learned rate for base calculation (falls back to nominal if no data)
        base_charge_rate = learned_charge_rate
        base_charge_energy_kwh = base_charge_rate * self.efficiency * self.slot_hours
        base_charge_cost_kwh = base_charge_rate * self.slot_hours
        load_kw = [self._predict_load_kw(p.hour) for p in hours_sorted_by_time]
        discharge_energy_kwh = [
            min(lk, self.discharge_rate) * self.slot_hours * slot_fractions[i]
            for i, lk in enumerate(load_kw)
        ]

        # DP tables
        neg_inf = -1e18
        max_charge_slots = n_slots
        dp = [[neg_inf] * n_states for _ in range(max_charge_slots + 1)]
        start_idx = int(round((start_energy - min_energy) / step_kwh))
        start_idx = min(max(start_idx, 0), n_states - 1)
        dp[0][start_idx] = 0.0
        prev_idx = [[[None] * n_states for _ in range(max_charge_slots + 1)] for _ in range(n_slots)]
        prev_c = [[[None] * n_states for _ in range(max_charge_slots + 1)] for _ in range(n_slots)]
        prev_action = [[[None] * n_states for _ in range(max_charge_slots + 1)] for _ in range(n_slots)]

        for t in range(n_slots):
            price = hours_sorted_by_time[t].price
            buy_price = price + self.grid_fee
            # For self-consumption (load-backed discharge), value is avoided import cost
            # Only use export_rate_multiplier if actually exporting to grid
            # Since discharge is modeled as min(load, discharge_rate), it's self-consumption
            discharge_value = buy_price  # Avoided import cost; charging already accounts energy cost
            fraction = slot_fractions[t]
            discharge_kwh = discharge_energy_kwh[t]
            # Use per-slot charge rate (temperature-aware if available)
            slot_charge_rate = charge_rates_per_slot[t]
            charge_energy_kwh = slot_charge_rate * self.efficiency * self.slot_hours * fraction
            charge_cost_kwh = slot_charge_rate * self.slot_hours * fraction
            # Treat current partial slot as a full slot for charge-counting decisions
            if current_slot_index is not None and t == current_slot_index and fraction < 0.999:
                charge_count_increment = 1
            else:
                charge_count_increment = 0 if fraction < 0.999 else 1
            next_dp = [[neg_inf] * n_states for _ in range(max_charge_slots + 1)]
            next_prev_idx = [[None] * n_states for _ in range(max_charge_slots + 1)]
            next_prev_c = [[None] * n_states for _ in range(max_charge_slots + 1)]
            next_prev_action = [[None] * n_states for _ in range(max_charge_slots + 1)]

            for c in range(max_charge_slots + 1):
                for idx, val in enumerate(dp[c]):
                    if val <= neg_inf / 2:
                        continue

                    # HOLD
                    if val > next_dp[c][idx]:
                        next_dp[c][idx] = val
                        next_prev_idx[c][idx] = idx
                        next_prev_c[c][idx] = c
                        next_prev_action[c][idx] = BatteryMode.HOLD

                    # CHARGE
                    # Only allow charging if:
                    # 1. Raw price is favorable vs battery avg cost (economically sensible), OR
                    # 2. There aren't enough favorable slots left to meet min_charge_slots
                    price_is_favorable = favorable_flags[t]
                    must_use_unfavorable = (c + favorable_remaining[t]) < min_charge_slots
                    allow_charge = price_is_favorable or must_use_unfavorable

                    if allow_charge and charge_energy_kwh > 0 and c + charge_count_increment <= max_charge_slots:
                        new_energy = energy_levels[idx] + charge_energy_kwh
                        if new_energy <= max_energy + 1e-6:
                            next_idx = int(round((new_energy - min_energy) / step_kwh))
                            next_idx = min(max(next_idx, 0), n_states - 1)
                            next_val = val - (buy_price * charge_cost_kwh)
                            c_next = c + charge_count_increment
                            if next_val > next_dp[c_next][next_idx]:
                                next_dp[c_next][next_idx] = next_val
                                next_prev_idx[c_next][next_idx] = idx
                                next_prev_c[c_next][next_idx] = c
                                next_prev_action[c_next][next_idx] = BatteryMode.CHARGE

                    # DISCHARGE
                    if discharge_kwh > 0:
                        new_energy = energy_levels[idx] - discharge_kwh
                        if new_energy >= min_energy - 1e-6:
                            next_idx = int(round((new_energy - min_energy) / step_kwh))
                            next_idx = min(max(next_idx, 0), n_states - 1)
                            next_val = val + (discharge_value * discharge_kwh)
                            if next_val > next_dp[c][next_idx]:
                                next_dp[c][next_idx] = next_val
                                next_prev_idx[c][next_idx] = idx
                                next_prev_c[c][next_idx] = c
                                next_prev_action[c][next_idx] = BatteryMode.DISCHARGE

            dp = next_dp
            prev_idx[t] = next_prev_idx
            prev_c[t] = next_prev_c
            prev_action[t] = next_prev_action

        # Choose best ending state (tie-break on higher SOC)
        best_val = neg_inf
        best_idx = None
        best_c = None
        max_charge_achieved = 0
        for c in range(max_charge_slots + 1):
            for i in range(n_states):
                if dp[c][i] > neg_inf / 2:
                    if c > max_charge_achieved:
                        max_charge_achieved = c
                    if c >= min_charge_slots and dp[c][i] > best_val:
                        best_val = dp[c][i]
                        best_idx = i
                        best_c = c

        if best_idx is None:
            # Fallback if min_charge_slots is infeasible
            for c in range(max_charge_slots + 1):
                for i in range(n_states):
                    if dp[c][i] > best_val:
                        best_val = dp[c][i]
                        best_idx = i
                        best_c = c
            self.log(f"Minimum charge slots not achievable (required {min_charge_slots}, achieved {max_charge_achieved})",
                     level="WARNING")

        # Backtrack actions
        actions: List[BatteryMode] = []
        idx = best_idx if best_idx is not None else start_idx
        c = best_c if best_c is not None else 0
        for t in range(n_slots - 1, -1, -1):
            action = prev_action[t][c][idx] or BatteryMode.HOLD
            actions.append(action)
            prev_i = prev_idx[t][c][idx]
            prev_c_val = prev_c[t][c][idx]
            if prev_i is None or prev_c_val is None:
                idx = idx
                c = c
            else:
                idx = prev_i
                c = prev_c_val
        actions.reverse()

        # Build schedule
        for price_point, action, lk in zip(hours_sorted_by_time, actions, load_kw):
            hour = price_point.hour
            price = price_point.price
            if action == BatteryMode.CHARGE:
                reason = f"Charge @ {price:.4f} EUR/kWh"
            elif action == BatteryMode.DISCHARGE:
                reason = f"Discharge @ {price:.4f} EUR/kWh (load~{lk:.2f}kW)"
            else:
                reason = f"Hold @ {price:.4f} EUR/kWh"
            schedule[hour] = ScheduleEntry(hour=hour, mode=action, reason=reason)

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
            # Cheapest charge candidates
            self.log(f"\nCheapest 5 charge candidates:")
            for i, p in enumerate(all_prices_sorted[:5]):
                marker = " *" if any(s["hour"] == p.hour for s in charge_slots) else ""
                self.log(f"  {i+1}. {p.hour.strftime('%H:%M')} @ {p.price:.4f} EUR/kWh{marker}")

            # Selected charge slots with rankings
            if charge_slots:
                self.log(f"\nSelected charge slots ({len(charge_slots)}):")
                for slot in sorted(charge_slots, key=lambda s: s["hour"]):
                    rank = price_rank.get(slot["hour"], "?")
                    total_prices = len(prices_sorted)
                    self.log(f"  {slot['hour'].strftime('%H:%M')} @ {slot['price']:.4f} EUR/kWh (rank {rank}/{total_prices})")

            # Selected discharge slots
            if discharge_slots:
                self.log(f"\nSelected discharge slots ({len(discharge_slots)}):")
                for slot in sorted(discharge_slots, key=lambda s: s["hour"]):
                    self.log(f"  {slot['hour'].strftime('%H:%M')} @ {slot['price']:.4f} EUR/kWh (load~{slot['load']:.2f}kW)")

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

    def calculate_expected_soc_schedule(self, schedule: Dict[datetime.datetime, ScheduleEntry],
                                        starting_soc: float) -> Dict[datetime.datetime, float]:
        """
        Calculate expected SOC at each slot based on schedule.
        Used for adaptive optimization to detect deviations.

        - Discharge drains at predicted load rate
        - Charge adds energy at charge_rate * efficiency
        """
        expected_soc = {}
        current_soc = starting_soc

        for hour in sorted(schedule.keys()):
            entry = schedule[hour]
            expected_soc[hour] = current_soc

            if entry.mode == BatteryMode.CHARGE:
                # Charging: grid energy * efficiency goes into battery
                energy_added = self.charge_rate * self.efficiency * self.slot_hours
                soc_increase = (energy_added / self.battery_capacity) * 100
                current_soc = min(self.max_soc, current_soc + soc_increase)

            elif entry.mode == BatteryMode.DISCHARGE:
                # Discharging: battery drains at predicted load rate (limited by discharge rate)
                load_kw = self._predict_load_kw(hour)
                energy_removed = min(load_kw, self.discharge_rate) * self.slot_hours
                soc_decrease = (energy_removed / self.battery_capacity) * 100
                current_soc = max(self.min_soc, current_soc - soc_decrease)

            else:  # HOLD
                # In hold mode, grid covers base load
                # Battery has minimal standby drain, negligible for hourly planning
                pass

        return expected_soc

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

        # Calculate expected SOC trajectory
        self.expected_soc_schedule = self.calculate_expected_soc_schedule(self.schedule, current_soc)

        # Log the generated schedule (with expected SOC)
        self._log_schedule(self.schedule, self.expected_soc_schedule)

        self.last_optimization = self.datetime()

        # Sync schedule to inverter TOU registers (if configured)
        if self.tou_sync_enabled and self.device_id:
            self.sync_schedule_to_inverter()

        # Apply current hour's mode
        self.execute_scheduled_mode(None)

        # Update sensor
        self._update_schedule_sensor()

        self.log("Full optimization complete")

    def adaptive_optimize(self, kwargs=None):
        """
        Adaptive re-evaluation on a configurable interval.
        Adjusts schedule based on actual SOC and PV production.
        """
        # Always update battery cost tracking based on SOC changes
        self._update_battery_cost_from_soc_change()

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
            self.set_mode(BatteryMode.HOLD)
            return

        # Check SOC deviation from expected
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

        if expected_soc is not None:
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
            if entry and fraction > 0:
                if entry.mode == BatteryMode.CHARGE:
                    energy_added = self.charge_rate * self.efficiency * self.slot_hours * fraction
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
                # Store trigger context for sensor exposure
                self._last_recalc_trigger = "soc_deviation"
                self._last_recalc_time = self.datetime()
                self._last_soc_deviation = soc_delta

                # Enhanced logging for decision transparency
                if self.decision_log_level >= 1:
                    self.log("=" * 70)
                    self.log("RECALCULATION TRIGGERED: SOC Deviation")
                    self.log("=" * 70)
                    self.log(f"  Expected SOC: {expected_soc_now:.1f}%")
                    self.log(f"  Actual SOC: {current_soc:.1f}%")
                    self.log(f"  Deviation: {soc_delta:+.1f}% (threshold: {self.soc_deviation_threshold}%)")
                    self.log("=" * 70)
                else:
                    self.log(f"SOC deviation detected: actual={current_soc}%, expected={expected_soc_now:.1f}%, delta={soc_delta}%")

                self._recalculate_remaining_schedule(current_soc)

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
                self._log_schedule(self.schedule, self.expected_soc_schedule)

        # Re-evaluate current mode based on updated schedule
        self.execute_scheduled_mode(None)

    def _recalculate_remaining_schedule(self, current_soc: float):
        """
        Recalculate schedule for remaining hours based on current SOC.
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

        self.expected_soc_schedule = self.calculate_expected_soc_schedule(
            {k: v for k, v in self.schedule.items() if is_current_or_future(k)},
            current_soc
        )

        # Sync updated schedule to inverter TOU registers (if configured)
        if self.tou_sync_enabled and self.device_id:
            self.sync_schedule_to_inverter()

        self._update_schedule_sensor()
        # Note: _log_schedule is called by adaptive_optimize after detecting changes

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

            # Update battery cost based on actual SOC change from previous hour
            self._update_battery_cost_from_soc_change()

            # When TOU sync is enabled, avoid hourly set_mode which clears TOU periods (30411=0)
            if self.tou_sync_enabled and self.device_id:
                self.log("TOU sync enabled; skipping hourly set_mode to preserve inverter TOU schedule")
                return
            self.set_mode(entry.mode)
        else:
            self.log(f"No schedule entry for {current_slot}, defaulting to HOLD")
            self._update_battery_cost_from_soc_change()
            if self.tou_sync_enabled and self.device_id:
                self.log("TOU sync enabled; skipping hourly set_mode to preserve inverter TOU schedule")
                return
            self.set_mode(BatteryMode.HOLD)

    def safety_check(self, kwargs=None):
        """
        Safety check every 5 minutes.
        Ensures SOC stays within bounds.
        """
        current_soc = self._get_current_soc()
        if current_soc is None:
            return

        # Stop discharge if SOC too low
        if current_soc <= self.min_soc and self.current_mode == BatteryMode.DISCHARGE:
            self.log(f"Safety: Stopping discharge, SOC at minimum ({current_soc}%)")
            self.set_mode(BatteryMode.HOLD)
            return

        # Stop charge if SOC full
        if current_soc >= self.max_soc and self.current_mode == BatteryMode.CHARGE:
            self.log(f"Safety: Stopping charge, SOC at maximum ({current_soc}%)")
            self.set_mode(BatteryMode.HOLD)
            return

    # =========================================================================
    # VPP Control
    # =========================================================================

    def set_mode(self, mode: BatteryMode, power_percent: int = 100):
        """
        Set the battery mode via VPP protocol registers (30xxx).

        WIT inverters use VPP protocol - legacy registers 201/202 do NOT work!

        VPP Register Reference:
        - 30100: VPP Control Authority (must be 1 to enable control)
        - 30407: Remote Power Control Enable (0=off, 1=on)
        - 30409: Remote Power Percent (-100 to +100, positive=charge, negative=discharge)
        - 30410: AC Charging Enable (1=PV first, required for grid charging)
        - 30411: Number of TOU periods
        - 30412-30414: TOU Period 1 (start, end, power)

        Mode Mapping:
        - CHARGE: 30407=1, 30409=+power_percent (enables AC charging)
        - DISCHARGE: 30407=1, 30409=-power_percent
        - HOLD: TOU with +1% charge - firmware quirk creates TRUE standby state!
                CRITICAL: Simply setting 30407=0 returns to self-consumption where
                battery WILL discharge to supply house load - this is NOT true HOLD!
                CRITICAL: +1% = HOLD, but -1% = FULL DISCHARGE (asymmetric behavior!)
        """
        if not self.device_id:
            self.log(f"No device_id configured, would set mode to {mode.name}", level="WARNING")
            self.current_mode = mode
            self._update_schedule_sensor()
            return

        try:
            # VPP Register addresses
            VPP_CONTROL_AUTHORITY = 30100
            VPP_REMOTE_POWER_ENABLE = 30407
            VPP_REMOTE_POWER_PERCENT = 30409
            VPP_AC_CHARGE_ENABLE = 30410
            VPP_TOU_NUM_PERIODS = 30411
            VPP_TOU_PERIOD1_BASE = 30412

            # Step 1: Enable VPP control authority (persists across power cycles)
            self.call_service("growatt_modbus/write_register",
                device_id=self.device_id,
                register=VPP_CONTROL_AUTHORITY,
                value=1
            )

            if mode == BatteryMode.HOLD:
                # HOLD: Use TOU +1% charge workaround for TRUE standby
                # This is a firmware quirk - +1% charge via TOU creates actual idle state
                # Simply disabling remote control (30407=0) returns to self-consumption
                # where battery WILL discharge to supply house load!
                # CRITICAL: +1% = HOLD, but -1% = FULL DISCHARGE (asymmetric!)
                self.log("Setting HOLD mode via TOU +1% workaround (true standby)")

                # Enable AC charging (required for TOU charge to work)
                try:
                    self.call_service("growatt_modbus/write_register",
                        device_id=self.device_id,
                        register=VPP_AC_CHARGE_ENABLE,
                        value=1
                    )
                except Exception as e:
                    self.log(f"AC charge enable (30410) failed: {e}", level="WARNING")

                # Get current time for TOU period (use HA timezone, not system time)
                now = self.datetime()
                local_tz = self._get_local_timezone()
                if now.tzinfo is not None and local_tz is not None:
                    now = now.astimezone(local_tz)
                current_minutes = now.hour * 60 + now.minute

                # Create TOU period: (now - 5min) to (now + 2 hours) at +1% charge
                start_min = max(0, current_minutes - 5)
                end_min = min(1439, current_minutes + 120)

                # Set num_periods BEFORE writing period data (required by Growatt firmware)
                self.call_service("growatt_modbus/write_register",
                    device_id=self.device_id,
                    register=VPP_TOU_NUM_PERIODS,
                    value=1
                )

                # Write TOU period using function 0x10 (write multiple registers)
                self.call_service("growatt_modbus/write_registers",
                    device_id=self.device_id,
                    register=VPP_TOU_PERIOD1_BASE,
                    values=[start_min, end_min, 1]  # +1% = HOLD (NOT -1% which = full discharge!)
                )

                self.log(f"Set battery mode to HOLD via TOU {start_min//60:02d}:{start_min%60:02d}-{end_min//60:02d}:{end_min%60:02d} @ +1%")

            elif mode == BatteryMode.CHARGE:
                # CHARGE: Enable AC charging, enable remote control, set positive power

                # Clear any HOLD TOU periods first
                self.call_service("growatt_modbus/write_register",
                    device_id=self.device_id,
                    register=VPP_TOU_NUM_PERIODS,
                    value=0
                )

                # Enable AC charging (PV priority - some firmware rejects AC priority=2)
                try:
                    self.call_service("growatt_modbus/write_register",
                        device_id=self.device_id,
                        register=VPP_AC_CHARGE_ENABLE,
                        value=1  # 1=PV priority (safer than 2=AC priority)
                    )
                except Exception as e:
                    self.log(f"AC charge enable (30410) failed: {e} - may not work on all firmware", level="WARNING")

                # Enable remote power control
                self.call_service("growatt_modbus/write_register",
                    device_id=self.device_id,
                    register=VPP_REMOTE_POWER_ENABLE,
                    value=1
                )

                # Set charge power (positive value)
                self.call_service("growatt_modbus/write_register",
                    device_id=self.device_id,
                    register=VPP_REMOTE_POWER_PERCENT,
                    value=power_percent
                )
                self.log(f"Set battery mode to CHARGE at {power_percent}%")

            elif mode == BatteryMode.DISCHARGE:
                # DISCHARGE: Enable remote control, set negative power

                # Clear any HOLD TOU periods first
                self.call_service("growatt_modbus/write_register",
                    device_id=self.device_id,
                    register=VPP_TOU_NUM_PERIODS,
                    value=0
                )

                # Enable remote power control
                self.call_service("growatt_modbus/write_register",
                    device_id=self.device_id,
                    register=VPP_REMOTE_POWER_ENABLE,
                    value=1
                )

                # Set discharge power (negative value, convert to unsigned 16-bit)
                # -100 becomes 65436 (65536 - 100)
                power_value = 65536 - power_percent
                self.call_service("growatt_modbus/write_register",
                    device_id=self.device_id,
                    register=VPP_REMOTE_POWER_PERCENT,
                    value=power_value
                )
                self.log(f"Set battery mode to DISCHARGE at {power_percent}%")

            self.current_mode = mode

            # Update sensor
            self._update_schedule_sensor()

        except Exception as e:
            self.log(f"Error setting battery mode: {e}", level="ERROR")

    # =========================================================================
    # TOU Schedule Sync
    # =========================================================================

    def schedule_to_tou_periods(self) -> List[TouPeriod]:
        """
        Convert the current schedule to TOU periods for inverter programming.

        Consolidates contiguous slots with same mode into single periods.
        All modes are written as TOU periods:
        - CHARGE: +100% (or configured power)
        - DISCHARGE: -100% (or configured power)
        - HOLD: +1% charge (firmware quirk creates TRUE standby)

        CRITICAL: TOU periods CANNOT overlap! End times use XX:59 format.
        Example: Period 1 ends at 05:59 (359 min), Period 2 starts at 06:00 (360 min)

        NOTE: TOU periods are limited to a single day (0-1439 minutes). The schedule
        may span today and tomorrow, so we include entries from both days mapped to
        time-of-day. If the same time-of-day appears on both days (e.g., 22:00 today
        and 22:00 tomorrow), we prefer today's entry since it executes first.

        Returns:
            List of TouPeriod objects (max 20 periods)
        """
        if not self.schedule:
            return []

        import datetime as dt

        periods = []
        current_period_mode = None
        current_period_start = None

        today = self.date()
        tomorrow = today + dt.timedelta(days=1)
        local_tz = self._get_local_timezone()

        def get_local_dt(hour_dt):
            """Convert to local timezone if needed."""
            if hour_dt.tzinfo is not None and local_tz is not None:
                return hour_dt.astimezone(local_tz)
            return hour_dt

        # Build time-of-day schedule from today and tomorrow
        # Key: minutes since midnight (0-1439), Value: (entry, date)
        # Prefer today's entry if same time-of-day appears on both days
        time_of_day_map = {}

        for hour_dt, entry in self.schedule.items():
            local_hour = get_local_dt(hour_dt)
            entry_date = local_hour.date()

            # Only include today and tomorrow
            if entry_date not in (today, tomorrow):
                continue

            minutes = local_hour.hour * 60 + local_hour.minute

            # If conflict, prefer the earlier date (today over tomorrow)
            if minutes in time_of_day_map:
                existing_date = time_of_day_map[minutes][1]
                if entry_date < existing_date:
                    time_of_day_map[minutes] = (entry, entry_date)
            else:
                time_of_day_map[minutes] = (entry, entry_date)

        if not time_of_day_map:
            self.log("No schedule entries for today/tomorrow, skipping TOU sync")
            return []

        self.log(f"TOU sync: {len(time_of_day_map)} time slots from schedule "
                 f"(today: {sum(1 for _, d in time_of_day_map.values() if d == today)}, "
                 f"tomorrow: {sum(1 for _, d in time_of_day_map.values() if d == tomorrow)})")

        # Sort by time-of-day (minutes since midnight)
        sorted_minutes = sorted(time_of_day_map.keys())

        def get_power_for_mode(mode: BatteryMode) -> int:
            """Get TOU power value for mode. HOLD uses +1% (true standby)."""
            if mode == BatteryMode.CHARGE:
                return 100
            elif mode == BatteryMode.DISCHARGE:
                return -100
            else:  # HOLD
                return 1  # +1% charge = TRUE HOLD (firmware quirk)

        for i, hour_minutes in enumerate(sorted_minutes):
            entry, _ = time_of_day_map[hour_minutes]
            mode = entry.mode

            if current_period_mode is None:
                # Start new period
                current_period_mode = mode
                current_period_start = hour_minutes
            elif current_period_mode == mode:
                # Continue current period (contiguous same mode)
                pass
            else:
                # Mode changed - close current period and start new one
                # CRITICAL: End at XX:59 to avoid overlap with next period
                period_end = hour_minutes - 1  # e.g., 06:00 -> 05:59 (359 min)
                power = get_power_for_mode(current_period_mode)
                periods.append(TouPeriod(
                    start=current_period_start,
                    end=period_end,
                    power=power
                ))
                # Start new period
                current_period_mode = mode
                current_period_start = hour_minutes

            # Check if this is the last slot - close any open period
            if i == len(sorted_minutes) - 1 and current_period_mode is not None:
                # End at end of the last slot
                period_end = hour_minutes + self.slot_minutes - 1
                # Handle case where schedule goes to midnight
                if period_end > 1439:
                    period_end = 1439
                power = get_power_for_mode(current_period_mode)
                periods.append(TouPeriod(
                    start=current_period_start,
                    end=period_end,
                    power=power
                ))

        # Limit to 20 periods (inverter maximum)
        if len(periods) > 20:
            self.log(f"TOU schedule has {len(periods)} periods, truncating to 20", level="WARNING")
            periods = periods[:20]

        return periods

    def sync_schedule_to_inverter(self) -> bool:
        """
        Sync the current schedule to the inverter's TOU registers.

        Uses the Growatt Modbus integration to write TOU period registers.
        This allows the inverter to operate autonomously even if HA goes offline.

        TOU Register Reference:
        - 30100: VPP Control Authority (1=enable)
        - 30407: Remote Power Control (0=disable, so TOU takes precedence!)
        - 30410: AC Charging Enable (1=enable, CRITICAL for charge periods!)
        - 30411: Number of active periods (0-20)
        - 30476: Default mode (0=load first/HOLD)
        - 30412-30471: Period data (3 registers per period: start, end, power)

        CRITICAL WRITE SEQUENCE (discovered through testing):
        1. Clear num_periods to 0
        2. Zero out all period registers (stale data causes overlap validation failures!)
        3. Write periods SEQUENTIALLY: write period N data, then set num_periods=N
        4. If any write fails, clear num_periods to deactivate partial schedule

        Key insight: Zeroed registers [0,0,0] are treated as "empty" by firmware.

        Returns:
            True if sync succeeded, False otherwise
        """
        import time

        if not self.device_id:
            self.log("No device_id configured, cannot sync TOU schedule", level="WARNING")
            return False

        try:
            # Convert schedule to TOU periods
            periods = self.schedule_to_tou_periods()
            num_periods = len(periods)

            self.log(f"Syncing {num_periods} TOU periods to inverter")

            # Step 1: Enable VPP control (register 30100 = 1)
            if not self._write_register_with_retry(30100, 1):
                self.log("Failed to enable VPP control", level="ERROR")
                return False

            # Step 2: Enable AC charging (register 30410 = 1) - CRITICAL for charge periods!
            self._write_register_with_retry(30410, 1)
            time.sleep(0.3)

            # Step 3: Disable remote control (register 30407 = 0) so TOU takes precedence
            # Without this, remote control overrides TOU schedule!
            self._write_register_with_retry(30407, 0)
            time.sleep(0.3)

            # Step 4: Set default mode to "load first" / HOLD (register 30476 = 0)
            self._write_register_with_retry(30476, 0)
            time.sleep(0.3)

            # Step 5: Clear existing schedule and zero out ALL period registers
            # CRITICAL: Stale non-zero data in ANY period register causes overlap validation failures!
            # The firmware validates writes against ALL registers, not just active ones.
            # Zeroed registers [0,0,0] are treated as "empty" and allow fresh writes.
            self.log("Clearing TOU schedule and zeroing period registers...")
            self._write_register_with_retry(30411, 0, verify=False)
            time.sleep(0.5)

            # Zero out ALL 20 period registers (not just the ones we'll use!)
            # This is crucial - leftover data from a previous schedule causes "Illegal data value" errors
            MAX_TOU_PERIODS = 20
            for i in range(MAX_TOU_PERIODS):
                base_addr = 30412 + (i * 3)
                # Zero all 3 registers atomically (best effort, don't fail on errors)
                try:
                    self.call_service("growatt_modbus/write_registers",
                        device_id=self.device_id, register=base_addr, values=[0, 0, 0])
                except Exception:
                    pass  # Best effort - continue even if zeroing fails
                time.sleep(0.1)
            time.sleep(0.5)

            # Step 6: Write periods SEQUENTIALLY, incrementing num_periods after each
            # This is the key sequence that works with Growatt firmware:
            # 1. Write period N data (using atomic multi-register write)
            # 2. Set num_periods=N
            # 3. Repeat for next period
            write_failures = 0
            for i, period in enumerate(periods):
                base_addr = 30412 + (i * 3)

                # Convert negative power to unsigned 16-bit
                power_unsigned = period.power if period.power >= 0 else 65536 + period.power

                # Write all 3 registers atomically using multi-register write (function 0x10)
                # This is more reliable than individual writes
                success = self._write_registers_with_retry(
                    base_addr, [period.start, period.end, power_unsigned]
                )

                if success:
                    # Increment num_periods to activate this period
                    if not self._write_register_with_retry(30411, i + 1):
                        self.log(f"Failed to set num_periods to {i+1}", level="WARNING")
                        success = False
                    else:
                        self.log(f"TOU Period {i+1}: {period.start//60:02d}:{period.start%60:02d} - "
                                 f"{period.end//60:02d}:{period.end%60:02d}, power={period.power}%")

                if not success:
                    write_failures += 1
                    self.log(f"TOU Period {i+1} write FAILED", level="ERROR")

                time.sleep(0.5)  # Allow inverter time to process before next period

            # If any period writes failed, clear num_periods to deactivate partial schedule
            if write_failures > 0:
                self.log(f"TOU sync failed: {write_failures} period(s) failed to write, clearing schedule", level="ERROR")
                self._write_register_with_retry(30411, 0, verify=False)
                return False

            # Step 7: Verify num_periods is correct
            time.sleep(0.5)
            try:
                verified = self._read_modbus_registers(30411, 1)
                if verified and len(verified) > 0 and verified[0] == num_periods:
                    self.log(f"TOU sync complete: {num_periods} periods written and verified")
                elif verified is not None:
                    self.log(f"TOU sync verification FAILED: expected {num_periods}, got {verified}", level="ERROR")
                    return False
                else:
                    # Verification not available, assume success based on no write errors
                    self.log(f"TOU sync complete: {num_periods} periods written (verification unavailable)")
            except Exception as e:
                # Verification failed but writes may have succeeded
                self.log(f"TOU sync complete: {num_periods} periods written (verification error: {e})")

            return True

        except Exception as e:
            self.log(f"Error syncing TOU schedule to inverter: {e}", level="ERROR")
            return False

    def _write_modbus_register(self, address: int, value: int):
        """Write a single register via Growatt Modbus integration"""
        if not self.device_id:
            self.log(f"No device_id configured, cannot write register {address}", level="WARNING")
            return

        self.call_service("growatt_modbus/write_register",
            device_id=self.device_id,
            register=address,
            value=value
        )

    def _write_modbus_registers(self, address: int, values: List[int]):
        """Write multiple registers via Growatt Modbus integration (function 0x10)"""
        if not self.device_id:
            self.log(f"No device_id configured, cannot write registers starting at {address}", level="WARNING")
            return

        self.call_service("growatt_modbus/write_registers",
            device_id=self.device_id,
            register=address,
            values=values
        )

    def _read_modbus_registers(self, address: int, count: int = 1) -> Optional[List[int]]:
        """Read holding registers via growatt_modbus service.

        Note: This uses HA service response feature. If the service doesn't
        return data (older HA versions), verification will be skipped.
        """
        if not self.device_id:
            return None
        # Prefer REST API for response-returning services (avoids return_result schema errors)
        rest_values = self._read_modbus_registers_rest(address, count)
        if rest_values is not None:
            return rest_values

        # Fallback to AppDaemon call_service without return_result
        try:
            result = self.call_service(
                "growatt_modbus/get_register_data",
                device_id=self.device_id,
                register_type="holding",
                start_address=address,
                count=count
            )
            if result and isinstance(result, dict) and result.get("success"):
                return result.get("values")
            return None
        except Exception as e:
            self.log(f"Failed to read registers {address}-{address+count-1}: {e}", level="WARNING")
            return None

    def _read_modbus_registers_rest(self, address: int, count: int) -> Optional[List[int]]:
        """Read holding registers via HA REST API with return_response."""
        if not REQUESTS_AVAILABLE:
            return None

        ha_url = self.args.get("ha_url", "").rstrip("/")
        token = self.args.get("ha_token", "")
        if not ha_url or not token:
            return None

        try:
            url = f"{ha_url}/api/services/growatt_modbus/get_register_data?return_response"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            payload = {
                "device_id": self.device_id,
                "register_type": "holding",
                "start_address": address,
                "count": count
            }

            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code != 200:
                self.log(f"REST API read returned status {response.status_code}: {response.text[:200]}", level="DEBUG")
                return None

            data = response.json()
            if isinstance(data, dict):
                if "service_response" in data:
                    data = data["service_response"]
                elif "response" in data:
                    data = data["response"]

            if isinstance(data, dict) and data.get("success"):
                return data.get("values")

            self.log(f"REST API read did not return success for {address}-{address+count-1}", level="DEBUG")
            return None

        except requests.exceptions.RequestException as e:
            self.log(f"REST API read failed for {address}-{address+count-1}: {e}", level="DEBUG")
            return None

    def _write_register_with_retry(self, address: int, value: int, max_retries: int = 3, verify: bool = True) -> bool:
        """Write single register with retry logic and optional verification.

        Args:
            address: Modbus register address
            value: Value to write (unsigned 16-bit)
            max_retries: Number of write attempts
            verify: If True, read back and verify the written value

        Returns:
            True if write (and verification if enabled) succeeded
        """
        import time
        for attempt in range(max_retries):
            try:
                self.call_service("growatt_modbus/write_register",
                    device_id=self.device_id, register=address, value=value)
                time.sleep(0.5)  # Allow inverter to process

                # Verify the write by reading back
                if verify:
                    readback = self._read_modbus_registers(address, 1)
                    if readback and len(readback) > 0:
                        if readback[0] == value:
                            return True
                        else:
                            self.log(f"Register {address} verify failed: wrote {value}, read {readback[0]}", level="WARNING")
                    else:
                        # Verification read failed, but write may have succeeded
                        self.log(f"Register {address} verify read failed (attempt {attempt+1})", level="WARNING")
                else:
                    return True  # No verification requested

            except Exception as e:
                self.log(f"Register {address} write attempt {attempt+1} failed: {e}", level="WARNING")

            if attempt < max_retries - 1:
                time.sleep(0.5)

        return False

    def _write_registers_with_retry(self, address: int, values: List[int], max_retries: int = 3) -> bool:
        """Write multiple registers atomically with retry logic and verification.

        Uses Modbus function 0x10 (write multiple registers) for atomic writes.
        This is more reliable than individual register writes.

        Retry strategy uses exponential backoff to handle:
        - Bus contention (concurrent reads from HA coordinator)
        - Firmware validation processing time
        - Transient communication errors

        Args:
            address: Starting Modbus register address
            values: List of values to write (unsigned 16-bit each)
            max_retries: Number of write attempts

        Returns:
            True if write and verification succeeded
        """
        import time
        base_delay = 0.7  # Initial delay after write

        for attempt in range(max_retries):
            try:
                self.call_service("growatt_modbus/write_registers",
                    device_id=self.device_id, register=address, values=values)

                # Exponential backoff for processing time
                delay = base_delay * (1.5 ** attempt)
                time.sleep(delay)

                # Verify the write by reading back
                readback = self._read_modbus_registers(address, len(values))
                if readback and len(readback) == len(values):
                    if readback == values:
                        return True
                    else:
                        self.log(f"Registers {address}-{address+len(values)-1} verify failed: "
                                 f"wrote {values}, read {readback}", level="WARNING")
                else:
                    self.log(f"Registers {address} verify read failed (attempt {attempt+1})", level="WARNING")

            except Exception as e:
                # Log the actual exception for debugging
                error_str = str(e)
                if "Illegal data value" in error_str or "exception 3" in error_str.lower():
                    self.log(f"Registers {address} write rejected by firmware (Modbus exception 3: Illegal data value). "
                             f"Values: {values}. This may indicate overlap with existing period data.", level="WARNING")
                else:
                    self.log(f"Registers {address} write attempt {attempt+1} failed: {e}", level="WARNING")

            if attempt < max_retries - 1:
                # Exponential backoff between retries
                retry_delay = 1.0 * (2 ** attempt)  # 1s, 2s, 4s
                time.sleep(retry_delay)

        return False

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
    # Battery Cost Tracking
    # =========================================================================

    def _init_battery_cost(self):
        """Initialize battery cost from persistent storage or estimate"""
        # Entity for persisting battery cost
        self.battery_cost_entity = self.args.get("battery_cost_entity", "input_number.battery_avg_cost")

        # Track SOC for measuring actual charge/discharge
        self._last_soc: Optional[float] = self._get_current_soc()
        self._last_mode: BatteryMode = BatteryMode.HOLD
        self._last_hour: Optional[datetime.datetime] = self.datetime().replace(minute=0, second=0, microsecond=0)

        # Try to load from persistent storage
        try:
            state = self.get_state(self.battery_cost_entity)
            if state and state not in ("unknown", "unavailable"):
                self.battery_avg_cost = float(state)
                self.log(f"Loaded battery avg cost from HA: {self.battery_avg_cost:.4f} EUR/kWh")
                return
        except (ValueError, TypeError) as e:
            self.log(f"Could not load battery cost from {self.battery_cost_entity}: {e}", level="WARNING")

        # Fallback: estimate from recent prices
        current_soc = self._get_current_soc()
        if current_soc is None or current_soc <= self.min_soc:
            self.battery_avg_cost = 0.0
            self._save_battery_cost()
            return

        prices = self.get_prices()
        if prices:
            avg_price = sum(p.price for p in prices) / len(prices)
            self.battery_avg_cost = avg_price
            self.log(f"Initialized battery avg cost to {self.battery_avg_cost:.4f} EUR/kWh (estimated from recent prices)")
        else:
            self.battery_avg_cost = 0.10  # Default fallback
            self.log(f"Initialized battery avg cost to {self.battery_avg_cost:.4f} EUR/kWh (default)")

        self._save_battery_cost()

    def _save_battery_cost(self):
        """Persist battery cost to Home Assistant entity"""
        try:
            self.call_service("input_number/set_value",
                entity_id=self.battery_cost_entity,
                value=round(self.battery_avg_cost, 4)
            )
        except Exception as e:
            self.log(f"Could not save battery cost to {self.battery_cost_entity}: {e}", level="DEBUG")

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

    def _update_battery_cost_from_soc_change(self):
        """
        Update battery cost based on actual SOC change since last check.
        Called periodically to track real charging/discharging.
        """
        current_soc = self._get_current_soc()
        current_hour = self._align_to_slot(self.datetime())

        if current_soc is None or self._last_soc is None:
            self._last_soc = current_soc
            return

        soc_change = current_soc - self._last_soc

        # Only process significant changes (> 1%)
        if abs(soc_change) < 1.0:
            self._last_soc = current_soc
            self._last_hour = current_hour  # Always update hour to prevent stale pricing
            return

        energy_change_kwh = abs(soc_change) / 100 * self.battery_capacity

        # Calculate time since last observation
        now = self.datetime()
        if self._last_hour:
            # Ensure consistent timezone handling to avoid naive/aware mismatch
            last_hour = self._last_hour
            if now.tzinfo is not None and last_hour.tzinfo is None:
                last_hour = last_hour.replace(tzinfo=now.tzinfo)
            elif now.tzinfo is None and last_hour.tzinfo is not None:
                now = now.replace(tzinfo=last_hour.tzinfo)
            duration_minutes = (now - last_hour).total_seconds() / 60
        else:
            duration_minutes = 60  # Default to 1 hour

        if soc_change > 0:
            # Battery charged - get price for the charging period
            charge_price = self._get_price_for_hour(self._last_hour) if self._last_hour else None
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
            battery_temp = self._get_battery_temp()
            self.learning_engine.record_charging(
                soc_start=self._last_soc,
                soc_end=current_soc,
                duration_minutes=duration_minutes,
                charge_price=charge_price,
                battery_temp=battery_temp
            )
            self._save_learning_data()
            self._update_learning_sensor()

        elif soc_change < 0:
            # Battery discharged - cost per kWh stays same, just less energy
            discharge_price = self._get_price_for_hour(self._last_hour) if self._last_hour else 0.0

            self.log(f"Battery discharged: {soc_change:.1f}% (-{energy_change_kwh:.2f} kWh), "
                     f"avg cost unchanged: {self.battery_avg_cost:.4f} EUR/kWh")

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
        self._last_hour = current_hour

    def _get_discharge_threshold(self) -> float:
        """Calculate discharge threshold based on actual battery cost"""
        # Threshold = (what we paid + grid fees) / efficiency + wear cost
        # Only discharge if we can "sell" above this price
        threshold = ((self.battery_avg_cost + self.grid_fee) / self.efficiency) + self.battery_wear_cost
        return threshold

    def _get_price_for_hour(self, hour: datetime.datetime) -> Optional[float]:
        """Get the electricity price for a specific slot"""
        local_tz = self._get_local_timezone()
        for price_point in self.cached_prices:
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
        tz = self.datetime().tzinfo
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
                    "prices_cached": len(self.cached_prices),
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

    def _log_schedule(self, schedule: Dict[datetime.datetime, ScheduleEntry],
                      expected_soc: Optional[Dict[datetime.datetime, float]] = None):
        """Log the full schedule in a readable format with optional expected SOC"""
        if not schedule:
            self.log("No schedule to log")
            return

        self.log("=" * 60)
        self.log("GENERATED SCHEDULE")
        self.log("=" * 60)

        local_tz = self._get_local_timezone()
        sorted_hours = sorted(schedule.keys())

        for i, hour in enumerate(sorted_hours):
            entry = schedule[hour]
            # Ensure time is displayed in local timezone
            display_hour = hour
            if hour.tzinfo is not None and local_tz is not None:
                display_hour = hour.astimezone(local_tz)
            time_str = display_hour.strftime("%Y-%m-%d %H:%M")
            mode_str = entry.mode.name.ljust(9)

            # Calculate end-of-slot SOC if expected_soc is provided
            soc_str = ""
            if expected_soc:
                start_soc = self._get_expected_soc_for_hour(expected_soc, hour, local_tz)
                if start_soc is not None:
                    # Calculate end-of-slot SOC based on the action
                    if entry.mode == BatteryMode.CHARGE:
                        energy_added = self.charge_rate * self.efficiency * self.slot_hours
                        end_soc = min(self.max_soc, start_soc + (energy_added / self.battery_capacity) * 100)
                    elif entry.mode == BatteryMode.DISCHARGE:
                        load_kw = self._predict_load_kw(hour)
                        energy_removed = min(load_kw, self.discharge_rate) * self.slot_hours
                        end_soc = max(self.min_soc, start_soc - (energy_removed / self.battery_capacity) * 100)
                    else:  # HOLD
                        end_soc = start_soc
                    soc_str = f" -> {end_soc:5.1f}%"

            self.log(f"  {time_str}  {mode_str}  {entry.reason}{soc_str}")

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
