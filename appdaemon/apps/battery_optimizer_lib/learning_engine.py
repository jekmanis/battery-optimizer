"""
Self-learning battery performance tracking engine.

Learns actual charge rates, SOC-dependent behavior, temperature effects,
and round-trip efficiency from observed battery performance.
"""

import datetime
import json
import statistics
from typing import Dict, List, Optional, Tuple

from .models import LearningStats


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
        - "<5" for temps below 5C
        - "5-10" for temps 5-10C
        - "10-15" for temps 10-15C
        - "15-20" for temps 15-20C
        - ">20" for temps above 20C
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
        battery_temp: Optional[float] = None,
        battery_temp_start: Optional[float] = None,
        battery_temp_end: Optional[float] = None,
        energy_to_battery_kwh: Optional[float] = None
    ):
        """
        Record a charging observation and update learned parameters.

        Args:
            soc_start: SOC at start of charging
            soc_end: SOC at end of charging
            duration_minutes: How long charging took
            energy_from_grid_kwh: Energy drawn from grid (if available from meter)
            charge_price: Price paid per kWh
            battery_temp: Battery temperature in Celsius (if available) - used for charge rate bucketing
            battery_temp_start: Battery temperature at start of charging (for warming rate tracking)
            battery_temp_end: Battery temperature at end of charging (for warming rate tracking)
            energy_to_battery_kwh: Actual energy stored in battery (from inverter sensor, more precise than SOC)
        """
        if duration_minutes <= 0 or soc_end <= soc_start:
            return

        # Use measured energy if available, otherwise calculate from SOC
        if energy_to_battery_kwh is not None and energy_to_battery_kwh > 0:
            energy_added = energy_to_battery_kwh
            energy_source = "inverter"
        else:
            energy_added = (soc_end - soc_start) / 100 * self.battery_capacity
            energy_source = "soc"

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

        # Update SOC-range specific charge rates (fallback when temperature unavailable)
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

        # Track temperature warming rate during charging
        # This helps predict when the inverter will switch to higher charge power
        if (battery_temp_start is not None and battery_temp_end is not None and
                duration_minutes >= 1.0):
            temp_change = battery_temp_end - battery_temp_start
            # Only record positive warming (battery heating up during charge)
            if temp_change > 0:
                warming_rate = temp_change / duration_minutes  # °C/minute
                start_temp_range = self._get_temp_range(battery_temp_start)

                if start_temp_range not in self.stats.temp_warming_rates:
                    self.stats.temp_warming_rates[start_temp_range] = []
                self.stats.temp_warming_rates[start_temp_range].append(warming_rate)

                # Keep last 50 observations per starting temp range
                if len(self.stats.temp_warming_rates[start_temp_range]) > 50:
                    self.stats.temp_warming_rates[start_temp_range] = \
                        self.stats.temp_warming_rates[start_temp_range][-50:]

        # Update totals
        self.stats.total_energy_charged_kwh += energy_added
        self.stats.total_charge_cost_eur += energy_added * charge_price

        # Update timestamps
        now = datetime.datetime.now().isoformat()
        if self.stats.first_observation is None:
            self.stats.first_observation = now
        self.stats.last_observation = now

        temp_str = f", temp={battery_temp:.1f}C" if battery_temp is not None else ""
        # Get observation count for this bucket
        obs_count = len(self.stats.charge_rates_by_soc.get(soc_range, []))
        self.log(f"Learning: Recorded charge {soc_start:.1f}%->{soc_end:.1f}% ({energy_added:.3f} kWh [{energy_source}]) "
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

        # Use measured energy if available, otherwise calculate from SOC
        if energy_delivered_kwh is not None and energy_delivered_kwh > 0:
            energy_source = "inverter"
        else:
            energy_delivered_kwh = (soc_start - soc_end) / 100 * self.battery_capacity
            energy_source = "soc"

        # Calculate observed discharge rate
        discharge_rate = energy_delivered_kwh / (duration_minutes / 60)

        # Update totals
        self.stats.total_energy_discharged_kwh += energy_delivered_kwh
        self.stats.total_discharge_revenue_eur += energy_delivered_kwh * price_eur_kwh

        self.log(f"Learning: Recorded discharge {soc_start:.1f}%->{soc_end:.1f}% ({energy_delivered_kwh:.3f} kWh [{energy_source}]) "
                 f"in {duration_minutes:.0f}min, rate={discharge_rate:.2f}kW")

    def record_temperature_observation(self, temp: float):
        """
        Record a battery temperature observation for ambient temperature estimation.

        Keeps a rolling window of recent observations to estimate ambient temperature
        (minimum temperature the battery reaches when idle).

        Args:
            temp: Current battery temperature (°C)
        """
        if temp is None:
            return

        self.stats.recent_min_temps.append(temp)

        # Keep last ~48 observations (assuming hourly recording, ~2 days of data)
        max_observations = 48
        if len(self.stats.recent_min_temps) > max_observations:
            self.stats.recent_min_temps = self.stats.recent_min_temps[-max_observations:]

    def get_estimated_ambient_temp(self, default: float = 10.0) -> float:
        """
        Get estimated ambient temperature based on recent minimum observations.

        Uses the minimum of recent battery temperature observations as a proxy
        for ambient temperature (coldest the battery gets when idle).

        Args:
            default: Default ambient if no observations available (°C)

        Returns:
            Estimated ambient temperature (°C)
        """
        if not self.stats.recent_min_temps:
            return default

        # Use minimum of recent observations as ambient estimate
        # Add small buffer (1°C) since battery may not fully reach ambient
        min_temp = min(self.stats.recent_min_temps)
        return min_temp

    def record_cooling(
        self,
        temp_start: float,
        temp_end: float,
        duration_minutes: float,
        ambient_temp: Optional[float] = None
    ):
        """
        Record a cooling observation during idle (HOLD/DISCHARGE) periods.

        Calculates the exponential decay rate and stores it by starting temperature range.

        Args:
            temp_start: Temperature at start of idle period (°C)
            temp_end: Temperature at end of idle period (°C)
            duration_minutes: Duration of idle period (minutes)
            ambient_temp: Ambient temperature estimate (°C), uses estimated if not provided
        """
        # Use estimated ambient if not provided
        if ambient_temp is None:
            ambient_temp = self.get_estimated_ambient_temp(default=10.0)

        # Only record if we have meaningful cooling (temp dropped toward ambient)
        if duration_minutes < 1.0:
            return
        if temp_start <= ambient_temp:
            return  # Already at or below ambient, no cooling to measure
        if temp_end >= temp_start:
            return  # Temperature didn't drop (maybe it even rose)
        if temp_end < ambient_temp:
            return  # Dropped below ambient - invalid data

        # Calculate the decay rate from the exponential decay formula:
        # T(t) = ambient + (start - ambient) * e^(-rate * t)
        # Solving for rate: rate = -ln((end - ambient) / (start - ambient)) / t
        import math
        temp_diff_start = temp_start - ambient_temp
        temp_diff_end = temp_end - ambient_temp

        if temp_diff_end <= 0 or temp_diff_start <= 0:
            return  # Avoid math errors

        ratio = temp_diff_end / temp_diff_start
        if ratio <= 0 or ratio >= 1:
            return  # Invalid ratio

        cooling_rate = -math.log(ratio) / duration_minutes

        # Sanity check: rate should be positive and reasonable (0.001 to 0.1 per minute)
        if cooling_rate <= 0.001 or cooling_rate > 0.1:
            return

        # Store by starting temperature range
        start_temp_range = self._get_temp_range(temp_start)

        if start_temp_range not in self.stats.temp_cooling_rates:
            self.stats.temp_cooling_rates[start_temp_range] = []
        self.stats.temp_cooling_rates[start_temp_range].append(cooling_rate)

        # Keep last 50 observations per starting temp range
        if len(self.stats.temp_cooling_rates[start_temp_range]) > 50:
            self.stats.temp_cooling_rates[start_temp_range] = \
                self.stats.temp_cooling_rates[start_temp_range][-50:]

        self.log(f"Learning: Recorded cooling {temp_start:.1f}C->{temp_end:.1f}C "
                 f"in {duration_minutes:.0f}min, rate={cooling_rate:.4f}/min, "
                 f"bucket={start_temp_range}")

    def get_cooling_rate(self, starting_temp: float) -> Optional[float]:
        """
        Get predicted cooling rate for a given starting temperature.

        Args:
            starting_temp: Battery temperature at start of idle period (°C)

        Returns:
            Cooling rate (decay per minute), or None if insufficient data
        """
        temp_range = self._get_temp_range(starting_temp)

        if temp_range in self.stats.temp_cooling_rates:
            rates = self.stats.temp_cooling_rates[temp_range]
            if len(rates) >= 3:
                return statistics.median(rates[-10:])

        # Try adjacent temperature ranges if exact match not found
        for try_temp in [starting_temp - 5, starting_temp + 5]:
            try_range = self._get_temp_range(try_temp)
            if try_range in self.stats.temp_cooling_rates:
                rates = self.stats.temp_cooling_rates[try_range]
                if len(rates) >= 3:
                    return statistics.median(rates[-10:])

        return None

    def get_charge_rate_for_soc(self, soc: float, battery_temp: Optional[float] = None) -> float:
        """
        Get predicted charge rate for a given SOC level and optional temperature.
        Uses learned data with fallback chain:
        1. Exact SOC+temp match (>=3 observations) -> median of last 10
        2. SOC match, aggregate all temps
        3. SOC-only data (when temperature unavailable)
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

        # Fallback 3: Use SOC-only data (when temperature unavailable)
        if soc_range in self.stats.charge_rates_by_soc:
            observations = self.stats.charge_rates_by_soc[soc_range]
            if len(observations) >= 3:
                return statistics.median(observations[-10:])

        # Fallback 4: Use configured nominal charge rate
        multiplier = self.soc_charge_multipliers.get(soc_range, 1.0)
        return self.nominal_charge_rate * multiplier

    def get_warming_rate(self, starting_temp: float) -> Optional[float]:
        """
        Get predicted battery warming rate during charging for a given starting temperature.

        Args:
            starting_temp: Battery temperature at start of charging (°C)

        Returns:
            Warming rate in °C/minute, or None if insufficient data
        """
        temp_range = self._get_temp_range(starting_temp)

        if temp_range in self.stats.temp_warming_rates:
            rates = self.stats.temp_warming_rates[temp_range]
            if len(rates) >= 3:
                return statistics.median(rates[-10:])

        # Try adjacent temperature ranges if exact match not found
        for try_temp in [starting_temp - 5, starting_temp + 5]:
            try_range = self._get_temp_range(try_temp)
            if try_range in self.stats.temp_warming_rates:
                rates = self.stats.temp_warming_rates[try_range]
                if len(rates) >= 3:
                    return statistics.median(rates[-10:])

        return None

    def predict_temp_after_duration(
        self,
        start_temp: float,
        duration_minutes: float
    ) -> float:
        """
        Predict battery temperature after charging for a given duration.

        Args:
            start_temp: Starting battery temperature (°C)
            duration_minutes: Charging duration in minutes

        Returns:
            Predicted temperature after charging
        """
        warming_rate = self.get_warming_rate(start_temp)
        if warming_rate is None:
            # Default warming rate if no data (conservative estimate)
            warming_rate = 0.1  # °C/minute

        return start_temp + (warming_rate * duration_minutes)

    def predict_temp_after_idle(
        self,
        start_temp: float,
        duration_minutes: float,
        ambient_temp: Optional[float] = None,
        default_cooling_rate: float = 0.012
    ) -> float:
        """
        Predict battery temperature after idle (not charging) for a given duration.

        Uses exponential decay toward ambient temperature. First tries to use
        learned cooling rate for the starting temperature, falls back to default.

        Args:
            start_temp: Starting battery temperature (°C)
            duration_minutes: Idle duration in minutes
            ambient_temp: Ambient temperature to decay toward (°C), uses estimated if not provided
            default_cooling_rate: Fallback rate if no learned data available
                         0.012 means battery loses ~50% of excess temp per hour
                         (e.g., 21°C → ~18°C after 1 hour with 15°C ambient)

        Returns:
            Predicted temperature after idle period
        """
        # Use estimated ambient if not provided
        if ambient_temp is None:
            ambient_temp = self.get_estimated_ambient_temp(default=10.0)

        if start_temp <= ambient_temp:
            # Already at or below ambient, won't cool further
            return start_temp

        # Try to use learned cooling rate, fall back to default
        learned_rate = self.get_cooling_rate(start_temp)
        cooling_rate = learned_rate if learned_rate is not None else default_cooling_rate

        # Exponential decay: temp approaches ambient
        # T(t) = ambient + (start - ambient) * e^(-rate * t)
        import math
        temp_diff = start_temp - ambient_temp
        decay_factor = math.exp(-cooling_rate * duration_minutes)
        return ambient_temp + (temp_diff * decay_factor)

    def get_time_to_reach_temp(
        self,
        start_temp: float,
        target_temp: float
    ) -> Optional[float]:
        """
        Predict how long it will take to reach a target temperature during charging.

        Args:
            start_temp: Starting battery temperature (°C)
            target_temp: Target temperature to reach (°C)

        Returns:
            Time in minutes to reach target temp, or None if won't warm up
        """
        if target_temp <= start_temp:
            return 0.0

        warming_rate = self.get_warming_rate(start_temp)
        if warming_rate is None or warming_rate <= 0:
            return None

        return (target_temp - start_temp) / warming_rate

    def predict_charge_energy_with_warming(
        self,
        current_soc: float,
        start_temp: float,
        duration_minutes: float,
        temp_threshold: float = 16.0
    ) -> Tuple[float, float]:
        """
        Predict charge energy accounting for temperature warming during charging.

        The inverter may charge faster once the battery warms above a threshold.
        This method calculates total energy by splitting the duration into
        cold and warm periods.

        Args:
            current_soc: Current state of charge (%)
            start_temp: Starting battery temperature (°C)
            duration_minutes: Total charging duration (minutes)
            temp_threshold: Temperature above which faster charging occurs (°C)

        Returns:
            Tuple of (total_energy_kwh, end_temperature)
        """
        total_energy = 0.0
        remaining_minutes = duration_minutes
        current_temp = start_temp

        # If already warm, use warm charge rate for entire duration
        if start_temp >= temp_threshold:
            charge_rate = self.get_charge_rate_for_soc(current_soc, start_temp)
            total_energy = charge_rate * (duration_minutes / 60)
            # Predict end temperature
            end_temp = self.predict_temp_after_duration(start_temp, duration_minutes)
            return total_energy, end_temp

        # Calculate time to reach threshold temperature
        time_to_warm = self.get_time_to_reach_temp(start_temp, temp_threshold)

        if time_to_warm is not None and time_to_warm < duration_minutes:
            # Phase 1: Cold charging until reaching threshold
            cold_rate = self.get_charge_rate_for_soc(current_soc, start_temp)
            cold_energy = cold_rate * (time_to_warm / 60)
            total_energy += cold_energy

            # Phase 2: Warm charging for remaining time
            warm_minutes = remaining_minutes - time_to_warm
            warm_rate = self.get_charge_rate_for_soc(current_soc, temp_threshold)
            warm_energy = warm_rate * (warm_minutes / 60)
            total_energy += warm_energy

            # End temperature continues warming
            end_temp = self.predict_temp_after_duration(temp_threshold, warm_minutes)
        else:
            # Won't reach threshold during this duration - use cold rate throughout
            cold_rate = self.get_charge_rate_for_soc(current_soc, start_temp)
            total_energy = cold_rate * (duration_minutes / 60)
            end_temp = self.predict_temp_after_duration(start_temp, duration_minutes)

        return total_energy, end_temp

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

        # Build warming rates summary
        warming_rates_summary = {}
        for temp_range, rates in self.stats.temp_warming_rates.items():
            if rates:
                count = len(rates)
                warming_rates_summary[temp_range] = {
                    "median_c_per_min": round(statistics.median(rates[-10:]), 3),
                    "observations": count
                }

        # Build cooling rates summary
        cooling_rates_summary = {}
        for temp_range, rates in self.stats.temp_cooling_rates.items():
            if rates:
                count = len(rates)
                cooling_rates_summary[temp_range] = {
                    "median_rate_per_min": round(statistics.median(rates[-10:]), 4),
                    "observations": count
                }

        # Estimated ambient temperature
        estimated_ambient = self.get_estimated_ambient_temp(default=10.0)

        return {
            "learned_efficiency": round(self.learned_efficiency, 3),
            "total_energy_charged_kwh": round(self.stats.total_energy_charged_kwh, 1),
            "total_energy_discharged_kwh": round(self.stats.total_energy_discharged_kwh, 1),
            "overall_efficiency": round(overall_efficiency, 3),
            "total_profit_eur": round(total_profit, 2),
            "total_observations": total_observations,
            "soc_charge_rates": soc_charge_rates,
            "temp_aware_rates": temp_aware_rates,
            "temp_warming_rates": warming_rates_summary,
            "temp_cooling_rates": cooling_rates_summary,
            "estimated_ambient_temp": round(estimated_ambient, 1),
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
