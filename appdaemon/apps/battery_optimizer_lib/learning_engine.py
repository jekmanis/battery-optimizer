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

        temp_str = f", temp={battery_temp:.1f}C" if battery_temp is not None else ""
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
