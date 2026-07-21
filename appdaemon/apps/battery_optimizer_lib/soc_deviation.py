"""
SOC Deviation Detection for Battery Optimizer.

This module handles detection of significant deviations between actual and expected
battery SOC, determining whether schedule recalculation is needed.
"""

import datetime
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .models import BatteryMode, ScheduleEntry
from .timezone_utils import ensure_local_tz, lookup_by_time


@dataclass
class SocDeviationConfig:
    """Configuration for SOC deviation detection."""
    slot_minutes: int
    charge_rate: float  # kW
    discharge_rate: float  # kW
    efficiency: float
    battery_capacity: float  # kWh
    min_soc: float  # %
    max_soc: float  # %
    soc_deviation_threshold: float  # %
    grid_fee: float  # EUR/kWh
    import_price_multiplier: float = 1.0
    inverter_efficiency: float = 1.0
    decision_log_level: int = 1

    @property
    def slot_hours(self) -> float:
        return self.slot_minutes / 60.0


@dataclass
class DeviationCheckResult:
    """Result of SOC deviation check."""
    should_recalculate: bool = False
    deviation: Optional[float] = None
    extra_charge_slots: int = 0
    # For logging context
    log_messages: List[str] = field(default_factory=list)


class SocDeviationDetector:
    """
    Detects significant SOC deviations from expected values.

    This class encapsulates the logic for:
    - Interpolating expected SOC within a time slot
    - Determining if deviation exceeds threshold
    - Checking if deviation during CHARGE can still reach target
    - Calculating extra charge slots needed for catch-up
    - Economic evaluation of catch-up charging
    """

    def __init__(
        self,
        config: SocDeviationConfig,
        learning_engine: Optional[object] = None,
        log_func: Optional[Callable[..., None]] = None,
    ):
        """
        Initialize the detector.

        Args:
            config: Configuration parameters
            learning_engine: Optional BatteryLearningEngine for temperature-aware rates
            log_func: Optional logging function
        """
        self.config = config
        self.learning_engine = learning_engine
        self._log = log_func or (lambda *args, **kwargs: None)

    def check_deviation(
        self,
        current_soc: float,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        expected_soc_schedule: Dict[datetime.datetime, float],
        now: datetime.datetime,
        current_slot: datetime.datetime,
        local_tz,
        current_temp: Optional[float] = None,
        predict_load_kw: Optional[Callable[[datetime.datetime], float]] = None,
        get_cheapest_upcoming_prices: Optional[Callable[[List[datetime.datetime], int], List[float]]] = None,
        get_discharge_threshold: Optional[Callable[[], float]] = None,
    ) -> DeviationCheckResult:
        """
        Check if current SOC deviates significantly from expected.

        Args:
            current_soc: Current battery state of charge (%)
            schedule: Current schedule mapping slot times to entries
            expected_soc_schedule: Expected SOC at start of each slot
            now: Current datetime
            current_slot: Current slot (aligned datetime)
            local_tz: Local timezone
            current_temp: Current battery temperature (Celsius), optional
            predict_load_kw: Function to predict load for a slot
            get_cheapest_upcoming_prices: Function to get N cheapest prices from remaining HOLD slots
            get_discharge_threshold: Function to get current discharge threshold

        Returns:
            DeviationCheckResult with recalculation decision and context
        """
        result = DeviationCheckResult()

        # Get expected SOC for current slot (with timezone-aware lookup)
        expected_soc = lookup_by_time(expected_soc_schedule, current_slot, local_tz)
        if expected_soc is None:
            return result  # No expected SOC, can't detect deviation

        # Get schedule entry for current slot
        entry = lookup_by_time(schedule, current_slot, local_tz)

        # Calculate time fraction into current slot
        minutes_into_slot = max(0.0, (now - current_slot).total_seconds() / 60.0)
        fraction = min(1.0, minutes_into_slot / max(1, self.config.slot_minutes))

        # Interpolate expected SOC based on elapsed time in slot
        expected_soc_now = self._interpolate_expected_soc(
            expected_soc, entry, fraction, current_soc, current_slot, current_temp,
            predict_load_kw,
        )

        soc_delta = current_soc - expected_soc_now
        result.deviation = soc_delta

        # Check if deviation exceeds threshold
        if abs(soc_delta) <= self.config.soc_deviation_threshold:
            return result  # Within tolerance

        # Check special cases where we should skip recalculation
        projected_final_soc = None

        # During CHARGE with negative deviation: check if we'll still reach max_soc
        if entry and entry.mode == BatteryMode.CHARGE and soc_delta < 0:
            projected_final_soc = self._project_charge_completion(
                current_soc, schedule, current_slot, fraction, current_temp, local_tz
            )

            # If we'll still reach max_soc (with 5% tolerance), skip recalculation
            if projected_final_soc >= self.config.max_soc - 5:
                result.log_messages.append(
                    f"SOC behind by {abs(soc_delta):.1f}% during CHARGE (actual={current_soc:.1f}%, "
                    f"expected={expected_soc_now:.1f}%), but projected to reach {projected_final_soc:.1f}% "
                    f"with remaining charge hours - skipping recalculation"
                )
                return result

        # During DISCHARGE with positive deviation: favorable, skip unless very large
        if entry and entry.mode == BatteryMode.DISCHARGE and soc_delta > 0:
            if soc_delta <= self.config.soc_deviation_threshold * 2:
                result.log_messages.append(
                    f"SOC ahead by {soc_delta:.1f}% during DISCHARGE (actual={current_soc:.1f}%, "
                    f"expected={expected_soc_now:.1f}%) - favorable deviation, skipping recalculation"
                )
                return result

        # Deviation is significant - need to recalculate
        result.should_recalculate = True

        # Calculate extra charge slots if behind during CHARGE
        if entry and entry.mode == BatteryMode.CHARGE and soc_delta < 0 and projected_final_soc is not None:
            result.extra_charge_slots = self._calculate_extra_charge_slots(
                current_soc,
                projected_final_soc,
                current_temp,
                schedule,
                current_slot,
                local_tz,
                get_cheapest_upcoming_prices,
                get_discharge_threshold,
                result.log_messages,
            )

        # Build decision logging
        if self.config.decision_log_level >= 1:
            result.log_messages.append("=" * 70)
            result.log_messages.append("RECALCULATION TRIGGERED: SOC Deviation")
            result.log_messages.append("=" * 70)
            result.log_messages.append(f"  Expected SOC: {expected_soc_now:.1f}%")
            result.log_messages.append(f"  Actual SOC: {current_soc:.1f}%")
            result.log_messages.append(f"  Deviation: {soc_delta:+.1f}% (threshold: {self.config.soc_deviation_threshold}%)")
            if result.extra_charge_slots > 0:
                result.log_messages.append(f"  Extra charge slots requested: {result.extra_charge_slots}")
            result.log_messages.append("=" * 70)
        else:
            result.log_messages.append(
                f"SOC deviation detected: actual={current_soc}%, expected={expected_soc_now:.1f}%, delta={soc_delta}%"
            )

        return result

    def _interpolate_expected_soc(
        self,
        expected_soc_start: float,
        entry: Optional[ScheduleEntry],
        fraction: float,
        current_soc: float,
        current_slot: datetime.datetime,
        current_temp: Optional[float],
        predict_load_kw: Optional[Callable[[datetime.datetime], float]],
    ) -> float:
        """
        Interpolate expected SOC based on elapsed time within the slot.

        Args:
            expected_soc_start: Expected SOC at start of slot
            entry: Schedule entry for current slot (may be None)
            fraction: Fraction of slot elapsed (0.0 to 1.0)
            current_soc: Current actual SOC (used for rate lookups)
            current_slot: Current slot datetime
            current_temp: Current battery temperature
            predict_load_kw: Load prediction function

        Returns:
            Interpolated expected SOC for current time within slot
        """
        if not entry or fraction <= 0:
            return expected_soc_start

        if entry.mode == BatteryMode.CHARGE:
            # Use learned charge rate if available
            effective_charge_rate = self.config.charge_rate
            if self.learning_engine:
                learned_rate = self.learning_engine.get_charge_rate_for_soc(current_soc, current_temp)
                if learned_rate is not None and learned_rate > 0:
                    effective_charge_rate = learned_rate

            energy_added = effective_charge_rate * self.config.efficiency * self.config.slot_hours * fraction
            return min(
                self.config.max_soc,
                expected_soc_start + (energy_added / self.config.battery_capacity) * 100
            )

        elif entry.mode == BatteryMode.DISCHARGE:
            if predict_load_kw:
                load_kw = predict_load_kw(current_slot)
            else:
                load_kw = self.config.discharge_rate  # Fallback
            energy_removed = min(load_kw, self.config.discharge_rate) * self.config.slot_hours * fraction
            return max(
                self.config.min_soc,
                expected_soc_start - (energy_removed / self.config.battery_capacity) * 100
            )

        # HOLD mode - no change
        return expected_soc_start

    def _project_charge_completion(
        self,
        current_soc: float,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        current_slot: datetime.datetime,
        fraction: float,
        current_temp: Optional[float],
        local_tz,
    ) -> float:
        """
        Project final SOC after all remaining scheduled charge slots complete.

        Accounts for temperature warming during charging which can increase charge rate.

        Args:
            current_soc: Current SOC
            schedule: Current schedule
            current_slot: Current slot
            fraction: Fraction of current slot elapsed
            current_temp: Current battery temperature
            local_tz: Local timezone

        Returns:
            Projected final SOC after all charge slots complete
        """
        remaining_charge_energy = 0.0
        projected_temp = current_temp if current_temp is not None else 15.0
        temp_threshold = 16.0  # Temperature where charge rate typically increases

        for future_hour in sorted(schedule.keys()):
            # Compare with timezone handling
            cmp_future = self._normalize_for_compare(future_hour, local_tz)
            cmp_current = self._normalize_for_compare(current_slot, local_tz)

            if cmp_future >= cmp_current:
                future_entry = schedule.get(future_hour)
                if future_entry and future_entry.mode == BatteryMode.CHARGE:
                    # For current slot, only count remaining time
                    if cmp_future == cmp_current:
                        remaining_minutes = (1.0 - fraction) * self.config.slot_minutes
                    else:
                        remaining_minutes = self.config.slot_minutes

                    # Use warming-aware projection if learning engine available
                    if self.learning_engine and current_temp is not None:
                        energy, projected_temp = self.learning_engine.predict_charge_energy_with_warming(
                            current_soc=current_soc,
                            start_temp=projected_temp,
                            duration_minutes=remaining_minutes,
                            temp_threshold=temp_threshold
                        )
                        remaining_charge_energy += energy * self.config.efficiency
                    else:
                        # Fallback: use simple rate-based calculation
                        effective_charge_rate = self.config.charge_rate
                        if self.learning_engine:
                            learned_rate = self.learning_engine.get_charge_rate_for_soc(current_soc, projected_temp)
                            if learned_rate is not None and learned_rate > 0:
                                effective_charge_rate = learned_rate
                        remaining_charge_energy += effective_charge_rate * self.config.efficiency * (remaining_minutes / 60)

        remaining_soc_gain = (remaining_charge_energy / self.config.battery_capacity) * 100
        return current_soc + remaining_soc_gain

    def _calculate_extra_charge_slots(
        self,
        current_soc: float,
        projected_final_soc: float,
        current_temp: Optional[float],
        schedule: Dict[datetime.datetime, ScheduleEntry],
        current_slot: datetime.datetime,
        local_tz,
        get_cheapest_upcoming_prices: Optional[Callable[[List[datetime.datetime], int], List[float]]],
        get_discharge_threshold: Optional[Callable[[], float]],
        log_messages: List[str],
    ) -> int:
        """
        Calculate extra charge slots needed for catch-up charging.

        Only adds slots if economically beneficial (charge cost < discharge threshold).

        Args:
            current_soc: Current SOC
            projected_final_soc: Projected final SOC with current schedule
            current_temp: Current battery temperature
            schedule: Current schedule
            current_slot: Current slot
            local_tz: Local timezone
            get_cheapest_upcoming_prices: Function to get cheap prices from HOLD slots
            get_discharge_threshold: Function to get discharge threshold
            log_messages: List to append log messages

        Returns:
            Number of extra charge slots to add (0 if not beneficial)
        """
        if get_cheapest_upcoming_prices is None or get_discharge_threshold is None:
            return 0

        # Calculate energy deficit
        soc_deficit = self.config.max_soc - projected_final_soc
        energy_deficit_kwh = (soc_deficit / 100) * self.config.battery_capacity

        # Get effective charge rate
        effective_charge_rate = self.config.charge_rate
        if self.learning_engine:
            learned_rate = self.learning_engine.get_charge_rate_for_soc(current_soc, current_temp)
            if learned_rate is not None and learned_rate > 0:
                effective_charge_rate = learned_rate

        energy_per_slot = effective_charge_rate * self.config.efficiency * self.config.slot_hours
        if energy_per_slot <= 0:
            return 0

        extra_slots_needed = math.ceil(energy_deficit_kwh / energy_per_slot)
        if extra_slots_needed <= 0:
            return 0

        # Get future hours for price lookup
        remaining_hours = []
        for h in sorted(schedule.keys()):
            cmp_h = self._normalize_for_compare(h, local_tz)
            cmp_current = self._normalize_for_compare(current_slot, local_tz)
            if cmp_h > cmp_current:
                remaining_hours.append(h)

        # Check if economically beneficial
        upcoming_prices = get_cheapest_upcoming_prices(remaining_hours, extra_slots_needed)
        if not upcoming_prices:
            log_messages.append(
                f"Charging behind schedule: projected {projected_final_soc:.1f}% vs target {self.config.max_soc}%, "
                f"but no HOLD slots available for extra charging"
            )
            return 0

        avg_extra_charge_price = sum(upcoming_prices) / len(upcoming_prices)
        discharge_threshold_ac = get_discharge_threshold()

        # Economic check in landed EUR per stored DC kWh. The persisted battery
        # cost and its discharge threshold use the same loss-aware basis.
        charge_cost = (
            (avg_extra_charge_price + self.config.grid_fee)
            * self.config.import_price_multiplier
            / max(1e-9, self.config.efficiency * self.config.inverter_efficiency)
        )
        discharge_value_dc = (
            discharge_threshold_ac * max(1e-9, self.config.inverter_efficiency)
        )
        if charge_cost < discharge_value_dc:
            log_messages.append(
                f"Charging behind schedule: projected {projected_final_soc:.1f}% vs target {self.config.max_soc}%, "
                f"adding {extra_slots_needed} slot(s) at avg {avg_extra_charge_price:.4f} EUR/kWh "
                f"(landed DC charge cost {charge_cost:.4f} < "
                f"stored-DC discharge value {discharge_value_dc:.4f})"
            )
            return extra_slots_needed
        else:
            log_messages.append(
                f"Charging behind schedule but extra charging not economical: "
                f"projected {projected_final_soc:.1f}% vs target {self.config.max_soc}%, "
                f"landed DC charge cost {charge_cost:.4f} >= "
                f"stored-DC discharge value {discharge_value_dc:.4f} "
                f"(avg price {avg_extra_charge_price:.4f}, fee {self.config.grid_fee:.4f})"
            )
            return 0

    def _normalize_for_compare(self, dt: datetime.datetime, local_tz) -> datetime.datetime:
        """Normalize datetime for comparison, handling mixed timezone-aware/naive."""
        if dt is None:
            return dt
        if local_tz is not None and dt.tzinfo is not None:
            dt = dt.astimezone(local_tz)
        # Strip timezone for comparison to handle mixed aware/naive
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
