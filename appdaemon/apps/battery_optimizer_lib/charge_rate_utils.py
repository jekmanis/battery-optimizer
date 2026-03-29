"""
Charge rate utility functions.

Provides shared utilities for computing temperature-aware charge rates per slot.
"""

from typing import Callable, List, Optional

from .models import PricePoint


def compute_charge_rates_per_slot(
    slots_sorted_by_time: List[PricePoint],
    slot_fractions: List[float],
    slot_minutes: int,
    current_soc: float,
    current_temp: Optional[float],
    get_charge_rate_for_soc: Callable[[float, Optional[float]], float],
    predict_temp_after_duration: Callable[[float, float], float],
    battery_capacity: float = 0.0,
    efficiency: float = 0.85,
    max_soc: float = 100.0,
) -> List[float]:
    """
    Pre-compute temperature and SOC-aware charge rates for each slot.

    Projects both temperature and SOC forward across slots so the DP sees
    realistic charge rates at each future SOC level (e.g. lower rates near
    full capacity, as enforced by the BMS).

    SOC is projected assuming continuous charging (worst-case for rate
    decline).  The DP will only use each rate when it actually picks CHARGE,
    so over-projection is conservative — it never makes the DP *over*-
    estimate the rate at a given SOC.

    Args:
        slots_sorted_by_time: List of price points sorted chronologically
        slot_fractions: Fraction of each slot that is available (0.0-1.0)
        slot_minutes: Duration of a full slot in minutes
        current_soc: Current battery state of charge (%)
        current_temp: Current battery temperature (Celsius), or None
        get_charge_rate_for_soc: Callback to get charge rate (kW) for given SOC and temperature
        predict_temp_after_duration: Callback to predict temperature after charging duration
        battery_capacity: Battery capacity in kWh (needed for SOC projection)
        efficiency: Charging efficiency 0-1 (needed for SOC projection)
        max_soc: Maximum SOC % (cap for projection)

    Returns:
        List of charge rates (kW) for each slot
    """
    charge_rates: List[float] = []
    slot_hours = slot_minutes / 60.0
    project_soc = battery_capacity > 0

    if current_temp is not None:
        projected_temp = current_temp
        projected_soc = current_soc
        for i, _ in enumerate(slots_sorted_by_time):
            rate = get_charge_rate_for_soc(projected_soc, projected_temp)
            charge_rates.append(rate)
            slot_duration_minutes = slot_minutes * slot_fractions[i]
            projected_temp = predict_temp_after_duration(projected_temp, slot_duration_minutes)
            if project_soc:
                energy_kwh = rate * efficiency * slot_hours * slot_fractions[i]
                projected_soc = min(max_soc, projected_soc + (energy_kwh / battery_capacity) * 100)
    else:
        # Fallback: project SOC only (no temperature data)
        projected_soc = current_soc
        for i, _ in enumerate(slots_sorted_by_time):
            rate = get_charge_rate_for_soc(projected_soc, None)
            charge_rates.append(rate)
            if project_soc:
                energy_kwh = rate * efficiency * slot_hours * slot_fractions[i]
                projected_soc = min(max_soc, projected_soc + (energy_kwh / battery_capacity) * 100)

    return charge_rates
