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
) -> List[float]:
    """
    Pre-compute temperature-aware charge rates for each slot.

    This function projects temperature changes across slots to predict
    how charge rate will vary due to thermal effects.

    Args:
        slots_sorted_by_time: List of price points sorted chronologically
        slot_fractions: Fraction of each slot that is available (0.0-1.0)
        slot_minutes: Duration of a full slot in minutes
        current_soc: Current battery state of charge (%)
        current_temp: Current battery temperature (Celsius), or None
        get_charge_rate_for_soc: Callback to get charge rate (kW) for given SOC and temperature
        predict_temp_after_duration: Callback to predict temperature after charging duration

    Returns:
        List of charge rates (kW) for each slot
    """
    charge_rates: List[float] = []

    if current_temp is not None:
        projected_temp = current_temp
        for i, _ in enumerate(slots_sorted_by_time):
            rate = get_charge_rate_for_soc(current_soc, projected_temp)
            charge_rates.append(rate)
            slot_duration_minutes = slot_minutes * slot_fractions[i]
            projected_temp = predict_temp_after_duration(projected_temp, slot_duration_minutes)
    else:
        # Fallback: use single rate for all slots (no temperature projection)
        rate = get_charge_rate_for_soc(current_soc, None)
        charge_rates = [rate for _ in slots_sorted_by_time]

    return charge_rates
