"""
Shared slot-level SOC transition model.

This module owns the ONE physics model for "given a SOC at the start of a slot,
a scheduled mode, and the load/PV forecast for that slot, what is the SOC at the
end of the slot?".

Historically three places modelled this independently:

* ``DPOptimizer._run_dp`` (the authority — it is what the schedule is chosen by)
* ``BatteryOptimizer.calculate_expected_soc_schedule`` (the expected trajectory)
* ``SocDeviationDetector._interpolate_expected_soc`` (the deviation detector)

They disagreed, which produced false "SOC behind plan" / "SOC ahead" events and
a recalculation loop. The semantics below are copied 1:1 from the DP so that the
two consumers cannot drift from it again. The DP keeps its own inlined
transition (it is fused with the value recursion and the discrete SOC grid);
``tests/test_soc_projection.py`` guards that the two agree within one DP grid
step per slot.

Invariants (see also docs/scheduling-algorithm.md):

1. A partial first slot is modelled by ``fraction``; nothing else changes.
2. DISCHARGE with ``pv >= load`` is a PV *charge*, not a discharge: the battery
   serves ``max(0, load - pv)`` and stores ``max(0, pv - load)``.
3. Battery-side (DC) energy is what moves the SOC. AC load served is converted
   with ``inverter_efficiency``; storage retention uses ``efficiency``.
"""

import datetime  # noqa: F401  (kept for type-hint friendliness of callers)
from dataclasses import dataclass
from typing import Optional

from .models import BatteryMode
from .thermal_model import battery_power_for_entry


@dataclass(frozen=True)
class SocProjectionParams:
    """Battery/inverter parameters needed to project one slot's SOC change."""

    battery_capacity: float          # kWh
    efficiency: float                # storage retention factor (0-1)
    charge_rate: float               # kW (nominal, used when no learned rate)
    discharge_rate: float            # kW (AC side, self-consumption cap)
    export_discharge_rate: float = 0.0   # kW (0 = fall back to discharge_rate)
    inverter_efficiency: float = 1.0     # AC<->DC conversion efficiency
    min_soc: float = 10.0            # %
    max_soc: float = 100.0           # %
    slot_minutes: int = 15

    @property
    def slot_hours(self) -> float:
        return self.slot_minutes / 60.0

    @property
    def effective_export_discharge_rate(self) -> float:
        """Discharge rate during grid export (kW); falls back to discharge_rate."""
        return (
            self.export_discharge_rate
            if self.export_discharge_rate > 0
            else self.discharge_rate
        )


@dataclass
class SocTransition:
    """Result of projecting a single slot."""

    soc_end: float
    temp_end: Optional[float] = None
    dc_energy_in_kwh: float = 0.0
    dc_energy_out_kwh: float = 0.0


def _effective_charge_rate(
    params: SocProjectionParams,
    rate_soc: float,
    temp_start: Optional[float],
    learning_engine,
) -> float:
    """Learned charge rate when available, else the configured nominal rate."""
    if learning_engine is not None:
        learned = learning_engine.get_charge_rate_for_soc(rate_soc, temp_start)
        if learned is not None and learned > 0:
            return learned
    return params.charge_rate


def project_slot_soc(
    *,
    soc_start: float,
    mode: BatteryMode,
    params: SocProjectionParams,
    load_kw: float = 0.0,
    pv_kw: float = 0.0,
    fraction: float = 1.0,
    export_rate: Optional[float] = None,
    temp_start: Optional[float] = None,
    learning_engine=None,
    temp_threshold: float = 16.0,
    rate_lookup_soc: Optional[float] = None,
    temp_projector=None,
    slot_time: Optional[datetime.datetime] = None,
) -> SocTransition:
    """Project the SOC (and temperature) after one — possibly partial — slot.

    Args:
        soc_start: SOC (%) at the start of the projected interval.
        mode: Scheduled BatteryMode for the slot.
        params: Battery/inverter parameters.
        load_kw: Predicted household load for the slot (kW, AC).
        pv_kw: Predicted PV production for the slot (kW, AC-equivalent).
        fraction: Fraction of the slot to project (0..1). Values outside the
            range are clamped; ``fraction <= 0`` is a no-op.
        export_rate: Direct-control export rate of the slot. A positive value
            marks a grid-export discharge slot (full export discharge rate).
        temp_start: Battery temperature (C) at the start, or None.
        learning_engine: Optional BatteryLearningEngine for learned charge
            rates, warming-aware charge energy and temperature evolution.
        temp_threshold: Temperature above which charging speeds up (learning
            engine warming model).
        rate_lookup_soc: SOC to use for charge-rate lookups instead of
            ``soc_start``. The deviation detector passes the *actual* SOC here
            because that is what the inverter's rate really depends on.
        temp_projector: Optional ``thermal_model.TemperatureProjector``. When
            given, the end temperature comes from the SHARED thermal model for
            every mode (relaxation toward a time-varying ambient plus
            ``k2*|P_bat|``) instead of the old mode-specific split where only
            CHARGE could warm the pack. The SOC/energy result is unaffected.
        slot_time: Slot timestamp, used by ``temp_projector`` to look up the
            ambient temperature at that point of the horizon.

    Returns:
        SocTransition with the end SOC, end temperature and the DC energy moved.
    """
    fraction = min(1.0, max(0.0, fraction))
    if fraction <= 0.0:
        return SocTransition(soc_end=soc_start, temp_end=temp_start)

    duration_hours = params.slot_hours * fraction
    duration_minutes = params.slot_minutes * fraction

    net_load_kw = max(0.0, load_kw - pv_kw)
    pv_surplus_kw = max(0.0, pv_kw - load_kw)

    rate_soc = rate_lookup_soc if rate_lookup_soc is not None else soc_start
    charge_rate = _effective_charge_rate(params, rate_soc, temp_start, learning_engine)

    capacity = params.battery_capacity if params.battery_capacity > 0 else 1e-9
    inv_eff = params.inverter_efficiency if params.inverter_efficiency > 0 else 1.0

    dc_in = 0.0
    dc_out = 0.0
    temp_end = temp_start

    if mode == BatteryMode.CHARGE:
        if temp_start is not None and learning_engine is not None:
            energy_ac, temp_end = learning_engine.predict_charge_energy_with_warming(
                rate_soc,
                temp_start,
                duration_minutes,
                temp_threshold=temp_threshold,
            )
            dc_in = energy_ac * params.efficiency
        else:
            # No temperature and/or no learning engine: flat rate, temp unknown.
            dc_in = charge_rate * params.efficiency * duration_hours
        soc_end = min(params.max_soc, soc_start + (dc_in / capacity) * 100)

    elif mode == BatteryMode.DISCHARGE and export_rate is not None and export_rate > 0:
        # Grid export: battery discharges at the export rate regardless of load.
        dc_out = params.effective_export_discharge_rate * duration_hours / inv_eff
        soc_end = max(params.min_soc, soc_start - (dc_out / capacity) * 100)
        temp_end = _idle_temp(learning_engine, temp_start, duration_minutes)

    elif mode == BatteryMode.DISCHARGE:
        # Self-consumption: battery serves the net load; PV surplus charges it.
        ac_served = min(net_load_kw, params.discharge_rate) * duration_hours
        dc_out = ac_served / inv_eff
        dc_in = min(pv_surplus_kw, charge_rate) * params.efficiency * duration_hours
        soc = min(params.max_soc, soc_start + (dc_in / capacity) * 100)
        soc_end = max(params.min_soc, soc - (dc_out / capacity) * 100)
        temp_end = _idle_temp(learning_engine, temp_start, duration_minutes)

    else:  # HOLD
        dc_in = min(pv_surplus_kw, charge_rate) * params.efficiency * duration_hours
        soc_end = min(params.max_soc, soc_start + (dc_in / capacity) * 100)
        temp_end = _idle_temp(learning_engine, temp_start, duration_minutes)

    if temp_projector is not None and temp_start is not None:
        # One thermal model for every mode. |P_bat| is derived by the same
        # helper the DP trajectory uses, so the two can never disagree.
        power_kw = battery_power_for_entry(
            mode,
            charge_rate_kw=charge_rate,
            load_kw=load_kw,
            pv_kw=pv_kw,
            discharge_rate_kw=params.discharge_rate,
            export_discharge_rate_kw=params.effective_export_discharge_rate,
            export_rate=export_rate,
            inverter_efficiency=inv_eff,
        )
        temp_end = temp_projector.project(
            temp_start, slot_time, duration_minutes, power_kw
        )

    return SocTransition(
        soc_end=soc_end,
        temp_end=temp_end,
        dc_energy_in_kwh=dc_in,
        dc_energy_out_kwh=dc_out,
    )


def _idle_temp(learning_engine, temp_start: Optional[float], duration_minutes: float):
    """Temperature after a non-charging interval (cools toward ambient)."""
    if learning_engine is not None and temp_start is not None:
        return learning_engine.predict_temp_after_idle(temp_start, duration_minutes)
    return temp_start
