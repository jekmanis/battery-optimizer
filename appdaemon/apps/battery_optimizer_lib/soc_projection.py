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
a recalculation loop. The physics is now delegated to
``slot_energy.simulate_slot``, so this function and the energy view of a slot
cannot drift apart at all. The DP keeps its own inlined transition (it is fused
with the value recursion and the discrete SOC grid), and THAT parity is proven
by replay: ``tests/test_dp_energy_conservation.py::
TestPrefixConservationAcrossConditions::test_no_prefix_creates_energy`` sweeps
210 (starting SOC, load, PV, partial-slot) combinations, replays each selected
plan through ``simulate_slot`` and requires the DP's own SOC trajectory to match
to 1e-6 %. ``tests/test_slot_energy_parity.py`` sweeps this function against
``simulate_slot`` directly.

Invariants (see also docs/scheduling-algorithm.md):

1. A partial first slot is modelled by ``fraction``; nothing else changes.
2. DISCHARGE with ``pv >= load`` is a PV *charge*, not a discharge: the battery
   serves ``max(0, load - pv)`` and stores ``max(0, pv - load)``.
3. Battery-side (DC) energy is what moves the SOC. AC load served is converted
   with ``inverter_efficiency``; storage retention uses ``efficiency``.

Charge-rate units: ``params.charge_rate`` and anything returned by
``learning_engine.get_charge_rate_for_soc`` are ``charge_input_dc_kw`` --
terminal power BEFORE retention. Stored energy is ``rate * efficiency *
duration``. See ``learning_engine``'s module docstring for the full contract and
``slot_energy.simulate_slot`` for the same transition with every grid/PV flow
named.
"""

import datetime  # noqa: F401  (kept for type-hint friendliness of callers)
from dataclasses import dataclass
from typing import Optional

from .models import BatteryMode
from .slot_energy import params_from_soc_projection, simulate_slot


@dataclass(frozen=True)
class SocProjectionParams:
    """Battery/inverter parameters needed to project one slot's SOC change."""

    battery_capacity: float          # kWh
    efficiency: float                # storage retention factor (0-1)
    charge_rate: float               # charge_input_dc_kw (nominal fallback)
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
    """Result of projecting a single slot.

    ``dc_energy_in_kwh`` / ``dc_energy_out_kwh`` are what the battery ACTUALLY
    stored and delivered, after the SOC limits truncated the request. The
    request is kept separately in ``requested_dc_energy_*`` for consumers that
    need to show "wanted 1.0 kWh, got 0.04". Reporting the request as delivered
    energy is how credited energy gets created out of nothing -- ``cost_tracker``
    used to re-derive the cap itself to work around it.
    """

    soc_end: float
    temp_end: Optional[float] = None
    dc_energy_in_kwh: float = 0.0
    dc_energy_out_kwh: float = 0.0
    requested_dc_energy_in_kwh: float = 0.0
    requested_dc_energy_out_kwh: float = 0.0
    unmet_battery_ac_kwh: float = 0.0


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
    rate_lookup_temp: Optional[float] = None,
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
        rate_lookup_temp: Temperature to use for charge-rate lookups instead of
            ``temp_start``. Consumers that are re-projecting a plan pass the
            temperature the PLAN was built with for that slot
            (``DPOptimizerResult.planning_temp_by_slot``). Without it the
            re-projection charges at whatever rate its own evolving temperature
            implies, which is a different plan: on the brief's Task 4 case,
            where the refinement falls back to a conservative idle profile, the
            two trajectories diverged by 7.5 SOC points after three slots -- the
            schedule log printing one and the deviation detector running on the
            other. The thermal projection itself still uses ``temp_start``; only
            the rate lookup is pinned.
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
    # The temperature the RATE is looked up at may be pinned to the one the plan
    # was built with, independently of the temperature the pack is projected
    # through. See `rate_lookup_temp` in the docstring.
    rate_temp = rate_lookup_temp if rate_lookup_temp is not None else temp_start
    charge_rate = _effective_charge_rate(params, rate_soc, rate_temp, learning_engine)

    inv_eff = params.inverter_efficiency if params.inverter_efficiency > 0 else 1.0
    temp_end = temp_start

    # The charge capability this slot is planned with, in charge_input_dc_kw.
    # With temperature AND a learning engine the rate varies WITHIN the slot
    # (cold phase then warm phase), so the warming-aware energy is converted
    # back to the equivalent constant terminal power for the shared transition.
    slot_charge_input_dc_kw = charge_rate
    if (
        mode == BatteryMode.CHARGE
        and temp_start is not None
        and learning_engine is not None
    ):
        # charge_input_dc_kwh: DC energy at the battery terminal, before
        # retention. Multiplying by `efficiency` gives stored energy. (The
        # variable used to be called `energy_ac`, which it never was.)
        charge_input_dc_kwh, warmed_temp = (
            learning_engine.predict_charge_input_dc_energy(
                rate_soc,
                rate_temp,
                duration_minutes,
                temp_threshold=temp_threshold,
            )
        )
        # When the rate lookup is pinned to the plan's temperature the pack's
        # own temperature still evolves from where it really is.
        temp_end = (
            warmed_temp
            if rate_lookup_temp is None
            else _idle_temp(learning_engine, temp_start, duration_minutes)
        )
        if duration_hours > 0:
            slot_charge_input_dc_kw = charge_input_dc_kwh / duration_hours
    elif mode != BatteryMode.CHARGE:
        temp_end = _idle_temp(learning_engine, temp_start, duration_minutes)

    # THE physical transition. Delegated rather than re-derived, so the SOC view
    # and the energy view of a slot cannot drift apart (they did: this function
    # reported the uncapped request as delivered energy).
    energy_params = params_from_soc_projection(params)
    outcome = simulate_slot(
        stored_energy_kwh=soc_start / 100.0 * params.battery_capacity,
        mode=mode,
        params=energy_params,
        charge_input_dc_kw=slot_charge_input_dc_kw,
        load_kw=load_kw,
        pv_kw=pv_kw,
        fraction=fraction,
        is_export=bool(export_rate is not None and export_rate > 0),
    )
    dc_in = outcome.stored_dc_in_kwh
    dc_out = outcome.stored_dc_out_kwh
    soc_end = energy_params.soc_of(outcome.energy_end_kwh)

    if temp_projector is not None and temp_start is not None:
        # One thermal model for every mode, driven by the energy that ACTUALLY
        # moved. This used to call `thermal_model.battery_power_for_entry`,
        # which models the REQUESTED flow and knows nothing about SOC limits --
        # three lines after `simulate_slot` had already computed the real one.
        # A full pack ordered to charge therefore warmed as if it had taken a
        # full slot of energy, in every consumer of this projection, and which
        # convention the published temperature used depended on which code path
        # built it. Imaginary power must not manufacture future capability.
        temp_end = temp_projector.project(
            temp_start, slot_time, duration_minutes, outcome.battery_power_kw
        )

    return SocTransition(
        soc_end=soc_end,
        temp_end=temp_end,
        dc_energy_in_kwh=dc_in,
        dc_energy_out_kwh=dc_out,
        requested_dc_energy_in_kwh=(
            outcome.requested_charge_input_dc_kwh * params.efficiency
        ),
        requested_dc_energy_out_kwh=outcome.requested_stored_dc_out_kwh,
        unmet_battery_ac_kwh=outcome.unmet_battery_ac_kwh,
    )


def _idle_temp(learning_engine, temp_start: Optional[float], duration_minutes: float):
    """Temperature after a non-charging interval (cools toward ambient)."""
    if learning_engine is not None and temp_start is not None:
        return learning_engine.predict_temp_after_idle(temp_start, duration_minutes)
    return temp_start
