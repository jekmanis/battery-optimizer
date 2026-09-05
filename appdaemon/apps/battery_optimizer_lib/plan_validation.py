"""Continuous replay of a FINAL plan through the shared physical model.

A schedule is chosen on a discrete SOC grid, and then it is published: the
schedule log, the expected-SOC trajectory, the projected cost column and the
deviation detector all describe it. If the discrete path credited energy the
pack never had, none of those consumers notice -- they inherit the same
arithmetic. Clamping the reported SOC into ``[min_soc, max_soc]`` hides it
completely, which is how a plan whose battery was empty after 14 slots kept
reporting 15 % after 15.

``replay_plan`` walks the FINAL action sequence -- after any orchestrator
conversion -- through ``slot_energy.simulate_slot``, in continuous energy, and
reports:

* the conservation check, evaluated on the REQUESTED flow, before clamping: a
  slot the continuous model cannot serve in full is either one the plan
  DECLARED energy-limited (``ScheduleEntry.energy_limited``, priced with the
  grid covering the remainder), or the plan credited the battery with service it
  does not have. Accumulating ``simulate_slot``'s own clamped outputs and
  comparing them with the bound those outputs are built to satisfy is an
  identity -- the brief's pre-fix defect produced zero violations that way --
  so it is reported (``credited_over_available_kwh``) and is not the test;

* AC energy actually served after inverter loss, not just raw DC totals;
* the AC demand the plan assigned to the battery that the battery could not
  supply (``unmet_battery_ac_kwh``) -- the grid pays for it;
* disagreement with the planner's own SOC trajectory, when one is supplied.

It is pure and deterministic: forecasts, the charge-rate lookup and the thermal
projector are all injected.
"""

import datetime
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .models import BatteryMode, ScheduleEntry
from .slot_energy import SlotEnergyParams, simulate_slot
from .timezone_utils import instant_key

# Energy tolerance (kWh) for the conservation inequality. It exists for
# floating-point accumulation only -- it is NOT a budget for modelling error.
CONSERVATION_EPS_KWH = 1e-7

# Default tolerance (SOC %) for planner-vs-replay disagreement. One DP grid
# step is the natural unit; callers pass their own.
DEFAULT_SOC_TOLERANCE = 1e-6


@dataclass
class SlotReplay:
    """What one slot of the final plan actually does."""

    time: datetime.datetime
    mode: BatteryMode
    is_export: bool
    fraction: float
    soc_start: float
    soc_end: float
    energy_start_kwh: float
    energy_end_kwh: float
    stored_dc_in_kwh: float
    stored_dc_out_kwh: float
    charge_input_dc_kwh: float
    grid_charge_ac_kwh: float
    battery_ac_served_kwh: float
    unmet_battery_ac_kwh: float
    grid_import_ac_kwh: float
    grid_export_ac_kwh: float
    value_eur: float
    temp_start: Optional[float] = None
    temp_end: Optional[float] = None
    energy_limited: bool = False


@dataclass
class PlanReplay:
    """Aggregate result of replaying a whole plan."""

    order: List[datetime.datetime] = field(default_factory=list)
    by_slot: Dict[datetime.datetime, SlotReplay] = field(default_factory=dict)
    conservation_violations: List[str] = field(default_factory=list)
    trajectory_disagreements: List[str] = field(default_factory=list)

    initial_usable_kwh: float = 0.0
    final_energy_kwh: float = 0.0
    total_stored_dc_in_kwh: float = 0.0
    total_stored_dc_out_kwh: float = 0.0
    total_battery_ac_served_kwh: float = 0.0
    total_unmet_battery_ac_kwh: float = 0.0
    total_grid_import_ac_kwh: float = 0.0
    total_grid_export_ac_kwh: float = 0.0
    total_grid_charge_ac_kwh: float = 0.0
    total_requested_stored_dc_out_kwh: float = 0.0
    credited_over_available_kwh: float = 0.0
    total_value_eur: float = 0.0
    terminal_value_eur: float = 0.0

    @property
    def max_battery_ac_available_kwh(self) -> float:
        """Upper bound on AC energy the battery could ever have served.

        Everything it started with plus everything it actually stored, after
        the DC->AC conversion. Checking AC energy matters because the DC totals
        can balance while the served load does not.
        """
        return (
            self.initial_usable_kwh + self.total_stored_dc_in_kwh
        ) * self._inv_eff

    _inv_eff: float = 1.0

    @property
    def ok(self) -> bool:
        return not self.conservation_violations and not self.trajectory_disagreements


def _params_from_config(config, min_soc=None, max_soc=None) -> SlotEnergyParams:
    return SlotEnergyParams(
        battery_capacity=config.battery_capacity,
        efficiency=config.efficiency,
        discharge_rate=config.discharge_rate,
        export_discharge_rate=getattr(config, "export_discharge_rate", 0.0),
        inverter_efficiency=getattr(config, "inverter_efficiency", 1.0),
        min_soc=config.min_soc if min_soc is None else min_soc,
        max_soc=config.max_soc if max_soc is None else max_soc,
        slot_minutes=config.slot_minutes,
    )


def replay_plan(
    *,
    schedule: Dict[datetime.datetime, ScheduleEntry],
    config,
    starting_soc: float,
    predict_load_kw: Callable[[datetime.datetime], float],
    predict_pv_kw: Optional[Callable[[datetime.datetime], float]] = None,
    charge_rate_for: Optional[
        Callable[[datetime.datetime, float, Optional[float]], float]
    ] = None,
    current_slot: Optional[datetime.datetime] = None,
    minutes_into_slot: float = 0.0,
    starting_temp: Optional[float] = None,
    temp_projector=None,
    prices_by_slot: Optional[Dict[datetime.datetime, float]] = None,
    terminal_rate: Optional[float] = None,
    min_soc: Optional[float] = None,
    max_soc: Optional[float] = None,
    planned_soc_by_slot: Optional[Dict[datetime.datetime, float]] = None,
    soc_tolerance: float = DEFAULT_SOC_TOLERANCE,
    slot_matcher: Optional[Callable[[datetime.datetime, datetime.datetime], bool]] = None,
) -> PlanReplay:
    """Replay ``schedule`` continuously and report its physical feasibility.

    Args:
        schedule: The FINAL action sequence, after any postprocessing.
        config: Anything with the ``DPOptimizerConfig`` attribute surface
            (capacity, efficiencies, rates, fees, slot_minutes, min/max SOC).
        starting_soc: SOC (%) at the replay instant.
        predict_load_kw / predict_pv_kw: Forecasts. Fixed for the whole pass --
            a caller must not let them change mid-replay, or a disagreement
            becomes indistinguishable from a moving input.
        charge_rate_for: ``(slot_time, soc, temp) -> charge_input_dc_kw``. This
            is the Task 4 contract: the rate depends on the SOC and temperature
            the plan actually reaches, not on a time-indexed array.
        current_slot / minutes_into_slot: Partial first slot, same formula as
            the planner.
        prices_by_slot: When given, each slot's cash flow is scored with the
            same tariff arithmetic the DP optimizes, so the replay's total value
            is comparable to the DP's objective.
        planned_soc_by_slot: The planner's own end-of-slot SOC per slot. Any
            disagreement beyond ``soc_tolerance`` is reported.
    """
    params = _params_from_config(config, min_soc=min_soc, max_soc=max_soc)
    inv_eff = params.inv_eff
    replay = PlanReplay()
    replay._inv_eff = inv_eff

    if not schedule:
        return replay

    order = sorted(schedule.keys(), key=instant_key)
    replay.order = order

    energy = min(
        params.max_energy_kwh,
        max(params.min_energy_kwh, params.energy_of(starting_soc)),
    )
    replay.initial_usable_kwh = max(0.0, energy - params.min_energy_kwh)

    first_fraction = min(
        1.0,
        max(
            0.0,
            (config.slot_minutes - minutes_into_slot) / max(1, config.slot_minutes),
        ),
    )
    partial_applied = current_slot is None

    def _matches(slot, target) -> bool:
        if slot_matcher is not None:
            return slot_matcher(slot, target)
        return instant_key(slot) == instant_key(target)

    grid_fee = getattr(config, "grid_fee", 0.0)
    import_mult = getattr(config, "import_price_multiplier", 1.0)
    export_mult = getattr(config, "export_rate_multiplier", 1.0)
    export_fee = getattr(config, "grid_export_fee", 0.0)
    wear = getattr(config, "battery_wear_cost", 0.0)

    cum_in = 0.0
    cum_out = 0.0
    cum_requested_out = 0.0
    cum_unmet = 0.0
    temp = starting_temp

    for slot in order:
        entry = schedule[slot]
        fraction = 1.0
        if not partial_applied and current_slot is not None and _matches(slot, current_slot):
            fraction = first_fraction
            partial_applied = True

        soc_start = params.soc_of(energy)
        rate = (
            charge_rate_for(slot, soc_start, temp)
            if charge_rate_for is not None
            else getattr(config, "charge_rate", 0.0)
        )
        is_export = bool(entry.export_rate is not None and entry.export_rate > 0)

        outcome = simulate_slot(
            stored_energy_kwh=energy,
            mode=entry.mode,
            params=params,
            charge_input_dc_kw=max(0.0, rate or 0.0),
            load_kw=max(0.0, predict_load_kw(slot)),
            pv_kw=max(0.0, predict_pv_kw(slot)) if predict_pv_kw is not None else 0.0,
            fraction=fraction,
            is_export=is_export,
        )

        temp_start = temp
        if temp_projector is not None and temp is not None:
            temp = temp_projector.project(
                temp,
                slot,
                config.slot_minutes * fraction,
                outcome.battery_power_kw,
            )

        value = 0.0
        if prices_by_slot is not None:
            price = prices_by_slot.get(slot)
            if price is None:
                price = prices_by_slot.get(instant_key(slot))
            if price is not None:
                buy = (price + grid_fee) * import_mult
                sell = max(0.0, price * export_mult - export_fee)
                value = (
                    -buy * outcome.grid_import_ac_kwh
                    + sell * outcome.grid_export_ac_kwh
                    - wear * outcome.stored_dc_out_kwh
                )

        cum_in += outcome.stored_dc_in_kwh
        cum_out += outcome.stored_dc_out_kwh
        cum_requested_out += outcome.requested_stored_dc_out_kwh
        cum_unmet += outcome.unmet_battery_ac_kwh

        # THE conservation check, and it has to be able to fail.
        #
        # Accumulating `simulate_slot`'s own clamped outputs and comparing them
        # with the bound those outputs are constructed to satisfy is an
        # identity: the brief's exact pre-fix defect (a planner deducting 1 %
        # per slot for a 1.4 %-per-slot load) produced zero violations that way.
        #
        # What can fail is the comparison against the PLAN's own statement about
        # its physics. A slot the continuous model cannot serve in full is
        # either one the plan DECLARED energy-limited -- priced with the grid
        # covering the remainder -- or the plan credited the battery with
        # service it does not have. This is evaluated on the requested flow,
        # BEFORE the SOC clamp, so clamping an impossible trajectory into range
        # cannot make it pass.
        if outcome.unmet_battery_ac_kwh > CONSERVATION_EPS_KWH and not getattr(
            entry, "energy_limited", False
        ):
            replay.conservation_violations.append(
                f"{slot}: the plan credits {outcome.unmet_battery_ac_kwh:.6f} kWh "
                f"more battery service than the pack holds and does not declare "
                f"the slot energy-limited (cumulative requested stored-DC out "
                f"{cum_requested_out:.6f} kWh vs initial usable "
                f"{replay.initial_usable_kwh:.6f} + charged {cum_in:.6f} kWh)"
            )

        energy = outcome.energy_end_kwh
        soc_end = params.soc_of(energy)

        replay.by_slot[slot] = SlotReplay(
            time=slot,
            mode=entry.mode,
            is_export=is_export,
            fraction=fraction,
            soc_start=soc_start,
            soc_end=soc_end,
            energy_start_kwh=params.energy_of(soc_start),
            energy_end_kwh=energy,
            stored_dc_in_kwh=outcome.stored_dc_in_kwh,
            stored_dc_out_kwh=outcome.stored_dc_out_kwh,
            charge_input_dc_kwh=outcome.charge_input_dc_kwh,
            grid_charge_ac_kwh=outcome.grid_charge_ac_kwh,
            battery_ac_served_kwh=outcome.battery_ac_served_kwh,
            unmet_battery_ac_kwh=outcome.unmet_battery_ac_kwh,
            grid_import_ac_kwh=outcome.grid_import_ac_kwh,
            grid_export_ac_kwh=outcome.grid_export_ac_kwh,
            value_eur=value,
            temp_start=temp_start,
            temp_end=temp,
            energy_limited=outcome.energy_limited,
        )

        replay.total_stored_dc_in_kwh += outcome.stored_dc_in_kwh
        replay.total_stored_dc_out_kwh += outcome.stored_dc_out_kwh
        replay.total_battery_ac_served_kwh += outcome.battery_ac_served_kwh
        replay.total_unmet_battery_ac_kwh += outcome.unmet_battery_ac_kwh
        replay.total_grid_import_ac_kwh += outcome.grid_import_ac_kwh
        replay.total_grid_export_ac_kwh += outcome.grid_export_ac_kwh
        replay.total_grid_charge_ac_kwh += outcome.grid_charge_ac_kwh
        replay.total_value_eur += value

        if planned_soc_by_slot is not None:
            planned = planned_soc_by_slot.get(slot)
            if planned is None:
                planned = planned_soc_by_slot.get(instant_key(slot))
            if planned is not None and abs(planned - soc_end) > soc_tolerance:
                replay.trajectory_disagreements.append(
                    f"{slot}: planned end SOC {planned:.3f}% vs replay "
                    f"{soc_end:.3f}%"
                )

    replay.final_energy_kwh = energy
    replay.total_requested_stored_dc_out_kwh = cum_requested_out
    # Reported for the log, not used as the pass/fail test above: with the
    # clamped flows this is an identity, which is precisely why it is not the
    # check. It is still the number a reader wants to see next to a violation.
    replay.credited_over_available_kwh = max(
        0.0,
        cum_requested_out - (replay.initial_usable_kwh + cum_in),
    )
    if terminal_rate:
        replay.terminal_value_eur = terminal_rate * max(
            0.0, energy - params.min_energy_kwh
        )
        replay.total_value_eur += replay.terminal_value_eur

    return replay


__all__ = [
    "CONSERVATION_EPS_KWH",
    "DEFAULT_SOC_TOLERANCE",
    "PlanReplay",
    "SlotReplay",
    "replay_plan",
]
