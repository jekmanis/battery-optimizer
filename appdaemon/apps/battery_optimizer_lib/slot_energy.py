"""The one PHYSICAL slot transition: energy flows for a single (partial) slot.

``soc_projection.project_slot_soc`` answers "what SOC does this slot end at?".
This module answers the larger question the planner and the plan validator both
need: *where did every kWh come from and go to*, in units that are named after
their measurement boundary.

Unit contract (see ``docs/scheduling-algorithm.md`` SS Charge-rate units)::

    charge_input_dc_kw   DC power at the battery terminal, BEFORE retention
    stored_charge_kw     rate at which STORED energy grows
                         = charge_input_dc_kw * efficiency
    grid_charge_ac_kwh   AC energy purchased to charge
                         = grid_charge_dc_kwh / inverter_efficiency
    stored_dc_out_kwh    DC energy drawn from the pack
    battery_ac_served_kwh= stored_dc_out_kwh * inverter_efficiency

So storing 1 kWh from the grid at ``efficiency=0.85`` and
``inverter_efficiency=0.97`` imports ``1 / (0.85 * 0.97) = 1.21286`` kWh AC.
DC-coupled PV surplus charges the pack without the grid inverter conversion, so
it is billed at neither the import price nor the inverter loss.

The function is PURE: no clock, no forecasts, no learning engine. Callers pass
the rate capability they decided on. It is used by

* ``DPOptimizer`` -- indirectly. The DP keeps an inlined transition fused with
  the value recursion and the discrete SOC grid, so parity is PROVEN rather than
  structural: ``tests/test_dp_energy_conservation.py::
  TestPrefixConservationAcrossConditions::test_no_prefix_creates_energy`` sweeps
  210 (starting SOC, load, PV, partial-slot) combinations, replays each selected
  plan through this module and requires the DP's own SOC trajectory to match to
  1e-6 %. ``tests/test_slot_energy_parity.py`` does the same for
  ``soc_projection``;
* the final-plan replay in ``plan_validation.py``, which is what actually
  decides whether a published plan is physically feasible;
* ``cost_tracker`` / ``soc_projection`` consumers via the SOC view.

Nothing here quantizes: the DP's discrete SOC grid is applied by the DP, on top
of these continuous flows (see ``docs/scheduling-algorithm.md`` SS Conservative
quantization).
"""

from dataclasses import dataclass

from .models import BatteryMode

# Tolerance used when comparing energies at a grid/limit boundary (kWh).
ENERGY_EPS = 1e-9

# How far back from the SOC a slot REACHES the end-of-span rate is probed, as a
# fraction of the span. See ``charge_rate_for_span``.
SPAN_ENDPOINT_EPS = 1e-9


@dataclass(frozen=True)
class SlotEnergyParams:
    """Battery/inverter parameters for one slot transition."""

    battery_capacity: float           # kWh
    efficiency: float                 # storage retention factor (0-1)
    discharge_rate: float             # kW, AC side, self-consumption cap
    export_discharge_rate: float = 0.0  # kW, AC side (0 = use discharge_rate)
    inverter_efficiency: float = 1.0  # AC<->DC conversion
    min_soc: float = 10.0             # %
    max_soc: float = 100.0            # %
    slot_minutes: int = 15

    @property
    def slot_hours(self) -> float:
        return self.slot_minutes / 60.0

    @property
    def effective_export_discharge_rate(self) -> float:
        return (
            self.export_discharge_rate
            if self.export_discharge_rate > 0
            else self.discharge_rate
        )

    @property
    def min_energy_kwh(self) -> float:
        return self.min_soc / 100.0 * self.battery_capacity

    @property
    def max_energy_kwh(self) -> float:
        return self.max_soc / 100.0 * self.battery_capacity

    @property
    def inv_eff(self) -> float:
        return self.inverter_efficiency if self.inverter_efficiency > 0 else 1.0

    def soc_of(self, energy_kwh: float) -> float:
        capacity = self.battery_capacity if self.battery_capacity > 0 else 1e-9
        return energy_kwh / capacity * 100.0

    def energy_of(self, soc: float) -> float:
        return soc / 100.0 * self.battery_capacity


@dataclass(frozen=True)
class SlotEnergyResult:
    """Every flow of one slot, in explicitly named units.

    ``requested_*`` is what the action asked for; ``*_kwh`` without the prefix
    is what the battery could actually deliver or absorb. A consumer that needs
    the request (for reporting) and the delivery (for accounting) gets both --
    treating an uncapped request as delivered energy is exactly how credited
    energy gets created out of nothing.
    """

    energy_end_kwh: float

    # Charging side
    charge_input_dc_kwh: float = 0.0
    requested_charge_input_dc_kwh: float = 0.0
    stored_dc_in_kwh: float = 0.0
    pv_charge_dc_kwh: float = 0.0
    grid_charge_dc_kwh: float = 0.0
    grid_charge_ac_kwh: float = 0.0

    # Discharging side
    stored_dc_out_kwh: float = 0.0
    requested_stored_dc_out_kwh: float = 0.0
    battery_ac_served_kwh: float = 0.0
    unmet_battery_ac_kwh: float = 0.0

    # Grid side
    grid_import_ac_kwh: float = 0.0
    grid_export_ac_kwh: float = 0.0

    # Thermal input
    battery_power_kw: float = 0.0

    @property
    def energy_limited(self) -> bool:
        """Whether a physical limit truncated what the action asked for."""
        return (
            self.unmet_battery_ac_kwh > ENERGY_EPS
            or self.requested_charge_input_dc_kwh - self.charge_input_dc_kwh
            > ENERGY_EPS
        )


def simulate_slot(
    *,
    stored_energy_kwh: float,
    mode: BatteryMode,
    params: SlotEnergyParams,
    charge_input_dc_kw: float = 0.0,
    load_kw: float = 0.0,
    pv_kw: float = 0.0,
    fraction: float = 1.0,
    is_export: bool = False,
) -> SlotEnergyResult:
    """Simulate one (possibly partial) slot from ``stored_energy_kwh``.

    Args:
        stored_energy_kwh: Absolute stored DC energy at the start (kWh), i.e.
            ``soc/100 * capacity`` -- not "usable above min_soc".
        mode: Scheduled action.
        params: Battery/inverter parameters.
        charge_input_dc_kw: Charge capability at the battery terminal for this
            slot (see the unit contract above). For HOLD and self-consumption
            DISCHARGE it caps how much PV surplus the pack can absorb. Callers
            must derive it with ``charge_rate_for_span`` rather than by a bare
            ``rate(soc_start, temp)`` lookup -- that is the ONE within-slot
            charge model, and freezing the rate at the start SOC over-credits
            every slot that crosses a learned SOC-taper boundary.
        load_kw: Household load (AC).
        pv_kw: PV production (AC-equivalent; DC-coupled surplus charges the pack).
        fraction: Fraction of the slot to simulate (0..1).
        is_export: True for a grid-export DISCHARGE slot.

    Returns:
        SlotEnergyResult with actual and requested flows.
    """
    fraction = min(1.0, max(0.0, fraction))
    if fraction <= 0.0:
        return SlotEnergyResult(energy_end_kwh=stored_energy_kwh)

    duration_h = params.slot_hours * fraction
    eff = params.efficiency if params.efficiency > 0 else 1e-9
    inv_eff = params.inv_eff

    min_energy = params.min_energy_kwh
    max_energy = params.max_energy_kwh
    energy = min(max_energy, max(min_energy, stored_energy_kwh))

    net_load_kw = max(0.0, load_kw - pv_kw)
    pv_surplus_kw = max(0.0, pv_kw - load_kw)
    net_load_kwh = net_load_kw * duration_h
    pv_surplus_kwh = pv_surplus_kw * duration_h
    load_kwh = max(0.0, load_kw) * duration_h
    pv_kwh = max(0.0, pv_kw) * duration_h

    charge_input_dc_kw = max(0.0, charge_input_dc_kw)
    headroom = max(0.0, max_energy - energy)
    available = max(0.0, energy - min_energy)

    charge_input_dc = 0.0
    requested_charge_input_dc = 0.0
    stored_in = 0.0
    pv_charge_dc = 0.0
    grid_charge_dc = 0.0
    grid_charge_ac = 0.0
    dc_out = 0.0
    requested_dc_out = 0.0
    ac_served = 0.0
    unmet_ac = 0.0
    grid_import = 0.0
    exported = 0.0

    def _absorb(cap_input_dc_kw: float, pv_only: bool):
        """Charge the pack at ``cap_input_dc_kw``, limited by headroom."""
        requested_input = cap_input_dc_kw * duration_h
        requested_stored = requested_input * eff
        actual_stored = min(requested_stored, headroom)
        actual_input = actual_stored / eff
        if pv_only:
            pv_dc = actual_input
            grid_dc = 0.0
        else:
            pv_dc = min(pv_surplus_kwh, actual_input)
            grid_dc = max(0.0, actual_input - pv_dc)
        return requested_input, actual_input, actual_stored, pv_dc, grid_dc

    if mode == BatteryMode.CHARGE:
        (
            requested_charge_input_dc,
            charge_input_dc,
            stored_in,
            pv_charge_dc,
            grid_charge_dc,
        ) = _absorb(charge_input_dc_kw, pv_only=False)
        grid_charge_ac = grid_charge_dc / inv_eff
        # The grid also covers the household's net load during a charge slot.
        grid_import = net_load_kwh + grid_charge_ac
        # PV surplus the pack could not take is exported.
        exported = max(0.0, pv_surplus_kwh - pv_charge_dc)

    elif mode == BatteryMode.DISCHARGE and is_export:
        requested_ac_out = params.effective_export_discharge_rate * duration_h
        requested_dc_out = requested_ac_out / inv_eff
        dc_out = min(requested_dc_out, available)
        ac_served = dc_out * inv_eff
        unmet_ac = max(0.0, requested_ac_out - ac_served)
        grid_import = max(0.0, load_kwh - pv_kwh - ac_served)
        exported = max(0.0, ac_served + pv_kwh - load_kwh)

    elif mode == BatteryMode.DISCHARGE:
        # Self-consumption. With pv >= load this is a PV charge, exactly as in
        # soc_projection (CLAUDE.md "One slot-SOC model").
        requested_ac = min(net_load_kw, params.discharge_rate) * duration_h
        requested_dc_out = requested_ac / inv_eff
        dc_out = min(requested_dc_out, available)
        ac_served = dc_out * inv_eff
        unmet_ac = max(0.0, requested_ac - ac_served)
        grid_import = max(0.0, net_load_kwh - ac_served)
        if pv_surplus_kwh > 0:
            (
                requested_charge_input_dc,
                charge_input_dc,
                stored_in,
                pv_charge_dc,
                _grid,
            ) = _absorb(min(pv_surplus_kw, charge_input_dc_kw), pv_only=True)
            exported = max(0.0, pv_surplus_kwh - pv_charge_dc)

    else:  # HOLD
        if pv_surplus_kwh > 0:
            (
                requested_charge_input_dc,
                charge_input_dc,
                stored_in,
                pv_charge_dc,
                _grid,
            ) = _absorb(min(pv_surplus_kw, charge_input_dc_kw), pv_only=True)
            exported = max(0.0, pv_surplus_kwh - pv_charge_dc)
        grid_import = net_load_kwh

    energy_end = energy + stored_in - dc_out
    # Guard against floating-point excursions past the physical bounds.
    energy_end = min(max_energy, max(min_energy, energy_end))

    net_dc_kw = (charge_input_dc - dc_out) / duration_h if duration_h > 0 else 0.0

    return SlotEnergyResult(
        energy_end_kwh=energy_end,
        charge_input_dc_kwh=charge_input_dc,
        requested_charge_input_dc_kwh=requested_charge_input_dc,
        stored_dc_in_kwh=stored_in,
        pv_charge_dc_kwh=pv_charge_dc,
        grid_charge_dc_kwh=grid_charge_dc,
        grid_charge_ac_kwh=grid_charge_ac,
        stored_dc_out_kwh=dc_out,
        requested_stored_dc_out_kwh=requested_dc_out,
        battery_ac_served_kwh=ac_served,
        unmet_battery_ac_kwh=unmet_ac,
        grid_import_ac_kwh=grid_import,
        grid_export_ac_kwh=exported,
        battery_power_kw=abs(net_dc_kw),
    )


def params_from_soc_projection(params) -> SlotEnergyParams:
    """Adapt a ``SocProjectionParams`` to ``SlotEnergyParams``.

    The two describe the same battery; keeping one conversion helper stops
    consumers from hand-copying six fields and silently dropping one.
    """
    return SlotEnergyParams(
        battery_capacity=params.battery_capacity,
        efficiency=params.efficiency,
        discharge_rate=params.discharge_rate,
        export_discharge_rate=params.export_discharge_rate,
        inverter_efficiency=params.inverter_efficiency,
        min_soc=params.min_soc,
        max_soc=params.max_soc,
        slot_minutes=params.slot_minutes,
    )


def clamp_soc(params: SlotEnergyParams, soc: float) -> float:
    return min(params.max_soc, max(params.min_soc, soc))


def charge_rate_for_span(
    rate_fn,
    soc_start: float,
    temp,
    duration_h: float,
    efficiency: float,
    capacity: float,
    max_soc: float = 100.0,
) -> float:
    """THE constant charge rate one (partial) slot is modelled at.

    A slot still runs at ONE rate -- the DP's inner loop, `simulate_slot`,
    `project_slot_soc` and both replays all depend on that. What changed is
    WHICH one: the rate is evaluated at the SOC the slot starts from *and* at
    the SOC that rate would reach (capped at ``max_soc``), and the MINIMUM of
    the two is used for the whole slot.

    Freezing it at the start SOC over-credited every slot that crossed one of
    the learning engine's SOC-taper buckets (25 / 50 / 75 / 90 %), and no
    validation could catch it: `plan_validation.replay_plan` and
    `DPOptimizer._replay_plan` evaluated the same frozen model. Measured on the
    reference 10 kWh pack with a 4 kW -> 1 kW taper at 90 % and a 15-minute
    slot from 88 %: sub-stepped truth 92.0 %, frozen model 98.0 % -- six SOC
    points of energy credited to a plan that could not take it.

    What this rule is, precisely:

    * **exact** for a piecewise-constant bucket rate when NO bucket boundary
      falls strictly inside ``[soc_start, reached_soc)``. Both probes then
      return the same rate and there is nothing to approximate. On the
      reference pack a 15-minute slot moves at most ~8 SOC points against
      25-point buckets, so most slots clear the buckets entirely -- but a slot
      that DOES cross one is not exact, it is merely conservative: 88 % over a
      4 kW -> 1 kW taper at 90 % gives 90.5 % against a sub-stepped truth of
      92.0 %, and 22 % over a 1 kW -> 4 kW step at 23 % gives 24.5 % against
      29.0 %. "At most one boundary" was the condition for the rule being
      well-behaved, never for it being exact;
    * **conservative** (never over-credits) when the rate is monotone over the
      span the slot covers -- the minimum of the endpoints is then a lower
      bound on every rate the slot visits;
    * an **approximation** otherwise. A curve that dips between the two
      endpoints and recovers is still over-credited; the only statement that
      always holds is the identity bound
      ``(max rate visited - min rate visited) * duration_h * efficiency``.

    Temperature is NOT spanned: the rate is looked up at the temperature the
    slot starts at, because temperature evolves between slots through
    ``thermal_model.TemperatureProjector`` and the DP's 1-D energy state cannot
    carry it. That is conservative when the pack WARMS during the slot -- the
    physical case while charging -- and over-credits a pack that cools while
    charging, bounded by
    ``(rate(T_start) - rate(T_end)) * duration_h * efficiency`` **only when the
    rate is NON-DECREASING in temperature over the range the slot traverses**.
    That is the direction that makes the bound mean anything: a rate that only
    ever falls as the pack cools stays between ``rate(T_end)`` and
    ``rate(T_start)``, so the difference of the endpoints caps the over-credit.
    Under the opposite condition the model cannot over-credit a cooling pack at
    all -- every temperature it visits is at least as fast as the one the rate
    was looked up at -- so the bound would be describing a case that does not
    arise. That bound compares two endpoints, so a rate that is non-monotone in
    temperature breaks it outright: a pack cooling 20 -> 5 C on a curve of
    2.0 kW at or above 19 C, 0.1 kW from 11 to 19 C and 1.9 kW below 11 C has
    both endpoints fast and the middle slow -- the model says 25.0 %, the
    sub-stepped truth is 22.4 %, and the endpoint bound allows 0.25. Outside
    that condition only the identity bound
    ``(max rate visited - min rate visited) * duration_h * efficiency`` holds.

    See ``docs/scheduling-algorithm.md`` SS Within-slot charge model and
    ``tests/test_within_slot_charge_model.py``.

    Args:
        rate_fn: ``(soc, temp) -> charge_input_dc_kw``.
        soc_start: SOC (%) the slot starts from -- the rate lookup basis.
        temp: Battery temperature at the start of the slot (may be None).
        duration_h: Length of the (possibly partial) slot in hours.
        efficiency: Storage retention factor, so the reached SOC is the SOC the
            pack would really hold.
        capacity: Battery capacity (kWh).
        max_soc: Upper SOC bound; the reached SOC is a physical SOC, never an
            extrapolation past the pack's own ceiling.
    """
    start_rate = max(0.0, float(rate_fn(soc_start, temp) or 0.0))
    if start_rate <= 0.0 or duration_h <= 0.0 or capacity <= 0.0:
        return start_rate
    eff = efficiency if efficiency > 0 else 0.0
    reached_soc = min(
        max_soc, soc_start + start_rate * duration_h * eff / capacity * 100.0
    )
    if reached_soc <= soc_start + 1e-12:
        return start_rate
    # The span the pack actually spends time in is the HALF-OPEN interval
    # ``[soc_start, reached_soc)``: it is at ``reached_soc`` for zero seconds.
    # Probing the closed endpoint would make a slot that exactly fills a learned
    # bucket -- the shape of a clean calibration observation, 40 % -> 50 % in
    # 15 minutes -- run at the NEXT bucket's rate for its whole length and stop
    # replaying its own measurement. The relative step back is large enough to
    # survive the float error in ``reached_soc`` (order 1e-15 of the span) and
    # small enough that any boundary strictly inside the span is still found.
    probe_soc = soc_start + (reached_soc - soc_start) * (1.0 - SPAN_ENDPOINT_EPS)
    end_rate = max(0.0, float(rate_fn(probe_soc, temp) or 0.0))
    return min(start_rate, end_rate)


__all__ = [
    "SPAN_ENDPOINT_EPS",
    "ENERGY_EPS",
    "SlotEnergyParams",
    "SlotEnergyResult",
    "simulate_slot",
    "charge_rate_for_span",
    "params_from_soc_projection",
    "clamp_soc",
]
