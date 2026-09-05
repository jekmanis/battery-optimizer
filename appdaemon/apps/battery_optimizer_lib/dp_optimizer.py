"""
Dynamic Programming optimizer for battery scheduling.

Chooses a charge/hold/discharge schedule by SOC-aware dynamic programming with
temperature-aware charge-rate predictions.

What is EXACT and what is APPROXIMATE is spelled out below the imports and in
``docs/scheduling-algorithm.md``. In one line: the physics of every transition
is exact and the published plan is replay-feasible; the SEARCH is not, because
one path per SOC bucket is kept and that is not a valid dominance rule.
"""

import datetime
import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import BatteryOptimizerConfig

from .models import NO_PRICE_REASON, BatteryMode, PricePoint, ScheduleEntry
from .slot_energy import (
    ENERGY_EPS,
    SlotEnergyParams,
    charge_rate_for_span,
    simulate_slot,
)
from .timezone_utils import canonical_slot_key, instant_key


# ---------------------------------------------------------------------------
# Charge rates that match the SOC and temperature the plan actually reaches
# ---------------------------------------------------------------------------
#
# The rate used to come from a time-indexed array built by advancing SOC and
# temperature "as if charging ran continuously from now", and `_run_dp` then
# applied that one number to every reachable state at that time. Both halves
# were wrong in opposite directions: imaginary charging warmed a cold pack, so
# a later planned charge looked faster than the selected path could achieve;
# and imaginary charging pushed SOC into a taper region, so paths that stayed
# low or discharged were denied capability they had.
#
# SOC dependence is now exact: the rate is evaluated per candidate transition,
# from that state's own energy, cached by (soc, slot temperature).
#
# Temperature depends on HISTORY, which a 1-D energy state cannot carry. Of the
# three designs available, this is a BOUNDED SOLVE / REPLAY / REFINE:
#
#   pass 0  plan with the IDLE temperature profile -- the pack is only as warm
#           as it would be with no battery activity at all, so no heat can come
#           from an action the plan has not committed to;
#   pass n  replay the selected plan through the shared TemperatureProjector
#           (warming from ACTUAL battery flow, so a full pack ordered to charge
#           contributes nothing), re-derive the profile, and solve again;
#   stop    when the plan is FEASIBLE at the temperatures the replay says it
#           reaches AND the profile is stable; else on a repeated profile
#           (oscillation) or on the pass budget.
#
# THE CRITERION IS FEASIBILITY AT THE REACHED TEMPERATURE, not a fixed point.
# `_replay_plan` walks the selected plan forward, looks the rate up at the
# temperature the pack has actually reached in that walk, and reports how much
# of the charge energy the plan credited the pack could not have taken. A fixed
# point alone is not enough and never was: the loop used to be skipped entirely
# by a sampled "is the rate temperature sensitive?" probe, and a learned bucket
# that varied only at a SOC between the probes then planned 4 kW of charging
# into a slot where the pack, at the temperature its own earlier CHARGE slot
# produced, could take 1 kW -- 0.75 kWh of the next slot's load uncovered, with
# `converged=True` and no fallback to hint at it
# (`tests/test_thermal_feasibility_refinement.py`).
#
# There is deliberately no sensitivity probe any more. `charge_rate_predictor`
# is an arbitrary callable and a finite sample of it proves nothing about the
# temperatures the refinement will reach -- the same reasoning that removed the
# SOC-independence hoist (see the NOTE on `_rate_for`). With a temperature
# reading the refinement always runs, bounded by the pass budget; without one
# there is nothing for it to refine and a single solve is the whole answer.
#
# On oscillation or exhaustion it falls back to a CONSERVATIVE solve: per
# (slot, SOC) the MINIMUM rate over every temperature profile seen in this
# call, then one more replay. Limits of that fallback, stated rather than
# implied:
#
#   * "Minimum over the profiles seen" is a genuine lower bound only over those
#     profiles. If the conservative plan reaches a temperature none of them
#     visited and the rate curve dips there, it can still be short -- which is
#     why the replay after it is not optional.
#   * If it IS still short, the branch DEGRADES: the credited charge energy is
#     reduced to what the replayed temperature actually allows and the whole
#     trajectory is rebuilt from that walk. Economic optimality is lost in that
#     branch -- the plan's actions were chosen for energy the pack will not
#     have -- and it is logged at WARNING with the shortfall in kWh. What is
#     preserved is the thing that matters: no published number credits energy
#     the model says is unavailable.
#   * A fixed point is not a proof of optimality of any kind. See the state
#     representation note below for what the solver does and does not guarantee.
#
# `DPOptimizerResult.rate_refinement_branch` says which of the four paths
# produced the plan: "single_solve", "converged", "conservative_fallback",
# "degraded".
#
# Rejected: putting discretized temperature in the DP state. It is the clearest
# formulation, but it multiplies the state count by the number of temperature
# buckets, and the partial-first-slot lookahead already runs the whole DP once
# per candidate -- the normal 132-slot horizon would go from ~125 ms to well
# over a second on the single AppDaemon thread.
#
# Also rejected: a fixed conservative temperature. "Coldest plausible" is not a
# valid bound over reachable conditions once SOC tapering and non-monotonic rate
# behaviour are in play, and it would refuse to plan the warm-pack charging the
# installation actually does.

# Solves per optimize() call: this many, plus at most one conservative final
# solve when the refinement oscillates or runs out of passes.
MAX_RATE_REFINEMENT_PASSES = 3

# Two temperature profiles are "the same" within this much (C). Coarser than
# the thermal model's resolution and far finer than the learning engine's 5 C
# temperature buckets, so it terminates without changing which bucket is used.
TEMP_FIXED_POINT_C = 0.25

# A plan is short of its own credited charge energy by more than this (kWh) or
# it is feasible. It is a float-accumulation allowance over a few hundred
# additions of order 1 kWh, NOT a budget for modelling error: a genuine
# disagreement is a rate lookup landing in a different bucket, which moves
# whole tenths of a kWh, never 1e-9.
CHARGE_FEASIBILITY_EPS_KWH = 1e-9

# The one place the "this slot runs the pack dry" prose is spelled. It is
# appended and removed together with `ScheduleEntry.energy_limited`, so the log
# text and the flag cannot disagree.
DEPLETION_NOTE = " (until depleted)"


# ---------------------------------------------------------------------------
# Intervals nobody published a price for, INSIDE the horizon
# ---------------------------------------------------------------------------
#
# The DP used to be handed only the intervals a source had published, so a hole
# in the middle of the horizon did not exist for planning: the slot after it was
# treated as following the slot before it, and the gap's PV, load, SOC and
# temperature were never modelled. On the reference reproduction (10 kWh pack at
# its 10 % minimum, 10:00 at 0.50 with nothing happening, 10:15 unpublished with
# 4 kW of PV, 10:30 at 1.00 with 4 kW of load) the plan charged at 10:00 and
# imported a kWh that the gap's own sunshine would have supplied for free.
#
# A missing interval is therefore a slot the horizon CONTAINS, entered with
# ``PricePoint.price is None``:
#
#   * only the HOLD transition is evaluated. The action is fixed -- nothing may
#     be chosen at a price that does not exist -- so there is no decision to
#     make, only physics to carry;
#   * PV absorption is modelled exactly as in any other HOLD slot (headroom,
#     the span rate), because that is physics and needs no price;
#   * the slot's grid IMPORT and EXPORT cash flows are left out of the
#     objective. Import is path-independent under a fixed action -- every path
#     through the slot draws the same net load -- so omitting it cannot change
#     which plan wins, only the absolute value the DP reports. Export at an
#     unknown price is valued at 0, which is the conservative direction: the
#     plan is never talked into spending energy for revenue nobody quoted.
#
# The terminal-rate median is taken over PRICED slots only, for the same
# reason: an unpriced slot has no price to be the median of.
#
# What the DP must NOT do is invent a number. There is no "carry the previous
# price forward", no "use the next one": that is the failure
# `_ensure_current_slot_price` was removed for, one slot further into the
# horizon.


def _is_priced(point: PricePoint) -> bool:
    """Whether a source published a price for this interval."""
    return point.price is not None


def _with_depletion_note(reason: str, energy_limited: bool) -> str:
    """``reason`` carrying the depletion note iff the slot is energy-limited."""
    reason = reason or ""
    has_note = DEPLETION_NOTE in reason
    if energy_limited and not has_note:
        return reason + DEPLETION_NOTE
    if not energy_limited and has_note:
        return reason.replace(DEPLETION_NOTE, "")
    return reason


def _energy_to_index(
    energy: float,
    min_energy: float,
    step_kwh: float,
    n_states: int,
    direction: str = "round",
) -> int:
    """Convert energy to DP state index with specified rounding.

    Args:
        energy: Energy level in kWh
        min_energy: Minimum energy bound in kWh
        step_kwh: Energy step size in kWh
        n_states: Number of DP states
        direction: "floor" for every state transition — a state's LABEL must
            never claim more energy than the path holds. "round"/"ceil" survive
            for reporting helpers only.

    Returns:
        Clamped index in [0, n_states - 1]
    """
    idx_float = (energy - min_energy) / step_kwh

    if direction == "floor":
        idx = int(math.floor(idx_float + 1e-9))
    elif direction == "ceil":
        idx = int(math.ceil(idx_float - 1e-9))
    else:
        idx = int(round(idx_float))

    return min(max(idx, 0), n_states - 1)


# ---------------------------------------------------------------------------
# The state representation: bucket LABEL + the path's EXACT energy
# ---------------------------------------------------------------------------
#
# A DP state used to be identified with the energy of its grid point, and a
# discharge was rounded to the NEAREST grid point. That is only unbiased for a
# random signal. A constant load on a constant slot length produces the same
# error, with the same sign, in every slot: at 0.14 kWh per slot on a 0.10 kWh
# grid the DP deducted 0.10 kWh twenty times and credited 2.8 kWh of service
# from a 2.0 kWh battery. `_discharge_index`'s no-free-lunch guard caught only
# the sub-step case, never the systematic one.
#
# Rounding DOWN instead is safe but far too pessimistic to use on its own:
# measured on the same reproduction, floor-to-grid serves 1.40 kWh of the
# 2.00 kWh the pack holds and throws 30 % of it away, discharging for 10 slots
# instead of 15. A 15-minute slot on the reference installation routinely moves
# barely two grid steps, so this is the normal case, not an edge case. Reaching
# an acceptable loss by shrinking `soc_step_percent` costs proportionally more
# states and CPU (measured over a 132-slot horizon with a partial first slot:
# 1 % -> 125 ms, 0.5 % -> 241 ms, 0.25 % -> 527 ms).
#
# So the grid stays where it is and the state carries BOTH:
#
#   * an index, floored, so a state's label never claims energy the path does
#     not hold; the index is what merges paths, i.e. it is the resolution of
#     the OPTIMIZATION;
#   * `dp_energy[idx]`, the EXACT continuous energy of the best path reaching
#     that bucket, which is what every transition is computed from.
#
# Physics is therefore exact: no transition can create a joule, and replaying
# the backtracked plan continuously reproduces the planner's own trajectory to
# floating-point precision.
#
# ---------------------------------------------------------------------------
# What this solver is, and is NOT
# ---------------------------------------------------------------------------
#
# EXACT:
#   * the physics of every transition -- energy in, energy out, grid import,
#     export, unmet demand, at the path's exact continuous energy;
#   * replay parity -- `plan_validation.replay_plan` walking the published
#     action sequence through `slot_energy.simulate_slot` reproduces the
#     planner's own SOC trajectory (proven by sweep in
#     `tests/test_dp_energy_conservation.py`);
#   * SOC dependence of the charge rate: evaluated per candidate transition;
#   * the value arithmetic of a given action sequence under a given temperature
#     profile.
#
# APPROXIMATE -- and this is a real gap, not a rounding remark:
#
#   THE BUCKET MERGE IS NOT AN EXACT STATE REDUCTION. Keeping one path per SOC
#   bucket -- the highest-valued one -- is NOT a valid dominance rule. A
#   higher-valued path does not dominate a lower-valued path that holds more
#   energy: the extra energy can be worth more later than the value gap is
#   worth now. This solver is therefore NOT "exact for its discretized model".
#   That claim used to be made here and in the docs; it was wrong.
#
#   The counterexample is small and is pinned as a regression test
#   (`tests/test_merge_approximation.py`). At the default 1 % step: 10 kWh pack,
#   min SOC 10 %, initial 10.9 %, two 15-minute slots drawing 0.2 kW at 0.10
#   then 1.00 EUR/kWh, unit efficiencies, no fees, wear or terminal value. The
#   solver returns DISCHARGE, DISCHARGE at 0.010 EUR; exhaustive enumeration of
#   the same action space finds HOLD, DISCHARGE at 0.005 EUR. Both PATHS land in
#   the same bucket after slot 1 (its 0.1 kWh span covers 1.00 to 1.10 kWh);
#   DISCHARGE ends there holding 1.04 kWh and wins on value (it saves
#   0.10 EUR/kWh of cheap import) and the HOLD path, which held 1.09 kWh -- 0.05
#   kWh more -- is discarded, and that 0.05 kWh was worth 1.00 EUR/kWh in slot 2.
#
#   Size of the gap:
#     * per merge, ENERGY loss < one step (a bucket is one step wide);
#     * per merge, VALUE loss <= step * marginal value of a kWh;
#     * over the horizon, the only bound established here is the sum,
#       `n_slots * step * marginal_value`. A discarded path can be discarded
#       again at every later slot. That bound is loose -- the errors are not
#       independent and a merged path usually rejoins -- but nothing in this
#       implementation proves anything tighter, so the loose statement is the
#       honest one.
#
#   Ties are broken toward the path holding more energy. THAT is a genuine
#   dominance rule (equal value, more energy can only widen later options); it
#   is what makes the merge deterministic, and it is not what is approximate.
#
#   The exact alternative -- Pareto-nondominated (value, energy) labels per
#   bucket -- and why it is not used are described in
#   `docs/scheduling-algorithm.md` SS Conservative quantization.
#
#   None of this is a licence to invent energy. The merge discards a plan; it
#   never credits a joule the pack does not hold.
#
#   On top of the merge, the temperature PROFILE is an outer approximation (see
#   the refinement note above), the within-slot charge rate is constant (see
#   `soc_projection`), and forecasts are forecasts.


@dataclass
class _PlanReplay:
    """One forward walk of a selected plan at the temperatures it reaches."""

    # Start-of-slot temperature the walk produced.
    temp_profile: List[Optional[float]]
    # (start, end) temperature per slot.
    temp_pairs: List[Tuple[Optional[float], Optional[float]]]
    # End-of-slot stored energy of the WALK. Meaningful as a trajectory only
    # when the walk was run with ``follow_plan_energy=False``.
    energies: List[float]
    # kWh of stored charge energy the plan credits that the reached
    # temperatures cannot supply.
    charge_shortfall_kwh: float
    shortfall_by_slot: List[float]
    # Slots where the walk could not serve the AC demand the plan assigned to
    # the battery. This is exactly what `plan_validation` requires the plan to
    # have DECLARED (`ScheduleEntry.energy_limited`).
    unmet_slots: List[bool]


@dataclass
class DPOptimizerConfig:
    """Static configuration for DP optimizer."""
    battery_capacity: float      # kWh
    min_soc: float               # % (e.g., 10.0)
    max_soc: float               # % (e.g., 100.0)
    efficiency: float            # 0-1 (e.g., 0.85)
    discharge_rate: float        # kW
    export_discharge_rate: float = 0.0  # kW — discharge rate during grid export (0 = use discharge_rate)
    slot_minutes: int = 15       # e.g., 60
    soc_step_percent: float = 1.0  # DP resolution (e.g., 1.0)
    grid_fee: float = 0.052     # EUR/kWh — trading margin + distribution on purchases
    battery_wear_cost: float = 0.0  # EUR/kWh
    grid_export_fee: float = 0.02  # EUR/kWh — fixed deduction from spot when selling
    export_rate_multiplier: float = 1.0   # Sell price = price * multiplier - export_fee
    inverter_efficiency: float = 1.0  # AC↔DC conversion efficiency (e.g., 0.97)
    import_price_multiplier: float = 1.0  # VAT/tax multiplier applied to spot + variable import fees
    # EUR per stored DC kWh at the end of the horizon. None derives a value
    # from the median forecast import price; 0 disables terminal valuation.
    terminal_energy_value_eur_kwh: Optional[float] = 0.0

    @property
    def slot_hours(self) -> float:
        return self.slot_minutes / 60.0

    @property
    def effective_export_discharge_rate(self) -> float:
        """Discharge rate during grid export (kW). Falls back to discharge_rate if not set."""
        return self.export_discharge_rate if self.export_discharge_rate > 0 else self.discharge_rate

    @classmethod
    def from_main_config(
        cls, cfg: "BatteryOptimizerConfig", *, min_soc: float, max_soc: float
    ) -> "DPOptimizerConfig":
        """Create from the central BatteryOptimizerConfig plus dynamic SOC limits."""
        return cls(
            battery_capacity=cfg.battery_capacity,
            min_soc=min_soc,
            max_soc=max_soc,
            efficiency=cfg.efficiency,
            discharge_rate=cfg.discharge_rate,
            export_discharge_rate=cfg.export_discharge_rate,
            slot_minutes=cfg.slot_minutes,
            soc_step_percent=cfg.soc_step_percent,
            grid_fee=cfg.grid_fee,
            grid_export_fee=cfg.grid_export_fee,
            battery_wear_cost=cfg.battery_wear_cost,
            export_rate_multiplier=cfg.export_rate_multiplier,
            inverter_efficiency=cfg.inverter_efficiency,
            import_price_multiplier=cfg.import_price_multiplier,
            terminal_energy_value_eur_kwh=cfg.terminal_energy_value_eur_kwh,
        )


@dataclass
class DPOptimizerResult:
    """Immutable result from optimization."""
    schedule: Dict[datetime.datetime, ScheduleEntry]
    soc_trajectory: Dict[datetime.datetime, Tuple[float, float]]
    temp_trajectory: Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]]
    charge_count: int
    hold_count: int
    export_slot_count: int = 0
    self_consume_slot_count: int = 0
    terminal_value_eur_kwh: Optional[float] = None
    # Rate/temperature refinement diagnostics (see the module header).
    rate_refinement_passes: int = 1
    rate_refinement_converged: bool = True
    rate_refinement_fallback: bool = False
    rate_refinement_degraded: bool = False
    # Which of the four paths produced this plan:
    #   "single_solve"          no temperature reading; nothing to refine
    #   "converged"             feasible at its replayed temperatures, profile
    #                           stable
    #   "conservative_fallback" oscillation or budget exhausted; solved on the
    #                           minimum rate over every profile seen, and that
    #                           plan IS feasible at its replayed temperatures
    #   "degraded"              the conservative solve was still short; the
    #                           credited charge energy has been reduced to what
    #                           the replayed temperature allows and the
    #                           trajectory rebuilt from that walk. Economic
    #                           optimality is lost; no unavailable energy is
    #                           credited.
    rate_refinement_branch: str = "single_solve"
    # kWh of charge energy the plan credited that the replayed temperature
    # could not supply, at the point the degrade branch was entered. Zero on
    # every other branch.
    rate_refinement_shortfall_kwh: float = 0.0
    # Start-of-slot temperatures the FINAL plan is feasible at -- the profile
    # the last replay produced, not a pinned planning assumption. Reporting and
    # diagnostics only: no consumer looks charge rates up at these any more.
    # Pinning them was how validation ended up checking the planner's
    # arithmetic against the planner's own assumption.
    planning_temp_by_slot: Dict[datetime.datetime, Optional[float]] = field(
        default_factory=dict
    )


class DPOptimizer:
    """
    SOC-aware dynamic programming optimizer for battery scheduling.

    Uses dependency injection for external functions:
    - load_predictor: predicts load (kW) for a given datetime
    - charge_rate_predictor: predicts charge rate (kW) for SOC and temperature
    - temp_after_charge_predictor: predicts temperature after charging
    - temp_after_idle_predictor: predicts temperature after idle period
    - pv_predictor: predicts PV production (kW) for a given datetime (optional)
    """

    def __init__(
        self,
        config: DPOptimizerConfig,
        load_predictor: Callable[[datetime.datetime], float],
        charge_rate_predictor: Callable[[float, Optional[float]], float],
        temp_after_charge_predictor: Callable[[float, float], float],
        temp_after_idle_predictor: Callable[[float, float], float],
        log_fn: Optional[Callable] = None,
        decision_log_level: int = 0,
        pv_predictor: Optional[Callable[[datetime.datetime], float]] = None,
        temp_projector=None,
        warn_degenerate_terminal: bool = True,
    ):
        self._config = config
        self._predict_load_kw = load_predictor
        self._get_charge_rate_for_soc = charge_rate_predictor
        self._predict_temp_after_duration = temp_after_charge_predictor
        self._predict_temp_after_idle = temp_after_idle_predictor
        self._log_fn = log_fn
        self._decision_log_level = decision_log_level
        self._predict_pv_kw = pv_predictor
        # Shared thermal model (thermal_model.TemperatureProjector). When absent
        # the legacy mode-specific predictors are used so existing callers and
        # tests keep working.
        self._temp_projector = temp_projector
        # (soc, temperature) -> charge_input_dc_kw, reset per optimize() call.
        self._rate_cache: Dict[Tuple[float, Optional[float]], float] = {}
        # The EXACT state SOCs `_run_dp` evaluated a rate at during the last
        # solve. `_profiles_agree`'s "do these two profiles imply the same
        # charge capability?" criterion is evaluated at these and not at the
        # grid: the grid is where a state's LABEL sits, `dp_energy[idx]` is what
        # every transition is computed from, and those are not the same points.
        self._visited_state_socs: set = set()
        # Whether a terminal value pinned to 0 (legacy/degenerate mode) is
        # surfaced at WARNING from this optimizer instance. The orchestrator
        # sets it only for the daily/full optimization so the adaptive
        # re-evaluations (every 15 min) don't produce 96 warnings a day.
        self._warn_degenerate_terminal = warn_degenerate_terminal

    def _log(self, message: str, level: str = "INFO"):
        """Log a message using the provided log function."""
        if self._log_fn:
            self._log_fn(message, level=level)

    def optimize(
        self,
        prices: List[PricePoint],
        current_slot: datetime.datetime,
        current_soc: float,
        current_temp: Optional[float] = None,
        minutes_into_slot: float = 0.0,
    ) -> DPOptimizerResult:
        """
        Run the DP and return the best schedule it finds under the discretized
        model with bucket merging.

        Not "the optimal schedule": the merge keeps one path per SOC bucket,
        which is not a valid dominance rule (see the module header and
        ``tests/test_merge_approximation.py``).

        Args:
            prices: The modelled horizon: one point per slot, contiguous. A
                point whose ``price`` is ``None`` is an interval nobody
                published — it is planned as a forced HOLD and scores no cash
                flow (see the note above ``_is_priced``). Building that
                sequence is the orchestrator's job; this method plans whatever
                slots it is handed.
            current_slot: Current time slot (aligned to slot boundary)
            current_soc: Current battery state of charge (%)
            current_temp: Current battery temperature (Celsius, optional)
            minutes_into_slot: Minutes elapsed in current slot

        Returns:
            DPOptimizerResult with schedule and trajectories
        """
        if not prices:
            return DPOptimizerResult(
                schedule={},
                soc_trajectory={},
                temp_trajectory={},
                charge_count=0,
                hold_count=0,
            )

        cfg = self._config
        # Preserve local wall-clock fields for load/PV predictors, but replace
        # region timezones with the concrete offset of each price interval.
        # This keeps the two repeated autumn-DST intervals distinct as dict
        # keys (ZoneInfo datetimes with different ``fold`` values can compare
        # equal when they share the same timezone object).
        canonical_prices = [
            PricePoint(
                time=(
                    canonical_slot_key(p.time)
                ),
                price=p.price,
            )
            for p in prices
        ]
        if current_slot.tzinfo is not None and current_slot.utcoffset() is not None:
            current_slot = canonical_slot_key(current_slot)

        slots_sorted_by_time = sorted(canonical_prices, key=lambda point: instant_key(point.time))
        n_slots = len(slots_sorted_by_time)

        # Energy bounds in kWh
        min_energy = (cfg.min_soc / 100) * cfg.battery_capacity
        max_energy = (cfg.max_soc / 100) * cfg.battery_capacity
        start_energy = min(max_energy, max(min_energy, (current_soc / 100) * cfg.battery_capacity))

        # DP resolution
        # soc_step_percent is an explicit accuracy/performance control. Coarser
        # grids are faster but can make short partial-slot transitions more
        # conservative because discharge residuals are rounded downward.
        configured_step_kwh = (cfg.soc_step_percent / 100) * cfg.battery_capacity
        step_kwh = max(0.01, configured_step_kwh)
        # Never create a discrete state above max SOC when the configured
        # range is not an exact multiple of the effective step.
        n_states = int(math.floor((max_energy - min_energy) / step_kwh + 1e-9)) + 1
        energy_levels = [min_energy + i * step_kwh for i in range(n_states)]

        # Per-slot energy changes (adjust first slot if partial)
        first_fraction = min(1.0, max(0.0, (cfg.slot_minutes - minutes_into_slot) / max(1, cfg.slot_minutes)))
        slot_fractions = [1.0] * n_slots
        current_slot_index = None

        for i, p in enumerate(slots_sorted_by_time):
            p_time = p.time
            compare_current = current_slot
            if p_time.tzinfo is not None and compare_current.tzinfo is None:
                p_time = p_time.replace(tzinfo=None)
            elif p_time.tzinfo is None and compare_current.tzinfo is not None:
                compare_current = compare_current.replace(tzinfo=None)
            if p_time == compare_current:
                slot_fractions[i] = first_fraction
                current_slot_index = i
                break

        # Pre-compute load and PV per slot. Fixed for the whole solve, including
        # every refinement pass: a forecast that moved mid-refinement would be
        # indistinguishable from a failure to converge.
        load_kw = [self._predict_load_kw(p.time) for p in slots_sorted_by_time]
        pv_kw = (
            [self._predict_pv_kw(p.time) for p in slots_sorted_by_time]
            if self._predict_pv_kw else None
        )

        # Rate lookups are cached for this call only; the cache key is what
        # actually determines the answer, (state SOC, slot temperature).
        self._rate_cache = {}

        terminal_rate = self._derive_terminal_rate(slots_sorted_by_time)
        if self._decision_log_level >= 1:
            if cfg.terminal_energy_value_eur_kwh == 0.0:
                # No-salvage mode: nothing is "worth less than 0", so the plan
                # spends the battery at the horizon edge. State the mode and its
                # trade-off; do not prescribe a value. Both settings have a real
                # failure mode and the right choice is installation-specific, so
                # this is INFO, not a warning.
                if self._warn_degenerate_terminal:
                    self._log(
                        "Terminal energy value: 0.0000 EUR/kWh (configured; "
                        "no-salvage mode) — stored energy has no value at the end "
                        "of the horizon, so the plan spends it there. Harmless "
                        "while the daily re-optimization extends the horizon "
                        "before those slots execute; see "
                        "terminal_energy_value_eur_kwh in apps.yaml for the "
                        "trade-off against \"auto\".",
                    )
            else:
                source = (
                    "auto: median horizon buy x inv_eff - wear"
                    if cfg.terminal_energy_value_eur_kwh is None
                    else "configured"
                )
                self._log(
                    f"Terminal energy value: {terminal_rate:.4f} EUR/kWh ({source}); "
                    f"net-load slots worth less than this are HELD"
                )

        # ------------------------------------------------------------------
        # Bounded solve / replay / refine over the temperature profile
        # ------------------------------------------------------------------
        temp_profile = self._idle_temp_profile(
            slots_sorted_by_time, slot_fractions, current_temp
        )
        # With a temperature reading the refinement ALWAYS runs. There is no
        # sampled "is this curve temperature sensitive?" shortcut: a finite
        # probe of an arbitrary callable proves nothing about the temperatures
        # the refinement itself will reach, and the probe that used to be here
        # skipped the whole loop for a curve that varied only between its
        # sample points.
        refine = current_temp is not None
        # Fallback only, for the case where a solve visited nothing (an empty
        # horizon). The real set is the exact state energies the solve reached,
        # collected during `_run_dp` and read back after each `_solve`.
        grid_state_socs = (
            [e / cfg.battery_capacity * 100 for e in energy_levels]
            if cfg.battery_capacity else []
        )
        state_socs = grid_state_socs
        seen_profiles: List[List[Optional[float]]] = []
        # Every profile any rate in this call was ever taken from. The
        # conservative fallback minimises over exactly these.
        profiles_used: List[List[Optional[float]]] = [list(temp_profile)]
        passes = 0
        branch = "single_solve"
        shortfall_kwh = 0.0
        replay = None

        def _solve(profile, rate_fns=None):
            return self._build_schedule(
                slots_sorted_by_time=slots_sorted_by_time,
                load_kw=load_kw,
                temp_profile=profile,
                slot_fractions=slot_fractions,
                current_slot_index=current_slot_index,
                start_energy=start_energy,
                min_energy=min_energy,
                max_energy=max_energy,
                step_kwh=step_kwh,
                n_states=n_states,
                energy_levels=energy_levels,
                pv_kw=pv_kw,
                rate_fns=rate_fns,
            )

        while True:
            passes += 1
            schedule, idx_trajectory, energy_trajectory, best_value = _solve(temp_profile)
            # Criterion 2 of `_profiles_agree` asks whether two profiles imply
            # the same charge capability. "The same" has to mean at the states
            # the solve actually visits, which are exact path energies, not grid
            # points.
            state_socs = self.last_visited_state_socs() or grid_state_socs
            if not refine:
                branch = "single_solve"
                break

            replay = self._replay_plan(
                slots_sorted_by_time=slots_sorted_by_time,
                slot_fractions=slot_fractions,
                schedule=schedule,
                energy_trajectory=energy_trajectory,
                start_energy=start_energy,
                current_temp=current_temp,
                load_kw=load_kw,
                pv_kw=pv_kw,
            )
            profiles_used.append(list(replay.temp_profile))
            feasible = replay.charge_shortfall_kwh <= CHARGE_FEASIBILITY_EPS_KWH
            stable = self._profiles_agree(
                replay.temp_profile, temp_profile, state_socs
            )
            if feasible and stable:
                branch = "converged"
                temp_profile = replay.temp_profile
                break

            oscillating = any(
                self._profiles_agree(replay.temp_profile, p, state_socs)
                for p in seen_profiles
            )
            if oscillating or passes >= MAX_RATE_REFINEMENT_PASSES:
                branch, shortfall_kwh, schedule, energy_trajectory, replay = (
                    self._conservative_solve(
                        solve=_solve,
                        profiles_used=profiles_used,
                        n_slots=n_slots,
                        oscillating=oscillating,
                        slots_sorted_by_time=slots_sorted_by_time,
                        slot_fractions=slot_fractions,
                        start_energy=start_energy,
                        current_temp=current_temp,
                        load_kw=load_kw,
                        pv_kw=pv_kw,
                    )
                )
                passes += 1
                temp_profile = replay.temp_profile
                break

            seen_profiles.append(list(temp_profile))
            temp_profile = replay.temp_profile

        # ------------------------------------------------------------------
        # ONE trajectory, and it is the PHYSICAL OUTCOME of the action sequence
        # ------------------------------------------------------------------
        #
        # Whatever branch produced the actions, what gets published is the walk
        # of those actions at the temperatures they reach, with the rate looked
        # up at the pack's own SOC and temperature. That is:
        #
        #   * identical to the DP's own energies in the "converged" branch, by
        #     construction -- the plan is feasible there and the profile is
        #     stable, so the walk repeats the planner's arithmetic;
        #   * the OUTCOME, not the assumption, in the conservative branch: the
        #     actions were chosen on deliberately pessimistic rates, and the
        #     inverter will nevertheless charge at whatever the pack can take;
        #   * the only honest answer in the degrade branch, where the plan
        #     credited energy the pack cannot take.
        #
        # It is also exactly what `plan_validation.replay_plan` and
        # `BatteryOptimizer.project_schedule_trajectory` compute, which is why
        # neither needs a pinned planning temperature to agree with the plan.
        if replay is not None:
            replay = self._replay_plan(
                slots_sorted_by_time=slots_sorted_by_time,
                slot_fractions=slot_fractions,
                schedule=schedule,
                energy_trajectory=energy_trajectory,
                start_energy=start_energy,
                current_temp=current_temp,
                load_kw=load_kw,
                pv_kw=pv_kw,
                follow_plan_energy=False,
            )
            energy_trajectory = list(replay.energies)
            temp_profile = replay.temp_profile
            # The FINAL walk decides `energy_limited`, in BOTH directions.
            #
            # Setting it only where the walk is short left the flag — and the
            # "(until depleted)" prose the schedule log prints — on every slot
            # the DP's own conservative pass thought would run the pack dry but
            # the walk serves in full. That is the normal outcome of the
            # conservative fallback: the plan is built on deliberately
            # pessimistic rates and the pack then has more energy than the plan
            # credited it with.
            for i, price_point in enumerate(slots_sorted_by_time):
                entry = schedule.get(price_point.time)
                if entry is None or i >= len(replay.unmet_slots):
                    continue
                entry.energy_limited = bool(replay.unmet_slots[i])
                entry.reason = _with_depletion_note(
                    entry.reason, entry.energy_limited
                )

        soc_trajectory = self._build_soc_trajectory(
            slots_sorted_by_time, energy_trajectory, start_energy
        )

        # The temperature trajectory is the final replay's — one thermal
        # implementation, and the same walk the feasibility criterion used.
        temp_trajectory = (
            {}
            if current_temp is None or replay is None
            else {
                p.time: replay.temp_pairs[i]
                for i, p in enumerate(slots_sorted_by_time)
                if i < len(replay.temp_pairs)
            }
        )

        # Count actions
        charge_count = sum(1 for e in schedule.values() if e.mode == BatteryMode.CHARGE)
        discharge_count = sum(1 for e in schedule.values() if e.mode == BatteryMode.DISCHARGE)
        hold_count = sum(1 for e in schedule.values() if e.mode == BatteryMode.HOLD)
        export_slot_count = sum(
            1 for e in schedule.values()
            if e.mode == BatteryMode.DISCHARGE and e.export_rate is not None and e.export_rate > 0
        )
        self_consume_slot_count = discharge_count - export_slot_count

        return DPOptimizerResult(
            schedule=schedule,
            soc_trajectory=soc_trajectory,
            temp_trajectory=temp_trajectory,
            charge_count=charge_count,
            hold_count=hold_count,
            export_slot_count=export_slot_count,
            self_consume_slot_count=self_consume_slot_count,
            terminal_value_eur_kwh=terminal_rate,
            rate_refinement_passes=passes,
            rate_refinement_converged=branch in ("single_solve", "converged"),
            rate_refinement_fallback=branch in (
                "conservative_fallback",
                "degraded",
            ),
            rate_refinement_degraded=(branch == "degraded"),
            rate_refinement_branch=branch,
            rate_refinement_shortfall_kwh=shortfall_kwh,
            planning_temp_by_slot={
                p.time: (temp_profile[i] if i < len(temp_profile) else None)
                for i, p in enumerate(slots_sorted_by_time)
            },
        )

    def _buy_price(self, price: float) -> float:
        """Marginal AC import price (EUR/kWh) including fees and taxes."""
        cfg = self._config
        return (price + cfg.grid_fee) * cfg.import_price_multiplier

    def _sell_price(self, price: float) -> float:
        """Marginal AC export revenue (EUR/kWh). NNS contract: floored at 0."""
        cfg = self._config
        return max(0.0, price * cfg.export_rate_multiplier - cfg.grid_export_fee)

    def _marginal_slot_value(
        self,
        action: BatteryMode,
        price: float,
        is_export: bool,
        terminal_rate: float,
    ) -> Tuple[float, str]:
        """EUR per battery DC kWh that this slot's decision is worth.

        REPORTING ONLY — the DP objective never reads this. It re-uses the very
        arithmetic `_run_dp` scores the slot with (`_buy_price`/`_sell_price`,
        inverter conversion, wear), normalized to one DC kWh so the schedule log
        can show why a slot was chosen. This is independent of the tracked
        stored-energy cost basis, which legitimately degenerates to 0.0000 when
        PV is booked at a zero export floor around midday.
        """
        cfg = self._config
        if action == BatteryMode.DISCHARGE:
            if is_export:
                return (
                    self._sell_price(price) * cfg.inverter_efficiency
                    - cfg.battery_wear_cost,
                    "export",
                )
            return (
                self._buy_price(price) * cfg.inverter_efficiency
                - cfg.battery_wear_cost,
                "avoided-import",
            )
        if action == BatteryMode.CHARGE:
            denom = max(1e-9, cfg.efficiency * cfg.inverter_efficiency)
            return -self._buy_price(price) / denom, "landed-charge"
        return terminal_rate, "kept"

    def _derive_terminal_rate(self, slots_list: List[PricePoint]) -> float:
        """EUR value of one stored DC kWh at the end of the horizon.

        Auto mode (config None) values it at the median avoided-import value
        over the given slots: any slot whose own discharge value falls below
        this is deliberately HELD — keeping the energy beats spending it there.

        The median is taken over PRICED slots only. An interval nobody
        published has no price to be the median of, and treating its absence as
        a number would be the invention this whole mechanism exists to avoid.
        With nothing priced at all there is no salvage value to derive.
        """
        cfg = self._config
        if cfg.terminal_energy_value_eur_kwh is not None:
            return cfg.terminal_energy_value_eur_kwh
        priced = [p.price for p in slots_list if _is_priced(p)]
        if not priced:
            return 0.0
        median_buy = statistics.median(
            (price + cfg.grid_fee) * cfg.import_price_multiplier
            for price in priced
        )
        return max(
            0.0,
            median_buy * cfg.inverter_efficiency - cfg.battery_wear_cost,
        )

    # ------------------------------------------------------------------
    # Charge rate: evaluated per candidate transition (see the module header)
    # ------------------------------------------------------------------

    def _rate_for(self, soc: float, temp: Optional[float]) -> float:
        """``charge_input_dc_kw`` at this SOC and temperature, memoized.

        The cache key is what actually determines the answer. A state's SOC is
        rounded to 1e-6 % only so that float noise does not defeat the cache;
        the temperature is used exactly, because a synthetic or learned rate
        curve may have a threshold anywhere.
        """
        key = (round(soc, 6), temp)
        cached = self._rate_cache.get(key)
        if cached is None:
            cached = max(0.0, float(self._get_charge_rate_for_soc(soc, temp) or 0.0))
            self._rate_cache[key] = cached
        return cached

    def last_visited_state_socs(self) -> List[float]:
        """The exact state SOCs the last solve evaluated a charge rate at.

        Not the DP's SOC grid: a state's bucket index is only its LABEL, and
        every transition is computed from ``dp_energy[idx]``, the path's exact
        continuous energy. Comparing two temperature profiles' implied charge
        capability at the grid was therefore comparing them at points the solve
        never asks about.
        """
        return sorted(self._visited_state_socs)

    def _span_rate(self, rate_fn, soc: float, temp, duration_h: float) -> float:
        """THE within-slot rate for a candidate transition.

        ``slot_energy.charge_rate_for_span``: the minimum of the rate at this
        state's SOC and the rate at the SOC that rate would reach. Two lookups
        per state per slot instead of one, and both go through ``_rate_for``'s
        cache. A single lookup at the start SOC over-credited every slot that
        crossed a learned SOC-taper bucket, and neither replay could catch it
        because both evaluated the same frozen model.
        """
        cfg = self._config
        return charge_rate_for_span(
            rate_fn,
            soc_start=soc,
            temp=temp,
            duration_h=duration_h,
            efficiency=cfg.efficiency,
            capacity=cfg.battery_capacity,
            max_soc=cfg.max_soc,
        )

    # NOTE: there is deliberately no "is the rate SOC-independent, so hoist it
    # out of the state loop" fast path. There was one, decided from a fixed
    # probe set around the CURRENT temperature, and it was wrong in exactly the
    # way a sampled test of an arbitrary callback is always eventually wrong: a
    # curve that is flat at every probe but tapers at a temperature the REFINED
    # profile reaches had its taper erased, because the hoisted value was
    # `rate(min_soc, slot_temp)` -- the fastest point of the curve -- applied to
    # every state. On the regression in
    # `tests/test_dp_rate_compatibility.py::TestRateIsEvaluatedAtTheTemperature
    # ProfileReaches` that invented 1.875 kWh in a single slot, with
    # `converged=True` and no fallback to hint at it.
    #
    # The probe cannot be repaired by widening it: `charge_rate_predictor` is an
    # arbitrary callable, and the temperatures the refinement will reach are not
    # known when the probe runs. The rate is therefore always evaluated per
    # state, memoized by `(soc, temperature)` globally and by state index within
    # a slot. Cost, measured on the 132-slot horizon at a 1 % step: 140 -> 184 ms
    # with a partial first slot. That is the price of the answer being right.
    #
    # The same reasoning removed the sampled "is this curve temperature
    # sensitive?" probe that used to gate the whole refinement loop. It read the
    # curve at three SOCs on a six-point temperature ladder around the CURRENT
    # temperature; a learned bucket that varies only between those SOCs made it
    # answer "no", and the plan was then built on the idle profile and never
    # checked. See `tests/test_thermal_feasibility_refinement.py`.

    def _min_rate_fns(
        self,
        profiles: List[List[Optional[float]]],
        n_slots: int,
    ) -> List[Callable[[float], float]]:
        """Per slot, ``soc -> min rate over every profile seen in this call``.

        The conservative fallback. It is a genuine lower bound over the
        temperatures it minimises across, and nothing more: a plan built on it
        can still be short at a temperature none of those profiles visited,
        which is why the caller replays it again afterwards.
        """
        fns: List[Callable[[float], float]] = []
        for i in range(n_slots):
            temps: List[Optional[float]] = []
            for profile in profiles:
                if i < len(profile) and profile[i] not in temps:
                    temps.append(profile[i])
            fns.append(self._make_min_rate_fn(tuple(temps)))
        return fns

    def _make_min_rate_fn(self, temps) -> Callable[[float], float]:
        rate_for = self._rate_for

        def _fn(soc: float) -> float:
            return min(rate_for(soc, t) for t in temps)

        return _fn

    def _conservative_solve(
        self,
        *,
        solve,
        profiles_used: List[List[Optional[float]]],
        n_slots: int,
        oscillating: bool,
        slots_sorted_by_time: List[PricePoint],
        slot_fractions: List[float],
        start_energy: float,
        current_temp: Optional[float],
        load_kw: List[float],
        pv_kw: Optional[List[float]],
    ):
        """The fallback, and the degrade branch behind it.

        Returns ``(branch, shortfall_kwh, schedule, energy_trajectory, replay)``.
        """
        rate_fns = self._min_rate_fns(profiles_used, n_slots)
        idle_profile = profiles_used[0]
        schedule, _idx, energy_trajectory, _value = solve(idle_profile, rate_fns)
        replay = self._replay_plan(
            slots_sorted_by_time=slots_sorted_by_time,
            slot_fractions=slot_fractions,
            schedule=schedule,
            energy_trajectory=energy_trajectory,
            start_energy=start_energy,
            current_temp=current_temp,
            load_kw=load_kw,
            pv_kw=pv_kw,
        )
        reason = "oscillated" if oscillating else (
            f"did not settle within {MAX_RATE_REFINEMENT_PASSES} passes"
        )
        shortfall = replay.charge_shortfall_kwh
        if shortfall <= CHARGE_FEASIBILITY_EPS_KWH:
            if self._decision_log_level >= 1:
                self._log(
                    f"Charge-rate refinement {reason}; planned on the minimum "
                    "charge rate over every temperature profile seen this call. "
                    "The result is feasible at the temperatures it reaches."
                )
            return "conservative_fallback", 0.0, schedule, energy_trajectory, replay

        # Still short. DEGRADE. The trajectory is rebuilt from the physical
        # walk by the common code in `optimize` (which does that in every
        # branch); what is specific here is that the plan's ACTIONS were chosen
        # for energy the pack will not have, so this is no longer an economic
        # optimum and it is said out loud.
        self._log(
            f"Charge-rate refinement {reason}, and the conservative solve is "
            f"still {shortfall:.3f} kWh short of the charge energy it credits "
            "at the temperatures the plan reaches. Degrading: the published "
            "trajectory now credits only the energy the replayed temperature "
            "allows. The ACTIONS were chosen for energy the pack will not have, "
            "so this plan is no longer economically optimal — it is only "
            "physically honest.",
            level="WARNING",
        )
        return "degraded", shortfall, schedule, energy_trajectory, replay

    def _idle_temp_profile(
        self,
        slots_sorted_by_time: List[PricePoint],
        slot_fractions: List[float],
        current_temp: Optional[float],
    ) -> List[Optional[float]]:
        """Start-of-slot temperatures with NO battery activity at all.

        This is pass 0 and the conservative fallback: heat can only come from
        actions the plan has actually committed to.
        """
        if current_temp is None:
            return [None] * len(slots_sorted_by_time)
        profile: List[Optional[float]] = []
        temp = current_temp
        for i, slot in enumerate(slots_sorted_by_time):
            profile.append(temp)
            duration = self._config.slot_minutes * slot_fractions[i]
            temp = self._step_temp(temp, slot.time, duration, 0.0, is_charge=False)
        return profile

    def _step_temp(
        self,
        temp: Optional[float],
        slot_time,
        duration_minutes: float,
        battery_power_kw: float,
        is_charge: bool,
    ) -> Optional[float]:
        """One thermal step, shared model first, legacy predictors second."""
        if temp is None:
            return None
        if self._temp_projector is not None:
            return self._temp_projector.project(
                temp, slot_time, duration_minutes, battery_power_kw
            )
        # Legacy two-predictor interface. Warming still follows ACTUAL battery
        # flow: an inverter told to charge a full pack moves no energy, so it
        # gets the idle predictor, not the charging one.
        if is_charge and battery_power_kw > 1e-9:
            return self._predict_temp_after_duration(temp, duration_minutes)
        return self._predict_temp_after_idle(temp, duration_minutes)

    def _replay_plan(
        self,
        *,
        slots_sorted_by_time: List[PricePoint],
        slot_fractions: List[float],
        schedule: Dict[datetime.datetime, ScheduleEntry],
        energy_trajectory: List[float],
        start_energy: float,
        current_temp: Optional[float],
        load_kw: List[float],
        pv_kw: Optional[List[float]],
        follow_plan_energy: bool = True,
    ) -> "_PlanReplay":
        """Walk the SELECTED plan forward at the temperatures it reaches.

        Self-consistent and single-pass: slot i's temperature depends only on
        slots before it, so looking the rate up at the pack's OWN evolving
        temperature needs no iteration. This is the same walk
        ``plan_validation.replay_plan`` and ``project_schedule_trajectory``
        perform, which is why they agree with the planner whenever it converged.

        It answers two questions at once:

        * the temperature PROFILE the plan produces. Warming is driven by
          ``simulate_slot``'s ``battery_power_kw``, i.e. the energy that really
          moved -- a full pack ordered to charge moves nothing and warms
          nothing;
        * the charge SHORTFALL: per slot, how much of the stored energy the
          PLAN credited exceeds what ``rate(soc_start, reached_temp)`` can
          deliver. That covers CHARGE and the PV absorption a HOLD or
          self-consumption DISCHARGE slot performs, because ``simulate_slot``
          caps all three with the same ``charge_input_dc_kw``.

        ``follow_plan_energy`` selects the anchor:

        * ``True`` -- energy advances by the PLANNER's trajectory, so each
          slot's comparison starts from the same SOC and the same headroom the
          planner used. This is the feasibility criterion.
        * ``False`` -- energy advances by what the walk itself achieved, so
          ``energies`` is the physical outcome of the action sequence. This is
          what the degrade branch republishes.
        """
        cfg = self._config
        params = SlotEnergyParams(
            battery_capacity=cfg.battery_capacity,
            efficiency=cfg.efficiency,
            discharge_rate=cfg.discharge_rate,
            export_discharge_rate=cfg.export_discharge_rate,
            inverter_efficiency=cfg.inverter_efficiency,
            min_soc=cfg.min_soc,
            max_soc=cfg.max_soc,
            slot_minutes=cfg.slot_minutes,
        )
        profile: List[Optional[float]] = []
        pairs: List[Tuple[Optional[float], Optional[float]]] = []
        energies: List[float] = []
        shortfall_by_slot: List[float] = []
        limited: List[bool] = []
        total_shortfall = 0.0
        temp = current_temp
        energy = start_energy
        for i, price_point in enumerate(slots_sorted_by_time):
            entry = schedule.get(price_point.time)
            mode = entry.mode if entry is not None else BatteryMode.HOLD
            is_export = bool(
                entry is not None
                and entry.export_rate is not None
                and entry.export_rate > 0
            )
            fraction = slot_fractions[i]
            profile.append(temp)
            soc = energy / cfg.battery_capacity * 100 if cfg.battery_capacity else 0.0
            # The capability the pack really has here: its own SOC, its own
            # temperature. Looking it up at the PLANNING temperature instead is
            # what made this check compare the planner with its own assumption.
            # Same span rule as `_run_dp`, or the feasibility check would be
            # measuring the plan against a different within-slot model.
            rate = self._span_rate(
                self._rate_for, soc, temp, cfg.slot_hours * fraction
            )
            outcome = simulate_slot(
                stored_energy_kwh=energy,
                mode=mode,
                params=params,
                charge_input_dc_kw=rate,
                load_kw=load_kw[i] if i < len(load_kw) else 0.0,
                pv_kw=pv_kw[i] if pv_kw is not None and i < len(pv_kw) else 0.0,
                fraction=fraction,
                is_export=is_export,
            )

            plan_start = (
                start_energy
                if i == 0
                else (energy_trajectory[i - 1] if i - 1 < len(energy_trajectory) else energy)
            )
            plan_end = (
                energy_trajectory[i]
                if i < len(energy_trajectory)
                else outcome.energy_end_kwh
            )
            planned_stored_in = max(0.0, plan_end - plan_start)
            slot_shortfall = max(0.0, planned_stored_in - outcome.stored_dc_in_kwh)
            shortfall_by_slot.append(slot_shortfall)
            total_shortfall += slot_shortfall
            limited.append(outcome.unmet_battery_ac_kwh > ENERGY_EPS)

            start_temp = temp
            temp = self._step_temp(
                temp,
                price_point.time,
                cfg.slot_minutes * fraction,
                outcome.battery_power_kw,
                is_charge=(mode == BatteryMode.CHARGE),
            )
            pairs.append((start_temp, temp))
            energies.append(outcome.energy_end_kwh)
            energy = plan_end if follow_plan_energy else outcome.energy_end_kwh

        return _PlanReplay(
            temp_profile=profile,
            temp_pairs=pairs,
            energies=energies,
            charge_shortfall_kwh=total_shortfall,
            shortfall_by_slot=shortfall_by_slot,
            unmet_slots=limited,
        )

    def _profiles_agree(
        self,
        a: List[Optional[float]],
        b: List[Optional[float]],
        state_socs: Optional[List[float]] = None,
    ) -> bool:
        """Whether two temperature profiles are the same plan.

        Two criteria, either of which stops the refinement:

        1. The temperatures themselves agree within ``TEMP_FIXED_POINT_C``.
        2. They imply the SAME charge capability at every state the last solve
           actually visited. Temperature only reaches the plan through the rate,
           so two profiles that produce identical rates produce identical plans
           -- and a pack that keeps warming inside one temperature bucket would
           otherwise never reach a fixed point at all.

        ``state_socs`` is ``last_visited_state_socs()``: the EXACT
        ``dp_energy[idx]`` of every state a rate was looked up for. It used to
        be the DP's SOC grid, which is where a state's LABEL sits and not where
        its transition is computed -- so a curve whose two profiles differ only
        at an energy strictly between two grid points read as "the same plan" at
        precisely the states the solve visits. A curve that differs only at an
        energy NO state reaches is genuinely the same plan, which is the
        question being asked.
        """
        if len(a) != len(b):
            return False
        for i, (x, y) in enumerate(zip(a, b)):
            if x is None or y is None:
                if x is not y:
                    return False
                continue
            if abs(x - y) <= TEMP_FIXED_POINT_C:
                continue
            if state_socs is None:
                return False
            for soc in state_socs:
                if abs(self._rate_for(soc, x) - self._rate_for(soc, y)) > 1e-9:
                    return False
        return True

    def _build_soc_trajectory(
        self,
        slots_sorted_by_time: List[PricePoint],
        energy_trajectory: List[float],
        start_energy: float,
    ) -> Dict[datetime.datetime, Tuple[float, float]]:
        """SOC trajectory from the chosen path's EXACT energies.

        Derived from ``energy_trajectory``, not from the grid indices: the
        published trajectory must describe the plan's real energy, or it
        reports 15 % for a pack the same plan has already emptied.
        """
        soc_trajectory: Dict[datetime.datetime, Tuple[float, float]] = {}
        cfg = self._config
        capacity = cfg.battery_capacity if cfg.battery_capacity > 0 else 1e-9

        if energy_trajectory and len(energy_trajectory) == len(slots_sorted_by_time):
            for t, price_point in enumerate(slots_sorted_by_time):
                hour = price_point.time
                slot_start_energy = (
                    start_energy if t == 0 else energy_trajectory[t - 1]
                )
                slot_end_energy = energy_trajectory[t]
                soc_trajectory[hour] = (
                    slot_start_energy / capacity * 100,
                    slot_end_energy / capacity * 100,
                )

        return soc_trajectory

    def _run_dp(
        self,
        slots_list: List[PricePoint],
        load_kw_list: List[float],
        temp_profile_list: List[Optional[float]],
        slot_fractions_list: List[float],
        start_energy_kwh: float,
        min_energy: float,
        max_energy: float,
        step_kwh: float,
        n_states: int,
        energy_levels: List[float],
        start_idx_override: Optional[int] = None,
        pv_kw_list: Optional[List[float]] = None,
        rate_fns_list: Optional[List[Callable[[float], float]]] = None,
    ) -> Tuple[List[BatteryMode], List[bool], List[bool], float, List[int], List[float]]:
        """
        Core DP algorithm.

        Returns:
            (actions, partial_flags, export_flags, best_value, idx_trajectory,
             energy_trajectory)

        ``energy_trajectory`` holds the EXACT end-of-slot stored energy of the
        backtracked path, which is what the reported SOC trajectory is built
        from. Deriving it from the grid index instead is what let a published
        trajectory read 15 % while the modelled pack was already empty.
        """
        cfg = self._config
        n_list_slots = len(slots_list)
        if n_list_slots == 0:
            return [], [], [], 0.0, [], []

        neg_inf = -1e18
        tie_val_eps = 1e-6
        tie_tie_eps = 1e-12
        tie_price_weight = 1e-5
        tie_time_weight = 1e-7

        # Discharge cost: only wear cost (battery degradation)
        discharge_cost_per_kwh = cfg.battery_wear_cost

        # Allocate 1D DP buffers and template row for efficient reset
        _neg_inf_row = [neg_inf] * n_states
        _min_energy_row = [min_energy] * n_states
        dp_a = [neg_inf] * n_states
        dp_b = [neg_inf] * n_states
        dp_tie_a = [neg_inf] * n_states
        dp_tie_b = [neg_inf] * n_states
        # Exact stored energy of the best path reaching each bucket.
        en_a = [min_energy] * n_states
        en_b = [min_energy] * n_states

        # The observed starting energy is kept EXACTLY; only its bucket label
        # is floored. `start_idx_override` is accepted for backward
        # compatibility and clamped, but the energy is never re-derived from it.
        start_energy_local = min(max_energy, max(min_energy, start_energy_kwh))
        if start_idx_override is None:
            start_idx_local = _energy_to_index(
                start_energy_local, min_energy, step_kwh, n_states, "floor"
            )
        else:
            start_idx_local = min(max(start_idx_override, 0), n_states - 1)

        # Initialize starting state in dp_a
        dp_a[start_idx_local] = 0.0
        dp_tie_a[start_idx_local] = 0.0
        en_a[start_idx_local] = start_energy_local
        dp, next_dp = dp_a, dp_b
        dp_tie, next_dp_tie = dp_tie_a, dp_tie_b
        dp_energy, next_energy = en_a, en_b

        prev_idx = [None] * n_list_slots
        prev_action = [None] * n_list_slots
        prev_partial = [None] * n_list_slots
        prev_export = [None] * n_list_slots
        # End-of-slot energy per state, kept so backtracking recovers the exact
        # energy of the chosen path rather than its grid label.
        energy_by_slot: List[List[float]] = [None] * n_list_slots

        def _should_update(
            curr_val: float,
            curr_tie: float,
            cand_val: float,
            cand_tie: float,
            curr_energy: float = 0.0,
            cand_energy: float = 0.0,
        ) -> bool:
            if cand_val > curr_val + tie_val_eps:
                return True
            if abs(cand_val - curr_val) <= tie_val_eps:
                if cand_tie > curr_tie + tie_tie_eps:
                    return True
                # Dominance: same value, same tie bias -> keep the path holding
                # more energy. It can only widen the options of every later
                # slot, and it makes the merge deterministic instead of
                # "whichever candidate was evaluated first".
                if (
                    abs(cand_tie - curr_tie) <= tie_tie_eps
                    and cand_energy > curr_energy + 1e-9
                ):
                    return True
            return False

        dp_trace_slots = []

        # Accumulated across every `_run_dp` of one solve, including the one per
        # partial-first-slot candidate. See `_visited_state_socs`.
        visited_socs = self._visited_state_socs

        inv_eff = cfg.inverter_efficiency

        for t in range(n_list_slots):
            price = slots_list[t].price
            # An interval nobody published: the action is FIXED to HOLD and no
            # cash flow is scored for it (see the note above `_is_priced`).
            # Zeroing the two tariffs is exactly that — HOLD's own arithmetic
            # below multiplies both the import and the export term by them —
            # and the CHARGE/DISCHARGE candidates are skipped outright.
            priced = price is not None
            buy_price = self._buy_price(price) if priced else 0.0
            fraction = slot_fractions_list[t]
            slot_load_kw = load_kw_list[t]
            slot_pv_kw = pv_kw_list[t] if pv_kw_list is not None else 0.0
            net_load_kw = max(0.0, slot_load_kw - slot_pv_kw)
            pv_surplus_kw = max(0.0, slot_pv_kw - slot_load_kw)
            # AC load battery can serve via self-consumption (capped by discharge rate)
            discharge_kwh = min(net_load_kw, cfg.discharge_rate) * cfg.slot_hours * fraction
            # DC energy battery must provide (higher due to inverter DC→AC loss)
            dc_discharge_kwh = discharge_kwh / inv_eff
            # The charge rate is NOT a slot constant: it depends on the SOC of
            # the candidate state. Only the temperature is fixed per slot (see
            # the module header), so the rate lookup happens inside the state
            # loop and everything derived from it with it.
            slot_temp = (
                temp_profile_list[t] if t < len(temp_profile_list) else None
            )
            # The conservative fallback replaces the (soc, slot temperature)
            # lookup with "the minimum rate over every profile seen this call".
            slot_rate_fn = (
                rate_fns_list[t]
                if rate_fns_list is not None and t < len(rate_fns_list)
                else None
            )
            slot_hours_fraction = cfg.slot_hours * fraction
            net_load_kwh = net_load_kw * cfg.slot_hours * fraction

            # Export variables — NNS contract: sell price floor at 0
            sell_price = self._sell_price(price) if priced else 0.0
            export_discharge_kwh = cfg.effective_export_discharge_rate * cfg.slot_hours * fraction
            dc_export_discharge_kwh = export_discharge_kwh / inv_eff
            load_kwh = slot_load_kw * cfg.slot_hours * fraction
            pv_kwh = slot_pv_kw * cfg.slot_hours * fraction
            exported_kwh_full = max(0.0, export_discharge_kwh + pv_kwh - load_kwh)

            # Reset next_dp buffers using slice assignment (faster than nested loop)
            next_dp[:] = _neg_inf_row
            next_dp_tie[:] = _neg_inf_row
            next_energy[:] = _min_energy_row
            next_prev_idx = [None] * n_states
            next_prev_action = [None] * n_states
            next_prev_partial = [False] * n_states
            next_prev_export = [False] * n_states

            # HOLD precomputation (the genuinely rate-independent part)
            hold_grid_cost = buy_price * net_load_kwh
            hold_sell = sell_price  # already floored at 0
            rate_for = self._rate_for
            # ``(soc, temp) -> rate`` for THIS slot. The conservative fallback
            # replaces the temperature lookup with "the minimum over every
            # profile seen this call"; the span rule is applied to whichever of
            # the two is in force, so both paths share one within-slot model.
            if slot_rate_fn is not None:
                def slot_rate_lookup(soc, _temp, _fn=slot_rate_fn):
                    return _fn(soc)
            else:
                slot_rate_lookup = rate_for
            span_rate = self._span_rate
            rate_cache_slot = {}

            slot_trace = []
            trace_this_slot = self._decision_log_level >= 3 and fraction > 0.5
            deep_trace_this_slot = self._decision_log_level >= 3 and t < 5

            for idx, val in enumerate(dp):
                if val <= neg_inf / 2:
                    continue
                curr_tie = dp_tie[idx]
                # The path's EXACT energy, not its grid point.
                curr_energy = dp_energy[idx]
                curr_soc = (curr_energy / cfg.battery_capacity) * 100
                headroom = max(0.0, max_energy - curr_energy)
                available = max(0.0, curr_energy - min_energy)

                # Charge capability of THIS state, at THIS slot's temperature,
                # over the SPAN this slot covers (see `_span_rate`).
                slot_charge_rate = rate_cache_slot.get(idx)
                if slot_charge_rate is None:
                    slot_charge_rate = span_rate(
                        slot_rate_lookup, curr_soc, slot_temp, slot_hours_fraction
                    )
                    rate_cache_slot[idx] = slot_charge_rate
                    visited_socs.add(round(curr_soc, 6))
                charge_dc_kwh = slot_charge_rate * slot_hours_fraction
                charge_energy_kwh = charge_dc_kwh * cfg.efficiency
                # PV surplus charges battery for free (pv_priority, DC→DC)
                pv_free_charge_kwh = (
                    min(pv_surplus_kw, slot_charge_rate) * slot_hours_fraction
                )
                hold_pv_charge_max = pv_free_charge_kwh * cfg.efficiency
                hold_excess_pv_kwh = (
                    max(0.0, pv_surplus_kw - slot_charge_rate) * slot_hours_fraction
                )

                # HOLD - no grid charge; PV covers load, surplus charges battery for free
                hold_updated = False
                if hold_pv_charge_max > 0:
                    actual_stored = min(hold_pv_charge_max, headroom)
                    new_energy = curr_energy + actual_stored
                    # PV that couldn't be stored (battery full) + excess beyond charge rate → export
                    unused_kwh = (hold_pv_charge_max - actual_stored) / cfg.efficiency
                    export_revenue = hold_sell * (hold_excess_pv_kwh + unused_kwh)
                    hold_next_idx = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "floor")
                else:
                    new_energy = curr_energy
                    export_revenue = hold_sell * hold_excess_pv_kwh
                    hold_next_idx = idx
                hold_val = val - hold_grid_cost + export_revenue
                if _should_update(
                    next_dp[hold_next_idx], next_dp_tie[hold_next_idx], hold_val, curr_tie,
                    next_energy[hold_next_idx], new_energy,
                ):
                    next_dp[hold_next_idx] = hold_val
                    next_dp_tie[hold_next_idx] = curr_tie
                    next_energy[hold_next_idx] = new_energy
                    next_prev_idx[hold_next_idx] = idx
                    next_prev_action[hold_next_idx] = BatteryMode.HOLD
                    next_prev_partial[hold_next_idx] = False
                    next_prev_export[hold_next_idx] = False
                    hold_updated = True

                discharge_attempted = False
                discharge_updated = False
                discharge_blocked_reason = None
                discharge_next_idx = None
                discharge_next_val = None

                # CHARGE (grid_charge): only when grid is actually needed to supplement PV
                # When PV surplus fully covers charge rate, HOLD already handles it
                if priced and charge_energy_kwh > 0 and pv_free_charge_kwh < charge_dc_kwh - 1e-6:
                    # Headroom truncates the stored energy; the terminal energy
                    # and the grid share follow from what was actually stored.
                    actual_charge_energy = min(charge_energy_kwh, headroom)
                    actual_charge_dc = actual_charge_energy / cfg.efficiency
                    grid_charge_dc = max(0.0, actual_charge_dc - pv_free_charge_kwh)
                    # Skip CHARGE when PV fully covers it — HOLD already handles PV charging
                    if actual_charge_energy > 1e-9 and grid_charge_dc >= 1e-6:
                        new_energy = curr_energy + actual_charge_energy
                        next_idx = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "floor")
                        # AC energy from grid = DC / inverter_efficiency
                        grid_charge_ac = grid_charge_dc / inv_eff
                        next_val = val - (buy_price * grid_charge_ac) - (buy_price * net_load_kwh)
                        charge_tie_bias = (-price * tie_price_weight) + (t * tie_time_weight)
                        next_tie = curr_tie + charge_tie_bias
                        if _should_update(
                            next_dp[next_idx], next_dp_tie[next_idx], next_val, next_tie,
                            next_energy[next_idx], new_energy,
                        ):
                            next_dp[next_idx] = next_val
                            next_dp_tie[next_idx] = next_tie
                            next_energy[next_idx] = new_energy
                            next_prev_idx[next_idx] = idx
                            next_prev_action[next_idx] = BatteryMode.CHARGE
                            next_prev_partial[next_idx] = False
                            next_prev_export[next_idx] = False

                # DISCHARGE (self-consumption)
                # Battery provides DC energy; inverter converts to AC to serve load.
                # dc_discharge_kwh = AC load / inverter_eff (battery works harder).
                if priced and dc_discharge_kwh > 0:
                    discharge_attempted = True
                    if available > 1e-9:
                        # A candidate delivers what the pack HAS. When that is
                        # less than the load asked for, the grid pays for the
                        # rest — the plan must not be scored as if the battery
                        # had covered it. No threshold decides whether the slot
                        # "counts": the energy does.
                        actual_dc_kwh = min(dc_discharge_kwh, available)
                        is_partial = actual_dc_kwh < dc_discharge_kwh - 1e-9
                        ac_served = actual_dc_kwh * inv_eff
                        grid_import_kwh = max(0.0, net_load_kwh - ac_served)
                        new_energy = curr_energy - actual_dc_kwh
                        next_idx = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "floor")
                        next_val = val - (buy_price * grid_import_kwh) - (discharge_cost_per_kwh * actual_dc_kwh)
                    else:
                        discharge_blocked_reason = (
                            f"at_min_soc ({curr_energy:.3f} <= {min_energy:.3f})"
                        )
                        next_val = None
                        next_idx = None

                    if next_val is not None:
                        discharge_next_idx = next_idx
                        discharge_next_val = next_val
                        if _should_update(
                            next_dp[next_idx], next_dp_tie[next_idx], next_val, curr_tie,
                            next_energy[next_idx], new_energy,
                        ):
                            next_dp[next_idx] = next_val
                            next_dp_tie[next_idx] = curr_tie
                            next_energy[next_idx] = new_energy
                            next_prev_idx[next_idx] = idx
                            next_prev_action[next_idx] = BatteryMode.DISCHARGE
                            next_prev_partial[next_idx] = is_partial
                            next_prev_export[next_idx] = False
                            discharge_updated = True
                        else:
                            discharge_blocked_reason = (
                                f"existing_val={next_dp[next_idx]:.4f} >= discharge_val={next_val:.4f}"
                            )

                # DISCHARGE_EXPORT (full rate discharge with grid export)
                # SOC transition uses DC energy; export revenue uses AC output.
                if priced and sell_price > 0 and exported_kwh_full > 0 and available > 1e-9:
                    actual_dc_export = min(dc_export_discharge_kwh, available)
                    is_partial_export = actual_dc_export < dc_export_discharge_kwh - 1e-9
                    ac_from_battery = actual_dc_export * inv_eff
                    actual_exported = max(0.0, ac_from_battery + pv_kwh - load_kwh)
                    remaining_load = max(0.0, load_kwh - pv_kwh - ac_from_battery)
                    new_energy = curr_energy - actual_dc_export
                    next_idx = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "floor")
                    next_val = (
                        val
                        + sell_price * actual_exported
                        - buy_price * remaining_load
                        - discharge_cost_per_kwh * actual_dc_export
                    )

                    if _should_update(
                        next_dp[next_idx], next_dp_tie[next_idx], next_val, curr_tie,
                        next_energy[next_idx], new_energy,
                    ):
                        next_dp[next_idx] = next_val
                        next_dp_tie[next_idx] = curr_tie
                        next_energy[next_idx] = new_energy
                        next_prev_idx[next_idx] = idx
                        next_prev_action[next_idx] = BatteryMode.DISCHARGE
                        next_prev_partial[next_idx] = is_partial_export
                        next_prev_export[next_idx] = True

                # Trace collection for logging
                if trace_this_slot and discharge_attempted:
                    next_soc_discharge = cfg.min_soc + (discharge_next_idx * step_kwh / cfg.battery_capacity) * 100 if discharge_next_idx is not None else 0
                    slot_trace.append({
                        "from_soc": curr_soc,
                        "from_idx": idx,
                        "from_val": val,
                        "hold_val": hold_val,
                        "hold_cost": hold_grid_cost,
                        "hold_updated": hold_updated,
                        "discharge_attempted": discharge_attempted,
                        "discharge_updated": discharge_updated,
                        "discharge_blocked": discharge_blocked_reason,
                        "discharge_to_soc": next_soc_discharge,
                        "discharge_to_idx": discharge_next_idx,
                        "discharge_val": discharge_next_val,
                    })

            if trace_this_slot and slot_trace:
                dp_trace_slots.append((slots_list[t].time, price, slot_trace))

            if deep_trace_this_slot:
                price_label = f"{price:.4f}" if priced else "unpriced"
                self._log(f"[DeepTrace] After slot {t} ({slots_list[t].time.strftime('%H:%M')} @ {price_label}):")
                active_states = [
                    (i, next_dp[i], next_prev_action[i])
                    for i in range(n_states)
                    if next_dp[i] > neg_inf / 2
                ]
                if active_states:
                    active_states.sort(key=lambda x: x[1], reverse=True)
                    top_states = active_states[:3]
                    self._log("  " + ", ".join(
                        f"idx={i} ({cfg.min_soc + i*step_kwh/cfg.battery_capacity*100:.1f}%) val={v:.4f} via {a.name if a else 'None'}"
                        for i, v, a in top_states
                    ))

            # Swap buffers instead of reassigning
            dp, next_dp = next_dp, dp
            dp_tie, next_dp_tie = next_dp_tie, dp_tie
            dp_energy, next_energy = next_energy, dp_energy
            energy_by_slot[t] = list(dp_energy)
            prev_idx[t] = next_prev_idx
            prev_action[t] = next_prev_action
            prev_partial[t] = next_prev_partial
            prev_export[t] = next_prev_export

        # Find best final state. Without a terminal value a finite-horizon
        # optimizer treats all remaining energy as worthless and drains the
        # battery at the forecast boundary.
        terminal_value_per_kwh = self._derive_terminal_rate(slots_list)

        best_val = neg_inf
        best_tie = neg_inf
        best_idx = None

        best_energy = min_energy
        for i in range(n_states):
            if dp[i] > neg_inf / 2:
                # Salvage the path's exact remaining energy, not its grid point.
                terminal_value = terminal_value_per_kwh * max(0.0, dp_energy[i] - min_energy)
                candidate_value = dp[i] + terminal_value
                if _should_update(
                    best_val, best_tie, candidate_value, dp_tie[i],
                    best_energy, dp_energy[i],
                ):
                    best_val = candidate_value
                    best_tie = dp_tie[i]
                    best_energy = dp_energy[i]
                    best_idx = i

        # Backtrack to extract actions
        actions: List[BatteryMode] = []
        partial_flags: List[bool] = []
        export_flags: List[bool] = []
        idx_trajectory: List[int] = []
        energy_trajectory: List[float] = []
        idx = best_idx if best_idx is not None else start_idx_local

        if self._decision_log_level >= 3:
            self._log(f"[DeepTrace] Backtracking from best final state: idx={idx}, val={best_val:.4f}")

        backtrack_trace = []
        for t in range(n_list_slots - 1, -1, -1):
            action = prev_action[t][idx] or BatteryMode.HOLD
            is_partial = prev_partial[t][idx] if action == BatteryMode.DISCHARGE else False
            is_export = prev_export[t][idx] if action == BatteryMode.DISCHARGE else False
            actions.append(action)
            partial_flags.append(is_partial)
            export_flags.append(is_export)
            idx_trajectory.append(idx)
            slot_energies = energy_by_slot[t]
            energy_trajectory.append(
                slot_energies[idx] if slot_energies is not None else min_energy
            )
            prev_i = prev_idx[t][idx]

            if t < 5 and self._decision_log_level >= 3:
                soc_at_t = cfg.min_soc + (idx * step_kwh / cfg.battery_capacity) * 100
                backtrack_trace.append(f"t={t} ({slots_list[t].time.strftime('%H:%M')}): action={action.name}, idx={idx} ({soc_at_t:.1f}%)->prev_i={prev_i}")

            if prev_i is None:
                idx = idx
            else:
                idx = prev_i

        actions.reverse()
        partial_flags.reverse()
        export_flags.reverse()
        idx_trajectory.reverse()
        energy_trajectory.reverse()

        if backtrack_trace and self._decision_log_level >= 3:
            self._log("[DeepTrace] Backtrack path (first 5 slots):")
            for line in reversed(backtrack_trace):
                self._log(f"  {line}")

        # Log DP trace
        if dp_trace_slots and self._decision_log_level >= 3:
            self._log("=" * 70)
            self._log("DP TRACE: Detailed state transitions for discharge-allowed slots")
            self._log("=" * 70)
            for slot_hour, slot_price, traces in dp_trace_slots:
                slot_idx = next((i for i, h in enumerate(slots_list) if h.time == slot_hour), -1)
                chosen_action = actions[slot_idx] if 0 <= slot_idx < len(actions) else None
                price_label = (
                    f"{slot_price:.4f} EUR/kWh" if slot_price is not None
                    else "no published price"
                )
                self._log(f"\n{slot_hour.strftime('%Y-%m-%d %H:%M')} @ {price_label} -> {chosen_action.name if chosen_action else '?'}")

                relevant_traces = [t for t in traces if t["from_val"] > -1e10]
                if relevant_traces:
                    relevant_traces.sort(key=lambda x: x["from_soc"], reverse=True)
                    for trace in relevant_traces[:5]:
                        status = ""
                        if trace["discharge_updated"]:
                            status = "[OK] DISCHARGE wins"
                        elif trace["discharge_blocked"]:
                            status = f"[X] blocked: {trace['discharge_blocked']}"
                        elif trace["hold_updated"]:
                            status = "-> HOLD set (no discharge attempted)"

                        delta = (trace['discharge_val'] - trace['hold_val']) if trace['discharge_val'] is not None else None
                        delta_str = f"+{delta:.4f}" if delta is not None and delta >= 0 else (f"{delta:.4f}" if delta is not None else "N/A")
                        discharge_val_str = f"{trace['discharge_val']:.4f}" if trace['discharge_val'] is not None else "N/A"
                        self._log(
                            f"  SOC {trace['from_soc']:.1f}% (idx={trace['from_idx']}): "
                            f"hold={trace['hold_val']:.4f} vs discharge={discharge_val_str} (delta={delta_str}) -> {status}"
                        )
            self._log("=" * 70)

        return (
            actions,
            partial_flags,
            export_flags,
            best_val,
            idx_trajectory,
            energy_trajectory,
        )

    def _build_schedule(
        self,
        slots_sorted_by_time: List[PricePoint],
        load_kw: List[float],
        temp_profile: List[Optional[float]],
        slot_fractions: List[float],
        current_slot_index: Optional[int],
        start_energy: float,
        min_energy: float,
        max_energy: float,
        step_kwh: float,
        n_states: int,
        energy_levels: List[float],
        pv_kw: Optional[List[float]] = None,
        rate_fns: Optional[List[Callable[[float], float]]] = None,
    ) -> Tuple[Dict[datetime.datetime, ScheduleEntry], List[int], List[float], float]:
        """
        Build schedule using DP with greedy lookahead for partial first slot.

        Returns:
            (schedule, idx_trajectory, energy_trajectory, best_value)
        """
        cfg = self._config
        neg_inf = -1e18

        # One solve: the exact state SOCs this solve visits start empty and
        # accumulate across the `_run_dp` call(s) below, including the one per
        # partial-first-slot candidate.
        self._visited_state_socs = set()

        # Discharge cost
        discharge_cost_per_kwh = cfg.battery_wear_cost

        schedule_local: Dict[datetime.datetime, ScheduleEntry] = {}
        partial_index = current_slot_index
        partial_fraction = slot_fractions[partial_index] if partial_index is not None else 1.0
        has_partial = partial_index is not None and partial_fraction < 0.999

        inv_eff = cfg.inverter_efficiency

        if has_partial:
            price_point = slots_sorted_by_time[partial_index]
            price = price_point.price
            # Same rule as `_run_dp`: an unpriced interval offers only HOLD and
            # scores no cash flow. The orchestrator normally resolves an
            # unpriced CURRENT interval before the solve, so this is the guard
            # that keeps the two mechanisms from contradicting each other if it
            # ever hands one through.
            partial_priced = price is not None
            buy_price = self._buy_price(price) if partial_priced else 0.0
            fraction = slot_fractions[partial_index]
            slot_load_kw = load_kw[partial_index]
            slot_pv_kw = pv_kw[partial_index] if pv_kw is not None else 0.0
            net_load_kw = max(0.0, slot_load_kw - slot_pv_kw)
            pv_surplus_kw = max(0.0, slot_pv_kw - slot_load_kw)
            discharge_kwh = min(net_load_kw, cfg.discharge_rate) * cfg.slot_hours * fraction
            dc_discharge_kwh = discharge_kwh / inv_eff
            # Same contract as the DP transition: the rate is evaluated at the
            # SOC this partial slot actually starts from, and at this slot's
            # temperature — not read out of a time-indexed array.
            partial_soc = (
                start_energy / cfg.battery_capacity * 100
                if cfg.battery_capacity else 0.0
            )
            partial_temp = (
                temp_profile[partial_index]
                if partial_index < len(temp_profile)
                else None
            )
            if rate_fns is not None and partial_index < len(rate_fns):
                _partial_fn = rate_fns[partial_index]

                def partial_rate_lookup(soc, _temp, _fn=_partial_fn):
                    return _fn(soc)
            else:
                partial_rate_lookup = self._rate_for
            slot_charge_rate = self._span_rate(
                partial_rate_lookup,
                partial_soc,
                partial_temp,
                cfg.slot_hours * fraction,
            )
            self._visited_state_socs.add(round(partial_soc, 6))
            charge_energy_kwh = slot_charge_rate * cfg.efficiency * cfg.slot_hours * fraction
            charge_dc_kwh = slot_charge_rate * cfg.slot_hours * fraction
            pv_free_charge_kwh = min(pv_surplus_kw, slot_charge_rate) * cfg.slot_hours * fraction
            net_load_kwh = net_load_kw * cfg.slot_hours * fraction

            # Remaining slots for DP
            remaining_slice = slice(partial_index + 1, None)
            slots_remaining = slots_sorted_by_time[remaining_slice]
            load_remaining = load_kw[remaining_slice]
            temp_profile_remaining = temp_profile[remaining_slice]
            rate_fns_remaining = (
                rate_fns[remaining_slice] if rate_fns is not None else None
            )
            slot_fractions_remaining = slot_fractions[remaining_slice]
            pv_kw_remaining = pv_kw[remaining_slice] if pv_kw is not None else None

            # Candidates: (action, new_energy, immediate_val, is_partial, is_export)
            #
            # The observed starting energy is used EXACTLY here — this is the
            # one explicitly continuous transition — and every candidate's
            # result is mapped conservatively (floored label, exact energy) by
            # `_run_dp` when the remaining horizon continues from it.
            candidates = []
            headroom = max(0.0, max_energy - start_energy)
            available = max(0.0, start_energy - min_energy)

            # HOLD — no grid charge; PV covers load, surplus charges battery for free
            sell_price = self._sell_price(price) if partial_priced else 0.0
            hold_pv_charge_kw = min(pv_surplus_kw, slot_charge_rate)
            hold_pv_energy = hold_pv_charge_kw * cfg.efficiency * cfg.slot_hours * fraction
            hold_actual_stored = min(hold_pv_energy, headroom)
            hold_new_energy = start_energy + hold_actual_stored
            # PV that couldn't be stored + excess beyond charge rate → export
            hold_unused_kwh = (hold_pv_energy - hold_actual_stored) / cfg.efficiency
            hold_excess_pv = max(0.0, pv_surplus_kw - slot_charge_rate) * cfg.slot_hours * fraction
            hold_export_revenue = sell_price * (hold_excess_pv + hold_unused_kwh)
            hold_grid_cost = buy_price * net_load_kwh
            candidates.append((
                BatteryMode.HOLD,
                hold_new_energy,
                -hold_grid_cost + hold_export_revenue,
                False,
                False,
            ))

            # CHARGE (grid_charge): only when grid is actually needed to supplement PV
            if partial_priced and charge_energy_kwh > 0 and pv_free_charge_kwh < charge_dc_kwh - 1e-6:
                actual_charge_energy = min(charge_energy_kwh, headroom)
                actual_charge_dc = actual_charge_energy / cfg.efficiency
                grid_charge_dc = max(0.0, actual_charge_dc - pv_free_charge_kwh)
                if actual_charge_energy > 1e-9 and grid_charge_dc >= 1e-6:
                    new_energy = start_energy + actual_charge_energy
                    grid_charge_ac = grid_charge_dc / inv_eff
                    charge_immediate_cost = -buy_price * grid_charge_ac - buy_price * net_load_kwh
                    candidates.append((
                        BatteryMode.CHARGE,
                        new_energy,
                        charge_immediate_cost,
                        False,
                        False,
                    ))

            # DISCHARGE (self-consumption)
            if partial_priced and dc_discharge_kwh > 0 and available > 1e-9:
                actual_dc_kwh = min(dc_discharge_kwh, available)
                is_partial_candidate = actual_dc_kwh < dc_discharge_kwh - 1e-9
                ac_served = actual_dc_kwh * inv_eff
                grid_import_kwh = max(0.0, net_load_kwh - ac_served)
                discharge_cost = -buy_price * grid_import_kwh - discharge_cost_per_kwh * actual_dc_kwh
                candidates.append((
                    BatteryMode.DISCHARGE,
                    start_energy - actual_dc_kwh,
                    discharge_cost,
                    is_partial_candidate,
                    False,
                ))

            # DISCHARGE_EXPORT (full rate with grid selling, PV adds to export)
            export_discharge_kwh = cfg.effective_export_discharge_rate * cfg.slot_hours * fraction
            dc_export_discharge_kwh = export_discharge_kwh / inv_eff
            load_kwh_slot = slot_load_kw * cfg.slot_hours * fraction
            pv_kwh_slot = slot_pv_kw * cfg.slot_hours * fraction
            exported_kwh = max(0.0, export_discharge_kwh + pv_kwh_slot - load_kwh_slot)

            if partial_priced and sell_price > 0 and exported_kwh > 0 and available > 1e-9:
                actual_dc_export = min(dc_export_discharge_kwh, available)
                is_partial_export_candidate = (
                    actual_dc_export < dc_export_discharge_kwh - 1e-9
                )
                ac_from_battery = actual_dc_export * inv_eff
                actual_exported = max(0.0, ac_from_battery + pv_kwh_slot - load_kwh_slot)
                remaining_load = max(0.0, load_kwh_slot - pv_kwh_slot - ac_from_battery)
                export_value = (
                    sell_price * actual_exported
                    - buy_price * remaining_load
                    - discharge_cost_per_kwh * actual_dc_export
                )
                candidates.append((
                    BatteryMode.DISCHARGE,
                    start_energy - actual_dc_export,
                    export_value,
                    is_partial_export_candidate,
                    True,
                ))

            best_action = BatteryMode.HOLD
            best_is_partial = False
            best_is_export = False
            best_actions_remaining: List[BatteryMode] = []
            best_partial_flags_remaining: List[bool] = []
            best_export_flags_remaining: List[bool] = []
            best_idx_trajectory_remaining: List[int] = []
            best_energy_trajectory_remaining: List[float] = []
            best_first_slot_end_energy: float = start_energy
            best_value = neg_inf

            if self._decision_log_level >= 3:
                self._log(f"[GreedyLookahead] Partial slot {price_point.time.strftime('%H:%M')} @ {price:.4f}")
                self._log(f"  Candidates: {[(c[0].name + ('[EXP]' if c[4] else ''), c[2]) for c in candidates]}")

            greedy_results = []
            for action, new_energy, immediate_val, is_partial, is_export in candidates:
                (
                    actions_remaining,
                    partial_flags_remaining,
                    export_flags_remaining,
                    future_val,
                    idx_traj_remaining,
                    energy_traj_remaining,
                ) = self._run_dp(
                    slots_remaining,
                    load_remaining,
                    temp_profile_remaining,
                    slot_fractions_remaining,
                    new_energy,
                    min_energy,
                    max_energy,
                    step_kwh,
                    n_states,
                    energy_levels,
                    pv_kw_list=pv_kw_remaining,
                    rate_fns_list=rate_fns_remaining,
                )
                if not slots_remaining:
                    terminal_rate = cfg.terminal_energy_value_eur_kwh
                    if terminal_rate is None:
                        # Auto mode with a one-slot horizon: the only price
                        # there is. Without one there is nothing to derive a
                        # salvage value from, and 0 is what `_derive_terminal_
                        # rate` answers in that case too.
                        terminal_rate = max(
                            0.0,
                            buy_price * cfg.inverter_efficiency - cfg.battery_wear_cost,
                        ) if partial_priced else 0.0
                    future_val = terminal_rate * max(0.0, new_energy - min_energy)
                total_val = immediate_val + future_val
                label = action.name + ("[EXP]" if is_export else "")
                greedy_results.append((label, immediate_val, future_val, total_val))
                if total_val > best_value:
                    best_value = total_val
                    best_action = action
                    best_is_partial = is_partial
                    best_is_export = is_export
                    best_actions_remaining = actions_remaining
                    best_partial_flags_remaining = partial_flags_remaining
                    best_export_flags_remaining = export_flags_remaining
                    best_idx_trajectory_remaining = idx_traj_remaining
                    best_energy_trajectory_remaining = energy_traj_remaining
                    best_first_slot_end_energy = new_energy

            if self._decision_log_level >= 3:
                for name, imm, fut, tot in greedy_results:
                    self._log(f"  {name}: immediate={imm:.4f}, future={fut:.4f}, total={tot:.4f}")
                self._log(f"  -> Best: {best_action.name}{'[EXP]' if best_is_export else ''} (val={best_value:.4f})")

                hold_result = next((r for r in greedy_results if r[0] == "HOLD"), None)
                discharge_result = next((r for r in greedy_results if r[0] == "DISCHARGE"), None)
                if hold_result and discharge_result:
                    hold_imm, hold_fut, hold_tot = hold_result[1], hold_result[2], hold_result[3]
                    disc_imm, disc_fut, disc_tot = discharge_result[1], discharge_result[2], discharge_result[3]
                    saved_by_discharge = disc_imm - hold_imm
                    extra_charge_cost = disc_fut - hold_fut
                    net_benefit = disc_tot - hold_tot

                    if net_benefit > 0.001:
                        self._log(f"  [DECISION] DISCHARGE wins: saves {saved_by_discharge:.4f} now, extra charge cost {-extra_charge_cost:.4f}, net benefit {net_benefit:.4f}")
                    elif net_benefit < -0.001:
                        self._log(f"  [DECISION] HOLD wins: would save {saved_by_discharge:.4f} by discharging, but recharging costs {-extra_charge_cost:.4f} extra")
                        if discharge_kwh > 0.01:
                            effective_recharge_cost_per_kwh = -extra_charge_cost / (discharge_kwh / cfg.efficiency)
                            self._log(f"             Overnight recharge cost: ~{effective_recharge_cost_per_kwh:.4f}/kWh vs discharge value {buy_price:.4f}/kWh")
                    else:
                        self._log(f"  [DECISION] Tie (within 0.001): HOLD preferred by default")

            actions = [best_action] + best_actions_remaining
            partial_flags = [best_is_partial] + best_partial_flags_remaining
            export_flags = [best_is_export] + best_export_flags_remaining
            best_first_slot_end_idx = _energy_to_index(
                best_first_slot_end_energy, min_energy, step_kwh, n_states, "floor"
            )
            idx_trajectory = [best_first_slot_end_idx] + best_idx_trajectory_remaining
            energy_trajectory = (
                [best_first_slot_end_energy] + best_energy_trajectory_remaining
            )
        else:
            (
                actions,
                partial_flags,
                export_flags,
                best_value,
                idx_trajectory,
                energy_trajectory,
            ) = self._run_dp(
                slots_sorted_by_time,
                load_kw,
                temp_profile,
                slot_fractions,
                start_energy,
                min_energy,
                max_energy,
                step_kwh,
                n_states,
                energy_levels,
                pv_kw_list=pv_kw,
                rate_fns_list=rate_fns,
            )

        terminal_rate = self._derive_terminal_rate(slots_sorted_by_time)
        for t, (price_point, action, lk, is_partial, is_export) in enumerate(zip(
            slots_sorted_by_time, actions, load_kw, partial_flags, export_flags
        )):
            hour = price_point.time
            price = price_point.price
            pv = pv_kw[t] if pv_kw is not None else 0.0
            if price is None:
                # A forced HOLD across an interval nobody published. It is an
                # ENTRY, not an absence: the plan advanced the pack across it
                # (absorbing PV), so the SOC trajectory, the final replay, the
                # mode census and the projected-cost column all have to see it.
                #
                # No provenance and no marginal value, on purpose. The reason
                # is the one `execute_scheduled_mode` already recognises, so
                # when this slot becomes current it applies HOLD, reports
                # `current_slot_entry: fallback` and keeps the price retry
                # armed - and the execution guard refuses to send it as
                # anything else, because it can show no published price.
                schedule_local[hour] = ScheduleEntry(
                    time=hour,
                    mode=BatteryMode.HOLD,
                    reason=NO_PRICE_REASON,
                    marginal_value_eur_kwh=None,
                    value_basis=None,
                    energy_limited=False,
                )
                continue
            reason = f"{price:.4f} EUR/kWh load~{lk:.2f}kW"
            if pv > 0:
                reason += f" pv~{pv:.2f}kW"
            if is_partial:
                reason += DEPLETION_NOTE
            if is_export:
                reason += " [EXPORT]"
            if action == BatteryMode.HOLD:
                # Annotate the two non-obvious HOLD causes so the schedule log
                # is self-explanatory (net load is what discharge serves).
                if pv > 0 and pv >= lk:
                    reason += " [pv>=load]"
                elif lk - pv > 0:
                    keep_value = (
                        self._buy_price(price)
                        * cfg.inverter_efficiency - cfg.battery_wear_cost
                    )
                    if keep_value < terminal_rate:
                        reason += (
                            f" [keep: {keep_value:.3f}<terminal {terminal_rate:.3f}]"
                        )
            marginal_value, value_basis = self._marginal_slot_value(
                action, price, bool(is_export), terminal_rate
            )
            entry = ScheduleEntry(
                time=hour,
                mode=action,
                reason=reason,
                marginal_value_eur_kwh=marginal_value,
                value_basis=value_basis,
                # The plan's own declaration that this slot runs the pack dry
                # and that the grid was priced for the remainder. `plan_validation`
                # checks the continuous replay against it.
                energy_limited=bool(is_partial),
            )
            if action == BatteryMode.DISCHARGE:
                entry.export_rate = 100 if is_export else 0
            elif action == BatteryMode.CHARGE and pv > 0:
                entry.ac_charge_mode = "pv_priority"
            schedule_local[hour] = entry

        return schedule_local, idx_trajectory, energy_trajectory, best_value
