"""
Dynamic Programming optimizer for battery scheduling.

Extracts the optimal charge/hold/discharge schedule using SOC-aware
dynamic programming with temperature-aware charge rate predictions.
"""

import datetime
import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import BatteryOptimizerConfig

from .models import BatteryMode, PricePoint, ScheduleEntry
from .charge_rate_utils import compute_charge_rates_per_slot
from .slot_energy import SlotEnergyParams, simulate_slot
from .thermal_model import battery_power_for_entry
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
#   stop    on a fixed point (profiles agree within TEMP_FIXED_POINT_C), on a
#           repeated profile (oscillation), or on the pass budget.
#
# On oscillation or exhaustion it falls back to the pass-0 idle profile and
# solves once more. Limits of that fallback, stated rather than implied:
#
#   * It is CONSERVATIVE only where the rate is non-decreasing in temperature --
#     the physical case, and the one the learning engine's temperature buckets
#     describe (a cold pack charges slower). With a non-monotonic learned curve
#     it is an approximation, not a bound, and what catches that is the final
#     replay in `plan_validation`, not this loop.
#   * A fixed point is not a proof of global optimality. The solver is exact for
#     its discretized model GIVEN a temperature profile; the profile itself is
#     an outer approximation.
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
# floating-point precision. What remains approximate is the MERGING: two paths
# in one bucket differ by up to one step, and only the better-valued one
# survives (ties are broken toward the one with more energy, which is a genuine
# dominance rule). That is an approximation of the optimum, bounded by one step
# times the marginal value of a kWh -- not a licence to invent energy.


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
    # Start-of-slot temperature the plan was BUILT with, per slot. The final
    # replay must look rates up at these, or a conservative fallback profile
    # would read as a trajectory disagreement.
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
        Run DP optimization to find optimal schedule.

        Args:
            prices: List of price points (must include current_slot)
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
        idle_profile = list(temp_profile)
        refine = self._rate_is_temperature_sensitive(current_temp)
        state_socs = (
            [e / cfg.battery_capacity * 100 for e in energy_levels]
            if cfg.battery_capacity else []
        )
        seen_profiles: List[List[Optional[float]]] = []
        passes = 0
        converged = True
        fallback_used = False

        while True:
            passes += 1
            schedule, idx_trajectory, energy_trajectory, best_value = self._build_schedule(
                slots_sorted_by_time=slots_sorted_by_time,
                load_kw=load_kw,
                temp_profile=temp_profile,
                slot_fractions=slot_fractions,
                current_slot_index=current_slot_index,
                start_energy=start_energy,
                min_energy=min_energy,
                max_energy=max_energy,
                step_kwh=step_kwh,
                n_states=n_states,
                energy_levels=energy_levels,
                pv_kw=pv_kw,
            )
            if not refine:
                break

            replayed_profile, temp_pairs = self._replay_plan_temps(
                slots_sorted_by_time,
                slot_fractions,
                schedule,
                energy_trajectory,
                start_energy,
                current_temp,
                load_kw,
                pv_kw,
                temp_profile,
            )
            if self._profiles_agree(replayed_profile, temp_profile, state_socs):
                temp_profile = replayed_profile
                break
            if any(self._profiles_agree(replayed_profile, p, state_socs) for p in seen_profiles) \
                    or passes >= MAX_RATE_REFINEMENT_PASSES:
                # Oscillating, or out of budget. Plan on the profile that
                # assumes no heat from any action the plan has not committed
                # to, and stop.
                converged = False
                fallback_used = True
                temp_profile = idle_profile
                passes += 1
                schedule, idx_trajectory, energy_trajectory, best_value = self._build_schedule(
                    slots_sorted_by_time=slots_sorted_by_time,
                    load_kw=load_kw,
                    temp_profile=temp_profile,
                    slot_fractions=slot_fractions,
                    current_slot_index=current_slot_index,
                    start_energy=start_energy,
                    min_energy=min_energy,
                    max_energy=max_energy,
                    step_kwh=step_kwh,
                    n_states=n_states,
                    energy_levels=energy_levels,
                    pv_kw=pv_kw,
                )
                break
            seen_profiles.append(list(temp_profile))
            temp_profile = replayed_profile

        if fallback_used and self._decision_log_level >= 1:
            self._log(
                "Charge-rate refinement did not settle within "
                f"{MAX_RATE_REFINEMENT_PASSES} passes; planning on the idle "
                "temperature profile (no heat credited to actions the plan has "
                "not committed to)."
            )

        # Build SOC trajectory
        soc_trajectory = self._build_soc_trajectory(
            slots_sorted_by_time, energy_trajectory, start_energy
        )

        # Build temperature trajectory from the FINAL plan, through the same
        # replay the refinement uses — one thermal implementation, not two.
        _final_profile, temp_pairs = self._replay_plan_temps(
            slots_sorted_by_time,
            slot_fractions,
            schedule,
            energy_trajectory,
            start_energy,
            current_temp,
            load_kw,
            pv_kw,
            temp_profile,
        )
        temp_trajectory = (
            {}
            if current_temp is None
            else {
                p.time: temp_pairs[i]
                for i, p in enumerate(slots_sorted_by_time)
                if i < len(temp_pairs)
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
            rate_refinement_converged=converged,
            rate_refinement_fallback=fallback_used,
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
        """
        cfg = self._config
        if cfg.terminal_energy_value_eur_kwh is not None:
            return cfg.terminal_energy_value_eur_kwh
        median_buy = statistics.median(
            (p.price + cfg.grid_fee) * cfg.import_price_multiplier
            for p in slots_list
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

    def _probe_temps(self, current_temp: Optional[float]) -> List[Optional[float]]:
        if current_temp is None:
            return [None]
        return [
            current_temp - 5, current_temp, current_temp + 5,
            current_temp + 10, current_temp + 15, current_temp + 25,
        ]

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

    def _rate_is_temperature_sensitive(self, current_temp: Optional[float]) -> bool:
        """Whether refinement can change anything at all.

        With no temperature reading, or a rate curve that ignores temperature,
        the profile cannot affect the plan and one solve is the whole answer.
        Probing costs a handful of lookups and saves a whole extra DP pass in
        the common case.
        """
        if current_temp is None:
            return False
        cfg = self._config
        socs = [cfg.min_soc, (cfg.min_soc + cfg.max_soc) / 2, cfg.max_soc]
        probes = self._probe_temps(current_temp)
        for soc in socs:
            base = self._rate_for(soc, probes[0])
            for temp in probes[1:]:
                if abs(self._rate_for(soc, temp) - base) > 1e-9:
                    return True
        return False

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

    def _replay_plan_temps(
        self,
        slots_sorted_by_time: List[PricePoint],
        slot_fractions: List[float],
        schedule: Dict[datetime.datetime, ScheduleEntry],
        energy_trajectory: List[float],
        start_energy: float,
        current_temp: Optional[float],
        load_kw: List[float],
        pv_kw: Optional[List[float]],
        planning_profile: List[Optional[float]],
    ) -> Tuple[List[Optional[float]], List[Tuple[Optional[float], Optional[float]]]]:
        """Temperatures the SELECTED plan actually produces.

        Warming is driven by ``simulate_slot``'s ``battery_power_kw``, i.e. the
        energy that really moved. A full pack ordered to charge, or an empty one
        ordered to discharge, moves nothing and warms nothing -- imaginary power
        must not manufacture future charging capability.

        Returns ``(start_of_slot_profile, [(start, end)] per slot)``.
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
            # The rate the plan was built with for this slot, evaluated at the
            # path's own SOC: the replay must not silently use a different
            # capability than the candidate transition did.
            rate = self._rate_for(
                soc,
                planning_profile[i] if i < len(planning_profile) else temp,
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
            start_temp = temp
            temp = self._step_temp(
                temp,
                price_point.time,
                cfg.slot_minutes * fraction,
                outcome.battery_power_kw,
                is_charge=(mode == BatteryMode.CHARGE),
            )
            pairs.append((start_temp, temp))
            # The DP's own path energy is authoritative for the SOC the plan
            # reaches; the simulation above exists for the battery POWER.
            energy = (
                energy_trajectory[i]
                if i < len(energy_trajectory)
                else outcome.energy_end_kwh
            )
        return profile, pairs

    def _profiles_agree(
        self,
        a: List[Optional[float]],
        b: List[Optional[float]],
        state_socs: Optional[List[float]] = None,
    ) -> bool:
        """Whether two temperature profiles are the same plan.

        Two criteria, either of which stops the refinement:

        1. The temperatures themselves agree within ``TEMP_FIXED_POINT_C``.
        2. They imply the SAME charge capability at every state of the DP's SOC
           grid. Temperature only reaches the plan through the rate, so two
           profiles that produce identical rates produce identical plans -- and
           a pack that keeps warming inside one temperature bucket would
           otherwise never reach a fixed point at all.

        Criterion 2 is evaluated at the grid SOCs. A rate curve with a
        discontinuity strictly between two adjacent grid points could be missed;
        that is the same 1 %-of-capacity resolution the whole DP works at, and
        the final replay in ``plan_validation`` is what catches a genuine
        disagreement.
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

    def _compute_charge_rates_per_slot(
        self,
        slots_sorted_by_time: List[PricePoint],
        slot_fractions: List[float],
        current_soc: float,
        current_temp: Optional[float],
    ) -> List[float]:
        """Legacy time-indexed rate array.

        NOT what the DP plans with any more -- ``_run_dp`` evaluates the rate
        per candidate transition. Kept because the orchestrator still uses the
        same helper as a fallback for the projected-cost column, where the real
        per-slot rate is re-derived from the learning engine anyway.
        """
        return compute_charge_rates_per_slot(
            slots_sorted_by_time=slots_sorted_by_time,
            slot_fractions=slot_fractions,
            slot_minutes=self._config.slot_minutes,
            current_soc=current_soc,
            current_temp=current_temp,
            get_charge_rate_for_soc=self._get_charge_rate_for_soc,
            predict_temp_after_duration=self._predict_temp_after_duration,
            project_temp=(
                self._temp_projector.project if self._temp_projector is not None else None
            ),
            battery_capacity=self._config.battery_capacity,
            efficiency=self._config.efficiency,
            max_soc=self._config.max_soc,
        )

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

        inv_eff = cfg.inverter_efficiency

        for t in range(n_list_slots):
            price = slots_list[t].price
            buy_price = self._buy_price(price)
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
            slot_hours_fraction = cfg.slot_hours * fraction
            net_load_kwh = net_load_kw * cfg.slot_hours * fraction

            # Export variables — NNS contract: sell price floor at 0
            sell_price = self._sell_price(price)
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

                # Charge capability of THIS state, at THIS slot's temperature.
                slot_charge_rate = rate_cache_slot.get(idx)
                if slot_charge_rate is None:
                    slot_charge_rate = rate_for(curr_soc, slot_temp)
                    rate_cache_slot[idx] = slot_charge_rate
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
                if charge_energy_kwh > 0 and pv_free_charge_kwh < charge_dc_kwh - 1e-6:
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
                if dc_discharge_kwh > 0:
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
                if sell_price > 0 and exported_kwh_full > 0 and available > 1e-9:
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
                self._log(f"[DeepTrace] After slot {t} ({slots_list[t].time.strftime('%H:%M')} @ {price:.4f}):")
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
                self._log(f"\n{slot_hour.strftime('%Y-%m-%d %H:%M')} @ {slot_price:.4f} EUR/kWh -> {chosen_action.name if chosen_action else '?'}")

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
    ) -> Tuple[Dict[datetime.datetime, ScheduleEntry], List[int], List[float], float]:
        """
        Build schedule using DP with greedy lookahead for partial first slot.

        Returns:
            (schedule, idx_trajectory, energy_trajectory, best_value)
        """
        cfg = self._config
        neg_inf = -1e18

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
            buy_price = self._buy_price(price)
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
            slot_charge_rate = self._rate_for(
                partial_soc,
                temp_profile[partial_index] if partial_index < len(temp_profile) else None,
            )
            charge_energy_kwh = slot_charge_rate * cfg.efficiency * cfg.slot_hours * fraction
            charge_dc_kwh = slot_charge_rate * cfg.slot_hours * fraction
            pv_free_charge_kwh = min(pv_surplus_kw, slot_charge_rate) * cfg.slot_hours * fraction
            net_load_kwh = net_load_kw * cfg.slot_hours * fraction

            # Remaining slots for DP
            remaining_slice = slice(partial_index + 1, None)
            slots_remaining = slots_sorted_by_time[remaining_slice]
            load_remaining = load_kw[remaining_slice]
            temp_profile_remaining = temp_profile[remaining_slice]
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
            sell_price = self._sell_price(price)
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
            if charge_energy_kwh > 0 and pv_free_charge_kwh < charge_dc_kwh - 1e-6:
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
            if dc_discharge_kwh > 0 and available > 1e-9:
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

            if sell_price > 0 and exported_kwh > 0 and available > 1e-9:
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
                )
                if not slots_remaining:
                    terminal_rate = cfg.terminal_energy_value_eur_kwh
                    if terminal_rate is None:
                        terminal_rate = max(
                            0.0,
                            buy_price * cfg.inverter_efficiency - cfg.battery_wear_cost,
                        )
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
            )

        terminal_rate = self._derive_terminal_rate(slots_sorted_by_time)
        for t, (price_point, action, lk, is_partial, is_export) in enumerate(zip(
            slots_sorted_by_time, actions, load_kw, partial_flags, export_flags
        )):
            hour = price_point.time
            price = price_point.price
            pv = pv_kw[t] if pv_kw is not None else 0.0
            reason = f"{price:.4f} EUR/kWh load~{lk:.2f}kW"
            if pv > 0:
                reason += f" pv~{pv:.2f}kW"
            if is_partial:
                reason += " (until depleted)"
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
            )
            if action == BatteryMode.DISCHARGE:
                entry.export_rate = 100 if is_export else 0
            elif action == BatteryMode.CHARGE and pv > 0:
                entry.ac_charge_mode = "pv_priority"
            schedule_local[hour] = entry

        return schedule_local, idx_trajectory, energy_trajectory, best_value
