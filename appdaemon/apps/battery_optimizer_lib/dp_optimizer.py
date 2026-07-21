"""
Dynamic Programming optimizer for battery scheduling.

Extracts the optimal charge/hold/discharge schedule using SOC-aware
dynamic programming with temperature-aware charge rate predictions.
"""

import datetime
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import BatteryOptimizerConfig

from .models import BatteryMode, PricePoint, ScheduleEntry
from .charge_rate_utils import compute_charge_rates_per_slot


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
        direction: "floor" (after charge), "ceil" (after discharge), "round" (neutral)

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
        )


@dataclass
class DPOptimizerResult:
    """Immutable result from optimization."""
    schedule: Dict[datetime.datetime, ScheduleEntry]
    soc_trajectory: Dict[datetime.datetime, Tuple[float, float]]
    temp_trajectory: Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]]
    charge_count: int
    discharge_count: int
    hold_count: int
    export_slot_count: int = 0
    self_consume_slot_count: int = 0


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
    ):
        self._config = config
        self._predict_load_kw = load_predictor
        self._get_charge_rate_for_soc = charge_rate_predictor
        self._predict_temp_after_duration = temp_after_charge_predictor
        self._predict_temp_after_idle = temp_after_idle_predictor
        self._log_fn = log_fn
        self._decision_log_level = decision_log_level
        self._predict_pv_kw = pv_predictor

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
                discharge_count=0,
                hold_count=0,
            )

        cfg = self._config
        slots_sorted_by_time = sorted(prices, key=lambda p: p.time)
        n_slots = len(slots_sorted_by_time)

        # Energy bounds in kWh
        min_energy = (cfg.min_soc / 100) * cfg.battery_capacity
        max_energy = (cfg.max_soc / 100) * cfg.battery_capacity
        start_energy = min(max_energy, max(min_energy, (current_soc / 100) * cfg.battery_capacity))

        # DP resolution
        step_kwh = max(0.01, (cfg.soc_step_percent / 100) * cfg.battery_capacity)
        n_states = int(round((max_energy - min_energy) / step_kwh)) + 1
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

        # Pre-compute temperature-aware charge rates for each slot
        charge_rates_per_slot = self._compute_charge_rates_per_slot(
            slots_sorted_by_time, slot_fractions, current_soc, current_temp
        )

        # Pre-compute load and PV per slot
        load_kw = [self._predict_load_kw(p.time) for p in slots_sorted_by_time]
        pv_kw = (
            [self._predict_pv_kw(p.time) for p in slots_sorted_by_time]
            if self._predict_pv_kw else None
        )

        # Run DP to build schedule
        schedule, idx_trajectory, best_value = self._build_schedule(
            slots_sorted_by_time=slots_sorted_by_time,
            load_kw=load_kw,
            charge_rates_per_slot=charge_rates_per_slot,
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

        # Build SOC trajectory
        soc_trajectory = self._build_soc_trajectory(
            slots_sorted_by_time, idx_trajectory, start_energy, min_energy, step_kwh, n_states
        )

        # Build temperature trajectory
        temp_trajectory = self._build_temp_trajectory(
            slots_sorted_by_time, schedule, slot_fractions, current_temp
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
            discharge_count=discharge_count,
            hold_count=hold_count,
            export_slot_count=export_slot_count,
            self_consume_slot_count=self_consume_slot_count,
        )

    def _compute_charge_rates_per_slot(
        self,
        slots_sorted_by_time: List[PricePoint],
        slot_fractions: List[float],
        current_soc: float,
        current_temp: Optional[float],
    ) -> List[float]:
        """Pre-compute temperature and SOC-aware charge rates for each slot."""
        return compute_charge_rates_per_slot(
            slots_sorted_by_time=slots_sorted_by_time,
            slot_fractions=slot_fractions,
            slot_minutes=self._config.slot_minutes,
            current_soc=current_soc,
            current_temp=current_temp,
            get_charge_rate_for_soc=self._get_charge_rate_for_soc,
            predict_temp_after_duration=self._predict_temp_after_duration,
            battery_capacity=self._config.battery_capacity,
            efficiency=self._config.efficiency,
            max_soc=self._config.max_soc,
        )

    def _build_soc_trajectory(
        self,
        slots_sorted_by_time: List[PricePoint],
        idx_trajectory: List[int],
        start_energy: float,
        min_energy: float,
        step_kwh: float,
        n_states: int,
    ) -> Dict[datetime.datetime, Tuple[float, float]]:
        """Convert idx_trajectory to SOC trajectory."""
        soc_trajectory: Dict[datetime.datetime, Tuple[float, float]] = {}
        cfg = self._config

        if idx_trajectory and len(idx_trajectory) == len(slots_sorted_by_time):
            start_idx = _energy_to_index(start_energy, min_energy, step_kwh, n_states, "round")

            for t, price_point in enumerate(slots_sorted_by_time):
                hour = price_point.time
                if t == 0:
                    slot_start_idx = start_idx
                else:
                    slot_start_idx = idx_trajectory[t - 1]
                slot_end_idx = idx_trajectory[t]

                start_soc = cfg.min_soc + (slot_start_idx * step_kwh / cfg.battery_capacity) * 100
                end_soc = cfg.min_soc + (slot_end_idx * step_kwh / cfg.battery_capacity) * 100
                soc_trajectory[hour] = (start_soc, end_soc)

        return soc_trajectory

    def _build_temp_trajectory(
        self,
        slots_sorted_by_time: List[PricePoint],
        schedule: Dict[datetime.datetime, ScheduleEntry],
        slot_fractions: List[float],
        current_temp: Optional[float],
    ) -> Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]]:
        """Build temperature trajectory based on scheduled modes."""
        temp_trajectory: Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]] = {}
        cfg = self._config

        if current_temp is not None:
            projected_temp = current_temp
            for t, price_point in enumerate(slots_sorted_by_time):
                hour = price_point.time
                start_temp = projected_temp
                slot_duration_minutes = cfg.slot_minutes * slot_fractions[t]

                entry = schedule.get(hour)
                if entry is not None and entry.mode == BatteryMode.CHARGE:
                    projected_temp = self._predict_temp_after_duration(projected_temp, slot_duration_minutes)
                else:
                    projected_temp = self._predict_temp_after_idle(projected_temp, slot_duration_minutes)
                temp_trajectory[hour] = (start_temp, projected_temp)

        return temp_trajectory

    def _run_dp(
        self,
        slots_list: List[PricePoint],
        load_kw_list: List[float],
        charge_rates_list: List[float],
        slot_fractions_list: List[float],
        start_energy_kwh: float,
        min_energy: float,
        max_energy: float,
        step_kwh: float,
        n_states: int,
        energy_levels: List[float],
        start_idx_override: Optional[int] = None,
        pv_kw_list: Optional[List[float]] = None,
    ) -> Tuple[List[BatteryMode], List[bool], List[bool], float, List[int]]:
        """
        Core DP algorithm.

        Returns:
            (actions, partial_flags, export_flags, best_value, idx_trajectory)
        """
        cfg = self._config
        n_list_slots = len(slots_list)
        if n_list_slots == 0:
            return [], [], [], 0.0, []

        neg_inf = -1e18
        tie_val_eps = 1e-6
        tie_tie_eps = 1e-12
        tie_price_weight = 1e-5
        tie_time_weight = 1e-7

        # Discharge cost: only wear cost (battery degradation)
        discharge_cost_per_kwh = cfg.battery_wear_cost

        # Allocate 1D DP buffers and template row for efficient reset
        _neg_inf_row = [neg_inf] * n_states
        dp_a = [neg_inf] * n_states
        dp_b = [neg_inf] * n_states
        dp_tie_a = [neg_inf] * n_states
        dp_tie_b = [neg_inf] * n_states

        if start_idx_override is None:
            start_idx_local = _energy_to_index(start_energy_kwh, min_energy, step_kwh, n_states, "round")
        else:
            start_idx_local = min(max(start_idx_override, 0), n_states - 1)

        # Initialize starting state in dp_a
        dp_a[start_idx_local] = 0.0
        dp_tie_a[start_idx_local] = 0.0
        dp, next_dp = dp_a, dp_b
        dp_tie, next_dp_tie = dp_tie_a, dp_tie_b

        prev_idx = [None] * n_list_slots
        prev_action = [None] * n_list_slots
        prev_partial = [None] * n_list_slots
        prev_export = [None] * n_list_slots

        def _should_update(curr_val: float, curr_tie: float, cand_val: float, cand_tie: float) -> bool:
            if cand_val > curr_val + tie_val_eps:
                return True
            if abs(cand_val - curr_val) <= tie_val_eps and cand_tie > curr_tie + tie_tie_eps:
                return True
            return False

        dp_trace_slots = []

        inv_eff = cfg.inverter_efficiency

        for t in range(n_list_slots):
            price = slots_list[t].price
            buy_price = price + cfg.grid_fee
            fraction = slot_fractions_list[t]
            slot_load_kw = load_kw_list[t]
            slot_pv_kw = pv_kw_list[t] if pv_kw_list is not None else 0.0
            net_load_kw = max(0.0, slot_load_kw - slot_pv_kw)
            pv_surplus_kw = max(0.0, slot_pv_kw - slot_load_kw)
            # AC load battery can serve via self-consumption (capped by discharge rate)
            discharge_kwh = min(net_load_kw, cfg.discharge_rate) * cfg.slot_hours * fraction
            # DC energy battery must provide (higher due to inverter DC→AC loss)
            dc_discharge_kwh = discharge_kwh / inv_eff
            slot_charge_rate = charge_rates_list[t]
            charge_energy_kwh = slot_charge_rate * cfg.efficiency * cfg.slot_hours * fraction
            # DC energy into battery from all sources (grid + PV)
            charge_dc_kwh = slot_charge_rate * cfg.slot_hours * fraction
            # PV surplus charges battery for free (pv_priority, DC→DC); grid covers rest
            pv_free_charge_kwh = min(pv_surplus_kw, slot_charge_rate) * cfg.slot_hours * fraction
            net_load_kwh = net_load_kw * cfg.slot_hours * fraction

            # Export variables — NNS contract: sell price floor at 0
            sell_price = max(0.0, price * cfg.export_rate_multiplier - cfg.grid_export_fee)
            export_discharge_kwh = cfg.effective_export_discharge_rate * cfg.slot_hours * fraction
            dc_export_discharge_kwh = export_discharge_kwh / inv_eff
            load_kwh = slot_load_kw * cfg.slot_hours * fraction
            pv_kwh = slot_pv_kw * cfg.slot_hours * fraction
            exported_kwh_full = max(0.0, export_discharge_kwh + pv_kwh - load_kwh)

            # Reset next_dp buffers using slice assignment (faster than nested loop)
            next_dp[:] = _neg_inf_row
            next_dp_tie[:] = _neg_inf_row
            next_prev_idx = [None] * n_states
            next_prev_action = [None] * n_states
            next_prev_partial = [False] * n_states
            next_prev_export = [False] * n_states

            # HOLD precomputation (slot-level constants)
            hold_grid_cost = buy_price * net_load_kwh
            # PV surplus charges battery for free (up to charge rate), rest exports
            hold_pv_charge_kw = min(pv_surplus_kw, slot_charge_rate)
            hold_pv_charge_max = hold_pv_charge_kw * cfg.efficiency * cfg.slot_hours * fraction
            hold_excess_pv_kwh = max(0.0, pv_surplus_kw - slot_charge_rate) * cfg.slot_hours * fraction
            hold_sell = sell_price  # already floored at 0

            slot_trace = []
            trace_this_slot = self._decision_log_level >= 3 and fraction > 0.5
            deep_trace_this_slot = self._decision_log_level >= 3 and t < 5

            for idx, val in enumerate(dp):
                if val <= neg_inf / 2:
                    continue
                curr_tie = dp_tie[idx]
                curr_soc = cfg.min_soc + (idx * step_kwh / cfg.battery_capacity) * 100

                # HOLD - no grid charge; PV covers load, surplus charges battery for free
                hold_updated = False
                if hold_pv_charge_max > 0:
                    new_energy = min(max_energy, energy_levels[idx] + hold_pv_charge_max)
                    actual_stored = new_energy - energy_levels[idx]
                    # PV that couldn't be stored (battery full) + excess beyond charge rate → export
                    if actual_stored < hold_pv_charge_max - 1e-9:
                        unused_kwh = (hold_pv_charge_max - actual_stored) / cfg.efficiency
                    else:
                        unused_kwh = 0.0
                    export_revenue = hold_sell * (hold_excess_pv_kwh + unused_kwh)
                    hold_next_idx = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "floor")
                else:
                    export_revenue = hold_sell * hold_excess_pv_kwh
                    hold_next_idx = idx
                hold_val = val - hold_grid_cost + export_revenue
                if _should_update(next_dp[hold_next_idx], next_dp_tie[hold_next_idx], hold_val, curr_tie):
                    next_dp[hold_next_idx] = hold_val
                    next_dp_tie[hold_next_idx] = curr_tie
                    next_prev_idx[hold_next_idx] = idx
                    next_prev_action[hold_next_idx] = BatteryMode.HOLD
                    hold_updated = True

                discharge_attempted = False
                discharge_updated = False
                discharge_blocked_reason = None
                discharge_next_idx = None
                discharge_next_val = None

                # CHARGE (grid_charge): only when grid is actually needed to supplement PV
                # When PV surplus fully covers charge rate, HOLD already handles it
                if charge_energy_kwh > 0 and pv_free_charge_kwh < charge_dc_kwh - 1e-6:
                    new_energy = energy_levels[idx] + charge_energy_kwh
                    actual_charge_energy = charge_energy_kwh
                    actual_charge_dc = charge_dc_kwh

                    if new_energy > max_energy + 1e-6:
                        headroom = max_energy - energy_levels[idx]
                        if headroom >= step_kwh:
                            actual_charge_energy = headroom
                            actual_charge_dc = headroom / cfg.efficiency
                            new_energy = max_energy
                        else:
                            actual_charge_energy = 0

                    if actual_charge_energy > 0:
                        next_idx = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "floor")
                        # DC energy from grid (total DC minus PV contribution)
                        grid_charge_dc = max(0.0, actual_charge_dc - pv_free_charge_kwh)
                        # Skip CHARGE when PV fully covers it — HOLD already handles PV charging
                        if grid_charge_dc < 1e-6:
                            actual_charge_energy = 0

                    if actual_charge_energy > 0:
                        # AC energy from grid = DC / inverter_efficiency
                        grid_charge_ac = grid_charge_dc / inv_eff
                        next_val = val - (buy_price * grid_charge_ac) - (buy_price * net_load_kwh)
                        charge_tie_bias = (-price * tie_price_weight) + (t * tie_time_weight)
                        next_tie = curr_tie + charge_tie_bias
                        if _should_update(next_dp[next_idx], next_dp_tie[next_idx], next_val, next_tie):
                            next_dp[next_idx] = next_val
                            next_dp_tie[next_idx] = next_tie
                            next_prev_idx[next_idx] = idx
                            next_prev_action[next_idx] = BatteryMode.CHARGE

                # DISCHARGE (self-consumption)
                # Battery provides DC energy; inverter converts to AC to serve load.
                # dc_discharge_kwh = AC load / inverter_eff (battery works harder).
                if dc_discharge_kwh > 0:
                    discharge_attempted = True
                    new_energy = energy_levels[idx] - dc_discharge_kwh
                    is_partial = False
                    actual_dc_kwh = dc_discharge_kwh

                    if new_energy >= min_energy - 1e-6:
                        next_idx = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "round")
                        # Grid covers any AC load that battery+inverter can't
                        grid_import_kwh = max(0.0, net_load_kwh - discharge_kwh)
                        next_val = val - (buy_price * grid_import_kwh) - (discharge_cost_per_kwh * actual_dc_kwh)
                    elif energy_levels[idx] > min_energy + dc_discharge_kwh * 0.5:
                        is_partial = True
                        actual_dc_kwh = energy_levels[idx] - min_energy
                        ac_served = actual_dc_kwh * inv_eff
                        grid_import_kwh = max(0.0, net_load_kwh - ac_served)
                        next_val = val - (buy_price * grid_import_kwh) - (discharge_cost_per_kwh * actual_dc_kwh)
                        next_idx = 0
                        new_energy = min_energy
                    else:
                        discharge_blocked_reason = f"would_hit_min_soc ({new_energy:.2f} < {min_energy:.2f})"
                        next_val = None
                        next_idx = None

                    if next_val is not None:
                        discharge_next_idx = next_idx
                        discharge_next_val = next_val
                        if _should_update(next_dp[next_idx], next_dp_tie[next_idx], next_val, curr_tie):
                            next_dp[next_idx] = next_val
                            next_dp_tie[next_idx] = curr_tie
                            next_prev_idx[next_idx] = idx
                            next_prev_action[next_idx] = BatteryMode.DISCHARGE
                            next_prev_partial[next_idx] = is_partial
                            discharge_updated = True
                        else:
                            discharge_blocked_reason = (
                                f"existing_val={next_dp[next_idx]:.4f} >= discharge_val={next_val:.4f}"
                            )

                # DISCHARGE_EXPORT (full rate discharge with grid export)
                # SOC transition uses DC energy; export revenue uses AC output.
                if sell_price > 0 and exported_kwh_full > 0:
                    new_energy = energy_levels[idx] - dc_export_discharge_kwh
                    is_partial_export = False
                    actual_dc_export = dc_export_discharge_kwh

                    if new_energy >= min_energy - 1e-6:
                        next_idx = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "round")
                        next_val = val + sell_price * exported_kwh_full - (discharge_cost_per_kwh * actual_dc_export)
                    elif energy_levels[idx] > min_energy + dc_export_discharge_kwh * 0.3:
                        is_partial_export = True
                        actual_dc_export = energy_levels[idx] - min_energy
                        ac_from_battery = actual_dc_export * inv_eff
                        remaining_load = max(0.0, load_kwh - pv_kwh - ac_from_battery)
                        actual_exported = max(0.0, ac_from_battery + pv_kwh - load_kwh)
                        if remaining_load > 0:
                            # Battery + PV can't cover load — grid covers rest
                            next_val = val - (buy_price * remaining_load) - (discharge_cost_per_kwh * actual_dc_export)
                        else:
                            next_val = val + sell_price * actual_exported - (discharge_cost_per_kwh * actual_dc_export)
                        next_idx = 0
                        new_energy = min_energy
                    else:
                        next_val = None
                        next_idx = None

                    if next_val is not None:
                        if _should_update(next_dp[next_idx], next_dp_tie[next_idx], next_val, curr_tie):
                            next_dp[next_idx] = next_val
                            next_dp_tie[next_idx] = curr_tie
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
            prev_idx[t] = next_prev_idx
            prev_action[t] = next_prev_action
            prev_partial[t] = next_prev_partial
            prev_export[t] = next_prev_export

        # Find best final state
        best_val = neg_inf
        best_tie = neg_inf
        best_idx = None

        for i in range(n_states):
            if dp[i] > neg_inf / 2:
                if _should_update(best_val, best_tie, dp[i], dp_tie[i]):
                    best_val = dp[i]
                    best_tie = dp_tie[i]
                    best_idx = i

        # Backtrack to extract actions
        actions: List[BatteryMode] = []
        partial_flags: List[bool] = []
        export_flags: List[bool] = []
        idx_trajectory: List[int] = []
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

        return actions, partial_flags, export_flags, best_val, idx_trajectory

    def _build_schedule(
        self,
        slots_sorted_by_time: List[PricePoint],
        load_kw: List[float],
        charge_rates_per_slot: List[float],
        slot_fractions: List[float],
        current_slot_index: Optional[int],
        start_energy: float,
        min_energy: float,
        max_energy: float,
        step_kwh: float,
        n_states: int,
        energy_levels: List[float],
        pv_kw: Optional[List[float]] = None,
    ) -> Tuple[Dict[datetime.datetime, ScheduleEntry], List[int], float]:
        """
        Build schedule using DP with greedy lookahead for partial first slot.

        Returns:
            (schedule, idx_trajectory, best_value)
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
            buy_price = price + cfg.grid_fee
            fraction = slot_fractions[partial_index]
            slot_load_kw = load_kw[partial_index]
            slot_pv_kw = pv_kw[partial_index] if pv_kw is not None else 0.0
            net_load_kw = max(0.0, slot_load_kw - slot_pv_kw)
            pv_surplus_kw = max(0.0, slot_pv_kw - slot_load_kw)
            discharge_kwh = min(net_load_kw, cfg.discharge_rate) * cfg.slot_hours * fraction
            dc_discharge_kwh = discharge_kwh / inv_eff
            slot_charge_rate = charge_rates_per_slot[partial_index]
            charge_energy_kwh = slot_charge_rate * cfg.efficiency * cfg.slot_hours * fraction
            charge_dc_kwh = slot_charge_rate * cfg.slot_hours * fraction
            pv_free_charge_kwh = min(pv_surplus_kw, slot_charge_rate) * cfg.slot_hours * fraction
            net_load_kwh = net_load_kw * cfg.slot_hours * fraction

            # Remaining slots for DP
            remaining_slice = slice(partial_index + 1, None)
            slots_remaining = slots_sorted_by_time[remaining_slice]
            load_remaining = load_kw[remaining_slice]
            charge_rates_remaining = charge_rates_per_slot[remaining_slice]
            slot_fractions_remaining = slot_fractions[remaining_slice]
            pv_kw_remaining = pv_kw[remaining_slice] if pv_kw is not None else None

            # Candidates: (action, new_energy, immediate_val, start_idx_override, is_partial, is_export)
            candidates = []

            # HOLD — no grid charge; PV covers load, surplus charges battery for free
            sell_price = max(0.0, price * cfg.export_rate_multiplier - cfg.grid_export_fee)
            hold_pv_charge_kw = min(pv_surplus_kw, slot_charge_rate)
            hold_pv_energy = hold_pv_charge_kw * cfg.efficiency * cfg.slot_hours * fraction
            hold_new_energy = min(max_energy, start_energy + hold_pv_energy)
            hold_actual_stored = hold_new_energy - start_energy
            # PV that couldn't be stored + excess beyond charge rate → export
            hold_unused_kwh = (hold_pv_energy - hold_actual_stored) / cfg.efficiency if hold_actual_stored < hold_pv_energy - 1e-9 else 0.0
            hold_excess_pv = max(0.0, pv_surplus_kw - slot_charge_rate) * cfg.slot_hours * fraction
            hold_export_revenue = sell_price * (hold_excess_pv + hold_unused_kwh)
            hold_grid_cost = buy_price * net_load_kwh
            hold_idx_override = _energy_to_index(hold_new_energy, min_energy, step_kwh, n_states, "floor") if hold_pv_energy > 0 else None
            candidates.append((BatteryMode.HOLD, hold_new_energy, -hold_grid_cost + hold_export_revenue, hold_idx_override, False, False))

            # CHARGE (grid_charge): only when grid is actually needed to supplement PV
            if charge_energy_kwh > 0 and pv_free_charge_kwh < charge_dc_kwh - 1e-6:
                new_energy = start_energy + charge_energy_kwh
                actual_charge_energy = charge_energy_kwh
                actual_charge_dc = charge_dc_kwh
                if new_energy > max_energy + 1e-6:
                    headroom = max_energy - start_energy
                    if headroom >= step_kwh:
                        actual_charge_energy = headroom
                        actual_charge_dc = headroom / cfg.efficiency
                        new_energy = max_energy
                    else:
                        actual_charge_energy = 0
                if actual_charge_energy > 0:
                    start_idx_override = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "floor")
                    grid_charge_dc = max(0.0, actual_charge_dc - pv_free_charge_kwh)
                    grid_charge_ac = grid_charge_dc / inv_eff
                    charge_immediate_cost = -buy_price * grid_charge_ac - buy_price * net_load_kwh
                    candidates.append((
                        BatteryMode.CHARGE,
                        new_energy,
                        charge_immediate_cost,
                        start_idx_override,
                        False,
                        False,
                    ))

            # DISCHARGE (self-consumption)
            if dc_discharge_kwh > 0:
                new_energy = start_energy - dc_discharge_kwh
                if new_energy >= min_energy - 1e-6:
                    start_idx_override = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "round")
                    grid_import_kwh = max(0.0, net_load_kwh - discharge_kwh)
                    discharge_cost = -buy_price * grid_import_kwh - discharge_cost_per_kwh * dc_discharge_kwh
                    candidates.append((
                        BatteryMode.DISCHARGE,
                        new_energy,
                        discharge_cost,
                        start_idx_override,
                        False,
                        False,
                    ))
                elif start_energy > min_energy + dc_discharge_kwh * 0.5:
                    dc_available = start_energy - min_energy
                    ac_served = dc_available * inv_eff
                    grid_import_kwh = max(0.0, net_load_kwh - ac_served)
                    partial_value = -buy_price * grid_import_kwh - discharge_cost_per_kwh * dc_available
                    candidates.append((
                        BatteryMode.DISCHARGE,
                        min_energy,
                        partial_value,
                        0,
                        True,
                        False,
                    ))

            # DISCHARGE_EXPORT (full rate with grid selling, PV adds to export)
            export_discharge_kwh = cfg.effective_export_discharge_rate * cfg.slot_hours * fraction
            dc_export_discharge_kwh = export_discharge_kwh / inv_eff
            load_kwh_slot = slot_load_kw * cfg.slot_hours * fraction
            pv_kwh_slot = slot_pv_kw * cfg.slot_hours * fraction
            exported_kwh = max(0.0, export_discharge_kwh + pv_kwh_slot - load_kwh_slot)

            if sell_price > 0 and exported_kwh > 0:
                new_energy = start_energy - dc_export_discharge_kwh
                if new_energy >= min_energy - 1e-6:
                    start_idx_override = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "round")
                    export_value = sell_price * exported_kwh - discharge_cost_per_kwh * dc_export_discharge_kwh
                    candidates.append((
                        BatteryMode.DISCHARGE,
                        new_energy,
                        export_value,
                        start_idx_override,
                        False,
                        True,
                    ))
                elif start_energy > min_energy + dc_export_discharge_kwh * 0.3:
                    dc_available = start_energy - min_energy
                    ac_from_battery = dc_available * inv_eff
                    remaining_load = max(0.0, load_kwh_slot - pv_kwh_slot - ac_from_battery)
                    actual_exported = max(0.0, ac_from_battery + pv_kwh_slot - load_kwh_slot)
                    if remaining_load > 0:
                        export_value = -buy_price * remaining_load - discharge_cost_per_kwh * dc_available
                    else:
                        export_value = sell_price * actual_exported - discharge_cost_per_kwh * dc_available
                    candidates.append((
                        BatteryMode.DISCHARGE,
                        min_energy,
                        export_value,
                        0,
                        True,
                        True,
                    ))

            best_action = BatteryMode.HOLD
            best_is_partial = False
            best_is_export = False
            best_actions_remaining: List[BatteryMode] = []
            best_partial_flags_remaining: List[bool] = []
            best_export_flags_remaining: List[bool] = []
            best_idx_trajectory_remaining: List[int] = []
            best_first_slot_end_idx: int = _energy_to_index(start_energy, min_energy, step_kwh, n_states, "round")
            best_value = neg_inf

            if self._decision_log_level >= 3:
                self._log(f"[GreedyLookahead] Partial slot {price_point.time.strftime('%H:%M')} @ {price:.4f}")
                self._log(f"  Candidates: {[(c[0].name + ('[EXP]' if c[5] else ''), c[2]) for c in candidates]}")

            greedy_results = []
            for action, new_energy, immediate_val, start_idx_override, is_partial, is_export in candidates:
                actions_remaining, partial_flags_remaining, export_flags_remaining, future_val, idx_traj_remaining = self._run_dp(
                    slots_remaining,
                    load_remaining,
                    charge_rates_remaining,
                    slot_fractions_remaining,
                    new_energy,
                    min_energy,
                    max_energy,
                    step_kwh,
                    n_states,
                    energy_levels,
                    start_idx_override=start_idx_override,
                    pv_kw_list=pv_kw_remaining,
                )
                total_val = immediate_val + future_val
                label = action.name + ("[EXP]" if is_export else "")
                greedy_results.append((label, immediate_val, future_val, total_val))
                if action == BatteryMode.CHARGE:
                    first_slot_end_idx = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "floor")
                elif action == BatteryMode.DISCHARGE:
                    first_slot_end_idx = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "round")
                else:
                    first_slot_end_idx = _energy_to_index(new_energy, min_energy, step_kwh, n_states, "round")
                if total_val > best_value:
                    best_value = total_val
                    best_action = action
                    best_is_partial = is_partial
                    best_is_export = is_export
                    best_actions_remaining = actions_remaining
                    best_partial_flags_remaining = partial_flags_remaining
                    best_export_flags_remaining = export_flags_remaining
                    best_idx_trajectory_remaining = idx_traj_remaining
                    best_first_slot_end_idx = first_slot_end_idx

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
            idx_trajectory = [best_first_slot_end_idx] + best_idx_trajectory_remaining
        else:
            actions, partial_flags, export_flags, best_value, idx_trajectory = self._run_dp(
                slots_sorted_by_time,
                load_kw,
                charge_rates_per_slot,
                slot_fractions,
                start_energy,
                min_energy,
                max_energy,
                step_kwh,
                n_states,
                energy_levels,
                pv_kw_list=pv_kw,
            )

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
            entry = ScheduleEntry(time=hour, mode=action, reason=reason)
            if action == BatteryMode.DISCHARGE:
                entry.export_rate = 100 if is_export else 0
            elif action == BatteryMode.CHARGE and pv > 0:
                entry.ac_charge_mode = "pv_priority"
            schedule_local[hour] = entry

        return schedule_local, idx_trajectory, best_value
