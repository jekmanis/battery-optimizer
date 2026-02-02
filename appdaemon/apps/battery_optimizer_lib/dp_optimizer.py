"""
Dynamic Programming optimizer for battery scheduling.

Extracts the optimal charge/hold/discharge schedule using SOC-aware
dynamic programming with temperature-aware charge rate predictions.
"""

import datetime
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .models import BatteryMode, PricePoint, ScheduleEntry
from .charge_rate_utils import compute_charge_rates_per_slot


@dataclass
class DPOptimizerConfig:
    """Static configuration for DP optimizer."""
    battery_capacity: float      # kWh
    min_soc: float               # % (e.g., 10.0)
    max_soc: float               # % (e.g., 100.0)
    efficiency: float            # 0-1 (e.g., 0.85)
    discharge_rate: float        # kW
    slot_minutes: int            # e.g., 60
    soc_step_percent: float      # DP resolution (e.g., 1.0)
    grid_fee: float              # EUR/kWh
    battery_wear_cost: float     # EUR/kWh

    @property
    def slot_hours(self) -> float:
        return self.slot_minutes / 60.0


@dataclass
class DPOptimizerResult:
    """Immutable result from optimization."""
    schedule: Dict[datetime.datetime, ScheduleEntry]
    soc_trajectory: Dict[datetime.datetime, Tuple[float, float]]
    temp_trajectory: Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]]
    projected_costs: Dict[datetime.datetime, float]
    min_charge_slots: int
    charge_count: int
    discharge_count: int
    hold_count: int
    dp_best_value: float


class DPOptimizer:
    """
    SOC-aware dynamic programming optimizer for battery scheduling.

    Uses dependency injection for external functions:
    - load_predictor: predicts load (kW) for a given datetime
    - charge_rate_predictor: predicts charge rate (kW) for SOC and temperature
    - temp_after_charge_predictor: predicts temperature after charging
    - temp_after_idle_predictor: predicts temperature after idle period
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
    ):
        self._config = config
        self._predict_load_kw = load_predictor
        self._get_charge_rate_for_soc = charge_rate_predictor
        self._predict_temp_after_duration = temp_after_charge_predictor
        self._predict_temp_after_idle = temp_after_idle_predictor
        self._log_fn = log_fn
        self._decision_log_level = decision_log_level

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
        min_charge_slots_hint: int = 0,
    ) -> DPOptimizerResult:
        """
        Run DP optimization to find optimal schedule.

        Args:
            prices: List of price points (must include current_slot)
            current_slot: Current time slot (aligned to slot boundary)
            current_soc: Current battery state of charge (%)
            current_temp: Current battery temperature (Celsius, optional)
            minutes_into_slot: Minutes elapsed in current slot
            min_charge_slots_hint: Informational minimum charge slots (not enforced)

        Returns:
            DPOptimizerResult with schedule and trajectories
        """
        if not prices:
            return DPOptimizerResult(
                schedule={},
                soc_trajectory={},
                temp_trajectory={},
                projected_costs={},
                min_charge_slots=0,
                charge_count=0,
                discharge_count=0,
                hold_count=0,
                dp_best_value=0.0,
            )

        cfg = self._config
        hours_sorted_by_time = sorted(prices, key=lambda p: p.hour)
        n_slots = len(hours_sorted_by_time)

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

        for i, p in enumerate(hours_sorted_by_time):
            p_hour = p.hour
            compare_current = current_slot
            if p_hour.tzinfo is not None and compare_current.tzinfo is None:
                p_hour = p_hour.replace(tzinfo=None)
            elif p_hour.tzinfo is None and compare_current.tzinfo is not None:
                compare_current = compare_current.replace(tzinfo=None)
            if p_hour == compare_current:
                slot_fractions[i] = first_fraction
                current_slot_index = i
                break

        # Pre-compute temperature-aware charge rates for each slot
        charge_rates_per_slot = self._compute_charge_rates_per_slot(
            hours_sorted_by_time, slot_fractions, current_soc, current_temp
        )

        # Pre-compute load per slot
        load_kw = [self._predict_load_kw(p.hour) for p in hours_sorted_by_time]

        # Run DP to build schedule
        schedule, idx_trajectory, best_value = self._build_schedule(
            hours_sorted_by_time=hours_sorted_by_time,
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
            n_slots=n_slots,
            min_charge_slots=min_charge_slots_hint,
        )

        # Build SOC trajectory
        soc_trajectory = self._build_soc_trajectory(
            hours_sorted_by_time, idx_trajectory, start_energy, min_energy, step_kwh, n_states
        )

        # Build temperature trajectory
        temp_trajectory = self._build_temp_trajectory(
            hours_sorted_by_time, schedule, slot_fractions, current_temp
        )

        # Count actions
        charge_count = sum(1 for e in schedule.values() if e.mode == BatteryMode.CHARGE)
        discharge_count = sum(1 for e in schedule.values() if e.mode == BatteryMode.DISCHARGE)
        hold_count = sum(1 for e in schedule.values() if e.mode == BatteryMode.HOLD)

        return DPOptimizerResult(
            schedule=schedule,
            soc_trajectory=soc_trajectory,
            temp_trajectory=temp_trajectory,
            projected_costs={},  # Populated by caller if needed
            min_charge_slots=min_charge_slots_hint,
            charge_count=charge_count,
            discharge_count=discharge_count,
            hold_count=hold_count,
            dp_best_value=best_value,
        )

    def _compute_charge_rates_per_slot(
        self,
        hours_sorted_by_time: List[PricePoint],
        slot_fractions: List[float],
        current_soc: float,
        current_temp: Optional[float],
    ) -> List[float]:
        """Pre-compute temperature-aware charge rates for each slot."""
        return compute_charge_rates_per_slot(
            hours_sorted_by_time=hours_sorted_by_time,
            slot_fractions=slot_fractions,
            slot_minutes=self._config.slot_minutes,
            current_soc=current_soc,
            current_temp=current_temp,
            get_charge_rate_for_soc=self._get_charge_rate_for_soc,
            predict_temp_after_duration=self._predict_temp_after_duration,
        )

    def _build_soc_trajectory(
        self,
        hours_sorted_by_time: List[PricePoint],
        idx_trajectory: List[int],
        start_energy: float,
        min_energy: float,
        step_kwh: float,
        n_states: int,
    ) -> Dict[datetime.datetime, Tuple[float, float]]:
        """Convert idx_trajectory to SOC trajectory."""
        soc_trajectory: Dict[datetime.datetime, Tuple[float, float]] = {}
        cfg = self._config

        if idx_trajectory and len(idx_trajectory) == len(hours_sorted_by_time):
            start_idx = int(round((start_energy - min_energy) / step_kwh))
            start_idx = min(max(start_idx, 0), n_states - 1)

            for t, price_point in enumerate(hours_sorted_by_time):
                hour = price_point.hour
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
        hours_sorted_by_time: List[PricePoint],
        schedule: Dict[datetime.datetime, ScheduleEntry],
        slot_fractions: List[float],
        current_temp: Optional[float],
    ) -> Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]]:
        """Build temperature trajectory based on scheduled modes."""
        temp_trajectory: Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]] = {}
        cfg = self._config

        if current_temp is not None:
            projected_temp = current_temp
            for t, price_point in enumerate(hours_sorted_by_time):
                hour = price_point.hour
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
        hours_list: List[PricePoint],
        load_kw_list: List[float],
        charge_rates_list: List[float],
        slot_fractions_list: List[float],
        start_energy_kwh: float,
        start_c: int,
        min_energy: float,
        max_energy: float,
        step_kwh: float,
        n_states: int,
        energy_levels: List[float],
        max_charge_slots: int,
        min_charge_slots: int,
        start_idx_override: Optional[int] = None,
    ) -> Tuple[List[BatteryMode], List[bool], float, bool, List[int]]:
        """
        Core DP algorithm.

        Returns:
            (actions, partial_flags, best_value, meets_min, idx_trajectory)
        """
        cfg = self._config
        n_list_slots = len(hours_list)
        if n_list_slots == 0:
            return [], [], 0.0, True, []

        neg_inf = -1e18
        tie_val_eps = 1e-6
        tie_tie_eps = 1e-12
        tie_price_weight = 1e-5
        tie_time_weight = 1e-7

        # Discharge cost: only wear cost (battery degradation)
        discharge_cost_per_kwh = cfg.battery_wear_cost

        dp = [[neg_inf] * n_states for _ in range(max_charge_slots + 1)]
        dp_tie = [[neg_inf] * n_states for _ in range(max_charge_slots + 1)]

        if start_idx_override is None:
            start_idx_local = int(round((start_energy_kwh - min_energy) / step_kwh))
        else:
            start_idx_local = start_idx_override
        start_idx_local = min(max(start_idx_local, 0), n_states - 1)
        start_c = min(max(start_c, 0), max_charge_slots)
        dp[start_c][start_idx_local] = 0.0
        dp_tie[start_c][start_idx_local] = 0.0

        prev_idx = [[[None] * n_states for _ in range(max_charge_slots + 1)] for _ in range(n_list_slots)]
        prev_c = [[[None] * n_states for _ in range(max_charge_slots + 1)] for _ in range(n_list_slots)]
        prev_action = [[[None] * n_states for _ in range(max_charge_slots + 1)] for _ in range(n_list_slots)]
        prev_partial = [[[False] * n_states for _ in range(max_charge_slots + 1)] for _ in range(n_list_slots)]

        def _should_update(curr_val: float, curr_tie: float, cand_val: float, cand_tie: float) -> bool:
            if cand_val > curr_val + tie_val_eps:
                return True
            if abs(cand_val - curr_val) <= tie_val_eps and cand_tie > curr_tie + tie_tie_eps:
                return True
            return False

        dp_trace_slots = []

        for t in range(n_list_slots):
            price = hours_list[t].price
            buy_price = price + cfg.grid_fee
            fraction = slot_fractions_list[t]
            discharge_kwh = min(load_kw_list[t], cfg.discharge_rate) * cfg.slot_hours * fraction
            slot_charge_rate = charge_rates_list[t]
            charge_energy_kwh = slot_charge_rate * cfg.efficiency * cfg.slot_hours * fraction
            charge_cost_kwh = slot_charge_rate * cfg.slot_hours * fraction
            charge_count_increment = 1

            next_dp = [[neg_inf] * n_states for _ in range(max_charge_slots + 1)]
            next_dp_tie = [[neg_inf] * n_states for _ in range(max_charge_slots + 1)]
            next_prev_idx = [[None] * n_states for _ in range(max_charge_slots + 1)]
            next_prev_c = [[None] * n_states for _ in range(max_charge_slots + 1)]
            next_prev_action = [[None] * n_states for _ in range(max_charge_slots + 1)]
            next_prev_partial = [[False] * n_states for _ in range(max_charge_slots + 1)]

            slot_trace = []
            trace_this_slot = self._decision_log_level >= 3 and fraction > 0.5
            deep_trace_this_slot = self._decision_log_level >= 3 and t < 5

            for c in range(max_charge_slots + 1):
                for idx, val in enumerate(dp[c]):
                    if val <= neg_inf / 2:
                        continue
                    curr_tie = dp_tie[c][idx]
                    curr_soc = cfg.min_soc + (idx * step_kwh / cfg.battery_capacity) * 100

                    # HOLD - costs grid price for load
                    hold_updated = False
                    hold_cost = buy_price * discharge_kwh
                    hold_val = val - hold_cost
                    if _should_update(next_dp[c][idx], next_dp_tie[c][idx], hold_val, curr_tie):
                        next_dp[c][idx] = hold_val
                        next_dp_tie[c][idx] = curr_tie
                        next_prev_idx[c][idx] = idx
                        next_prev_c[c][idx] = c
                        next_prev_action[c][idx] = BatteryMode.HOLD
                        hold_updated = True

                    discharge_attempted = False
                    discharge_updated = False
                    discharge_blocked_reason = None
                    discharge_next_idx = None
                    discharge_next_val = None

                    # CHARGE
                    if charge_energy_kwh > 0 and c + charge_count_increment <= max_charge_slots:
                        new_energy = energy_levels[idx] + charge_energy_kwh
                        actual_charge_energy = charge_energy_kwh
                        actual_charge_cost = charge_cost_kwh

                        if new_energy > max_energy + 1e-6:
                            headroom = max_energy - energy_levels[idx]
                            if headroom >= step_kwh:
                                actual_charge_energy = headroom
                                actual_charge_cost = headroom / cfg.efficiency
                                new_energy = max_energy
                            else:
                                actual_charge_energy = 0

                        if actual_charge_energy > 0:
                            next_idx = int(round((new_energy - min_energy) / step_kwh))
                            next_idx = min(max(next_idx, 0), n_states - 1)
                            next_val = val - (buy_price * actual_charge_cost) - (buy_price * discharge_kwh)
                            charge_tie_bias = (-price * tie_price_weight) + (t * tie_time_weight)
                            next_tie = curr_tie + charge_tie_bias
                            c_next = c + charge_count_increment
                            if _should_update(next_dp[c_next][next_idx], next_dp_tie[c_next][next_idx], next_val, next_tie):
                                next_dp[c_next][next_idx] = next_val
                                next_dp_tie[c_next][next_idx] = next_tie
                                next_prev_idx[c_next][next_idx] = idx
                                next_prev_c[c_next][next_idx] = c
                                next_prev_action[c_next][next_idx] = BatteryMode.CHARGE

                    # DISCHARGE
                    if discharge_kwh > 0:
                        discharge_attempted = True
                        new_energy = energy_levels[idx] - discharge_kwh
                        is_partial = False
                        actual_discharge_kwh = discharge_kwh

                        if new_energy >= min_energy - 1e-6:
                            next_idx = int(round((new_energy - min_energy) / step_kwh))
                            next_idx = min(max(next_idx, 0), n_states - 1)
                            next_val = val - (discharge_cost_per_kwh * actual_discharge_kwh)
                        elif energy_levels[idx] > min_energy + discharge_kwh * 0.5:
                            is_partial = True
                            actual_discharge_kwh = energy_levels[idx] - min_energy
                            grid_import = discharge_kwh - actual_discharge_kwh
                            next_val = val - (buy_price * grid_import) - (discharge_cost_per_kwh * actual_discharge_kwh)
                            next_idx = 0
                            new_energy = min_energy
                        else:
                            discharge_blocked_reason = f"would_hit_min_soc ({new_energy:.2f} < {min_energy:.2f})"
                            next_val = None
                            next_idx = None

                        if next_val is not None:
                            discharge_next_idx = next_idx
                            discharge_next_val = next_val
                            if _should_update(next_dp[c][next_idx], next_dp_tie[c][next_idx], next_val, curr_tie):
                                next_dp[c][next_idx] = next_val
                                next_dp_tie[c][next_idx] = curr_tie
                                next_prev_idx[c][next_idx] = idx
                                next_prev_c[c][next_idx] = c
                                next_prev_action[c][next_idx] = BatteryMode.DISCHARGE
                                next_prev_partial[c][next_idx] = is_partial
                                discharge_updated = True
                            else:
                                discharge_blocked_reason = (
                                    f"existing_val={next_dp[c][next_idx]:.4f} >= discharge_val={next_val:.4f}"
                                )

                    # Trace collection for logging
                    trace_high_soc = curr_soc >= 95.0 and c >= min_charge_slots
                    if trace_this_slot and discharge_attempted and (c == 0 or trace_high_soc):
                        next_soc_discharge = cfg.min_soc + (discharge_next_idx * step_kwh / cfg.battery_capacity) * 100 if discharge_next_idx is not None else 0
                        slot_trace.append({
                            "charge_count": c,
                            "from_soc": curr_soc,
                            "from_idx": idx,
                            "from_val": val,
                            "hold_val": hold_val,
                            "hold_cost": hold_cost,
                            "hold_updated": hold_updated,
                            "discharge_attempted": discharge_attempted,
                            "discharge_updated": discharge_updated,
                            "discharge_blocked": discharge_blocked_reason,
                            "discharge_to_soc": next_soc_discharge,
                            "discharge_to_idx": discharge_next_idx,
                            "discharge_val": discharge_next_val,
                        })

            if trace_this_slot and slot_trace:
                dp_trace_slots.append((hours_list[t].hour, price, slot_trace))

            if deep_trace_this_slot:
                self._log(f"[DeepTrace] After slot {t} ({hours_list[t].hour.strftime('%H:%M')} @ {price:.4f}):")
                for c in range(min(3, max_charge_slots + 1)):
                    active_states = [
                        (i, next_dp[c][i], next_prev_action[c][i])
                        for i in range(n_states)
                        if next_dp[c][i] > neg_inf / 2
                    ]
                    if active_states:
                        active_states.sort(key=lambda x: x[1], reverse=True)
                        top_states = active_states[:3]
                        self._log(f"  c={c}: " + ", ".join(
                            f"idx={i} ({cfg.min_soc + i*step_kwh/cfg.battery_capacity*100:.1f}%) val={v:.4f} via {a.name if a else 'None'}"
                            for i, v, a in top_states
                        ))

            dp = next_dp
            dp_tie = next_dp_tie
            prev_idx[t] = next_prev_idx
            prev_c[t] = next_prev_c
            prev_action[t] = next_prev_action
            prev_partial[t] = next_prev_partial

        # Find best final state
        best_val = neg_inf
        best_tie = neg_inf
        best_idx = None
        best_c = None
        max_charge_achieved = 0

        for c in range(max_charge_slots + 1):
            for i in range(n_states):
                if dp[c][i] > neg_inf / 2:
                    if c > max_charge_achieved:
                        max_charge_achieved = c
                    if _should_update(best_val, best_tie, dp[c][i], dp_tie[c][i]):
                        best_val = dp[c][i]
                        best_tie = dp_tie[c][i]
                        best_idx = i
                        best_c = c

        meets_min = best_c is not None and best_c >= min_charge_slots
        if not meets_min and min_charge_slots > 0:
            self._log(
                f"Charge slots below calculated minimum ({best_c or 0} vs {min_charge_slots}) - "
                f"using grid during cheap hours instead",
                level="INFO",
            )

        # Backtrack to extract actions
        actions: List[BatteryMode] = []
        partial_flags: List[bool] = []
        idx_trajectory: List[int] = []
        idx = best_idx if best_idx is not None else start_idx_local
        c = best_c if best_c is not None else start_c

        if self._decision_log_level >= 3:
            self._log(f"[DeepTrace] Backtracking from best final state: c={c}, idx={idx}, val={best_val:.4f}")

        backtrack_trace = []
        for t in range(n_list_slots - 1, -1, -1):
            action = prev_action[t][c][idx] or BatteryMode.HOLD
            is_partial = prev_partial[t][c][idx] if action == BatteryMode.DISCHARGE else False
            actions.append(action)
            partial_flags.append(is_partial)
            idx_trajectory.append(idx)
            prev_i = prev_idx[t][c][idx]
            prev_c_val = prev_c[t][c][idx]

            if t < 5 and self._decision_log_level >= 3:
                soc_at_t = cfg.min_soc + (idx * step_kwh / cfg.battery_capacity) * 100
                backtrack_trace.append(f"t={t} ({hours_list[t].hour.strftime('%H:%M')}): action={action.name}, c={c}->prev_c={prev_c_val}, idx={idx} ({soc_at_t:.1f}%)->prev_i={prev_i}")

            if prev_i is None or prev_c_val is None:
                idx = idx
                c = c
            else:
                idx = prev_i
                c = prev_c_val

        actions.reverse()
        partial_flags.reverse()
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
                slot_idx = next((i for i, h in enumerate(hours_list) if h.hour == slot_hour), -1)
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

                        c_info = f"c={trace['charge_count']}, " if trace.get('charge_count', 0) > 0 else ""
                        delta = (trace['discharge_val'] - trace['hold_val']) if trace['discharge_val'] is not None else None
                        delta_str = f"+{delta:.4f}" if delta is not None and delta >= 0 else (f"{delta:.4f}" if delta is not None else "N/A")
                        discharge_val_str = f"{trace['discharge_val']:.4f}" if trace['discharge_val'] is not None else "N/A"
                        self._log(
                            f"  SOC {trace['from_soc']:.1f}% ({c_info}idx={trace['from_idx']}): "
                            f"hold={trace['hold_val']:.4f} vs discharge={discharge_val_str} (delta={delta_str}) -> {status}"
                        )
            self._log("=" * 70)

        return actions, partial_flags, best_val, meets_min, idx_trajectory

    def _build_schedule(
        self,
        hours_sorted_by_time: List[PricePoint],
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
        n_slots: int,
        min_charge_slots: int,
    ) -> Tuple[Dict[datetime.datetime, ScheduleEntry], List[int], float]:
        """
        Build schedule using DP with greedy lookahead for partial first slot.

        Returns:
            (schedule, idx_trajectory, best_value)
        """
        cfg = self._config
        max_charge_slots = n_slots
        neg_inf = -1e18

        # Discharge cost
        discharge_cost_per_kwh = cfg.battery_wear_cost

        schedule_local: Dict[datetime.datetime, ScheduleEntry] = {}
        partial_index = current_slot_index
        partial_fraction = slot_fractions[partial_index] if partial_index is not None else 1.0
        has_partial = partial_index is not None and partial_fraction < 0.999

        if has_partial:
            price_point = hours_sorted_by_time[partial_index]
            price = price_point.price
            buy_price = price + cfg.grid_fee
            fraction = slot_fractions[partial_index]
            slot_load_kw = load_kw[partial_index]
            discharge_kwh = min(slot_load_kw, cfg.discharge_rate) * cfg.slot_hours * fraction
            slot_charge_rate = charge_rates_per_slot[partial_index]
            charge_energy_kwh = slot_charge_rate * cfg.efficiency * cfg.slot_hours * fraction
            charge_cost_kwh = slot_charge_rate * cfg.slot_hours * fraction

            # Remaining slots for DP
            remaining_slice = slice(partial_index + 1, None)
            hours_remaining = hours_sorted_by_time[remaining_slice]
            load_remaining = load_kw[remaining_slice]
            charge_rates_remaining = charge_rates_per_slot[remaining_slice]
            slot_fractions_remaining = slot_fractions[remaining_slice]

            # Candidates: (action, new_energy, immediate_val, start_c, start_idx_override, is_partial)
            candidates = []

            # HOLD
            hold_cost = buy_price * discharge_kwh
            candidates.append((BatteryMode.HOLD, start_energy, -hold_cost, 0, None, False))

            # CHARGE
            partial_charge_increment = 1
            if charge_energy_kwh > 0 and partial_charge_increment <= max_charge_slots:
                new_energy = start_energy + charge_energy_kwh
                actual_charge_energy = charge_energy_kwh
                actual_charge_cost = charge_cost_kwh
                if new_energy > max_energy + 1e-6:
                    headroom = max_energy - start_energy
                    if headroom >= step_kwh:
                        actual_charge_energy = headroom
                        actual_charge_cost = headroom / cfg.efficiency
                        new_energy = max_energy
                    else:
                        actual_charge_energy = 0
                if actual_charge_energy > 0:
                    idx_float = (new_energy - min_energy) / step_kwh
                    start_idx_override = int(math.floor(idx_float + 1e-9))
                    charge_immediate_cost = -buy_price * actual_charge_cost - buy_price * discharge_kwh
                    candidates.append((
                        BatteryMode.CHARGE,
                        new_energy,
                        charge_immediate_cost,
                        partial_charge_increment,
                        start_idx_override,
                        False,
                    ))

            # DISCHARGE
            if discharge_kwh > 0:
                new_energy = start_energy - discharge_kwh
                if new_energy >= min_energy - 1e-6:
                    idx_float = (new_energy - min_energy) / step_kwh
                    start_idx_override = int(math.ceil(idx_float - 1e-9))
                    discharge_cost = -discharge_cost_per_kwh * discharge_kwh
                    candidates.append((
                        BatteryMode.DISCHARGE,
                        new_energy,
                        discharge_cost,
                        0,
                        start_idx_override,
                        False,
                    ))
                elif start_energy > min_energy + discharge_kwh * 0.5:
                    actual_discharge_kwh = start_energy - min_energy
                    grid_import = discharge_kwh - actual_discharge_kwh
                    partial_value = -buy_price * grid_import - discharge_cost_per_kwh * actual_discharge_kwh
                    candidates.append((
                        BatteryMode.DISCHARGE,
                        min_energy,
                        partial_value,
                        0,
                        0,
                        True,
                    ))

            best_action = BatteryMode.HOLD
            best_is_partial = False
            best_actions_remaining: List[BatteryMode] = []
            best_partial_flags_remaining: List[bool] = []
            best_idx_trajectory_remaining: List[int] = []
            best_first_slot_end_idx: int = int(round((start_energy - min_energy) / step_kwh))
            best_value = neg_inf

            if self._decision_log_level >= 3:
                self._log(f"[GreedyLookahead] Partial slot {price_point.hour.strftime('%H:%M')} @ {price:.4f}")
                self._log(f"  Candidates: {[(c[0].name, c[2]) for c in candidates]}")

            greedy_results = []
            for action, new_energy, immediate_val, start_c, start_idx_override, is_partial in candidates:
                actions_remaining, partial_flags_remaining, future_val, _, idx_traj_remaining = self._run_dp(
                    hours_remaining,
                    load_remaining,
                    charge_rates_remaining,
                    slot_fractions_remaining,
                    new_energy,
                    start_c,
                    min_energy,
                    max_energy,
                    step_kwh,
                    n_states,
                    energy_levels,
                    max_charge_slots,
                    min_charge_slots,
                    start_idx_override=start_idx_override,
                )
                total_val = immediate_val + future_val
                greedy_results.append((action.name, immediate_val, future_val, total_val))
                first_slot_end_idx = int(round((new_energy - min_energy) / step_kwh))
                first_slot_end_idx = min(max(first_slot_end_idx, 0), n_states - 1)
                if total_val > best_value:
                    best_value = total_val
                    best_action = action
                    best_is_partial = is_partial
                    best_actions_remaining = actions_remaining
                    best_partial_flags_remaining = partial_flags_remaining
                    best_idx_trajectory_remaining = idx_traj_remaining
                    best_first_slot_end_idx = first_slot_end_idx

            if self._decision_log_level >= 3:
                for name, imm, fut, tot in greedy_results:
                    self._log(f"  {name}: immediate={imm:.4f}, future={fut:.4f}, total={tot:.4f}")
                self._log(f"  -> Best: {best_action.name} (val={best_value:.4f})")

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
            idx_trajectory = [best_first_slot_end_idx] + best_idx_trajectory_remaining
        else:
            actions, partial_flags, best_value, _, idx_trajectory = self._run_dp(
                hours_sorted_by_time,
                load_kw,
                charge_rates_per_slot,
                slot_fractions,
                start_energy,
                0,
                min_energy,
                max_energy,
                step_kwh,
                n_states,
                energy_levels,
                max_charge_slots,
                min_charge_slots,
            )

        for price_point, action, lk, is_partial in zip(hours_sorted_by_time, actions, load_kw, partial_flags):
            hour = price_point.hour
            price = price_point.price
            reason = f"{price:.4f} EUR/kWh load~{lk:.2f}kW"
            if is_partial:
                reason += " (until depleted)"
            schedule_local[hour] = ScheduleEntry(hour=hour, mode=action, reason=reason)

        return schedule_local, idx_trajectory, best_value
