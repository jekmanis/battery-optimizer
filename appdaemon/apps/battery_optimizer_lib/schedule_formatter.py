"""
Schedule formatting and logging for the Battery Optimizer.

This module handles all presentation logic for battery optimization schedules:
- Logging schedules with SOC/temperature trajectories
- Decision transparency logging
- Formatting schedule data for Home Assistant sensors
- Human-readable schedule summaries

Extracted from BatteryOptimizer to separate presentation from business logic.
"""

import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .models import BatteryMode, PricePoint, ScheduleEntry
from .timezone_utils import lookup_by_time


# WIT mode display names and icons for markdown rendering
_WIT_MODE_DISPLAY = {
    "grid_charge":        ("Grid Charge",        "🔋", "Yes"),
    "discharge_to_load":  ("Discharge to Load",  "🏠", "No"),
    "discharge_to_grid":  ("Discharge to Grid",  "💰", "Yes"),
    "max_export":         ("Max Export",          "⚡", "Yes"),
    "hold":               ("Hold",               "⏸️", "—"),
    "self_consumption":   ("Self-Consumption",    "☀️", "Auto"),
}


def resolve_wit_mode(entry: ScheduleEntry, default_power_percent: int = 100) -> str:
    """Map a ScheduleEntry to the WIT mode string that DirectControl would send.

    Mirrors DirectControl._resolve_charge_mode / _resolve_discharge_mode logic
    so the display layer can show the same mode without importing DirectControl.
    """
    mode = entry.mode
    if mode == BatteryMode.CHARGE:
        return "grid_charge"
    elif mode == BatteryMode.DISCHARGE:
        export_rate = entry.export_rate
        if export_rate is not None and export_rate > 0:
            power = entry.power_percent if entry.power_percent is not None else default_power_percent
            if export_rate >= 100 and power >= 100:
                return "max_export"
            return "discharge_to_grid"
        return "discharge_to_load"
    elif mode == BatteryMode.SELF_CONSUMPTION:
        return "self_consumption"
    else:
        return "hold"


@dataclass
class ScheduleFormatterConfig:
    """Configuration for schedule formatting."""

    slot_minutes: int
    slot_hours: float
    battery_capacity: float
    charge_rate: float
    discharge_rate: float
    export_discharge_rate: float  # kW — discharge rate during grid export (0 = use discharge_rate)
    efficiency: float
    battery_wear_cost: float
    decision_log_level: int


class ScheduleFormatter:
    """
    Formats and logs battery optimization schedules.

    Handles all presentation concerns:
    - Schedule logging with SOC/temperature trajectories
    - Decision context logging for transparency
    - Schedule formatting for HA sensor attributes
    - Human-readable summaries
    """

    def __init__(
        self,
        config: ScheduleFormatterConfig,
        log_func: Callable[[str], None],
        learning_engine=None,
    ):
        """
        Initialize the schedule formatter.

        Args:
            config: Static configuration for formatting
            log_func: Function to call for logging (typically self.log from AppDaemon)
            learning_engine: Optional BatteryLearningEngine for charge rate predictions
        """
        self.config = config
        self.log = log_func
        self.learning_engine = learning_engine

    def log_schedule(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        expected_soc: Optional[Dict[datetime.datetime, float]] = None,
        expected_temp: Optional[Dict[datetime.datetime, float]] = None,
        dp_soc_trajectory: Optional[Dict[datetime.datetime, Tuple[float, float]]] = None,
        dp_temp_trajectory: Optional[
            Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]]
        ] = None,
        projected_costs: Optional[Dict[datetime.datetime, float]] = None,
        local_tz=None,
        predict_load_kw: Optional[Callable[[datetime.datetime], float]] = None,
        predict_pv_kw: Optional[Callable[[datetime.datetime], float]] = None,
        min_soc: float = 10.0,
        max_soc: float = 100.0,
    ):
        """
        Log the full schedule in a readable format with optional expected SOC and temperature.

        Prefers the DP optimizer's actual SOC trajectory (dp_soc_trajectory) when available,
        as this reflects the exact values the optimizer computed. Falls back to expected_soc
        (from calculate_expected_soc_schedule) when DP trajectory is not available.

        Args:
            schedule: Schedule entries keyed by slot datetime
            expected_soc: Expected SOC at start of each slot (fallback)
            expected_temp: Expected temperature at start of each slot (fallback)
            dp_soc_trajectory: DP optimizer's SOC trajectory (start, end) per slot
            dp_temp_trajectory: DP optimizer's temp trajectory (start, end) per slot
            projected_costs: Projected battery cost at each slot
            local_tz: Local timezone for display
            predict_load_kw: Function to predict load for a given datetime
            min_soc: Minimum SOC constraint (dynamic property)
            max_soc: Maximum SOC constraint (dynamic property)
        """
        if not schedule:
            self.log("No schedule to log")
            return

        self.log("=" * 60)
        self.log("GENERATED SCHEDULE")
        self.log("=" * 60)

        sorted_hours = sorted(schedule.keys())

        for hour in sorted_hours:
            entry = schedule[hour]
            # Ensure time is displayed in local timezone
            display_hour = hour
            if hour.tzinfo is not None and local_tz is not None:
                display_hour = hour.astimezone(local_tz)
            time_str = display_hour.strftime("%Y-%m-%d %H:%M")
            mode_str = entry.mode.name.ljust(9)

            # Get SOC and temperature trajectory string
            soc_str = self._format_soc_trajectory(
                hour=hour,
                entry=entry,
                dp_soc_trajectory=dp_soc_trajectory,
                dp_temp_trajectory=dp_temp_trajectory,
                expected_soc=expected_soc,
                expected_temp=expected_temp,
                local_tz=local_tz,
                predict_load_kw=predict_load_kw,
                predict_pv_kw=predict_pv_kw,
                min_soc=min_soc,
                max_soc=max_soc,
            )

            # For discharge, show projected battery cost as primary, grid price in parentheses
            reason_display = self._format_reason_with_cost(
                entry, hour, projected_costs
            )

            self.log(f"  {time_str}  {mode_str}  {reason_display}{soc_str}")

        self.log("=" * 60)

        # Summary counts
        charge_count = len(
            [e for e in schedule.values() if e.mode == BatteryMode.CHARGE]
        )
        export_count = len(
            [e for e in schedule.values()
             if e.mode == BatteryMode.DISCHARGE and e.export_rate is not None and e.export_rate > 0]
        )
        self_consume_count = len(
            [e for e in schedule.values()
             if e.mode == BatteryMode.DISCHARGE and (e.export_rate is None or e.export_rate == 0)]
        )
        hold_count = len([e for e in schedule.values() if e.mode == BatteryMode.HOLD])
        sc_count = len([e for e in schedule.values() if e.mode == BatteryMode.SELF_CONSUMPTION])
        parts = [f"{charge_count} charge"]
        if self_consume_count:
            parts.append(f"{self_consume_count} discharge(self)")
        if export_count:
            parts.append(f"{export_count} discharge(export)")
        if sc_count:
            parts.append(f"{sc_count} self_consumption")
        parts.append(f"{hold_count} hold")
        self.log(f"Total: {', '.join(parts)} slots")

    def _format_soc_trajectory(
        self,
        hour: datetime.datetime,
        entry: ScheduleEntry,
        dp_soc_trajectory: Optional[Dict[datetime.datetime, Tuple[float, float]]],
        dp_temp_trajectory: Optional[
            Dict[datetime.datetime, Tuple[Optional[float], Optional[float]]]
        ],
        expected_soc: Optional[Dict[datetime.datetime, float]],
        expected_temp: Optional[Dict[datetime.datetime, float]],
        local_tz,
        predict_load_kw: Optional[Callable[[datetime.datetime], float]],
        predict_pv_kw: Optional[Callable[[datetime.datetime], float]] = None,
        min_soc: float = 10.0,
        max_soc: float = 100.0,
    ) -> str:
        """Format the SOC/temperature trajectory string for a schedule entry."""
        # Try DP trajectory first (exact values from optimizer)
        dp_soc_data = (
            lookup_by_time(dp_soc_trajectory, hour, local_tz)
            if dp_soc_trajectory
            else None
        )
        dp_temp_data = (
            lookup_by_time(dp_temp_trajectory, hour, local_tz)
            if dp_temp_trajectory
            else None
        )

        if dp_soc_data is not None:
            return self._format_dp_trajectory(dp_soc_data, dp_temp_data)

        if expected_soc:
            return self._format_expected_trajectory(
                hour=hour,
                entry=entry,
                expected_soc=expected_soc,
                expected_temp=expected_temp,
                local_tz=local_tz,
                predict_load_kw=predict_load_kw,
                predict_pv_kw=predict_pv_kw,
                min_soc=min_soc,
                max_soc=max_soc,
            )

        return ""

    def _format_dp_trajectory(
        self,
        dp_soc_data: Tuple[float, float],
        dp_temp_data: Optional[Tuple[Optional[float], Optional[float]]],
    ) -> str:
        """Format trajectory using DP optimizer's computed values."""
        start_soc, end_soc = dp_soc_data
        if dp_temp_data is not None:
            start_temp, end_temp = dp_temp_data
            if start_temp is not None and end_temp is not None:
                return f" {start_soc:5.1f}%->{end_soc:5.1f}% ({start_temp:.0f}C->{end_temp:.0f}C)"
        return f" {start_soc:5.1f}%->{end_soc:5.1f}%"

    def _format_expected_trajectory(
        self,
        hour: datetime.datetime,
        entry: ScheduleEntry,
        expected_soc: Dict[datetime.datetime, float],
        expected_temp: Optional[Dict[datetime.datetime, float]],
        local_tz,
        predict_load_kw: Optional[Callable[[datetime.datetime], float]],
        predict_pv_kw: Optional[Callable[[datetime.datetime], float]] = None,
        min_soc: float = 10.0,
        max_soc: float = 100.0,
    ) -> str:
        """Format trajectory using recalculated expected values (fallback)."""
        start_soc = lookup_by_time(expected_soc, hour, local_tz)
        start_temp = (
            lookup_by_time(expected_temp, hour, local_tz) if expected_temp else None
        )

        if start_soc is None:
            return ""

        if entry.mode == BatteryMode.CHARGE:
            return self._format_charge_trajectory(start_soc, start_temp, max_soc)
        elif entry.mode == BatteryMode.DISCHARGE:
            return self._format_discharge_trajectory(
                hour, start_soc, start_temp, min_soc, predict_load_kw, entry=entry
            )
        elif entry.mode == BatteryMode.SELF_CONSUMPTION:
            pv_kw = predict_pv_kw(hour) if predict_pv_kw else 0.0
            load_kw = predict_load_kw(hour) if predict_load_kw else 0.5
            pv_surplus = max(0.0, pv_kw - load_kw)
            pv_charge_kw = min(pv_surplus, self.config.charge_rate)
            pv_charge_kwh = pv_charge_kw * self.config.slot_hours * self.config.efficiency
            load_deficit = max(0.0, load_kw - pv_kw)
            battery_discharge_kwh = min(load_deficit, self.config.discharge_rate) * self.config.slot_hours
            net_energy = pv_charge_kwh - battery_discharge_kwh
            soc_change = (net_energy / self.config.battery_capacity) * 100
            end_soc = max(min_soc, min(max_soc, start_soc + soc_change))
            return f" {start_soc:5.1f}%->{end_soc:5.1f}%"
        else:  # HOLD
            return self._format_hold_trajectory(start_soc, start_temp)

    def _format_charge_trajectory(
        self,
        start_soc: float,
        start_temp: Optional[float],
        max_soc: float,
    ) -> str:
        """Format trajectory for a CHARGE slot."""
        if start_temp is not None and self.learning_engine:
            # Use temperature-aware charging
            energy_added, end_temp = self.learning_engine.predict_charge_energy_with_warming(
                start_soc, start_temp, self.config.slot_minutes, temp_threshold=16.0
            )
            energy_to_battery = energy_added * self.config.efficiency
            end_soc = min(
                max_soc,
                start_soc + (energy_to_battery / self.config.battery_capacity) * 100,
            )
            return f" {start_soc:5.1f}%->{end_soc:5.1f}% ({start_temp:.0f}C->{end_temp:.0f}C)"
        else:
            # Fallback: Use learned charge rate without temperature
            effective_charge_rate = self.config.charge_rate
            if self.learning_engine:
                learned_rate = self.learning_engine.get_charge_rate_for_soc(start_soc)
                if learned_rate is not None and learned_rate > 0:
                    effective_charge_rate = learned_rate
            energy_added = (
                effective_charge_rate * self.config.efficiency * self.config.slot_hours
            )
            end_soc = min(
                max_soc,
                start_soc + (energy_added / self.config.battery_capacity) * 100,
            )
            return f" {start_soc:5.1f}%->{end_soc:5.1f}%"

    def _format_discharge_trajectory(
        self,
        hour: datetime.datetime,
        start_soc: float,
        start_temp: Optional[float],
        min_soc: float,
        predict_load_kw: Optional[Callable[[datetime.datetime], float]],
        entry: Optional[ScheduleEntry] = None,
    ) -> str:
        """Format trajectory for a DISCHARGE slot."""
        if entry is not None and entry.export_rate is not None and entry.export_rate > 0:
            edr = self.config.export_discharge_rate if self.config.export_discharge_rate > 0 else self.config.discharge_rate
            energy_removed = edr * self.config.slot_hours
        else:
            load_kw = predict_load_kw(hour) if predict_load_kw else 0.5
            energy_removed = (
                min(load_kw, self.config.discharge_rate) * self.config.slot_hours
            )
        end_soc = max(
            min_soc, start_soc - (energy_removed / self.config.battery_capacity) * 100
        )

        if start_temp is not None and self.learning_engine:
            end_temp = self.learning_engine.predict_temp_after_idle(
                start_temp, self.config.slot_minutes
            )
            return f" {start_soc:5.1f}%->{end_soc:5.1f}% ({start_temp:.0f}C->{end_temp:.0f}C)"
        return f" {start_soc:5.1f}%->{end_soc:5.1f}%"

    def _format_hold_trajectory(
        self,
        start_soc: float,
        start_temp: Optional[float],
    ) -> str:
        """Format trajectory for a HOLD slot."""
        end_soc = start_soc

        if start_temp is not None and self.learning_engine:
            end_temp = self.learning_engine.predict_temp_after_idle(
                start_temp, self.config.slot_minutes
            )
            return f" {start_soc:5.1f}%->{end_soc:5.1f}% ({start_temp:.0f}C->{end_temp:.0f}C)"
        return f" {start_soc:5.1f}%->{end_soc:5.1f}%"

    def _format_reason_with_cost(
        self,
        entry: ScheduleEntry,
        hour: datetime.datetime,
        projected_costs: Optional[Dict[datetime.datetime, float]],
    ) -> str:
        """Format reason string, showing projected cost for discharge slots."""
        if entry.mode != BatteryMode.DISCHARGE or not projected_costs:
            return entry.reason

        proj_cost = projected_costs.get(hour)
        if proj_cost is None:
            return entry.reason

        # Parse grid price from reason: "X.XXXX EUR/kWh load~Y.YYkW"
        parts = entry.reason.split(" EUR/kWh")
        if len(parts) == 2:
            grid_price = parts[0]
            rest = parts[1]
            return f"{proj_cost:.4f} EUR/kWh (grid {grid_price}){rest}"
        return entry.reason

    def log_decision_context(
        self,
        prices_sorted: List[PricePoint],
        schedule: Dict[datetime.datetime, ScheduleEntry],
        load_kw: List[float],
        current_soc: float,
        min_charge_slots: int,
        battery_avg_cost: float,
        min_soc: float,
    ) -> List[Dict]:
        """
        Log detailed decision context for transparency.
        Shows why specific charge/discharge slots were selected.

        Args:
            prices_sorted: Prices sorted by time
            schedule: Generated schedule
            load_kw: Predicted load for each price point
            current_soc: Current SOC percentage
            min_charge_slots: Minimum charge slots calculated
            battery_avg_cost: Current weighted average battery cost
            min_soc: Minimum SOC constraint

        Returns:
            List of charge slot dicts for storage in sensor attributes
        """
        # Extract charge and discharge slots from schedule
        charge_slots = []
        discharge_slots = []
        for hour, entry in schedule.items():
            price_point = next((p for p in prices_sorted if p.time == hour), None)
            price = price_point.price if price_point else 0.0
            if entry.mode == BatteryMode.CHARGE:
                charge_slots.append({"hour": hour, "price": price})
            elif entry.mode == BatteryMode.DISCHARGE:
                # Find corresponding load
                idx = next(
                    (i for i, p in enumerate(prices_sorted) if p.time == hour), 0
                )
                load = load_kw[idx] if idx < len(load_kw) else 0.0
                discharge_slots.append({"hour": hour, "price": price, "load": load})

        # Sort all prices to rank candidates
        all_prices_sorted = sorted(prices_sorted, key=lambda p: p.price)
        price_rank = {p.time: i + 1 for i, p in enumerate(all_prices_sorted)}

        # Build charge slots list for sensor exposure
        formatted_charge_slots = [
            {"time": s["hour"].isoformat(), "price": round(s["price"], 4)}
            for s in sorted(charge_slots, key=lambda x: x["hour"])
        ]

        # Build decision context log
        if self.config.decision_log_level >= 1:
            self.log("=" * 70)
            self.log("DECISION CONTEXT")
            self.log("=" * 70)

            # Input state
            self.log("Input State:")
            self.log(f"  Current SOC: {current_soc:.1f}%")
            self.log(f"  Min SOC target: {min_soc:.1f}%")
            self.log(f"  Min charge slots (informational): {min_charge_slots}")
            self.log(f"  Battery avg cost: {battery_avg_cost:.4f} EUR/kWh")
            self.log(
                f"  Discharge wear cost: {self.config.battery_wear_cost:.4f} EUR/kWh"
            )
            self.log(
                "  Note: DP evaluates all options; discharge only costs wear (no double-counting)"
            )

        # Verbose logging (level 2): show candidates and analysis
        if self.config.decision_log_level >= 2:
            self._log_verbose_decision_context(
                all_prices_sorted,
                charge_slots,
                discharge_slots,
                price_rank,
                prices_sorted,
            )

        if self.config.decision_log_level >= 1:
            self.log("=" * 70)

        return formatted_charge_slots

    def _log_verbose_decision_context(
        self,
        all_prices_sorted: List[PricePoint],
        charge_slots: List[Dict],
        discharge_slots: List[Dict],
        price_rank: Dict[datetime.datetime, int],
        prices_sorted: List[PricePoint],
    ):
        """Log verbose decision context (level 2)."""

        def _fmt_dt(dt: datetime.datetime) -> str:
            return dt.strftime("%Y-%m-%d %H:%M")

        # Cheapest charge candidates
        self.log("\nCheapest 5 charge candidates:")
        for i, p in enumerate(all_prices_sorted[:5]):
            marker = " *" if any(s["hour"] == p.time for s in charge_slots) else ""
            self.log(f"  {i+1}. {_fmt_dt(p.time)} @ {p.price:.4f} EUR/kWh{marker}")

        # Selected charge slots with rankings
        if charge_slots:
            self.log(f"\nSelected charge slots ({len(charge_slots)}):")
            for slot in sorted(charge_slots, key=lambda s: s["hour"]):
                rank = price_rank.get(slot["hour"], "?")
                total_prices = len(prices_sorted)
                self.log(
                    f"  {_fmt_dt(slot['hour'])} @ {slot['price']:.4f} EUR/kWh (rank {rank}/{total_prices})"
                )

        # Selected discharge slots
        if discharge_slots:
            self.log(f"\nSelected discharge slots ({len(discharge_slots)}):")
            for slot in sorted(discharge_slots, key=lambda s: s["hour"]):
                self.log(
                    f"  {_fmt_dt(slot['hour'])} @ {slot['price']:.4f} EUR/kWh (load~{slot['load']:.2f}kW)"
                )

        # Arbitrage analysis
        if charge_slots and discharge_slots:
            avg_charge_price = sum(s["price"] for s in charge_slots) / len(charge_slots)
            avg_discharge_price = sum(s["price"] for s in discharge_slots) / len(
                discharge_slots
            )
            spread = avg_discharge_price - avg_charge_price
            effective_spread = spread - (
                avg_charge_price * (1 - self.config.efficiency)
            )

            self.log("\nArbitrage Analysis:")
            self.log(f"  Avg charge price: {avg_charge_price:.4f} EUR/kWh")
            self.log(f"  Avg discharge price: {avg_discharge_price:.4f} EUR/kWh")
            self.log(
                f"  Spread: {spread:.4f} EUR/kWh (after {self.config.efficiency*100:.0f}% efficiency: {effective_spread:.4f} EUR/kWh)"
            )

    def format_schedule_list(
        self, schedule: Dict[datetime.datetime, ScheduleEntry]
    ) -> List[Dict]:
        """
        Format schedule as a list of dicts for sensor attributes.

        Args:
            schedule: Schedule entries keyed by slot datetime

        Returns:
            List of {time, mode, wit_mode, reason, export} dicts sorted by time
        """
        schedule_data = []
        for hour in sorted(schedule.keys()):
            entry = schedule[hour]
            wit_mode = resolve_wit_mode(entry)
            display = _WIT_MODE_DISPLAY.get(wit_mode, (wit_mode, "", "—"))
            schedule_data.append({
                "time": hour.isoformat(),
                "mode": entry.mode.name,
                "wit_mode": wit_mode,
                "wit_mode_name": display[0],
                "export": display[2],
                "reason": entry.reason,
            })
        return schedule_data

    def format_schedule_markdown(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        now: datetime.datetime,
        local_tz,
        align_to_slot_func: Callable[[datetime.datetime], datetime.datetime],
        dp_soc_trajectory: Optional[Dict[datetime.datetime, Tuple[float, float]]] = None,
        predict_load_kw: Optional[Callable[[datetime.datetime], float]] = None,
        predict_pv_kw: Optional[Callable[[datetime.datetime], float]] = None,
    ) -> str:
        """Generate a markdown table of the schedule for HA dashboard display.

        Shows the actual WIT mode that will be sent to the inverter, export
        status, SOC trajectory, price, and estimated load/PV — giving full
        visibility into what the optimizer is doing.

        Args:
            schedule: Schedule entries keyed by slot datetime
            now: Current datetime
            local_tz: Local timezone for display
            align_to_slot_func: Function to align datetime to slot boundary
            dp_soc_trajectory: DP optimizer's SOC trajectory (start, end) per slot
            predict_load_kw: Load predictor function (kW) per slot
            predict_pv_kw: PV production predictor function (kW) per slot
        """
        if not schedule:
            return "No schedule available"

        # Determine the current slot for highlighting
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        now_slot = align_to_slot_func(now)

        lines: List[str] = []
        lines.append("| Time | Mode | Export | SOC | Load | PV | Price |")
        lines.append("|------|------|--------|-----|------|-----|-------|")

        for hour in sorted(schedule.keys()):
            entry = schedule[hour]

            # Display time in local tz
            display_hour = hour
            if hour.tzinfo is not None and local_tz is not None:
                display_hour = hour.astimezone(local_tz)
            time_str = display_hour.strftime("%H:%M")

            # Current-slot marker
            is_current = self._is_same_slot(hour, now_slot, local_tz)
            if is_current:
                time_str = f"**{time_str}**"

            # WIT mode
            wit_mode = resolve_wit_mode(entry)
            display = _WIT_MODE_DISPLAY.get(wit_mode, (wit_mode, "", "—"))
            icon = display[1]
            mode_name = display[0]
            export_str = display[2]

            # SOC trajectory
            soc_str = ""
            if dp_soc_trajectory:
                soc_data = lookup_by_time(dp_soc_trajectory, hour, local_tz)
                if soc_data is not None:
                    start_soc, end_soc = soc_data
                    soc_str = f"{start_soc:.0f}→{end_soc:.0f}%"

            # Estimated load
            load_str = ""
            if predict_load_kw is not None:
                load_val = predict_load_kw(hour)
                load_str = f"{load_val:.2f}"

            # Estimated PV
            pv_str = ""
            if predict_pv_kw is not None:
                pv_val = predict_pv_kw(hour)
                if pv_val > 0:
                    pv_str = f"{pv_val:.2f}"

            # Price from reason (format: "X.XXXX EUR/kWh ...")
            price_str = self._extract_price_from_reason(entry.reason)

            lines.append(
                f"| {time_str} | {icon} {mode_name} | {export_str} | {soc_str} | {load_str} | {pv_str} | {price_str} |"
            )

        return "\n".join(lines)

    @staticmethod
    def _is_same_slot(
        hour: datetime.datetime,
        now_slot: datetime.datetime,
        local_tz,
    ) -> bool:
        """Check if hour refers to the same slot as now_slot."""
        compare_hour = hour
        compare_now = now_slot
        if local_tz is not None:
            if hour.tzinfo is not None:
                compare_hour = hour.astimezone(local_tz)
            if now_slot.tzinfo is not None:
                compare_now = now_slot.astimezone(local_tz)
        if compare_hour.tzinfo is not None and compare_now.tzinfo is None:
            compare_hour = compare_hour.replace(tzinfo=None)
        elif compare_hour.tzinfo is None and compare_now.tzinfo is not None:
            compare_now = compare_now.replace(tzinfo=None)
        return compare_hour == compare_now

    @staticmethod
    def _extract_price_from_reason(reason: str) -> str:
        """Pull the price out of a reason string like '0.1234 EUR/kWh load~0.5kW'."""
        if not reason:
            return ""
        parts = reason.split(" EUR/kWh")
        if len(parts) >= 2:
            try:
                price_val = float(parts[0])
                return f"{price_val:.4f}"
            except ValueError:
                pass
        # Try just taking leading numeric
        for token in reason.split():
            try:
                val = float(token)
                return f"{val:.4f}"
            except ValueError:
                break
        return ""

    def find_next_events(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        now: datetime.datetime,
        local_tz,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the next charge and discharge times.

        Args:
            schedule: Schedule entries
            now: Current datetime (should be in local timezone)
            local_tz: Local timezone for comparison

        Returns:
            Tuple of (next_charge_iso, next_discharge_iso), either may be None
        """
        next_charge = None
        next_discharge = None

        for hour in sorted(schedule.keys()):
            # Convert both to local timezone for comparison
            compare_hour = hour
            compare_now = now
            if local_tz is not None:
                if hour.tzinfo is not None:
                    compare_hour = hour.astimezone(local_tz)
                if compare_now.tzinfo is not None:
                    compare_now = compare_now.astimezone(local_tz)
            # Handle mixed timezone-aware/naive
            if compare_hour.tzinfo is not None and compare_now.tzinfo is None:
                compare_hour = compare_hour.replace(tzinfo=None)
            elif compare_hour.tzinfo is None and compare_now.tzinfo is not None:
                compare_now = compare_now.replace(tzinfo=None)
            if compare_hour < compare_now:
                continue

            entry = schedule[hour]
            if entry.mode == BatteryMode.CHARGE and next_charge is None:
                next_charge = hour.isoformat()
            if entry.mode == BatteryMode.DISCHARGE and next_discharge is None:
                next_discharge = hour.isoformat()

        return next_charge, next_discharge

    def format_summary(
        self,
        schedule: Dict[datetime.datetime, ScheduleEntry],
        now: datetime.datetime,
        local_tz,
        align_to_slot_func: Callable[[datetime.datetime], datetime.datetime],
    ) -> str:
        """
        Generate a human-readable schedule summary.

        Args:
            schedule: Schedule entries
            now: Current datetime
            local_tz: Local timezone
            align_to_slot_func: Function to align datetime to slot boundary

        Returns:
            Human-readable summary string
        """
        if not schedule:
            return "No schedule available"

        # Convert now to local timezone
        if now.tzinfo is not None and local_tz is not None:
            now = now.astimezone(local_tz)
        now_slot = align_to_slot_func(now)

        def is_future_or_current(k):
            compare_k = k
            compare_now = now_slot
            if local_tz is not None:
                if k.tzinfo is not None:
                    compare_k = k.astimezone(local_tz)
                if compare_now.tzinfo is not None:
                    compare_now = compare_now.astimezone(local_tz)
            # Handle mixed timezone-aware/naive
            if compare_k.tzinfo is not None and compare_now.tzinfo is None:
                compare_k = compare_k.replace(tzinfo=None)
            elif compare_k.tzinfo is None and compare_now.tzinfo is not None:
                compare_now = compare_now.replace(tzinfo=None)
            return compare_k >= compare_now

        future_schedule = {k: v for k, v in schedule.items() if is_future_or_current(k)}

        charge_hours = [
            h for h, e in future_schedule.items() if e.mode == BatteryMode.CHARGE
        ]
        discharge_hours = [
            h for h, e in future_schedule.items() if e.mode == BatteryMode.DISCHARGE
        ]
        hold_hours = [
            h for h, e in future_schedule.items() if e.mode == BatteryMode.HOLD
        ]

        summary = (
            f"Schedule: {len(charge_hours)} slots charge, "
            f"{len(discharge_hours)} slots discharge, "
            f"{len(hold_hours)} slots hold"
        )

        if charge_hours:
            next_charge = min(charge_hours)
            if next_charge.tzinfo is not None and local_tz is not None:
                next_charge = next_charge.astimezone(local_tz)
            summary += f"\nNext charge: {next_charge.strftime('%H:%M')}"
        if discharge_hours:
            next_discharge = min(discharge_hours)
            if next_discharge.tzinfo is not None and local_tz is not None:
                next_discharge = next_discharge.astimezone(local_tz)
            summary += f"\nNext discharge: {next_discharge.strftime('%H:%M')}"

        return summary
