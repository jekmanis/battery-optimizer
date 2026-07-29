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
from .soc_projection import SocProjectionParams, project_slot_soc
from .timezone_utils import lookup_by_time


# WIT mode display names and icons for markdown rendering
_WIT_MODE_DISPLAY = {
    "grid_charge":        ("Grid Charge",        "🔋", "Yes"),
    "discharge_to_load":  ("Discharge to Load",  "🏠", "No"),
    "discharge_to_grid":  ("Discharge to Grid",  "💰", "Yes"),
    "max_export":         ("Max Export",          "⚡", "Yes"),
    "hold":               ("Hold",               "⏸️", "—"),
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
            # DirectControl always sends config.default_power_percent; ScheduleEntry
            # no longer carries a per-slot power_percent.
            if export_rate >= 100 and default_power_percent >= 100:
                return "max_export"
            return "discharge_to_grid"
        return "discharge_to_load"
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
    # AC<->DC conversion efficiency. Needed because the fallback trajectory goes
    # through the shared soc_projection model, which moves DC energy.
    inverter_efficiency: float = 1.0


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
        temp_projector=None,
    ):
        """
        Initialize the schedule formatter.

        Args:
            config: Static configuration for formatting
            log_func: Function to call for logging (typically self.log from AppDaemon)
            learning_engine: Optional BatteryLearningEngine for charge rate predictions
            temp_projector: Optional shared thermal_model.TemperatureProjector.
                Without it this fallback path would show a DIFFERENT temperature
                model from the DP trajectory depending on which branch of
                ``_format_soc_trajectory`` is taken.
        """
        self.config = config
        self.log = log_func
        self.learning_engine = learning_engine
        self._temp_projector = temp_projector

    def _projection_params(
        self, min_soc: float, max_soc: float
    ) -> SocProjectionParams:
        """Battery parameters for the ONE shared slot-SOC transition model."""
        return SocProjectionParams(
            battery_capacity=self.config.battery_capacity,
            efficiency=self.config.efficiency,
            charge_rate=self.config.charge_rate,
            discharge_rate=self.config.discharge_rate,
            export_discharge_rate=self.config.export_discharge_rate,
            inverter_efficiency=self.config.inverter_efficiency,
            min_soc=min_soc,
            max_soc=max_soc,
            slot_minutes=self.config.slot_minutes,
        )

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
            predict_pv_kw: Function to predict PV production for a given datetime.
                Required for the fallback trajectory to agree with
                ``soc_projection`` — PV surplus charges the battery in HOLD and
                in self-consumption DISCHARGE.
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
        parts = [f"{charge_count} charge"]
        if self_consume_count:
            parts.append(f"{self_consume_count} discharge(self)")
        if export_count:
            parts.append(f"{export_count} discharge(export)")
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
        """Format trajectory using recalculated expected values (fallback).

        The slot transition goes through ``soc_projection.project_slot_soc`` —
        the SAME model as the expected-SOC trajectory and the deviation
        detector. This path used to carry a third, private transition model:
        HOLD ignored PV surplus charging entirely (``end_soc = start_soc``) and
        DISCHARGE drained at ``min(load, discharge_rate)`` on raw load without
        subtracting PV or dividing by the inverter efficiency. On a sunny slot
        (PV 4.0 kW, load 0.8 kW, SOC 50%) it printed 50.0%->50.0% for HOLD and
        50.0%->48.6% for DISCHARGE where the shared model gives 55.3% for both —
        a 5-7 point contradiction in the very log used to diagnose SOC
        deviations.
        """
        start_soc = lookup_by_time(expected_soc, hour, local_tz)
        start_temp = (
            lookup_by_time(expected_temp, hour, local_tz) if expected_temp else None
        )

        if start_soc is None:
            return ""

        transition = project_slot_soc(
            soc_start=start_soc,
            mode=entry.mode,
            params=self._projection_params(min_soc, max_soc),
            load_kw=predict_load_kw(hour) if predict_load_kw else 0.0,
            pv_kw=predict_pv_kw(hour) if predict_pv_kw else 0.0,
            export_rate=entry.export_rate,
            temp_start=start_temp,
            learning_engine=self.learning_engine,
            temp_projector=self._temp_projector,
            slot_time=hour,
        )

        end_soc = transition.soc_end
        end_temp = transition.temp_end
        if start_temp is not None and end_temp is not None:
            return (
                f" {start_soc:5.1f}%->{end_soc:5.1f}% "
                f"({start_temp:.0f}C->{end_temp:.0f}C)"
            )
        return f" {start_soc:5.1f}%->{end_soc:5.1f}%"

    def _format_reason_with_cost(
        self,
        entry: ScheduleEntry,
        hour: datetime.datetime,
        projected_costs: Optional[Dict[datetime.datetime, float]],
    ) -> str:
        """Format the reason string so the leading number explains the decision.

        Preferred form (DP supplied a marginal value):

            0.1234 EUR/kWh avoided-import (grid 0.0712, stored 0.0000) load~...

        The leading number is THIS slot's marginal economics as scored by the
        DP. The tracked stored-energy basis stays visible as a separate figure —
        it degenerates to 0.0000 whenever PV was booked at the zero export floor
        (midday spot at/below the export fee), which is correct but explains
        nothing about the decision. Before the fix that degenerate basis WAS the
        only number shown, so every slot logged "0.0000 EUR/kWh".

        Falls back to the legacy discharge-only formatting when no marginal
        value is present (e.g. schedules restored from the HA sensor).
        """
        marginal = getattr(entry, "marginal_value_eur_kwh", None)
        proj_cost = projected_costs.get(hour) if projected_costs else None

        if marginal is None:
            if entry.mode != BatteryMode.DISCHARGE or proj_cost is None:
                return entry.reason
            parts = entry.reason.split(" EUR/kWh")
            if len(parts) == 2:
                return f"{proj_cost:.4f} EUR/kWh (grid {parts[0]}){parts[1]}"
            return entry.reason

        parts = entry.reason.split(" EUR/kWh")
        if len(parts) != 2:
            return entry.reason
        grid_price, rest = parts

        basis = getattr(entry, "value_basis", None) or "value"
        detail = f"grid {grid_price}"
        if proj_cost is not None:
            detail += f", stored {proj_cost:.4f}"
        line = f"{marginal:.4f} EUR/kWh {basis} ({detail}){rest}"
        if proj_cost is not None and abs(proj_cost) <= 0.0005:
            line += " [stored basis ~0: PV booked at export floor]"
        return line

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
            marginal = getattr(entry, "marginal_value_eur_kwh", None)
            schedule_data.append({
                "time": hour.isoformat(),
                "mode": entry.mode.name,
                "wit_mode": wit_mode,
                "wit_mode_name": display[0],
                "export": display[2],
                "reason": entry.reason,
                # Decision economics per battery DC kWh (reporting only)
                "value": round(marginal, 4) if marginal is not None else None,
                "value_basis": getattr(entry, "value_basis", None),
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
        lines.append("| Time | Mode | Export | SOC | Load | PV | Price | Value |")
        lines.append("|------|------|--------|-----|------|-----|-------|-------|")

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

            # Decision economics per battery DC kWh (reporting only). Distinct
            # from Price: it already includes conversion, wear and — for HOLD —
            # the end-of-horizon salvage value.
            marginal = getattr(entry, "marginal_value_eur_kwh", None)
            value_str = f"{marginal:.4f}" if marginal is not None else ""

            lines.append(
                f"| {time_str} | {icon} {mode_name} | {export_str} | {soc_str} | {load_str} | {pv_str} | {price_str} | {value_str} |"
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
