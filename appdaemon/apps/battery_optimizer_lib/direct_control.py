"""Direct inverter control via growatt_modbus/set_wit_mode service.

Replaces tou_sync.py. Sends mode commands to the WIT inverter via HA
service calls instead of writing TOU registers.
"""

import datetime
from typing import Optional

from .models import BatteryMode, ScheduleEntry


class DirectControl:
    """Sends mode commands to WIT inverter via HA service calls."""

    def __init__(self, app, config):
        """
        Args:
            app: AppDaemon app instance (for call_service, get_state, log)
            config: BatteryOptimizerConfig instance
        """
        self.app = app
        self.config = config
        self._last_mode_sent: Optional[str] = None
        self._last_mode_time: Optional[datetime.datetime] = None
        self._last_params: dict = {}

    @property
    def device_id(self) -> str:
        return self.config.device_id

    def _duration_for_slot(self) -> int:
        """Override duration: slot_minutes + safety buffer.

        If the optimizer misses a refresh, the override expires and
        the inverter reverts to its panel-configured base mode.
        """
        return self.config.slot_minutes + self.config.direct_control_buffer_minutes

    def apply_mode(self, entry: ScheduleEntry) -> bool:
        """Send mode command to inverter via set_wit_mode service.

        Args:
            entry: Schedule entry with mode, and optional export_rate,
                   ac_charge_mode, power_percent, cutoff SOC values.

        Returns:
            True if service call succeeded (or dry-run), False otherwise.
        """
        if not self.device_id:
            self.app.log(
                f"DirectControl: dry-run {entry.mode.name} ({entry.reason})"
            )
            return True

        mode = entry.mode
        duration = self._duration_for_slot()

        mode_str = {
            BatteryMode.CHARGE: self._resolve_charge_mode(entry),
            BatteryMode.DISCHARGE: self._resolve_discharge_mode(entry),
            BatteryMode.HOLD: "hold",
        }.get(mode, "hold")

        params = {
            "device_id": self.device_id,
            "mode": mode_str,
            "duration_minutes": duration,
        }

        # Power percent
        power = entry.power_percent if entry.power_percent is not None else self.config.default_power_percent
        if mode_str not in ("self_consumption", "passthrough"):
            params["power_percent"] = power

        # Export rate
        if entry.export_rate is not None:
            params["export_rate"] = entry.export_rate

        # AC charge mode
        ac_mode = self._ac_charge_mode_for_entry(entry)
        if ac_mode:
            params["ac_charge_mode"] = ac_mode

        # SOC limits
        if entry.charge_cutoff_soc is not None:
            params["charge_cutoff_soc"] = entry.charge_cutoff_soc
        elif mode == BatteryMode.CHARGE:
            params["charge_cutoff_soc"] = self._get_max_soc()

        if entry.discharge_cutoff_soc is not None:
            params["discharge_cutoff_soc"] = entry.discharge_cutoff_soc
        elif mode == BatteryMode.DISCHARGE:
            params["discharge_cutoff_soc"] = self._get_min_soc()

        # Duplicate detection
        if self._is_duplicate(mode_str, params):
            self.app.log(
                f"DirectControl: skipping duplicate {mode_str} "
                f"(last sent {self._seconds_since_last():.0f}s ago)",
                level="DEBUG",
            )
            return True

        try:
            self.app.log(
                f"DirectControl: {mode_str} "
                f"power={params.get('power_percent', '-')}% "
                f"duration={duration}min "
                f"export={params.get('export_rate', '-')} "
                f"ac={params.get('ac_charge_mode', '-')} "
                f"soc=[{params.get('discharge_cutoff_soc', '-')}"
                f"-{params.get('charge_cutoff_soc', '-')}]"
            )

            self.app.call_service(
                "growatt_modbus/set_wit_mode", **params
            )

            self._last_mode_sent = mode_str
            self._last_mode_time = datetime.datetime.now()
            self._last_params = params.copy()
            return True

        except Exception as e:
            self.app.log(
                f"DirectControl: set_wit_mode failed: {e}",
                level="ERROR",
            )
            return False

    def release_control(self) -> bool:
        """Release all overrides — inverter reverts to base mode."""
        if not self.device_id:
            self.app.log("DirectControl: dry-run release_control (passthrough)")
            return True

        try:
            self.app.call_service(
                "growatt_modbus/set_wit_mode",
                device_id=self.device_id,
                mode="passthrough",
            )
            self._last_mode_sent = "passthrough"
            self._last_mode_time = datetime.datetime.now()
            self.app.log("DirectControl: released all overrides (passthrough)")
            return True
        except Exception as e:
            self.app.log(f"DirectControl: release failed: {e}", level="ERROR")
            return False

    def _ac_charge_mode_for_entry(self, entry: ScheduleEntry) -> str:
        """Determine AC charge mode based on entry and PV conditions."""
        if entry.ac_charge_mode is not None:
            return entry.ac_charge_mode

        if entry.mode == BatteryMode.CHARGE:
            pv_power = self._get_pv_power()
            if pv_power is not None and pv_power > self._get_pv_threshold():
                return "pv_priority"
            return "ac_priority"

        return "disabled"

    def _resolve_charge_mode(self, entry: ScheduleEntry) -> str:
        """Map CHARGE to service mode string."""
        return "grid_charge"

    def _resolve_discharge_mode(self, entry: ScheduleEntry) -> str:
        """Map DISCHARGE + export_rate to specific mode string."""
        export_rate = entry.export_rate

        if export_rate is not None and export_rate > 0:
            power = entry.power_percent if entry.power_percent is not None else self.config.default_power_percent
            if export_rate >= 100 and power >= 100:
                return "max_export"
            return "discharge_to_grid"
        # Default: no accidental export
        return "discharge_to_load"

    def _get_pv_power(self) -> Optional[float]:
        """Read current PV power from HA sensor."""
        try:
            state = self.app.get_state(self.config.pv_power_sensor)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return None

    def _get_pv_threshold(self) -> float:
        """Get current PV threshold from HA entity or config default."""
        try:
            state = self.app.get_state(self.config.pv_threshold_entity)
            if state and state not in ("unknown", "unavailable"):
                return float(state)
        except (ValueError, TypeError):
            pass
        return self.config.default_pv_threshold

    def _get_min_soc(self) -> int:
        """Get current min SOC from HA entity or config default."""
        try:
            state = self.app.get_state(self.config.min_soc_entity)
            if state and state not in ("unknown", "unavailable"):
                return int(float(state))
        except (ValueError, TypeError):
            pass
        return int(self.config.default_min_soc)

    def _get_max_soc(self) -> int:
        """Get current max SOC from HA entity or config default."""
        try:
            state = self.app.get_state(self.config.max_soc_entity)
            if state and state not in ("unknown", "unavailable"):
                return int(float(state))
        except (ValueError, TypeError):
            pass
        return int(self.config.default_max_soc)

    def _is_duplicate(self, mode_str: str, params: dict) -> bool:
        """Check if this command is identical to the last one sent recently."""
        if self._last_mode_sent != mode_str:
            return False
        if not self._last_mode_time:
            return False

        elapsed = (datetime.datetime.now() - self._last_mode_time).total_seconds()
        half_slot = self.config.slot_minutes * 60 / 2
        if elapsed > half_slot:
            return False  # Time to refresh even if same mode

        for key in ("mode", "power_percent", "export_rate", "ac_charge_mode",
                     "charge_cutoff_soc", "discharge_cutoff_soc"):
            if params.get(key) != self._last_params.get(key):
                return False

        return True

    def _seconds_since_last(self) -> float:
        if self._last_mode_time:
            return (datetime.datetime.now() - self._last_mode_time).total_seconds()
        return float('inf')
