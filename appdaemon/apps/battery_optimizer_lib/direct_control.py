"""Direct inverter control via growatt_modbus/set_wit_mode service.

Sends mode commands to the WIT inverter via HA service calls
instead of writing TOU registers.
"""

import datetime
from typing import Optional

from .models import BatteryMode, ScheduleEntry

# Default entity id of the integration's "Inverter Mode" sensor
# (GrowattWitModeStatusSensor). Its friendly name is "<entry> Inverter Mode"
# and, because that sensor does NOT use has_entity_name, the entity_id is
# derived directly from that name — e.g. "Growatt Inverter Mode" ->
# sensor.growatt_inverter_mode. Overridable via config.inverter_mode_sensor.
DEFAULT_MODE_STATUS_ENTITY = "sensor.growatt_inverter_mode"

# Expected value of the Inverter Mode sensor for each mode string we send.
# Mirrors _compute_wit_mode_status() in the integration's coordinator.
MODE_STATUS_MAP = {
    "grid_charge": "Grid Charge",
    "hold": "Preserve SOC",
    "preserve_soc": "Preserve SOC",
    "max_export": "Max Export",
    "discharge_to_grid": "Discharge to Grid",
    "discharge_to_load": "Discharge to Load",
    "passthrough": "Passthrough",
}

# Delay before verifying a sent mode against the inverter's reported status.
# The status sensor is recomputed on each coordinator poll (~30-60s), so 90s
# gives at least one poll cycle after the write settles.
VERIFY_DELAY_SECONDS = 90

# Per-call websocket timeout for set_wit_mode (AppDaemon >= 4.4 HASS kwarg).
# The handler performs 6-9 sequential Modbus writes behind a shared lock and
# can legitimately exceed AppDaemon's 10s default before HA finishes.
SET_WIT_MODE_TIMEOUT_SECONDS = 30


class DirectControl:
    """Sends mode commands to WIT inverter via HA service calls."""

    def __init__(self, app, config):
        """
        Args:
            app: AppDaemon app instance (for call_service, get_state, log,
                 run_in, cancel_timer)
            config: BatteryOptimizerConfig instance
        """
        self.app = app
        self.config = config
        self._last_mode_sent: Optional[str] = None
        self._last_mode_time: Optional[datetime.datetime] = None
        self._last_params: dict = {}
        # Handle of the pending one-shot verification timer (from run_in), or
        # None. Superseded whenever a new mode is applied.
        self._verify_timer = None

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
            entry: Schedule entry with mode, and optional export_rate
                   and ac_charge_mode.

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
        params["power_percent"] = self.config.default_power_percent

        # Export rate
        if entry.export_rate is not None:
            params["export_rate"] = entry.export_rate

        # AC charge mode
        ac_mode = self._ac_charge_mode_for_entry(entry)
        if ac_mode:
            params["ac_charge_mode"] = ac_mode

        # SOC limits
        if mode == BatteryMode.CHARGE:
            params["charge_cutoff_soc"] = self._get_max_soc()

        if mode == BatteryMode.DISCHARGE:
            params["discharge_cutoff_soc"] = self._get_min_soc()

        # Duplicate detection
        if self._is_duplicate(mode_str, params):
            self.app.log(
                f"DirectControl: skipping duplicate {mode_str} "
                f"(last sent {self._seconds_since_last():.0f}s ago)",
                level="DEBUG",
            )
            return True

        self.app.log(
            f"DirectControl: {mode_str} "
            f"power={params.get('power_percent', '-')}% "
            f"duration={duration}min "
            f"export={params.get('export_rate', '-')} "
            f"ac={params.get('ac_charge_mode', '-')} "
            f"soc=[{params.get('discharge_cutoff_soc', '-')}"
            f"-{params.get('charge_cutoff_soc', '-')}]"
        )

        outcome = self._call_set_wit_mode(params)

        if outcome is False:
            # Confirmed failure: do NOT record last-sent, so duplicate
            # suppression can't mask an immediate resend on the next slot.
            self.app.log(
                f"DirectControl: set_wit_mode reported failure for {mode_str}; "
                "not recording last-sent so a resend can correct it",
                level="ERROR",
            )
            return False

        # outcome is True (confirmed) or None (unconfirmed — client-side
        # timeout, command usually still applied). In both cases record the
        # last-sent marker so the schedule isn't spammed with resends;
        # verify-after-set (below) catches genuine losses.
        self._last_mode_sent = mode_str
        self._last_mode_time = datetime.datetime.now()
        self._last_params = params.copy()

        if outcome is None:
            self.app.log(
                f"DirectControl: {mode_str} unconfirmed (client-side timeout); "
                "will verify against inverter mode sensor",
                level="WARNING",
            )

        self._schedule_verification(mode_str, params)
        return True

    def release_control(self) -> bool:
        """Release all overrides — inverter reverts to base mode."""
        if not self.device_id:
            self.app.log("DirectControl: dry-run release_control (passthrough)")
            return True

        params = {"device_id": self.device_id, "mode": "passthrough"}

        outcome = self._call_set_wit_mode(params)

        if outcome is False:
            self.app.log("DirectControl: release failed", level="ERROR")
            return False

        self._last_mode_sent = "passthrough"
        self._last_mode_time = datetime.datetime.now()
        self._last_params = params.copy()

        if outcome is None:
            self.app.log(
                "DirectControl: passthrough unconfirmed (client-side timeout); "
                "will verify against inverter mode sensor",
                level="WARNING",
            )
        else:
            self.app.log("DirectControl: released all overrides (passthrough)")

        self._schedule_verification("passthrough", params)
        return True

    def _call_set_wit_mode(self, params: dict) -> Optional[bool]:
        """Call the set_wit_mode service and classify the outcome.

        Returns:
            True  - confirmed success (handler returned success=True, or a
                    truthy response we couldn't disprove)
            False - confirmed failure (handler raised -> exception here, or an
                    explicit success=False in the response)
            None  - unconfirmed. AppDaemon returns None on a client-side
                    websocket timeout even at hass_timeout=30; the command
                    usually still executes on the inverter. We can't tell slow
                    from lost, so we defer to verify-after-set.
        """
        try:
            # hass_timeout: consumed by the AppDaemon HASS plugin (>= 4.4),
            #   NOT forwarded as service data. Raises the 10s default to 30s.
            # return_result: makes AppDaemon wait for and surface the service
            #   response dict (set_wit_mode is SupportsResponse.OPTIONAL) and
            #   lets handler exceptions propagate so real failures are caught.
            result = self.app.call_service(
                "growatt_modbus/set_wit_mode",
                hass_timeout=SET_WIT_MODE_TIMEOUT_SECONDS,
                return_result=True,
                **params,
            )
        except Exception as e:
            self.app.log(
                f"DirectControl: set_wit_mode failed: {e}",
                level="ERROR",
            )
            return False

        if result is None:
            return None

        # Response shape from the handler on success:
        #   {"success": True, "mode_applied": ..., "registers_written": {...},
        #    "timestamp": ..., "override_expires": ...}
        # The handler raises on failure (caught above), so success=False is
        # unusual, but honour it if a future version returns it. Tolerate the
        # response being nested under a "result" key on some AD versions.
        if isinstance(result, dict):
            success = result.get("success")
            if success is None and isinstance(result.get("result"), dict):
                success = result["result"].get("success")
            if success is False:
                return False

        return True

    def _mode_status_entity(self) -> str:
        """Entity id of the inverter mode-status sensor used for verification."""
        return getattr(self.config, "inverter_mode_sensor", "") \
            or DEFAULT_MODE_STATUS_ENTITY

    def _cancel_verification(self) -> None:
        """Cancel any pending verification timer."""
        if self._verify_timer is not None:
            try:
                self.app.cancel_timer(self._verify_timer)
            except Exception:
                pass
            self._verify_timer = None

    def _schedule_verification(self, mode_str: str, params: dict) -> None:
        """Schedule a one-shot verification ~90s after a mode was sent.

        Supersedes any previously pending verification, so a mode applied
        between send and verify cancels the stale check.
        """
        self._cancel_verification()
        try:
            self._verify_timer = self.app.run_in(
                self._verify_mode,
                VERIFY_DELAY_SECONDS,
                mode_str=mode_str,
                params=params.copy(),
            )
        except Exception as e:
            self.app.log(
                f"DirectControl: could not schedule verification: {e}",
                level="ERROR",
            )
            self._verify_timer = None

    def _verify_mode(self, kwargs=None) -> None:
        """Verify the inverter reached the last-sent mode; resend once if not.

        AppDaemon scheduler callback: receives a single kwargs dict. Max one
        resend per apply_mode invocation (no re-verification of the resend),
        so a persistent mismatch can't loop.
        """
        self._verify_timer = None
        kwargs = kwargs or {}
        mode_str = kwargs.get("mode_str")
        params = kwargs.get("params", {})

        try:
            expected = MODE_STATUS_MAP.get(mode_str)
            if expected is None:
                return  # Unknown mode string — nothing to verify against.

            entity = self._mode_status_entity()
            state = self.app.get_state(entity)

            if state is None or state in ("unknown", "unavailable"):
                # Cannot verify — don't resend blindly.
                self.app.log(
                    f"DirectControl: cannot verify {mode_str} — "
                    f"{entity} is {state}",
                    level="DEBUG",
                )
                return

            if str(state) == expected:
                self.app.log(
                    f"DirectControl: verified {mode_str} — "
                    f"inverter reports '{state}'",
                    level="DEBUG",
                )
                return

            # Mismatch — resend the same params ONCE, bypassing duplicate
            # suppression by clearing the last-sent timestamp.
            self.app.log(
                f"DirectControl: mode mismatch — expected '{expected}' for "
                f"{mode_str}, inverter reports '{state}'; resending once",
                level="WARNING",
            )
            self._last_mode_time = None  # bypass _is_duplicate
            outcome = self._call_set_wit_mode(params)

            if outcome is False:
                self.app.log(
                    f"DirectControl: resend of {mode_str} failed",
                    level="ERROR",
                )
                return

            # Record last-sent again. Deliberately do NOT schedule another
            # verification — one resend per apply_mode invocation.
            self._last_mode_sent = mode_str
            self._last_mode_time = datetime.datetime.now()
            self._last_params = params.copy()

        except Exception as e:
            self.app.log(
                f"DirectControl: verification error for {mode_str}: {e}",
                level="ERROR",
            )

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
            power = self.config.default_power_percent
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

        for key in ("mode", "export_rate", "ac_charge_mode",
                     "charge_cutoff_soc", "discharge_cutoff_soc"):
            if params.get(key) != self._last_params.get(key):
                return False

        return True

    def _seconds_since_last(self) -> float:
        if self._last_mode_time:
            return (datetime.datetime.now() - self._last_mode_time).total_seconds()
        return float('inf')
