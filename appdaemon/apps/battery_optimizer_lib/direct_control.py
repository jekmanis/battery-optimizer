"""Direct inverter control via growatt_modbus/set_wit_mode service.

Sends mode commands to the WIT inverter via HA service calls
instead of writing TOU registers.
"""

import datetime
import enum
from typing import Optional

from .models import BatteryMode, ScheduleEntry

# Default entity id of the integration's "Inverter Mode" sensor
# (GrowattWitModeStatusSensor). Its friendly name is "<entry> Inverter Mode"
# and, because that sensor does NOT use has_entity_name, the entity_id is
# derived directly from that slugified name. The id therefore depends on the
# config entry NAME:
#   entry "Growatt"     -> sensor.growatt_inverter_mode      (this deployment)
#   entry "Growatt WIT" -> sensor.growatt_wit_inverter_mode
# This deployment's other entities are prefixed "growatt_" (e.g.
# sensor.growatt_battery_battery_soc), so the entry is named "Growatt" and the
# default below matches. If yours differs, set config.inverter_mode_sensor —
# a mismatch is surfaced by a WARNING the first time verification can't read
# the sensor.
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

# Default delay before verifying a sent mode against the inverter's reported
# status. The status sensor is recomputed on each coordinator poll (~30-60s), so
# 90s gives at least one poll cycle after the write settles. Overridable via
# config.verify_delay_seconds: a lagging sensor and a lost command look
# identical at a fixed delay.
VERIFY_DELAY_SECONDS = 90

# Default delay for the SINGLE re-check performed after a resend. Shorter than
# the first check — the resend goes out immediately, only one poll cycle is
# needed to see it.
VERIFY_RECHECK_SECONDS = 60

# Per-call websocket timeout for set_wit_mode (AppDaemon >= 4.4 HASS kwarg).
# The handler performs 6-9 sequential Modbus writes behind a shared lock and can
# exceed AppDaemon's 10s default. This call is SYNCHRONOUS on the callback
# thread, so every second here blocks every other callback of this app; the
# unconfirmed (None) path is safe because verify-after-set covers it. Overridable
# via config.set_wit_mode_timeout_seconds.
SET_WIT_MODE_TIMEOUT_SECONDS = 15


class ApplyOutcome(enum.Enum):
    """What actually happened to one ``apply_mode`` command.

    The boolean returned by ``apply_mode`` cannot separate these, and THREE of
    them are True: a dry run, a duplicate that was never transmitted, and a
    client-side timeout nobody confirmed. Treating all three as "the inverter
    obeyed" is what let a hung growatt_modbus publish climbing apply_successes
    while the "inverter is NOT following the schedule" escalation could never
    fire — every call timed out at ``hass_timeout``, logged a WARNING and
    returned True.
    """

    SENT = "sent"                        # confirmed by the service response
    UNCONFIRMED_TIMEOUT = "unconfirmed"  # client-side timeout, outcome unknown
    SKIPPED_DUPLICATE = "duplicate"      # identical command, nothing transmitted
    DRY_RUN = "dry_run"                  # device_id == "" — nothing transmitted
    FAILED = "failed"                    # confirmed failure

    @property
    def confirmed(self) -> bool:
        """True only when the inverter actually acknowledged the command."""
        return self is ApplyOutcome.SENT


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
        # Whether the "cannot verify — mode sensor unreadable" condition has
        # been logged at WARNING yet. First occurrence per app start is a
        # WARNING (a wrong entity id must be visible); the rest are DEBUG.
        self._verify_unreadable_warned = False

        # Timing (configurable — a lagging modbus sensor must be compensable
        # from apps.yaml, not by editing this module).
        self._verify_delay = int(
            getattr(config, "verify_delay_seconds", VERIFY_DELAY_SECONDS)
        )
        self._verify_recheck_delay = int(
            getattr(config, "verify_recheck_seconds", VERIFY_RECHECK_SECONDS)
        )
        self._set_mode_timeout = int(
            getattr(config, "set_wit_mode_timeout_seconds",
                    SET_WIT_MODE_TIMEOUT_SECONDS)
        )

        # Diagnostics counters. They exist to separate "the HA sensor lags"
        # (mismatch_count high, resend_recovered_count high, persistent 0) from
        # "the inverter really falls back to Passthrough" (persistent grows).
        self._mismatch_count = 0
        self._resend_count = 0
        self._resend_recovered_count = 0
        self._resend_failed_count = 0
        self._persistent_mismatch_count = 0
        self._unverifiable_count = 0
        self._verified_count = 0
        self._last_mismatch: Optional[dict] = None

        # Outcome of the last apply_mode command, and a tally per outcome.
        # These are what separate "the inverter acknowledged N commands" from
        # "N commands timed out unconfirmed" / "N were never transmitted".
        self.last_apply_outcome: Optional[ApplyOutcome] = None
        self._apply_outcome_counts: dict = {}

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

        Backward-compatible boolean wrapper around
        :meth:`apply_mode_with_outcome`: False only for a CONFIRMED failure.
        Callers that need to distinguish "the inverter acknowledged" from
        "nothing was transmitted" or "we never found out" must use
        ``apply_mode_with_outcome`` (or read ``last_apply_outcome``).

        Args:
            entry: Schedule entry with mode, and optional export_rate
                   and ac_charge_mode.

        Returns:
            True unless the service call confirmed a failure.
        """
        return self.apply_mode_with_outcome(entry) is not ApplyOutcome.FAILED

    def apply_mode_with_outcome(self, entry: ScheduleEntry) -> ApplyOutcome:
        """Send mode command to inverter and report what actually happened.

        Args:
            entry: Schedule entry with mode, and optional export_rate
                   and ac_charge_mode.

        Returns:
            The :class:`ApplyOutcome` for this command. Also stored on
            ``self.last_apply_outcome`` and counted in ``get_diagnostics()``.
        """
        if not self.device_id:
            self.app.log(
                f"DirectControl: dry-run {entry.mode.name} ({entry.reason})"
            )
            return self._record_outcome(ApplyOutcome.DRY_RUN)

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
            return self._record_outcome(ApplyOutcome.SKIPPED_DUPLICATE)

        # Supersede any verification pending from a previous send BEFORE we send
        # (regardless of this send's outcome). Otherwise a confirmed failure
        # here would return without cancelling, and the stale timer could later
        # resend the now-superseded older mode.
        self._cancel_verification()

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
            return self._record_outcome(ApplyOutcome.FAILED)

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
        return self._record_outcome(
            ApplyOutcome.SENT if outcome is True
            else ApplyOutcome.UNCONFIRMED_TIMEOUT
        )

    def _record_outcome(self, outcome: ApplyOutcome) -> ApplyOutcome:
        """Store and count one apply outcome, then return it unchanged."""
        self.last_apply_outcome = outcome
        self._apply_outcome_counts[outcome] = (
            self._apply_outcome_counts.get(outcome, 0) + 1
        )
        return outcome

    def release_control(self) -> bool:
        """Release all overrides — inverter reverts to base mode."""
        if not self.device_id:
            self.app.log("DirectControl: dry-run release_control (passthrough)")
            return True

        params = {"device_id": self.device_id, "mode": "passthrough"}

        # Cancel any pending verification from a previous send before sending,
        # so a failed release can't leave a stale timer that resends an older
        # mode ~90s later.
        self._cancel_verification()

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
                    websocket timeout regardless of how high hass_timeout is;
                    the command usually still executes on the inverter. We can't
                    tell slow from lost, so we defer to verify-after-set. This is
                    why a SHORT timeout is safe: the failure mode of a too-short
                    timeout is one extra verification, while a long one blocks
                    the whole app's callback thread.
        """
        try:
            # hass_timeout: a formal parameter of the AppDaemon HASS plugin
            #   (>= 4.4). It is consumed by the plugin, NOT forwarded as
            #   service data, and raises the 10s default to 30s.
            # No return_response/return_result kwarg is passed: set_wit_mode is
            #   registered SupportsResponse.OPTIONAL, and AppDaemon auto-enables
            #   return_response for such services, so call_service already
            #   surfaces the handler's response dict (or None on client-side
            #   timeout) and propagates handler exceptions. Passing an unknown
            #   kwarg like return_result would be forwarded to HA's strict
            #   voluptuous schema and fail every call with "extra keys not
            #   allowed".
            result = self.app.call_service(
                "growatt_modbus/set_wit_mode",
                hass_timeout=self._set_mode_timeout,
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

    def _schedule_verification(
        self, mode_str: str, params: dict, attempt: int = 1
    ) -> None:
        """Schedule a one-shot verification after a mode was sent.

        Supersedes any previously pending verification, so a mode applied
        between send and verify cancels the stale check.

        Args:
            attempt: 1 for the check after the original send (delayed by
                verify_delay_seconds), 2 for the single re-check after a resend
                (delayed by verify_recheck_seconds). Attempt is capped at 2 in
                _verify_mode, so this can never become a resend loop.
        """
        self._cancel_verification()
        delay = self._verify_delay if attempt <= 1 else self._verify_recheck_delay
        try:
            self._verify_timer = self.app.run_in(
                self._verify_mode,
                delay,
                mode_str=mode_str,
                params=params.copy(),
                attempt=attempt,
            )
        except Exception as e:
            self.app.log(
                f"DirectControl: could not schedule verification: {e}",
                level="ERROR",
            )
            self._verify_timer = None

    def _verify_mode(self, kwargs=None) -> None:
        """Verify the inverter reached the last-sent mode; resend once if not.

        AppDaemon scheduler callback: receives a single kwargs dict.

        Attempt ladder (bounded — never a loop):
          attempt 1: mismatch -> WARNING, resend once, schedule attempt 2
          attempt 2: match    -> INFO "recovered after resend"
                     mismatch -> ERROR, NO further resend, NO further timer

        The second check is what makes the diagnostics meaningful: without it we
        never learned whether the resend helped, so a lagging HA modbus sensor
        was indistinguishable from an inverter that genuinely drops back to
        Passthrough.
        """
        self._verify_timer = None
        kwargs = kwargs or {}
        mode_str = kwargs.get("mode_str")
        params = kwargs.get("params", {})
        attempt = int(kwargs.get("attempt", 1) or 1)

        try:
            expected = MODE_STATUS_MAP.get(mode_str)
            if expected is None:
                return  # Unknown mode string — nothing to verify against.

            entity = self._mode_status_entity()
            state = self.app.get_state(entity)

            if state is None or state in ("unknown", "unavailable"):
                # Cannot verify — don't resend blindly. Warn the FIRST time so a
                # wrong/misconfigured entity id is visible; stay DEBUG after that
                # to avoid log spam when the sensor is merely briefly offline.
                self._unverifiable_count += 1
                if not self._verify_unreadable_warned:
                    self._verify_unreadable_warned = True
                    self.app.log(
                        f"DirectControl: cannot verify {mode_str} — mode sensor "
                        f"'{entity}' is {state}. If this persists, check that "
                        "inverter_mode_sensor points at the integration's "
                        "Inverter Mode sensor (verification is disabled until "
                        "it reads a value).",
                        level="WARNING",
                    )
                else:
                    self.app.log(
                        f"DirectControl: cannot verify {mode_str} — "
                        f"{entity} is {state}",
                        level="DEBUG",
                    )
                return

            if str(state) == expected:
                self._verified_count += 1
                if attempt > 1:
                    self._resend_recovered_count += 1
                    self.app.log(
                        f"DirectControl: {mode_str} recovered after resend — "
                        f"inverter now reports '{state}'"
                    )
                else:
                    self.app.log(
                        f"DirectControl: verified {mode_str} — "
                        f"inverter reports '{state}'",
                        level="DEBUG",
                    )
                return

            self._mismatch_count += 1
            self._last_mismatch = {
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "mode": mode_str,
                "expected": expected,
                "actual": str(state),
                "attempt": attempt,
            }

            if attempt >= 2:
                # Already resent once and the inverter still disagrees. This is
                # no longer sensor lag — escalate and STOP (no third send, no
                # third timer).
                self._persistent_mismatch_count += 1
                self.app.log(
                    f"DirectControl: persistent mode mismatch after resend — "
                    f"expected '{expected}' for {mode_str}, inverter still "
                    f"reports '{state}'. The inverter is not honouring the "
                    f"command; not resending again (retry happens next slot).",
                    level="ERROR",
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
            self._resend_count += 1
            outcome = self._call_set_wit_mode(params)

            if outcome is False:
                self._resend_failed_count += 1
                self.app.log(
                    f"DirectControl: resend of {mode_str} failed",
                    level="ERROR",
                )
                return

            # Record last-sent again, then re-check exactly ONCE so we learn
            # whether the resend actually took effect.
            self._last_mode_sent = mode_str
            self._last_mode_time = datetime.datetime.now()
            self._last_params = params.copy()
            self._schedule_verification(mode_str, params, attempt=2)

        except Exception as e:
            self.app.log(
                f"DirectControl: verification error for {mode_str}: {e}",
                level="ERROR",
            )

    def get_diagnostics(self) -> dict:
        """Counters that make inverter-control health observable in HA.

        Interpretation:
          * mismatch_count high, resend_recovered_count ~= resend_count,
            persistent_mismatch_count == 0  -> the HA mode sensor merely LAGS.
            Raise verify_delay_seconds.
          * persistent_mismatch_count growing -> the inverter genuinely drops
            the override (e.g. back to Passthrough). A configuration/firmware
            problem, not a timing one.
          * resend_failed_count growing -> the set_wit_mode service itself is
            failing; check the Modbus connection.
          * unconfirmed_count growing while sent_count stays flat -> every
            set_wit_mode call is hitting its client-side timeout. That is a hung
            growatt_modbus, even though apply_mode keeps returning True.
        """
        counts = self._apply_outcome_counts
        return {
            "sent_count": counts.get(ApplyOutcome.SENT, 0),
            "unconfirmed_count": counts.get(ApplyOutcome.UNCONFIRMED_TIMEOUT, 0),
            "duplicate_skipped_count": counts.get(
                ApplyOutcome.SKIPPED_DUPLICATE, 0
            ),
            "dry_run_count": counts.get(ApplyOutcome.DRY_RUN, 0),
            "failed_count": counts.get(ApplyOutcome.FAILED, 0),
            "last_apply_outcome": (
                self.last_apply_outcome.value if self.last_apply_outcome else None
            ),
            "mismatch_count": self._mismatch_count,
            "resend_count": self._resend_count,
            "resend_recovered_count": self._resend_recovered_count,
            "resend_failed_count": self._resend_failed_count,
            "persistent_mismatch_count": self._persistent_mismatch_count,
            "unverifiable_count": self._unverifiable_count,
            "verified_count": self._verified_count,
            "last_mismatch": self._last_mismatch,
            "verify_delay_seconds": self._verify_delay,
            "verify_recheck_seconds": self._verify_recheck_delay,
            "set_wit_mode_timeout_seconds": self._set_mode_timeout,
        }

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
