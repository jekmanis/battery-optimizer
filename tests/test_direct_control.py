"""Tests for DirectControl set_wit_mode reliability behavior.

Covers:
- AppDaemon 4.5.x timeout classification: an ``ad_status: TIMEOUT`` response is
  UNCONFIRMED, not a confirmed failure (the request was already on the wire)
- confirmed failure (exception or a genuine success=False) -> returns False,
  does NOT record last-sent, schedules no verification
- verify-after-set mismatch -> resends once (bypassing duplicate suppression),
  then re-checks exactly once; a persistent mismatch escalates to ERROR without
  ever sending a third time (bounded ladder, no resend loop)
- verify/timeout delays are configurable and counters are exposed
- verify-after-set match -> no resend
- verify-after-set unavailable sensor -> cannot verify, no resend
- pending verification timer is superseded when a new mode is applied
- verification is OFF unless a source is configured, and the comparison itself
  is a pluggable strategy

NOTE: ``make_dc()`` sets ``inverter_mode_sensor`` explicitly. DirectControl no
longer falls back to DEFAULT_MODE_STATUS_ENTITY, so an unset value means "no
verification at all" — see test_verification_disabled_when_no_sensor_configured.
"""

import datetime
import threading
import time

import pytest

from battery_optimizer_lib.config import BatteryOptimizerConfig
from battery_optimizer_lib.direct_control import (
    ApplyOutcome,
    DirectControl,
    DEFAULT_MODE_STATUS_ENTITY,
    ModeSensorVerifier,
    RegisterVerifier,
    VerificationOutcome,
    VerificationResult,
    decode_signed_power,
    decode_wit_mode_status,
    expected_registers,
)
from battery_optimizer_lib.models import BatteryMode, ScheduleEntry


class FakeApp:
    """Minimal AppDaemon app double exposing the methods DirectControl uses."""

    def __init__(self):
        # call_service behavior
        self.call_service_return = {"success": True}
        self.call_service_raise = None  # set to an Exception instance to raise
        self.service_calls = []  # list of (service, kwargs)

        # Concurrency instrumentation. AppDaemon dispatches this app's
        # callbacks across worker threads, so the double has to be able to
        # model a slow, overlapping inverter write.
        self.sleep_seconds = 0.0        # how long call_service blocks
        self.call_intervals = []        # [(monotonic_start, monotonic_end)]
        self.call_started = threading.Event()   # set when a call begins
        self._bookkeeping = threading.Lock()

        # get_state backing store: {entity_id: state}
        self.states = {}

        # scheduler
        self._next_handle = 0
        self.run_in_calls = []  # list of (callback, delay, kwargs, handle)
        self.cancelled = []  # list of cancelled handles

        # logging
        self.logs = []  # list of (message, level)

    # --- AppDaemon API surface used by DirectControl ---
    def call_service(self, service, **kwargs):
        started = time.monotonic()
        with self._bookkeeping:
            self.service_calls.append((service, kwargs))
        self.call_started.set()
        try:
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
            if self.call_service_raise is not None:
                raise self.call_service_raise
            return self.call_service_return
        finally:
            with self._bookkeeping:
                self.call_intervals.append((started, time.monotonic()))

    def get_state(self, entity):
        return self.states.get(entity)

    def run_in(self, callback, delay, **kwargs):
        with self._bookkeeping:
            self._next_handle += 1
            handle = f"timer_{self._next_handle}"
            self.run_in_calls.append((callback, delay, kwargs, handle))
        return handle

    def cancel_timer(self, handle):
        with self._bookkeeping:
            self.cancelled.append(handle)

    def log(self, message, level="INFO"):
        with self._bookkeeping:
            self.logs.append((message, level))

    # --- concurrency test helpers ---
    def overlapping_calls(self):
        """Pairs of service calls whose execution windows overlapped."""
        spans = sorted(self.call_intervals)
        return [
            (spans[i], spans[i + 1])
            for i in range(len(spans) - 1)
            if spans[i][1] > spans[i + 1][0]
        ]

    # --- test helpers ---
    def levels(self):
        return [lvl for _, lvl in self.logs]

    def fire_last_timer(self):
        """Invoke the most recently scheduled run_in callback."""
        callback, _delay, kwargs, _handle = self.run_in_calls[-1]
        callback(kwargs)


def make_dc(device_id="dev123", verify_source="mode_sensor", **overrides):
    """DirectControl wired to the mode-sensor strategy unless told otherwise.

    Two things are opt-in now and both are named explicitly here:

    * verification needs a SOURCE — without one there is no verifier and no
      timers at all;
    * ``verify_source`` picks WHICH source. The production default is
      ``"registers"``; these legacy tests pin ``"mode_sensor"`` so they keep
      exercising that strategy's ladder. Register tests pass
      ``verify_source="registers"`` and use ``FakeApp.registers``.

    ``verify_source`` is set on the config object rather than passed to the
    constructor because config.py is owned by another agent — see the proposed
    diff in the report.
    """
    overrides.setdefault("inverter_mode_sensor", DEFAULT_MODE_STATUS_ENTITY)
    verify_enabled = overrides.pop("verify_enabled", True)
    config = BatteryOptimizerConfig(device_id=device_id, **overrides)
    config.verify_source = verify_source
    app = FakeApp()
    return DirectControl(app, config, verify_enabled=verify_enabled), app


def hold_entry():
    return ScheduleEntry(
        time=datetime.datetime(2024, 1, 1, 12, 0, 0),
        mode=BatteryMode.HOLD,
        reason="test",
    )


def charge_entry():
    return ScheduleEntry(
        time=datetime.datetime(2024, 1, 1, 12, 0, 0),
        mode=BatteryMode.CHARGE,
        reason="test",
    )


# ---------------------------------------------------------------------------
# Task 1: timeout / failure detection
# ---------------------------------------------------------------------------

def test_timeout_none_is_unconfirmed_but_records_and_schedules():
    """call_service returning None (client-side timeout) -> unconfirmed."""
    dc, app = make_dc()
    app.call_service_return = None

    result = dc.apply_mode(hold_entry())

    assert result is True
    # hass_timeout was passed on the service call, at the CONFIGURED value.
    # It is deliberately not hard-coded any more: the call is synchronous on the
    # AppDaemon callback thread, so the timeout is a blocking budget.
    assert app.service_calls[0][1].get("hass_timeout") == dc._set_mode_timeout
    assert dc._set_mode_timeout == BatteryOptimizerConfig().set_wit_mode_timeout_seconds
    # last-sent recorded so the schedule isn't spammed
    assert dc._last_mode_sent == "hold"
    assert dc._last_mode_time is not None
    # WARNING about unconfirmed state
    assert "WARNING" in app.levels()
    # verification scheduled
    assert len(app.run_in_calls) == 1
    assert app.run_in_calls[0][1] == dc._verify_delay
    assert app.run_in_calls[0][2]["attempt"] == 1


def test_failure_exception_returns_false_and_does_not_record():
    """Handler raising -> exception here -> confirmed failure."""
    dc, app = make_dc()
    app.call_service_raise = RuntimeError("boom")

    result = dc.apply_mode(hold_entry())

    assert result is False
    assert dc._last_mode_sent is None  # not recorded -> resend not suppressed
    assert dc._last_mode_time is None
    assert len(app.run_in_calls) == 0  # no verification scheduled
    assert "ERROR" in app.levels()


def test_failure_success_false_returns_false_and_does_not_record():
    """Explicit success=False in the response dict -> confirmed failure."""
    dc, app = make_dc()
    app.call_service_return = {"success": False, "error": "nope"}

    result = dc.apply_mode(hold_entry())

    assert result is False
    assert dc._last_mode_sent is None
    assert len(app.run_in_calls) == 0
    assert "ERROR" in app.levels()


def test_service_call_kwargs_only_hass_timeout_plus_schema_fields():
    """No return_result/return_response key; only schema-valid service fields.

    An unknown kwarg would be forwarded to HA's strict voluptuous schema and
    fail every set_wit_mode call with "extra keys not allowed".
    """
    dc, app = make_dc()
    dc.apply_mode(hold_entry())

    service, kwargs = app.service_calls[0]
    assert service == "growatt_modbus/set_wit_mode"
    # The legacy/incorrect kwargs must never be present.
    assert "return_result" not in kwargs
    assert "return_response" not in kwargs
    # hass_timeout is the only non-service-data kwarg (consumed by the plugin).
    assert kwargs.get("hass_timeout") == dc._set_mode_timeout
    # Everything else must be a field the set_wit_mode voluptuous schema allows.
    allowed = {
        "hass_timeout",  # AppDaemon HASS plugin formal parameter
        "device_id", "mode", "power_percent", "duration_minutes",
        "export_rate", "ac_charge_mode", "charge_cutoff_soc",
        "discharge_cutoff_soc",
    }
    assert set(kwargs).issubset(allowed), f"unexpected kwargs: {set(kwargs) - allowed}"


def test_first_unverifiable_logs_warning_then_debug():
    """First cannot-verify occurrence is WARNING; subsequent ones are DEBUG."""
    dc, app = make_dc()
    # mode sensor absent -> get_state returns None -> cannot verify

    dc.apply_mode(hold_entry())
    app.fire_last_timer()  # first verification: sensor unreadable

    cannot_verify_warnings = [
        m for m, lvl in app.logs
        if lvl == "WARNING" and "cannot verify" in m
    ]
    assert len(cannot_verify_warnings) == 1

    # A second send + verify with the sensor still unreadable stays at DEBUG.
    logs_before = len(app.logs)
    dc.apply_mode(charge_entry())  # different mode -> not a duplicate
    app.fire_last_timer()
    new_logs = app.logs[logs_before:]
    assert not any(
        lvl == "WARNING" and "cannot verify" in m for m, lvl in new_logs
    )
    assert any(
        lvl == "DEBUG" and "cannot verify" in m for m, lvl in new_logs
    )


def test_failed_apply_cancels_pending_verification_timer():
    """A confirmed-failure send cancels a timer from a previous good send."""
    dc, app = make_dc()

    # First send succeeds and schedules timer_1.
    dc.apply_mode(hold_entry())
    first_handle = app.run_in_calls[0][3]
    assert len(app.run_in_calls) == 1

    # Second send (different mode) fails -> must cancel the stale timer and
    # NOT schedule a new one.
    app.call_service_raise = RuntimeError("boom")
    result = dc.apply_mode(charge_entry())

    assert result is False
    assert first_handle in app.cancelled
    assert len(app.run_in_calls) == 1  # no new verification scheduled


def test_failed_release_cancels_pending_verification_timer():
    """A failed release_control also cancels a previously pending timer."""
    dc, app = make_dc()

    dc.apply_mode(hold_entry())  # schedules timer_1
    first_handle = app.run_in_calls[0][3]

    app.call_service_raise = RuntimeError("boom")
    result = dc.release_control()

    assert result is False
    assert first_handle in app.cancelled
    assert len(app.run_in_calls) == 1


def test_success_records_and_schedules_verification():
    """Confirmed success records last-sent and schedules verification."""
    dc, app = make_dc()
    app.call_service_return = {"success": True, "mode_applied": "hold"}

    result = dc.apply_mode(hold_entry())

    assert result is True
    assert dc._last_mode_sent == "hold"
    assert len(app.run_in_calls) == 1


# ---------------------------------------------------------------------------
# Task 2: verify-after-set
# ---------------------------------------------------------------------------

def test_verify_match_does_not_resend():
    """Sensor reports the expected status -> no resend."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Preserve SOC"  # hold -> Preserve SOC

    dc.apply_mode(hold_entry())
    calls_before = len(app.service_calls)

    app.fire_last_timer()

    assert len(app.service_calls) == calls_before  # no extra service call


def test_verify_mismatch_resends_once_and_rechecks():
    """Sensor reports a different status -> resend once, then re-check ONCE.

    Changed semantics (was: resend and never look again). Without the re-check
    the log could never distinguish "the HA modbus sensor merely lagged" from
    "the inverter really dropped back to Passthrough" — the production log shows
    30 mismatches with no evidence either way.
    """
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"  # expected Preserve SOC

    dc.apply_mode(hold_entry())
    calls_before = len(app.service_calls)

    app.fire_last_timer()

    # exactly one additional set_wit_mode call (the resend)
    assert len(app.service_calls) == calls_before + 1
    assert "WARNING" in app.levels()
    # exactly one follow-up verification, at the (shorter) re-check delay
    assert len(app.run_in_calls) == 2
    assert app.run_in_calls[1][2]["attempt"] == 2
    assert app.run_in_calls[1][1] == dc._verify_recheck_delay
    assert dc.get_diagnostics()["mismatch_count"] == 1
    assert dc.get_diagnostics()["resend_count"] == 1


def test_second_verification_after_resend_matches_logs_recovery():
    """The sensor catches up after the resend -> recovery is recorded."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.apply_mode(hold_entry())
    app.fire_last_timer()                       # attempt 1: mismatch -> resend
    calls_after_resend = len(app.service_calls)

    # Sensor now reflects the mode (it was simply lagging the coordinator poll).
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Preserve SOC"
    app.fire_last_timer()                       # attempt 2: match

    assert len(app.service_calls) == calls_after_resend   # no third send
    assert len(app.run_in_calls) == 2                     # no third timer
    diag = dc.get_diagnostics()
    assert diag["resend_recovered_count"] == 1
    assert diag["persistent_mismatch_count"] == 0
    assert any("recovered after resend" in m for m, _ in app.logs)


def test_persistent_mismatch_escalates_to_error_and_does_not_loop():
    """Sensor never agrees -> ERROR after the re-check, and the ladder STOPS."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.apply_mode(hold_entry())
    sends_after_apply = len(app.service_calls)
    app.fire_last_timer()   # attempt 1: mismatch -> resend + schedule attempt 2
    app.fire_last_timer()   # attempt 2: still mismatch -> ERROR, stop

    # Exactly two sends in total: the original and one resend.
    assert len(app.service_calls) == sends_after_apply + 1
    # Exactly two timers: the first check and the single re-check.
    assert len(app.run_in_calls) == 2
    diag = dc.get_diagnostics()
    assert diag["persistent_mismatch_count"] == 1
    assert diag["mismatch_count"] == 2
    assert diag["resend_recovered_count"] == 0
    assert any(
        lvl == "ERROR" and "persistent mode mismatch" in m for m, lvl in app.logs
    )
    assert diag["last_mismatch"]["actual"] == "Passthrough"
    assert diag["last_mismatch"]["attempt"] == 2


def test_failed_resend_is_counted_and_stops_the_ladder():
    """A resend that fails outright is counted and does not re-check."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.apply_mode(hold_entry())
    app.call_service_raise = RuntimeError("boom")
    app.fire_last_timer()

    assert dc.get_diagnostics()["resend_failed_count"] == 1
    assert len(app.run_in_calls) == 1  # no re-check after a failed resend
    assert any(lvl == "ERROR" and "resend of hold failed" in m for m, lvl in app.logs)


def test_verify_delay_and_timeout_are_configurable():
    """apps.yaml can compensate a lagging modbus sensor without a code change."""
    dc, app = make_dc(
        verify_delay_seconds=30,
        verify_recheck_seconds=20,
        set_wit_mode_timeout_seconds=10,
    )
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.apply_mode(hold_entry())

    assert app.service_calls[0][1]["hass_timeout"] == 10
    assert app.run_in_calls[0][1] == 30

    app.fire_last_timer()
    assert app.run_in_calls[1][1] == 20


def test_get_diagnostics_shape():
    """The diagnostics sensor payload is stable and starts at zero."""
    dc, _app = make_dc()
    diag = dc.get_diagnostics()

    expected_keys = {
        "mismatch_count", "resend_count", "resend_recovered_count",
        "resend_failed_count", "persistent_mismatch_count",
        "unverifiable_count", "verified_count", "last_mismatch",
        "verify_delay_seconds", "verify_recheck_seconds",
        "set_wit_mode_timeout_seconds",
        # Per-outcome tally: a dry run, a suppressed duplicate and an
        # unconfirmed timeout are all "True" from apply_mode, and only
        # sent_count means the inverter acknowledged anything.
        "sent_count", "unconfirmed_count", "duplicate_skipped_count",
        "dry_run_count", "failed_count", "last_apply_outcome",
        # Which source (if any) the mismatch counters were produced by. Without
        # it the 2026-09-02 log's 73 mismatches were unattributable.
        "verification_enabled", "verification_source",
        # Matches that would have happened anyway (passthrough releases).
        "unprobative_match_count",
        # Timer identity under multi-thread dispatch (cancel_timer cannot stop
        # a callback another worker has already dequeued).
        "verify_generation",
    }
    assert set(diag) == expected_keys
    assert diag["last_mismatch"] is None
    assert diag["last_apply_outcome"] is None
    assert diag["verification_enabled"] is True
    assert diag["verification_source"] == f"mode_sensor:{DEFAULT_MODE_STATUS_ENTITY}"
    assert all(
        diag[k] == 0 for k in expected_keys
        if k.endswith("_count")
    )


def test_unverifiable_reads_are_counted():
    """A sensor that can't be read is counted separately from a mismatch."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "unavailable"

    dc.apply_mode(hold_entry())
    app.fire_last_timer()

    diag = dc.get_diagnostics()
    assert diag["unverifiable_count"] == 1
    assert diag["mismatch_count"] == 0
    assert diag["resend_count"] == 0


def test_verify_mismatch_bypasses_duplicate_suppression():
    """The resend goes out even though params match the last send."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.apply_mode(hold_entry())
    # last_mode_time is recent -> _is_duplicate would normally suppress.
    assert dc._is_duplicate("hold", dc._last_params) is True

    calls_before = len(app.service_calls)
    app.fire_last_timer()
    assert len(app.service_calls) == calls_before + 1


def test_verify_unavailable_sensor_does_not_resend():
    """Unavailable/unknown sensor -> cannot verify -> no resend."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "unavailable"

    dc.apply_mode(hold_entry())
    calls_before = len(app.service_calls)

    app.fire_last_timer()

    assert len(app.service_calls) == calls_before
    # First unreadable occurrence surfaces at WARNING (see dedicated test for
    # the WARNING-then-DEBUG progression).
    assert "WARNING" in app.levels()


def test_verify_missing_sensor_state_none_does_not_crash():
    """get_state returns None -> cannot verify, no crash, no resend."""
    dc, app = make_dc()
    # DEFAULT_MODE_STATUS_ENTITY not in states -> get_state returns None

    dc.apply_mode(hold_entry())
    calls_before = len(app.service_calls)

    app.fire_last_timer()

    assert len(app.service_calls) == calls_before


def test_verify_uses_configured_entity_when_set():
    """A configured inverter_mode_sensor overrides the default entity id."""
    dc, app = make_dc(inverter_mode_sensor="sensor.custom_mode")
    app.states["sensor.custom_mode"] = "Preserve SOC"

    dc.apply_mode(hold_entry())
    calls_before = len(app.service_calls)
    app.fire_last_timer()

    assert len(app.service_calls) == calls_before  # matched -> no resend


# ---------------------------------------------------------------------------
# Timer superseding
# ---------------------------------------------------------------------------

def test_new_apply_supersedes_pending_verification():
    """Applying a new mode cancels the previous verification timer."""
    dc, app = make_dc()

    dc.apply_mode(hold_entry())          # schedules timer_1
    first_handle = app.run_in_calls[0][3]

    dc.apply_mode(charge_entry())        # different mode -> not a duplicate

    assert first_handle in app.cancelled
    assert len(app.run_in_calls) == 2    # a fresh timer for the new mode


# ---------------------------------------------------------------------------
# release_control (passthrough) also verifies
# ---------------------------------------------------------------------------

def test_release_control_schedules_verification():
    dc, app = make_dc()
    app.call_service_return = {"success": True}

    result = dc.release_control()

    assert result is True
    assert dc._last_mode_sent == "passthrough"
    assert len(app.run_in_calls) == 1
    assert app.run_in_calls[0][2]["mode_str"] == "passthrough"


def test_release_control_timeout_none_is_unconfirmed():
    dc, app = make_dc()
    app.call_service_return = None

    result = dc.release_control()

    assert result is True
    assert dc._last_mode_sent == "passthrough"
    assert "WARNING" in app.levels()
    assert len(app.run_in_calls) == 1


def test_release_control_failure_returns_false():
    dc, app = make_dc()
    app.call_service_raise = RuntimeError("boom")

    result = dc.release_control()

    assert result is False
    assert dc._last_mode_sent is None
    assert len(app.run_in_calls) == 0


# ---------------------------------------------------------------------------
# apply_mode_with_outcome: the boolean's three flavours of "True"
#
# apply_mode returns True for a dry run, for a duplicate that was never
# transmitted and for an unconfirmed client-side timeout. The orchestrator used
# that boolean for health accounting, so a hung growatt_modbus (timeouts, no
# exception) published climbing apply_successes and the "inverter is NOT
# following the schedule" escalation could never fire.
# ---------------------------------------------------------------------------

def test_outcome_confirmed_send_is_sent():
    dc, app = make_dc()
    app.call_service_return = {"success": True}

    assert dc.apply_mode_with_outcome(hold_entry()) is ApplyOutcome.SENT
    assert dc.last_apply_outcome is ApplyOutcome.SENT
    assert ApplyOutcome.SENT.confirmed is True
    assert dc.get_diagnostics()["sent_count"] == 1
    assert dc.get_diagnostics()["last_apply_outcome"] == "sent"


def test_outcome_timeout_is_unconfirmed_not_sent():
    dc, app = make_dc()
    app.call_service_return = None

    outcome = dc.apply_mode_with_outcome(hold_entry())

    assert outcome is ApplyOutcome.UNCONFIRMED_TIMEOUT
    assert outcome.confirmed is False
    # Backward compatible: still "not a failure" for the boolean caller.
    assert dc.apply_mode(charge_entry()) is True
    assert dc.get_diagnostics()["unconfirmed_count"] == 2
    assert dc.get_diagnostics()["sent_count"] == 0


def test_outcome_duplicate_is_skipped_and_nothing_is_transmitted():
    dc, app = make_dc()

    assert dc.apply_mode_with_outcome(hold_entry()) is ApplyOutcome.SENT
    assert (dc.apply_mode_with_outcome(hold_entry())
            is ApplyOutcome.SKIPPED_DUPLICATE)

    assert len(app.service_calls) == 1  # the duplicate never went out
    assert dc.get_diagnostics()["duplicate_skipped_count"] == 1


def test_outcome_dry_run_when_no_device_id():
    dc, app = make_dc(device_id="")

    assert dc.apply_mode_with_outcome(hold_entry()) is ApplyOutcome.DRY_RUN

    assert app.service_calls == []
    assert dc.get_diagnostics()["dry_run_count"] == 1
    assert dc.get_diagnostics()["sent_count"] == 0


def test_outcome_success_false_response_is_failed():
    dc, app = make_dc()
    app.call_service_return = {"success": False, "error": "nope"}

    assert dc.apply_mode_with_outcome(hold_entry()) is ApplyOutcome.FAILED
    assert dc.apply_mode(hold_entry()) is False  # boolean wrapper agrees
    assert dc.get_diagnostics()["failed_count"] == 2


def test_outcome_exception_is_failed():
    dc, app = make_dc()
    app.call_service_raise = RuntimeError("boom")

    assert dc.apply_mode_with_outcome(hold_entry()) is ApplyOutcome.FAILED
    assert dc.get_diagnostics()["failed_count"] == 1


# ---------------------------------------------------------------------------
# Defect A: verification is opt-in, and the comparison is pluggable
#
# 2026-09-02 production log: 73/73 verifications read 'Passthrough' from
# sensor.growatt_inverter_mode while the battery physically executed every
# command (grid charge 9 -> 21 % SOC, max_export 0.7 kWh in 4 min, hold flat for
# 27 min under load). Every mismatch was false and every resend blocked the
# single AppDaemon thread for ~10 s — 36 useless inverter writes in 9 h.
# ---------------------------------------------------------------------------

def test_verification_disabled_when_no_sensor_configured():
    """Empty inverter_mode_sensor DISABLES verification (no silent fallback).

    It used to fall back to DEFAULT_MODE_STATUS_ENTITY, so every deployment
    verified against an entity that may not reflect the override at all.
    """
    dc, app = make_dc(inverter_mode_sensor="")

    assert dc.verification_enabled is False
    assert dc.verification_source is None

    dc.apply_mode(hold_entry())

    assert len(app.service_calls) == 1   # the command still goes out
    assert app.run_in_calls == []        # but nothing is scheduled to check it
    diag = dc.get_diagnostics()
    assert diag["verification_enabled"] is False
    assert diag["verification_source"] is None


def test_verification_disabled_by_flag_even_with_a_sensor():
    """verify_enabled=False is a master switch, independent of the sensor."""
    dc, app = make_dc(verify_enabled=False)
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    assert dc.verification_enabled is False

    dc.apply_mode(hold_entry())

    assert app.run_in_calls == []
    assert dc.get_diagnostics()["mismatch_count"] == 0


def test_disabled_verification_says_so_on_an_unconfirmed_send():
    """The unconfirmed WARNING must not promise a check that will not happen."""
    dc, app = make_dc(inverter_mode_sensor="")
    app.call_service_return = {"success": False, "ad_status": "TIMEOUT"}

    dc.apply_mode(hold_entry())

    warnings = [m for m, lvl in app.logs if lvl == "WARNING"]
    assert any("verification is disabled" in m for m in warnings)


def test_custom_verifier_strategy_is_used():
    """Any callable returning a VerificationOutcome plugs into the ladder."""
    seen = []

    def physical_verifier(mode_str, params):
        seen.append((mode_str, params["mode"]))
        return VerificationOutcome(
            result=VerificationResult.MATCH,
            source="soc_slope",
            observed="+3.2 %/15min",
            expected="rising",
        )

    config = BatteryOptimizerConfig(device_id="dev123")
    config.verify_source = "none"  # an explicit verifier= always wins
    app = FakeApp()
    dc = DirectControl(app, config, verifier=physical_verifier)

    assert dc.verification_enabled is True
    # A bare function declares no `.source`, so before it has run the callable's
    # own name identifies it in the diagnostics.
    assert dc.verification_source == "physical_verifier"

    dc.apply_mode(charge_entry())
    app.fire_last_timer()

    assert seen == [("grid_charge", "grid_charge")]
    assert dc.get_diagnostics()["verified_count"] == 1
    # After it has run, the source it actually reported is what shows up.
    assert dc.verification_source == "soc_slope"
    assert dc.get_diagnostics()["verification_source"] == "soc_slope"
    assert any("soc_slope reports '+3.2 %/15min'" in m for m, _ in app.logs)


def test_custom_verifier_mismatch_drives_the_same_bounded_ladder():
    """A non-sensor strategy gets exactly the documented 2 checks / 2 sends."""
    def always_mismatch(mode_str, params):
        return VerificationOutcome(
            result=VerificationResult.MISMATCH,
            source="register:0x1234",
            observed="0",
            expected="1",
        )

    config = BatteryOptimizerConfig(device_id="dev123")
    config.verify_source = "none"  # an explicit verifier= always wins
    app = FakeApp()
    dc = DirectControl(app, config, verifier=always_mismatch)

    dc.apply_mode(hold_entry())
    app.fire_last_timer()   # attempt 1 -> resend
    app.fire_last_timer()   # attempt 2 -> ERROR, stop

    assert len(app.service_calls) == 2   # original + one resend, never a third
    assert len(app.run_in_calls) == 2    # first check + single re-check
    assert dc.get_diagnostics()["persistent_mismatch_count"] == 1
    assert dc.get_diagnostics()["last_mismatch"]["source"] == "register:0x1234"


def test_set_verifier_swaps_strategy_and_cancels_pending_check():
    """A pending timer must not be evaluated by a strategy it was not made for."""
    dc, app = make_dc()
    dc.apply_mode(hold_entry())
    pending = app.run_in_calls[0][3]

    dc.set_verifier(None)

    assert pending in app.cancelled
    assert dc.verification_enabled is False


def test_mismatch_log_names_the_entity_and_the_raw_state():
    """A mismatch line must be diagnosable without guessing the source."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.apply_mode(hold_entry())
    app.fire_last_timer()

    warn = [m for m, lvl in app.logs if lvl == "WARNING" and "mismatch" in m]
    assert len(warn) == 1
    assert DEFAULT_MODE_STATUS_ENTITY in warn[0]   # which entity
    assert "'Passthrough'" in warn[0]              # what it actually said
    assert "'Preserve SOC'" in warn[0]             # what we expected


def test_persistent_mismatch_error_states_both_possible_causes():
    """The ERROR must not assert an inverter fault it cannot observe."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.apply_mode(hold_entry())
    app.fire_last_timer()
    app.fire_last_timer()

    errors = [m for m, lvl in app.logs if lvl == "ERROR"]
    assert len(errors) == 1
    msg = errors[0]
    # The old wording — "The inverter is not honouring the command" — asserted
    # one of two indistinguishable causes as fact.
    assert "is not honouring the command" not in msg
    assert "does not track the override" in msg
    assert DEFAULT_MODE_STATUS_ENTITY in msg
    assert "'Passthrough'" in msg


def test_verify_duration_is_reported_to_the_app_when_available():
    """_verify_mode's wall time reaches the app's slow-callback accounting."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"
    recorded = []
    app.record_external_callback_duration = (
        lambda name, seconds: recorded.append((name, seconds))
    )

    dc.apply_mode(hold_entry())
    app.fire_last_timer()

    assert len(recorded) == 1
    assert recorded[0][0] == "DirectControl._verify_mode"
    assert recorded[0][1] >= 0.0


def test_verify_duration_reporting_is_optional():
    """DirectControl still works against an app without the hook (tests/mocks)."""
    dc, app = make_dc()
    assert not hasattr(app, "record_external_callback_duration")

    dc.apply_mode(hold_entry())
    app.fire_last_timer()  # must not raise


def test_verify_duration_reported_even_when_verification_raises():
    """The timing report is in a finally: a broken strategy still gets counted."""
    def exploding(mode_str, params):
        raise RuntimeError("probe blew up")

    config = BatteryOptimizerConfig(device_id="dev123")
    app = FakeApp()
    recorded = []
    app.record_external_callback_duration = (
        lambda name, seconds: recorded.append(name)
    )
    dc = DirectControl(app, config, verifier=exploding)

    dc.apply_mode(hold_entry())
    app.fire_last_timer()

    assert recorded == ["DirectControl._verify_mode"]
    assert any(lvl == "ERROR" and "verification error" in m for m, lvl in app.logs)


# ---------------------------------------------------------------------------
# Defect B: AppDaemon 4.5.x timeout classification
#
# HassPlugin.websocket_send_json awaits the response future AFTER writing the
# request to the websocket. On asyncio.TimeoutError it logs
# "Timed out [0:00:15] waiting for request: {'type': 'call_service' ...}" and
# RETURNS {"success": False, "ad_status": "TIMEOUT", "ad_duration": ...} — it
# does not raise and does not return None. Reading only `success` therefore
# classified all four 2026-09-02 timeouts as confirmed failures, while the
# "Request already timed out for <id>" pairs prove HA did answer.
# ---------------------------------------------------------------------------

def test_ad_status_timeout_is_unconfirmed_not_failed():
    dc, app = make_dc()
    app.call_service_return = {
        "success": False, "ad_status": "TIMEOUT", "ad_duration": 15.02,
    }

    outcome = dc.apply_mode_with_outcome(hold_entry())

    assert outcome is ApplyOutcome.UNCONFIRMED_TIMEOUT
    assert dc.apply_mode(charge_entry()) is True  # boolean caller: not a failure
    diag = dc.get_diagnostics()
    assert diag["unconfirmed_count"] == 2
    assert diag["failed_count"] == 0


def test_ad_status_timeout_records_last_sent_and_verifies():
    """Unconfirmed means probably-applied: record it, then let verify decide."""
    dc, app = make_dc()
    app.call_service_return = {"success": False, "ad_status": "TIMEOUT"}

    dc.apply_mode(hold_entry())

    assert dc._last_mode_sent == "hold"
    assert dc._last_mode_time is not None
    assert len(app.run_in_calls) == 1
    warnings = [m for m, lvl in app.logs if lvl == "WARNING"]
    assert any("unconfirmed" in m for m in warnings)
    # It must NOT be logged as a confirmed failure.
    assert not any("reported failure" in m for m, _ in app.logs)


def test_ad_status_terminating_is_unconfirmed():
    """Cancelled during AppDaemon shutdown — the request was still on the wire."""
    dc, app = make_dc()
    app.call_service_return = {"success": False, "ad_status": "TERMINATING"}

    assert dc.apply_mode_with_outcome(hold_entry()) is ApplyOutcome.UNCONFIRMED_TIMEOUT


def test_ad_status_ok_with_success_false_is_still_a_failure():
    """A genuine HA error response must stay FAILED."""
    dc, app = make_dc()
    app.call_service_return = {
        "success": False,
        "ad_status": "OK",
        "error": {"code": "unknown_error", "message": "modbus write refused"},
    }

    assert dc.apply_mode_with_outcome(hold_entry()) is ApplyOutcome.FAILED
    assert dc._last_mode_sent is None
    assert app.run_in_calls == []
    errors = [m for m, lvl in app.logs if lvl == "ERROR"]
    assert any("modbus write refused" in m for m in errors)


def test_connection_error_exception_is_still_a_failure():
    """Connection refused / handler raised -> confirmed failure, unchanged."""
    dc, app = make_dc()
    app.call_service_raise = ConnectionRefusedError("no route to inverter")

    assert dc.apply_mode_with_outcome(hold_entry()) is ApplyOutcome.FAILED
    assert dc.get_diagnostics()["failed_count"] == 1


def test_ad_status_timeout_on_release_is_unconfirmed():
    dc, app = make_dc()
    app.call_service_return = {"success": False, "ad_status": "TIMEOUT"}

    assert dc.release_control() is True
    assert dc._last_mode_sent == "passthrough"


def test_ad_status_timeout_counts_toward_the_three_in_a_row_escalation():
    """The orchestrator escalates on 3 consecutive non-SENT outcomes."""
    dc, app = make_dc()
    app.call_service_return = {"success": False, "ad_status": "TIMEOUT"}

    outcomes = []
    for entry in (hold_entry(), charge_entry(), hold_entry()):
        outcomes.append(dc.apply_mode_with_outcome(entry))

    assert all(o is ApplyOutcome.UNCONFIRMED_TIMEOUT for o in outcomes)
    assert all(o.confirmed is False for o in outcomes)
    assert dc.get_diagnostics()["sent_count"] == 0


def test_nested_result_ad_status_timeout_is_unconfirmed():
    """Some AD versions nest the payload under a "result" key."""
    dc, app = make_dc()
    app.call_service_return = {"result": {"success": False, "ad_status": "TIMEOUT"}}

    assert dc.apply_mode_with_outcome(hold_entry()) is ApplyOutcome.UNCONFIRMED_TIMEOUT


# ---------------------------------------------------------------------------
# Defect C: rest-of-module review against the log
# ---------------------------------------------------------------------------

def test_failed_resend_leaves_duplicate_suppression_off_on_purpose():
    """After a resend we know failed, the next slot must not be suppressed.

    Log 06:53:49 "resend of hold failed" leaves _last_mode_time None. That is
    intended and mirrors apply_mode's confirmed-failure path: suppression stays
    off until a send succeeds, so the schedule can correct the inverter.
    """
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.apply_mode(hold_entry())
    app.call_service_raise = RuntimeError("boom")
    app.fire_last_timer()                      # resend fails

    assert dc._last_mode_time is None
    assert dc._is_duplicate("hold", dc._last_params) is False

    # The very next slot's identical command really does go out.
    app.call_service_raise = None
    calls_before = len(app.service_calls)
    assert dc.apply_mode_with_outcome(hold_entry()) is ApplyOutcome.SENT
    assert len(app.service_calls) == calls_before + 1
    assert any(
        lvl == "ERROR" and "leaving last-sent cleared" in m for m, lvl in app.logs
    )


def test_release_passthrough_match_is_not_counted_as_verified():
    """'Passthrough' is also a non-tracking sensor's reading — not evidence."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.release_control()
    app.fire_last_timer()

    diag = dc.get_diagnostics()
    assert diag["verified_count"] == 0
    assert diag["unprobative_match_count"] == 1
    assert diag["resend_recovered_count"] == 0
    # No misleading "verified"/"recovered" INFO line.
    infos = [m for m, lvl in app.logs if lvl == "INFO"]
    assert not any("verified passthrough" in m for m in infos)
    assert not any("recovered after resend" in m for m, _ in app.logs)
    assert any(
        lvl == "DEBUG" and "not counted as verified" in m for m, lvl in app.logs
    )


def test_release_passthrough_mismatch_is_still_real_evidence():
    """The source reporting an ACTIVE override after a release is a real miss."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Grid Charge"

    dc.release_control()
    calls_before = len(app.service_calls)
    app.fire_last_timer()

    assert len(app.service_calls) == calls_before + 1  # resent once
    assert dc.get_diagnostics()["mismatch_count"] == 1


def test_duplicate_skip_keeps_the_pending_verification():
    """A duplicate wants the SAME mode: its pending check is still the right one."""
    dc, app = make_dc()

    dc.apply_mode(hold_entry())
    pending = app.run_in_calls[0][3]

    assert dc.apply_mode_with_outcome(hold_entry()) is ApplyOutcome.SKIPPED_DUPLICATE

    assert pending not in app.cancelled
    assert len(app.run_in_calls) == 1


def test_dry_run_never_leaves_a_pending_timer():
    """A dry run returns before any send, and never scheduled one to begin with."""
    dc, app = make_dc(device_id="")

    dc.apply_mode(hold_entry())
    dc.release_control()

    assert app.run_in_calls == []
    assert app.service_calls == []


def test_mode_sensor_verifier_is_usable_standalone():
    """The strategy is a plain callable — testable without DirectControl."""
    app = FakeApp()
    verifier = ModeSensorVerifier(app, "sensor.x")

    app.states["sensor.x"] = "Grid Charge"
    out = verifier("grid_charge", {})
    assert out.result is VerificationResult.MATCH
    assert out.probative is True
    assert out.source == "mode_sensor:sensor.x"

    app.states["sensor.x"] = "Passthrough"
    assert verifier("grid_charge", {}).result is VerificationResult.MISMATCH
    assert verifier("passthrough", {}).probative is False

    app.states["sensor.x"] = "unavailable"
    assert verifier("grid_charge", {}).result is VerificationResult.UNVERIFIABLE

    del app.states["sensor.x"]
    assert verifier("grid_charge", {}).result is VerificationResult.UNVERIFIABLE
    # An unmapped mode string is unverifiable, not a mismatch.
    app.states["sensor.x"] = "Grid Charge"
    assert verifier("nonsense_mode", {}).result is VerificationResult.UNVERIFIABLE


# ===========================================================================
# RegisterVerifier — the recommended strategy
#
# Root cause of the 73/73 false mismatches: growatt_modbus computes
# sensor.growatt_inverter_mode from registers 30100, 30200-30201 and
# 30407-30410, but reads them behind a `_failed_optional_holding_addrs`
# blacklist that is never cleared. One transient failure on 2026-09-01T03:46:34Z
# blacklisted all three blocks, freezing the sensor at its dataclass default
# "Passthrough". The registers themselves were correct throughout.
# ===========================================================================

# Probe, 2026-09-02 09:25, 9 minutes into a discharge_to_load command.
PROBE_DISCHARGE_TO_LOAD = {
    30407: 1,      # remote power enable
    30408: 20,     # duration minutes (does NOT count down)
    30409: 65436,  # -100 % (signed)
    30410: 0,      # ac charge disabled
    30200: 1,      # export limit enabled
    30201: 0,      # export rate 0 % -> no grid export
}


class FakeRegisterApp(FakeApp):
    """FakeApp that also answers growatt_modbus/get_register_data.

    Register reads are recorded in ``reads`` and kept OUT of ``service_calls``,
    so the existing "how many set_wit_mode sends happened" assertions stay
    readable.
    """

    _UNSET = object()

    def __init__(self):
        super().__init__()
        self.registers = {}                 # {address: value}
        self.reads = []                     # [(start_address, count)]
        self.read_raise = None              # Exception to raise on a read
        self.read_response = self._UNSET    # full envelope override

    def call_service(self, service, **kwargs):
        if service != "growatt_modbus/get_register_data":
            return super().call_service(service, **kwargs)

        self.reads.append((kwargs.get("start_address"), kwargs.get("count")))
        self.read_kwargs = kwargs
        if self.read_raise is not None:
            raise self.read_raise
        if self.read_response is not self._UNSET:
            return self.read_response

        start = kwargs["start_address"]
        count = kwargs["count"]
        values = [self.registers.get(start + i, 0) for i in range(count)]
        # Home Assistant websocket envelope as AppDaemon hands it back.
        return {
            "success": True,
            "ad_status": "OK",
            "result": {"response": {"success": True, "values": values}},
        }


def make_register_dc(device_id="dev123", registers=None, **overrides):
    config = BatteryOptimizerConfig(device_id=device_id, **overrides)
    config.verify_source = "registers"
    app = FakeRegisterApp()
    if registers:
        app.registers.update(registers)
    return DirectControl(app, config), app


def discharge_entry(export_rate=None):
    return ScheduleEntry(
        time=datetime.datetime(2024, 1, 1, 12, 0, 0),
        mode=BatteryMode.DISCHARGE,
        reason="test",
        export_rate=export_rate,
    )


# --- pure decode helpers ---------------------------------------------------

def test_decode_signed_power_two_s_complement():
    assert decode_signed_power(65436) == -100     # the probe's real value
    assert decode_signed_power(65535) == -1
    assert decode_signed_power(32768) == -32768
    assert decode_signed_power(32767) == 32767
    assert decode_signed_power(100) == 100
    assert decode_signed_power(1) == 1
    assert decode_signed_power(0) == 0


def test_decode_wit_mode_status_matches_the_integration():
    """Port of coordinator._compute_wit_mode_status (coordinator.py:525-570)."""
    assert decode_wit_mode_status({30407: 0}) == "Passthrough"
    assert decode_wit_mode_status({30407: 1, 30409: 100}) == "Grid Charge"
    assert decode_wit_mode_status({30407: 1, 30409: 1}) == "Preserve SOC"
    assert decode_wit_mode_status({30407: 1, 30409: 0}) == "Preserve SOC"
    # -100 % with the limiter off -> Max Export
    assert decode_wit_mode_status(
        {30407: 1, 30409: 65436, 30200: 0}) == "Max Export"
    # -100 % with a non-zero limit -> still an export mode
    assert decode_wit_mode_status(
        {30407: 1, 30409: 65436, 30200: 1, 30201: 40}) == "Max Export"
    assert decode_wit_mode_status(
        {30407: 1, 30409: 65486, 30200: 1, 30201: 40}) == "Discharge to Grid"
    # zero export rate -> the battery cannot reach the grid
    assert decode_wit_mode_status(PROBE_DISCHARGE_TO_LOAD) == "Discharge to Load"


def test_expected_registers_follows_the_params_not_a_static_table():
    """discharge_to_grid writes different export registers per export_rate."""
    limited = expected_registers(
        "discharge_to_grid",
        {"power_percent": 100, "export_rate": 40, "ac_charge_mode": "disabled"},
    )
    assert limited[30200] == (1,)
    assert limited[30201] == (40,)
    assert limited[30409] == (65436,)

    unlimited = expected_registers(
        "discharge_to_grid",
        {"power_percent": 50, "export_rate": 100, "ac_charge_mode": "disabled"},
    )
    assert unlimited[30200] == (0,)
    assert 30201 not in unlimited          # the service never writes it
    assert unlimited[30409] == (65536 - 50,)


def test_expected_registers_tolerates_the_ac_priority_firmware_fallback():
    """30410=2 is retried as 1 when the firmware rejects it."""
    exp = expected_registers(
        "grid_charge", {"power_percent": 100, "ac_charge_mode": "ac_priority"})
    assert exp[30410] == (2, 1)
    assert expected_registers(
        "grid_charge", {"ac_charge_mode": "pv_priority"})[30410] == (1,)


def test_expected_registers_hold_is_plus_one_percent_not_zero():
    """30409=1 (tiny charge) is what actually idles the pack; 0 clips PV."""
    exp = expected_registers("hold", {"power_percent": 100})
    assert exp[30407] == (1,)
    assert exp[30409] == (1,)
    assert expected_registers("preserve_soc", {})[30409] == (1,)


def test_expected_registers_unknown_mode_is_none():
    assert expected_registers("teleport", {}) is None


# --- the ladder, driven by registers ---------------------------------------

def test_probe_tuple_verifies_discharge_to_load():
    """The exact registers the probe read must verify the command that set them."""
    dc, app = make_register_dc(registers=PROBE_DISCHARGE_TO_LOAD)

    dc.apply_mode(discharge_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()

    assert len(app.service_calls) == sends          # verified -> no resend
    assert dc.get_diagnostics()["verified_count"] == 1
    assert dc.get_diagnostics()["mismatch_count"] == 0
    # Exactly two blocking reads, in the documented blocks.
    assert app.reads == [(30407, 4), (30200, 2)]


def test_frozen_mode_sensor_no_longer_causes_a_false_mismatch():
    """The 2026-09-02 regression: sensor stuck at Passthrough, registers right."""
    dc, app = make_register_dc(registers=PROBE_DISCHARGE_TO_LOAD)
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"   # frozen entity

    dc.apply_mode(discharge_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()

    assert len(app.service_calls) == sends            # no useless resend
    assert dc.get_diagnostics()["persistent_mismatch_count"] == 0
    assert dc.get_diagnostics()["verified_count"] == 1


def test_register_verify_grid_charge():
    dc, app = make_register_dc(registers={
        30407: 1, 30408: 20, 30409: 100, 30410: 2, 30200: 0, 30201: 0,
    })

    dc.apply_mode(charge_entry())
    app.fire_last_timer()

    assert dc.get_diagnostics()["verified_count"] == 1


def test_register_verify_grid_charge_accepts_the_pv_priority_fallback():
    """30410 came back as 1 instead of the requested 2 — still correct."""
    dc, app = make_register_dc(registers={
        30407: 1, 30408: 20, 30409: 100, 30410: 1, 30200: 0, 30201: 0,
    })

    dc.apply_mode(charge_entry())
    app.fire_last_timer()

    assert dc.get_diagnostics()["verified_count"] == 1
    assert dc.get_diagnostics()["mismatch_count"] == 0


def test_register_verify_hold():
    dc, app = make_register_dc(registers={
        30407: 1, 30408: 20, 30409: 1, 30410: 0, 30200: 0, 30201: 0,
    })

    dc.apply_mode(hold_entry())
    app.fire_last_timer()

    assert dc.get_diagnostics()["verified_count"] == 1


def test_register_verify_max_export():
    dc, app = make_register_dc(registers={
        30407: 1, 30408: 20, 30409: 65436, 30410: 0, 30200: 0, 30201: 0,
    })

    dc.apply_mode(discharge_entry(export_rate=100))
    app.fire_last_timer()

    assert app.service_calls[0][1]["mode"] == "max_export"
    assert dc.get_diagnostics()["verified_count"] == 1


def test_register_verify_discharge_to_grid_with_a_limit():
    dc, app = make_register_dc(registers={
        30407: 1, 30408: 20, 30409: 65436, 30410: 0, 30200: 1, 30201: 40,
    })

    dc.apply_mode(discharge_entry(export_rate=40))
    app.fire_last_timer()

    assert app.service_calls[0][1]["mode"] == "discharge_to_grid"
    assert dc.get_diagnostics()["verified_count"] == 1


def test_register_verify_real_mismatch_still_resends_once():
    """A genuinely dropped override (30407 back to 0) drives the same ladder."""
    dc, app = make_register_dc(registers={
        30407: 0, 30408: 0, 30409: 0, 30410: 0, 30200: 0, 30201: 0,
    })

    dc.apply_mode(hold_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()      # attempt 1 -> resend
    app.fire_last_timer()      # attempt 2 -> ERROR, stop

    assert len(app.service_calls) == sends + 1     # exactly one resend
    assert len(app.run_in_calls) == 2              # bounded ladder unchanged
    diag = dc.get_diagnostics()
    assert diag["persistent_mismatch_count"] == 1
    assert diag["verification_source"] == "registers:30407-30410,30200-30201"
    # The observed tuple and the decoded status are both in the record.
    assert "30407=0" in diag["last_mismatch"]["actual"]
    assert "Passthrough" in diag["last_mismatch"]["actual"]
    warn = [m for m, lvl in app.logs if lvl == "WARNING" and "mismatch" in m]
    assert "30409" in warn[0]


def test_register_duration_30408_is_informational_only():
    """A duration that differs must not be called a mismatch."""
    regs = dict(PROBE_DISCHARGE_TO_LOAD)
    regs[30408] = 7                      # not what we asked for
    dc, app = make_register_dc(registers=regs)

    dc.apply_mode(discharge_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()

    assert len(app.service_calls) == sends
    assert dc.get_diagnostics()["verified_count"] == 1


def test_register_stale_export_rate_is_ignored_when_the_limiter_is_off():
    """30201 is not written when 30200=0, so a stale value proves nothing."""
    dc, app = make_register_dc(registers={
        30407: 1, 30408: 20, 30409: 100, 30410: 1, 30200: 0, 30201: 55,
    })

    dc.apply_mode(charge_entry())
    app.fire_last_timer()

    assert dc.get_diagnostics()["verified_count"] == 1


# --- degraded reads are UNVERIFIABLE, never MISMATCH ------------------------

def _assert_unverifiable_and_silent(dc, app, sends_before):
    assert len(app.service_calls) == sends_before      # no resend
    diag = dc.get_diagnostics()
    assert diag["unverifiable_count"] == 1
    assert diag["mismatch_count"] == 0
    assert diag["resend_count"] == 0


def test_register_read_none_is_unverifiable():
    dc, app = make_register_dc()
    app.read_response = None

    dc.apply_mode(hold_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()

    _assert_unverifiable_and_silent(dc, app, sends)


def test_register_read_service_error_is_unverifiable():
    """The integration's own failure shape: success=False, values=[]."""
    dc, app = make_register_dc()
    app.read_response = {
        "success": True,
        "result": {"response": {
            "success": False, "values": [], "error": "Register read failed",
        }},
    }

    dc.apply_mode(hold_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()

    _assert_unverifiable_and_silent(dc, app, sends)


def test_register_read_timeout_is_unverifiable():
    dc, app = make_register_dc()
    app.read_response = {"success": False, "ad_status": "TIMEOUT"}

    dc.apply_mode(hold_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()

    _assert_unverifiable_and_silent(dc, app, sends)


def test_register_read_exception_is_unverifiable():
    dc, app = make_register_dc()
    app.read_raise = RuntimeError("modbus offline")

    dc.apply_mode(hold_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()

    _assert_unverifiable_and_silent(dc, app, sends)


def test_register_read_short_values_is_unverifiable():
    """A truncated read must not be padded into a verdict."""
    dc, app = make_register_dc()
    app.read_response = {"result": {"response": {
        "success": True, "values": [1, 20],
    }}}

    dc.apply_mode(hold_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()

    _assert_unverifiable_and_silent(dc, app, sends)


def test_register_read_malformed_payload_is_unverifiable():
    dc, app = make_register_dc()
    app.read_response = {"result": {"response": {"success": True,
                                                 "values": "not-a-list"}}}

    dc.apply_mode(hold_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()

    _assert_unverifiable_and_silent(dc, app, sends)


def test_register_read_non_numeric_values_is_unverifiable():
    dc, app = make_register_dc()
    app.read_response = {"result": {"response": {
        "success": True, "values": ["a", "b", "c", "d"],
    }}}

    dc.apply_mode(hold_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()

    _assert_unverifiable_and_silent(dc, app, sends)


def test_register_second_block_failure_is_unverifiable():
    """The export block failing alone is still no verdict."""
    app = FakeRegisterApp()
    config = BatteryOptimizerConfig(device_id="dev123")
    config.verify_source = "registers"
    dc = DirectControl(app, config)
    app.registers.update(PROBE_DISCHARGE_TO_LOAD)

    real_call = app.call_service

    def only_first_block(service, **kwargs):
        if (service == "growatt_modbus/get_register_data"
                and kwargs.get("start_address") == 30200):
            app.reads.append((30200, kwargs.get("count")))
            return None
        return real_call(service, **kwargs)

    app.call_service = only_first_block

    dc.apply_mode(discharge_entry())
    sends = len(app.service_calls)
    app.fire_last_timer()

    _assert_unverifiable_and_silent(dc, app, sends)


def test_register_read_envelope_shapes_are_all_accepted():
    """Top-level, result-nested and response-nested payloads all decode."""
    values = [1, 20, 1, 0]
    envelopes = [
        {"success": True, "values": values},
        {"result": {"success": True, "values": values}},
        {"result": {"response": {"success": True, "values": values}}},
        {"response": {"success": True, "values": values}},
    ]
    for envelope in envelopes:
        app = FakeRegisterApp()
        verifier = RegisterVerifier(app, "dev123", timeout=15)
        app.read_response = envelope
        assert verifier._read_block(30407, 4) == values, envelope


def test_register_read_uses_the_documented_service_schema():
    """Field is start_address (not address); hass_timeout is the send timeout."""
    dc, app = make_register_dc(
        registers=PROBE_DISCHARGE_TO_LOAD, set_wit_mode_timeout_seconds=12)

    dc.apply_mode(discharge_entry())
    app.fire_last_timer()

    kwargs = app.read_kwargs
    assert kwargs["device_id"] == "dev123"
    assert kwargs["register_type"] == "holding"
    assert kwargs["start_address"] == 30200      # last read
    assert kwargs["count"] == 2
    assert kwargs["return_response"] is True
    assert kwargs["hass_timeout"] == 12
    assert set(kwargs) == {
        "device_id", "register_type", "start_address", "count",
        "return_response", "hass_timeout",
    }


# --- passthrough / dry run --------------------------------------------------

def test_register_passthrough_match_is_not_probative():
    """All-zero registers read the same whether or not the release arrived."""
    dc, app = make_register_dc(registers={
        30407: 0, 30408: 0, 30409: 0, 30410: 0, 30200: 0, 30201: 0,
    })

    dc.release_control()
    app.fire_last_timer()

    diag = dc.get_diagnostics()
    assert diag["verified_count"] == 0
    assert diag["unprobative_match_count"] == 1
    assert diag["mismatch_count"] == 0
    assert not any("recovered after resend" in m for m, _ in app.logs)


def test_register_passthrough_mismatch_is_real_evidence():
    """An override still active after a release is a genuine miss."""
    dc, app = make_register_dc(registers=PROBE_DISCHARGE_TO_LOAD)

    dc.release_control()
    sends = len(app.service_calls)
    app.fire_last_timer()

    assert len(app.service_calls) == sends + 1
    assert dc.get_diagnostics()["mismatch_count"] == 1


def test_register_verifier_without_device_id_is_unverifiable():
    app = FakeRegisterApp()
    verifier = RegisterVerifier(app, "")

    out = verifier("hold", {})

    assert out.result is VerificationResult.UNVERIFIABLE
    assert "dry run" in out.detail
    assert app.reads == []


def test_register_verifier_unknown_mode_is_unverifiable_not_mismatch():
    app = FakeRegisterApp()
    verifier = RegisterVerifier(app, "dev123")

    out = verifier("teleport", {})

    assert out.result is VerificationResult.UNVERIFIABLE
    assert app.reads == []   # no blocking read wasted on an unknown mode


# --- verify_source selection ------------------------------------------------

def test_verify_source_registers_selects_the_register_strategy():
    dc, _app = make_register_dc()
    assert dc.verification_source == "registers:30407-30410,30200-30201"


def test_verify_source_auto_prefers_registers_over_a_configured_sensor():
    """'auto' must NOT fall back to the mode sensor just because it is set."""
    config = BatteryOptimizerConfig(
        device_id="dev123", inverter_mode_sensor=DEFAULT_MODE_STATUS_ENTITY)
    dc = DirectControl(FakeRegisterApp(), config)   # no verify_source attribute

    assert dc.verification_source == "registers:30407-30410,30200-30201"


def test_verify_source_auto_without_device_id_is_disabled():
    config = BatteryOptimizerConfig(device_id="")
    dc = DirectControl(FakeRegisterApp(), config)

    assert dc.verification_enabled is False


def test_verify_source_registers_without_device_id_is_disabled():
    dc, app = make_register_dc(device_id="")

    assert dc.verification_enabled is False
    dc.apply_mode(hold_entry())
    assert app.run_in_calls == []


def test_verify_source_none_disables_everything():
    dc, app = make_register_dc()
    dc.config.verify_source = "none"
    dc2 = DirectControl(app, dc.config)

    assert dc2.verification_enabled is False


def test_verify_source_mode_sensor_without_an_entity_warns_and_disables():
    config = BatteryOptimizerConfig(device_id="dev123", inverter_mode_sensor="")
    config.verify_source = "mode_sensor"
    app = FakeRegisterApp()
    dc = DirectControl(app, config)

    assert dc.verification_enabled is False
    assert any(
        lvl == "WARNING" and "inverter_mode_sensor is empty" in m
        for m, lvl in app.logs
    )


def test_register_verify_duration_is_reported_to_the_app():
    """Two blocking reads per check must land in the slow-callback accounting."""
    dc, app = make_register_dc(registers=PROBE_DISCHARGE_TO_LOAD)
    recorded = []
    app.record_external_callback_duration = (
        lambda name, seconds: recorded.append(name)
    )

    dc.apply_mode(discharge_entry())
    app.fire_last_timer()

    assert recorded == ["DirectControl._verify_mode"]


# ===========================================================================
# Thread safety (design §2.5, tests 9-16)
#
# AppDaemon 4.5.13 dispatches this app's callbacks across worker threads once
# `total_threads` is set AND `pin_app: false` — apply_mode, release_control and
# the verification timer can then all run at once. DirectControl carries two
# locks with a fixed order: app lock -> _io_lock -> _state_lock.
# ===========================================================================

def run_threads(targets, timeout=5.0):
    """Start every callable at once on a Barrier, join, re-raise failures."""
    barrier = threading.Barrier(len(targets))
    errors = []

    def runner(fn):
        def wrapped():
            try:
                barrier.wait(timeout=timeout)
                fn()
            except Exception as exc:      # pragma: no cover - surfaced below
                errors.append(exc)
        return wrapped

    threads = [threading.Thread(target=runner(fn)) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    assert not any(t.is_alive() for t in threads), "thread did not finish"
    assert not errors, errors
    return threads


# --- 9: the duplicate decision is atomic with the send ---------------------

def test_concurrent_same_mode_applies_send_exactly_once():
    """R6: read-before-send / write-after-send let both threads through.

    _is_duplicate used to run before a call that can block for 15 s and the
    last-sent marker was written only after it returned, so two same-mode
    applies on two workers both reached the inverter. _io_lock now spans both.
    """
    dc, app = make_dc(inverter_mode_sensor="")     # no verification timers
    app.sleep_seconds = 0.02
    outcomes = []
    lock = threading.Lock()

    def apply_once():
        outcome = dc.apply_mode_with_outcome(hold_entry())
        with lock:
            outcomes.append(outcome)

    run_threads([apply_once] * 8)

    assert outcomes.count(ApplyOutcome.SENT) == 1
    assert outcomes.count(ApplyOutcome.SKIPPED_DUPLICATE) == 7
    assert len(app.service_calls) == 1
    diag = dc.get_diagnostics()
    assert diag["sent_count"] == 1
    assert diag["duplicate_skipped_count"] == 7


# --- 10: different modes serialize ------------------------------------------

def test_concurrent_different_modes_serialize_and_last_finished_wins():
    """R3: competing commands must not interleave on the wire."""
    dc, app = make_dc(inverter_mode_sensor="")
    app.sleep_seconds = 0.03
    finished = []
    lock = threading.Lock()

    def apply(entry, name):
        def go():
            dc.apply_mode_with_outcome(entry)
            with lock:
                finished.append(name)
        return go

    run_threads([
        apply(hold_entry(), "hold"),
        apply(charge_entry(), "grid_charge"),
        apply(discharge_entry(), "discharge_to_load"),
    ])

    assert len(app.service_calls) == 3
    assert app.overlapping_calls() == []          # serialized on _io_lock
    # The mode that stands is the one whose send COMPLETED last.
    assert dc._last_mode_sent == finished[-1]


# --- 11: diagnostics never wait on the inverter -----------------------------

def test_get_diagnostics_does_not_wait_for_an_in_flight_send():
    """_state_lock is never held across a service call.

    get_diagnostics runs from _update_control_health_sensor under the APP lock;
    if it could block on a 15 s inverter write it would freeze every callback.
    """
    dc, app = make_dc(inverter_mode_sensor="")
    app.sleep_seconds = 0.5

    sender = threading.Thread(
        target=lambda: dc.apply_mode_with_outcome(hold_entry()))
    sender.start()
    assert app.call_started.wait(timeout=2.0)

    started = time.monotonic()
    diag = dc.get_diagnostics()
    elapsed = time.monotonic() - started

    assert elapsed < 0.05, f"get_diagnostics blocked for {elapsed:.3f}s"
    assert isinstance(diag, dict)

    sender.join(timeout=5.0)
    assert not sender.is_alive()
    assert dc.get_diagnostics()["sent_count"] == 1


# --- 12: counters are exact -------------------------------------------------

def test_outcome_counters_are_exact_under_concurrency():
    """R8: `counts[k] = counts.get(k, 0) + 1` loses increments unguarded."""
    dc, app = make_dc(inverter_mode_sensor="")

    def apply_many():
        for _ in range(50):
            dc.apply_mode_with_outcome(hold_entry())

    run_threads([apply_many] * 8, timeout=20.0)

    diag = dc.get_diagnostics()
    total = (diag["sent_count"] + diag["duplicate_skipped_count"]
             + diag["failed_count"] + diag["unconfirmed_count"]
             + diag["dry_run_count"])
    assert total == 8 * 50
    # Only the first send is not a duplicate: same mode, same params, and the
    # whole run is far inside half a slot.
    assert diag["sent_count"] == 1
    assert diag["duplicate_skipped_count"] == 8 * 50 - 1
    assert len(app.service_calls) == 1


# --- 13: verify-timer generation (R7) ---------------------------------------

def test_stale_verification_returns_early_and_leaves_the_new_timer_alone():
    """cancel_timer cannot stop a callback another worker already dequeued.

    The orphan used to run: it nulled `_verify_timer` (which by then belonged to
    a NEWER verification) and resent the superseded mode.
    """
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"  # would mismatch

    dc.apply_mode(hold_entry())
    stale_kwargs = dict(app.run_in_calls[0][2])
    assert stale_kwargs["generation"] == dc._verify_generation

    # A second apply supersedes it (bumping the generation) and schedules its
    # own check.
    dc.apply_mode(charge_entry())
    fresh_handle = app.run_in_calls[-1][3]
    fresh_generation = app.run_in_calls[-1][2]["generation"]
    assert fresh_generation != stale_kwargs["generation"]
    sends_before = len(app.service_calls)

    # Now the orphan finally runs.
    dc._verify_mode(stale_kwargs)

    assert len(app.service_calls) == sends_before      # no superseded resend
    assert dc._verify_timer == fresh_handle            # newer handle intact
    assert dc.get_diagnostics()["mismatch_count"] == 0
    assert dc.get_diagnostics()["resend_count"] == 0

    # The current verification still works.
    dc._verify_mode(app.run_in_calls[-1][2])
    assert dc.get_diagnostics()["mismatch_count"] == 1


def test_cancel_verification_alone_invalidates_a_dequeued_callback():
    """release_control supersedes a pending check even mid-dispatch."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.apply_mode(hold_entry())
    stale_kwargs = dict(app.run_in_calls[0][2])

    dc._cancel_verification()
    sends_before = len(app.service_calls)

    dc._verify_mode(stale_kwargs)

    assert len(app.service_calls) == sends_before
    assert dc.get_diagnostics()["resend_count"] == 0


# --- 14: the resend serializes against an apply -----------------------------

def test_resend_and_apply_never_overlap_on_the_wire():
    """The verification READ is lock-free, but its resend takes _io_lock."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.apply_mode(hold_entry())               # schedules verification
    verify_kwargs = dict(app.run_in_calls[-1][2])
    app.sleep_seconds = 0.05
    app.call_started.clear()

    run_threads([
        lambda: dc._verify_mode(verify_kwargs),
        lambda: dc.apply_mode_with_outcome(charge_entry()),
    ])

    assert app.overlapping_calls() == [], "a resend overlapped an apply"
    # Whoever lost the race must not have corrupted the ladder.
    assert dc.get_diagnostics()["resend_count"] <= 1


def test_verification_read_does_not_block_an_apply():
    """A slow read must not make an apply queue behind it."""
    reading = threading.Event()
    release = threading.Event()

    def slow_verifier(mode_str, params):
        reading.set()
        release.wait(timeout=2.0)
        return VerificationOutcome(
            result=VerificationResult.MATCH, source="slow",
            observed="ok", expected="ok",
        )

    config = BatteryOptimizerConfig(device_id="dev123")
    config.verify_source = "none"
    app = FakeApp()
    dc = DirectControl(app, config, verifier=slow_verifier)

    dc.apply_mode(hold_entry())
    verify_kwargs = dict(app.run_in_calls[-1][2])

    verifier_thread = threading.Thread(
        target=lambda: dc._verify_mode(verify_kwargs))
    verifier_thread.start()
    assert reading.wait(timeout=2.0)

    # The read is in progress and holds no lock: this apply must complete now.
    started = time.monotonic()
    outcome = dc.apply_mode_with_outcome(charge_entry())
    elapsed = time.monotonic() - started

    assert outcome is ApplyOutcome.SENT
    assert elapsed < 0.5, f"apply waited {elapsed:.3f}s on a read"

    release.set()
    verifier_thread.join(timeout=2.0)
    assert not verifier_thread.is_alive()


# --- 15: the bounded ladder survives concurrency ----------------------------

def test_ladder_bounds_survive_concurrent_verifications():
    """Max 2 checks and 2 sends, however many workers pile on."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"

    dc.apply_mode(hold_entry())
    sends_after_apply = len(app.service_calls)
    verify_kwargs = dict(app.run_in_calls[-1][2])

    # Six workers all dequeue the SAME verification callback.
    run_threads([lambda: dc._verify_mode(dict(verify_kwargs))] * 6)

    diag = dc.get_diagnostics()
    # Exactly one of them owned the generation; the rest were stale.
    assert diag["resend_count"] == 1
    assert len(app.service_calls) == sends_after_apply + 1

    # Drive the re-check the same way.
    recheck_kwargs = dict(app.run_in_calls[-1][2])
    assert recheck_kwargs["attempt"] == 2
    run_threads([lambda: dc._verify_mode(dict(recheck_kwargs))] * 6)

    assert len(app.service_calls) == sends_after_apply + 1   # never a third
    assert dc.get_diagnostics()["persistent_mismatch_count"] == 1


# --- 16: no deadlock against the app lock -----------------------------------

def test_report_duration_cannot_deadlock_against_get_diagnostics():
    """Lock order app -> io -> state, enforced by WHERE _report_duration runs.

    _report_duration calls back into the app, which runs it under the APP lock.
    If it were called while DirectControl still held _state_lock, a thread
    holding the app lock and asking for get_diagnostics() would deadlock. It
    lives in _verify_mode's finally, after _run_verification released both.
    """
    app_lock = threading.RLock()
    reported = []
    in_verifier = threading.Event()
    let_verifier_finish = threading.Event()

    def slow_verifier(mode_str, params):
        in_verifier.set()
        let_verifier_finish.wait(timeout=3.0)
        return VerificationOutcome(
            result=VerificationResult.MATCH, source="probe",
            observed="ok", expected="ok",
        )

    class AppLockedApp(FakeApp):
        def record_external_callback_duration(self, name, seconds):
            with app_lock:                     # what the real app does
                reported.append(name)

    config = BatteryOptimizerConfig(device_id="dev123")
    config.verify_source = "none"
    app = AppLockedApp()
    dc = DirectControl(app, config, verifier=slow_verifier)

    dc.apply_mode(hold_entry())
    verify_kwargs = dict(app.run_in_calls[-1][2])

    verifier_thread = threading.Thread(
        target=lambda: dc._verify_mode(verify_kwargs))

    with app_lock:                              # main thread holds the app lock
        verifier_thread.start()
        assert in_verifier.wait(timeout=2.0)
        # DirectControl holds no lock during the read, so this returns at once.
        started = time.monotonic()
        diag = dc.get_diagnostics()
        assert time.monotonic() - started < 0.5
        # The strategy has not returned yet, so it is still identified by the
        # callable's name rather than the source it will report.
        assert diag["verification_source"] == "slow_verifier"
        # Release the verifier while we still hold the app lock: its
        # _report_duration will block on it, but only AFTER dropping both
        # DirectControl locks — so nothing is waiting on it.
        let_verifier_finish.set()
        time.sleep(0.05)
        assert dc.get_diagnostics()["verified_count"] in (0, 1)

    verifier_thread.join(timeout=3.0)
    assert not verifier_thread.is_alive(), "deadlock: _verify_mode never finished"
    assert reported == ["DirectControl._verify_mode"]


def test_concurrent_release_and_apply_serialize():
    """release_control shares _io_lock with apply_mode."""
    dc, app = make_dc(inverter_mode_sensor="")
    app.sleep_seconds = 0.03

    run_threads([
        lambda: dc.release_control(),
        lambda: dc.apply_mode_with_outcome(charge_entry()),
    ])

    assert len(app.service_calls) == 2
    assert app.overlapping_calls() == []
    assert dc._last_mode_sent in ("passthrough", "grid_charge")
