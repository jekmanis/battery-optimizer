"""Tests for DirectControl set_wit_mode reliability behavior.

Covers:
- timeout (call_service returns None) -> unconfirmed, still records last-sent,
  schedules verification, logs WARNING
- confirmed failure (exception or success=False) -> returns False, does NOT
  record last-sent, schedules no verification
- verify-after-set mismatch -> resends once (bypassing duplicate suppression)
- verify-after-set match -> no resend
- verify-after-set unavailable sensor -> cannot verify, no resend
- pending verification timer is superseded when a new mode is applied
"""

import datetime

import pytest

from battery_optimizer_lib.config import BatteryOptimizerConfig
from battery_optimizer_lib.direct_control import (
    DirectControl,
    DEFAULT_MODE_STATUS_ENTITY,
)
from battery_optimizer_lib.models import BatteryMode, ScheduleEntry


class FakeApp:
    """Minimal AppDaemon app double exposing the methods DirectControl uses."""

    def __init__(self):
        # call_service behavior
        self.call_service_return = {"success": True}
        self.call_service_raise = None  # set to an Exception instance to raise
        self.service_calls = []  # list of (service, kwargs)

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
        self.service_calls.append((service, kwargs))
        if self.call_service_raise is not None:
            raise self.call_service_raise
        return self.call_service_return

    def get_state(self, entity):
        return self.states.get(entity)

    def run_in(self, callback, delay, **kwargs):
        self._next_handle += 1
        handle = f"timer_{self._next_handle}"
        self.run_in_calls.append((callback, delay, kwargs, handle))
        return handle

    def cancel_timer(self, handle):
        self.cancelled.append(handle)

    def log(self, message, level="INFO"):
        self.logs.append((message, level))

    # --- test helpers ---
    def levels(self):
        return [lvl for _, lvl in self.logs]

    def fire_last_timer(self):
        """Invoke the most recently scheduled run_in callback."""
        callback, _delay, kwargs, _handle = self.run_in_calls[-1]
        callback(kwargs)


def make_dc(device_id="dev123", **overrides):
    config = BatteryOptimizerConfig(device_id=device_id, **overrides)
    app = FakeApp()
    return DirectControl(app, config), app


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
    # hass_timeout was passed on the service call
    assert app.service_calls[0][1].get("hass_timeout") == 30
    assert app.service_calls[0][1].get("return_result") is True
    # last-sent recorded so the schedule isn't spammed
    assert dc._last_mode_sent == "hold"
    assert dc._last_mode_time is not None
    # WARNING about unconfirmed state
    assert "WARNING" in app.levels()
    # verification scheduled
    assert len(app.run_in_calls) == 1
    assert app.run_in_calls[0][1] == 90


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


def test_verify_mismatch_resends_once():
    """Sensor reports a different status -> resend exactly once."""
    dc, app = make_dc()
    app.states[DEFAULT_MODE_STATUS_ENTITY] = "Passthrough"  # expected Preserve SOC

    dc.apply_mode(hold_entry())
    calls_before = len(app.service_calls)

    app.fire_last_timer()

    # exactly one additional set_wit_mode call (the resend)
    assert len(app.service_calls) == calls_before + 1
    assert "WARNING" in app.levels()
    # the resend must NOT schedule another verification (no infinite loop)
    assert len(app.run_in_calls) == 1


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
    assert "DEBUG" in app.levels()


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
