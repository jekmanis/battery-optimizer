"""Regressions for DEFECT 11(b) and the DEFECT 10 diagnostics sensor.

Production evidence (33h AppDaemon window): 70 x "Excessive time spent in
callback ... (limit=10.0s)" at 10-34s, ALL on thread-0, alongside 10 x 30s
`set_wit_mode` timeouts and 9 x "Failed to apply mode - will retry next slot".
Nothing in the app measured any of it, nothing counted the failures, and nothing
told the operator that a single AppDaemon thread is the reason one blocking
service call stalls everything else.

CLAUDE.md keeps the orchestrator out of unit tests, so these exercise only the
small, pure pieces added for this: the timing decorator, the overrun accounting,
the rate-limited terminal warning gate, and the health-sensor payload.
"""

import datetime
import inspect

import pytest

import battery_optimizer as bo
from battery_optimizer_lib.direct_control import ApplyOutcome


class FakeOptimizer(bo.BatteryOptimizer):
    """Minimal stand-in: no AppDaemon initialize(), just the added state."""

    def __init__(self, **config_overrides):
        self.config = bo.BatteryOptimizerConfig(**config_overrides)
        self.logs = []
        self.states = {}
        self._callback_overrun_count = 0
        self._slowest_callback = None
        self._threads_hint_logged = False
        self._apply_failure_count = 0
        self._consecutive_apply_failures = 0
        self._apply_success_count = 0
        self._apply_unconfirmed_count = 0
        self._consecutive_apply_unconfirmed = 0
        self._apply_duplicate_count = 0
        self._apply_dry_run_count = 0
        self._last_terminal_warning_time = None
        self._now = datetime.datetime(2026, 7, 28, 16, 0)

    def log(self, message, level="INFO"):
        self.logs.append((message, level))

    def set_state(self, entity, state=None, attributes=None):
        self.states[entity] = {"state": state, "attributes": attributes or {}}

    def datetime(self):
        return self._now

    def levels(self):
        return [lvl for _m, lvl in self.logs]


# ---------------------------------------------------------------------------
# The decorator must not disturb AppDaemon's calling convention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("execute_scheduled_mode", "(self, kwargs, force: bool = False)"),
        ("full_optimize", "(self, kwargs=None)"),
        ("adaptive_optimize", "(self, kwargs=None)"),
        ("_on_soc_change", "(self, entity, attribute, old, new, kwargs)"),
        ("_on_energy_sensor_change", "(self, entity, attribute, old, new, kwargs)"),
        # The two PV/ambient timer callbacks run on the same shared worker
        # thread and were shipped undecorated, so a slow HA read from either
        # was invisible in the overrun accounting.
        ("_record_ambient_observation", "(self, kwargs=None)"),
        ("_sample_pv", "(self, kwargs=None)"),
    ],
)
def test_timed_callbacks_keep_their_signatures(name, expected):
    """AppDaemon calls these positionally; functools.wraps must preserve them."""
    method = getattr(bo.BatteryOptimizer, name)
    assert str(inspect.signature(method)) == expected
    assert method.__name__ == name


def test_decorator_records_duration_and_propagates_return_and_errors():
    calls = []

    class Probe:
        config = bo.BatteryOptimizerConfig()

        def _record_callback_duration(self, name, seconds):
            calls.append((name, seconds))

        @bo._timed_callback
        def work(self, a, b=2):
            return a + b

        @bo._timed_callback
        def boom(self):
            raise ValueError("nope")

    probe = Probe()
    assert probe.work(1, b=5) == 6
    with pytest.raises(ValueError):
        probe.boom()

    assert [name for name, _ in calls] == ["work", "boom"]
    assert all(seconds >= 0 for _n, seconds in calls)


# ---------------------------------------------------------------------------
# Overrun accounting
# ---------------------------------------------------------------------------

def test_fast_callbacks_are_silent():
    app = FakeOptimizer()

    app._record_callback_duration("adaptive_optimize", 0.4)

    assert app._callback_overrun_count == 0
    assert app.logs == []
    assert app._slowest_callback == ("adaptive_optimize", 0.4)


def test_slow_callback_warns_and_names_itself():
    app = FakeOptimizer()

    app._record_callback_duration("execute_scheduled_mode", 34.0)

    assert app._callback_overrun_count == 1
    message, level = app.logs[0]
    assert level == "WARNING"
    assert "execute_scheduled_mode" in message
    assert "34.0s" in message


def test_third_overrun_advises_more_appdaemon_threads_once():
    app = FakeOptimizer()

    for seconds in (12.0, 30.0, 11.0, 15.0, 20.0):
        app._record_callback_duration("full_optimize", seconds)

    hints = [m for m, _l in app.logs if "total_threads" in m]
    assert len(hints) == 1
    assert "set_wit_mode is a blocking service call" in hints[0]
    assert app._callback_overrun_count == 5


def test_overrun_limit_is_configurable():
    app = FakeOptimizer(callback_warn_seconds=30.0)

    app._record_callback_duration("full_optimize", 20.0)

    assert app._callback_overrun_count == 0


# ---------------------------------------------------------------------------
# The degenerate-terminal warning must be rate limited, not spammed
# ---------------------------------------------------------------------------

def test_terminal_warning_gate_is_silent_for_auto():
    app = FakeOptimizer(terminal_energy_value_eur_kwh=None)
    assert app._should_warn_degenerate_terminal() is False


def test_terminal_warning_gate_fires_once_per_six_hours():
    app = FakeOptimizer(terminal_energy_value_eur_kwh=0.0)

    assert app._should_warn_degenerate_terminal() is True
    # The schedule is rebuilt every 15 minutes -> ~96 warnings/day without this.
    app._now += datetime.timedelta(minutes=15)
    assert app._should_warn_degenerate_terminal() is False
    app._now += datetime.timedelta(hours=6)
    assert app._should_warn_degenerate_terminal() is True


# ---------------------------------------------------------------------------
# Diagnostics sensor (DEFECT 10c)
# ---------------------------------------------------------------------------

class StubDirectControl:
    def __init__(self, **diag):
        self._diag = {
            "mismatch_count": 0,
            "resend_count": 0,
            "resend_recovered_count": 0,
            "resend_failed_count": 0,
            "persistent_mismatch_count": 0,
            "unverifiable_count": 0,
            "verified_count": 0,
            "last_mismatch": None,
            "verify_delay_seconds": 90,
            "verify_recheck_seconds": 60,
            "set_wit_mode_timeout_seconds": 15,
        }
        self._diag.update(diag)

    def get_diagnostics(self):
        return dict(self._diag)


def test_control_health_sensor_publishes_counters():
    app = FakeOptimizer()
    app._direct_control = StubDirectControl(
        mismatch_count=30, resend_count=30, persistent_mismatch_count=2
    )
    app._apply_failure_count = 9
    app._callback_overrun_count = 70
    app._slowest_callback = ("execute_scheduled_mode", 34.0)

    app._update_control_health_sensor()

    sensor = app.states["sensor.battery_inverter_control_health"]
    # HA states are strings. This assertion used to read `== 2` (an int), which
    # is what shipped — and in production an int state of 0 is falsy, got
    # dropped from the POST body, and HA rejected every publish with
    # "[400] HTTP POST: Bad Request" on each mode apply.
    assert sensor["state"] == "2"  # persistent mismatches are the headline number
    assert isinstance(sensor["state"], str)
    attrs = sensor["attributes"]
    assert attrs["mismatch_count"] == 30
    assert attrs["apply_failures"] == 9
    assert attrs["callback_overruns"] == 70
    assert attrs["slowest_callback"] == "execute_scheduled_mode 34.0s"
    assert attrs["set_wit_mode_timeout_seconds"] == 15


def test_control_health_state_is_a_truthy_string_when_there_are_no_mismatches():
    """Regression for the production "[400] Bad Request" on every mode apply.

    The healthy case is persistent_mismatch_count == 0. Published as an int that
    is falsy, so it never reached Home Assistant and the sensor errored every
    slot instead of reporting "all good".
    """
    app = FakeOptimizer()
    app._direct_control = StubDirectControl(
        mismatch_count=0, resend_count=0, persistent_mismatch_count=0
    )

    app._update_control_health_sensor()

    state = app.states["sensor.battery_inverter_control_health"]["state"]
    assert state == "0"
    assert isinstance(state, str)
    assert state, "a falsy state is dropped from the HA POST body -> 400"


def test_control_health_sensor_survives_a_broken_direct_control():
    """A diagnostics failure must never take down mode execution."""
    class Broken:
        def get_diagnostics(self):
            raise RuntimeError("boom")

    app = FakeOptimizer()
    app._direct_control = Broken()

    app._update_control_health_sensor()  # must not raise

    assert "sensor.battery_inverter_control_health" not in app.states
    assert "WARNING" in app.levels()


# ---------------------------------------------------------------------------
# Every scheduled callback is instrumented
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "execute_scheduled_mode",
        "full_optimize",
        "adaptive_optimize",
        "_on_soc_change",
        "_on_energy_sensor_change",
        "_record_ambient_observation",
        "_sample_pv",
    ],
)
def test_scheduled_callbacks_are_timed(name):
    """`@_timed_callback` sets `__wrapped__` via functools.wraps."""
    method = getattr(bo.BatteryOptimizer, name)
    assert hasattr(method, "__wrapped__"), f"{name} is not instrumented"


# ---------------------------------------------------------------------------
# Every apply_mode goes through one accounting helper
# ---------------------------------------------------------------------------

class RecordingDirectControl:
    """apply_mode_with_outcome returning a scripted sequence of outcomes.

    Accepts booleans for readability: True -> SENT, False -> FAILED.
    """

    def __init__(self, *results):
        self.results = [
            ApplyOutcome.SENT if r is True
            else ApplyOutcome.FAILED if r is False
            else r
            for r in results
        ]
        self.calls = []
        self._diag = {"persistent_mismatch_count": 0}

    def apply_mode_with_outcome(self, entry):
        self.calls.append(entry)
        return self.results.pop(0) if self.results else ApplyOutcome.SENT

    def get_diagnostics(self):
        return dict(self._diag)


def _entry(reason="safety_min_soc", mode=None):
    return bo.ScheduleEntry(
        time=datetime.datetime(2026, 7, 28, 16, 0),
        mode=mode or bo.BatteryMode.HOLD,
        reason=reason,
    )


def test_successful_apply_resets_the_consecutive_failure_counter():
    """A safety HOLD that WORKED used to leave the counter climbing.

    Only `execute_scheduled_mode` accounted for its result, so the three
    `_check_soc_boundaries` paths, the solar override and manual mode never
    reset it — the "inverter is NOT following the schedule" ERROR eventually
    fired while every command was in fact being obeyed.
    """
    app = FakeOptimizer()
    app._direct_control = RecordingDirectControl(True)
    app._consecutive_apply_failures = 2

    assert app._apply_mode_tracked(_entry()) is True
    assert app._consecutive_apply_failures == 0
    assert app._apply_success_count == 1
    assert "ERROR" not in app.levels()


def test_failed_apply_is_never_silent():
    """A failed safety apply used to produce no log line at all."""
    app = FakeOptimizer()
    app._direct_control = RecordingDirectControl(False)

    assert app._apply_mode_tracked(_entry("safety_min_soc")) is False
    assert app._apply_failure_count == 1
    assert app._consecutive_apply_failures == 1
    warnings = [m for m, lvl in app.logs if lvl == "WARNING"]
    assert any("Failed to apply mode" in m for m in warnings)
    # The message names WHICH command failed.
    assert any("safety_min_soc" in m for m in warnings)


def test_three_consecutive_failures_escalate_to_error():
    app = FakeOptimizer()
    app._direct_control = RecordingDirectControl(False, False, False)

    for _ in range(3):
        app._apply_mode_tracked(_entry())

    assert app._consecutive_apply_failures == 3
    errors = [m for m, lvl in app.logs if lvl == "ERROR"]
    assert len(errors) == 1
    assert "NOT following the schedule" in errors[0]


def test_a_success_between_failures_prevents_the_error():
    app = FakeOptimizer()
    app._direct_control = RecordingDirectControl(False, True, False, False)

    for _ in range(4):
        app._apply_mode_tracked(_entry())

    assert app._consecutive_apply_failures == 2
    assert "ERROR" not in app.levels()


def test_health_sensor_is_published_after_every_apply():
    app = FakeOptimizer()
    app._direct_control = RecordingDirectControl(True)

    app._apply_mode_tracked(_entry())

    assert "sensor.battery_inverter_control_health" in app.states


# ---------------------------------------------------------------------------
# Only a CONFIRMED send counts as health
#
# apply_mode returns True for three outcomes the inverter never acknowledged:
# a dry run, a duplicate that was never transmitted, and a client-side timeout.
# Counting them as successes reset _consecutive_apply_failures on every call, so
# with growatt_modbus hung-but-not-raising the health sensor reported climbing
# apply_successes and the "NOT following the schedule" ERROR could never fire.
# ---------------------------------------------------------------------------

def test_unconfirmed_timeout_is_not_a_success():
    app = FakeOptimizer()
    app._direct_control = RecordingDirectControl(ApplyOutcome.UNCONFIRMED_TIMEOUT)
    app._consecutive_apply_failures = 2

    # Still True: DirectControl recorded it as sent and verify-after-set is what
    # resolves it — but it must not be booked as a success.
    assert app._apply_mode_tracked(_entry()) is True
    assert app._apply_success_count == 0
    assert app._apply_unconfirmed_count == 1
    assert app._consecutive_apply_failures == 2, "an unconfirmed send reset the streak"
    assert app._consecutive_apply_unconfirmed == 1


def test_three_consecutive_unconfirmed_timeouts_escalate_to_error():
    """The hung-modbus case: every call times out, nothing is ever confirmed."""
    app = FakeOptimizer()
    app._direct_control = RecordingDirectControl(
        *([ApplyOutcome.UNCONFIRMED_TIMEOUT] * 3)
    )

    for _ in range(3):
        app._apply_mode_tracked(_entry())

    errors = [m for m, lvl in app.logs if lvl == "ERROR"]
    assert len(errors) == 1
    assert "NOT following the schedule" in errors[0]
    assert app._apply_unconfirmed_count == 3
    assert app._apply_success_count == 0


def test_a_confirmed_send_clears_the_unconfirmed_streak():
    app = FakeOptimizer()
    app._direct_control = RecordingDirectControl(
        ApplyOutcome.UNCONFIRMED_TIMEOUT,
        ApplyOutcome.UNCONFIRMED_TIMEOUT,
        ApplyOutcome.SENT,
        ApplyOutcome.UNCONFIRMED_TIMEOUT,
    )

    for _ in range(4):
        app._apply_mode_tracked(_entry())

    assert app._consecutive_apply_unconfirmed == 1
    assert app._apply_success_count == 1
    assert "ERROR" not in app.levels()


def test_duplicate_skip_is_neutral_and_does_not_reset_the_failure_streak():
    """Nothing was transmitted, so it is evidence of neither health nor failure."""
    app = FakeOptimizer()
    app._direct_control = RecordingDirectControl(
        ApplyOutcome.FAILED,
        ApplyOutcome.SKIPPED_DUPLICATE,
        ApplyOutcome.FAILED,
        ApplyOutcome.SKIPPED_DUPLICATE,
        ApplyOutcome.FAILED,
    )

    for _ in range(5):
        app._apply_mode_tracked(_entry())

    assert app._consecutive_apply_failures == 3
    assert app._apply_duplicate_count == 2
    assert app._apply_success_count == 0
    errors = [m for m, lvl in app.logs if lvl == "ERROR"]
    assert len(errors) == 1


def test_dry_run_does_not_report_perfect_health():
    app = FakeOptimizer()
    app._direct_control = RecordingDirectControl(*([ApplyOutcome.DRY_RUN] * 3))

    for _ in range(3):
        assert app._apply_mode_tracked(_entry()) is True

    assert app._apply_dry_run_count == 3
    assert app._apply_success_count == 0
    assert app._apply_failure_count == 0
    attrs = app.states["sensor.battery_inverter_control_health"]["attributes"]
    assert attrs["apply_dry_runs"] == 3
    assert attrs["apply_successes"] == 0


def test_health_sensor_exposes_the_new_counters_without_renaming_the_old():
    app = FakeOptimizer()
    app._direct_control = RecordingDirectControl(ApplyOutcome.UNCONFIRMED_TIMEOUT)

    app._apply_mode_tracked(_entry())

    attrs = app.states["sensor.battery_inverter_control_health"]["attributes"]
    for old in ("apply_failures", "consecutive_apply_failures", "apply_successes",
                "callback_overruns", "slowest_callback"):
        assert old in attrs, f"{old} was renamed — the sensor must stay compatible"
    assert attrs["apply_unconfirmed"] == 1
    assert attrs["consecutive_apply_unconfirmed"] == 1
    assert attrs["apply_duplicates_skipped"] == 0
    assert attrs["apply_dry_runs"] == 0
