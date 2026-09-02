"""Execution guards driven by the new config knobs (2026-09-02 production log).

Three defects, all in the orchestrator's execution path:

* **Duplicate slot execution.** `_recalculate_remaining_schedule` applies the
  current slot itself, then the quarter-hour timer applied the identical entry
  seconds later (07:30:06 -> 07:30:12, 08:30:06 -> 08:30:15). DirectControl's
  duplicate suppression absorbed the command, but the call still cost a
  blocking `set_wit_mode` round trip on the single AppDaemon thread.
  -> `execute_dedupe_seconds`.
* **Self-inflicted depletion recalculation.** The DP planned
  `EXPORT (until depleted) -> 10.0%` at 06:45, the battery reached min_soc as
  planned at 06:51, and the safety net paid for a full 17 s re-optimization of
  a state the plan had asked for.
  -> `planned_depletion_margin_percent`.
* **A DISCHARGE re-applied at min SOC.** That re-optimization then re-executed
  the OLD 06:45 DISCHARGE entry at 06:54:05 with SOC at 10.0 %, overriding the
  safety HOLD sent three minutes earlier. At min SOC the SOC stops changing, so
  no further boundary check could correct it.

Plus the grid-charge window that feeds the cost tracker's source attribution.
"""

import datetime

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import BatteryMode, CallbackLock, ScheduleEntry


TZ = datetime.timezone(datetime.timedelta(hours=3))


class FakeOutcomeTracker:
    def __init__(self):
        self.starts = []
        self.ends = []

    def record_slot_end(self, **kwargs):
        self.ends.append(kwargs)

    def record_slot_start(self, **kwargs):
        self.starts.append(kwargs)


class _FakeDirectControl:
    """Only what `_on_enabled_change` touches."""

    def __init__(self, app):
        self._app = app

    def release_control(self):
        self._app.released = True
        return True


class ExecOptimizer(bo.BatteryOptimizer):
    """Minimal stand-in: no AppDaemon initialize(), just the execution path."""

    def __init__(self, now, soc=50.0, **config_overrides):
        self.config = bo.BatteryOptimizerConfig(slot_minutes=15, **config_overrides)
        # initialize() creates this as its very first statement; the doubles
        # skip initialize(), so they build it themselves.
        self._lock = CallbackLock()
        self._now = now
        self._soc = soc
        self.logs = []
        self.schedule = {}
        self.expected_soc_schedule = {}
        self.expected_temp_schedule = {}
        self.current_mode = None
        self.applied = []
        self.transitions = []
        self._outcome_tracker = FakeOutcomeTracker()
        self._apply_result = True
        self._last_executed_slot = None
        self._last_executed_monotonic = None
        self._grid_charge_active_until = None
        self._last_depletion_recalc_time = None
        self.run_in_calls = []
        self.released = False
        self._direct_control = _FakeDirectControl(self)

    # --- AppDaemon surface -------------------------------------------------
    def log(self, message, level="INFO"):
        self.logs.append((message, level))

    def datetime(self):
        return self._now

    def run_in(self, callback, delay, **kwargs):
        self.run_in_calls.append((callback, delay))

    # --- app surface -------------------------------------------------------
    def _get_local_timezone(self):
        return TZ

    def _is_enabled(self):
        return True

    def _is_override_active(self):
        return False

    def _get_current_soc(self):
        return self._soc

    def _get_battery_temp(self):
        return 25.0

    def _get_pv_power(self):
        return 0.0

    def _get_load_power(self):
        return 600.0

    def _predict_load_kw(self, dt):
        return 0.6

    def _predict_pv_kw(self, dt):
        return 0.0

    def _handle_mode_transition(self, mode):
        self.transitions.append(mode)
        self.current_mode = mode

    def _apply_mode_tracked(self, entry):
        self.applied.append(entry)
        return self._apply_result

    @property
    def min_soc(self):
        return 10.0

    @property
    def max_soc(self):
        return 100.0

    def messages(self):
        return [m for m, _lvl in self.logs]


def _slot(hour, minute=0):
    return datetime.datetime(2026, 9, 2, hour, minute, tzinfo=TZ)


def _note(app, slot, mode, outcome=None):
    """`_note_applied_mode` with the default outcome of a real transmission."""
    app._note_applied_mode(
        _entry(slot, mode),
        outcome if outcome is not None else bo.ApplyOutcome.SENT,
    )


def _entry(slot, mode=BatteryMode.HOLD, reason="dp_optimal"):
    return ScheduleEntry(time=slot, mode=mode, reason=reason)


# ---------------------------------------------------------------------------
# Duplicate slot execution
# ---------------------------------------------------------------------------

class TestTimerDedupe:
    def _app(self, **kw):
        app = ExecOptimizer(now=_slot(7, 30) + datetime.timedelta(seconds=6), **kw)
        app.schedule = {_slot(7, 30): _entry(_slot(7, 30))}
        return app

    def test_timer_repeat_of_a_just_applied_slot_is_skipped(self):
        app = self._app()

        app.execute_scheduled_mode(None)              # the recalculation's apply
        app.execute_scheduled_mode({})                # the timer, 6 s later

        assert len(app.applied) == 1
        assert any("Skipping timer execution" in m for m in app.messages())

    def test_an_internal_call_is_never_skipped(self):
        """A recalculation must always be able to apply its new decision."""
        app = self._app()

        app.execute_scheduled_mode({})   # timer first
        app.execute_scheduled_mode(None)  # then a recalculation

        assert len(app.applied) == 2

    def test_a_forced_call_is_never_skipped(self):
        app = self._app()

        app.execute_scheduled_mode(None)
        app.execute_scheduled_mode({}, force=True)

        assert len(app.applied) == 2

    def test_the_next_slot_always_executes(self):
        app = self._app()
        app.execute_scheduled_mode(None)

        app._now = _slot(7, 45) + datetime.timedelta(seconds=5)
        app.schedule[_slot(7, 45)] = _entry(_slot(7, 45))
        app.execute_scheduled_mode({})

        assert len(app.applied) == 2

    def test_a_confirmed_failure_stays_retryable(self):
        """Nothing reached the inverter, so the timer must try again."""
        app = self._app()
        app._apply_result = False

        app.execute_scheduled_mode(None)
        app.execute_scheduled_mode({})

        assert len(app.applied) == 2

    def test_zero_disables_the_dedupe(self):
        app = self._app(execute_dedupe_seconds=0)

        app.execute_scheduled_mode(None)
        app.execute_scheduled_mode({})

        assert len(app.applied) == 2

    def test_a_timer_delayed_past_the_window_executes(self, monkeypatch):
        app = self._app(execute_dedupe_seconds=30)
        clock = {"t": 1000.0}
        monkeypatch.setattr(bo.time, "monotonic", lambda: clock["t"])

        app.execute_scheduled_mode(None)
        clock["t"] += 45.0
        app.execute_scheduled_mode({})

        assert len(app.applied) == 2

    def test_the_skip_also_avoids_a_second_outcome_record(self):
        app = self._app()

        app.execute_scheduled_mode(None)
        app.execute_scheduled_mode({})

        assert len(app._outcome_tracker.starts) == 1


# ---------------------------------------------------------------------------
# Planned depletion
# ---------------------------------------------------------------------------

class TestPlannedDepletion:
    def _app(self, planned_end_soc=None, **kw):
        app = ExecOptimizer(now=_slot(6, 51), soc=10.0, **kw)
        app.current_mode = BatteryMode.DISCHARGE
        app.schedule = {_slot(6, 45): _entry(_slot(6, 45), BatteryMode.DISCHARGE)}
        if planned_end_soc is not None:
            app.expected_soc_schedule = {
                _slot(6, 45): 15.0,
                _slot(7, 0): planned_end_soc,
            }
        return app

    def test_depletion_the_plan_asked_for_does_not_schedule_a_recalc(self):
        app = self._app(planned_end_soc=10.0)

        assert app._check_soc_boundaries(10.0) is True
        assert app.run_in_calls == []
        # The safety HOLD is still applied — only the re-optimization is skipped.
        assert app.applied[-1].mode is BatteryMode.HOLD
        assert app.applied[-1].reason == "safety_min_soc"

    def test_unexpected_depletion_still_schedules_the_recalc(self):
        app = self._app(planned_end_soc=40.0)

        assert app._check_soc_boundaries(10.0) is True
        assert len(app.run_in_calls) == 1
        assert app.run_in_calls[0][1] == 120

    def test_the_margin_is_configurable(self):
        tight = self._app(planned_end_soc=11.5)
        assert len(tight.run_in_calls) == 0 or True
        tight._check_soc_boundaries(10.0)
        assert len(tight.run_in_calls) == 1, "1.5% is outside the 1.0% default margin"

        loose = self._app(planned_end_soc=11.5, planned_depletion_margin_percent=2.0)
        loose._check_soc_boundaries(10.0)
        assert loose.run_in_calls == []

    def test_no_trajectory_falls_back_to_the_old_behaviour(self):
        app = self._app(planned_end_soc=None)

        app._check_soc_boundaries(10.0)

        assert len(app.run_in_calls) == 1

    def test_horizon_end_falls_back_to_the_old_behaviour(self):
        """No entry for the next slot: the answer is unknown, so re-optimize."""
        app = self._app(planned_end_soc=10.0)
        del app.expected_soc_schedule[_slot(7, 0)]

        app._check_soc_boundaries(10.0)

        assert len(app.run_in_calls) == 1


# ---------------------------------------------------------------------------
# DISCHARGE must not be re-applied at min SOC
# ---------------------------------------------------------------------------

class TestMinSocDischargeOverride:
    def test_a_discharge_entry_at_min_soc_becomes_hold(self):
        app = ExecOptimizer(now=_slot(6, 54), soc=10.0)
        app.schedule = {_slot(6, 45): _entry(_slot(6, 45), BatteryMode.DISCHARGE)}

        app.execute_scheduled_mode(None)

        assert app.applied[-1].mode is BatteryMode.HOLD
        assert app.applied[-1].reason == "safety_min_soc"
        assert any("min SOC" in m for m in app.messages())

    def test_a_discharge_entry_above_min_soc_is_untouched(self):
        app = ExecOptimizer(now=_slot(6, 54), soc=11.0)
        app.schedule = {_slot(6, 45): _entry(_slot(6, 45), BatteryMode.DISCHARGE)}

        app.execute_scheduled_mode(None)

        assert app.applied[-1].mode is BatteryMode.DISCHARGE

    def test_an_export_entry_at_min_soc_becomes_hold(self):
        app = ExecOptimizer(now=_slot(6, 45), soc=10.0)
        entry = _entry(_slot(6, 45), BatteryMode.DISCHARGE)
        entry.export_rate = 100
        app.schedule = {_slot(6, 45): entry}

        app.execute_scheduled_mode(None)

        assert app.applied[-1].mode is BatteryMode.HOLD

    def test_charge_at_min_soc_is_untouched(self):
        app = ExecOptimizer(now=_slot(5, 0), soc=10.0)
        app.schedule = {_slot(5, 0): _entry(_slot(5, 0), BatteryMode.CHARGE)}

        app.execute_scheduled_mode(None)

        assert app.applied[-1].mode is BatteryMode.CHARGE


# ---------------------------------------------------------------------------
# Grid-charge window feeding the cost tracker
# ---------------------------------------------------------------------------

class TestGridChargeWindow:
    def test_no_command_yet_means_no_window(self):
        app = ExecOptimizer(now=_slot(5, 0))

        assert app._grid_charge_active() is False

    def test_a_charge_command_opens_a_slot_plus_buffer_window(self):
        app = ExecOptimizer(now=_slot(5, 0))

        _note(app, _slot(5, 0), BatteryMode.CHARGE)

        # slot_minutes 15 + direct_control_buffer_minutes 5 = 20 minutes
        assert app._grid_charge_active_until == _slot(5, 20)
        assert app._grid_charge_active() is True

    def test_the_window_survives_the_transition_to_hold(self):
        """The exact production case: HOLD at 05:15, charge measured 05:15:10."""
        app = ExecOptimizer(now=_slot(5, 0))
        _note(app, _slot(5, 0), BatteryMode.CHARGE)

        app._now = _slot(5, 15)
        _note(app, _slot(5, 15), BatteryMode.HOLD)

        app._now = _slot(5, 15) + datetime.timedelta(seconds=10)
        assert app._grid_charge_active() is True

    def test_the_grace_period_bounds_it_after_a_supersede(self):
        app = ExecOptimizer(now=_slot(5, 0), cost_grid_charge_grace_seconds=120)
        _note(app, _slot(5, 0), BatteryMode.CHARGE)

        app._now = _slot(5, 15)
        _note(app, _slot(5, 15), BatteryMode.HOLD)

        assert app._grid_charge_active_until == _slot(5, 17)
        app._now = _slot(5, 18)
        assert app._grid_charge_active() is False

    def test_the_grace_never_extends_beyond_the_original_command(self):
        app = ExecOptimizer(now=_slot(5, 0), cost_grid_charge_grace_seconds=3600)
        _note(app, _slot(5, 0), BatteryMode.CHARGE)

        app._now = _slot(5, 15)
        _note(app, _slot(5, 15), BatteryMode.HOLD)

        assert app._grid_charge_active_until == _slot(5, 20)

    def test_a_hold_with_no_prior_charge_opens_nothing(self):
        app = ExecOptimizer(now=_slot(5, 0))

        _note(app, _slot(5, 0), BatteryMode.HOLD)

        assert app._grid_charge_active_until is None

    def test_the_window_expires_on_its_own(self):
        app = ExecOptimizer(now=_slot(5, 0))
        _note(app, _slot(5, 0), BatteryMode.CHARGE)

        app._now = _slot(5, 21)

        assert app._grid_charge_active() is False


class TestOnlyATransmittedCommandOpensTheWindow:
    """The window claims the inverter IS grid-charging; only a real send may.

    `apply_mode` returns True for three outcomes the inverter never
    acknowledged, and two of them transmitted nothing at all.
    """

    def test_a_duplicate_charge_does_not_slide_the_window_forward(self):
        app = ExecOptimizer(now=_slot(5, 0))
        _note(app, _slot(5, 0), BatteryMode.CHARGE)
        assert app._grid_charge_active_until == _slot(5, 20)

        # The same CHARGE re-applied 15 min later; DirectControl suppressed it,
        # so the ORIGINAL command's expiry is still the true one.
        app._now = _slot(5, 15)
        _note(app, _slot(5, 15), BatteryMode.CHARGE,
              bo.ApplyOutcome.SKIPPED_DUPLICATE)

        assert app._grid_charge_active_until == _slot(5, 20)
        app._now = _slot(5, 21)
        assert app._grid_charge_active() is False

    def test_a_dry_run_charge_opens_no_window_at_all(self):
        app = ExecOptimizer(now=_slot(5, 0))

        _note(app, _slot(5, 0), BatteryMode.CHARGE, bo.ApplyOutcome.DRY_RUN)

        assert app._grid_charge_active_until is None
        assert app._grid_charge_active() is False

    def test_an_unconfirmed_timeout_still_opens_one(self):
        """The request was already on the wire when the client stopped waiting."""
        app = ExecOptimizer(now=_slot(5, 0))

        _note(app, _slot(5, 0), BatteryMode.CHARGE,
              bo.ApplyOutcome.UNCONFIRMED_TIMEOUT)

        assert app._grid_charge_active_until == _slot(5, 20)

    def test_a_duplicate_hold_still_bounds_an_open_window(self):
        """Superseding is about the app's mode, not about the send."""
        app = ExecOptimizer(now=_slot(5, 0), cost_grid_charge_grace_seconds=120)
        _note(app, _slot(5, 0), BatteryMode.CHARGE)

        app._now = _slot(5, 15)
        _note(app, _slot(5, 15), BatteryMode.HOLD,
              bo.ApplyOutcome.SKIPPED_DUPLICATE)

        assert app._grid_charge_active_until == _slot(5, 17)

    def test_shrinking_twice_never_extends(self):
        app = ExecOptimizer(now=_slot(5, 0), cost_grid_charge_grace_seconds=120)
        _note(app, _slot(5, 0), BatteryMode.CHARGE)

        app._now = _slot(5, 15)
        app._shrink_grid_charge_window()
        app._now = _slot(5, 16)
        app._shrink_grid_charge_window()

        assert app._grid_charge_active_until == _slot(5, 17)


class TestReleaseControlEndsTheWindow:
    """Disabling the optimizer hands the inverter back to passthrough."""

    def test_release_shrinks_the_window_like_a_supersede(self):
        app = ExecOptimizer(now=_slot(5, 0), cost_grid_charge_grace_seconds=120)
        _note(app, _slot(5, 0), BatteryMode.CHARGE)

        app._now = _slot(5, 5)
        app._on_enabled_change("input_boolean.x", "state", "on", "off", {})

        assert app.released is True
        assert app._grid_charge_active_until == _slot(5, 7)
        app._now = _slot(5, 8)
        assert app._grid_charge_active() is False

    def test_release_with_no_open_window_is_a_no_op(self):
        app = ExecOptimizer(now=_slot(5, 0))

        app._on_enabled_change("input_boolean.x", "state", "on", "off", {})

        assert app.released is True
        assert app._grid_charge_active_until is None


# ---------------------------------------------------------------------------
# Callback instrumentation coverage
# ---------------------------------------------------------------------------

class TestCallbackDecoration:
    """Every run_in / run_every / listen_* entry point must be measured.

    `_on_depletion_recalc` was not: its 17.0 s overrun on 2026-09-02 appeared
    only in AppDaemon's own generic warning, never in this app's counters.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "_on_depletion_recalc",
            "record_load_observation",
            "_on_enabled_change",
            "on_override_change",
            "on_manual_mode_change",
            "_on_ha_start",
            # Already decorated before this change — guard against regressions.
            "execute_scheduled_mode",
            "full_optimize",
            "adaptive_optimize",
            "_sample_pv",
            "_record_ambient_observation",
            "_on_soc_change",
            "_on_energy_sensor_change",
        ],
    )
    def test_callback_is_instrumented(self, name):
        func = getattr(bo.BatteryOptimizer, name)

        assert getattr(func, "__wrapped__", None) is not None, (
            f"{name} is an AppDaemon callback but is not wrapped by "
            f"_timed_callback"
        )

    def test_the_decorator_preserves_the_calling_convention(self):
        import inspect

        sig = inspect.signature(bo.BatteryOptimizer._on_depletion_recalc)

        assert str(sig) == "(self, kwargs=None)"

    def test_external_callback_durations_are_recorded(self):
        """DirectControl's run_in callbacks report through this public hook."""
        app = ExecOptimizer(now=_slot(5, 0), callback_warn_seconds=10.0)
        app._callback_overrun_count = 0
        app._slowest_callback = None
        app._threads_hint_logged = False

        app.record_external_callback_duration("DirectControl._verify_mode", 15.8)

        assert app._slowest_callback == ("DirectControl._verify_mode", 15.8)
        assert app._callback_overrun_count == 1
        assert any("_verify_mode took 15.8s" in m for m in app.messages())
