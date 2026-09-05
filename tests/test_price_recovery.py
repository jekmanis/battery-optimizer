"""Task 5 reproductions: price recovery and schedule horizon health.

Defect (pre-fix):

* `full_optimize` returns when prices or future prices are missing and never
  schedules a retry.
* `adaptive_optimize` only handles the solar override and the reactive PV
  shortfall; it is not a price-refresh or horizon-extension pass.
* `execute_scheduled_mode` applies `HOLD/no_schedule` when the current slot has
  no entry and nothing ever asks for prices again.

A transient empty fetch, or a response that contains today but no tomorrow
after the configured publication window, therefore leaves the optimizer on an
old or absent plan until an unrelated trigger or the next daily optimization.

Everything here is deterministic: a settable clock, a scripted price service,
and a stubbed planner.  No AppDaemon, no HA, no network.
"""

from __future__ import annotations

import datetime
import inspect

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import BatteryMode, CallbackLock, PricePoint, ScheduleEntry


TZ = datetime.timezone(datetime.timedelta(hours=3))
UTC = datetime.timezone.utc
SLOT = 15


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_prices(start, count, tz=TZ, slot_minutes=SLOT, price=0.10):
    """`count` contiguous intervals from `start`, stepped as UTC instants."""
    base = start.astimezone(UTC)
    return [
        PricePoint(
            time=(base + datetime.timedelta(minutes=i * slot_minutes)).astimezone(tz),
            price=price,
        )
        for i in range(count)
    ]


def slots_between(start, end, slot_minutes=SLOT):
    """Number of `slot_minutes` intervals between two aware instants."""
    delta = end.astimezone(UTC) - start.astimezone(UTC)
    return int(delta.total_seconds() // (slot_minutes * 60))


def day_start(dt, tz=TZ):
    local = dt.astimezone(tz) if dt.tzinfo else dt.replace(tzinfo=tz)
    return datetime.datetime(local.year, local.month, local.day, tzinfo=tz)


class ScriptedPriceService:
    """Price source under the test's control."""

    def __init__(self, result=None):
        self.result = list(result or [])
        self.calls = 0
        self.cached_prices = list(result or [])

    def get_prices(self):
        self.calls += 1
        self.cached_prices = list(self.result)
        return list(self.result)


class FakeFormatter:
    def log_schedule(self, **kwargs):
        pass

    def format_schedule_list(self, schedule):
        return []

    def find_next_events(self, *a, **kw):
        return None, None

    def format_schedule_markdown(self, **kwargs):
        return ""


class FakePvForecast:
    has_forecast = True


class FakeDirectControl:
    """Only what `_on_enabled_change` touches."""

    def __init__(self):
        self.released = 0

    def release_control(self):
        self.released += 1
        return True


class FakeOutcomeTracker:
    def __init__(self):
        self.starts = []
        self.ends = []

    def record_slot_end(self, **kwargs):
        self.ends.append(kwargs)

    def record_slot_start(self, **kwargs):
        self.starts.append(kwargs)


class RecoveryOptimizer(bo.BatteryOptimizer):
    """Orchestrator with AppDaemon and the planner replaced by test doubles.

    The planner is stubbed on purpose: Task 5 is about *whether* a plan is
    rebuilt and applied after prices come back, not about its economics.
    """

    def __init__(self, now, prices=None, tz=TZ, soc=50.0, zone=None, **config_overrides):
        config_overrides.setdefault("slot_minutes", SLOT)
        config_overrides.setdefault("tomorrow_prices_hour", 14)
        config_overrides.setdefault("decision_log_level", 0)
        self.config = bo.BatteryOptimizerConfig(**config_overrides)
        self._lock = CallbackLock()
        self._tz = tz
        # What AppDaemon's own `get_timezone()` offers: a zone name, a tzinfo,
        # an exception, or nothing. `_tz` stays the FIXED-OFFSET value
        # production gets from `_get_local_timezone()` when `self.datetime()`
        # is naive.
        self._zone_source = zone
        self._now = now
        self._soc = soc
        self.enabled = True
        self.override = False
        self.shortfall_checks = 0

        self.logs = []
        self.schedule = {}
        self.expected_soc_schedule = {}
        self.expected_temp_schedule = {}
        self._expected_soc_anchor = None
        self.current_mode = BatteryMode.HOLD
        self.last_optimization = None
        self.applied = []
        self.transitions = []
        self.sensor_updates = 0
        self.planner_calls = []

        self._price_service = ScriptedPriceService(prices)
        self._schedule_formatter = FakeFormatter()
        self._pv_forecast_service = FakePvForecast()
        self._outcome_tracker = FakeOutcomeTracker()
        self._direct_control = FakeDirectControl()

        self._callback_overrun_count = 0
        self._slowest_callback = None
        self._threads_hint_logged = False

        self._last_recalc_trigger = "startup"
        self._last_recalc_time = None
        self._last_soc_deviation = None
        self._last_min_charge_slots = 0
        self._last_charge_slots = []
        self._last_projected_costs = {}
        self._last_dp_soc_trajectory = {}
        self._last_dp_temp_trajectory = {}
        self._last_executed_slot = None
        self._last_executed_monotonic = None
        self._grid_charge_active_until = None
        self._pv_bias_factor = 1.0

        self.run_in_calls = []
        self.cancelled_timers = []
        self._handle_seq = 0

        # State that initialize() builds for the price-recovery owner.
        self._init_price_recovery_state()

    # --- AppDaemon surface -------------------------------------------------
    def log(self, message, level="INFO"):
        self.logs.append((message, level))

    def datetime(self):
        return self._now

    def advance(self, seconds):
        self._now = self._now + datetime.timedelta(seconds=seconds)

    def run_in(self, callback, delay, **kwargs):
        self._handle_seq += 1
        handle = f"h{self._handle_seq}"
        self.run_in_calls.append(
            {"callback": callback, "delay": delay, "kwargs": kwargs, "handle": handle}
        )
        return handle

    def cancel_timer(self, handle):
        self.cancelled_timers.append(handle)

    def get_timezone(self):
        """AppDaemon's own accessor - a configured zone NAME in production."""
        if isinstance(self._zone_source, BaseException):
            raise self._zone_source
        return self._zone_source

    # --- app surface -------------------------------------------------------
    def _get_local_timezone(self):
        return self._tz

    def _is_enabled(self):
        return self.enabled

    def _is_override_active(self):
        return self.override

    def _get_current_soc(self):
        return self._soc

    def _get_battery_temp(self):
        return 20.0

    def _get_pv_power(self):
        return 0.0

    def _get_load_power(self):
        return 500.0

    def _get_inverter_mode(self):
        return "Hold"

    def _predict_load_kw(self, dt):
        return 0.5

    def _predict_pv_kw(self, dt):
        return 0.0

    def _predict_pv_kw_raw(self, dt):
        return 0.0

    def _refresh_pv_bias_factor(self):
        return 1.0

    def _handle_mode_transition(self, mode):
        self.transitions.append(mode)
        self.current_mode = mode

    def _apply_mode_tracked(self, entry):
        self.applied.append(entry)
        return True

    def _update_schedule_sensor(self):
        self.sensor_updates += 1

    def _preserve_mode_on_restart(self, current_slot):
        pass

    def _check_pv_shortfall(self, current_soc):
        self.shortfall_checks += 1
        return False

    # --- planner stub ------------------------------------------------------
    def calculate_min_charge_slots_for_horizon(self, current_soc, prices):
        return 0

    def find_optimal_schedule(self, prices, charge_hours_needed, current_soc=None):
        self.planner_calls.append(
            {"count": len(prices), "soc": current_soc, "now": self._now}
        )
        return {
            bo.canonical_slot_key(p.time): ScheduleEntry(
                time=p.time, mode=BatteryMode.HOLD, reason="planned"
            )
            for p in prices
        }

    def calculate_expected_soc_schedule(self, *args, **kwargs):
        return {}, {}

    @property
    def min_soc(self):
        return 10.0

    @property
    def max_soc(self):
        return 100.0

    @property
    def pv_threshold(self):
        return 500.0

    # --- test conveniences -------------------------------------------------
    def pending_retries(self):
        return [
            call for call in self.run_in_calls
            if getattr(call["callback"], "__name__", "") == "_price_recovery_retry"
        ]

    def fire_latest_retry(self):
        call = self.pending_retries()[-1]
        return call["callback"](dict(call["kwargs"]))

    def set_prices(self, prices):
        self._price_service.result = list(prices)


@pytest.fixture
def base_now():
    """A plain (non-DST) day, before the tomorrow-price publication window."""
    return datetime.datetime(2024, 1, 15, 10, 0, tzinfo=TZ)


def full_day_prices(now, tz=TZ, days=2):
    """Contiguous intervals from local midnight to `days` local midnights on."""
    start = day_start(now, tz)
    end = day_start(now + datetime.timedelta(days=days), tz)
    return make_prices(start, slots_between(start, end), tz=tz)


# ===========================================================================
# Reproduction 1: empty fetch, then a good one
# ===========================================================================

class TestEmptyThenSuccessRecovery:
    def test_full_optimize_with_no_prices_arms_a_bounded_retry(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])

        app.full_optimize(None)

        assert app.schedule == {}, "no schedule can be generated without prices"
        retries = app.pending_retries()
        assert len(retries) == 1, "a price retry must be scheduled"
        assert retries[0]["delay"] == 30

    def test_retry_recovers_without_soc_change_or_daily_optimization(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        soc_before = app._soc
        planner_calls_before = len(app.planner_calls)

        # The service can serve prices again. Nothing else changes: same SOC,
        # no manual action, no second full_optimize.
        app.advance(30)
        app.set_prices(full_day_prices(app.datetime()))
        app.fire_latest_retry()

        assert app._soc == soc_before
        assert len(app.planner_calls) == planner_calls_before + 1
        assert app.schedule, "a schedule must exist after recovery"

    def test_recovered_schedule_is_applied_through_the_normal_path(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        app.advance(30)
        app.set_prices(full_day_prices(app.datetime()))
        app.fire_latest_retry()

        assert app.applied, "the recovered current slot must be applied"
        applied_slot = bo.canonical_slot_key(app.applied[-1].time)
        assert applied_slot == bo.canonical_slot_key(app._align_to_slot(app.datetime()))
        assert app.applied[-1].reason != "no_schedule"

    def test_successful_recovery_stops_retrying(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        before = len(app.pending_retries())
        app.advance(30)
        app.set_prices(full_day_prices(app.datetime()))
        app.fire_latest_retry()

        assert len(app.pending_retries()) == before, "no retry after recovery"
        assert app._price_horizon.attempts == 0

    def test_execute_without_a_schedule_holds_and_asks_for_prices(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])

        app.execute_scheduled_mode({})

        assert app.applied and app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "no_schedule"
        assert len(app.pending_retries()) == 1, "the HOLD must trigger recovery"


# ===========================================================================
# Reproduction 2: today only, after the publication window
# ===========================================================================

class TestMissingTomorrowAfterWindow:
    def test_today_only_after_the_window_is_incomplete(self):
        now = datetime.datetime(2024, 1, 15, 15, 0, tzinfo=TZ)
        today_only = full_day_prices(now, days=1)
        app = RecoveryOptimizer(now, prices=today_only)

        app.full_optimize(None)

        assert app.schedule, "today's prices still produce a plan"
        assert len(app.pending_retries()) == 1, (
            "missing tomorrow after the publication window is a coverage failure"
        )

    def test_today_only_before_the_window_is_not_retried(self):
        now = datetime.datetime(2024, 1, 15, 10, 0, tzinfo=TZ)
        app = RecoveryOptimizer(now, prices=full_day_prices(now, days=1))

        app.full_optimize(None)

        assert app.pending_retries() == [], (
            "tomorrow is legitimately unpublished before the window"
        )

    def test_retries_stop_once_tomorrow_arrives(self):
        now = datetime.datetime(2024, 1, 15, 15, 0, tzinfo=TZ)
        app = RecoveryOptimizer(now, prices=full_day_prices(now, days=1))
        app.full_optimize(None)
        assert len(app.pending_retries()) == 1

        app.advance(30)
        app.set_prices(full_day_prices(app.datetime(), days=2))
        app.fire_latest_retry()

        assert len(app.pending_retries()) == 1, "no further retry once covered"
        assert app._price_horizon.attempts == 0

    def test_backoff_grows_and_is_capped(self):
        now = datetime.datetime(2024, 1, 15, 15, 0, tzinfo=TZ)
        app = RecoveryOptimizer(now, prices=full_day_prices(now, days=1))

        delays = []
        app.full_optimize(None)
        for _ in range(6):
            call = app.pending_retries()[-1]
            delays.append(call["delay"])
            app.advance(call["delay"])
            app.set_prices(full_day_prices(app.datetime(), days=1))
            app.fire_latest_retry()

        assert delays[:3] == [30, 120, 300]
        assert all(d <= app.config.price_retry_max_seconds for d in delays)
        assert delays[3:] == [app.config.price_retry_max_seconds] * len(delays[3:])


# ===========================================================================
# Retained coverage across a failed or shortened refresh
# ===========================================================================

class TestRetainedCoverage:
    def test_a_shortened_response_does_not_destroy_tomorrow(self):
        now = datetime.datetime(2024, 1, 15, 15, 0, tzinfo=TZ)
        app = RecoveryOptimizer(now, prices=full_day_prices(now, days=2))

        first = app.get_prices()
        assert app._price_horizon.evaluate(first, now).ok

        app.advance(900)
        app.set_prices(full_day_prices(app.datetime(), days=1))
        second = app.get_prices()

        health = app._price_horizon.evaluate(second, app.datetime())
        assert health.ok, "the known tomorrow horizon must survive a today-only reply"
        assert len(second) > len(app._price_service.result)

    def test_an_empty_response_keeps_future_coverage(self):
        now = datetime.datetime(2024, 1, 15, 15, 0, tzinfo=TZ)
        app = RecoveryOptimizer(now, prices=full_day_prices(now, days=2))
        app.get_prices()

        app.advance(900)
        app.set_prices([])
        retained = app.get_prices()

        assert retained, "a failed refresh must not discard valid future prices"
        assert app._price_horizon.evaluate(retained, app.datetime()).ok

    def test_fresh_values_win_over_retained_ones(self):
        now = datetime.datetime(2024, 1, 15, 15, 0, tzinfo=TZ)
        app = RecoveryOptimizer(now, prices=full_day_prices(now, days=2))
        app.get_prices()

        updated = [
            PricePoint(time=p.time, price=0.99) for p in full_day_prices(now, days=1)
        ]
        app.set_prices(updated)
        merged = app.get_prices()

        current = bo.canonical_slot_key(app._align_to_slot(now))
        by_key = {bo.instant_key(p.time): p.price for p in merged}
        assert by_key[bo.instant_key(current)] == 0.99

    def test_retained_prices_expire(self):
        now = datetime.datetime(2024, 1, 15, 15, 0, tzinfo=TZ)
        app = RecoveryOptimizer(now, prices=full_day_prices(now, days=2))
        app.get_prices()

        app.advance(int(app.config.price_retain_max_age_hours * 3600) + 3600)
        app.set_prices([])
        assert app.get_prices() == []

    def test_retained_prices_never_cover_the_past(self):
        now = datetime.datetime(2024, 1, 15, 15, 0, tzinfo=TZ)
        app = RecoveryOptimizer(now, prices=full_day_prices(now, days=2))
        app.get_prices()

        app.advance(3600)
        app.set_prices([])
        retained = app.get_prices()
        current = bo.instant_key(app._align_to_slot(app.datetime()))
        assert all(bo.instant_key(p.time) >= current for p in retained)


# ===========================================================================
# One pending retry, generations, disable/enable/override/restart
# ===========================================================================

class TestRetryLifecycle:
    def test_repeated_failures_keep_one_pending_retry(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])

        for _ in range(5):
            app.full_optimize(None)
            app.execute_scheduled_mode({})
            app.adaptive_optimize(None)

        assert len(app.pending_retries()) == 1, "retry storm"

    def test_a_stale_retry_callback_is_inert(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        stale = app.pending_retries()[-1]

        # A newer generation supersedes it (e.g. the optimizer was disabled and
        # re-enabled, or a successful fetch already recovered the horizon).
        app._cancel_price_retry()
        app.set_prices(full_day_prices(app.datetime()))
        planner_before = len(app.planner_calls)

        stale["callback"](dict(stale["kwargs"]))

        assert len(app.planner_calls) == planner_before, "stale timer must be inert"

    def test_disable_cancels_the_pending_retry(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        pending = app.pending_retries()[-1]

        app.enabled = False
        app._on_enabled_change(None, None, "on", "off", {})

        app.set_prices(full_day_prices(app.datetime()))
        pending["callback"](dict(pending["kwargs"]))
        assert app.planner_calls == [], "a disabled app must not plan"

    def test_re_enable_restarts_recovery(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        app.advance(30)
        app.fire_latest_retry()  # still no prices: backoff has escalated
        assert app.pending_retries()[-1]["delay"] == 120

        app.enabled = False
        app._on_enabled_change(None, None, "on", "off", {})
        armed_while_disabled = len(app.pending_retries())

        app.enabled = True
        app._on_enabled_change(None, None, "off", "on", {})

        assert len(app.pending_retries()) == armed_while_disabled + 1
        assert app.pending_retries()[-1]["delay"] == 30, (
            "the backoff restarts on re-enable; the price service state before "
            "the toggle says nothing about it now"
        )
        assert app._price_horizon.attempts == 1

    def test_terminate_makes_retries_inert(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        pending = app.pending_retries()[-1]

        app.terminate()

        app.set_prices(full_day_prices(app.datetime()))
        pending["callback"](dict(pending["kwargs"]))
        assert app.planner_calls == []

    def test_override_recovers_prices_without_commanding_the_inverter(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        app.override = True

        app.advance(30)
        app.set_prices(full_day_prices(app.datetime()))
        app.fire_latest_retry()

        assert app.applied == [], "no automatic command during manual override"
        assert app.schedule, "the plan is still refreshed in the background"

    def test_restart_with_no_schedule_recovers(self, base_now):
        """A fresh app instance (restart) has no schedule and must recover."""
        app = RecoveryOptimizer(base_now, prices=full_day_prices(base_now))
        app.execute_scheduled_mode({})
        assert app.applied[-1].reason == "no_schedule"

        app.advance(30)
        app.fire_latest_retry()
        assert app.schedule
        assert app.applied[-1].reason != "no_schedule"

    def test_disabled_app_does_not_arm_retries_from_execute(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.enabled = False

        app.execute_scheduled_mode({})
        app.adaptive_optimize(None)

        assert app.pending_retries() == []


# ===========================================================================
# The periodic adaptive pass checks horizon health
# ===========================================================================

class TestAdaptiveHorizonCheck:
    def test_adaptive_arms_recovery_when_the_horizon_is_gone(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])

        app.adaptive_optimize(None)

        assert len(app.pending_retries()) == 1

    def test_adaptive_rebuilds_an_exhausted_schedule(self, base_now):
        app = RecoveryOptimizer(base_now, prices=full_day_prices(base_now))
        app.full_optimize(None)
        planner_before = len(app.planner_calls)

        # The plan ran out (every entry is in the past) but prices are fine.
        app.schedule = {}
        app.adaptive_optimize(None)

        assert len(app.planner_calls) > planner_before
        assert app.schedule

    def test_adaptive_is_quiet_when_everything_is_healthy(self, base_now):
        app = RecoveryOptimizer(base_now, prices=full_day_prices(base_now))
        app.full_optimize(None)
        planner_before = len(app.planner_calls)
        fetches_before = app._price_service.calls

        app.adaptive_optimize(None)

        assert len(app.planner_calls) == planner_before
        assert app.pending_retries() == []
        assert app._price_service.calls == fetches_before, (
            "the periodic pass must not add a blocking fetch when the horizon is fine"
        )


# ===========================================================================
# Horizon evaluation: gaps, missing current slot, midnight, DST
# ===========================================================================

class TestHorizonEvaluation:
    def _monitor(self, now, tz=TZ, **overrides):
        app = RecoveryOptimizer(now, prices=[], tz=tz, **overrides)
        return app._price_horizon

    def test_no_prices_is_never_fresh(self, base_now):
        health = self._monitor(base_now).evaluate([], base_now)
        assert not health.ok
        assert health.reason == "no_prices"
        assert health.horizon_end is None

    def test_missing_current_interval_is_detected(self, base_now):
        monitor = self._monitor(base_now)
        start = base_now + datetime.timedelta(hours=1)
        health = monitor.evaluate(make_prices(start, 96), base_now)
        assert not health.ok
        assert health.reason == "missing_current_interval"

    def test_internal_gap_is_reported_as_a_gap(self, base_now):
        monitor = self._monitor(base_now)
        start = day_start(base_now)
        prices = full_day_prices(base_now, days=2)
        cut = bo.instant_key(base_now + datetime.timedelta(hours=2))
        prices = [p for p in prices if bo.instant_key(p.time) != cut]
        health = monitor.evaluate(prices, base_now)
        assert not health.ok
        assert health.reason == "gap"
        assert health.horizon_end == cut

    def test_full_coverage_is_fresh(self, base_now):
        monitor = self._monitor(base_now)
        health = monitor.evaluate(full_day_prices(base_now, days=2), base_now)
        assert health.ok
        assert health.reason == "ok"

    def test_midnight_boundary(self):
        now = datetime.datetime(2024, 1, 15, 23, 45, tzinfo=TZ)
        monitor = self._monitor(now)
        # After the publication window: tomorrow must be covered.
        assert monitor.evaluate(full_day_prices(now, days=1), now).ok is False
        assert monitor.evaluate(full_day_prices(now, days=2), now).ok is True

    def test_spring_forward_day_is_23_hours(self, riga_timezone):
        now = datetime.datetime(2024, 3, 31, 1, 0, tzinfo=riga_timezone)
        monitor = self._monitor(now, tz=riga_timezone)
        start = day_start(now, riga_timezone)
        end = day_start(now + datetime.timedelta(days=1), riga_timezone)
        count = slots_between(start, end)
        assert count == 92, "a spring-forward local day has 23 hours"
        health = monitor.evaluate(make_prices(start, count, tz=riga_timezone), now)
        assert health.ok, "23 hours of intervals cover a 23-hour day"

    def test_autumn_fold_day_is_25_hours(self, riga_timezone):
        now = datetime.datetime(2024, 10, 27, 1, 0, tzinfo=riga_timezone)
        monitor = self._monitor(now, tz=riga_timezone)
        start = day_start(now, riga_timezone)
        end = day_start(now + datetime.timedelta(days=1), riga_timezone)
        count = slots_between(start, end)
        assert count == 100, "an autumn-fold local day has 25 hours"
        assert monitor.evaluate(make_prices(start, count - 4, tz=riga_timezone), now).ok is False
        assert monitor.evaluate(make_prices(start, count, tz=riga_timezone), now).ok is True

    def test_naive_clock_is_tolerated(self):
        now = datetime.datetime(2024, 1, 15, 10, 0)
        app = RecoveryOptimizer(now, prices=[], tz=TZ)
        health = app._price_horizon.evaluate(full_day_prices(now.replace(tzinfo=TZ)), now)
        assert health.ok


# ===========================================================================
# Diagnostics
# ===========================================================================

class TestDiagnostics:
    def test_diagnostics_report_failure_and_pending_retry(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)

        diag = app._price_horizon_diagnostics()
        assert diag["ok"] is False
        assert diag["reason"] == "no_prices"
        assert diag["retry_pending"] is True
        assert diag["retry_attempts"] == 1
        assert diag["retry_next_seconds"] == 120
        assert diag["last_failure_reason"] == "no_prices"
        assert diag["last_success_horizon_end"] is None

    def test_diagnostics_name_a_missing_schedule_separately(self, base_now):
        """An empty current slot is a coverage failure even with valid prices."""
        app = RecoveryOptimizer(base_now, prices=full_day_prices(base_now))
        app.get_prices()
        app.execute_scheduled_mode({})

        diag = app._price_horizon_diagnostics()
        assert diag["last_failure_reason"] == "no_schedule"
        assert diag["retry_pending"] is True

    def test_diagnostics_report_the_recovered_horizon(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        app.advance(30)
        app.set_prices(full_day_prices(app.datetime()))
        app.fire_latest_retry()

        diag = app._price_horizon_diagnostics()
        assert diag["ok"] is True
        assert diag["retry_pending"] is False
        assert diag["retry_attempts"] == 0
        assert diag["last_success_horizon_end"] is not None

    def test_the_schedule_sensor_publishes_the_horizon(self):
        source = inspect.getsource(bo.BatteryOptimizer._update_schedule_sensor)
        assert "_price_horizon_diagnostics()" in source
        assert '"price_horizon"' in source


# ===========================================================================
# Bounded latency / logging
# ===========================================================================

class TestBoundedCost:
    def test_recovery_never_waits_longer_than_the_cap(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        for _ in range(12):
            call = app.pending_retries()[-1]
            assert call["delay"] <= app.config.price_retry_max_seconds
            app.advance(call["delay"])
            app.fire_latest_retry()

    def test_a_raising_price_service_is_a_failure_not_a_crash(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])

        def boom():
            raise RuntimeError("HA restarting")

        app._price_service.get_prices = boom

        app.full_optimize(None)

        assert app.schedule == {}
        assert len(app.pending_retries()) == 1
        assert app._price_horizon.last_health.ok is False

    def test_repeated_failures_do_not_spam_the_log(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        warnings_after_first = sum(1 for _m, lvl in app.logs if lvl == "WARNING")

        for _ in range(6):
            call = app.pending_retries()[-1]
            app.advance(call["delay"])
            app.fire_latest_retry()

        warnings_total = sum(1 for _m, lvl in app.logs if lvl == "WARNING")
        assert warnings_total - warnings_after_first <= 2, (
            "repeated identical coverage failures must be rate-limited"
        )


# ===========================================================================
# Review findings
# ===========================================================================

class TestUnreadableSocDoesNotResetTheBackoff:
    """Finding 1: an unreadable SOC used to restart the backoff every retry.

    `_review_price_horizon` ran FIRST; healthy coverage called `record_success`
    (attempts -> 0) and only then did the SOC check fail and record attempt 1
    again.  The result was a permanent 30 s loop, each iteration paying for a
    blocking price fetch under the app lock.
    """

    def test_delays_grow_while_the_soc_is_unreadable(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        app.set_prices(full_day_prices(app.datetime()))
        app._soc = None

        delays = []
        for _ in range(5):
            call = app.pending_retries()[-1]
            delays.append(call["delay"])
            app.advance(call["delay"])
            app.fire_latest_retry()

        assert delays == [30, 120, 300, 900, 900]

    def test_an_unreadable_soc_does_not_pay_for_a_fetch(self, base_now):
        app = RecoveryOptimizer(base_now, prices=full_day_prices(base_now))
        app.full_optimize(None)
        app.schedule = {}
        # None, not {}: a timer call moments after full_optimize applied the
        # same slot is deduped and would never reach the no_schedule branch.
        app.execute_scheduled_mode(None)
        app._soc = None
        fetches = app._price_service.calls

        for _ in range(3):
            call = app.pending_retries()[-1]
            app.advance(call["delay"])
            app.fire_latest_retry()

        assert app._price_service.calls == fetches, (
            "nothing can be planned without an SOC; do not re-fetch prices "
            "under the app lock to find that out"
        )

    def test_recovery_resumes_once_the_soc_returns(self, base_now):
        app = RecoveryOptimizer(base_now, prices=full_day_prices(base_now))
        app.execute_scheduled_mode({})
        app._soc = None
        call = app.pending_retries()[-1]
        app.advance(call["delay"])
        app.fire_latest_retry()

        app._soc = 42.0
        call = app.pending_retries()[-1]
        app.advance(call["delay"])
        app.fire_latest_retry()

        assert app.schedule
        assert app._price_horizon.attempts == 0


class TestCoverageFailureDoesNotSuppressTheShortfallCheck:
    """Finding 2: `tomorrow_missing` is routine, and it blocked the PV replan.

    From `tomorrow_prices_hour` until tomorrow's intervals publish, the horizon
    is legitimately incomplete - and that window sits inside the PV day.  The
    adaptive pass returned early, so the reactive PV-shortfall replan never ran
    for hours.
    """

    def test_today_only_after_the_window_still_checks_pv_shortfall(self):
        now = datetime.datetime(2024, 1, 15, 14, 30, tzinfo=TZ)
        app = RecoveryOptimizer(now, prices=full_day_prices(now, days=1))
        app.full_optimize(None)
        assert app.schedule, "a plan exists; only tomorrow is missing"
        app.shortfall_checks = 0

        app.adaptive_optimize(None)

        assert app.shortfall_checks == 1
        assert app._price_horizon.last_health.reason == "tomorrow_missing"

    def test_a_rebuild_still_suppresses_the_duplicate_shortfall_pass(self, base_now):
        app = RecoveryOptimizer(base_now, prices=full_day_prices(base_now))
        app.full_optimize(None)
        app.schedule = {}
        app.shortfall_checks = 0

        app.adaptive_optimize(None)

        assert app.schedule, "the exhausted plan was rebuilt"
        assert app.shortfall_checks == 0, (
            "the rebuild already re-planned; a second pass in the same "
            "callback would be wasted work"
        )

    def test_missing_prices_still_check_pv_shortfall(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])

        app.adaptive_optimize(None)

        assert app.shortfall_checks == 1
        assert app._price_retry_pending()


class TestDstBoundaryWithAProductionTimezone:
    """Finding 3: local midnight computed with a FIXED offset is 1 h wrong.

    `_get_local_timezone()` returns `datetime.now().astimezone().tzinfo` - a
    fixed `datetime.timezone` - whenever AppDaemon hands the app a naive
    `self.datetime()`.  Combining a date with that offset gives the wrong
    instant for the midnight on the far side of a DST transition, so a complete
    horizon was judged `tomorrow_missing` all afternoon (and, with finding 2,
    silenced the PV shortfall check too).
    """

    def _spring_prices(self, riga):
        start = datetime.datetime(2024, 3, 30, tzinfo=riga)
        end = datetime.datetime(2024, 4, 1, tzinfo=riga)
        return make_prices(start, slots_between(start, end), tz=riga), end

    def _autumn_prices(self, riga):
        start = datetime.datetime(2024, 10, 26, tzinfo=riga)
        end = datetime.datetime(2024, 10, 28, tzinfo=riga)
        return make_prices(start, slots_between(start, end), tz=riga), end

    def test_spring_forward_required_end_uses_real_dst_rules(self, riga_timezone):
        fixed = datetime.timezone(datetime.timedelta(hours=2))  # Riga on 03-30
        now = datetime.datetime(2024, 3, 30, 15, 0, tzinfo=fixed)
        prices, end = self._spring_prices(riga_timezone)
        app = RecoveryOptimizer(now, prices=prices, tz=fixed, zone=riga_timezone)

        health = app._price_horizon.evaluate(prices, now)

        assert health.required_end == bo.instant_key(end)
        assert health.ok, "a complete 47-hour horizon must not read as incomplete"

    def test_autumn_back_required_end_uses_real_dst_rules(self, riga_timezone):
        fixed = datetime.timezone(datetime.timedelta(hours=3))  # Riga on 10-26
        now = datetime.datetime(2024, 10, 26, 15, 0, tzinfo=fixed)
        prices, end = self._autumn_prices(riga_timezone)
        app = RecoveryOptimizer(now, prices=prices, tz=fixed, zone=riga_timezone)

        health = app._price_horizon.evaluate(prices, now)

        assert health.required_end == bo.instant_key(end)
        assert health.ok

    def test_a_short_horizon_is_still_detected_on_a_transition_day(self, riga_timezone):
        fixed = datetime.timezone(datetime.timedelta(hours=2))
        now = datetime.datetime(2024, 3, 30, 15, 0, tzinfo=fixed)
        prices, _end = self._spring_prices(riga_timezone)
        app = RecoveryOptimizer(now, prices=prices[:-8], tz=fixed, zone=riga_timezone)

        health = app._price_horizon.evaluate(prices[:-8], now)

        assert not health.ok
        assert health.reason == "tomorrow_missing"

    def test_full_optimize_does_not_retry_a_complete_dst_horizon(self, riga_timezone):
        fixed = datetime.timezone(datetime.timedelta(hours=2))
        now = datetime.datetime(2024, 3, 30, 15, 0, tzinfo=fixed)
        prices, _end = self._spring_prices(riga_timezone)
        app = RecoveryOptimizer(now, prices=prices, tz=fixed, zone=riga_timezone)

        app.full_optimize(None)

        assert app.schedule
        assert not app._price_retry_pending()

    def test_a_zone_name_is_resolved(self, base_now):
        zoneinfo = pytest.importorskip("zoneinfo")
        try:
            expected = zoneinfo.ZoneInfo("Europe/Riga")
        except Exception:
            pytest.skip("no tz database available")
        app = RecoveryOptimizer(base_now, prices=[], zone="Europe/Riga")
        assert app._get_region_timezone() == expected

    def test_an_unusable_zone_falls_back_and_says_so(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[], zone="Not/AZone")
        assert app._get_region_timezone() is app._tz
        assert any("Not/AZone" in m for m, _lvl in app.logs)

    def test_a_missing_or_broken_accessor_falls_back(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[], zone=RuntimeError("nope"))
        assert app._get_region_timezone() is app._tz

        app2 = RecoveryOptimizer(base_now, prices=[], zone=None)
        assert app2._get_region_timezone() is app2._tz

    def test_a_fixed_offset_boundary_is_reported_once(self, base_now):
        """No region zone available: the app must say the boundary is degraded."""
        app = RecoveryOptimizer(base_now, prices=[], zone=None)

        for _ in range(3):
            app._price_horizon.evaluate(full_day_prices(base_now), base_now)

        degraded = [m for m, lvl in app.logs if "DST rules" in m]
        assert len(degraded) == 1


class TestReEnableKeepsTheRetryItArmed:
    """Finding 4: the healthy-coverage review cancelled the `no_schedule` retry.

    Re-enable executed the current slot first (no plan -> HOLD/no_schedule ->
    retry armed) and then reviewed the horizon; healthy prices cancelled that
    retry, leaving the app holding until the next adaptive pass.
    """

    def test_re_enable_with_healthy_prices_and_no_schedule(self, base_now):
        app = RecoveryOptimizer(base_now, prices=full_day_prices(base_now))
        app.get_prices()  # populate the retained snapshot
        app.schedule = {}

        app.enabled = False
        app._on_enabled_change(None, None, "on", "off", {})
        app.enabled = True
        app._on_enabled_change(None, None, "off", "on", {})

        assert app._price_retry_pending(), (
            "the empty schedule armed a retry; a healthy price verdict must "
            "not disarm it"
        )

        app.advance(30)
        app.fire_latest_retry()
        assert app.schedule
        assert app.applied[-1].reason != "no_schedule"

    def test_a_non_coverage_retry_survives_a_healthy_review(self, base_now):
        app = RecoveryOptimizer(base_now, prices=full_day_prices(base_now))
        app.get_prices()
        app.execute_scheduled_mode({})  # no schedule -> arms a retry
        assert app._price_horizon.last_failure_reason == "no_schedule"

        app._review_price_horizon(app._price_horizon.retained_prices)

        assert app._price_retry_pending()
        assert app._price_horizon.attempts == 1

    def test_a_coverage_retry_is_still_cancelled_on_recovery(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[])
        app.full_optimize(None)
        assert app._price_retry_pending()

        app.set_prices(full_day_prices(app.datetime()))
        app._review_price_horizon(app.get_prices())

        assert not app._price_retry_pending()
        assert app._price_horizon.attempts == 0


# ===========================================================================
# Configuration
# ===========================================================================

class TestPriceRecoveryConfig:
    def test_defaults(self):
        config = bo.BatteryOptimizerConfig()
        assert config.price_retry_enabled is True
        assert config.price_retry_delays_seconds == (30, 120, 300)
        assert config.price_retry_max_seconds == 900
        assert config.price_retain_max_age_hours == 36.0

    def test_from_args_accepts_a_list(self):
        config = bo.BatteryOptimizerConfig.from_args(
            {"price_retry_delays_seconds": [10, 20]}
        )
        assert config.price_retry_delays_seconds == (10, 20)

    def test_from_args_accepts_a_comma_separated_string(self):
        config = bo.BatteryOptimizerConfig.from_args(
            {"price_retry_delays_seconds": "45, 90, 180"}
        )
        assert config.price_retry_delays_seconds == (45, 90, 180)

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("false", False), ("False", False), ("no", False), ("off", False),
            ("0", False), ("", False),
            ("true", True), ("yes", True), ("on", True), ("1", True),
            (False, False), (True, True), (0, False), (1, True),
        ],
    )
    def test_yaml_strings_are_parsed_as_booleans(self, raw, expected):
        """`price_retry_enabled: "false"` used to be True: bool("false")."""
        config = bo.BatteryOptimizerConfig.from_args({"price_retry_enabled": raw})
        assert config.price_retry_enabled is expected

    def test_a_blank_key_uses_the_default(self):
        config = bo.BatteryOptimizerConfig.from_args(
            {"price_retry_delays_seconds": None, "price_retry_enabled": None}
        )
        assert config.price_retry_delays_seconds == (30, 120, 300)
        assert config.price_retry_enabled is True

    def test_delays_are_floored_and_capped(self):
        config = bo.BatteryOptimizerConfig(
            price_retry_delays_seconds=(0, 100000),
            price_retry_max_seconds=600,
        )
        assert config.price_retry_delays_seconds == (5, 600)

    def test_disabled_recovery_arms_nothing(self, base_now):
        app = RecoveryOptimizer(base_now, prices=[], price_retry_enabled=False)

        app.full_optimize(None)
        app.execute_scheduled_mode({})

        assert app.pending_retries() == []
        assert app.applied[-1].reason == "no_schedule", "still a safe HOLD"
