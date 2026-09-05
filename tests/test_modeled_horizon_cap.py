"""The runaway guard on the modelled horizon is a TIME budget, and it says so.

Defect (pre-fix)
----------------

``modeled_horizon`` stopped after ``len(priced) * 4 + 1000`` slots. The bound
existed for one reason - "a source handing us an interval a decade out must not
spin here" - but it was expressed in a count that has nothing to do with the
horizon the app plans over, and it truncated in SILENCE: the returned sequence
simply ended, the last priced interval was missing from it, and every consumer
downstream (the schedule, the replay, the decision log) described a horizon
that had been shortened without a word.

The count also scaled with the input, so what the guard actually allowed
depended on how many intervals a reply happened to contain: 252 hours for two
points, 346 hours for a full day of quarter hours.

Policy under test
-----------------

The bound is ELAPSED TIME from the first modelled slot. Nothing a source
legitimately publishes comes close to it: the horizon the app can even require
ends at the end of TOMORROW (``PriceHorizonMonitor.required_horizon_end``),
under 48 hours out, and the retention clamp tops out at a week.

When the budget would stop the sequence before the last priced interval, the
priced points beyond it are DROPPED - out of the modelled sequence and out of
everything the caller derives from the price list - and a WARNING says how many
and from when. A shortened horizon is a fact about the plan, not an
implementation detail.
"""

from __future__ import annotations

import datetime

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import BatteryMode, PricePoint

from tests.test_current_slot_price import PlanningOptimizer, attach_service_path
from tests.test_price_recovery import TZ, UTC


SLOT = 15

DAY = datetime.date(2024, 1, 15)
SLOT_10_00 = datetime.datetime(2024, 1, 15, 10, 0, tzinfo=TZ)
SLOT_10_15 = datetime.datetime(2024, 1, 15, 10, 15, tzinfo=TZ)
NOW = datetime.datetime(2024, 1, 15, 10, 0, tzinfo=TZ)

# Ten days out: inside the old count bound (1008 slots for two points), far
# outside anything a price source publishes.
FAR = SLOT_10_00 + datetime.timedelta(days=10)


class _Log:
    def __init__(self):
        self.lines = []

    def __call__(self, message, level="INFO"):
        self.lines.append((message, level))

    def warnings(self):
        return [m for m, level in self.lines if level == "WARNING"]


def _point(when, price=0.10):
    return PricePoint(
        time=when, price=price, end=when + datetime.timedelta(minutes=SLOT)
    )


# ===========================================================================
# Branch 1: a real horizon fits, and nothing is said about it
# ===========================================================================

class TestAPublishedHorizonIsModelledInFull:
    def test_two_days_of_quarter_hours_survive_the_budget(self):
        log = _Log()
        first = datetime.datetime(2024, 1, 15, 0, 0, tzinfo=TZ)
        last = datetime.datetime(2024, 1, 16, 23, 45, tzinfo=TZ)

        modeled = bo.modeled_horizon(
            [_point(first), _point(last)], SLOT, log_fn=log,
        )

        assert len(modeled) == 192, "today plus tomorrow is 192 quarter hours"
        assert modeled[0].time.astimezone(UTC) == first.astimezone(UTC)
        assert modeled[-1].time.astimezone(UTC) == last.astimezone(UTC)
        assert modeled[-1].price == pytest.approx(0.10), (
            "the LAST priced interval is the end of the horizon, not a "
            "casualty of the guard"
        )
        assert log.warnings() == []


# ===========================================================================
# Branch 2: beyond the budget, the points are dropped and it is logged
# ===========================================================================

class TestAnAbsurdTimestampIsDroppedLoudly:
    def _run(self, log, max_hours=None):
        kwargs = {} if max_hours is None else {"max_hours": max_hours}
        return bo.modeled_horizon(
            [_point(SLOT_10_00), _point(FAR, price=9.99)],
            SLOT,
            log_fn=log,
            **kwargs,
        )

    def test_the_sequence_stops_at_the_time_budget(self):
        log = _Log()

        modeled = self._run(log)

        budget = datetime.timedelta(hours=bo.MODELED_HORIZON_MAX_HOURS)
        assert len(modeled) == int(budget.total_seconds() // (SLOT * 60)) + 1
        span = modeled[-1].time.astimezone(UTC) - modeled[0].time.astimezone(UTC)
        assert span == budget

    def test_the_unreachable_priced_point_is_dropped(self):
        log = _Log()

        modeled = self._run(log)

        assert FAR.astimezone(UTC) not in {
            p.time.astimezone(UTC) for p in modeled
        }
        assert all(p.price is None for p in modeled[1:]), (
            "everything after the one real interval is an unpublished slot"
        )

    def test_the_truncation_is_reported(self):
        log = _Log()

        self._run(log)

        warnings = log.warnings()
        assert len(warnings) == 1, (
            "a horizon that stops short of the data is not an implementation "
            "detail; pre-fix this was silent"
        )
        assert "1 priced" in warnings[0]
        assert str(int(bo.MODELED_HORIZON_MAX_HOURS)) in warnings[0]

    def test_the_budget_is_a_parameter_not_a_count(self):
        """Same two points, a one-hour budget: five slots, not 1008."""
        log = _Log()

        modeled = self._run(log, max_hours=1.0)

        assert len(modeled) == 5
        assert len(log.warnings()) == 1


# ===========================================================================
# The caller drops them too
# ===========================================================================

class TestThePlannerDropsWhatItCannotModel:
    def test_no_entry_is_planned_for_an_unreachable_interval(self):
        app = PlanningOptimizer(NOW, soc=50.0)
        # One corrupt timestamp inside an otherwise ordinary reply.
        attach_service_path(
            app,
            {DAY: [_point(SLOT_10_00), _point(SLOT_10_15), _point(FAR, price=9.99)]},
        )

        app.full_optimize(None)

        assert bo.lookup_by_time(
            app.schedule, FAR, app._get_local_timezone()
        ) is None, "a priced interval past the budget is dropped, not planned"
        assert any(
            "priced" in message and "budget" in message.lower()
            for message in [m for m, level in app.logs if level == "WARNING"]
        ), "the planner says the horizon was shortened"
        assert bo.lookup_by_time(
            app.schedule, SLOT_10_00, app._get_local_timezone()
        ) is not None
