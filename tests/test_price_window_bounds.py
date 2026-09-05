"""A published interval is bounded BEFORE it is expanded onto the slot grid.

Defect (pre-fix)
----------------

``NordPoolPriceService._normalize_prices`` walked every slot inside a record's
declared ``[start, end)`` and allocated a bucket for each one. Nothing bounded
that span. ``modeled_horizon``'s 168-hour budget is the only guard the price
path had, and it runs much later - on the RESULT, after every bucket has been
built.

A single record declaring a year therefore produced 35,040 quarter-hour
buckets, silently, inside the app lock and on a callback that is supposed to
finish in milliseconds. The guard that would eventually discard them had
already been paid for in memory and time.

Policy under test
-----------------

Two bounds, both applied to the SOURCE window before any expansion:

* ``MAX_RECORD_SPAN_HOURS`` (24) - the longest span a single published record
  may declare. Nord Pool publishes 15- or 60-minute intervals and a day-ahead
  block never exceeds a day. A longer record is rejected with the same
  once-an-hour malformed-record WARNING.
* ``MAX_NORMALIZED_WINDOW_HOURS`` (168) - the widest window the reply as a
  whole may be mapped onto, measured from its EARLIEST record. It is the same
  week as ``battery_optimizer.MODELED_HORIZON_MAX_HOURS``, which imports the
  constant so the two cannot drift. Records past it are dropped and one WARNING
  says how many slots went with them.

Ordinary replies - 15-minute records, hourly records, a two-day horizon, a
25-hour DST day - are untouched.
"""

from __future__ import annotations

import datetime

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import PricePoint
from battery_optimizer_lib.price_service import (
    MAX_NORMALIZED_WINDOW_HOURS,
    MAX_RECORD_SPAN_HOURS,
)

from tests.test_current_slot_price import PlanningOptimizer
from tests.test_price_record_validation import SLOT, _service
from tests.test_price_recovery import TZ, UTC
from tests.test_unparseable_interval_end import attach_raw_service_path


DAY = datetime.date(2024, 1, 15)
MIDNIGHT = datetime.datetime(2024, 1, 15, 0, 0, tzinfo=TZ)
NOW = datetime.datetime(2024, 1, 15, 10, 7, tzinfo=TZ)

HOUR = datetime.timedelta(hours=1)
STEP = datetime.timedelta(minutes=SLOT)

# Slots inside the 168-hour window at the 15-minute resolution.
WINDOW_SLOTS = int(MAX_NORMALIZED_WINDOW_HOURS * 60 // SLOT)


def _tz_service(tz=TZ):
    """The capturing-log service from ``test_price_record_validation``."""
    return _service(tz=tz, now=NOW)


def _count_shifts(service):
    """Record every ``_shift`` call; the expansion walk makes one per slot."""
    calls = []
    original = service._shift

    def counting(dt, minutes):
        calls.append(minutes)
        return original(dt, minutes)

    service._shift = counting
    return calls


def _record(start, span, price=0.10):
    return PricePoint(time=start, price=price, end=start + span)


def _quarter_hours(start, count, price=0.10):
    return [_record(start + i * STEP, STEP, price) for i in range(count)]


def _hours(start, count, price=0.10):
    return [_record(start + i * HOUR, HOUR, price) for i in range(count)]


# ===========================================================================
# 1. A single record may not declare more than a day
# ===========================================================================

class TestOneRecordCannotDeclareAnUnboundedSpan:
    def test_a_one_year_record_is_rejected_before_any_expansion(self):
        service = _tz_service()
        shifts = _count_shifts(service)
        year = datetime.datetime(2025, 1, 15, 0, 0, tzinfo=TZ) - MIDNIGHT

        normalized = service._normalize_prices([_record(MIDNIGHT, year)])

        assert normalized == [], (
            "pre-fix this allocated 35,040 buckets - 8,760 hours of them - "
            "before anything downstream got the chance to discard them"
        )
        assert shifts == [], (
            "the expansion walk must never have started: it makes one _shift "
            "call per slot, and the record was rejected on its declared span"
        )
        assert len(service.warnings()) == 1

    def test_a_twenty_five_hour_record_is_rejected(self):
        service = _tz_service()

        normalized = service._normalize_prices(
            [_record(MIDNIGHT, datetime.timedelta(hours=25))]
        )

        assert normalized == []
        assert len(service.warnings()) == 1

    def test_a_twenty_four_hour_record_is_accepted(self):
        service = _tz_service()

        normalized = service._normalize_prices(
            [_record(MIDNIGHT, datetime.timedelta(hours=24))]
        )

        assert len(normalized) == 96, (
            "a day-ahead block for a whole day is at the limit, not past it"
        )
        assert service.warnings() == []

    def test_the_healthy_records_around_it_survive(self):
        service = _tz_service()
        ordinary = _quarter_hours(MIDNIGHT, 4, price=0.20)
        absurd = _record(
            MIDNIGHT + 4 * STEP, datetime.timedelta(days=365), price=9.99
        )

        normalized = service._normalize_prices(ordinary + [absurd])

        assert [p.time for p in normalized] == [p.time for p in ordinary]
        assert all(p.price == pytest.approx(0.20) for p in normalized)
        assert len(service.warnings()) == 1

    def test_the_warning_is_rate_limited_to_once_an_hour(self):
        service = _tz_service()
        record = [_record(MIDNIGHT, datetime.timedelta(days=365))]

        service._normalize_prices(record)
        service.clock.advance(SLOT)
        service._normalize_prices(record)

        assert len(service.warnings()) == 1

        service.clock.advance(61)
        service._normalize_prices(record)

        assert len(service.warnings()) == 2


# ===========================================================================
# 2. Ordinary replies are untouched
# ===========================================================================

class TestOrdinaryRepliesAreUnaffected:
    def test_a_day_of_quarter_hour_records(self):
        service = _tz_service()

        normalized = service._normalize_prices(_quarter_hours(MIDNIGHT, 96))

        assert len(normalized) == 96
        assert service.warnings() == []

    def test_a_day_of_hourly_records(self):
        service = _tz_service()

        normalized = service._normalize_prices(_hours(MIDNIGHT, 24))

        assert len(normalized) == 96
        assert service.warnings() == []

    def test_a_legitimate_two_day_horizon_is_fully_expanded(self):
        service = _tz_service()

        normalized = service._normalize_prices(_hours(MIDNIGHT, 48))

        assert len(normalized) == 192, "today plus tomorrow, at 15 minutes"
        assert service.warnings() == []

    def test_the_spring_dst_day_of_hourly_records(self, riga_timezone):
        """23 local hours, published as 23 hourly records."""
        service = _service(tz=riga_timezone, now=NOW)
        start = datetime.datetime(2024, 3, 31, 0, 0, tzinfo=riga_timezone)
        end = datetime.datetime(2024, 4, 1, 0, 0, tzinfo=riga_timezone)
        hours = int(
            (end.astimezone(UTC) - start.astimezone(UTC)).total_seconds() // 3600
        )
        assert hours == 23, "the fixture day really is 23 hours long"
        hourly = [
            _record(
                (start.astimezone(UTC) + i * HOUR).astimezone(riga_timezone),
                HOUR,
                price=float(i),
            )
            for i in range(hours)
        ]

        normalized = service._normalize_prices(hourly)

        assert len(normalized) == 92
        assert service.warnings() == []

    def test_the_autumn_dst_day_of_hourly_records(self, riga_timezone):
        """25 local hours - and no record longer than one of them."""
        service = _service(tz=riga_timezone, now=NOW)
        start = datetime.datetime(2024, 10, 27, 0, 0, tzinfo=riga_timezone)
        end = datetime.datetime(2024, 10, 28, 0, 0, tzinfo=riga_timezone)
        hours = int(
            (end.astimezone(UTC) - start.astimezone(UTC)).total_seconds() // 3600
        )
        assert hours == 25
        hourly = [
            _record(
                (start.astimezone(UTC) + i * HOUR).astimezone(riga_timezone),
                HOUR,
                price=float(i),
            )
            for i in range(hours)
        ]

        normalized = service._normalize_prices(hourly)

        assert len(normalized) == 100
        assert service.warnings() == []


# ===========================================================================
# 3. The reply as a whole is bounded to one week from its earliest record
# ===========================================================================

class TestTheNormalizedWindowIsBounded:
    def test_it_is_the_same_week_the_planner_models(self):
        assert MAX_NORMALIZED_WINDOW_HOURS == bo.MODELED_HORIZON_MAX_HOURS, (
            "expanding past what the planner will model spends memory to "
            "produce slots that are then discarded"
        )

    def test_a_fortnight_of_hourly_records_stops_at_the_bound(self):
        service = _tz_service()

        normalized = service._normalize_prices(_hours(MIDNIGHT, 24 * 14))

        assert len(normalized) == WINDOW_SLOTS
        span = (
            normalized[-1].time.astimezone(UTC)
            - normalized[0].time.astimezone(UTC)
        )
        assert span == datetime.timedelta(hours=MAX_NORMALIZED_WINDOW_HOURS) - STEP
        warnings = service.warnings()
        assert len(warnings) == 1
        assert str(WINDOW_SLOTS) in warnings[0], (
            "a truncated horizon says how much of it went: pre-fix the whole "
            "fortnight was expanded and then thrown away downstream"
        )

    def test_a_far_future_record_is_dropped(self):
        service = _tz_service()
        far = MIDNIGHT + datetime.timedelta(days=10)

        normalized = service._normalize_prices(
            _quarter_hours(MIDNIGHT, 4) + [_record(far, STEP, price=9.99)]
        )

        assert far.astimezone(UTC) not in {
            p.time.astimezone(UTC) for p in normalized
        }
        assert len(normalized) == 4
        assert len(service.warnings()) == 1

    def test_a_record_straddling_the_bound_is_truncated_not_dropped(self):
        service = _tz_service()
        bound = MIDNIGHT + datetime.timedelta(hours=MAX_NORMALIZED_WINDOW_HOURS)
        straddler = bound - HOUR

        normalized = service._normalize_prices(
            [_record(MIDNIGHT, STEP)]
            + [_record(straddler, datetime.timedelta(hours=24), price=0.50)]
        )

        inside = [
            p for p in normalized
            if p.time.astimezone(UTC) >= straddler.astimezone(UTC)
        ]
        assert len(inside) == 4, "the hour inside the window survives"
        assert all(p.price == pytest.approx(0.50) for p in inside)
        assert max(
            p.time.astimezone(UTC) for p in normalized
        ) < bound.astimezone(UTC)
        assert len(service.warnings()) == 1

    def test_the_window_is_measured_from_the_earliest_record(self):
        """A reply that starts in the past still reaches a week forward."""
        service = _tz_service()
        late = MIDNIGHT + datetime.timedelta(hours=100)

        normalized = service._normalize_prices(
            [_record(MIDNIGHT, HOUR), _record(late, HOUR)]
        )

        assert len(normalized) == 8, "both hours, four quarter hours each"
        assert service.warnings() == []


# ===========================================================================
# 4. Nothing downstream is handed an unbounded list
# ===========================================================================

class TestTheHorizonMonitorIsNeverHandedAnUnboundedList:
    def _entries(self):
        """An ordinary day plus one record declaring a year."""
        entries = [
            (MIDNIGHT + i * STEP, MIDNIGHT + (i + 1) * STEP, 0.10)
            for i in range(96)
        ]
        entries.append(
            (MIDNIGHT, datetime.datetime(2025, 1, 15, 0, 0, tzinfo=TZ), 0.01)
        )
        return entries

    def test_get_prices_is_bounded(self):
        app = PlanningOptimizer(NOW, soc=50.0)
        attach_raw_service_path(app, {DAY: self._entries()})

        prices = app.get_prices()

        assert len(prices) == 96, (
            "the year-long record is rejected; the ordinary day survives"
        )
        assert len(prices) <= WINDOW_SLOTS

    def test_the_retained_set_stays_bounded(self):
        """`merge_with_retained` only ever sees the bounded list.

        Retained intervals are pruned to the FUTURE on every merge and dropped
        wholesale past `price_retain_max_age_hours`, so the set cannot grow
        beyond what one bounded reply contains.
        """
        app = PlanningOptimizer(NOW, soc=50.0)
        attach_raw_service_path(app, {DAY: self._entries()})

        app.get_prices()
        app.get_prices()

        retained = app._price_horizon.retained_prices
        assert len(retained) <= WINDOW_SLOTS
        assert all(
            p.time.astimezone(UTC) >= app._align_to_slot(NOW).astimezone(UTC)
            for p in retained
        ), "only future intervals are ever retained"

    def test_evaluate_sees_the_bounded_list(self):
        app = PlanningOptimizer(NOW, soc=50.0)
        attach_raw_service_path(app, {DAY: self._entries()})

        health = app._price_horizon.evaluate(app.get_prices(), app.datetime())

        assert health.interval_count == 96
        assert health.has_current is True
