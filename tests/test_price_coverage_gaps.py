"""A sparse price reply must not manufacture a price for the slots it omits.

Defect (pre-fix)
----------------

``NordPoolPriceService._normalize_prices`` inferred the source interval width
from the MINIMUM SPACING between the timestamps that survived, then expanded
every point by that factor.  Both parsers threw the explicit ``end`` the
sources publish away (``{start, end, price}`` on the service path,
``{start, end, value}`` on the sensor raw path), so spacing was all it had.

The maintainer's reproduction: a reply containing only 10:00-10:15 at 0.01 and
10:30-10:45 at 1.00.  The minimum spacing between the two surviving timestamps
is 30 minutes, so the "expansion" invented four quarter hours - and 10:15, the
interval nobody published, was published at 0.01.  At 10:17 the monitor then
reported ``has_current=True`` and the planner sent CHARGE with
``price_source="market"``.

Policy under test
-----------------

Source interval boundaries are preserved and a point is expanded only across
its own ``[start, end)``.  A missing record stays a GAP.  Spacing establishes
nothing.  A point that carries no ``end`` covers exactly one ``slot_minutes``
slot; the simple sensor list is the one exception, because the format's own
resolution (24 values for a day, 96 values for a day) IS explicit coverage.
"""

from __future__ import annotations

import datetime

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import BatteryMode, PricePoint

from tests.test_current_slot_price import (
    PlanningOptimizer,
    SLOT,
    attach_sensor_path,
    attach_service_path,
    current_entry,
)
from tests.test_price_recovery import TZ, UTC, day_start, make_prices, slots_between


# ---------------------------------------------------------------------------
# The maintainer's sparse reply
# ---------------------------------------------------------------------------

CHEAP = 0.01
EXPENSIVE = 1.00

DAY = datetime.date(2024, 1, 15)
SLOT_10_00 = datetime.datetime(2024, 1, 15, 10, 0, tzinfo=TZ)
SLOT_10_15 = datetime.datetime(2024, 1, 15, 10, 15, tzinfo=TZ)
SLOT_10_30 = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ)
SLOT_10_45 = datetime.datetime(2024, 1, 15, 10, 45, tzinfo=TZ)

# 10:17 - inside the interval the reply does NOT contain.
NOW = datetime.datetime(2024, 1, 15, 10, 17, tzinfo=TZ)


def _sparse_points():
    """Exactly the maintainer's two records: 10:00-10:15 and 10:30-10:45."""
    return [
        PricePoint(time=SLOT_10_00, price=CHEAP, end=SLOT_10_15),
        PricePoint(time=SLOT_10_30, price=EXPENSIVE, end=SLOT_10_45),
    ]


def _price_service(app, points):
    """Wire the real service to whichever fetch path the test wants."""
    return attach_service_path(app, {DAY: points})


def _utc(dt):
    return dt.astimezone(UTC)


def _instants(points):
    return {_utc(p.time) for p in points}


# ===========================================================================
# 1. The normalization itself
# ===========================================================================

class TestSparseRepliesStaySparse:
    def test_the_omitted_interval_is_not_manufactured(self):
        app = PlanningOptimizer(NOW, soc=50.0)
        service = _price_service(app, _sparse_points())

        normalized = service._normalize_prices(_sparse_points())

        assert _utc(SLOT_10_15) not in _instants(normalized), (
            "10:15 was never published; expanding 10:00 across it invents a "
            "price for the interval the app is living in"
        )
        assert _instants(normalized) == {_utc(SLOT_10_00), _utc(SLOT_10_30)}
        prices = {_utc(p.time): p.price for p in normalized}
        assert prices[_utc(SLOT_10_00)] == pytest.approx(CHEAP)
        assert prices[_utc(SLOT_10_30)] == pytest.approx(EXPENSIVE)

    def test_a_point_without_an_end_covers_exactly_one_slot(self):
        app = PlanningOptimizer(NOW, soc=50.0)
        service = _price_service(app, [])

        normalized = service._normalize_prices(
            [PricePoint(time=SLOT_10_00, price=CHEAP)]
        )

        assert len(normalized) == 1
        assert _utc(normalized[0].time) == _utc(SLOT_10_00)

    def test_an_hourly_interval_expands_to_four_quarters(self):
        app = PlanningOptimizer(NOW, soc=50.0)
        service = _price_service(app, [])
        hour_end = SLOT_10_00 + datetime.timedelta(hours=1)

        normalized = service._normalize_prices(
            [PricePoint(time=SLOT_10_00, price=0.25, end=hour_end)]
        )

        assert _instants(normalized) == {
            _utc(SLOT_10_00), _utc(SLOT_10_15),
            _utc(SLOT_10_30), _utc(SLOT_10_45),
        }
        assert all(p.price == pytest.approx(0.25) for p in normalized)

    def test_a_thirty_minute_interval_expands_to_two(self):
        app = PlanningOptimizer(NOW, soc=50.0)
        service = _price_service(app, [])

        normalized = service._normalize_prices(
            [PricePoint(time=SLOT_10_00, price=0.4, end=SLOT_10_30)]
        )

        assert _instants(normalized) == {_utc(SLOT_10_00), _utc(SLOT_10_15)}

    def test_finer_records_are_aggregated_only_when_they_fill_the_slot(self):
        app = PlanningOptimizer(NOW, soc=50.0)
        service = _price_service(app, [])
        five = datetime.timedelta(minutes=5)
        thirds = [
            PricePoint(
                time=SLOT_10_00 + i * five,
                price=price,
                end=SLOT_10_00 + (i + 1) * five,
            )
            for i, price in enumerate((0.30, 0.60, 0.90))
        ]

        full = service._normalize_prices(thirds)
        assert _instants(full) == {_utc(SLOT_10_00)}
        assert full[0].price == pytest.approx(0.60)

        # Drop the middle five minutes: the slot is no longer covered.
        partial = service._normalize_prices([thirds[0], thirds[2]])
        assert partial == [], (
            "two thirds of a slot is not a price for the slot"
        )


# ===========================================================================
# 2. Both fetch paths, end to end
# ===========================================================================

class TestBothFetchPathsPreserveTheGap:
    def test_service_path(self):
        app = PlanningOptimizer(NOW, soc=50.0)
        _price_service(app, _sparse_points())

        fetched = app.get_prices()

        assert _utc(SLOT_10_15) not in _instants(fetched)
        assert _utc(SLOT_10_30) in _instants(fetched)

    def test_sensor_path(self):
        app = PlanningOptimizer(NOW, soc=50.0)
        attach_sensor_path(app, _sparse_points())

        fetched = app.get_prices()

        assert _utc(SLOT_10_15) not in _instants(fetched)
        assert _utc(SLOT_10_30) in _instants(fetched)

    @pytest.mark.parametrize("attach", ["service", "sensor"])
    def test_the_monitor_reports_the_current_interval_missing(self, attach):
        app = PlanningOptimizer(NOW, soc=50.0)
        if attach == "service":
            _price_service(app, _sparse_points())
        else:
            attach_sensor_path(app, _sparse_points())

        health = app._price_horizon.evaluate(app.get_prices(), app.datetime())

        assert health.has_current is False
        assert health.reason == "missing_current_interval"


# ===========================================================================
# 3. Both planning paths: HOLD/no_price, and the retry stays armed
# ===========================================================================

class TestNeitherPlanningPathChargesOnTheInventedPrice:
    @pytest.mark.parametrize("attach", ["service", "sensor"])
    def test_full_optimize(self, attach):
        app = PlanningOptimizer(NOW, soc=50.0)
        if attach == "service":
            _price_service(app, _sparse_points())
        else:
            attach_sensor_path(app, _sparse_points())

        app.full_optimize(None)

        entry = current_entry(app, SLOT_10_15)
        assert entry is not None
        assert entry.mode == BatteryMode.HOLD
        assert entry.reason == "no_price"
        assert entry.price_source is None
        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "no_price"
        assert len(app.pending_retries()) == 1

    @pytest.mark.parametrize("attach", ["service", "sensor"])
    def test_recalculate_remaining_schedule(self, attach):
        app = PlanningOptimizer(NOW, soc=50.0)
        if attach == "service":
            _price_service(app, _sparse_points())
        else:
            attach_sensor_path(app, _sparse_points())

        app._recalculate_remaining_schedule(app._soc)

        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "no_price"
        assert len(app.pending_retries()) == 1


# ===========================================================================
# 4. DST: the repeated autumn interval keeps its own coverage
# ===========================================================================

class TestAutumnFold:
    def test_the_repeated_interval_stays_distinct(self, riga_timezone):
        app = PlanningOptimizer(
            datetime.datetime(2024, 10, 27, 12, 0, tzinfo=riga_timezone),
            tz=riga_timezone,
            soc=50.0,
        )
        service = _price_service(app, [])
        service.get_timezone = lambda: riga_timezone
        start = datetime.datetime(2024, 10, 27, 0, 0, tzinfo=riga_timezone)
        end = datetime.datetime(2024, 10, 28, 0, 0, tzinfo=riga_timezone)
        hours = int((end.astimezone(UTC) - start.astimezone(UTC)).total_seconds() // 3600)
        assert hours == 25, "the fixture day really is 25 hours long"
        hourly = [
            PricePoint(
                time=(start.astimezone(UTC) + datetime.timedelta(hours=i))
                .astimezone(riga_timezone),
                price=float(i),
                end=(start.astimezone(UTC) + datetime.timedelta(hours=i + 1))
                .astimezone(riga_timezone),
            )
            for i in range(hours)
        ]

        normalized = service._normalize_prices(hourly)

        assert len(normalized) == 100
        repeated = [
            p for p in normalized if p.time.hour == 3 and p.time.minute == 0
        ]
        assert len(repeated) == 2
        assert _utc(repeated[0].time) != _utc(repeated[1].time)
        assert repeated[0].price != repeated[1].price
