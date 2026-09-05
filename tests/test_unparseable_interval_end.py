"""An ``end`` the source published but nobody can read is not "no end".

Defect (pre-fix)
----------------

``NordPoolPriceService._parse_interval_end`` returned ``None`` for a value it
could not parse - the SAME answer it returns for a record that carries no
``end`` at all. The two are not the same statement:

* No ``end`` is a normal case with a documented answer: the record covers
  exactly one ``slot_minutes`` slot (``docs/scheduling-algorithm.md``, "What
  counts as coverage").
* An ``end`` of ``"not-a-timestamp"`` is a record that TRIED to say how far it
  reaches and produced garbage. Nothing about it can be trusted, least of all
  a width invented on its behalf.

Collapsing the second into the first granted one slot of coverage to a corrupt
record. The maintainer's reproduction: a cheap current record whose ``end`` is
``"not-a-timestamp"``, with every later interval expensive. The monitor
reported ``has_current=True`` for it, the planner charged the live slot at
0.01 EUR/kWh, and the entry carried ``price_source="market"`` - the provenance
marker that exists to make exactly that impossible.

Policy under test
-----------------

``_parse_interval_end`` is TRI-STATE: ``None`` for an absent end,
``MALFORMED_END`` for one that was published and cannot be used, a datetime
otherwise. A record with a malformed end is DROPPED on both fetch paths, with
the same once-an-hour rate limit the other corrupt-record WARNINGs use. The
interval stays a gap.
"""

from __future__ import annotations

import datetime

import pytest

from battery_optimizer_lib import BatteryMode, NordPoolPriceService, PricePoint
from battery_optimizer_lib.price_service import MALFORMED_END

from tests.test_current_slot_price import PlanningOptimizer, current_entry
from tests.test_price_record_validation import NAIVE_10_00, SLOT, _service
from tests.test_price_recovery import TZ, UTC


CHEAP = 0.01
EXPENSIVE = 1.00

DAY = datetime.date(2024, 1, 15)
DAY_END = datetime.datetime(2024, 1, 16, 0, 0, tzinfo=TZ)
SLOT_10_15 = datetime.datetime(2024, 1, 15, 10, 15, tzinfo=TZ)
SLOT_10_30 = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ)

# 10:17 - inside the interval whose declared end is garbage.
NOW = datetime.datetime(2024, 1, 15, 10, 17, tzinfo=TZ)

GARBAGE = "not-a-timestamp"

# Marker for "this record carries no `end` key at all", distinct from a record
# whose `end` is explicitly null.
ABSENT = object()


# ---------------------------------------------------------------------------
# Fetch paths driven from RAW payloads, so the `end` field survives untouched
# ---------------------------------------------------------------------------

def _raw_end(value):
    """The value as a source would put it on the wire."""
    if isinstance(value, datetime.datetime):
        return value.astimezone(UTC).isoformat()
    return value


def service_payload(entries, area="LV"):
    """`nordpool.get_price_indices_for_date` shape, EUR/MWh, raw ends."""
    rows = []
    for start, end, price in entries:
        row = {
            "start": start.astimezone(UTC).isoformat(),
            "price": price * 1000.0,
        }
        if end is not ABSENT:
            row["end"] = _raw_end(end)
        rows.append(row)
    return {area: rows}


def sensor_payload(entries):
    """HACS `raw_today` / `raw_tomorrow` shape, EUR/kWh, raw ends."""
    rows = []
    for start, end, price in entries:
        row = {"start": start.isoformat(), "value": price}
        if end is not ABSENT:
            row["end"] = _raw_end(end)
        rows.append(row)
    return rows


def attach_raw_service_path(app, entries_by_date):
    """The REAL price service on its built-in-integration path."""
    def call_service(service, **kwargs):
        if not service.endswith("get_price_indices_for_date"):
            return None
        date = datetime.date.fromisoformat(kwargs["date"])
        entries = entries_by_date.get(date, [])
        return service_payload(entries) if entries else None

    app._price_service = NordPoolPriceService(
        nordpool_config_entry="cfg",
        nordpool_area="LV",
        nordpool_sensor="",
        ha_url="",
        ha_token="",
        tomorrow_prices_hour=app.config.tomorrow_prices_hour,
        slot_minutes=app.config.slot_minutes,
        get_state_func=lambda *a, **k: None,
        call_service_func=call_service,
        get_datetime_func=app.datetime,
        get_date_func=lambda: app.datetime().date(),
        get_timezone_func=app._get_local_timezone,
        log_func=lambda *a, **k: None,
    )
    return app._price_service


def attach_raw_sensor_path(app, today_entries, tomorrow_entries=()):
    """The REAL price service on its HACS-sensor path."""
    def get_state(entity, attribute=None):
        return {
            "attributes": {
                "raw_today": sensor_payload(today_entries),
                "raw_tomorrow": sensor_payload(tomorrow_entries),
            }
        }

    app._price_service = NordPoolPriceService(
        nordpool_config_entry="",
        nordpool_area="LV",
        nordpool_sensor="sensor.nordpool",
        ha_url="",
        ha_token="",
        tomorrow_prices_hour=app.config.tomorrow_prices_hour,
        slot_minutes=app.config.slot_minutes,
        get_state_func=get_state,
        call_service_func=lambda *a, **k: None,
        get_datetime_func=app.datetime,
        get_date_func=lambda: app.datetime().date(),
        get_timezone_func=app._get_local_timezone,
        log_func=lambda *a, **k: None,
    )
    return app._price_service


def maintainer_entries(end=GARBAGE):
    """The current interval cheap with an unreadable end; the rest expensive.

    Charging at 0.01 and discharging into a day of 1.00 is overwhelmingly
    profitable, so a planner that accepts the invented coverage WILL choose
    CHARGE for the live slot.
    """
    entries = [(SLOT_10_15, end, CHEAP)]
    cursor = SLOT_10_30
    step = datetime.timedelta(minutes=SLOT)
    while cursor < DAY_END:
        entries.append((cursor, cursor + step, EXPENSIVE))
        cursor = cursor + step
    return entries


def _attach(app, attach, entries):
    if attach == "service":
        return attach_raw_service_path(app, {DAY: entries})
    return attach_raw_sensor_path(app, entries)


def _utc(dt):
    return dt.astimezone(UTC)


def _instants(points):
    return {_utc(p.time) for p in points}


# ===========================================================================
# 1. The parser itself is tri-state
# ===========================================================================

class TestTheParserDistinguishesAbsentFromUnreadable:
    def test_an_absent_end_is_none(self):
        service = _service()

        assert service._parse_interval_end(None, None) is None

    @pytest.mark.parametrize("value", [
        GARBAGE,                       # garbage string
        "",                            # empty string
        "   ",                         # whitespace only
        "2024-13-45T99:99:99",         # shaped like a timestamp, isn't one
        1705312800,                    # numeric (a unix epoch, unannounced)
        1705312800.0,
        True,                          # non-string, non-datetime
        ["2024-01-15T10:30:00"],
        {"end": "2024-01-15T10:30:00"},
        datetime.date(2024, 1, 15),    # a date is not an interval end
    ])
    def test_a_published_but_unusable_end_is_not_none(self, value):
        service = _service()

        assert service._parse_interval_end(value, None) is MALFORMED_END, (
            "returning None here is indistinguishable from 'no end published', "
            "and 'no end published' buys the record a slot of coverage"
        )

    def test_a_usable_end_still_parses(self):
        service = _service()

        parsed = service._parse_interval_end("2024-01-15T11:00:00", None)

        assert parsed == datetime.datetime(2024, 1, 15, 11, 0)


# ===========================================================================
# 2. Normalization drops the record and says so
# ===========================================================================

class TestNormalizationDropsAnUnreadableEnd:
    def test_the_record_is_dropped(self):
        service = _service()

        normalized = service._normalize_prices(
            [PricePoint(time=NAIVE_10_00, price=CHEAP, end=GARBAGE)]
        )

        assert normalized == [], (
            "a record whose declared end is garbage states nothing about "
            "coverage; one slot of it is a price invented for 10:00"
        )
        assert len(service.warnings()) == 1

    def test_an_absent_end_still_covers_exactly_one_slot(self):
        service = _service()

        normalized = service._normalize_prices(
            [PricePoint(time=NAIVE_10_00, price=CHEAP)]
        )

        assert [p.time for p in normalized] == [NAIVE_10_00]
        assert service.warnings() == []

    def test_the_healthy_records_around_it_survive(self):
        service = _service()
        step = datetime.timedelta(minutes=SLOT)
        good = NAIVE_10_00
        bad = NAIVE_10_00 + step
        later = NAIVE_10_00 + 2 * step

        normalized = service._normalize_prices([
            PricePoint(time=good, price=0.25, end=bad),
            PricePoint(time=bad, price=9.99, end=GARBAGE),
            PricePoint(time=later, price=0.75, end=later + step),
        ])

        assert [p.time for p in normalized] == [good, later]
        assert [p.price for p in normalized] == [0.25, 0.75]

    def test_the_warning_is_rate_limited_to_once_an_hour(self):
        service = _service()
        record = [PricePoint(time=NAIVE_10_00, price=CHEAP, end=GARBAGE)]

        service._normalize_prices(record)
        service.clock.advance(SLOT)
        service._normalize_prices(record)
        service.clock.advance(SLOT)
        service._normalize_prices(record)

        assert len(service.warnings()) == 1, (
            "every poll re-reads the same corrupt reply; one line per poll is "
            "one line nobody reads"
        )

        service.clock.advance(61)
        service._normalize_prices(record)

        assert len(service.warnings()) == 2, (
            "rate-limited is not silenced: it says so again an hour later"
        )


# ===========================================================================
# 3. Both fetch paths, end to end
# ===========================================================================

class TestBothFetchPathsDropTheRecord:
    @pytest.mark.parametrize("attach", ["service", "sensor"])
    def test_the_current_interval_is_a_gap(self, attach):
        app = PlanningOptimizer(NOW, soc=50.0)
        _attach(app, attach, maintainer_entries())

        fetched = app.get_prices()

        assert _utc(SLOT_10_15) not in _instants(fetched), (
            "the record declaring 10:15 published an unreadable end; giving it "
            "one slot prices the interval the app is living in"
        )
        assert _utc(SLOT_10_30) in _instants(fetched)

    @pytest.mark.parametrize("attach", ["service", "sensor"])
    def test_the_monitor_reports_the_current_interval_missing(self, attach):
        app = PlanningOptimizer(NOW, soc=50.0)
        _attach(app, attach, maintainer_entries())

        health = app._price_horizon.evaluate(app.get_prices(), app.datetime())

        assert health.has_current is False
        assert health.reason == "missing_current_interval"

    @pytest.mark.parametrize("attach", ["service", "sensor"])
    def test_an_absent_end_is_still_one_slot_through_the_fetch(self, attach):
        """The documented rule is untouched by the fix."""
        app = PlanningOptimizer(NOW, soc=50.0)
        _attach(app, attach, maintainer_entries(end=ABSENT))

        fetched = app.get_prices()

        assert _utc(SLOT_10_15) in _instants(fetched)
        assert {_utc(p.time): p.price for p in fetched}[
            _utc(SLOT_10_15)
        ] == pytest.approx(CHEAP)


# ===========================================================================
# 4. Both planning paths: HOLD/no_price, never CHARGE
# ===========================================================================

class TestNeitherPlanningPathChargesOnTheDroppedRecord:
    @pytest.mark.parametrize("attach", ["service", "sensor"])
    def test_full_optimize(self, attach):
        app = PlanningOptimizer(NOW, soc=50.0)
        _attach(app, attach, maintainer_entries())

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
        _attach(app, attach, maintainer_entries())

        app._recalculate_remaining_schedule(app._soc)

        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "no_price"
        assert len(app.pending_retries()) == 1
