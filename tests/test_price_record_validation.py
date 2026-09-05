"""A price record has to be usable before it can be coverage.

Defect (pre-fix)
----------------

``_interval_end`` fell back to ONE SLOT of coverage for any ``end`` that was
not strictly after its ``start``. An absent end is a normal, documented case
and one slot is the right answer for it. An end that comes BEFORE its start is
not: it is evidence the record is corrupt, and answering "one slot" published a
price for an interval on the strength of a field that says the opposite -
coverage being what the schedule, the horizon monitor and the provenance guard
are all built on.

Policy under test
-----------------

* No ``end``: exactly one ``slot_minutes`` slot (unchanged).
* An ``end`` at or before its ``start``: the record is DROPPED, with a WARNING
  rate-limited to once an hour - prices are re-fetched every slot and again on
  every bounded retry, so an unlimited line would be scrolled past exactly like
  the ones that matter.
"""

from __future__ import annotations

import datetime

from battery_optimizer_lib import NordPoolPriceService, PricePoint


SLOT = 15
UTC = datetime.timezone.utc
TZ = datetime.timezone(datetime.timedelta(hours=3))

NAIVE_10_00 = datetime.datetime(2024, 1, 15, 10, 0)


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, minutes):
        self.now = self.now + datetime.timedelta(minutes=minutes)


def _service(tz=None, now=None, sensor_state=None):
    """A price service with a settable clock and a capturing log."""
    clock = Clock(now or datetime.datetime(2024, 1, 15, 10, 0, tzinfo=UTC))
    logs = []
    service = NordPoolPriceService(
        nordpool_config_entry="",
        nordpool_area="LV",
        nordpool_sensor="sensor.nordpool",
        ha_url="",
        ha_token="",
        tomorrow_prices_hour=13,
        slot_minutes=SLOT,
        get_state_func=lambda *a, **k: sensor_state,
        call_service_func=lambda *a, **k: None,
        get_datetime_func=clock,
        get_date_func=lambda: clock().date(),
        get_timezone_func=lambda: tz,
        log_func=lambda message, level="INFO": logs.append((message, level)),
    )
    service.clock = clock
    service.logs = logs
    service.warnings = lambda: [m for m, level in logs if level == "WARNING"]
    return service


def _instants(points):
    return [p.time for p in points]


# ===========================================================================
# 1. An absent end is one slot; a REVERSED end is corruption
# ===========================================================================

class TestAnEndThatDoesNotFollowItsStart:
    def test_a_missing_end_still_covers_exactly_one_slot(self):
        service = _service()

        normalized = service._normalize_prices(
            [PricePoint(time=NAIVE_10_00, price=0.25)]
        )

        assert _instants(normalized) == [NAIVE_10_00]
        assert service.warnings() == [], "an absent end is normal, not corrupt"

    def test_a_reversed_end_drops_the_record(self):
        service = _service()

        normalized = service._normalize_prices([
            PricePoint(
                time=NAIVE_10_00,
                price=0.25,
                end=NAIVE_10_00 - datetime.timedelta(minutes=SLOT),
            )
        ])

        assert normalized == [], (
            "a record whose end precedes its start states nothing about "
            "coverage; one slot of it is a price invented for 10:00"
        )
        assert len(service.warnings()) == 1

    def test_an_end_equal_to_the_start_drops_the_record(self):
        service = _service()

        normalized = service._normalize_prices(
            [PricePoint(time=NAIVE_10_00, price=0.25, end=NAIVE_10_00)]
        )

        assert normalized == []
        assert len(service.warnings()) == 1

    def test_the_healthy_records_around_it_survive(self):
        service = _service()
        good = NAIVE_10_00
        bad = NAIVE_10_00 + datetime.timedelta(minutes=SLOT)
        later = NAIVE_10_00 + datetime.timedelta(minutes=2 * SLOT)

        normalized = service._normalize_prices([
            PricePoint(time=good, price=0.25, end=bad),
            PricePoint(time=bad, price=9.99, end=good),      # reversed
            PricePoint(
                time=later, price=0.75,
                end=later + datetime.timedelta(minutes=SLOT),
            ),
        ])

        assert _instants(normalized) == [good, later]
        assert [p.price for p in normalized] == [0.25, 0.75]

    def test_the_warning_is_rate_limited_to_once_an_hour(self):
        service = _service()
        record = [
            PricePoint(
                time=NAIVE_10_00,
                price=0.25,
                end=NAIVE_10_00 - datetime.timedelta(minutes=SLOT),
            )
        ]

        service._normalize_prices(record)
        service.clock.advance(SLOT)          # the next poll
        service._normalize_prices(record)
        service.clock.advance(SLOT)          # and the one after
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
