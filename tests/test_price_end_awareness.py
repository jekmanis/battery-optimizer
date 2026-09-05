"""A mixed aware/naive record must not take the whole fetch out.

Defect (pre-fix)
----------------

``_interval_end`` compared ``instant_key(end)`` with ``instant_key(point.time)``
directly. ``instant_key`` returns aware values as UTC and leaves naive ones
alone, so when one of the pair carried a timezone and the other did not the
comparison raised ``TypeError`` - out of ``_normalize_prices``, out of
``get_prices``, which does not catch it, and out of whichever callback was
fetching. One malformed record therefore lost the whole reply AND the fetch
that would have noticed the missing horizon.

The combination is reachable in production, not hypothetical: when AppDaemon
reports no timezone, ``_parse_sensor_prices`` and ``_parse_service_prices``
leave ``start`` and ``end`` with whatever awareness their own ISO strings had,
so a source publishing ``start`` bare and ``end`` with an offset produces
exactly this pair.

Policy under test
-----------------

``end`` is normalized to the awareness of its own ``start`` before anything is
compared: converted to local time and stripped when the start is naive,
interpreted as local wall time when the start is aware. It then covers what it
declares - and if what it declares is still not after its start, it is dropped
by the ordinary rule (``tests/test_price_record_validation.py``).
"""

from __future__ import annotations

import datetime

import pytest

from battery_optimizer_lib import PricePoint

from tests.test_price_record_validation import NAIVE_10_00, SLOT, TZ, UTC, _service


AWARE_10_00 = datetime.datetime(2024, 1, 15, 10, 0, tzinfo=TZ)


class TestMixedAwarenessEnds:
    def test_a_naive_start_with_an_aware_end_is_usable(self):
        service = _service(tz=None)

        normalized = service._normalize_prices([
            PricePoint(
                time=NAIVE_10_00,
                price=0.40,
                end=datetime.datetime(2024, 1, 15, 11, 0, tzinfo=UTC),
            )
        ])

        assert len(normalized) == 4, (
            "the record declares an hour and covers four quarter hours; "
            "pre-fix the comparison raised TypeError out of get_prices"
        )
        assert all(p.price == pytest.approx(0.40) for p in normalized)
        assert normalized[0].time == NAIVE_10_00

    def test_an_aware_start_with_a_naive_end_is_usable(self):
        service = _service(tz=TZ)

        normalized = service._normalize_prices([
            PricePoint(
                time=AWARE_10_00,
                price=0.40,
                end=datetime.datetime(2024, 1, 15, 11, 0),
            )
        ])

        assert len(normalized) == 4
        assert normalized[0].time.astimezone(UTC) == AWARE_10_00.astimezone(UTC)

    def test_a_mixed_end_is_read_in_local_time_not_as_a_bare_instant(self):
        """The aware end is CONVERTED, not stripped where it stands.

        With a +03:00 clock, an end published as 08:00Z is 11:00 locally: the
        record covers the hour from 10:00, not a negative span.
        """
        service = _service(tz=TZ)

        normalized = service._normalize_prices([
            PricePoint(
                time=datetime.datetime(2024, 1, 15, 10, 0),
                price=0.40,
                end=datetime.datetime(2024, 1, 15, 8, 0, tzinfo=UTC),
            )
        ])

        assert len(normalized) == 4
        assert service.warnings() == []

    def test_a_mixed_reversed_end_is_still_dropped(self):
        service = _service(tz=None)

        normalized = service._normalize_prices([
            PricePoint(
                time=NAIVE_10_00,
                price=0.40,
                end=datetime.datetime(2024, 1, 15, 9, 0, tzinfo=UTC),
            )
        ])

        assert normalized == []
        assert len(service.warnings()) == 1

    def test_the_whole_fetch_survives_one_mixed_record(self):
        """The sensor path with no configured timezone: exactly the production
        combination, end to end through ``get_prices``."""
        state = {
            "attributes": {
                # `start` without an offset, `end` with one.
                "raw_today": [
                    {
                        "start": "2024-01-15T10:00:00",
                        "end": "2024-01-15T11:00:00+00:00",
                        "value": 0.40,
                    },
                ],
                "raw_tomorrow": [],
            }
        }
        service = _service(tz=None, sensor_state=state)

        prices = service.get_prices()

        assert len(prices) == 4
        assert all(p.price == pytest.approx(0.40) for p in prices)
        assert [p.time for p in prices] == [
            NAIVE_10_00 + datetime.timedelta(minutes=SLOT * i) for i in range(4)
        ]
