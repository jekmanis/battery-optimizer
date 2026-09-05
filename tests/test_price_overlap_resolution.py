"""Overlapping price records: the MOST SPECIFIC one owns the minutes.

Defect (pre-fix)
----------------

``_normalize_prices`` documented its overlap rule as publication order - "a
later interval contributes only the minutes an earlier one did not already
cover" - but it sorted the records by instant first, which discards input
order entirely, and then clipped every record to ``covered_until``. The rule it
actually implemented was EARLIEST START WINS.

The two only differ where they matter. A coarse interval published first and a
finer correction nested inside it (10:00-11:00 at 0.10, then 10:15-10:30 at
0.90) came out as four quarter hours at 0.10: the correction was discarded, and
nothing said so. That is the shape a correction takes.

Decision (see the report and ``docs/scheduling-algorithm.md``): implement the
rule, do not soften the docstring. "Publication order" is not recoverable from
a reply - the records arrive as a list, with no publication timestamps - but
SPECIFICITY is a property of the record itself, and a narrower interval nested
inside a wider one is a more specific statement about the minutes they share.

Policy under test
-----------------

For every minute an interval covers, the winner is decided by a deterministic
key: the NARROWER span first, then the LATER position in the reply. Each minute
is then attributed to exactly one record, so:

* a finer record overrides the coarser one it sits inside, for its own minutes
  only - the rest of the coarse interval survives;
* two records of the same span take the later one;
* nothing is SUMMED. Two intervals that both start inside a slot cannot add up
  to its width and mark it covered when its first minutes never were.
"""

from __future__ import annotations

import datetime

import pytest

from battery_optimizer_lib import PricePoint

from tests.test_price_record_validation import NAIVE_10_00, SLOT, _service


def _at(minutes):
    return NAIVE_10_00 + datetime.timedelta(minutes=minutes)


def _record(start_minutes, end_minutes, price):
    return PricePoint(
        time=_at(start_minutes), price=price, end=_at(end_minutes)
    )


def _prices(normalized):
    return {p.time: round(p.price, 6) for p in normalized}


class TestAFinerRecordOverridesACoarserOne:
    def test_a_correction_nested_in_an_hourly_interval_takes_effect(self):
        service = _service()

        normalized = service._normalize_prices([
            _record(0, 60, 0.10),      # 10:00-11:00, the coarse interval
            _record(15, 30, 0.90),     # 10:15-10:30, the correction
        ])

        assert _prices(normalized) == {
            _at(0): 0.10,
            _at(15): 0.90,
            _at(30): 0.10,
            _at(45): 0.10,
        }, "the correction owns its own quarter hour and only that one"

    def test_the_result_does_not_depend_on_the_order_of_the_reply(self):
        service = _service()
        coarse = _record(0, 60, 0.10)
        fine = _record(15, 30, 0.90)

        first = service._normalize_prices([coarse, fine])
        second = service._normalize_prices([fine, coarse])

        assert _prices(first) == _prices(second), (
            "specificity is a property of the records, not of their position"
        )

    def test_a_correction_finer_than_a_slot_is_weighted_into_it(self):
        """Five minutes of 0.90 inside an hour of 0.10.

        The slot is still fully covered - the coarse record owns the other ten
        minutes - so it is emitted, at the mean of what actually covers it.
        """
        service = _service()

        normalized = service._normalize_prices([
            _record(0, 60, 0.10),
            _record(20, 25, 0.90),
        ])

        assert _prices(normalized)[_at(15)] == pytest.approx(
            (5 * 0.10 + 5 * 0.90 + 5 * 0.10) / 15
        )
        assert _prices(normalized)[_at(0)] == pytest.approx(0.10)
        assert len(normalized) == 4


class TestEqualSpansTakeTheLaterRecord:
    def test_the_second_of_two_identical_intervals_wins(self):
        service = _service()

        normalized = service._normalize_prices([
            _record(0, 15, 0.10),
            _record(0, 15, 0.90),
        ])

        assert _prices(normalized) == {_at(0): 0.90}, (
            "with nothing to tell them apart but their position, the later "
            "record is the correction"
        )

    def test_the_first_still_wins_when_it_comes_last(self):
        service = _service()

        normalized = service._normalize_prices([
            _record(0, 15, 0.90),
            _record(0, 15, 0.10),
        ])

        assert _prices(normalized) == {_at(0): 0.10}


class TestOverlapsAreNeverSummed:
    def test_two_records_starting_inside_a_slot_do_not_fill_it(self):
        """10:05-10:15 and 10:07-10:17: eighteen minutes of records, ten of
        coverage for the 10:00 slot, whose first five minutes nobody
        published."""
        service = _service()

        normalized = service._normalize_prices([
            _record(5, 15, 1.00),
            _record(7, 17, 1.00),
        ])

        assert normalized == [], (
            "a slot whose beginning nobody covered is a gap, however many "
            "overlapping records touch the rest of it"
        )

    def test_the_union_of_two_records_can_still_cover_a_slot(self):
        service = _service()

        normalized = service._normalize_prices([
            _record(0, 10, 0.20),
            _record(5, 15, 0.60),
        ])

        # The narrower span wins ties by lateness; both are 10 minutes, so the
        # LATER record owns 10:05-10:15 and the earlier one keeps 10:00-10:05.
        assert _prices(normalized) == {
            _at(0): round((5 * 0.20 + 10 * 0.60) / 15, 6)
        }
