"""Tests for timezone_utils module."""

import datetime
import pytest

from battery_optimizer_lib.timezone_utils import (
    normalize_tz_pair,
    datetimes_match_slot,
    dt_ge,
    ensure_local_tz,
    align_to_slot,
    canonical_slot_key,
    next_slot_time,
    next_interval_time,
    lookup_by_time,
)


# Create simple timezone offsets for testing (works without tzdata)
TZ_UTC = datetime.timezone.utc
TZ_PLUS2 = datetime.timezone(datetime.timedelta(hours=2))
TZ_PLUS3 = datetime.timezone(datetime.timedelta(hours=3))


class TestNormalizeTzPair:
    """Tests for normalize_tz_pair function."""

    def test_both_naive_unchanged(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30)
        dt2 = datetime.datetime(2024, 1, 15, 11, 00)
        cmp1, cmp2 = normalize_tz_pair(dt1, dt2)
        assert cmp1 == dt1
        assert cmp2 == dt2

    def test_both_aware_same_tz(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ_PLUS2)
        dt2 = datetime.datetime(2024, 1, 15, 11, 00, tzinfo=TZ_PLUS2)
        cmp1, cmp2 = normalize_tz_pair(dt1, dt2, TZ_PLUS2)
        assert cmp1.tzinfo == TZ_PLUS2
        assert cmp2.tzinfo == TZ_PLUS2

    def test_mixed_aware_naive_strips_tz(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ_PLUS2)
        dt2 = datetime.datetime(2024, 1, 15, 11, 00)  # naive
        cmp1, cmp2 = normalize_tz_pair(dt1, dt2)
        assert cmp1.tzinfo is None
        assert cmp2.tzinfo is None

    def test_converts_to_local_tz(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 0, tzinfo=TZ_UTC)
        dt2 = datetime.datetime(2024, 1, 15, 12, 0, tzinfo=TZ_PLUS2)
        cmp1, cmp2 = normalize_tz_pair(dt1, dt2, TZ_PLUS2)
        # Both should now be in TZ_PLUS2
        assert cmp1.tzinfo == TZ_PLUS2
        assert cmp2.tzinfo == TZ_PLUS2


class TestDatetimesMatchSlot:
    """Tests for datetimes_match_slot function."""

    def test_exact_match(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30)
        dt2 = datetime.datetime(2024, 1, 15, 10, 30)
        assert datetimes_match_slot(dt1, dt2) is True

    def test_different_seconds_match(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30, 0)
        dt2 = datetime.datetime(2024, 1, 15, 10, 30, 45)
        assert datetimes_match_slot(dt1, dt2) is True

    def test_different_minutes_no_match(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30)
        dt2 = datetime.datetime(2024, 1, 15, 10, 31)
        assert datetimes_match_slot(dt1, dt2) is False

    def test_different_hours_no_match(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30)
        dt2 = datetime.datetime(2024, 1, 15, 11, 30)
        assert datetimes_match_slot(dt1, dt2) is False

    def test_different_dates_no_match(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30)
        dt2 = datetime.datetime(2024, 1, 16, 10, 30)
        assert datetimes_match_slot(dt1, dt2) is False

    def test_mixed_tz_naive_matches(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ_PLUS2)
        dt2 = datetime.datetime(2024, 1, 15, 10, 30)  # naive
        assert datetimes_match_slot(dt1, dt2) is True

    def test_dst_fold_slots_are_distinct_instants(self, riga_timezone):
        riga = riga_timezone
        first = datetime.datetime(2024, 10, 27, 3, 0, tzinfo=riga, fold=0)
        second = datetime.datetime(2024, 10, 27, 3, 0, tzinfo=riga, fold=1)
        assert first.utcoffset() != second.utcoffset()
        assert datetimes_match_slot(first, second, riga) is False


class TestDtComparisons:
    """Tests for dt_ge function."""

    def test_dt_ge_equal(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30)
        dt2 = datetime.datetime(2024, 1, 15, 10, 30)
        assert dt_ge(dt1, dt2) is True

    def test_dt_ge_greater(self):
        dt1 = datetime.datetime(2024, 1, 15, 11, 30)
        dt2 = datetime.datetime(2024, 1, 15, 10, 30)
        assert dt_ge(dt1, dt2) is True

    def test_dt_ge_less(self):
        dt1 = datetime.datetime(2024, 1, 15, 9, 30)
        dt2 = datetime.datetime(2024, 1, 15, 10, 30)
        assert dt_ge(dt1, dt2) is False

    def test_comparisons_with_mixed_tz(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ_PLUS2)
        dt2 = datetime.datetime(2024, 1, 15, 10, 30)  # naive
        # After normalization, they should be equal
        assert dt_ge(dt1, dt2) is True
        assert dt_ge(dt2, dt1) is True


class TestAlignToSlot:
    """Tests for align_to_slot function."""

    def test_align_30min_slots(self):
        dt = datetime.datetime(2024, 1, 15, 10, 45, 30)
        aligned = align_to_slot(dt, 30)
        assert aligned == datetime.datetime(2024, 1, 15, 10, 30, 0)

    def test_align_60min_slots(self):
        dt = datetime.datetime(2024, 1, 15, 10, 45, 30)
        aligned = align_to_slot(dt, 60)
        assert aligned == datetime.datetime(2024, 1, 15, 10, 0, 0)

    def test_align_already_aligned(self):
        dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        aligned = align_to_slot(dt, 30)
        assert aligned == dt

    def test_align_preserves_date(self):
        dt = datetime.datetime(2024, 6, 20, 23, 55, 30)
        aligned = align_to_slot(dt, 30)
        assert aligned.date() == datetime.date(2024, 6, 20)
        assert aligned.hour == 23
        assert aligned.minute == 30


class TestNextSlotTime:
    """Tests for next_slot_time function."""

    def test_next_30min_slot(self):
        now = datetime.datetime(2024, 1, 15, 10, 15, 0)
        next_slot = next_slot_time(now, 30)
        assert next_slot.hour == 10
        assert next_slot.minute == 30
        assert next_slot.second == 5  # 5 second offset

    def test_next_60min_slot(self):
        now = datetime.datetime(2024, 1, 15, 10, 15, 0)
        next_slot = next_slot_time(now, 60)
        assert next_slot.hour == 11
        assert next_slot.minute == 0

    def test_next_slot_crosses_midnight(self):
        now = datetime.datetime(2024, 1, 15, 23, 45, 0)
        next_slot = next_slot_time(now, 30)
        assert next_slot.date() == datetime.date(2024, 1, 16)
        assert next_slot.hour == 0
        assert next_slot.minute == 0

    def test_next_slot_enters_second_dst_fold(self, riga_timezone):
        riga = riga_timezone
        now = datetime.datetime(2024, 10, 27, 3, 45, tzinfo=riga, fold=0)
        next_slot = next_slot_time(now, 15, riga)
        assert (next_slot.hour, next_slot.minute, next_slot.fold) == (3, 0, 1)

    def test_next_slot_skips_spring_gap(self, riga_timezone):
        riga = riga_timezone
        now = datetime.datetime(2024, 3, 31, 2, 45, tzinfo=riga)
        next_slot = next_slot_time(now, 15, riga)
        assert (next_slot.hour, next_slot.minute) == (4, 0)


class TestNextIntervalTime:
    """Tests for next_interval_time function."""

    def test_next_15min_interval(self):
        now = datetime.datetime(2024, 1, 15, 10, 8, 0)
        next_int = next_interval_time(now, 15)
        assert next_int.hour == 10
        assert next_int.minute == 15

    def test_minimum_interval_clamped(self):
        now = datetime.datetime(2024, 1, 15, 10, 0, 0)
        next_int = next_interval_time(now, 0)  # Should be clamped to 1
        assert next_int.minute == 1


class TestLookupByHour:
    """Tests for lookup_by_time function."""

    def test_direct_lookup(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30)
        dt2 = datetime.datetime(2024, 1, 15, 11, 00)
        data = {dt1: "value1", dt2: "value2"}
        assert lookup_by_time(data, dt1) == "value1"
        assert lookup_by_time(data, dt2) == "value2"

    def test_fallback_matching(self):
        dt_aware = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ_PLUS2)
        dt_naive = datetime.datetime(2024, 1, 15, 10, 30)
        data = {dt_aware: "value1"}
        # Should find via slot matching
        result = lookup_by_time(data, dt_naive, TZ_PLUS2)
        assert result == "value1"

    def test_not_found(self):
        dt1 = datetime.datetime(2024, 1, 15, 10, 30)
        dt_missing = datetime.datetime(2024, 1, 15, 12, 30)
        data = {dt1: "value1"}
        assert lookup_by_time(data, dt_missing) is None

    def test_empty_dict(self):
        dt = datetime.datetime(2024, 1, 15, 10, 30)
        assert lookup_by_time({}, dt) is None

    def test_lookup_distinguishes_dst_fold_with_utc_keys(self, riga_timezone):
        riga = riga_timezone
        first = datetime.datetime(2024, 10, 27, 3, 0, tzinfo=riga, fold=0)
        second = datetime.datetime(2024, 10, 27, 3, 0, tzinfo=riga, fold=1)
        data = {
            first.astimezone(datetime.timezone.utc): "summer",
            second.astimezone(datetime.timezone.utc): "winter",
        }
        assert lookup_by_time(data, first, riga) == "summer"
        assert lookup_by_time(data, second, riga) == "winter"

    def test_canonical_slot_keys_keep_both_dst_fold_slots(self, riga_timezone):
        first = datetime.datetime(2024, 10, 27, 3, 0, tzinfo=riga_timezone, fold=0)
        second = datetime.datetime(2024, 10, 27, 3, 0, tzinfo=riga_timezone, fold=1)
        data = {
            canonical_slot_key(first): "summer",
            canonical_slot_key(second): "winter",
        }
        assert len(data) == 2
        assert lookup_by_time(data, first, riga_timezone) == "summer"
        assert lookup_by_time(data, second, riga_timezone) == "winter"
