"""
Timezone utilities for the Battery Optimizer.

Provides functions for handling timezone-aware and naive datetime comparisons,
slot alignment, and dictionary lookups with timezone normalization.
"""

import datetime
from typing import Dict, Optional, Tuple, TypeVar

T = TypeVar("T")
UTC = datetime.timezone.utc


def is_aware(dt: datetime.datetime) -> bool:
    """Return whether *dt* identifies an unambiguous instant.

    Public because "can these two values be compared at all?" is a question
    outside this module too: a naive app clock alongside aware price
    timestamps is a supported combination (it is why :func:`dt_ge` and
    :func:`datetimes_match_slot` exist), and code that compares the two
    directly raises ``TypeError``.
    """
    return dt.tzinfo is not None and dt.utcoffset() is not None


def instant_key(dt: datetime.datetime) -> datetime.datetime:
    """Return an aware datetime as UTC, leaving legacy naive values unchanged."""
    return dt.astimezone(UTC) if is_aware(dt) else dt


def canonical_slot_key(dt: datetime.datetime) -> datetime.datetime:
    """Return a datetime safe for use as a slot dictionary key.

    Region-zone datetimes in an autumn fold can compare and hash equal even
    when their ``fold`` values identify different instants.  A fixed-offset
    representation preserves the local clock fields while keeping both
    physical intervals distinct.
    """
    if not is_aware(dt):
        return dt
    return dt.astimezone(datetime.timezone(dt.utcoffset()))


def normalize_tz_pair(
    dt1: datetime.datetime,
    dt2: datetime.datetime,
    local_tz: Optional[datetime.tzinfo] = None
) -> Tuple[datetime.datetime, datetime.datetime]:
    """
    Normalize two datetimes to be comparable.

    If both have timezone info and local_tz is provided, converts both to local_tz.
    If one has timezone and one doesn't, strips timezone from the aware one.

    Args:
        dt1: First datetime
        dt2: Second datetime
        local_tz: Optional local timezone to convert to

    Returns:
        Tuple of (normalized_dt1, normalized_dt2) that can be safely compared
    """
    cmp1, cmp2 = dt1, dt2

    # Convert to local timezone if both are aware and local_tz is provided
    if local_tz is not None:
        if dt1.tzinfo is not None:
            cmp1 = dt1.astimezone(local_tz)
        if dt2.tzinfo is not None:
            cmp2 = dt2.astimezone(local_tz)

    # Handle mixed aware/naive by stripping tzinfo
    if cmp1.tzinfo is not None and cmp2.tzinfo is None:
        cmp1 = cmp1.replace(tzinfo=None)
    elif cmp1.tzinfo is None and cmp2.tzinfo is not None:
        cmp2 = cmp2.replace(tzinfo=None)

    return cmp1, cmp2


def datetimes_match_slot(
    dt1: datetime.datetime,
    dt2: datetime.datetime,
    local_tz: Optional[datetime.tzinfo] = None
) -> bool:
    """
    Check if two datetimes refer to the same time slot.

    Compares date, hour, and minute after timezone normalization.

    Args:
        dt1: First datetime
        dt2: Second datetime
        local_tz: Optional local timezone for conversion

    Returns:
        True if both datetimes represent the same slot
    """
    # Aware values identify physical instants.  Comparing local components loses
    # that identity during the autumn DST fold (there are two 03:00 slots).
    if is_aware(dt1) and is_aware(dt2):
        cmp1, cmp2 = dt1.astimezone(UTC), dt2.astimezone(UTC)
    else:
        # Preserve compatibility for legacy naive schedule keys: a naive value
        # is interpreted as local wall time.
        cmp1, cmp2 = normalize_tz_pair(dt1, dt2, local_tz)
    return (
        cmp1.date() == cmp2.date() and
        cmp1.hour == cmp2.hour and
        cmp1.minute == cmp2.minute
    )


def dt_ge(
    dt1: datetime.datetime,
    dt2: datetime.datetime,
    local_tz: Optional[datetime.tzinfo] = None
) -> bool:
    """
    Check if dt1 >= dt2 with timezone normalization.

    Args:
        dt1: First datetime
        dt2: Second datetime
        local_tz: Optional local timezone for conversion

    Returns:
        True if dt1 >= dt2
    """
    if is_aware(dt1) and is_aware(dt2):
        cmp1, cmp2 = dt1.astimezone(UTC), dt2.astimezone(UTC)
    else:
        cmp1, cmp2 = normalize_tz_pair(dt1, dt2, local_tz)
    return cmp1 >= cmp2


def ensure_local_tz(
    dt: datetime.datetime,
    local_tz: Optional[datetime.tzinfo]
) -> datetime.datetime:
    """
    Ensure datetime is in local timezone.

    If dt has timezone info and local_tz is provided, converts to local_tz.
    If dt is naive and local_tz is provided, adds local_tz.

    Args:
        dt: Datetime to normalize
        local_tz: Local timezone

    Returns:
        Datetime in local timezone (or unchanged if local_tz is None)
    """
    if local_tz is None:
        return dt
    if dt.tzinfo is not None:
        return dt.astimezone(local_tz)
    return dt.replace(tzinfo=local_tz)


def align_to_slot(
    dt: datetime.datetime,
    slot_minutes: int,
    local_tz: Optional[datetime.tzinfo] = None
) -> datetime.datetime:
    """
    Floor datetime to the start of its time slot.

    Args:
        dt: Datetime to align
        slot_minutes: Slot duration in minutes (e.g., 30 or 60)
        local_tz: Optional local timezone

    Returns:
        Datetime floored to slot boundary with seconds/microseconds zeroed
    """
    dt = ensure_local_tz(dt, local_tz)
    minutes = dt.hour * 60 + dt.minute
    slot_start = (minutes // slot_minutes) * slot_minutes
    return dt.replace(
        hour=int(slot_start // 60),
        minute=int(slot_start % 60),
        second=0,
        microsecond=0
    )


def slot_offset(
    dt: datetime.datetime,
    slot_minutes: int,
    slots: int,
    local_tz: Optional[datetime.tzinfo] = None
) -> datetime.datetime:
    """
    Return the slot boundary ``slots`` slots away from the one containing ``dt``.

    Wall-clock arithmetic on an aware datetime is wrong across a DST
    transition: ``dt - timedelta(minutes=15)`` around the Europe/Riga autumn
    fold is a 1h15m step (and a -45min step in spring), so the caller's slot
    lookup misses the slot it meant.  Elapsed time is therefore added as a UTC
    instant, the same idiom ``next_slot_time`` has always used.

    Args:
        dt: Any datetime inside the reference slot
        slot_minutes: Slot duration in minutes
        slots: Number of slots to move (negative moves backwards)
        local_tz: Optional local timezone

    Returns:
        Slot-aligned datetime with seconds/microseconds zeroed
    """
    dt = ensure_local_tz(dt, local_tz)
    current_slot = align_to_slot(dt, slot_minutes)
    delta = datetime.timedelta(minutes=slot_minutes * slots)
    if is_aware(current_slot):
        # Move in UTC so a spring gap is skipped and both autumn fold slots
        # remain reachable.
        shifted = (current_slot.astimezone(UTC) + delta).astimezone(
            current_slot.tzinfo
        )
    else:
        shifted = current_slot + delta
    return shifted.replace(second=0, microsecond=0)


def prev_slot_time(
    now: datetime.datetime,
    slot_minutes: int,
    local_tz: Optional[datetime.tzinfo] = None
) -> datetime.datetime:
    """
    Get the start of the slot immediately BEFORE the one containing ``now``.

    DST-safe counterpart of :func:`next_slot_time` (see :func:`slot_offset`).

    Args:
        now: Current datetime
        slot_minutes: Slot duration in minutes
        local_tz: Optional local timezone

    Returns:
        Datetime of the previous slot boundary
    """
    return slot_offset(now, slot_minutes, -1, local_tz)


def next_slot_time(
    now: datetime.datetime,
    slot_minutes: int,
    local_tz: Optional[datetime.tzinfo] = None
) -> datetime.datetime:
    """
    Get the next slot boundary time.

    Args:
        now: Current datetime
        slot_minutes: Slot duration in minutes
        local_tz: Optional local timezone

    Returns:
        Datetime of next slot boundary (with 5 seconds offset for safety)
    """
    return slot_offset(now, slot_minutes, 1, local_tz).replace(
        second=5, microsecond=0
    )


def next_interval_time(
    now: datetime.datetime,
    interval_minutes: int,
    local_tz: Optional[datetime.tzinfo] = None
) -> datetime.datetime:
    """
    Get the next boundary time for a given interval.

    Args:
        now: Current datetime
        interval_minutes: Interval duration in minutes
        local_tz: Optional local timezone

    Returns:
        Datetime of next interval boundary (with 5 seconds offset for safety)
    """
    return next_slot_time(now, max(1, int(interval_minutes)), local_tz)


def lookup_by_time(
    data: Dict[datetime.datetime, T],
    slot_time: datetime.datetime,
    local_tz: Optional[datetime.tzinfo] = None
) -> Optional[T]:
    """
    Look up a value in a datetime-keyed dict, handling timezone mismatches.

    First tries direct lookup, then falls back to matching by date/hour/minute
    after timezone normalization.

    Args:
        data: Dictionary keyed by datetime
        slot_time: The slot time to look up
        local_tz: Optional local timezone for normalization

    Returns:
        The value if found, None otherwise
    """
    if not data:
        return None

    # Canonical fixed-offset schedule maps have a safe O(1) lookup even during
    # an autumn fold. Legacy ZoneInfo-keyed maps fall through to instant scan.
    canonical_key = canonical_slot_key(slot_time)
    if canonical_key in data:
        return data[canonical_key]
    if not is_aware(slot_time) and slot_time in data:
        return data[slot_time]

    # Fallback: match by local time components
    for sched_time, value in data.items():
        if datetimes_match_slot(sched_time, slot_time, local_tz):
            return value

    return None


