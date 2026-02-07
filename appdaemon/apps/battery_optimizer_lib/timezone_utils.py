"""
Timezone utilities for the Battery Optimizer.

Provides functions for handling timezone-aware and naive datetime comparisons,
slot alignment, and dictionary lookups with timezone normalization.
"""

import datetime
from typing import Dict, Optional, Tuple, TypeVar, Any

T = TypeVar("T")


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
    now = ensure_local_tz(now, local_tz)
    minutes = now.hour * 60 + now.minute
    next_slot = ((minutes // slot_minutes) + 1) * slot_minutes

    if next_slot >= 1440:
        next_slot = 0
        now = now + datetime.timedelta(days=1)

    return now.replace(
        hour=int(next_slot // 60),
        minute=int(next_slot % 60),
        second=5,
        microsecond=0
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
    interval_minutes = max(1, int(interval_minutes))
    now = ensure_local_tz(now, local_tz)
    minutes = now.hour * 60 + now.minute
    next_boundary = ((minutes // interval_minutes) + 1) * interval_minutes

    if next_boundary >= 1440:
        next_boundary = 0
        now = now + datetime.timedelta(days=1)

    return now.replace(
        hour=int(next_boundary // 60),
        minute=int(next_boundary % 60),
        second=5,
        microsecond=0
    )


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

    # Direct lookup first (fast path)
    if slot_time in data:
        return data[slot_time]

    # Fallback: match by local time components
    for sched_time, value in data.items():
        if datetimes_match_slot(sched_time, slot_time, local_tz):
            return value

    return None


