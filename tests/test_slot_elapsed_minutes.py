"""Regression: elapsed-minutes-into-slot must survive naive/aware datetime mixing.

Production incident (2026-07-28 22:12, live AppDaemon):

    File "/config/apps/battery_optimizer.py", line 1014, in full_optimize
      minutes_into_slot = max(0.0, (now - current_slot).total_seconds() / 60.0)
    TypeError: can't subtract offset-naive and offset-aware datetimes

`BatteryOptimizer.full_optimize` and `_recalculate_remaining_schedule` compute
the partial first slot (the DEFECT 1 fix) as `now - slot_start`, where

  * `now = self.datetime()`      -> AppDaemon returns NAIVE local time
  * `slot_start = _align_to_slot(now)` -> align_to_slot() runs the value through
    ensure_local_tz(), so it comes back TZ-AWARE

Subtracting those raises, which killed full_optimize on every run and left the
inverter uncontrolled. The unit suite missed it because the orchestrator is not
unit-tested (see CLAUDE.md) — so this file pins the arithmetic contract the
orchestrator depends on instead.
"""

import datetime

try:  # pragma: no cover - zoneinfo is stdlib on the target runtime
    from zoneinfo import ZoneInfo
    RIGA = ZoneInfo("Europe/Riga")
except Exception:  # pragma: no cover
    RIGA = datetime.timezone(datetime.timedelta(hours=3))

from battery_optimizer_lib.timezone_utils import align_to_slot, normalize_tz_pair


SLOT = 15


def _elapsed_minutes(now, slot_start):
    """Exactly what the orchestrator does after the fix."""
    now_cmp, slot_cmp = normalize_tz_pair(now, slot_start)
    return max(0.0, (now_cmp - slot_cmp).total_seconds() / 60.0)


def test_align_to_slot_makes_a_naive_input_aware():
    """The trap itself: the two operands do NOT have matching awareness."""
    naive_now = datetime.datetime(2026, 7, 28, 22, 12, 36)

    slot_start = align_to_slot(naive_now, SLOT, RIGA)

    assert naive_now.tzinfo is None
    assert slot_start.tzinfo is not None, "align_to_slot attaches a timezone"


def test_raw_subtraction_is_the_bug_we_are_guarding_against():
    naive_now = datetime.datetime(2026, 7, 28, 22, 12, 36)
    slot_start = align_to_slot(naive_now, SLOT, RIGA)

    try:
        (naive_now - slot_start).total_seconds()
    except TypeError:
        pass  # expected — this is what crashed in production
    else:  # pragma: no cover
        raise AssertionError(
            "raw subtraction no longer raises; the guard below may be obsolete"
        )


def test_elapsed_minutes_from_naive_now_and_aware_slot():
    """The incident timestamp: 22:12:36 is 12.6 min into the 22:00 slot."""
    naive_now = datetime.datetime(2026, 7, 28, 22, 12, 36)
    slot_start = align_to_slot(naive_now, SLOT, RIGA)

    assert _elapsed_minutes(naive_now, slot_start) == 12.6


def test_elapsed_minutes_matches_the_11_59_recalculation_from_the_log():
    """11:59:18 leaves ~42 s of the 11:45 slot — the case DEFECT 1 was about."""
    naive_now = datetime.datetime(2026, 7, 27, 11, 59, 18)
    slot_start = align_to_slot(naive_now, SLOT, RIGA)

    elapsed = _elapsed_minutes(naive_now, slot_start)
    assert elapsed == 14.3
    remaining_fraction = max(0.0, 1.0 - elapsed / SLOT)
    assert round(remaining_fraction, 4) == 0.0467  # ~42 s of a 15 min slot


def test_elapsed_minutes_works_when_both_are_aware():
    aware_now = datetime.datetime(2026, 7, 28, 22, 12, 36, tzinfo=RIGA)
    slot_start = align_to_slot(aware_now, SLOT, RIGA)

    assert _elapsed_minutes(aware_now, slot_start) == 12.6


def test_elapsed_minutes_works_when_both_are_naive():
    naive_now = datetime.datetime(2026, 7, 28, 22, 12, 36)
    naive_slot = datetime.datetime(2026, 7, 28, 22, 0, 0)

    assert _elapsed_minutes(naive_now, naive_slot) == 12.6


def test_elapsed_minutes_never_goes_negative():
    """A slot boundary in the future must clamp to 0, not produce a negative."""
    naive_now = datetime.datetime(2026, 7, 28, 22, 0, 0)
    future_slot = datetime.datetime(2026, 7, 28, 22, 15, 0, tzinfo=RIGA)

    assert _elapsed_minutes(naive_now, future_slot) == 0.0


def test_elapsed_minutes_at_an_exact_slot_boundary_is_zero():
    naive_now = datetime.datetime(2026, 7, 28, 22, 15, 0)
    slot_start = align_to_slot(naive_now, SLOT, RIGA)

    assert _elapsed_minutes(naive_now, slot_start) == 0.0
