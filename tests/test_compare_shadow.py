"""scripts/compare_shadow.py: the S5 acceptance check, on synthetic plans."""

from __future__ import annotations

import datetime
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "compare_shadow", REPO / "scripts" / "compare_shadow.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

UTC = datetime.timezone.utc
T0 = datetime.datetime(2026, 10, 5, 11, 15, tzinfo=UTC)
WIT = {"HOLD": "hold", "CHARGE": "grid_charge", "DISCHARGE": "discharge_to_load"}


def plan(modes, start=T0, offset_hours=3):
    tz = datetime.timezone(datetime.timedelta(hours=offset_hours))
    return [
        {"time": (start + i * cmp.SLOT).astimezone(tz).isoformat(),
         "mode": mode, "wit_mode": WIT[mode]}
        for i, mode in enumerate(modes)
    ]


def test_identical_plans_pass():
    modes = ["HOLD"] * 4 + ["CHARGE"] * 4 + ["DISCHARGE"] * 4
    result = cmp.compare(plan(modes), plan(modes), T0 + datetime.timedelta(minutes=5))
    assert result["ok"] and len(result["near"]) == 5


def test_near_difference_fails_even_if_totals_match():
    a = ["HOLD", "CHARGE"] + ["HOLD"] * 10
    b = ["CHARGE", "HOLD"] + ["HOLD"] * 10
    result = cmp.compare(plan(a), plan(b), T0)
    assert result["totals_ok"] and not result["near_ok"] and not result["ok"]


def test_one_slot_total_difference_is_tolerated_two_is_not():
    base = ["HOLD"] * 6 + ["CHARGE"] * 4 + ["HOLD"] * 4
    one = ["HOLD"] * 6 + ["CHARGE"] * 5 + ["HOLD"] * 3
    two = ["HOLD"] * 6 + ["CHARGE"] * 6 + ["HOLD"] * 2
    assert cmp.compare(plan(base), plan(one), T0)["ok"]
    assert not cmp.compare(plan(base), plan(two), T0)["totals_ok"]


def test_slots_are_matched_by_instant_not_by_wall_clock_text():
    """The same instants written with different offsets still line up."""
    modes = ["HOLD"] * 6
    result = cmp.compare(plan(modes, offset_hours=3), plan(modes, offset_hours=0), T0)
    assert len(result["common"]) == 6 and result["ok"]


def test_past_slots_are_not_the_current_one():
    live = ["CHARGE"] * 2 + ["HOLD"] * 8
    shadow = ["HOLD"] * 10
    result = cmp.compare(plan(live), plan(shadow), T0 + datetime.timedelta(minutes=31))
    assert result["near"][0]["time"] == T0 + 2 * cmp.SLOT
    assert result["near_ok"]


def test_too_short_a_common_horizon_fails():
    result = cmp.compare(plan(["HOLD"] * 3), plan(["HOLD"] * 3), T0)
    assert not result["near_ok"]
