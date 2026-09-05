"""A LEADING run of unpriced intervals is modelled, not skipped.

Defect (pre-fix)
----------------

``modeled_horizon`` was handed the priced points and started at the FIRST of
them.  ``_resolve_unpriced_current_slot`` owns exactly ONE interval - the one
running now.  So when the current interval and one or more intervals after it
are unpublished before the next published one, every slot in between belonged
to neither mechanism: it was not modelled by the DP, not replayed by
``_validate_final_plan``, and not present in the schedule at all.

The maintainer's reproduction, and the fixture below: the reply holds only
10:00-10:15 at 0.50 and 10:45-11:00 at 1.00; it is 10:17; a 10 kWh pack sits at
its 10 % minimum with unit efficiencies; 4 kW of forecast PV at 10:30 and 4 kW
of forecast load at 10:45.

Pre-fix the plan was 10:15 HOLD/``no_price``, NO ENTRY AT ALL for 10:30, and a
HOLD at 10:45 that imported a full kWh at 1.00 EUR/kWh - the kWh the 10:30 sun
would have stored for free.  The replay covered two slots and agreed with a
trajectory that was ten SOC points low from 10:45 onwards.  Moving the same
physics one slot earlier, so that the current interval is the priced one and
the identical hole falls INSIDE the horizon, produced the correct plan: this is
a defect of where the modelled sequence STARTS, nothing else.

Policy under test
-----------------

The modelled sequence runs from

* the slot AFTER the current one, when the current interval is unpriced and
  therefore owned by ``_resolve_unpriced_current_slot``; else
* the current slot itself,

up to the last priced slot, with every unpublished interval in between entering
the DP as ``price=None`` - a forced HOLD that still carries PV, load, SOC and
temperature (``docs/scheduling-algorithm.md``, "A missing interval INSIDE the
horizon").  The lower bound is passed explicitly rather than inferred from the
first priced point.
"""

from __future__ import annotations

import datetime

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import BatteryMode, PricePoint, ScheduleEntry
from battery_optimizer_lib.models import PRICE_SOURCE_MARKET

from tests.test_current_slot_price import (
    attach_sensor_path,
    attach_service_path,
    current_entry,
)
from tests.test_unpriced_gap_slots import GapOptimizer
from tests.test_price_recovery import TZ, UTC


SLOT = 15

DAY = datetime.date(2024, 1, 15)
SLOT_10_00 = datetime.datetime(2024, 1, 15, 10, 0, tzinfo=TZ)
SLOT_10_15 = datetime.datetime(2024, 1, 15, 10, 15, tzinfo=TZ)
SLOT_10_30 = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ)
SLOT_10_45 = datetime.datetime(2024, 1, 15, 10, 45, tzinfo=TZ)
SLOT_11_00 = datetime.datetime(2024, 1, 15, 11, 0, tzinfo=TZ)

# 10:17 - inside the interval the reply does NOT contain, with one further
# unpublished interval between it and the next published one.
NOW = datetime.datetime(2024, 1, 15, 10, 17, tzinfo=TZ)
# 10:02 - the same physics observed one slot earlier, where the identical hole
# falls inside the horizon instead of leading it.
NOW_ONE_SLOT_EARLIER = datetime.datetime(2024, 1, 15, 10, 2, tzinfo=TZ)

MIN_SOC = 10.0
CHEAP = 0.50
DEAR = 1.00
PV_KW = 4.0
LOAD_KW = 4.0


def _sparse_points():
    """The maintainer's two records: 10:00-10:15 and 10:45-11:00."""
    return [
        PricePoint(time=SLOT_10_00, price=CHEAP, end=SLOT_10_15),
        PricePoint(time=SLOT_10_45, price=DEAR, end=SLOT_11_00),
    ]


def _attach(app, path, points):
    """Wire the REAL price service to whichever fetch path the test wants."""
    if path == "service":
        return attach_service_path(app, {DAY: points})
    return attach_sensor_path(app, points)


def _repro_app(path, now=NOW, points=None):
    """10:00 priced, 10:15 (current) and 10:30 missing, 10:45 priced."""
    app = GapOptimizer(
        now,
        soc=MIN_SOC,
        pv_by_slot={SLOT_10_30: PV_KW},
        load_by_slot={SLOT_10_45: LOAD_KW},
    )
    _attach(app, path, _sparse_points() if points is None else points)
    return app


def _run(app, planning_path):
    if planning_path == "full_optimize":
        app.full_optimize(None)
    else:
        app._recalculate_remaining_schedule(app._soc)


def _entry(app, slot):
    return current_entry(app, slot)


def _row(app, slot):
    """The FINAL validated replay's row for *slot*, or None."""
    for key, row in app._last_plan_replay.by_slot.items():
        if key.astimezone(UTC) == slot.astimezone(UTC):
            return row
    return None


def _soc(app, slot):
    row = _row(app, slot)
    return None if row is None else row.soc_end


# ===========================================================================
# The reproduction, on both fetch paths and both planning paths
# ===========================================================================

@pytest.mark.parametrize("fetch_path", ["service", "sensor"])
@pytest.mark.parametrize("planning_path", ["full_optimize", "recalculate"])
class TestTheLeadingRunIsPlannedThrough:
    def test_the_intermediate_slot_is_in_the_plan(self, fetch_path, planning_path):
        app = _repro_app(fetch_path)

        _run(app, planning_path)

        current = _entry(app, SLOT_10_15)
        assert current is not None
        assert current.mode == BatteryMode.HOLD
        assert current.reason == "no_price"
        assert current.price_source is None

        gap = _entry(app, SLOT_10_30)
        assert gap is not None, (
            "the interval between the unpriced current slot and the next "
            "published one belongs to the DP, not to nobody"
        )
        assert gap.mode == BatteryMode.HOLD
        assert gap.reason == "no_price"
        assert gap.price_source is None
        assert gap.marginal_value_eur_kwh is None

    def test_the_free_pv_is_stored_and_serves_the_load(self, fetch_path, planning_path):
        app = _repro_app(fetch_path)

        _run(app, planning_path)

        assert _entry(app, SLOT_10_45).mode == BatteryMode.DISCHARGE, (
            "the kWh the 10:30 sun stored covers the 10:45 load"
        )
        assert app._last_plan_replay.total_grid_import_ac_kwh == pytest.approx(
            0.0, abs=1e-9
        ), "importing at 1.00 EUR/kWh buys energy the plan already has"
        assert app._last_plan_replay.total_grid_charge_ac_kwh == pytest.approx(
            0.0, abs=1e-9
        )

    def test_the_replay_covers_every_slot_of_the_run(self, fetch_path, planning_path):
        app = _repro_app(fetch_path)

        _run(app, planning_path)

        replayed = {k.astimezone(UTC) for k in app._last_plan_replay.by_slot}
        assert replayed == {
            SLOT_10_15.astimezone(UTC),
            SLOT_10_30.astimezone(UTC),
            SLOT_10_45.astimezone(UTC),
        }, "a slot the plan is not replayed over is one the validator cannot contradict"
        assert app._last_plan_replay.ok

    def test_the_trajectory_walks_the_whole_run(self, fetch_path, planning_path):
        app = _repro_app(fetch_path)

        _run(app, planning_path)

        # 10 % of a 10 kWh pack, held through the rest of 10:15; 4 kW x 0.25 h
        # of PV at 10:30 stores 1.0 kWh (+10 points); the 10:45 load takes it
        # back out.
        assert _soc(app, SLOT_10_15) == pytest.approx(MIN_SOC)
        assert _soc(app, SLOT_10_30) == pytest.approx(20.0)
        assert _soc(app, SLOT_10_45) == pytest.approx(MIN_SOC)

        trajectory = {
            k.astimezone(UTC): v for k, v in app._last_dp_soc_trajectory.items()
        }
        assert SLOT_10_30.astimezone(UTC) in trajectory
        assert trajectory[SLOT_10_30.astimezone(UTC)][1] == pytest.approx(20.0)

    def test_the_run_is_counted_and_costed(self, fetch_path, planning_path):
        app = _repro_app(fetch_path)

        _run(app, planning_path)

        assert app._last_schedule_counts.hold == 2
        assert app._last_schedule_counts.discharge == 1
        costed = {k.astimezone(UTC) for k in app._last_projected_costs}
        assert SLOT_10_30.astimezone(UTC) in costed


# ===========================================================================
# The same run behind a RETAINED real-priced current entry
# ===========================================================================

class TestTwoUnpricedSlotsAfterARetainedEntry:
    """10:15 retained on a real price, 10:30 and 10:45 unpublished, 11:00 priced."""

    def _points(self):
        return [
            PricePoint(time=SLOT_10_00, price=CHEAP, end=SLOT_10_15),
            PricePoint(
                time=SLOT_11_00,
                price=DEAR,
                end=SLOT_11_00 + datetime.timedelta(minutes=SLOT),
            ),
        ]

    def _app(self, fetch_path="service"):
        app = GapOptimizer(
            NOW,
            soc=MIN_SOC,
            pv_by_slot={SLOT_10_30: PV_KW, SLOT_10_45: PV_KW},
            load_by_slot={SLOT_11_00: LOAD_KW},
        )
        app.schedule = {
            bo.canonical_slot_key(SLOT_10_15): ScheduleEntry(
                time=SLOT_10_15,
                mode=BatteryMode.HOLD,
                reason="0.5000 EUR/kWh load~0.00kW",
                price_source=PRICE_SOURCE_MARKET,
            )
        }
        _attach(app, fetch_path, self._points())
        return app

    def test_both_unpriced_slots_are_modelled(self):
        app = self._app()

        app.full_optimize(None)

        retained = _entry(app, SLOT_10_15)
        assert retained is not None
        assert retained.price_source == PRICE_SOURCE_MARKET, (
            "the entry planned on a real price is retained, not replaced"
        )
        for slot in (SLOT_10_30, SLOT_10_45):
            entry = _entry(app, slot)
            assert entry is not None, f"{slot:%H:%M} is missing from the plan"
            assert entry.mode == BatteryMode.HOLD
            assert entry.reason == "no_price"
            assert entry.price_source is None

    def test_the_pv_of_both_slots_reaches_the_priced_load(self):
        app = self._app()

        app.full_optimize(None)

        assert _soc(app, SLOT_10_15) == pytest.approx(MIN_SOC)
        assert _soc(app, SLOT_10_30) == pytest.approx(20.0)
        assert _soc(app, SLOT_10_45) == pytest.approx(30.0)
        assert _entry(app, SLOT_11_00).mode == BatteryMode.DISCHARGE
        # 11:00 discharges at the full 4.5 kW rate - 1.125 kWh, 11.25 points of
        # a 10 kWh pack - and the pack only HAS that charge because both
        # unpriced slots were modelled.
        assert _soc(app, SLOT_11_00) == pytest.approx(18.75)
        assert app._last_plan_replay.total_grid_import_ac_kwh == pytest.approx(
            0.0, abs=1e-9
        )

    def test_the_replay_covers_the_retained_slot_and_the_run(self):
        app = self._app()

        app.full_optimize(None)

        replayed = {k.astimezone(UTC) for k in app._last_plan_replay.by_slot}
        assert replayed == {
            SLOT_10_15.astimezone(UTC),
            SLOT_10_30.astimezone(UTC),
            SLOT_10_45.astimezone(UTC),
            SLOT_11_00.astimezone(UTC),
        }


# ===========================================================================
# Metamorphic: where the horizon STARTS must not change the physics
# ===========================================================================

class TestLeadingAndInsideGapsAgree:
    """The same prices, PV and load, observed one slot apart.

    At 10:17 the hole (10:15, 10:30) LEADS the modelled sequence, because the
    current interval is unpriced. At 10:02 the current interval is the
    published 10:00 one, so the identical hole falls INSIDE the horizon. The
    slots both plans share must get the same action and the same energies:
    otherwise the plan depends on when it was made rather than on what the
    physics and the prices are.
    """

    @staticmethod
    def _shared(app):
        return {
            slot: (
                _entry(app, slot).mode,
                round(_row(app, slot).grid_import_ac_kwh, 9),
                round(_row(app, slot).soc_end, 9),
            )
            for slot in (SLOT_10_30, SLOT_10_45)
        }

    def test_the_shared_slots_get_the_same_plan(self):
        leading = _repro_app("service", now=NOW)
        leading.full_optimize(None)

        inside = _repro_app("service", now=NOW_ONE_SLOT_EARLIER)
        inside.full_optimize(None)

        assert _entry(inside, SLOT_10_15) is not None, (
            "at 10:02 the hole really is INSIDE the modelled horizon"
        )
        assert _entry(inside, SLOT_10_15).reason == "no_price"
        assert self._shared(leading) == self._shared(inside)
        assert leading._last_plan_replay.total_grid_import_ac_kwh == pytest.approx(
            inside._last_plan_replay.total_grid_import_ac_kwh, abs=1e-9
        )


# ===========================================================================
# The horizon still ends at the last PRICED interval
# ===========================================================================

class TestNothingIsModelledPastTheLastPricedSlot:
    def test_a_leading_run_does_not_extend_the_end(self):
        app = _repro_app("service")

        app.full_optimize(None)

        modelled = sorted(k.astimezone(UTC) for k in app.schedule)
        assert modelled == [
            SLOT_10_15.astimezone(UTC),
            SLOT_10_30.astimezone(UTC),
            SLOT_10_45.astimezone(UTC),
        ], "nothing is planned past the last interval a source published"
