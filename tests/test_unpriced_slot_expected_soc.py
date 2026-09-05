"""The unpriced current slot exists in the plan, so every consumer can see it.

Defect (pre-fix)
----------------

The pre-solve step advances SOC across the current interval when nobody
published a price for it -- including the PV a HOLD still absorbs -- and hands
the DP the SOC that walk ends at.  But when there was nothing to retain, the
fallback left that interval OUT of ``schedule`` entirely.  Both calling paths
then rebuilt ``expected_soc_schedule`` from the MEASURED SOC over a schedule
that skipped the interval, so the two disagreed about the same quarter hour.

The maintainer's reproduction, and the fixture below: 10:07, SOC 40 %, 3 kW of
PV, eight minutes of the slot left.  The DP starts 10:15 at 42.3776 % and
``expected_soc_schedule`` says 40 %.

Design
------

ONE source of truth: ``schedule`` says what runs, and every consumer walks it.
The fallback interval is a real ``HOLD`` entry with reason ``no_price`` and no
price provenance -- exactly what ``execute_scheduled_mode`` was constructing on
the fly anyway.  The alternative (threading an advanced SOC, temperature and
time anchor through ``calculate_expected_soc_schedule``,
``project_schedule_trajectory``, the cost projection and the deviation
detector's anchor) is four more places that can disagree.

An entry existing must not silence the recovery it stands in for, so the tests
below also pin the ``no_price`` retry arming, the ``current_slot_entry:
fallback`` diagnostics and the horizon extension that rebuilds over a stale
fallback once prices come back.
"""

from __future__ import annotations

import datetime

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import BatteryMode, ScheduleEntry
from battery_optimizer_lib.models import PRICE_SOURCE_MARKET

from tests.test_current_slot_price import (
    PlanningOptimizer,
    attach_service_path,
    maintainer_scenario,
)
from tests.test_price_recovery import TZ, day_start, make_prices, slots_between


SLOT = 15
CAPACITY = 14.3
CHARGE_KW = 4.5
EFFICIENCY = 0.85
START_SOC = 40.0
# 10:07 leaves eight minutes of the 10:00-10:15 interval.
REMAINING_H = 8.0 / 60.0

NOW = datetime.datetime(2024, 1, 15, 10, 7, tzinfo=TZ)
SLOT_10_00 = datetime.datetime(2024, 1, 15, 10, 0, tzinfo=TZ)
SLOT_10_15 = datetime.datetime(2024, 1, 15, 10, 15, tzinfo=TZ)


def _soc_gain(kw):
    return kw * REMAINING_H * EFFICIENCY / CAPACITY * 100.0


HOLD_FALLBACK_SOC = START_SOC + _soc_gain(3.0)      # 42.3776 %
RETAINED_CHARGE_SOC = START_SOC + _soc_gain(CHARGE_KW)


class UnpricedSlotOptimizer(PlanningOptimizer):
    """``PlanningOptimizer`` with the published SOC trajectory un-stubbed.

    ``RecoveryOptimizer`` returns an empty ``calculate_expected_soc_schedule``;
    here it is the thing under test.
    """

    calculate_expected_soc_schedule = (
        bo.BatteryOptimizer.calculate_expected_soc_schedule
    )

    def __init__(self, now, *, pv_kw=0.0, load_kw=0.0, **kwargs):
        kwargs.setdefault("battery_capacity", CAPACITY)
        kwargs.setdefault("charge_rate", CHARGE_KW)
        kwargs.setdefault("discharge_rate", CHARGE_KW)
        kwargs.setdefault("efficiency", EFFICIENCY)
        kwargs.setdefault("inverter_efficiency", 1.0)
        super().__init__(now, **kwargs)
        self._pv_kw = pv_kw
        self._load_kw = load_kw

    def _predict_pv_kw(self, dt):
        # Only the interval the app is living in; the DP never sees this slot,
        # so PV anywhere else would only muddy the plan it does see.
        return self._pv_kw if dt == SLOT_10_00 else 0.0

    def _predict_pv_kw_raw(self, dt):
        return self._predict_pv_kw(dt)

    def _predict_load_kw(self, dt):
        return self._load_kw


def _app(*, pv_kw=0.0, load_kw=0.0, previous_mode=None, previous_priced=True):
    app = UnpricedSlotOptimizer(NOW, soc=START_SOC, pv_kw=pv_kw, load_kw=load_kw)
    _, today_points, _ = maintainer_scenario(NOW)
    attach_service_path(app, {NOW.date(): today_points})
    if previous_mode is not None:
        app.schedule = {
            bo.canonical_slot_key(SLOT_10_00): ScheduleEntry(
                time=SLOT_10_00,
                mode=previous_mode,
                reason="prior plan",
                export_rate=0 if previous_mode == BatteryMode.DISCHARGE else None,
                price_source=PRICE_SOURCE_MARKET if previous_priced else None,
            )
        }
    return app


def _assert_the_two_trajectories_agree(app, expected_at_10_15):
    """The DP solved from this SOC; the published trajectory must reach it."""
    assert app._last_dp_soc_trajectory[SLOT_10_15][0] == pytest.approx(
        expected_at_10_15, abs=1e-6
    )
    assert app.expected_soc_schedule[SLOT_10_15] == pytest.approx(
        expected_at_10_15, abs=1e-6
    )
    # And the interval itself is described, starting from what was measured.
    assert app.expected_soc_schedule[SLOT_10_00] == pytest.approx(
        START_SOC, abs=1e-9
    )


# ===========================================================================
# The reproduction, on both calling paths
# ===========================================================================

class TestTheHoldFallbackIsInTheExpectedSocProjection:
    def test_full_optimize(self):
        app = _app(pv_kw=3.0)

        app.full_optimize(None)

        entry = bo.lookup_by_time(app.schedule, SLOT_10_00, app._get_local_timezone())
        assert entry is not None, "the interval that is RUNNING must be in the plan"
        assert entry.mode == BatteryMode.HOLD
        assert entry.reason == "no_price"
        assert entry.price_source is None, "nothing can vouch for this interval"
        _assert_the_two_trajectories_agree(app, HOLD_FALLBACK_SOC)

    def test_recalculate_remaining_schedule(self):
        app = _app(pv_kw=3.0)

        app._recalculate_remaining_schedule(START_SOC)

        entry = bo.lookup_by_time(app.schedule, SLOT_10_00, app._get_local_timezone())
        assert entry is not None
        assert entry.mode == BatteryMode.HOLD
        assert entry.reason == "no_price"
        _assert_the_two_trajectories_agree(app, HOLD_FALLBACK_SOC)


class TestARetainedEntryProjectsTheSameWay:
    """The retained variants: already consistent, and they must stay that way."""

    def test_full_optimize_retained_charge(self):
        app = _app(previous_mode=BatteryMode.CHARGE)

        app.full_optimize(None)

        assert app.schedule[SLOT_10_00].mode == BatteryMode.CHARGE
        _assert_the_two_trajectories_agree(app, RETAINED_CHARGE_SOC)

    def test_recalculate_retained_charge(self):
        app = _app(previous_mode=BatteryMode.CHARGE)

        app._recalculate_remaining_schedule(START_SOC)

        assert app.schedule[SLOT_10_00].mode == BatteryMode.CHARGE
        _assert_the_two_trajectories_agree(app, RETAINED_CHARGE_SOC)

    def test_full_optimize_retained_discharge(self):
        app = _app(load_kw=2.0, previous_mode=BatteryMode.DISCHARGE)

        app.full_optimize(None)

        assert app.schedule[SLOT_10_00].mode == BatteryMode.DISCHARGE
        drained = 2.0 * REMAINING_H / CAPACITY * 100.0
        _assert_the_two_trajectories_agree(app, START_SOC - drained)

    def test_recalculate_retained_discharge(self):
        app = _app(load_kw=2.0, previous_mode=BatteryMode.DISCHARGE)

        app._recalculate_remaining_schedule(START_SOC)

        assert app.schedule[SLOT_10_00].mode == BatteryMode.DISCHARGE
        drained = 2.0 * REMAINING_H / CAPACITY * 100.0
        _assert_the_two_trajectories_agree(app, START_SOC - drained)


class TestAnUnretainableEntryFallsBack:
    def test_a_previous_entry_without_provenance_is_not_retained(self):
        app = _app(
            pv_kw=3.0,
            previous_mode=BatteryMode.CHARGE,
            previous_priced=False,
        )

        app.full_optimize(None)

        entry = app.schedule[SLOT_10_00]
        assert entry.mode == BatteryMode.HOLD
        assert entry.reason == "no_price"
        _assert_the_two_trajectories_agree(app, HOLD_FALLBACK_SOC)


# ===========================================================================
# An entry that exists must not silence what it stands in for
# ===========================================================================

class TestTheFallbackEntryStillBehavesLikeAFallback:
    def test_execution_applies_hold_no_price_and_keeps_the_retry_armed(self):
        app = _app(pv_kw=3.0)
        app.full_optimize(None)
        app.applied.clear()

        app.execute_scheduled_mode(None)

        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "no_price"
        assert app._price_retry_pending(), (
            "an entry standing in for a missing price must not stop it being "
            "chased"
        )

    def test_the_diagnostics_still_call_it_a_fallback(self):
        app = _app(pv_kw=3.0)

        app.full_optimize(None)

        diag = app._price_horizon_diagnostics()
        assert diag["current_slot_priced"] is False
        assert diag["current_slot_entry"] == "fallback"

    def test_the_horizon_extension_rebuilds_over_a_stale_fallback(self):
        """A fallback HOLD is not a plan: once prices arrive it must be replaced."""
        app = _app(pv_kw=3.0)
        app.full_optimize(None)
        assert app.schedule[SLOT_10_00].reason == "no_price"

        # The interval is published now: a complete day, this one included.
        today = day_start(NOW)
        tomorrow = day_start(NOW + datetime.timedelta(days=1))
        attach_service_path(
            app,
            {NOW.date(): make_prices(today, slots_between(today, tomorrow))},
        )
        app.get_prices()  # refresh the snapshot the periodic pass reads

        rebuilt = app._check_price_horizon_health(START_SOC)

        assert rebuilt is True
        assert app.schedule[SLOT_10_00].reason != "no_price"
        assert app.schedule[SLOT_10_00].price_source == PRICE_SOURCE_MARKET
