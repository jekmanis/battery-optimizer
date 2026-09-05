"""Restart continuity is decided BEFORE the plan is validated, not after it.

Defect (pre-fix)
----------------

``full_optimize`` called ``_preserve_mode_on_restart`` *after*
``find_optimal_schedule`` had validated, replayed, counted, costed and logged
its result.  That method replaced the current slot's ``HOLD`` with the
``CHARGE``/``DISCHARGE`` the previous run had been executing, and let it inherit
the slot's published-price provenance -- so the rewritten entry executed.

The maintainer's reproduction, and the fixture below: a validated
``HOLD, DISCHARGE`` plan reserves the pack for a 1.00 EUR/kWh slot and imports
during the 0.10 one.  Restart preservation turns it into ``DISCHARGE,
DISCHARGE``: the import moves to the expensive slot and the plan credits the
battery with service it no longer has, while ``_last_dp_soc_trajectory``, the
mode census, the projected-cost column, the decision log and
``_last_plan_replay`` all still describe the plan that was replaced.

Policy under test
-----------------

Restart continuity is a constraint on the solve, not an edit of its answer.
When the previous run was CHARGE or DISCHARGE for the interval the app wakes up
in, that action is FIXED for the remainder of the slot, the SOC and temperature
are advanced across it through the shared slot model, and the DP solves the
rest of the horizon from there -- exactly as the unpriced current slot is
handled.  Nothing changes an action after ``find_optimal_schedule`` has
validated one.

Everything here is deterministic: the settable clock and AppDaemon double from
``test_price_recovery``, the REAL planner and the REAL final-plan validation.
"""

from __future__ import annotations

import datetime

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import BatteryMode, PricePoint
from battery_optimizer_lib.models import count_schedule_modes

from tests.test_current_slot_price import PlanningOptimizer
from tests.test_price_recovery import TZ


# ---------------------------------------------------------------------------
# A double that runs the real planner AND the real expected-SOC projection
# ---------------------------------------------------------------------------

class RestartOptimizer(PlanningOptimizer):
    """``PlanningOptimizer`` with the published SOC trajectory un-stubbed.

    ``RecoveryOptimizer`` returns an empty ``calculate_expected_soc_schedule``
    because Task 5 never looked at it.  Here it is half the evidence: the
    trajectory the app publishes has to describe the plan the app executes.
    """

    calculate_expected_soc_schedule = (
        bo.BatteryOptimizer.calculate_expected_soc_schedule
    )

    def __init__(self, now, prices=None, load_kw=1.0, **kwargs):
        kwargs.setdefault("battery_capacity", 10.0)
        kwargs.setdefault("charge_rate", 4.0)
        kwargs.setdefault("discharge_rate", 4.0)
        kwargs.setdefault("efficiency", 1.0)
        kwargs.setdefault("inverter_efficiency", 1.0)
        kwargs.setdefault("grid_fee", 0.0)
        kwargs.setdefault("grid_export_fee", 0.0)
        kwargs.setdefault("export_rate_multiplier", 0.0)
        kwargs.setdefault("import_price_multiplier", 1.0)
        kwargs.setdefault("battery_wear_cost", 0.0)
        kwargs.setdefault("terminal_energy_value_eur_kwh", 0.0)
        kwargs.setdefault("slot_minutes", 60)
        kwargs.setdefault("soc_step_percent", 1.0)
        super().__init__(now, prices=prices, **kwargs)
        self._load_kw = load_kw
        # What ``find_optimal_schedule`` returned, before anything downstream
        # of it could touch the plan.
        self.validated_modes = None

    def _predict_load_kw(self, dt):
        return self._load_kw

    def _predict_pv_kw(self, dt):
        return 0.0

    def find_optimal_schedule(self, *args, **kwargs):
        schedule = super().find_optimal_schedule(*args, **kwargs)
        self.validated_modes = {slot: e.mode for slot, e in schedule.items()}
        return schedule


SLOT_MIN = 60
CAPACITY = 10.0
MIN_SOC = 10.0


def _slots(now, count=2):
    base = now.replace(minute=0, second=0, microsecond=0)
    return [base + datetime.timedelta(hours=i) for i in range(count)]


def _app(now, prices, *, previous_mode=None, soc=25.0, load_kw=1.0):
    app = RestartOptimizer(now, prices=prices, soc=soc, load_kw=load_kw)
    if previous_mode is not None:
        current_slot = app._align_to_slot(now)
        app._previous_schedule_from_sensor = {current_slot: previous_mode}
    return app


def _replay_final_plan(app):
    """Walk the plan the app ended up with, from the SOC it really started at."""
    now = app.datetime()
    current_slot = app._align_to_slot(now)
    minutes_into_slot = max(
        0.0, (now - current_slot).total_seconds() / 60.0
    )
    return app._replay_schedule(
        schedule=app.schedule,
        starting_soc=app._get_current_soc(),
        starting_temp=app._get_battery_temp(),
        current_slot=current_slot,
        minutes_into_slot=minutes_into_slot,
        prices_sorted=sorted(
            app._price_service.result, key=lambda p: p.time
        ),
    )


def _assert_everything_describes_the_executed_plan(app):
    """The five things a plan publishes must all describe the same plan."""
    schedule = app.schedule

    # 1. Nothing changed an action after the plan was validated.
    assert app.validated_modes is not None
    for slot, mode in app.validated_modes.items():
        assert schedule[slot].mode == mode, (
            f"{slot}: the plan was validated as {mode.name} and executes as "
            f"{schedule[slot].mode.name}"
        )

    # 2. The published SOC trajectory is the trajectory of THIS plan.
    for slot, entry in schedule.items():
        assert slot in app._last_dp_soc_trajectory, slot
    assert {
        slot: pair[0] for slot, pair in app._last_dp_soc_trajectory.items()
    } == pytest.approx(app.expected_soc_schedule)

    # 3. The mode census counts THIS plan.
    assert app._last_schedule_counts is not None
    assert (
        app._last_schedule_counts.summary_parts()
        == count_schedule_modes(schedule).summary_parts()
    )

    # 4. The validation replay described THIS plan...
    replay = app._last_plan_replay
    assert replay is not None
    assert set(replay.by_slot) == set(schedule)
    for slot, row in replay.by_slot.items():
        assert row.mode == schedule[slot].mode, slot

    # 5. ...and re-walking it credits no battery service the pack cannot give.
    fresh = _replay_final_plan(app)
    assert fresh is not None
    assert fresh.conservation_violations == []

    # 6. The command that went to the inverter is this plan's current slot.
    current_slot = app._align_to_slot(app.datetime())
    assert app.applied, "the current slot was never applied"
    assert app.applied[-1].mode == schedule[current_slot].mode


# ===========================================================================
# The maintainer's reproduction
# ===========================================================================

@pytest.fixture
def cheap_then_expensive():
    """0.10 EUR/kWh now, 1.00 next: the plan must import now and hold."""
    now = datetime.datetime(2024, 1, 15, 10, 0, tzinfo=TZ)
    slot_0, slot_1 = _slots(now)
    prices = [
        PricePoint(time=slot_0, price=0.10),
        PricePoint(time=slot_1, price=1.00),
    ]
    return now, slot_0, slot_1, prices


class TestRestartPreservationNeverRewritesAValidatedPlan:
    def test_the_plan_the_dp_validated_is_the_plan_that_executes(
        self, cheap_then_expensive
    ):
        now, slot_0, slot_1, prices = cheap_then_expensive
        app = _app(now, prices, previous_mode=BatteryMode.DISCHARGE)

        app.full_optimize(None)

        _assert_everything_describes_the_executed_plan(app)

    def test_no_unavailable_battery_service_is_credited(
        self, cheap_then_expensive
    ):
        """The pre-fix rewrite spent the pack an hour early.

        With 1.5 kWh usable above ``min_soc`` and 1 kWh of load per slot, a
        plan that discharges in BOTH slots asks the battery for 2 kWh. The DP
        never priced that plan, so its ``DISCHARGE`` for the second slot was
        not declared energy-limited -- the replay of the plan that actually
        executes finds service the pack does not have.
        """
        now, slot_0, slot_1, prices = cheap_then_expensive
        app = _app(now, prices, previous_mode=BatteryMode.DISCHARGE)

        app.full_optimize(None)

        replay = _replay_final_plan(app)
        assert replay is not None
        assert replay.conservation_violations == [], (
            "the executing plan credits battery service that was never "
            "validated"
        )

    def test_the_current_slot_continues_the_previous_action(
        self, cheap_then_expensive
    ):
        """Continuity is kept -- as a constraint the DP solved around."""
        now, slot_0, slot_1, prices = cheap_then_expensive
        app = _app(now, prices, previous_mode=BatteryMode.DISCHARGE)

        app.full_optimize(None)

        entry = app.schedule[slot_0]
        assert entry.mode == BatteryMode.DISCHARGE
        assert "restart" in entry.reason
        # And the DP knew: the slot it planned starts from the SOC the
        # continued discharge leaves behind, not from the measured 25 %.
        assert app._last_dp_soc_trajectory[slot_1][0] == pytest.approx(
            15.0, abs=1e-6
        )

    def test_the_previous_schedule_is_consumed_once(self, cheap_then_expensive):
        now, slot_0, slot_1, prices = cheap_then_expensive
        app = _app(now, prices, previous_mode=BatteryMode.DISCHARGE)

        app.full_optimize(None)
        assert app._previous_schedule_from_sensor is None

        # A second pass is an ordinary optimization: nothing forces the slot.
        app.full_optimize(None)
        assert app.schedule[slot_0].mode == BatteryMode.HOLD
        _assert_everything_describes_the_executed_plan(app)


# ===========================================================================
# A restart in the middle of a CHARGE slot
# ===========================================================================

class TestARestartMidChargeKeepsCharging:
    def test_the_dp_plans_the_horizon_from_the_charge_it_continues(self):
        """Half a slot of charging is 2 kWh the rest of the plan must see."""
        now = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ)
        slot_0, slot_1 = _slots(now)
        # Expensive throughout and the pack already at min SOC, so the DP left
        # to itself would HOLD both slots: any mode change here is the restart
        # continuation, not an economic choice.
        prices = [
            PricePoint(time=slot_0, price=1.00),
            PricePoint(time=slot_1, price=1.00),
        ]
        app = _app(now, prices, previous_mode=BatteryMode.CHARGE, soc=MIN_SOC)

        app.full_optimize(None)

        assert app.schedule[slot_0].mode == BatteryMode.CHARGE
        # 4 kW for the remaining 30 minutes on a 10 kWh pack at efficiency 1.0.
        assert app._last_dp_soc_trajectory[slot_1][0] == pytest.approx(
            MIN_SOC + 4.0 * 0.5 / CAPACITY * 100.0, abs=1e-6
        )
        _assert_everything_describes_the_executed_plan(app)


# ===========================================================================
# Control: no restart state at all
# ===========================================================================

class TestNoPreviousScheduleChangesNothing:
    def test_the_dp_owns_the_current_slot(self, cheap_then_expensive):
        now, slot_0, slot_1, prices = cheap_then_expensive
        app = _app(now, prices, previous_mode=None)

        app.full_optimize(None)

        assert app.schedule[slot_0].mode == BatteryMode.HOLD
        assert app.schedule[slot_1].mode == BatteryMode.DISCHARGE
        _assert_everything_describes_the_executed_plan(app)

    def test_a_previous_hold_is_not_a_continuation(self, cheap_then_expensive):
        """Only CHARGE and DISCHARGE are worth continuing across a restart."""
        now, slot_0, slot_1, prices = cheap_then_expensive
        app = _app(now, prices, previous_mode=BatteryMode.HOLD)

        app.full_optimize(None)

        assert app.schedule[slot_0].mode == BatteryMode.HOLD
        assert app.schedule[slot_1].mode == BatteryMode.DISCHARGE
        _assert_everything_describes_the_executed_plan(app)
