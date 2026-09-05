"""A restart does not override the DP: the partial slot IS the continuity.

Defect (pre-fix)
----------------

``find_optimal_schedule`` read the previous plan back from
``sensor.battery_optimizer`` and FIXED the CHARGE or DISCHARGE it woke up in for
the whole remainder of the current slot, unconditionally, before solving. Two
measured counterexamples:

* prices 2.00 / 0.05 / 0.05, previous CHARGE, restart one minute into the slot
  with 59 minutes left and 50 % on a 10 kWh pack: the forced CHARGE imports
  3.93 kWh at 2.00 EUR/kWh -- about 7.87 EUR -- where the DP left to itself
  discharges and buys the same energy two slots later at 0.05.
* prices -0.50 / 1.00 / 1.00, previous DISCHARGE: the forced DISCHARGE spends
  the pack while the grid is PAYING to take energy, where the DP charges.

And on an interval nobody published a price for, the continuation had no
provenance, so ``execute_scheduled_mode`` refused it and applied HOLD -- while
the plan had already advanced the pack across a CHARGE that never ran. On the
fixture below that is a 20-point SOC error in the published trajectory, and
every later discharge is scheduled on energy that will not exist.

Policy under test
-----------------

There is no restart override. The DP's partial-slot fraction is the continuity
mechanism: it evaluates the remaining minutes of the current slot at the real
price, from the measured SOC, so a forced continuation can only duplicate its
answer or contradict it. Whatever the DP decides for the interval the app woke
up in is what runs.

Everything here is deterministic: the settable clock and AppDaemon double from
``test_price_recovery``, the REAL planner and the REAL final-plan validation.
"""

from __future__ import annotations

import datetime
import inspect

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import BatteryMode, PricePoint, ScheduleEntry
from battery_optimizer_lib.models import PRICE_SOURCE_MARKET, count_schedule_modes

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
    """An app that has just restarted mid-slot, previously running *previous_mode*.

    The previous plan is installed BOTH ways a restart could offer it: the
    parsed snapshot the deleted ``_restore_previous_schedule_from_sensor``
    used to leave behind, and a real priced entry in ``self.schedule`` for the
    interval the app woke up in. Neither may select an action any more --
    setting them is how these tests stay red if the override comes back under
    any name.
    """
    app = RestartOptimizer(now, prices=prices, soc=soc, load_kw=load_kw)
    if previous_mode is not None:
        current_slot = app._align_to_slot(now)
        app._previous_schedule_from_sensor = {current_slot: previous_mode}
        app.schedule = {
            bo.canonical_slot_key(current_slot): ScheduleEntry(
                time=current_slot,
                mode=previous_mode,
                reason="prior plan",
                export_rate=0 if previous_mode == BatteryMode.DISCHARGE else None,
                price_source=PRICE_SOURCE_MARKET,
            )
        }
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
# The maintainer's reproduction: a reservation the restart used to spend
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


class TestThePlanTheDpValidatedIsThePlanThatExecutes:
    def test_the_reservation_survives_a_previous_discharge(
        self, cheap_then_expensive
    ):
        """``HOLD, DISCHARGE`` stays ``HOLD, DISCHARGE``.

        The pack is reserved for the 1.00 EUR/kWh slot and the grid covers the
        0.10 one. The forced continuation turned this into
        ``DISCHARGE, DISCHARGE``, moving the import into the expensive slot.
        """
        now, slot_0, slot_1, prices = cheap_then_expensive
        app = _app(now, prices, previous_mode=BatteryMode.DISCHARGE)

        app.full_optimize(None)

        assert app.schedule[slot_0].mode == BatteryMode.HOLD
        assert app.schedule[slot_1].mode == BatteryMode.DISCHARGE
        assert "restart" not in app.schedule[slot_0].reason
        _assert_everything_describes_the_executed_plan(app)

    def test_no_unavailable_battery_service_is_credited(
        self, cheap_then_expensive
    ):
        """The forced rewrite spent the pack an hour early.

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
        assert replay.conservation_violations == []

    def test_a_previous_charge_does_not_import_at_two_euros(self):
        """Measured: 3.93 kWh at 2.00 EUR/kWh, about 7.87 EUR of forced import.

        59 minutes of the slot are left, the pack charges at 4 kW, and the two
        following slots are priced at 0.05. Continuing the previous CHARGE
        buys now what the DP buys later for one fortieth of the price.
        """
        now = datetime.datetime(2024, 1, 15, 10, 1, tzinfo=TZ)
        slot_0, slot_1, slot_2 = _slots(now, 3)
        prices = [
            PricePoint(time=slot_0, price=2.00),
            PricePoint(time=slot_1, price=0.05),
            PricePoint(time=slot_2, price=0.05),
        ]
        app = _app(now, prices, previous_mode=BatteryMode.CHARGE, soc=50.0)

        app.full_optimize(None)

        assert app.schedule[slot_0].mode != BatteryMode.CHARGE, (
            "the previous CHARGE was forced through a 2.00 EUR/kWh interval"
        )
        assert app.applied[-1].mode != BatteryMode.CHARGE
        # The DP's own answer: serve the load from the pack while the grid is
        # expensive, and refill in the cheap slots.
        assert app.schedule[slot_0].mode == BatteryMode.DISCHARGE
        _assert_everything_describes_the_executed_plan(app)

    def test_a_previous_discharge_does_not_block_a_negative_price(self):
        """Measured: -0.50 EUR/kWh now, 1.00 later. The grid PAYS to charge."""
        now = datetime.datetime(2024, 1, 15, 10, 1, tzinfo=TZ)
        slot_0, slot_1, slot_2 = _slots(now, 3)
        prices = [
            PricePoint(time=slot_0, price=-0.50),
            PricePoint(time=slot_1, price=1.00),
            PricePoint(time=slot_2, price=1.00),
        ]
        app = _app(now, prices, previous_mode=BatteryMode.DISCHARGE, soc=50.0)

        app.full_optimize(None)

        assert app.schedule[slot_0].mode == BatteryMode.CHARGE, (
            "the previous DISCHARGE was forced through a negative price"
        )
        assert app.applied[-1].mode == BatteryMode.CHARGE
        _assert_everything_describes_the_executed_plan(app)


# ===========================================================================
# A restart in the middle of a CHARGE the DP itself wants to continue
# ===========================================================================

class TestARestartMidChargeStartsFromTheMeasuredSoc:
    def test_the_partial_fraction_is_the_continuity_mechanism(self):
        """Cheap now, expensive later: the DP charges the current slot itself.

        Continuity needs no override -- the DP solves the remaining 30 minutes
        of the interval at its real price, from the SOC that was MEASURED, and
        the rest of the horizon starts where that leaves the pack.
        """
        now = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ)
        slot_0, slot_1, slot_2 = _slots(now, 3)
        prices = [
            PricePoint(time=slot_0, price=0.05),
            PricePoint(time=slot_1, price=2.00),
            PricePoint(time=slot_2, price=2.00),
        ]
        # Nearly empty and 2 kW of load ahead through two 2.00 EUR/kWh slots:
        # buying now at 0.05 is the DP's own answer, not a continuation.
        app = _app(now, prices, previous_mode=BatteryMode.CHARGE, soc=15.0,
                   load_kw=2.0)

        app.full_optimize(None)

        entry = app.schedule[slot_0]
        assert entry.mode == BatteryMode.CHARGE
        assert "restart" not in entry.reason, (
            "the DP chose this CHARGE; no override may claim it"
        )
        # Starts from the MEASURED SOC ...
        assert app._last_dp_soc_trajectory[slot_0][0] == pytest.approx(
            15.0, abs=1e-6
        )
        # ... and gains exactly the remaining half slot: 4 kW x 0.5 h on a
        # 10 kWh pack at efficiency 1.0 is 20 SOC points.
        assert app._last_dp_soc_trajectory[slot_1][0] == pytest.approx(
            15.0 + 4.0 * 0.5 / CAPACITY * 100.0, abs=1e-6
        )
        _assert_everything_describes_the_executed_plan(app)


# ===========================================================================
# A restart into an interval nobody published a price for
# ===========================================================================

class TestARestartIntoAnUnpricedIntervalHoldsConsistently:
    def test_the_fallback_is_hold_no_price_and_the_trajectory_matches(self):
        """No advance through an action ``execute_scheduled_mode`` refuses.

        The continuation carried no provenance on an unpriced interval, so the
        provenance guard degraded it to HOLD -- after the plan had already
        walked the pack through 30 minutes of 4 kW CHARGE. 20 SOC points the
        pack never gained, and every later DISCHARGE planned on them.
        """
        now = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ)
        slot_0, slot_1, slot_2 = _slots(now, 3)
        # The interval the app woke up in is absent from the data.
        prices = [
            PricePoint(time=slot_1, price=1.00),
            PricePoint(time=slot_2, price=1.00),
        ]
        app = _app(now, prices, previous_mode=BatteryMode.CHARGE, soc=50.0)
        # A restart has no plan at all: the sensor snapshot is the only thing
        # that survived, and nothing may be retained from it.
        app.schedule = {}

        app.full_optimize(None)

        entry = bo.lookup_by_time(
            app.schedule, slot_0, app._get_local_timezone()
        )
        assert entry is not None, "the interval that is RUNNING must be in the plan"
        assert entry.mode == BatteryMode.HOLD
        assert entry.reason == "no_price"
        assert entry.price_source is None
        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "no_price", (
            "a continuation the guard refuses is not a fallback the retry sees"
        )
        assert app._price_retry_pending()
        # HOLD without PV moves nothing: the pack is where it was measured.
        assert app.expected_soc_schedule[slot_1] == pytest.approx(50.0, abs=1e-6)
        assert app._last_dp_soc_trajectory[slot_1][0] == pytest.approx(
            50.0, abs=1e-6
        )


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

    def test_the_answer_is_identical_with_and_without_restart_state(
        self, cheap_then_expensive
    ):
        """The previous plan is not an input to anything."""
        now, slot_0, slot_1, prices = cheap_then_expensive
        plain = _app(now, prices, previous_mode=None)
        restarted = _app(now, prices, previous_mode=BatteryMode.DISCHARGE)

        plain.full_optimize(None)
        restarted.full_optimize(None)

        assert {s: e.mode for s, e in restarted.schedule.items()} == {
            s: e.mode for s, e in plain.schedule.items()
        }


# ===========================================================================
# The override is gone, not merely unreachable
# ===========================================================================

class TestNothingSelectsAnActionFromTheSensorSnapshot:
    def test_the_restart_override_and_its_state_are_removed(self):
        assert not hasattr(bo.BatteryOptimizer, "_restart_continuation_entry")
        assert not hasattr(
            bo.BatteryOptimizer, "_restore_previous_schedule_from_sensor"
        )
        source = inspect.getsource(bo)
        assert "_previous_schedule_from_sensor" not in source, (
            "the restored sensor snapshot has no consumer; it must not be "
            "carried as state either"
        )
