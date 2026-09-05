"""A missing future interval is a forced HOLD, not an absence from the timeline.

Defect (pre-fix)
----------------

``find_optimal_schedule`` handed the DP only the intervals a source had
published.  A hole inside the horizon therefore did not exist for planning at
all: the slot after the hole was treated as following the slot before it, and
the PV, the load, the SOC and the temperature of the missing quarter hour were
never modelled.  The final replay skipped it for the same reason and reported
success.

The maintainer's reproduction, and the fixture below: a 10 kWh pack sitting at
its 10 % minimum with unit efficiencies; 10:00 published at 0.50 with no load
and no PV; 10:15 NOT published, with 4 kW of forecast PV; 10:30 published at
1.00 with 4 kW of forecast load.  Both planning paths chose CHARGE at 10:00 and
imported a kWh for nothing - HOLD through the gap stores exactly the kWh that
10:30 needs, for free.  ``_validate_final_plan`` agreed with the plan because it
walked the same two slots, and its SOC from 10:30 on was ten points low.

Policy under test
-----------------

The modelled horizon is the CONTIGUOUS slot sequence from the first interval
the DP is given to the last PRICED one.  An unpriced slot enters the DP with
``price=None``:

* only the HOLD transition is evaluated - the action is fixed, so there is
  nothing to choose;
* PV absorption is modelled normally (headroom, span rate), because that is
  physics and does not need a price;
* the slot's grid import and export cash flows are omitted from the objective.
  Import is path-independent under a fixed action, so omitting it cannot change
  which plan wins; export at an unknown price is valued at 0, which is the
  conservative direction;
* the schedule carries a ``HOLD`` entry with reason ``no_price``, no price
  provenance and no marginal value, so ``execute_scheduled_mode`` applies HOLD
  and arms the ``no_price`` retry when the slot becomes current.

Nothing is extended beyond the last priced interval: a gap at the end of the
horizon is the horizon ending.
"""

from __future__ import annotations

import datetime

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import BatteryMode, PricePoint, ScheduleEntry
from battery_optimizer_lib.models import PRICE_SOURCE_MARKET

from tests.test_current_slot_price import PlanningOptimizer, current_entry
from tests.test_price_recovery import TZ, UTC


SLOT = 15

SLOT_10_00 = datetime.datetime(2024, 1, 15, 10, 0, tzinfo=TZ)
SLOT_10_15 = datetime.datetime(2024, 1, 15, 10, 15, tzinfo=TZ)
SLOT_10_30 = datetime.datetime(2024, 1, 15, 10, 30, tzinfo=TZ)
SLOT_10_45 = datetime.datetime(2024, 1, 15, 10, 45, tzinfo=TZ)

CAPACITY = 10.0
MIN_SOC = 10.0
CHEAP = 0.50
DEAR = 1.00
PV_KW = 4.0
LOAD_KW = 4.0


class GapOptimizer(PlanningOptimizer):
    """The maintainer's fixture: unit efficiencies, no fees, no salvage value.

    ``load_by_slot`` / ``pv_by_slot`` are keyed by UTC instant so the autumn
    fold's two 03:00 intervals stay distinct.

    ``calculate_expected_soc_schedule`` is un-stubbed: the published SOC map is
    one of the consumers that has to see the gap.
    """

    calculate_expected_soc_schedule = (
        bo.BatteryOptimizer.calculate_expected_soc_schedule
    )

    def __init__(self, now, *, load_by_slot=None, pv_by_slot=None, **kwargs):
        kwargs.setdefault("battery_capacity", CAPACITY)
        kwargs.setdefault("charge_rate", 4.5)
        kwargs.setdefault("discharge_rate", 4.5)
        kwargs.setdefault("efficiency", 1.0)
        kwargs.setdefault("inverter_efficiency", 1.0)
        kwargs.setdefault("grid_fee", 0.0)
        kwargs.setdefault("grid_export_fee", 0.0)
        kwargs.setdefault("battery_wear_cost", 0.0)
        kwargs.setdefault("terminal_energy_value_eur_kwh", 0.0)
        kwargs.setdefault("soc_step_percent", 0.25)
        super().__init__(now, **kwargs)
        self._load_by_slot = {
            k.astimezone(UTC): v for k, v in (load_by_slot or {}).items()
        }
        self._pv_by_slot = {
            k.astimezone(UTC): v for k, v in (pv_by_slot or {}).items()
        }

    def _predict_load_kw(self, dt):
        return self._load_by_slot.get(dt.astimezone(UTC), 0.0)

    def _predict_pv_kw(self, dt):
        return self._pv_by_slot.get(dt.astimezone(UTC), 0.0)

    def _predict_pv_kw_raw(self, dt):
        return self._predict_pv_kw(dt)


class ScriptedService:
    """Returns exactly the intervals the test scripted, gaps and all."""

    def __init__(self, points):
        self.points = list(points)
        self.cached_prices = list(points)

    def get_prices(self):
        return list(self.points)


def _repro_app(now=SLOT_10_00):
    """10:00 priced, 10:15 missing with PV, 10:30 priced with load."""
    app = GapOptimizer(
        now,
        soc=MIN_SOC,
        pv_by_slot={SLOT_10_15: PV_KW},
        load_by_slot={SLOT_10_30: LOAD_KW},
    )
    app._price_service = ScriptedService([
        PricePoint(time=SLOT_10_00, price=CHEAP, end=SLOT_10_15),
        PricePoint(time=SLOT_10_30, price=DEAR, end=SLOT_10_45),
    ])
    return app


def _entry(app, slot):
    return current_entry(app, slot)


def _soc(app, slot):
    """End-of-slot SOC the FINAL validated replay says the plan reaches."""
    replay = app._last_plan_replay
    for key, row in replay.by_slot.items():
        if key.astimezone(UTC) == slot.astimezone(UTC):
            return row.soc_end
    return None


# ===========================================================================
# The reproduction, on both planning paths
# ===========================================================================

@pytest.mark.parametrize("path", ["full_optimize", "recalculate"])
class TestTheGapIsPlannedThrough:
    @staticmethod
    def _run(app, path):
        if path == "full_optimize":
            app.full_optimize(None)
        else:
            app._recalculate_remaining_schedule(app._soc)

    def test_the_free_pv_is_used_instead_of_importing(self, path):
        app = _repro_app()

        self._run(app, path)

        assert _entry(app, SLOT_10_00).mode == BatteryMode.HOLD, (
            "charging at 10:00 imports a kWh the 10:15 PV supplies for free"
        )
        gap = _entry(app, SLOT_10_15)
        assert gap is not None, "the missing interval is still part of the plan"
        assert gap.mode == BatteryMode.HOLD
        assert gap.reason == "no_price"
        assert gap.price_source is None
        assert gap.marginal_value_eur_kwh is None
        assert _entry(app, SLOT_10_30).mode == BatteryMode.DISCHARGE

    def test_the_gap_stores_the_pv_and_the_load_is_served_from_it(self, path):
        app = _repro_app()

        self._run(app, path)

        # 1.0 kWh at 10 % -> 4 kW x 0.25 h stored -> 2.0 kWh = 20 %, then the
        # whole kWh is served back at 10:30.
        assert _soc(app, SLOT_10_00) == pytest.approx(MIN_SOC)
        assert _soc(app, SLOT_10_15) == pytest.approx(20.0)
        assert _soc(app, SLOT_10_30) == pytest.approx(MIN_SOC)

    def test_nothing_is_imported(self, path):
        app = _repro_app()

        self._run(app, path)

        replay = app._last_plan_replay
        assert replay.total_grid_import_ac_kwh == pytest.approx(0.0, abs=1e-9)
        assert replay.total_grid_charge_ac_kwh == pytest.approx(0.0, abs=1e-9)

    def test_the_validator_covers_the_gap(self, path):
        app = _repro_app()

        self._run(app, path)

        replayed = {k.astimezone(UTC) for k in app._last_plan_replay.by_slot}
        assert SLOT_10_15.astimezone(UTC) in replayed, (
            "a slot the plan is not replayed over is a slot the validator "
            "cannot contradict"
        )
        assert app._last_plan_replay.ok

    def test_the_published_trajectory_includes_the_gap(self, path):
        app = _repro_app()

        self._run(app, path)

        trajectory = {
            k.astimezone(UTC): v for k, v in app._last_dp_soc_trajectory.items()
        }
        assert SLOT_10_15.astimezone(UTC) in trajectory
        assert trajectory[SLOT_10_15.astimezone(UTC)][1] == pytest.approx(20.0)
        # The expected-SOC map the deviation detector compares against is the
        # same walk, and it must not skip the quarter hour either.
        expected = {
            k.astimezone(UTC): v for k, v in app.expected_soc_schedule.items()
        }
        assert SLOT_10_15.astimezone(UTC) in expected
        assert expected[SLOT_10_15.astimezone(UTC)] == pytest.approx(MIN_SOC)
        assert expected[SLOT_10_30.astimezone(UTC)] == pytest.approx(20.0), (
            "10:30 starts from the SOC the gap's PV left the pack at"
        )

    def test_the_gap_is_counted_and_costed(self, path):
        app = _repro_app()

        self._run(app, path)

        assert app._last_schedule_counts.hold == 2
        assert app._last_schedule_counts.discharge == 1
        costed = {k.astimezone(UTC) for k in app._last_projected_costs}
        assert SLOT_10_15.astimezone(UTC) in costed


# ===========================================================================
# The gap with LOAD: held, and its unknown import is not scored
# ===========================================================================

class TestAGapWithLoad:
    def _app(self):
        app = GapOptimizer(
            SLOT_10_00,
            soc=60.0,
            load_by_slot={SLOT_10_15: LOAD_KW},
        )
        app._price_service = ScriptedService([
            PricePoint(time=SLOT_10_00, price=CHEAP, end=SLOT_10_15),
            PricePoint(time=SLOT_10_30, price=DEAR, end=SLOT_10_45),
        ])
        return app

    def test_the_slot_is_held_and_the_grid_covers_it(self):
        app = self._app()

        app.full_optimize(None)

        gap = _entry(app, SLOT_10_15)
        assert gap.mode == BatteryMode.HOLD
        # HOLD leaves the pack alone; the house draws the whole kWh from the
        # grid, at a price nobody published.
        row = app._last_plan_replay.by_slot[
            next(k for k in app._last_plan_replay.by_slot
                 if k.astimezone(UTC) == SLOT_10_15.astimezone(UTC))
        ]
        assert row.grid_import_ac_kwh == pytest.approx(1.0)
        assert row.soc_start == pytest.approx(row.soc_end)

    def test_the_unknown_import_is_not_priced(self):
        """The cash flow is omitted, deliberately, and documented as such.

        The action is FIXED, so the import is the same on every path through
        the slot: scoring it with a made-up number could only change the plan
        by accident, and scoring it with a real one is impossible.
        """
        app = self._app()

        app.full_optimize(None)

        row = next(
            r for k, r in app._last_plan_replay.by_slot.items()
            if k.astimezone(UTC) == SLOT_10_15.astimezone(UTC)
        )
        assert row.grid_import_ac_kwh > 0
        assert row.value_eur == pytest.approx(0.0), (
            "an unpriced slot contributes no cash flow to the plan's value"
        )


# ===========================================================================
# The cloud-safe hedge has nothing to price the trade with
# ===========================================================================

class TestTheHedgeSkipsUnpricedSlots:
    def test_the_gap_stays_a_hold_although_it_is_an_eligible_candidate(self):
        """4 kW of PV against no load is exactly the hedge's own criterion.

        `_cloud_safe_candidates` selects the slot; the hedge then refuses it
        because there is no price to establish that the avoided import beats
        wear and the value of keeping the kWh.
        """
        app = _repro_app()

        app.full_optimize(None)

        assert bo._cloud_safe_candidates(
            app.schedule, app._predict_load_kw, app._predict_pv_kw
        ), "the gap slot IS eligible on the PV test"
        gap = _entry(app, SLOT_10_15)
        assert gap.mode == BatteryMode.HOLD
        assert "[cloud-safe]" not in gap.reason
        assert not any("Cloud-safe" in message for message, _ in app.logs)


# ===========================================================================
# The schedule log and the sensor attributes tolerate a valueless entry
# ===========================================================================

class TestTheFormatterHandlesAGapEntry:
    def _formatter(self, app):
        return bo.ScheduleFormatter(
            config=bo.ScheduleFormatterConfig(
                slot_minutes=app.config.slot_minutes,
                slot_hours=app.config.slot_minutes / 60.0,
                battery_capacity=app.config.battery_capacity,
                charge_rate=app.config.charge_rate,
                discharge_rate=app.config.discharge_rate,
                export_discharge_rate=app.config.export_discharge_rate,
                efficiency=app.config.efficiency,
                battery_wear_cost=app.config.battery_wear_cost,
                decision_log_level=1,
                inverter_efficiency=app.config.inverter_efficiency,
            ),
            log_func=lambda *a, **k: None,
        )

    def test_the_entry_survives_every_presentation_path(self):
        app = _repro_app()
        app.full_optimize(None)
        formatter = self._formatter(app)

        rows = formatter.format_schedule_list(app.schedule)
        gap_row = next(
            r for r in rows
            if r["time"] == _entry(app, SLOT_10_15).time.isoformat()
        )
        assert gap_row["value"] is None, (
            "no marginal value can be reported for a slot with no price"
        )
        assert gap_row["reason"] == "no_price"

        markdown = formatter.format_schedule_markdown(
            schedule=app.schedule,
            now=app.datetime(),
            local_tz=TZ,
            align_to_slot_func=app._align_to_slot,
            dp_soc_trajectory=app._last_dp_soc_trajectory,
            predict_load_kw=app._predict_load_kw,
            predict_pv_kw=app._predict_pv_kw,
        )
        assert "10:15" in markdown

        # The decision-log and DP-trace paths format the price directly.
        verbose = _repro_app()
        verbose.config.decision_log_level = 3
        verbose._schedule_formatter = formatter
        verbose.full_optimize(None)
        assert _entry(verbose, SLOT_10_15).reason == "no_price"
        assert any(
            "unpriced" in message for message, _ in verbose.logs
        ), "the DP trace really did format the gap slot's missing price"

        # The full schedule log walks every entry through the value column.
        formatter.log_schedule(
            schedule=app.schedule,
            expected_soc=app.expected_soc_schedule,
            expected_temp=app.expected_temp_schedule,
            dp_soc_trajectory=app._last_dp_soc_trajectory,
            dp_temp_trajectory=app._last_dp_temp_trajectory,
            projected_costs=app._last_projected_costs,
            local_tz=TZ,
            predict_load_kw=app._predict_load_kw,
            predict_pv_kw=app._predict_pv_kw,
            min_soc=app.min_soc,
            max_soc=app.max_soc,
        )


# ===========================================================================
# The horizon ends at the last PRICED interval
# ===========================================================================

class TestTheHorizonIsNotExtendedIntoTheGap:
    def test_a_trailing_gap_is_not_modelled(self):
        app = GapOptimizer(SLOT_10_00, soc=50.0)
        app._price_service = ScriptedService([
            PricePoint(time=SLOT_10_00, price=CHEAP, end=SLOT_10_15),
            PricePoint(time=SLOT_10_30, price=DEAR, end=SLOT_10_45),
        ])

        app.full_optimize(None)

        modelled = sorted(k.astimezone(UTC) for k in app.schedule)
        assert modelled == [
            SLOT_10_00.astimezone(UTC),
            SLOT_10_15.astimezone(UTC),
            SLOT_10_30.astimezone(UTC),
        ], "nothing is planned past the last interval a source published"


# ===========================================================================
# Temperature keeps evolving across the gap
# ===========================================================================

class TestTemperatureEvolvesThroughTheGap:
    def test_the_thermal_walk_is_continuous(self):
        app = _repro_app()

        app.full_optimize(None)

        temps = {
            k.astimezone(UTC): v
            for k, v in app._last_dp_temp_trajectory.items()
        }
        assert SLOT_10_15.astimezone(UTC) in temps, (
            "a slot missing from the thermal walk hands the next slot the "
            "temperature of the one before it"
        )
        gap_start, gap_end = temps[SLOT_10_15.astimezone(UTC)]
        after_start, _ = temps[SLOT_10_30.astimezone(UTC)]
        before_start, before_end = temps[SLOT_10_00.astimezone(UTC)]
        assert gap_start == pytest.approx(before_end)
        assert after_start == pytest.approx(gap_end)


# ===========================================================================
# Execution and recovery
# ===========================================================================

class TestExecutionOfAGapSlot:
    def test_it_applies_hold_and_arms_the_no_price_retry(self):
        app = _repro_app()
        app.full_optimize(None)
        armed_after_planning = len(app.pending_retries())

        app.advance(SLOT * 60)          # the gap slot is now current
        app.execute_scheduled_mode(None)

        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "no_price"
        assert len(app.pending_retries()) == armed_after_planning, (
            "one armed retry already covers this failure"
        )

    def test_a_gap_entry_can_never_be_sent_as_anything_else(self):
        """The entry carries no provenance, so the execution guard refuses it
        even if something downstream changes its mode."""
        app = _repro_app()
        app.full_optimize(None)
        app.advance(SLOT * 60)
        entry = _entry(app, SLOT_10_15)
        entry.mode = BatteryMode.CHARGE

        app.execute_scheduled_mode(None)

        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "unpriced_slot"

    def test_the_retry_is_armed_once_although_the_current_slot_is_priced(self):
        app = _repro_app()

        app.full_optimize(None)

        assert app._price_horizon.last_health.reason == "gap"
        assert app._price_horizon.last_health.has_current is True
        assert len(app.pending_retries()) == 1
        assert app._price_horizon.attempts == 1


# ===========================================================================
# DST: the modelled horizon steps in elapsed time, not on the clock face
# ===========================================================================

class TestAutumnFoldGap:
    def test_the_gap_spans_the_repeated_hour(self, riga_timezone):
        first_03 = datetime.datetime(
            2024, 10, 27, 3, 0, tzinfo=riga_timezone, fold=0
        )
        second_03 = datetime.datetime(
            2024, 10, 27, 3, 0, tzinfo=riga_timezone, fold=1
        )
        app = GapOptimizer(first_03, tz=riga_timezone, soc=50.0)
        app._price_service = ScriptedService([
            PricePoint(time=first_03, price=CHEAP),
            PricePoint(time=second_03, price=DEAR),
        ])

        app.full_optimize(None)

        modelled = sorted(k.astimezone(UTC) for k in app.schedule)
        assert len(modelled) == 5, (
            "the repeated hour is four quarter-hours of elapsed time, so the "
            "two 03:00 intervals are five slots apart, not zero"
        )
        assert modelled[0] == first_03.astimezone(UTC)
        assert modelled[-1] == second_03.astimezone(UTC)
        assert _entry(app, first_03).price_source == PRICE_SOURCE_MARKET
        assert _entry(app, second_03).price_source == PRICE_SOURCE_MARKET
        for key in modelled[1:-1]:
            entry = bo.lookup_by_time(app.schedule, key, riga_timezone)
            assert entry.reason == "no_price"
            assert entry.mode == BatteryMode.HOLD
