"""A4 and A5: what ``find_optimal_schedule`` publishes, and what it solves from.

A4 -- the mode census (``_last_schedule_counts``), the projected-cost column
(``_last_projected_costs``) and the decision log (``_last_charge_slots``) were
derived BEFORE ``_validate_final_plan``, whose ``_resolve_plan_shortfall`` can
revert a cloud-safe hedge slot back to HOLD. Everything published then described
a plan that is not the one returned.

A5 -- when nobody published a price for the current interval, the DP was handed
the SOC measured mid-slot and solved the remaining horizon from it, while the
current slot went on executing a retained entry the plan never saw. A retained
CHARGE running 10:07 -> 10:15 at 4.5 kW x 0.85 adds ~3.6 SOC points on the
14.3 kWh reference pack; a retained DISCHARGE errs the other way.
"""

from __future__ import annotations

import copy
import datetime
from typing import Dict, List, Optional

import pytest

from battery_optimizer import (
    BatteryLearningEngine,
    BatteryMode,
    PricePoint,
    ScheduleEntry,
)
from battery_optimizer_lib import (
    BatteryCostTracker,
    BatteryCostConfig,
    BatteryOptimizerConfig,
    ScheduleFormatter,
    ScheduleFormatterConfig,
    TemperatureProjector,
)
from battery_optimizer_lib.models import PRICE_SOURCE_MARKET

from battery_optimizer import BatteryOptimizer


class _NoPvForecast:
    def refresh(self, force: bool = False) -> bool:
        return False


class _NoPrices:
    def get_prices_for_date(self, date, tz):
        return []


class PlannerApp:
    """A BatteryOptimizer surface with fully controlled forecasts.

    Everything ``find_optimal_schedule`` reaches for is real -- the DP, the
    cloud-safe hedge, the cost tracker, the formatter and the final-plan
    validation. Only the HA I/O is replaced.
    """

    def __init__(
        self,
        *,
        battery_capacity: float = 14.3,
        charge_rate: float = 4.5,
        discharge_rate: float = 4.5,
        efficiency: float = 0.85,
        inverter_efficiency: float = 1.0,
        min_soc: float = 10.0,
        max_soc: float = 100.0,
        slot_minutes: int = 15,
        battery_temp: Optional[float] = 20.0,
        decision_log_level: int = 1,
        terminal_energy_value_eur_kwh: Optional[float] = 0.0,
        battery_wear_cost: float = 0.0,
        grid_fee: float = 0.0,
        export_rate_multiplier: float = 0.0,
        now: Optional[datetime.datetime] = None,
    ):
        self.config = BatteryOptimizerConfig(
            battery_capacity=battery_capacity,
            charge_rate=charge_rate,
            discharge_rate=discharge_rate,
            efficiency=efficiency,
            inverter_efficiency=inverter_efficiency,
            grid_fee=grid_fee,
            grid_export_fee=0.0,
            export_rate_multiplier=export_rate_multiplier,
            import_price_multiplier=1.0,
            battery_wear_cost=battery_wear_cost,
            terminal_energy_value_eur_kwh=terminal_energy_value_eur_kwh,
            slot_minutes=slot_minutes,
            soc_step_percent=1.0,
            default_min_soc=min_soc,
            default_max_soc=max_soc,
            decision_log_level=decision_log_level,
        )
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.battery_avg_cost = 0.05

        self._current_time = now or datetime.datetime(2026, 6, 21, 10, 0)
        self._battery_temp = battery_temp
        self._ambient_service = None

        self.load_by_slot: Dict[datetime.datetime, float] = {}
        self.pv_by_slot: Dict[datetime.datetime, float] = {}
        self.default_load_kw = 0.0
        self.default_pv_kw = 0.0

        self.learning_engine = BatteryLearningEngine(
            battery_capacity_kwh=self.config.battery_capacity,
            nominal_charge_rate_kw=self.config.charge_rate,
            nominal_efficiency=self.config.efficiency,
        )

        self.schedule: Dict[datetime.datetime, ScheduleEntry] = {}
        self.messages: List[tuple] = []
        self._last_min_charge_slots = 0
        self._last_charge_slots = []
        self._last_projected_costs = {}
        self._last_schedule_counts = None
        self._last_dp_soc_trajectory = {}
        self._last_dp_temp_trajectory = {}
        self._last_plan_replay = None
        self.replay_calls = 0

        self._schedule_formatter = ScheduleFormatter(
            config=ScheduleFormatterConfig(
                slot_minutes=self.config.slot_minutes,
                slot_hours=self.config.slot_hours,
                battery_capacity=self.config.battery_capacity,
                charge_rate=self.config.charge_rate,
                discharge_rate=self.config.discharge_rate,
                export_discharge_rate=self.config.export_discharge_rate,
                efficiency=self.config.efficiency,
                battery_wear_cost=self.config.battery_wear_cost,
                decision_log_level=self.config.decision_log_level,
            ),
            log_func=self.log,
            learning_engine=self.learning_engine,
        )
        self._cost_tracker = BatteryCostTracker(
            config=BatteryCostConfig(
                battery_capacity=self.config.battery_capacity,
                efficiency=self.config.efficiency,
                slot_minutes=self.config.slot_minutes,
                charge_rate=self.config.charge_rate,
                discharge_rate=self.config.discharge_rate,
                grid_fee=self.config.grid_fee,
                battery_wear_cost=self.config.battery_wear_cost,
            ),
            get_state_func=lambda e: None,
            call_service_func=lambda *a, **k: None,
            get_datetime_func=self.datetime,
            get_timezone_func=self._get_local_timezone,
            align_to_slot_func=self._align_to_slot,
            get_min_soc_func=lambda: self.min_soc,
            get_max_soc_func=lambda: self.max_soc,
            get_current_soc_func=lambda: 50.0,
            get_battery_temp_func=lambda: self._battery_temp,
            learning_engine=self.learning_engine,
            get_cached_prices_func=lambda: [],
            save_learning_data_func=lambda: None,
            update_learning_sensor_func=lambda: None,
            log_func=self.log,
        )
        self._pv_forecast_service = _NoPvForecast()

    # --- app surface ---------------------------------------------------
    @property
    def _temp_projector(self):
        return TemperatureProjector(
            learning_engine=self.learning_engine,
            ambient_provider=self._ambient_service,
        )

    @property
    def _price_service(self):
        return _NoPrices()

    def datetime(self):
        return self._current_time

    def log(self, message: str, level: str = "INFO"):
        self.messages.append((level, message))

    def _align_to_slot(self, dt: datetime.datetime) -> datetime.datetime:
        minutes = dt.hour * 60 + dt.minute
        start = (minutes // self.config.slot_minutes) * self.config.slot_minutes
        return dt.replace(
            hour=start // 60, minute=start % 60, second=0, microsecond=0
        )

    def _get_local_timezone(self):
        return None

    def _get_battery_temp(self):
        return self._battery_temp

    def _predict_load_kw(self, dt):
        return self.load_by_slot.get(dt, self.default_load_kw)

    def _predict_pv_kw(self, dt):
        return self.pv_by_slot.get(dt, self.default_pv_kw)


for _name in (
    "find_optimal_schedule",
    "project_schedule_trajectory",
    "calculate_expected_soc_schedule",
    "_replay_schedule",
    "_validate_final_plan",
    "_resolve_plan_shortfall",
    "_compute_slot_fractions",
    "_entry_has_real_price",
    "_retain_current_slot_if_unpriced",
    "_advance_across_unpriced_current_slot",
    "_advance_current_slot",
    "_rate_refinement_diagnostics",
):
    setattr(PlannerApp, _name, getattr(BatteryOptimizer, _name, None))


def _entry(slot, mode, *, priced=True, export_rate=None):
    entry = ScheduleEntry(time=slot, mode=mode, reason="prior plan")
    entry.export_rate = export_rate
    if priced:
        entry.price_source = PRICE_SOURCE_MARKET
    return entry


# ---------------------------------------------------------------------------
# A4
# ---------------------------------------------------------------------------

HEDGE_BASE = datetime.datetime(2026, 6, 21, 10, 0)


class _RevertingApp(PlannerApp):
    """The PV forecast the plan validator sees is not the one the hedge saw.

    The hedge only converts a HOLD slot when the forecast says PV covers the
    whole load, so with one self-consistent forecast the converted slot serves
    nothing from the pack and can never be short. The revert branch exists for
    the case where the two disagree; this double is what makes it reachable
    deterministically. ``_replay_schedule`` is the seam because
    ``find_optimal_schedule`` calls it once for the hedge's export test and then
    again inside ``_validate_final_plan`` -- so from the validation replay
    onwards, PV is gone.
    """

    collapse_slot: Optional[datetime.datetime] = None

    def _replay_schedule(self, **kwargs):
        self.replay_calls += 1
        if self.replay_calls >= 2 and self.collapse_slot is not None:
            self.pv_by_slot = {
                k: v for k, v in self.pv_by_slot.items() if k != self.collapse_slot
            }
        return BatteryOptimizer._replay_schedule(self, **kwargs)


def _hedge_case():
    app = _RevertingApp(
        battery_capacity=10.0,
        charge_rate=4.0,
        discharge_rate=4.0,
        efficiency=1.0,
        min_soc=10.0,
        slot_minutes=60,
        grid_fee=0.0,
        decision_log_level=1,
    )
    slots = [HEDGE_BASE + datetime.timedelta(hours=i) for i in range(2)]
    # Slot 0: PV more than covers the load, so the DP calls it HOLD and the
    # hedge converts it — and it is that slot's PV that vanishes before
    # validation. The surplus is deliberate rather than an exact match: it is
    # what makes the projected-cost column DIFFER between the pre-validation
    # plan (a discharge_to_load slot banking 1 kWh of surplus at the export
    # floor) and the returned one (a HOLD slot whose PV never arrived). Without
    # it both orderings produce the same numbers and the column cannot tell
    # anyone which plan it describes.
    # Slot 1 keeps a PV surplus, so the plan can still gain stored energy and
    # the projected-cost column is not trivially empty.
    app.load_by_slot = {slots[0]: 1.0, slots[1]: 0.0}
    app.pv_by_slot = {slots[0]: 2.0, slots[1]: 1.0}
    app.collapse_slot = slots[0]
    prices = [
        PricePoint(time=slots[0], price=1.00),
        PricePoint(time=slots[1], price=0.01),
    ]
    return app, slots, prices


def _project(app, schedule, prices, starting_soc=10.0):
    """A fresh projected-cost column for *schedule*, as the app would build it."""
    from battery_optimizer_lib.timezone_utils import canonical_slot_key

    costs, _ = app._cost_tracker.project_costs(
        schedule,
        starting_soc,
        app.battery_avg_cost,
        {canonical_slot_key(p.time): p.price for p in prices},
        predict_load_func=app._predict_load_kw,
        predict_pv_func=app._predict_pv_kw,
        starting_temp=app._get_battery_temp(),
        learning_engine=app.learning_engine,
        temp_projector=app._temp_projector,
    )
    return costs


class TestTheCensusDescribesThePlanThatIsReturned:
    def test_a_reverted_hedge_slot_is_counted_as_hold(self):
        app, slots, prices = _hedge_case()

        # Everything the old ordering would have derived from, captured at the
        # instant before validation: the plan with the hedge conversion still
        # in it, and the PV view that has not collapsed yet.
        pre_validation = {}
        _real_validate = app._validate_final_plan

        def _capture(**kwargs):
            pre_validation["schedule"] = copy.deepcopy(kwargs["schedule"])
            pre_validation["pv"] = dict(app.pv_by_slot)
            return _real_validate(**kwargs)

        app._validate_final_plan = _capture

        # The pack is at min SOC: the moment PV disappears, the converted
        # DISCHARGE slot cannot serve a joule.
        schedule = app.find_optimal_schedule(prices, 0, current_soc=10.0)

        assert schedule[slots[0]].mode == BatteryMode.HOLD, (
            "the unserviceable hedge conversion was not reverted -- this test "
            "no longer exercises the ordering it is about"
        )
        assert pre_validation["schedule"][slots[0]].mode == BatteryMode.DISCHARGE, (
            "the pre-validation plan must really differ, or nothing below can "
            "distinguish the two orderings"
        )

        # --- the census -----------------------------------------------------
        counts = app._last_schedule_counts
        assert counts is not None
        returned_hold = sum(
            1 for e in schedule.values() if e.mode == BatteryMode.HOLD
        )
        returned_self_consume = sum(
            1
            for e in schedule.values()
            if e.mode == BatteryMode.DISCHARGE
            and not (e.export_rate is not None and e.export_rate > 0)
        )
        assert counts.hold == returned_hold
        assert counts.self_consume == returned_self_consume

        # --- the projected-cost column --------------------------------------
        # Same three derivations, same ordering question. This one is the
        # sensitive half: the numbers a pre-validation source would publish are
        # measurably different, so the assertion cannot pass by coincidence.
        assert set(app._last_projected_costs) == set(schedule)
        assert app._last_projected_costs == pytest.approx(
            _project(app, schedule, prices)
        )

        saved_pv = app.pv_by_slot
        app.pv_by_slot = pre_validation["pv"]
        try:
            stale = _project(app, pre_validation["schedule"], prices)
        finally:
            app.pv_by_slot = saved_pv
        assert app._last_projected_costs != pytest.approx(stale), (
            "the projected-cost column would be identical either way -- this "
            "scenario no longer proves the column is derived after validation"
        )

        # --- the decision log ------------------------------------------------
        # `_last_charge_slots` lists the plan's CHARGE slots. It must describe
        # the plan that is returned: the reverted slot is not one of them, and
        # nothing appears that the returned plan does not call CHARGE.
        #
        # Unlike the column above this one cannot be made sensitive by any
        # scenario: `_resolve_plan_shortfall` only ever turns DISCHARGE into
        # HOLD, and it mutates the entries in place, so the CHARGE set is the
        # same object graph before and after. The correspondence is asserted
        # anyway, because it is the property that would break if the rows were
        # ever built from a genuine pre-validation COPY of the plan.
        logged = {row["time"] for row in app._last_charge_slots}
        returned_charge = {
            slot.isoformat()
            for slot, entry in schedule.items()
            if entry.mode == BatteryMode.CHARGE
        }
        assert logged == returned_charge
        assert slots[0].isoformat() not in logged

    def test_the_decision_log_rows_describe_the_returned_plan(self):
        app, slots, prices = _hedge_case()
        schedule = app.find_optimal_schedule(prices, 0, current_soc=10.0)
        assert schedule[slots[0]].mode == BatteryMode.HOLD
        # The decision log records the CHARGE slots it selected; whatever it
        # recorded must exist in the returned plan with that mode.
        for row in app._last_charge_slots:
            slot_time = row.get("time") if isinstance(row, dict) else None
            if slot_time is None:
                continue
            assert schedule[slot_time].mode == BatteryMode.CHARGE

    def test_the_projected_cost_column_describes_the_returned_plan(self):
        app, slots, prices = _hedge_case()
        schedule = app.find_optimal_schedule(prices, 0, current_soc=10.0)
        assert schedule[slots[0]].mode == BatteryMode.HOLD
        # Recomputed after the revert: the projected cost of the plan that is
        # returned, not of the one the hedge produced. Compare against a fresh
        # projection of the FINAL schedule.
        from battery_optimizer_lib.timezone_utils import canonical_slot_key

        expected, _ = app._cost_tracker.project_costs(
            schedule,
            10.0,
            app.battery_avg_cost,
            {canonical_slot_key(p.time): p.price for p in prices},
            predict_load_func=app._predict_load_kw,
            predict_pv_func=app._predict_pv_kw,
            starting_temp=app._get_battery_temp(),
            learning_engine=app.learning_engine,
            temp_projector=app._temp_projector,
        )
        assert app._last_projected_costs == pytest.approx(expected)


# ---------------------------------------------------------------------------
# A5
# ---------------------------------------------------------------------------

DAY = datetime.datetime(2026, 6, 21)
SLOT_10_00 = DAY.replace(hour=10, minute=0)
SLOT_10_15 = DAY.replace(hour=10, minute=15)
SLOT_10_30 = DAY.replace(hour=10, minute=30)
NOW = DAY.replace(hour=10, minute=7)

START_SOC = 40.0
CAPACITY = 14.3
CHARGE_KW = 4.5
EFFICIENCY = 0.85
# 8 minutes of the 15-minute slot are left at 10:07.
REMAINING_H = 8.0 / 60.0


def _unpriced_app(**kwargs):
    app = PlannerApp(
        battery_capacity=CAPACITY,
        charge_rate=CHARGE_KW,
        discharge_rate=CHARGE_KW,
        efficiency=EFFICIENCY,
        slot_minutes=15,
        now=NOW,
        decision_log_level=0,
        **kwargs,
    )
    app.default_load_kw = 0.0
    app.default_pv_kw = 0.0
    return app


def _future_prices():
    return [
        PricePoint(time=SLOT_10_15, price=0.50),
        PricePoint(time=SLOT_10_30, price=0.50),
    ]


class TestTheUnpricedCurrentSlotIsResolvedBeforeSolving:
    """The DP must start from the SOC the retained action will actually leave."""

    def test_a_retained_charge_advances_the_starting_soc(self):
        app = _unpriced_app()
        retained = _entry(SLOT_10_00, BatteryMode.CHARGE)
        schedule = app.find_optimal_schedule(
            _future_prices(), 0, current_soc=START_SOC,
            previous_current_entry=retained,
        )
        assert SLOT_10_00 not in schedule  # unpriced: the planner does not own it

        gained = CHARGE_KW * REMAINING_H * EFFICIENCY / CAPACITY * 100
        expected = START_SOC + gained
        assert expected == pytest.approx(43.5664, abs=1e-3)
        assert app._last_dp_soc_trajectory[SLOT_10_15][0] == pytest.approx(
            expected, abs=1e-6
        )

    def test_a_retained_discharge_advances_the_other_way(self):
        app = _unpriced_app()
        app.default_load_kw = 2.0
        retained = _entry(SLOT_10_00, BatteryMode.DISCHARGE, export_rate=0)
        schedule = app.find_optimal_schedule(
            _future_prices(), 0, current_soc=START_SOC,
            previous_current_entry=retained,
        )
        assert SLOT_10_00 not in schedule
        drained = 2.0 * REMAINING_H / CAPACITY * 100
        expected = START_SOC - drained
        assert app._last_dp_soc_trajectory[SLOT_10_15][0] == pytest.approx(
            expected, abs=1e-6
        )
        assert app._last_dp_soc_trajectory[SLOT_10_15][0] < START_SOC

    def test_the_hold_fallback_still_absorbs_pv_surplus(self):
        """No retainable entry: HOLD -- which is not the same as "nothing"."""
        app = _unpriced_app()
        app.pv_by_slot = {SLOT_10_00: 3.0}
        unpriced_previous = _entry(SLOT_10_00, BatteryMode.CHARGE, priced=False)
        app.find_optimal_schedule(
            _future_prices(), 0, current_soc=START_SOC,
            previous_current_entry=unpriced_previous,
        )
        gained = 3.0 * REMAINING_H * EFFICIENCY / CAPACITY * 100
        assert app._last_dp_soc_trajectory[SLOT_10_15][0] == pytest.approx(
            START_SOC + gained, abs=1e-6
        )

    def test_no_previous_entry_holds_and_changes_nothing_without_pv(self):
        app = _unpriced_app()
        app.find_optimal_schedule(
            _future_prices(), 0, current_soc=START_SOC, previous_current_entry=None
        )
        assert app._last_dp_soc_trajectory[SLOT_10_15][0] == pytest.approx(
            START_SOC, abs=1e-9
        )

    def test_the_priced_path_keeps_its_partial_first_slot(self):
        """When the current interval IS priced, nothing about it changes."""
        app = _unpriced_app()
        prices = [PricePoint(time=SLOT_10_00, price=0.50)] + _future_prices()
        schedule = app.find_optimal_schedule(prices, 0, current_soc=START_SOC)
        assert SLOT_10_00 in schedule
        # The first slot starts at the measured SOC and is solved at the
        # remaining 8/15 of its width, exactly as before.
        assert app._last_dp_soc_trajectory[SLOT_10_00][0] == pytest.approx(
            START_SOC, abs=1e-9
        )
