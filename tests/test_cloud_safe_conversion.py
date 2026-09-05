"""Orchestrator-level regressions for the cloud-safe HOLD -> DISCHARGE hedge.

``BatteryOptimizer.find_optimal_schedule`` may rewrite the DP's HOLD actions
into ``discharge_to_load`` so that a cloud, rather than the grid, is covered by
the battery. The rewrite happens AFTER the DP has already chosen the rest of the
horizon on the assumption that those HOLD slots preserve energy, so it is only
sound where the DP's own model says the two actions are identical.

The reproduction below is the counter-example that motivated the restriction:
two slots, exact forecasts, no cloud anywhere. The unconditional
``pv > 0 and buy_price > wear`` override spent the battery in a 0.10 EUR/kWh
slot and left the 1.00 EUR/kWh slot to the grid.

The equivalence conditions the production code now requires are asserted class
by class below:

* ``pv >= load`` — with a net load the DISCHARGE transition drains the pack and
  the HOLD one does not (``soc_projection.project_slot_soc``), so the actions
  are not interchangeable at any price.
* no export revenue at risk — ``discharge_to_load`` pins the export limiter to
  0 % (``direct_control.expected_registers``), so PV surplus the plan expected
  to SELL would be curtailed instead.
* the avoided import must beat wear AND what the plan itself says the kWh is
  worth kept — the better of the horizon-end terminal rate and the best later
  DISCHARGE slot's own marginal value. Otherwise the hedge, when the cloud
  actually arrives, spends energy earmarked for a dearer slot.
"""

from __future__ import annotations

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
    DPOptimizerConfig,
    ScheduleFormatter,
    ScheduleFormatterConfig,
    TemperatureProjector,
)


BASE = datetime.datetime(2026, 6, 21, 10, 0)


class _NoPvForecast:
    """``find_optimal_schedule`` refreshes this before every run; nothing to do."""

    def refresh(self, force: bool = False) -> bool:
        return False


class _NoPrices:
    def get_prices_for_date(self, date, tz):
        return []


class HedgeOptimizer:
    """Mock BatteryOptimizer with fully controlled load/PV/price forecasts.

    Deliberately not shared with ``tests/test_algorithm.py``: those doubles
    predict load from a ``LoadProfile``, and every assertion here depends on an
    exactly known per-slot load and PV value.
    """

    def __init__(
        self,
        *,
        battery_capacity: float = 10.0,
        charge_rate: float = 4.0,
        discharge_rate: float = 1.0,
        efficiency: float = 1.0,
        inverter_efficiency: float = 1.0,
        min_soc: float = 10.0,
        max_soc: float = 100.0,
        grid_fee: float = 0.0,
        grid_export_fee: float = 0.0,
        export_rate_multiplier: float = 0.0,
        import_price_multiplier: float = 1.0,
        battery_wear_cost: float = 0.0,
        terminal_energy_value_eur_kwh: Optional[float] = 0.0,
        slot_minutes: int = 60,
        soc_step_percent: float = 1.0,
        battery_temp: Optional[float] = 20.0,
    ):
        self.config = BatteryOptimizerConfig(
            battery_capacity=battery_capacity,
            charge_rate=charge_rate,
            discharge_rate=discharge_rate,
            efficiency=efficiency,
            inverter_efficiency=inverter_efficiency,
            grid_fee=grid_fee,
            grid_export_fee=grid_export_fee,
            export_rate_multiplier=export_rate_multiplier,
            import_price_multiplier=import_price_multiplier,
            battery_wear_cost=battery_wear_cost,
            terminal_energy_value_eur_kwh=terminal_energy_value_eur_kwh,
            slot_minutes=slot_minutes,
            soc_step_percent=soc_step_percent,
            default_min_soc=min_soc,
            default_max_soc=max_soc,
            decision_log_level=0,
        )
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.battery_avg_cost = 0.05

        self._current_time = BASE
        self._battery_temp = battery_temp
        self._ambient_service = None

        # Per-slot forecasts, keyed by slot start.
        self.load_by_slot: Dict[datetime.datetime, float] = {}
        self.pv_by_slot: Dict[datetime.datetime, float] = {}
        self.default_load_kw = 0.0
        self.default_pv_kw = 0.0

        self.learning_engine = BatteryLearningEngine(
            battery_capacity_kwh=self.config.battery_capacity,
            nominal_charge_rate_kw=self.config.charge_rate,
            nominal_efficiency=self.config.efficiency,
        )

        self._last_min_charge_slots = 0
        self._last_charge_slots = []
        self._last_projected_costs = {}
        self._last_schedule_counts = None
        self._last_dp_soc_trajectory = {}
        self._last_dp_temp_trajectory = {}

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

    # --- AppDaemon / app surface ------------------------------------------
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

    def set_datetime(self, dt: datetime.datetime):
        self._current_time = dt

    def log(self, message: str, level: str = "INFO"):
        pass

    def _align_to_slot(self, dt: datetime.datetime) -> datetime.datetime:
        minutes = dt.hour * 60 + dt.minute
        slot_start = (minutes // self.config.slot_minutes) * self.config.slot_minutes
        return dt.replace(
            hour=slot_start // 60, minute=slot_start % 60, second=0, microsecond=0
        )

    def _get_local_timezone(self):
        return None

    def _get_battery_temp(self):
        return self._battery_temp

    def _predict_load_kw(self, dt: datetime.datetime) -> float:
        return self.load_by_slot.get(dt, self.default_load_kw)

    def _predict_pv_kw(self, dt: datetime.datetime) -> float:
        return self.pv_by_slot.get(dt, self.default_pv_kw)


import sys  # noqa: E402
from pathlib import Path  # noqa: E402

apps_dir = Path(__file__).parent.parent / "appdaemon" / "apps"
sys.path.insert(0, str(apps_dir))

from battery_optimizer import BatteryOptimizer  # noqa: E402

HedgeOptimizer.find_optimal_schedule = BatteryOptimizer.find_optimal_schedule
HedgeOptimizer.project_schedule_trajectory = BatteryOptimizer.project_schedule_trajectory
HedgeOptimizer._replay_schedule = BatteryOptimizer._replay_schedule
HedgeOptimizer._ensure_current_slot_price = BatteryOptimizer._ensure_current_slot_price
HedgeOptimizer._compute_slot_fractions = BatteryOptimizer._compute_slot_fractions


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def slot(i: int) -> datetime.datetime:
    return BASE + datetime.timedelta(hours=i)


def build(opt: HedgeOptimizer, rows: List[tuple]) -> List[PricePoint]:
    """rows = [(price, load_kw, pv_kw), ...] for consecutive hourly slots."""
    prices = []
    for i, (price, load_kw, pv_kw) in enumerate(rows):
        t = slot(i)
        opt.load_by_slot[t] = load_kw
        opt.pv_by_slot[t] = pv_kw
        prices.append(PricePoint(time=t, price=price))
    return prices


def replay_grid_cost(
    opt: HedgeOptimizer, schedule, rows: List[tuple], start_soc: float
) -> float:
    """Independent continuous replay of the FINAL plan (efficiencies 1.0).

    Written from the physics, not from the optimizer's arithmetic: PV serves
    load first, a self-consumption DISCHARGE covers what is left of the load
    from the battery down to min_soc, and the grid pays for the remainder.
    ``discharge_to_load`` clamps export to 0, so a converted slot earns nothing
    for its surplus.
    """
    cfg = opt.config
    capacity = cfg.battery_capacity
    energy = start_soc / 100.0 * capacity
    min_energy = opt.min_soc / 100.0 * capacity
    max_energy = opt.max_soc / 100.0 * capacity
    hours = cfg.slot_hours
    cost = 0.0

    for i, (price, load_kw, pv_kw) in enumerate(rows):
        entry = schedule.get(slot(i))
        if entry is None:
            continue
        buy = (price + cfg.grid_fee) * cfg.import_price_multiplier
        sell = max(0.0, price * cfg.export_rate_multiplier - cfg.grid_export_fee)
        net_load_kwh = max(0.0, load_kw - pv_kw) * hours
        surplus_kwh = max(0.0, pv_kw - load_kw) * hours
        exports_allowed = not (
            entry.mode == BatteryMode.DISCHARGE
            and (entry.export_rate is None or entry.export_rate == 0)
        )

        if entry.mode == BatteryMode.CHARGE:
            room = max(0.0, max_energy - energy)
            charged = min(cfg.charge_rate * hours, room)
            from_pv = min(charged, surplus_kwh)
            cost += buy * (charged - from_pv) + buy * net_load_kwh
            energy += charged
            surplus_kwh -= from_pv
        elif entry.mode == BatteryMode.DISCHARGE and (entry.export_rate or 0) == 0:
            served = min(net_load_kwh, cfg.discharge_rate * hours, energy - min_energy)
            served = max(0.0, served)
            energy -= served
            cost += buy * (net_load_kwh - served)
            stored = min(surplus_kwh, cfg.charge_rate * hours, max_energy - energy)
            energy += stored
            surplus_kwh -= stored
        else:  # HOLD (export handled below)
            cost += buy * net_load_kwh
            stored = min(surplus_kwh, cfg.charge_rate * hours, max_energy - energy)
            energy += stored
            surplus_kwh -= stored

        if exports_allowed:
            cost -= sell * surplus_kwh

    return cost


def modes(schedule) -> List[BatteryMode]:
    return [schedule[t].mode for t in sorted(schedule)]


# ---------------------------------------------------------------------------
# The brief's reproduction
# ---------------------------------------------------------------------------


class TestEconomicReservationSurvivesPostprocessing:
    """Two slots, exact forecasts, no cloud: the DP's HOLD must stand.

    | slot | price | load | PV |
    |------|-------|------|----|
    | 1    | 0.10  | 2 kW | 1 kW |
    | 2    | 1.00  | 1 kW | 0 kW |

    10 kWh pack, 10/100 % SOC limits, 20 % initial SOC (1 kWh usable),
    4 kW charge / 1 kW discharge, efficiencies 1.0, zero fees/wear/terminal
    value, export disabled.
    """

    ROWS = [(0.10, 2.0, 1.0), (1.00, 1.0, 0.0)]
    START_SOC = 20.0

    def _run(self):
        opt = HedgeOptimizer()
        prices = build(opt, self.ROWS)
        opt.set_datetime(BASE)
        schedule = opt.find_optimal_schedule(prices, 0, current_soc=self.START_SOC)
        return opt, schedule

    def test_the_cheap_pv_slot_stays_hold(self):
        _opt, schedule = self._run()
        assert modes(schedule) == [BatteryMode.HOLD, BatteryMode.DISCHARGE]

    def test_the_reported_trajectory_reserves_the_energy(self):
        opt, schedule = self._run()
        traj = opt._last_dp_soc_trajectory
        assert traj[slot(0)][0] == pytest.approx(20.0)
        assert traj[slot(0)][1] == pytest.approx(20.0)
        assert traj[slot(1)][1] == pytest.approx(10.0)

    def test_the_replayed_plan_costs_the_cheap_import(self):
        """Independent replay: 1 kWh at 0.10, not 1 kWh at 1.00."""
        opt, schedule = self._run()
        cost = replay_grid_cost(opt, schedule, self.ROWS, self.START_SOC)
        assert cost == pytest.approx(0.10)

    def test_no_slot_carries_the_cloud_safe_tag(self):
        _opt, schedule = self._run()
        assert not [e for e in schedule.values() if "[cloud-safe]" in e.reason]

    def test_the_published_plan_matches_an_exhaustive_search(self):
        """Small-horizon enumeration over the whole action space.

        Nine sequences, replayed continuously and priced independently of the
        DP's arithmetic. With a terminal value of zero there is no salvage term
        to add, so the cheapest replay IS the optimum of the modeled problem.
        The point is the comparison against the FINAL schedule: postprocessing
        is inside the loop being checked.
        """
        opt, schedule = self._run()
        published = replay_grid_cost(opt, schedule, self.ROWS, self.START_SOC)

        best = None
        for first in BatteryMode:
            for second in BatteryMode:
                plan = {
                    slot(0): ScheduleEntry(time=slot(0), mode=first, export_rate=0),
                    slot(1): ScheduleEntry(time=slot(1), mode=second, export_rate=0),
                }
                cost = replay_grid_cost(opt, plan, self.ROWS, self.START_SOC)
                best = cost if best is None else min(best, cost)

        assert best == pytest.approx(0.10)
        assert published == pytest.approx(best)


# ---------------------------------------------------------------------------
# PV vs load: the three regimes
# ---------------------------------------------------------------------------

# Two horizon shapes, because the two questions pull in opposite directions.
#
# SUNNY: three slots where PV covers the load and nothing follows. The DP has
# no discharge candidate (net load is zero) and no reason to charge, so every
# slot comes back HOLD with no later slot the energy is owed to. Whatever
# happens to them is the hedge's doing and nothing else's.
#
# SUNNY_THEN_PEAK: the same three slots followed by a 2.00 EUR/kWh evening
# slot, with one slot's worth of usable energy in the pack. This is the only
# shape in which the DP HOLDs a slot whose PV does NOT cover the load — and it
# is also, by construction, a plan whose energy is spoken for, so the hedge is
# refused twice over.
PEAK_SLOT = (2.00, 1.0, 0.0)


def sunny(pv_kw: float, load_kw: float = 1.0, slots: int = 3):
    return [(0.10, load_kw, pv_kw)] * slots


def sunny_then_peak(pv_kw: float, load_kw: float = 1.0):
    return sunny(pv_kw, load_kw) + [PEAK_SLOT]


def hedged(schedule):
    return [t for t in sorted(schedule) if "[cloud-safe]" in schedule[t].reason]


class TestPvVersusLoadRegimes:
    """A HOLD slot is convertible only when forecast PV covers forecast load."""

    def _sunny(self, pv_kw: float, soc: float = 50.0, **kw):
        opt = HedgeOptimizer(discharge_rate=1.0, **kw)
        rows = sunny(pv_kw)
        prices = build(opt, rows)
        opt.set_datetime(BASE)
        return opt, opt.find_optimal_schedule(prices, 0, current_soc=soc), rows

    def test_pv_below_load_is_not_converted(self):
        """0 < PV < load: DISCHARGE drains the pack, HOLD does not.

        The reproduction's regime. Refused before any price is looked at — the
        assertion on ``_cloud_safe_candidates`` pins that it is condition 1
        doing the work, not the value gate that also happens to apply here.
        """
        opt = HedgeOptimizer(discharge_rate=1.0)
        rows = sunny_then_peak(0.3)
        prices = build(opt, rows)
        opt.set_datetime(BASE)
        schedule = opt.find_optimal_schedule(prices, 0, current_soc=20.0)

        assert hedged(schedule) == []
        assert _cloud_safe_candidates(
            schedule, opt._predict_load_kw, opt._predict_pv_kw
        ) == []
        assert modes(schedule)[:3] == [BatteryMode.HOLD] * 3
        assert schedule[slot(3)].mode is BatteryMode.DISCHARGE
        # 3 x 0.7 kWh imported at 0.10, and nothing at 2.00.
        assert replay_grid_cost(opt, schedule, rows, 20.0) == pytest.approx(0.21)

    def test_pv_equal_to_load_is_converted(self):
        """PV == load: net load is 0, so neither action moves any energy."""
        _opt, schedule, _rows = self._sunny(pv_kw=1.0)

        assert hedged(schedule) == [slot(0), slot(1), slot(2)]
        assert all(schedule[t].export_rate == 0 for t in hedged(schedule))

    def test_pv_above_load_is_converted_when_the_surplus_is_stored(self):
        """PV > load, surplus below the charge rate, room in the pack."""
        _opt, schedule, _rows = self._sunny(pv_kw=2.0)

        assert hedged(schedule) == [slot(0), slot(1), slot(2)]

    def test_the_dps_own_discharge_slots_are_never_touched(self):
        opt = HedgeOptimizer(discharge_rate=1.0)
        prices = build(opt, sunny_then_peak(1.0))
        opt.set_datetime(BASE)
        schedule = opt.find_optimal_schedule(prices, 0, current_soc=20.0)

        assert schedule[slot(3)].mode is BatteryMode.DISCHARGE
        assert slot(3) not in hedged(schedule)


# ---------------------------------------------------------------------------
# Export equivalence
# ---------------------------------------------------------------------------


class TestExportRevenueIsNotCurtailed:
    """``discharge_to_load`` pins the export limiter to 0 %.

    A surplus the DP priced as export revenue would therefore be curtailed, so
    those slots stay HOLD however sunny they are. The two cases differ only in
    whether the surplus fits inside the charge rate.

    Sell price 0.10 - 0.09 = 0.01 EUR/kWh, and a terminal value of 0.05 so the
    DP neither dumps the pack for that penny nor grid-charges at 0.10. Both sit
    below the 0.10 avoided import, so only the export test can block the hedge.
    """

    SELLABLE = dict(
        export_rate_multiplier=1.0,
        grid_export_fee=0.09,
        terminal_energy_value_eur_kwh=0.05,
    )

    def _schedule(self, pv_kw: float, **kw):
        opt = HedgeOptimizer(discharge_rate=1.0, **dict(self.SELLABLE, **kw))
        prices = build(opt, sunny(pv_kw))
        opt.set_datetime(BASE)
        return opt, opt.find_optimal_schedule(prices, 0, current_soc=20.0)

    def test_a_surplus_beyond_the_charge_rate_blocks_the_hedge(self):
        """9 kW PV, 1 kW load, 4 kW charge rate: 4 kWh/slot would be sold."""
        _opt, schedule = self._schedule(pv_kw=9.0)
        assert hedged(schedule) == []
        assert all(m is BatteryMode.HOLD for m in modes(schedule))

    def test_a_surplus_the_pack_absorbs_is_still_hedged(self):
        """2 kW PV against a 1 kW load: the 1 kWh surplus never reaches the grid."""
        _opt, schedule = self._schedule(pv_kw=2.0)
        assert hedged(schedule) == [slot(0), slot(1), slot(2)]

    def test_an_unsellable_surplus_is_hedged_even_when_curtailed(self):
        """With no export remuneration, curtailing the surplus costs nothing."""
        _opt, schedule = self._schedule(
            pv_kw=9.0, export_rate_multiplier=0.0, grid_export_fee=0.0
        )
        assert hedged(schedule) == [slot(0), slot(1), slot(2)]


# ---------------------------------------------------------------------------
# Opportunity cost, not just wear
# ---------------------------------------------------------------------------


class TestTheHedgeMustBeatTheValueOfKeepingTheEnergy:
    ROWS = [(0.10, 1.0, 1.0)] * 3

    def _schedule(self, **kw):
        opt = HedgeOptimizer(discharge_rate=1.0, **kw)
        prices = build(opt, self.ROWS)
        opt.set_datetime(BASE)
        schedule = opt.find_optimal_schedule(prices, 0, current_soc=50.0)
        return opt, schedule

    def test_a_positive_terminal_value_above_the_slot_blocks_the_hedge(self):
        """Stored energy is worth 0.30/kWh; avoiding this import saves 0.10.

        Wear is zero here, so only the opportunity cost of the energy can stop
        the conversion.
        """
        _opt, schedule = self._schedule(terminal_energy_value_eur_kwh=0.30)
        assert hedged(schedule) == []
        assert schedule[slot(0)].mode is BatteryMode.HOLD

    def test_a_terminal_value_below_the_slot_allows_it(self):
        _opt, schedule = self._schedule(terminal_energy_value_eur_kwh=0.05)
        assert hedged(schedule) == [slot(0), slot(1), slot(2)]

    def test_wear_above_the_avoided_import_blocks_the_hedge(self):
        _opt, schedule = self._schedule(battery_wear_cost=0.15)
        assert hedged(schedule) == []
        assert all(m is BatteryMode.HOLD for m in modes(schedule))

    def _two_slot(self, first_price, second_price):
        """A convertible sunny slot followed by a DISCHARGE slot of its own.

        One slot's worth of usable energy, terminal value 0 — the reference
        installation's setting, and the one that makes the horizon-end salvage
        value useless as an opportunity cost.
        """
        opt = HedgeOptimizer(discharge_rate=1.0)
        rows = [(first_price, 1.0, 1.0), (second_price, 1.0, 0.0)]
        prices = build(opt, rows)
        opt.set_datetime(BASE)
        schedule = opt.find_optimal_schedule(prices, 0, current_soc=20.0)
        assert schedule[slot(1)].mode is BatteryMode.DISCHARGE
        return opt, schedule, rows

    def test_a_dearer_later_slot_blocks_the_hedge(self):
        """The hedge must not eat the energy the evening peak is planned on.

        Slot 1 is a sunny 0.10 EUR/kWh slot with PV == load, so converting it
        is free under the forecast — but if the cloud comes, the pack pays for
        0.10 of import that the plan had earmarked for a 1.00 EUR/kWh slot.
        Terminal value is 0, so only the later slot's own marginal value can
        price that kWh.
        """
        opt, schedule, rows = self._two_slot(0.10, 1.00)

        assert hedged(schedule) == []
        assert schedule[slot(0)].mode is BatteryMode.HOLD
        assert schedule[slot(1)].marginal_value_eur_kwh == pytest.approx(1.00)
        # The battery still serves the expensive slot: nothing was imported.
        assert replay_grid_cost(opt, schedule, rows, 20.0) == pytest.approx(0.0)

    def test_a_cheaper_later_slot_leaves_the_hedge_alone(self):
        """Same shape, prices swapped: spending the kWh here is the better use."""
        _opt, schedule, _rows = self._two_slot(0.30, 0.10)

        assert hedged(schedule) == [slot(0)]
        assert schedule[slot(0)].mode is BatteryMode.DISCHARGE
        assert schedule[slot(1)].marginal_value_eur_kwh == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# What a retained conversion must preserve
# ---------------------------------------------------------------------------


class TestARetainedConversionPreservesTheModel:
    ROWS = [(0.10, 1.0, 2.0)] * 4
    START_SOC = 50.0

    def _run(self):
        opt = HedgeOptimizer(discharge_rate=1.0)
        prices = build(opt, self.ROWS)
        opt.set_datetime(BASE)
        schedule = opt.find_optimal_schedule(prices, 0, current_soc=self.START_SOC)
        assert all(m is BatteryMode.DISCHARGE for m in modes(schedule))
        return opt, schedule

    def test_the_soc_transition_is_identical_to_the_hold_plan(self):
        """Same shared-model trajectory as the plan the DP actually scored."""
        opt, schedule = self._run()

        hold_plan = {}
        for t, entry in schedule.items():
            clone = ScheduleEntryClone(entry)
            clone.mode = BatteryMode.HOLD
            clone.export_rate = None
            hold_plan[t] = clone

        hold_soc, hold_temp = opt.project_schedule_trajectory(
            hold_plan,
            self.START_SOC,
            starting_temp=opt._get_battery_temp(),
            current_slot=BASE,
            minutes_into_slot=0.0,
        )
        assert opt._last_dp_soc_trajectory == hold_soc
        assert opt._last_dp_temp_trajectory == hold_temp

    def test_the_replayed_grid_cost_is_unchanged(self):
        opt, schedule = self._run()
        converted_cost = replay_grid_cost(opt, schedule, self.ROWS, self.START_SOC)

        hold_plan = {}
        for t, entry in schedule.items():
            clone = ScheduleEntryClone(entry)
            clone.mode = BatteryMode.HOLD
            clone.export_rate = None
            hold_plan[t] = clone
        hold_cost = replay_grid_cost(opt, hold_plan, self.ROWS, self.START_SOC)

        assert converted_cost == pytest.approx(hold_cost)

    def test_the_entry_is_labelled_as_a_discharge_slot(self):
        _opt, schedule = self._run()
        entry = schedule[slot(0)]
        assert entry.mode is BatteryMode.DISCHARGE
        assert entry.export_rate == 0
        assert "[cloud-safe]" in entry.reason
        # The reported value must not silently keep an unexplained HOLD label.
        assert entry.value_basis is not None
        assert "cloud-safe" in entry.value_basis

    def test_the_mode_census_counts_the_converted_slots(self):
        opt, schedule = self._run()
        counts = opt._last_schedule_counts
        assert counts.hold == 0
        assert counts.self_consume == len(schedule)
        assert counts.export == 0


# ---------------------------------------------------------------------------
# The equivalence predicates on their own
# ---------------------------------------------------------------------------
#
# The end-to-end cases above depend on which action the DP picks, which makes
# some regimes awkward to reach (a full pack in a sunny slot is usually an
# EXPORT decision, not a HOLD). These exercise the predicates directly, so the
# boundaries stay covered whatever the DP decides.

from battery_optimizer import (  # noqa: E402
    _best_later_discharge_value,
    _cloud_safe_candidates,
    _cloud_safe_hedge,
    _hold_sells_nothing,
)


def _entry(t, mode, reason="0.1000 EUR/kWh load~1.00kW pv~2.00kW"):
    return ScheduleEntry(
        time=t, mode=mode, reason=reason, marginal_value_eur_kwh=0.0,
        value_basis="kept",
    )


class TestCandidateSelection:
    def _candidates(self, mode, pv_kw, load_kw):
        schedule = {slot(0): _entry(slot(0), mode)}
        return _cloud_safe_candidates(
            schedule, lambda t: load_kw, lambda t: pv_kw
        )

    def test_hold_with_pv_above_load(self):
        assert self._candidates(BatteryMode.HOLD, 2.0, 1.0) == [slot(0)]

    def test_hold_with_pv_equal_to_load(self):
        assert self._candidates(BatteryMode.HOLD, 1.0, 1.0) == [slot(0)]

    def test_hold_with_pv_below_load(self):
        assert self._candidates(BatteryMode.HOLD, 0.9999, 1.0) == []

    def test_hold_without_pv_is_not_a_hedge(self):
        """No PV forecast to be wrong about; a discharge here is the DP's call."""
        assert self._candidates(BatteryMode.HOLD, 0.0, 0.0) == []

    def test_other_modes_are_never_touched(self):
        assert self._candidates(BatteryMode.CHARGE, 2.0, 1.0) == []
        assert self._candidates(BatteryMode.DISCHARGE, 2.0, 1.0) == []


class TestSurplusAbsorption:
    """One hour, 10 kWh pack: 1 kW of surplus is 1 kWh, i.e. 10 SOC points.

    The gate reads the PRE-HEDGE replay's own ``grid_export_ac_kwh``, produced
    with the charge-rate lookup pinned to the temperatures the DP priced the
    plan at. It used to infer absorption from ``project_schedule_trajectory``'s
    SOC span, which looks the rate up at the projector's own evolving
    temperature -- a different plan whenever the rate refinement falls back to
    its conservative idle profile.
    """

    @staticmethod
    def _replay(
        *, soc=50.0, pv_kw=2.0, load_kw=1.0, fraction=1.0, efficiency=1.0,
        charge_rate=4.0,
    ):
        from battery_optimizer_lib.plan_validation import replay_plan

        opt = HedgeOptimizer(efficiency=efficiency, charge_rate=charge_rate)
        cfg = DPOptimizerConfig.from_main_config(
            opt.config, min_soc=opt.min_soc, max_soc=opt.max_soc
        )
        schedule = {slot(0): _entry(slot(0), BatteryMode.HOLD)}
        return replay_plan(
            schedule=schedule,
            config=cfg,
            starting_soc=soc,
            predict_load_kw=lambda t: load_kw,
            predict_pv_kw=lambda t: pv_kw,
            charge_rate_for=lambda t, s, temp: charge_rate,
            current_slot=slot(0) if fraction < 1.0 else None,
            minutes_into_slot=(1.0 - fraction) * opt.config.slot_minutes,
        )

    def _check(self, **kwargs):
        return _hold_sells_nothing(
            slot_time=slot(0), pre_hedge_replay=self._replay(**kwargs)
        )

    def test_the_whole_surplus_is_stored(self):
        assert self._check() is True

    def test_a_full_pack_stores_nothing(self):
        """At max SOC the surplus can only be exported - HOLD is what allows it.

        The planning-side counterpart of the execution-time
        ``DISCHARGE -> HOLD at max SOC with PV > load`` override.
        """
        assert self._check(soc=100.0) is False

    def test_a_rate_below_the_surplus_leaves_something_to_sell(self):
        """0.25 kW of capability against 1 kW of surplus: 0.75 kWh is sold."""
        assert self._check(charge_rate=0.25) is False

    def test_a_rate_that_exactly_covers_the_surplus_sells_nothing(self):
        assert self._check(charge_rate=1.0) is True

    def test_no_surplus_is_trivially_equivalent(self):
        assert self._check(pv_kw=1.0, load_kw=1.0) is True

    def test_a_partial_slot_does_not_change_the_verdict(self):
        """Both sides are powers: shortening the slot scales them together."""
        assert self._check(fraction=0.5, charge_rate=1.0) is True
        assert self._check(fraction=0.5, charge_rate=0.9) is False
        assert self._check(fraction=1.0, charge_rate=1.0) is True
        assert self._check(fraction=1.0, charge_rate=0.9) is False

    def test_storage_loss_does_not_change_what_is_sold(self):
        """Retention shrinks what is STORED, not what the pack took in."""
        assert self._check(efficiency=0.8, charge_rate=1.0) is True
        assert self._check(efficiency=0.8, charge_rate=0.9) is False

    def test_an_unknown_slot_is_refused(self):
        from battery_optimizer_lib.plan_validation import PlanReplay

        assert _hold_sells_nothing(
            slot_time=slot(0), pre_hedge_replay=PlanReplay()
        ) is False
        assert _hold_sells_nothing(slot_time=slot(0), pre_hedge_replay=None) is False


class TestBestLaterDischargeValue:
    """The opportunity cost the value gate uses, read off the plan itself."""

    def _plan(self, spec):
        schedule = {}
        for i, (mode, value) in enumerate(spec):
            entry = _entry(slot(i), mode)
            entry.marginal_value_eur_kwh = value
            schedule[slot(i)] = entry
        return schedule

    def test_only_later_slots_count(self):
        plan = self._plan([
            (BatteryMode.DISCHARGE, 1.00),
            (BatteryMode.HOLD, 0.0),
            (BatteryMode.DISCHARGE, 0.40),
        ])
        best = _best_later_discharge_value(plan)

        assert best[slot(0)] == pytest.approx(0.40)
        assert best[slot(1)] == pytest.approx(0.40)
        assert best[slot(2)] == pytest.approx(0.0)

    def test_the_maximum_wins_not_the_next_one(self):
        plan = self._plan([
            (BatteryMode.HOLD, 0.0),
            (BatteryMode.DISCHARGE, 0.10),
            (BatteryMode.DISCHARGE, 1.00),
        ])
        assert _best_later_discharge_value(plan)[slot(0)] == pytest.approx(1.00)

    def test_charge_and_hold_slots_are_not_opportunities(self):
        plan = self._plan([
            (BatteryMode.HOLD, 0.0),
            (BatteryMode.CHARGE, -5.0),
            (BatteryMode.HOLD, 9.0),
        ])
        assert _best_later_discharge_value(plan)[slot(0)] == pytest.approx(0.0)

    def test_a_missing_value_counts_as_zero(self):
        plan = self._plan([(BatteryMode.HOLD, 0.0), (BatteryMode.DISCHARGE, None)])
        assert _best_later_discharge_value(plan)[slot(0)] == pytest.approx(0.0)

    def test_a_negative_value_never_drags_it_below_zero(self):
        plan = self._plan([(BatteryMode.HOLD, 0.0), (BatteryMode.DISCHARGE, -0.5)])
        assert _best_later_discharge_value(plan)[slot(0)] == pytest.approx(0.0)


class TestHedgeAppliedToAFullPack:
    """The case the DP rarely reaches as HOLD, driven through the helper."""

    def _hedge(self, soc, **cfg_kw):
        from battery_optimizer_lib.plan_validation import replay_plan

        opt = HedgeOptimizer(**cfg_kw)
        schedule = {slot(0): _entry(slot(0), BatteryMode.HOLD)}
        pre_hedge_replay = replay_plan(
            schedule=schedule,
            config=DPOptimizerConfig.from_main_config(
                opt.config, min_soc=opt.min_soc, max_soc=opt.max_soc
            ),
            starting_soc=soc,
            predict_load_kw=lambda t: 1.0,
            predict_pv_kw=lambda t: 2.0,
            charge_rate_for=lambda t, s, temp: opt.config.charge_rate,
        )
        converted = _cloud_safe_hedge(
            schedule,
            candidates=[slot(0)],
            config=opt.config,
            prices_by_slot={slot(0): 0.10},
            predict_load_kw=lambda t: 1.0,
            predict_pv_kw=lambda t: 2.0,
            slot_fractions_by_slot={slot(0): 1.0},
            pre_hedge_replay=pre_hedge_replay,
            terminal_rate=0.0,
        )
        return schedule[slot(0)], converted

    def test_a_full_pack_with_sellable_surplus_stays_hold(self):
        entry, converted = self._hedge(
            100.0, export_rate_multiplier=1.0, grid_export_fee=0.0
        )
        assert converted == []
        assert entry.mode is BatteryMode.HOLD
        assert entry.value_basis == "kept"

    def test_a_full_pack_with_worthless_surplus_is_converted(self):
        entry, converted = self._hedge(100.0, export_rate_multiplier=0.0)
        assert converted == [slot(0)]
        assert entry.mode is BatteryMode.DISCHARGE
        assert entry.export_rate == 0
        assert entry.value_basis == "kept (cloud-safe)"


class TestTheExportTestUsesThePlansOwnPricedExport:
    """The export gate must read the DP's numbers, not a re-projection.

    ``_hold_stores_all_pv_surplus`` judged "HOLD stored all the surplus" from
    ``project_schedule_trajectory``'s continuous SOC span -- a SECOND
    computation of the same quantity, and one that disagreed with the DP
    whenever the rate refinement fell back to a conservative profile. The
    projection then "absorbed" surplus the DP had booked as export revenue, and
    ``discharge_to_load`` pins the export limiter to 0 %, so the hedge silently
    curtailed a sale the schedule was chosen for.

    The gate reads the pre-hedge replay's own ``grid_export_ac_kwh``, produced
    by the same ``_replay_schedule`` construction ``_validate_final_plan`` uses.
    That is still the point: one number, one source. (Both now look the rate up
    at the temperature the replay reaches -- there is no planning-temperature
    pin left anywhere -- so the two would agree even if the gate went back to
    re-projecting. Reading the plan's own priced export is still the right
    contract, and it is what this class pins.)
    """

    @staticmethod
    def _replay_with_export(exported_kwh, absorbed_soc_span=(50.0, 60.0)):
        """A pre-hedge replay saying the slot sells ``exported_kwh``."""
        from battery_optimizer_lib.plan_validation import PlanReplay, SlotReplay

        replay = PlanReplay()
        replay.order = [slot(0)]
        replay.by_slot[slot(0)] = SlotReplay(
            time=slot(0),
            mode=BatteryMode.HOLD,
            is_export=False,
            fraction=1.0,
            soc_start=absorbed_soc_span[0],
            soc_end=absorbed_soc_span[1],
            energy_start_kwh=absorbed_soc_span[0] / 100.0 * 10.0,
            energy_end_kwh=absorbed_soc_span[1] / 100.0 * 10.0,
            stored_dc_in_kwh=1.0 - exported_kwh,
            stored_dc_out_kwh=0.0,
            charge_input_dc_kwh=1.0 - exported_kwh,
            grid_charge_ac_kwh=0.0,
            battery_ac_served_kwh=0.0,
            unmet_battery_ac_kwh=0.0,
            grid_import_ac_kwh=0.0,
            grid_export_ac_kwh=exported_kwh,
            value_eur=0.0,
        )
        return replay

    def _hedge(self, exported_kwh, **cfg_kw):
        opt = HedgeOptimizer(export_rate_multiplier=1.0, grid_export_fee=0.0, **cfg_kw)
        schedule = {slot(0): _entry(slot(0), BatteryMode.HOLD)}
        converted = _cloud_safe_hedge(
            schedule,
            candidates=[slot(0)],
            config=opt.config,
            prices_by_slot={slot(0): 0.20},
            predict_load_kw=lambda t: 1.0,
            predict_pv_kw=lambda t: 2.0,
            slot_fractions_by_slot={slot(0): 1.0},
            pre_hedge_replay=self._replay_with_export(exported_kwh),
            terminal_rate=0.0,
        )
        return schedule[slot(0)], converted

    def test_a_slot_the_plan_sells_from_is_not_converted(self):
        """The DP absorbed 0.25 kWh and priced 0.75 kWh of export revenue.

        A re-projection at a warmer temperature would have absorbed the whole
        1.0 kWh and called the slot equivalent to HOLD. It is not: converting
        it curtails the sale.
        """
        entry, converted = self._hedge(0.75)
        assert converted == []
        assert entry.mode is BatteryMode.HOLD
        assert entry.value_basis == "kept"

    def test_a_slot_with_nothing_to_sell_is_still_converted(self):
        """The gate must not degenerate into "never hedge"."""
        entry, converted = self._hedge(0.0)
        assert converted == [slot(0)]
        assert entry.mode is BatteryMode.DISCHARGE
        assert entry.export_rate == 0

    def test_an_unknown_slot_is_refused(self):
        from battery_optimizer_lib.plan_validation import PlanReplay

        opt = HedgeOptimizer(export_rate_multiplier=1.0, grid_export_fee=0.0)
        schedule = {slot(0): _entry(slot(0), BatteryMode.HOLD)}
        converted = _cloud_safe_hedge(
            schedule,
            candidates=[slot(0)],
            config=opt.config,
            prices_by_slot={slot(0): 0.20},
            predict_load_kw=lambda t: 1.0,
            predict_pv_kw=lambda t: 2.0,
            slot_fractions_by_slot={slot(0): 1.0},
            pre_hedge_replay=PlanReplay(),
            terminal_rate=0.0,
        )
        assert converted == []

    def test_a_worthless_surplus_is_converted_whatever_the_replay_says(self):
        """With no sale possible there is no export revenue to protect."""
        opt = HedgeOptimizer(export_rate_multiplier=0.0)
        schedule = {slot(0): _entry(slot(0), BatteryMode.HOLD)}
        converted = _cloud_safe_hedge(
            schedule,
            candidates=[slot(0)],
            config=opt.config,
            prices_by_slot={slot(0): 0.20},
            predict_load_kw=lambda t: 1.0,
            predict_pv_kw=lambda t: 2.0,
            slot_fractions_by_slot={slot(0): 1.0},
            pre_hedge_replay=self._replay_with_export(0.75),
            terminal_rate=0.0,
        )
        assert converted == [slot(0)]


class ScheduleEntryClone:
    """Shallow copy of a ScheduleEntry so a test can vary one field."""

    def __init__(self, entry):
        self.time = entry.time
        self.mode = entry.mode
        self.reason = entry.reason
        self.export_rate = entry.export_rate
        self.ac_charge_mode = entry.ac_charge_mode
        self.marginal_value_eur_kwh = entry.marginal_value_eur_kwh
        self.value_basis = entry.value_basis
