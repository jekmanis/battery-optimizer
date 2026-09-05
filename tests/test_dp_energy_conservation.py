"""Task 3: the DP must not create available energy by SOC rounding.

The defect: ``_discharge_index`` rounded a post-discharge energy to the NEAREST
grid state and only fell back to conservative rounding when the index would not
move. With a 1 % grid on a 10 kWh pack (0.10 kWh) and a 0.14 kWh per-slot
discharge, every slot deducted 0.10 kWh instead of 0.14 kWh -- a 0.04 kWh error
of the SAME SIGN in every slot. Twenty slots credited 2.8 kWh of battery service
from a battery holding 2.0 kWh.

"Zero-mean quantization error" is a property of a random signal, not of a
constant load repeated on a constant slot length.

The checks here are deliberately independent of the DP's own arithmetic: the
selected schedule is replayed continuously through ``slot_energy.simulate_slot``
and the prefix conservation inequality is evaluated on the replay, BEFORE any
SOC clamping.
"""

import datetime
import itertools
from typing import Optional

import pytest

from battery_optimizer_lib import (
    BatteryMode,
    DPOptimizer,
    DPOptimizerConfig,
    PricePoint,
)
from battery_optimizer_lib.plan_validation import replay_plan


BASE = datetime.datetime(2026, 3, 2, 0, 0, 0)


def _prices(n, price=1.00, slot_minutes=15):
    return [
        PricePoint(time=BASE + datetime.timedelta(minutes=slot_minutes * i), price=price)
        for i in range(n)
    ]


def _optimizer(
    config,
    load_kw=0.0,
    charge_rate_kw=0.0,
    pv_kw=0.0,
    load_fn=None,
    pv_fn=None,
):
    return DPOptimizer(
        config=config,
        load_predictor=load_fn or (lambda dt: load_kw),
        charge_rate_predictor=lambda soc, temp: charge_rate_kw,
        temp_after_charge_predictor=lambda t, m: t,
        temp_after_idle_predictor=lambda t, m: t,
        pv_predictor=pv_fn or (lambda dt: pv_kw),
    )


def _config(**kwargs):
    base = dict(
        battery_capacity=10.0,
        min_soc=10.0,
        max_soc=100.0,
        efficiency=1.0,
        discharge_rate=4.0,
        slot_minutes=15,
        soc_step_percent=1.0,
        grid_fee=0.0,
        battery_wear_cost=0.0,
        grid_export_fee=0.0,
        export_rate_multiplier=0.0,
        inverter_efficiency=1.0,
        import_price_multiplier=1.0,
        terminal_energy_value_eur_kwh=0.0,
    )
    base.update(kwargs)
    return DPOptimizerConfig(**base)


class TestTwentySlotReproduction:
    """The brief's reproduction, checked against a continuous replay."""

    @staticmethod
    def _run(soc_step_percent=1.0, initial_soc=30.0):
        cfg = _config(soc_step_percent=soc_step_percent)
        opt = _optimizer(cfg, load_kw=0.56)
        result = opt.optimize(
            prices=_prices(20),
            current_slot=BASE,
            current_soc=initial_soc,
        )
        replay = replay_plan(
            schedule=result.schedule,
            config=cfg,
            starting_soc=initial_soc,
            predict_load_kw=lambda dt: 0.56,
            predict_pv_kw=lambda dt: 0.0,
            charge_rate_for=lambda dt, soc, temp: 0.0,
        )
        return cfg, result, replay

    def test_credited_battery_service_fits_in_the_battery(self):
        """20 slots x 0.14 kWh = 2.8 kWh of load, 2.0 kWh of usable energy."""
        cfg, result, replay = self._run()
        usable = (30.0 - cfg.min_soc) / 100 * cfg.battery_capacity
        assert usable == pytest.approx(2.0)
        assert replay.total_battery_ac_served_kwh <= usable + 1e-9
        # Only the FINAL discharge may run the pack dry mid-slot ("until
        # depleted"). Six energy-limited slots means the plan kept ordering
        # service from an empty battery.
        limited = [s for s in replay.by_slot.values() if s.unmet_battery_ac_kwh > 1e-9]
        assert len(limited) <= 1, [str(s.time) for s in limited]

    def test_planned_grid_import_covers_the_unmet_demand(self):
        cfg, result, replay = self._run()
        total_load = 20 * 0.56 * 0.25
        assert total_load == pytest.approx(2.8)
        assert (
            replay.total_battery_ac_served_kwh + replay.total_grid_import_ac_kwh
            == pytest.approx(total_load, abs=1e-9)
        )
        assert replay.total_grid_import_ac_kwh >= 0.8 - 1e-9

    def test_prefix_conservation_holds_at_every_slot(self):
        cfg, result, replay = self._run()
        assert replay.conservation_violations == []

    def test_the_dp_trajectory_agrees_with_the_replay(self):
        """The reported SOC after 15 slots was 15 % while the pack was empty."""
        cfg, result, replay = self._run()
        slots = sorted(result.schedule.keys())
        for slot in slots:
            planned_start, planned_end = result.soc_trajectory[slot]
            replayed = replay.by_slot[slot]
            assert planned_end == pytest.approx(replayed.soc_end, abs=1e-6), slot

    def test_the_battery_is_empty_after_about_fourteen_slots(self):
        """2.0 kWh / 0.14 kWh = 14.28 slots. It used to last all 20."""
        cfg, result, replay = self._run()
        discharging = [
            slot
            for slot, entry in sorted(result.schedule.items())
            if entry.mode == BatteryMode.DISCHARGE
        ]
        # At most one extra slot for the partial final discharge.
        assert len(discharging) <= 15
        assert len(discharging) >= 14


class TestPrefixConservationAcrossConditions:
    """The inequality must hold for every starting point and every load."""

    CASES = list(
        itertools.product(
            [10.0, 10.5, 20.0, 30.0, 33.7, 99.9, 100.0],  # starting SOC
            [0.0, 0.56, 1.0, 4.0, 9.0],                   # load kW
            [0.0, 2.0, 6.0],                              # PV kW
            [0.0, 45.0],                                  # minutes into slot
        )
    )

    @pytest.mark.parametrize("soc,load,pv,minutes_in", CASES)
    def test_no_prefix_creates_energy(self, soc, load, pv, minutes_in):
        cfg = _config(
            efficiency=0.85,
            inverter_efficiency=0.97,
            export_rate_multiplier=1.0,
            grid_export_fee=0.01,
            export_discharge_rate=5.0,
            battery_wear_cost=0.01,
        )
        opt = _optimizer(cfg, load_kw=load, pv_kw=pv, charge_rate_kw=4.0)
        prices = [
            PricePoint(time=BASE + datetime.timedelta(minutes=15 * i), price=p)
            for i, p in enumerate([0.30, 0.05, 0.02, 0.40, 0.90, 0.10, 0.60, 0.03])
        ]
        result = opt.optimize(
            prices=prices,
            current_slot=BASE,
            current_soc=soc,
            minutes_into_slot=minutes_in,
        )
        replay = replay_plan(
            schedule=result.schedule,
            config=cfg,
            starting_soc=soc,
            predict_load_kw=lambda dt: load,
            predict_pv_kw=lambda dt: pv,
            charge_rate_for=lambda dt, s, t: 4.0,
            current_slot=BASE,
            minutes_into_slot=minutes_in,
            # This is the DP-vs-simulate_slot parity proof: the DP keeps an
            # inlined transition fused with the value recursion, and the only
            # thing standing between the two implementations is this sweep.
            planned_soc_by_slot={
                slot: pair[1] for slot, pair in result.soc_trajectory.items()
            },
            soc_tolerance=1e-6,
        )
        assert replay.conservation_violations == []
        assert replay.trajectory_disagreements == []
        # AC energy served after inverter loss, not just raw DC totals.
        assert (
            replay.total_battery_ac_served_kwh
            <= replay.max_battery_ac_available_kwh + 1e-9
        )

    @pytest.mark.parametrize(
        "soc",
        # Immediately on either side of a 1 % grid boundary (0.10 kWh steps).
        [19.999, 20.0, 20.001, 20.499, 20.5, 20.501],
    )
    def test_starting_soc_around_a_rounding_boundary(self, soc):
        cfg = _config()
        opt = _optimizer(cfg, load_kw=0.56)
        result = opt.optimize(prices=_prices(20), current_slot=BASE, current_soc=soc)
        replay = replay_plan(
            schedule=result.schedule,
            config=cfg,
            starting_soc=soc,
            predict_load_kw=lambda dt: 0.56,
            predict_pv_kw=lambda dt: 0.0,
            charge_rate_for=lambda dt, s, t: 0.0,
        )
        assert replay.conservation_violations == []
        usable = (soc - cfg.min_soc) / 100 * cfg.battery_capacity
        assert replay.total_battery_ac_served_kwh <= usable + 1e-9


class TestMetamorphic:
    """Less energy cannot buy more credited battery service."""

    @staticmethod
    def _served(initial_soc):
        cfg = _config()
        opt = _optimizer(cfg, load_kw=0.56)
        result = opt.optimize(
            prices=_prices(20), current_slot=BASE, current_soc=initial_soc
        )
        replay = replay_plan(
            schedule=result.schedule,
            config=cfg,
            starting_soc=initial_soc,
            predict_load_kw=lambda dt: 0.56,
            predict_pv_kw=lambda dt: 0.0,
            charge_rate_for=lambda dt, s, t: 0.0,
        )
        return replay.total_battery_ac_served_kwh

    def test_service_is_monotone_in_starting_energy(self):
        served = [self._served(soc) for soc in (12.0, 20.0, 30.0, 50.0, 80.0)]
        for lower, higher in zip(served, served[1:]):
            assert higher >= lower - 1e-9


class TestExhaustiveEnumeration:
    """Small horizon: compare the DP against every feasible action sequence.

    This separates a wrong recurrence from quantization error. The enumeration
    evaluates each action sequence with the SHARED continuous model, so the DP's
    value may fall short of the enumerated optimum by the quantization gap --
    but it must never EXCEED it, which would mean it scored a plan that cannot
    physically happen.
    """

    @staticmethod
    def _enumerate(cfg, prices, load, pv, starting_soc):
        best = None
        modes = [
            (BatteryMode.HOLD, None),
            (BatteryMode.CHARGE, None),
            (BatteryMode.DISCHARGE, 0),
        ]
        for combo in itertools.product(modes, repeat=len(prices)):
            schedule = {}
            for point, (mode, export_rate) in zip(prices, combo):
                from battery_optimizer_lib import ScheduleEntry

                entry = ScheduleEntry(time=point.time, mode=mode, reason="")
                entry.export_rate = export_rate
                schedule[point.time] = entry
            replay = replay_plan(
                schedule=schedule,
                config=cfg,
                starting_soc=starting_soc,
                predict_load_kw=lambda dt: load,
                predict_pv_kw=lambda dt: pv,
                charge_rate_for=lambda dt, s, t: 4.0,
                prices_by_slot={p.time: p.price for p in prices},
            )
            if best is None or replay.total_value_eur > best:
                best = replay.total_value_eur
        return best

    def test_dp_value_never_exceeds_the_enumerated_optimum(self):
        cfg = _config(
            efficiency=0.85,
            inverter_efficiency=0.97,
            battery_wear_cost=0.01,
            soc_step_percent=1.0,
        )
        prices = [
            PricePoint(time=BASE + datetime.timedelta(minutes=15 * i), price=p)
            for i, p in enumerate([0.10, 0.02, 0.50, 0.30, 0.80])
        ]
        opt = _optimizer(cfg, load_kw=1.2, charge_rate_kw=4.0)
        result = opt.optimize(prices=prices, current_slot=BASE, current_soc=40.0)
        replay = replay_plan(
            schedule=result.schedule,
            config=cfg,
            starting_soc=40.0,
            predict_load_kw=lambda dt: 1.2,
            predict_pv_kw=lambda dt: 0.0,
            charge_rate_for=lambda dt, s, t: 4.0,
            prices_by_slot={p.time: p.price for p in prices},
        )
        best = self._enumerate(cfg, prices, 1.2, 0.0, 40.0)
        assert replay.total_value_eur <= best + 1e-9
        # And the quantization gap must be small, not an excuse for a bad plan.
        assert replay.total_value_eur >= best - 0.02


class TestTheConservationCheckIsFalsifiable:
    """The prefix check must be able to FAIL, not restate the replay.

    It used to accumulate `simulate_slot`'s own clamped outputs and compare
    them with the bound those same outputs are constructed to satisfy -- an
    identity. Injecting the brief's exact pre-fix defect (a planner deducting
    1 % per slot for a 1.4 %-per-slot load) produced zero violations.

    What makes it falsifiable is the plan's own declaration: a slot the replay
    cannot serve in full is either one the plan DECLARED energy-limited
    ("until depleted", priced with the grid covering the rest), or the plan
    credited the battery with service it does not have.
    """

    @staticmethod
    def _defective_plan():
        """20 DISCHARGE slots, none declared limited -- the pre-fix DP."""
        from battery_optimizer_lib import ScheduleEntry

        cfg = _config()
        schedule = {}
        planned = {}
        soc = 30.0
        for i in range(20):
            slot = BASE + datetime.timedelta(minutes=15 * i)
            entry = ScheduleEntry(time=slot, mode=BatteryMode.DISCHARGE, reason="")
            entry.export_rate = 0
            schedule[slot] = entry
            # 1 % per slot: what nearest-rounding on a 0.10 kWh grid deducted
            # for a 0.14 kWh load.
            soc = max(cfg.min_soc, soc - 1.0)
            planned[slot] = soc
        return cfg, schedule, planned

    @staticmethod
    def _replay(cfg, schedule, planned):
        return replay_plan(
            schedule=schedule,
            config=cfg,
            starting_soc=30.0,
            predict_load_kw=lambda dt: 0.56,
            predict_pv_kw=lambda dt: 0.0,
            charge_rate_for=lambda dt, soc, temp: 0.0,
            planned_soc_by_slot=planned,
            soc_tolerance=1e-6,
        )

    def test_the_injected_defect_violates_conservation(self):
        cfg, schedule, planned = self._defective_plan()
        replay = self._replay(cfg, schedule, planned)
        assert replay.conservation_violations, (
            "a plan that credits 2.8 kWh of battery service from a 2.0 kWh "
            "battery must violate the conservation check"
        )
        assert not replay.ok

    def test_a_declared_partial_slot_is_not_a_violation(self):
        """Running the pack dry mid-slot is legitimate WHEN the plan says so."""
        from battery_optimizer_lib import ScheduleEntry

        cfg = _config()
        schedule = {}
        planned = {}
        energy = 3.0
        for i in range(20):
            slot = BASE + datetime.timedelta(minutes=15 * i)
            available = max(0.0, energy - 1.0)
            draw = min(0.14, available)
            mode = BatteryMode.DISCHARGE if draw > 1e-9 else BatteryMode.HOLD
            entry = ScheduleEntry(time=slot, mode=mode, reason="")
            entry.export_rate = 0 if mode == BatteryMode.DISCHARGE else None
            entry.energy_limited = draw < 0.14 - 1e-9 and mode == BatteryMode.DISCHARGE
            schedule[slot] = entry
            energy -= draw
            planned[slot] = energy / cfg.battery_capacity * 100
        replay = self._replay(cfg, schedule, planned)
        assert replay.conservation_violations == []
        assert replay.trajectory_disagreements == []

    def test_the_real_dp_plan_declares_every_limited_slot(self):
        cfg = _config()
        opt = _optimizer(cfg, load_kw=0.56)
        result = opt.optimize(
            prices=_prices(20), current_slot=BASE, current_soc=30.0
        )
        replay = replay_plan(
            schedule=result.schedule,
            config=cfg,
            starting_soc=30.0,
            predict_load_kw=lambda dt: 0.56,
            predict_pv_kw=lambda dt: 0.0,
            charge_rate_for=lambda dt, soc, temp: 0.0,
            planned_soc_by_slot={
                slot: pair[1] for slot, pair in result.soc_trajectory.items()
            },
            soc_tolerance=1e-6,
        )
        assert replay.conservation_violations == []
        assert replay.trajectory_disagreements == []


class _StubApp:
    """Minimal surface `_validate_final_plan` needs, nothing more."""

    def __init__(self, cfg, load=0.56, pv=0.0, rate=0.0):
        from battery_optimizer_lib import BatteryLearningEngine

        self.config = cfg
        self.min_soc = cfg.min_soc
        self.max_soc = cfg.max_soc
        self._temp_projector = None
        self._load = load
        self._pv = pv
        self.messages = []
        self.learning_engine = BatteryLearningEngine(
            battery_capacity_kwh=cfg.battery_capacity,
            nominal_charge_rate_kw=rate,
            nominal_efficiency=cfg.efficiency,
            log_func=lambda *a, **k: None,
        )

    def _predict_load_kw(self, dt):
        return self._load

    def _predict_pv_kw(self, dt):
        return self._pv

    def _get_local_timezone(self):
        return None

    def log(self, message, level="INFO"):
        self.messages.append((level, message))


class TestFinalPlanValidationHook:
    """`find_optimal_schedule` publishes only what the shared model confirms."""

    @staticmethod
    def _validate(app, schedule, soc_trajectory, starting_soc):
        import battery_optimizer as bo

        return bo.BatteryOptimizer._validate_final_plan(
            app,
            schedule=schedule,
            soc_trajectory=soc_trajectory,
            starting_soc=starting_soc,
            starting_temp=None,
            current_slot=None,
            minutes_into_slot=0.0,
            prices_sorted=None,
        )

    @staticmethod
    def _plan():
        cfg = _config()
        opt = _optimizer(cfg, load_kw=0.56)
        result = opt.optimize(
            prices=_prices(20), current_slot=BASE, current_soc=30.0
        )
        return cfg, result

    def test_a_consistent_plan_reports_nothing(self):
        cfg, result = self._plan()
        app = _StubApp(cfg)
        replay = self._validate(app, result.schedule, result.soc_trajectory, 30.0)
        assert replay is not None
        assert replay.conservation_violations == []
        assert replay.trajectory_disagreements == []
        assert not [m for m in app.messages if m[0] in ("WARNING", "ERROR")]

    def test_a_trajectory_that_overstates_the_soc_is_reported(self):
        """The published 15 %-after-15-slots case, injected deliberately."""
        cfg, result = self._plan()
        app = _StubApp(cfg)
        inflated = {
            slot: (pair[0], pair[1] + 5.0)
            for slot, pair in result.soc_trajectory.items()
        }
        replay = self._validate(app, result.schedule, inflated, 30.0)
        assert replay.trajectory_disagreements
        warnings = [m for m in app.messages if m[0] == "WARNING"]
        assert warnings and "disagrees with the shared" in warnings[0][1]

    def test_the_hook_never_breaks_planning(self):
        cfg, result = self._plan()
        app = _StubApp(cfg)
        app._predict_load_kw = lambda dt: (_ for _ in ()).throw(RuntimeError("boom"))
        assert self._validate(app, result.schedule, result.soc_trajectory, 30.0) is None
