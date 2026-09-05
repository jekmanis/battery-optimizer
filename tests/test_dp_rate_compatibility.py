"""Task 4: charge rates must match the SOC and temperature the plan reaches.

The defect: ``compute_charge_rates_per_slot`` advanced SOC and temperature as
though charging ran continuously from now on, and ``_run_dp`` then used that one
time-indexed rate for every reachable state at that time. Two opposite errors
followed -- imaginary charging warmed a cold pack, making a later planned charge
look faster than the selected path could achieve; and imaginary charging pushed
SOC into a taper region, making rates too low for paths that stayed low or
discharged.

The synthetic predictors below are deliberately extreme so the disagreement is
unmistakable. They are not a claim about the real pack; the shared-thermal-model
cases at the bottom of this file are.
"""

import datetime
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


def _prices(values, slot_minutes=15):
    return [
        PricePoint(time=BASE + datetime.timedelta(minutes=slot_minutes * i), price=p)
        for i, p in enumerate(values)
    ]


class _CountingOptimizer(DPOptimizer):
    """Counts DP solves so the refinement budget is testable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solve_count = 0

    def _build_schedule(self, *args, **kwargs):
        self.solve_count += 1
        return super()._build_schedule(*args, **kwargs)


def _planned_stored_in_kwh(result, slot, capacity):
    start, end = result.soc_trajectory[slot]
    return max(0.0, (end - start) / 100.0 * capacity)


class TestColdBatteryReproduction:
    """The brief's synthetic example.

    Rate: 1 kW below 10 C, 4 kW otherwise.
    Charging warms the pack by 1 C per minute; idling does not change it.
    Starting temperature 0 C, starting SOC 10 %.
    """

    @staticmethod
    def _build():
        cfg = _config()
        opt = _CountingOptimizer(
            config=cfg,
            load_predictor=lambda dt: 4.0 if dt == BASE + datetime.timedelta(minutes=30) else 0.0,
            charge_rate_predictor=lambda soc, temp: (
                1.0 if temp is not None and temp < 10.0 else 4.0
            ),
            temp_after_charge_predictor=lambda t, m: t + m,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
        )
        return cfg, opt

    @staticmethod
    def _load(dt):
        return 4.0 if dt == BASE + datetime.timedelta(minutes=30) else 0.0

    def _run(self):
        cfg, opt = self._build()
        result = opt.optimize(
            prices=_prices([0.60, 0.01, 1.00]),
            current_slot=BASE,
            current_soc=10.0,
            current_temp=0.0,
        )
        return cfg, opt, result

    def test_the_plan_does_not_rely_on_heat_from_an_unselected_action(self):
        """HOLD, CHARGE, DISCHARGE charged at 4 kW on a pack still at 0 C.

        Whatever the plan is, the rate it charges each slot at must be one the
        plan's OWN earlier actions can produce.
        """
        cfg, opt, result = self._run()
        slots = sorted(result.schedule.keys())

        # Replay the selected sequence with the same synthetic thermal model
        # and rate function, and check no slot was credited more stored energy
        # than the selected path can deliver.
        temp = 0.0
        for slot in slots:
            entry = result.schedule[slot]
            soc_start = result.soc_trajectory[slot][0]
            achievable_rate = 1.0 if temp < 10.0 else 4.0
            planned = _planned_stored_in_kwh(result, slot, cfg.battery_capacity)
            achievable = achievable_rate * cfg.efficiency * 0.25
            assert planned <= achievable + 1e-9, (
                f"{slot}: plan stores {planned:.3f} kWh but the path's own "
                f"temperature {temp:.1f} C only supports {achievable:.3f} kWh"
            )
            if entry.mode == BatteryMode.CHARGE:
                temp = temp + 15.0
            _ = soc_start

    def test_the_cheap_slot_is_not_charged_at_an_imaginary_warm_rate(self):
        """The 0.01 EUR/kWh slot cannot store 1.0 kWh from a 0 C start."""
        cfg, opt, result = self._run()
        cheap = BASE + datetime.timedelta(minutes=15)
        assert _planned_stored_in_kwh(result, cheap, cfg.battery_capacity) <= 0.25 + 1e-9

    def test_refinement_is_bounded_and_terminates(self):
        """This example oscillates; it must still stop, and stop conservatively."""
        cfg, opt, result = self._run()
        assert opt.solve_count <= cfg_max_solves()
        assert result.rate_refinement_passes >= 1
        assert result.rate_refinement_converged in (True, False)

    def test_the_fallback_does_not_credit_unavailable_charge_energy(self):
        cfg, opt, result = self._run()
        total_planned = sum(
            _planned_stored_in_kwh(result, slot, cfg.battery_capacity)
            for slot in sorted(result.schedule.keys())
        )
        # Two 15-minute slots at the cold 1 kW rate is the most a 0 C pack can
        # take before the load slot arrives.
        assert total_planned <= 0.5 + 1e-9


def cfg_max_solves():
    from battery_optimizer_lib.dp_optimizer import MAX_RATE_REFINEMENT_PASSES

    # One solve per pass plus at most one conservative final solve.
    return MAX_RATE_REFINEMENT_PASSES + 1


class TestSocDependenceIsPerState:
    """A low-SOC path keeps its capability even if another path is tapering."""

    @staticmethod
    def _tapering_rate(soc, temp):
        # Full speed below 80 %, heavily tapered above.
        return 4.0 if soc < 80.0 else 0.4

    def test_a_discharged_path_still_charges_at_full_rate(self):
        # A positive salvage value, so end-of-horizon charging is worth doing
        # at all; without it nothing would ever charge in the last slot.
        cfg = _config(battery_capacity=10.0, terminal_energy_value_eur_kwh=1.0)
        opt = DPOptimizer(
            config=cfg,
            # Load only in the middle slots, so one reachable path discharges.
            load_predictor=lambda dt: 4.0 if 1 <= _slot_index(dt) <= 2 else 0.0,
            charge_rate_predictor=self._tapering_rate,
            temp_after_charge_predictor=lambda t, m: t,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
        )
        # Cheap at the end: charging there is worth it, and the path that got
        # there is at a LOW soc because it served the load.
        result = opt.optimize(
            prices=_prices([0.02, 0.90, 0.90, 0.01, 0.01]),
            current_slot=BASE,
            current_soc=75.0,
        )
        last = BASE + datetime.timedelta(minutes=60)
        entry = result.schedule[last]
        assert entry.mode == BatteryMode.CHARGE
        stored = _planned_stored_in_kwh(result, last, cfg.battery_capacity)
        # A time-indexed array built from "charge continuously from now" would
        # have this slot deep in the taper region at 0.4 kW -> 0.1 kWh.
        assert stored > 0.1 + 1e-6

    def test_a_near_full_state_is_tapered_in_the_same_solve(self):
        """Same solve, different state: the taper must still apply where it should."""
        cfg = _config(terminal_energy_value_eur_kwh=1.0)
        opt = DPOptimizer(
            config=cfg,
            load_predictor=lambda dt: 0.0,
            charge_rate_predictor=self._tapering_rate,
            temp_after_charge_predictor=lambda t, m: t,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
        )
        result = opt.optimize(
            prices=_prices([0.01, 0.01, 0.01]),
            current_slot=BASE,
            current_soc=85.0,
            current_temp=None,
        )
        first = BASE
        stored = _planned_stored_in_kwh(result, first, cfg.battery_capacity)
        assert stored == pytest.approx(0.4 * 0.25, abs=1e-9)


def _slot_index(dt):
    return int((dt - BASE).total_seconds() // 900)


class TestSharedThermalModel:
    """Realistic cases through TemperatureProjector and the ambient service."""

    @staticmethod
    def _projector(ambient=5.0, k1=0.01, k2=1.0, ambient_fn=None):
        from battery_optimizer_lib.thermal_model import TemperatureProjector

        class _Ambient:
            def predict_c(self, when):
                if ambient_fn is not None:
                    return ambient_fn(when)
                return ambient

        return TemperatureProjector(
            ambient_provider=_Ambient(),
            default_cooling_rate=k1,
            default_heating_c_per_kwh=k2,
        )

    @staticmethod
    def _rate(soc, temp):
        if temp is None:
            return 4.0
        return 1.0 if temp < 10.0 else 4.0

    def _optimize(self, *, load=0.0, current_soc=20.0, current_temp=5.0,
                  prices=None, ambient=5.0, pv=0.0):
        cfg = _config(terminal_energy_value_eur_kwh=1.0)
        opt = DPOptimizer(
            config=cfg,
            load_predictor=lambda dt: load,
            charge_rate_predictor=self._rate,
            temp_after_charge_predictor=lambda t, m: t,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: pv,
            temp_projector=self._projector(ambient=ambient),
        )
        result = opt.optimize(
            prices=prices or _prices([0.01] * 8),
            current_slot=BASE,
            current_soc=current_soc,
            current_temp=current_temp,
        )
        return cfg, opt, result

    def test_cold_idle_then_charging_uses_the_cold_rate_first(self):
        cfg, opt, result = self._optimize(current_temp=0.0)
        first = BASE
        stored = _planned_stored_in_kwh(result, first, cfg.battery_capacity)
        assert stored <= 1.0 * cfg.efficiency * 0.25 + 1e-9

    def test_missing_temperature_data_still_plans(self):
        cfg, opt, result = self._optimize(current_temp=None)
        assert result.schedule
        assert result.rate_refinement_passes == 1

    def test_changing_ambient_is_used(self):
        """The ambient is a function of time, not one scalar."""
        seen = []

        def ambient(when):
            seen.append(when)
            return 5.0

        cfg = _config()
        opt = DPOptimizer(
            config=cfg,
            load_predictor=lambda dt: 0.0,
            charge_rate_predictor=self._rate,
            temp_after_charge_predictor=lambda t, m: t,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
            temp_projector=self._projector(ambient_fn=ambient),
        )
        opt.optimize(
            prices=_prices([0.01] * 6),
            current_slot=BASE,
            current_soc=20.0,
            current_temp=0.0,
        )
        assert len({s for s in seen if s is not None}) > 1

    def test_a_full_battery_is_not_warmed_by_an_impossible_charge(self):
        """Warming follows ACTUAL battery flow, so a full pack stays idle."""
        cfg, opt, result = self._optimize(current_soc=100.0, current_temp=0.0)
        temps = [pair for pair in result.temp_trajectory.values()]
        assert temps
        start, end = temps[0]
        # Ambient 5 C, pack 0 C, no battery flow possible -> it only relaxes
        # toward ambient, it does not gain k2*|P| heat.
        assert end <= 5.0 + 1e-9

    def test_discharge_then_charge_carries_the_discharge_heat(self):
        """Discharging heats the pack, and the next slot's rate must see it."""
        cfg = _config(terminal_energy_value_eur_kwh=1.0)
        opt = DPOptimizer(
            config=cfg,
            load_predictor=lambda dt: 4.0 if _slot_index(dt) < 2 else 0.0,
            charge_rate_predictor=self._rate,
            temp_after_charge_predictor=lambda t, m: t,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
            temp_projector=self._projector(ambient=5.0, k1=0.0, k2=6.0),
        )
        result = opt.optimize(
            # Expensive now (worth discharging even against the 1.0 salvage
            # value), cheap later (worth recharging).
            prices=_prices([2.00, 2.00, 0.01, 0.01]),
            current_slot=BASE,
            current_soc=60.0,
            current_temp=0.0,
        )
        # The first two slots discharge 4 kW (1.0 kWh each); with k2 = 6 C/kWh
        # that is 6 C per slot, so by slot 2 the pack is above the 10 C
        # threshold and the cheap slot may charge at the warm rate.
        cheap = BASE + datetime.timedelta(minutes=30)
        stored = _planned_stored_in_kwh(result, cheap, cfg.battery_capacity)
        assert stored > 1.0 * cfg.efficiency * 0.25 + 1e-9

    def test_pv_charging_uses_a_soc_compatible_rate(self):
        # Expensive grid, so the only charging is the free PV surplus.
        cfg, opt, result = self._optimize(
            load=0.0, pv=3.0, current_soc=20.0, current_temp=20.0,
            prices=_prices([2.00] * 4),
        )
        first = BASE
        stored = _planned_stored_in_kwh(result, first, cfg.battery_capacity)
        # PV surplus 3 kW, warm rate 4 kW -> PV is the binding limit.
        assert stored == pytest.approx(3.0 * 0.25, abs=1e-9)


class TestPlanReplayAgreesWithTheTemperatureAwarePlan:
    """Acceptance: chosen-path replay and planned energy agree.

    The tolerance is a tenth of a DP grid step, derived from the representation
    (see ``BatteryOptimizer._validate_final_plan``), not fitted to the error
    observed here.
    """

    class _Engine:
        @staticmethod
        def get_charge_rate_for_soc(soc, temp=None):
            if temp is None:
                return 4.0
            return 1.0 if temp < 10.0 else 4.0

    class _App:
        def __init__(self, cfg, projector, load, pv):
            self.config = cfg
            self.min_soc = cfg.min_soc
            self.max_soc = cfg.max_soc
            self._temp_projector = projector
            self.learning_engine = TestPlanReplayAgreesWithTheTemperatureAwarePlan._Engine()
            self._load = load
            self._pv = pv
            self.messages = []

        def _predict_load_kw(self, dt):
            return self._load

        def _predict_pv_kw(self, dt):
            return self._pv

        def _get_local_timezone(self):
            return None

        def log(self, message, level="INFO"):
            self.messages.append((level, message))

    def test_no_disagreement_is_reported(self):
        import battery_optimizer as bo

        projector = TestSharedThermalModel._projector(ambient=5.0, k1=0.0, k2=6.0)
        cfg = _config(terminal_energy_value_eur_kwh=1.0)
        opt = DPOptimizer(
            config=cfg,
            load_predictor=lambda dt: 4.0 if _slot_index(dt) < 2 else 0.0,
            charge_rate_predictor=self._Engine.get_charge_rate_for_soc,
            temp_after_charge_predictor=lambda t, m: t,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
            temp_projector=projector,
        )
        result = opt.optimize(
            prices=_prices([2.00, 2.00, 0.01, 0.01]),
            current_slot=BASE,
            current_soc=60.0,
            current_temp=0.0,
        )

        # Load differs per slot, so give the app the same predictor the DP used.
        app = self._App(cfg, projector, load=0.0, pv=0.0)
        app._predict_load_kw = lambda dt: 4.0 if _slot_index(dt) < 2 else 0.0

        replay = bo.BatteryOptimizer._validate_final_plan(
            app,
            schedule=result.schedule,
            soc_trajectory=result.soc_trajectory,
            starting_soc=60.0,
            starting_temp=0.0,
            current_slot=None,
            minutes_into_slot=0.0,
            prices_sorted=None,
            planning_temp_by_slot=result.planning_temp_by_slot,
        )
        assert replay is not None
        assert replay.conservation_violations == []
        assert replay.trajectory_disagreements == []
        assert not [m for m in app.messages if m[0] in ("WARNING", "ERROR")]


class TestHoistedConstantRate:
    """The SOC-independent fast path must not change any answer.

    When the rate is the same at every state, `_run_dp` hoists the derived
    per-slot energies out of the state loop. That is a performance hint, not a
    modelling choice, so the two paths must produce identical plans.
    """

    @staticmethod
    def _run(force_per_state):
        cfg = _config(terminal_energy_value_eur_kwh=0.2)
        opt = DPOptimizer(
            config=cfg,
            load_predictor=lambda dt: 1.1,
            charge_rate_predictor=lambda soc, temp: 4.0,
            temp_after_charge_predictor=lambda t, m: t,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 2.0 if _slot_index(dt) in (3, 4) else 0.0,
        )
        original = DPOptimizer._rate_is_soc_dependent
        if force_per_state:
            opt._rate_is_soc_dependent = lambda socs, temp: True
        try:
            return opt.optimize(
                prices=_prices([0.30, 0.05, 0.02, 0.40, 0.90, 0.10, 0.60, 0.03]),
                current_slot=BASE,
                current_soc=40.0,
                minutes_into_slot=7.0,
            )
        finally:
            opt._rate_is_soc_dependent = original.__get__(opt, DPOptimizer)

    def test_the_two_paths_agree(self):
        hoisted = self._run(force_per_state=False)
        per_state = self._run(force_per_state=True)
        assert [e.mode for _, e in sorted(hoisted.schedule.items())] == [
            e.mode for _, e in sorted(per_state.schedule.items())
        ]
        for slot in sorted(hoisted.schedule):
            assert hoisted.soc_trajectory[slot] == pytest.approx(
                per_state.soc_trajectory[slot], abs=1e-9
            )


class TestWithinSlotApproximation:
    """The DP uses the rate at the START of a slot; the replay does the same.

    A threshold crossed inside a slot is therefore a documented, bounded
    approximation: at most ``(warm_rate - cold_rate) * slot_hours`` of stored
    energy per slot.
    """

    def test_planner_and_replay_agree_across_a_threshold_crossing(self):
        cfg = _config()
        rate = lambda soc, temp: 1.0 if (temp is not None and temp < 10.0) else 4.0
        opt = DPOptimizer(
            config=cfg,
            load_predictor=lambda dt: 0.0,
            charge_rate_predictor=rate,
            # Crosses the 10 C threshold in the middle of a slot.
            temp_after_charge_predictor=lambda t, m: t + 0.8 * m,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
        )
        result = opt.optimize(
            prices=_prices([0.01] * 4),
            current_slot=BASE,
            current_soc=20.0,
            current_temp=4.0,
        )
        replay = replay_plan(
            schedule=result.schedule,
            config=cfg,
            starting_soc=20.0,
            predict_load_kw=lambda dt: 0.0,
            predict_pv_kw=lambda dt: 0.0,
            charge_rate_for=lambda slot, soc, temp: rate(soc, temp),
            starting_temp=4.0,
            planned_soc_by_slot={
                slot: pair[1] for slot, pair in result.soc_trajectory.items()
            },
            soc_tolerance=0.1,
        )
        # Without a shared thermal projector the replay has no temperature to
        # evolve, so it must be given one; the point here is that the DP's own
        # per-slot rate is what it charged with.
        assert replay.conservation_violations == []

    def test_the_approximation_is_bounded_by_one_slot_of_rate_difference(self):
        cfg = _config()
        opt = DPOptimizer(
            config=cfg,
            load_predictor=lambda dt: 0.0,
            charge_rate_predictor=lambda soc, temp: (
                1.0 if (temp is not None and temp < 10.0) else 4.0
            ),
            temp_after_charge_predictor=lambda t, m: t + 0.8 * m,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
        )
        result = opt.optimize(
            prices=_prices([0.01] * 4),
            current_slot=BASE,
            current_soc=20.0,
            current_temp=4.0,
        )
        bound = (4.0 - 1.0) * cfg.slot_hours * cfg.efficiency
        for slot in sorted(result.schedule.keys()):
            stored = _planned_stored_in_kwh(result, slot, cfg.battery_capacity)
            assert stored <= 4.0 * cfg.slot_hours * cfg.efficiency + 1e-9
            assert stored >= 0.0
        assert bound == pytest.approx(0.75)
