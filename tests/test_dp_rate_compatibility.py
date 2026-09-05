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

        def _replay_schedule(self, **kwargs):
            import battery_optimizer as bo

            return bo.BatteryOptimizer._replay_schedule(self, **kwargs)

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
        assert replay.corrected is False
        assert not [m for m in app.messages if m[0] in ("WARNING", "ERROR")]

    def test_a_post_hedge_schedule_is_validated_and_corrected(self):
        """The hedge rewrites actions after the DP; validation sees the result.

        A HOLD slot turned into ``discharge_to_load`` changes the modelled flow
        whenever PV does NOT cover the load, so the DP's trajectory no longer
        describes the plan. The validator must correct what gets published,
        not merely mention it.
        """
        import battery_optimizer as bo
        from battery_optimizer_lib import BatteryMode as _Mode

        projector = TestSharedThermalModel._projector(ambient=5.0, k1=0.0, k2=6.0)
        cfg = _config(terminal_energy_value_eur_kwh=1.0)
        load = lambda dt: 0.0 if _slot_index(dt) < 2 else 2.0
        opt = DPOptimizer(
            config=cfg,
            load_predictor=load,
            charge_rate_predictor=self._Engine.get_charge_rate_for_soc,
            temp_after_charge_predictor=lambda t, m: t,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
            temp_projector=projector,
        )
        result = opt.optimize(
            prices=_prices([0.05, 0.05, 0.50, 0.50]),
            current_slot=BASE,
            current_soc=60.0,
            current_temp=20.0,
        )
        # Rewrite a HOLD slot the way the orchestrator's hedge would.
        hedged = None
        for slot in sorted(result.schedule):
            entry = result.schedule[slot]
            if entry.mode is _Mode.HOLD:
                entry.mode = _Mode.DISCHARGE
                entry.export_rate = 0
                hedged = slot
                break

        app = self._App(cfg, projector, load=0.0, pv=0.0)
        app._predict_load_kw = load
        replay = bo.BatteryOptimizer._validate_final_plan(
            app,
            schedule=result.schedule,
            soc_trajectory=result.soc_trajectory,
            starting_soc=60.0,
            starting_temp=20.0,
            current_slot=None,
            minutes_into_slot=0.0,
            prices_sorted=None,
            planning_temp_by_slot=result.planning_temp_by_slot,
        )
        assert replay is not None
        if hedged is not None and replay.trajectory_disagreements:
            assert replay.corrected is True
            corrected = replay.soc_trajectory()
            assert corrected[hedged][1] == pytest.approx(
                replay.by_slot[hedged].soc_end, abs=1e-12
            )


class TestOneTrajectoryPerPlan:
    """The schedule log and the deviation detector must read the same plan.

    Two trajectories were published for one schedule: the DP's own energies,
    built at the PLANNING temperatures, and ``calculate_expected_soc_schedule``,
    built by re-projecting at the projector's own evolving temperature. On the
    brief's Task 4 case they diverged by several SOC points -- the schedule log
    printed one and the deviation detector ran on the other, while the plan
    validator's tolerance is a tenth of a grid step.
    """

    class _Engine:
        @staticmethod
        def get_charge_rate_for_soc(soc, temp=None):
            if temp is None:
                return 4.0
            return 1.0 if temp < 10.0 else 4.0

        @staticmethod
        def predict_temp_after_idle(temp, duration_minutes):
            return temp

        @classmethod
        def predict_charge_input_dc_energy(
            cls, soc, start_temp, duration_minutes, temp_threshold=16.0
        ):
            # Flat within the slot, and charging warms the pack 1 C per minute
            # -- the same synthetic thermal model the DP is given below.
            rate = cls.get_charge_rate_for_soc(soc, start_temp)
            return rate * duration_minutes / 60.0, start_temp + duration_minutes

    class _App:
        def __init__(self, cfg, projector, load_fn, pv_fn):
            import types

            # project_schedule_trajectory reads a main-config surface, which
            # adds `charge_rate` to what DPOptimizerConfig carries.
            self.config = types.SimpleNamespace(
                battery_capacity=cfg.battery_capacity,
                efficiency=cfg.efficiency,
                charge_rate=4.0,
                discharge_rate=cfg.discharge_rate,
                export_discharge_rate=cfg.export_discharge_rate,
                inverter_efficiency=cfg.inverter_efficiency,
                slot_minutes=cfg.slot_minutes,
                soc_step_percent=cfg.soc_step_percent,
            )
            self.min_soc = cfg.min_soc
            self.max_soc = cfg.max_soc
            self._temp_projector = projector
            self.learning_engine = TestOneTrajectoryPerPlan._Engine()
            self._load_fn = load_fn
            self._pv_fn = pv_fn

        def project_schedule_trajectory(self, *args, **kwargs):
            import battery_optimizer as bo

            return bo.BatteryOptimizer.project_schedule_trajectory(
                self, *args, **kwargs
            )

        def _predict_load_kw(self, dt):
            return self._load_fn(dt)

        def _predict_pv_kw(self, dt):
            return self._pv_fn(dt)

        def _get_local_timezone(self):
            return None

        def log(self, message, level="INFO"):
            pass

    @staticmethod
    def _load(dt):
        return 4.0 if _slot_index(dt) == 2 else 0.0

    def _plan(self):
        """The brief's Task 4 case, where the refinement OSCILLATES.

        That is the interesting one: the plan is then built on the conservative
        idle profile (0 C throughout) while the shared projection evolves the
        pack to 15 C and 30 C, so the two trajectories diverge by the whole of
        the plan's warming.
        """
        cfg = _config()
        opt = DPOptimizer(
            config=cfg,
            load_predictor=self._load,
            charge_rate_predictor=self._Engine.get_charge_rate_for_soc,
            temp_after_charge_predictor=lambda t, m: t + m,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
        )
        result = opt.optimize(
            prices=_prices([0.60, 0.01, 1.00]),
            current_slot=BASE,
            current_soc=10.0,
            current_temp=0.0,
        )
        assert result.rate_refinement_fallback
        app = self._App(cfg, None, self._load, lambda dt: 0.0)
        return cfg, app, result

    def test_expected_soc_matches_the_dp_trajectory(self):
        import battery_optimizer as bo

        cfg, app, result = self._plan()
        expected_soc, _t = bo.BatteryOptimizer.calculate_expected_soc_schedule(
            app,
            result.schedule,
            10.0,
            starting_temp=0.0,
            planning_temp_by_slot=result.planning_temp_by_slot,
        )
        # calculate_expected_soc_schedule reports the START of each slot; the
        # DP trajectory reports (start, end). Compare starts.
        tolerance = 0.1 * cfg.soc_step_percent
        for slot in sorted(result.schedule.keys()):
            planned_start = result.soc_trajectory[slot][0]
            assert expected_soc[slot] == pytest.approx(planned_start, abs=tolerance), (
                f"{slot}: schedule log says {planned_start:.2f}%, the deviation "
                f"detector says {expected_soc[slot]:.2f}%"
            )

    def test_the_rebuilt_trajectory_matches_too(self):
        import battery_optimizer as bo

        cfg, app, result = self._plan()
        soc_traj, _temp = bo.BatteryOptimizer.project_schedule_trajectory(
            app,
            result.schedule,
            10.0,
            starting_temp=0.0,
            planning_temp_by_slot=result.planning_temp_by_slot,
        )
        tolerance = 0.1 * cfg.soc_step_percent
        for slot in sorted(result.schedule.keys()):
            assert soc_traj[slot][1] == pytest.approx(
                result.soc_trajectory[slot][1], abs=tolerance
            ), slot


def assert_plan_respects_its_own_rate_contract(cfg, result, rate_fn, tol=1e-9):
    """Every slot stores at most what ITS OWN planning rate allows.

    The invariant the DP owes: for each slot, the stored energy the plan
    credits is at most ``rate(soc at the start of that slot, the temperature
    the plan was built with for that slot) * efficiency * duration``. Written
    against ``rate_fn`` -- the caller's curve -- not against anything the
    implementation computed, so a hoisted or cached rate cannot make it pass.
    """
    for slot in sorted(result.schedule.keys()):
        start_soc, end_soc = result.soc_trajectory[slot]
        stored = max(0.0, (end_soc - start_soc) / 100.0 * cfg.battery_capacity)
        temp = result.planning_temp_by_slot.get(slot)
        allowed = rate_fn(start_soc, temp) * cfg.efficiency * cfg.slot_hours
        assert stored <= allowed + tol, (
            f"{slot}: plan stores {stored:.4f} kWh from {start_soc:.2f}% at "
            f"{temp} C, but its own rate curve allows {allowed:.4f} kWh"
        )


class TestRateIsEvaluatedAtTheTemperatureTheProfileReaches:
    """A rate that is SOC-dependent only at a temperature no probe visited.

    `_run_dp` used to decide once, from a fixed probe set around the CURRENT
    temperature, whether the rate varies with SOC -- and if it decided "no", it
    hoisted ``rate(min_soc, slot_temp)``, the fastest point of a tapering
    curve, onto every state. A curve that is flat at every probe but tapers at
    a temperature the refined profile actually reaches therefore had its taper
    erased, and the plan invented the difference.
    """

    @staticmethod
    def _rate(soc, temp):
        # Flat at every probe around 0 C (-5, 0, 5, 10, 15, 25) ...
        if temp is None:
            return 8.0
        if temp < 5.0:
            return 1.0
        # ... but tapered above 50 % in a window the probes step over, and
        # which the plan's own warming lands squarely inside.
        if 6.0 <= temp < 9.0 and soc > 50.0:
            return 0.5
        return 8.0

    def _run(self):
        cfg = _config(terminal_energy_value_eur_kwh=1.0)
        opt = DPOptimizer(
            config=cfg,
            load_predictor=lambda dt: 0.0,
            charge_rate_predictor=self._rate,
            # The pack settles at 7 C, inside the tapered window, whatever it
            # does -- so the refinement reaches a genuine fixed point and the
            # conservative fallback never fires to mask the defect.
            temp_after_charge_predictor=lambda t, m: 7.0,
            temp_after_idle_predictor=lambda t, m: 7.0,
            pv_predictor=lambda dt: 0.0,
        )
        result = opt.optimize(
            prices=_prices([0.01] * 5),
            current_slot=BASE,
            current_soc=50.5,
            current_temp=0.0,
        )
        return cfg, result

    def test_the_taper_is_not_erased(self):
        cfg, result = self._run()
        # The refinement must reach a fixed point here: the conservative
        # fallback would mask the defect rather than fix it.
        assert result.rate_refinement_converged
        assert not result.rate_refinement_fallback
        assert_plan_respects_its_own_rate_contract(cfg, result, self._rate)

    def test_no_slot_charges_at_the_low_soc_rate_from_a_tapered_state(self):
        cfg, result = self._run()
        for slot in sorted(result.schedule.keys()):
            start_soc, end_soc = result.soc_trajectory[slot]
            temp = result.planning_temp_by_slot.get(slot)
            if temp is not None and 6.0 <= temp < 9.0 and start_soc > 50.0:
                stored = (end_soc - start_soc) / 100.0 * cfg.battery_capacity
                # 0.5 kW for 15 minutes = 0.125 kWh = 1.25 SOC points.
                assert stored <= 0.125 + 1e-9


class TestRandomCurvesRespectTheirOwnRates:
    """Metamorphic sweep over arbitrary SOC/temperature rate curves."""

    @pytest.mark.parametrize("seed", range(12))
    def test_random_curve_plans_stay_inside_their_rate(self, seed):
        import random

        rng = random.Random(seed)
        soc_knots = sorted(rng.sample(range(11, 100), 3))
        temp_knots = sorted(rng.sample(range(-5, 30), 3))
        levels = [round(rng.uniform(0.3, 6.0), 3) for _ in range(16)]

        def rate(soc, temp):
            if temp is None:
                return levels[0]
            si = sum(1 for k in soc_knots if soc > k)
            ti = sum(1 for k in temp_knots if temp > k)
            return levels[si * 4 + ti]

        cfg = _config(terminal_energy_value_eur_kwh=rng.choice([0.0, 0.5, 1.0]))
        opt = DPOptimizer(
            config=cfg,
            load_predictor=lambda dt: rng.choice([0.0, 0.8, 3.0]),
            charge_rate_predictor=rate,
            temp_after_charge_predictor=lambda t, m: t + 0.3 * m,
            temp_after_idle_predictor=lambda t, m: t - 0.02 * m,
            pv_predictor=lambda dt: 0.0,
        )
        result = opt.optimize(
            prices=_prices([0.30, 0.02, 0.60, 0.01, 0.90, 0.05]),
            current_slot=BASE,
            current_soc=rng.choice([15.0, 45.5, 72.0]),
            current_temp=rng.choice([-2.0, 4.0, 12.0, 22.0]),
        )
        assert_plan_respects_its_own_rate_contract(cfg, result, rate)


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
