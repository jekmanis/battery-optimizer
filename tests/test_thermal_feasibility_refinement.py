"""Defect 1: a plan must be feasible at the temperature it actually reaches.

Two halves of one hole.

**The planner.** ``DPOptimizer._rate_is_temperature_sensitive`` probed the rate
curve at three SOCs (min, mid, max) on a fixed temperature ladder and, if none of
those eighteen lookups moved, skipped the whole solve/replay/refine loop. A
learned bucket that is temperature dependent only at a SOC *between* the probes
therefore bypassed refinement entirely, and ``optimize`` reported
``converged=True`` after a single pass on the idle profile.

**The validator.** ``BatteryOptimizer._replay_schedule`` looked the charge rate
up at ``planning_temp_by_slot[slot]`` -- the temperature the planner had assumed
-- so it compared the planner's arithmetic against the planner's own assumption
instead of against the capability the shared thermal model says the pack has.
Nothing reported the difference.

The maintainer's reproduction is below: a three-slot CHARGE, CHARGE, DISCHARGE
plan reported converged after one pass, whose replay at the temperatures it
actually reaches leaves 0.75 kWh of the final load uncovered.

The rate curves here are deliberately extreme, and one of them DECREASES with
temperature. That is not exotic: a BMS derates a hot pack, and it is the only
direction in which a plan built on the idle (coldest) profile can be
infeasible at all.
"""

import datetime
import math
import random
from typing import List, Optional

import pytest

from battery_optimizer_lib import (
    BatteryMode,
    DPOptimizer,
    DPOptimizerConfig,
    PricePoint,
)
from battery_optimizer_lib.slot_energy import SlotEnergyParams, simulate_slot


BASE = datetime.datetime(2026, 3, 2, 0, 0, 0)


def _config(**kwargs) -> DPOptimizerConfig:
    base = dict(
        battery_capacity=10.0,
        min_soc=10.0,
        max_soc=100.0,
        efficiency=1.0,
        discharge_rate=8.0,
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


def _prices(values) -> List[PricePoint]:
    return [
        PricePoint(time=BASE + datetime.timedelta(minutes=15 * i), price=p)
        for i, p in enumerate(values)
    ]


class _CountingOptimizer(DPOptimizer):
    """Counts DP solves so the pass budget is testable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solve_count = 0

    def _build_schedule(self, *args, **kwargs):
        self.solve_count += 1
        return super()._build_schedule(*args, **kwargs)


# ---------------------------------------------------------------------------
# The independent feasibility check the tests assert on.
#
# Deliberately NOT the production helper: it walks the published plan forward
# through `simulate_slot`, looks the rate up at the temperature the pack has
# reached in this walk, and reports how much of the plan's own credited charge
# energy the pack could not have taken -- plus the AC demand left uncovered.
# ---------------------------------------------------------------------------


def replay_at_reached_temperatures(
    cfg: DPOptimizerConfig,
    result,
    rate,
    temp_after_charge,
    temp_after_idle,
    starting_soc: float,
    starting_temp: Optional[float],
    load_fn,
    pv_fn=None,
):
    params = SlotEnergyParams(
        battery_capacity=cfg.battery_capacity,
        efficiency=cfg.efficiency,
        discharge_rate=cfg.discharge_rate,
        export_discharge_rate=cfg.export_discharge_rate,
        inverter_efficiency=cfg.inverter_efficiency,
        min_soc=cfg.min_soc,
        max_soc=cfg.max_soc,
        slot_minutes=cfg.slot_minutes,
    )
    energy = starting_soc / 100.0 * cfg.battery_capacity
    temp = starting_temp
    charge_shortfall = 0.0
    unmet_ac = 0.0
    temps = []
    for slot in sorted(result.schedule.keys()):
        entry = result.schedule[slot]
        temps.append(temp)
        soc = energy / cfg.battery_capacity * 100
        achievable_rate = rate(soc, temp)
        outcome = simulate_slot(
            stored_energy_kwh=energy,
            mode=entry.mode,
            params=params,
            charge_input_dc_kw=achievable_rate,
            load_kw=load_fn(slot),
            pv_kw=pv_fn(slot) if pv_fn is not None else 0.0,
            is_export=bool(entry.export_rate),
        )
        planned_start, planned_end = result.soc_trajectory[slot]
        planned_in = max(0.0, (planned_end - planned_start) / 100.0 * cfg.battery_capacity)
        charge_shortfall += max(0.0, planned_in - outcome.stored_dc_in_kwh)
        unmet_ac += outcome.unmet_battery_ac_kwh
        if temp is not None:
            if entry.mode == BatteryMode.CHARGE and outcome.battery_power_kw > 1e-9:
                temp = temp_after_charge(temp, cfg.slot_minutes)
            else:
                temp = temp_after_idle(temp, cfg.slot_minutes)
        energy = outcome.energy_end_kwh
    return charge_shortfall, unmet_ac, temps


# ---------------------------------------------------------------------------
# The maintainer's reproduction
# ---------------------------------------------------------------------------


class TestTemperatureSensitivityBetweenTheProbes:
    """A learned bucket that only varies at 15-25 % SOC.

    ``_rate_is_temperature_sensitive`` probed 10 %, 55 % and 100 % SOC, so the
    whole refinement loop was skipped and the plan was built on the idle profile
    at 0 C. The pack reaches 15 C after the first CHARGE slot, where this curve
    derates to 1 kW -- and the plan had credited 4 kW.
    """

    CAPACITY = 10.0
    START_SOC = 10.0
    START_TEMP = 0.0

    @staticmethod
    def rate(soc, temp):
        # Flat 4 kW at every SOC the probe visits (10 / 55 / 100), at every
        # temperature. Between 15 % and 25 % it derates once the pack is warm.
        if 15.0 <= soc < 25.0 and temp is not None and temp >= 10.0:
            return 1.0
        return 4.0

    @staticmethod
    def temp_after_charge(t, m):
        return t + m  # 1 C per minute: 0 C -> 15 C over one 15-minute slot

    @staticmethod
    def temp_after_idle(t, m):
        return t

    @classmethod
    def load(cls, dt):
        return 8.0 if dt == BASE + datetime.timedelta(minutes=30) else 0.0

    def _run(self):
        cfg = _config()
        opt = _CountingOptimizer(
            config=cfg,
            load_predictor=self.load,
            charge_rate_predictor=self.rate,
            temp_after_charge_predictor=self.temp_after_charge,
            temp_after_idle_predictor=self.temp_after_idle,
            pv_predictor=lambda dt: 0.0,
        )
        result = opt.optimize(
            prices=_prices([0.60, 0.01, 1.00]),
            current_slot=BASE,
            current_soc=self.START_SOC,
            current_temp=self.START_TEMP,
        )
        return cfg, opt, result

    def test_the_reproduction_still_bites_without_refinement(self):
        """Guards the fixture: with the plan pinned to the idle profile the
        pack is 0.75 kWh short. Computed here, not taken from the planner."""
        cfg = _config()
        params = SlotEnergyParams(
            battery_capacity=cfg.battery_capacity,
            efficiency=cfg.efficiency,
            discharge_rate=cfg.discharge_rate,
            inverter_efficiency=cfg.inverter_efficiency,
            min_soc=cfg.min_soc,
            max_soc=cfg.max_soc,
            slot_minutes=cfg.slot_minutes,
        )
        # Planned on the idle profile (0 C throughout): 4 kW in both slots.
        planned = [4.0 * 0.25, 4.0 * 0.25]
        # Reached: slot 0 at 0 C is 4 kW; slot 1 starts at 15 C and 20 % SOC.
        assert self.rate(10.0, 0.0) == 4.0
        assert self.rate(20.0, 15.0) == 1.0
        achievable = [4.0 * 0.25, 1.0 * 0.25]
        assert sum(planned) - sum(achievable) == pytest.approx(0.75)
        # And the DISCHARGE slot then cannot cover its 2.0 kWh of load.
        energy = params.energy_of(10.0) + sum(achievable)
        served = simulate_slot(
            stored_energy_kwh=energy,
            mode=BatteryMode.DISCHARGE,
            params=params,
            charge_input_dc_kw=0.0,
            load_kw=8.0,
        )
        assert served.unmet_battery_ac_kwh == pytest.approx(0.75)

    def test_refinement_is_not_skipped(self):
        cfg, opt, result = self._run()
        assert result.rate_refinement_passes > 1, (
            "the sampled temperature-sensitivity probe skipped refinement"
        )
        assert opt.solve_count == result.rate_refinement_passes

    def test_the_published_plan_is_feasible_at_the_temperatures_it_reaches(self):
        cfg, opt, result = self._run()
        shortfall, unmet, _temps = replay_at_reached_temperatures(
            cfg,
            result,
            self.rate,
            self.temp_after_charge,
            self.temp_after_idle,
            self.START_SOC,
            self.START_TEMP,
            self.load,
        )
        assert shortfall == pytest.approx(0.0, abs=1e-9)

    def test_the_branch_that_produced_the_plan_is_exposed(self):
        cfg, opt, result = self._run()
        assert result.rate_refinement_branch in (
            "converged",
            "conservative_fallback",
            "degraded",
        )
        if result.rate_refinement_branch == "converged":
            assert result.rate_refinement_converged is True
            assert result.rate_refinement_fallback is False
        else:
            assert result.rate_refinement_converged is False
            assert result.rate_refinement_fallback is True

    def test_no_slot_credits_more_charge_than_its_reached_temperature_allows(self):
        cfg, opt, result = self._run()
        _short, _unmet, temps = replay_at_reached_temperatures(
            cfg,
            result,
            self.rate,
            self.temp_after_charge,
            self.temp_after_idle,
            self.START_SOC,
            self.START_TEMP,
            self.load,
        )
        for i, slot in enumerate(sorted(result.schedule.keys())):
            start, end = result.soc_trajectory[slot]
            stored = max(0.0, (end - start) / 100.0 * cfg.battery_capacity)
            allowed = self.rate(start, temps[i]) * cfg.slot_hours * cfg.efficiency
            assert stored <= allowed + 1e-9


# ---------------------------------------------------------------------------
# Metamorphic sweep
# ---------------------------------------------------------------------------


class TestEveryPublishedPlanIsAchievableAtItsReachedTemperature:
    """Random piecewise (soc, temp) curves, including derating ones."""

    @pytest.mark.parametrize("seed", range(20))
    def test_random_curve(self, seed):
        rng = random.Random(1000 + seed)
        soc_knots = sorted(rng.sample(range(12, 99), 3))
        temp_knots = sorted(rng.sample(range(-4, 40), 3))
        levels = [round(rng.uniform(0.2, 6.0), 3) for _ in range(16)]

        def rate(soc, temp):
            if temp is None:
                return levels[0]
            si = sum(1 for k in soc_knots if soc > k)
            ti = sum(1 for k in temp_knots if temp > k)
            return levels[si * 4 + ti]

        warm_per_min = rng.choice([0.2, 0.8, 2.0])
        cool_per_min = rng.choice([0.0, 0.05])

        def temp_after_charge(t, m):
            return t + warm_per_min * m

        def temp_after_idle(t, m):
            return t - cool_per_min * m

        load_by_slot = [rng.choice([0.0, 0.5, 4.0]) for _ in range(6)]

        def load(dt):
            idx = int((dt - BASE).total_seconds() // 900)
            return load_by_slot[idx % len(load_by_slot)]

        cfg = _config(
            terminal_energy_value_eur_kwh=rng.choice([0.0, 0.5]),
            discharge_rate=rng.choice([4.0, 8.0]),
        )
        opt = DPOptimizer(
            config=cfg,
            load_predictor=load,
            charge_rate_predictor=rate,
            temp_after_charge_predictor=temp_after_charge,
            temp_after_idle_predictor=temp_after_idle,
            pv_predictor=lambda dt: 0.0,
        )
        start_soc = rng.choice([10.0, 30.5, 62.0])
        start_temp = rng.choice([-2.0, 6.0, 18.0, 30.0])
        result = opt.optimize(
            prices=_prices([0.30, 0.02, 0.60, 0.01, 0.90, 0.05]),
            current_slot=BASE,
            current_soc=start_soc,
            current_temp=start_temp,
        )
        shortfall, _unmet, _temps = replay_at_reached_temperatures(
            cfg,
            result,
            rate,
            temp_after_charge,
            temp_after_idle,
            start_soc,
            start_temp,
            load,
        )
        assert shortfall == pytest.approx(0.0, abs=1e-9), (
            f"seed {seed} published {shortfall:.6f} kWh of charge energy that is "
            f"not available at the temperature the plan reaches "
            f"(branch={result.rate_refinement_branch})"
        )

    @pytest.mark.parametrize("seed", range(8))
    def test_a_partial_first_slot_is_covered_too(self, seed):
        rng = random.Random(7000 + seed)
        levels = [round(rng.uniform(0.2, 6.0), 3) for _ in range(4)]

        def rate(soc, temp):
            if temp is None:
                return levels[0]
            return levels[min(3, max(0, int((temp + 5) // 12)))]

        def temp_after_charge(t, m):
            return t + 1.2 * m

        def temp_after_idle(t, m):
            return t

        def load(dt):
            return 4.0 if dt >= BASE + datetime.timedelta(minutes=60) else 0.0

        cfg = _config()
        opt = DPOptimizer(
            config=cfg,
            load_predictor=load,
            charge_rate_predictor=rate,
            temp_after_charge_predictor=temp_after_charge,
            temp_after_idle_predictor=temp_after_idle,
            pv_predictor=lambda dt: 0.0,
        )
        result = opt.optimize(
            prices=_prices([0.02, 0.05, 0.02, 0.90, 0.90]),
            current_slot=BASE,
            current_soc=20.0,
            current_temp=rng.choice([-3.0, 5.0, 17.0]),
            minutes_into_slot=7.0,
        )
        # The published trajectory's first slot is the partial one; check the
        # remaining slots, where a full-slot replay is the right comparison.
        slots = sorted(result.schedule.keys())
        params = SlotEnergyParams(
            battery_capacity=cfg.battery_capacity,
            efficiency=cfg.efficiency,
            discharge_rate=cfg.discharge_rate,
            inverter_efficiency=cfg.inverter_efficiency,
            min_soc=cfg.min_soc,
            max_soc=cfg.max_soc,
            slot_minutes=cfg.slot_minutes,
        )
        energy = 20.0 / 100.0 * cfg.battery_capacity
        temp = result.temp_trajectory[slots[0]][0]
        for i, slot in enumerate(slots):
            entry = result.schedule[slot]
            fraction = (15.0 - 7.0) / 15.0 if i == 0 else 1.0
            soc = energy / cfg.battery_capacity * 100
            outcome = simulate_slot(
                stored_energy_kwh=energy,
                mode=entry.mode,
                params=params,
                charge_input_dc_kw=rate(soc, temp),
                load_kw=load(slot),
                fraction=fraction,
            )
            start, end = result.soc_trajectory[slot]
            planned_in = max(
                0.0, (end - start) / 100.0 * cfg.battery_capacity
            )
            assert planned_in <= outcome.stored_dc_in_kwh + 1e-9
            if entry.mode == BatteryMode.CHARGE and outcome.battery_power_kw > 1e-9:
                temp = temp_after_charge(temp, cfg.slot_minutes * fraction)
            else:
                temp = temp_after_idle(temp, cfg.slot_minutes * fraction)
            energy = (end / 100.0) * cfg.battery_capacity


# ---------------------------------------------------------------------------
# Oscillation, budget exhaustion, and the degrade branch
# ---------------------------------------------------------------------------


class _ScriptedProfileOptimizer(_CountingOptimizer):
    """Feeds the refinement loop a scripted sequence of replayed profiles.

    The refinement is a fixed-point iteration over a CAUSAL system -- slot i's
    temperature depends only on actions 0..i-1 -- so an ordinary rate curve
    settles in a couple of passes and neither the oscillation branch nor the
    budget branch is reachable from physics alone in a small deterministic
    fixture. This subclass therefore drives the loop directly, which is what
    these three tests are about: the CONTROL FLOW of the fallback, not the
    thermal model. The physical behaviour is covered by the reproduction and
    the metamorphic sweep above.
    """

    def __init__(self, *args, scripted_profiles=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._scripted = list(scripted_profiles or [])
        self._script_index = 0
        self.replay_calls = 0

    def _replay_plan(self, *args, **kwargs):
        """Only the reported PROFILE is scripted.

        The shortfall, the energies and the temperature pairs are the real
        walk's, so what the loop then does about the scripted profile is
        measured against genuine physics.
        """
        replay = super()._replay_plan(*args, **kwargs)
        self.replay_calls += 1
        if kwargs.get("follow_plan_energy", True) is False:
            return replay
        if self._script_index < len(self._scripted):
            temps = self._scripted[self._script_index]
            self._script_index += 1
            replay.temp_profile = [temps] * len(replay.temp_profile)
        return replay

    @staticmethod
    def _rate(soc, temp):
        # Strictly decreasing in temperature, so two profiles that differ at
        # all imply different charge capability -- otherwise `_profiles_agree`
        # would call a scripted 12 C and 19 C the same plan and the loop would
        # stop before reaching the branch under test.
        if temp is None:
            return 4.0
        return max(0.2, 4.0 - 0.15 * temp)

    @classmethod
    def build(cls, **kwargs):
        cfg = _config()
        opt = cls(
            config=cfg,
            load_predictor=lambda dt: (
                8.0 if dt == BASE + datetime.timedelta(minutes=45) else 0.0
            ),
            charge_rate_predictor=cls._rate,
            temp_after_charge_predictor=lambda t, m: t + m,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
            **kwargs,
        )
        return cfg, opt

    @staticmethod
    def run(opt):
        return opt.optimize(
            prices=_prices([0.30, 0.01, 0.02, 1.00]),
            current_slot=BASE,
            current_soc=10.0,
            current_temp=0.0,
        )


class TestOscillationAndBudget:
    # Profiles that include a temperature at or above everything the real walk
    # reaches: the minimum rate over the profiles seen is then a true bound and
    # the conservative solve is feasible.
    COVERING = [0.0, 60.0, 0.0, 60.0, 0.0, 60.0]
    # Profiles that all sit BELOW the temperatures the real walk reaches, so the
    # minimum over them is not a bound and the conservative solve is still short.
    UNCOVERING = [5.0, 12.0, 19.0, 26.0, 33.0, 40.0]

    def test_an_oscillating_profile_falls_back_conservatively(self):
        cfg, opt = _ScriptedProfileOptimizer.build(scripted_profiles=self.COVERING)
        result = _ScriptedProfileOptimizer.run(opt)
        assert result.rate_refinement_branch == "conservative_fallback"
        assert result.rate_refinement_converged is False
        assert result.rate_refinement_fallback is True
        assert result.rate_refinement_degraded is False
        assert opt.solve_count <= 5

    def test_budget_exhaustion_falls_back_conservatively(self):
        cfg, opt = _ScriptedProfileOptimizer.build(
            scripted_profiles=[5.0, 60.0, 12.0, 60.0, 12.0]
        )
        result = _ScriptedProfileOptimizer.run(opt)
        assert result.rate_refinement_branch == "conservative_fallback"
        assert result.rate_refinement_fallback is True
        # Bounded: at most MAX_RATE_REFINEMENT_PASSES solves plus the one
        # conservative solve.
        from battery_optimizer_lib.dp_optimizer import MAX_RATE_REFINEMENT_PASSES

        assert opt.solve_count <= MAX_RATE_REFINEMENT_PASSES + 1
        assert result.rate_refinement_passes <= MAX_RATE_REFINEMENT_PASSES + 1

    def test_the_conservative_fallback_is_feasible_at_its_reached_temperatures(self):
        cfg, opt = _ScriptedProfileOptimizer.build(scripted_profiles=self.COVERING)
        result = _ScriptedProfileOptimizer.run(opt)
        assert result.rate_refinement_branch == "conservative_fallback"
        shortfall, _unmet, _temps = replay_at_reached_temperatures(
            cfg,
            result,
            _ScriptedProfileOptimizer._rate,
            lambda t, m: t + m,
            lambda t, m: t,
            10.0,
            0.0,
            lambda dt: 8.0 if dt == BASE + datetime.timedelta(minutes=45) else 0.0,
        )
        assert shortfall == pytest.approx(0.0, abs=1e-9)

    def test_a_conservative_solve_that_is_still_short_degrades(self):
        """The min over the profiles SEEN is not a bound over all of them.

        Every scripted profile here is colder than the temperatures the real
        walk reaches, so the conservative solve still credits charge energy the
        pack cannot take -- and the branch degrades rather than publishing it.
        """
        logged = []
        cfg, opt = _ScriptedProfileOptimizer.build(
            scripted_profiles=self.UNCOVERING,
            log_fn=lambda msg, level="INFO": logged.append((level, msg)),
            decision_log_level=1,
        )
        result = _ScriptedProfileOptimizer.run(opt)
        assert result.rate_refinement_branch == "degraded"
        assert result.rate_refinement_degraded is True
        assert result.rate_refinement_converged is False
        assert result.rate_refinement_shortfall_kwh > 0.0
        warnings = [m for lvl, m in logged if lvl == "WARNING"]
        assert warnings, "the degrade branch must log at WARNING"
        assert any("kWh" in m for m in warnings)

    def test_the_degraded_trajectory_credits_only_achievable_energy(self):
        cfg, opt = _ScriptedProfileOptimizer.build(
            scripted_profiles=self.UNCOVERING,
        )
        result = _ScriptedProfileOptimizer.run(opt)
        assert result.rate_refinement_branch == "degraded"
        shortfall, _unmet, _temps = replay_at_reached_temperatures(
            cfg,
            result,
            _ScriptedProfileOptimizer._rate,
            lambda t, m: t + m,
            lambda t, m: t,
            10.0,
            0.0,
            lambda dt: 8.0 if dt == BASE + datetime.timedelta(minutes=45) else 0.0,
        )
        assert shortfall == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Orchestrator validation, after the cloud-safe hedge
# ---------------------------------------------------------------------------


class _ValidationApp:
    """The surface ``_validate_final_plan`` needs, and nothing else."""

    class Engine:
        """1 kW below 10 C, 4 kW at or above."""

        @staticmethod
        def get_charge_rate_for_soc(soc, temp=None):
            if temp is None:
                return 4.0
            return 1.0 if temp < 10.0 else 4.0

        @staticmethod
        def predict_temp_after_idle(temp, duration_minutes):
            return temp

        @staticmethod
        def predict_temp_after_duration(temp, duration_minutes):
            return temp + duration_minutes

    @staticmethod
    def projector():
        """No relaxation, strong self-heating: 1 kW for 15 minutes is +15 C.

        ``plan_validation.replay_plan`` evolves temperature only through the
        shared projector, which is what the orchestrator always supplies.
        """
        from battery_optimizer_lib.thermal_model import TemperatureProjector

        return TemperatureProjector(
            learning_engine=None,
            ambient_provider=None,
            default_cooling_rate=0.0,
            default_heating_c_per_kwh=60.0,
            fallback_ambient_c=0.0,
        )

    def __init__(self, cfg, load_fn, pv_fn=None, projector=None):
        self.config = cfg
        self.min_soc = cfg.min_soc
        self.max_soc = cfg.max_soc
        self._temp_projector = projector if projector is not None else self.projector()
        self.learning_engine = self.Engine()
        self._load_fn = load_fn
        self._pv_fn = pv_fn or (lambda dt: 0.0)
        self.messages = []

    def _predict_load_kw(self, dt):
        return self._load_fn(dt)

    def _predict_pv_kw(self, dt):
        return self._pv_fn(dt)

    def _get_local_timezone(self):
        return None

    def log(self, message, level="INFO"):
        self.messages.append((level, message))

    def _replay_schedule(self, **kwargs):
        import battery_optimizer as bo

        return bo.BatteryOptimizer._replay_schedule(self, **kwargs)

    def _resolve_plan_shortfall(self, **kwargs):
        import battery_optimizer as bo

        return bo.BatteryOptimizer._resolve_plan_shortfall(self, **kwargs)

    def validate(self, **kwargs):
        import battery_optimizer as bo

        return bo.BatteryOptimizer._validate_final_plan(self, **kwargs)


def _entry(slot, mode, export_rate=None):
    from battery_optimizer_lib import ScheduleEntry

    entry = ScheduleEntry(time=slot, mode=mode, reason="test")
    entry.export_rate = export_rate
    return entry


class TestValidationLooksUpTheRateAtTheReachedTemperature:
    """The other half of the defect.

    ``_replay_schedule`` pinned the charge-rate lookup to
    ``planning_temp_by_slot``, so a plan that credited warm-pack charging in a
    cold slot validated clean: the check was the planner's arithmetic against
    the planner's own assumption.
    """

    def _plan(self):
        cfg = _config(discharge_rate=4.0)
        slots = [BASE + datetime.timedelta(minutes=15 * i) for i in range(2)]
        schedule = {
            slots[0]: _entry(slots[0], BatteryMode.CHARGE),
            slots[1]: _entry(slots[1], BatteryMode.CHARGE),
        }
        return cfg, slots, schedule

    def test_a_warm_rate_credited_in_a_cold_slot_is_reported(self):
        cfg, slots, schedule = self._plan()
        app = _ValidationApp(cfg, lambda dt: 0.0)
        # A trajectory that credits the WARM rate (4 kW = 10 SOC points) in the
        # first slot, which the pack at 0 C cannot take.
        published = {
            slots[0]: (20.0, 30.0),
            slots[1]: (30.0, 40.0),
        }
        replay = app.validate(
            schedule=schedule,
            soc_trajectory=published,
            starting_soc=20.0,
            starting_temp=0.0,
            current_slot=None,
            minutes_into_slot=0.0,
        )
        assert replay is not None
        assert replay.trajectory_disagreements, (
            "validation accepted warm-rate charging on a 0 C pack"
        )
        assert replay.corrected is True
        # The corrected trajectory is the cold rate: 1 kW for 15 minutes.
        corrected = replay.soc_trajectory()
        assert corrected[slots[0]][1] == pytest.approx(22.5, abs=1e-9)

    def test_the_pack_warming_itself_is_credited(self):
        """The second slot legitimately charges at 4 kW: its own first slot
        warmed the pack past the threshold."""
        cfg, slots, schedule = self._plan()
        app = _ValidationApp(cfg, lambda dt: 0.0)
        replay = app.validate(
            schedule=schedule,
            soc_trajectory={slots[0]: (20.0, 22.5), slots[1]: (22.5, 32.5)},
            starting_soc=20.0,
            starting_temp=0.0,
            current_slot=None,
            minutes_into_slot=0.0,
        )
        assert replay is not None
        assert replay.trajectory_disagreements == []
        assert replay.conservation_violations == []


class TestPostHedgeShortfallIsResolved:
    """A hedge conversion the pack cannot serve is reverted, then degraded."""

    @staticmethod
    def _setup():
        cfg = _config(discharge_rate=4.0)
        slots = [BASE + datetime.timedelta(minutes=15 * i) for i in range(2)]
        # The pack sits at min SOC: a DISCHARGE slot can serve nothing.
        schedule = {
            slots[0]: _entry(slots[0], BatteryMode.DISCHARGE, export_rate=0),
            slots[1]: _entry(slots[1], BatteryMode.HOLD),
        }
        app = _ValidationApp(cfg, lambda dt: 4.0)
        return cfg, slots, schedule, app

    def test_the_conversion_is_reverted_and_the_plan_then_validates(self):
        cfg, slots, schedule, app = self._setup()
        replay = app.validate(
            schedule=schedule,
            soc_trajectory={slots[0]: (10.0, 10.0), slots[1]: (10.0, 10.0)},
            starting_soc=10.0,
            starting_temp=20.0,
            current_slot=None,
            minutes_into_slot=0.0,
            hedge_converted_slots=[slots[0]],
        )
        assert replay is not None
        assert schedule[slots[0]].mode == BatteryMode.HOLD, (
            "the unserviceable hedge conversion was not reverted"
        )
        assert replay.conservation_violations == []
        assert any(
            lvl == "WARNING" and "reverting" in msg for lvl, msg in app.messages
        )
        # No ERROR: reverting the hedge was enough.
        assert not [m for m in app.messages if m[0] == "ERROR"]

    def test_a_shortfall_that_is_not_the_hedge_degrades_and_logs_error(self):
        cfg, slots, schedule, app = self._setup()
        replay = app.validate(
            schedule=schedule,
            soc_trajectory={slots[0]: (10.0, 10.0), slots[1]: (10.0, 10.0)},
            starting_soc=10.0,
            starting_temp=20.0,
            current_slot=None,
            minutes_into_slot=0.0,
            hedge_converted_slots=None,
        )
        assert replay is not None
        # The action stays; the plan now DECLARES what it really does, so
        # nothing credits the battery with service it does not have.
        assert schedule[slots[0]].mode == BatteryMode.DISCHARGE
        assert schedule[slots[0]].energy_limited is True
        assert replay.conservation_violations == []
        assert replay.corrected is True
        errors = [m for lvl, m in app.messages if lvl == "ERROR"]
        assert errors and "kWh" in errors[0]


class TestNoTemperatureReadingStillPlansInOneSolve:
    def test_one_solve_without_a_temperature(self):
        cfg = _config()
        opt = _CountingOptimizer(
            config=cfg,
            load_predictor=lambda dt: 0.0,
            charge_rate_predictor=lambda soc, temp: 4.0,
            temp_after_charge_predictor=lambda t, m: t + m,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
        )
        result = opt.optimize(
            prices=_prices([0.01] * 4),
            current_slot=BASE,
            current_soc=20.0,
            current_temp=None,
        )
        assert result.rate_refinement_passes == 1
        assert opt.solve_count == 1
        assert result.rate_refinement_branch == "single_solve"
        assert result.rate_refinement_converged is True
        assert result.temp_trajectory == {}
