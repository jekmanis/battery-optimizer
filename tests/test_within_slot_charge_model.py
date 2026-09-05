"""Defect 2: ONE within-slot charge model, in every consumer.

The DP charged a slot at a constant rate taken from the temperature at the
slot's START (``dp_optimizer._run_dp``). ``soc_projection.project_slot_soc``
called ``learning_engine.predict_charge_input_dc_energy``, which split the slot
into a cold phase and a warm phase using the learning engine's legacy warming
model (``get_time_to_reach_temp`` / ``predict_temp_after_duration``) -- a second
thermal model, and a second answer for the same slot.

The maintainer's reproduction: 10 kWh pack, 10 % SOC, one 15-minute CHARGE that
crosses from 1 kW to 4 kW halfway. The DP said 12.5 %, the projector said
16.25 %. Nothing in production was wrong with the battery; the expected-SOC
trajectory simply disagreed with the plan by 3.75 SOC points, and the deviation
detector chased the difference.

The chosen model is the DP's: a CONSTANT rate at the start-of-slot temperature,
in candidate evaluation, ``simulate_slot``, ``plan_validation.replay_plan``,
``project_slot_soc`` (expected SOC and the deviation detector) and
``cost_tracker.project_costs``. Temperature evolves BETWEEN slots only, through
``thermal_model.TemperatureProjector``.
"""

import datetime
from typing import Optional

import pytest

from battery_optimizer_lib import (
    BatteryMode,
    DPOptimizer,
    DPOptimizerConfig,
    PricePoint,
    ScheduleEntry,
)
from battery_optimizer_lib.plan_validation import replay_plan
from battery_optimizer_lib.soc_projection import (
    SocProjectionParams,
    project_slot_soc,
)


BASE = datetime.datetime(2026, 3, 2, 0, 0, 0)
SLOT_0 = BASE
SLOT_1 = BASE + datetime.timedelta(minutes=15)

CAPACITY = 10.0
START_SOC = 10.0
START_TEMP = 10.0

# 1 kW cold, 4 kW warm, threshold 16 C, warming 0.8 C/min: the pack crosses the
# threshold exactly 7.5 minutes into a 15-minute slot.
COLD_RATE = 1.0
WARM_RATE = 4.0
THRESHOLD_C = 16.0
WARMING_C_PER_MIN = 0.8

# The one within-slot model: 1 kW for the whole slot.
#   0.25 h * 1 kW * efficiency 1.0 / 10 kWh = 2.5 SOC points
CONSTANT_RATE_END_SOC = 12.5
# What the removed cold/warm split produced:
#   7.5 min at 1 kW + 7.5 min at 4 kW = 0.625 kWh = 6.25 SOC points
SPLIT_END_SOC = 16.25


class CrossingEngine:
    """Learning-engine double whose rate steps up mid-slot.

    Deliberately extreme so the disagreement is unmistakable; it is not a claim
    about the real pack's warming rate.
    """

    storage_efficiency = 1.0

    def __init__(self):
        # Every call to the legacy two-phase predictor, so a test can assert
        # that planning and projection no longer reach it at all.
        self.split_calls = []

    def get_charge_rate_for_soc(self, soc: float, battery_temp: Optional[float] = None) -> float:
        if battery_temp is None:
            return WARM_RATE
        return COLD_RATE if battery_temp < THRESHOLD_C else WARM_RATE

    def get_warming_rate(self, starting_temp: float) -> float:
        return WARMING_C_PER_MIN

    def get_time_to_reach_temp(self, start_temp: float, target_temp: float):
        if target_temp <= start_temp:
            return 0.0
        return (target_temp - start_temp) / WARMING_C_PER_MIN

    def predict_temp_after_duration(
        self, start_temp, duration_minutes, ambient_temp=None, battery_power_kw=None
    ):
        return start_temp + WARMING_C_PER_MIN * duration_minutes

    def predict_temp_after_idle(
        self, start_temp, duration_minutes, ambient_temp=None, default_cooling_rate=0.012
    ):
        return start_temp

    # The legacy two-phase predictor, under the name the learning engine still
    # exposes it by. Kept on the double so this module can compute what the
    # projector used to answer -- and so the tests can assert that planning and
    # projection never call it.
    def predict_charge_input_dc_energy(
        self, current_soc, start_temp, duration_minutes, temp_threshold=THRESHOLD_C
    ):
        self.split_calls.append((current_soc, start_temp, duration_minutes))
        return (
            self.split_phase_energy(current_soc, start_temp, duration_minutes),
            self.predict_temp_after_duration(start_temp, duration_minutes),
        )

    def split_phase_energy(self, soc, start_temp, duration_minutes):
        if start_temp >= THRESHOLD_C:
            return WARM_RATE * duration_minutes / 60.0
        warm_after = self.get_time_to_reach_temp(start_temp, THRESHOLD_C)
        if warm_after >= duration_minutes:
            return COLD_RATE * duration_minutes / 60.0
        return (
            COLD_RATE * warm_after / 60.0
            + WARM_RATE * (duration_minutes - warm_after) / 60.0
        )


def _dp_config(**kwargs) -> DPOptimizerConfig:
    base = dict(
        battery_capacity=CAPACITY,
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


def _projection_params() -> SocProjectionParams:
    return SocProjectionParams(
        battery_capacity=CAPACITY,
        efficiency=1.0,
        charge_rate=WARM_RATE,
        discharge_rate=4.0,
        inverter_efficiency=1.0,
        min_soc=10.0,
        max_soc=100.0,
        slot_minutes=15,
    )


def _load(dt):
    """4 kW in the second slot only, so the DP has a reason to charge first."""
    return 4.0 if dt == SLOT_1 else 0.0


class TestTheCrossingSlotHasOneAnswer:
    """1 kW -> 4 kW halfway through a 15-minute CHARGE, from four consumers."""

    def test_the_split_and_the_constant_rate_really_do_differ(self):
        """Guards the reproduction itself: 12.5 % vs 16.25 %."""
        engine = CrossingEngine()
        split_kwh = engine.split_phase_energy(START_SOC, START_TEMP, 15.0)
        constant_kwh = COLD_RATE * 0.25
        assert constant_kwh == pytest.approx(0.25)
        assert split_kwh == pytest.approx(0.625)
        assert START_SOC + constant_kwh / CAPACITY * 100 == pytest.approx(
            CONSTANT_RATE_END_SOC
        )
        assert START_SOC + split_kwh / CAPACITY * 100 == pytest.approx(SPLIT_END_SOC)

    def test_dp_optimize(self):
        engine = CrossingEngine()
        cfg = _dp_config()
        opt = DPOptimizer(
            config=cfg,
            load_predictor=_load,
            charge_rate_predictor=engine.get_charge_rate_for_soc,
            temp_after_charge_predictor=engine.predict_temp_after_duration,
            temp_after_idle_predictor=engine.predict_temp_after_idle,
            pv_predictor=lambda dt: 0.0,
        )
        result = opt.optimize(
            prices=[
                PricePoint(time=SLOT_0, price=0.01),
                PricePoint(time=SLOT_1, price=1.00),
            ],
            current_slot=SLOT_0,
            current_soc=START_SOC,
            current_temp=START_TEMP,
        )
        assert result.schedule[SLOT_0].mode == BatteryMode.CHARGE
        assert result.soc_trajectory[SLOT_0][1] == pytest.approx(
            CONSTANT_RATE_END_SOC, abs=1e-9
        )

    def test_project_slot_soc(self):
        engine = CrossingEngine()
        transition = project_slot_soc(
            soc_start=START_SOC,
            mode=BatteryMode.CHARGE,
            params=_projection_params(),
            load_kw=0.0,
            pv_kw=0.0,
            fraction=1.0,
            temp_start=START_TEMP,
            learning_engine=engine,
        )
        assert transition.soc_end == pytest.approx(CONSTANT_RATE_END_SOC, abs=1e-9)
        assert engine.split_calls == [], (
            "project_slot_soc still calls the legacy cold/warm split"
        )

    def test_replay_plan(self):
        engine = CrossingEngine()
        cfg = _dp_config()
        schedule = {
            SLOT_0: ScheduleEntry(time=SLOT_0, mode=BatteryMode.CHARGE, reason="t"),
            SLOT_1: ScheduleEntry(time=SLOT_1, mode=BatteryMode.HOLD, reason="t"),
        }
        replay = replay_plan(
            schedule=schedule,
            config=cfg,
            starting_soc=START_SOC,
            predict_load_kw=lambda dt: 0.0,
            predict_pv_kw=lambda dt: 0.0,
            charge_rate_for=lambda slot, soc, temp: engine.get_charge_rate_for_soc(
                soc, temp
            ),
            starting_temp=START_TEMP,
        )
        assert replay.by_slot[SLOT_0].soc_end == pytest.approx(
            CONSTANT_RATE_END_SOC, abs=1e-9
        )

    def test_project_costs(self, monkeypatch):
        """``project_costs`` walks the same transition; capture the SOC it sees."""
        from battery_optimizer_lib import cost_tracker as cost_tracker_module
        from battery_optimizer_lib import soc_projection as soc_projection_module

        seen = {}
        real = soc_projection_module.project_slot_soc

        def spy(**kwargs):
            seen[kwargs.get("slot_time")] = kwargs["soc_start"]
            return real(**kwargs)

        monkeypatch.setattr(soc_projection_module, "project_slot_soc", spy)

        engine = CrossingEngine()
        tracker = _cost_tracker(cost_tracker_module)
        schedule = {
            SLOT_0: ScheduleEntry(time=SLOT_0, mode=BatteryMode.CHARGE, reason="t"),
            SLOT_1: ScheduleEntry(time=SLOT_1, mode=BatteryMode.HOLD, reason="t"),
        }
        tracker.project_costs(
            schedule,
            START_SOC,
            0.10,
            {SLOT_0: 0.01, SLOT_1: 1.00},
            predict_load_func=lambda dt: 0.0,
            predict_pv_func=lambda dt: 0.0,
            starting_temp=START_TEMP,
            learning_engine=engine,
        )
        # The SOC the second slot starts from IS the first slot's end SOC.
        assert seen[SLOT_1] == pytest.approx(CONSTANT_RATE_END_SOC, abs=1e-9)
        assert engine.split_calls == [], (
            "project_costs still reaches the legacy cold/warm split"
        )


def _cost_tracker(cost_tracker_module):
    """A real BatteryCostTracker on the same battery, with no HA surface."""
    from battery_optimizer_lib.learning_engine import BatteryLearningEngine

    return cost_tracker_module.BatteryCostTracker(
        config=cost_tracker_module.BatteryCostConfig(
            battery_capacity=CAPACITY,
            efficiency=1.0,
            slot_minutes=15,
            charge_rate=WARM_RATE,
            discharge_rate=4.0,
            grid_fee=0.0,
            inverter_efficiency=1.0,
            battery_wear_cost=0.0,
            default_cost=0.10,
        ),
        get_state_func=lambda e: None,
        call_service_func=lambda *a, **k: None,
        get_datetime_func=lambda: BASE,
        get_timezone_func=lambda: None,
        align_to_slot_func=lambda dt: dt,
        get_min_soc_func=lambda: 10.0,
        get_max_soc_func=lambda: 100.0,
        get_current_soc_func=lambda: START_SOC,
        get_battery_temp_func=lambda: START_TEMP,
        learning_engine=BatteryLearningEngine(
            battery_capacity_kwh=CAPACITY,
            nominal_charge_rate_kw=WARM_RATE,
            nominal_efficiency=1.0,
        ),
        get_cached_prices_func=lambda: [],
        save_learning_data_func=lambda: None,
        update_learning_sensor_func=lambda: None,
        log_func=lambda *a, **k: None,
    )


class TestTheApproximationIsBounded:
    """How far the one model can be from a fine-grained reference.

    Documented bound, per slot, in stored energy::

        (warm_rate - cold_rate) * slot_hours * efficiency

    i.e. ``(warm_rate - cold_rate) * slot_hours * efficiency / capacity * 100``
    SOC points. The reference is 1-minute sub-stepping through the same rate
    curve and the same warming model.
    """

    @staticmethod
    def _fine_grained_end_soc(engine, minutes=15, substep=1.0):
        soc = START_SOC
        temp = START_TEMP
        elapsed = 0.0
        while elapsed < minutes - 1e-9:
            step = min(substep, minutes - elapsed)
            rate = engine.get_charge_rate_for_soc(soc, temp)
            soc += rate * (step / 60.0) / CAPACITY * 100
            temp = engine.predict_temp_after_duration(temp, step)
            elapsed += step
        return soc

    def test_constant_rate_is_within_one_slot_of_rate_difference(self):
        engine = CrossingEngine()
        reference = self._fine_grained_end_soc(engine)
        bound_soc = (
            (WARM_RATE - COLD_RATE) * 0.25 * 1.0 / CAPACITY * 100
        )
        assert bound_soc == pytest.approx(7.5)
        assert abs(reference - CONSTANT_RATE_END_SOC) <= bound_soc + 1e-9
        # It is a real difference, not a vacuous bound.
        assert abs(reference - CONSTANT_RATE_END_SOC) > 1.0

    def test_the_constant_rate_is_the_conservative_side(self):
        """Charging at the start-of-slot rate never over-credits a warming pack."""
        engine = CrossingEngine()
        reference = self._fine_grained_end_soc(engine)
        assert CONSTANT_RATE_END_SOC <= reference + 1e-9


class TestTemperatureStillEvolvesBetweenSlots:
    """Removing the within-slot split must not freeze the pack's temperature."""

    def test_the_projector_still_warms_the_pack_across_slots(self):
        engine = CrossingEngine()
        params = _projection_params()
        first = project_slot_soc(
            soc_start=START_SOC,
            mode=BatteryMode.CHARGE,
            params=params,
            fraction=1.0,
            temp_start=START_TEMP,
            learning_engine=engine,
        )
        assert first.temp_end is not None
        assert first.temp_end > START_TEMP
        second = project_slot_soc(
            soc_start=first.soc_end,
            mode=BatteryMode.CHARGE,
            params=params,
            fraction=1.0,
            temp_start=first.temp_end,
            learning_engine=engine,
        )
        # The second slot starts warm, so it charges at the warm rate: the
        # temperature dependence survives, it just lives between slots.
        gained = (second.soc_end - first.soc_end) / 100.0 * CAPACITY
        assert gained == pytest.approx(WARM_RATE * 0.25, abs=1e-9)
