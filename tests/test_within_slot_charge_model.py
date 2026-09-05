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


def _fine_grained_end_soc(
    rate_fn,
    soc_start=START_SOC,
    temp_start=START_TEMP,
    warm_per_min=0.0,
    minutes=15.0,
    substep=1.0,
    efficiency=1.0,
    capacity=CAPACITY,
):
    """1-minute sub-stepped truth for a slot, through the same rate curve."""
    soc = soc_start
    temp = temp_start
    elapsed = 0.0
    while elapsed < minutes - 1e-9:
        step = min(substep, minutes - elapsed)
        rate = rate_fn(soc, temp)
        soc += rate * (step / 60.0) * efficiency / capacity * 100
        temp += warm_per_min * step
        elapsed += step
    return soc


def _rates_visited(
    rate_fn,
    soc_start=START_SOC,
    temp_start=START_TEMP,
    warm_per_min=0.0,
    minutes=15.0,
    substep=1.0,
    efficiency=1.0,
    capacity=CAPACITY,
):
    """Every rate the sub-stepped truth evaluates during the slot."""
    soc = soc_start
    temp = temp_start
    elapsed = 0.0
    rates = []
    while elapsed < minutes - 1e-9:
        step = min(substep, minutes - elapsed)
        rate = rate_fn(soc, temp)
        rates.append(rate)
        soc += rate * (step / 60.0) * efficiency / capacity * 100
        temp += warm_per_min * step
        elapsed += step
    return rates


def _span_end_soc(
    rate_fn,
    soc_start=START_SOC,
    temp_start=START_TEMP,
    minutes=15.0,
    efficiency=1.0,
    capacity=CAPACITY,
    max_soc=100.0,
):
    """The model, computed straight from the documented rule.

    Kept independent of the implementation so the scenarios below measure the
    RULE, not ``charge_rate_for_span``'s own arithmetic; a separate test asserts
    the shipped helper agrees with this.
    """
    duration_h = minutes / 60.0
    r0 = rate_fn(soc_start, temp_start)
    reached = min(
        max_soc, soc_start + r0 * duration_h * efficiency / capacity * 100
    )
    rate = min(r0, rate_fn(reached, temp_start))
    return soc_start + rate * duration_h * efficiency / capacity * 100


def _constant_end_soc(
    rate_fn,
    soc_start=START_SOC,
    temp_start=START_TEMP,
    minutes=15.0,
    efficiency=1.0,
    capacity=CAPACITY,
):
    """The old model: the start-SOC, start-temperature rate for the whole slot."""
    return (
        soc_start
        + rate_fn(soc_start, temp_start) * (minutes / 60.0) * efficiency / capacity * 100
    )


def _soc_taper(boundary, fast=4.0, slow=1.0):
    """A learned SOC-taper bucket boundary (the engine uses 25/50/75/90 %)."""

    def _fn(soc, temp=None):
        return slow if soc >= boundary else fast

    return _fn


class TestTheApproximationIsBounded:
    """How far the one model can be from a 1-minute sub-stepped reference.

    Two different statements, and they are different kinds of statement:

    * an IDENTITY of the model -- the slot runs at one rate drawn from the
      rates the truth visits, and the truth is a duration-weighted average of
      those same rates, so the gap cannot exceed
      ``(max rate visited - min rate visited) * slot_hours * efficiency``.
      Nothing can violate it and it proves nothing on its own;
    * the falsifiable DIRECTION claims, which hold only under their stated
      monotonicity assumptions (see ``TestTheRateSpanErrsConservative`` and
      ``TestTheDirectionClaimNeedsMonotonicity``).

    The bound that used to be documented here -- ``(rate at end - rate at
    start) * slot_hours * efficiency`` -- is not an identity and is violated by
    an order of magnitude by a non-monotonic curve; that counterexample is
    pinned below.
    """

    def test_the_identity_bound_holds_for_the_crossing_curve(self):
        engine = CrossingEngine()
        rate_fn = engine.get_charge_rate_for_soc

        def warming(soc, temp):
            return rate_fn(soc, temp)

        reference = _fine_grained_end_soc(
            warming, warm_per_min=WARMING_C_PER_MIN
        )
        visited = _rates_visited(warming, warm_per_min=WARMING_C_PER_MIN)
        bound_soc = (max(visited) - min(visited)) * 0.25 * 1.0 / CAPACITY * 100
        assert bound_soc == pytest.approx(7.5)
        assert abs(reference - CONSTANT_RATE_END_SOC) <= bound_soc + 1e-9

    def test_the_identity_bound_holds_for_the_non_monotonic_curve(self):
        """The only bound that survives a non-monotonic rate curve."""
        reference = _fine_grained_end_soc(
            _BANDED_RATE, warm_per_min=BAND_WARMING_C_PER_MIN
        )
        visited = _rates_visited(
            _BANDED_RATE, warm_per_min=BAND_WARMING_C_PER_MIN
        )
        bound_soc = (max(visited) - min(visited)) * 0.25 * 1.0 / CAPACITY * 100
        model = _span_end_soc(_BANDED_RATE)
        assert bound_soc == pytest.approx(12.5)
        assert abs(reference - model) <= bound_soc + 1e-9


# The A3 counterexample: a NON-MONOTONIC rate curve. 1.0 kW below 14 C,
# 6.0 kW from 14 C to 20 C, 1.2 kW above 20 C, pack warming 1 C/min from 10 C.
# Sub-stepped truth 17.667 %, model 12.5 % -- an error of 5.17 SOC points,
# against the 0.5 points the "rate at the end minus rate at the start" reading
# of the old bound allows.
BAND_WARMING_C_PER_MIN = 1.0


def _BANDED_RATE(soc, temp):
    if temp is None:
        return 6.0
    if temp < 14.0:
        return 1.0
    if temp < 20.0:
        return 6.0
    return 1.2


class TestTheDirectionClaimNeedsMonotonicity:
    """The falsifiable half, and the case that falsifies the general version."""

    def test_conservative_when_the_rate_is_monotone_over_the_span(self):
        """Warming pack, rate non-decreasing in temperature: never over-credits."""
        engine = CrossingEngine()
        reference = _fine_grained_end_soc(
            engine.get_charge_rate_for_soc, warm_per_min=WARMING_C_PER_MIN
        )
        model = _span_end_soc(engine.get_charge_rate_for_soc)
        assert model <= reference + 1e-9

    def test_a_non_monotonic_curve_over_credits_and_only_the_identity_holds(self):
        """DOCUMENTED counterexample -- pinned, not wished away.

        The old bound (rate at the end minus rate at the start) allows 0.5 SOC
        points here. The real error is 5.17.
        """
        reference = _fine_grained_end_soc(
            _BANDED_RATE, warm_per_min=BAND_WARMING_C_PER_MIN
        )
        model = _span_end_soc(_BANDED_RATE)
        assert reference == pytest.approx(17.6667, abs=1e-3)
        assert model == pytest.approx(12.5, abs=1e-9)
        old_bound = (
            abs(_BANDED_RATE(model, START_TEMP + 15 * BAND_WARMING_C_PER_MIN)
                - _BANDED_RATE(START_SOC, START_TEMP))
            * 0.25 / CAPACITY * 100
        )
        assert old_bound == pytest.approx(0.5, abs=1e-9)
        assert abs(reference - model) > 10 * old_bound

    def test_a_cooling_pack_can_be_over_credited_within_the_stated_bound(self):
        """The one direction the span rule does NOT fix, stated out loud.

        The rate is evaluated at the temperature the slot STARTS at, so a pack
        that cools while charging is credited at the warmer (faster) rate. The
        over-credit is bounded by
        ``(rate(T_start) - rate(T_end)) * slot_hours * efficiency``.
        """
        engine = CrossingEngine()
        cooling = -0.8

        reference = _fine_grained_end_soc(
            engine.get_charge_rate_for_soc,
            temp_start=THRESHOLD_C,
            warm_per_min=cooling,
        )
        model = _span_end_soc(engine.get_charge_rate_for_soc, temp_start=THRESHOLD_C)
        over_credit = model - reference
        assert over_credit > 0  # it really does over-credit here
        bound = (
            (
                engine.get_charge_rate_for_soc(START_SOC, THRESHOLD_C)
                - engine.get_charge_rate_for_soc(
                    START_SOC, THRESHOLD_C + 15 * cooling
                )
            )
            * 0.25
            / CAPACITY
            * 100
        )
        assert over_credit <= bound + 1e-9


class TestTheRateSpanErrsConservative:
    """A2: the rate is evaluated over the SPAN, not frozen at the start SOC.

    Freezing the rate at the start SOC over-credits every slot that crosses a
    learned SOC-taper boundary, and ``replay_plan`` could not catch it because
    it evaluated the same frozen model. The rule now is: evaluate the rate at
    the start SOC and at the end SOC the start rate would reach (capped at
    ``max_soc``) and use the MINIMUM of the two for the whole slot.
    """

    def test_taper_at_90_percent(self):
        rate_fn = _soc_taper(90.0)
        truth = _fine_grained_end_soc(rate_fn, soc_start=88.0)
        old = _constant_end_soc(rate_fn, soc_start=88.0)
        new = _span_end_soc(rate_fn, soc_start=88.0)
        assert truth == pytest.approx(92.0, abs=1e-9)
        assert old == pytest.approx(98.0, abs=1e-9)  # +6 SOC points of credit
        assert new <= truth + 1e-9
        assert new == pytest.approx(90.5, abs=1e-9)

    def test_taper_at_25_percent(self):
        rate_fn = _soc_taper(25.0)
        truth = _fine_grained_end_soc(rate_fn, soc_start=22.0)
        old = _constant_end_soc(rate_fn, soc_start=22.0)
        new = _span_end_soc(rate_fn, soc_start=22.0)
        assert truth == pytest.approx(27.0, abs=1e-9)
        assert old == pytest.approx(32.0, abs=1e-9)  # +5 SOC points of credit
        assert new <= truth + 1e-9

    def test_a_cooling_pack_whose_curve_also_tapers(self):
        """Cooling 0.8 C/min, learned curve keyed on temperature AND SOC."""

        def rate_fn(soc, temp):
            if temp is None or temp < 14.0:
                return 1.0
            return 1.0 if soc >= 25.0 else 4.0

        truth = _fine_grained_end_soc(
            rate_fn, soc_start=22.0, temp_start=16.0, warm_per_min=-0.8
        )
        old = _constant_end_soc(rate_fn, soc_start=22.0, temp_start=16.0)
        new = _span_end_soc(rate_fn, soc_start=22.0, temp_start=16.0)
        assert old > truth + 1e-9  # the frozen rate over-credits
        assert new <= truth + 1e-9

    def test_the_shipped_helper_implements_the_rule(self):
        from battery_optimizer_lib.slot_energy import charge_rate_for_span

        rate_fn = _soc_taper(90.0)
        assert charge_rate_for_span(
            rate_fn,
            soc_start=88.0,
            temp=None,
            duration_h=0.25,
            efficiency=1.0,
            capacity=CAPACITY,
            max_soc=100.0,
        ) == pytest.approx(1.0)
        # No boundary in the span: the start rate stands.
        assert charge_rate_for_span(
            rate_fn,
            soc_start=10.0,
            temp=None,
            duration_h=0.25,
            efficiency=1.0,
            capacity=CAPACITY,
            max_soc=100.0,
        ) == pytest.approx(4.0)

    def test_the_span_is_capped_at_max_soc(self):
        """The reachable end SOC is a physical SOC, not an extrapolation."""
        from battery_optimizer_lib.slot_energy import charge_rate_for_span

        # A curve that only tapers ABOVE 100 % must never be sampled there.
        def rate_fn(soc, temp=None):
            return 0.0 if soc > 100.0 else 4.0

        assert charge_rate_for_span(
            rate_fn,
            soc_start=99.0,
            temp=None,
            duration_h=0.25,
            efficiency=1.0,
            capacity=CAPACITY,
            max_soc=100.0,
        ) == pytest.approx(4.0)


class TaperEngine:
    """Learning-engine double with a 4 kW -> 1 kW taper at 90 % SOC."""

    storage_efficiency = 1.0

    def get_charge_rate_for_soc(self, soc, battery_temp=None):
        return 1.0 if soc >= 90.0 else 4.0

    def predict_temp_after_duration(self, start_temp, duration_minutes, **kw):
        return start_temp

    def predict_temp_after_idle(self, start_temp, duration_minutes, **kw):
        return start_temp


class TestEveryConsumerAgreesOnATaperCrossingSlot:
    """One rule, one answer: 88 % -> 90.5 % in every consumer, to 1e-9."""

    START = 88.0
    EXPECTED = 90.5
    # A high floor so the pack cannot simply serve slot 1 from what it already
    # holds: the DP has to buy the cheap slot's energy, which is the transition
    # under test. Nothing else here depends on the floor.
    MIN_SOC = 85.0

    def test_dp_optimize(self):
        engine = TaperEngine()
        opt = DPOptimizer(
            config=_dp_config(min_soc=self.MIN_SOC),
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
            current_soc=self.START,
            current_temp=START_TEMP,
        )
        assert result.schedule[SLOT_0].mode == BatteryMode.CHARGE
        assert result.soc_trajectory[SLOT_0][1] == pytest.approx(
            self.EXPECTED, abs=1e-9
        )

    def test_project_slot_soc(self):
        engine = TaperEngine()
        transition = project_slot_soc(
            soc_start=self.START,
            mode=BatteryMode.CHARGE,
            params=_projection_params(),
            temp_start=START_TEMP,
            learning_engine=engine,
        )
        assert transition.soc_end == pytest.approx(self.EXPECTED, abs=1e-9)

    def test_replay_plan(self):
        engine = TaperEngine()
        schedule = {
            SLOT_0: ScheduleEntry(time=SLOT_0, mode=BatteryMode.CHARGE, reason="t"),
        }
        replay = replay_plan(
            schedule=schedule,
            config=_dp_config(min_soc=self.MIN_SOC),
            starting_soc=self.START,
            predict_load_kw=lambda dt: 0.0,
            predict_pv_kw=lambda dt: 0.0,
            charge_rate_for=lambda slot, soc, temp: engine.get_charge_rate_for_soc(
                soc, temp
            ),
            starting_temp=START_TEMP,
        )
        assert replay.by_slot[SLOT_0].soc_end == pytest.approx(
            self.EXPECTED, abs=1e-9
        )

    def test_partial_slot_lookahead(self):
        """The DP's separate first-slot candidate path uses the same rule."""
        engine = TaperEngine()
        opt = DPOptimizer(
            config=_dp_config(min_soc=self.MIN_SOC),
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
            current_soc=self.START,
            current_temp=START_TEMP,
            minutes_into_slot=7.5,
        )
        # Half a slot at the span rate (1 kW): 88 % + 0.125 kWh -> 89.25 %.
        assert result.schedule[SLOT_0].mode == BatteryMode.CHARGE
        assert result.soc_trajectory[SLOT_0][1] == pytest.approx(89.25, abs=1e-9)


class TestTemperatureStillEvolvesBetweenSlots:
    """Removing the within-slot split must not freeze the pack's temperature.

    It evolves through the SHARED projector, and only through it. This test
    used to pass a bare ``learning_engine`` and rely on ``project_slot_soc``
    falling through to ``predict_temp_after_duration`` — the second thermal
    model that the deviation detector was also reaching (A1). There is no such
    fallback any more: no projector, no temperature movement.
    """

    @staticmethod
    def _projector():
        from battery_optimizer_lib.thermal_model import TemperatureProjector

        class _Ambient:
            def predict_c(self, dt=None):
                return 30.0

        return TemperatureProjector(
            learning_engine=None,
            ambient_provider=_Ambient(),
            default_cooling_rate=0.1,
            default_heating_c_per_kwh=0.0,
        )

    def test_without_a_projector_the_temperature_is_unchanged(self):
        engine = CrossingEngine()
        transition = project_slot_soc(
            soc_start=START_SOC,
            mode=BatteryMode.CHARGE,
            params=_projection_params(),
            fraction=1.0,
            temp_start=START_TEMP,
            learning_engine=engine,
        )
        assert transition.temp_end == pytest.approx(START_TEMP)

    def test_the_projector_still_warms_the_pack_across_slots(self):
        engine = CrossingEngine()
        params = _projection_params()
        projector = self._projector()
        first = project_slot_soc(
            soc_start=START_SOC,
            mode=BatteryMode.CHARGE,
            params=params,
            fraction=1.0,
            temp_start=START_TEMP,
            learning_engine=engine,
            temp_projector=projector,
            slot_time=SLOT_0,
        )
        assert first.temp_end is not None
        assert first.temp_end > THRESHOLD_C
        second = project_slot_soc(
            soc_start=first.soc_end,
            mode=BatteryMode.CHARGE,
            params=params,
            fraction=1.0,
            temp_start=first.temp_end,
            learning_engine=engine,
            temp_projector=projector,
            slot_time=SLOT_1,
        )
        # The second slot starts warm, so it charges at the warm rate: the
        # temperature dependence survives, it just lives between slots.
        gained = (second.soc_end - first.soc_end) / 100.0 * CAPACITY
        assert gained == pytest.approx(WARM_RATE * 0.25, abs=1e-9)
