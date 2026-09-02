"""
``SocDeviationDetector._project_charge_completion`` must stay physical.

Production 2026-09-02, 05:00-05:15 (one 15-minute CHARGE slot):

    05:01:11  SOC behind by 90.0% during CHARGE (actual=10.0%, expected=100.0%),
              but projected to reach 21894.1% ... - skipping recalculation
    ...
    05:14:33  ... projected to reach 716.7% ... - skipping recalculation

Twelve times, the projection decaying linearly with the minutes left in the
slot. The projection summed ``rate * efficiency * minutes/60`` over the
remaining charge slots and added it to the current SOC with NO cap, and the rate
came from a bucket whose median was 14308.71 kW. The caller's guard is
``projected_final_soc >= max_soc - 5``, so it could never be false: catch-up
charging was structurally unreachable during CHARGE.

Two fixes are pinned here:
  * the projection goes through the shared ``project_slot_soc`` model and is
    clamped to ``max_soc``;
  * the learned rate it uses is bounded by the learning engine.
"""

import datetime

import pytest

from battery_optimizer_lib import BatteryLearningEngine, BatteryMode, ScheduleEntry
from battery_optimizer_lib.soc_deviation import (
    SocDeviationConfig,
    SocDeviationDetector,
)

CAPACITY = 14.3
SLOT = 15

# The live 0-25%/>20C bucket that produced 14308.71 kW.
LIVE_POISON = [2.806, 34535.687, 44653.932, 14308.71, 5.959]


def _config(**overrides) -> SocDeviationConfig:
    base = dict(
        slot_minutes=SLOT,
        charge_rate=4.5,
        discharge_rate=5.9,
        efficiency=0.95,
        battery_capacity=CAPACITY,
        min_soc=10.0,
        max_soc=100.0,
        soc_deviation_threshold=4.0,
        grid_fee=0.05,
        inverter_efficiency=0.97,
    )
    base.update(overrides)
    return SocDeviationConfig(**base)


def _poisoned_engine() -> BatteryLearningEngine:
    """A learning engine whose in-memory stats carry the live poison.

    Injected directly (not through ``load_from_json``) so the test proves the
    detector is safe even if the sanitising load path is bypassed.
    """
    engine = BatteryLearningEngine(
        battery_capacity_kwh=CAPACITY,
        nominal_charge_rate_kw=4.5,
        log_func=lambda *a, **k: None,
    )
    engine.stats.charge_rates_by_soc["0-25"] = list(LIVE_POISON)
    engine.stats.charge_rates_by_soc_temp["0-25"] = {">20": list(LIVE_POISON)}
    return engine


def _schedule(start: datetime.datetime, modes) -> dict:
    return {
        start + datetime.timedelta(minutes=SLOT * i): ScheduleEntry(
            time=start + datetime.timedelta(minutes=SLOT * i), mode=mode, reason="test"
        )
        for i, mode in enumerate(modes)
    }


class TestProjectionIsClamped:
    def test_projection_never_exceeds_max_soc(self):
        detector = SocDeviationDetector(_config(), learning_engine=_poisoned_engine())
        slot = datetime.datetime(2026, 9, 2, 5, 0)
        # 20 CHARGE slots is far more than the pack can absorb.
        schedule = _schedule(slot, [BatteryMode.CHARGE] * 20)

        projected = detector._project_charge_completion(
            current_soc=10.0,
            schedule=schedule,
            current_slot=slot,
            fraction=0.07,
            current_temp=21.9,
            local_tz=None,
        )
        # The old code added the whole (unbounded, unclamped) energy sum and
        # returned 21894.1% for this shape.
        assert projected == pytest.approx(100.0)

    def test_projection_uses_the_bounded_rate(self):
        detector = SocDeviationDetector(_config(), learning_engine=_poisoned_engine())
        slot = datetime.datetime(2026, 9, 2, 5, 0)
        schedule = _schedule(slot, [BatteryMode.CHARGE] * 4)

        projected = detector._project_charge_completion(
            current_soc=10.0,
            schedule=schedule,
            current_slot=slot,
            fraction=0.07,
            current_temp=21.9,
            local_tz=None,
        )
        # 4 slots x <= 9.0 kW x 0.95 x 0.25 h = <= 8.55 kWh = <= 59.8% SOC.
        assert 10.0 < projected <= 10.0 + 60.0

    def test_single_remaining_slot_cannot_fill_from_ten_percent(self):
        """The real 05:00 case: one CHARGE slot, 14 of 15 minutes left."""
        detector = SocDeviationDetector(_config(), learning_engine=_poisoned_engine())
        slot = datetime.datetime(2026, 9, 2, 5, 0)
        schedule = _schedule(
            slot, [BatteryMode.CHARGE, BatteryMode.HOLD, BatteryMode.HOLD]
        )

        projected = detector._project_charge_completion(
            current_soc=10.0,
            schedule=schedule,
            current_slot=slot,
            fraction=1.0 / 15.0,
            current_temp=21.9,
            local_tz=None,
        )
        # Bounded rate <= 9.0 kW over 14 min => at most ~14% SOC gain.
        assert 10.0 < projected < 30.0

    def test_guard_is_not_trivially_satisfied_anymore(self):
        """The full production scenario must now TRIGGER a recalculation."""
        config = _config()
        detector = SocDeviationDetector(config, learning_engine=_poisoned_engine())
        slot = datetime.datetime(2026, 9, 2, 5, 0)
        now = datetime.datetime(2026, 9, 2, 5, 1)
        schedule = _schedule(
            slot, [BatteryMode.CHARGE, BatteryMode.HOLD, BatteryMode.HOLD]
        )

        result = detector.check_deviation(
            current_soc=10.0,
            schedule=schedule,
            expected_soc_schedule={slot: 100.0},
            now=now,
            current_slot=slot,
            local_tz=None,
            current_temp=21.9,
            predict_load_kw=lambda _dt: 0.64,
            predict_pv_kw=lambda _dt: 0.0,
            expected_soc_anchor=None,
        )
        assert result.should_recalculate is True
        assert not any(
            "skipping recalculation" in msg for msg in result.log_messages
        )

    def test_genuinely_recoverable_shortfall_still_skips(self):
        """The skip must survive for the case it was written for."""
        detector = SocDeviationDetector(
            _config(), learning_engine=None
        )
        slot = datetime.datetime(2026, 9, 2, 5, 0)
        now = datetime.datetime(2026, 9, 2, 5, 1)
        # 8 full CHARGE slots at the nominal 4.5 kW: 8 * 4.5 * 0.95 * 0.25 kWh
        # = 8.55 kWh = 59.8% SOC, comfortably filling from 45%.
        schedule = _schedule(slot, [BatteryMode.CHARGE] * 8)

        result = detector.check_deviation(
            current_soc=45.0,
            schedule=schedule,
            expected_soc_schedule={slot: 55.0},
            now=now,
            current_slot=slot,
            local_tz=None,
            current_temp=None,
            predict_load_kw=lambda _dt: 0.64,
            predict_pv_kw=lambda _dt: 0.0,
            expected_soc_anchor=None,
        )
        assert result.should_recalculate is False
        assert any("skipping recalculation" in msg for msg in result.log_messages)


class TestProjectionUsesTheSharedModel:
    def test_matches_project_slot_soc_slot_by_slot(self):
        from battery_optimizer_lib import SocProjectionParams, project_slot_soc

        engine = BatteryLearningEngine(
            battery_capacity_kwh=CAPACITY,
            nominal_charge_rate_kw=4.5,
            log_func=lambda *a, **k: None,
        )
        for _ in range(5):
            engine.record_charging(
                soc_start=20.0, soc_end=24.0, duration_minutes=15.0, battery_temp=22.0
            )

        config = _config()
        detector = SocDeviationDetector(config, learning_engine=engine)
        slot = datetime.datetime(2026, 9, 2, 5, 0)
        schedule = _schedule(slot, [BatteryMode.CHARGE] * 3)

        projected = detector._project_charge_completion(
            current_soc=20.0,
            schedule=schedule,
            current_slot=slot,
            fraction=0.0,
            current_temp=22.0,
            local_tz=None,
        )

        params = SocProjectionParams(
            battery_capacity=CAPACITY,
            efficiency=config.efficiency,
            charge_rate=config.charge_rate,
            discharge_rate=config.discharge_rate,
            inverter_efficiency=config.inverter_efficiency,
            min_soc=config.min_soc,
            max_soc=config.max_soc,
            slot_minutes=SLOT,
        )
        soc, temp = 20.0, 22.0
        for i in range(3):
            transition = project_slot_soc(
                soc_start=soc,
                mode=BatteryMode.CHARGE,
                params=params,
                temp_start=temp,
                learning_engine=engine,
                rate_lookup_soc=20.0,
            )
            soc, temp = transition.soc_end, transition.temp_end

        assert projected == pytest.approx(soc)
