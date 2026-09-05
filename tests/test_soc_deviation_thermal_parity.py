"""A1: the deviation detector must not run a second thermal model.

``SocDeviationDetector._project_charge_completion`` walks the remaining CHARGE
slots forward. It used to call ``project_slot_soc`` WITHOUT a
``temp_projector``, which routed the temperature through
``learning_engine.predict_temp_after_duration`` -- linear learned warming, no
ambient relaxation, no dependence on the power that actually flowed. Every
other consumer of the shared projection uses ``TemperatureProjector``.

Because the detector's walk is multi-slot, the two models diverge and the error
compounds: with learned warming 0.5 C/min, k1 = 0.012/min, k2 = 1.0 C/kWh,
ambient 10 C and a rate curve of 1 kW below 16 C / 4 kW at or above it, three
15-minute CHARGE slots from 10 % and 10 C gave

    _project_charge_completion      32.5 %
    project_schedule_trajectory     17.5 %

and 32.5 % is what drove ``projected_final_soc``, the ``>= max_soc - 5`` guard,
the "projected to reach X%" log line and ``_calculate_extra_charge_slots``.
"""

import datetime

import pytest

from battery_optimizer_lib.models import BatteryMode, ScheduleEntry
from battery_optimizer_lib.soc_deviation import (
    SocDeviationConfig,
    SocDeviationDetector,
)
from battery_optimizer_lib.thermal_model import TemperatureProjector

from battery_optimizer import BatteryOptimizer


BASE = datetime.datetime(2026, 5, 4, 0, 0, 0)
SLOTS = [BASE + datetime.timedelta(minutes=15 * i) for i in range(3)]

CAPACITY = 10.0
START_SOC = 10.0
START_TEMP = 10.0
AMBIENT_C = 10.0

COLD_RATE = 1.0
WARM_RATE = 4.0
THRESHOLD_C = 16.0
LEARNED_WARMING_C_PER_MIN = 0.5
K1 = 0.012
K2 = 1.0


class Engine:
    """Learning-engine double: a temperature-keyed rate and linear warming."""

    storage_efficiency = 1.0

    def get_charge_rate_for_soc(self, soc, battery_temp=None):
        if battery_temp is None:
            return WARM_RATE
        return COLD_RATE if battery_temp < THRESHOLD_C else WARM_RATE

    # The legacy (non-projector) warming model. Linear, unbounded, and blind to
    # the power that actually flowed -- exactly what must no longer be reached.
    def predict_temp_after_duration(
        self, start_temp, duration_minutes, ambient_temp=None, battery_power_kw=None
    ):
        return start_temp + LEARNED_WARMING_C_PER_MIN * duration_minutes

    def predict_temp_after_idle(
        self, start_temp, duration_minutes, ambient_temp=None, default_cooling_rate=K1
    ):
        return start_temp

    # TemperatureProjector coefficient sourcing.
    def get_cooling_rate_estimate(self, start_temp, default=K1):
        return K1

    def get_heating_coefficient(self, default=1.0):
        return K2


class Ambient:
    def refresh(self, force=False):
        return False

    def predict_c(self, dt=None):
        return AMBIENT_C


class _Config:
    battery_capacity = CAPACITY
    efficiency = 1.0
    charge_rate = WARM_RATE
    discharge_rate = 4.0
    export_discharge_rate = 0.0
    inverter_efficiency = 1.0
    slot_minutes = 15


class _App:
    """The smallest surface ``project_schedule_trajectory`` needs."""

    def __init__(self, engine, projector):
        self.config = _Config()
        self.min_soc = 10.0
        self.max_soc = 100.0
        self.learning_engine = engine
        self._temp_projector = projector

    def _predict_load_kw(self, dt):
        return 0.0

    def _predict_pv_kw(self, dt):
        return 0.0

    def _get_local_timezone(self):
        return None


_App.project_schedule_trajectory = BatteryOptimizer.project_schedule_trajectory


def _schedule():
    return {
        slot: ScheduleEntry(time=slot, mode=BatteryMode.CHARGE, reason="t")
        for slot in SLOTS
    }


def _detector(engine, projector):
    return SocDeviationDetector(
        config=SocDeviationConfig(
            slot_minutes=15,
            charge_rate=WARM_RATE,
            discharge_rate=4.0,
            efficiency=1.0,
            battery_capacity=CAPACITY,
            min_soc=10.0,
            max_soc=100.0,
            soc_deviation_threshold=5.0,
            grid_fee=0.0,
            inverter_efficiency=1.0,
        ),
        learning_engine=engine,
        temp_projector=projector,
    )


class TestTheDetectorUsesTheSharedThermalModel:
    def test_charge_completion_agrees_with_the_published_trajectory(self):
        engine = Engine()
        projector = TemperatureProjector(
            learning_engine=engine, ambient_provider=Ambient()
        )
        schedule = _schedule()

        detector_soc = _detector(engine, projector)._project_charge_completion(
            START_SOC, schedule, SLOTS[0], 0.0, START_TEMP, None
        )
        soc_traj, _temp_traj = _App(engine, projector).project_schedule_trajectory(
            schedule, START_SOC, starting_temp=START_TEMP
        )
        trajectory_soc = soc_traj[SLOTS[-1]][1]

        assert detector_soc == pytest.approx(trajectory_soc, abs=1e-6)

    def test_the_shared_model_is_the_one_that_answers(self):
        """Pins the numbers: 17.5 % (shared model) and not 32.5 % (legacy)."""
        engine = Engine()
        projector = TemperatureProjector(
            learning_engine=engine, ambient_provider=Ambient()
        )
        detector_soc = _detector(engine, projector)._project_charge_completion(
            START_SOC, _schedule(), SLOTS[0], 0.0, START_TEMP, None
        )
        assert detector_soc == pytest.approx(17.5, abs=1e-6)

    def test_interpolation_also_uses_the_shared_model(self):
        """``_interpolate_expected_soc`` must carry the projector too.

        Its temperature answer is what a caller of ``expected_soc_at`` compares
        a live reading against.
        """
        engine = Engine()
        projector = TemperatureProjector(
            learning_engine=engine, ambient_provider=Ambient()
        )
        detector = _detector(engine, projector)
        entry = ScheduleEntry(time=SLOTS[0], mode=BatteryMode.CHARGE, reason="t")
        expected = detector._interpolate_expected_soc(
            START_SOC, entry, 1.0, START_SOC, SLOTS[0], START_TEMP, None, None
        )
        soc_traj, _ = _App(engine, projector).project_schedule_trajectory(
            {SLOTS[0]: entry}, START_SOC, starting_temp=START_TEMP
        )
        assert expected == pytest.approx(soc_traj[SLOTS[0]][1], abs=1e-9)


class TestNoSecondThermalModelSurvives:
    def test_soc_projection_has_no_charge_temperature_helper(self):
        from battery_optimizer_lib import soc_projection

        assert not hasattr(soc_projection, "_charge_temp")
        assert not hasattr(soc_projection, "_idle_temp")

    def test_without_a_projector_the_temperature_does_not_move(self):
        """No projector means no temperature model at all -- not a second one."""
        from battery_optimizer_lib.soc_projection import (
            SocProjectionParams,
            project_slot_soc,
        )

        engine = Engine()
        transition = project_slot_soc(
            soc_start=START_SOC,
            mode=BatteryMode.CHARGE,
            params=SocProjectionParams(
                battery_capacity=CAPACITY,
                efficiency=1.0,
                charge_rate=WARM_RATE,
                discharge_rate=4.0,
                inverter_efficiency=1.0,
                min_soc=10.0,
                max_soc=100.0,
                slot_minutes=15,
            ),
            temp_start=START_TEMP,
            learning_engine=engine,
        )
        assert transition.temp_end == pytest.approx(START_TEMP)
