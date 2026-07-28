"""
Tests for thermal data collection through BatteryCostTracker.

DEFECT 5 had a self-reinforcing loop: ``record_cooling`` fell back to
``min(recent battery temps)`` for ambient, which in summer is ~the current
battery temperature, so ``temp_start <= ambient_temp`` discarded nearly every
observation. ``temp_cooling_rates`` therefore stayed empty forever and the
default 0.012/min was always used.

DEFECT 6 had a data problem underneath it: ``record_discharging`` took no
temperatures at all and neither call site passed any, so the learning data
contained ZERO discharge thermal samples and the heating coefficient could not
be fitted.
"""

import datetime

import pytest

from battery_optimizer_lib import (
    BatteryCostConfig,
    BatteryCostTracker,
    BatteryLearningEngine,
)


def make_tracker(learning_engine, ambient_temp=None, battery_temp=33.0):
    """Cost tracker in SOC-fallback mode with a real learning engine."""
    config = BatteryCostConfig(
        battery_cost_entity="input_number.battery_avg_cost",
        battery_charge_sensor="sensor.charge",
        battery_discharge_sensor="sensor.discharge",
        use_inverter_energy_sensors=False,
        battery_capacity=14.3,
        efficiency=1.0,
        slot_minutes=60,
        grid_fee=0.05,
        battery_wear_cost=0.0,
        default_cost=0.20,
    )
    now = {"t": datetime.datetime(2026, 7, 27, 20, 0)}

    tracker = BatteryCostTracker(
        config=config,
        get_state_func=lambda entity: None,
        call_service_func=lambda *a, **k: None,
        get_datetime_func=lambda: now["t"],
        get_timezone_func=lambda: None,
        align_to_slot_func=lambda dt: dt.replace(minute=0, second=0, microsecond=0),
        get_min_soc_func=lambda: 10.0,
        get_max_soc_func=lambda: 100.0,
        get_current_soc_func=lambda: 50.0,
        get_battery_temp_func=lambda: battery_temp,
        learning_engine=learning_engine,
        get_cached_prices_func=lambda: [],
        save_learning_data_func=lambda: None,
        update_learning_sensor_func=lambda: None,
        log_func=lambda msg, level="INFO": None,
        get_ambient_temp_func=(
            None if ambient_temp is None else (lambda: ambient_temp)
        ),
    )
    tracker._avg_cost = 0.20
    tracker._cost_from_fallback = False
    tracker._energy_sensor_available = False
    return tracker, now


@pytest.fixture
def engine():
    return BatteryLearningEngine(
        battery_capacity_kwh=14.3,
        nominal_charge_rate_kw=4.5,
        log_func=lambda *a, **k: None,
    )


class TestDischargeThermalSamples:
    def test_soc_fallback_discharge_records_temperatures(self, engine):
        """A discharge must now produce a thermal sample."""
        tracker, now = make_tracker(engine, ambient_temp=27.0, battery_temp=33.5)

        tracker.process_soc_change(80.0)
        tracker._last_sig_temp = 33.0
        now["t"] += datetime.timedelta(minutes=30)
        tracker.process_soc_change(70.0)

        assert len(engine.stats.thermal_samples) == 1
        sample = engine.stats.thermal_samples[0]
        assert sample[0] == 33.0     # temp_start
        assert sample[1] == 33.5     # temp_end
        assert sample[2] == pytest.approx(30.0)
        assert sample[3] > 0.0       # non-zero discharge power
        assert sample[4] == 27.0     # injected ambient

    def test_discharge_power_matches_energy_over_time(self, engine):
        tracker, now = make_tracker(engine, ambient_temp=27.0, battery_temp=33.5)

        tracker.process_soc_change(80.0)
        tracker._last_sig_temp = 33.0
        now["t"] += datetime.timedelta(minutes=60)
        tracker.process_soc_change(60.0)

        # 20% of 14.3 kWh in one hour = 2.86 kW
        assert engine.stats.thermal_samples[0][3] == pytest.approx(2.86, abs=0.01)


class TestCoolingObservationsUseRealAmbient:
    def test_injected_ambient_lets_cooling_be_learned(self, engine):
        """With a real ambient the cooling observation is accepted."""
        from battery_optimizer_lib import BatteryMode

        tracker, now = make_tracker(engine, ambient_temp=27.0, battery_temp=33.0)
        tracker._idle_start_time = now["t"]
        tracker._idle_start_temp = 35.0
        now["t"] += datetime.timedelta(minutes=60)
        tracker._get_battery_temp = lambda: 33.0

        tracker.on_mode_transition(BatteryMode.HOLD, BatteryMode.CHARGE, current_soc=50.0)

        assert engine.stats.temp_cooling_rates, "cooling observation was discarded"
        assert engine.get_cooling_rate(35.0) is None or engine.get_cooling_rate(35.0) > 0

    def test_without_ambient_the_summer_observation_is_discarded(self, engine):
        """Documents the self-reinforcing loop the injection breaks.

        The battery has never been colder than 33C, so min(recent) = 33C and
        the 35C -> 33C cooling observation is rejected as "already at ambient".
        """
        from battery_optimizer_lib import BatteryMode

        for temp in [33.0, 33.5, 34.0, 35.0]:
            engine.record_temperature_observation(temp)

        tracker, now = make_tracker(engine, ambient_temp=None, battery_temp=33.0)
        tracker._idle_start_time = now["t"]
        tracker._idle_start_temp = 35.0
        now["t"] += datetime.timedelta(minutes=60)

        tracker.on_mode_transition(BatteryMode.HOLD, BatteryMode.CHARGE, current_soc=50.0)

        assert engine.stats.temp_cooling_rates == {}

    def test_real_ambient_service_fallback_accepts_a_summer_cooling_sample(self, engine):
        """The injected ambient must be COLDER than the pack, not hotter.

        With the diurnal fallback anchored the wrong way round (mean = min + A)
        the service returned ~36 C at 20:00 for a pack whose rolling minimum was
        31 C, so ``record_cooling`` hit ``temp_end < ambient_temp`` and dropped
        the sample — the injection made cooling learning strictly WORSE than the
        old min(recent) fallback it replaced.
        """
        from battery_optimizer_lib import BatteryMode
        from battery_optimizer_lib.ambient_service import (
            AmbientServiceConfig,
            AmbientTemperatureService,
        )

        for temp in [31.0, 31.5, 32.0, 33.0]:
            engine.record_temperature_observation(temp)

        at = {"t": datetime.datetime(2026, 7, 27, 20, 0)}
        ambient = AmbientTemperatureService(
            config=AmbientServiceConfig(
                diurnal_amplitude_c=4.0, diurnal_peak_hour=15.0, slot_minutes=60
            ),
            get_datetime_func=lambda: at["t"],
            get_timezone_func=lambda: None,
            min_temp_provider=engine.get_estimated_ambient_min_temp,
        )
        assert ambient.predict_c(at["t"]) < 31.0

        tracker, now = make_tracker(engine, battery_temp=31.0)
        tracker._get_ambient_temp = lambda: ambient.predict_c(now["t"])
        at["t"] = now["t"]
        tracker._idle_start_time = now["t"]
        tracker._idle_start_temp = 33.0
        now["t"] += datetime.timedelta(minutes=60)
        at["t"] = now["t"]

        tracker.on_mode_transition(BatteryMode.HOLD, BatteryMode.CHARGE, current_soc=50.0)

        assert engine.stats.temp_cooling_rates, "cooling observation was discarded"
