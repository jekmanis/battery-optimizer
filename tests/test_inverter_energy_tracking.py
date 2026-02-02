import datetime

from battery_optimizer import BatteryOptimizer


class DummyLearningEngine:
    def record_charging(self, **kwargs):
        pass

    def record_discharging(self, **kwargs):
        pass


class MockEnergyOptimizer:
    def __init__(self):
        self.battery_capacity = 10.0
        self.min_soc = 0.0
        self.max_soc = 100.0
        self.battery_avg_cost = 0.20
        self.efficiency = 1.0

        self._last_sig_soc_time = None
        self._last_price_slot = datetime.datetime(2024, 1, 1, 0, 0)
        self._last_soc = 50.0
        self._last_sig_temp = None
        self._stored_energy_kwh = 5.0

        self.learning_engine = DummyLearningEngine()

        self.battery_charge_sensor = "sensor.charge"
        self.battery_discharge_sensor = "sensor.discharge"
        self._energy_sensor_available = False

    def log(self, message: str, level: str = "INFO"):
        pass

    def datetime(self):
        return datetime.datetime(2024, 1, 1, 12, 0)

    def _align_to_slot(self, now: datetime.datetime) -> datetime.datetime:
        return now.replace(minute=0, second=0, microsecond=0)

    def _get_price_for_hour(self, hour: datetime.datetime) -> float:
        return 0.10

    def _get_battery_temp(self):
        return None

    def _save_battery_cost(self):
        pass

    def _save_learning_data(self):
        pass

    def _update_learning_sensor(self):
        pass

    def _get_current_soc(self):
        return 50.0

    def _is_midnight_reset(self, current: float, previous: float, now: datetime.datetime) -> bool:
        return False

    def _get_inverter_energy_readings(self):
        return 1.0, 1.0


MockEnergyOptimizer._process_energy_change = BatteryOptimizer._process_energy_change
MockEnergyOptimizer._on_energy_sensor_change = BatteryOptimizer._on_energy_sensor_change


def test_energy_delta_ignored_when_unavailable():
    optimizer = MockEnergyOptimizer()
    optimizer._energy_sensor_available = False
    calls = {"count": 0}

    def _process_energy_change(*args, **kwargs):
        calls["count"] += 1

    optimizer._process_energy_change = _process_energy_change

    optimizer._on_energy_sensor_change(
        optimizer.battery_charge_sensor, None, 1.0, 1.2, {}
    )

    assert calls["count"] == 0


def test_avg_cost_accumulates_with_multiple_energy_deltas():
    optimizer = MockEnergyOptimizer()
    optimizer._energy_sensor_available = True
    optimizer._stored_energy_kwh = 5.0

    now = datetime.datetime(2024, 1, 1, 1, 0)
    optimizer._process_energy_change(
        energy_kwh=1.0,
        is_charge=True,
        current_soc=50.0,
        now=now,
    )
    optimizer._process_energy_change(
        energy_kwh=1.0,
        is_charge=True,
        current_soc=50.0,
        now=now + datetime.timedelta(minutes=10),
    )

    expected_first = (5.0 * 0.20 + 1.0 * 0.10) / 6.0
    expected_second = (6.0 * expected_first + 1.0 * 0.10) / 7.0

    assert abs(optimizer.battery_avg_cost - expected_second) < 0.0001
