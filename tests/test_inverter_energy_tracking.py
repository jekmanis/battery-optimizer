"""Tests for inverter energy tracking via BatteryCostTracker."""

import datetime

from battery_optimizer_lib import BatteryCostTracker, BatteryCostConfig, BatteryMode


class DummyLearningEngine:
    """Mock learning engine for testing."""

    def record_charging(self, **kwargs):
        pass

    def record_discharging(self, **kwargs):
        pass

    def record_temperature_observation(self, temp):
        pass

    def record_cooling(self, **kwargs):
        pass

    def get_charge_rate_for_soc(self, soc, temp=None):
        return 4.5


def make_cost_tracker(
    battery_capacity=10.0,
    min_soc=0.0,
    max_soc=100.0,
    efficiency=1.0,
    use_inverter_energy_sensors=True,
    initial_cost=0.20,
):
    """Create a BatteryCostTracker with mocked dependencies."""
    config = BatteryCostConfig(
        battery_cost_entity="input_number.battery_avg_cost",
        battery_charge_sensor="sensor.charge",
        battery_discharge_sensor="sensor.discharge",
        use_inverter_energy_sensors=use_inverter_energy_sensors,
        battery_capacity=battery_capacity,
        efficiency=efficiency,
        slot_minutes=60,
        grid_fee=0.05,
        battery_wear_cost=0.0,
        default_cost=initial_cost,
    )

    # Mock state storage
    state = {
        "sensor.charge": "1.0",
        "sensor.discharge": "1.0",
    }
    current_time = datetime.datetime(2024, 1, 1, 12, 0)
    current_soc = [50.0]  # Use list for mutability

    def get_state(entity):
        return state.get(entity)

    def call_service(*args, **kwargs):
        pass

    def get_datetime():
        return current_time

    def get_timezone():
        return None

    def align_to_slot(dt):
        return dt.replace(minute=0, second=0, microsecond=0)

    def get_min_soc():
        return min_soc

    def get_max_soc():
        return max_soc

    def get_current_soc():
        return current_soc[0]

    def get_battery_temp():
        return None

    def get_cached_prices():
        return []

    def save_learning_data():
        pass

    def update_learning_sensor():
        pass

    def log(msg, level="INFO"):
        pass

    learning_engine = DummyLearningEngine()

    tracker = BatteryCostTracker(
        config=config,
        get_state_func=get_state,
        call_service_func=call_service,
        get_datetime_func=get_datetime,
        get_timezone_func=get_timezone,
        align_to_slot_func=align_to_slot,
        get_min_soc_func=get_min_soc,
        get_max_soc_func=get_max_soc,
        get_current_soc_func=get_current_soc,
        get_battery_temp_func=get_battery_temp,
        learning_engine=learning_engine,
        get_cached_prices_func=get_cached_prices,
        save_learning_data_func=save_learning_data,
        update_learning_sensor_func=update_learning_sensor,
        log_func=log,
    )

    # Initialize and set up for testing
    tracker.initialize()
    tracker._avg_cost = initial_cost
    tracker._cost_from_fallback = False
    tracker._stored_energy_kwh = 5.0
    tracker._energy_sensor_available = use_inverter_energy_sensors

    return tracker, state, current_soc


def test_energy_delta_ignored_when_unavailable():
    """Energy changes should be ignored when sensors are unavailable."""
    tracker, state, current_soc = make_cost_tracker(use_inverter_energy_sensors=False)

    initial_cost = tracker.avg_cost
    tracker.on_energy_sensor_change("sensor.charge", "1.0", "1.2")

    # Cost should not change when sensors are unavailable
    assert tracker.avg_cost == initial_cost


def test_avg_cost_accumulates_with_multiple_energy_deltas():
    """Multiple energy changes should accumulate properly in weighted average."""
    tracker, state, current_soc = make_cost_tracker(
        battery_capacity=10.0,
        min_soc=0.0,
        initial_cost=0.20,
    )

    # First charge: 5.0 kWh @ 0.20 + 1.0 kWh @ price
    # Since we don't have cached prices, it will use the current avg cost
    now = datetime.datetime(2024, 1, 1, 1, 0)
    tracker._process_energy_change(
        energy_kwh=1.0,
        is_charge=True,
        current_soc=50.0,
        now=now,
    )

    # The price lookup returns None (no cached prices), so it uses avg_cost (0.20)
    # New cost = (5.0 * 0.20 + 1.0 * 0.20) / 6.0 = 0.20
    expected_first = (5.0 * 0.20 + 1.0 * 0.20) / 6.0

    tracker._process_energy_change(
        energy_kwh=1.0,
        is_charge=True,
        current_soc=50.0,
        now=now + datetime.timedelta(minutes=10),
    )

    # Second charge: stored_energy is now 6.0 kWh
    # New cost = (6.0 * expected_first + 1.0 * expected_first) / 7.0 = expected_first
    expected_second = (6.0 * expected_first + 1.0 * expected_first) / 7.0

    assert abs(tracker.avg_cost - expected_second) < 0.0001
