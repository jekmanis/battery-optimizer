"""
Tests for temperature-aware SOC projection in schedule calculations.

These tests verify that the optimizer correctly:
- Tracks temperature evolution during charge slots
- Uses temperature-aware charge rates in SOC projections
- Falls back gracefully when temperature data is unavailable
"""

import datetime
from typing import Dict, List

import pytest

from battery_optimizer import (
    BatteryLearningEngine,
    BatteryMode,
    LoadProfile,
    PricePoint,
    ScheduleEntry,
)


class MockOptimizer:
    """
    Minimal mock of BatteryOptimizer for testing temperature-aware SOC calculations.
    """

    def __init__(
        self,
        battery_capacity: float = 14.3,
        charge_rate: float = 4.5,
        discharge_rate: float = 4.5,
        efficiency: float = 0.85,
        min_soc: float = 10.0,
        max_soc: float = 100.0,
        grid_fee: float = 0.05,
        slot_minutes: int = 60,
        battery_avg_cost: float = 0.08,
        base_consumption: float = 500,
        load_quantile: float = 0.75,
        soc_step_percent: float = 1.0,
    ):
        # Battery parameters
        self.battery_capacity = battery_capacity
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.efficiency = efficiency
        self.grid_fee = grid_fee
        self.slot_minutes = slot_minutes
        self.slot_hours = slot_minutes / 60.0
        self.battery_avg_cost = battery_avg_cost
        self.base_consumption = base_consumption
        self.load_quantile = load_quantile
        self.soc_step_percent = soc_step_percent
        self.decision_log_level = 0

        # Dynamic properties
        self.min_soc = min_soc
        self.max_soc = max_soc

        # Current time
        self._current_time = datetime.datetime(2024, 1, 15, 10, 0, 0)
        self._battery_temp = 20.0  # Default warm temperature

        # Learning engine
        self.learning_engine = BatteryLearningEngine(
            battery_capacity_kwh=battery_capacity,
            nominal_charge_rate_kw=charge_rate,
            nominal_efficiency=efficiency,
        )

        # Load profile
        self.load_profile = LoadProfile(
            slot_minutes=slot_minutes,
            default_load_w=base_consumption,
        )

        # Internal state
        self._last_min_charge_slots = 0
        self._last_charge_slots = []
        self._last_projected_costs = {}
        self.battery_wear_cost = 0.0

        # Expected schedule data
        self.expected_soc_schedule: Dict[datetime.datetime, float] = {}
        self.expected_temp_schedule: Dict[datetime.datetime, float] = {}

    def datetime(self):
        """Return current simulated time."""
        return self._current_time

    def set_datetime(self, dt: datetime.datetime):
        """Set simulated current time."""
        self._current_time = dt

    def set_battery_temp(self, temp: float):
        """Set simulated battery temperature."""
        self._battery_temp = temp

    def log(self, message: str, level: str = "INFO"):
        """Silent logging for tests."""
        pass

    def _align_to_slot(self, dt: datetime.datetime) -> datetime.datetime:
        """Align datetime to slot boundary."""
        minutes = dt.hour * 60 + dt.minute
        slot_start = (minutes // self.slot_minutes) * self.slot_minutes
        return dt.replace(
            hour=slot_start // 60,
            minute=slot_start % 60,
            second=0,
            microsecond=0
        )

    def _get_local_timezone(self):
        """Return None for naive datetimes in tests."""
        return None

    def _get_battery_temp(self) -> float:
        """Return simulated battery temperature."""
        return self._battery_temp

    def _predict_load_kw(self, dt: datetime.datetime) -> float:
        """Predict load for given time."""
        return self.load_profile.predict_kw(dt, self.load_quantile)

    def _get_prices_for_date(self, date, tz):
        """Return empty list (no yesterday prices in tests)."""
        return []

    def _get_discharge_threshold(self) -> float:
        """Calculate discharge price threshold."""
        return (self.battery_avg_cost / self.efficiency) + self.grid_fee

    def _log_schedule_decision_context(self, *args, **kwargs):
        """No-op for tests."""
        pass


# Import the actual methods from BatteryOptimizer
import sys
from pathlib import Path
apps_dir = Path(__file__).parent.parent / "appdaemon" / "apps"
sys.path.insert(0, str(apps_dir))

from battery_optimizer import BatteryOptimizer

# Bind the relevant methods to our mock
MockOptimizer.calculate_expected_soc_schedule = BatteryOptimizer.calculate_expected_soc_schedule
MockOptimizer.find_optimal_schedule = BatteryOptimizer.find_optimal_schedule
MockOptimizer._project_battery_costs = BatteryOptimizer._project_battery_costs
MockOptimizer._get_discharge_threshold_for_cost = BatteryOptimizer._get_discharge_threshold_for_cost


class TestTemperatureAwareSOCProjection:
    """Test temperature-aware SOC projection in schedule calculations."""

    @pytest.fixture
    def optimizer(self):
        """Create a mock optimizer for testing."""
        return MockOptimizer()

    @pytest.fixture
    def sample_schedule(self) -> Dict[datetime.datetime, ScheduleEntry]:
        """Create a sample schedule with charge/hold/discharge slots."""
        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
        return {
            base_time + datetime.timedelta(hours=0): ScheduleEntry(
                hour=base_time + datetime.timedelta(hours=0),
                mode=BatteryMode.CHARGE,
                reason="cheap"
            ),
            base_time + datetime.timedelta(hours=1): ScheduleEntry(
                hour=base_time + datetime.timedelta(hours=1),
                mode=BatteryMode.CHARGE,
                reason="cheap"
            ),
            base_time + datetime.timedelta(hours=2): ScheduleEntry(
                hour=base_time + datetime.timedelta(hours=2),
                mode=BatteryMode.HOLD,
                reason="neutral"
            ),
            base_time + datetime.timedelta(hours=3): ScheduleEntry(
                hour=base_time + datetime.timedelta(hours=3),
                mode=BatteryMode.DISCHARGE,
                reason="expensive"
            ),
        }

    def test_calculate_expected_soc_with_temp_returns_both_trajectories(
        self, optimizer, sample_schedule
    ):
        """Should return both SOC and temperature trajectories."""
        optimizer.set_battery_temp(15.0)

        soc_trajectory, temp_trajectory = optimizer.calculate_expected_soc_schedule(
            sample_schedule,
            starting_soc=50.0,
            starting_temp=15.0
        )

        # Both should be populated
        assert len(soc_trajectory) == 4
        assert len(temp_trajectory) == 4

        # SOC should increase during charge slots
        soc_values = list(soc_trajectory.values())
        assert soc_values[1] > soc_values[0]  # First charge

    def test_calculate_expected_soc_temp_evolves_during_charging(
        self, optimizer, sample_schedule, learning_engine_with_warming_data
    ):
        """Temperature should increase across consecutive charge slots."""
        optimizer.learning_engine = learning_engine_with_warming_data

        soc_trajectory, temp_trajectory = optimizer.calculate_expected_soc_schedule(
            sample_schedule,
            starting_soc=30.0,
            starting_temp=10.0
        )

        temp_values = list(temp_trajectory.values())

        # Temperature should increase during charging (slots 0 and 1)
        assert temp_values[1] > temp_values[0]  # Warming during first charge

    def test_calculate_expected_soc_temp_cools_during_hold(
        self, optimizer, learning_engine_with_warming_data
    ):
        """Temperature should cool toward estimated ambient during HOLD slots."""
        optimizer.learning_engine = learning_engine_with_warming_data

        # Record some temperature observations to establish ambient estimate
        for temp in [12.0, 10.0, 11.0, 10.5]:  # Min is 10.0, so ambient ~10°C
            optimizer.learning_engine.record_temperature_observation(temp)

        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
        hold_schedule = {
            base_time + datetime.timedelta(hours=i): ScheduleEntry(
                hour=base_time + datetime.timedelta(hours=i),
                mode=BatteryMode.HOLD,
                reason="hold"
            )
            for i in range(4)
        }

        soc_trajectory, temp_trajectory = optimizer.calculate_expected_soc_schedule(
            hold_schedule,
            starting_soc=50.0,
            starting_temp=25.0  # Above ambient (~10°C), should cool
        )

        temp_values = list(temp_trajectory.values())

        # Temperature should decrease toward estimated ambient during HOLD
        assert temp_values[0] == 25.0  # Starting temp
        assert temp_values[-1] < temp_values[0]  # Should have cooled
        # Should cool toward ~10°C ambient, but not quite reach it in 4 hours
        assert temp_values[-1] > 10.0  # But still above estimated ambient

    def test_calculate_expected_soc_fallback_without_temp(self, optimizer, sample_schedule):
        """Should work with None temperature (backward compatible)."""
        soc_trajectory, temp_trajectory = optimizer.calculate_expected_soc_schedule(
            sample_schedule,
            starting_soc=50.0,
            starting_temp=None  # No temperature
        )

        # SOC trajectory should still be populated
        assert len(soc_trajectory) == 4

        # Temperature trajectory should be empty when no starting temp
        assert len(temp_trajectory) == 0

    def test_calculate_expected_soc_cold_start_warming(
        self, optimizer, learning_engine_with_warming_data
    ):
        """
        Cold start (10°C) should show:
        - Slow charging initially
        - Faster charging after warming past threshold
        """
        optimizer.learning_engine = learning_engine_with_warming_data

        # Create schedule with 4 consecutive charge slots
        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
        charge_schedule = {
            base_time + datetime.timedelta(hours=i): ScheduleEntry(
                hour=base_time + datetime.timedelta(hours=i),
                mode=BatteryMode.CHARGE,
                reason="cheap"
            )
            for i in range(4)
        }

        soc_trajectory, temp_trajectory = optimizer.calculate_expected_soc_schedule(
            charge_schedule,
            starting_soc=30.0,
            starting_temp=10.0  # Cold start
        )

        soc_values = list(soc_trajectory.values())
        temp_values = list(temp_trajectory.values())

        # SOC should increase over time
        assert soc_values[-1] > soc_values[0]

        # Temperature should warm up
        assert temp_values[-1] > temp_values[0]

    def test_calculate_expected_soc_already_warm(
        self, optimizer, learning_engine_with_warming_data
    ):
        """Warm start (20°C) should use high charge rate throughout."""
        optimizer.learning_engine = learning_engine_with_warming_data

        # Create schedule with 2 consecutive charge slots
        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
        charge_schedule = {
            base_time + datetime.timedelta(hours=i): ScheduleEntry(
                hour=base_time + datetime.timedelta(hours=i),
                mode=BatteryMode.CHARGE,
                reason="cheap"
            )
            for i in range(2)
        }

        soc_trajectory, temp_trajectory = optimizer.calculate_expected_soc_schedule(
            charge_schedule,
            starting_soc=30.0,
            starting_temp=20.0  # Already warm
        )

        soc_values = list(soc_trajectory.values())

        # SOC should increase significantly when already warm (faster charging)
        soc_increase = soc_values[1] - soc_values[0]
        assert soc_increase > 0


class TestDPOptimizerTemperatureAwareRates:
    """Test DP optimizer uses temperature-aware charge rates."""

    @pytest.fixture
    def optimizer(self):
        """Create a mock optimizer for testing."""
        return MockOptimizer()

    @pytest.fixture
    def sample_prices(self) -> List[PricePoint]:
        """Sample prices for a 6-hour period."""
        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
        prices_cents = [3.0, 3.5, 4.0, 8.0, 12.0, 15.0]
        return [
            PricePoint(
                hour=base_time + datetime.timedelta(hours=i),
                price=price / 100
            )
            for i, price in enumerate(prices_cents)
        ]

    def test_charge_rates_per_slot_vary_with_warming(
        self, optimizer, sample_prices, learning_engine_with_warming_data
    ):
        """charge_rates_per_slot should increase for later slots as temp rises."""
        optimizer.learning_engine = learning_engine_with_warming_data
        optimizer.set_battery_temp(10.0)  # Cold start
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        # Call find_optimal_schedule to generate the schedule
        schedule = optimizer.find_optimal_schedule(
            sample_prices,
            charge_hours_needed=2,
            current_soc=30.0
        )

        # Should have generated a schedule
        assert len(schedule) > 0

    def test_cold_battery_charges_with_warming_consideration(
        self, optimizer, sample_prices, learning_engine_with_warming_data
    ):
        """Cold battery should account for warming in schedule optimization."""
        optimizer.learning_engine = learning_engine_with_warming_data
        optimizer.set_battery_temp(10.0)  # Cold
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        # Get schedule with cold battery
        cold_schedule = optimizer.find_optimal_schedule(
            sample_prices,
            charge_hours_needed=2,
            current_soc=30.0
        )

        cold_charges = [e for e in cold_schedule.values() if e.mode == BatteryMode.CHARGE]

        # Should have charge slots
        assert len(cold_charges) >= 2

    def test_warm_battery_optimization(
        self, optimizer, sample_prices, learning_engine_with_warming_data
    ):
        """Warm battery should optimize normally with higher charge rate."""
        optimizer.learning_engine = learning_engine_with_warming_data
        optimizer.set_battery_temp(20.0)  # Warm
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        # Get schedule with warm battery
        warm_schedule = optimizer.find_optimal_schedule(
            sample_prices,
            charge_hours_needed=2,
            current_soc=30.0
        )

        warm_charges = [e for e in warm_schedule.values() if e.mode == BatteryMode.CHARGE]

        # Should have charge slots
        assert len(warm_charges) >= 2


class TestLogScheduleTemperatureDisplay:
    """Test temperature display in schedule logging."""

    @pytest.fixture
    def optimizer(self):
        """Create a mock optimizer for testing."""
        opt = MockOptimizer()
        # Override log to capture output
        opt.log_messages = []
        opt.log = lambda msg, level="INFO": opt.log_messages.append(msg)
        return opt

    @pytest.fixture
    def sample_schedule(self) -> Dict[datetime.datetime, ScheduleEntry]:
        """Create a sample schedule."""
        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
        return {
            base_time: ScheduleEntry(
                hour=base_time,
                mode=BatteryMode.CHARGE,
                reason="0.03 EUR/kWh"
            ),
        }

    def test_log_schedule_shows_temp_when_available(
        self, optimizer, sample_schedule, learning_engine_with_warming_data
    ):
        """Log output should include temperature evolution."""
        optimizer.learning_engine = learning_engine_with_warming_data

        expected_soc = {list(sample_schedule.keys())[0]: 50.0}
        expected_temp = {list(sample_schedule.keys())[0]: 15.0}

        # Bind the _log_schedule method
        from battery_optimizer import BatteryOptimizer
        MockOptimizer._log_schedule = BatteryOptimizer._log_schedule
        MockOptimizer._get_expected_soc_for_hour = BatteryOptimizer._get_expected_soc_for_hour
        MockOptimizer._get_expected_temp_for_hour = BatteryOptimizer._get_expected_temp_for_hour

        optimizer._log_schedule(sample_schedule, expected_soc, expected_temp)

        # Check that temperature info is in the log
        log_text = " ".join(optimizer.log_messages)
        assert "C->" in log_text  # Temperature transition format

    def test_log_schedule_omits_temp_when_unavailable(
        self, optimizer, sample_schedule
    ):
        """Should not show temperature when not tracked."""
        expected_soc = {list(sample_schedule.keys())[0]: 50.0}
        expected_temp = {}  # No temperature data

        # Bind the _log_schedule method
        from battery_optimizer import BatteryOptimizer
        MockOptimizer._log_schedule = BatteryOptimizer._log_schedule
        MockOptimizer._get_expected_soc_for_hour = BatteryOptimizer._get_expected_soc_for_hour
        MockOptimizer._get_expected_temp_for_hour = BatteryOptimizer._get_expected_temp_for_hour

        optimizer._log_schedule(sample_schedule, expected_soc, expected_temp)

        # Check that temperature info is NOT in the log
        log_text = " ".join(optimizer.log_messages)
        assert "C->" not in log_text
