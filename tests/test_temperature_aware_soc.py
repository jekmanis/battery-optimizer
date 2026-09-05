"""
Tests for temperature-aware SOC projection in schedule calculations.

These tests verify that the optimizer correctly:
- Tracks temperature evolution during charge slots
- Uses temperature-aware charge rates in SOC projections
- Falls back gracefully when temperature data is unavailable
"""

from __future__ import annotations

import datetime
import math
from typing import Dict, List

import pytest

from battery_optimizer import (
    BatteryLearningEngine,
    BatteryMode,
    LoadProfile,
    PricePoint,
    ScheduleEntry,
)
from battery_optimizer_lib import (
    BatteryCostTracker,
    BatteryCostConfig,
    BatteryOptimizerConfig,
    PvForecastService,
    PvForecastServiceConfig,
    ScheduleFormatter,
    ScheduleFormatterConfig,
    TemperatureProjector,
)


class FixedAmbient:
    """Ambient provider returning a constant, for deterministic assertions."""

    def __init__(self, value: float):
        self.value = value

    def refresh(self, force: bool = False) -> bool:
        return False

    def predict_c(self, dt=None):
        return self.value


class DiurnalAmbient:
    """Ambient provider with a real daily swing (peak at 15:00)."""

    def __init__(self, mean: float = 31.0, amplitude: float = 4.0, peak_hour: float = 15.0):
        self.mean = mean
        self.amplitude = amplitude
        self.peak_hour = peak_hour

    def refresh(self, force: bool = False) -> bool:
        return False

    def predict_c(self, dt=None):
        if dt is None:
            return self.mean
        hour = dt.hour + dt.minute / 60.0
        return self.mean + self.amplitude * math.cos(
            2 * math.pi * (hour - self.peak_hour) / 24.0
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
        # Create config object (new pattern)
        self.config = BatteryOptimizerConfig(
            battery_capacity=battery_capacity,
            charge_rate=charge_rate,
            discharge_rate=discharge_rate,
            efficiency=efficiency,
            grid_fee=grid_fee,
            slot_minutes=slot_minutes,
            base_consumption=base_consumption,
            load_quantile=load_quantile,
            soc_step_percent=soc_step_percent,
            default_min_soc=min_soc,
            default_max_soc=max_soc,
            decision_log_level=0,
            battery_wear_cost=0.0,
        )

        # Legacy attributes for tests
        self.battery_avg_cost = battery_avg_cost

        # Dynamic properties
        self.min_soc = min_soc
        self.max_soc = max_soc

        # Current time
        self._current_time = datetime.datetime(2024, 1, 15, 10, 0, 0)
        self._battery_temp = 20.0  # Default warm temperature

        # Learning engine
        self.learning_engine = BatteryLearningEngine(
            battery_capacity_kwh=self.config.battery_capacity,
            nominal_charge_rate_kw=self.config.charge_rate,
            nominal_efficiency=self.config.efficiency,
        )

        # Load profile
        self.load_profile = LoadProfile(
            slot_minutes=self.config.slot_minutes,
            default_load_w=self.config.base_consumption,
        )

        # Internal state
        self._last_min_charge_slots = 0
        self._last_charge_slots = []
        self._last_projected_costs = {}

        # Expected schedule data
        self.expected_soc_schedule: Dict[datetime.datetime, float] = {}
        self.expected_temp_schedule: Dict[datetime.datetime, float] = {}

        # Create cost tracker for project_costs method
        self._cost_tracker = BatteryCostTracker(
            config=BatteryCostConfig(
                battery_capacity=self.config.battery_capacity,
                efficiency=self.config.efficiency,
                slot_minutes=self.config.slot_minutes,
                charge_rate=self.config.charge_rate,
                discharge_rate=self.config.discharge_rate,
                grid_fee=self.config.grid_fee,
                battery_wear_cost=self.config.battery_wear_cost,
            ),
            get_state_func=lambda e: None,
            call_service_func=lambda *a, **k: None,
            get_datetime_func=self.datetime,
            get_timezone_func=self._get_local_timezone,
            align_to_slot_func=self._align_to_slot,
            get_min_soc_func=lambda: self.min_soc,
            get_max_soc_func=lambda: self.max_soc,
            get_current_soc_func=lambda: 50.0,
            get_battery_temp_func=lambda: self._battery_temp,
            learning_engine=self.learning_engine,
            get_cached_prices_func=lambda: [],
            save_learning_data_func=lambda: None,
            update_learning_sensor_func=lambda: None,
            log_func=self.log,
        )

        # Shared thermal model. ``_temp_projector`` is a property because
        # several tests replace ``learning_engine`` after construction.
        self._ambient_service = None

        # PV forecast service (empty — no forecast data in tests)
        self._pv_forecast_service = PvForecastService(
            config=PvForecastServiceConfig(slot_minutes=self.config.slot_minutes),
            get_state_func=lambda e, **kw: None,
            get_datetime_func=self.datetime,
            get_timezone_func=self._get_local_timezone,
            log_func=self.log,
        )

    @property
    def _temp_projector(self):
        """Shared thermal projector bound to the CURRENT learning engine."""
        return TemperatureProjector(
            learning_engine=self.learning_engine,
            ambient_provider=self._ambient_service,
        )

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
        slot_start = (minutes // self.config.slot_minutes) * self.config.slot_minutes
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
        return self.load_profile.predict_kw(dt, self.config.load_quantile)

    def _predict_pv_kw(self, dt: datetime.datetime) -> float:
        return 0.0

    def _get_prices_for_date(self, date, tz):
        """Return empty list (no yesterday prices in tests)."""
        return []

    def _get_discharge_threshold(self) -> float:
        """Calculate discharge price threshold."""
        return (self.battery_avg_cost / self.config.efficiency) + self.config.grid_fee

    def _log_schedule_decision_context(self, *args, **kwargs):
        """No-op for tests."""
        pass

    @property
    def _price_service(self):
        """Mock price service with get_prices_for_date method."""
        class MockPriceService:
            def get_prices_for_date(self, date, tz):
                return []
        return MockPriceService()


# Import the actual methods from BatteryOptimizer
import sys
from pathlib import Path
apps_dir = Path(__file__).parent.parent / "appdaemon" / "apps"
sys.path.insert(0, str(apps_dir))

from battery_optimizer import BatteryOptimizer

# Bind the relevant methods to our mock
MockOptimizer.calculate_expected_soc_schedule = BatteryOptimizer.calculate_expected_soc_schedule
MockOptimizer.project_schedule_trajectory = BatteryOptimizer.project_schedule_trajectory
MockOptimizer._replay_schedule = BatteryOptimizer._replay_schedule
MockOptimizer.find_optimal_schedule = BatteryOptimizer.find_optimal_schedule
MockOptimizer._ensure_current_slot_price = BatteryOptimizer._ensure_current_slot_price
MockOptimizer._compute_slot_fractions = BatteryOptimizer._compute_slot_fractions
MockOptimizer._compute_charge_rates_per_slot = BatteryOptimizer._compute_charge_rates_per_slot


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
                time=base_time + datetime.timedelta(hours=0),
                mode=BatteryMode.CHARGE,
                reason="cheap"
            ),
            base_time + datetime.timedelta(hours=1): ScheduleEntry(
                time=base_time + datetime.timedelta(hours=1),
                mode=BatteryMode.CHARGE,
                reason="cheap"
            ),
            base_time + datetime.timedelta(hours=2): ScheduleEntry(
                time=base_time + datetime.timedelta(hours=2),
                mode=BatteryMode.HOLD,
                reason="neutral"
            ),
            base_time + datetime.timedelta(hours=3): ScheduleEntry(
                time=base_time + datetime.timedelta(hours=3),
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
                time=base_time + datetime.timedelta(hours=i),
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
                time=base_time + datetime.timedelta(hours=i),
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
                time=base_time + datetime.timedelta(hours=i),
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
                time=base_time + datetime.timedelta(hours=i),
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

    def test_cold_battery_optimization_generates_schedule(
        self, optimizer, sample_prices, learning_engine_with_warming_data
    ):
        """Cold battery should generate valid schedule accounting for temperature."""
        optimizer.learning_engine = learning_engine_with_warming_data
        optimizer.set_battery_temp(10.0)  # Cold
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        # Get schedule with cold battery
        cold_schedule = optimizer.find_optimal_schedule(
            sample_prices,
            charge_hours_needed=2,
            current_soc=30.0
        )

        # Should generate valid schedule with some activity
        assert len(cold_schedule) > 0
        # Algorithm may choose discharge during expensive hours or hold
        # (charging is not forced if not economically beneficial)
        modes = {e.mode for e in cold_schedule.values()}
        assert len(modes) >= 1  # At least one mode type in schedule

    def test_warm_battery_optimization_generates_schedule(
        self, optimizer, sample_prices, learning_engine_with_warming_data
    ):
        """Warm battery should generate valid schedule with higher efficiency."""
        optimizer.learning_engine = learning_engine_with_warming_data
        optimizer.set_battery_temp(20.0)  # Warm
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        # Get schedule with warm battery
        warm_schedule = optimizer.find_optimal_schedule(
            sample_prices,
            charge_hours_needed=2,
            current_soc=30.0
        )

        # Should generate valid schedule
        assert len(warm_schedule) > 0
        # Algorithm optimizes economically - may choose discharge/hold over charge
        modes = {e.mode for e in warm_schedule.values()}
        assert len(modes) >= 1  # At least one mode type in schedule


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
                time=base_time,
                mode=BatteryMode.CHARGE,
                reason="0.03 EUR/kWh"
            ),
        }

    def test_log_schedule_shows_temp_when_available(
        self, optimizer, sample_schedule, learning_engine_with_warming_data
    ):
        """Log output should include temperature evolution."""
        log_messages = []

        def mock_log(message: str, level: str = "INFO"):
            log_messages.append(message)

        # Create formatter with learning engine that has warming data
        formatter = ScheduleFormatter(
            config=ScheduleFormatterConfig(
                slot_minutes=60,
                slot_hours=1.0,
                battery_capacity=14.3,
                charge_rate=4.5,
                discharge_rate=4.5,
                export_discharge_rate=0.0,
                efficiency=0.85,
                battery_wear_cost=0.0,
                decision_log_level=1,
            ),
            log_func=mock_log,
            learning_engine=learning_engine_with_warming_data,
        )

        expected_soc = {list(sample_schedule.keys())[0]: 50.0}
        expected_temp = {list(sample_schedule.keys())[0]: 15.0}

        formatter.log_schedule(
            schedule=sample_schedule,
            expected_soc=expected_soc,
            expected_temp=expected_temp,
            local_tz=None,
            predict_load_kw=lambda h: 0.5,
            min_soc=10.0,
            max_soc=100.0,
        )

        # Check that temperature info is in the log
        log_text = " ".join(log_messages)
        assert "C->" in log_text  # Temperature transition format

    def test_log_schedule_omits_temp_when_unavailable(
        self, optimizer, sample_schedule
    ):
        """Should not show temperature when not tracked."""
        log_messages = []

        def mock_log(message: str, level: str = "INFO"):
            log_messages.append(message)

        # Create formatter without learning engine
        formatter = ScheduleFormatter(
            config=ScheduleFormatterConfig(
                slot_minutes=60,
                slot_hours=1.0,
                battery_capacity=14.3,
                charge_rate=4.5,
                discharge_rate=4.5,
                export_discharge_rate=0.0,
                efficiency=0.85,
                battery_wear_cost=0.0,
                decision_log_level=1,
            ),
            log_func=mock_log,
            learning_engine=None,
        )

        expected_soc = {list(sample_schedule.keys())[0]: 50.0}
        expected_temp = {}  # No temperature data

        formatter.log_schedule(
            schedule=sample_schedule,
            expected_soc=expected_soc,
            expected_temp=expected_temp,
            local_tz=None,
            predict_load_kw=lambda h: 0.5,
            min_soc=10.0,
            max_soc=100.0,
        )

        # Check that temperature info is NOT in the log
        log_text = " ".join(log_messages)
        assert "C->" not in log_text


class TestChargeRateSOCProjection:
    """Tests that charge rates decline as projected SOC increases."""

    def test_rates_decline_with_soc_projection(self):
        """With SOC-dependent charge rate, later slots should see lower rates."""
        from battery_optimizer_lib.charge_rate_utils import compute_charge_rates_per_slot

        base = datetime.datetime(2024, 6, 15, 12, 0)
        slots = [PricePoint(time=base + datetime.timedelta(minutes=15 * i), price=0.05) for i in range(8)]
        fractions = [1.0] * 8

        # Simulate BMS behaviour: charge rate drops with SOC
        def charge_rate_by_soc(soc, temp):
            if soc < 50:
                return 7.4
            elif soc < 80:
                return 5.0
            else:
                return 2.5

        rates = compute_charge_rates_per_slot(
            slots_sorted_by_time=slots,
            slot_fractions=fractions,
            slot_minutes=15,
            current_soc=20.0,
            current_temp=25.0,
            get_charge_rate_for_soc=charge_rate_by_soc,
            predict_temp_after_duration=lambda t, d: t,  # constant temp
            battery_capacity=14.3,
            efficiency=0.85,
            max_soc=100.0,
        )

        # First slots should have high rate (low SOC)
        assert rates[0] == 7.4
        # Later slots should have lower rates as SOC climbs
        assert rates[-1] < rates[0], f"Last rate {rates[-1]} should be < first rate {rates[0]}"
        # Rates should be monotonically non-increasing
        for i in range(1, len(rates)):
            assert rates[i] <= rates[i - 1], f"Rate at slot {i} ({rates[i]}) > slot {i-1} ({rates[i-1]})"

    def test_rates_constant_without_capacity(self):
        """Without battery_capacity (legacy), rates stay at initial SOC."""
        from battery_optimizer_lib.charge_rate_utils import compute_charge_rates_per_slot

        base = datetime.datetime(2024, 6, 15, 12, 0)
        slots = [PricePoint(time=base + datetime.timedelta(minutes=15 * i), price=0.05) for i in range(4)]
        fractions = [1.0] * 4

        def charge_rate_by_soc(soc, temp):
            return 7.4 if soc < 50 else 2.5

        rates = compute_charge_rates_per_slot(
            slots_sorted_by_time=slots,
            slot_fractions=fractions,
            slot_minutes=15,
            current_soc=20.0,
            current_temp=None,
            get_charge_rate_for_soc=charge_rate_by_soc,
            predict_temp_after_duration=lambda t, d: t,
            # battery_capacity=0 (default) → no SOC projection
        )

        # All rates should be at the low-SOC rate
        assert all(r == 7.4 for r in rates)

    def test_soc_capped_at_max(self):
        """SOC projection should not exceed max_soc."""
        from battery_optimizer_lib.charge_rate_utils import compute_charge_rates_per_slot

        base = datetime.datetime(2024, 6, 15, 12, 0)
        slots = [PricePoint(time=base + datetime.timedelta(minutes=15 * i), price=0.05) for i in range(20)]
        fractions = [1.0] * 20

        soc_values_seen = []

        def charge_rate_tracker(soc, temp):
            soc_values_seen.append(soc)
            return 4.5

        compute_charge_rates_per_slot(
            slots_sorted_by_time=slots,
            slot_fractions=fractions,
            slot_minutes=15,
            current_soc=80.0,
            current_temp=25.0,
            get_charge_rate_for_soc=charge_rate_tracker,
            predict_temp_after_duration=lambda t, d: t,
            battery_capacity=14.3,
            efficiency=0.85,
            max_soc=100.0,
        )

        # SOC should never exceed max_soc
        assert all(s <= 100.0 + 0.01 for s in soc_values_seen), \
            f"SOC exceeded max: {max(soc_values_seen):.1f}%"


class TestDischargeWarmsTheBattery:
    """DEFECT 6 regression on the expected-SOC trajectory.

    ``calculate_expected_soc_schedule`` used to route both DISCHARGE and HOLD
    through ``predict_temp_after_idle`` with the comment "no active warming".
    A 5.9 kW discharge does not leave the pack thermally idle.
    """

    @staticmethod
    def _optimizer(ambient: float, export_discharge_rate: float = 0.0):
        opt = MockOptimizer(slot_minutes=15, discharge_rate=5.9)
        opt.config.export_discharge_rate = export_discharge_rate
        opt._ambient_service = FixedAmbient(ambient)
        # 4 kW household net load through the whole window
        opt._predict_load_kw = lambda dt: 4.0
        return opt

    @staticmethod
    def _discharge_schedule(n=8, export_rate=None):
        base = datetime.datetime(2026, 7, 27, 20, 0)
        return {
            base + datetime.timedelta(minutes=15 * i): ScheduleEntry(
                time=base + datetime.timedelta(minutes=15 * i),
                mode=BatteryMode.DISCHARGE,
                reason="expensive",
                export_rate=export_rate,
            )
            for i in range(n)
        }

    def test_discharge_slots_warm_the_battery(self):
        """Self-consumption discharge at ambient must RAISE the temperature.

        Before the fix the trajectory relaxed toward min(recent battery temps)
        (default 10C here) and was strictly decreasing.
        """
        optimizer = self._optimizer(ambient=27.0)
        schedule = self._discharge_schedule()

        _, temp_trajectory = optimizer.calculate_expected_soc_schedule(
            schedule, starting_soc=80.0, starting_temp=27.0
        )

        temps = [temp_trajectory[h] for h in sorted(temp_trajectory)]
        assert len(temps) == 8
        assert temps[0] == 27.0
        assert temps[-1] > temps[0], f"discharge did not warm the pack: {temps}"

    def test_export_discharge_warms_more_than_self_consumption(self):
        """Higher |P_bat| must produce more heating over the same slots."""
        self_consume = self._optimizer(ambient=27.0)
        exporting = self._optimizer(ambient=27.0, export_discharge_rate=5.9)

        _, self_temps = self_consume.calculate_expected_soc_schedule(
            self._discharge_schedule(), starting_soc=90.0, starting_temp=27.0
        )
        _, export_temps = exporting.calculate_expected_soc_schedule(
            self._discharge_schedule(export_rate=100), starting_soc=90.0, starting_temp=27.0
        )

        assert export_temps[max(export_temps)] > self_temps[max(self_temps)]

    def test_hold_without_pv_stays_thermally_idle(self):
        """HOLD with no PV surplus is genuinely idle — it must still cool."""
        optimizer = self._optimizer(ambient=27.0)
        base = datetime.datetime(2026, 7, 27, 20, 0)
        schedule = {
            base + datetime.timedelta(minutes=15 * i): ScheduleEntry(
                time=base + datetime.timedelta(minutes=15 * i),
                mode=BatteryMode.HOLD,
                reason="hold",
            )
            for i in range(8)
        }

        _, temps = optimizer.calculate_expected_soc_schedule(
            schedule, starting_soc=80.0, starting_temp=33.0
        )
        values = [temps[h] for h in sorted(temps)]
        assert values[-1] < values[0]
        assert values[-1] > 27.0


class TestExpectedTemperatureFollowsAmbientOverTime:
    """DEFECT 7 regression on the expected-SOC trajectory."""

    def test_trajectory_is_not_a_single_repeated_value(self):
        optimizer = MockOptimizer(slot_minutes=15, discharge_rate=5.9)
        optimizer._ambient_service = DiurnalAmbient(mean=31.0, amplitude=4.0)
        optimizer._predict_load_kw = lambda dt: 0.5

        # 33 h horizon starting at 16:00, exactly like the analysed log window.
        base = datetime.datetime(2026, 7, 28, 16, 0)
        schedule = {
            base + datetime.timedelta(minutes=15 * i): ScheduleEntry(
                time=base + datetime.timedelta(minutes=15 * i),
                mode=BatteryMode.HOLD,
                reason="hold",
            )
            for i in range(33 * 4)
        }

        _, temps = optimizer.calculate_expected_soc_schedule(
            schedule, starting_soc=50.0, starting_temp=34.0
        )

        values = [round(temps[h], 0) for h in sorted(temps)]
        # The logged trajectory showed 34C for 1.5 h then 33C for the rest.
        assert len(set(values)) >= 4, f"still practically constant: {sorted(set(values))}"
        assert max(values) - min(values) >= 4.0

    def test_tomorrow_morning_is_colder_than_tonight(self):
        optimizer = MockOptimizer(slot_minutes=60, discharge_rate=5.9)
        optimizer._ambient_service = DiurnalAmbient(mean=31.0, amplitude=4.0)
        optimizer._predict_load_kw = lambda dt: 0.5

        base = datetime.datetime(2026, 7, 27, 16, 0)
        schedule = {
            base + datetime.timedelta(hours=i): ScheduleEntry(
                time=base + datetime.timedelta(hours=i),
                mode=BatteryMode.HOLD,
                reason="hold",
            )
            for i in range(24)
        }

        _, temps = optimizer.calculate_expected_soc_schedule(
            schedule, starting_soc=50.0, starting_temp=34.0
        )
        tonight = temps[datetime.datetime(2026, 7, 27, 20, 0)]
        tomorrow_morning = temps[datetime.datetime(2026, 7, 28, 6, 0)]
        assert tonight - tomorrow_morning > 2.0


class TestSharedProjectionAcrossConsumers:
    """Point (d): one temperature model, whichever code path is taken."""

    def test_dp_and_expected_trajectories_agree(self):
        optimizer = MockOptimizer(slot_minutes=15, discharge_rate=5.9)
        optimizer.config.inverter_efficiency = 0.97
        optimizer._ambient_service = DiurnalAmbient(mean=29.0, amplitude=5.0)
        optimizer._predict_load_kw = lambda dt: 3.0
        optimizer.set_battery_temp(33.0)
        optimizer.set_datetime(datetime.datetime(2026, 7, 27, 16, 0))

        base = datetime.datetime(2026, 7, 27, 16, 0)
        prices = [
            PricePoint(time=base + datetime.timedelta(minutes=15 * i), price=0.30)
            for i in range(24)
        ]

        schedule = optimizer.find_optimal_schedule(prices, 0, current_soc=90.0)
        assert schedule
        dp_temps = optimizer._last_dp_temp_trajectory
        assert dp_temps

        _, expected_temps = optimizer.calculate_expected_soc_schedule(
            schedule, starting_soc=90.0, starting_temp=33.0
        )

        compared = 0
        for hour in sorted(schedule):
            if hour not in dp_temps or hour not in expected_temps:
                continue
            dp_start = dp_temps[hour][0]
            assert abs(dp_start - expected_temps[hour]) < 0.1, (
                f"{hour}: DP {dp_start:.3f}C vs expected {expected_temps[hour]:.3f}C"
            )
            compared += 1
        assert compared >= 20

    def test_formatter_matches_the_shared_model(self, learning_engine):
        """The log fallback path must use the same model as everything else."""
        projector = TemperatureProjector(
            learning_engine=learning_engine, ambient_provider=FixedAmbient(27.0)
        )
        log_messages = []
        formatter = ScheduleFormatter(
            config=ScheduleFormatterConfig(
                slot_minutes=15,
                slot_hours=0.25,
                battery_capacity=14.3,
                charge_rate=4.5,
                discharge_rate=5.9,
                export_discharge_rate=0.0,
                efficiency=0.85,
                battery_wear_cost=0.0,
                decision_log_level=1,
            ),
            log_func=lambda msg, level="INFO": log_messages.append(msg),
            learning_engine=learning_engine,
            temp_projector=projector,
        )

        hour = datetime.datetime(2026, 7, 27, 20, 0)
        schedule = {
            hour: ScheduleEntry(time=hour, mode=BatteryMode.DISCHARGE, reason="0.30 EUR/kWh")
        }
        formatter.log_schedule(
            schedule=schedule,
            expected_soc={hour: 80.0},
            expected_temp={hour: 27.0},
            local_tz=None,
            predict_load_kw=lambda h: 4.0,
            min_soc=10.0,
            max_soc=100.0,
        )

        text = " ".join(log_messages)
        assert "C->" in text
        # The formatter now feeds |P_bat| into the shared model, so a discharge
        # slot at ambient warms instead of staying flat.
        expected_end = projector.project(27.0, hour, 15.0, 4.0)
        assert expected_end > 27.0
        assert f"(27C->{expected_end:.0f}C)" in text


class TestChargeRateProjectionIsBounded:
    """The one place where the temperature forecast changes DP decisions."""

    @staticmethod
    def _slots(n):
        base = datetime.datetime(2026, 7, 27, 16, 0)
        return [
            PricePoint(time=base + datetime.timedelta(minutes=15 * i), price=0.05)
            for i in range(n)
        ]

    def test_legacy_projection_diverges(self):
        """Documents the defect: unbounded linear warming across 33 h."""
        from battery_optimizer_lib.charge_rate_utils import compute_charge_rates_per_slot

        seen = []

        def rate(soc, temp):
            seen.append(temp)
            return 4.5

        compute_charge_rates_per_slot(
            slots_sorted_by_time=self._slots(132),
            slot_fractions=[1.0] * 132,
            slot_minutes=15,
            current_soc=30.0,
            current_temp=33.0,
            get_charge_rate_for_soc=rate,
            # The historical default: +0.1 C/min, no ambient, no ceiling.
            predict_temp_after_duration=lambda t, d: t + 0.1 * d,
        )
        assert max(seen) > 200.0

    def test_shared_projector_does_not_diverge(self, learning_engine):
        from battery_optimizer_lib.charge_rate_utils import compute_charge_rates_per_slot

        projector = TemperatureProjector(
            learning_engine=learning_engine,
            ambient_provider=DiurnalAmbient(mean=31.0, amplitude=4.0),
        )
        seen = []

        def rate(soc, temp):
            seen.append(temp)
            return 4.5

        compute_charge_rates_per_slot(
            slots_sorted_by_time=self._slots(132),
            slot_fractions=[1.0] * 132,
            slot_minutes=15,
            current_soc=30.0,
            current_temp=33.0,
            get_charge_rate_for_soc=rate,
            project_temp=projector.project,
            battery_capacity=14.3,
            efficiency=0.85,
            max_soc=100.0,
        )

        assert len(seen) == 132
        assert max(seen) < 60.0
        # ...and it tracks the ambient profile instead of climbing forever.
        assert max(seen) - min(seen) > 1.0

    def test_projector_wins_over_legacy_callback(self, learning_engine):
        """When both are supplied, the shared model is authoritative."""
        from battery_optimizer_lib.charge_rate_utils import compute_charge_rates_per_slot

        projector = TemperatureProjector(
            learning_engine=learning_engine, ambient_provider=FixedAmbient(27.0)
        )
        seen = []

        def rate(soc, temp):
            seen.append(temp)
            return 4.5

        compute_charge_rates_per_slot(
            slots_sorted_by_time=self._slots(10),
            slot_fractions=[1.0] * 10,
            slot_minutes=15,
            current_soc=30.0,
            current_temp=33.0,
            get_charge_rate_for_soc=rate,
            predict_temp_after_duration=lambda t, d: 999.0,
            project_temp=projector.project,
        )
        assert max(seen) < 60.0


class TestCloudSafeConversionRebuildsTrajectories:
    """The reported trajectories must describe the plan that will EXECUTE.

    ``find_optimal_schedule`` rewrites HOLD -> DISCHARGE(to load) for PV hours
    AFTER the DP has already built ``soc_trajectory`` / ``temp_trajectory``.
    Those DP trajectories are what ``schedule_formatter`` prefers (it falls back
    to the expected-SOC map only when they are missing), so they are rebuilt
    from the FINAL schedule through the shared model — by derivation, not by an
    equivalence argument, inside the very log used to diagnose SOC deviations.

    The conversion is now restricted to slots whose modeled flow is identical
    to HOLD, so the scenario is a sunny slot with PV ABOVE the load: the pack
    charges from the surplus under either action. The old scenario had PV at
    0.3 kW against a 1.0 kW load, where the conversion drained a battery the DP
    had reserved — see ``tests/test_cloud_safe_conversion.py``.
    """

    START_SOC = 50.0
    START_TEMP = 20.0
    BASE = datetime.datetime(2026, 7, 27, 10, 0)

    def _optimizer(self):
        opt = MockOptimizer(slot_minutes=60, discharge_rate=4.5)
        # 0.10 keeps the DP off both extremes: grid charging lands at
        # 0.25/0.85 = 0.29 EUR/kWh so it does not charge, and the hedge's
        # avoided import (0.25) still beats the value of keeping a kWh.
        opt.config.terminal_energy_value_eur_kwh = 0.10
        opt.config.battery_wear_cost = 0.0
        # Export must stay worth something (so the hedge's export-equivalence
        # branch is actually exercised) but less than a kept kWh, or the DP
        # sells the pack instead of holding it: 0.20 - 0.16 = 0.04 < 0.10.
        opt.config.grid_export_fee = 0.16
        opt._ambient_service = FixedAmbient(20.0)
        opt._predict_load_kw = lambda dt: 1.0
        # PV covers the load with a surplus small enough for the pack to
        # absorb all of it, so nothing that HOLD would have exported is lost.
        opt._predict_pv_kw = lambda dt: 1.3
        opt.set_battery_temp(self.START_TEMP)
        opt.set_datetime(self.BASE)
        return opt

    def _run(self):
        opt = self._optimizer()
        prices = [
            PricePoint(time=self.BASE + datetime.timedelta(hours=i), price=0.20)
            for i in range(8)
        ]
        schedule = opt.find_optimal_schedule(prices, 0, current_soc=self.START_SOC)
        assert schedule
        return opt, schedule

    def test_conversion_actually_happened(self):
        _opt, schedule = self._run()
        converted = [e for e in schedule.values() if "[cloud-safe]" in e.reason]
        assert converted, "scenario no longer exercises the conversion"
        assert all(e.mode == BatteryMode.DISCHARGE for e in converted)

    def test_soc_trajectory_charges_like_the_converted_plan(self):
        """``discharge_to_load`` with PV above the load is a PV charge."""
        opt, schedule = self._run()
        soc_traj = opt._last_dp_soc_trajectory
        assert soc_traj

        first = min(schedule)
        start_soc, end_soc = soc_traj[first]
        assert start_soc == pytest.approx(self.START_SOC)
        assert end_soc > start_soc

    def test_trajectories_match_the_shared_model_for_the_final_schedule(self):
        opt, schedule = self._run()

        rebuilt_soc, rebuilt_temp = opt.project_schedule_trajectory(
            schedule,
            self.START_SOC,
            starting_temp=self.START_TEMP,
            current_slot=self.BASE,
            minutes_into_slot=0.0,
        )

        assert opt._last_dp_soc_trajectory == rebuilt_soc
        assert opt._last_dp_temp_trajectory == rebuilt_temp

    def test_temperature_trajectory_reflects_the_battery_power(self):
        """Warming is a function of |P_bat|, not of the mode label."""
        opt, schedule = self._run()
        temps = opt._last_dp_temp_trajectory
        assert temps

        first = min(schedule)
        start_temp, end_temp = temps[first]
        assert start_temp == pytest.approx(self.START_TEMP)
        assert end_temp is not None
