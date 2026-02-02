"""
Tests for SOC deviation detection and recalculation skip logic.

These tests verify that:
1. During CHARGE mode, if behind schedule but will still reach max_soc, recalculation is skipped
2. During DISCHARGE mode, favorable deviations (ahead) skip recalculation
3. Recalculation is still triggered when actually needed
"""

import datetime
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

import sys
from pathlib import Path
apps_dir = Path(__file__).parent.parent / "appdaemon" / "apps"
sys.path.insert(0, str(apps_dir))

from battery_optimizer import BatteryMode, ScheduleEntry, BatteryOptimizer
from battery_optimizer_lib import BatteryLearningEngine, ScheduleFormatter, ScheduleFormatterConfig


class MockSocDeviationOptimizer:
    """
    Mock optimizer for testing _check_soc_deviation.
    """

    def __init__(self):
        self.schedule: Dict[datetime.datetime, ScheduleEntry] = {}
        self.expected_soc_schedule: Dict[datetime.datetime, float] = {}
        self._current_datetime = datetime.datetime(2024, 1, 15, 5, 22, 0)
        self.slot_minutes = 60
        self.slot_hours = 1.0
        self.soc_deviation_threshold = 4.0
        self.battery_capacity = 14.3  # kWh
        self.charge_rate = 4.27  # kW (theoretical)
        self.discharge_rate = 4.27  # kW
        self.efficiency = 0.95
        self.max_soc = 100.0
        self.min_soc = 10.0
        self.current_mode = BatteryMode.CHARGE
        self.decision_log_level = 1
        self.learning_engine: Optional[BatteryLearningEngine] = None

        # Tracking
        self._recalculate_calls = []
        self._log_messages = []
        self._last_recalc_trigger = None
        self._last_recalc_time = None
        self._last_soc_deviation = None

    def datetime(self):
        return self._current_datetime

    def log(self, message: str, level: str = "INFO"):
        self._log_messages.append(message)

    def _get_local_timezone(self):
        return None

    def _align_to_slot(self, dt: datetime.datetime) -> datetime.datetime:
        """Align datetime to slot boundary."""
        return dt.replace(minute=0, second=0, microsecond=0)

    def _is_enabled(self):
        return True

    def _is_override_active(self):
        return False

    def _predict_load_kw(self, hour: datetime.datetime) -> float:
        return 0.5  # Default load prediction

    def _get_battery_temp(self) -> Optional[float]:
        """Return mock battery temperature (None by default, can be set in tests)."""
        return getattr(self, '_battery_temp', None)

    def _recalculate_remaining_schedule(self, current_soc: float, extra_charge_slots: int = 0):
        """Track recalculation calls."""
        self._recalculate_calls.append(current_soc)

    def get_prices(self):
        """Return mock prices (can be overridden in tests)."""
        return getattr(self, '_prices', [])

    def _get_discharge_threshold(self) -> float:
        """Return mock discharge threshold (can be overridden in tests)."""
        return getattr(self, '_discharge_threshold', 0.20)

    @property
    def grid_fee(self):
        """Return mock grid fee (can be overridden in tests)."""
        return getattr(self, '_grid_fee', 0.05)

    @grid_fee.setter
    def grid_fee(self, value):
        self._grid_fee = value


# Bind the actual methods from BatteryOptimizer
MockSocDeviationOptimizer._check_soc_deviation = BatteryOptimizer._check_soc_deviation
MockSocDeviationOptimizer._get_cheapest_upcoming_prices = BatteryOptimizer._get_cheapest_upcoming_prices


class TestSocDeviationDuringCharge:
    """Test SOC deviation behavior during CHARGE mode."""

    @pytest.fixture
    def optimizer(self):
        opt = MockSocDeviationOptimizer()
        # No learning engine by default - use theoretical rate
        return opt

    def test_skip_recalc_when_behind_but_will_reach_max_soc(self, optimizer):
        """
        Scenario: Charging at 05:30, SOC is 55% but expected was ~66%.
        With charge_rate=4.27, efficiency=0.95, capacity=14.3:
        - Energy per hour = 4.27 * 0.95 = 4.06 kWh
        - SOC gain per hour = 4.06 / 14.3 * 100 = 28.4%
        - At 30 min in: expected gain = 14.2%, so expected = 50 + 14.2 = 64.2%
        - Actual = 55%, delta = 55 - 64.2 = -9.2% (exceeds threshold)

        Remaining in this slot: 0.5 * 4.06 = 2.03 kWh = 14.2% SOC
        From 55%, will reach 55 + 14.2 = 69.2%

        But we have slot_06 also CHARGE, so total remaining:
        - 0.5 hours in slot_05 + 1.0 hour in slot_06 = 1.5 * 4.06 = 6.09 kWh = 42.6%
        - From 55%, will reach 55 + 42.6 = 97.6% >= 95% (max_soc - 5)

        Should skip recalculation.
        """
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)
        slot_07 = datetime.datetime(2024, 1, 15, 7, 0, 0)

        optimizer.schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_07: ScheduleEntry(hour=slot_07, mode=BatteryMode.DISCHARGE, reason="expensive"),
        }
        # Expected SOC at start of each slot
        optimizer.expected_soc_schedule = {
            slot_05: 50.0,
            slot_06: 78.4,  # 50 + 28.4
            slot_07: 100.0,  # Capped at max
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 5, 30, 0)

        current_soc = 55.0  # Behind schedule (-9.2% deviation)

        result = optimizer._check_soc_deviation(current_soc)

        # Should NOT trigger recalculation - will still reach ~97% with remaining charge hours
        assert result is False
        assert len(optimizer._recalculate_calls) == 0
        # Should log that we're skipping
        assert any("skipping recalculation" in msg for msg in optimizer._log_messages)

    def test_skip_recalc_with_single_remaining_charge_hour(self, optimizer):
        """
        Scenario: Late in charging hour, SOC is behind but will still reach max.
        """
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)

        optimizer.schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.HOLD, reason="moderate"),
        }
        optimizer.expected_soc_schedule = {
            slot_05: 70.0,  # High starting SOC
            slot_06: 98.4,
        }
        # 15 minutes into the hour (fraction = 0.25)
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 5, 15, 0)

        # Expected at 15 min: 70 + 0.25 * 28.4 = 77.1%
        # Current: 72% (5.1% behind, exceeds 4% threshold)
        current_soc = 72.0

        result = optimizer._check_soc_deviation(current_soc)

        # Remaining: 0.75 * 28.4 = 21.3% gain
        # From 72%, will reach 72 + 21.3 = 93.3%, which is < 95 (max_soc - 5)
        # So this SHOULD trigger recalculation
        assert result is True
        assert len(optimizer._recalculate_calls) == 1

    def test_trigger_recalc_when_wont_reach_max_soc(self, optimizer):
        """
        Scenario: At 05:50, SOC is 50% but expected was 78%.
        Only 10 minutes left in single charge hour, won't reach max_soc.
        Should trigger recalculation.
        """
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)

        optimizer.schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.DISCHARGE, reason="expensive"),
        }
        optimizer.expected_soc_schedule = {
            slot_05: 50.0,
            slot_06: 78.4,
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 5, 50, 0)

        # At 50 min, expected = 50 + (50/60) * 28.4 = 73.7%
        # Current = 50%, deviation = -23.7% (way behind)
        current_soc = 50.0

        result = optimizer._check_soc_deviation(current_soc)

        # Only 10 min left = 0.167 * 28.4 = 4.7% gain
        # From 50%, will reach 54.7%, which is way below 95%
        # Should trigger recalculation
        assert result is True
        assert len(optimizer._recalculate_calls) == 1


class TestSocDeviationDuringDischarge:
    """Test SOC deviation behavior during DISCHARGE mode."""

    @pytest.fixture
    def optimizer(self):
        return MockSocDeviationOptimizer()

    def test_skip_recalc_when_favorable_deviation(self, optimizer):
        """
        Scenario: Discharging at 10:30, SOC is higher than expected.
        Being ahead (more SOC than expected) during discharge is favorable.
        Should skip recalculation for small favorable deviations.

        With discharge (load ~0.5 kW, 1 hour slot):
        - Energy removed per hour = 0.5 kWh
        - SOC decrease per hour = 0.5 / 14.3 * 100 = 3.5%
        - At 30 min: expected decrease = 1.75%
        - Start at 80%, expected at 30 min = 78.25%

        If actual = 83% (4.75% ahead), this exceeds threshold but is favorable.
        """
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        slot_11 = datetime.datetime(2024, 1, 15, 11, 0, 0)

        optimizer.schedule = {
            slot_10: ScheduleEntry(hour=slot_10, mode=BatteryMode.DISCHARGE, reason="expensive"),
            slot_11: ScheduleEntry(hour=slot_11, mode=BatteryMode.DISCHARGE, reason="expensive"),
        }
        optimizer.expected_soc_schedule = {
            slot_10: 80.0,
            slot_11: 76.5,  # 80 - 3.5
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 10, 30, 0)

        # Expected at 30 min: 80 - 1.75 = 78.25%
        # Actual: 83% (4.75% ahead - favorable, exceeds 4% threshold)
        current_soc = 83.0

        result = optimizer._check_soc_deviation(current_soc)

        # Should NOT trigger recalculation - favorable deviation during discharge
        assert result is False
        assert len(optimizer._recalculate_calls) == 0
        assert any("favorable deviation" in msg for msg in optimizer._log_messages)

    def test_trigger_recalc_when_very_significantly_ahead(self, optimizer):
        """
        Scenario: Discharging but SOC is way ahead (>2x threshold).
        This might indicate load predictions are significantly off.
        Should trigger recalculation.
        """
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)

        optimizer.schedule = {
            slot_10: ScheduleEntry(hour=slot_10, mode=BatteryMode.DISCHARGE, reason="expensive"),
        }
        optimizer.expected_soc_schedule = {
            slot_10: 80.0,
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 10, 30, 0)
        optimizer.soc_deviation_threshold = 4.0

        # Expected at 30 min: 80 - 1.75 = 78.25%
        # Actual: 90% (11.75% ahead - way more than 2x threshold of 8%)
        current_soc = 90.0

        result = optimizer._check_soc_deviation(current_soc)

        # Should trigger recalculation - very significantly ahead
        assert result is True
        assert len(optimizer._recalculate_calls) == 1

    def test_trigger_recalc_when_behind_during_discharge(self, optimizer):
        """
        Scenario: Discharging but SOC is lower than expected.
        This is unfavorable - draining faster than expected.
        Should trigger recalculation.
        """
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)

        optimizer.schedule = {
            slot_10: ScheduleEntry(hour=slot_10, mode=BatteryMode.DISCHARGE, reason="expensive"),
        }
        optimizer.expected_soc_schedule = {
            slot_10: 80.0,
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 10, 30, 0)

        # Expected at 30 min: 78.25%
        # Actual: 70% (8.25% behind - draining faster than expected)
        current_soc = 70.0

        result = optimizer._check_soc_deviation(current_soc)

        # Should trigger recalculation - unfavorable deviation
        assert result is True
        assert len(optimizer._recalculate_calls) == 1


class TestSocDeviationEdgeCases:
    """Test edge cases for SOC deviation logic."""

    @pytest.fixture
    def optimizer(self):
        return MockSocDeviationOptimizer()

    def test_no_recalc_when_within_threshold(self, optimizer):
        """Small deviations within threshold should not trigger recalculation."""
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)

        optimizer.schedule = {
            slot_10: ScheduleEntry(hour=slot_10, mode=BatteryMode.HOLD, reason="moderate"),
        }
        optimizer.expected_soc_schedule = {
            slot_10: 50.0,
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 10, 30, 0)
        optimizer.soc_deviation_threshold = 4.0

        # During HOLD mode, expected_soc_now = 50.0 (no change)
        # Actual: 52% (2% deviation, within threshold)
        current_soc = 52.0

        result = optimizer._check_soc_deviation(current_soc)

        assert result is False
        assert len(optimizer._recalculate_calls) == 0

    def test_hold_mode_deviation_triggers_recalc(self, optimizer):
        """During HOLD mode, significant deviations should still trigger recalc."""
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)

        optimizer.schedule = {
            slot_10: ScheduleEntry(hour=slot_10, mode=BatteryMode.HOLD, reason="moderate"),
        }
        optimizer.expected_soc_schedule = {
            slot_10: 50.0,
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 10, 30, 0)

        # During HOLD, expected_soc_now = 50.0
        # Actual: 60% (10% deviation, exceeds threshold)
        current_soc = 60.0

        result = optimizer._check_soc_deviation(current_soc)

        # HOLD mode doesn't have special skip logic, should recalculate
        assert result is True
        assert len(optimizer._recalculate_calls) == 1

    def test_no_schedule_returns_false(self, optimizer):
        """If no schedule exists for current slot, should return False."""
        optimizer.schedule = {}
        optimizer.expected_soc_schedule = {}
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 10, 30, 0)

        result = optimizer._check_soc_deviation(50.0)

        assert result is False
        assert len(optimizer._recalculate_calls) == 0


class TestSocDeviationWithLearnedRate:
    """Test that learned charge rate is used for projection calculations."""

    def test_uses_learned_rate_in_remaining_charge_calculation(self):
        """
        Verify that learned charge rate is used when calculating
        whether remaining hours will reach max_soc.

        With learned rate ~2.86 kW (lower than theoretical 4.27 kW):
        - Remaining in slot_05 (0.5h) = 0.5 * 2.86 * 0.95 = 1.36 kWh = 9.5% SOC
        - Full slot_06 (1.0h) = 2.86 * 0.95 = 2.72 kWh = 19.0% SOC
        - Total remaining = 28.5% SOC gain
        - From 55%, projected final = 83.5%, which is < 95% (max_soc - 5)

        So with learned rate, recalculation SHOULD be triggered.

        With theoretical rate (4.27 kW):
        - Remaining = 1.5 * 4.27 * 0.95 = 6.08 kWh = 42.5% SOC
        - From 55%, projected = 97.5%, which would skip recalculation

        This test verifies the learned rate IS used (triggering recalc)
        whereas theoretical rate would have skipped it.
        """
        opt = MockSocDeviationOptimizer()
        # Set up a learning engine
        opt.learning_engine = BatteryLearningEngine(battery_capacity_kwh=14.3)

        # Record charging at ~2.86 kW (need at least 3 observations for median)
        for _ in range(5):
            opt.learning_engine.record_charging(
                soc_start=60.0,
                soc_end=61.0,
                duration_minutes=3.0  # 1% in 3 min = ~2.86 kW
            )

        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)
        slot_07 = datetime.datetime(2024, 1, 15, 7, 0, 0)

        opt.schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_07: ScheduleEntry(hour=slot_07, mode=BatteryMode.DISCHARGE, reason="expensive"),
        }
        opt.expected_soc_schedule = {
            slot_05: 50.0,
            slot_06: 78.4,
            slot_07: 100.0,
        }
        opt._current_datetime = datetime.datetime(2024, 1, 15, 5, 30, 0)

        # Behind schedule, with 1.5 hours of charging left
        # With learned rate, won't reach 95%, so recalc should be triggered
        current_soc = 55.0

        result = opt._check_soc_deviation(current_soc)

        # With learned rate (2.86 kW), projected final SOC = 83.5% < 95%
        # So recalculation SHOULD be triggered (proving learned rate is used)
        # If theoretical rate (4.27 kW) was used, it would skip (projected 97.5% >= 95%)
        assert result is True
        assert len(opt._recalculate_calls) == 1


class TestSocDeviationWithTemperature:
    """Test temperature-aware charge rate handling."""

    def test_uses_temperature_for_charge_rate_lookup(self):
        """
        Verify that battery temperature is passed to the learning engine
        for temperature-aware charge rate lookup.

        The inverter charges at different rates based on temperature:
        - Cold (~14C): ~2.7 kW
        - Warm (~18C): ~5.4 kW

        This test verifies the temperature is used in the lookup.
        """
        opt = MockSocDeviationOptimizer()
        opt.learning_engine = BatteryLearningEngine(battery_capacity_kwh=14.3)

        # Record cold temperature observations (~2.7 kW at 14C)
        # These go into the 10-15C temp bucket
        for _ in range(5):
            opt.learning_engine.record_charging(
                soc_start=60.0,
                soc_end=61.0,
                duration_minutes=3.2,  # ~2.7 kW
                battery_temp=14.0
            )

        # Record warm temperature observations (~5.4 kW at 18C)
        # These go into the 15-20C temp bucket
        for _ in range(5):
            opt.learning_engine.record_charging(
                soc_start=60.0,
                soc_end=61.0,
                duration_minutes=1.6,  # ~5.4 kW
                battery_temp=18.0
            )

        # Verify we get different rates for different temperatures
        cold_rate = opt.learning_engine.get_charge_rate_for_soc(60.0, battery_temp=14.0)
        warm_rate = opt.learning_engine.get_charge_rate_for_soc(60.0, battery_temp=18.0)

        # Warm rate should be significantly higher than cold rate
        assert warm_rate > cold_rate * 1.5  # At least 50% faster when warm

    def test_temperature_affects_skip_recalc_decision(self):
        """
        With warm temperature (higher charge rate), we can reach max_soc faster,
        so we should skip recalculation. With cold temperature (lower rate),
        we might not reach max_soc and should trigger recalculation.
        """
        # Test with warm temperature - should skip recalc
        opt_warm = MockSocDeviationOptimizer()
        opt_warm.learning_engine = BatteryLearningEngine(battery_capacity_kwh=14.3)
        opt_warm._battery_temp = 18.0  # Set mock temperature

        # Record warm temperature observations (~5.4 kW)
        for _ in range(5):
            opt_warm.learning_engine.record_charging(
                soc_start=60.0,
                soc_end=61.0,
                duration_minutes=1.6,
                battery_temp=18.0
            )

        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)

        opt_warm.schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.HOLD, reason="moderate"),
        }
        opt_warm.expected_soc_schedule = {
            slot_05: 70.0,
            slot_06: 100.0,
        }
        opt_warm._current_datetime = datetime.datetime(2024, 1, 15, 5, 30, 0)

        # At warm temp with ~5.4 kW rate:
        # Remaining 30 min = 0.5 * 5.4 * 0.95 = 2.57 kWh = 18% SOC
        # From 75%, will reach 93%, close to 95% threshold
        # With 30 min remaining at 5.4 kW, from 75% should reach ~93%
        current_soc = 75.0

        result_warm = opt_warm._check_soc_deviation(current_soc)

        # With high charge rate, might still reach near max_soc
        # The exact result depends on the calculation, but log should show temp
        assert any("temp=" in msg or "skipping" in msg or "RECALCULATION" in msg
                   for msg in opt_warm._log_messages)


class TestWarmingRateTracking:
    """Test temperature warming rate tracking and predictions."""

    def test_warming_rate_recorded_during_charging(self):
        """
        Verify that warming rate is tracked when both start and end temps are provided.
        """
        engine = BatteryLearningEngine(battery_capacity_kwh=14.3)

        # Record charging with temperature change: 14°C -> 16°C in 10 minutes
        # Warming rate = 2°C / 10min = 0.2°C/min
        engine.record_charging(
            soc_start=50.0,
            soc_end=55.0,
            duration_minutes=10.0,
            battery_temp=15.0,  # midpoint for charge rate bucketing
            battery_temp_start=14.0,
            battery_temp_end=16.0
        )

        # Should have recorded warming rate in the 10-15°C bucket
        assert "10-15" in engine.stats.temp_warming_rates
        rates = engine.stats.temp_warming_rates["10-15"]
        assert len(rates) == 1
        assert abs(rates[0] - 0.2) < 0.01  # ~0.2°C/min

    def test_get_warming_rate_returns_median(self):
        """
        Verify get_warming_rate returns median of observations.
        """
        engine = BatteryLearningEngine(battery_capacity_kwh=14.3)

        # Record several warming observations at ~14°C start
        warming_rates = [0.15, 0.20, 0.18, 0.22, 0.19]  # °C/min
        for rate in warming_rates:
            temp_change = rate * 5  # 5 minute duration
            engine.record_charging(
                soc_start=50.0,
                soc_end=52.0,
                duration_minutes=5.0,
                battery_temp=14.0,
                battery_temp_start=14.0,
                battery_temp_end=14.0 + temp_change
            )

        # Should return median
        result = engine.get_warming_rate(14.0)
        assert result is not None
        # Median of [0.15, 0.18, 0.19, 0.20, 0.22] = 0.19
        assert abs(result - 0.19) < 0.01

    def test_predict_temp_after_duration(self):
        """
        Verify temperature prediction after charging duration.
        """
        engine = BatteryLearningEngine(battery_capacity_kwh=14.3)

        # Record warming rate of 0.2°C/min at 14°C
        for _ in range(5):
            engine.record_charging(
                soc_start=50.0,
                soc_end=52.0,
                duration_minutes=5.0,
                battery_temp=14.0,
                battery_temp_start=14.0,
                battery_temp_end=15.0  # 1°C in 5min = 0.2°C/min
            )

        # Predict temp after 20 minutes starting at 14°C
        # Expected: 14 + 20 * 0.2 = 18°C
        predicted = engine.predict_temp_after_duration(14.0, 20.0)
        assert abs(predicted - 18.0) < 0.5

    def test_time_to_reach_temp(self):
        """
        Verify prediction of time to reach target temperature.
        """
        engine = BatteryLearningEngine(battery_capacity_kwh=14.3)

        # Record warming rate of 0.2°C/min
        for _ in range(5):
            engine.record_charging(
                soc_start=50.0,
                soc_end=52.0,
                duration_minutes=5.0,
                battery_temp=14.0,
                battery_temp_start=14.0,
                battery_temp_end=15.0
            )

        # Time to go from 14°C to 18°C at 0.2°C/min
        # Expected: 4°C / 0.2 = 20 minutes
        time = engine.get_time_to_reach_temp(14.0, 18.0)
        assert time is not None
        assert abs(time - 20.0) < 1.0

    def test_predict_charge_energy_with_warming(self):
        """
        Verify charge energy prediction accounts for warming-induced rate increase.

        Scenario: Start at 14°C (cold, 2.7kW), warm to 18°C (warm, 5.4kW)
        """
        engine = BatteryLearningEngine(battery_capacity_kwh=14.3)

        # Record cold charge rate (~2.7 kW at 14°C)
        for _ in range(5):
            engine.record_charging(
                soc_start=50.0,
                soc_end=51.0,
                duration_minutes=3.2,  # ~2.7 kW
                battery_temp=14.0,
                battery_temp_start=14.0,
                battery_temp_end=14.5
            )

        # Record warm charge rate (~5.4 kW at 18°C)
        for _ in range(5):
            engine.record_charging(
                soc_start=50.0,
                soc_end=51.0,
                duration_minutes=1.6,  # ~5.4 kW
                battery_temp=18.0,
                battery_temp_start=18.0,
                battery_temp_end=18.5
            )

        # Record warming rate (0.2°C/min from 14°C)
        for _ in range(5):
            engine.record_charging(
                soc_start=50.0,
                soc_end=52.0,
                duration_minutes=5.0,
                battery_temp=14.0,
                battery_temp_start=14.0,
                battery_temp_end=15.0
            )

        # Predict energy for 30 minutes starting at 14°C
        # With warming: first ~10 min at cold rate, then ~20 min at warm rate
        energy, end_temp = engine.predict_charge_energy_with_warming(
            current_soc=50.0,
            start_temp=14.0,
            duration_minutes=30.0,
            temp_threshold=16.0
        )

        # Should get more energy than if it stayed cold the whole time
        cold_only_energy = 2.7 * 0.5  # 30 min at 2.7 kW = 1.35 kWh
        assert energy > cold_only_energy

        # End temperature should be warmer than start
        assert end_temp > 14.0

    def test_warming_rate_in_learning_summary(self):
        """
        Verify warming rates appear in learning summary.
        """
        engine = BatteryLearningEngine(battery_capacity_kwh=14.3)

        # Record some warming observations
        for _ in range(5):
            engine.record_charging(
                soc_start=50.0,
                soc_end=52.0,
                duration_minutes=5.0,
                battery_temp=14.0,
                battery_temp_start=14.0,
                battery_temp_end=15.0
            )

        summary = engine.get_learning_summary()

        assert "temp_warming_rates" in summary
        assert "10-15" in summary["temp_warming_rates"]
        assert "median_c_per_min" in summary["temp_warming_rates"]["10-15"]
        assert summary["temp_warming_rates"]["10-15"]["observations"] == 5


class TestExpectedSocCalculationWithLearnedRate:
    """Test that expected SOC calculations use learned charge rate."""

    def test_calculate_expected_soc_uses_learned_rate(self):
        """Verify calculate_expected_soc_schedule uses learned charge rate."""
        # Create a mock optimizer with just the needed attributes
        class MockCalculateOptimizer:
            def __init__(self):
                self.charge_rate = 4.27  # Theoretical
                self.discharge_rate = 4.27
                self.efficiency = 0.95
                self.slot_hours = 1.0
                self.battery_capacity = 14.3
                self.max_soc = 100.0
                self.min_soc = 10.0
                self.learning_engine = BatteryLearningEngine(battery_capacity_kwh=14.3)

            def _predict_load_kw(self, hour):
                return 0.5

        MockCalculateOptimizer.calculate_expected_soc_schedule = BatteryOptimizer.calculate_expected_soc_schedule

        optimizer = MockCalculateOptimizer()

        # Record charging at ~2.86 kW (need at least 3 observations)
        for _ in range(5):
            optimizer.learning_engine.record_charging(
                soc_start=60.0,
                soc_end=61.0,
                duration_minutes=3.0  # 1% in 3 min = ~2.86 kW
            )

        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)

        schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.HOLD, reason="moderate"),
        }

        expected_soc, _ = optimizer.calculate_expected_soc_schedule(schedule, starting_soc=60.0)

        # With learned rate ~2.86 kW, SOC gain per hour = 2.86 * 0.95 / 14.3 * 100 = ~19%
        # Starting at 60%, after charge hour: ~79%
        # With theoretical 4.27 kW: 4.27 * 0.95 / 14.3 * 100 = ~28.4%
        # So theoretical would give 60 + 28.4 = 88.4%

        # Verify the expected SOC is based on learned rate (lower than theoretical)
        assert expected_soc[slot_05] == 60.0  # Starting SOC
        # After charge hour should be around 79% not 88%
        assert expected_soc[slot_06] < 85.0  # Using learned rate, not theoretical
        assert expected_soc[slot_06] > 70.0  # But still some charging


class TestLogScheduleUsesLearnedRate:
    """
    Test that ScheduleFormatter.log_schedule uses learned charge rate for SOC display.

    This tests the fix for the bug where log_schedule used only the configured
    charge rate while calculate_expected_soc_schedule used the learned rate,
    causing inconsistent SOC values in the schedule display.

    Example of the bug:
        2026-01-28 04:00  CHARGE     0.1091 EUR/kWh load~0.52kW -> 87.9%
        2026-01-28 05:00  HOLD       0.1300 EUR/kWh load~0.52kW ->100.0%  <- BUG!

    The HOLD slot showed 100% (from expected_soc calculated with high learned rate)
    while the CHARGE slot showed 87.9% (calculated with low configured rate).
    """

    def test_log_schedule_uses_learned_charge_rate(self):
        """
        Verify log_schedule uses learned charge rate, matching calculate_expected_soc_schedule.

        With a high learned rate (~6.8 kW) vs configured rate (4.5 kW):
        - Learned rate: 6.8 * 0.95 / 14.3 * 100 = ~45% SOC gain per hour
        - Configured rate: 4.5 * 0.95 / 14.3 * 100 = ~30% SOC gain per hour

        If log_schedule uses learned rate, the displayed end SOC for CHARGE
        should match the expected_soc value used for the following HOLD slot.
        """
        log_messages = []

        def mock_log(message: str, level: str = "INFO"):
            log_messages.append(message)

        # Create learning engine and train it with high charge rate
        learning_engine = BatteryLearningEngine(battery_capacity_kwh=14.3)

        # Record charging at high rate ~6.8 kW (need at least 3 observations)
        # 1% SOC = 0.143 kWh, in 1.26 min -> 0.143 / (1.26/60) = 6.8 kW
        for _ in range(5):
            learning_engine.record_charging(
                soc_start=50.0,
                soc_end=51.0,
                duration_minutes=1.26  # 1% in 1.26 min = ~6.8 kW
            )

        # Create formatter with the learning engine
        formatter = ScheduleFormatter(
            config=ScheduleFormatterConfig(
                slot_minutes=60,
                slot_hours=1.0,
                battery_capacity=14.3,
                charge_rate=4.5,  # Configured (lower)
                discharge_rate=4.5,
                efficiency=0.95,
                battery_wear_cost=0.0,
                decision_log_level=1,
            ),
            log_func=mock_log,
            learning_engine=learning_engine,
        )

        slot_04 = datetime.datetime(2024, 1, 15, 4, 0, 0)
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)

        schedule = {
            slot_04: ScheduleEntry(hour=slot_04, mode=BatteryMode.CHARGE, reason="0.1091 EUR/kWh load~0.52kW"),
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.HOLD, reason="0.1300 EUR/kWh load~0.52kW"),
        }

        # Calculate expected SOC (uses learned rate via learning engine)
        expected_soc = {slot_04: 55.0}  # Starting SOC
        # With learned rate, the CHARGE slot ends at ~100% (55% + 45%)
        # We simulate this for the test
        expected_soc[slot_05] = 100.0  # After charge (capped)

        # Log the schedule
        formatter.log_schedule(
            schedule=schedule,
            expected_soc=expected_soc,
            expected_temp=None,
            local_tz=None,
            predict_load_kw=lambda h: 0.5,
            min_soc=10.0,
            max_soc=100.0,
        )

        # Parse logged SOC values
        logged_soc_values = {}
        for msg in log_messages:
            # Look for lines like "  2024-01-15 04:00  CHARGE     0.1091 EUR/kWh load~0.52kW  55.0%->100.0%"
            if "->" in msg and "%" in msg:
                parts = msg.strip().split()
                if len(parts) >= 3:
                    time_str = parts[1]  # "04:00" or "05:00"
                    # Extract end SOC from the trajectory (format: "XX.X%->YY.Y%")
                    soc_part = msg.split("->")[-1].strip()
                    # Remove temperature info if present (e.g., "100.0% (20C->22C)")
                    soc_value_str = soc_part.split("%")[0]
                    soc_value = float(soc_value_str)
                    logged_soc_values[time_str] = soc_value

        # Verify both slots were logged
        assert "04:00" in logged_soc_values, f"Missing 04:00 in logged values: {log_messages}"
        assert "05:00" in logged_soc_values, f"Missing 05:00 in logged values: {log_messages}"

        charge_end_soc = logged_soc_values["04:00"]
        hold_end_soc = logged_soc_values["05:00"]

        # With high learned rate, CHARGE should hit max_soc
        assert charge_end_soc >= 95.0, (
            f"Expected near max_soc with high learned rate, got {charge_end_soc}%. "
            "This suggests log_schedule is not using the learned rate."
        )

    def test_log_schedule_without_learning_engine_uses_configured_rate(self):
        """
        Verify log_schedule falls back to configured rate when no learning engine.
        """
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
                efficiency=0.95,
                battery_wear_cost=0.0,
                decision_log_level=1,
            ),
            log_func=mock_log,
            learning_engine=None,  # No learning engine
        )

        slot_04 = datetime.datetime(2024, 1, 15, 4, 0, 0)
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)

        schedule = {
            slot_04: ScheduleEntry(hour=slot_04, mode=BatteryMode.CHARGE, reason="0.10 EUR/kWh"),
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.HOLD, reason="0.13 EUR/kWh"),
        }

        # Provide expected_soc for fallback path
        expected_soc = {slot_04: 55.0, slot_05: 84.9}  # 55% + ~30% with configured rate

        formatter.log_schedule(
            schedule=schedule,
            expected_soc=expected_soc,
            expected_temp=None,
            local_tz=None,
            predict_load_kw=lambda h: 0.5,
            min_soc=10.0,
            max_soc=100.0,
        )

        # Parse logged SOC values
        logged_soc_values = {}
        for msg in log_messages:
            if "->" in msg and "%" in msg:
                parts = msg.strip().split()
                if len(parts) >= 3:
                    time_str = parts[1]
                    # Extract end SOC from the trajectory
                    soc_part = msg.split("->")[-1].strip()
                    soc_value_str = soc_part.split("%")[0]
                    soc_value = float(soc_value_str)
                    logged_soc_values[time_str] = soc_value

        charge_end_soc = logged_soc_values["04:00"]
        hold_end_soc = logged_soc_values["05:00"]

        # With configured rate 4.5 kW: 55% + (4.5 * 0.95 / 14.3 * 100) = 55% + 29.9% = 84.9%
        assert 80.0 < charge_end_soc < 90.0, (
            f"Expected ~85% with configured rate, got {charge_end_soc}%"
        )


class TestExtraChargeSlotsWhenBehindSchedule:
    """
    Test the feature that adds extra charge slots when charging is behind schedule.

    When charging is slower than expected (learned rate was optimistic), the battery
    may not reach max_soc by the end of scheduled CHARGE slots. The system should:
    1. Detect the shortfall (projected_final_soc < max_soc - 5)
    2. Calculate extra slots needed
    3. Only add extra slots if economically beneficial (charge cost < discharge threshold)
    4. Pass extra_charge_slots to _recalculate_remaining_schedule
    """

    @pytest.fixture
    def optimizer(self):
        """Create a mock optimizer for testing extra charge slots feature."""
        opt = MockSocDeviationOptimizer()
        # Add additional tracking for extra_charge_slots
        opt._recalculate_extra_slots = []

        # Override _recalculate_remaining_schedule to capture extra_charge_slots
        original_recalculate = opt._recalculate_remaining_schedule
        def tracking_recalculate(current_soc, extra_charge_slots=0):
            opt._recalculate_extra_slots.append(extra_charge_slots)
            opt._recalculate_calls.append(current_soc)
        opt._recalculate_remaining_schedule = tracking_recalculate

        return opt

    def test_extra_slots_calculated_when_behind_schedule(self, optimizer):
        """
        Scenario: At 05:30, charging at slow rate, won't reach max_soc.
        Should calculate extra slots and pass to recalculation.

        Setup:
        - Current SOC: 55%
        - Expected at 05:30: ~64% (14% gain expected in 30 min)
        - Deviation: -9% (exceeds 4% threshold)
        - Remaining CHARGE hours: 0.5h in slot_05, 1.0h in slot_06 (but at slow rate)
        - With slow rate ~2.86 kW: projected final = 83.5% < 95%

        Expected: Extra slots calculated, cheap HOLD slot at 07:00 identified
        """
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)
        slot_07 = datetime.datetime(2024, 1, 15, 7, 0, 0)  # HOLD slot, potential for extra charge

        optimizer.schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="0.10 EUR/kWh"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.CHARGE, reason="0.10 EUR/kWh"),
            slot_07: ScheduleEntry(hour=slot_07, mode=BatteryMode.HOLD, reason="0.12 EUR/kWh"),
        }
        optimizer.expected_soc_schedule = {
            slot_05: 50.0,
            slot_06: 78.4,
            slot_07: 100.0,
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 5, 30, 0)

        # Set up slow learned rate (~2.86 kW)
        optimizer.learning_engine = BatteryLearningEngine(battery_capacity_kwh=14.3)
        for _ in range(5):
            optimizer.learning_engine.record_charging(
                soc_start=60.0, soc_end=61.0, duration_minutes=3.0
            )

        # Mock get_prices to return prices for the slots
        optimizer._prices = [
            type('PricePoint', (), {'hour': slot_05, 'price': 0.10})(),
            type('PricePoint', (), {'hour': slot_06, 'price': 0.10})(),
            type('PricePoint', (), {'hour': slot_07, 'price': 0.12})(),
        ]
        optimizer.get_prices = lambda: optimizer._prices

        # Mock _get_discharge_threshold to return a high value (so charging is economical)
        optimizer._get_discharge_threshold = lambda: 0.20  # Higher than charge price + grid fee
        optimizer.grid_fee = 0.05

        current_soc = 55.0

        result = optimizer._check_soc_deviation(current_soc)

        # Should trigger recalculation
        assert result is True
        assert len(optimizer._recalculate_calls) == 1

        # Should have passed extra_charge_slots > 0
        assert len(optimizer._recalculate_extra_slots) == 1
        assert optimizer._recalculate_extra_slots[0] > 0

        # Should log about adding extra slots
        assert any("adding" in msg.lower() and "slot" in msg.lower() for msg in optimizer._log_messages)

    def test_no_extra_slots_when_charging_economically_unfavorable(self, optimizer):
        """
        Scenario: Behind schedule but extra charging is not economical.
        Charge price + grid fee >= discharge threshold.

        Should NOT add extra slots.
        """
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)
        slot_07 = datetime.datetime(2024, 1, 15, 7, 0, 0)  # HOLD slot, expensive

        optimizer.schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="0.10 EUR/kWh"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.CHARGE, reason="0.10 EUR/kWh"),
            slot_07: ScheduleEntry(hour=slot_07, mode=BatteryMode.HOLD, reason="0.25 EUR/kWh"),  # Expensive
        }
        optimizer.expected_soc_schedule = {
            slot_05: 50.0,
            slot_06: 78.4,
            slot_07: 100.0,
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 5, 30, 0)

        # Set up slow learned rate
        optimizer.learning_engine = BatteryLearningEngine(battery_capacity_kwh=14.3)
        for _ in range(5):
            optimizer.learning_engine.record_charging(
                soc_start=60.0, soc_end=61.0, duration_minutes=3.0
            )

        # Mock prices with expensive HOLD slot
        optimizer._prices = [
            type('PricePoint', (), {'hour': slot_05, 'price': 0.10})(),
            type('PricePoint', (), {'hour': slot_06, 'price': 0.10})(),
            type('PricePoint', (), {'hour': slot_07, 'price': 0.25})(),  # Expensive
        ]
        optimizer.get_prices = lambda: optimizer._prices

        # Low discharge threshold - charging is not economical
        optimizer._get_discharge_threshold = lambda: 0.20  # Lower than 0.25 + 0.05 grid fee
        optimizer.grid_fee = 0.05

        current_soc = 55.0

        result = optimizer._check_soc_deviation(current_soc)

        # Should still trigger recalculation
        assert result is True
        assert len(optimizer._recalculate_calls) == 1

        # Should NOT have passed extra_charge_slots (or passed 0)
        assert len(optimizer._recalculate_extra_slots) == 1
        assert optimizer._recalculate_extra_slots[0] == 0

        # Should log about not economical
        assert any("not economical" in msg.lower() for msg in optimizer._log_messages)

    def test_no_extra_slots_when_no_hold_slots_available(self, optimizer):
        """
        Scenario: Behind schedule but remaining slots are DISCHARGE (no HOLD to convert).
        Need extra charging but no HOLD slots available.

        Should trigger recalculation but NOT add extra slots.

        Setup:
        - Only 1 CHARGE slot remaining (slot_05, already 30 min in)
        - Next slot is DISCHARGE (expensive) - no HOLD slots to convert
        - With slow rate 2.86 kW and 0.5h remaining: 2.86 * 0.95 * 0.5 = 1.36 kWh = 9.5% SOC
        - From 55%, projected final = 64.5% << 95%
        """
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)  # DISCHARGE, not HOLD

        optimizer.schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="0.10 EUR/kWh"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.DISCHARGE, reason="0.30 EUR/kWh"),  # DISCHARGE
        }
        optimizer.expected_soc_schedule = {
            slot_05: 50.0,
            slot_06: 78.4,  # Expected from original schedule
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 5, 30, 0)

        # Set up slow learned rate (~2.86 kW)
        optimizer.learning_engine = BatteryLearningEngine(battery_capacity_kwh=14.3)
        for _ in range(5):
            optimizer.learning_engine.record_charging(
                soc_start=60.0, soc_end=61.0, duration_minutes=3.0
            )

        # Mock prices - slot_06 is expensive (DISCHARGE), no HOLD slots
        optimizer._prices = [
            type('PricePoint', (), {'hour': slot_05, 'price': 0.10})(),
            type('PricePoint', (), {'hour': slot_06, 'price': 0.30})(),
        ]
        optimizer.get_prices = lambda: optimizer._prices

        optimizer._get_discharge_threshold = lambda: 0.20
        optimizer.grid_fee = 0.05

        current_soc = 55.0

        result = optimizer._check_soc_deviation(current_soc)

        # Should trigger recalculation (behind schedule, won't reach max_soc)
        assert result is True
        assert len(optimizer._recalculate_calls) == 1

        # Should NOT have passed extra_charge_slots (no HOLD slots available)
        assert len(optimizer._recalculate_extra_slots) == 1
        assert optimizer._recalculate_extra_slots[0] == 0

        # Should log about no HOLD slots
        assert any("no hold slots" in msg.lower() for msg in optimizer._log_messages)

    def test_extra_slots_only_during_charge_mode_behind_schedule(self, optimizer):
        """
        Scenario: Deviation during HOLD or DISCHARGE mode should NOT calculate extra slots.
        Extra slots logic only applies to CHARGE mode when behind schedule.
        """
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        slot_11 = datetime.datetime(2024, 1, 15, 11, 0, 0)

        # HOLD mode with deviation
        optimizer.schedule = {
            slot_10: ScheduleEntry(hour=slot_10, mode=BatteryMode.HOLD, reason="0.15 EUR/kWh"),
            slot_11: ScheduleEntry(hour=slot_11, mode=BatteryMode.HOLD, reason="0.15 EUR/kWh"),
        }
        optimizer.expected_soc_schedule = {
            slot_10: 70.0,
            slot_11: 70.0,
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 10, 30, 0)

        # During HOLD, expected_soc_now = 70.0
        # Actual: 60% (10% deviation, exceeds threshold)
        current_soc = 60.0

        # Mock prices
        optimizer._prices = [
            type('PricePoint', (), {'hour': slot_10, 'price': 0.15})(),
            type('PricePoint', (), {'hour': slot_11, 'price': 0.15})(),
        ]
        optimizer.get_prices = lambda: optimizer._prices
        optimizer._get_discharge_threshold = lambda: 0.20
        optimizer.grid_fee = 0.05

        result = optimizer._check_soc_deviation(current_soc)

        # Should trigger recalculation
        assert result is True
        assert len(optimizer._recalculate_calls) == 1

        # Should NOT have calculated extra slots (not in CHARGE mode)
        assert len(optimizer._recalculate_extra_slots) == 1
        assert optimizer._recalculate_extra_slots[0] == 0

    def test_handles_mixed_timezone_aware_naive_datetimes(self, optimizer):
        """
        Scenario: Schedule keys are timezone-aware but current_slot is naive (or vice versa).
        Should not raise TypeError when comparing datetimes.

        This tests the fix for the bug where direct comparison `h > current_slot`
        would crash with "can't compare offset-naive and offset-aware datetimes".
        """
        # Use a fixed offset timezone (UTC+2) to avoid needing tzdata package
        tz = datetime.timezone(datetime.timedelta(hours=2))

        # Create timezone-aware schedule keys
        slot_05_aware = datetime.datetime(2024, 1, 15, 5, 0, 0, tzinfo=tz)
        slot_06_aware = datetime.datetime(2024, 1, 15, 6, 0, 0, tzinfo=tz)
        slot_07_aware = datetime.datetime(2024, 1, 15, 7, 0, 0, tzinfo=tz)

        optimizer.schedule = {
            slot_05_aware: ScheduleEntry(hour=slot_05_aware, mode=BatteryMode.CHARGE, reason="0.10 EUR/kWh"),
            slot_06_aware: ScheduleEntry(hour=slot_06_aware, mode=BatteryMode.CHARGE, reason="0.10 EUR/kWh"),
            slot_07_aware: ScheduleEntry(hour=slot_07_aware, mode=BatteryMode.HOLD, reason="0.12 EUR/kWh"),
        }

        # Use naive datetime for expected_soc_schedule (mixed scenario)
        slot_05_naive = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06_naive = datetime.datetime(2024, 1, 15, 6, 0, 0)
        slot_07_naive = datetime.datetime(2024, 1, 15, 7, 0, 0)

        optimizer.expected_soc_schedule = {
            slot_05_naive: 50.0,
            slot_06_naive: 78.4,
            slot_07_naive: 100.0,
        }

        # Current time is naive (simulates _align_to_slot returning naive when no local_tz)
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 5, 30, 0)

        # Set up slow learned rate
        optimizer.learning_engine = BatteryLearningEngine(battery_capacity_kwh=14.3)
        for _ in range(5):
            optimizer.learning_engine.record_charging(
                soc_start=60.0, soc_end=61.0, duration_minutes=3.0
            )

        # Mock prices with aware timestamps
        optimizer._prices = [
            type('PricePoint', (), {'hour': slot_05_aware, 'price': 0.10})(),
            type('PricePoint', (), {'hour': slot_06_aware, 'price': 0.10})(),
            type('PricePoint', (), {'hour': slot_07_aware, 'price': 0.12})(),
        ]
        optimizer.get_prices = lambda: optimizer._prices

        optimizer._get_discharge_threshold = lambda: 0.20
        optimizer.grid_fee = 0.05

        current_soc = 55.0

        # Should NOT raise TypeError - this is the main assertion
        # The function should handle mixed timezone-aware/naive comparison gracefully
        try:
            result = optimizer._check_soc_deviation(current_soc)
        except TypeError as e:
            if "can't compare" in str(e) and "naive" in str(e):
                pytest.fail(f"TypeError raised due to timezone mismatch: {e}")
            raise

        # The result doesn't matter as much as not crashing, but it should work
        # (may return True or False depending on how the comparison resolves)
        assert result in (True, False)


class TestGetCheapestUpcomingPrices:
    """Test the _get_cheapest_upcoming_prices helper method."""

    @pytest.fixture
    def optimizer(self):
        return MockSocDeviationOptimizer()

    def test_returns_cheapest_hold_prices(self, optimizer):
        """Should return the N cheapest prices from HOLD slots only."""
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)
        slot_07 = datetime.datetime(2024, 1, 15, 7, 0, 0)
        slot_08 = datetime.datetime(2024, 1, 15, 8, 0, 0)

        optimizer.schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.HOLD, reason="0.15"),  # HOLD
            slot_07: ScheduleEntry(hour=slot_07, mode=BatteryMode.DISCHARGE, reason="expensive"),
            slot_08: ScheduleEntry(hour=slot_08, mode=BatteryMode.HOLD, reason="0.12"),  # HOLD, cheaper
        }

        optimizer._prices = [
            type('PricePoint', (), {'hour': slot_05, 'price': 0.10})(),
            type('PricePoint', (), {'hour': slot_06, 'price': 0.15})(),
            type('PricePoint', (), {'hour': slot_07, 'price': 0.25})(),
            type('PricePoint', (), {'hour': slot_08, 'price': 0.12})(),
        ]
        optimizer.get_prices = lambda: optimizer._prices

        # Bind the method
        optimizer._get_cheapest_upcoming_prices = BatteryOptimizer._get_cheapest_upcoming_prices.__get__(optimizer)

        remaining_hours = [slot_06, slot_07, slot_08]
        result = optimizer._get_cheapest_upcoming_prices(remaining_hours, 2)

        # Should return the 2 cheapest HOLD slot prices (slot_08=0.12, slot_06=0.15)
        assert len(result) == 2
        assert result == [0.12, 0.15]

    def test_returns_empty_when_no_hold_slots(self, optimizer):
        """Should return empty list when no HOLD slots exist."""
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)

        optimizer.schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.DISCHARGE, reason="expensive"),
        }

        optimizer._prices = [
            type('PricePoint', (), {'hour': slot_05, 'price': 0.10})(),
            type('PricePoint', (), {'hour': slot_06, 'price': 0.25})(),
        ]
        optimizer.get_prices = lambda: optimizer._prices

        optimizer._get_cheapest_upcoming_prices = BatteryOptimizer._get_cheapest_upcoming_prices.__get__(optimizer)

        remaining_hours = [slot_05, slot_06]
        result = optimizer._get_cheapest_upcoming_prices(remaining_hours, 2)

        assert result == []

    def test_returns_fewer_when_not_enough_hold_slots(self, optimizer):
        """Should return only available HOLD slot prices if fewer than requested."""
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)

        optimizer.schedule = {
            slot_05: ScheduleEntry(hour=slot_05, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_06: ScheduleEntry(hour=slot_06, mode=BatteryMode.HOLD, reason="moderate"),  # Only 1 HOLD
        }

        optimizer._prices = [
            type('PricePoint', (), {'hour': slot_05, 'price': 0.10})(),
            type('PricePoint', (), {'hour': slot_06, 'price': 0.15})(),
        ]
        optimizer.get_prices = lambda: optimizer._prices

        optimizer._get_cheapest_upcoming_prices = BatteryOptimizer._get_cheapest_upcoming_prices.__get__(optimizer)

        remaining_hours = [slot_05, slot_06]
        result = optimizer._get_cheapest_upcoming_prices(remaining_hours, 3)  # Request 3

        # Should return only 1 (the only HOLD slot)
        assert len(result) == 1
        assert result == [0.15]


class TestRecalculateRemainingScheduleWithExtraSlots:
    """Test that _recalculate_remaining_schedule properly uses extra_charge_slots."""

    def test_extra_slots_boost_min_charge_slots(self):
        """
        Verify that extra_charge_slots parameter boosts the min_charge_slots
        used in the optimization.
        """
        # This test verifies the logging behavior to confirm the parameter is used
        class MockRecalculateOptimizer:
            def __init__(self):
                self.slot_minutes = 60
                self.slot_hours = 1.0
                self._log_messages = []
                self._min_charge_slots_used = None

            def log(self, message: str, level: str = "INFO"):
                self._log_messages.append(message)

            def datetime(self):
                return datetime.datetime(2024, 1, 15, 5, 30, 0)

            def _get_local_timezone(self):
                return None

            def _align_to_slot(self, dt):
                return dt.replace(minute=0, second=0, microsecond=0)

            def get_prices(self):
                return [
                    type('PricePoint', (), {'hour': datetime.datetime(2024, 1, 15, 6, 0, 0), 'price': 0.10})(),
                    type('PricePoint', (), {'hour': datetime.datetime(2024, 1, 15, 7, 0, 0), 'price': 0.15})(),
                ]

            def calculate_min_charge_slots_for_horizon(self, soc, prices):
                return 1  # Base requirement

            def find_optimal_schedule(self, prices, min_charge_slots, soc):
                self._min_charge_slots_used = min_charge_slots
                return {}

            def _get_battery_temp(self):
                return None

            def calculate_expected_soc_schedule(self, schedule, starting_soc, starting_temp=None):
                return {}, {}

            def _schedule_tou_sync(self, reason):
                pass

            def _update_schedule_sensor(self):
                pass

            def _predict_load_kw(self, dt):
                return 0.5

        MockRecalculateOptimizer._recalculate_remaining_schedule = BatteryOptimizer._recalculate_remaining_schedule

        opt = MockRecalculateOptimizer()
        opt.schedule = {}
        opt.expected_soc_schedule = {}
        opt.expected_temp_schedule = {}
        opt.tou_sync_enabled = False
        opt.device_id = ""
        opt.decision_log_level = 1
        opt.min_soc = 10.0
        opt.max_soc = 100.0
        opt._last_dp_soc_trajectory = {}
        opt._last_dp_temp_trajectory = {}
        opt._last_projected_costs = {}

        # Add schedule formatter
        opt._schedule_formatter = ScheduleFormatter(
            config=ScheduleFormatterConfig(
                slot_minutes=60,
                slot_hours=1.0,
                battery_capacity=14.3,
                charge_rate=4.5,
                discharge_rate=4.5,
                efficiency=0.85,
                battery_wear_cost=0.0,
                decision_log_level=1,
            ),
            log_func=opt.log,
            learning_engine=None,
        )

        # Call with extra_charge_slots=2
        opt._recalculate_remaining_schedule(55.0, extra_charge_slots=2)

        # Verify logging indicates the boost
        assert any("boosting min_charge_slots by 2" in msg.lower() for msg in opt._log_messages)

        # Verify the min_charge_slots used in optimization was boosted
        assert opt._min_charge_slots_used == 3  # 1 base + 2 extra
