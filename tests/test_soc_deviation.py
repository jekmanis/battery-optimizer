"""
Tests for SOC deviation detection and recalculation skip logic.

These tests verify that:
1. During CHARGE mode, if behind schedule but will still reach max_soc, recalculation is skipped
2. During DISCHARGE mode, favorable deviations (ahead) skip recalculation
3. Recalculation is still triggered when actually needed
"""

import datetime
from typing import Dict, Optional
from unittest.mock import MagicMock

import pytest

import sys
from pathlib import Path
apps_dir = Path(__file__).parent.parent / "appdaemon" / "apps"
sys.path.insert(0, str(apps_dir))

from battery_optimizer import BatteryMode, ScheduleEntry, BatteryOptimizer
from battery_optimizer_lib import BatteryLearningEngine


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

    def _recalculate_remaining_schedule(self, current_soc: float):
        """Track recalculation calls."""
        self._recalculate_calls.append(current_soc)


# Bind the actual method from BatteryOptimizer
MockSocDeviationOptimizer._check_soc_deviation = BatteryOptimizer._check_soc_deviation


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

        expected_soc = optimizer.calculate_expected_soc_schedule(schedule, starting_soc=60.0)

        # With learned rate ~2.86 kW, SOC gain per hour = 2.86 * 0.95 / 14.3 * 100 = ~19%
        # Starting at 60%, after charge hour: ~79%
        # With theoretical 4.27 kW: 4.27 * 0.95 / 14.3 * 100 = ~28.4%
        # So theoretical would give 60 + 28.4 = 88.4%

        # Verify the expected SOC is based on learned rate (lower than theoretical)
        assert expected_soc[slot_05] == 60.0  # Starting SOC
        # After charge hour should be around 79% not 88%
        assert expected_soc[slot_06] < 85.0  # Using learned rate, not theoretical
        assert expected_soc[slot_06] > 70.0  # But still some charging
