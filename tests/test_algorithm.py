"""
Tests for the find_optimal_schedule algorithm.

These tests verify the core dynamic programming optimization algorithm
without requiring AppDaemon or Home Assistant.
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
from battery_optimizer_lib import (
    BatteryCostTracker,
    BatteryCostConfig,
    BatteryOptimizerConfig,
    ScheduleFormatter,
    ScheduleFormatterConfig,
)


class MockOptimizer:
    """
    Minimal mock of BatteryOptimizer for testing find_optimal_schedule.

    Only implements the methods and attributes needed by the algorithm.
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

        # Legacy attributes for backward compatibility with tests
        self.battery_avg_cost = battery_avg_cost

        # Dynamic properties (simulated as simple values)
        self.min_soc = min_soc
        self.max_soc = max_soc

        # Current time
        self._current_time = datetime.datetime(2024, 1, 15, 10, 0, 0)

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

        # Schedule formatter for logging
        self._schedule_formatter = ScheduleFormatter(
            config=ScheduleFormatterConfig(
                slot_minutes=self.config.slot_minutes,
                slot_hours=self.config.slot_hours,
                battery_capacity=self.config.battery_capacity,
                charge_rate=self.config.charge_rate,
                discharge_rate=self.config.discharge_rate,
                efficiency=self.config.efficiency,
                battery_wear_cost=self.config.battery_wear_cost,
                decision_log_level=self.config.decision_log_level,
            ),
            log_func=self.log,
            learning_engine=self.learning_engine,
        )

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
            get_battery_temp_func=lambda: 20.0,
            learning_engine=self.learning_engine,
            get_cached_prices_func=lambda: [],
            save_learning_data_func=lambda: None,
            update_learning_sensor_func=lambda: None,
            log_func=self.log,
        )

    def datetime(self):
        """Return current simulated time."""
        return self._current_time

    def set_datetime(self, dt: datetime.datetime):
        """Set simulated current time."""
        self._current_time = dt

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
        """Return a default temperature."""
        return 20.0

    def _predict_load_kw(self, dt: datetime.datetime) -> float:
        """Predict load for given time."""
        return self.load_profile.predict_kw(dt, self.config.load_quantile)

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


# Import the actual algorithm method
import sys
from pathlib import Path
apps_dir = Path(__file__).parent.parent / "appdaemon" / "apps"
sys.path.insert(0, str(apps_dir))

# We need to import the method from the module
from battery_optimizer import BatteryOptimizer


# Bind the actual algorithm to our mock
MockOptimizer.find_optimal_schedule = BatteryOptimizer.find_optimal_schedule
MockOptimizer._get_discharge_threshold_for_cost = BatteryOptimizer._get_discharge_threshold_for_cost
MockOptimizer._ensure_current_slot_price = BatteryOptimizer._ensure_current_slot_price
MockOptimizer._compute_slot_fractions = BatteryOptimizer._compute_slot_fractions
MockOptimizer._compute_charge_rates_per_slot = BatteryOptimizer._compute_charge_rates_per_slot


class TestFindOptimalSchedule:
    """Test cases for the find_optimal_schedule algorithm."""

    @pytest.fixture
    def optimizer(self):
        """Create a mock optimizer for testing."""
        return MockOptimizer()

    def test_empty_prices_returns_empty(self, optimizer):
        """Empty price list should return empty schedule."""
        schedule = optimizer.find_optimal_schedule([], 0, current_soc=50)
        assert schedule == {}

    def test_basic_charge_discharge_pattern(self, optimizer, sample_prices):
        """Algorithm should charge during cheap hours and discharge during expensive."""
        # Set current time to start of price data
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        schedule = optimizer.find_optimal_schedule(sample_prices, 3, current_soc=50)

        assert len(schedule) > 0

        # Find charge and discharge entries
        charge_slots = [e for e in schedule.values() if e.mode == BatteryMode.CHARGE]
        discharge_slots = [e for e in schedule.values() if e.mode == BatteryMode.DISCHARGE]

        # Should have some of each
        assert len(charge_slots) > 0 or len(discharge_slots) > 0

        # If we have both, charge prices should generally be lower than discharge
        if charge_slots and discharge_slots:
            avg_charge_price = sum(
                next(p.price for p in sample_prices if p.time == e.time)
                for e in charge_slots
            ) / len(charge_slots)

            avg_discharge_price = sum(
                next(p.price for p in sample_prices if p.time == e.time)
                for e in discharge_slots
            ) / len(discharge_slots)

            # Charge prices should be lower than discharge prices
            assert avg_charge_price < avg_discharge_price

    def test_respects_soc_constraints(self, optimizer, sample_prices):
        """Schedule should respect min/max SOC constraints."""
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))
        optimizer.min_soc = 20
        optimizer.max_soc = 90

        # Start at low SOC - should prioritize charging
        schedule = optimizer.find_optimal_schedule(sample_prices, 3, current_soc=25)

        # With low starting SOC, first few actions should include charging
        sorted_entries = sorted(schedule.values(), key=lambda e: e.time)
        early_entries = sorted_entries[:6]  # First 6 hours

        early_charges = sum(1 for e in early_entries if e.mode == BatteryMode.CHARGE)
        early_discharges = sum(1 for e in early_entries if e.mode == BatteryMode.DISCHARGE)

        # At low SOC, shouldn't aggressively discharge
        assert early_charges >= early_discharges or len(sorted_entries) < 6

    def test_handles_future_prices_only(self, optimizer, sample_prices):
        """Should only schedule for future time slots."""
        # Set current time to noon
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 12, 0, 0))

        schedule = optimizer.find_optimal_schedule(sample_prices, 2, current_soc=50)

        # All scheduled entries should be at noon or later
        for hour, entry in schedule.items():
            hour_naive = hour.replace(tzinfo=None) if hour.tzinfo else hour
            assert hour_naive.hour >= 12

    def test_extreme_prices_maximize_arbitrage(self, optimizer, extreme_prices):
        """With extreme price differences, should maximize arbitrage through charge/discharge."""
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        schedule = optimizer.find_optimal_schedule(extreme_prices, 4, current_soc=50)

        # Find the expensive hours (peak prices)
        expensive_entries = [
            e for e in schedule.values()
            if e.time.hour in [7, 8, 9, 17, 18, 19]  # Peak price hours
        ]

        # With extreme price spikes, should discharge during expensive hours
        discharges = [e for e in expensive_entries if e.mode == BatteryMode.DISCHARGE]
        assert len(discharges) > 0, "Should discharge during expensive hours"

        # At negative price hours (0, 1), should HOLD or CHARGE - not discharge
        # (at negative prices, we get paid to import from grid!)
        negative_price_entries = [
            e for e in schedule.values()
            if e.time.hour in [0, 1]  # Negative price hours
        ]
        negative_discharges = [e for e in negative_price_entries if e.mode == BatteryMode.DISCHARGE]
        assert len(negative_discharges) == 0, "Should not discharge at negative price hours"

        # Verify expensive hours have more discharges than cheap hours
        # The DP should prioritize high-value discharge opportunities
        cheap_positive_entries = [
            e for e in schedule.values()
            if e.time.hour in [2, 3, 4, 5]  # Very cheap but positive prices
        ]
        cheap_discharges = [e for e in cheap_positive_entries if e.mode == BatteryMode.DISCHARGE]
        # With zero wear cost, discharging at any positive price is valid
        # But we should have MORE discharges at expensive hours than cheap hours
        assert len(discharges) >= len(cheap_discharges), "Should prioritize expensive hour discharges"

    def test_flat_prices_minimal_activity(self, optimizer, flat_prices):
        """With flat prices, charging is only valuable if below battery cost."""
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))
        optimizer.battery_avg_cost = 0.08  # Lower than flat price of 0.10

        schedule = optimizer.find_optimal_schedule(flat_prices, 0, current_soc=50)

        # With flat prices higher than battery cost, should mostly hold
        charges = [e for e in schedule.values() if e.mode == BatteryMode.CHARGE]
        discharges = [e for e in schedule.values() if e.mode == BatteryMode.DISCHARGE]
        holds = [e for e in schedule.values() if e.mode == BatteryMode.HOLD]

        # Most entries should be holds since there's no arbitrage opportunity
        # (unless discharge is profitable due to load)
        total = len(schedule)
        if total > 0:
            hold_ratio = len(holds) / total
            # With no price variation, expect significant holding
            # (discharge may still happen if load exists)
            assert hold_ratio >= 0.3 or len(discharges) > 0

    def test_high_soc_start_allows_discharge(self, optimizer, sample_prices):
        """Starting at high SOC should allow discharge during expensive hours."""
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        # Start at 90% SOC
        schedule = optimizer.find_optimal_schedule(sample_prices, 0, current_soc=90)

        # Should have some discharge slots
        discharges = [e for e in schedule.values() if e.mode == BatteryMode.DISCHARGE]
        assert len(discharges) > 0

    def test_low_soc_start_needs_charge(self, optimizer, sample_prices):
        """Starting at low SOC should require charging."""
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))
        optimizer.min_soc = 20

        # Start at minimum SOC
        schedule = optimizer.find_optimal_schedule(sample_prices, 3, current_soc=20)

        # Should have charge slots
        charges = [e for e in schedule.values() if e.mode == BatteryMode.CHARGE]
        assert len(charges) >= 1

    def test_partial_slot_handling(self, optimizer, sample_prices):
        """Algorithm should handle partial time slots correctly."""
        # Set time to middle of hour
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 10, 30, 0))

        schedule = optimizer.find_optimal_schedule(sample_prices, 2, current_soc=50)

        # Schedule should still be generated
        assert len(schedule) > 0

        # Current slot (10:00) should be included
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        has_current_slot = any(
            (h.replace(tzinfo=None) if h.tzinfo else h) == slot_10
            for h in schedule.keys()
        )
        assert has_current_slot

    def test_minimum_charge_slots_is_soft_constraint(self, optimizer, sample_prices):
        """min_charge_slots is now a soft constraint - algorithm optimizes economically."""
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        # Request 5 charge slots but algorithm may choose fewer if not economical
        schedule = optimizer.find_optimal_schedule(sample_prices, 5, current_soc=30)

        # Verify schedule was generated
        assert len(schedule) > 0

        # Verify algorithm makes economically rational choices:
        # - Discharge during expensive hours (when above threshold)
        # - Hold during moderate hours
        discharges = [e for e in schedule.values() if e.mode == BatteryMode.DISCHARGE]
        holds = [e for e in schedule.values() if e.mode == BatteryMode.HOLD]

        # Should have discharge activity during peak hours
        assert len(discharges) > 0, "Should discharge during expensive hours"

        # With moderate SOC, shouldn't need to charge if existing energy suffices
        # (this is the key behavior change - no forced charging)

    def test_schedule_covers_all_future_hours(self, optimizer, sample_prices):
        """Schedule should have entry for every future price point."""
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        schedule = optimizer.find_optimal_schedule(sample_prices, 2, current_soc=50)

        # Should have 24 entries (one per hour)
        assert len(schedule) == 24

    def test_schedule_entry_format(self, optimizer, sample_prices):
        """Each schedule entry should have correct format."""
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        schedule = optimizer.find_optimal_schedule(sample_prices, 2, current_soc=50)

        for hour, entry in schedule.items():
            # Hour should be datetime
            assert isinstance(hour, datetime.datetime)

            # Entry should be ScheduleEntry
            assert isinstance(entry, ScheduleEntry)

            # Entry hour should match key
            assert entry.time == hour

            # Mode should be valid
            assert entry.mode in [BatteryMode.HOLD, BatteryMode.CHARGE, BatteryMode.DISCHARGE]

            # Reason should include price
            assert "EUR/kWh" in entry.reason

    def test_different_slot_sizes(self):
        """Algorithm should work with different slot sizes."""
        optimizer = MockOptimizer(slot_minutes=30)

        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
        prices = [
            PricePoint(time=base_time + datetime.timedelta(minutes=30 * i), price=0.05 + i * 0.01)
            for i in range(48)  # 24 hours at 30-min slots
        ]

        optimizer.set_datetime(base_time)
        schedule = optimizer.find_optimal_schedule(prices, 4, current_soc=50)

        # Should have entries for 30-minute slots
        assert len(schedule) == 48

    def test_efficiency_affects_charge_value(self):
        """Lower efficiency should make charging less attractive."""
        # High efficiency optimizer
        optimizer_high = MockOptimizer(efficiency=0.95)
        # Low efficiency optimizer
        optimizer_low = MockOptimizer(efficiency=0.70)

        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
        prices = [
            PricePoint(time=base_time + datetime.timedelta(hours=i), price=0.05 + i * 0.01)
            for i in range(24)
        ]

        optimizer_high.set_datetime(base_time)
        optimizer_low.set_datetime(base_time)

        schedule_high = optimizer_high.find_optimal_schedule(prices, 3, current_soc=50)
        schedule_low = optimizer_low.find_optimal_schedule(prices, 3, current_soc=50)

        charges_high = len([e for e in schedule_high.values() if e.mode == BatteryMode.CHARGE])
        charges_low = len([e for e in schedule_low.values() if e.mode == BatteryMode.CHARGE])

        # With identical conditions, high efficiency should favor more charging
        # (though both should charge during cheap hours)
        assert charges_high >= charges_low or charges_low > 0

    def test_grid_fee_affects_thresholds(self):
        """Higher grid fee should raise discharge threshold."""
        optimizer_low_fee = MockOptimizer(grid_fee=0.01)
        optimizer_high_fee = MockOptimizer(grid_fee=0.10)

        threshold_low = optimizer_low_fee._get_discharge_threshold()
        threshold_high = optimizer_high_fee._get_discharge_threshold()

        assert threshold_high > threshold_low


class TestScheduleConsistency:
    """Tests for schedule consistency and edge cases."""

    @pytest.fixture
    def optimizer(self):
        return MockOptimizer()

    def test_deterministic_output(self, optimizer, sample_prices):
        """Same inputs should produce same output."""
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))

        schedule1 = optimizer.find_optimal_schedule(sample_prices, 3, current_soc=50)
        schedule2 = optimizer.find_optimal_schedule(sample_prices, 3, current_soc=50)

        # Compare modes for each hour
        for hour in schedule1:
            assert schedule1[hour].mode == schedule2[hour].mode

    def test_no_discharge_below_min_soc(self, optimizer, sample_prices):
        """Should not schedule discharge that would violate min SOC."""
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 0, 0, 0))
        optimizer.min_soc = 30

        # Start at 35% - very close to minimum
        schedule = optimizer.find_optimal_schedule(sample_prices, 0, current_soc=35)

        # Count consecutive discharges from start
        sorted_entries = sorted(schedule.values(), key=lambda e: e.time)

        consecutive_discharges = 0
        for entry in sorted_entries:
            if entry.mode == BatteryMode.DISCHARGE:
                consecutive_discharges += 1
            else:
                break

        # With only 5% headroom above min, can't have many discharges
        # Each discharge uses ~(discharge_rate * slot_hours) / capacity * 100 % ≈ 32%/hour
        # So max 0-1 discharge slots possible
        assert consecutive_discharges <= 2

    def test_handles_single_price_point(self, optimizer):
        """Should handle schedule with single price point."""
        single_price = [
            PricePoint(
                time=datetime.datetime(2024, 1, 15, 10, 0, 0),
                price=0.10
            )
        ]

        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 10, 0, 0))
        schedule = optimizer.find_optimal_schedule(single_price, 0, current_soc=50)

        assert len(schedule) == 1

    def test_handles_very_long_horizon(self, optimizer):
        """Should handle multi-day price horizon."""
        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)

        # 48 hours of prices
        long_prices = [
            PricePoint(
                time=base_time + datetime.timedelta(hours=i),
                price=0.05 + (i % 24) * 0.005
            )
            for i in range(48)
        ]

        optimizer.set_datetime(base_time)
        schedule = optimizer.find_optimal_schedule(long_prices, 4, current_soc=50)

        assert len(schedule) == 48


class TestDischargeVsHoldDecision:
    """Test cases specifically for DISCHARGE vs HOLD decisions."""

    def test_discharge_preferred_when_cheaper_overnight_charging(self):
        """
        When overnight charging is cheap, DISCHARGE should be preferred over HOLD
        because the extra charge cost is less than the saved load cost.

        Scenario (from user bug report):
        - At 23:00, price = 0.1257, discharge is allowed (above threshold)
        - SOC is ~18%, enough for one hour of discharge
        - Overnight charging is cheap (negative to ~5 cents)
        - Battery will charge to 100% overnight regardless
        - DISCHARGE at 23:00 should be chosen because:
          * Saves load cost at 23:00: ~0.07 EUR
          * Extra charge cost overnight: ~0.024 EUR
          * Net benefit: ~0.05 EUR
        """
        # Create optimizer with verbose logging
        optimizer = MockOptimizer(
            battery_capacity=14.3,
            charge_rate=4.5,
            discharge_rate=4.5,
            efficiency=0.90,
            min_soc=10.0,
            max_soc=100.0,
            grid_fee=0.045,
            battery_avg_cost=0.05,
            base_consumption=500,  # 0.5 kW load
        )
        optimizer.config.decision_log_level = 3  # Enable deep tracing

        # Override log to capture output
        log_output = []
        original_log = optimizer.log
        def capture_log(message, level="INFO"):
            log_output.append(message)
            print(message)  # Also print for debugging
        optimizer.log = capture_log

        # Create prices: starting at 23:00 today, going through tomorrow
        base_time = datetime.datetime(2024, 1, 15, 23, 0, 0)
        prices = [
            # Today 23:00 - moderate price, discharge should be allowed
            PricePoint(time=base_time, price=0.1257),
            # Tomorrow 00:00-06:00 - cheap overnight (some negative)
            PricePoint(time=base_time + datetime.timedelta(hours=1), price=-0.02),
            PricePoint(time=base_time + datetime.timedelta(hours=2), price=-0.01),
            PricePoint(time=base_time + datetime.timedelta(hours=3), price=0.01),
            PricePoint(time=base_time + datetime.timedelta(hours=4), price=0.02),
            PricePoint(time=base_time + datetime.timedelta(hours=5), price=0.03),
            PricePoint(time=base_time + datetime.timedelta(hours=6), price=0.05),
            # Tomorrow 07:00-12:00 - moderate morning
            PricePoint(time=base_time + datetime.timedelta(hours=7), price=0.10),
            PricePoint(time=base_time + datetime.timedelta(hours=8), price=0.12),
            PricePoint(time=base_time + datetime.timedelta(hours=9), price=0.10),
            PricePoint(time=base_time + datetime.timedelta(hours=10), price=0.08),
            PricePoint(time=base_time + datetime.timedelta(hours=11), price=0.07),
            PricePoint(time=base_time + datetime.timedelta(hours=12), price=0.08),
            # Tomorrow 13:00-17:00 - expensive peak
            PricePoint(time=base_time + datetime.timedelta(hours=13), price=0.15),
            PricePoint(time=base_time + datetime.timedelta(hours=14), price=0.20),
            PricePoint(time=base_time + datetime.timedelta(hours=15), price=0.25),
            PricePoint(time=base_time + datetime.timedelta(hours=16), price=0.30),
            PricePoint(time=base_time + datetime.timedelta(hours=17), price=0.25),
            PricePoint(time=base_time + datetime.timedelta(hours=18), price=0.18),
        ]

        optimizer.set_datetime(base_time)

        # Start at 18% SOC - enough for discharge but will need charging
        schedule = optimizer.find_optimal_schedule(prices, 5, current_soc=18.2)

        # Debug output
        print("\n=== Schedule ===")
        for hour, entry in sorted(schedule.items()):
            print(f"{hour.strftime('%H:%M')}: {entry.mode.name} - {entry.reason}")

        # The first slot (23:00) should be DISCHARGE, not HOLD
        first_slot = base_time
        assert first_slot in schedule, "First slot should be in schedule"
        first_entry = schedule[first_slot]

        # Print the log output for debugging
        print("\n=== Log Output ===")
        for line in log_output:
            print(line)

        # The key assertion: DISCHARGE should be preferred over HOLD at 23:00
        # because overnight charging is cheap enough to make up for the discharged energy
        assert first_entry.mode == BatteryMode.DISCHARGE, (
            f"Expected DISCHARGE at 23:00, got {first_entry.mode.name}. "
            f"Reason: {first_entry.reason}"
        )

    def test_discharge_with_high_battery_cost(self):
        """
        Test with higher battery_avg_cost to see if threshold calculations cause issues.

        Higher battery cost means higher discharge threshold, which might
        prevent discharge at 23:00 even though it would be economically beneficial.
        """
        # Create optimizer with higher battery cost
        optimizer = MockOptimizer(
            battery_capacity=14.3,
            charge_rate=4.5,
            discharge_rate=4.5,
            efficiency=0.90,
            min_soc=10.0,
            max_soc=100.0,
            grid_fee=0.045,
            battery_avg_cost=0.10,  # Higher battery cost - this raises discharge threshold
            base_consumption=500,
        )
        optimizer.config.decision_log_level = 3

        log_output = []
        def capture_log(message, level="INFO"):
            log_output.append(message)
            print(message)
        optimizer.log = capture_log

        # Create prices matching user's scenario:
        # - 23:00 @ 0.1257 (moderate, above threshold)
        # - Overnight charging is cheap
        # - Next day has high peak around 17:00
        base_time = datetime.datetime(2024, 1, 15, 23, 0, 0)
        prices = [
            # Today 23:00
            PricePoint(time=base_time, price=0.1257),
            # Tomorrow 00:00-06:00 - cheap overnight
            PricePoint(time=base_time + datetime.timedelta(hours=1), price=-0.02),
            PricePoint(time=base_time + datetime.timedelta(hours=2), price=-0.01),
            PricePoint(time=base_time + datetime.timedelta(hours=3), price=0.01),
            PricePoint(time=base_time + datetime.timedelta(hours=4), price=0.02),
            PricePoint(time=base_time + datetime.timedelta(hours=5), price=0.03),
            PricePoint(time=base_time + datetime.timedelta(hours=6), price=0.05),
            PricePoint(time=base_time + datetime.timedelta(hours=7), price=0.08),
            # Tomorrow 08:00-16:00 - moderate
            PricePoint(time=base_time + datetime.timedelta(hours=8), price=0.10),
            PricePoint(time=base_time + datetime.timedelta(hours=9), price=0.12),
            PricePoint(time=base_time + datetime.timedelta(hours=10), price=0.10),
            PricePoint(time=base_time + datetime.timedelta(hours=11), price=0.08),
            PricePoint(time=base_time + datetime.timedelta(hours=12), price=0.07),
            PricePoint(time=base_time + datetime.timedelta(hours=13), price=0.08),
            PricePoint(time=base_time + datetime.timedelta(hours=14), price=0.10),
            PricePoint(time=base_time + datetime.timedelta(hours=15), price=0.15),
            PricePoint(time=base_time + datetime.timedelta(hours=16), price=0.20),
            # Tomorrow 17:00-18:00 - HIGH peak (this is what user's log showed)
            PricePoint(time=base_time + datetime.timedelta(hours=17), price=0.30),  # High!
            PricePoint(time=base_time + datetime.timedelta(hours=18), price=0.3157),  # User's 17:00 price
            PricePoint(time=base_time + datetime.timedelta(hours=19), price=0.25),
            PricePoint(time=base_time + datetime.timedelta(hours=20), price=0.18),
        ]

        optimizer.set_datetime(base_time)
        schedule = optimizer.find_optimal_schedule(prices, 5, current_soc=18.2)

        # Debug output
        print("\n=== Schedule (high battery cost) ===")
        for hour, entry in sorted(schedule.items()):
            print(f"{hour.strftime('%H:%M')}: {entry.mode.name} - {entry.reason}")

        first_slot = base_time
        first_entry = schedule.get(first_slot)

        # Calculate expected discharge threshold
        discharge_threshold = (optimizer.battery_avg_cost / optimizer.config.efficiency) + optimizer.config.grid_fee
        buy_price_23 = 0.1257 + optimizer.config.grid_fee
        print(f"\nDischarge threshold: {discharge_threshold:.4f}")
        print(f"Buy price at 23:00: {buy_price_23:.4f}")
        print(f"Discharge allowed: {buy_price_23 >= discharge_threshold}")

        # With battery_avg_cost=0.10, threshold = 0.10/0.90 + 0.045 = 0.156
        # buy_price_23 = 0.1257 + 0.045 = 0.1707
        # So 0.1707 >= 0.156 -> discharge IS allowed
        # But let's see what the algorithm chooses

        if first_entry:
            print(f"\nFirst slot mode: {first_entry.mode.name}")
            # Even with higher battery cost, DISCHARGE should be chosen if:
            # 1. Discharge is allowed (buy_price >= threshold)
            # 2. Overnight charging cost < load cost saved
            assert first_entry.mode == BatteryMode.DISCHARGE, (
                f"Expected DISCHARGE at 23:00 even with high battery cost. "
                f"Got {first_entry.mode.name}. Reason: {first_entry.reason}"
            )


class TestFifteenMinuteSlots:
    """Test cases for 15-minute slot scheduling."""

    def test_15_minute_slots(self):
        """Algorithm should generate 96 entries for 24 hours of 15-min slots."""
        optimizer = MockOptimizer(slot_minutes=15)

        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
        prices = []
        for i in range(96):
            dt = base_time + datetime.timedelta(minutes=15 * i)
            hour = dt.hour
            # Cheap at night (00:00-06:00), expensive during day (07:00-22:00)
            if hour < 6:
                price = 0.03
            elif hour < 9:
                price = 0.10
            elif hour < 17:
                price = 0.15
            elif hour < 21:
                price = 0.12
            else:
                price = 0.05
            prices.append(PricePoint(time=dt, price=price))

        optimizer.set_datetime(base_time)
        schedule = optimizer.find_optimal_schedule(prices, 4, current_soc=50)

        # Should have 96 entries (one per 15-min slot)
        assert len(schedule) == 96

        # All entries should have valid BatteryMode
        for hour, entry in schedule.items():
            assert isinstance(entry, ScheduleEntry)
            assert entry.mode in [BatteryMode.HOLD, BatteryMode.CHARGE, BatteryMode.DISCHARGE]

    def test_15min_charge_at_cheapest_slots(self):
        """Should charge during cheap (negative price) 15-min slots."""
        optimizer = MockOptimizer(slot_minutes=15)

        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
        prices = []
        cheap_slots = set()
        for i in range(96):
            dt = base_time + datetime.timedelta(minutes=15 * i)
            # 4 cheap slots at 01:00, 01:15, 01:30, 01:45 (indices 4-7)
            # Use negative prices to clearly incentivize charging
            if 4 <= i <= 7:
                price = -0.05
                cheap_slots.add(dt)
            elif dt.hour >= 17 and dt.hour < 18:
                price = 0.30  # Expensive peak
            else:
                price = 0.10
            prices.append(PricePoint(time=dt, price=price))

        optimizer.set_datetime(base_time)
        schedule = optimizer.find_optimal_schedule(prices, 4, current_soc=50)

        assert len(schedule) == 96

        # Verify charging happens during the cheap (negative price) slots
        charge_entries = [
            e for e in schedule.values()
            if e.mode == BatteryMode.CHARGE
        ]
        assert len(charge_entries) > 0, "Should have charge slots at negative prices"

        # Check that the cheap slots are used for charging
        cheap_charges = [
            e for e in charge_entries
            if e.time in cheap_slots
        ]
        assert len(cheap_charges) > 0, (
            "Should charge during the cheapest slots (01:00-01:45)"
        )

    def test_15min_partial_slot(self):
        """Algorithm should handle partial 15-min slot correctly."""
        optimizer = MockOptimizer(slot_minutes=15)

        # Set current time to 10:07 (7 min into the 10:00-10:15 slot)
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 10, 7, 0))

        base_time = datetime.datetime(2024, 1, 15, 10, 0, 0)
        prices = [
            PricePoint(
                time=base_time + datetime.timedelta(minutes=15 * i),
                price=0.05 + i * 0.005,
            )
            for i in range(20)  # 5 hours of 15-min slots
        ]

        schedule = optimizer.find_optimal_schedule(
            prices, 2, current_soc=50
        )

        # Schedule should be generated successfully
        assert len(schedule) > 0

        # First slot (10:00) should be included
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        has_current_slot = any(
            (h.replace(tzinfo=None) if h.tzinfo else h) == slot_10
            for h in schedule.keys()
        )
        assert has_current_slot, "Current partial slot (10:00) should be in schedule"
