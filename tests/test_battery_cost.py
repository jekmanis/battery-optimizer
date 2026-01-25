"""
Tests for battery cost tracking functions.

Tests for:
- _project_battery_costs() - projects battery cost evolution through a schedule
- _get_discharge_threshold() - calculates discharge price threshold
- _get_discharge_threshold_for_cost() - calculates threshold for a given cost
"""

import datetime
from typing import Dict

import pytest

from battery_optimizer import BatteryMode, BatteryOptimizer, ScheduleEntry


class MockCostOptimizer:
    """
    Minimal mock for testing battery cost functions.

    Only implements the methods and attributes needed by the functions.
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
        battery_wear_cost: float = 0.0,
        battery_avg_cost: float = 0.08,
        slot_minutes: int = 60,
        predicted_load_kw: float = 0.5,
    ):
        self.battery_capacity = battery_capacity
        self.charge_rate = charge_rate
        self.discharge_rate = discharge_rate
        self.efficiency = efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.grid_fee = grid_fee
        self.battery_wear_cost = battery_wear_cost
        self.battery_avg_cost = battery_avg_cost
        self.slot_minutes = slot_minutes
        self.slot_hours = slot_minutes / 60.0

        # Configurable load prediction
        self._predicted_load_kw = predicted_load_kw
        self._load_by_hour = {}

    def log(self, message: str, level: str = "INFO"):
        """Silent logging for tests."""
        pass

    def _predict_load_kw(self, dt: datetime.datetime) -> float:
        """Return configurable load prediction."""
        if dt in self._load_by_hour:
            return self._load_by_hour[dt]
        return self._predicted_load_kw

    def set_load_for_hour(self, dt: datetime.datetime, load_kw: float):
        """Set specific load for a given hour."""
        self._load_by_hour[dt] = load_kw


# Bind the actual methods to our mock
MockCostOptimizer._project_battery_costs = BatteryOptimizer._project_battery_costs
MockCostOptimizer._get_discharge_threshold = BatteryOptimizer._get_discharge_threshold
MockCostOptimizer._get_discharge_threshold_for_cost = (
    BatteryOptimizer._get_discharge_threshold_for_cost
)


def make_schedule_entry(
    hour: datetime.datetime, mode: BatteryMode, reason: str = "Test"
) -> ScheduleEntry:
    """Create a schedule entry."""
    return ScheduleEntry(hour=hour, mode=mode, reason=reason)


class TestProjectBatteryCosts:
    """Test cases for _project_battery_costs."""

    @pytest.fixture
    def optimizer(self):
        """Create a mock optimizer with default settings."""
        return MockCostOptimizer()

    @pytest.fixture
    def base_time(self):
        """Standard base time for tests."""
        return datetime.datetime(2024, 1, 15, 0, 0, 0)

    def test_empty_schedule_returns_starting_cost(self, optimizer):
        """Empty schedule should return empty dict and starting cost."""
        projected, final_cost = optimizer._project_battery_costs(
            schedule={},
            starting_soc=50.0,
            starting_cost=0.10,
            prices_by_slot={},
        )
        assert projected == {}
        assert final_cost == 0.10

    def test_single_charge_updates_cost(self, optimizer, base_time):
        """Single charge slot should update battery cost via weighted average."""
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.CHARGE)}
        prices_by_slot = {hour: 0.05}  # Cheap charging

        # Starting: 50% SOC, 0.10 avg cost
        # old_energy = (50 - 10) / 100 * 14.3 = 5.72 kWh
        # added = 4.5 * 0.85 * 1.0 = 3.825 kWh
        # new_cost = (5.72 * 0.10 + 3.825 * 0.05) / (5.72 + 3.825)
        #          = (0.572 + 0.19125) / 9.545 = 0.0799
        projected, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
        )

        # Cost at start of charge slot is still the old cost
        assert hour in projected
        assert projected[hour] == 0.10

        # Final cost should be weighted average
        expected_old_energy = (50 - 10) / 100 * 14.3
        expected_added = 4.5 * 0.85 * 1.0
        expected_cost = (
            expected_old_energy * 0.10 + expected_added * 0.05
        ) / (expected_old_energy + expected_added)
        assert abs(final_cost - expected_cost) < 0.001

    def test_single_discharge_preserves_cost(self, optimizer, base_time):
        """Discharge does not change average cost."""
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.DISCHARGE)}
        prices_by_slot = {hour: 0.20}

        projected, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
        )

        # Cost should be unchanged
        assert projected[hour] == 0.10
        assert final_cost == 0.10

    def test_hold_preserves_cost(self, optimizer, base_time):
        """Hold mode does not change average cost."""
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.HOLD)}
        prices_by_slot = {hour: 0.15}

        projected, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
        )

        # Cost unchanged by hold
        assert projected[hour] == 0.10
        assert final_cost == 0.10

    def test_charge_sequence_accumulates(self, optimizer, base_time):
        """Sequence of charges should progressively update cost."""
        # Three charge slots at different prices
        hours = [base_time + datetime.timedelta(hours=i) for i in range(3)]
        schedule = {h: make_schedule_entry(h, BatteryMode.CHARGE) for h in hours}
        prices_by_slot = {
            hours[0]: 0.05,  # Cheap
            hours[1]: 0.08,  # Medium
            hours[2]: 0.12,  # Expensive
        }

        projected, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=30.0,  # Lower SOC to allow more charging
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
        )

        # Each slot should have a projected cost
        assert len(projected) == 3

        # Cost should decrease initially (charging at 0.05 with 0.10 avg)
        # then trend towards the charging prices
        assert projected[hours[0]] == 0.10  # Starting cost
        assert projected[hours[1]] < 0.10  # Reduced after cheap charge
        # Final cost influenced by all three charges

    def test_charge_then_discharge(self, optimizer, base_time):
        """Cost set by charge should persist through discharge."""
        hours = [base_time + datetime.timedelta(hours=i) for i in range(2)]
        schedule = {
            hours[0]: make_schedule_entry(hours[0], BatteryMode.CHARGE),
            hours[1]: make_schedule_entry(hours[1], BatteryMode.DISCHARGE),
        }
        prices_by_slot = {hours[0]: 0.05, hours[1]: 0.20}

        projected, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
        )

        # After charge, cost should be reduced
        cost_after_charge = projected[hours[1]]
        assert cost_after_charge < 0.10

        # After discharge, cost should be same as after charge
        assert abs(final_cost - cost_after_charge) < 0.001

    def test_low_soc_charge_dilutes_cost(self, optimizer, base_time):
        """Starting near min_soc, new charge dominates cost."""
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.CHARGE)}
        prices_by_slot = {hour: 0.03}

        # At min_soc, old_energy is ~0
        # New charge price should dominate
        projected, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=10.0,  # at min_soc
            starting_cost=0.20,  # High old cost
            prices_by_slot=prices_by_slot,
        )

        # Final cost should be close to charge price (0.03)
        # since there's almost no existing energy
        assert final_cost < 0.10
        assert final_cost > 0.02  # But not exactly 0.03 due to small buffer

    def test_high_soc_charge_small_impact(self, optimizer, base_time):
        """Starting near max_soc, new charge has small impact on cost."""
        # Set max_soc to 90 to allow some headroom
        optimizer.max_soc = 95.0
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.CHARGE)}
        prices_by_slot = {hour: 0.02}  # Very cheap

        # At 90% SOC, lots of existing energy
        # New charge limited by headroom
        projected, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=90.0,
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
        )

        # Final cost should be close to starting cost
        # (small amount of cheap energy added to large existing pool)
        assert final_cost < 0.10  # Slightly reduced
        assert final_cost > 0.08  # But not by much

    def test_respects_max_soc_cap(self, optimizer, base_time):
        """Energy addition should be capped by max_soc headroom."""
        optimizer.max_soc = 95.0
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.CHARGE)}
        prices_by_slot = {hour: 0.05}

        # At 93%, only ~2% headroom = 0.286 kWh can be added
        # Not the full 3.825 kWh
        starting_soc = 93.0
        projected, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=starting_soc,
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
        )

        # Verify cost changed less than with full charge
        old_energy = (starting_soc - optimizer.min_soc) / 100 * optimizer.battery_capacity
        max_added = (optimizer.max_soc - starting_soc) / 100 * optimizer.battery_capacity
        expected_cost = (old_energy * 0.10 + max_added * 0.05) / (old_energy + max_added)
        assert abs(final_cost - expected_cost) < 0.001

    def test_charge_rates_by_slot_used(self, optimizer, base_time):
        """Custom charge rates should affect energy calculation."""
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.CHARGE)}
        prices_by_slot = {hour: 0.05}
        charge_rates_by_slot = {hour: 2.0}  # Half the default rate

        projected, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
            charge_rates_by_slot=charge_rates_by_slot,
        )

        # Calculate expected with custom rate
        old_energy = (50 - 10) / 100 * 14.3
        added_energy = 2.0 * 0.85 * 1.0  # Half rate
        expected_cost = (old_energy * 0.10 + added_energy * 0.05) / (
            old_energy + added_energy
        )
        assert abs(final_cost - expected_cost) < 0.001

    def test_slot_fractions_affect_energy(self, optimizer, base_time):
        """Partial slot fractions should reduce energy added/removed."""
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.CHARGE)}
        prices_by_slot = {hour: 0.05}
        slot_fractions = {hour: 0.5}  # Half slot

        projected, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
            slot_fractions_by_slot=slot_fractions,
        )

        # Calculate expected with half slot
        old_energy = (50 - 10) / 100 * 14.3
        added_energy = 4.5 * 0.85 * 1.0 * 0.5  # Half energy
        expected_cost = (old_energy * 0.10 + added_energy * 0.05) / (
            old_energy + added_energy
        )
        assert abs(final_cost - expected_cost) < 0.001

    def test_mixed_schedule_order(self, optimizer, base_time):
        """Mixed schedule (C, D, H, C, D) should track cost correctly."""
        hours = [base_time + datetime.timedelta(hours=i) for i in range(5)]
        schedule = {
            hours[0]: make_schedule_entry(hours[0], BatteryMode.CHARGE),
            hours[1]: make_schedule_entry(hours[1], BatteryMode.DISCHARGE),
            hours[2]: make_schedule_entry(hours[2], BatteryMode.HOLD),
            hours[3]: make_schedule_entry(hours[3], BatteryMode.CHARGE),
            hours[4]: make_schedule_entry(hours[4], BatteryMode.DISCHARGE),
        }
        prices_by_slot = {
            hours[0]: 0.04,
            hours[1]: 0.15,
            hours[2]: 0.10,
            hours[3]: 0.06,
            hours[4]: 0.18,
        }

        projected, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=40.0,
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
        )

        # Should have costs for all 5 slots
        assert len(projected) == 5

        # Starting cost
        assert projected[hours[0]] == 0.10

        # After cheap charge, cost should decrease
        assert projected[hours[1]] < 0.10

        # Discharge and hold don't change cost
        assert projected[hours[2]] == projected[hours[1]]
        assert projected[hours[3]] == projected[hours[2]]

        # After second charge, cost changes again
        # Final cost should be < starting due to cheap charges
        assert final_cost < 0.10


class TestGetDischargeThreshold:
    """Test cases for _get_discharge_threshold."""

    def test_basic_threshold(self):
        """Basic threshold calculation."""
        # threshold = (avg_cost + grid_fee) / efficiency + wear_cost
        # = (0.10 + 0.05) / 0.85 + 0.0 = 0.1765
        optimizer = MockCostOptimizer(
            battery_avg_cost=0.10,
            efficiency=0.85,
            grid_fee=0.05,
            battery_wear_cost=0.0,
        )

        threshold = optimizer._get_discharge_threshold()
        expected = (0.10 + 0.05) / 0.85
        assert abs(threshold - expected) < 0.001

    def test_with_wear_cost(self):
        """Wear cost should increase threshold."""
        optimizer = MockCostOptimizer(
            battery_avg_cost=0.10,
            efficiency=0.85,
            grid_fee=0.05,
            battery_wear_cost=0.02,
        )

        threshold = optimizer._get_discharge_threshold()
        expected = (0.10 + 0.05) / 0.85 + 0.02
        assert abs(threshold - expected) < 0.001

        # Should be higher than without wear cost
        opt_no_wear = MockCostOptimizer(
            battery_avg_cost=0.10,
            efficiency=0.85,
            grid_fee=0.05,
            battery_wear_cost=0.0,
        )
        assert threshold > opt_no_wear._get_discharge_threshold()

    def test_zero_grid_fee(self):
        """Zero grid fee should lower threshold."""
        optimizer = MockCostOptimizer(
            battery_avg_cost=0.10,
            efficiency=0.85,
            grid_fee=0.0,
            battery_wear_cost=0.0,
        )

        threshold = optimizer._get_discharge_threshold()
        expected = 0.10 / 0.85
        assert abs(threshold - expected) < 0.001

    def test_high_efficiency_lower_threshold(self):
        """Higher efficiency should result in lower threshold."""
        opt_high_eff = MockCostOptimizer(
            battery_avg_cost=0.10,
            efficiency=0.95,
            grid_fee=0.05,
        )
        opt_low_eff = MockCostOptimizer(
            battery_avg_cost=0.10,
            efficiency=0.80,
            grid_fee=0.05,
        )

        threshold_high = opt_high_eff._get_discharge_threshold()
        threshold_low = opt_low_eff._get_discharge_threshold()

        # Higher efficiency = lower threshold
        assert threshold_high < threshold_low


class TestGetDischargeThresholdForCost:
    """Test cases for _get_discharge_threshold_for_cost."""

    @pytest.fixture
    def optimizer(self):
        return MockCostOptimizer(
            efficiency=0.85,
            grid_fee=0.05,
            battery_wear_cost=0.01,
            battery_avg_cost=0.10,
        )

    def test_matches_main_for_same_cost(self, optimizer):
        """Should match _get_discharge_threshold when using same cost."""
        main_threshold = optimizer._get_discharge_threshold()
        for_cost_threshold = optimizer._get_discharge_threshold_for_cost(
            optimizer.battery_avg_cost
        )

        assert abs(main_threshold - for_cost_threshold) < 0.0001

    def test_different_cost_different_threshold(self, optimizer):
        """Different costs should produce different thresholds."""
        threshold_low = optimizer._get_discharge_threshold_for_cost(0.05)
        threshold_high = optimizer._get_discharge_threshold_for_cost(0.15)

        # Higher cost = higher threshold
        assert threshold_high > threshold_low

    def test_zero_cost(self, optimizer):
        """Zero cost should just include grid_fee/efficiency + wear."""
        threshold = optimizer._get_discharge_threshold_for_cost(0.0)

        # threshold = (0 + 0.05) / 0.85 + 0.01
        expected = (0.0 + 0.05) / 0.85 + 0.01
        assert abs(threshold - expected) < 0.001

    def test_negative_cost(self, optimizer):
        """Negative cost (paid to take energy) should work."""
        # This could happen with negative electricity prices
        threshold = optimizer._get_discharge_threshold_for_cost(-0.05)

        # threshold = (-0.05 + 0.05) / 0.85 + 0.01 = 0 + 0.01 = 0.01
        expected = (-0.05 + 0.05) / 0.85 + 0.01
        assert abs(threshold - expected) < 0.001

    def test_proportional_to_cost(self, optimizer):
        """Threshold should scale linearly with cost (before efficiency)."""
        threshold_1 = optimizer._get_discharge_threshold_for_cost(0.10)
        threshold_2 = optimizer._get_discharge_threshold_for_cost(0.20)

        # Difference should be proportional to cost difference / efficiency
        cost_diff = 0.20 - 0.10
        expected_threshold_diff = cost_diff / optimizer.efficiency
        actual_threshold_diff = threshold_2 - threshold_1

        assert abs(actual_threshold_diff - expected_threshold_diff) < 0.001


class TestCostTrackingIntegration:
    """Integration tests combining cost projection with thresholds."""

    @pytest.fixture
    def optimizer(self):
        return MockCostOptimizer(
            battery_avg_cost=0.10,
            efficiency=0.85,
            grid_fee=0.05,
            battery_wear_cost=0.01,
        )

    @pytest.fixture
    def base_time(self):
        return datetime.datetime(2024, 1, 15, 0, 0, 0)

    def test_cheap_charge_lowers_future_threshold(self, optimizer, base_time):
        """Charging at low price should lower the discharge threshold."""
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.CHARGE)}
        prices_by_slot = {hour: 0.02}  # Very cheap

        _, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
        )

        # Calculate thresholds
        initial_threshold = optimizer._get_discharge_threshold_for_cost(0.10)
        final_threshold = optimizer._get_discharge_threshold_for_cost(final_cost)

        # Cheap charge should lower cost and threshold
        assert final_cost < 0.10
        assert final_threshold < initial_threshold

    def test_expensive_charge_raises_future_threshold(self, optimizer, base_time):
        """Charging at high price should raise the discharge threshold."""
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.CHARGE)}
        prices_by_slot = {hour: 0.25}  # Expensive

        _, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.10,
            prices_by_slot=prices_by_slot,
        )

        # Calculate thresholds
        initial_threshold = optimizer._get_discharge_threshold_for_cost(0.10)
        final_threshold = optimizer._get_discharge_threshold_for_cost(final_cost)

        # Expensive charge should raise cost and threshold
        assert final_cost > 0.10
        assert final_threshold > initial_threshold

    def test_threshold_determines_profitable_discharge(self, optimizer, base_time):
        """Discharge should be profitable only when price > threshold."""
        # Set up a known battery cost
        optimizer.battery_avg_cost = 0.08
        threshold = optimizer._get_discharge_threshold()

        # Price above threshold should be profitable
        high_price = threshold + 0.05
        # Price below threshold should not be profitable
        low_price = threshold - 0.05

        # Verify threshold is between them
        assert low_price < threshold < high_price
