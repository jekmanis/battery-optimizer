"""
Tests for calculate_min_charge_slots_for_horizon().

This function calculates minimum charge slots needed to avoid hitting min_soc
during the planning horizon based on predicted load and available battery energy.
"""

import datetime
from typing import List

import pytest

from battery_optimizer import BatteryOptimizer, PricePoint
from battery_optimizer_lib import BatteryOptimizerConfig


class MockMinChargeOptimizer:
    """
    Minimal mock for testing calculate_min_charge_slots_for_horizon.

    Only implements the methods and attributes needed by the function.
    """

    def __init__(
        self,
        battery_capacity: float = 14.3,
        charge_rate: float = 4.5,
        discharge_rate: float = 4.5,
        efficiency: float = 0.85,
        min_soc: float = 10.0,
        slot_minutes: int = 60,
        predicted_load_kw: float = 0.5,  # Default constant load
    ):
        # Create config object
        self.config = BatteryOptimizerConfig(
            battery_capacity=battery_capacity,
            charge_rate=charge_rate,
            discharge_rate=discharge_rate,
            efficiency=efficiency,
            default_min_soc=min_soc,
            slot_minutes=slot_minutes,
            decision_log_level=0,
        )

        # Dynamic properties
        self.min_soc = min_soc

        # Configurable load prediction
        self._predicted_load_kw = predicted_load_kw
        self._load_by_hour = {}  # For varying load tests

    def log(self, message: str, level: str = "INFO"):
        """Silent logging for tests."""
        pass

    def _predict_load_kw(self, dt: datetime.datetime) -> float:
        """Return configurable load prediction."""
        # Check for hour-specific load first
        if dt in self._load_by_hour:
            return self._load_by_hour[dt]
        return self._predicted_load_kw

    def set_load_for_hour(self, dt: datetime.datetime, load_kw: float):
        """Set specific load for a given hour."""
        self._load_by_hour[dt] = load_kw


# Bind the actual method to our mock
MockMinChargeOptimizer.calculate_min_charge_slots_for_horizon = (
    BatteryOptimizer.calculate_min_charge_slots_for_horizon
)


def make_prices(start_time: datetime.datetime, count: int) -> List[PricePoint]:
    """Create a list of price points starting from given time."""
    return [
        PricePoint(time=start_time + datetime.timedelta(hours=i), price=0.10)
        for i in range(count)
    ]


class TestMinChargeSlots:
    """Test cases for calculate_min_charge_slots_for_horizon."""

    @pytest.fixture
    def optimizer(self):
        """Create a mock optimizer with default settings."""
        return MockMinChargeOptimizer()

    @pytest.fixture
    def base_time(self):
        """Standard base time for tests."""
        return datetime.datetime(2024, 1, 15, 0, 0, 0)

    def test_empty_prices_returns_zero(self, optimizer):
        """Empty price list should return 0 slots."""
        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=50.0, prices=[]
        )
        assert result == 0

    def test_sufficient_soc_no_charge_needed(self, optimizer, base_time):
        """With high SOC and low load, no charging needed."""
        # 80% SOC with 14.3 kWh battery
        # Usable energy = (80 - 10) / 100 * 14.3 = 10.01 kWh
        # With 0.5 kW load over 4 hours = 2.0 kWh total
        # 10.01 > 2.0, so no charging needed
        optimizer._predicted_load_kw = 0.5
        prices = make_prices(base_time, 4)

        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=80.0, prices=prices
        )
        assert result == 0

    def test_low_soc_high_load_needs_charge(self, optimizer, base_time):
        """With low SOC and high load, charging is required."""
        # 30% SOC: usable = (30 - 10) / 100 * 14.3 = 2.86 kWh
        # With 2.0 kW load over 10 hours = 20.0 kWh (capped by discharge rate)
        # But discharge_rate is 4.5 kW, so load is uncapped at 2.0 kW
        # 2.0 * 10 = 20.0 kWh needed
        # Deficit = 20.0 - 2.86 = 17.14 kWh
        # Energy per slot = 4.5 * 0.85 * 1 = 3.825 kWh
        # Slots = ceil(17.14 / 3.825) = 5
        optimizer._predicted_load_kw = 2.0
        prices = make_prices(base_time, 10)

        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=30.0, prices=prices
        )
        assert result > 0
        assert result == 5

    def test_at_min_soc_always_needs_charge(self, optimizer, base_time):
        """At min_soc with any load, charging is always needed."""
        # At min_soc (10%), usable energy = 0
        # Any load creates a deficit
        optimizer._predicted_load_kw = 0.5
        prices = make_prices(base_time, 6)  # 6 hours * 0.5 kW = 3.0 kWh needed

        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=10.0,  # at min_soc
            prices=prices,
        )
        # Deficit = 3.0 - 0 = 3.0 kWh
        # Energy per slot = 4.5 * 0.85 = 3.825 kWh
        # Slots = ceil(3.0 / 3.825) = 1
        assert result >= 1

    def test_charge_slots_proportional_to_deficit(self, optimizer, base_time):
        """More deficit should require more charge slots."""
        optimizer._predicted_load_kw = 1.0  # 1 kW constant load

        # Short horizon: 4 hours = 4 kWh load
        prices_short = make_prices(base_time, 4)
        result_short = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=20.0, prices=prices_short
        )

        # Longer horizon: 12 hours = 12 kWh load
        prices_long = make_prices(base_time, 12)
        result_long = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=20.0, prices=prices_long
        )

        # Longer horizon should need more slots
        assert result_long >= result_short

    def test_efficiency_affects_slots_needed(self, base_time):
        """Lower efficiency requires more charge slots for same deficit."""
        # High efficiency optimizer
        opt_high_eff = MockMinChargeOptimizer(efficiency=0.95)
        opt_high_eff._predicted_load_kw = 1.5

        # Low efficiency optimizer
        opt_low_eff = MockMinChargeOptimizer(efficiency=0.70)
        opt_low_eff._predicted_load_kw = 1.5

        prices = make_prices(base_time, 12)

        result_high = opt_high_eff.calculate_min_charge_slots_for_horizon(
            current_soc=30.0, prices=prices
        )
        result_low = opt_low_eff.calculate_min_charge_slots_for_horizon(
            current_soc=30.0, prices=prices
        )

        # Lower efficiency should need equal or more slots
        # (deficit/slot is lower with lower efficiency)
        # energy_per_slot = charge_rate * efficiency * slot_hours
        # Higher efficiency = more energy per slot = fewer slots needed
        assert result_low >= result_high

    def test_long_horizon_more_slots(self, optimizer, base_time):
        """Longer horizons generally need more charge slots."""
        optimizer._predicted_load_kw = 1.0

        # 12 hour horizon
        prices_12h = make_prices(base_time, 12)
        result_12h = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=25.0, prices=prices_12h
        )

        # 24 hour horizon
        prices_24h = make_prices(base_time, 24)
        result_24h = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=25.0, prices=prices_24h
        )

        # 48 hour horizon
        prices_48h = make_prices(base_time, 48)
        result_48h = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=25.0, prices=prices_48h
        )

        # Longer horizons should need more or equal slots
        assert result_24h >= result_12h
        assert result_48h >= result_24h

    def test_zero_load_no_charge_needed(self, optimizer, base_time):
        """With zero load prediction, no charging is needed."""
        optimizer._predicted_load_kw = 0.0
        prices = make_prices(base_time, 24)

        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=50.0, prices=prices
        )
        # No load = no deficit
        assert result == 0

    def test_exact_boundary_energy(self, optimizer, base_time):
        """When usable energy exactly equals predicted load, no charging needed."""
        # Set up so usable energy exactly matches load
        # Usable at 50% SOC = (50 - 10) / 100 * 14.3 = 5.72 kWh
        # Set load so total = 5.72 kWh over 8 hours
        # 5.72 / 8 = 0.715 kW
        optimizer._predicted_load_kw = 0.715
        prices = make_prices(base_time, 8)

        # At exactly boundary, deficit = 0
        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=50.0, prices=prices
        )
        assert result == 0

    def test_small_deficit_rounds_up(self, optimizer, base_time):
        """Small deficit should still round up to at least 1 slot."""
        # Usable at 50% = (50 - 10) / 100 * 14.3 = 5.72 kWh
        # Load = 0.75 kW * 8 hours = 6.0 kWh
        # Deficit = 6.0 - 5.72 = 0.28 kWh
        # Energy per slot = 4.5 * 0.85 = 3.825 kWh
        # Slots = ceil(0.28 / 3.825) = 1
        optimizer._predicted_load_kw = 0.75
        prices = make_prices(base_time, 8)

        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=50.0, prices=prices
        )
        # Even tiny deficit requires 1 slot
        assert result == 1

    def test_varying_load_by_hour(self, optimizer, base_time):
        """Function should sum varying load predictions correctly."""
        # Set different loads for different hours
        prices = make_prices(base_time, 6)

        # Varying loads: 0.2, 0.5, 1.0, 2.0, 1.0, 0.3 kW = 5.0 kWh total
        loads = [0.2, 0.5, 1.0, 2.0, 1.0, 0.3]
        for i, load in enumerate(loads):
            dt = base_time + datetime.timedelta(hours=i)
            optimizer.set_load_for_hour(dt, load)

        # Usable at 30% = (30 - 10) / 100 * 14.3 = 2.86 kWh
        # Deficit = 5.0 - 2.86 = 2.14 kWh
        # Energy per slot = 4.5 * 0.85 = 3.825 kWh
        # Slots = ceil(2.14 / 3.825) = 1
        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=30.0, prices=prices
        )
        assert result == 1

    def test_load_capped_by_discharge_rate(self, optimizer, base_time):
        """Load prediction should be capped by discharge_rate."""
        # discharge_rate is 4.5 kW
        # Setting load above this should be capped
        optimizer._predicted_load_kw = 10.0  # Well above discharge rate
        optimizer.config.discharge_rate = 4.5
        prices = make_prices(base_time, 4)

        # Load is capped: min(10.0, 4.5) * 4 hours = 18.0 kWh
        # Usable at 50% = (50 - 10) / 100 * 14.3 = 5.72 kWh
        # Deficit = 18.0 - 5.72 = 12.28 kWh
        # Energy per slot = 4.5 * 0.85 = 3.825 kWh
        # Slots = ceil(12.28 / 3.825) = 4
        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=50.0, prices=prices
        )
        assert result == 4

    def test_different_slot_sizes(self, base_time):
        """Function should work with different slot sizes (e.g., 30 min)."""
        optimizer = MockMinChargeOptimizer(slot_minutes=30)
        optimizer._predicted_load_kw = 1.0

        # 12 half-hour slots = 6 hours
        prices = make_prices(base_time, 12)

        # Load = 1.0 kW * 0.5 hours * 12 slots = 6.0 kWh
        # Usable at 30% = (30 - 10) / 100 * 14.3 = 2.86 kWh
        # Deficit = 6.0 - 2.86 = 3.14 kWh
        # Energy per slot = 4.5 * 0.85 * 0.5 = 1.9125 kWh
        # Slots = ceil(3.14 / 1.9125) = 2
        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=30.0, prices=prices
        )
        assert result == 2

    def test_very_small_battery(self, base_time):
        """Function should work with small battery capacity."""
        optimizer = MockMinChargeOptimizer(battery_capacity=5.0)
        optimizer._predicted_load_kw = 0.5

        prices = make_prices(base_time, 8)

        # Usable at 50% = (50 - 10) / 100 * 5.0 = 2.0 kWh
        # Load = 0.5 * 8 = 4.0 kWh
        # Deficit = 4.0 - 2.0 = 2.0 kWh
        # Energy per slot = 4.5 * 0.85 = 3.825 kWh
        # Slots = ceil(2.0 / 3.825) = 1
        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=50.0, prices=prices
        )
        assert result == 1

    def test_very_large_battery(self, base_time):
        """Large battery with low load should need no charging."""
        optimizer = MockMinChargeOptimizer(battery_capacity=50.0)
        optimizer._predicted_load_kw = 0.5

        prices = make_prices(base_time, 24)

        # Usable at 50% = (50 - 10) / 100 * 50 = 20 kWh
        # Load = 0.5 * 24 = 12 kWh
        # 20 > 12, no deficit
        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=50.0, prices=prices
        )
        assert result == 0

    def test_high_min_soc_reduces_usable_energy(self, base_time):
        """Higher min_soc means less usable energy and more charging needed."""
        opt_low_min = MockMinChargeOptimizer(min_soc=10.0)
        opt_low_min._predicted_load_kw = 1.0

        opt_high_min = MockMinChargeOptimizer(min_soc=30.0)
        opt_high_min._predicted_load_kw = 1.0

        prices = make_prices(base_time, 12)

        # At 50% SOC:
        # Low min: usable = (50 - 10) / 100 * 14.3 = 5.72 kWh
        # High min: usable = (50 - 30) / 100 * 14.3 = 2.86 kWh
        result_low = opt_low_min.calculate_min_charge_slots_for_horizon(
            current_soc=50.0, prices=prices
        )
        result_high = opt_high_min.calculate_min_charge_slots_for_horizon(
            current_soc=50.0, prices=prices
        )

        # Higher min_soc should need more charging
        assert result_high >= result_low

    def test_below_min_soc_maximum_charging(self, optimizer, base_time):
        """When current SOC is below min_soc, usable energy is 0."""
        # This is a degenerate case but should be handled
        prices = make_prices(base_time, 4)
        optimizer._predicted_load_kw = 0.5

        # SOC below min_soc (which shouldn't happen in practice)
        # usable = max(0, (5 - 10) / 100 * 14.3) = 0 kWh
        # But the code uses: (current_soc - self.min_soc) which could be negative
        # Load = 0.5 * 4 = 2.0 kWh
        result = optimizer.calculate_min_charge_slots_for_horizon(
            current_soc=5.0,  # Below min_soc of 10%
            prices=prices,
        )
        # Should still calculate slots needed
        assert result >= 1
