"""
Unit tests for the DPOptimizer class.
"""

import datetime
import pytest
from typing import Optional

import sys
sys.path.insert(0, "appdaemon/apps")

from battery_optimizer_lib import (
    BatteryMode,
    PricePoint,
    ScheduleEntry,
    DPOptimizer,
    DPOptimizerConfig,
    DPOptimizerResult,
)


# === Test Fixtures ===

@pytest.fixture
def default_config():
    """Default configuration for tests."""
    return DPOptimizerConfig(
        battery_capacity=14.3,
        min_soc=10.0,
        max_soc=100.0,
        efficiency=0.85,
        discharge_rate=4.5,
        slot_minutes=60,
        soc_step_percent=1.0,
        grid_fee=0.05,
        battery_wear_cost=0.02,
    )


@pytest.fixture
def simple_prices():
    """Simple price list: cheap at night, expensive during day."""
    base = datetime.datetime(2024, 1, 15, 0, 0, 0)
    return [
        PricePoint(hour=base + datetime.timedelta(hours=i), price=price)
        for i, price in enumerate([
            0.05,  # 00:00 - cheap
            0.04,  # 01:00 - cheapest
            0.05,  # 02:00 - cheap
            0.06,  # 03:00
            0.08,  # 04:00
            0.10,  # 05:00
            0.15,  # 06:00
            0.20,  # 07:00 - expensive
            0.25,  # 08:00 - most expensive
            0.22,  # 09:00
            0.18,  # 10:00
            0.15,  # 11:00
        ])
    ]


@pytest.fixture
def constant_load_predictor():
    """Load predictor that returns constant 0.5 kW."""
    def predict(dt: datetime.datetime) -> float:
        return 0.5
    return predict


@pytest.fixture
def constant_charge_rate_predictor():
    """Charge rate predictor that returns constant 4.5 kW."""
    def predict(soc: float, temp: Optional[float]) -> float:
        return 4.5
    return predict


@pytest.fixture
def identity_temp_predictor():
    """Temperature predictor that returns the same temperature."""
    def predict(temp: float, duration: float) -> float:
        return temp
    return predict


# === Config Tests ===

class TestDPOptimizerConfig:
    def test_slot_hours_calculation(self, default_config):
        assert default_config.slot_hours == 1.0

    def test_slot_hours_30min(self):
        config = DPOptimizerConfig(
            battery_capacity=14.3,
            min_soc=10.0,
            max_soc=100.0,
            efficiency=0.85,
            discharge_rate=4.5,
            slot_minutes=30,
            soc_step_percent=1.0,
            grid_fee=0.05,
            battery_wear_cost=0.02,
        )
        assert config.slot_hours == 0.5


# === DPOptimizer Tests ===

class TestDPOptimizerInit:
    def test_init(
        self,
        default_config,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        optimizer = DPOptimizer(
            config=default_config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        assert optimizer is not None


class TestDPOptimizerOptimize:
    def test_empty_prices_returns_empty_result(
        self,
        default_config,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        optimizer = DPOptimizer(
            config=default_config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        result = optimizer.optimize(
            prices=[],
            current_slot=datetime.datetime(2024, 1, 15, 0, 0, 0),
            current_soc=50.0,
        )
        assert result.schedule == {}
        assert result.charge_count == 0
        assert result.discharge_count == 0
        assert result.hold_count == 0

    def test_single_slot_hold(
        self,
        default_config,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        """Single slot with moderate price should result in HOLD at high SOC."""
        optimizer = DPOptimizer(
            config=default_config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        current_slot = datetime.datetime(2024, 1, 15, 12, 0, 0)
        prices = [PricePoint(hour=current_slot, price=0.10)]
        result = optimizer.optimize(
            prices=prices,
            current_slot=current_slot,
            current_soc=90.0,  # High SOC, no need to charge
        )
        assert len(result.schedule) == 1
        # At high SOC, low price, should HOLD
        entry = result.schedule[current_slot]
        assert entry.mode in [BatteryMode.HOLD, BatteryMode.DISCHARGE]

    def test_charges_during_cheapest_hours(
        self,
        default_config,
        simple_prices,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        """With low SOC, optimizer should charge during cheapest hours."""
        optimizer = DPOptimizer(
            config=default_config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        current_slot = simple_prices[0].hour
        result = optimizer.optimize(
            prices=simple_prices,
            current_slot=current_slot,
            current_soc=15.0,  # Low SOC, needs charging
        )
        # Should have some charge slots
        assert result.charge_count > 0

        # Check that charging happens during cheap hours (first few hours)
        charge_hours = [
            h for h, e in result.schedule.items()
            if e.mode == BatteryMode.CHARGE
        ]
        if charge_hours:
            # Cheapest hour is 01:00 (price 0.04)
            cheapest_time = current_slot + datetime.timedelta(hours=1)
            assert any(h == cheapest_time for h in charge_hours), \
                f"Expected charging at {cheapest_time}, got {charge_hours}"

    def test_discharges_during_expensive_hours(
        self,
        default_config,
        simple_prices,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        """With high SOC, optimizer should discharge during expensive hours."""
        optimizer = DPOptimizer(
            config=default_config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        current_slot = simple_prices[0].hour
        result = optimizer.optimize(
            prices=simple_prices,
            current_slot=current_slot,
            current_soc=95.0,  # High SOC, can discharge
        )
        # Should have some discharge slots
        assert result.discharge_count > 0

        # Check that discharge happens during expensive hours
        discharge_hours = [
            h for h, e in result.schedule.items()
            if e.mode == BatteryMode.DISCHARGE
        ]
        if discharge_hours:
            # Most expensive hour is 08:00 (price 0.25)
            most_expensive = current_slot + datetime.timedelta(hours=8)
            assert any(h == most_expensive for h in discharge_hours), \
                f"Expected discharge at {most_expensive}, got {discharge_hours}"

    def test_respects_min_soc(
        self,
        default_config,
        simple_prices,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        """Optimizer should not discharge below min_soc."""
        # Configure high min_soc
        config = DPOptimizerConfig(
            battery_capacity=14.3,
            min_soc=50.0,  # High min SOC
            max_soc=100.0,
            efficiency=0.85,
            discharge_rate=4.5,
            slot_minutes=60,
            soc_step_percent=1.0,
            grid_fee=0.05,
            battery_wear_cost=0.02,
        )
        optimizer = DPOptimizer(
            config=config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        current_slot = simple_prices[0].hour
        result = optimizer.optimize(
            prices=simple_prices,
            current_slot=current_slot,
            current_soc=55.0,  # Just above min_soc
        )
        # SOC trajectory should never go below min_soc
        for hour, (start_soc, end_soc) in result.soc_trajectory.items():
            assert end_soc >= config.min_soc - 0.1, \
                f"SOC dropped below min at {hour}: {end_soc} < {config.min_soc}"

    def test_respects_max_soc(
        self,
        default_config,
        simple_prices,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        """Optimizer should not charge above max_soc."""
        # Configure low max_soc
        config = DPOptimizerConfig(
            battery_capacity=14.3,
            min_soc=10.0,
            max_soc=80.0,  # Low max SOC
            efficiency=0.85,
            discharge_rate=4.5,
            slot_minutes=60,
            soc_step_percent=1.0,
            grid_fee=0.05,
            battery_wear_cost=0.02,
        )
        optimizer = DPOptimizer(
            config=config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        current_slot = simple_prices[0].hour
        result = optimizer.optimize(
            prices=simple_prices,
            current_slot=current_slot,
            current_soc=75.0,  # Near max_soc
        )
        # SOC trajectory should never exceed max_soc
        for hour, (start_soc, end_soc) in result.soc_trajectory.items():
            assert end_soc <= config.max_soc + 0.1, \
                f"SOC exceeded max at {hour}: {end_soc} > {config.max_soc}"

    def test_partial_first_slot(
        self,
        default_config,
        simple_prices,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        """Optimizer handles partial first slot correctly."""
        optimizer = DPOptimizer(
            config=default_config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        current_slot = simple_prices[0].hour
        result = optimizer.optimize(
            prices=simple_prices,
            current_slot=current_slot,
            current_soc=50.0,
            minutes_into_slot=30.0,  # Half into the slot
        )
        # Should still produce a valid schedule
        assert len(result.schedule) == len(simple_prices)


class TestDPOptimizerResult:
    def test_result_has_all_fields(
        self,
        default_config,
        simple_prices,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        optimizer = DPOptimizer(
            config=default_config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        result = optimizer.optimize(
            prices=simple_prices,
            current_slot=simple_prices[0].hour,
            current_soc=50.0,
        )
        # Check all fields exist
        assert hasattr(result, 'schedule')
        assert hasattr(result, 'soc_trajectory')
        assert hasattr(result, 'temp_trajectory')
        assert hasattr(result, 'projected_costs')
        assert hasattr(result, 'min_charge_slots')
        assert hasattr(result, 'charge_count')
        assert hasattr(result, 'discharge_count')
        assert hasattr(result, 'hold_count')
        assert hasattr(result, 'dp_best_value')

    def test_counts_match_schedule(
        self,
        default_config,
        simple_prices,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        optimizer = DPOptimizer(
            config=default_config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        result = optimizer.optimize(
            prices=simple_prices,
            current_slot=simple_prices[0].hour,
            current_soc=50.0,
        )
        # Verify counts match schedule
        actual_charge = sum(1 for e in result.schedule.values() if e.mode == BatteryMode.CHARGE)
        actual_discharge = sum(1 for e in result.schedule.values() if e.mode == BatteryMode.DISCHARGE)
        actual_hold = sum(1 for e in result.schedule.values() if e.mode == BatteryMode.HOLD)

        assert result.charge_count == actual_charge
        assert result.discharge_count == actual_discharge
        assert result.hold_count == actual_hold


class TestTemperatureAwareness:
    def test_uses_temperature_predictor(
        self,
        default_config,
        simple_prices,
        constant_load_predictor,
        constant_charge_rate_predictor,
    ):
        """Verify temperature predictors are called."""
        temp_calls = []

        def tracking_temp_predictor(temp: float, duration: float) -> float:
            temp_calls.append((temp, duration))
            return temp + 1.0  # Warming up

        optimizer = DPOptimizer(
            config=default_config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=tracking_temp_predictor,
            temp_after_idle_predictor=tracking_temp_predictor,
        )
        result = optimizer.optimize(
            prices=simple_prices,
            current_slot=simple_prices[0].hour,
            current_soc=50.0,
            current_temp=15.0,
        )
        # Temperature predictor should have been called
        assert len(temp_calls) > 0

    def test_temperature_trajectory_built(
        self,
        default_config,
        simple_prices,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        """Verify temperature trajectory is built when temp is provided."""
        optimizer = DPOptimizer(
            config=default_config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        result = optimizer.optimize(
            prices=simple_prices,
            current_slot=simple_prices[0].hour,
            current_soc=50.0,
            current_temp=15.0,
        )
        # Temperature trajectory should be populated
        assert len(result.temp_trajectory) > 0

    def test_no_temperature_trajectory_without_temp(
        self,
        default_config,
        simple_prices,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        """Verify temperature trajectory is empty when no temp provided."""
        optimizer = DPOptimizer(
            config=default_config,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        result = optimizer.optimize(
            prices=simple_prices,
            current_slot=simple_prices[0].hour,
            current_soc=50.0,
            current_temp=None,
        )
        # Temperature trajectory should be empty
        assert len(result.temp_trajectory) == 0
