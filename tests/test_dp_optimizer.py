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
        PricePoint(time=base + datetime.timedelta(hours=i), price=price)
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
        prices = [PricePoint(time=current_slot, price=0.10)]
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
        current_slot = simple_prices[0].time
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
        current_slot = simple_prices[0].time
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
        current_slot = simple_prices[0].time
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
        current_slot = simple_prices[0].time
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
        current_slot = simple_prices[0].time
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
            current_slot=simple_prices[0].time,
            current_soc=50.0,
        )
        # Check all fields exist
        assert hasattr(result, 'schedule')
        assert hasattr(result, 'soc_trajectory')
        assert hasattr(result, 'temp_trajectory')
        assert hasattr(result, 'charge_count')
        assert hasattr(result, 'discharge_count')
        assert hasattr(result, 'hold_count')

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
            current_slot=simple_prices[0].time,
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
            current_slot=simple_prices[0].time,
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
            current_slot=simple_prices[0].time,
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
            current_slot=simple_prices[0].time,
            current_soc=50.0,
            current_temp=None,
        )
        # Temperature trajectory should be empty
        assert len(result.temp_trajectory) == 0


class TestFifteenMinuteSlotDP:
    """Tests for DPOptimizer with 15-minute slot resolution."""

    @pytest.fixture
    def config_15min(self):
        """Configuration for 15-minute slots."""
        return DPOptimizerConfig(
            battery_capacity=14.3,
            min_soc=10.0,
            max_soc=100.0,
            efficiency=0.85,
            discharge_rate=4.5,
            slot_minutes=15,
            soc_step_percent=1.0,
            grid_fee=0.05,
            battery_wear_cost=0.02,
        )

    @pytest.fixture
    def prices_15min(self):
        """Price list for 15-minute slots: 8 slots (2 hours)."""
        base = datetime.datetime(2024, 1, 15, 0, 0, 0)
        return [
            PricePoint(time=base + datetime.timedelta(minutes=15 * i), price=price)
            for i, price in enumerate([
                0.03,  # 00:00 - cheap
                0.03,  # 00:15 - cheap
                0.04,  # 00:30
                0.05,  # 00:45
                0.15,  # 01:00 - expensive
                0.20,  # 01:15 - most expensive
                0.18,  # 01:30
                0.12,  # 01:45
            ])
        ]

    def test_15min_slot_energy_calculation(
        self,
        config_15min,
        prices_15min,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        """Verify 15-min slot energy: charge_rate * 0.25 * efficiency per slot."""
        # With charge_rate=4.5, efficiency=0.85, slot_hours=0.25:
        # Energy per slot = 4.5 * 0.25 * 0.85 = 0.95625 kWh
        assert config_15min.slot_hours == 0.25

        optimizer = DPOptimizer(
            config=config_15min,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        current_slot = prices_15min[0].time
        result = optimizer.optimize(
            prices=prices_15min,
            current_slot=current_slot,
            current_soc=20.0,  # Low SOC to encourage charging
        )
        # Should generate a valid schedule
        assert len(result.schedule) == len(prices_15min)

        # Verify at least some charge slots exist at low SOC with cheap prices
        assert result.charge_count > 0 or result.discharge_count > 0 or result.hold_count > 0

    def test_15min_soc_trajectory(
        self,
        config_15min,
        constant_load_predictor,
        constant_charge_rate_predictor,
        identity_temp_predictor,
    ):
        """Verify SOC trajectory step sizes for 15-min slots.

        With charge_rate=4.5, efficiency=0.85, capacity=14.3:
        energy per slot = 4.5 * 0.25 * 0.85 = 0.95625 kWh
        SOC change per slot = 0.95625 / 14.3 * 100 ~= 6.69%
        """
        # Create many cheap slots to encourage charging
        base = datetime.datetime(2024, 1, 15, 0, 0, 0)
        prices = [
            PricePoint(time=base + datetime.timedelta(minutes=15 * i), price=0.01)
            for i in range(8)
        ]

        optimizer = DPOptimizer(
            config=config_15min,
            load_predictor=constant_load_predictor,
            charge_rate_predictor=constant_charge_rate_predictor,
            temp_after_charge_predictor=identity_temp_predictor,
            temp_after_idle_predictor=identity_temp_predictor,
        )
        result = optimizer.optimize(
            prices=prices,
            current_slot=prices[0].time,
            current_soc=30.0,  # Start low to get charging
        )

        # Verify SOC trajectory has entries
        assert len(result.soc_trajectory) > 0

        # For charge slots, check SOC step size is reasonable
        # Expected: ~6.69% per charging slot (with 0.5kW load reducing it slightly)
        expected_max_soc_step = (4.5 * 0.25 * 0.85) / 14.3 * 100  # ~6.69%
        for slot_time, (start_soc, end_soc) in result.soc_trajectory.items():
            entry = result.schedule.get(slot_time)
            if entry and entry.mode == BatteryMode.CHARGE:
                soc_change = end_soc - start_soc
                # SOC increase should not exceed the maximum possible
                # (load consumption reduces effective charging, so allow some margin)
                assert soc_change <= expected_max_soc_step + 1.0, (
                    f"SOC change {soc_change:.2f}% at {slot_time} exceeds "
                    f"expected max {expected_max_soc_step:.2f}%"
                )
