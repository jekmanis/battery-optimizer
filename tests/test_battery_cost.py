"""
Tests for battery cost tracking functions.

Tests for:
- _project_battery_costs() - projects battery cost evolution through a schedule
- _get_discharge_threshold() - calculates discharge price threshold
- BatteryCostTracker.get_discharge_threshold_for_cost() - threshold for a given cost
"""

import datetime

import pytest

from battery_optimizer import BatteryMode, BatteryOptimizer, ScheduleEntry
from battery_optimizer_lib import BatteryCostTracker, BatteryCostConfig, BatteryLearningEngine


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
        inverter_efficiency: float = 1.0,
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
        self.inverter_efficiency = inverter_efficiency
        self.battery_wear_cost = battery_wear_cost
        self._battery_avg_cost = battery_avg_cost
        self.slot_minutes = slot_minutes
        self.slot_hours = slot_minutes / 60.0

        # Configurable load prediction
        self._predicted_load_kw = predicted_load_kw
        self._load_by_hour = {}

        # Create learning engine (needed by cost tracker)
        self._learning_engine = BatteryLearningEngine(
            battery_capacity_kwh=battery_capacity,
            nominal_charge_rate_kw=charge_rate,
            nominal_efficiency=efficiency,
        )

        # Create real cost tracker
        self._cost_tracker = BatteryCostTracker(
            config=BatteryCostConfig(
                battery_capacity=battery_capacity,
                efficiency=efficiency,
                slot_minutes=slot_minutes,
                charge_rate=charge_rate,
                discharge_rate=discharge_rate,
                grid_fee=grid_fee,
                inverter_efficiency=inverter_efficiency,
                battery_wear_cost=battery_wear_cost,
                default_cost=battery_avg_cost,
            ),
            get_state_func=lambda e: None,
            call_service_func=lambda *a, **k: None,
            get_datetime_func=lambda: datetime.datetime(2024, 1, 1, 12, 0),
            get_timezone_func=lambda: None,
            align_to_slot_func=lambda dt: dt.replace(minute=0, second=0, microsecond=0),
            get_min_soc_func=lambda: self.min_soc,
            get_max_soc_func=lambda: self.max_soc,
            get_current_soc_func=lambda: 50.0,
            get_battery_temp_func=lambda: 20.0,
            learning_engine=self._learning_engine,
            get_cached_prices_func=lambda: [],
            save_learning_data_func=lambda: None,
            update_learning_sensor_func=lambda: None,
            log_func=self.log,
        )
        # Set the avg_cost directly (bypassing the default load)
        self._cost_tracker._avg_cost = battery_avg_cost

    @property
    def battery_avg_cost(self) -> float:
        """Get battery average cost from cost tracker."""
        return self._cost_tracker.avg_cost

    @battery_avg_cost.setter
    def battery_avg_cost(self, value: float):
        """Set battery average cost in cost tracker."""
        self._cost_tracker._avg_cost = value

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

    def _get_discharge_threshold_for_cost(self, avg_cost: float) -> float:
        """Delegate to the cost tracker's threshold-for-cost calculation."""
        return self._cost_tracker.get_discharge_threshold_for_cost(avg_cost)

    def _project_battery_costs(
        self,
        schedule,
        starting_soc,
        starting_cost,
        prices_by_slot,
        slot_fractions_by_slot=None,
    ):
        """Wrapper that calls cost tracker's project_costs with predict_load_func."""
        return self._cost_tracker.project_costs(
            schedule=schedule,
            starting_soc=starting_soc,
            starting_cost=starting_cost,
            prices_by_slot=prices_by_slot,
            predict_load_func=self._predict_load_kw,
            slot_fractions_by_slot=slot_fractions_by_slot,
        )


# Bind the actual method to our mock
MockCostOptimizer._get_discharge_threshold = BatteryOptimizer._get_discharge_threshold


def make_schedule_entry(
    hour: datetime.datetime, mode: BatteryMode, reason: str = "Test"
) -> ScheduleEntry:
    """Create a schedule entry."""
    return ScheduleEntry(time=hour, mode=mode, reason=reason)


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
        landed_cost = (0.05 + optimizer.grid_fee) / optimizer.efficiency
        expected_cost = (
            expected_old_energy * 0.10 + expected_added * landed_cost
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

        # Cost should change to reflect fees and charge losses.
        assert projected[hours[0]] == 0.10  # Starting cost
        assert projected[hours[1]] != 0.10
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

        # After charge, landed cost is reflected.
        cost_after_charge = projected[hours[1]]
        assert cost_after_charge != 0.10

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
        assert final_cost > 0.02

    def test_hold_pv_charge_uses_foregone_export_value(self, optimizer, base_time):
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.HOLD)}

        projected, final_cost = optimizer._cost_tracker.project_costs(
            schedule=schedule,
            starting_soc=optimizer.min_soc,
            starting_cost=0.20,
            prices_by_slot={hour: 0.10},
            predict_load_func=lambda _: 0.5,
            predict_pv_func=lambda _: 2.0,
        )

        assert projected[hour] == 0.20
        # Export would earn 0.10 - 0.02; charge loss is applied per stored kWh.
        assert final_cost == pytest.approx(0.08 / optimizer.efficiency)

    def test_hybrid_charge_blends_pv_and_grid_cost(self, optimizer, base_time):
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.CHARGE)}

        _, final_cost = optimizer._cost_tracker.project_costs(
            schedule=schedule,
            starting_soc=optimizer.min_soc,
            starting_cost=0.20,
            prices_by_slot={hour: 0.10},
            predict_load_func=lambda _: 0.5,
            predict_pv_func=lambda _: 2.0,
        )

        # 4.5 DC kWh charge: 1.5 PV surplus + 3.0 grid DC kWh.
        expected = (1.5 * 0.08 + 3.0 * 0.15) / (4.5 * optimizer.efficiency)
        assert final_cost == pytest.approx(expected)

    def test_grid_landed_cost_includes_multiplier_and_inverter_loss(self, base_time):
        optimizer = MockCostOptimizer(inverter_efficiency=0.90)
        optimizer._cost_tracker._config.import_price_multiplier = 1.21
        hour = base_time

        _, final_cost = optimizer._project_battery_costs(
            schedule={hour: make_schedule_entry(hour, BatteryMode.CHARGE)},
            starting_soc=optimizer.min_soc,
            starting_cost=0.20,
            prices_by_slot={hour: 0.10},
        )

        expected = (0.10 + optimizer.grid_fee) * 1.21 / (
            optimizer.efficiency * optimizer.inverter_efficiency
        )
        assert final_cost == pytest.approx(expected)

    def test_observed_charge_in_discharge_mode_is_pv_opportunity_cost(self, optimizer):
        """Growatt discharge-to-load can still store surplus PV."""
        optimizer._cost_tracker._current_mode = BatteryMode.DISCHARGE

        cost, source = optimizer._cost_tracker._observed_charge_cost(0.10)

        assert source == "pv"
        assert cost == pytest.approx((0.10 - 0.02) / optimizer.efficiency)

    def test_projected_discharge_to_load_can_store_surplus_pv(self, optimizer, base_time):
        hour = base_time
        entry = make_schedule_entry(hour, BatteryMode.DISCHARGE)
        entry.export_rate = 0

        _, final_cost = optimizer._cost_tracker.project_costs(
            schedule={hour: entry},
            starting_soc=optimizer.min_soc,
            starting_cost=0.20,
            prices_by_slot={hour: 0.10},
            predict_load_func=lambda _: 0.5,
            predict_pv_func=lambda _: 2.0,
        )

        assert final_cost == pytest.approx((0.10 - 0.02) / optimizer.efficiency)

    def test_observed_charge_before_mode_callback_uses_conservative_landed_cost(self):
        """An unknown source must not inject raw AC spot into the DC average."""
        optimizer = MockCostOptimizer(efficiency=0.90, inverter_efficiency=0.95)
        optimizer._cost_tracker._config.import_price_multiplier = 1.21

        cost, source = optimizer._cost_tracker._observed_charge_cost(0.10)

        assert source == "unknown-grid"
        assert cost == pytest.approx((0.10 + 0.05) * 1.21 / (0.90 * 0.95))

    def test_loading_persisted_cost_does_not_emit_permanent_migration_warning(self, optimizer):
        messages = []
        optimizer._cost_tracker._get_state = lambda entity: (
            "2" if entity.endswith("basis_version") else "0.1234"
        )
        optimizer._cost_tracker._log = lambda message, **kwargs: messages.append(
            (message, kwargs.get("level"))
        )

        assert optimizer._cost_tracker.load_from_ha() is True
        assert optimizer.battery_avg_cost == pytest.approx(0.1234)
        assert not [message for message, level in messages if level == "WARNING"]

    def test_legacy_persisted_cost_is_migrated_once_to_landed_basis(self, optimizer):
        states = {
            "input_number.battery_avg_cost": "0.10",
            "input_number.battery_cost_basis_version": "1",
        }
        calls = []
        optimizer._cost_tracker._get_state = states.get
        optimizer._cost_tracker._call_service = lambda service, **data: calls.append(
            (service, data)
        )

        assert optimizer._cost_tracker.load_from_ha() is True
        expected = (0.10 + optimizer.grid_fee) / (
            optimizer.efficiency * optimizer.inverter_efficiency
        )
        assert optimizer.battery_avg_cost == pytest.approx(expected)
        assert any(
            data.get("entity_id") == "input_number.battery_cost_basis_version"
            and data.get("value") == 2
            for _, data in calls
        )

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
        landed_cost = (0.05 + optimizer.grid_fee) / optimizer.efficiency
        expected_cost = (old_energy * 0.10 + max_added * landed_cost) / (old_energy + max_added)
        assert abs(final_cost - expected_cost) < 0.001

    # NOTE: test_charge_rates_by_slot_used lived here. `project_costs` took a
    # per-slot rate array that never reached the column:
    # `soc_projection._effective_charge_rate` always prefers the learning
    # engine, which is passed in production. The parameter is gone; what the
    # rate does to the projected cost is covered by
    # TestProjectedCostsUseTheSharedChargeModel below, through the engine.

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
        landed_cost = (0.05 + optimizer.grid_fee) / optimizer.efficiency
        expected_cost = (old_energy * 0.10 + added_energy * landed_cost) / (
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

        # After charging, landed acquisition cost changes the average.
        assert projected[hours[1]] != 0.10

        # Discharge and hold don't change cost
        assert projected[hours[2]] == projected[hours[1]]
        assert projected[hours[3]] == projected[hours[2]]

        # After second charge, cost changes again
        # Fees and charge losses are included, so raw spot below 0.10 does not
        # necessarily mean stored energy costs less than 0.10.
        assert final_cost != 0.10


class TestGetDischargeThreshold:
    """Test cases for _get_discharge_threshold."""

    def test_basic_threshold(self):
        """Basic threshold calculation."""
        # Average cost is already landed per stored DC kWh.
        optimizer = MockCostOptimizer(
            battery_avg_cost=0.10,
            efficiency=0.85,
            grid_fee=0.05,
            battery_wear_cost=0.0,
        )

        threshold = optimizer._get_discharge_threshold()
        expected = 0.10
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
        expected = 0.10 + 0.02
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
        expected = 0.10
        assert abs(threshold - expected) < 0.001

    def test_charge_efficiency_does_not_double_count_threshold(self):
        """Charge efficiency is already represented in landed average cost."""
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

        assert threshold_high == threshold_low

    def test_inverter_efficiency_raises_threshold(self):
        optimizer = MockCostOptimizer(
            battery_avg_cost=0.10,
            inverter_efficiency=0.90,
        )
        assert optimizer._get_discharge_threshold() == pytest.approx(0.10 / 0.90)


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
        """Zero acquisition cost should include only wear."""
        threshold = optimizer._get_discharge_threshold_for_cost(0.0)

        expected = 0.01
        assert abs(threshold - expected) < 0.001

    def test_negative_cost(self, optimizer):
        """Negative cost (paid to take energy) should work."""
        # This could happen with negative electricity prices
        threshold = optimizer._get_discharge_threshold_for_cost(-0.05)

        expected = -0.05 + 0.01
        assert abs(threshold - expected) < 0.001

    def test_proportional_to_cost(self, optimizer):
        """Threshold should scale linearly with stored-energy cost."""
        threshold_1 = optimizer._get_discharge_threshold_for_cost(0.10)
        threshold_2 = optimizer._get_discharge_threshold_for_cost(0.20)

        # With unity inverter efficiency, difference equals stored-cost difference.
        cost_diff = 0.20 - 0.10
        expected_threshold_diff = cost_diff
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


class TestFifteenMinCostTracking:
    """Test battery cost calculations with 15-minute slots."""

    def test_15min_charge_energy_calculation(self):
        """Cost tracker should compute energy for a 15-min slot correctly.

        With charge_rate=4.5 kW, efficiency=0.85, slot_minutes=15:
            energy = 4.5 * 0.85 * 0.25 = 0.95625 kWh per full slot

        Not 0.5h (30-min) or 1.0h (60-min).
        """
        optimizer = MockCostOptimizer(
            charge_rate=4.5,
            efficiency=0.85,
            slot_minutes=15,
        )

        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
        hour = base_time
        schedule = {hour: make_schedule_entry(hour, BatteryMode.CHARGE)}
        prices_by_slot = {hour: 0.05}

        # Starting at 50% SOC with 0.10 avg cost
        starting_soc = 50.0
        starting_cost = 0.10

        _, final_cost = optimizer._project_battery_costs(
            schedule=schedule,
            starting_soc=starting_soc,
            starting_cost=starting_cost,
            prices_by_slot=prices_by_slot,
        )

        # Calculate expected values with 15-min slot (0.25 hours)
        old_energy = (starting_soc - optimizer.min_soc) / 100 * optimizer.battery_capacity
        energy_added = 4.5 * 0.85 * 0.25  # 0.95625 kWh for 15-min slot
        landed_cost = (0.05 + optimizer.grid_fee) / optimizer.efficiency
        expected_cost = (old_energy * starting_cost + energy_added * landed_cost) / (
            old_energy + energy_added
        )
        assert abs(final_cost - expected_cost) < 0.001

        # Verify the energy is specifically for a 15-min slot, not 30-min or 60-min
        energy_30min = 4.5 * 0.85 * 0.5  # would be 1.9125 for 30-min
        energy_60min = 4.5 * 0.85 * 1.0  # would be 3.825 for 60-min

        cost_if_30min = (old_energy * starting_cost + energy_30min * landed_cost) / (
            old_energy + energy_30min
        )
        cost_if_60min = (old_energy * starting_cost + energy_60min * landed_cost) / (
            old_energy + energy_60min
        )

        # final_cost should NOT match 30-min or 60-min calculations
        assert abs(final_cost - cost_if_30min) > 0.001
        assert abs(final_cost - cost_if_60min) > 0.001


class TestProjectCostsUsesTheSharedSocModel:
    """``project_costs`` must not own a fourth slot-SOC transition.

    It used to re-implement the transition inline, and the copy differed from
    ``soc_projection.project_slot_soc``: it clamped DISCHARGE at ``min_soc``
    BEFORE adding PV surplus and capped charging with its own headroom
    arithmetic.  Anything the projected-cost column disagreed with the
    expected-SOC and deviation columns about landed in the same log, which is
    the class of contradiction that caused the production recalculation loop.
    """

    @pytest.fixture
    def optimizer(self):
        return MockCostOptimizer(inverter_efficiency=0.95, slot_minutes=15)

    @staticmethod
    def _params(tracker):
        from battery_optimizer_lib.soc_projection import SocProjectionParams

        cfg = tracker._config
        return SocProjectionParams(
            battery_capacity=cfg.battery_capacity,
            efficiency=cfg.efficiency,
            charge_rate=cfg.charge_rate,
            discharge_rate=cfg.discharge_rate,
            export_discharge_rate=cfg.export_discharge_rate,
            inverter_efficiency=cfg.inverter_efficiency,
            min_soc=tracker._get_min_soc(),
            max_soc=tracker._get_max_soc(),
            slot_minutes=cfg.slot_minutes,
        )

    def _mixed_schedule(self, base):
        """CHARGE, PV-surplus DISCHARGE, net-load DISCHARGE, export, HOLD."""
        slots = [base + datetime.timedelta(minutes=15 * i) for i in range(6)]
        modes = [
            BatteryMode.DISCHARGE,   # net load -> discharges into the min_soc clamp
            BatteryMode.CHARGE,
            BatteryMode.DISCHARGE,   # pv > load -> PV charges the pack
            BatteryMode.DISCHARGE,   # export
            BatteryMode.HOLD,        # pv > load
            BatteryMode.CHARGE,
        ]
        schedule = {}
        for slot, mode in zip(slots, modes):
            entry = make_schedule_entry(slot, mode)
            schedule[slot] = entry
        schedule[slots[3]].export_rate = 100.0
        return slots, schedule

    @pytest.mark.parametrize("starting_soc", [10.5, 99.0])
    def test_end_soc_per_slot_equals_project_slot_soc(
        self, optimizer, monkeypatch, starting_soc
    ):
        from battery_optimizer_lib import soc_projection

        base = datetime.datetime(2024, 1, 1, 10, 0)
        slots, schedule = self._mixed_schedule(base)
        # starting_soc=10.5 drives the first DISCHARGE into the min_soc clamp;
        # starting_soc=99.0 drives the CHARGE slots into the max_soc clamp.
        load_by_slot = {
            slots[0]: 9.0, slots[1]: 0.5, slots[2]: 0.4,
            slots[3]: 0.5, slots[4]: 0.3, slots[5]: 0.5,
        }
        pv_by_slot = {
            slots[0]: 0.0, slots[1]: 0.0, slots[2]: 4.0,
            slots[3]: 3.0, slots[4]: 5.0, slots[5]: 0.0,
        }
        prices = {slot: 0.10 for slot in slots}

        real = soc_projection.project_slot_soc
        seen = []

        def spy(**kwargs):
            result = real(**kwargs)
            seen.append((kwargs["soc_start"], result.soc_end))
            return result

        monkeypatch.setattr(soc_projection, "project_slot_soc", spy)

        tracker = optimizer._cost_tracker
        tracker.project_costs(
            schedule=schedule,
            starting_soc=starting_soc,
            starting_cost=0.12,
            prices_by_slot=prices,
            predict_load_func=lambda s: load_by_slot[s],
            predict_pv_func=lambda s: pv_by_slot[s],
        )

        assert len(seen) == len(slots), "project_costs bypassed the shared model"

        # Independently chained reference trajectory.
        params = self._params(tracker)
        soc = starting_soc
        expected = []
        for slot in slots:
            transition = real(
                soc_start=soc,
                mode=schedule[slot].mode,
                params=params,
                load_kw=load_by_slot[slot],
                pv_kw=pv_by_slot[slot],
                fraction=1.0,
                export_rate=schedule[slot].export_rate,
            )
            soc = transition.soc_end
            expected.append(soc)

        assert [end for _, end in seen] == pytest.approx(expected)
        # Each slot starts where the previous one ended: one continuous model.
        assert [start for start, _ in seen] == pytest.approx(
            [starting_soc] + expected[:-1]
        )

        # The clamp cases really were exercised.
        assert min(expected) >= tracker._get_min_soc() - 1e-9
        assert max(expected) <= tracker._get_max_soc() + 1e-9
        if starting_soc < 20.0:
            assert expected[0] == pytest.approx(tracker._get_min_soc())
        else:
            assert max(expected) == pytest.approx(tracker._get_max_soc())

    def test_pv_surplus_during_discharge_charges_before_the_min_soc_clamp(
        self, optimizer
    ):
        """A cloud-safe HOLD->discharge_to_load slot at min_soc must still charge.

        The old inline copy clamped at ``min_soc`` first and only then added PV
        surplus; the shared model adds PV, clamps at ``max_soc``, then
        subtracts. With the battery sitting at the floor and PV covering the
        load, the projected column must show the pack RISING.
        """
        base = datetime.datetime(2024, 1, 1, 12, 0)
        schedule = {base: make_schedule_entry(base, BatteryMode.DISCHARGE)}
        tracker = optimizer._cost_tracker

        projected, final_cost = tracker.project_costs(
            schedule=schedule,
            starting_soc=tracker._get_min_soc(),
            starting_cost=0.20,
            prices_by_slot={base: 0.10},
            predict_load_func=lambda _: 0.4,
            predict_pv_func=lambda _: 4.0,
        )

        assert projected[base] == pytest.approx(0.20)
        # PV surplus was stored, so the cost basis moved toward the foregone
        # export value instead of staying put.
        assert final_cost < 0.20

    def test_slot_fractions_are_honoured_by_the_shared_model(self, optimizer):
        base = datetime.datetime(2024, 1, 1, 10, 0)
        schedule = {base: make_schedule_entry(base, BatteryMode.CHARGE)}
        tracker = optimizer._cost_tracker

        _, full = tracker.project_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.30,
            prices_by_slot={base: 0.10},
            predict_load_func=lambda _: 0.5,
        )
        _, half = tracker.project_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.30,
            prices_by_slot={base: 0.10},
            predict_load_func=lambda _: 0.5,
            slot_fractions_by_slot={base: 0.5},
        )
        # Half the energy at the same landed cost moves the average half as far.
        assert 0.30 > half > full
class _RecordingLearningEngine:
    """Learning-engine stub that records what the cost tracker feeds it."""

    def __init__(self):
        self.charge_calls = []
        self.discharge_calls = []

    def record_charging(self, **kwargs):
        self.charge_calls.append(kwargs)

    def record_discharging(self, **kwargs):
        self.discharge_calls.append(kwargs)

    def record_temperature_observation(self, temp):
        pass

    def record_cooling(self, **kwargs):
        pass

    def get_charge_rate_for_soc(self, soc, temp=None):
        return None

    def predict_temp_after_idle(self, temp, duration_minutes):
        return temp


class TestDepletedSocStartIsNotFalsy:
    """A measured SOC of 0.0 % is an observation, not "unset".

    ``_process_energy_change`` used ``self._last_soc if self._last_soc else
    current_soc``.  Right after a genuine depletion ``_last_soc == 0.0`` is
    falsy, so the tracker substituted the *current* SOC and called
    ``record_charging(soc_start=current_soc, soc_end=current_soc)``.  That trips
    ``learning_engine.record_charging``'s ``soc_end <= soc_start`` early return,
    so the charge-rate, efficiency and thermal samples of the deep-discharge
    curve - exactly the ones the model has fewest of - were silently dropped.
    """

    @staticmethod
    def _make_tracker(engine, soc_holder, state):
        clock = {"now": datetime.datetime(2024, 1, 15, 10, 0)}

        tracker = BatteryCostTracker(
            config=BatteryCostConfig(
                battery_capacity=10.0,
                efficiency=1.0,
                slot_minutes=15,
                charge_rate=4.0,
                discharge_rate=4.0,
                use_inverter_energy_sensors=True,
            ),
            get_state_func=lambda e: state.get(e),
            call_service_func=lambda *a, **k: None,
            get_datetime_func=lambda: clock["now"],
            get_timezone_func=lambda: None,
            align_to_slot_func=lambda dt: dt.replace(
                minute=(dt.minute // 15) * 15, second=0, microsecond=0
            ),
            get_min_soc_func=lambda: 0.0,
            get_max_soc_func=lambda: 100.0,
            get_current_soc_func=lambda: soc_holder["soc"],
            get_battery_temp_func=lambda: 20.0,
            learning_engine=engine,
            get_cached_prices_func=lambda: [],
            save_learning_data_func=lambda: None,
            update_learning_sensor_func=lambda: None,
            log_func=lambda *a, **k: None,
        )
        tracker.initialize()
        return tracker, clock

    def test_charge_after_depletion_keeps_soc_start_at_zero(self):
        engine = _RecordingLearningEngine()
        soc_holder = {"soc": 20.0}
        state = {
            "sensor.growatt_battery_charge_today": "5.0",
            "sensor.growatt_battery_discharge_today": "3.0",
        }
        tracker, clock = self._make_tracker(engine, soc_holder, state)

        # Discharge the pack all the way to 0 %.
        clock["now"] += datetime.timedelta(minutes=30)
        soc_holder["soc"] = 0.0
        state["sensor.growatt_battery_discharge_today"] = "5.0"
        tracker.on_energy_sensor_change(
            "sensor.growatt_battery_discharge_today", "3.0", "5.0"
        )
        assert tracker.last_soc == 0.0

        # Now charge: the SOC sensor has moved off the floor.
        clock["now"] += datetime.timedelta(minutes=15)
        soc_holder["soc"] = 2.0
        state["sensor.growatt_battery_charge_today"] = "6.0"
        tracker.on_energy_sensor_change(
            "sensor.growatt_battery_charge_today", "5.0", "6.0"
        )

        assert len(engine.charge_calls) == 1
        call = engine.charge_calls[0]
        assert call["soc_start"] == 0.0
        # The observation must be usable: record_charging drops soc_end <= soc_start.
        assert call["soc_end"] > call["soc_start"]

    def test_discharge_after_depletion_keeps_soc_start_at_zero(self):
        engine = _RecordingLearningEngine()
        soc_holder = {"soc": 20.0}
        state = {
            "sensor.growatt_battery_charge_today": "5.0",
            "sensor.growatt_battery_discharge_today": "3.0",
        }
        tracker, clock = self._make_tracker(engine, soc_holder, state)

        clock["now"] += datetime.timedelta(minutes=30)
        soc_holder["soc"] = 0.0
        state["sensor.growatt_battery_discharge_today"] = "5.0"
        tracker.on_energy_sensor_change(
            "sensor.growatt_battery_discharge_today", "3.0", "5.0"
        )

        # A second discharge event while the SOC sensor reports 1 % (PV briefly
        # lifted it off the floor): the baseline is still the recorded 0.0 %.
        clock["now"] += datetime.timedelta(minutes=15)
        soc_holder["soc"] = 1.0
        state["sensor.growatt_battery_discharge_today"] = "5.5"
        tracker.on_energy_sensor_change(
            "sensor.growatt_battery_discharge_today", "5.0", "5.5"
        )

        assert len(engine.discharge_calls) == 2
        assert engine.discharge_calls[1]["soc_start"] == 0.0


class _WarmingChargeEngine:
    """Learning engine whose learned rate differs sharply from the nominal one.

    The rate is CONSTANT within a slot (the one within-slot model) and depends
    on the temperature the slot STARTS at; the pack warms between slots. It used
    to expose ``predict_charge_input_dc_energy`` and the projection called it,
    which split a slot into a cold and a warm phase -- a second thermal model.
    """

    def __init__(self, rate_kw: float = 12.0, warming_per_15min: float = 2.0):
        self._rate = rate_kw
        self._warming = warming_per_15min
        self.rate_calls = []

    def get_charge_rate_for_soc(self, soc, temp=None):
        self.rate_calls.append((soc, temp))
        return self._rate

    def predict_temp_after_duration(self, temp, duration_minutes):
        return temp + self._warming * duration_minutes / 15.0

    def predict_temp_after_idle(self, temp, duration_minutes):
        return temp


class TestProjectCostsUsesTheLearnedChargeModel:
    """The projected-cost column must use the same CHARGE model as the SOC column.

    ``project_costs`` called ``project_slot_soc`` without ``learning_engine``
    and ``temp_start``, so its CHARGE slots silently fell back to the nominal
    ``charge_rate * efficiency * duration`` while
    ``project_schedule_trajectory`` - which passes both - used the LEARNED
    rate.  One log then carried two different charge models, the contradiction
    that adopting the shared model was supposed to remove.

    Retargeted when the within-slot cold/warm split was removed: the mechanism
    is now ``get_charge_rate_for_soc`` at the start-of-slot temperature plus the
    thermal model between slots, not ``predict_charge_input_dc_energy``.
    """

    @staticmethod
    def _params(tracker, charge_rate=None):
        from battery_optimizer_lib.soc_projection import SocProjectionParams

        cfg = tracker._config
        return SocProjectionParams(
            battery_capacity=cfg.battery_capacity,
            efficiency=cfg.efficiency,
            charge_rate=cfg.charge_rate if charge_rate is None else charge_rate,
            discharge_rate=cfg.discharge_rate,
            export_discharge_rate=cfg.export_discharge_rate,
            inverter_efficiency=cfg.inverter_efficiency,
            min_soc=tracker._get_min_soc(),
            max_soc=tracker._get_max_soc(),
            slot_minutes=cfg.slot_minutes,
        )

    def test_charge_slot_matches_project_slot_soc_with_the_engine(self):
        from battery_optimizer_lib.soc_projection import project_slot_soc

        optimizer = MockCostOptimizer(slot_minutes=15)
        tracker = optimizer._cost_tracker
        engine = _WarmingChargeEngine()

        base = datetime.datetime(2024, 1, 15, 3, 0)
        schedule = {base: make_schedule_entry(base, BatteryMode.CHARGE)}
        prices = {base: 0.10}

        _, learned_cost = tracker.project_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.30,
            prices_by_slot=prices,
            predict_load_func=lambda _: 0.5,
            starting_temp=20.0,
            learning_engine=engine,
        )
        assert engine.rate_calls, "project_costs never reached the learned rate"

        def cost_for(dc_in: float) -> float:
            old_energy = tracker._soc_to_energy_kwh(50.0)
            landed = tracker._grid_landed_cost(0.10)
            return (old_energy * 0.30 + dc_in * landed) / (old_energy + dc_in)

        params = self._params(tracker)
        with_engine = project_slot_soc(
            soc_start=50.0,
            mode=BatteryMode.CHARGE,
            params=params,
            load_kw=0.5,
            pv_kw=0.0,
            temp_start=20.0,
            learning_engine=engine,
        )
        flat = project_slot_soc(
            soc_start=50.0,
            mode=BatteryMode.CHARGE,
            params=params,
            load_kw=0.5,
            pv_kw=0.0,
        )
        assert with_engine.dc_energy_in_kwh != pytest.approx(flat.dc_energy_in_kwh)

        assert learned_cost == pytest.approx(cost_for(with_engine.dc_energy_in_kwh))
        assert learned_cost != pytest.approx(cost_for(flat.dc_energy_in_kwh))

    def test_temperature_evolves_across_slots_like_the_soc_trajectory(self, monkeypatch):
        from battery_optimizer_lib import soc_projection

        optimizer = MockCostOptimizer(slot_minutes=15)
        tracker = optimizer._cost_tracker
        engine = _WarmingChargeEngine()

        base = datetime.datetime(2024, 1, 15, 3, 0)
        slots = [base + datetime.timedelta(minutes=15 * i) for i in range(3)]
        schedule = {s: make_schedule_entry(s, BatteryMode.CHARGE) for s in slots}

        real = soc_projection.project_slot_soc
        seen = []

        def spy(**kwargs):
            result = real(**kwargs)
            seen.append((kwargs, result))
            return result

        monkeypatch.setattr(soc_projection, "project_slot_soc", spy)

        tracker.project_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.30,
            prices_by_slot={s: 0.10 for s in slots},
            predict_load_func=lambda _: 0.5,
            starting_temp=20.0,
            learning_engine=engine,
        )

        assert len(seen) == 3
        # Each slot starts at the previous slot end temperature (+2 C/slot here).
        temps_in = [kwargs["temp_start"] for kwargs, _ in seen]
        assert temps_in == pytest.approx([20.0, 22.0, 24.0])
        assert all(kwargs["learning_engine"] is engine for kwargs, _ in seen)
        assert [kwargs["slot_time"] for kwargs, _ in seen] == slots

    def test_defaults_preserve_the_flat_rate_behaviour(self):
        """Callers that pass neither temp nor engine keep the old projection."""
        from battery_optimizer_lib.soc_projection import project_slot_soc

        optimizer = MockCostOptimizer(slot_minutes=15)
        tracker = optimizer._cost_tracker

        base = datetime.datetime(2024, 1, 15, 3, 0)
        schedule = {base: make_schedule_entry(base, BatteryMode.CHARGE)}

        _, final_cost = tracker.project_costs(
            schedule=schedule,
            starting_soc=50.0,
            starting_cost=0.30,
            prices_by_slot={base: 0.10},
            predict_load_func=lambda _: 0.5,
        )

        flat = project_slot_soc(
            soc_start=50.0,
            mode=BatteryMode.CHARGE,
            params=self._params(tracker),
            load_kw=0.5,
            pv_kw=0.0,
        )
        old_energy = tracker._soc_to_energy_kwh(50.0)
        landed = tracker._grid_landed_cost(0.10)
        expected = (old_energy * 0.30 + flat.dc_energy_in_kwh * landed) / (
            old_energy + flat.dc_energy_in_kwh
        )
        assert final_cost == pytest.approx(expected)
