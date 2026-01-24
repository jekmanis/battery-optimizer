"""
Tests for BatteryLearningEngine class.
"""

import json

import pytest

from battery_optimizer import BatteryLearningEngine, LearningStats


class TestLearningStats:
    """Test cases for LearningStats dataclass."""

    def test_default_initialization(self):
        """Default values should be set correctly."""
        stats = LearningStats()
        assert stats.charge_rates_by_soc == {}
        assert stats.discharge_rates_by_soc == {}
        assert stats.efficiency_history == []
        assert stats.total_energy_charged_kwh == 0.0
        assert stats.total_energy_discharged_kwh == 0.0
        assert stats.total_charge_cost_eur == 0.0
        assert stats.first_observation is None
        assert stats.charge_rates_by_soc_temp == {}

    def test_to_dict_roundtrip(self):
        """Data should survive dict serialization."""
        stats = LearningStats(
            charge_rates_by_soc={"25-50": [3.5, 3.6, 3.7]},
            total_energy_charged_kwh=100.0,
            total_cycles=10,
        )

        data = stats.to_dict()
        restored = LearningStats.from_dict(data)

        assert restored.charge_rates_by_soc == {"25-50": [3.5, 3.6, 3.7]}
        assert restored.total_energy_charged_kwh == 100.0
        assert restored.total_cycles == 10

    def test_from_dict_backward_compatibility(self):
        """Should handle missing fields from older versions."""
        old_data = {
            "charge_rates_by_soc": {"25-50": [3.5]},
            "total_energy_charged_kwh": 50.0,
            # Missing: charge_rates_by_soc_temp (added in later version)
        }

        stats = LearningStats.from_dict(old_data)
        assert stats.charge_rates_by_soc == {"25-50": [3.5]}
        assert stats.charge_rates_by_soc_temp == {}  # Default

    def test_from_dict_ignores_unknown_fields(self):
        """Should ignore unknown fields from future versions."""
        future_data = {
            "charge_rates_by_soc": {"25-50": [3.5]},
            "unknown_future_field": "some value",
            "another_new_field": [1, 2, 3],
        }

        # Should not raise
        stats = LearningStats.from_dict(future_data)
        assert stats.charge_rates_by_soc == {"25-50": [3.5]}


class TestBatteryLearningEngine:
    """Test cases for BatteryLearningEngine."""

    def test_initialization_defaults(self, learning_engine):
        """Engine should initialize with correct defaults."""
        assert learning_engine.battery_capacity == 14.3
        assert learning_engine.nominal_charge_rate == 4.5
        assert learning_engine.nominal_efficiency == 0.85
        assert learning_engine.learned_efficiency == 0.85

    def test_get_soc_range_boundaries(self, learning_engine):
        """SOC range mapping should handle boundaries correctly."""
        assert learning_engine._get_soc_range(0) == "0-25"
        assert learning_engine._get_soc_range(24.9) == "0-25"
        assert learning_engine._get_soc_range(25) == "25-50"
        assert learning_engine._get_soc_range(49.9) == "25-50"
        assert learning_engine._get_soc_range(50) == "50-75"
        assert learning_engine._get_soc_range(74.9) == "50-75"
        assert learning_engine._get_soc_range(75) == "75-90"
        assert learning_engine._get_soc_range(89.9) == "75-90"
        assert learning_engine._get_soc_range(90) == "90-100"
        assert learning_engine._get_soc_range(100) == "90-100"

    def test_get_temp_range_boundaries(self, learning_engine):
        """Temperature range mapping with default boundaries [5, 10, 15, 20]."""
        assert learning_engine._get_temp_range(-5) == "<5"
        assert learning_engine._get_temp_range(4.9) == "<5"
        assert learning_engine._get_temp_range(5) == "5-10"
        assert learning_engine._get_temp_range(9.9) == "5-10"
        assert learning_engine._get_temp_range(10) == "10-15"
        assert learning_engine._get_temp_range(14.9) == "10-15"
        assert learning_engine._get_temp_range(15) == "15-20"
        assert learning_engine._get_temp_range(19.9) == "15-20"
        assert learning_engine._get_temp_range(20) == ">20"
        assert learning_engine._get_temp_range(30) == ">20"

    def test_get_temp_range_custom_boundaries(self):
        """Custom temperature boundaries should work."""
        engine = BatteryLearningEngine(
            temp_ranges=[0, 15, 30],  # Custom: <0, 0-15, 15-30, >30
        )
        assert engine._get_temp_range(-10) == "<0"
        assert engine._get_temp_range(10) == "0-15"
        assert engine._get_temp_range(20) == "15-30"
        assert engine._get_temp_range(35) == ">30"

    def test_get_temp_range_no_boundaries(self):
        """Empty temp_ranges should return 'all'."""
        engine = BatteryLearningEngine(temp_ranges=[])
        assert engine._get_temp_range(15) == "all"

    def test_record_charging_basic(self, learning_engine):
        """Recording a charge should update statistics."""
        learning_engine.record_charging(
            soc_start=30,
            soc_end=50,
            duration_minutes=60,
            charge_price=0.10,
        )

        # Energy added: (50-30)/100 * 14.3 = 2.86 kWh
        assert learning_engine.stats.total_energy_charged_kwh > 0
        assert learning_engine.stats.first_observation is not None
        assert "25-50" in learning_engine.stats.charge_rates_by_soc

    def test_record_charging_with_temperature(self, learning_engine):
        """Recording with temperature should populate temp-aware stats."""
        learning_engine.record_charging(
            soc_start=30,
            soc_end=50,
            duration_minutes=60,
            battery_temp=12.0,
        )

        # Should have entry in temp-aware dict
        assert "25-50" in learning_engine.stats.charge_rates_by_soc_temp
        assert "10-15" in learning_engine.stats.charge_rates_by_soc_temp["25-50"]

    def test_record_charging_invalid_duration(self, learning_engine):
        """Zero or negative duration should be ignored."""
        learning_engine.record_charging(soc_start=30, soc_end=50, duration_minutes=0)
        learning_engine.record_charging(soc_start=30, soc_end=50, duration_minutes=-10)
        assert learning_engine.stats.total_energy_charged_kwh == 0

    def test_record_charging_invalid_soc(self, learning_engine):
        """End SOC <= start SOC should be ignored."""
        learning_engine.record_charging(soc_start=50, soc_end=30, duration_minutes=60)
        learning_engine.record_charging(soc_start=50, soc_end=50, duration_minutes=60)
        assert learning_engine.stats.total_energy_charged_kwh == 0

    def test_record_charging_limits_observations(self, learning_engine):
        """Should keep only last 50 observations per bucket."""
        for i in range(60):
            learning_engine.record_charging(
                soc_start=30,
                soc_end=40,
                duration_minutes=60,
            )

        observations = learning_engine.stats.charge_rates_by_soc.get("25-50", [])
        assert len(observations) == 50

    def test_record_charging_updates_efficiency(self, learning_engine):
        """Efficiency should update from grid energy measurement."""
        # 2.86 kWh added to battery from 3.0 kWh grid = 95% efficiency
        learning_engine.record_charging(
            soc_start=30,
            soc_end=50,
            duration_minutes=60,
            energy_from_grid_kwh=3.0,
        )

        # Should have moved toward 0.95 from 0.85
        assert learning_engine.learned_efficiency > 0.85

    def test_record_discharging_basic(self, learning_engine):
        """Recording discharge should update statistics."""
        learning_engine.record_discharging(
            soc_start=80,
            soc_end=60,
            duration_minutes=60,
            price_eur_kwh=0.15,
        )

        # Energy: (80-60)/100 * 14.3 = 2.86 kWh
        assert learning_engine.stats.total_energy_discharged_kwh > 0
        assert learning_engine.stats.total_discharge_revenue_eur > 0

    def test_record_discharging_invalid(self, learning_engine):
        """Invalid discharge should be ignored."""
        learning_engine.record_discharging(soc_start=60, soc_end=80, duration_minutes=60)
        assert learning_engine.stats.total_energy_discharged_kwh == 0

    def test_get_charge_rate_fallback_chain(self, learning_engine):
        """Should fall back through: temp+soc -> soc -> nominal."""
        # No data: should return nominal
        rate = learning_engine.get_charge_rate_for_soc(40)
        assert rate == 4.5  # nominal

        # Add SOC-only data
        for _ in range(5):
            learning_engine.record_charging(soc_start=30, soc_end=50, duration_minutes=60)

        # Should now use learned rate
        rate_with_soc_data = learning_engine.get_charge_rate_for_soc(40)
        assert rate_with_soc_data != 4.5  # Should be learned value

    def test_get_charge_rate_temp_aware(self, learning_engine_with_data):
        """Temperature-aware lookup should work when data exists."""
        engine = learning_engine_with_data

        # Add temp-aware data
        for _ in range(5):
            engine.record_charging(
                soc_start=30,
                soc_end=50,
                duration_minutes=60,
                battery_temp=15.0,
            )

        # Should use temp-aware data when available
        rate = engine.get_charge_rate_for_soc(40, battery_temp=15.0)
        assert rate > 0

    def test_get_confidence_no_data(self, learning_engine):
        """No data should return 0 confidence."""
        conf = learning_engine.get_confidence_for_soc(40)
        assert conf == 0.0

    def test_get_confidence_with_data(self, learning_engine_with_data):
        """With data, confidence should be > 0."""
        conf = learning_engine_with_data.get_confidence_for_soc(40)
        assert conf > 0

    def test_predict_charge_time_zero_delta(self, learning_engine):
        """No charging needed should return zero time."""
        expected, min_t, max_t = learning_engine.predict_charge_time(50, 50)
        assert expected == 0.0
        assert min_t == 0.0
        assert max_t == 0.0

    def test_predict_charge_time_already_above(self, learning_engine):
        """Already above target should return zero."""
        expected, min_t, max_t = learning_engine.predict_charge_time(80, 50)
        assert expected == 0.0

    def test_predict_charge_time_basic(self, learning_engine):
        """Basic charge time prediction."""
        expected, min_t, max_t = learning_engine.predict_charge_time(30, 80)

        # Should take some time
        assert expected > 0
        # Should have reasonable bounds
        assert min_t <= expected <= max_t
        # With no learned data, uncertainty should be ~30%
        assert min_t >= expected * 0.7
        assert max_t <= expected * 1.3

    def test_predict_charge_time_with_temperature(self, learning_engine_with_data):
        """Temperature should affect prediction."""
        engine = learning_engine_with_data

        expected_warm, _, _ = engine.predict_charge_time(30, 80, battery_temp=20)
        expected_cold, _, _ = engine.predict_charge_time(30, 80, battery_temp=5)

        # Both should be positive
        assert expected_warm > 0
        assert expected_cold > 0

    def test_get_learning_summary(self, learning_engine_with_data):
        """Summary should include all expected fields."""
        summary = learning_engine_with_data.get_learning_summary()

        assert "learned_efficiency" in summary
        assert "total_energy_charged_kwh" in summary
        assert "total_energy_discharged_kwh" in summary
        assert "overall_efficiency" in summary
        assert "total_profit_eur" in summary
        assert "total_observations" in summary
        assert "soc_charge_rates" in summary
        assert "temp_aware_rates" in summary

        # Should have some observations
        assert summary["total_observations"] > 0

    def test_save_and_load_json(self, learning_engine_with_data):
        """JSON serialization should preserve state."""
        engine = learning_engine_with_data

        # Save
        json_str = engine.save_to_json()
        assert json_str is not None

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["version"] == 5

        # Create new engine and load
        new_engine = BatteryLearningEngine()
        success = new_engine.load_from_json(json_str)

        assert success
        assert new_engine.stats.total_energy_charged_kwh == engine.stats.total_energy_charged_kwh

    def test_load_json_invalid(self, learning_engine):
        """Invalid JSON should return False."""
        assert learning_engine.load_from_json("not valid json") is False
        assert learning_engine.load_from_json("{}") is False  # Missing version
        assert learning_engine.load_from_json('{"version": 0}') is False

    def test_load_json_older_version(self, learning_engine):
        """Should handle older version data."""
        old_json = json.dumps({
            "version": 1,
            "learned_efficiency": 0.88,
            "stats": {
                "charge_rates_by_soc": {"25-50": [3.5, 3.6]},
                "discharge_rates_by_soc": {},
                "efficiency_history": [0.85, 0.88],
                "prediction_errors": [],
                "total_energy_charged_kwh": 100.0,
                "total_energy_discharged_kwh": 50.0,
                "total_charge_cost_eur": 5.0,
                "total_discharge_revenue_eur": 10.0,
                "total_cycles": 5,
                "first_observation": "2024-01-01T00:00:00",
                "last_observation": "2024-01-15T12:00:00",
            }
        })

        success = learning_engine.load_from_json(old_json)
        assert success
        assert learning_engine.learned_efficiency == 0.88
        assert learning_engine.stats.total_energy_charged_kwh == 100.0
