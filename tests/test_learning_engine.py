"""
Tests for BatteryLearningEngine class.
"""

import json

import pytest

from battery_optimizer_lib import BatteryLearningEngine, LearningStats
from battery_optimizer_lib.thermal_model import step_temperature


class TestLearningStats:
    """Test cases for LearningStats dataclass."""

    def test_default_initialization(self):
        """Default values should be set correctly."""
        stats = LearningStats()
        assert stats.charge_rates_by_soc == {}
        assert stats.total_energy_charged_kwh == 0.0
        assert stats.total_energy_discharged_kwh == 0.0
        assert stats.total_charge_cost_eur == 0.0
        assert stats.charge_rates_by_soc_temp == {}

    def test_to_dict_roundtrip(self):
        """Data should survive dict serialization."""
        stats = LearningStats(
            charge_rates_by_soc={"25-50": [3.5, 3.6, 3.7]},
            total_energy_charged_kwh=100.0,
        )

        data = stats.to_dict()
        restored = LearningStats.from_dict(data)

        assert restored.charge_rates_by_soc == {"25-50": [3.5, 3.6, 3.7]}
        assert restored.total_energy_charged_kwh == 100.0

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
        assert data["version"] == 6  # v6 adds thermal_samples / thermal_coeffs

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


class TestCoolingRateLearning:
    """Test cases for cooling rate learning."""

    def test_record_cooling_basic(self, learning_engine):
        """Recording cooling should store data in temp_cooling_rates."""
        learning_engine.record_cooling(
            temp_start=21.0,
            temp_end=18.0,
            duration_minutes=60,
            ambient_temp=15.0
        )

        assert ">20" in learning_engine.stats.temp_cooling_rates
        assert len(learning_engine.stats.temp_cooling_rates[">20"]) == 1

    def test_record_cooling_calculates_rate(self, learning_engine):
        """Cooling rate should be calculated correctly from exponential decay."""
        # 21°C -> 18°C in 60 min with ambient 15°C
        # rate = -ln((18-15)/(21-15)) / 60 = -ln(0.5) / 60 ≈ 0.0116
        learning_engine.record_cooling(
            temp_start=21.0,
            temp_end=18.0,
            duration_minutes=60,
            ambient_temp=15.0
        )

        rate = learning_engine.stats.temp_cooling_rates[">20"][0]
        assert 0.010 < rate < 0.013  # Should be around 0.0116

    def test_record_cooling_ignored_when_no_drop(self, learning_engine):
        """No cooling recorded if temperature didn't drop."""
        learning_engine.record_cooling(
            temp_start=21.0,
            temp_end=22.0,  # Temp increased
            duration_minutes=60
        )

        assert len(learning_engine.stats.temp_cooling_rates) == 0

    def test_record_cooling_ignored_below_ambient(self, learning_engine):
        """No cooling recorded if start temp at or below ambient."""
        learning_engine.record_cooling(
            temp_start=14.0,  # Below ambient
            temp_end=13.0,
            duration_minutes=60,
            ambient_temp=15.0
        )

        assert len(learning_engine.stats.temp_cooling_rates) == 0

    def test_record_cooling_ignored_short_duration(self, learning_engine):
        """No cooling recorded for very short durations."""
        learning_engine.record_cooling(
            temp_start=21.0,
            temp_end=20.0,
            duration_minutes=0.5  # Too short
        )

        assert len(learning_engine.stats.temp_cooling_rates) == 0

    def test_get_cooling_rate_returns_none_without_data(self, learning_engine):
        """Should return None when no cooling data available."""
        rate = learning_engine.get_cooling_rate(21.0)
        assert rate is None

    def test_get_cooling_rate_returns_median(self, learning_engine):
        """Should return median of last observations."""
        # Record multiple cooling observations
        for end_temp in [18.0, 17.5, 18.5, 17.8, 18.2]:
            learning_engine.record_cooling(
                temp_start=21.0,
                temp_end=end_temp,
                duration_minutes=60,
                ambient_temp=15.0
            )

        rate = learning_engine.get_cooling_rate(21.0)
        assert rate is not None
        assert 0.008 < rate < 0.016  # Should be in reasonable range

    def test_predict_temp_after_idle_uses_learned_rate(self, learning_engine):
        """Prediction should use learned rate when available."""
        # Record enough cooling observations
        for _ in range(5):
            learning_engine.record_cooling(
                temp_start=21.0,
                temp_end=18.0,
                duration_minutes=60,
                ambient_temp=15.0
            )

        # Predict with learned rate
        predicted = learning_engine.predict_temp_after_idle(21.0, 60, ambient_temp=15.0)

        # Should be around 18°C (matching the learned data)
        assert 17.5 < predicted < 18.5

    def test_predict_temp_after_idle_uses_default_when_no_data(self, learning_engine):
        """Prediction should use default rate when no learned data."""
        predicted = learning_engine.predict_temp_after_idle(
            21.0, 60, ambient_temp=15.0, default_cooling_rate=0.012
        )

        # With default rate 0.012, should cool to ~17.9°C
        assert 17.5 < predicted < 18.5

    def test_cooling_rates_in_learning_summary(self, learning_engine):
        """Learning summary should include cooling rates."""
        # First record ambient observations so cooling recording doesn't fail
        for temp in [15.0, 14.0, 15.5]:
            learning_engine.record_temperature_observation(temp)

        for _ in range(5):
            learning_engine.record_cooling(
                temp_start=21.0,
                temp_end=18.0,
                duration_minutes=60
            )

        summary = learning_engine.get_learning_summary()

        assert "temp_cooling_rates" in summary
        assert ">20" in summary["temp_cooling_rates"]
        assert "median_rate_per_min" in summary["temp_cooling_rates"][">20"]
        assert "observations" in summary["temp_cooling_rates"][">20"]


class TestAmbientTemperatureEstimation:
    """Test cases for ambient temperature estimation."""

    def test_record_temperature_observation(self, learning_engine):
        """Should record temperature observations."""
        learning_engine.record_temperature_observation(12.0)
        learning_engine.record_temperature_observation(15.0)

        assert len(learning_engine.stats.recent_min_temps) == 2
        assert 12.0 in learning_engine.stats.recent_min_temps
        assert 15.0 in learning_engine.stats.recent_min_temps

    def test_record_temperature_observation_limits_history(self, learning_engine):
        """Should limit history to a 48 h window (192 samples at 15 min).

        The window used to be 48 samples. Once observations became event driven
        that covered only a few hours, which can never contain a diurnal
        minimum — the value the ambient estimate is supposed to be.
        """
        for i in range(250):
            learning_engine.record_temperature_observation(10.0 + i * 0.1)

        assert len(learning_engine.stats.recent_min_temps) == 192

    def test_get_estimated_ambient_returns_minimum(self, learning_engine):
        """Should return minimum of recent observations as ambient."""
        learning_engine.record_temperature_observation(15.0)
        learning_engine.record_temperature_observation(12.0)
        learning_engine.record_temperature_observation(18.0)
        learning_engine.record_temperature_observation(10.0)
        learning_engine.record_temperature_observation(14.0)

        ambient = learning_engine.get_estimated_ambient_temp()

        assert ambient == 10.0

    def test_get_estimated_ambient_returns_default_when_empty(self, learning_engine):
        """Should return default when no observations."""
        ambient = learning_engine.get_estimated_ambient_temp(default=10.0)
        assert ambient == 10.0

        ambient = learning_engine.get_estimated_ambient_temp(default=15.0)
        assert ambient == 15.0

    def test_estimated_ambient_in_learning_summary(self, learning_engine):
        """Learning summary should include estimated ambient temperature."""
        learning_engine.record_temperature_observation(12.0)
        learning_engine.record_temperature_observation(10.0)

        summary = learning_engine.get_learning_summary()

        assert "estimated_ambient_temp" in summary
        assert summary["estimated_ambient_temp"] == 10.0

    def test_cooling_uses_estimated_ambient(self, learning_engine):
        """Cooling predictions should use estimated ambient, not hardcoded default."""
        # Set ambient to 8°C via observations
        for temp in [10.0, 8.0, 9.0]:
            learning_engine.record_temperature_observation(temp)

        # Predict cooling from 20°C - should cool toward 8°C
        predicted = learning_engine.predict_temp_after_idle(20.0, 60)

        # With ambient 8°C and default rate, should cool significantly
        # but still be well above 8°C after 1 hour
        assert predicted < 20.0
        assert predicted > 8.0

    def test_min_temp_alias_matches_new_name(self, learning_engine):
        """The renamed accessor and its backwards-compatible alias agree."""
        for temp in [15.0, 12.0, 18.0]:
            learning_engine.record_temperature_observation(temp)

        assert learning_engine.get_estimated_ambient_min_temp() == 12.0
        assert learning_engine.get_estimated_ambient_temp() == 12.0

    def test_has_ambient_observations(self, learning_engine):
        assert learning_engine.has_ambient_observations() is False
        learning_engine.record_temperature_observation(27.4)
        assert learning_engine.has_ambient_observations() is True


class TestThermalCalibration:
    """k1 / k2 calibration from raw thermal samples."""

    def test_calibrate_recovers_known_coefficients(self, learning_engine):
        """Least squares must recover synthetic coefficients without noise.

        The samples are generated by ``thermal_model.step_temperature`` — the
        model the coefficients are consumed by — NOT by its Euler
        linearisation, which is what the fit used to assume.
        """
        k1_true = 0.02      # per minute
        k2_true = 0.5       # C per kWh

        # Vary BOTH regressors so they are not collinear.
        samples = []
        for i in range(60):
            t_start = 25.0 + (i % 12)
            ambient = 20.0 + (i % 5)
            power = [0.0, 1.5, 3.0, 4.5, 5.9][i % 5]
            dt = 15.0
            t_end = step_temperature(
                start_temp=t_start,
                ambient_temp=ambient,
                duration_minutes=dt,
                battery_power_kw=power,
                cooling_rate_per_min=k1_true,
                heating_c_per_kwh=k2_true,
            )
            samples.append((t_start, t_end, dt, power, ambient))

        for t_start, t_end, dt, power, ambient in samples:
            assert learning_engine.record_thermal_observation(
                temp_start=t_start,
                temp_end=t_end,
                duration_minutes=dt,
                avg_power_kw=power,
                ambient_temp=ambient,
            )

        result = learning_engine.calibrate_thermal_coefficients()
        assert result is not None
        k1, k2 = result
        assert abs(k1 - k1_true) / k1_true < 0.01
        assert abs(k2 - k2_true) / k2_true < 0.01

        assert learning_engine.stats.thermal_coeffs["n"] == 60
        assert learning_engine.get_heating_coefficient() == pytest.approx(k2, rel=1e-6)
        assert learning_engine.get_cooling_rate_estimate(30.0) == pytest.approx(k1, rel=1e-6)

    def test_calibration_needs_enough_samples(self, learning_engine):
        for i in range(10):
            learning_engine.record_thermal_observation(
                temp_start=30.0, temp_end=30.5, duration_minutes=15.0,
                avg_power_kw=4.0, ambient_temp=25.0,
            )
        assert learning_engine.calibrate_thermal_coefficients() is None
        assert learning_engine.stats.thermal_coeffs == {}

    def test_collinear_samples_do_not_produce_coefficients(self, learning_engine):
        """All samples at the same power/delta cannot separate k1 from k2."""
        for _ in range(30):
            learning_engine.record_thermal_observation(
                temp_start=30.0, temp_end=30.2, duration_minutes=15.0,
                avg_power_kw=4.0, ambient_temp=25.0,
            )
        assert learning_engine.calibrate_thermal_coefficients() is None

    def test_bootstrap_k2_from_warming_rates(self, learning_engine_with_warming_data):
        """Already-collected charge warming rates seed k2 before calibration.

        The task assumed k2 could be fitted from the ~250 stored observations.
        It cannot: they carry no power. But their median C/min divided by the
        nominal charge rate is an honest first estimate, so the model is not
        stuck on the hardcoded default while new samples accumulate.
        """
        engine = learning_engine_with_warming_data
        engine.stats.thermal_samples = []
        engine.stats.thermal_coeffs = {}

        k2 = engine.get_heating_coefficient()

        # Fixture warms 10->14C and 16->20C over 60 min => 0.0667 C/min
        expected = 0.0667 / engine.nominal_charge_rate * 60.0
        assert k2 == pytest.approx(expected, rel=0.02)
        assert k2 > 0.0
        assert k2 != 0.35  # not the hardcoded default

    def test_default_k2_without_any_data(self, learning_engine):
        assert learning_engine.get_heating_coefficient() == pytest.approx(0.35)
        assert learning_engine.get_heating_coefficient(default=0.5) == pytest.approx(0.5)

    def test_record_discharging_captures_temps(self, learning_engine):
        """Discharge observations must now carry thermal data.

        Before the fix ``record_discharging`` had no temperature parameters at
        all, so the learning data contained ZERO discharge thermal samples and
        the heating coefficient was unfittable.
        """
        before = len(learning_engine.stats.thermal_samples)

        learning_engine.record_discharging(
            soc_start=80.0,
            soc_end=70.0,
            duration_minutes=30.0,
            energy_delivered_kwh=1.43,
            price_eur_kwh=0.1,
            battery_temp_start=30.0,
            battery_temp_end=32.0,
            ambient_temp=27.0,
        )

        assert len(learning_engine.stats.thermal_samples) == before + 1
        sample = learning_engine.stats.thermal_samples[-1]
        assert sample[0] == 30.0
        assert sample[1] == 32.0
        assert sample[2] == 30.0
        # 1.43 kWh over 0.5 h = 2.86 kW
        assert sample[3] == pytest.approx(2.86, abs=0.01)
        assert sample[4] == 27.0

    def test_record_discharging_without_temps_is_unchanged(self, learning_engine):
        learning_engine.record_discharging(
            soc_start=80.0, soc_end=70.0, duration_minutes=30.0
        )
        assert learning_engine.stats.thermal_samples == []

    def test_charging_also_feeds_thermal_samples(self, learning_engine):
        """Warming is a function of power, not of mode — charge samples count."""
        learning_engine.record_charging(
            soc_start=30.0, soc_end=50.0, duration_minutes=60.0,
            battery_temp=12.0, battery_temp_start=10.0, battery_temp_end=14.0,
        )
        assert len(learning_engine.stats.thermal_samples) == 1

    def test_invalid_thermal_samples_rejected(self, learning_engine):
        assert not learning_engine.record_thermal_observation(30.0, 31.0, 0.5, 4.0)
        assert not learning_engine.record_thermal_observation(30.0, 60.0, 15.0, 4.0)
        assert not learning_engine.record_thermal_observation(30.0, 31.0, 15.0, 25.0)
        assert learning_engine.stats.thermal_samples == []

    def test_thermal_samples_are_capped(self, learning_engine):
        for i in range(400):
            learning_engine.record_thermal_observation(
                temp_start=25.0 + (i % 10),
                temp_end=25.5 + (i % 10),
                duration_minutes=15.0,
                avg_power_kw=(i % 5),
                ambient_temp=20.0,
            )
        assert len(learning_engine.stats.thermal_samples) == 300

    def test_thermal_state_survives_json_round_trip(self, learning_engine):
        for i in range(30):
            learning_engine.record_thermal_observation(
                temp_start=25.0 + (i % 8),
                temp_end=25.4 + (i % 8),
                duration_minutes=15.0,
                avg_power_kw=[0.0, 2.0, 4.5][i % 3],
                ambient_temp=22.0,
            )
        json_str = learning_engine.save_to_json()

        restored = BatteryLearningEngine(battery_capacity_kwh=14.3)
        assert restored.load_from_json(json_str)
        assert len(restored.stats.thermal_samples) == 30
        assert restored.stats.thermal_coeffs == learning_engine.stats.thermal_coeffs

    def test_old_json_without_thermal_fields_still_loads(self, learning_engine):
        old_json = json.dumps({
            "version": 5,
            "learned_efficiency": 0.88,
            "stats": {"charge_rates_by_soc": {}, "total_energy_charged_kwh": 100.0},
        })
        assert learning_engine.load_from_json(old_json)
        assert learning_engine.stats.thermal_samples == []
        assert learning_engine.stats.thermal_coeffs == {}


class TestBoundedTemperatureProjection:
    """predict_temp_after_duration must not diverge."""

    def test_projection_is_capped(self, learning_engine):
        """132 slots x 15 min of linear warming used to reach ~230C."""
        engine = learning_engine
        for _ in range(5):
            engine.record_charging(
                soc_start=50.0, soc_end=52.0, duration_minutes=5.0,
                battery_temp=33.0, battery_temp_start=33.0, battery_temp_end=33.5,
            )

        temp = 33.0
        for _ in range(132):
            temp = engine.predict_temp_after_duration(temp, 15.0)
        assert temp <= 55.0

    def test_thermal_fallback_without_warming_data(self, learning_engine):
        """No learned warming rate -> thermal model, not a flat +0.1 C/min."""
        for t in [27.0, 28.0, 30.0]:
            learning_engine.record_temperature_observation(t)

        predicted = learning_engine.predict_temp_after_duration(33.0, 15.0)
        # Old model: 33 + 0.1*15 = 34.5 unconditionally.
        assert predicted < 34.5
        assert predicted > 27.0


class TestThermalCalibrationMatchesStepTemperature:
    """The fit must invert the model the coefficients are consumed by.

    ``step_temperature`` applies exponential relaxation
    ``Ta + (T0-Ta)*exp(-k1*dt)``; the calibration used to fit the Euler form
    ``(T_end-T_start)/dt = -k1*(T_start-Ta) + k2'*|P|``.  The two only agree as
    ``k1*dt -> 0``, so the recovered k1 was low by roughly ``k1*dt/2``: ~3 % at
    dt=5 min, 16 % at 30 min, 29 % at 60 min for k1=0.012/min.  Thermal samples
    are whole charge/discharge sessions, so 20-40 min intervals are the norm.
    """

    @staticmethod
    def _feed(engine, k1_true, k2_true, durations):
        for i in range(60):
            t_start = 24.0 + (i % 11)
            ambient = 18.0 + (i % 7)
            power = [0.0, 1.2, 2.7, 4.5, 5.9][i % 5]
            dt = durations[i % len(durations)]
            t_end = step_temperature(
                start_temp=t_start,
                ambient_temp=ambient,
                duration_minutes=dt,
                battery_power_kw=power,
                cooling_rate_per_min=k1_true,
                heating_c_per_kwh=k2_true,
            )
            assert engine.record_thermal_observation(
                temp_start=t_start,
                temp_end=t_end,
                duration_minutes=dt,
                avg_power_kw=power,
                ambient_temp=ambient,
            )

    @pytest.mark.parametrize("dt", [5.0, 30.0, 60.0])
    def test_recovers_exponential_coefficients_at_each_interval(
        self, learning_engine, dt
    ):
        """This FAILS on the Euler fit at dt=60 (k1 low by ~29 %)."""
        k1_true = 0.012
        k2_true = 0.42
        self._feed(learning_engine, k1_true, k2_true, [dt])

        result = learning_engine.calibrate_thermal_coefficients()
        assert result is not None
        k1, k2 = result
        assert abs(k1 - k1_true) / k1_true < 0.01, f"dt={dt}: k1={k1}"
        assert abs(k2 - k2_true) / k2_true < 0.01, f"dt={dt}: k2={k2}"

    def test_recovers_coefficients_from_mixed_interval_lengths(self, learning_engine):
        k1_true = 0.012
        k2_true = 0.42
        self._feed(learning_engine, k1_true, k2_true, [5.0, 30.0, 60.0])

        k1, k2 = learning_engine.calibrate_thermal_coefficients()
        assert abs(k1 - k1_true) / k1_true < 0.01
        assert abs(k2 - k2_true) / k2_true < 0.01

    def test_calibrated_coefficients_reproduce_step_temperature(self, learning_engine):
        """End-to-end: replaying the samples through the projector's own model."""
        k1_true = 0.02
        k2_true = 0.6
        self._feed(learning_engine, k1_true, k2_true, [12.0, 45.0, 60.0])
        k1, k2 = learning_engine.calibrate_thermal_coefficients()

        for t_start, t_end, dt, power, ambient in learning_engine.stats.thermal_samples:
            predicted = step_temperature(
                start_temp=t_start,
                ambient_temp=ambient,
                duration_minutes=dt,
                battery_power_kw=power,
                cooling_rate_per_min=k1,
                heating_c_per_kwh=k2,
            )
            assert predicted == pytest.approx(t_end, abs=0.02)


class TestThermalAmbientSourceIsShared:
    """Both recorders feed ONE pooled k1/k2 regression, so both need one ambient.

    ``record_discharging`` took a real ambient-service value while
    ``record_charging`` silently fell back to the rolling battery-temperature
    minimum, and the regression's x1 is ``-(t_start - ambient)``.  In summer the
    rolling minimum sits ~10 C above the real ambient, so two thermally
    identical samples entered the fit as x1=-3 (charge) and x1=-13 (discharge)
    — making k1 a proxy for the mode.
    """

    def test_record_charging_accepts_ambient(self, learning_engine):
        learning_engine.record_charging(
            soc_start=40, soc_end=55, duration_minutes=30,
            battery_temp_start=33.0, battery_temp_end=34.5,
            ambient_temp=20.0,
        )
        assert len(learning_engine.stats.thermal_samples) == 1
        assert learning_engine.stats.thermal_samples[0][4] == pytest.approx(20.0)

    def test_both_recorders_store_the_same_ambient(self, learning_engine):
        # A rolling battery-temperature window far above the real ambient is
        # exactly the summer situation that made the two disagree.
        for _ in range(20):
            learning_engine.record_temperature_observation(30.0)
        assert learning_engine.get_estimated_ambient_min_temp() == pytest.approx(30.0)

        learning_engine.record_charging(
            soc_start=40, soc_end=55, duration_minutes=30,
            battery_temp_start=33.0, battery_temp_end=34.5,
            ambient_temp=20.0,
        )
        learning_engine.record_discharging(
            soc_start=55, soc_end=40, duration_minutes=30,
            battery_temp_start=33.0, battery_temp_end=34.5,
            ambient_temp=20.0,
        )

        charge_sample, discharge_sample = learning_engine.stats.thermal_samples
        assert charge_sample[4] == discharge_sample[4] == pytest.approx(20.0)
        # Identical physics -> identical sample, mode included.
        assert charge_sample == discharge_sample

    def test_omitted_ambient_still_falls_back_for_both(self, learning_engine):
        for _ in range(20):
            learning_engine.record_temperature_observation(30.0)

        learning_engine.record_charging(
            soc_start=40, soc_end=55, duration_minutes=30,
            battery_temp_start=33.0, battery_temp_end=34.5,
        )
        learning_engine.record_discharging(
            soc_start=55, soc_end=40, duration_minutes=30,
            battery_temp_start=33.0, battery_temp_end=34.5,
        )
        charge_sample, discharge_sample = learning_engine.stats.thermal_samples
        assert charge_sample[4] == discharge_sample[4] == pytest.approx(30.0)
