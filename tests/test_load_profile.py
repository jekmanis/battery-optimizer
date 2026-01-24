"""
Tests for LoadProfile class.
"""

import datetime
import json

import pytest

from battery_optimizer import LoadProfile, LoadProfileStats


class TestLoadProfileStats:
    """Test cases for LoadProfileStats dataclass."""

    def test_default_initialization(self):
        """Default values should be set correctly."""
        stats = LoadProfileStats()
        assert stats.samples_by_slot == {}
        assert stats.observation_count == 0
        assert stats.last_observation is None

    def test_to_dict_roundtrip(self):
        """Data should survive dict serialization."""
        stats = LoadProfileStats(
            samples_by_slot={"5": [100.0, 150.0], "10": [200.0]},
            observation_count=3,
            last_observation="2024-01-15T12:00:00",
        )

        data = stats.to_dict()
        restored = LoadProfileStats.from_dict(data)

        assert restored.samples_by_slot == {"5": [100.0, 150.0], "10": [200.0]}
        assert restored.observation_count == 3
        assert restored.last_observation == "2024-01-15T12:00:00"


class TestLoadProfile:
    """Test cases for LoadProfile class."""

    def test_initialization(self, load_profile):
        """LoadProfile should initialize correctly."""
        assert load_profile.slot_minutes == 60
        assert load_profile.slots_per_day == 24
        assert load_profile.default_load_w == 500.0
        assert load_profile.max_samples == 60
        assert load_profile.min_samples == 6

    def test_initialization_different_slot_size(self):
        """Different slot sizes should calculate slots_per_day correctly."""
        profile_30min = LoadProfile(slot_minutes=30, default_load_w=500.0)
        assert profile_30min.slots_per_day == 48

        profile_15min = LoadProfile(slot_minutes=15, default_load_w=500.0)
        assert profile_15min.slots_per_day == 96

    def test_initialization_minimum_slot_size(self):
        """Slot size should be at least 1 minute."""
        profile = LoadProfile(slot_minutes=0, default_load_w=500.0)
        assert profile.slot_minutes == 1

        profile_neg = LoadProfile(slot_minutes=-5, default_load_w=500.0)
        assert profile_neg.slot_minutes == 1

    def test_slot_index_mapping(self, load_profile):
        """Slot index should map correctly for hourly slots."""
        dt_midnight = datetime.datetime(2024, 1, 15, 0, 0, 0)
        assert load_profile._slot_index(dt_midnight) == 0

        dt_noon = datetime.datetime(2024, 1, 15, 12, 0, 0)
        assert load_profile._slot_index(dt_noon) == 12

        dt_11pm = datetime.datetime(2024, 1, 15, 23, 0, 0)
        assert load_profile._slot_index(dt_11pm) == 23

    def test_slot_index_30min_slots(self):
        """Slot index for 30-minute slots."""
        profile = LoadProfile(slot_minutes=30, default_load_w=500.0)

        dt = datetime.datetime(2024, 1, 15, 0, 0, 0)
        assert profile._slot_index(dt) == 0

        dt = datetime.datetime(2024, 1, 15, 0, 30, 0)
        assert profile._slot_index(dt) == 1

        dt = datetime.datetime(2024, 1, 15, 12, 0, 0)
        assert profile._slot_index(dt) == 24

    def test_record_basic(self, load_profile):
        """Recording should store samples."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        load_profile.record(dt, 600.0)

        assert "10" in load_profile.stats.samples_by_slot
        assert 600.0 in load_profile.stats.samples_by_slot["10"]
        assert load_profile.stats.observation_count == 1
        assert load_profile.stats.last_observation is not None

    def test_record_multiple_same_slot(self, load_profile):
        """Multiple recordings in same slot should accumulate."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        load_profile.record(dt, 500.0)
        load_profile.record(dt, 600.0)
        load_profile.record(dt, 700.0)

        samples = load_profile.stats.samples_by_slot["10"]
        assert len(samples) == 3
        assert samples == [500.0, 600.0, 700.0]

    def test_record_zero_or_negative_ignored(self, load_profile):
        """Zero or negative load should be ignored."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        load_profile.record(dt, 0)
        load_profile.record(dt, -100)

        assert "10" not in load_profile.stats.samples_by_slot

    def test_record_limits_samples(self, load_profile):
        """Should keep only last max_samples per slot."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        # Record more than max_samples (60)
        for i in range(70):
            load_profile.record(dt, float(i))

        samples = load_profile.stats.samples_by_slot["10"]
        assert len(samples) == 60
        # Should be the last 60 values
        assert samples[0] == 10.0  # First kept
        assert samples[-1] == 69.0  # Last recorded

    def test_predict_no_data_returns_default(self, load_profile):
        """No data should return default load."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        prediction = load_profile.predict_kw(dt)

        # Default is 500W = 0.5 kW
        assert prediction == 0.5

    def test_predict_with_data(self, load_profile_with_data):
        """With data, should return quantile-based prediction."""
        profile = load_profile_with_data

        # Morning (hour 6) - recorded ~300W
        dt_morning = datetime.datetime(2024, 1, 15, 6, 0, 0)
        pred_morning = profile.predict_kw(dt_morning, quantile=0.75)

        # Evening (hour 18) - recorded ~1200W
        dt_evening = datetime.datetime(2024, 1, 15, 18, 0, 0)
        pred_evening = profile.predict_kw(dt_evening, quantile=0.75)

        # Evening should be higher than morning
        assert pred_evening > pred_morning

    def test_predict_quantile_effect(self, load_profile):
        """Higher quantile should give higher prediction."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        # Add varied samples
        for val in [100, 200, 300, 400, 500, 600, 700]:
            load_profile.record(dt, float(val))

        pred_25 = load_profile.predict_kw(dt, quantile=0.25)
        pred_50 = load_profile.predict_kw(dt, quantile=0.50)
        pred_75 = load_profile.predict_kw(dt, quantile=0.75)

        assert pred_25 < pred_50 < pred_75

    def test_predict_confidence_blending(self, load_profile):
        """Low sample count should blend with default."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        # Add just 2 samples (below min_samples=6)
        load_profile.record(dt, 1000.0)
        load_profile.record(dt, 1000.0)

        pred = load_profile.predict_kw(dt, quantile=0.5)

        # Should be blended between 1000W and 500W default
        # Confidence = 2/6 = 0.33
        # Expected: 500 * 0.67 + 1000 * 0.33 ≈ 667W = 0.667 kW
        assert 0.5 < pred < 1.0

    def test_predict_full_confidence(self, load_profile):
        """At min_samples, should use full sample data."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        # Add exactly min_samples
        for _ in range(6):
            load_profile.record(dt, 1000.0)

        pred = load_profile.predict_kw(dt, quantile=0.5)

        # Should be close to 1.0 kW (1000W)
        assert 0.9 < pred <= 1.0

    def test_predict_never_negative(self, load_profile):
        """Prediction should never be negative."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        # Even with weird data, should be non-negative
        load_profile.stats.samples_by_slot["10"] = [-100.0, -200.0]

        pred = load_profile.predict_kw(dt)
        assert pred >= 0

    def test_json_roundtrip(self, load_profile_with_data):
        """JSON serialization should preserve state."""
        profile = load_profile_with_data

        json_str = profile.to_json()
        assert json_str is not None

        # Verify valid JSON
        data = json.loads(json_str)
        assert data["version"] == 1
        assert data["slot_minutes"] == 60

        # Create new profile and load
        new_profile = LoadProfile(slot_minutes=60, default_load_w=500.0)
        success = new_profile.load_from_json(json_str)

        assert success
        assert new_profile.stats.observation_count == profile.stats.observation_count

    def test_load_json_different_slot_size(self, load_profile):
        """Loading with different slot size should fail."""
        # Save with 60-minute slots
        json_str = load_profile.to_json()

        # Try to load into 30-minute profile
        profile_30 = LoadProfile(slot_minutes=30, default_load_w=500.0)
        success = profile_30.load_from_json(json_str)

        assert success is False

    def test_load_json_invalid(self, load_profile):
        """Invalid JSON should return False."""
        assert load_profile.load_from_json("not json") is False
        assert load_profile.load_from_json("{}") is False
        assert load_profile.load_from_json('{"version": 0}') is False

    def test_different_days_same_slot(self, load_profile):
        """Different days should map to same slot."""
        day1 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        day2 = datetime.datetime(2024, 1, 16, 10, 0, 0)

        load_profile.record(day1, 500.0)
        load_profile.record(day2, 600.0)

        # Both should be in slot 10
        samples = load_profile.stats.samples_by_slot["10"]
        assert 500.0 in samples
        assert 600.0 in samples
