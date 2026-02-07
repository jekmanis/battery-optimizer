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

    def test_load_json_migrates_coarser_to_finer_slots(self, load_profile):
        """Loading from coarser to finer divisible slots should migrate."""
        # Record some data in 60-minute profile (slot 10 = hour 10)
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        load_profile.record(dt, 500.0)
        json_str = load_profile.to_json()

        # Load into 30-minute profile (factor=2)
        profile_30 = LoadProfile(slot_minutes=30, default_load_w=500.0)
        success = profile_30.load_from_json(json_str)

        assert success is True
        # Old slot 10 should be split into new slots 20 and 21
        assert "20" in profile_30.stats.samples_by_slot
        assert "21" in profile_30.stats.samples_by_slot
        assert profile_30.stats.samples_by_slot["20"] == [500.0]
        assert profile_30.stats.samples_by_slot["21"] == [500.0]

    def test_load_json_rejects_incompatible_slot_size(self, load_profile):
        """Loading from finer to coarser slots should fail (cannot migrate)."""
        # Save with 60-minute slots
        json_str = load_profile.to_json()

        # Try to load into 120-minute profile (60 < 120, so not migrateable)
        profile_120 = LoadProfile(slot_minutes=120, default_load_w=500.0)
        success = profile_120.load_from_json(json_str)

        assert success is False

    def test_load_json_rejects_non_divisible_slot_size(self):
        """Loading from non-divisible coarser slots should fail."""
        # Create and save a 45-minute profile
        profile_45 = LoadProfile(slot_minutes=45, default_load_w=500.0)
        json_str = profile_45.to_json()

        # Try to load into 20-minute profile (45 > 20 but 45 % 20 != 0)
        profile_20 = LoadProfile(slot_minutes=20, default_load_w=500.0)
        success = profile_20.load_from_json(json_str)

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


class TestFifteenMinuteSlotLoadProfile:
    """Test cases for LoadProfile with 15-minute slots."""

    def test_slot_index_15min_slots(self):
        """Verify slot index mapping for 15-minute slots."""
        profile = LoadProfile(slot_minutes=15, default_load_w=500)

        # Verify slots_per_day
        assert profile.slots_per_day == 96

        # Test specific slot indices
        assert profile._slot_index(datetime.datetime(2024, 1, 15, 0, 0, 0)) == 0
        assert profile._slot_index(datetime.datetime(2024, 1, 15, 0, 15, 0)) == 1
        assert profile._slot_index(datetime.datetime(2024, 1, 15, 0, 30, 0)) == 2
        assert profile._slot_index(datetime.datetime(2024, 1, 15, 0, 45, 0)) == 3
        assert profile._slot_index(datetime.datetime(2024, 1, 15, 1, 0, 0)) == 4
        assert profile._slot_index(datetime.datetime(2024, 1, 15, 12, 0, 0)) == 48
        assert profile._slot_index(datetime.datetime(2024, 1, 15, 23, 45, 0)) == 95

    def test_15min_record_and_predict(self):
        """Record observations at 15-min boundaries and verify predictions."""
        profile = LoadProfile(
            slot_minutes=15,
            default_load_w=500.0,
            min_samples=3,
        )

        # Record observations at 15-min boundaries
        base = datetime.datetime(2024, 1, 15, 10, 0, 0)
        for day in range(5):
            dt_slot0 = base + datetime.timedelta(days=day)
            profile.record(dt_slot0, 800.0 + day * 10)  # 10:00 slot

            dt_slot1 = base + datetime.timedelta(days=day, minutes=15)
            profile.record(dt_slot1, 600.0 + day * 10)  # 10:15 slot

            dt_slot2 = base + datetime.timedelta(days=day, minutes=30)
            profile.record(dt_slot2, 400.0 + day * 10)  # 10:30 slot

        # Predict for each slot and verify they differ
        pred_1000 = profile.predict_kw(
            datetime.datetime(2024, 1, 20, 10, 0, 0), quantile=0.5
        )
        pred_1015 = profile.predict_kw(
            datetime.datetime(2024, 1, 20, 10, 15, 0), quantile=0.5
        )
        pred_1030 = profile.predict_kw(
            datetime.datetime(2024, 1, 20, 10, 30, 0), quantile=0.5
        )

        # Predictions should reflect the recorded data pattern
        # 10:00 had highest load, 10:30 had lowest
        assert pred_1000 > pred_1015, (
            f"10:00 prediction ({pred_1000:.3f}) should exceed 10:15 ({pred_1015:.3f})"
        )
        assert pred_1015 > pred_1030, (
            f"10:15 prediction ({pred_1015:.3f}) should exceed 10:30 ({pred_1030:.3f})"
        )

    def test_migrate_30min_to_15min(self):
        """Migrating from 30-min slots to 15-min slots should split data."""
        profile = LoadProfile(slot_minutes=15, default_load_w=500.0)

        # Create JSON data with 30-min slots
        json_data = json.dumps({
            "version": 1,
            "slot_minutes": 30,
            "stats": {
                "samples_by_slot": {
                    "0": [300.0, 350.0],
                    "1": [400.0],
                },
                "observation_count": 3,
                "last_observation": "2024-01-15T01:00:00",
            },
        })

        success = profile.load_from_json(json_data)
        assert success is True

        # Old slot "0" (00:00-00:30) -> new slots "0" (00:00-00:15) and "1" (00:15-00:30)
        assert "0" in profile.stats.samples_by_slot
        assert "1" in profile.stats.samples_by_slot
        assert profile.stats.samples_by_slot["0"] == [300.0, 350.0]
        assert profile.stats.samples_by_slot["1"] == [300.0, 350.0]

        # Old slot "1" (00:30-01:00) -> new slots "2" (00:30-00:45) and "3" (00:45-01:00)
        assert "2" in profile.stats.samples_by_slot
        assert "3" in profile.stats.samples_by_slot
        assert profile.stats.samples_by_slot["2"] == [400.0]
        assert profile.stats.samples_by_slot["3"] == [400.0]

    def test_migrate_60min_to_15min(self):
        """Migrating from 60-min slots to 15-min slots (factor=4)."""
        profile = LoadProfile(slot_minutes=15, default_load_w=500.0)

        json_data = json.dumps({
            "version": 1,
            "slot_minutes": 60,
            "stats": {
                "samples_by_slot": {
                    "0": [250.0, 275.0],   # Hour 0 (00:00-01:00)
                    "12": [900.0, 950.0],  # Hour 12 (12:00-13:00)
                },
                "observation_count": 4,
                "last_observation": "2024-01-15T12:00:00",
            },
        })

        success = profile.load_from_json(json_data)
        assert success is True

        # Old slot "0" (hour 0) -> new slots "0", "1", "2", "3" (factor=4)
        for new_slot in ["0", "1", "2", "3"]:
            assert new_slot in profile.stats.samples_by_slot, (
                f"Slot {new_slot} should exist after migration"
            )
            assert profile.stats.samples_by_slot[new_slot] == [250.0, 275.0]

        # Old slot "12" (hour 12) -> new slots "48", "49", "50", "51"
        for new_slot in ["48", "49", "50", "51"]:
            assert new_slot in profile.stats.samples_by_slot, (
                f"Slot {new_slot} should exist after migration"
            )
            assert profile.stats.samples_by_slot[new_slot] == [900.0, 950.0]

    def test_migrate_incompatible_rejected(self):
        """Migrating from 20-min to 15-min slots should fail (20 % 15 != 0)."""
        profile = LoadProfile(slot_minutes=15, default_load_w=500.0)

        json_data = json.dumps({
            "version": 1,
            "slot_minutes": 20,
            "stats": {
                "samples_by_slot": {
                    "0": [100.0],
                },
                "observation_count": 1,
                "last_observation": "2024-01-15T00:00:00",
            },
        })

        success = profile.load_from_json(json_data)
        assert success is False
