"""
Tests for LoadPredictionTracker.
"""

import datetime
import json

import pytest

from battery_optimizer_lib import (
    BatteryMode,
    PredictionAccuracyStats,
    ScheduleEntry,
)
from battery_optimizer_lib.load_prediction_tracker import LoadPredictionTracker, _median


class TestMedian:
    """Test the _median helper."""

    def test_empty(self):
        assert _median([]) == 0.0

    def test_single(self):
        assert _median([5.0]) == 5.0

    def test_odd(self):
        assert _median([1.0, 3.0, 2.0]) == 2.0

    def test_even(self):
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5


class TestPredictionAccuracyStats:
    """Test PredictionAccuracyStats dataclass."""

    def test_default(self):
        stats = PredictionAccuracyStats()
        assert stats.ratios_by_slot == {}
        assert stats.global_ratios == []
        assert stats.total_comparisons == 0

    def test_to_dict_roundtrip(self):
        stats = PredictionAccuracyStats(
            ratios_by_slot={"10": [1.5, 2.0]},
            global_ratios=[1.5, 2.0],
            total_comparisons=2,
        )
        d = stats.to_dict()
        restored = PredictionAccuracyStats.from_dict(d)
        assert restored.ratios_by_slot == {"10": [1.5, 2.0]}
        assert restored.total_comparisons == 2

    def test_from_dict_ignores_unknown_fields(self):
        d = {
            "ratios_by_slot": {},
            "global_ratios": [],
            "total_comparisons": 0,
            "last_comparison": None,
            "future_field": "ignored",
        }
        stats = PredictionAccuracyStats.from_dict(d)
        assert stats.total_comparisons == 0


class TestLoadPredictionTracker:
    """Test LoadPredictionTracker."""

    @pytest.fixture
    def tracker(self):
        return LoadPredictionTracker(slot_minutes=15, log_func=lambda msg: None)

    def test_no_data_returns_factor_1(self, tracker):
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        assert tracker.get_correction_factor(dt) == 1.0

    def test_basic_record_retrieve_cycle(self, tracker):
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        # Record prediction, then actual
        tracker.record_prediction(dt, 0.5)
        tracker.record_actual(dt, 1.0)

        assert tracker.stats.total_comparisons == 1
        # Ratio = 1.0 / 0.5 = 2.0
        assert tracker.stats.global_ratios == [2.0]

    def test_correction_factor_with_per_slot_data(self, tracker):
        """With enough per-slot data, correction factor uses slot median."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        # Record 6 observations (>= MIN_SLOT_SAMPLES=5) with ratio ~2.0
        for day in range(6):
            slot_dt = dt + datetime.timedelta(days=day)
            tracker.record_prediction(slot_dt, 0.5)
            tracker.record_actual(slot_dt, 1.0)

        factor = tracker.get_correction_factor(dt)
        assert factor == 2.0

    def test_correction_factor_adjacent_slot_fallback(self, tracker):
        """With few per-slot samples, falls back to adjacent slots."""
        dt_1000 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        dt_1015 = datetime.datetime(2024, 1, 15, 10, 15, 0)

        # Record 3 samples each for 10:00 and 10:15 (below per-slot threshold
        # but combined = 6 >= MIN_SLOT_SAMPLES)
        for day in range(3):
            t1 = dt_1000 + datetime.timedelta(days=day)
            tracker.record_prediction(t1, 0.5)
            tracker.record_actual(t1, 1.5)  # ratio = 3.0

            t2 = dt_1015 + datetime.timedelta(days=day)
            tracker.record_prediction(t2, 0.5)
            tracker.record_actual(t2, 1.5)  # ratio = 3.0

        # 10:00 has 3 per-slot + 3 from adjacent 10:15 = 6, use adjacent fallback
        factor = tracker.get_correction_factor(dt_1000)
        assert factor == 3.0

    def test_correction_factor_global_fallback(self, tracker):
        """With no per-slot or adjacent data, falls back to global."""
        # Put data in a completely different slot
        dt_other = datetime.datetime(2024, 1, 15, 22, 0, 0)
        for day in range(3):
            t = dt_other + datetime.timedelta(days=day)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 1.0)

        # Query a slot with no data nearby
        dt_query = datetime.datetime(2024, 1, 15, 6, 0, 0)
        factor = tracker.get_correction_factor(dt_query)
        # Global median of [2.0, 2.0, 2.0] = 2.0
        assert factor == 2.0

    def test_ratio_clamping(self, tracker):
        """Ratios should be clamped to [0.1, 10.0]."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        # Extreme under-prediction: actual 20x higher
        tracker.record_prediction(dt, 0.1)
        tracker.record_actual(dt, 5.0)
        assert tracker.stats.global_ratios[-1] == 10.0  # clamped

        # Extreme over-prediction: actual 0.005x
        dt2 = datetime.datetime(2024, 1, 15, 10, 15, 0)
        tracker.record_prediction(dt2, 10.0)
        tracker.record_actual(dt2, 0.05)
        assert tracker.stats.global_ratios[-1] == 0.1  # clamped

    def test_rolling_window_eviction_per_slot(self, tracker):
        """Per-slot ratios should not exceed MAX_SAMPLES_PER_SLOT."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        for day in range(40):
            t = dt + datetime.timedelta(days=day)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 1.0)

        slot_key = tracker._slot_key(dt)
        assert len(tracker.stats.ratios_by_slot[slot_key]) == 30

    def test_rolling_window_eviction_global(self, tracker):
        """Global ratios should not exceed MAX_GLOBAL_SAMPLES."""
        dt = datetime.datetime(2024, 1, 15, 0, 0, 0)

        for i in range(250):
            t = dt + datetime.timedelta(minutes=15 * i)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 1.0)

        assert len(tracker.stats.global_ratios) == 200

    def test_zero_actual_ignored(self, tracker):
        """Zero actual load should not create a ratio."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        tracker.record_prediction(dt, 0.5)
        tracker.record_actual(dt, 0.0)
        assert tracker.stats.total_comparisons == 0

    def test_zero_prediction_ignored(self, tracker):
        """Zero or negative prediction should not be stored."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        tracker.record_prediction(dt, 0.0)
        assert tracker._pending == {}

        tracker.record_prediction(dt, -1.0)
        assert tracker._pending == {}

    def test_no_pending_prediction_no_ratio(self, tracker):
        """Actual without a pending prediction should be ignored."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        tracker.record_actual(dt, 1.0)
        assert tracker.stats.total_comparisons == 0


class TestRiskMetrics:
    """Test risk metrics computation."""

    @pytest.fixture
    def tracker(self):
        return LoadPredictionTracker(slot_minutes=15, log_func=lambda msg: None)

    def test_empty_risk_metrics(self, tracker):
        metrics = tracker.get_risk_metrics()
        assert metrics["overall_bias"] == 1.0
        assert metrics["underestimate_pct"] == 0.0
        assert metrics["confidence"] == 0.0
        assert metrics["worst_slot"] is None

    def test_risk_metrics_bias(self, tracker):
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        # All predictions underestimate by 2x
        for day in range(10):
            t = dt + datetime.timedelta(days=day)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 1.0)

        metrics = tracker.get_risk_metrics()
        assert metrics["overall_bias"] == 2.0
        assert metrics["underestimate_pct"] == 100.0
        assert metrics["p90_ratio"] == 2.0
        assert metrics["confidence"] == 10 / 50  # 0.2

    def test_risk_metrics_mixed(self, tracker):
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        # Half underestimate, half overestimate
        for day in range(5):
            t = dt + datetime.timedelta(days=day)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 1.0)  # ratio 2.0 (underestimate)

        dt2 = datetime.datetime(2024, 1, 15, 12, 0, 0)
        for day in range(5):
            t = dt2 + datetime.timedelta(days=day)
            tracker.record_prediction(t, 1.0)
            tracker.record_actual(t, 0.5)  # ratio 0.5 (overestimate)

        metrics = tracker.get_risk_metrics()
        assert metrics["underestimate_pct"] == 50.0

    def test_risk_metrics_worst_slot(self, tracker):
        # Slot A: mild underestimate
        dt_a = datetime.datetime(2024, 1, 15, 8, 0, 0)
        for day in range(5):
            t = dt_a + datetime.timedelta(days=day)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 0.6)  # ratio = 1.2

        # Slot B: severe underestimate
        dt_b = datetime.datetime(2024, 1, 15, 22, 0, 0)
        for day in range(5):
            t = dt_b + datetime.timedelta(days=day)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 1.5)  # ratio = 3.0

        metrics = tracker.get_risk_metrics()
        assert metrics["worst_slot"] == "22:00"
        assert metrics["worst_slot_ratio"] == 3.0

    def test_full_confidence(self, tracker):
        dt = datetime.datetime(2024, 1, 15, 0, 0, 0)
        for i in range(60):
            t = dt + datetime.timedelta(minutes=15 * i)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 0.5)

        metrics = tracker.get_risk_metrics()
        assert metrics["confidence"] == 1.0


class TestScheduleRiskAssessment:
    """Test schedule risk assessment."""

    @pytest.fixture
    def tracker(self):
        return LoadPredictionTracker(slot_minutes=60, log_func=lambda msg: None)

    def test_no_data_no_risk(self, tracker):
        schedule = {
            datetime.datetime(2024, 1, 15, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="test",
            )
        }
        risk = tracker.get_schedule_risk_assessment(schedule)
        assert risk["overall_risk"] == "low"
        assert risk["discharge_slot_risks"] == {}

    def test_high_risk_discharge(self, tracker):
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        # All historical data shows underestimate
        for day in range(5):
            t = dt + datetime.timedelta(days=day)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 1.0)

        schedule = {
            dt: ScheduleEntry(time=dt, mode=BatteryMode.DISCHARGE, reason="test"),
        }
        risk = tracker.get_schedule_risk_assessment(schedule)
        assert risk["discharge_slot_risks"]["10:00"] == 100.0
        assert risk["overall_risk"] == "high"

    def test_ignores_non_discharge_slots(self, tracker):
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        for day in range(5):
            t = dt + datetime.timedelta(days=day)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 1.0)

        schedule = {
            dt: ScheduleEntry(time=dt, mode=BatteryMode.HOLD, reason="test"),
        }
        risk = tracker.get_schedule_risk_assessment(schedule)
        assert risk["discharge_slot_risks"] == {}
        assert risk["overall_risk"] == "low"


class TestPersistence:
    """Test JSON persistence."""

    @pytest.fixture
    def tracker(self):
        return LoadPredictionTracker(slot_minutes=15, log_func=lambda msg: None)

    def test_json_roundtrip(self, tracker):
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        for day in range(5):
            t = dt + datetime.timedelta(days=day)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 1.0)

        json_str = tracker.to_json()
        data = json.loads(json_str)
        assert data["version"] == 1
        assert data["slot_minutes"] == 15

        new_tracker = LoadPredictionTracker(slot_minutes=15, log_func=lambda msg: None)
        success = new_tracker.load_from_json(json_str)

        assert success
        assert new_tracker.stats.total_comparisons == 5
        assert len(new_tracker.stats.global_ratios) == 5

    def test_json_slot_mismatch_rejected(self, tracker):
        json_str = tracker.to_json()

        new_tracker = LoadPredictionTracker(slot_minutes=30, log_func=lambda msg: None)
        success = new_tracker.load_from_json(json_str)
        assert success is False

    def test_json_invalid(self, tracker):
        assert tracker.load_from_json("not json") is False
        assert tracker.load_from_json("{}") is False
        assert tracker.load_from_json('{"version": 0}') is False

    def test_json_empty_tracker(self, tracker):
        """Empty tracker should serialize and deserialize cleanly."""
        json_str = tracker.to_json()
        new_tracker = LoadPredictionTracker(slot_minutes=15, log_func=lambda msg: None)
        assert new_tracker.load_from_json(json_str)
        assert new_tracker.stats.total_comparisons == 0


class TestRealisticScenario:
    """Test a realistic scenario matching the Feb 13 incident."""

    def test_night_underestimation_correction(self):
        """Predictions at 0.5kW, actuals at 1.5kW -> correction ~3.0."""
        tracker = LoadPredictionTracker(slot_minutes=15, log_func=lambda msg: None)

        # Simulate a week of night observations (21:00-06:00)
        for day in range(7):
            for hour in [21, 22, 23, 0, 1, 2, 3, 4, 5]:
                dt = datetime.datetime(2024, 1, 15 + day, hour, 0, 0)
                tracker.record_prediction(dt, 0.5)
                tracker.record_actual(dt, 1.5)

        # Check correction at a night slot
        dt_night = datetime.datetime(2024, 1, 22, 22, 0, 0)
        factor = tracker.get_correction_factor(dt_night)
        assert factor == pytest.approx(3.0)

        # Risk metrics should show systematic underestimation
        metrics = tracker.get_risk_metrics()
        assert metrics["overall_bias"] == 3.0
        assert metrics["underestimate_pct"] == 100.0

    def test_self_correcting_loop(self):
        """After correction improves predictions, factor naturally decays."""
        tracker = LoadPredictionTracker(slot_minutes=60, log_func=lambda msg: None)

        dt = datetime.datetime(2024, 1, 15, 22, 0, 0)

        # Phase 1: bad predictions (ratio = 3.0)
        for day in range(6):
            t = dt + datetime.timedelta(days=day)
            tracker.record_prediction(t, 0.5)
            tracker.record_actual(t, 1.5)

        factor1 = tracker.get_correction_factor(dt)
        assert factor1 == 3.0

        # Phase 2: correction applied, predictions now accurate (ratio ~1.0)
        for day in range(6, 12):
            t = dt + datetime.timedelta(days=day)
            tracker.record_prediction(t, 1.5)  # corrected prediction
            tracker.record_actual(t, 1.5)

        factor2 = tracker.get_correction_factor(dt)
        # Median of [3,3,3,3,3,3, 1,1,1,1,1,1] = 2.0
        assert factor2 < factor1

        # Phase 3: more accurate data pushes out old ratios
        for day in range(12, 42):
            t = dt + datetime.timedelta(days=day)
            tracker.record_prediction(t, 1.5)
            tracker.record_actual(t, 1.5)

        factor3 = tracker.get_correction_factor(dt)
        # After 30 more good predictions, rolling window only has ratio=1.0
        assert factor3 == 1.0
