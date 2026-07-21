"""Unit tests for SlotOutcomeTracker module."""

import datetime
import sys
sys.path.insert(0, "appdaemon/apps")

from battery_optimizer_lib.slot_outcome_tracker import SlotOutcomeTracker


class TestSlotOutcomeTracker:
    def test_record_start_and_end(self):
        tracker = SlotOutcomeTracker(slot_minutes=15)
        dt = datetime.datetime(2024, 7, 15, 12, 0, 0)
        tracker.record_slot_start(
            slot_time=dt, mode="DISCHARGE", predicted_soc_end=45.0,
            predicted_load_kw=0.5, predicted_pv_kw=0.0,
        )
        tracker.record_slot_end(actual_soc=44.0, actual_pv_w=0.0, actual_mode="Discharge to Load")
        outcomes = tracker.get_recent_outcomes(10)
        assert len(outcomes) == 1
        assert outcomes[0]["scheduled_mode"] == "DISCHARGE"
        assert outcomes[0]["soc_error"] == -1.0  # 44 - 45

    def test_pending_finalized_on_next_start(self):
        tracker = SlotOutcomeTracker(slot_minutes=15)
        dt1 = datetime.datetime(2024, 7, 15, 12, 0, 0)
        dt2 = datetime.datetime(2024, 7, 15, 12, 15, 0)
        # First slot — no end recorded
        tracker.record_slot_start(
            slot_time=dt1, mode="HOLD", predicted_soc_end=50.0,
            predicted_load_kw=0.5,
        )
        # Second slot — should finalize first
        tracker.record_slot_start(
            slot_time=dt2, mode="CHARGE", predicted_soc_end=55.0,
            predicted_load_kw=0.5,
        )
        outcomes = tracker.get_recent_outcomes(10)
        assert len(outcomes) == 1  # First was finalized
        assert outcomes[0]["scheduled_mode"] == "HOLD"

    def test_rolling_window(self):
        tracker = SlotOutcomeTracker(slot_minutes=15, max_outcomes=5)
        base = datetime.datetime(2024, 7, 15, 12, 0, 0)
        for i in range(10):
            dt = base + datetime.timedelta(minutes=i * 15)
            tracker.record_slot_start(
                slot_time=dt, mode="HOLD", predicted_soc_end=50.0,
                predicted_load_kw=0.5,
            )
            tracker.record_slot_end(actual_soc=50.0)
        outcomes = tracker.get_recent_outcomes(100)
        assert len(outcomes) == 5

    def test_accuracy_stats_empty(self):
        tracker = SlotOutcomeTracker()
        stats = tracker.get_accuracy_stats()
        assert stats["outcome_count"] == 0
        assert stats["soc_mae"] is None

    def test_accuracy_stats_soc_mae(self):
        tracker = SlotOutcomeTracker()
        for i, error in enumerate([1.0, -2.0, 3.0]):
            dt = datetime.datetime(2024, 7, 15, 12, i * 15, 0)
            tracker.record_slot_start(
                slot_time=dt, mode="HOLD", predicted_soc_end=50.0,
                predicted_load_kw=0.5,
            )
            tracker.record_slot_end(actual_soc=50.0 + error)
        stats = tracker.get_accuracy_stats()
        assert stats["outcome_count"] == 3
        assert abs(stats["soc_mae"] - 2.0) < 0.01  # Mean of |1|, |2|, |3| = 2.0

    def test_pv_error_tracking(self):
        tracker = SlotOutcomeTracker()
        dt = datetime.datetime(2024, 7, 15, 12, 0, 0)
        tracker.record_slot_start(
            slot_time=dt, mode="HOLD", predicted_soc_end=55.0,
            predicted_load_kw=0.5, predicted_pv_kw=3.0,
        )
        tracker.record_slot_end(actual_soc=56.0, actual_pv_w=3600.0)
        outcomes = tracker.get_recent_outcomes()
        assert outcomes[0]["actual_pv_kw"] == 3.6
        assert outcomes[0]["pv_error_pct"] is not None
        # (3.6 - 3.0) / 3.0 * 100 = 20%
        assert abs(outcomes[0]["pv_error_pct"] - 20.0) < 0.1
