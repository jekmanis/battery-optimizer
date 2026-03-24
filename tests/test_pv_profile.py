"""Unit tests for PvProfile module."""

import datetime
import json
import sys
sys.path.insert(0, "appdaemon/apps")

from battery_optimizer_lib.pv_profile import PvProfile


class TestPvProfile:
    def test_record_and_predict(self):
        profile = PvProfile(slot_minutes=15, default_pv_w=0.0, min_samples=2)
        dt = datetime.datetime(2024, 7, 15, 12, 0, 0)
        profile.record(dt, 3000.0)
        profile.record(dt, 4000.0)
        predicted = profile.predict_kw(dt)
        assert predicted > 0, "Should predict PV after recording observations"
        assert predicted < 5.0, "Prediction should be reasonable"

    def test_default_zero_at_night(self):
        profile = PvProfile(slot_minutes=15, default_pv_w=0.0)
        dt = datetime.datetime(2024, 7, 15, 2, 0, 0)  # 2 AM
        predicted = profile.predict_kw(dt)
        assert predicted == 0.0

    def test_negative_pv_ignored(self):
        profile = PvProfile(slot_minutes=15)
        dt = datetime.datetime(2024, 7, 15, 12, 0, 0)
        profile.record(dt, -100.0)
        assert profile.stats.observation_count == 0

    def test_json_persistence(self):
        profile = PvProfile(slot_minutes=15, min_samples=2)
        dt = datetime.datetime(2024, 7, 15, 12, 0, 0)
        profile.record(dt, 3000.0)
        profile.record(dt, 4000.0)

        json_str = profile.to_json()
        profile2 = PvProfile(slot_minutes=15, min_samples=2)
        assert profile2.load_from_json(json_str)
        assert profile2.stats.observation_count == 2
        assert profile2.predict_kw(dt) > 0

    def test_slot_migration(self):
        profile30 = PvProfile(slot_minutes=30, min_samples=1)
        dt = datetime.datetime(2024, 7, 15, 12, 0, 0)
        profile30.record(dt, 3000.0)
        json_str = profile30.to_json()

        profile15 = PvProfile(slot_minutes=15, min_samples=1)
        assert profile15.load_from_json(json_str)
        assert profile15.predict_kw(dt) > 0

    def test_max_samples_enforced(self):
        profile = PvProfile(slot_minutes=15, max_samples=5)
        dt = datetime.datetime(2024, 7, 15, 12, 0, 0)
        for i in range(10):
            profile.record(dt, 1000.0 + i * 100)
        slot = str(profile._slot_index(dt))
        assert len(profile.stats.samples_by_slot[slot]) == 5

    def test_median_quantile_default(self):
        profile = PvProfile(slot_minutes=15, min_samples=1)
        dt = datetime.datetime(2024, 7, 15, 12, 0, 0)
        profile.record(dt, 1000.0)
        profile.record(dt, 3000.0)
        profile.record(dt, 5000.0)
        predicted = profile.predict_kw(dt, quantile=0.5)
        assert abs(predicted - 3.0) < 0.5, f"Median should be ~3kW, got {predicted}"
