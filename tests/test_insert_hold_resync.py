"""
Tests for _insert_hold_and_resync functionality.

These tests verify that safety HOLD insertions correctly modify the schedule
and preserve the rest of the TOU periods when resynced to the inverter.
"""

import datetime
from typing import Dict, Optional
from unittest.mock import MagicMock, patch, call

import pytest

import sys
from pathlib import Path
apps_dir = Path(__file__).parent.parent / "appdaemon" / "apps"
sys.path.insert(0, str(apps_dir))

from battery_optimizer import BatteryMode, ScheduleEntry, TouPeriod, BatteryOptimizer
from battery_optimizer_lib import TouSyncManager, BatteryOptimizerConfig


class MockInsertHoldOptimizer:
    """
    Mock optimizer for testing _insert_hold_and_resync.
    """

    def __init__(self):
        self.schedule: Dict[datetime.datetime, ScheduleEntry] = {}
        self._current_datetime = datetime.datetime(2024, 1, 15, 9, 55, 0)
        self._current_date = datetime.date(2024, 1, 15)
        self.current_mode = BatteryMode.CHARGE
        self._schedule_tou_sync_calls = []
        self._handle_mode_transition_calls = []
        self._update_schedule_sensor_calls = 0

        # Create config object
        self.config = BatteryOptimizerConfig(
            slot_minutes=60,
            tou_sync_enabled=True,
            device_id="test_device",
        )

        # Create TouSyncManager for delegation
        self._tou_sync_manager = TouSyncManager(
            device_id=self.config.device_id,
            slot_minutes=self.config.slot_minutes,
            ha_url="",
            ha_token="",
            call_service_func=lambda *args, **kwargs: None,
            get_datetime_func=self.datetime,
            get_timezone_func=self._get_local_timezone,
            sleep_func=lambda x: None,
            create_task_func=lambda x: None,
            log_func=self.log,
            get_schedule_func=lambda: self.schedule,
        )

    def datetime(self):
        return self._current_datetime

    def date(self):
        return self._current_date

    def log(self, message: str, level: str = "INFO"):
        pass

    def _get_local_timezone(self):
        return None

    def _align_to_slot(self, dt: datetime.datetime) -> datetime.datetime:
        """Align datetime to slot boundary."""
        return dt.replace(minute=0, second=0, microsecond=0)

    def _handle_mode_transition(self, new_mode: BatteryMode):
        """Track mode transition calls."""
        self._handle_mode_transition_calls.append(new_mode)
        self.current_mode = new_mode

    def _update_schedule_sensor(self):
        """Track sensor update calls."""
        self._update_schedule_sensor_calls += 1

    def _schedule_tou_sync(self, **kwargs):
        """Track TOU sync calls."""
        self._schedule_tou_sync_calls.append(kwargs)


# Bind the actual method from BatteryOptimizer
MockInsertHoldOptimizer._insert_hold_and_resync = BatteryOptimizer._insert_hold_and_resync
MockInsertHoldOptimizer.schedule_to_tou_periods = BatteryOptimizer.schedule_to_tou_periods


class TestInsertHoldAndResync:
    """Test cases for _insert_hold_and_resync."""

    @pytest.fixture
    def optimizer(self):
        return MockInsertHoldOptimizer()

    def test_modifies_charge_slot_to_hold(self, optimizer):
        """Inserting HOLD on a CHARGE slot should change it to HOLD."""
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        optimizer.schedule = {
            slot_09: ScheduleEntry(time=slot_09, mode=BatteryMode.CHARGE, reason="cheap"),
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 55, 0)

        optimizer._insert_hold_and_resync("safety_max_soc")

        assert optimizer.schedule[slot_09].mode == BatteryMode.HOLD
        assert "safety_max_soc_hold" in optimizer.schedule[slot_09].reason
        assert "was CHARGE" in optimizer.schedule[slot_09].reason

    def test_modifies_discharge_slot_to_hold(self, optimizer):
        """Inserting HOLD on a DISCHARGE slot should change it to HOLD."""
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        optimizer.schedule = {
            slot_09: ScheduleEntry(time=slot_09, mode=BatteryMode.DISCHARGE, reason="expensive"),
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 30, 0)

        optimizer._insert_hold_and_resync("safety_min_soc")

        assert optimizer.schedule[slot_09].mode == BatteryMode.HOLD
        assert "safety_min_soc_hold" in optimizer.schedule[slot_09].reason
        assert "was DISCHARGE" in optimizer.schedule[slot_09].reason

    def test_already_hold_schedule_and_mode_does_nothing(self, optimizer):
        """If both schedule AND current_mode are HOLD, should not modify or resync."""
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        original_entry = ScheduleEntry(time=slot_09, mode=BatteryMode.HOLD, reason="already_hold")
        optimizer.schedule = {slot_09: original_entry}
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 30, 0)
        optimizer.current_mode = BatteryMode.HOLD  # Both schedule and mode are HOLD

        optimizer._insert_hold_and_resync("safety")

        # Entry should be unchanged
        assert optimizer.schedule[slot_09] is original_entry
        # No TOU sync should be triggered
        assert len(optimizer._schedule_tou_sync_calls) == 0
        # No mode transition
        assert len(optimizer._handle_mode_transition_calls) == 0

    def test_schedule_hold_but_mode_differs_enforces_hold(self, optimizer):
        """
        If schedule is HOLD but current_mode is CHARGE/DISCHARGE, must enforce HOLD.

        This is a critical safety scenario: schedule says HOLD but inverter is
        still charging (e.g., manual override or desync). Safety check triggers
        and we MUST force HOLD even though schedule already shows HOLD.
        """
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        original_entry = ScheduleEntry(time=slot_09, mode=BatteryMode.HOLD, reason="already_hold")
        optimizer.schedule = {slot_09: original_entry}
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 30, 0)
        optimizer.current_mode = BatteryMode.CHARGE  # Inverter is charging despite HOLD schedule!

        optimizer._insert_hold_and_resync("safety_max_soc")

        # Schedule entry should remain HOLD (unchanged)
        assert optimizer.schedule[slot_09].mode == BatteryMode.HOLD

        # CRITICAL: Mode transition MUST be called to stop charging
        assert len(optimizer._handle_mode_transition_calls) == 1
        assert optimizer._handle_mode_transition_calls[0] == BatteryMode.HOLD

        # TOU sync MUST be triggered to update inverter
        assert len(optimizer._schedule_tou_sync_calls) == 1

    def test_schedule_hold_but_discharging_enforces_hold(self, optimizer):
        """
        Similar to above but with DISCHARGE mode - must enforce HOLD.
        """
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        original_entry = ScheduleEntry(time=slot_09, mode=BatteryMode.HOLD, reason="already_hold")
        optimizer.schedule = {slot_09: original_entry}
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 30, 0)
        optimizer.current_mode = BatteryMode.DISCHARGE  # Inverter is discharging despite HOLD schedule!

        optimizer._insert_hold_and_resync("safety_min_soc")

        # Mode transition MUST be called to stop discharging
        assert len(optimizer._handle_mode_transition_calls) == 1
        assert optimizer._handle_mode_transition_calls[0] == BatteryMode.HOLD

        # TOU sync MUST be triggered
        assert len(optimizer._schedule_tou_sync_calls) == 1

    def test_missing_slot_creates_hold_entry(self, optimizer):
        """If current slot is not in schedule, should create HOLD entry."""
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        optimizer.schedule = {
            slot_10: ScheduleEntry(time=slot_10, mode=BatteryMode.DISCHARGE, reason="later"),
        }
        # Current time is 09:30, but 09:00 slot is missing
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 30, 0)

        optimizer._insert_hold_and_resync("safety")

        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        assert slot_09 in optimizer.schedule
        assert optimizer.schedule[slot_09].mode == BatteryMode.HOLD
        assert "safety_hold" in optimizer.schedule[slot_09].reason

    def test_triggers_mode_transition(self, optimizer):
        """Should call _handle_mode_transition with HOLD."""
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        optimizer.schedule = {
            slot_09: ScheduleEntry(time=slot_09, mode=BatteryMode.CHARGE, reason="charge"),
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 55, 0)

        optimizer._insert_hold_and_resync("safety")

        assert BatteryMode.HOLD in optimizer._handle_mode_transition_calls

    def test_triggers_schedule_sensor_update(self, optimizer):
        """Should call _update_schedule_sensor."""
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        optimizer.schedule = {
            slot_09: ScheduleEntry(time=slot_09, mode=BatteryMode.CHARGE, reason="charge"),
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 55, 0)

        optimizer._insert_hold_and_resync("safety")

        assert optimizer._update_schedule_sensor_calls == 1

    def test_triggers_tou_sync_when_enabled(self, optimizer):
        """Should call _schedule_tou_sync when TOU sync is enabled."""
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        optimizer.schedule = {
            slot_09: ScheduleEntry(time=slot_09, mode=BatteryMode.CHARGE, reason="charge"),
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 55, 0)

        optimizer._insert_hold_and_resync("safety_max_soc")

        assert len(optimizer._schedule_tou_sync_calls) == 1
        call_kwargs = optimizer._schedule_tou_sync_calls[0]
        assert call_kwargs.get("skip_fit_check") is True
        assert "safety_max_soc_hold_resync" in call_kwargs.get("reason", "")

    def test_no_tou_sync_when_disabled(self, optimizer):
        """Should NOT call _schedule_tou_sync when TOU sync is disabled."""
        optimizer.config.tou_sync_enabled = False
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        optimizer.schedule = {
            slot_09: ScheduleEntry(time=slot_09, mode=BatteryMode.CHARGE, reason="charge"),
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 55, 0)

        optimizer._insert_hold_and_resync("safety")

        assert len(optimizer._schedule_tou_sync_calls) == 0

    def test_no_tou_sync_when_no_device(self, optimizer):
        """Should NOT call _schedule_tou_sync when device_id is empty."""
        optimizer.config.device_id = ""
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        optimizer.schedule = {
            slot_09: ScheduleEntry(time=slot_09, mode=BatteryMode.CHARGE, reason="charge"),
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 55, 0)

        optimizer._insert_hold_and_resync("safety")

        assert len(optimizer._schedule_tou_sync_calls) == 0


class TestInsertHoldPreservesSchedule:
    """Test that inserting HOLD preserves the rest of the schedule in TOU conversion."""

    @pytest.fixture
    def optimizer(self):
        return MockInsertHoldOptimizer()

    def test_hold_insertion_preserves_future_discharge(self, optimizer):
        """
        Scenario: At 09:55, SOC reaches 100% during CHARGE.
        - 09:00 was CHARGE, should become HOLD
        - 10:00, 11:00, 12:00 are DISCHARGE, should remain unchanged
        """
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        slot_11 = datetime.datetime(2024, 1, 15, 11, 0, 0)
        slot_12 = datetime.datetime(2024, 1, 15, 12, 0, 0)

        optimizer.schedule = {
            slot_09: ScheduleEntry(time=slot_09, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_10: ScheduleEntry(time=slot_10, mode=BatteryMode.DISCHARGE, reason="expensive"),
            slot_11: ScheduleEntry(time=slot_11, mode=BatteryMode.DISCHARGE, reason="expensive"),
            slot_12: ScheduleEntry(time=slot_12, mode=BatteryMode.DISCHARGE, reason="expensive"),
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 55, 0)

        optimizer._insert_hold_and_resync("safety_max_soc")

        # 09:00 should be HOLD now
        assert optimizer.schedule[slot_09].mode == BatteryMode.HOLD

        # Future slots should be unchanged
        assert optimizer.schedule[slot_10].mode == BatteryMode.DISCHARGE
        assert optimizer.schedule[slot_11].mode == BatteryMode.DISCHARGE
        assert optimizer.schedule[slot_12].mode == BatteryMode.DISCHARGE

    def test_tou_periods_after_hold_insertion(self, optimizer):
        """
        Verify TOU periods are correctly generated after HOLD insertion.

        Before: 09:00-09:59 CHARGE, 10:00-12:59 DISCHARGE
        After:  09:00-09:59 HOLD,   10:00-12:59 DISCHARGE
        """
        slot_09 = datetime.datetime(2024, 1, 15, 9, 0, 0)
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        slot_11 = datetime.datetime(2024, 1, 15, 11, 0, 0)
        slot_12 = datetime.datetime(2024, 1, 15, 12, 0, 0)

        optimizer.schedule = {
            slot_09: ScheduleEntry(time=slot_09, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_10: ScheduleEntry(time=slot_10, mode=BatteryMode.DISCHARGE, reason="expensive"),
            slot_11: ScheduleEntry(time=slot_11, mode=BatteryMode.DISCHARGE, reason="expensive"),
            slot_12: ScheduleEntry(time=slot_12, mode=BatteryMode.DISCHARGE, reason="expensive"),
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 55, 0)

        # Insert HOLD
        optimizer._insert_hold_and_resync("safety_max_soc")

        # Convert to TOU periods
        periods = optimizer.schedule_to_tou_periods()

        # Find the HOLD period at 09:00 (may also have midnight HOLD pad)
        hold_periods = [p for p in periods if p.power == 1]
        assert len(hold_periods) >= 1
        hold_period = next(p for p in hold_periods if p.start == 9 * 60)
        assert hold_period.end == 9 * 60 + 59  # 09:59

        # Find the DISCHARGE period (power=-100)
        discharge_periods = [p for p in periods if p.power == -100]
        assert len(discharge_periods) >= 1

        # Verify discharge covers 10:00-12:59
        discharge_period = next(p for p in discharge_periods if p.start == 10 * 60)
        assert discharge_period.start == 10 * 60  # 10:00
        assert discharge_period.end == 12 * 60 + 59  # 12:59

    def test_full_day_schedule_with_hold_insertion(self, optimizer):
        """
        Full day scenario with HOLD insertion preserving the complete schedule.

        Original: 00-07 DISCHARGE, 08-09 CHARGE, 10-23 DISCHARGE
        After HOLD at 09:00: 00-07 DISCHARGE, 08:00 CHARGE, 09:00 HOLD, 10-23 DISCHARGE
        """
        # Build full day schedule
        for hour in range(24):
            dt = datetime.datetime(2024, 1, 15, hour, 0, 0)
            if hour < 8:
                mode = BatteryMode.DISCHARGE
                reason = "night_discharge"
            elif hour < 10:
                mode = BatteryMode.CHARGE
                reason = "morning_charge"
            else:
                mode = BatteryMode.DISCHARGE
                reason = "day_discharge"

            optimizer.schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=reason)

        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 55, 0)

        # Insert HOLD at 09:00
        optimizer._insert_hold_and_resync("safety_max_soc")

        # Verify schedule
        assert optimizer.schedule[datetime.datetime(2024, 1, 15, 8, 0, 0)].mode == BatteryMode.CHARGE
        assert optimizer.schedule[datetime.datetime(2024, 1, 15, 9, 0, 0)].mode == BatteryMode.HOLD
        assert optimizer.schedule[datetime.datetime(2024, 1, 15, 10, 0, 0)].mode == BatteryMode.DISCHARGE

        # Convert to TOU and verify
        periods = optimizer.schedule_to_tou_periods()

        # Should have: DISCHARGE (00-07), CHARGE (08), HOLD (09), DISCHARGE (10-23)
        # These should consolidate to 4 periods
        assert len(periods) == 4

        sorted_periods = sorted(periods, key=lambda p: p.start)

        # Period 1: 00:00-07:59 DISCHARGE
        assert sorted_periods[0].start == 0
        assert sorted_periods[0].end == 7 * 60 + 59
        assert sorted_periods[0].power == -100

        # Period 2: 08:00-08:59 CHARGE
        assert sorted_periods[1].start == 8 * 60
        assert sorted_periods[1].end == 8 * 60 + 59
        assert sorted_periods[1].power == 100

        # Period 3: 09:00-09:59 HOLD
        assert sorted_periods[2].start == 9 * 60
        assert sorted_periods[2].end == 9 * 60 + 59
        assert sorted_periods[2].power == 1

        # Period 4: 10:00-23:59 DISCHARGE
        assert sorted_periods[3].start == 10 * 60
        assert sorted_periods[3].end == 23 * 60 + 59
        assert sorted_periods[3].power == -100


class TestInsertHoldWithRollingBoundary:
    """Test HOLD insertion works correctly with rolling TOU boundaries."""

    @pytest.fixture
    def optimizer(self):
        return MockInsertHoldOptimizer()

    def test_hold_insertion_with_rolling_boundary(self, optimizer):
        """
        Scenario: Rolling boundary at 09:00, HOLD inserted at 09:55.

        Schedule spans today and tomorrow:
        - Today 09:00: CHARGE (at/after boundary, use today)
        - Today 10:00-12:00: DISCHARGE (after boundary, use today)
        - Tomorrow 00:00-08:00: DISCHARGE (before boundary, use tomorrow)

        After HOLD insertion at 09:00:
        - Today 09:00 becomes HOLD
        - Rest of schedule preserved
        """
        today = datetime.date(2024, 1, 15)
        tomorrow = datetime.date(2024, 1, 16)
        optimizer._current_date = today

        # Today's schedule
        for hour in range(9, 13):
            dt = datetime.datetime(2024, 1, 15, hour, 0, 0)
            mode = BatteryMode.CHARGE if hour == 9 else BatteryMode.DISCHARGE
            optimizer.schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason="today")

        # Tomorrow's schedule (early hours)
        for hour in range(0, 9):
            dt = datetime.datetime(2024, 1, 16, hour, 0, 0)
            optimizer.schedule[dt] = ScheduleEntry(
                time=dt, mode=BatteryMode.DISCHARGE, reason="tomorrow"
            )

        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 9, 55, 0)

        # Insert HOLD
        optimizer._insert_hold_and_resync("safety_max_soc")

        # Today's 09:00 should be HOLD
        assert optimizer.schedule[datetime.datetime(2024, 1, 15, 9, 0, 0)].mode == BatteryMode.HOLD

        # Today's future slots unchanged
        assert optimizer.schedule[datetime.datetime(2024, 1, 15, 10, 0, 0)].mode == BatteryMode.DISCHARGE

        # Tomorrow's slots unchanged
        assert optimizer.schedule[datetime.datetime(2024, 1, 16, 6, 0, 0)].mode == BatteryMode.DISCHARGE

        # Convert with rolling boundary at 09:00 (540 minutes)
        periods = optimizer.schedule_to_tou_periods(boundary_minute=540)

        # Verify periods
        # Before boundary (00:00-08:59): Use tomorrow's DISCHARGE
        # At/after boundary (09:00+): Use today's schedule

        # Find period at 06:00 (before boundary) - should be DISCHARGE from tomorrow
        period_at_6 = next((p for p in periods if p.start <= 360 <= p.end), None)
        assert period_at_6 is not None
        assert period_at_6.power == -100  # DISCHARGE

        # Find period at 09:00 (at boundary) - should be HOLD from today
        period_at_9 = next((p for p in periods if p.start <= 540 <= p.end), None)
        assert period_at_9 is not None
        assert period_at_9.power == 1  # HOLD

        # Find period at 10:00 (after boundary) - should be DISCHARGE from today
        period_at_10 = next((p for p in periods if p.start <= 600 <= p.end), None)
        assert period_at_10 is not None
        assert period_at_10.power == -100  # DISCHARGE

    def test_hold_insertion_before_rolling_boundary(self, optimizer):
        """
        Edge case: HOLD inserted at slot that's before the rolling boundary.

        If current time is 05:55 with boundary at 06:00:
        - 05:00 slot is BEFORE boundary
        - But we're modifying today's schedule, not tomorrow's

        This tests that schedule modification happens in the correct day's schedule.
        """
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Today's early morning schedule
        slot_05 = datetime.datetime(2024, 1, 15, 5, 0, 0)
        slot_06 = datetime.datetime(2024, 1, 15, 6, 0, 0)

        optimizer.schedule = {
            slot_05: ScheduleEntry(time=slot_05, mode=BatteryMode.CHARGE, reason="early_charge"),
            slot_06: ScheduleEntry(time=slot_06, mode=BatteryMode.DISCHARGE, reason="morning"),
        }

        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 5, 55, 0)

        # Insert HOLD
        optimizer._insert_hold_and_resync("safety_max_soc")

        # 05:00 should be HOLD now
        assert optimizer.schedule[slot_05].mode == BatteryMode.HOLD

        # 06:00 unchanged
        assert optimizer.schedule[slot_06].mode == BatteryMode.DISCHARGE


class TestSolarOverrideHoldInsertion:
    """Test HOLD insertion for solar override scenarios."""

    @pytest.fixture
    def optimizer(self):
        return MockInsertHoldOptimizer()

    def test_solar_override_inserts_hold(self, optimizer):
        """Solar override should insert HOLD and preserve rest of schedule."""
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        slot_11 = datetime.datetime(2024, 1, 15, 11, 0, 0)
        slot_12 = datetime.datetime(2024, 1, 15, 12, 0, 0)

        optimizer.schedule = {
            slot_10: ScheduleEntry(time=slot_10, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_11: ScheduleEntry(time=slot_11, mode=BatteryMode.CHARGE, reason="cheap"),
            slot_12: ScheduleEntry(time=slot_12, mode=BatteryMode.DISCHARGE, reason="expensive"),
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 10, 30, 0)

        optimizer._insert_hold_and_resync("solar_override")

        # 10:00 should be HOLD (solar override)
        assert optimizer.schedule[slot_10].mode == BatteryMode.HOLD
        assert "solar_override" in optimizer.schedule[slot_10].reason

        # 11:00 and 12:00 unchanged
        assert optimizer.schedule[slot_11].mode == BatteryMode.CHARGE
        assert optimizer.schedule[slot_12].mode == BatteryMode.DISCHARGE

    def test_solar_override_reason_in_tou_sync(self, optimizer):
        """Solar override should pass correct reason to TOU sync."""
        slot_10 = datetime.datetime(2024, 1, 15, 10, 0, 0)
        optimizer.schedule = {
            slot_10: ScheduleEntry(time=slot_10, mode=BatteryMode.CHARGE, reason="cheap"),
        }
        optimizer._current_datetime = datetime.datetime(2024, 1, 15, 10, 30, 0)

        optimizer._insert_hold_and_resync("solar_override")

        assert len(optimizer._schedule_tou_sync_calls) == 1
        assert "solar_override_hold_resync" in optimizer._schedule_tou_sync_calls[0].get("reason", "")
