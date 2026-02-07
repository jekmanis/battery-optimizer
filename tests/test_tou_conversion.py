"""
Tests for schedule_to_tou_periods conversion.

These tests verify the conversion of schedule entries to TOU periods
for inverter programming.
"""

import datetime
from typing import Dict, List

import pytest

from battery_optimizer import BatteryMode, ScheduleEntry, TouPeriod
from battery_optimizer_lib import TouSyncManager


class MockTouOptimizer:
    """
    Minimal mock for testing schedule_to_tou_periods.
    """

    def __init__(self):
        self.schedule: Dict[datetime.datetime, ScheduleEntry] = {}
        self._current_date = datetime.date(2024, 1, 15)
        self._current_datetime = datetime.datetime(2024, 1, 15, 12, 0, 0)
        self.slot_minutes = 60  # Default hourly slots

        # Create TouSyncManager for delegation
        self._tou_sync_manager = TouSyncManager(
            device_id="",
            slot_minutes=self.slot_minutes,
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

    def date(self):
        return self._current_date

    def datetime(self):
        return self._current_datetime

    def log(self, message: str, level: str = "INFO"):
        pass

    def _get_local_timezone(self):
        return None


# Import and bind the method
import sys
from pathlib import Path
apps_dir = Path(__file__).parent.parent / "appdaemon" / "apps"
sys.path.insert(0, str(apps_dir))

from battery_optimizer import BatteryOptimizer

MockTouOptimizer.schedule_to_tou_periods = BatteryOptimizer.schedule_to_tou_periods


class TestScheduleToTouPeriods:
    """Test cases for schedule_to_tou_periods conversion."""

    @pytest.fixture
    def optimizer(self):
        return MockTouOptimizer()

    def test_empty_schedule_returns_empty(self, optimizer):
        """Empty schedule should return empty list."""
        optimizer.schedule = {}
        periods = optimizer.schedule_to_tou_periods()
        assert periods == []

    def test_single_charge_hour(self, optimizer):
        """Single charge hour should create one period."""
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 2, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 2, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Charge"
            )
        }

        periods = optimizer.schedule_to_tou_periods()

        assert len(periods) >= 1
        # Find the charge period
        charge_periods = [p for p in periods if p.power == 100]
        assert len(charge_periods) == 1

    def test_single_discharge_hour(self, optimizer):
        """Single discharge hour should create period with negative power."""
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 18, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 18, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Discharge"
            )
        }

        periods = optimizer.schedule_to_tou_periods()

        discharge_periods = [p for p in periods if p.power == -100]
        assert len(discharge_periods) == 1

    def test_single_hold_hour(self, optimizer):
        """Single hold hour should create period with power=1 (standby)."""
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 12, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 12, 0, 0),
                mode=BatteryMode.HOLD,
                reason="Hold"
            )
        }

        periods = optimizer.schedule_to_tou_periods()

        hold_periods = [p for p in periods if p.power == 1]
        assert len(hold_periods) >= 1
        # 12:00 (720 min) must be covered
        assert any(p.start <= 720 <= p.end for p in hold_periods)

    def test_contiguous_same_mode_consolidated(self, optimizer):
        """Contiguous hours with same mode should consolidate."""
        base_dt = datetime.datetime(2024, 1, 15, 2, 0, 0)
        optimizer.schedule = {
            base_dt: ScheduleEntry(time=base_dt, mode=BatteryMode.CHARGE, reason="C1"),
            base_dt + datetime.timedelta(hours=1): ScheduleEntry(
                time=base_dt + datetime.timedelta(hours=1),
                mode=BatteryMode.CHARGE,
                reason="C2"
            ),
            base_dt + datetime.timedelta(hours=2): ScheduleEntry(
                time=base_dt + datetime.timedelta(hours=2),
                mode=BatteryMode.CHARGE,
                reason="C3"
            ),
        }

        periods = optimizer.schedule_to_tou_periods()

        # Should consolidate into 1 charge period
        charge_periods = [p for p in periods if p.power == 100]
        assert len(charge_periods) == 1

        # Period should span 3 hours (02:00 - 04:59)
        period = charge_periods[0]
        assert period.start == 2 * 60  # 02:00
        assert period.end == 4 * 60 + 59  # 04:59

    def test_mode_change_creates_new_period(self, optimizer):
        """Mode changes should create separate periods."""
        base_dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        optimizer.schedule = {
            base_dt: ScheduleEntry(time=base_dt, mode=BatteryMode.CHARGE, reason="C"),
            base_dt + datetime.timedelta(hours=1): ScheduleEntry(
                time=base_dt + datetime.timedelta(hours=1),
                mode=BatteryMode.HOLD,
                reason="H"
            ),
            base_dt + datetime.timedelta(hours=2): ScheduleEntry(
                time=base_dt + datetime.timedelta(hours=2),
                mode=BatteryMode.DISCHARGE,
                reason="D"
            ),
        }

        periods = optimizer.schedule_to_tou_periods()

        # 3 mode periods + midnight HOLD pad = 4
        assert len(periods) == 4

        # Verify mode-specific power values
        powers = sorted([p.power for p in periods])
        assert -100 in powers  # Discharge
        assert 1 in powers     # Hold
        assert 100 in powers   # Charge

    def test_periods_do_not_overlap(self, optimizer):
        """Generated periods should not overlap."""
        base_dt = datetime.datetime(2024, 1, 15, 0, 0, 0)
        optimizer.schedule = {}

        # Create full 24-hour schedule with varying modes
        for i in range(24):
            mode = [BatteryMode.CHARGE, BatteryMode.HOLD, BatteryMode.DISCHARGE][i % 3]
            dt = base_dt + datetime.timedelta(hours=i)
            optimizer.schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=f"Mode {i}")

        periods = optimizer.schedule_to_tou_periods()

        # Sort by start time
        sorted_periods = sorted(periods, key=lambda p: p.start)

        # Check no overlaps
        for i in range(len(sorted_periods) - 1):
            current = sorted_periods[i]
            next_period = sorted_periods[i + 1]
            # Current end should be less than next start
            assert current.end < next_period.start, \
                f"Period ending at {current.end} overlaps with period starting at {next_period.start}"

    def test_minutes_since_midnight_correct(self, optimizer):
        """Start/end times should be correct minutes since midnight."""
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 6, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 6, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Morning charge"
            )
        }

        periods = optimizer.schedule_to_tou_periods()

        charge_period = next(p for p in periods if p.power == 100)

        # 06:00 = 360 minutes
        assert charge_period.start == 360
        # 06:59 = 419 minutes
        assert charge_period.end == 419

    def test_handles_tomorrow_schedule(self, optimizer):
        """Should include tomorrow's schedule entries."""
        today = datetime.date(2024, 1, 15)
        tomorrow = datetime.date(2024, 1, 16)
        optimizer._current_date = today

        # Schedule spanning today and tomorrow
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 22, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 22, 0, 0),
                mode=BatteryMode.HOLD,
                reason="Tonight"
            ),
            datetime.datetime(2024, 1, 16, 2, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, 2, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Tomorrow early"
            ),
        }

        periods = optimizer.schedule_to_tou_periods()

        # Should have entries from both days
        assert len(periods) >= 2

    def test_today_takes_precedence_over_tomorrow(self, optimizer):
        """Same time-of-day from today should override tomorrow."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Both today and tomorrow have entry at 10:00
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.CHARGE,  # Today: charge
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, 10, 0, 0),
                mode=BatteryMode.DISCHARGE,  # Tomorrow: discharge
                reason="Tomorrow"
            ),
        }

        periods = optimizer.schedule_to_tou_periods()

        # Find period at 10:00 (600 minutes)
        period_at_10 = next(
            (p for p in periods if p.start <= 600 <= p.end),
            None
        )

        # Should be charge (today's entry), not discharge
        assert period_at_10 is not None
        assert period_at_10.power == 100  # Charge, not -100 (discharge)

    def test_excludes_past_dates(self, optimizer):
        """Should exclude entries from before today."""
        optimizer._current_date = datetime.date(2024, 1, 15)

        optimizer.schedule = {
            # Yesterday - should be excluded
            datetime.datetime(2024, 1, 14, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 14, 10, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Yesterday"
            ),
            # Today - should be included
            datetime.datetime(2024, 1, 15, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.HOLD,
                reason="Today"
            ),
        }

        periods = optimizer.schedule_to_tou_periods()

        # Should only have today's entry
        period_at_10 = next(
            (p for p in periods if p.start <= 600 <= p.end),
            None
        )
        assert period_at_10 is not None
        assert period_at_10.power == 1  # Hold (today), not 100 (yesterday's charge)

    def test_max_20_periods(self, optimizer):
        """Should not exceed 20 periods (inverter limit)."""
        base_dt = datetime.datetime(2024, 1, 15, 0, 0, 0)
        optimizer.schedule = {}

        # Create 24 hours alternating modes = 24 periods
        for i in range(24):
            mode = BatteryMode.CHARGE if i % 2 == 0 else BatteryMode.DISCHARGE
            dt = base_dt + datetime.timedelta(hours=i)
            optimizer.schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=f"Hour {i}")

        periods = optimizer.schedule_to_tou_periods()

        # Should be limited to 20
        assert len(periods) <= 20

    def test_full_day_coverage(self, optimizer):
        """Full day schedule should cover 0-1439 minutes."""
        base_dt = datetime.datetime(2024, 1, 15, 0, 0, 0)
        optimizer.schedule = {}

        # Create full 24-hour hold schedule
        for i in range(24):
            dt = base_dt + datetime.timedelta(hours=i)
            optimizer.schedule[dt] = ScheduleEntry(
                time=dt, mode=BatteryMode.HOLD, reason=f"Hour {i}"
            )

        periods = optimizer.schedule_to_tou_periods()

        # All HOLD — may be 1 or 2 periods depending on walk start
        assert len(periods) <= 2
        assert all(p.power == 1 for p in periods)

        # Should cover full day (0-1439)
        covered = set()
        for p in periods:
            for m in range(p.start, p.end + 1):
                covered.add(m)
        assert covered == set(range(1440))


class TestRollingBoundary:
    """Test cases for rolling boundary TOU schedule sync."""

    @pytest.fixture
    def optimizer(self):
        return MockTouOptimizer()

    def test_no_boundary_prefers_today(self, optimizer):
        """Without boundary_minute, today's entry takes precedence."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Both days have entry at 10:00 with different modes
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, 10, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow"
            ),
        }

        periods = optimizer.schedule_to_tou_periods()  # No boundary

        period_at_10 = next((p for p in periods if p.start <= 600 <= p.end), None)
        assert period_at_10 is not None
        assert period_at_10.power == 100  # Today's CHARGE

    def test_boundary_before_uses_tomorrow(self, optimizer):
        """Hours before boundary should use tomorrow's schedule."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Both days have entry at 06:00 with different modes
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 6, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 6, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 6, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, 6, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow"
            ),
        }

        # Boundary at 14:00 (840 minutes) - 06:00 is BEFORE boundary
        periods = optimizer.schedule_to_tou_periods(boundary_minute=840)

        period_at_6 = next((p for p in periods if p.start <= 360 <= p.end), None)
        assert period_at_6 is not None
        assert period_at_6.power == -100  # Tomorrow's DISCHARGE

    def test_boundary_at_or_after_uses_today(self, optimizer):
        """Hours at or after boundary should use today's schedule."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Both days have entry at 18:00 with different modes
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 18, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 18, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 18, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, 18, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow"
            ),
        }

        # Boundary at 14:00 (840 minutes) - 18:00 is AFTER boundary
        periods = optimizer.schedule_to_tou_periods(boundary_minute=840)

        period_at_18 = next((p for p in periods if p.start <= 1080 <= p.end), None)
        assert period_at_18 is not None
        assert period_at_18.power == 100  # Today's CHARGE

    def test_boundary_exactly_at_hour(self, optimizer):
        """Hour exactly at boundary should use today's schedule."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Both days have entry at 14:00
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 14, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 14, 0, 0),
                mode=BatteryMode.HOLD,
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 14, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, 14, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow"
            ),
        }

        # Boundary at exactly 14:00 (840 minutes)
        periods = optimizer.schedule_to_tou_periods(boundary_minute=840)

        period_at_14 = next((p for p in periods if p.start <= 840 <= p.end), None)
        assert period_at_14 is not None
        assert period_at_14.power == 1  # Today's HOLD

    def test_boundary_zero_uses_today_for_all(self, optimizer):
        """With boundary=0, all overlapping hours use today's schedule."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Both days have entries at multiple hours
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 6, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 6, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today morning"
            ),
            datetime.datetime(2024, 1, 15, 18, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 18, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today evening"
            ),
            datetime.datetime(2024, 1, 16, 6, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, 6, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow morning"
            ),
            datetime.datetime(2024, 1, 16, 18, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, 18, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow evening"
            ),
        }

        # Boundary at 0 - all hours are >= 0, so all use today
        periods = optimizer.schedule_to_tou_periods(boundary_minute=0)

        # Scheduled time slots should be CHARGE (today's schedule)
        # Midnight HOLD pad may be present at start
        charge_periods = [p for p in periods if p.power == 100]
        assert len(charge_periods) >= 2  # 06:00 and 18:00
        # Non-charge periods should only be the HOLD pad at start
        for period in periods:
            if period.power != 100:
                assert period.start == 0  # Only the midnight pad

    def test_boundary_end_of_day_uses_tomorrow_for_all(self, optimizer):
        """With boundary=1440, all hours use tomorrow's schedule."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Both days have entries
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, 10, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow"
            ),
        }

        # Boundary at 1440 (end of day) - all hours are < 1440
        periods = optimizer.schedule_to_tou_periods(boundary_minute=1440)

        period_at_10 = next((p for p in periods if p.start <= 600 <= p.end), None)
        assert period_at_10 is not None
        assert period_at_10.power == -100  # Tomorrow's DISCHARGE

    def test_boundary_splits_day_correctly(self, optimizer):
        """Boundary should correctly split day between today and tomorrow."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Full day schedule for both days with different patterns
        for hour in range(24):
            # Today: CHARGE for all hours
            optimizer.schedule[datetime.datetime(2024, 1, 15, hour, 0, 0)] = ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, hour, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today"
            )
            # Tomorrow: DISCHARGE for all hours
            optimizer.schedule[datetime.datetime(2024, 1, 16, hour, 0, 0)] = ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, hour, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow"
            )

        # Boundary at 12:00 (720 minutes)
        periods = optimizer.schedule_to_tou_periods(boundary_minute=720)

        # Should have 2 periods: DISCHARGE 00:00-11:59, CHARGE 12:00-23:59
        assert len(periods) == 2

        sorted_periods = sorted(periods, key=lambda p: p.start)

        # First period: 00:00-11:59 should be DISCHARGE (tomorrow)
        assert sorted_periods[0].start == 0
        assert sorted_periods[0].end == 719  # 11:59
        assert sorted_periods[0].power == -100  # DISCHARGE

        # Second period: 12:00-23:59 should be CHARGE (today)
        assert sorted_periods[1].start == 720
        assert sorted_periods[1].end == 1439  # 23:59
        assert sorted_periods[1].power == 100  # CHARGE

    def test_boundary_with_only_today_schedule(self, optimizer):
        """Boundary with only today's schedule should work normally."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Only today's schedule
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today"
            ),
        }

        # Boundary at 14:00 - 10:00 is before, but no tomorrow entry exists
        periods = optimizer.schedule_to_tou_periods(boundary_minute=840)

        # Should still include today's entry (it's the only one)
        assert len(periods) >= 1
        period_at_10 = next((p for p in periods if p.start <= 600 <= p.end), None)
        assert period_at_10 is not None
        assert period_at_10.power == 100

    def test_boundary_with_only_tomorrow_schedule(self, optimizer):
        """Boundary with only tomorrow's schedule should work normally."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Only tomorrow's schedule
        optimizer.schedule = {
            datetime.datetime(2024, 1, 16, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, 10, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow"
            ),
        }

        # Boundary at 14:00 - 10:00 is before, so prefer tomorrow (which is all we have)
        periods = optimizer.schedule_to_tou_periods(boundary_minute=840)

        assert len(periods) >= 1
        period_at_10 = next((p for p in periods if p.start <= 600 <= p.end), None)
        assert period_at_10 is not None
        assert period_at_10.power == -100  # Tomorrow's DISCHARGE

    def test_boundary_non_overlapping_hours(self, optimizer):
        """Non-overlapping hours from both days should both appear."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Today has 18:00, tomorrow has 06:00 (no overlap)
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 18, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 18, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today evening"
            ),
            datetime.datetime(2024, 1, 16, 6, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 16, 6, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow morning"
            ),
        }

        # Boundary at 12:00
        periods = optimizer.schedule_to_tou_periods(boundary_minute=720)

        # Both entries + midnight HOLD pad
        assert len(periods) == 3

        # 06:00 (before boundary) from tomorrow
        period_at_6 = next((p for p in periods if p.start <= 360 <= p.end), None)
        assert period_at_6 is not None
        assert period_at_6.power == -100  # DISCHARGE

        # 18:00 (after boundary) from today
        period_at_18 = next((p for p in periods if p.start <= 1080 <= p.end), None)
        assert period_at_18 is not None
        assert period_at_18.power == 100  # CHARGE


class TestTouPeriodEdgeCases:
    """Edge case tests for TOU period handling."""

    @pytest.fixture
    def optimizer(self):
        return MockTouOptimizer()

    def test_midnight_boundary(self, optimizer):
        """Handle periods around midnight correctly."""
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 23, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 23, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Late night"
            ),
            datetime.datetime(2024, 1, 15, 0, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 0, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Midnight"
            ),
        }

        periods = optimizer.schedule_to_tou_periods()

        # Should have periods for both times
        assert len(periods) >= 1

        # Verify times are valid (0-1439)
        for period in periods:
            assert 0 <= period.start <= 1439
            assert 0 <= period.end <= 1439

    def test_single_minute_slot(self, optimizer):
        """Handle very short slots (edge case)."""
        # Single entry at midnight
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 0, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 0, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Brief"
            )
        }

        periods = optimizer.schedule_to_tou_periods()

        # Should still create valid period
        assert len(periods) >= 1
        assert all(p.end >= p.start for p in periods)


class TestFifteenMinuteTOU:
    """Test cases for 15-minute slot TOU period conversion."""

    @pytest.fixture
    def optimizer(self):
        opt = MockTouOptimizer()
        opt.slot_minutes = 15
        # Recreate TouSyncManager with 15-minute slots
        opt._tou_sync_manager = TouSyncManager(
            device_id="",
            slot_minutes=15,
            ha_url="",
            ha_token="",
            call_service_func=lambda *args, **kwargs: None,
            get_datetime_func=opt.datetime,
            get_timezone_func=opt._get_local_timezone,
            sleep_func=lambda x: None,
            create_task_func=lambda x: None,
            log_func=opt.log,
            get_schedule_func=lambda: opt.schedule,
        )
        return opt

    def test_15min_consolidation(self, optimizer):
        """Four consecutive 15-min CHARGE slots should consolidate into one 60-min period."""
        base_dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        optimizer.schedule = {}
        for i in range(4):
            dt = base_dt + datetime.timedelta(minutes=15 * i)
            optimizer.schedule[dt] = ScheduleEntry(
                time=dt, mode=BatteryMode.CHARGE, reason=f"C{i}"
            )

        periods = optimizer.schedule_to_tou_periods()

        # Should consolidate into 1 charge period
        charge_periods = [p for p in periods if p.power == 100]
        assert len(charge_periods) == 1

        # Period should span 60 minutes: 10:00 (600) to 10:59 (659)
        period = charge_periods[0]
        assert period.start == 600
        assert period.end == 659

    def test_15min_many_mode_changes(self, optimizer):
        """Alternating CHARGE/DISCHARGE every 15 min for 24h should respect 20-period max."""
        base_dt = datetime.datetime(2024, 1, 15, 0, 0, 0)
        optimizer.schedule = {}

        # 96 entries alternating every 15 minutes
        for i in range(96):
            dt = base_dt + datetime.timedelta(minutes=15 * i)
            mode = BatteryMode.CHARGE if i % 2 == 0 else BatteryMode.DISCHARGE
            optimizer.schedule[dt] = ScheduleEntry(
                time=dt, mode=mode, reason=f"Slot {i}"
            )

        periods = optimizer.schedule_to_tou_periods()

        # Should respect the 20-period maximum
        assert len(periods) <= 20

    def test_15min_period_boundaries(self, optimizer):
        """A single 15-min CHARGE slot at 10:15 should have correct minute boundaries."""
        dt = datetime.datetime(2024, 1, 15, 10, 15, 0)
        optimizer.schedule = {
            dt: ScheduleEntry(time=dt, mode=BatteryMode.CHARGE, reason="Test")
        }

        periods = optimizer.schedule_to_tou_periods()

        charge_periods = [p for p in periods if p.power == 100]
        assert len(charge_periods) == 1

        period = charge_periods[0]
        # 10:15 = 615 minutes, end = 615 + 15 - 1 = 629
        assert period.start == 615
        assert period.end == 629


def _make_tou_sync(current_datetime, slot_minutes=15):
    """Helper to create a TouSyncManager for testing."""
    return TouSyncManager(
        device_id="",
        slot_minutes=slot_minutes,
        ha_url="",
        ha_token="",
        call_service_func=lambda *args, **kwargs: None,
        get_datetime_func=lambda: current_datetime,
        get_timezone_func=lambda: None,
        sleep_func=lambda x: None,
        create_task_func=lambda x: None,
        log_func=lambda msg, level="INFO": None,
    )


class TestForwardWalk:
    """Test that schedule_to_tou_periods walks forward from now, collecting up to 20 periods."""

    def test_simple_schedule_fits_entirely(self):
        """A schedule with few mode changes fits within 20 periods."""
        now = datetime.datetime(2024, 1, 15, 14, 0, 0)
        tou = _make_tou_sync(now, slot_minutes=60)

        schedule = {}
        for hour in range(24):
            dt = datetime.datetime(2024, 1, 15, hour, 0, 0)
            mode = BatteryMode.CHARGE if hour < 6 else BatteryMode.HOLD
            schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=f"H{hour}")

        periods = tou.schedule_to_tou_periods(schedule)

        # Walk from 14:00: HOLD 14:00-23:59, CHARGE 00:00-05:59, HOLD 06:00-13:59
        # The HOLD zone is split by CHARGE, so 3 periods
        assert len(periods) <= 20
        assert all(p.start >= 0 and p.end <= 1439 for p in periods)

        # Verify correct modes
        minute_power = {}
        for p in periods:
            for m in range(p.start, p.end + 1):
                minute_power[m] = p.power
        # Hour 2 (120 min) should be CHARGE
        assert minute_power.get(120) == 100
        # Hour 10 (600 min) should be HOLD
        assert minute_power.get(600) == 1

    def test_walks_forward_from_now(self):
        """Periods nearest to 'now' should be included first."""
        now = datetime.datetime(2024, 1, 15, 14, 0, 0)
        tou = _make_tou_sync(now, slot_minutes=15)

        # Build a highly fragmented schedule: alternating every 15 min
        schedule = {}
        for i in range(96):
            dt = datetime.datetime(2024, 1, 15, 0, 0, 0) + datetime.timedelta(minutes=15 * i)
            mode = BatteryMode.CHARGE if i % 2 == 0 else BatteryMode.DISCHARGE
            schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=f"S{i}")

        periods = tou.schedule_to_tou_periods(schedule)

        # 96 alternating slots = 96 periods, but capped at 20
        assert len(periods) == 20

        # The 20 periods should cover time near 14:00, not start of day
        minute_power = {}
        for p in periods:
            for m in range(p.start, p.end + 1):
                minute_power[m] = p.power

        # 14:00 (840) must be covered — it's the current slot
        assert 840 in minute_power
        # 14:15 (855) should also be covered — next slot
        assert 855 in minute_power

    def test_far_future_dropped_when_over_20(self):
        """When >20 periods, far-future entries are the ones dropped."""
        now = datetime.datetime(2024, 1, 15, 14, 0, 0)
        tou = _make_tou_sync(now, slot_minutes=15)

        # Alternating every 15 min for full day = 96 periods needed
        schedule = {}
        for i in range(96):
            dt = datetime.datetime(2024, 1, 15, 0, 0, 0) + datetime.timedelta(minutes=15 * i)
            mode = BatteryMode.CHARGE if i % 2 == 0 else BatteryMode.DISCHARGE
            schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=f"S{i}")

        periods = tou.schedule_to_tou_periods(schedule)
        assert len(periods) == 20

        covered_minutes = set()
        for p in periods:
            for m in range(p.start, p.end + 1):
                covered_minutes.add(m)

        # The 20 periods starting from 14:00 cover 20*15=300 min = 5 hours
        # So 14:00 to ~19:00 should be covered
        assert 840 in covered_minutes   # 14:00
        assert 1080 in covered_minutes  # 18:00

        # Morning hours (far future from 14:00) should NOT be covered
        # 06:00 is about 16h forward — well past the 20 period window
        assert 360 not in covered_minutes  # 06:00

    def test_wraps_through_midnight(self):
        """Forward walk at 22:00 should wrap through midnight into tomorrow."""
        now = datetime.datetime(2024, 1, 15, 22, 0, 0)
        tou = _make_tou_sync(now, slot_minutes=60)

        schedule = {}
        for hour in range(24):
            dt_today = datetime.datetime(2024, 1, 15, hour, 0, 0)
            schedule[dt_today] = ScheduleEntry(
                time=dt_today, mode=BatteryMode.CHARGE, reason="Today"
            )
            dt_tomorrow = datetime.datetime(2024, 1, 16, hour, 0, 0)
            schedule[dt_tomorrow] = ScheduleEntry(
                time=dt_tomorrow, mode=BatteryMode.DISCHARGE, reason="Tomorrow"
            )

        # No boundary → today preferred for conflicts
        periods = tou.schedule_to_tou_periods(schedule)

        minute_power = {}
        for p in periods:
            for m in range(p.start, p.end + 1):
                minute_power[m] = p.power

        # 22:00 (1320) is today's CHARGE
        assert minute_power.get(1320) == 100
        # 02:00 (120) is today's entry (today preferred), so CHARGE
        assert minute_power.get(120) == 100

    def test_sorted_by_clock_time(self):
        """Output periods should be sorted by start minute regardless of walk order."""
        now = datetime.datetime(2024, 1, 15, 22, 0, 0)
        tou = _make_tou_sync(now, slot_minutes=60)

        schedule = {}
        for hour in [22, 23]:
            dt = datetime.datetime(2024, 1, 15, hour, 0, 0)
            schedule[dt] = ScheduleEntry(time=dt, mode=BatteryMode.CHARGE, reason="Late")
        for hour in [0, 1, 2]:
            dt = datetime.datetime(2024, 1, 16, hour, 0, 0)
            schedule[dt] = ScheduleEntry(time=dt, mode=BatteryMode.DISCHARGE, reason="Early")

        periods = tou.schedule_to_tou_periods(schedule)

        # Periods should be sorted by start time (clock order)
        starts = [p.start for p in periods]
        assert starts == sorted(starts)

    def test_15min_fragmented_schedule_respects_limit(self):
        """A highly fragmented 15-min schedule should produce exactly 20 periods."""
        now = datetime.datetime(2024, 1, 15, 14, 0, 0)
        tou = _make_tou_sync(now, slot_minutes=15)

        # 96 slots alternating every 15 min
        schedule = {}
        for i in range(96):
            dt = datetime.datetime(2024, 1, 15, 0, 0, 0) + datetime.timedelta(minutes=15 * i)
            mode = BatteryMode.CHARGE if i % 2 == 0 else BatteryMode.HOLD
            schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=f"Slot {i}")

        periods = tou.schedule_to_tou_periods(schedule)
        assert len(periods) == 20

    def test_boundary_used_as_reference(self):
        """boundary_minute should be used as the forward walk start instead of now."""
        now = datetime.datetime(2024, 1, 15, 14, 0, 0)
        tou = _make_tou_sync(now, slot_minutes=15)

        # Alternating every 15 min
        schedule = {}
        for i in range(96):
            dt = datetime.datetime(2024, 1, 15, 0, 0, 0) + datetime.timedelta(minutes=15 * i)
            mode = BatteryMode.CHARGE if i % 2 == 0 else BatteryMode.DISCHARGE
            schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=f"S{i}")

        # boundary at 10:00 — walk should start from 10:00, not 14:00
        periods = tou.schedule_to_tou_periods(schedule, boundary_minute=600)

        covered = set()
        for p in periods:
            for m in range(p.start, p.end + 1):
                covered.add(m)

        # 10:00 (600) must be covered — it's the walk start
        assert 600 in covered
        # 10:15 (615) should also be covered
        assert 615 in covered

    def test_gap_in_schedule_creates_separate_periods(self):
        """Non-contiguous slots should produce separate periods with a gap."""
        now = datetime.datetime(2024, 1, 15, 10, 0, 0)
        tou = _make_tou_sync(now, slot_minutes=60)

        # Two clusters: 10:00-11:00 CHARGE, then 14:00-15:00 CHARGE (gap 12:00-13:00)
        schedule = {
            datetime.datetime(2024, 1, 15, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.CHARGE, reason="A"),
            datetime.datetime(2024, 1, 15, 11, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 11, 0, 0),
                mode=BatteryMode.CHARGE, reason="B"),
            datetime.datetime(2024, 1, 15, 14, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 14, 0, 0),
                mode=BatteryMode.CHARGE, reason="C"),
            datetime.datetime(2024, 1, 15, 15, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 15, 0, 0),
                mode=BatteryMode.CHARGE, reason="D"),
        }

        periods = tou.schedule_to_tou_periods(schedule)

        # HOLD pad + 2 separate CHARGE periods with a gap between them
        assert len(periods) == 3
        assert periods[0].power == 1    # Midnight HOLD pad
        assert periods[0].start == 0
        assert periods[1].power == 100  # First CHARGE cluster
        assert periods[2].power == 100  # Second CHARGE cluster
        # Gap: periods[1].end < periods[2].start
        assert periods[1].end < periods[2].start

    def test_period_1_starts_at_midnight(self):
        """Firmware requires period 1 to start at 00:00 — HOLD pad prepended."""
        now = datetime.datetime(2024, 1, 15, 10, 0, 0)
        tou = _make_tou_sync(now, slot_minutes=60)

        # Single CHARGE entry at 10:00 — period starts at 600, not 0
        schedule = {
            datetime.datetime(2024, 1, 15, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.CHARGE, reason="Morning"),
        }

        periods = tou.schedule_to_tou_periods(schedule)

        # First period must start at 00:00 (firmware requirement)
        assert periods[0].start == 0
        assert periods[0].power == 1  # HOLD pad
        assert periods[0].end == 599  # Ends just before the CHARGE period

        # CHARGE period follows
        assert periods[1].start == 600
        assert periods[1].power == 100

    def test_no_midnight_pad_when_already_at_zero(self):
        """No HOLD pad needed when first period already starts at 00:00."""
        now = datetime.datetime(2024, 1, 15, 0, 0, 0)
        tou = _make_tou_sync(now, slot_minutes=60)

        schedule = {}
        for hour in range(24):
            dt = datetime.datetime(2024, 1, 15, hour, 0, 0)
            mode = BatteryMode.CHARGE if hour < 6 else BatteryMode.HOLD
            schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=f"H{hour}")

        periods = tou.schedule_to_tou_periods(schedule)

        # Walk from 00:00 — first period should start at 0 naturally
        assert periods[0].start == 0
        # No extra HOLD pad: first real period IS at 0
        assert periods[0].power == 100  # CHARGE (0-5)

    def test_no_midnight_pad_when_at_max_periods(self):
        """Don't add HOLD pad if already at 20 periods — it would exceed the limit."""
        now = datetime.datetime(2024, 1, 15, 14, 0, 0)
        tou = _make_tou_sync(now, slot_minutes=15)

        # Alternating every 15 min: produces max 20 periods
        schedule = {}
        for i in range(96):
            dt = datetime.datetime(2024, 1, 15, 0, 0, 0) + datetime.timedelta(minutes=15 * i)
            mode = BatteryMode.CHARGE if i % 2 == 0 else BatteryMode.HOLD
            schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=f"S{i}")

        periods = tou.schedule_to_tou_periods(schedule)

        # Should be capped at 20 — no room for pad
        assert len(periods) == 20


class TestRollingTouInteraction:
    """Test check_and_sync_rolling_tou interaction with forward walk."""

    def test_stable_schedule_does_not_churn(self):
        """Rolling check should not trigger sync when schedule hasn't changed."""
        now = datetime.datetime(2024, 1, 15, 14, 0, 0)
        syncs_scheduled = []

        schedule = {}
        for hour in range(24):
            dt = datetime.datetime(2024, 1, 15, hour, 0, 0)
            mode = BatteryMode.CHARGE if hour < 6 else BatteryMode.HOLD
            schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=f"H{hour}")

        tou = TouSyncManager(
            device_id="test_device",
            slot_minutes=60,
            ha_url="",
            ha_token="",
            call_service_func=lambda *args, **kwargs: None,
            get_datetime_func=lambda: now,
            get_timezone_func=lambda: None,
            sleep_func=lambda x: None,
            create_task_func=lambda x: syncs_scheduled.append(x),
            log_func=lambda msg, level="INFO": None,
            get_schedule_func=lambda: schedule,
        )

        # Compute the periods that would be on the inverter
        existing_periods = tou.schedule_to_tou_periods(schedule, boundary_minute=840)

        # Mock read_current_tou_periods to return these exact periods
        tou.read_current_tou_periods = lambda: existing_periods

        # Rolling check should find no difference
        tou.check_and_sync_rolling_tou()

        # No sync should have been scheduled
        assert len(syncs_scheduled) == 0

    def test_changed_schedule_triggers_sync(self):
        """Rolling check should trigger sync when schedule differs from inverter."""
        now = datetime.datetime(2024, 1, 15, 14, 0, 0)
        syncs_scheduled = []

        schedule = {}
        for hour in range(24):
            dt = datetime.datetime(2024, 1, 15, hour, 0, 0)
            mode = BatteryMode.CHARGE if hour < 6 else BatteryMode.HOLD
            schedule[dt] = ScheduleEntry(time=dt, mode=mode, reason=f"H{hour}")

        tou = TouSyncManager(
            device_id="test_device",
            slot_minutes=60,
            ha_url="",
            ha_token="",
            call_service_func=lambda *args, **kwargs: None,
            get_datetime_func=lambda: now,
            get_timezone_func=lambda: None,
            sleep_func=lambda x: None,
            create_task_func=lambda x: syncs_scheduled.append(x),
            log_func=lambda msg, level="INFO": None,
            get_schedule_func=lambda: schedule,
        )

        # Simulate stale inverter: all DISCHARGE
        stale_periods = [TouPeriod(start=0, end=1439, power=-100)]
        tou.read_current_tou_periods = lambda: stale_periods

        # Rolling check should detect mismatch and schedule sync
        tou.check_and_sync_rolling_tou()

        assert len(syncs_scheduled) == 1

    def test_gap_schedule_stable_across_rolling_checks(self):
        """Schedule with gaps should produce same periods when re-computed at same time."""
        now = datetime.datetime(2024, 1, 15, 10, 0, 0)

        schedule = {
            datetime.datetime(2024, 1, 15, 10, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.CHARGE, reason="A"),
            datetime.datetime(2024, 1, 15, 11, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 11, 0, 0),
                mode=BatteryMode.CHARGE, reason="B"),
            datetime.datetime(2024, 1, 15, 14, 0, 0): ScheduleEntry(
                time=datetime.datetime(2024, 1, 15, 14, 0, 0),
                mode=BatteryMode.CHARGE, reason="C"),
        }

        tou = _make_tou_sync(now, slot_minutes=60)

        # Compute twice with same boundary — should produce identical results
        periods1 = tou.schedule_to_tou_periods(schedule, boundary_minute=600)
        periods2 = tou.schedule_to_tou_periods(schedule, boundary_minute=600)

        assert len(periods1) == len(periods2)
        for p1, p2 in zip(periods1, periods2):
            assert p1.start == p2.start
            assert p1.end == p2.end
            assert p1.power == p2.power
