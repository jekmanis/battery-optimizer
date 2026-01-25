"""
Tests for schedule_to_tou_periods conversion.

These tests verify the conversion of schedule entries to TOU periods
for inverter programming.
"""

import datetime
from typing import Dict, List

import pytest

from battery_optimizer import BatteryMode, ScheduleEntry, TouPeriod


class MockTouOptimizer:
    """
    Minimal mock for testing schedule_to_tou_periods.
    """

    def __init__(self):
        self.schedule: Dict[datetime.datetime, ScheduleEntry] = {}
        self._current_date = datetime.date(2024, 1, 15)
        self._current_datetime = datetime.datetime(2024, 1, 15, 12, 0, 0)
        self.slot_minutes = 60  # Default hourly slots

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
                hour=datetime.datetime(2024, 1, 15, 2, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, 18, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, 12, 0, 0),
                mode=BatteryMode.HOLD,
                reason="Hold"
            )
        }

        periods = optimizer.schedule_to_tou_periods()

        hold_periods = [p for p in periods if p.power == 1]
        assert len(hold_periods) == 1

    def test_contiguous_same_mode_consolidated(self, optimizer):
        """Contiguous hours with same mode should consolidate."""
        base_dt = datetime.datetime(2024, 1, 15, 2, 0, 0)
        optimizer.schedule = {
            base_dt: ScheduleEntry(hour=base_dt, mode=BatteryMode.CHARGE, reason="C1"),
            base_dt + datetime.timedelta(hours=1): ScheduleEntry(
                hour=base_dt + datetime.timedelta(hours=1),
                mode=BatteryMode.CHARGE,
                reason="C2"
            ),
            base_dt + datetime.timedelta(hours=2): ScheduleEntry(
                hour=base_dt + datetime.timedelta(hours=2),
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
            base_dt: ScheduleEntry(hour=base_dt, mode=BatteryMode.CHARGE, reason="C"),
            base_dt + datetime.timedelta(hours=1): ScheduleEntry(
                hour=base_dt + datetime.timedelta(hours=1),
                mode=BatteryMode.HOLD,
                reason="H"
            ),
            base_dt + datetime.timedelta(hours=2): ScheduleEntry(
                hour=base_dt + datetime.timedelta(hours=2),
                mode=BatteryMode.DISCHARGE,
                reason="D"
            ),
        }

        periods = optimizer.schedule_to_tou_periods()

        # Should have 3 separate periods
        assert len(periods) == 3

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
            optimizer.schedule[dt] = ScheduleEntry(hour=dt, mode=mode, reason=f"Mode {i}")

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
                hour=datetime.datetime(2024, 1, 15, 6, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, 22, 0, 0),
                mode=BatteryMode.HOLD,
                reason="Tonight"
            ),
            datetime.datetime(2024, 1, 16, 2, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 16, 2, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.CHARGE,  # Today: charge
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 10, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 16, 10, 0, 0),
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
                hour=datetime.datetime(2024, 1, 14, 10, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Yesterday"
            ),
            # Today - should be included
            datetime.datetime(2024, 1, 15, 10, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 15, 10, 0, 0),
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
            optimizer.schedule[dt] = ScheduleEntry(hour=dt, mode=mode, reason=f"Hour {i}")

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
                hour=dt, mode=BatteryMode.HOLD, reason=f"Hour {i}"
            )

        periods = optimizer.schedule_to_tou_periods()

        # Should consolidate to 1 period
        assert len(periods) == 1

        # Should cover full day
        assert periods[0].start == 0
        assert periods[0].end == 1439  # 23:59


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
                hour=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 10, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 16, 10, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, 6, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 6, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 16, 6, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, 18, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 18, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 16, 18, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, 14, 0, 0),
                mode=BatteryMode.HOLD,
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 14, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 16, 14, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, 6, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today morning"
            ),
            datetime.datetime(2024, 1, 15, 18, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 15, 18, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today evening"
            ),
            datetime.datetime(2024, 1, 16, 6, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 16, 6, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow morning"
            ),
            datetime.datetime(2024, 1, 16, 18, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 16, 18, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow evening"
            ),
        }

        # Boundary at 0 - all hours are >= 0, so all use today
        periods = optimizer.schedule_to_tou_periods(boundary_minute=0)

        # All periods should be CHARGE (today's schedule)
        for period in periods:
            assert period.power == 100  # Today's CHARGE

    def test_boundary_end_of_day_uses_tomorrow_for_all(self, optimizer):
        """With boundary=1440, all hours use tomorrow's schedule."""
        today = datetime.date(2024, 1, 15)
        optimizer._current_date = today

        # Both days have entries
        optimizer.schedule = {
            datetime.datetime(2024, 1, 15, 10, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 15, 10, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today"
            ),
            datetime.datetime(2024, 1, 16, 10, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 16, 10, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, hour, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today"
            )
            # Tomorrow: DISCHARGE for all hours
            optimizer.schedule[datetime.datetime(2024, 1, 16, hour, 0, 0)] = ScheduleEntry(
                hour=datetime.datetime(2024, 1, 16, hour, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, 10, 0, 0),
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
                hour=datetime.datetime(2024, 1, 16, 10, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, 18, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Today evening"
            ),
            datetime.datetime(2024, 1, 16, 6, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 16, 6, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Tomorrow morning"
            ),
        }

        # Boundary at 12:00
        periods = optimizer.schedule_to_tou_periods(boundary_minute=720)

        # Both entries should appear
        assert len(periods) == 2

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
                hour=datetime.datetime(2024, 1, 15, 23, 0, 0),
                mode=BatteryMode.CHARGE,
                reason="Late night"
            ),
            datetime.datetime(2024, 1, 15, 0, 0, 0): ScheduleEntry(
                hour=datetime.datetime(2024, 1, 15, 0, 0, 0),
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
                hour=datetime.datetime(2024, 1, 15, 0, 0, 0),
                mode=BatteryMode.DISCHARGE,
                reason="Brief"
            )
        }

        periods = optimizer.schedule_to_tou_periods()

        # Should still create valid period
        assert len(periods) >= 1
        assert all(p.end >= p.start for p in periods)
