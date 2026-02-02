"""
Tests for dataclasses: BatteryMode, PricePoint, ScheduleEntry, TouPeriod.
"""

import datetime

import pytest

from battery_optimizer import BatteryMode, PricePoint, ScheduleEntry, TouPeriod


class TestBatteryMode:
    """Test cases for BatteryMode enum."""

    def test_enum_values(self):
        """Enum values should match expected integers."""
        assert BatteryMode.HOLD.value == 0
        assert BatteryMode.CHARGE.value == 1
        assert BatteryMode.DISCHARGE.value == 2

    def test_enum_from_value(self):
        """Should be able to create from integer value."""
        assert BatteryMode(0) == BatteryMode.HOLD
        assert BatteryMode(1) == BatteryMode.CHARGE
        assert BatteryMode(2) == BatteryMode.DISCHARGE

    def test_enum_names(self):
        """Enum names should be correct."""
        assert BatteryMode.HOLD.name == "HOLD"
        assert BatteryMode.CHARGE.name == "CHARGE"
        assert BatteryMode.DISCHARGE.name == "DISCHARGE"

    def test_enum_comparison(self):
        """Enum comparison should work."""
        assert BatteryMode.CHARGE == BatteryMode.CHARGE
        assert BatteryMode.CHARGE != BatteryMode.DISCHARGE

    def test_enum_iteration(self):
        """Should be able to iterate over all modes."""
        modes = list(BatteryMode)
        assert len(modes) == 3
        assert BatteryMode.HOLD in modes
        assert BatteryMode.CHARGE in modes
        assert BatteryMode.DISCHARGE in modes


class TestPricePoint:
    """Test cases for PricePoint dataclass."""

    def test_basic_creation(self):
        """Basic creation should work."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        pp = PricePoint(hour=dt, price=0.15)

        assert pp.hour == dt
        assert pp.price == 0.15

    def test_negative_price(self):
        """Negative prices should be supported."""
        dt = datetime.datetime(2024, 1, 15, 2, 0, 0)
        pp = PricePoint(hour=dt, price=-0.02)

        assert pp.price == -0.02

    def test_zero_price(self):
        """Zero price should be supported."""
        dt = datetime.datetime(2024, 1, 15, 2, 0, 0)
        pp = PricePoint(hour=dt, price=0.0)

        assert pp.price == 0.0


class TestScheduleEntry:
    """Test cases for ScheduleEntry dataclass."""

    def test_basic_creation(self):
        """Basic creation should work."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        entry = ScheduleEntry(
            hour=dt,
            mode=BatteryMode.CHARGE,
            reason="Cheap electricity"
        )

        assert entry.hour == dt
        assert entry.mode == BatteryMode.CHARGE
        assert entry.reason == "Cheap electricity"

    def test_all_modes(self):
        """Should work with all battery modes."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        hold = ScheduleEntry(hour=dt, mode=BatteryMode.HOLD, reason="Hold")
        charge = ScheduleEntry(hour=dt, mode=BatteryMode.CHARGE, reason="Charge")
        discharge = ScheduleEntry(hour=dt, mode=BatteryMode.DISCHARGE, reason="Discharge")

        assert hold.mode == BatteryMode.HOLD
        assert charge.mode == BatteryMode.CHARGE
        assert discharge.mode == BatteryMode.DISCHARGE

    def test_reason_with_price(self):
        """Reason can include formatted price info."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        entry = ScheduleEntry(
            hour=dt,
            mode=BatteryMode.CHARGE,
            reason=f"Charge @ {0.0523:.4f} EUR/kWh"
        )

        assert "0.0523" in entry.reason
        assert "EUR/kWh" in entry.reason


class TestTouPeriod:
    """Test cases for TouPeriod dataclass."""

    def test_basic_creation(self):
        """Basic creation should work."""
        period = TouPeriod(start=0, end=359, power=100)

        assert period.start == 0
        assert period.end == 359
        assert period.power == 100

    def test_full_day_coverage(self):
        """Periods can cover full day (0-1439 minutes)."""
        period = TouPeriod(start=0, end=1439, power=50)

        assert period.start == 0
        assert period.end == 1439  # 23:59

    def test_charge_power(self):
        """Positive power means charge."""
        period = TouPeriod(start=0, end=60, power=100)
        assert period.power > 0  # Charge

    def test_discharge_power(self):
        """Negative power means discharge."""
        period = TouPeriod(start=0, end=60, power=-100)
        assert period.power < 0  # Discharge

    def test_hold_power(self):
        """Small positive power (1%) is used for hold."""
        period = TouPeriod(start=0, end=60, power=1)
        assert period.power == 1  # Hold (firmware quirk)

    def test_time_conversion_examples(self):
        """Test common time-to-minutes conversions."""
        # 00:00 = 0 minutes
        assert 0 == 0 * 60 + 0

        # 06:00 = 360 minutes
        assert 360 == 6 * 60 + 0

        # 12:00 = 720 minutes
        assert 720 == 12 * 60 + 0

        # 18:00 = 1080 minutes
        assert 1080 == 18 * 60 + 0

        # 23:59 = 1439 minutes
        assert 1439 == 23 * 60 + 59

    def test_non_overlapping_periods(self):
        """Adjacent periods should not overlap."""
        period1 = TouPeriod(start=0, end=359, power=100)  # 00:00-05:59
        period2 = TouPeriod(start=360, end=719, power=-100)  # 06:00-11:59

        # End of period1 should be less than start of period2
        assert period1.end < period2.start

    def test_period_duration(self):
        """Period duration calculation."""
        period = TouPeriod(start=0, end=59, power=100)  # 1 hour

        duration_minutes = period.end - period.start + 1
        assert duration_minutes == 60
