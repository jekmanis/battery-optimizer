"""
Tests for dataclasses: BatteryMode, PricePoint, ScheduleEntry.
"""

import datetime

import pytest

from battery_optimizer import BatteryMode, PricePoint, ScheduleEntry


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
        pp = PricePoint(time=dt, price=0.15)

        assert pp.time == dt
        assert pp.price == 0.15

    def test_negative_price(self):
        """Negative prices should be supported."""
        dt = datetime.datetime(2024, 1, 15, 2, 0, 0)
        pp = PricePoint(time=dt, price=-0.02)

        assert pp.price == -0.02

    def test_zero_price(self):
        """Zero price should be supported."""
        dt = datetime.datetime(2024, 1, 15, 2, 0, 0)
        pp = PricePoint(time=dt, price=0.0)

        assert pp.price == 0.0


class TestScheduleEntry:
    """Test cases for ScheduleEntry dataclass."""

    def test_basic_creation(self):
        """Basic creation should work."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        entry = ScheduleEntry(
            time=dt,
            mode=BatteryMode.CHARGE,
            reason="Cheap electricity"
        )

        assert entry.time == dt
        assert entry.mode == BatteryMode.CHARGE
        assert entry.reason == "Cheap electricity"

    def test_all_modes(self):
        """Should work with all battery modes."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)

        hold = ScheduleEntry(time=dt, mode=BatteryMode.HOLD, reason="Hold")
        charge = ScheduleEntry(time=dt, mode=BatteryMode.CHARGE, reason="Charge")
        discharge = ScheduleEntry(time=dt, mode=BatteryMode.DISCHARGE, reason="Discharge")

        assert hold.mode == BatteryMode.HOLD
        assert charge.mode == BatteryMode.CHARGE
        assert discharge.mode == BatteryMode.DISCHARGE

    def test_reason_with_price(self):
        """Reason can include formatted price info."""
        dt = datetime.datetime(2024, 1, 15, 10, 0, 0)
        entry = ScheduleEntry(
            time=dt,
            mode=BatteryMode.CHARGE,
            reason=f"Charge @ {0.0523:.4f} EUR/kWh"
        )

        assert "0.0523" in entry.reason
        assert "EUR/kWh" in entry.reason
