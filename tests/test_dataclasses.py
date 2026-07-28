"""
Tests for dataclasses: BatteryMode, PricePoint, ScheduleEntry, config fields.
"""

import datetime

import pytest

from battery_optimizer import BatteryMode, PricePoint, ScheduleEntry
from battery_optimizer_lib.config import BatteryOptimizerConfig


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


class TestPvBiasConfigFields:
    """Config fields introduced for PV shortfall measurement and forecast bias."""

    def test_defaults(self):
        cfg = BatteryOptimizerConfig()
        assert cfg.pv_reactive_consecutive_slots == 2
        assert cfg.pv_reactive_min_samples == 3
        assert cfg.pv_sample_seconds == 60
        assert cfg.pv_bias_enabled is True
        assert cfg.pv_bias_window_minutes == 120
        assert cfg.pv_bias_min_slots == 2
        assert cfg.pv_bias_min_factor == pytest.approx(0.2)
        assert cfg.pv_bias_max_factor == pytest.approx(1.5)
        assert cfg.pv_bias_decay_slots == 8

    def test_from_args_parses_all_new_fields(self):
        cfg = BatteryOptimizerConfig.from_args({
            "pv_reactive_consecutive_slots": 4,
            "pv_reactive_min_samples": 5,
            "pv_sample_seconds": 30,
            "pv_bias_enabled": False,
            "pv_bias_window_minutes": 60,
            "pv_bias_min_slots": 3,
            "pv_bias_min_factor": 0.3,
            "pv_bias_max_factor": 1.2,
            "pv_bias_decay_slots": 4,
        })
        assert cfg.pv_reactive_consecutive_slots == 4
        assert cfg.pv_reactive_min_samples == 5
        assert cfg.pv_sample_seconds == 30
        assert cfg.pv_bias_enabled is False
        assert cfg.pv_bias_window_minutes == 60
        assert cfg.pv_bias_min_slots == 3
        assert cfg.pv_bias_min_factor == pytest.approx(0.3)
        assert cfg.pv_bias_max_factor == pytest.approx(1.2)
        assert cfg.pv_bias_decay_slots == 4

    def test_from_args_defaults_when_absent(self):
        cfg = BatteryOptimizerConfig.from_args({})
        assert cfg.pv_reactive_consecutive_slots == 2
        assert cfg.pv_reactive_min_samples == 3
        assert cfg.pv_sample_seconds == 60
        assert cfg.pv_bias_enabled is True

    def test_sample_seconds_clamped_to_slot_length(self):
        """A sample interval longer than a slot could never build a mean."""
        cfg = BatteryOptimizerConfig(slot_minutes=15, pv_sample_seconds=3600)
        assert cfg.pv_sample_seconds == 900
        cfg_low = BatteryOptimizerConfig(pv_sample_seconds=1)
        assert cfg_low.pv_sample_seconds == 10

    def test_bias_factor_bounds_are_ordered(self):
        cfg = BatteryOptimizerConfig(pv_bias_min_factor=0.8, pv_bias_max_factor=0.4)
        assert cfg.pv_bias_max_factor > cfg.pv_bias_min_factor
        assert cfg.pv_bias_max_factor == pytest.approx(0.81)

    def test_bias_min_factor_clamped_to_unit_range(self):
        assert BatteryOptimizerConfig(pv_bias_min_factor=-1.0).pv_bias_min_factor == 0.0
        assert BatteryOptimizerConfig(pv_bias_min_factor=5.0).pv_bias_min_factor == 1.0

    def test_bias_window_at_least_one_slot(self):
        cfg = BatteryOptimizerConfig(slot_minutes=30, pv_bias_window_minutes=5)
        assert cfg.pv_bias_window_minutes == 30

    def test_counters_have_a_floor_of_one(self):
        cfg = BatteryOptimizerConfig(
            pv_reactive_consecutive_slots=0,
            pv_reactive_min_samples=0,
            pv_bias_min_slots=0,
            pv_bias_decay_slots=0,
        )
        assert cfg.pv_reactive_consecutive_slots == 1
        assert cfg.pv_reactive_min_samples == 1
        assert cfg.pv_bias_min_slots == 1
        assert cfg.pv_bias_decay_slots == 1

    def test_log_summary_mentions_bias(self):
        lines = []
        BatteryOptimizerConfig().log_summary(lambda msg: lines.append(msg))
        assert any("PV bias:" in line for line in lines)
        assert any("consecutive_slots=2" in line for line in lines)
