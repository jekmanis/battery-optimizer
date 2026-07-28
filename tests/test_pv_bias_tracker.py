"""
Tests for PvBiasTracker — slot-energy PV sampling and sliding forecast bias.

These cover the two production defects from the 33-hour AppDaemon log window
(2026-07-27 07:30 .. 07-28 16:08):

DEFECT 3 — the reactive shortfall trigger compared ONE instantaneous PV reading
taken 5 s after a slot boundary against the slot AVERAGE forecast, producing 43
"PV below forecast" events in 33 hours.

DEFECT 4 — the correction was applied to the current slot only, so each of the
43 recalculations re-planned the remaining horizon on the same 3-5x optimistic
Solcast forecast and emitted a near-identical schedule ~20 times.
"""

import datetime

import pytest

from battery_optimizer_lib.config import BatteryOptimizerConfig
from battery_optimizer_lib.pv_bias_tracker import (
    ClosedSlot,
    PvBiasConfig,
    PvBiasTracker,
    _median,
)
from battery_optimizer_lib.timezone_utils import align_to_slot, canonical_slot_key


TZ_PLUS2 = datetime.timezone(datetime.timedelta(hours=2))
TZ_PLUS3 = datetime.timezone(datetime.timedelta(hours=3))


def _make_tracker(**overrides) -> PvBiasTracker:
    """Create a tracker with a fixed +02:00 slot alignment."""
    cfg_kwargs = dict(
        enabled=True,
        slot_minutes=15,
        window_minutes=120,
        min_slots=2,
        min_forecast_kw=0.2,
        min_factor=0.2,
        max_factor=1.5,
        decay_slots=8,
        min_samples_per_slot=3,
        shortfall_threshold=0.5,
    )
    cfg_kwargs.update(overrides)
    config = PvBiasConfig(**cfg_kwargs)
    return PvBiasTracker(
        config=config,
        align_to_slot_func=lambda dt: align_to_slot(dt, config.slot_minutes, TZ_PLUS2),
        log_func=lambda msg, **kw: None,
    )


def _dt(hour, minute=0, second=0, day=27):
    return datetime.datetime(2026, 7, day, hour, minute, second, tzinfo=TZ_PLUS2)


def _fill_slot(tracker, slot, forecast_kw, actual_kw, samples=15):
    """Snapshot a forecast and add *samples* readings of *actual_kw*."""
    tracker.ensure_slot_forecast(slot, forecast_kw)
    for i in range(samples):
        tracker.add_sample(slot + datetime.timedelta(seconds=30 * i), actual_kw)


# ---------------------------------------------------------------------------
# Sampling — slot ENERGY instead of one instantaneous reading (DEFECT 3)
# ---------------------------------------------------------------------------

class TestSampling:

    def test_slot_mean_is_average_of_samples(self):
        """The slot value is the MEAN of many samples, not the last reading."""
        t = _make_tracker()
        slot = _dt(15, 15)
        # Real log shape: 457 W reported at the boundary, but the slot really
        # produced ~3.3 kW (SOC 84%->100% in 51 min plus ~0.6 kW house load).
        t.add_sample(_dt(15, 15, 5), 0.457)
        for i in range(1, 15):
            t.add_sample(slot + datetime.timedelta(minutes=i), 3.3)

        mean = t.slot_mean_kw(slot)
        assert mean == pytest.approx((0.457 + 14 * 3.3) / 15)
        # The instantaneous boundary read would have been 0.457 kW.
        assert mean > 3.0
        assert t.slot_sample_count(slot) == 15

    def test_empty_slot_has_no_mean(self):
        t = _make_tracker()
        assert t.slot_mean_kw(_dt(12)) is None
        assert t.slot_sample_count(_dt(12)) == 0

    def test_negative_samples_are_ignored(self):
        t = _make_tracker()
        slot = _dt(12)
        t.add_sample(slot, -500.0)
        t.add_sample(slot + datetime.timedelta(minutes=1), 2.0)
        assert t.slot_sample_count(slot) == 1
        assert t.slot_mean_kw(slot) == pytest.approx(2.0)

    def test_non_numeric_sample_is_ignored(self):
        t = _make_tracker()
        slot = _dt(12)
        t.add_sample(slot, None)
        t.add_sample(slot, "unavailable")
        assert t.slot_sample_count(slot) == 0

    def test_samples_land_in_their_own_slot(self):
        t = _make_tracker()
        t.add_sample(_dt(12, 14, 59), 1.0)
        t.add_sample(_dt(12, 15, 1), 5.0)
        assert t.slot_mean_kw(_dt(12, 0)) == pytest.approx(1.0)
        assert t.slot_mean_kw(_dt(12, 15)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Forecast snapshot — first write wins (guards against self-erasing bias)
# ---------------------------------------------------------------------------

class TestForecastSnapshot:

    def test_first_forecast_wins(self):
        """refresh_for_shortfall caps the cache; the snapshot must not follow."""
        t = _make_tracker()
        slot = _dt(15, 15)
        assert t.ensure_slot_forecast(slot, 2.891) == pytest.approx(2.891)
        # Second (capped) read — as PvForecastService would report after
        # refresh_for_shortfall wrote min(forecast, observed).
        assert t.ensure_slot_forecast(slot, 0.457) == pytest.approx(2.891)
        assert t.get_slot_forecast(slot) == pytest.approx(2.891)

    def test_capped_forecast_would_erase_the_signal(self):
        """With a capped snapshot the ratio would be ~1.0 — the bug we avoid."""
        t = _make_tracker()
        slot = _dt(15, 15)
        t.ensure_slot_forecast(slot, 2.891)
        t.ensure_slot_forecast(slot, 0.457)  # ignored
        for i in range(15):
            t.add_sample(slot + datetime.timedelta(seconds=30 * i), 0.457)
        closed = t.close_slots_before(_dt(15, 30))
        assert len(closed) == 1
        assert closed[0].ratio == pytest.approx(0.457 / 2.891, abs=1e-6)
        assert closed[0].ratio < 0.2

    def test_negative_forecast_is_floored(self):
        t = _make_tracker()
        assert t.ensure_slot_forecast(_dt(12), -1.0) == 0.0

    def test_unknown_slot_has_no_forecast(self):
        t = _make_tracker()
        assert t.get_slot_forecast(_dt(12)) is None


# ---------------------------------------------------------------------------
# Slot closing
# ---------------------------------------------------------------------------

class TestCloseSlots:

    def test_close_is_idempotent(self):
        t = _make_tracker()
        _fill_slot(t, _dt(12, 0), 3.0, 1.0)
        _fill_slot(t, _dt(12, 15), 3.0, 1.0)

        closed = t.close_slots_before(_dt(12, 30))
        assert [c.slot.hour * 60 + c.slot.minute for c in closed] == [720, 735]
        # Second call returns nothing new.
        assert t.close_slots_before(_dt(12, 30)) == []

    def test_current_slot_is_not_closed(self):
        t = _make_tracker()
        _fill_slot(t, _dt(12, 30), 3.0, 1.0)
        assert t.close_slots_before(_dt(12, 30)) == []
        assert t.slot_sample_count(_dt(12, 30)) == 15

    def test_closed_slot_reports_mean_and_ratio(self):
        t = _make_tracker()
        _fill_slot(t, _dt(12, 0), 4.0, 1.2, samples=12)
        closed = t.close_slots_before(_dt(12, 15))
        assert len(closed) == 1
        entry = closed[0]
        assert isinstance(entry, ClosedSlot)
        assert entry.forecast_kw == pytest.approx(4.0)
        assert entry.actual_kw == pytest.approx(1.2)
        assert entry.samples == 12
        assert entry.ratio == pytest.approx(0.3)
        assert t.get_closed(_dt(12, 0)) is entry

    def test_zero_forecast_gives_zero_ratio(self):
        t = _make_tracker()
        _fill_slot(t, _dt(2, 0), 0.0, 0.0)
        closed = t.close_slots_before(_dt(2, 15))
        assert closed[0].ratio == 0.0

    def test_slot_with_forecast_but_no_samples_closes_with_zero_samples(self):
        """Sensor unavailable for a whole slot: must not become a fake 0 W obs."""
        t = _make_tracker()
        t.ensure_slot_forecast(_dt(16, 45), 2.022)
        closed = t.close_slots_before(_dt(17, 0))
        assert len(closed) == 1
        assert closed[0].samples == 0
        # Untrustworthy — no ratio recorded, streak untouched.
        assert t.ratio_count(_dt(17, 0)) == 0
        assert t.shortfall_streak == 0

    def test_ratio_recorded_only_when_forecast_and_samples_suffice(self):
        t = _make_tracker(min_forecast_kw=0.2, min_samples_per_slot=3)
        # Below the forecast floor — no ratio
        _fill_slot(t, _dt(5, 0), 0.05, 0.0, samples=15)
        # Enough forecast but too few samples — no ratio
        _fill_slot(t, _dt(5, 15), 3.0, 0.5, samples=2)
        # Trustworthy — ratio recorded
        _fill_slot(t, _dt(5, 30), 3.0, 0.9, samples=15)
        t.close_slots_before(_dt(5, 45))
        assert t.ratio_count(_dt(5, 45)) == 1

    def test_closed_history_is_bounded(self):
        t = _make_tracker()
        for i in range(20):
            _fill_slot(t, _dt(6, 0) + datetime.timedelta(minutes=15 * i), 3.0, 1.0)
        t.close_slots_before(_dt(6, 0) + datetime.timedelta(minutes=15 * 20))
        assert t.get_closed(_dt(6, 0)) is None  # evicted
        assert t.get_closed(_dt(6, 0) + datetime.timedelta(minutes=15 * 19)) is not None


# ---------------------------------------------------------------------------
# Consecutive-shortfall streak (part (a) of the fix)
# ---------------------------------------------------------------------------

class TestShortfallStreak:

    def test_two_consecutive_shortfalls(self):
        t = _make_tracker()
        _fill_slot(t, _dt(11, 0), 3.0, 0.9)   # ratio 0.30
        t.close_slots_before(_dt(11, 15))
        assert t.shortfall_streak == 1
        _fill_slot(t, _dt(11, 15), 3.0, 0.6)  # ratio 0.20
        t.close_slots_before(_dt(11, 30))
        assert t.shortfall_streak == 2

    def test_good_slot_resets_streak(self):
        t = _make_tracker()
        _fill_slot(t, _dt(11, 0), 3.0, 0.9)
        t.close_slots_before(_dt(11, 15))
        assert t.shortfall_streak == 1
        _fill_slot(t, _dt(11, 15), 3.0, 2.8)  # ratio 0.93
        t.close_slots_before(_dt(11, 30))
        assert t.shortfall_streak == 0

    def test_untrusted_slot_does_not_change_streak(self):
        t = _make_tracker(min_samples_per_slot=3)
        _fill_slot(t, _dt(11, 0), 3.0, 0.9)
        t.close_slots_before(_dt(11, 15))
        assert t.shortfall_streak == 1
        # Only 1 sample — cannot confirm or deny
        _fill_slot(t, _dt(11, 15), 3.0, 3.0, samples=1)
        t.close_slots_before(_dt(11, 30))
        assert t.shortfall_streak == 1

    def test_night_slot_resets_streak(self):
        t = _make_tracker()
        _fill_slot(t, _dt(20, 0), 3.0, 0.5)
        t.close_slots_before(_dt(20, 15))
        assert t.shortfall_streak == 1
        _fill_slot(t, _dt(20, 15), 0.02, 0.0)  # below min_forecast_kw
        t.close_slots_before(_dt(20, 30))
        assert t.shortfall_streak == 0

    def test_single_boundary_dip_does_not_reach_default_threshold(self):
        """DEFECT 3: one dip must not be enough for a full recalculation."""
        t = _make_tracker()
        # Morning ramp: the slot really produced 2.7 kW of a 3.0 kW forecast,
        # even though the 00:05 boundary sample read 0.2 kW.
        t.ensure_slot_forecast(_dt(9, 0), 3.0)
        t.add_sample(_dt(9, 0, 5), 0.2)
        for i in range(1, 15):
            t.add_sample(_dt(9, 0) + datetime.timedelta(minutes=i), 2.9)
        t.close_slots_before(_dt(9, 15))
        assert t.shortfall_streak == 0
        assert t.get_closed(_dt(9, 0)).ratio > 0.5


# ---------------------------------------------------------------------------
# Bias factor (part (b) of the fix — DEFECT 4)
# ---------------------------------------------------------------------------

class TestBiasFactor:

    def test_below_min_slots_returns_one(self):
        t = _make_tracker(min_slots=2)
        _fill_slot(t, _dt(11, 0), 3.0, 0.9)
        t.close_slots_before(_dt(11, 15))
        assert t.ratio_count(_dt(11, 15)) == 1
        assert t.get_factor(_dt(11, 15)) == 1.0

    def test_median_of_ratios(self):
        t = _make_tracker(min_slots=2)
        for i, actual in enumerate([0.9, 0.75, 1.05]):  # ratios 0.30/0.25/0.35
            _fill_slot(t, _dt(11, 0) + datetime.timedelta(minutes=15 * i), 3.0, actual)
        t.close_slots_before(_dt(11, 45))
        assert t.get_factor(_dt(11, 45)) == pytest.approx(0.3, abs=1e-3)

    def test_clamped_at_lower_bound(self):
        t = _make_tracker(min_slots=2, min_factor=0.2)
        for i in range(3):
            _fill_slot(t, _dt(11, 0) + datetime.timedelta(minutes=15 * i), 3.0, 0.15)
        t.close_slots_before(_dt(11, 45))
        assert t.get_factor(_dt(11, 45)) == pytest.approx(0.2)

    def test_clamped_at_upper_bound(self):
        t = _make_tracker(min_slots=2, max_factor=1.5)
        for i in range(3):
            _fill_slot(t, _dt(11, 0) + datetime.timedelta(minutes=15 * i), 1.0, 3.0)
        t.close_slots_before(_dt(11, 45))
        assert t.get_factor(_dt(11, 45)) == pytest.approx(1.5)

    def test_window_pruning_drops_old_ratios(self):
        t = _make_tracker(min_slots=2, window_minutes=120)
        for i in range(3):
            _fill_slot(t, _dt(8, 0) + datetime.timedelta(minutes=15 * i), 3.0, 0.9)
        t.close_slots_before(_dt(8, 45))
        assert t.ratio_count(_dt(8, 45)) == 3
        # 3 hours later everything is out of the 120-minute window
        assert t.ratio_count(_dt(11, 45)) == 0
        assert t.get_factor(_dt(11, 45)) == 1.0

    def test_factor_relaxes_towards_one(self):
        t = _make_tracker(min_slots=2, decay_slots=8, window_minutes=600)
        for i in range(3):
            _fill_slot(t, _dt(11, 0) + datetime.timedelta(minutes=15 * i), 3.0, 0.9)
        t.close_slots_before(_dt(11, 45))
        fresh = t.get_factor(_dt(11, 45))
        assert fresh == pytest.approx(0.3, abs=1e-3)
        # 4 slots later: partially relaxed
        mid = t.get_factor(_dt(12, 30))
        assert fresh < mid < 1.0
        # 9+ slots after the newest observation: fully relaxed
        assert t.get_factor(_dt(14, 0)) == pytest.approx(1.0)

    def test_disabled_always_returns_one(self):
        t = _make_tracker(enabled=False, min_slots=2)
        for i in range(3):
            _fill_slot(t, _dt(11, 0) + datetime.timedelta(minutes=15 * i), 3.0, 0.9)
        t.close_slots_before(_dt(11, 45))
        assert t.get_factor(_dt(11, 45)) == 1.0
        assert t.describe(_dt(11, 45)) == "disabled"

    def test_describe_reports_evidence(self):
        t = _make_tracker(min_slots=2)
        assert t.describe(_dt(11, 45)) == "no observations"
        for i in range(3):
            _fill_slot(t, _dt(11, 0) + datetime.timedelta(minutes=15 * i), 3.0, 0.9)
        t.close_slots_before(_dt(11, 45))
        text = t.describe(_dt(11, 45))
        assert "median of 3 slots" in text
        assert "120min" in text
        assert "15min ago" in text

    def test_log_scenario_bias_scales_whole_horizon(self):
        """DEFECT 4: the 07-27 shortfall must scale FUTURE slots, not just one.

        Log: 15:15 forecast 2891 W vs ~457 W measured; 15:30 and 15:45 kept the
        untouched 2076 W Solcast value.  With the bias factor those later slots
        collapse to ~0.3x.
        """
        t = _make_tracker(min_slots=2)
        _fill_slot(t, _dt(14, 45), 2.891, 0.90)   # ratio 0.311
        _fill_slot(t, _dt(15, 0), 2.891, 0.85)    # ratio 0.294
        _fill_slot(t, _dt(15, 15), 2.891, 0.87)   # ratio 0.301
        t.close_slots_before(_dt(15, 30))

        factor = t.get_factor(_dt(15, 30))
        assert 0.25 < factor < 0.35
        # Untouched provider value for the *next* slots
        raw_1530 = raw_1545 = 2.076
        assert raw_1530 * factor < 0.8
        assert raw_1545 * factor < 0.8


# ---------------------------------------------------------------------------
# DST safety
# ---------------------------------------------------------------------------

class TestDstSafety:

    def test_autumn_fold_slots_stay_distinct(self):
        """The two 03:00 local slots are different physical intervals."""

        def align(dt):
            return dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)

        config = PvBiasConfig(slot_minutes=15, min_slots=1, min_samples_per_slot=1)
        t = PvBiasTracker(config=config, align_to_slot_func=align)

        first = datetime.datetime(2024, 10, 27, 3, 0, tzinfo=TZ_PLUS3)   # summer
        second = datetime.datetime(2024, 10, 27, 3, 0, tzinfo=TZ_PLUS2)  # winter
        assert canonical_slot_key(first) != canonical_slot_key(second)

        t.ensure_slot_forecast(first, 1.0)
        t.ensure_slot_forecast(second, 2.0)
        t.add_sample(first, 0.5)
        t.add_sample(second, 1.5)

        assert t.get_slot_forecast(first) == pytest.approx(1.0)
        assert t.get_slot_forecast(second) == pytest.approx(2.0)
        assert t.slot_mean_kw(first) == pytest.approx(0.5)
        assert t.slot_mean_kw(second) == pytest.approx(1.5)

        # Closing at 04:00 winter time retires both, in instant order.
        closed = t.close_slots_before(
            datetime.datetime(2024, 10, 27, 4, 0, tzinfo=TZ_PLUS2)
        )
        assert len(closed) == 2
        assert closed[0].forecast_kw == pytest.approx(1.0)
        assert closed[1].forecast_kw == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Config bridging
# ---------------------------------------------------------------------------

class TestPvBiasConfigFromMainConfig:

    def test_values_are_carried_over(self):
        cfg = BatteryOptimizerConfig(
            slot_minutes=15,
            pv_bias_enabled=True,
            pv_bias_window_minutes=90,
            pv_bias_min_slots=3,
            pv_bias_min_factor=0.25,
            pv_bias_max_factor=1.4,
            pv_bias_decay_slots=6,
            pv_reactive_min_forecast_w=250.0,
            pv_reactive_min_samples=4,
            pv_reactive_threshold=0.6,
        )
        bias_cfg = PvBiasConfig.from_main_config(cfg)
        assert bias_cfg.enabled is True
        assert bias_cfg.slot_minutes == 15
        assert bias_cfg.window_minutes == 90
        assert bias_cfg.min_slots == 3
        assert bias_cfg.min_factor == pytest.approx(0.25)
        assert bias_cfg.max_factor == pytest.approx(1.4)
        assert bias_cfg.decay_slots == 6
        assert bias_cfg.min_forecast_kw == pytest.approx(0.25)
        assert bias_cfg.min_samples_per_slot == 4
        assert bias_cfg.shortfall_threshold == pytest.approx(0.6)

    def test_disabled_flag_is_carried_over(self):
        cfg = BatteryOptimizerConfig(pv_bias_enabled=False)
        assert PvBiasConfig.from_main_config(cfg).enabled is False


class TestMedianHelper:

    def test_odd_and_even(self):
        assert _median([3.0, 1.0, 2.0]) == 2.0
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
        assert _median([]) == 0.0
