"""
Orchestrator-level tests for the reactive PV shortfall path and bias application.

These bind the real ``BatteryOptimizer`` methods onto a lightweight mock (the
same pattern ``test_soc_deviation.py`` uses) so the two production defects are
covered end-to-end:

DEFECT 3 — the trigger compared ONE instantaneous reading taken 5 s after the
slot boundary against the slot AVERAGE forecast and recalculated immediately.
Every test in ``TestReactiveShortfallTrigger`` fails against that behaviour.

DEFECT 4 — the correction applied to the current slot only.  The tests in
``TestBiasAppliedToHorizon`` fail against the old ``_predict_pv_kw``, which had
no bias multiplier at all.
"""

import datetime
import sys
from pathlib import Path

import pytest

apps_dir = Path(__file__).parent.parent / "appdaemon" / "apps"
sys.path.insert(0, str(apps_dir))

from battery_optimizer import BatteryOptimizer
from battery_optimizer_lib import (
    BatteryOptimizerConfig,
    PvBiasConfig,
    PvBiasTracker,
)
from battery_optimizer_lib.timezone_utils import align_to_slot


TZ = datetime.timezone(datetime.timedelta(hours=3))


def _dt(hour, minute=0, second=0):
    return datetime.datetime(2026, 7, 27, hour, minute, second, tzinfo=TZ)


class _FakeForecastService:
    def __init__(self):
        self.shortfall_calls = []

    def refresh_for_shortfall(self, dt, actual_kw):
        self.shortfall_calls.append((dt, actual_kw))
        return False


class MockPvOptimizer:
    """Minimal host for the real PV bias / shortfall methods."""

    def __init__(self, now=None, raw_forecast_kw=2.891, pv_power_w=457.0, **cfg):
        self.config = BatteryOptimizerConfig(slot_minutes=15, decision_log_level=1, **cfg)
        self._now = now or _dt(15, 30, 5)
        self._raw_forecast_kw = raw_forecast_kw
        self._pv_power_w = pv_power_w
        self._log_messages = []
        self._pv_bias_factor = 1.0
        self._last_recalc_trigger = "startup"
        self._last_recalc_time = None
        self.recalc_calls = []
        self._pv_forecast_service = _FakeForecastService()
        self._pv_bias = PvBiasTracker(
            config=PvBiasConfig.from_main_config(self.config),
            align_to_slot_func=self._align_to_slot,
            log_func=self.log,
        )

    # --- AppDaemon-ish surface -------------------------------------------
    def log(self, message, level="INFO"):
        self._log_messages.append(message)

    def datetime(self):
        return self._now

    def _get_local_timezone(self):
        return TZ

    def _align_to_slot(self, dt):
        return align_to_slot(dt, self.config.slot_minutes, TZ)

    # --- collaborators ----------------------------------------------------
    def _predict_pv_kw_raw(self, dt):
        return self._raw_forecast_kw

    def _get_pv_power_optional(self):
        return self._pv_power_w

    def _recalculate_remaining_schedule(self, current_soc, extra_charge_slots=0):
        self.recalc_calls.append(current_soc)

    # --- helpers ----------------------------------------------------------
    def messages(self):
        return "\n".join(self._log_messages)


# Bind the real implementations under test
MockPvOptimizer._check_pv_shortfall = BatteryOptimizer._check_pv_shortfall
MockPvOptimizer._refresh_pv_bias_factor = BatteryOptimizer._refresh_pv_bias_factor
MockPvOptimizer._sample_pv = BatteryOptimizer._sample_pv
MockPvOptimizer._predict_pv_kw = BatteryOptimizer._predict_pv_kw


def _measure_slot(opt, slot, forecast_kw, actual_kw, samples=15):
    """Simulate a whole slot's worth of PV sampling."""
    opt._pv_bias.ensure_slot_forecast(slot, forecast_kw)
    for i in range(samples):
        opt._pv_bias.add_sample(slot + datetime.timedelta(seconds=60 * i), actual_kw)


# ---------------------------------------------------------------------------
# DEFECT 3 — trigger on measured slot energy, not one boundary sample
# ---------------------------------------------------------------------------

class TestReactiveShortfallTrigger:

    def test_single_boundary_reading_does_not_trigger_recalc(self):
        """The old code recalculated instantly on actual=457W vs forecast=2891W."""
        opt = MockPvOptimizer(now=_dt(15, 30, 5))
        # Nothing measured yet for the completed slot: only the boundary read.
        opt._pv_bias.ensure_slot_forecast(_dt(15, 15), 2.891)
        opt._pv_bias.add_sample(_dt(15, 15, 5), 0.457)

        triggered = opt._check_pv_shortfall(current_soc=84.0)

        assert triggered is False
        assert opt.recalc_calls == []
        assert opt._pv_forecast_service.shortfall_calls == []

    def test_first_measured_shortfall_only_logs(self):
        opt = MockPvOptimizer(now=_dt(15, 30, 5))
        _measure_slot(opt, _dt(15, 15), 2.891, 0.457)

        assert opt._check_pv_shortfall(current_soc=84.0) is False
        assert opt.recalc_calls == []
        assert "streak 1/2" in opt.messages()
        assert "no recalc" in opt.messages()

    def test_second_consecutive_shortfall_triggers_recalc(self):
        opt = MockPvOptimizer(now=_dt(15, 30, 5))
        _measure_slot(opt, _dt(15, 0), 2.891, 0.40)
        _measure_slot(opt, _dt(15, 15), 2.891, 0.457)

        assert opt._check_pv_shortfall(current_soc=84.0) is True
        assert opt.recalc_calls == [84.0]
        assert "recalculating" in opt.messages()

    def test_recalc_records_its_trigger(self):
        """The old PV branch never set _last_recalc_trigger — the sensor lied."""
        opt = MockPvOptimizer(now=_dt(15, 30, 5))
        _measure_slot(opt, _dt(15, 0), 2.891, 0.40)
        _measure_slot(opt, _dt(15, 15), 2.891, 0.457)

        opt._check_pv_shortfall(current_soc=84.0)

        assert opt._last_recalc_trigger == "pv_shortfall"
        assert opt._last_recalc_time == _dt(15, 30, 5)

    def test_shortfall_uses_slot_mean_not_boundary_dip(self):
        """A ramp slot dipping only at its boundary is NOT a shortfall."""
        opt = MockPvOptimizer(now=_dt(9, 15, 5))
        opt._pv_bias.ensure_slot_forecast(_dt(9, 0), 3.0)
        opt._pv_bias.add_sample(_dt(9, 0, 5), 0.2)  # the old instantaneous read
        for i in range(1, 15):
            opt._pv_bias.add_sample(_dt(9, 0) + datetime.timedelta(minutes=i), 2.9)

        assert opt._check_pv_shortfall(current_soc=50.0) is False
        assert opt.recalc_calls == []

    def test_too_few_samples_is_not_a_shortfall(self):
        opt = MockPvOptimizer(now=_dt(15, 30, 5), pv_reactive_consecutive_slots=1)
        _measure_slot(opt, _dt(15, 15), 2.891, 0.457, samples=2)

        assert opt._check_pv_shortfall(current_soc=84.0) is False
        assert opt.recalc_calls == []

    def test_night_slot_below_forecast_floor_is_ignored(self):
        opt = MockPvOptimizer(now=_dt(22, 15, 5), pv_reactive_consecutive_slots=1)
        _measure_slot(opt, _dt(22, 0), 0.05, 0.0)

        assert opt._check_pv_shortfall(current_soc=40.0) is False
        assert opt.recalc_calls == []

    def test_forecast_snapshot_is_taken_before_any_capping(self):
        """refresh_for_shortfall caps the cache; the snapshot must predate it."""
        opt = MockPvOptimizer(now=_dt(15, 30, 5))
        _measure_slot(opt, _dt(15, 0), 2.891, 0.40)
        _measure_slot(opt, _dt(15, 15), 2.891, 0.457)
        # The provider now reports the CAPPED value for the new current slot.
        opt._raw_forecast_kw = 0.457

        opt._check_pv_shortfall(current_soc=84.0)
        # Snapshot of the current slot came from the (capped) provider read,
        # but the ratio that fired came from the un-capped 15:15 snapshot.
        closed = opt._pv_bias.get_closed(_dt(15, 15))
        assert closed.forecast_kw == pytest.approx(2.891)
        assert closed.ratio == pytest.approx(0.457 / 2.891, abs=1e-6)

    def test_consecutive_requirement_is_configurable(self):
        opt = MockPvOptimizer(now=_dt(15, 30, 5), pv_reactive_consecutive_slots=1)
        _measure_slot(opt, _dt(15, 15), 2.891, 0.457)

        assert opt._check_pv_shortfall(current_soc=84.0) is True

    def test_streak_restarts_after_a_recalculation(self):
        """The streak must not keep growing past the threshold forever.

        It is only ever cleared by a GOOD slot, so during a long cloudy
        afternoon it went 2, 3, 4, 5 ... and `streak < consecutive_slots` could
        never hold again — every 15 min slot recalculated.
        """
        opt = MockPvOptimizer(now=_dt(15, 30, 5))
        _measure_slot(opt, _dt(15, 0), 2.891, 0.40)
        _measure_slot(opt, _dt(15, 15), 2.891, 0.457)

        assert opt._check_pv_shortfall(current_soc=84.0) is True
        assert opt._pv_bias.shortfall_streak == 0

    def test_persistent_clouds_do_not_recalculate_every_slot(self):
        """Four hours of unbroken cloud: at most one recalc per 2 slots."""
        opt = MockPvOptimizer(now=_dt(12, 0, 5))

        slot = _dt(11, 45)
        for _ in range(16):  # 16 x 15 min = 4 h
            _measure_slot(opt, slot, 2.891, 0.40)
            opt._now = slot + datetime.timedelta(minutes=15, seconds=5)
            opt._check_pv_shortfall(current_soc=84.0)
            slot += datetime.timedelta(minutes=15)

        # Old behaviour: 15 recalcs (every slot after the first two).
        assert len(opt.recalc_calls) == 8

    def test_a_good_slot_still_clears_the_streak(self):
        opt = MockPvOptimizer(now=_dt(15, 30, 5))
        _measure_slot(opt, _dt(15, 0), 2.891, 0.40)
        _measure_slot(opt, _dt(15, 15), 2.891, 2.80)  # sun came back

        assert opt._check_pv_shortfall(current_soc=84.0) is False
        assert opt._pv_bias.shortfall_streak == 0


# ---------------------------------------------------------------------------
# DEFECT 4 — bias applies to the whole remaining horizon
# ---------------------------------------------------------------------------

class TestBiasAppliedToHorizon:

    def _biased(self, now=None, **cfg):
        opt = MockPvOptimizer(now=now or _dt(15, 30, 5), **cfg)
        _measure_slot(opt, _dt(14, 45), 2.891, 0.90)
        _measure_slot(opt, _dt(15, 0), 2.891, 0.85)
        _measure_slot(opt, _dt(15, 15), 2.891, 0.87)
        opt._pv_bias.close_slots_before(_dt(15, 30))
        opt._refresh_pv_bias_factor()
        return opt

    def test_factor_is_derived_from_measurements(self):
        opt = self._biased()
        assert 0.25 < opt._pv_bias_factor < 0.35

    def test_current_slot_is_scaled(self):
        opt = self._biased()
        opt._raw_forecast_kw = 2.076
        assert opt._predict_pv_kw(_dt(15, 30)) == pytest.approx(
            2.076 * opt._pv_bias_factor
        )

    def test_remaining_slots_today_are_scaled_at_full_strength(self):
        """The old code left 15:30, 15:45 and 17:30 untouched."""
        opt = self._biased()
        opt._raw_forecast_kw = 2.076
        for offset_minutes in (15, 30, 120, 8 * 60):
            slot = _dt(15, 30) + datetime.timedelta(minutes=offset_minutes)
            assert opt._predict_pv_kw(slot) == pytest.approx(
                2.076 * opt._pv_bias_factor
            )
            assert opt._predict_pv_kw(slot) < 0.8  # was 2.076 kW before the fix

    def test_past_slots_stay_raw(self):
        opt = self._biased()
        opt._raw_forecast_kw = 2.891
        assert opt._predict_pv_kw(_dt(14, 45)) == pytest.approx(2.891)

    def test_zero_forecast_stays_zero(self):
        opt = self._biased()
        opt._raw_forecast_kw = 0.0
        assert opt._predict_pv_kw(_dt(16, 0)) == 0.0

    def test_disabled_flag_keeps_forecast_untouched(self):
        opt = self._biased(pv_bias_enabled=False)
        opt._raw_forecast_kw = 2.076
        assert opt._pv_bias_factor == 1.0
        assert opt._predict_pv_kw(_dt(16, 0)) == pytest.approx(2.076)

    def test_factor_change_is_logged(self):
        opt = self._biased()
        assert any("PV bias factor" in m for m in opt._log_messages)


# ---------------------------------------------------------------------------
# Day-boundary attenuation — today's weather is not tomorrow's calibration
# ---------------------------------------------------------------------------

class TestBiasAttenuatesAcrossDayBoundary:
    """Two cloudy hours today must not scale ALL of tomorrow to the clamp.

    The daily full optimization runs at 13:15 and plans ~33 h ahead. Without
    attenuation a 0.2-clamped factor made the DP plan tomorrow with 5x too
    little PV, systematically shifting it onto paid grid charging. `get_factor`
    only relaxes once observations go STALE, which cannot happen within a day.
    """

    def _biased(self, now=None, **cfg):
        opt = MockPvOptimizer(now=now or _dt(13, 15, 5), **cfg)
        for slot in (_dt(12, 15), _dt(12, 30), _dt(12, 45), _dt(13, 0)):
            _measure_slot(opt, slot, 3.0, 0.30)  # ratio 0.10 -> clamped to 0.2
        opt._pv_bias.close_slots_before(_dt(13, 15))
        opt._refresh_pv_bias_factor()
        return opt

    def test_today_keeps_the_full_factor(self):
        opt = self._biased()
        assert opt._pv_bias_factor == pytest.approx(0.2, abs=0.01)
        opt._raw_forecast_kw = 2.0
        assert opt._predict_pv_kw(_dt(17, 0)) == pytest.approx(
            2.0 * opt._pv_bias_factor
        )

    def test_tomorrow_is_attenuated_and_floored(self):
        opt = self._biased()
        opt._raw_forecast_kw = 2.0
        tomorrow_noon = _dt(13, 0) + datetime.timedelta(days=1)
        # 1 + (0.2 - 1) * 0.5 = 0.6, floored at the next-day clamp 0.7
        assert opt._predict_pv_kw(tomorrow_noon) == pytest.approx(2.0 * 0.7)

    def test_the_day_after_relaxes_further(self):
        opt = self._biased(pv_bias_next_day_min_factor=0.2)
        opt._raw_forecast_kw = 2.0
        f = opt._pv_bias_factor
        d1 = opt._predict_pv_kw(_dt(13, 0) + datetime.timedelta(days=1))
        d2 = opt._predict_pv_kw(_dt(13, 0) + datetime.timedelta(days=2))
        assert d1 == pytest.approx(2.0 * (1 + (f - 1) * 0.5), abs=0.01)
        assert d2 == pytest.approx(2.0 * (1 + (f - 1) * 0.25), abs=0.01)
        assert d2 > d1

    def test_attenuation_can_be_disabled(self):
        """weight=1.0 restores the old whole-horizon behaviour."""
        opt = self._biased(
            pv_bias_next_day_weight=1.0, pv_bias_next_day_min_factor=0.0
        )
        opt._raw_forecast_kw = 2.0
        tomorrow = _dt(13, 0) + datetime.timedelta(days=1)
        assert opt._predict_pv_kw(tomorrow) == pytest.approx(
            2.0 * opt._pv_bias_factor
        )

    def test_overproduction_bias_also_relaxes_toward_one(self):
        opt = MockPvOptimizer(now=_dt(13, 15, 5))
        for slot in (_dt(12, 15), _dt(12, 30), _dt(12, 45), _dt(13, 0)):
            _measure_slot(opt, slot, 1.0, 1.4)
        opt._pv_bias.close_slots_before(_dt(13, 15))
        opt._refresh_pv_bias_factor()
        assert opt._pv_bias_factor == pytest.approx(1.4)

        opt._raw_forecast_kw = 2.0
        today = opt._predict_pv_kw(_dt(17, 0))
        tomorrow = opt._predict_pv_kw(_dt(13, 0) + datetime.timedelta(days=1))
        assert today == pytest.approx(2.0 * 1.4)
        assert tomorrow == pytest.approx(2.0 * 1.2)  # 1 + (1.4-1)*0.5

    def test_past_slots_are_still_raw(self):
        opt = self._biased()
        opt._raw_forecast_kw = 3.0
        assert opt._predict_pv_kw(_dt(12, 30)) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Sampling timer
# ---------------------------------------------------------------------------

class TestSamplePv:

    def test_sample_accumulates_into_the_open_slot(self):
        opt = MockPvOptimizer(now=_dt(12, 3), pv_power_w=1500.0)
        opt._sample_pv()
        opt._now = _dt(12, 4)
        opt._pv_power_w = 2500.0
        opt._sample_pv()

        assert opt._pv_bias.slot_sample_count(_dt(12, 0)) == 2
        assert opt._pv_bias.slot_mean_kw(_dt(12, 0)) == pytest.approx(2.0)

    def test_unavailable_sensor_is_not_recorded_as_zero(self):
        """The log had 'actual=0W vs forecast=573W (0%)' from a dead sensor."""
        opt = MockPvOptimizer(now=_dt(7, 45, 5), pv_power_w=None)
        opt._sample_pv()
        assert opt._pv_bias.slot_sample_count(_dt(7, 45)) == 0
        assert opt._pv_bias.slot_mean_kw(_dt(7, 45)) is None

    def test_crossing_a_boundary_closes_and_logs_the_slot(self):
        opt = MockPvOptimizer(now=_dt(12, 0), pv_power_w=600.0)
        for minute in range(0, 15):
            opt._now = _dt(12, minute)
            opt._sample_pv()
        opt._now = _dt(12, 15)
        opt._sample_pv()

        closed = opt._pv_bias.get_closed(_dt(12, 0))
        assert closed is not None
        assert closed.samples == 15
        assert closed.actual_kw == pytest.approx(0.6)
        assert closed.forecast_kw == pytest.approx(2.891)
        assert "PV slot" in opt.messages()
        assert "n=15" in opt.messages()

    def test_sampling_failure_does_not_propagate(self):
        opt = MockPvOptimizer()

        def boom():
            raise RuntimeError("sensor exploded")

        opt._get_pv_power_optional = boom
        opt._sample_pv()  # must not raise
        assert "PV sampling failed" in opt.messages()
