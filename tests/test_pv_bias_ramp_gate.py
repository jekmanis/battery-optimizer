"""The sunrise ramp must not be able to set the whole-horizon PV bias.

Production 2026-09-02, log lines 743-749:

```
07:15:05 PV below forecast at 07:00: actual=0W vs forecast=292W (0%, n=15), streak 1/2
07:30:05 PV bias factor 1.00 -> 0.20 (median of 2 slots over 120min, newest 0min ago)
07:30:05 PV below forecast at 07:15: actual=0W vs forecast=292W (0%, n=15), streak 2 - recalculating
```

Two dawn slots forecast at 292 W and measuring 0 W were the entire evidence
base, and their median dropped the bias onto the `min_factor` clamp — a 5x
under-forecast of a genuinely sunny day (Solcast peak 5.11 kW), applied to the
whole remaining horizon and systematically pushing the DP toward paid grid
charging.

`pv_reactive_min_forecast_w` gates BOTH the reactive shortfall check and the
ratio history (`PvBiasConfig.min_forecast_kw` is derived from it), so raising it
above the ramp fixes both at once.
"""

import datetime

import pytest

from battery_optimizer_lib.config import BatteryOptimizerConfig
from battery_optimizer_lib.pv_bias_tracker import PvBiasConfig, PvBiasTracker
from battery_optimizer_lib.timezone_utils import align_to_slot


TZ = datetime.timezone(datetime.timedelta(hours=3))


def _tracker(min_forecast_kw):
    config = PvBiasConfig(
        enabled=True,
        slot_minutes=15,
        window_minutes=120,
        min_slots=2,
        min_forecast_kw=min_forecast_kw,
        min_factor=0.2,
        max_factor=1.5,
        decay_slots=8,
        min_samples_per_slot=3,
        shortfall_threshold=0.5,
    )
    return PvBiasTracker(
        config=config,
        align_to_slot_func=lambda dt: align_to_slot(dt, 15, TZ),
        log_func=lambda msg, **kw: None,
    )


def _slot(hour, minute):
    return datetime.datetime(2026, 9, 2, hour, minute, tzinfo=TZ)


def _fill(tracker, slot, forecast_kw, actual_kw, samples=15):
    tracker.ensure_slot_forecast(slot, forecast_kw)
    for i in range(samples):
        tracker.add_sample(slot + datetime.timedelta(seconds=30 * i), actual_kw)


def _replay_dawn(tracker):
    """The two 292 W / 0 W slots exactly as they were logged."""
    _fill(tracker, _slot(7, 0), 0.292, 0.0)
    _fill(tracker, _slot(7, 15), 0.292, 0.0)
    tracker.close_slots_before(_slot(7, 30))


class TestRampSlotsAreExcluded:
    def test_the_old_200w_gate_reproduces_the_defect(self):
        """Regression witness: with the old default the clamp is reached."""
        t = _tracker(min_forecast_kw=0.2)

        _replay_dawn(t)

        assert t.get_factor(_slot(7, 30)) == pytest.approx(0.2)

    def test_the_600w_gate_leaves_the_forecast_alone(self):
        t = _tracker(min_forecast_kw=0.6)

        _replay_dawn(t)

        assert t.get_factor(_slot(7, 30)) == 1.0
        assert t.ratio_count(_slot(7, 30)) == 0

    def test_ramp_slots_do_not_build_a_shortfall_streak(self):
        t = _tracker(min_forecast_kw=0.6)

        _replay_dawn(t)

        assert t.shortfall_streak == 0

    def test_a_real_daytime_shortfall_still_registers(self):
        """08:30: forecast 2566 W, measured 437 W — a genuine cloud event."""
        t = _tracker(min_forecast_kw=0.6)

        _fill(t, _slot(8, 30), 2.566, 0.437)
        _fill(t, _slot(8, 45), 2.566, 0.437)
        t.close_slots_before(_slot(9, 0))

        assert t.shortfall_streak == 2
        # 437/2566 = 0.17, below the 0.2 clamp — the clamp, not the gate, is
        # what limits it here.
        assert t.get_factor(_slot(9, 0)) == pytest.approx(0.2)

    def test_a_mixed_window_ignores_only_the_ramp(self):
        t = _tracker(min_forecast_kw=0.6)

        _replay_dawn(t)                              # excluded
        _fill(t, _slot(7, 30), 1.0, 0.5)             # counted
        _fill(t, _slot(7, 45), 1.0, 0.5)             # counted
        t.close_slots_before(_slot(8, 0))

        assert t.ratio_count(_slot(8, 0)) == 2
        assert t.get_factor(_slot(8, 0)) == pytest.approx(0.5)


class TestMinSlotsStillGuardsTheMedian:
    """The factor stays at 1.0 until `min_slots` qualifying slots are in."""

    def test_one_qualifying_slot_is_not_enough(self):
        t = _tracker(min_forecast_kw=0.6)

        _fill(t, _slot(9, 0), 3.0, 0.3)
        t.close_slots_before(_slot(9, 15))

        assert t.ratio_count(_slot(9, 15)) == 1
        assert t.get_factor(_slot(9, 15)) == 1.0

    def test_the_second_qualifying_slot_releases_it(self):
        t = _tracker(min_forecast_kw=0.6)

        _fill(t, _slot(9, 0), 3.0, 0.3)
        _fill(t, _slot(9, 15), 3.0, 0.3)
        t.close_slots_before(_slot(9, 30))

        assert t.get_factor(_slot(9, 30)) == pytest.approx(0.2)


class TestConfigWiring:
    def test_the_main_config_default_reaches_the_bias_config(self):
        bias = PvBiasConfig.from_main_config(BatteryOptimizerConfig())

        assert bias.min_forecast_kw == pytest.approx(0.6)

    def test_an_override_reaches_the_bias_config(self):
        main = BatteryOptimizerConfig.from_args({"pv_reactive_min_forecast_w": 800})
        bias = PvBiasConfig.from_main_config(main)

        assert bias.min_forecast_kw == pytest.approx(0.8)

    def test_one_threshold_gates_both_consumers(self):
        """The reactive check and the bias window must not drift apart."""
        main = BatteryOptimizerConfig()
        bias = PvBiasConfig.from_main_config(main)

        assert bias.min_forecast_kw * 1000.0 == pytest.approx(
            main.pv_reactive_min_forecast_w
        )
