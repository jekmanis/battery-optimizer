"""Regressions for DEFECT 8: a terminal energy value of 0 was silently accepted.

Production evidence (33h AppDaemon window, 2026-07-27..28): the deployed
apps.yaml pinned `terminal_energy_value_eur_kwh: 0`, so every plan ended with
"07-30 00:30 EXPORT (until depleted) -> 11.2%". The only trace in the log was 70
identical INFO lines reading

    "Terminal energy value: 0.0000 EUR/kWh (configured); net-load slots worth
     less than this are HELD"

which is not just quiet but wrong — no slot is worth less than zero, so the
sentence describes a rule that can never fire.

Before the fix: from_args() emitted nothing, log_summary() never mentioned the
terminal value at all, and the DP line above was INFO. All assertions here fail
against that code.
"""

import datetime

import pytest

from battery_optimizer_lib import BatteryMode, DPOptimizer, DPOptimizerConfig, PricePoint
from battery_optimizer_lib.config import (
    TERMINAL_VALUE_ZERO_NOTICE,
    BatteryOptimizerConfig,
)


class RecordingLog:
    def __init__(self):
        self.entries = []

    def __call__(self, message, level="INFO"):
        self.entries.append((message, level))

    def at(self, level):
        return [m for m, lvl in self.entries if lvl == level]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def test_zero_terminal_value_is_announced_at_config_load():
    log = RecordingLog()

    config = BatteryOptimizerConfig.from_args(
        {"terminal_energy_value_eur_kwh": 0}, log_func=log
    )

    assert config.terminal_energy_value_eur_kwh == 0.0
    assert TERMINAL_VALUE_ZERO_NOTICE in log.at("INFO")
    # Both modes have a real failure mode, so this must not be a warning and
    # must not prescribe one of them.
    assert not log.at("WARNING")
    assert "no-salvage mode" in TERMINAL_VALUE_ZERO_NOTICE
    assert "Neither is" in TERMINAL_VALUE_ZERO_NOTICE


def test_zero_terminal_value_without_a_logger_does_not_raise():
    """log_func is documented as optional — every call site must be guarded.

    Regression: the zero-salvage notice was emitted with a bare `log_func(...)`
    instead of the guarded local helper, so constructing the config with the
    documented default `log_func=None` raised
    `TypeError: 'NoneType' object is not callable` and no config could be built
    at all. Only the WARNING paths were guarded; the INFO path was not.
    """
    config = BatteryOptimizerConfig.from_args({"terminal_energy_value_eur_kwh": 0})

    assert config.terminal_energy_value_eur_kwh == 0.0


def test_every_from_args_log_path_tolerates_a_missing_logger():
    """The other guarded paths (invalid slot/recalc/observation minutes)."""
    config = BatteryOptimizerConfig.from_args({
        "terminal_energy_value_eur_kwh": 0,
        "slot_minutes": 7,                 # not a divisor of 1440
        "adaptive_recalc_minutes": 7,
        "load_observation_minutes": 7,
    })

    assert config.terminal_energy_value_eur_kwh == 0.0
    assert config.slot_minutes == 15       # fell back, silently


def test_string_zero_terminal_value_is_also_announced():
    """apps.yaml may quote the value; the mode is the same."""
    log = RecordingLog()

    BatteryOptimizerConfig.from_args(
        {"terminal_energy_value_eur_kwh": "0.0"}, log_func=log
    )

    assert TERMINAL_VALUE_ZERO_NOTICE in log.at("INFO")


def test_default_and_auto_are_silent_and_mean_auto():
    log = RecordingLog()

    default_cfg = BatteryOptimizerConfig.from_args({}, log_func=log)
    auto_cfg = BatteryOptimizerConfig.from_args(
        {"terminal_energy_value_eur_kwh": "auto"}, log_func=log
    )

    assert default_cfg.terminal_energy_value_eur_kwh is None
    assert auto_cfg.terminal_energy_value_eur_kwh is None
    assert TERMINAL_VALUE_ZERO_NOTICE not in log.at("INFO")


def test_positive_terminal_value_is_silent():
    log = RecordingLog()

    config = BatteryOptimizerConfig.from_args(
        {"terminal_energy_value_eur_kwh": 0.12}, log_func=log
    )

    assert config.terminal_energy_value_eur_kwh == pytest.approx(0.12)
    assert TERMINAL_VALUE_ZERO_NOTICE not in log.at("INFO")


# ---------------------------------------------------------------------------
# Startup summary
# ---------------------------------------------------------------------------

def test_log_summary_reports_terminal_value_mode():
    """The startup summary never mentioned the terminal value before the fix."""
    info = []
    warn = []

    BatteryOptimizerConfig().log_summary(
        lambda msg: info.append(msg), warn_func=warn.append
    )
    assert any("Terminal energy value: auto" in m for m in info)
    assert not warn

    info.clear()
    BatteryOptimizerConfig(terminal_energy_value_eur_kwh=0.0).log_summary(
        lambda msg: info.append(msg), warn_func=warn.append
    )
    assert TERMINAL_VALUE_ZERO_NOTICE in info
    assert not warn


def test_log_summary_without_warn_func_still_reports_the_mode():
    """Single-argument callers must still see which mode is active."""
    info = []

    BatteryOptimizerConfig(terminal_energy_value_eur_kwh=0.0).log_summary(info.append)

    assert TERMINAL_VALUE_ZERO_NOTICE in info


def test_log_summary_reports_inverter_control_timing():
    info = []
    BatteryOptimizerConfig().log_summary(info.append)

    line = next(m for m in info if "Inverter control timing" in m)
    assert "timeout=15s" in line
    assert "verify after 90s" in line


# ---------------------------------------------------------------------------
# DP logging
# ---------------------------------------------------------------------------

def _dp(terminal_value, log_fn, warn_degenerate=True):
    config = DPOptimizerConfig(
        battery_capacity=10.0,
        min_soc=0.0,
        max_soc=100.0,
        efficiency=1.0,
        discharge_rate=5.0,
        slot_minutes=60,
        soc_step_percent=10.0,
        grid_fee=0.0,
        battery_wear_cost=0.0,
        export_rate_multiplier=0.0,
        inverter_efficiency=1.0,
        terminal_energy_value_eur_kwh=terminal_value,
    )
    return DPOptimizer(
        config=config,
        load_predictor=lambda _when: 1.0,
        charge_rate_predictor=lambda _soc, _temp: 0.0,
        temp_after_charge_predictor=lambda temp, _minutes: temp,
        temp_after_idle_predictor=lambda temp, _minutes: temp,
        log_fn=log_fn,
        decision_log_level=1,
        warn_degenerate_terminal=warn_degenerate,
    )


def test_dp_logs_zero_terminal_value_as_info():
    slot = datetime.datetime(2026, 7, 28, 16, 0)
    log = RecordingLog()

    _dp(0.0, log).optimize([PricePoint(time=slot, price=0.05)], slot, current_soc=60.0)

    info = log.at("INFO")
    assert any("no-salvage mode" in m and "spends it there" in m for m in info)
    # Both terminal-value settings have a real failure mode, so the DP states
    # the active one without recommending the other.
    assert not log.at("WARNING")
    # The misleading "worth less than this are HELD" sentence must be gone.
    assert not any("worth less than this" in m for m, _ in log.entries)


def test_dp_terminal_warning_can_be_suppressed_per_run():
    """Adaptive re-optimizations (every 15 min) must not spam the warning."""
    slot = datetime.datetime(2026, 7, 28, 16, 0)
    log = RecordingLog()

    _dp(0.0, log, warn_degenerate=False).optimize(
        [PricePoint(time=slot, price=0.05)], slot, current_soc=60.0
    )

    assert log.at("WARNING") == []


def test_dp_keeps_informative_line_for_non_degenerate_values():
    slot = datetime.datetime(2026, 7, 28, 16, 0)
    log = RecordingLog()

    _dp(None, log).optimize([PricePoint(time=slot, price=0.05)], slot, current_soc=60.0)

    assert any(
        "Terminal energy value" in m and "auto" in m and lvl == "INFO"
        for m, lvl in log.entries
    )
    assert log.at("WARNING") == []


def test_zero_terminal_value_still_drains_the_horizon():
    """The fix WARNS; it does not change behaviour. apps.yaml must be edited."""
    slot = datetime.datetime(2026, 7, 28, 16, 0)
    log = RecordingLog()

    result = _dp(0.0, log).optimize(
        [PricePoint(time=slot, price=0.05)], slot, current_soc=60.0
    )

    assert result.schedule[slot].mode == BatteryMode.DISCHARGE
