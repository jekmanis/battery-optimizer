"""Business-level regressions for finite-horizon DP energy accounting."""

import datetime

from battery_optimizer_lib import (
    BatteryMode,
    DPOptimizer,
    DPOptimizerConfig,
    PricePoint,
)


def _optimizer(*, terminal_value, load_kw=1.0, soc_step_percent=10.0):
    config = DPOptimizerConfig(
        battery_capacity=10.0,
        min_soc=0.0,
        max_soc=100.0,
        efficiency=1.0,
        discharge_rate=5.0,
        slot_minutes=60,
        soc_step_percent=soc_step_percent,
        grid_fee=0.0,
        battery_wear_cost=0.0,
        export_rate_multiplier=0.0,
        inverter_efficiency=1.0,
        terminal_energy_value_eur_kwh=terminal_value,
    )
    return DPOptimizer(
        config=config,
        load_predictor=lambda _when: load_kw,
        charge_rate_predictor=lambda _soc, _temp: 0.0,
        temp_after_charge_predictor=lambda temp, _minutes: temp,
        temp_after_idle_predictor=lambda temp, _minutes: temp,
    )


def test_terminal_value_prevents_unprofitable_horizon_depletion():
    """Stored energy is retained when its terminal value exceeds avoided import."""
    slot = datetime.datetime(2026, 1, 1, 12, 0)
    prices = [PricePoint(time=slot, price=0.50)]

    legacy = _optimizer(terminal_value=0.0).optimize(prices, slot, current_soc=60.0)
    valued = _optimizer(terminal_value=1.0).optimize(prices, slot, current_soc=60.0)

    assert legacy.schedule[slot].mode == BatteryMode.DISCHARGE
    assert valued.schedule[slot].mode == BatteryMode.HOLD
    assert valued.soc_trajectory[slot][1] == 60.0


def test_auto_terminal_value_is_supported_and_non_negative():
    """Application auto mode (None) produces a valid finite-horizon schedule."""
    slot = datetime.datetime(2026, 1, 1, 12, 0)
    result = _optimizer(terminal_value=None).optimize(
        [PricePoint(time=slot, price=0.20)],
        slot,
        current_soc=60.0,
    )

    assert result.schedule[slot].mode in {BatteryMode.HOLD, BatteryMode.DISCHARGE}
    assert 0.0 <= result.soc_trajectory[slot][1] <= 60.0


def test_discharge_quantization_never_creates_stored_energy():
    """A sub-step discharge must round SOC down, not back to its start state."""
    slot = datetime.datetime(2026, 1, 1, 12, 0)
    optimizer = _optimizer(
        terminal_value=0.0,
        load_kw=0.4,          # 0.4 kWh removed from a 1 kWh DP step
        soc_step_percent=10.0,
    )
    result = optimizer.optimize(
        [PricePoint(time=slot, price=1.0)],
        slot,
        current_soc=60.0,
    )

    assert result.schedule[slot].mode == BatteryMode.DISCHARGE
    theoretical_end_soc = 56.0
    assert result.soc_trajectory[slot][1] <= theoretical_end_soc + 1e-9
