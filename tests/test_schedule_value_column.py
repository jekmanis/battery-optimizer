"""Regressions for DEFECT 9(a): the schedule log's value column read 0.0000.

Production evidence (33h AppDaemon window): the first column of every DISCHARGE
line came from the tracked stored-energy cost basis, which legitimately decays
to zero around midday (PV is booked at the foregone export revenue, and when
spot <= export fee that revenue IS zero). The trajectory in the log was

    07-27 07:45  avg 0.0540 -> first DISCHARGE slot printed 0.0540
    07-27 11:59  avg 0.0090 -> printed 0.0090
    07-28 16:00  avg 0.0000 -> EVERY slot printed 0.0000

At which point the log no longer explained a single decision.

The fix leaves both the DP objective and `_pv_opportunity_cost` untouched (the
zero basis is correct) and instead reports the slot's MARGINAL value from the
DP's own tariff arithmetic, keeping the stored basis visible alongside it.
"""

import datetime

import pytest

from battery_optimizer_lib import (
    BatteryMode,
    DPOptimizer,
    DPOptimizerConfig,
    PricePoint,
    ScheduleEntry,
    ScheduleFormatter,
    ScheduleFormatterConfig,
)


def make_formatter():
    return ScheduleFormatter(
        config=ScheduleFormatterConfig(
            slot_minutes=15,
            slot_hours=0.25,
            battery_capacity=14.3,
            charge_rate=4.5,
            discharge_rate=4.5,
            export_discharge_rate=0.0,
            efficiency=0.85,
            battery_wear_cost=0.0,
            decision_log_level=1,
        ),
        log_func=lambda *_a, **_k: None,
    )


HOUR = datetime.datetime(2026, 7, 28, 16, 0)


# ---------------------------------------------------------------------------
# Reason formatting
# ---------------------------------------------------------------------------

def test_degenerate_stored_basis_no_longer_hides_the_decision():
    """The logged 07-28 16:00 case: stored basis 0.0000, real value 0.1234."""
    formatter = make_formatter()
    entry = ScheduleEntry(
        time=HOUR,
        mode=BatteryMode.DISCHARGE,
        reason="0.0712 EUR/kWh load~1.20kW",
        marginal_value_eur_kwh=0.1234,
        value_basis="avoided-import",
    )

    line = formatter._format_reason_with_cost(entry, HOUR, {HOUR: 0.0})

    assert line.startswith("0.1234 EUR/kWh avoided-import")
    assert "grid 0.0712" in line          # spot price still visible
    assert "stored 0.0000" in line        # basis still visible, but demoted
    assert "[stored basis ~0: PV booked at export floor]" in line
    assert "load~1.20kW" in line


def test_charge_slots_now_carry_a_value_too():
    """Before the fix only DISCHARGE slots were annotated at all."""
    formatter = make_formatter()
    entry = ScheduleEntry(
        time=HOUR,
        mode=BatteryMode.CHARGE,
        reason="0.0300 EUR/kWh load~0.80kW",
        marginal_value_eur_kwh=-0.0847,
        value_basis="landed-charge",
    )

    line = formatter._format_reason_with_cost(entry, HOUR, {HOUR: 0.0540})

    assert line.startswith("-0.0847 EUR/kWh landed-charge")
    assert "stored 0.0540" in line
    assert "[stored basis ~0" not in line


def test_hold_slot_keeps_its_annotations():
    formatter = make_formatter()
    entry = ScheduleEntry(
        time=HOUR,
        mode=BatteryMode.HOLD,
        reason="0.0100 EUR/kWh load~0.50kW pv~2.00kW [pv>=load]",
        marginal_value_eur_kwh=0.0921,
        value_basis="kept",
    )

    line = formatter._format_reason_with_cost(entry, HOUR, None)

    assert line.startswith("0.0921 EUR/kWh kept")
    assert "[pv>=load]" in line
    assert "stored" not in line  # no projected costs supplied


def test_entries_without_a_marginal_value_keep_the_legacy_format():
    """Schedules restored from the HA sensor carry no DP fields."""
    formatter = make_formatter()
    entry = ScheduleEntry(
        time=HOUR,
        mode=BatteryMode.DISCHARGE,
        reason="0.0712 EUR/kWh load~1.20kW",
    )

    line = formatter._format_reason_with_cost(entry, HOUR, {HOUR: 0.0540})

    assert line == "0.0540 EUR/kWh (grid 0.0712) load~1.20kW"


# ---------------------------------------------------------------------------
# Sensor / markdown output
# ---------------------------------------------------------------------------

def test_schedule_list_exposes_value_and_basis():
    formatter = make_formatter()
    schedule = {
        HOUR: ScheduleEntry(
            time=HOUR,
            mode=BatteryMode.DISCHARGE,
            reason="0.0712 EUR/kWh load~1.20kW",
            export_rate=0,
            marginal_value_eur_kwh=0.123456,
            value_basis="avoided-import",
        )
    }

    row = formatter.format_schedule_list(schedule)[0]

    assert row["value"] == pytest.approx(0.1235)
    assert row["value_basis"] == "avoided-import"


def test_markdown_gains_a_value_column():
    formatter = make_formatter()
    schedule = {
        HOUR: ScheduleEntry(
            time=HOUR,
            mode=BatteryMode.DISCHARGE,
            reason="0.0712 EUR/kWh load~1.20kW",
            export_rate=0,
            marginal_value_eur_kwh=0.1234,
            value_basis="avoided-import",
        )
    }

    md = formatter.format_schedule_markdown(
        schedule=schedule,
        now=HOUR,
        local_tz=None,
        align_to_slot_func=lambda dt: dt.replace(minute=0, second=0, microsecond=0),
    )

    header, _sep, row = md.split("\n")
    assert header.strip().endswith("| Price | Value |")
    assert row.strip().endswith("| 0.0712 | 0.1234 |")


# ---------------------------------------------------------------------------
# DP is the source of the number, and it is REPORTING ONLY
# ---------------------------------------------------------------------------

def _dp(**overrides):
    params = dict(
        battery_capacity=10.0,
        min_soc=0.0,
        max_soc=100.0,
        efficiency=0.9,
        discharge_rate=5.0,
        slot_minutes=60,
        soc_step_percent=10.0,
        grid_fee=0.052,
        battery_wear_cost=0.01,
        grid_export_fee=0.02,
        export_rate_multiplier=1.0,
        inverter_efficiency=0.97,
        import_price_multiplier=1.21,
        terminal_energy_value_eur_kwh=0.0,
    )
    params.update(overrides)
    config = DPOptimizerConfig(**params)
    return DPOptimizer(
        config=config,
        load_predictor=lambda _when: 2.0,
        charge_rate_predictor=lambda _soc, _temp: 0.0,
        temp_after_charge_predictor=lambda temp, _minutes: temp,
        temp_after_idle_predictor=lambda temp, _minutes: temp,
    )


def test_dp_reports_marginal_value_per_slot():
    slot = datetime.datetime(2026, 7, 28, 16, 0)
    optimizer = _dp(export_rate_multiplier=0.0)  # self-consumption only
    price = 0.20

    result = optimizer.optimize(
        [PricePoint(time=slot, price=price)], slot, current_soc=60.0
    )
    entry = result.schedule[slot]

    assert entry.mode == BatteryMode.DISCHARGE
    assert not entry.export_rate
    expected = (price + 0.052) * 1.21 * 0.97 - 0.01
    assert entry.marginal_value_eur_kwh == pytest.approx(expected)
    assert entry.value_basis == "avoided-import"


def test_dp_marginal_value_for_charge_is_negative_landed_cost():
    optimizer = _dp()
    price = 0.05

    value, basis = optimizer._marginal_slot_value(
        BatteryMode.CHARGE, price, is_export=False, terminal_rate=0.1
    )

    assert basis == "landed-charge"
    assert value == pytest.approx(-((price + 0.052) * 1.21) / (0.9 * 0.97))
    assert value < 0


def test_dp_marginal_value_for_hold_is_the_terminal_rate():
    optimizer = _dp()

    value, basis = optimizer._marginal_slot_value(
        BatteryMode.HOLD, 0.05, is_export=False, terminal_rate=0.1234
    )

    assert basis == "kept"
    assert value == pytest.approx(0.1234)


def test_dp_marginal_value_for_export_uses_the_sell_price():
    optimizer = _dp()
    price = 0.30

    value, basis = optimizer._marginal_slot_value(
        BatteryMode.DISCHARGE, price, is_export=True, terminal_rate=0.0
    )

    assert basis == "export"
    assert value == pytest.approx(max(0.0, price * 1.0 - 0.02) * 0.97 - 0.01)


def test_marginal_value_does_not_change_the_schedule():
    """CLAUDE.md invariant: the value column is reporting, never an objective.

    Same scenario as test_terminal_value_prevents_unprofitable_horizon_depletion:
    modes and SOC trajectory must be exactly what they were before the field was
    added.
    """
    slot = datetime.datetime(2026, 1, 1, 12, 0)
    prices = [PricePoint(time=slot, price=0.50)]

    legacy = _dp(
        efficiency=1.0, grid_fee=0.0, battery_wear_cost=0.0,
        export_rate_multiplier=0.0, inverter_efficiency=1.0,
        import_price_multiplier=1.0, terminal_energy_value_eur_kwh=0.0,
    ).optimize(prices, slot, current_soc=60.0)
    valued = _dp(
        efficiency=1.0, grid_fee=0.0, battery_wear_cost=0.0,
        export_rate_multiplier=0.0, inverter_efficiency=1.0,
        import_price_multiplier=1.0, terminal_energy_value_eur_kwh=1.0,
    ).optimize(prices, slot, current_soc=60.0)

    assert legacy.schedule[slot].mode == BatteryMode.DISCHARGE
    assert valued.schedule[slot].mode == BatteryMode.HOLD
    assert valued.soc_trajectory[slot][1] == 60.0
    # ... and both still carry a value the log can print.
    assert legacy.schedule[slot].value_basis == "avoided-import"
    assert valued.schedule[slot].value_basis == "kept"
    assert valued.schedule[slot].marginal_value_eur_kwh == pytest.approx(1.0)
