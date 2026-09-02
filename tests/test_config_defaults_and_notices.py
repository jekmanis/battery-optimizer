"""Config defaults and startup notices fixed after the 2026-09-02 production log.

Covers:
* the PV ramp gate (`pv_reactive_min_forecast_w`) raised out of the sunrise ramp;
* the new cost-attribution / execution-guard knobs and their clamping;
* `verify_enabled` and the "verification DISABLED" summary line;
* the no-salvage notice being STATED once per initialization, not twice.
"""

import pytest

from battery_optimizer_lib.config import (
    TERMINAL_VALUE_ZERO_NOTICE,
    BatteryOptimizerConfig,
)


class TestPvRampGate:
    """Two dawn slots forecast at 292 W must not be able to set the bias.

    On 2026-09-02 the 07:00 and 07:15 slots (forecast 292 W, measured 0 W)
    were the only two observations in the window, and with the old 200 W gate
    their median slammed the whole-horizon PV bias onto the 0.20 clamp.
    """

    def test_default_excludes_a_292w_sunrise_ramp_slot(self):
        cfg = BatteryOptimizerConfig()

        assert cfg.pv_reactive_min_forecast_w == 600.0
        assert 292.0 <= cfg.pv_reactive_min_forecast_w

    def test_from_args_default_matches_the_dataclass_default(self):
        assert (
            BatteryOptimizerConfig.from_args({}).pv_reactive_min_forecast_w
            == BatteryOptimizerConfig().pv_reactive_min_forecast_w
        )

    def test_explicit_value_still_wins(self):
        cfg = BatteryOptimizerConfig.from_args({"pv_reactive_min_forecast_w": 250})

        assert cfg.pv_reactive_min_forecast_w == 250.0

    def test_negative_value_is_clamped_to_zero(self):
        cfg = BatteryOptimizerConfig(pv_reactive_min_forecast_w=-50)

        assert cfg.pv_reactive_min_forecast_w == 0.0

    def test_the_bias_window_uses_the_same_gate(self):
        """One threshold gates both the shortfall check and the ratio history."""
        from battery_optimizer_lib.pv_bias_tracker import PvBiasConfig

        bias = PvBiasConfig.from_main_config(BatteryOptimizerConfig())

        assert bias.min_forecast_kw == pytest.approx(0.6)


class TestExecutionAndAttributionKnobs:
    def test_defaults(self):
        cfg = BatteryOptimizerConfig()

        assert cfg.execute_dedupe_seconds == 60
        assert cfg.planned_depletion_margin_percent == 1.0
        assert cfg.cost_pv_attribution_min_w == 100.0
        assert cfg.cost_grid_charge_grace_seconds == 120

    def test_from_args_reads_all_four(self):
        cfg = BatteryOptimizerConfig.from_args(
            {
                "execute_dedupe_seconds": 30,
                "planned_depletion_margin_percent": 2.5,
                "cost_pv_attribution_min_w": 250,
                "cost_grid_charge_grace_seconds": 45,
            }
        )

        assert cfg.execute_dedupe_seconds == 30
        assert cfg.planned_depletion_margin_percent == 2.5
        assert cfg.cost_pv_attribution_min_w == 250.0
        assert cfg.cost_grid_charge_grace_seconds == 45

    def test_dedupe_window_can_never_swallow_a_whole_slot(self):
        """Half a slot is the hard ceiling: the next timer tick must execute."""
        cfg = BatteryOptimizerConfig(slot_minutes=15, execute_dedupe_seconds=99999)

        assert cfg.execute_dedupe_seconds == 15 * 30  # 450s = half a slot

    def test_dedupe_can_be_disabled_with_zero(self):
        assert BatteryOptimizerConfig(execute_dedupe_seconds=0).execute_dedupe_seconds == 0

    def test_negative_values_are_clamped(self):
        cfg = BatteryOptimizerConfig(
            execute_dedupe_seconds=-5,
            planned_depletion_margin_percent=-1.0,
            cost_pv_attribution_min_w=-10,
            cost_grid_charge_grace_seconds=-1,
        )

        assert cfg.execute_dedupe_seconds == 0
        assert cfg.planned_depletion_margin_percent == 0.0
        assert cfg.cost_pv_attribution_min_w == 0.0
        assert cfg.cost_grid_charge_grace_seconds == 0

    def test_log_summary_states_the_attribution_rules(self):
        lines = []
        BatteryOptimizerConfig().log_summary(lines.append)
        text = "\n".join(lines)

        assert "Cost attribution:" in text
        assert "100W measured PV" in text
        assert "120s" in text
        assert "deduped within 60s" in text


def _timing_line(cfg):
    lines = []
    cfg.log_summary(lines.append)
    timing = [l for l in lines if "Inverter control timing" in l]
    assert len(timing) == 1
    return timing[0]


class TestVerifyEnabled:
    def test_default_is_on(self):
        assert BatteryOptimizerConfig().verify_enabled is True
        assert BatteryOptimizerConfig.from_args({}).verify_enabled is True

    def test_can_be_turned_off(self):
        cfg = BatteryOptimizerConfig.from_args({"verify_enabled": False})

        assert cfg.verify_enabled is False

    def test_no_fallback_sensor_is_guessed(self):
        """A guessed entity produced 73/73 false 'Passthrough' mismatches."""
        assert BatteryOptimizerConfig().inverter_mode_sensor == ""

    def test_the_master_switch_beats_the_source(self):
        timing = _timing_line(
            BatteryOptimizerConfig(
                verify_enabled=False, verify_source="registers", device_id="dev1"
            )
        )

        assert "verification DISABLED" in timing


class TestVerifySource:
    """Which source is armed must be visible at startup.

    The 2026-09-02 log carried 73 mismatches with no way to tell from the
    startup lines WHAT was being compared. "registers" and "mode_sensor" fail
    in completely different ways.
    """

    def test_default_is_auto(self):
        assert BatteryOptimizerConfig().verify_source == "auto"
        assert BatteryOptimizerConfig.from_args({}).verify_source == "auto"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("registers", "registers"),
            ("  Registers ", "registers"),
            ("MODE_SENSOR", "mode_sensor"),
            ("none", "none"),
            ("auto", "auto"),
        ],
    )
    def test_from_args_normalizes_case_and_whitespace(self, raw, expected):
        cfg = BatteryOptimizerConfig.from_args({"verify_source": raw})

        assert cfg.verify_source == expected

    @pytest.mark.parametrize("bad", ["sensor", "yes", "", "regsters", "1"])
    def test_an_unknown_value_falls_back_to_auto(self, bad):
        """It must never quietly disable, nor quietly pick the mode sensor."""
        assert BatteryOptimizerConfig(verify_source=bad).verify_source == "auto"
        assert (
            BatteryOptimizerConfig.from_args({"verify_source": bad}).verify_source
            == "auto"
        )

    def test_none_is_preserved_and_not_coerced_to_auto(self):
        assert BatteryOptimizerConfig(verify_source="none").verify_source == "none"

    def test_summary_always_names_the_source(self):
        timing = _timing_line(
            BatteryOptimizerConfig(verify_source="registers", device_id="dev1")
        )

        assert "verification source=registers" in timing

    def test_registers_summary_names_the_registers(self):
        timing = _timing_line(
            BatteryOptimizerConfig(verify_source="registers", device_id="dev1")
        )

        assert "via registers 30407-30410/30200-30201" in timing

    def test_auto_with_a_device_reports_the_registers(self):
        timing = _timing_line(
            BatteryOptimizerConfig(verify_source="auto", device_id="dev1")
        )

        assert "via registers 30407-30410/30200-30201" in timing

    def test_auto_in_dry_run_reports_disabled(self):
        timing = _timing_line(BatteryOptimizerConfig(verify_source="auto"))

        assert "verification DISABLED (dry run, no device_id)" in timing

    def test_registers_without_a_device_reports_disabled(self):
        timing = _timing_line(BatteryOptimizerConfig(verify_source="registers"))

        assert "verification DISABLED (dry run, no device_id)" in timing

    def test_mode_sensor_summary_names_the_entity(self):
        timing = _timing_line(
            BatteryOptimizerConfig(
                verify_source="mode_sensor",
                inverter_mode_sensor="sensor.growatt_wit_inverter_mode",
                device_id="dev1",
            )
        )

        assert "verification via sensor.growatt_wit_inverter_mode" in timing

    def test_mode_sensor_without_an_entity_reports_disabled(self):
        timing = _timing_line(
            BatteryOptimizerConfig(verify_source="mode_sensor", device_id="dev1")
        )

        assert "verification DISABLED (no inverter_mode_sensor)" in timing

    def test_none_reports_disabled_even_with_a_device(self):
        timing = _timing_line(
            BatteryOptimizerConfig(verify_source="none", device_id="dev1")
        )

        assert "verification source=none" in timing
        assert "verification DISABLED" in timing

    def test_auto_never_falls_back_to_the_mode_sensor(self):
        """Choosing the frozen entity must be explicit, never implicit."""
        timing = _timing_line(
            BatteryOptimizerConfig(
                verify_source="auto",
                inverter_mode_sensor="sensor.growatt_inverter_mode",
                device_id="dev1",
            )
        )

        assert "sensor.growatt_inverter_mode" not in timing
        assert "via registers" in timing

    def test_the_example_yaml_selects_registers(self):
        import yaml

        data = yaml.safe_load(
            open("appdaemon/apps/apps.yaml.example", encoding="utf-8")
        )
        args = data[list(data.keys())[0]]
        cfg = BatteryOptimizerConfig.from_args(args)

        assert cfg.verify_source == "registers"
        assert cfg.verify_enabled is True


class TestTerminalNoticeIsStatedOnce:
    """CLAUDE.md: the mode is STATED once at config load — once, not twice.

    Production 2026-09-02 printed the whole ~600-character paragraph twice, 4 ms
    apart (01:59:31.084 from from_args, .088 from log_summary).
    """

    def test_from_args_plus_log_summary_emit_the_notice_only_once(self):
        lines = []
        cfg = BatteryOptimizerConfig.from_args(
            {"terminal_energy_value_eur_kwh": 0}, log_func=lines.append
        )
        cfg.log_summary(lines.append)

        assert sum(1 for l in lines if l == TERMINAL_VALUE_ZERO_NOTICE) == 1

    def test_the_summary_still_names_the_active_mode(self):
        lines = []
        cfg = BatteryOptimizerConfig.from_args(
            {"terminal_energy_value_eur_kwh": 0}, log_func=lines.append
        )
        after_args = len(lines)
        cfg.log_summary(lines.append)
        summary = lines[after_args:]

        assert any("no-salvage mode" in l for l in summary)

    def test_a_config_built_directly_still_gets_the_full_notice(self):
        lines = []
        BatteryOptimizerConfig(terminal_energy_value_eur_kwh=0.0).log_summary(
            lines.append
        )

        assert TERMINAL_VALUE_ZERO_NOTICE in lines

    def test_from_args_without_a_logger_does_not_mark_the_notice_emitted(self):
        """No logger means nothing was stated, so log_summary must state it."""
        cfg = BatteryOptimizerConfig.from_args({"terminal_energy_value_eur_kwh": 0})
        lines = []
        cfg.log_summary(lines.append)

        assert TERMINAL_VALUE_ZERO_NOTICE in lines

    def test_auto_mode_never_emits_the_zero_notice(self):
        lines = []
        cfg = BatteryOptimizerConfig.from_args(
            {"terminal_energy_value_eur_kwh": "auto"}, log_func=lines.append
        )
        cfg.log_summary(lines.append)

        assert TERMINAL_VALUE_ZERO_NOTICE not in lines

    def test_the_flag_is_not_a_from_args_keyword(self):
        """init=False: it must never look like a configuration knob."""
        cfg = BatteryOptimizerConfig.from_args(
            {"terminal_zero_notice_emitted": True}
        )

        assert cfg.terminal_zero_notice_emitted is False
