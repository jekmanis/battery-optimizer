"""HA recorder attribute-limit regressions.

Production evidence (2026-09-07 19:39 onwards, 64 occurrences): "State
attributes for sensor.battery_optimizer exceed maximum size of 16384 bytes ...
Attributes will not be stored". Measured on the live installation with only 63
schedule slots (before tomorrow's prices are published) the blob was 18 821
bytes, of which `schedule` was 13 693 B, `load_profile_stats` 2 439 B and
`temp_aware_rates` 1 008 B. After 14:15 the plan holds ~140 slots, so `schedule`
alone is ~30 KB.

The recorder drops ALL attributes above the limit, so the sensor keeps a state
history and loses every attribute. These tests pin the split that fixes it: a
lean, recorded `sensor.battery_optimizer` carrying a compact summary, and the
bulky payloads on their own recorder-excluded entities.
"""

import ast
import datetime
import inspect
import textwrap

import pytest
import yaml

import battery_optimizer as bo
from battery_optimizer_lib import (
    BatteryMode,
    ScheduleEntry,
    ScheduleFormatter,
    ScheduleFormatterConfig,
)


PACKAGE_YAML = "homeassistant/packages/battery_optimizer.yaml"

BULKY_ENTITIES = (
    "sensor.battery_optimizer_schedule",
    "sensor.battery_optimizer_schedule_markdown",
    "sensor.battery_optimizer_load_profile",
)


def make_formatter(slot_minutes: int = 15) -> ScheduleFormatter:
    return ScheduleFormatter(
        config=ScheduleFormatterConfig(
            slot_minutes=slot_minutes,
            slot_hours=slot_minutes / 60.0,
            battery_capacity=14.3,
            charge_rate=4.5,
            discharge_rate=4.5,
            export_discharge_rate=0.0,
            efficiency=0.95,
            battery_wear_cost=0.017,
            decision_log_level=1,
        ),
        log_func=lambda *_a, **_k: None,
    )


def aware(year, month, day, hour, minute=0):
    return datetime.datetime(
        year, month, day, hour, minute, tzinfo=datetime.timezone.utc
    )


def entry(time, mode, export_rate=None):
    return ScheduleEntry(
        time=time,
        mode=mode,
        reason="test",
        export_rate=export_rate,
    )


# ===========================================================================
# ScheduleFormatter.summarize_schedule
# ===========================================================================

class TestSummarizeSchedule:
    def test_empty_schedule(self):
        summary = make_formatter().summarize_schedule({})
        assert summary == {
            "schedule_slots": 0,
            "schedule_start": None,
            "schedule_end": None,
            "charge_hours": 0.0,
            "discharge_hours": 0.0,
            "hold_hours": 0.0,
            "charge_slots_count": 0,
            "discharge_slots_count": 0,
        }

    def test_mixed_schedule_at_15_minute_slots(self):
        base = aware(2026, 9, 8, 10)
        schedule = {}
        # 4 CHARGE, 2 DISCHARGE, 2 HOLD = 8 slots = 2 hours
        modes = [
            BatteryMode.CHARGE,
            BatteryMode.CHARGE,
            BatteryMode.CHARGE,
            BatteryMode.CHARGE,
            BatteryMode.HOLD,
            BatteryMode.HOLD,
            BatteryMode.DISCHARGE,
            BatteryMode.DISCHARGE,
        ]
        for i, mode in enumerate(modes):
            t = base + datetime.timedelta(minutes=15 * i)
            schedule[t] = entry(t, mode)

        summary = make_formatter().summarize_schedule(schedule)

        assert summary["schedule_slots"] == 8
        assert summary["schedule_start"] == base.isoformat()
        # LAST slot's start + slot_minutes: the exclusive end of the plan.
        assert summary["schedule_end"] == (base + datetime.timedelta(hours=2)).isoformat()
        assert summary["charge_hours"] == 1.0
        assert summary["hold_hours"] == 0.5
        assert summary["discharge_hours"] == 0.5
        assert summary["charge_slots_count"] == 4
        assert summary["discharge_slots_count"] == 2

    def test_hours_follow_the_configured_slot_length(self):
        base = aware(2026, 9, 8, 10)
        schedule = {}
        for i in range(3):
            t = base + datetime.timedelta(minutes=30 * i)
            schedule[t] = entry(t, BatteryMode.CHARGE)

        summary = make_formatter(slot_minutes=30).summarize_schedule(schedule)
        assert summary["charge_hours"] == 1.5
        assert summary["schedule_end"] == (base + datetime.timedelta(hours=1.5)).isoformat()

    def test_export_and_self_consumption_both_count_as_discharge(self):
        """The summary is per BatteryMode; export_rate does not split it."""
        base = aware(2026, 9, 8, 18)
        schedule = {
            base: entry(base, BatteryMode.DISCHARGE, export_rate=100),
            base + datetime.timedelta(minutes=15): entry(
                base + datetime.timedelta(minutes=15), BatteryMode.DISCHARGE
            ),
        }
        summary = make_formatter().summarize_schedule(schedule)
        assert summary["discharge_slots_count"] == 2
        assert summary["discharge_hours"] == 0.5

    def test_unsorted_keys_still_give_the_true_bounds(self):
        base = aware(2026, 9, 8, 10)
        times = [base + datetime.timedelta(minutes=15 * i) for i in range(4)]
        schedule = {t: entry(t, BatteryMode.HOLD) for t in reversed(times)}
        summary = make_formatter().summarize_schedule(schedule)
        assert summary["schedule_start"] == times[0].isoformat()
        assert summary["schedule_end"] == (times[-1] + datetime.timedelta(minutes=15)).isoformat()

    def test_the_summary_stays_small(self):
        """Whatever the horizon, this is what the recorded sensor carries."""
        base = aware(2026, 9, 8, 0)
        schedule = {}
        for i in range(200):
            t = base + datetime.timedelta(minutes=15 * i)
            schedule[t] = entry(t, BatteryMode.CHARGE if i % 3 else BatteryMode.HOLD)
        summary = make_formatter().summarize_schedule(schedule)
        assert bo._attributes_size_bytes(summary) < 512


# ===========================================================================
# Why the split exists
# ===========================================================================

class TestAttributeSizes:
    def _realistic_schedule(self, slots: int):
        base = aware(2026, 9, 8, 0)
        schedule = {}
        for i in range(slots):
            t = base + datetime.timedelta(minutes=15 * i)
            if i % 5 == 0:
                mode, reason, rate = (
                    BatteryMode.CHARGE,
                    "cheap slot: landed charge below evening avoided import",
                    None,
                )
            elif i % 5 == 1:
                mode, reason, rate = (
                    BatteryMode.DISCHARGE,
                    "expensive slot: serve net load from the battery",
                    None,
                )
            elif i % 5 == 2:
                mode, reason, rate = (
                    BatteryMode.DISCHARGE,
                    "export: spot above wear plus export fee",
                    100,
                )
            else:
                mode, reason, rate = (
                    BatteryMode.HOLD,
                    "hold: keeping energy for a higher-valued slot",
                    None,
                )
            e = entry(t, mode, export_rate=rate)
            e.reason = reason
            e.marginal_value_eur_kwh = 0.1234
            e.value_basis = "avoided-import"
            schedule[t] = e
        return schedule

    def test_the_schedule_list_alone_exceeds_the_recorder_limit(self):
        formatter = make_formatter()
        schedule = self._realistic_schedule(200)
        blob = {"schedule": formatter.format_schedule_list(schedule)}
        size = bo._attributes_size_bytes(blob)
        assert size > bo.HA_MAX_STATE_ATTRS_BYTES, size

    def test_the_summary_of_the_same_plan_is_under_512_bytes(self):
        formatter = make_formatter()
        summary = formatter.summarize_schedule(self._realistic_schedule(200))
        assert bo._attributes_size_bytes(summary) < 512

    def test_the_budget_is_half_the_recorder_limit(self):
        assert bo.HA_MAX_STATE_ATTRS_BYTES == 16384
        assert bo.MAIN_SENSOR_ATTRS_BUDGET_BYTES == 8192
        assert bo.MAIN_SENSOR_ATTRS_BUDGET_BYTES < bo.HA_MAX_STATE_ATTRS_BYTES

    def test_size_helper_survives_non_json_values(self):
        """HA tolerates datetimes and enums; the guard must not raise on them."""
        blob = {"t": aware(2026, 9, 8, 10), "mode": BatteryMode.CHARGE}
        assert bo._attributes_size_bytes(blob) > 0


# ===========================================================================
# Source scan: the orchestrator is not unit-tested (CLAUDE.md)
# ===========================================================================

def _main_sensor_attribute_keys() -> set:
    """Keys of the dict published as sensor.battery_optimizer's attributes.

    The method builds `main_attributes` and hands it to set_state, so parse the
    assignment rather than the call.
    """
    source = inspect.getsource(bo.BatteryOptimizer._update_schedule_sensor)
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "main_attributes"
            for t in node.targets
        ):
            assert isinstance(node.value, ast.Dict)
            return {
                k.value
                for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
    raise AssertionError("main_attributes dict not found in _update_schedule_sensor")


class TestMainSensorStaysLean:
    @pytest.mark.parametrize(
        "key", ["schedule", "load_profile_stats", "temp_aware_rates"]
    )
    def test_the_bulky_payloads_are_gone(self, key):
        assert key not in _main_sensor_attribute_keys()

    def test_the_summary_and_the_pointer_are_published(self):
        source = inspect.getsource(bo.BatteryOptimizer._update_schedule_sensor)
        assert "summarize_schedule" in source
        assert '"schedule_entity": "sensor.battery_optimizer_schedule"' in source

    def test_the_schedule_entity_is_published_by_the_same_method(self):
        source = inspect.getsource(bo.BatteryOptimizer._update_schedule_sensor)
        assert 'set_state("sensor.battery_optimizer_schedule"' in source

    def test_main_sensor_replaces_attributes_instead_of_merging(self):
        """The host's set_state merges attributes by default; a removed key
        would survive in HA until the next HA restart. Measured on the first
        deploy: 18 550 bytes with `schedule` still present."""
        source = inspect.getsource(bo.BatteryOptimizer._update_schedule_sensor)
        main_call = source.split('set_state("sensor.battery_optimizer",', 1)[1]
        main_call = main_call.split(")", 1)[0]
        assert "replace=True" in main_call
        assert '"schedule": schedule_data' in source
        # HA states are strings; an int 0 is falsy and gets dropped from the POST.
        assert "state=str(len(self.schedule))" in source

    def test_the_budget_guard_runs_before_the_publish(self):
        source = inspect.getsource(bo.BatteryOptimizer._update_schedule_sensor)
        assert "_check_main_sensor_attr_budget(main_attributes)" in source
        guard = inspect.getsource(bo.BatteryOptimizer._check_main_sensor_attr_budget)
        assert "MAIN_SENSOR_ATTRS_BUDGET_BYTES" in guard
        assert "HA_MAX_STATE_ATTRS_BYTES" in guard
        # Never truncate or drop silently.
        assert "level=\"WARNING\"" in guard

    def test_the_diagnostics_attributes_survived_the_split(self):
        """Guarded elsewhere too (test_price_recovery / test_refinement_reporting)."""
        keys = _main_sensor_attribute_keys()
        for key in ("price_horizon", "rate_refinement", "current_mode",
                    "next_charge", "next_discharge", "app_version", "slot_minutes"):
            assert key in keys

    def test_the_load_profile_table_has_its_own_entity(self):
        source = inspect.getsource(
            bo.BatteryOptimizer._update_load_profile_stats_sensor
        )
        assert '"sensor.battery_optimizer_load_profile"' in source
        assert '"load_profile_stats": self._get_load_profile_stats()' in source
        assert "state=str(" in source
        # Reached after every observation, so the entity exists after a restart.
        wired = inspect.getsource(bo.BatteryOptimizer._update_load_profile_sensors)
        assert "_update_load_profile_stats_sensor()" in wired

    def test_nothing_reads_a_schedule_back_out_of_ha(self):
        """CLAUDE.md: there is no restart override at all."""
        source = inspect.getsource(bo)
        assert "get_state(\"sensor.battery_optimizer_schedule\"" not in source
        assert "state_attr" not in source


# ===========================================================================
# HA package
# ===========================================================================

class TestPackageYaml:
    @pytest.fixture(scope="class")
    def package(self):
        with open(PACKAGE_YAML, encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def test_it_parses(self, package):
        assert isinstance(package, dict)
        assert "template" in package
        assert "group" in package

    def test_the_bulky_entities_are_excluded_from_the_recorder(self, package):
        excluded = package["recorder"]["exclude"]["entities"]
        for entity in BULKY_ENTITIES:
            assert entity in excluded

    def test_no_template_iterates_the_schedule_list_any_more(self):
        with open(PACKAGE_YAML, encoding="utf-8") as fh:
            raw = fh.read()
        assert "state_attr('sensor.battery_optimizer', 'schedule')" not in raw
        assert 'state_attr("sensor.battery_optimizer", "schedule")' not in raw

    def test_the_schedule_hours_sensor_reads_the_summary(self, package):
        sensors = [
            sensor
            for block in package["template"]
            for sensor in block.get("sensor", [])
        ]
        by_name = {s["name"]: s for s in sensors}
        hours = by_name["Battery Schedule Hours"]
        assert "charge_hours" in hours["state"]
        assert "discharge_hours" in hours["state"]
        assert "h charge /" in hours["state"]
        for attr in ("charge_hours", "discharge_hours", "hold_hours"):
            assert attr in hours["attributes"]
            assert attr in hours["attributes"][attr]

    def test_the_new_entities_are_in_the_group(self, package):
        entities = package["group"]["battery_optimizer"]["entities"]
        assert "sensor.battery_optimizer_schedule" in entities
        assert "sensor.battery_optimizer_load_profile" in entities
