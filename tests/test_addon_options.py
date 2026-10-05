"""Add-on options -> args -> BatteryOptimizerConfig.

The add-on replaces apps.yaml with Supervisor-validated options. That swap is
only safe if (a) every key the config loader reads can be set as an option,
(b) every example value survives the Supervisor's type coercion unchanged, and
(c) the args built from the options give the IDENTICAL config apps.yaml gave.
The connection keys are the one deliberate difference: SUPERVISOR_TOKEN and
http://supervisor/core replace ha_url / ha_token.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest
import yaml

from battery_optimizer_lib.addon_main import (
    CONNECTION_KEYS,
    HOST_OPTIONS,
    SUPERVISOR_CORE_URL,
    options_to_args,
    shadow_policy,
)
from battery_optimizer_lib.config import BatteryOptimizerConfig

REPO = Path(__file__).resolve().parent.parent
ADDON_CONFIG = REPO / "addon" / "battery_optimizer" / "config.yaml"
EXAMPLES = [
    REPO / "appdaemon" / "apps" / "apps.yaml.example",
    REPO / "addon" / "battery_optimizer" / "options.example.yaml",
]
# Keys only AppDaemon ever consumed; meaningless to the add-on.
APPDAEMON_ONLY_KEYS = {"module", "class", "pin_app", "pin_thread"}


def _schema():
    with open(ADDON_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["schema"]


def _known_config_keys():
    return _load_script("smoke_config").known_config_keys()


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADDON_OPTIONS = _load_script("addon_options")
supervisor_coerce = ADDON_OPTIONS.supervisor_coerce


def _example_args(path):
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if "battery_optimizer" in data and isinstance(data["battery_optimizer"], dict):
        data = data["battery_optimizer"]
    return data


def _config_dict(args):
    cfg = BatteryOptimizerConfig.from_args(dict(args))
    return dataclasses.asdict(cfg)


def test_every_config_key_is_an_option():
    schema = _schema()
    readable = _known_config_keys() - CONNECTION_KEYS
    missing = sorted(readable - set(schema))
    assert not missing, f"from_args reads keys the add-on cannot set: {missing}"


def test_no_dead_options():
    schema = _schema()
    dead = sorted(set(schema) - HOST_OPTIONS - _known_config_keys())
    assert not dead, f"options nothing reads: {dead}"


@pytest.mark.parametrize("example", [p for p in EXAMPLES if p.exists()],
                         ids=lambda p: p.name)
def test_example_yields_identical_config(example):
    schema = _schema()
    raw = _example_args(example)
    keys = set(raw) - APPDAEMON_ONLY_KEYS - CONNECTION_KEYS - HOST_OPTIONS
    missing = sorted(k for k in keys if k not in schema)
    assert not missing, f"{example.name} keys missing from the schema: {missing}"

    # The example is the LIVE-mode configuration.
    options = ADDON_OPTIONS.apps_yaml_to_options(raw, schema, shadow=False)
    token = "supervisor-token"
    via_addon = options_to_args(options, supervisor_token=token)

    via_appsyaml = {k: v for k, v in raw.items() if k not in APPDAEMON_ONLY_KEYS}
    via_appsyaml["ha_url"] = SUPERVISOR_CORE_URL
    via_appsyaml["ha_token"] = token
    assert _config_dict(via_addon) == _config_dict(via_appsyaml)


def test_connection_comes_from_the_supervisor():
    args = options_to_args({"ha_url": "http://old", "ha_token": "old",
                            "shadow_mode": False}, supervisor_token="T")
    assert args["ha_url"] == SUPERVISOR_CORE_URL
    assert args["ha_token"] == "T"
    assert "shadow_mode" not in args


def test_shadow_mode_forces_dry_run_and_a_suffix():
    options = {"shadow_mode": True, "device_id": "05005d2c", "entity_suffix": "_x"}
    args = options_to_args(options, supervisor_token="T")
    assert args["device_id"] == ""
    assert BatteryOptimizerConfig.from_args(args).device_id == ""
    assert shadow_policy(options).entity("sensor.battery_optimizer") == \
        "sensor.battery_optimizer_x"
    assert shadow_policy({"shadow_mode": False}) is None


def test_live_mode_keeps_device_id():
    args = options_to_args({"shadow_mode": False, "device_id": "dev"},
                           supervisor_token="T")
    assert args["device_id"] == "dev"


def test_terminal_value_zero_survives_string_coercion():
    """The live policy is terminal_energy_value_eur_kwh: 0; the option is a
    string (it also takes "auto"), so 0 arrives as "0"."""
    schema = _schema()
    value = supervisor_coerce("terminal_energy_value_eur_kwh", 0,
                              schema["terminal_energy_value_eur_kwh"])
    cfg = BatteryOptimizerConfig.from_args(
        options_to_args({"terminal_energy_value_eur_kwh": value},
                        supervisor_token="T"))
    assert cfg.terminal_energy_value_eur_kwh == 0.0


def test_default_options_validate_against_the_schema():
    with open(ADDON_CONFIG, encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    for key, value in manifest["options"].items():
        supervisor_coerce(key, value, manifest["schema"][key])
    assert manifest["options"]["shadow_mode"] is True, \
        "a fresh install must start in shadow mode"
    assert manifest["homeassistant_api"] is True
    assert "addon_config:rw" in manifest["map"]


def test_fractional_value_under_an_int_type_is_rejected():
    with pytest.raises(ADDON_OPTIONS.OptionError):
        supervisor_coerce("slot_minutes", 7.5, "int?")


def test_unknown_apps_yaml_key_is_rejected():
    with pytest.raises(ADDON_OPTIONS.OptionError):
        ADDON_OPTIONS.apps_yaml_to_options({"slot_minuts": 15}, _schema(), shadow=True)


def test_conversion_drops_appdaemon_and_connection_keys():
    options = ADDON_OPTIONS.apps_yaml_to_options(
        {"module": "battery_optimizer", "class": "BatteryOptimizer",
         "pin_app": False, "ha_url": "http://x", "ha_token": "secret",
         "efficiency": 0.92, "terminal_energy_value_eur_kwh": 0,
         "device_id": "05005d2c"},
        _schema(), shadow=True)
    assert options == {"efficiency": 0.92, "terminal_energy_value_eur_kwh": "0",
                       "shadow_mode": True, "entity_suffix": "_shadow",
                       "device_id": ""}
