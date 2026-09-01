#!/usr/bin/env python
"""
Smoke-test the battery optimizer code against a real apps.yaml.

The unit suite does not cover ``battery_optimizer.py`` (the AppDaemon
orchestrator), so a config or wiring break only shows up at import time on the
HA machine.  This helper reproduces that import in the repo:

  1. installs the ``appdaemon.plugins.hass.hassapi`` mock exactly the way
     ``tests/conftest.py`` does (by importing conftest),
  2. imports ``battery_optimizer`` and every module in ``battery_optimizer_lib``,
  3. loads the given apps.yaml, finds the app whose ``module`` is
     ``battery_optimizer`` and calls ``BatteryOptimizerConfig.from_args()``,
     then derives ``AmbientServiceConfig`` and ``PvForecastServiceConfig``,
  4. prints a REDACTED summary (never a token/key/password/secret).

Exit code 0 = the deployed code can load that config; non-zero otherwise.

Usage:
    uv run python scripts/smoke_config.py <path-to-apps.yaml>
    uv run python scripts/smoke_config.py appdaemon/apps/apps.yaml.example
"""

import argparse
import importlib
import pkgutil
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
APPS_DIR = REPO_ROOT / "appdaemon" / "apps"
CONFIG_PY = APPS_DIR / "battery_optimizer_lib" / "config.py"

# Any key whose NAME contains one of these is never printed.
SECRET_HINTS = ("token", "key", "password", "secret", "passwd", "credential")

# Keys AppDaemon itself consumes, so they are not "unknown" config keys.
APPDAEMON_KEYS = {
    "module", "class", "dependencies", "plugin", "priority", "pin_app",
    "pin_thread", "log_level", "log", "disable", "global_dependencies",
    "constrain_days", "constrain_input_boolean", "constrain_input_select",
    "constrain_presence", "namespace", "sequence",
}


def is_secret(name):
    lowered = str(name).lower()
    return any(hint in lowered for hint in SECRET_HINTS)


def redact(name, value):
    if is_secret(name):
        if value in (None, ""):
            return "<empty>"
        return "<set>"
    text = str(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return text


def fail(message, exc=None):
    print("FAIL: %s" % message)
    if exc is not None:
        traceback.print_exc()
    return 1


def known_config_keys():
    """Every ``args.get("...")`` literal read by BatteryOptimizerConfig."""
    try:
        source = CONFIG_PY.read_text(encoding="utf-8")
    except OSError:
        return set()
    return set(re.findall(r"""args\.get\(\s*["']([A-Za-z0-9_]+)["']""", source))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Import the optimizer and load an apps.yaml through "
                    "BatteryOptimizerConfig.from_args()."
    )
    parser.add_argument("apps_yaml", help="Path to the apps.yaml to validate")
    parser.add_argument(
        "--app-name", default=None,
        help="Explicit app key in apps.yaml (default: the entry whose "
             "module is battery_optimizer)",
    )
    args = parser.parse_args(argv)

    apps_yaml = Path(args.apps_yaml)
    if not apps_yaml.is_file():
        return fail("apps.yaml not found: %s" % apps_yaml)

    print("Smoke test")
    print("  repo      : %s" % REPO_ROOT)
    print("  apps.yaml : %s" % apps_yaml)
    print("  python    : %s" % sys.version.split()[0])

    # --- 1. hassapi mock, exactly as tests/conftest.py installs it ----------
    sys.path.insert(0, str(TESTS_DIR))
    try:
        import conftest  # noqa: F401  (installs the appdaemon mock modules)
    except Exception as exc:
        return fail("could not import tests/conftest.py (hassapi mock): %r" % exc, exc)
    if "appdaemon.plugins.hass.hassapi" not in sys.modules:
        return fail("tests/conftest.py did not install the hassapi mock")
    print("  hassapi mock installed via tests/conftest.py")

    # --- 2. import the orchestrator and every library module ---------------
    sys.path.insert(0, str(APPS_DIR))
    try:
        import yaml
    except ImportError as exc:
        return fail(
            "PyYAML is missing. Install it with: uv pip install pyyaml "
            "(it is declared in the project's dev extra)", exc,
        )

    imported = []
    try:
        importlib.import_module("battery_optimizer")
        imported.append("battery_optimizer")
        lib = importlib.import_module("battery_optimizer_lib")
        imported.append("battery_optimizer_lib")
        for mod in pkgutil.iter_modules(lib.__path__):
            name = "battery_optimizer_lib.%s" % mod.name
            importlib.import_module(name)
            imported.append(name)
    except Exception as exc:
        last = imported[-1] if imported else "-"
        return fail("import failed after %d modules (last ok: %s): %r"
                    % (len(imported), last, exc), exc)
    print("  imported %d modules (orchestrator + battery_optimizer_lib)"
          % len(imported))

    # --- 3. load the YAML and build the config -----------------------------
    try:
        raw = yaml.safe_load(apps_yaml.read_text(encoding="utf-8"))
    except Exception as exc:
        return fail("could not parse YAML: %r" % exc, exc)
    if not isinstance(raw, dict):
        return fail("apps.yaml did not parse to a mapping")

    app_name = args.app_name
    if app_name is None:
        candidates = [
            key for key, value in raw.items()
            if isinstance(value, dict) and value.get("module") == "battery_optimizer"
        ]
        if not candidates:
            return fail("no app entry with module: battery_optimizer found")
        if len(candidates) > 1:
            print("  NOTE: %d battery_optimizer entries, using '%s'"
                  % (len(candidates), candidates[0]))
        app_name = candidates[0]
    if app_name not in raw or not isinstance(raw[app_name], dict):
        return fail("app '%s' not found in %s" % (app_name, apps_yaml))

    app_args = raw[app_name]
    print("  app entry : %s (class=%s)" % (app_name, app_args.get("class")))

    messages = []

    def log_func(msg, level="INFO"):
        messages.append("    [%s] %s" % (level, msg))

    from battery_optimizer_lib.config import BatteryOptimizerConfig
    from battery_optimizer_lib.ambient_service import AmbientServiceConfig
    from battery_optimizer_lib.pv_forecast_service import PvForecastServiceConfig

    try:
        cfg = BatteryOptimizerConfig.from_args(app_args, log_func=log_func)
    except Exception as exc:
        return fail("BatteryOptimizerConfig.from_args() raised: %r" % exc, exc)
    try:
        ambient_cfg = AmbientServiceConfig.from_main_config(cfg)
        pv_cfg = PvForecastServiceConfig.from_main_config(cfg)
    except Exception as exc:
        return fail("derived service config failed: %r" % exc, exc)

    # --- 4. redacted summary ----------------------------------------------
    print("")
    print("Config loaded OK")
    summary = []
    cfg.log_summary(summary.append, summary.append)
    for line in summary:
        print("  %s" % line)
    if messages:
        print("  from_args messages:")
        for line in messages:
            print(line)

    print("")
    print("Derived services")
    print("  ambient: weather='%s' outdoor='%s' amplitude=%sC peak=%sh "
          "cache=%dmin retry=%dmin slot=%dmin"
          % (ambient_cfg.weather_entity or "-",
             ambient_cfg.outdoor_temp_sensor or "-",
             ambient_cfg.diurnal_amplitude_c, ambient_cfg.diurnal_peak_hour,
             ambient_cfg.cache_minutes, ambient_cfg.failure_retry_minutes,
             ambient_cfg.slot_minutes))
    print("  pv     : solcast='%s' field='%s' forecast_solar_kwp=%s api_key=%s "
          "cache=%dmin retry=%dmin"
          % (pv_cfg.solcast_today_entity or "-", pv_cfg.solcast_estimate_field,
             pv_cfg.forecast_solar_kwp,
             "<set>" if pv_cfg.forecast_solar_api_key else "<empty>",
             pv_cfg.pv_forecast_cache_minutes, pv_cfg.failure_retry_minutes))

    print("")
    print("Direct control: device_id=%s (%s), inverter_mode_sensor='%s'"
          % ("<set>" if cfg.device_id else "<empty>",
             "live" if cfg.device_id else "DRY-RUN",
             cfg.inverter_mode_sensor or "(default sensor.growatt_inverter_mode)"))

    # --- 5. keys the current code does not read ---------------------------
    known = known_config_keys()
    if known:
        recognised = known | APPDAEMON_KEYS
        unknown = sorted(k for k in app_args if k not in recognised)
        if unknown:
            print("")
            print("NOTE: %d key(s) in apps.yaml are not read by this version of "
                  "the config loader (typo or stale?):" % len(unknown))
            for key in unknown:
                print("  %s: %s" % (key, redact(key, app_args[key])))
        missing = sorted(k for k in known if k not in app_args)
        if missing:
            print("")
            print("INFO: %d supported key(s) are absent from apps.yaml "
                  "(defaults apply):" % len(missing))
            print("  %s" % ", ".join(missing))

    print("")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
