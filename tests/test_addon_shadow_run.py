"""The real orchestrator, in shadow mode, against a fake Home Assistant.

Shadow mode is the parallel run next to the live instance, so it
must be impossible for it to act: no inverter command, no write to a shared
`input_*` helper, nothing published under a name the live instance owns.

This runs `BatteryOptimizer.initialize()` and the startup `full_optimize`
through `ha_host` over a real socket, then drives every path that WOULD
write - a slot execution, a manual "Auto" (input_boolean/turn_off), the
battery-cost save (input_number/set_value), the load-profile mirror
(input_text/set_value) - and checks what actually reached HA. It is also the
orchestrator's first end-to-end wiring test on the new host.
"""

from __future__ import annotations

import datetime
import time

import pytest

from battery_optimizer_lib.addon_main import options_to_args, shadow_policy
from battery_optimizer_lib.ha_host import HAHost

from tests.fake_ha import FakeHA

try:
    from zoneinfo import ZoneInfo

    RIGA = ZoneInfo("Europe/Riga")
except Exception:  # pragma: no cover - tzdata missing
    RIGA = None

pytestmark = pytest.mark.skipif(RIGA is None, reason="no tz database")

SUFFIX = "_shadow"


def _nordpool(msg):
    """96 quarter-hours for the requested LOCAL date, cheap nights, dear evenings."""
    day = datetime.date.fromisoformat(msg["service_data"]["date"])
    start = datetime.datetime.combine(day, datetime.time(0), tzinfo=RIGA)
    entries = []
    for q in range(96):
        s = (start + datetime.timedelta(minutes=15 * q)).astimezone(datetime.timezone.utc)
        hour = q // 4
        price = 20.0 if hour < 6 else 150.0 if 17 <= hour < 21 else 60.0
        entries.append({
            "start": s.isoformat(),
            "end": (s + datetime.timedelta(minutes=15)).isoformat(),
            "price": price,
        })
    return {"LV": entries}


def _seed(fake: FakeHA) -> None:
    states = {
        "sun.sun": "above_horizon",
        "sensor.growatt_battery_battery_soc": "55",
        "sensor.growatt_solar_solar_total_power": "1200",
        "sensor.growatt_battery_battery_temperature": "21",
        "sensor.growatt_battery_battery_charge_today": "3.2",
        "sensor.growatt_battery_battery_discharge_today": "2.1",
        "sensor.growatt_load_house_consumption": "650",
        "input_boolean.battery_optimizer_enabled": "on",
        "input_boolean.battery_optimizer_override": "off",
        "input_select.battery_manual_mode": "Auto",
        "input_number.battery_min_soc": "10",
        "input_number.battery_max_soc": "100",
        "input_number.battery_pv_threshold": "500",
        "input_number.battery_avg_cost": "0.08",
        "input_number.battery_cost_basis_version": "2",
    }
    for entity, state in states.items():
        fake.set_entity(entity, state, notify=False)
    fake.register_service("nordpool/get_price_indices_for_date", _nordpool,
                          response="only")
    for service in ("input_boolean/turn_off", "input_number/set_value",
                    "input_text/set_value"):
        fake.register_service(service, lambda msg: None)
    fake.register_service("growatt_modbus/set_wit_mode",
                          lambda msg: {"success": True}, response="optional")


def _wait(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


@pytest.fixture
def shadow_app(tmp_path):
    import battery_optimizer

    fake = FakeHA().start()
    _seed(fake)
    options = {
        "shadow_mode": True,
        "entity_suffix": SUFFIX,
        # A REAL device id: shadow mode must neutralise it, not rely on it
        # being empty.
        "device_id": "05005d2cc8b5b7acce146af1698e9fb3",
        "nordpool_config_entry": "entry",
        "nordpool_area": "LV",
        "soc_sensor": "sensor.growatt_battery_battery_soc",
        "pv_power_sensor": "sensor.growatt_solar_solar_total_power",
        "battery_temp_sensor": "sensor.growatt_battery_battery_temperature",
        "battery_charge_sensor": "sensor.growatt_battery_battery_charge_today",
        "battery_discharge_sensor": "sensor.growatt_battery_battery_discharge_today",
        "load_power_sensor": "sensor.growatt_load_house_consumption",
        "load_profile_entity": "input_text.battery_load_profile",
        "load_profile_file": str(tmp_path / "load_profile.json"),
        "learning_data_file": str(tmp_path / "battery_learning_data.json"),
        "prediction_tracker_file": str(tmp_path / "prediction_tracker.json"),
        "pv_profile_file": str(tmp_path / "pv_profile.json"),
        "soc_step_percent": 2.0,
        "efficiency": 0.92,
        "terminal_energy_value_eur_kwh": "0",
        "verify_source": "registers",
    }
    host = HAHost(ws_url=fake.ws_url, rest_url=fake.rest_url, token=fake.token,
                  time_zone="Europe/Riga", shadow=shadow_policy(options),
                  reconnect_delays=(0.05,))
    assert host.start(wait_timeout=5)
    args = options_to_args(options, supervisor_token=fake.token,
                           core_url=fake.rest_url)
    app = battery_optimizer.BatteryOptimizer(host=host, args=args)
    app.initialize()
    yield fake, host, app
    try:
        app.terminate()
    finally:
        host.stop()
        fake.stop()


def test_shadow_run_never_writes_outside_its_own_names(shadow_app):
    fake, host, app = shadow_app

    # Startup full_optimize (run_in 1 s) publishes a plan under the suffix.
    assert _wait(
        lambda: (fake.states.get("sensor.battery_optimizer" + SUFFIX, {})
                 .get("attributes", {}).get("schedule_slots", 0) > 0),
        timeout=90,
    ), "startup optimization never published a schedule"
    main = fake.states["sensor.battery_optimizer" + SUFFIX]["attributes"]
    import battery_optimizer

    assert main["app_version"] == battery_optimizer.APP_VERSION

    # Every path that writes.
    app.execute_scheduled_mode(None)
    # Manual "Auto" only turns the override off while the override is on.
    fake.set_entity("input_boolean.battery_optimizer_override", "on")
    assert _wait(lambda: host.get_state("input_boolean.battery_optimizer_override") == "on", 5)
    app.on_manual_mode_change("input_select.battery_manual_mode", "state",
                              "Charge", "Auto", {})
    app._cost_tracker._cost_from_fallback = False
    app._cost_tracker.save_to_ha()
    app._save_load_profile()
    fake.set_entity("sensor.growatt_battery_battery_soc", "54")
    fake.set_entity("sensor.growatt_battery_battery_charge_today", "3.4")
    time.sleep(1.0)

    # Those paths really were exercised - and stopped at the host.
    suppressed = host.shadow.suppressed
    for service in ("input_boolean/turn_off", "input_number/set_value",
                    "input_text/set_value"):
        assert suppressed.get(service), f"{service} was never attempted"

    # Nothing but reads reached HA.
    called = {f"{m['domain']}/{m['service']}" for m in fake.calls()}
    assert not any(s.startswith("input_") for s in called), called
    assert "growatt_modbus/set_wit_mode" not in called
    assert called <= {"nordpool/get_price_indices_for_date",
                      "weather/get_forecasts",
                      "growatt_modbus/get_register_data"}, called

    # Every published entity carries the suffix; none of the live names moved.
    posted = [r["path"][len("/api/states/"):] for r in fake.rest_requests
              if r["method"] == "POST" and r["path"].startswith("/api/states/")]
    assert posted, "nothing was published"
    unsuffixed = sorted({e for e in posted if not e.endswith(SUFFIX)})
    assert not unsuffixed, unsuffixed
    assert "sensor.battery_optimizer" not in fake.states
    assert fake.states["input_number.battery_avg_cost"]["state"] == "0.08"

    # The dry run is the reason the inverter was never called, AND the
    # host would have refused it anyway.
    assert app.config.device_id == ""
