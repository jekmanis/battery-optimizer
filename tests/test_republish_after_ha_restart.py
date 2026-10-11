"""After an HA restart the app republishes its entities from memory.

Home Assistant keeps a state created through ``POST /api/states`` only until
its own restart. On 2026-10-11 HA restarted at 21:21, the host reconnected at
21:22:29 (connection #3) and kept executing slots, but
``sensor.battery_optimizer`` and every other app-published entity answered
"Entity not found" until the next re-plan happened to publish them again.

The fix: ``HAHost.listen_connected`` reports every (re)connect, and the app's
``_republish_entities`` (also bound to ``homeassistant_started``) re-runs the
existing publishers. These tests run the real orchestrator against the fake
HA, simulate the restart (state store wiped, websocket dropped) and check that
the entities come back with the same plan, that nothing was re-planned and that
no service at all - least of all ``growatt_modbus/set_wit_mode`` - was called.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time

import pytest

from battery_optimizer_lib.addon_main import options_to_args, shadow_policy
from battery_optimizer_lib.ha_host import HOST_CONNECTED_EVENT, HAHost

from tests.fake_ha import FakeHA
from tests.test_addon_shadow_run import RIGA, _seed, _wait

pytestmark = pytest.mark.skipif(RIGA is None, reason="no tz database")

SUFFIX = "_shadow"

# The entities the task is about; the republish covers more (see
# `_republish_entities`), these are the ones a user notices.
KEY_ENTITIES = (
    "sensor.battery_optimizer",
    "sensor.battery_optimizer_schedule",
    "sensor.battery_optimizer_schedule_markdown",
    "sensor.battery_optimizer_load_profile",
    "sensor.battery_inverter_control_health",
    "sensor.battery_learning_stats",
    "sensor.battery_prediction_accuracy",
)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append((record.levelname, record.getMessage()))


def _options(tmp_path, shadow: bool) -> dict:
    options = {
        "device_id": "05005d2cc8b5b7acce146af1698e9fb3",
        "nordpool_config_entry": "entry",
        "nordpool_area": "LV",
        "soc_sensor": "sensor.growatt_battery_battery_soc",
        "pv_power_sensor": "sensor.growatt_solar_solar_total_power",
        "battery_temp_sensor": "sensor.growatt_battery_battery_temperature",
        "battery_charge_sensor": "sensor.growatt_battery_battery_charge_today",
        "battery_discharge_sensor": "sensor.growatt_battery_battery_discharge_today",
        "load_power_sensor": "sensor.growatt_load_house_consumption",
        "load_profile_file": str(tmp_path / "load_profile.json"),
        "learning_data_file": str(tmp_path / "battery_learning_data.json"),
        "prediction_tracker_file": str(tmp_path / "prediction_tracker.json"),
        "pv_profile_file": str(tmp_path / "pv_profile.json"),
        "soc_step_percent": 2.0,
        "efficiency": 0.92,
        "terminal_energy_value_eur_kwh": "0",
        # No delayed verification timer: the test stops the scheduler.
        "verify_source": "none",
    }
    if shadow:
        options.update(shadow_mode=True, entity_suffix=SUFFIX)
    return options


def _start_app(tmp_path, shadow: bool):
    import battery_optimizer

    fake = FakeHA().start()
    _seed(fake)
    options = _options(tmp_path, shadow)
    host = HAHost(ws_url=fake.ws_url, rest_url=fake.rest_url, token=fake.token,
                  time_zone="Europe/Riga", shadow=shadow_policy(options),
                  reconnect_delays=(0.05,))
    assert host.start(wait_timeout=5)
    logs = _ListHandler()
    host._logger.addHandler(logs)
    args = options_to_args(options, supervisor_token=fake.token,
                           core_url=fake.rest_url)
    app = battery_optimizer.BatteryOptimizer(host=host, args=args)
    app.initialize()
    return fake, host, app, logs


@pytest.fixture(params=[False, True], ids=["live", "shadow"])
def planned_app(request, tmp_path):
    """The real orchestrator after its startup optimization, timers stopped.

    The scheduler is stopped once the startup plan is out so no periodic
    callback (adaptive pass, slot execution) can run during the test: whatever
    is published or called afterwards came from the republish alone.
    """
    shadow = request.param
    suffix = SUFFIX if shadow else ""
    fake, host, app, logs = _start_app(tmp_path, shadow)
    try:
        main = "sensor.battery_optimizer" + suffix
        assert _wait(
            lambda: (fake.states.get(main, {}).get("attributes", {})
                     .get("schedule_slots", 0) > 0),
            timeout=90,
        ), "startup optimization never published a schedule"
        # Let the startup slot execution (and its set_wit_mode) settle.
        time.sleep(1.0)
        host.scheduler.stop()
        # The learning sensor is only published by a charge/discharge record;
        # publish it once so "it existed before the restart" holds for all.
        with app._lock:
            app._update_learning_sensor()
            app._update_prediction_accuracy_sensor()
        yield fake, host, app, logs, suffix
    finally:
        try:
            app.terminate()
        finally:
            host.stop()
            fake.stop()


def _app_entities(fake: FakeHA):
    return sorted({
        r["path"][len("/api/states/"):] for r in fake.rest_requests
        if r["method"] == "POST" and r["path"].startswith("/api/states/")
    })


def _forbid_planning(app, monkeypatch):
    """Any planning or execution entry point reached afterwards is recorded."""
    reached = []
    for name in ("full_optimize", "adaptive_optimize",
                 "_recalculate_remaining_schedule", "execute_scheduled_mode",
                 "find_optimal_schedule", "_apply_mode_tracked"):
        original = getattr(app, name)

        def _spy(*args, _name=name, _original=original, **kwargs):
            reached.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(app, name, _spy)
    return reached


def _plan_fingerprint(fake: FakeHA, suffix: str):
    main = fake.states["sensor.battery_optimizer" + suffix]["attributes"]
    schedule = fake.states["sensor.battery_optimizer_schedule" + suffix]
    return {
        "mode": fake.states["sensor.battery_optimizer" + suffix]["state"],
        "census": {k: main.get(k) for k in (
            "schedule_slots", "schedule_start", "schedule_end",
            "charge_slots_count", "discharge_slots_count", "last_optimization",
            "app_version",
        )},
        "slots": schedule["state"],
        "schedule": schedule["attributes"]["schedule"],
        "health": fake.states["sensor.battery_inverter_control_health" + suffix]["state"],
        "load_profile_obs": fake.states[
            "sensor.battery_optimizer_load_profile" + suffix]["state"],
    }


def test_ha_restart_republishes_every_entity_without_replanning(planned_app, monkeypatch):
    fake, host, app, logs, suffix = planned_app

    published = _app_entities(fake)
    for entity in KEY_ENTITIES:
        assert entity + suffix in published, entity
    before = _plan_fingerprint(fake, suffix)
    last_optimization = app.last_optimization
    schedule_before = dict(app.schedule)
    calls_before = list(fake.calls())
    if not suffix:
        # The live run really did command the inverter at startup, so "no
        # set_wit_mode after the restart" below is not vacuous.
        assert fake.calls("growatt_modbus/set_wit_mode")
    reached = _forbid_planning(app, monkeypatch)
    connections = host.connection.connect_count

    # HA restarts: it forgets every API-created state and drops the socket.
    with fake._lock:
        for entity in published:
            fake.states.pop(entity, None)
    fake.drop_connections()

    assert _wait(lambda: host.connection.connect_count == connections + 1, 10)
    assert _wait(lambda: all(e in fake.states for e in published), 10), (
        sorted(set(published) - set(fake.states))
    )
    time.sleep(0.5)  # anything else the reconnect might have set off

    assert _plan_fingerprint(fake, suffix) == before
    assert app.last_optimization == last_optimization
    assert app.schedule == schedule_before
    assert reached == []
    # Not one service call: no set_wit_mode, no input_* write, not even a
    # price fetch.
    assert fake.calls() == calls_before
    # Shadow mode: everything came back under the suffix, nothing under the
    # live names.
    if suffix:
        assert all(e.endswith(suffix) for e in _app_entities(fake))
        assert "sensor.battery_optimizer" not in fake.states

    lines = [m for _lvl, m in logs.lines if m.startswith("Republished")]
    assert lines == [
        f"Republished {len(published)} entities after HA reconnect "
        f"#{connections + 1} (from memory: nothing re-planned, nothing sent)"
    ], lines


def test_homeassistant_started_republishes(planned_app, monkeypatch):
    fake, host, app, logs, suffix = planned_app

    published = _app_entities(fake)
    before = _plan_fingerprint(fake, suffix)
    calls_before = list(fake.calls())
    reached = _forbid_planning(app, monkeypatch)

    with fake._lock:
        for entity in published:
            fake.states.pop(entity, None)
    fake.fire_event("homeassistant_started")

    assert _wait(lambda: all(e in fake.states for e in published), 10), (
        sorted(set(published) - set(fake.states))
    )
    assert _plan_fingerprint(fake, suffix) == before
    assert reached == []
    assert fake.calls() == calls_before
    assert any(
        m.startswith(f"Republished {len(published)} entities after "
                     "homeassistant_started event")
        for _lvl, m in logs.lines
    ), logs.lines


def test_first_connection_and_an_uninitialized_app_publish_nothing(planned_app):
    import battery_optimizer

    fake, host, app, logs, suffix = planned_app
    posts = len(fake.rest_requests)

    # Connection #1 is the one `initialize` already published after.
    app._republish_entities(HOST_CONNECTED_EVENT, {"connection": 1}, {})
    assert len(fake.rest_requests) == posts

    # A connect that lands before `initialize` (or on a bare test double) is
    # a no-op, not a crash.
    bare = battery_optimizer.BatteryOptimizer()
    bare._republish_entities(HOST_CONNECTED_EVENT, {"connection": 2}, {})
    bare._republish_entities("homeassistant_started", {}, {})

    # After terminate() the app republishes nothing either.
    app.terminate()
    app._republish_entities(HOST_CONNECTED_EVENT, {"connection": 5}, {})
    assert len(fake.rest_requests) == posts


# ---------------------------------------------------------------------------
# The host hook itself
# ---------------------------------------------------------------------------


def test_listen_connected_fires_on_a_worker_after_every_reconnect(fake_ha, ha_host):
    seen = []
    done = threading.Event()

    def _on_connected(event, data, kwargs):
        # The fresh state cache is already in place when the hook fires.
        seen.append((event, data, kwargs, threading.current_thread().name,
                     ha_host.get_state("sensor.after_restart")))
        done.set()

    first = ha_host.connection.connect_count
    ha_host.listen_connected(_on_connected, tag="x")
    assert seen == []  # registration does not replay the connect it missed

    fake_ha.set_entity("sensor.after_restart", "fresh", notify=False)
    fake_ha.drop_connections()
    assert done.wait(10)
    event, data, kwargs, thread, cached = seen[0]
    assert event == HOST_CONNECTED_EVENT
    assert data == {"connection": first + 1}
    assert kwargs == {"tag": "x"}
    # Dispatched to the worker pool, never run on the websocket thread.
    assert thread.startswith("worker"), thread
    assert cached == "fresh"


def test_cancel_listen_connected(fake_ha, ha_host):
    seen = []
    handle = ha_host.listen_connected(lambda e, d, k: seen.append(d))
    assert ha_host.cancel_listen_connected(handle) is True
    connections = ha_host.connection.connect_count
    fake_ha.drop_connections()
    assert _wait(lambda: ha_host.connection.connect_count == connections + 1, 10)
    time.sleep(0.3)
    assert seen == []
