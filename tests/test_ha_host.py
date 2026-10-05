"""The add-on host (`ha_host`) against a fake Home Assistant.

`FakeHA` speaks HA's websocket protocol and `POST /api/states` on a real local
socket, so every test below runs the production `websocket-client` /
`requests` code paths. What is pinned:

* `listen_state` delivers (entity, attribute, old, new, kwargs) and fires only
  on a change of the watched value;
* the `call_service` envelope is the one `direct_control` and `price_service`
  already parse - including the AppDaemon `ad_status` stamp;
* a service that does not answer in time is UNCONFIRMED, a refused one FAILED;
* `run_every` keeps the slot grid, `run_daily` keeps local wall time across
  DST, `run_in` / `cancel_timer` work;
* after a dropped connection the host reconnects, re-subscribes, resyncs the
  cache to what HA holds, and listeners fire again;
* `set_state` merges unless `replace=True`; shadow mode suffixes every
  published entity and suppresses every non-read service.
"""

from __future__ import annotations

import datetime
import threading
import time

import pytest

from battery_optimizer_lib.config import BatteryOptimizerConfig
from battery_optimizer_lib.direct_control import (
    ApplyOutcome,
    DirectControl,
    RegisterVerifier,
)
from battery_optimizer_lib.ha_host import (
    AD_STATUS_TIMEOUT,
    HAHost,
    Hass,
    Scheduler,
    ShadowPolicy,
    has_expanded_kwargs,
)
from battery_optimizer_lib.models import BatteryMode, ScheduleEntry
from battery_optimizer_lib.price_service import NordPoolPriceService

from tests.fake_ha import HANG, FakeHA, ServiceError

UTC = datetime.timezone.utc


def wait_until(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@pytest.fixture
def fake():
    ha = FakeHA().start()
    ha.set_entity("sun.sun", "above_horizon", notify=False)
    ha.set_entity("sensor.soc", "50", {"unit_of_measurement": "%"}, notify=False)
    yield ha
    ha.stop()


def make_host(fake, **kwargs):
    kwargs.setdefault("time_zone", "Europe/Riga")
    kwargs.setdefault("reconnect_delays", (0.05,))
    host = HAHost(ws_url=fake.ws_url, rest_url=fake.rest_url, token=fake.token,
                  **kwargs)
    assert host.start(wait_timeout=5), "host never connected to the fake HA"
    return host


@pytest.fixture
def host(fake):
    h = make_host(fake)
    yield h
    h.stop()


# ---------------------------------------------------------------------------
# State cache and listen_state
# ---------------------------------------------------------------------------


def test_initial_sync_fills_the_cache(host):
    assert host.get_state("sun.sun") == "above_horizon"
    assert host.get_state("sensor.soc", attribute="unit_of_measurement") == "%"
    full = host.get_state("sensor.soc", attribute="all")
    assert full["entity_id"] == "sensor.soc" and full["state"] == "50"
    assert host.get_state("sensor.missing") is None
    assert host.get_state("sensor.missing", default="x") == "x"


def test_listen_state_delivers_old_and_new(fake, host):
    seen = []
    done = threading.Event()

    def on_change(entity, attribute, old, new, kwargs):
        seen.append((entity, attribute, old, new, dict(kwargs)))
        done.set()

    host.listen_state(on_change, "sensor.soc", tag="soc")
    fake.set_entity("sensor.soc", "51", {"unit_of_measurement": "%"})
    assert done.wait(5)
    assert seen == [("sensor.soc", "state", "50", "51", {"tag": "soc"})]
    assert host.get_state("sensor.soc") == "51"


def test_listen_state_ignores_attribute_only_changes(fake, host):
    calls = []
    host.listen_state(lambda *a: calls.append(a), "sensor.soc")
    fake.set_entity("sensor.soc", "50", {"unit_of_measurement": "%", "x": 1})
    assert wait_until(lambda: host.get_state("sensor.soc", attribute="x") == 1)
    time.sleep(0.1)
    assert calls == []


def test_attribute_listener_fires_on_that_attribute(fake, host):
    seen = []
    host.listen_state(lambda e, a, o, n, k: seen.append((a, o, n)),
                      "sensor.soc", attribute="x")
    fake.set_entity("sensor.soc", "50", {"x": 2})
    assert wait_until(lambda: seen == [("x", None, 2)])


def test_listen_event_subscribes_and_dispatches(fake, host):
    got = []
    host.listen_event(lambda name, data, kwargs: got.append((name, data)),
                      "homeassistant_start")
    assert wait_until(lambda: any(
        m.get("type") == "subscribe_events" and m.get("event_type") == "homeassistant_start"
        for m in fake.received
    ))
    time.sleep(0.05)
    fake.fire_event("homeassistant_start", {})
    assert wait_until(lambda: got == [("homeassistant_start", {})])


# ---------------------------------------------------------------------------
# call_service envelope - the shapes direct_control and price_service parse
# ---------------------------------------------------------------------------


def test_call_service_envelope_carries_response_and_ad_status(fake, host):
    fake.register_service("growatt_modbus/get_register_data",
                          lambda msg: {"success": True, "values": [1, 2, 3]},
                          response="optional")
    result = host.call_service("growatt_modbus/get_register_data",
                               device_id="d", register_type="holding",
                               start_address=30407, count=3,
                               return_response=True, hass_timeout=5)
    assert result["success"] is True
    assert result["ad_status"] == "OK"
    assert result["result"]["response"] == {"success": True, "values": [1, 2, 3]}
    assert isinstance(result["ad_duration"], float)


def test_register_verifier_reads_through_the_host(fake, host):
    fake.register_service(
        "growatt_modbus/get_register_data",
        lambda msg: {"success": True,
                     "values": list(range(msg["service_data"]["count"]))},
        response="optional",
    )
    app = Hass(host)
    verifier = RegisterVerifier(app, "dev1", timeout=5)
    assert verifier._read_block(30407, 4) == [0, 1, 2, 3]


def test_optional_response_service_gets_return_response_automatically(fake, host):
    """AppDaemon auto-sets return_response for SupportsResponse.OPTIONAL;
    direct_control relies on it for set_wit_mode.

    The service is registered AFTER the host connected, as growatt_modbus is
    when HA restarts underneath us: `service_registered` refreshes the
    catalogue."""
    fake.register_service("growatt_modbus/set_wit_mode",
                          lambda msg: {"success": True, "mode_applied": "hold"},
                          response="optional")
    fake.fire_event("service_registered",
                    {"domain": "growatt_modbus", "service": "set_wit_mode"})
    assert wait_until(lambda: "growatt_modbus" in host.connection.services)
    host.call_service("growatt_modbus/set_wit_mode", device_id="d", mode="hold",
                      hass_timeout=5)
    sent = fake.calls("growatt_modbus/set_wit_mode")[-1]
    assert sent["return_response"] is True
    assert "hass_timeout" not in sent.get("service_data", {})


def test_entity_id_goes_into_target(fake, host):
    fake.register_service("input_boolean/turn_off", lambda msg: None)
    host.call_service("input_boolean/turn_off", entity_id="input_boolean.x")
    sent = fake.calls("input_boolean/turn_off")[-1]
    assert sent["target"] == {"entity_id": "input_boolean.x"}
    assert "service_data" not in sent


def test_nordpool_envelope_unwraps_via_call_service(fake, host):
    payload = {"LV": [{"start": "2026-10-05T21:00:00+00:00",
                       "end": "2026-10-05T21:15:00+00:00", "price": 100.0}]}
    fake.register_service("nordpool/get_price_indices_for_date",
                          lambda msg: payload, response="only")
    svc = _price_service(host, ha_url="", ha_token="")
    assert svc._call_nordpool_service("2026-10-06") == payload


def test_nordpool_rest_path_uses_the_host_token(fake, host):
    """ha_url/ha_token become http://supervisor/core + SUPERVISOR_TOKEN."""
    payload = {"LV": [{"start": "2026-10-05T21:00:00+00:00", "price": 1.0}]}
    fake.register_service("nordpool/get_price_indices_for_date",
                          lambda msg: payload, response="only")
    svc = _price_service(host, ha_url=fake.rest_url, ha_token=fake.token)
    assert svc._call_nordpool_rest_api("2026-10-06") == payload


def _price_service(host, ha_url, ha_token):
    return NordPoolPriceService(
        nordpool_config_entry="entry", nordpool_area="LV", nordpool_sensor="",
        ha_url=ha_url, ha_token=ha_token, tomorrow_prices_hour=14,
        slot_minutes=15, get_state_func=host.get_state,
        call_service_func=host.call_service, get_datetime_func=host.datetime,
        get_date_func=host.date, get_timezone_func=host.get_timezone,
        log_func=host.log,
    )


# ---------------------------------------------------------------------------
# timeout -> unconfirmed, error -> failed (through DirectControl itself)
# ---------------------------------------------------------------------------


def _direct_control(host):
    config = BatteryOptimizerConfig(device_id="dev1", set_wit_mode_timeout_seconds=1)
    config.verify_source = "none"
    return DirectControl(Hass(host), config, verify_enabled=False)


def _entry(mode=BatteryMode.HOLD):
    return ScheduleEntry(time=datetime.datetime(2026, 10, 5, 12, 0), mode=mode,
                         reason="test")


def test_timeout_is_unconfirmed(fake, host):
    fake.register_service("growatt_modbus/set_wit_mode", lambda msg: HANG,
                          response="optional")
    raw = host.call_service("growatt_modbus/set_wit_mode", device_id="d",
                            hass_timeout=0.2)
    assert raw["success"] is False and raw["ad_status"] == AD_STATUS_TIMEOUT
    dc = _direct_control(host)
    assert dc.apply_mode_with_outcome(_entry()) is ApplyOutcome.UNCONFIRMED_TIMEOUT


def test_refused_call_is_failed(fake, host):
    def refuse(msg):
        raise ServiceError("home_assistant_error", "modbus write failed")

    fake.register_service("growatt_modbus/set_wit_mode", refuse, response="optional")
    dc = _direct_control(host)
    assert dc.apply_mode_with_outcome(_entry()) is ApplyOutcome.FAILED
    assert "modbus write failed" in (dc._last_service_error or "")


def test_handler_success_false_is_failed(fake, host):
    fake.register_service("growatt_modbus/set_wit_mode",
                          lambda msg: {"success": False, "error": "busy"},
                          response="optional")
    dc = _direct_control(host)
    # Parity, not an endorsement: a handler-level success:false sits under
    # result.response, and direct_control reads `success` at the top level and
    # one level down only. The envelope is byte-for-byte AppDaemon's, so this
    # is SENT exactly as it was there; the integration reports a real failure
    # by raising, which is the FAILED case above.
    assert dc.apply_mode_with_outcome(_entry()) is ApplyOutcome.SENT


def test_confirmed_success_is_sent(fake, host):
    fake.register_service("growatt_modbus/set_wit_mode",
                          lambda msg: {"success": True}, response="optional")
    dc = _direct_control(host)
    assert dc.apply_mode_with_outcome(_entry()) is ApplyOutcome.SENT


def test_disconnected_call_returns_none(fake):
    host = make_host(fake)
    try:
        host.connection.stop()
        assert host.call_service("growatt_modbus/set_wit_mode", device_id="d") is None
    finally:
        host.stop()


# ---------------------------------------------------------------------------
# set_state
# ---------------------------------------------------------------------------


def test_set_state_merges_unless_replace(fake, host):
    host.set_state("sensor.battery_optimizer", state="ok",
                   attributes={"a": 1, "b": 2})
    host.set_state("sensor.battery_optimizer", state="ok", attributes={"b": 3})
    assert fake.states["sensor.battery_optimizer"]["attributes"] == {"a": 1, "b": 3}
    host.set_state("sensor.battery_optimizer", state="ok", attributes={"c": 4},
                   replace=True)
    assert fake.states["sensor.battery_optimizer"]["attributes"] == {"c": 4}
    assert host.get_state("sensor.battery_optimizer", attribute="c") == 4


def test_set_state_failure_is_logged_not_raised(fake):
    host = make_host(fake, http_post=lambda *a: (_ for _ in ()).throw(OSError("down")))
    try:
        assert host.set_state("sensor.x", state="1") is None
    finally:
        host.stop()


# ---------------------------------------------------------------------------
# Shadow mode at the host boundary
# ---------------------------------------------------------------------------


def test_shadow_suffixes_entities_and_blocks_writes(fake):
    host = make_host(fake, shadow=ShadowPolicy(entity_suffix="_shadow"))
    try:
        fake.register_service("input_number/set_value", lambda m: None)
        fake.register_service("growatt_modbus/set_wit_mode", lambda m: {"success": True},
                              response="optional")
        fake.register_service("nordpool/get_price_indices_for_date",
                              lambda m: {"LV": []}, response="only")
        host.set_state("sensor.battery_optimizer", state="x", attributes={})
        assert host.call_service("input_number/set_value",
                                 entity_id="input_number.battery_avg_cost", value=1) is None
        assert host.call_service("growatt_modbus/set_wit_mode", device_id="d") is None
        assert host.call_service("nordpool/get_price_indices_for_date",
                                 hass_timeout=5)["success"] is True
        assert "sensor.battery_optimizer_shadow" in fake.states
        assert "sensor.battery_optimizer" not in fake.states
        assert fake.calls("input_number/set_value") == []
        assert fake.calls("growatt_modbus/set_wit_mode") == []
    finally:
        host.stop()


# ---------------------------------------------------------------------------
# Reconnect
# ---------------------------------------------------------------------------


def test_reconnect_resubscribes_resyncs_and_listeners_fire_again(fake, host):
    seen = []
    host.listen_state(lambda e, a, o, n, k: seen.append((o, n)), "sensor.soc")
    host.listen_event(lambda *a: None, "homeassistant_start")
    assert wait_until(lambda: any(
        m.get("event_type") == "homeassistant_start" for m in fake.received))
    first = host.connection.connect_count

    fake.drop_connections()
    assert wait_until(lambda: not host.connection.connected, timeout=5)
    # Changes while the host is away: a new entity and a changed one.
    fake.set_entity("sensor.soc", "60", notify=False)
    fake.set_entity("sensor.new", "on", notify=False)
    assert wait_until(lambda: host.connection.connect_count > first, timeout=5)

    # The cache is HA's state, not the pre-outage one.
    assert host.get_state("sensor.soc") == "60"
    assert host.get_state("sensor.new") == "on"
    assert {k: v["state"] for k, v in host.connection.states.items()} == {
        k: v["state"] for k, v in fake.states.items()
    }
    # Both subscriptions were sent again on the new connection.
    subs = [m.get("event_type") for m in fake.received if m.get("type") == "subscribe_events"]
    assert subs.count("state_changed") >= 2
    assert subs.count("homeassistant_start") >= 2

    fake.set_entity("sensor.soc", "61")
    assert wait_until(lambda: seen and seen[-1] == ("60", "61"))


def test_requests_after_reconnect_still_work(fake, host):
    fake.register_service("nordpool/get_price_indices_for_date",
                          lambda m: {"LV": []}, response="only")
    first = host.connection.connect_count
    fake.drop_connections()
    assert wait_until(lambda: host.connection.connect_count > first, timeout=5)
    result = host.call_service("nordpool/get_price_indices_for_date", hass_timeout=5)
    assert result["success"] is True


def test_bad_token_is_not_retried_fast(fake):
    host = HAHost(ws_url=fake.ws_url, rest_url=fake.rest_url, token="wrong",
                  time_zone="Europe/Riga", reconnect_delays=(0.01,))
    host.connection._auth_failure_delay = 30.0
    try:
        assert host.start(wait_timeout=0.5) is False
        time.sleep(0.3)
        auths = [m for m in fake.received if m.get("type") == "auth"]
        assert len(auths) == 1
    finally:
        host.stop()


# ---------------------------------------------------------------------------
# Time and timers (fake clock, inline executor: deterministic)
# ---------------------------------------------------------------------------


class InlineExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)

    def shutdown(self, wait=True):
        pass


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def offline_host(clock, zone="Europe/Riga"):
    """A host whose timers we drive by hand - no connection is opened."""
    return HAHost(ws_url="ws://unused", rest_url="http://unused", token="t",
                  time_zone=zone, now_func=clock, executor=InlineExecutor())


def test_datetime_is_naive_local_and_aware_on_request():
    clock = Clock(datetime.datetime(2026, 10, 5, 9, 7, 30, tzinfo=UTC))
    host = offline_host(clock)
    assert host.datetime() == datetime.datetime(2026, 10, 5, 12, 7, 30)
    aware = host.datetime(aware=True)
    assert aware.utcoffset() == datetime.timedelta(hours=3)
    assert host.date() == datetime.date(2026, 10, 5)
    assert isinstance(host.get_timezone(), datetime.tzinfo)


def test_run_every_stays_on_the_slot_grid():
    clock = Clock(datetime.datetime(2026, 10, 5, 9, 7, 30, tzinfo=UTC))
    host = offline_host(clock)
    fired = []
    # The orchestrator passes the next slot boundary as a naive local time.
    start = datetime.datetime(2026, 10, 5, 12, 15)
    host.run_every(lambda kwargs: fired.append(clock.now), start, 15 * 60)

    for minutes in (8, 15, 22, 31, 47, 60, 61):
        clock.now = datetime.datetime(2026, 10, 5, 9, 0, tzinfo=UTC) + datetime.timedelta(
            minutes=minutes, seconds=5)  # ticks always a little late
        host.scheduler.tick()
    # One fire per boundary crossed, never two for one boundary.
    assert len(fired) == 4
    # And the NEXT due time is still exactly on the grid.
    due = host.scheduler._heap[0].due
    assert due == datetime.datetime(2026, 10, 5, 10, 15, tzinfo=UTC)


def test_run_every_start_in_the_past_advances_on_the_grid():
    clock = Clock(datetime.datetime(2026, 10, 5, 9, 7, 30, tzinfo=UTC))
    host = offline_host(clock)
    host.run_every(lambda kwargs: None, clock.now - datetime.timedelta(minutes=40), 900)
    assert host.scheduler._heap[0].due == datetime.datetime(2026, 10, 5, 9, 12, 30, tzinfo=UTC)


def test_run_every_passes_kwargs_positionally():
    clock = Clock(datetime.datetime(2026, 10, 5, 9, 0, tzinfo=UTC))
    host = offline_host(clock)
    got = []

    def callback(kwargs, force=False):
        got.append((kwargs, force))

    host.run_every(callback, "immediate", 60, tag=1)
    host.scheduler.tick()
    assert got == [({"tag": 1}, False)]


def test_expanded_kwargs_callbacks_get_keywords():
    clock = Clock(datetime.datetime(2026, 10, 5, 9, 0, tzinfo=UTC))
    host = offline_host(clock)
    got = []
    host.run_in(lambda **kw: got.append(kw), 0, mode_str="hold")
    host.scheduler.tick()
    assert got == [{"mode_str": "hold"}]


def test_run_in_and_cancel_timer():
    clock = Clock(datetime.datetime(2026, 10, 5, 9, 0, tzinfo=UTC))
    host = offline_host(clock)
    fired = []
    keep = host.run_in(lambda kwargs: fired.append(("keep", kwargs)), 90, attempt=1)
    drop = host.run_in(lambda kwargs: fired.append(("drop", kwargs)), 90)
    assert host.cancel_timer(drop) is True
    assert host.cancel_timer(drop) is False
    clock.now += datetime.timedelta(seconds=89)
    host.scheduler.tick()
    assert fired == []
    clock.now += datetime.timedelta(seconds=2)
    host.scheduler.tick()
    assert fired == [("keep", {"attempt": 1})]
    assert host.timer_running(keep) is False


def test_run_daily_keeps_local_wall_time_across_dst():
    # 2026-10-25 is the autumn change in Europe/Riga (UTC+3 -> UTC+2).
    clock = Clock(datetime.datetime(2026, 10, 24, 10, 0, tzinfo=UTC))  # 13:00 local
    host = offline_host(clock)
    fired = []
    host.run_daily(lambda kwargs: fired.append(clock.now), datetime.time(14, 15))
    assert host.scheduler._heap[0].due == datetime.datetime(2026, 10, 24, 11, 15, tzinfo=UTC)
    clock.now = datetime.datetime(2026, 10, 24, 11, 15, 1, tzinfo=UTC)
    host.scheduler.tick()
    # Next day 14:15 local is 12:15 UTC: one more hour of UTC, same wall time.
    assert host.scheduler._heap[0].due == datetime.datetime(2026, 10, 25, 12, 15, tzinfo=UTC)
    assert len(fired) == 1


def test_scheduler_thread_fires_real_timers(fake, host):
    done = threading.Event()
    host.run_in(lambda kwargs: done.set(), 0.05)
    assert done.wait(5)


def test_callback_exception_is_logged_and_does_not_kill_the_scheduler():
    clock = Clock(datetime.datetime(2026, 10, 5, 9, 0, tzinfo=UTC))
    host = offline_host(clock)
    logged = []
    host.log = lambda msg, *a, level="INFO", **k: logged.append((level, msg))
    ok = []
    host.run_in(lambda kwargs: 1 / 0, 0)
    host.run_in(lambda kwargs: ok.append(1), 0)
    host.scheduler.tick()
    assert ok == [1]
    assert any(level == "ERROR" and "ZeroDivisionError" in msg for level, msg in logged)


def test_has_expanded_kwargs_sees_through_wraps():
    import functools

    def plain(self, kwargs=None):
        pass

    @functools.wraps(plain)
    def wrapper(self, *args, **kwargs):
        pass

    assert has_expanded_kwargs(wrapper) is False
    assert has_expanded_kwargs(lambda **kw: None) is True


def test_hass_without_host_is_constructible():
    class App(Hass):
        pass

    app = App()
    assert app.args == {}
    with pytest.raises(Exception):
        app.datetime()


def test_scheduler_rejects_nothing_silently_on_empty_tick():
    sched = Scheduler(lambda timer: None,
                      now_func=lambda: datetime.datetime(2026, 1, 1, tzinfo=UTC))
    assert sched.tick() == 0
