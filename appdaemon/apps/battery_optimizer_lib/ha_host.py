"""
Home Assistant host for the optimizer: the AppDaemon API surface, without AppDaemon.

The orchestrator was written against AppDaemon 4.5's ``hass.Hass``. This module
provides exactly the part of that API the app uses, with the same semantics,
over Home Assistant's websocket API (state cache, events, service calls) and
REST ``POST /api/states`` (entity publication). It runs inside the add-on; the
optimizer code above it is unchanged.

Semantics copied from AppDaemon 4.5.13 on purpose, because the callers were
built and debugged against them:

* ``call_service`` returns HA's websocket result envelope with ``ad_status``
  and ``ad_duration`` stamped on it. A request that was written to the socket
  but not answered within ``hass_timeout`` (default 10 s, AppDaemon's
  ``ws_timeout``) returns ``{"success": False, "ad_status": "TIMEOUT"}`` -
  which ``direct_control`` and ``price_service`` already classify as
  UNCONFIRMED, never as a failure. A refused call returns HA's own
  ``success: False`` envelope (``ad_status: "OK"``): a confirmed failure. A
  call made while disconnected returns ``None`` (unconfirmed, nothing sent);
  a send that breaks mid-write raises (confirmed: nothing reached HA).
  ``return_response`` is set automatically for services whose HA definition
  declares a response, as AppDaemon does.
* ``listen_state`` callbacks receive ``(entity, attribute, old, new, kwargs)``
  and fire only when the watched value CHANGED.
* Timer callbacks receive ``(kwargs)`` - or ``**kwargs`` when the function
  declares a ``**`` parameter - and event callbacks ``(event, data, kwargs)``.
* ``set_state`` merges attributes into the cached ones unless
  ``replace=True``, then POSTs the result, which HA stores as given.
* ``datetime()`` is NAIVE local time, ``datetime(aware=True)`` aware.

Threading: callbacks run on a small worker pool, concurrently, exactly like
AppDaemon with ``total_threads: 4`` and ``pin_app: false``. The host does NOT
serialize them: the app's own ``CallbackLock`` (taken by ``@_timed_callback``)
does, and ``DirectControl``'s verification callback deliberately runs outside
it. Adding a host-level lock would invert the documented lock order.

Reconnects: the reader thread reconnects with backoff, re-authenticates,
re-subscribes every event type, reloads the service catalogue and replaces the
state cache with a fresh ``get_states``. Listeners and timers stay registered;
they are not re-fired for what changed during the outage (AppDaemon does not
either), the next real change reaches them.

Shadow mode (``ShadowPolicy``) is enforced HERE, at the only two exits to HA,
so no caller can forget it: every ``set_state`` entity gets a suffix and every
service outside a read-only allowlist is suppressed.
"""

from __future__ import annotations

import copy
import datetime
import functools
import heapq
import inspect
import itertools
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

try:  # zoneinfo is stdlib; the tz database comes from the OS or `tzdata`.
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python < 3.9 is not supported anyway
    ZoneInfo = None  # type: ignore

# AppDaemon's `ws_timeout` default: the wait for a service result when the
# caller passes no `hass_timeout`.
DEFAULT_WS_TIMEOUT_S = 10.0
# REST calls (set_state) - AppDaemon's http_method default.
DEFAULT_REST_TIMEOUT_S = 10.0
# Reconnect backoff. Bounded and never faster than 1 s: a refused token must
# not hammer HA (repeated auth failures can get the client IP banned).
RECONNECT_DELAYS_S: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 30.0)
AUTH_FAILURE_DELAY_S = 60.0
# Heartbeat: a ping after this much silence, a reconnect after twice that.
HEARTBEAT_IDLE_S = 30.0

AD_STATUS_OK = "OK"
AD_STATUS_TIMEOUT = "TIMEOUT"

LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Services that only READ. In shadow mode nothing else leaves the add-on.
SHADOW_READ_ONLY_SERVICES: FrozenSet[str] = frozenset({
    "nordpool/get_price_indices_for_date",
    "weather/get_forecasts",
    "growatt_modbus/get_register_data",
    "calendar/get_events",
})


class HostError(Exception):
    """A host-level failure the caller should see (bad config, auth refused)."""


@dataclass
class ShadowPolicy:
    """Parallel-run guard: publish under suffixed names, never act.

    ``entity_suffix`` is appended to every entity the app publishes
    (``sensor.battery_optimizer`` -> ``sensor.battery_optimizer_shadow``).
    Every service call outside ``allowed_services`` is suppressed and answered
    with ``None`` - the "nothing was confirmed" value, so no caller books it
    as a success. That covers the inverter (``growatt_modbus/set_wit_mode``)
    as well as the shared helpers (``input_number``/``input_boolean``/
    ``input_text``), whichever path the app takes to reach them.
    """

    entity_suffix: str = "_shadow"
    allowed_services: FrozenSet[str] = SHADOW_READ_ONLY_SERVICES
    suppressed: Dict[str, int] = field(default_factory=dict)

    def entity(self, entity_id: str) -> str:
        if not self.entity_suffix or entity_id.endswith(self.entity_suffix):
            return entity_id
        return f"{entity_id}{self.entity_suffix}"

    def allows(self, service: str) -> bool:
        return service in self.allowed_services


def has_expanded_kwargs(func: Callable) -> bool:
    """AppDaemon's rule: ``**kwargs`` in the (unwrapped) signature -> expand."""
    func = inspect.unwrap(func)
    if isinstance(func, functools.partial):
        func = func.func
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind == p.VAR_KEYWORD for p in params)


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def resolve_zone(name: Optional[str]):
    """A tzinfo for an IANA name, or None."""
    if not name or ZoneInfo is None:
        return None
    try:
        return ZoneInfo(name)
    except Exception:
        return None


def apply_process_timezone(name: Optional[str]) -> bool:
    """Make the process-local zone HA's zone.

    The orchestrator's ``_get_local_timezone`` falls back to
    ``datetime.now().astimezone().tzinfo`` because ``self.datetime()`` is
    naive. Under AppDaemon that was the add-on container's zone; here it must
    be HA's configured zone as well, or slot alignment and the daily schedule
    would run in UTC. Returns True when the zone was applied.
    """
    if not name or not hasattr(time, "tzset"):
        return False
    os.environ["TZ"] = name
    time.tzset()
    return True


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _Timer:
    due: datetime.datetime
    seq: int
    handle: str = field(compare=False)
    callback: Callable = field(compare=False)
    kwargs: dict = field(compare=False, default_factory=dict)
    interval: Optional[datetime.timedelta] = field(compare=False, default=None)
    daily_time: Optional[datetime.time] = field(compare=False, default=None)
    cancelled: bool = field(compare=False, default=False)


class Scheduler:
    """Wall-clock timers, fired on a background thread (or by ``tick``).

    All instants are aware UTC. A repeating timer's next fire is computed from
    its previous DUE time, never from when the callback finished, so a
    slot-aligned ``run_every`` stays on the slot grid however long a callback
    takes. A daily timer recomputes the next local occurrence of its wall time
    each day, which keeps it at 14:15 local across DST changes.
    """

    def __init__(self, submit: Callable[[_Timer], None],
                 now_func: Callable[[], datetime.datetime] = _now_utc,
                 zone=None):
        self._submit = submit
        self._now = now_func
        self._zone = zone
        self._heap: List[_Timer] = []
        self._by_handle: Dict[str, _Timer] = {}
        self._seq = itertools.count()
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stopping = False

    # -- registration --------------------------------------------------
    def add(self, callback, due: datetime.datetime, *,
            interval: Optional[datetime.timedelta] = None,
            daily_time: Optional[datetime.time] = None,
            kwargs: Optional[dict] = None) -> str:
        handle = uuid.uuid4().hex
        timer = _Timer(
            due=due, seq=next(self._seq), handle=handle, callback=callback,
            kwargs=dict(kwargs or {}), interval=interval, daily_time=daily_time,
        )
        with self._cond:
            heapq.heappush(self._heap, timer)
            self._by_handle[handle] = timer
            self._cond.notify()
        return handle

    def cancel(self, handle) -> bool:
        with self._cond:
            timer = self._by_handle.pop(handle, None)
            if timer is None:
                return False
            timer.cancelled = True
            self._cond.notify()
            return True

    def running(self, handle) -> bool:
        with self._cond:
            return handle in self._by_handle

    def cancel_all(self) -> None:
        with self._cond:
            for timer in self._by_handle.values():
                timer.cancelled = True
            self._by_handle.clear()
            self._heap.clear()
            self._cond.notify()

    def next_daily(self, wall: datetime.time,
                   after: datetime.datetime) -> datetime.datetime:
        """First local ``wall`` strictly after ``after`` (aware UTC)."""
        zone = self._zone or datetime.timezone.utc
        local_after = after.astimezone(zone)
        day = local_after.date()
        for _ in range(3):
            candidate = datetime.datetime.combine(day, wall).replace(tzinfo=zone)
            candidate_utc = candidate.astimezone(datetime.timezone.utc)
            if candidate_utc > after:
                return candidate_utc
            day = day + datetime.timedelta(days=1)
        return (after + datetime.timedelta(days=1))  # pragma: no cover

    # -- firing --------------------------------------------------------
    def tick(self, now: Optional[datetime.datetime] = None) -> int:
        """Submit every due timer; reschedule repeating ones. Returns count."""
        now = now or self._now()
        fired = 0
        while True:
            with self._cond:
                if not self._heap or self._heap[0].due > now:
                    return fired
                timer = heapq.heappop(self._heap)
                if timer.cancelled:
                    continue
                if timer.interval is not None:
                    nxt = timer.due + timer.interval
                    while nxt <= now:
                        # Missed beats (host suspended) collapse into one fire,
                        # and the grid is kept.
                        nxt += timer.interval
                    timer.due = nxt
                    timer.seq = next(self._seq)
                    heapq.heappush(self._heap, timer)
                elif timer.daily_time is not None:
                    timer.due = self.next_daily(timer.daily_time, now)
                    timer.seq = next(self._seq)
                    heapq.heappush(self._heap, timer)
                else:
                    self._by_handle.pop(timer.handle, None)
            fired += 1
            self._submit(timer)

    def start(self) -> None:
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="scheduler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            with self._cond:
                if self._stopping:
                    return
                if self._heap:
                    wait = (self._heap[0].due - self._now()).total_seconds()
                else:
                    wait = 3600.0
                if wait > 0:
                    # Bounded wait: re-reads the clock at least once a second
                    # so a wall-clock step (NTP) cannot strand a timer.
                    self._cond.wait(timeout=min(wait, 1.0))
                    continue
            self.tick()


# ---------------------------------------------------------------------------
# Websocket client
# ---------------------------------------------------------------------------


def _default_ws_factory(url: str, timeout: float):
    import websocket  # websocket-client

    return websocket.create_connection(url, timeout=timeout)


class _Pending:
    __slots__ = ("event", "response")

    def __init__(self):
        self.event = threading.Event()
        self.response: Optional[dict] = None


class HAConnection:
    """One logical connection to HA's websocket API, reconnecting forever.

    ``on_state_changed(entity_id, old_state, new_state)`` and
    ``on_event(event_type, data)`` are called on the reader thread; they must
    not block (the host hands them to the worker pool).
    """

    def __init__(self, url: str, token: str, *,
                 log: Callable[..., None],
                 on_state_changed: Callable[[str, Optional[dict], Optional[dict]], None],
                 on_event: Callable[[str, dict], None],
                 on_connected: Optional[Callable[[], None]] = None,
                 ws_factory: Callable[[str, float], Any] = _default_ws_factory,
                 reconnect_delays: Tuple[float, ...] = RECONNECT_DELAYS_S,
                 auth_failure_delay: float = AUTH_FAILURE_DELAY_S,
                 heartbeat_idle: float = HEARTBEAT_IDLE_S):
        self.url = url
        self._token = token
        self._log = log
        self._on_state_changed = on_state_changed
        self._on_event = on_event
        self._on_connected = on_connected
        self._ws_factory = ws_factory
        self._reconnect_delays = tuple(reconnect_delays) or (1.0,)
        self._auth_failure_delay = auth_failure_delay
        self._heartbeat_idle = heartbeat_idle

        self._ws = None
        self._send_lock = threading.Lock()
        self._id_lock = threading.Lock()
        self._next_id = 0
        self._pending: Dict[int, _Pending] = {}
        # service_registered keeps the catalogue current: an integration that
        # loads after we connected (growatt_modbus during an HA restart) must
        # still get return_response set automatically.
        self._event_types: Dict[str, Optional[int]] = {
            "state_changed": None, "service_registered": None,
        }
        self._connected = threading.Event()
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.states: Dict[str, dict] = {}
        self.states_lock = threading.RLock()
        self.services: Dict[str, Dict[str, dict]] = {}
        self.config: Dict[str, Any] = {}
        self.connect_count = 0

    # -- lifecycle -----------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> None:
        self._stopping.clear()
        self._thread = threading.Thread(
            target=self._run, name="ha-websocket", daemon=True
        )
        self._thread.start()

    def wait_connected(self, timeout: Optional[float] = None) -> bool:
        return self._connected.wait(timeout)

    def stop(self) -> None:
        self._stopping.set()
        self._connected.clear()
        self._close_ws()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def drop(self) -> None:
        """Close the socket and let the reader reconnect (tests, diagnostics)."""
        self._close_ws()

    def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    # -- subscriptions -------------------------------------------------
    def ensure_event_subscription(self, event_type: str) -> None:
        """Subscribe to ``event_type`` now (if connected) and on every reconnect."""
        if event_type in self._event_types:
            return
        self._event_types[event_type] = None
        if self.connected:
            try:
                self._send_raw({"type": "subscribe_events", "event_type": event_type},
                               track=False)
            except Exception as e:
                self._log(f"subscribe_events {event_type} failed: {e}",
                          level="WARNING")

    # -- requests ------------------------------------------------------
    def _allocate_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _send_raw(self, message: dict, *, track: bool) -> Tuple[int, Optional[_Pending]]:
        msg_id = self._allocate_id()
        message = dict(message, id=msg_id)
        pending = None
        if track:
            pending = _Pending()
            self._pending[msg_id] = pending
        ws = self._ws
        if ws is None:
            self._pending.pop(msg_id, None)
            raise ConnectionError("websocket is not connected")
        try:
            with self._send_lock:
                ws.send(json.dumps(message, default=str))
        except Exception:
            self._pending.pop(msg_id, None)
            raise
        return msg_id, pending

    def request(self, message: dict, timeout: Optional[float]) -> Optional[dict]:
        """Send and wait for the matching result, AppDaemon-style envelope.

        None when not connected (nothing was sent). Raises when the write
        itself fails (nothing reached HA). On timeout the request WAS written,
        so the answer is ``ad_status: TIMEOUT`` - unconfirmed, not failed.
        """
        if not self.connected:
            return None
        wait = DEFAULT_WS_TIMEOUT_S if timeout is None else float(timeout)
        started = time.perf_counter()
        msg_id, pending = self._send_raw(message, track=True)
        if not pending.event.wait(wait):
            self._pending.pop(msg_id, None)
            self._log(
                f"Timed out [{wait:g}s] waiting for request: "
                f"{_describe_request(message)}",
                level="WARNING",
            )
            return {
                "success": False,
                "ad_status": AD_STATUS_TIMEOUT,
                "ad_duration": time.perf_counter() - started,
            }
        response = dict(pending.response or {})
        response["ad_status"] = AD_STATUS_OK
        response["ad_duration"] = time.perf_counter() - started
        return response

    # -- reader --------------------------------------------------------
    def _run(self) -> None:
        attempt = 0
        while not self._stopping.is_set():
            try:
                ws = self._ws_factory(self.url, 10.0)
                self._ws = ws
                self._handshake(ws)
                attempt = 0
                self._read_loop(ws)
            except _AuthRefused as e:
                self._log(f"Home Assistant refused the token: {e}", level="ERROR")
                self._sleep(self._auth_failure_delay)
                continue
            except Exception as e:
                if not self._stopping.is_set():
                    self._log(f"Home Assistant websocket error: {e}",
                              level="WARNING" if attempt == 0 else "DEBUG")
            finally:
                was_connected = self._connected.is_set()
                self._connected.clear()
                self._close_ws()
                if was_connected and not self._stopping.is_set():
                    self._log("Disconnected from Home Assistant; reconnecting",
                              level="WARNING")
            if self._stopping.is_set():
                return
            delay = self._reconnect_delays[min(attempt, len(self._reconnect_delays) - 1)]
            attempt += 1
            self._sleep(delay)

    def _sleep(self, seconds: float) -> None:
        self._stopping.wait(seconds)

    def _recv_json(self, ws) -> dict:
        raw = ws.recv()
        if raw is None or raw == "":
            raise ConnectionError("websocket closed")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    def _handshake(self, ws) -> None:
        """Auth, then subscribe + catalogue + full state, BEFORE going live.

        Subscriptions go first so no change is lost between the snapshot and
        the stream; events that arrive while the snapshot is in flight are
        applied afterwards, and only if they are newer than the snapshot.
        """
        hello = self._recv_json(ws)
        if hello.get("type") != "auth_required":
            raise ConnectionError(f"unexpected greeting: {hello.get('type')}")
        with self._send_lock:
            ws.send(json.dumps({"type": "auth", "access_token": self._token}))
        reply = self._recv_json(ws)
        if reply.get("type") != "auth_ok":
            raise _AuthRefused(reply.get("message") or reply.get("type"))

        sub_ids = {}
        for event_type in list(self._event_types):
            sub_ids[event_type] = self._send_raw(
                {"type": "subscribe_events", "event_type": event_type}, track=False
            )[0]
        config_id = self._send_raw({"type": "get_config"}, track=False)[0]
        services_id = self._send_raw({"type": "get_services"}, track=False)[0]
        states_id = self._send_raw({"type": "get_states"}, track=False)[0]

        buffered: List[dict] = []
        waiting = {config_id, services_id, states_id}
        fresh_states: Dict[str, dict] = {}
        while waiting:
            msg = self._recv_json(ws)
            mtype = msg.get("type")
            if mtype == "event":
                buffered.append(msg)
                continue
            if mtype != "result":
                continue
            mid = msg.get("id")
            if mid in waiting:
                waiting.discard(mid)
                if not msg.get("success"):
                    raise ConnectionError(
                        f"startup request {mid} refused: {msg.get('error')}"
                    )
                result = msg.get("result")
                if mid == config_id:
                    self.config = result or {}
                elif mid == services_id:
                    self.services = result or {}
                else:
                    fresh_states = {
                        s["entity_id"]: s for s in (result or []) if "entity_id" in s
                    }

        with self.states_lock:
            self.states = fresh_states
        for msg in buffered:
            self._handle_event(msg, only_if_newer=True)

        self.connect_count += 1
        self._connected.set()
        self._log(
            f"Connected to Home Assistant {self.config.get('version', '?')} "
            f"({len(fresh_states)} entities, connection #{self.connect_count})"
        )
        if self._on_connected is not None:
            try:
                self._on_connected()
            except Exception as e:  # pragma: no cover - defensive
                self._log(f"on_connected hook failed: {e}", level="ERROR")

    def _read_loop(self, ws) -> None:
        import websocket  # websocket-client: for its timeout exception type

        idle_pings = 0
        ws.settimeout(self._heartbeat_idle)
        while not self._stopping.is_set():
            try:
                msg = self._recv_json(ws)
            except websocket.WebSocketTimeoutException:
                idle_pings += 1
                if idle_pings > 1:
                    raise ConnectionError("no answer to heartbeat ping")
                self._send_raw({"type": "ping"}, track=False)
                continue
            idle_pings = 0
            mtype = msg.get("type")
            if mtype == "result":
                pending = self._pending.pop(msg.get("id"), None)
                if pending is not None:
                    pending.response = msg
                    pending.event.set()
            elif mtype == "event":
                self._handle_event(msg)
            # "pong" and anything else: liveness only.

    def _handle_event(self, msg: dict, only_if_newer: bool = False) -> None:
        event = msg.get("event") or {}
        event_type = event.get("event_type")
        data = event.get("data") or {}
        if event_type == "state_changed":
            entity_id = data.get("entity_id")
            if not entity_id:
                return
            new_state = data.get("new_state")
            with self.states_lock:
                cached = self.states.get(entity_id)
                if only_if_newer and cached is not None and new_state is not None:
                    if str(new_state.get("last_updated", "")) <= str(
                        cached.get("last_updated", "")
                    ):
                        return
                old_state = cached if cached is not None else data.get("old_state")
                if new_state is None:
                    self.states.pop(entity_id, None)
                else:
                    self.states[entity_id] = new_state
            self._on_state_changed(entity_id, old_state, new_state)
        elif event_type == "service_registered" and not only_if_newer:
            # Not on the reader thread: the refresh waits for a result that
            # only the reader thread can deliver.
            threading.Thread(target=self.refresh_services, name="services",
                             daemon=True).start()
        elif event_type:
            self._on_event(event_type, data)

    def refresh_services(self) -> bool:
        """Reload the service catalogue (response metadata for return_response)."""
        result = self.request({"type": "get_services"}, DEFAULT_WS_TIMEOUT_S)
        if isinstance(result, dict) and result.get("success") and isinstance(
            result.get("result"), dict
        ):
            self.services = result["result"]
            return True
        return False


class _AuthRefused(Exception):
    pass


def _describe_request(message: dict) -> str:
    """A request for a log line - service and keys, never values like tokens."""
    if message.get("type") == "call_service":
        data = message.get("service_data") or {}
        return (f"call_service {message.get('domain')}/{message.get('service')} "
                f"keys={sorted(data)}")
    return str(message.get("type"))


# ---------------------------------------------------------------------------
# Host
# ---------------------------------------------------------------------------


class HAHost:
    """Connection + state cache + scheduler + worker pool + logging."""

    def __init__(self, *, ws_url: str, rest_url: str, token: str,
                 time_zone: Optional[str] = None,
                 threads: int = 4,
                 log_level: str = "INFO",
                 app_name: str = "battery_optimizer",
                 shadow: Optional[ShadowPolicy] = None,
                 now_func: Callable[[], datetime.datetime] = _now_utc,
                 ws_factory: Callable[[str, float], Any] = _default_ws_factory,
                 http_post: Optional[Callable[..., Any]] = None,
                 reconnect_delays: Tuple[float, ...] = RECONNECT_DELAYS_S,
                 heartbeat_idle: float = HEARTBEAT_IDLE_S,
                 executor=None,
                 logger: Optional[logging.Logger] = None):
        self.app_name = app_name
        self.rest_url = rest_url.rstrip("/")
        self._token = token
        self.shadow = shadow
        self._now = now_func
        self._http_post = http_post
        self._logger = logger or _default_logger(app_name, log_level)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max(1, int(threads)), thread_name_prefix="worker"
        )
        self.zone = resolve_zone(time_zone)
        self._time_zone_name = time_zone

        self._listen_lock = threading.Lock()
        self._state_listeners: Dict[str, Tuple[str, Optional[str], Callable, dict]] = {}
        self._event_listeners: Dict[str, Tuple[str, Callable, dict]] = {}

        self.connection = HAConnection(
            ws_url, token,
            log=self.log,
            on_state_changed=self._on_state_changed,
            on_event=self._on_event,
            ws_factory=ws_factory,
            reconnect_delays=reconnect_delays,
            heartbeat_idle=heartbeat_idle,
        )
        self.scheduler = Scheduler(self._dispatch_timer, now_func=now_func,
                                   zone=self.zone)

    # -- lifecycle -----------------------------------------------------
    def start(self, wait_timeout: Optional[float] = None) -> bool:
        """Connect; adopt HA's time zone if none was given. True once synced."""
        self.connection.start()
        ok = self.connection.wait_connected(wait_timeout)
        if ok and self.zone is None:
            name = self.connection.config.get("time_zone")
            self.zone = resolve_zone(name)
            self._time_zone_name = name
            self.scheduler._zone = self.zone
        self.scheduler.start()
        return ok

    def stop(self) -> None:
        self.scheduler.stop()
        self.connection.stop()
        self._executor.shutdown(wait=False)

    @property
    def time_zone_name(self) -> Optional[str]:
        return self._time_zone_name

    # -- logging -------------------------------------------------------
    def log(self, msg, *args, level: str = "INFO", **kwargs) -> None:
        if args:
            try:
                msg = msg % args
            except Exception:
                msg = " ".join([str(msg)] + [str(a) for a in args])
        self._logger.log(LEVELS.get(str(level).upper(), logging.INFO), msg)

    # -- time ----------------------------------------------------------
    def now_local(self) -> datetime.datetime:
        now = self._now()
        return now.astimezone(self.zone) if self.zone else now.astimezone()

    def datetime(self, aware: bool = False) -> datetime.datetime:
        now = self.now_local()
        return now if aware else now.replace(tzinfo=None)

    def date(self) -> datetime.date:
        return self.now_local().date()

    def get_timezone(self):
        return self.zone if self.zone is not None else self._time_zone_name

    def _to_utc(self, when) -> datetime.datetime:
        if isinstance(when, datetime.datetime):
            if when.tzinfo is None:
                zone = self.zone
                if zone is not None:
                    when = when.replace(tzinfo=zone)
                else:
                    when = when.astimezone()
            return when.astimezone(datetime.timezone.utc)
        raise TypeError(f"not a datetime: {when!r}")

    # -- state ---------------------------------------------------------
    def get_state(self, entity_id: Optional[str] = None, attribute: Optional[str] = None,
                  default: Any = None, copy_: bool = True, **kwargs):
        conn = self.connection
        with conn.states_lock:
            if entity_id is None:
                data = conn.states
                return copy.deepcopy(data) if copy_ else dict(data)
            if "." not in entity_id:
                found = {k: v for k, v in conn.states.items()
                         if k.split(".", 1)[0] == entity_id}
                return copy.deepcopy(found) if copy_ else found
            state = conn.states.get(entity_id)
            if state is None:
                return default
            if attribute == "all":
                return copy.deepcopy(state) if copy_ else state
            if attribute is None:
                return state.get("state", default)
            if attribute in state:
                value = state[attribute]
            else:
                value = (state.get("attributes") or {}).get(attribute, default)
            return copy.deepcopy(value) if copy_ else value

    def set_state(self, entity_id: str, state: Any = None,
                  attributes: Optional[dict] = None, replace: bool = False,
                  **kwargs) -> Optional[dict]:
        if self.shadow is not None:
            entity_id = self.shadow.entity(entity_id)
        conn = self.connection
        with conn.states_lock:
            cached = copy.deepcopy(conn.states.get(entity_id) or {})
        new_attrs = dict(attributes or {})
        new_attrs.update(kwargs)
        merged = dict(cached.get("attributes") or {})
        if new_attrs:
            merged = new_attrs if replace else {**merged, **new_attrs}
        new_state = cached.get("state") if state is None else state
        body = {"state": new_state, "attributes": merged}
        url = f"{self.rest_url}/api/states/{entity_id}"
        try:
            result = self._post(url, body)
        except Exception as e:
            self.log(f"Error setting state of {entity_id}: {e}", level="ERROR")
            return None
        if isinstance(result, dict):
            with conn.states_lock:
                conn.states[entity_id] = result
        return result

    def _post(self, url: str, body: dict):
        payload = json.dumps(body, default=str)
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        if self._http_post is not None:
            return self._http_post(url, payload, headers)
        import requests

        response = requests.post(url, data=payload, headers=headers,
                                 timeout=DEFAULT_REST_TIMEOUT_S)
        if response.status_code not in (200, 201):
            raise HostError(f"HTTP {response.status_code}: {response.text[:200]}")
        return response.json()

    # -- services ------------------------------------------------------
    def call_service(self, service: str, *, hass_timeout=None,
                     return_response: Optional[bool] = None,
                     target=None, entity_id=None, namespace=None, timeout=None,
                     callback=None, suppress_log_messages: bool = False,
                     **data):
        if "/" not in service:
            raise ValueError(f"service must be 'domain/service', got {service!r}")
        if self.shadow is not None and not self.shadow.allows(service):
            count = self.shadow.suppressed.get(service, 0) + 1
            self.shadow.suppressed[service] = count
            self.log(
                f"shadow mode: suppressed {service} (#{count})",
                level="INFO" if count == 1 else "DEBUG",
            )
            return None
        domain, svc = service.split("/", 1)
        request: Dict[str, Any] = {"type": "call_service", "domain": domain,
                                   "service": svc}
        if return_response is not None:
            request["return_response"] = bool(return_response)
        service_data = dict(data.pop("service_data", None) or {})
        service_data.update(data)
        service_data = {k: v for k, v in service_data.items() if v is not None}
        if service_data:
            request["service_data"] = service_data
        response_spec = (
            (self.connection.services.get(domain) or {}).get(svc) or {}
        ).get("response")
        if isinstance(response_spec, dict):
            if response_spec.get("optional") is False:
                request["return_response"] = True
            elif response_spec.get("optional") is True and "return_response" not in request:
                request["return_response"] = True
        if target is None and entity_id is not None:
            request["target"] = {"entity_id": entity_id}
        elif target is not None:
            request["target"] = target
        result = self.connection.request(request, hass_timeout)
        if (isinstance(result, dict) and result.get("success") is False
                and result.get("ad_status") == AD_STATUS_OK
                and not suppress_log_messages):
            error = result.get("error") or {}
            self.log(
                f"Error with websocket result: {error.get('code')}: "
                f"{error.get('message')}: {_describe_request(request)}",
                level="WARNING",
            )
        return result

    # -- listeners -----------------------------------------------------
    def listen_state(self, callback: Callable, entity_id: str,
                     attribute: Optional[str] = None, **kwargs) -> str:
        handle = uuid.uuid4().hex
        with self._listen_lock:
            self._state_listeners[handle] = (entity_id, attribute, callback, kwargs)
        return handle

    def cancel_listen_state(self, handle) -> bool:
        with self._listen_lock:
            return self._state_listeners.pop(handle, None) is not None

    def listen_event(self, callback: Callable, event: str, **kwargs) -> str:
        handle = uuid.uuid4().hex
        with self._listen_lock:
            self._event_listeners[handle] = (event, callback, kwargs)
        self.connection.ensure_event_subscription(event)
        return handle

    def cancel_listen_event(self, handle) -> bool:
        with self._listen_lock:
            return self._event_listeners.pop(handle, None) is not None

    @staticmethod
    def _value_of(state: Optional[dict], attribute: Optional[str]):
        if state is None:
            return None
        key = attribute or "state"
        if key in state:
            return state[key]
        return (state.get("attributes") or {}).get(key)

    def _on_state_changed(self, entity_id, old_state, new_state) -> None:
        with self._listen_lock:
            listeners = [l for l in self._state_listeners.values() if l[0] == entity_id]
        for _, attribute, callback, kwargs in listeners:
            if attribute == "all":
                old, new = old_state, new_state
            else:
                old = self._value_of(old_state, attribute)
                new = self._value_of(new_state, attribute)
                if old == new:
                    continue
            self._submit(callback, (entity_id, attribute or "state", old, new),
                         dict(kwargs))

    def _on_event(self, event_type: str, data: dict) -> None:
        with self._listen_lock:
            listeners = [l for l in self._event_listeners.values() if l[0] == event_type]
        for _, callback, kwargs in listeners:
            self._submit(callback, (event_type, data), dict(kwargs))

    # -- timers --------------------------------------------------------
    def run_in(self, callback: Callable, delay, *args, **kwargs) -> str:
        seconds = delay.total_seconds() if isinstance(delay, datetime.timedelta) else float(delay)
        due = self._now() + datetime.timedelta(seconds=seconds)
        return self.scheduler.add(callback, due, kwargs=kwargs)

    def run_every(self, callback: Callable, start=None, interval=0,
                  *args, **kwargs) -> str:
        step = (interval if isinstance(interval, datetime.timedelta)
                else datetime.timedelta(seconds=float(interval)))
        if step.total_seconds() <= 0:
            raise ValueError("run_every needs a positive interval")
        now = self._now()
        if start is None or start == "now":
            due = now + step
        elif start == "immediate":
            due = now
        else:
            due = self._to_utc(start)
            while due < now:
                due += step
        return self.scheduler.add(callback, due, interval=step, kwargs=kwargs)

    def run_daily(self, callback: Callable, start, *args, **kwargs) -> str:
        if isinstance(start, datetime.datetime):
            start = start.time()
        if not isinstance(start, datetime.time):
            raise TypeError(f"run_daily needs a datetime.time, got {start!r}")
        due = self.scheduler.next_daily(start, self._now())
        return self.scheduler.add(callback, due, daily_time=start, kwargs=kwargs)

    def run_at(self, callback: Callable, start, *args, **kwargs) -> str:
        return self.scheduler.add(callback, self._to_utc(start), kwargs=kwargs)

    def cancel_timer(self, handle, silent: bool = False) -> bool:
        return self.scheduler.cancel(handle)

    def timer_running(self, handle) -> bool:
        return self.scheduler.running(handle)

    # -- dispatch ------------------------------------------------------
    def _dispatch_timer(self, timer: _Timer) -> None:
        self._submit(timer.callback, (), dict(timer.kwargs))

    def _submit(self, callback: Callable, pos_args: tuple, kwargs: dict) -> None:
        try:
            self._executor.submit(self._invoke, callback, pos_args, kwargs)
        except RuntimeError:
            # Executor shut down: the host is stopping.
            pass

    def _invoke(self, callback: Callable, pos_args: tuple, kwargs: dict) -> None:
        name = getattr(callback, "__qualname__", repr(callback))
        try:
            if has_expanded_kwargs(callback):
                callback(*pos_args, **kwargs)
            else:
                callback(*pos_args, kwargs)
        except Exception:
            self.log(
                f"Unexpected error in callback {name}:\n{traceback.format_exc()}",
                level="ERROR",
            )


def _default_logger(name: str, level: str) -> logging.Logger:
    logger = logging.getLogger(f"ha_host.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(threadName)s: %(message)s"
        ))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(LEVELS.get(str(level).upper(), logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# App base class
# ---------------------------------------------------------------------------


class Hass:
    """AppDaemon-compatible base class; every call delegates to an ``HAHost``.

    Constructible without a host so test doubles that subclass the app and
    never run ``initialize`` keep working; a call that needs HA then raises.
    """

    def __init__(self, host: Optional[HAHost] = None,
                 args: Optional[dict] = None, name: str = "battery_optimizer"):
        self._ha_host = host
        self.args = dict(args or {})
        self.name = name

    def _require_host(self) -> HAHost:
        host = getattr(self, "_ha_host", None)
        if host is None:
            raise HostError("no Home Assistant host attached to this app")
        return host

    # lifecycle hooks the app overrides
    def initialize(self):  # pragma: no cover - overridden
        pass

    def terminate(self):  # pragma: no cover - overridden
        pass

    # logging / time
    def log(self, msg, *args, level: str = "INFO", **kwargs):
        host = getattr(self, "_ha_host", None)
        if host is None:
            print(f"{level} {msg}")
            return
        host.log(msg, *args, level=level, **kwargs)

    def error(self, msg, *args, level: str = "ERROR", **kwargs):
        self.log(msg, *args, level=level, **kwargs)

    def datetime(self, aware: bool = False):
        return self._require_host().datetime(aware=aware)

    def get_now(self, aware: bool = True):
        return self._require_host().datetime(aware=aware)

    def date(self):
        return self._require_host().date()

    def get_timezone(self):
        return self._require_host().get_timezone()

    # state / services
    def get_state(self, entity_id=None, attribute=None, default=None, copy=True, **kwargs):
        return self._require_host().get_state(entity_id, attribute, default, copy)

    def set_state(self, entity_id, **kwargs):
        return self._require_host().set_state(entity_id, **kwargs)

    def call_service(self, service, **kwargs):
        return self._require_host().call_service(service, **kwargs)

    def entity_exists(self, entity_id) -> bool:
        return self._require_host().get_state(entity_id, attribute="all") is not None

    # listeners
    def listen_state(self, callback, entity_id, **kwargs):
        return self._require_host().listen_state(callback, entity_id, **kwargs)

    def cancel_listen_state(self, handle):
        return self._require_host().cancel_listen_state(handle)

    def listen_event(self, callback, event, **kwargs):
        return self._require_host().listen_event(callback, event, **kwargs)

    def cancel_listen_event(self, handle):
        return self._require_host().cancel_listen_event(handle)

    # timers
    def run_in(self, callback, delay, *args, **kwargs):
        return self._require_host().run_in(callback, delay, *args, **kwargs)

    def run_every(self, callback, start=None, interval=0, *args, **kwargs):
        return self._require_host().run_every(callback, start, interval, *args, **kwargs)

    def run_daily(self, callback, start, *args, **kwargs):
        return self._require_host().run_daily(callback, start, *args, **kwargs)

    def run_at(self, callback, start, *args, **kwargs):
        return self._require_host().run_at(callback, start, *args, **kwargs)

    def cancel_timer(self, handle, silent: bool = False):
        return self._require_host().cancel_timer(handle, silent)

    def timer_running(self, handle):
        return self._require_host().timer_running(handle)
