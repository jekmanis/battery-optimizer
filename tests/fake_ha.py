"""
A fake Home Assistant: websocket API + ``POST /api/states`` on one local port.

Standard library only (RFC 6455 framing by hand), so the host under test talks
to it through the real ``websocket-client`` and ``requests`` code paths - the
handshake, the auth exchange, the id matching and the REST call are the
production ones, not a stub of them.

What it speaks is the subset of HA's protocol the host uses: ``auth``,
``subscribe_events``, ``get_states``, ``get_services``, ``get_config``,
``call_service`` (with ``return_response``), ``ping``. Every request it
receives is recorded for assertions.
"""

from __future__ import annotations

import base64
import copy
import datetime
import hashlib
import json
import socket
import socketserver
import struct
import threading
from typing import Callable, Dict, List, Optional

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

HANG = object()  # a service handler returning this never answers


class ServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class _Session:
    def __init__(self, server: "FakeHA", sock: socket.socket):
        self.server = server
        self.sock = sock
        self.send_lock = threading.Lock()
        self.subscriptions: Dict[int, str] = {}
        self.authenticated = False
        self.closed = False

    # -- framing -------------------------------------------------------
    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("client closed")
            buf += chunk
        return buf

    def recv_message(self) -> Optional[str]:
        while True:
            b1, b2 = self._recv_exact(2)
            opcode = b1 & 0x0F
            length = b2 & 0x7F
            if length == 126:
                (length,) = struct.unpack("!H", self._recv_exact(2))
            elif length == 127:
                (length,) = struct.unpack("!Q", self._recv_exact(8))
            mask = self._recv_exact(4) if b2 & 0x80 else b"\x00\x00\x00\x00"
            payload = bytearray(self._recv_exact(length))
            for i in range(len(payload)):
                payload[i] ^= mask[i % 4]
            if opcode == 0x8:  # close
                self._send_frame(0x8, b"")
                return None
            if opcode == 0x9:  # ping
                self._send_frame(0xA, bytes(payload))
                continue
            if opcode in (0x1, 0x2):
                return payload.decode("utf-8")

    def _send_frame(self, opcode: int, data: bytes) -> None:
        header = bytes([0x80 | opcode])
        n = len(data)
        if n < 126:
            header += bytes([n])
        elif n < 65536:
            header += bytes([126]) + struct.pack("!H", n)
        else:
            header += bytes([127]) + struct.pack("!Q", n)
        with self.send_lock:
            self.sock.sendall(header + data)

    def send_json(self, message: dict) -> None:
        if self.closed:
            return
        try:
            self._send_frame(0x1, json.dumps(message).encode("utf-8"))
        except OSError:
            self.closed = True

    def close(self) -> None:
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    # -- protocol ------------------------------------------------------
    def serve(self) -> None:
        self.send_json({"type": "auth_required", "ha_version": "2026.10.0"})
        while not self.closed:
            try:
                raw = self.recv_message()
            except (ConnectionError, OSError):
                break
            if raw is None:
                break
            msg = json.loads(raw)
            self.server.received.append(msg)
            if not self.authenticated:
                if msg.get("type") == "auth" and msg.get("access_token") == self.server.token:
                    self.authenticated = True
                    self.send_json({"type": "auth_ok", "ha_version": "2026.10.0"})
                else:
                    self.send_json({"type": "auth_invalid", "message": "Invalid access token"})
                    break
                continue
            self.server._handle(self, msg)
        self.closed = True


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        server: FakeHA = self.server.fake  # type: ignore[attr-defined]
        sock: socket.socket = self.request
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                return
            data += chunk
        head, _, rest = data.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        method, path, _ = lines[0].split(" ", 2)
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        if headers.get("upgrade", "").lower() == "websocket":
            key = headers["sec-websocket-key"]
            accept = base64.b64encode(
                hashlib.sha1((key + _WS_GUID).encode()).digest()
            ).decode()
            sock.sendall(
                (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                ).encode()
            )
            session = _Session(server, sock)
            server._sessions.append(session)
            session.serve()
            return

        length = int(headers.get("content-length", "0"))
        body = rest
        while len(body) < length:
            body += sock.recv(length - len(body))
        status, payload = server._rest(method, path, headers, body)
        out = json.dumps(payload).encode()
        reason = {200: "OK", 201: "Created", 401: "Unauthorized", 404: "Not Found"}.get(status, "X")
        sock.sendall(
            (
                f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(out)}\r\nConnection: close\r\n\r\n"
            ).encode() + out
        )


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class FakeHA:
    """Start with ``FakeHA().start()``; ``ws_url`` / ``rest_url`` point at it."""

    def __init__(self, token: str = "test-token", time_zone: str = "Europe/Riga"):
        self.token = token
        self.time_zone = time_zone
        self.states: Dict[str, dict] = {}
        self.services: Dict[str, Dict[str, dict]] = {}
        self._handlers: Dict[str, Callable[[dict], object]] = {}
        self.received: List[dict] = []
        self.rest_requests: List[dict] = []
        self._sessions: List[_Session] = []
        self._lock = threading.RLock()
        self._srv: Optional[_Server] = None

    # -- lifecycle -----------------------------------------------------
    def start(self) -> "FakeHA":
        self._srv = _Server(("127.0.0.1", 0), _Handler)
        self._srv.fake = self  # type: ignore[attr-defined]
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.drop_connections()
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()

    @property
    def port(self) -> int:
        return self._srv.server_address[1]

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/api/websocket"

    @property
    def rest_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def drop_connections(self) -> None:
        for session in list(self._sessions):
            session.close()
        self._sessions = [s for s in self._sessions if not s.closed]

    def live_sessions(self) -> List[_Session]:
        return [s for s in self._sessions if not s.closed and s.authenticated]

    # -- scripting -----------------------------------------------------
    def register_service(self, service: str, handler: Callable[[dict], object],
                         response: Optional[str] = None) -> None:
        """``response``: None, "optional" or "only" (HA's SupportsResponse)."""
        domain, name = service.split("/", 1)
        spec: dict = {"name": name, "fields": {}}
        if response == "optional":
            spec["response"] = {"optional": True}
        elif response == "only":
            spec["response"] = {"optional": False}
        self.services.setdefault(domain, {})[name] = spec
        self._handlers[service] = handler

    def set_entity(self, entity_id: str, state, attributes: Optional[dict] = None,
                   notify: bool = True) -> dict:
        with self._lock:
            old = copy.deepcopy(self.states.get(entity_id))
            now = _iso_now()
            new = {
                "entity_id": entity_id,
                "state": str(state),
                "attributes": dict(attributes or {}),
                "last_changed": now if not old or old.get("state") != str(state) else old["last_changed"],
                "last_updated": now,
                "context": {"id": "ctx"},
            }
            self.states[entity_id] = new
        if notify:
            self._broadcast("state_changed", {
                "entity_id": entity_id, "old_state": old, "new_state": new,
            })
        return new

    def fire_event(self, event_type: str, data: Optional[dict] = None) -> None:
        self._broadcast(event_type, dict(data or {}))

    def _broadcast(self, event_type: str, data: dict) -> None:
        for session in self.live_sessions():
            for sub_id, sub_type in list(session.subscriptions.items()):
                if sub_type == event_type:
                    session.send_json({
                        "id": sub_id, "type": "event",
                        "event": {"event_type": event_type, "data": data,
                                  "origin": "LOCAL", "time_fired": _iso_now()},
                    })

    def calls(self, service: Optional[str] = None) -> List[dict]:
        out = [m for m in self.received if m.get("type") == "call_service"]
        if service is not None:
            domain, name = service.split("/", 1)
            out = [m for m in out if m.get("domain") == domain and m.get("service") == name]
        return out

    # -- websocket handling --------------------------------------------
    def _handle(self, session: _Session, msg: dict) -> None:
        mtype = msg.get("type")
        mid = msg.get("id")
        if mtype == "ping":
            session.send_json({"id": mid, "type": "pong"})
        elif mtype == "subscribe_events":
            session.subscriptions[mid] = msg.get("event_type")
            session.send_json({"id": mid, "type": "result", "success": True, "result": None})
        elif mtype == "get_states":
            with self._lock:
                states = copy.deepcopy(list(self.states.values()))
            session.send_json({"id": mid, "type": "result", "success": True, "result": states})
        elif mtype == "get_services":
            session.send_json({"id": mid, "type": "result", "success": True,
                               "result": copy.deepcopy(self.services)})
        elif mtype == "get_config":
            session.send_json({"id": mid, "type": "result", "success": True,
                               "result": {"time_zone": self.time_zone, "version": "2026.10.0"}})
        elif mtype == "call_service":
            service = f"{msg.get('domain')}/{msg.get('service')}"
            handler = self._handlers.get(service)
            if handler is None:
                session.send_json({"id": mid, "type": "result", "success": False,
                                   "error": {"code": "not_found",
                                             "message": f"Service {service} not found."}})
                return
            try:
                response = handler(msg)
            except ServiceError as e:
                session.send_json({"id": mid, "type": "result", "success": False,
                                   "error": {"code": e.code, "message": e.message}})
                return
            if response is HANG:
                return
            result = {"context": {"id": "ctx"},
                      "response": response if msg.get("return_response") else None}
            session.send_json({"id": mid, "type": "result", "success": True, "result": result})
        else:
            session.send_json({"id": mid, "type": "result", "success": False,
                               "error": {"code": "unknown_command", "message": mtype}})

    # -- REST ----------------------------------------------------------
    def _rest(self, method: str, path: str, headers: dict, body: bytes):
        self.rest_requests.append({"method": method, "path": path,
                                   "body": json.loads(body or b"null")})
        if headers.get("authorization") != f"Bearer {self.token}":
            return 401, {"message": "401: Unauthorized"}
        if method == "POST" and path.startswith("/api/states/"):
            entity_id = path[len("/api/states/"):]
            data = json.loads(body or b"{}")
            existed = entity_id in self.states
            # HA stores exactly what was posted: no attribute merge.
            new = self.set_entity(entity_id, data.get("state"), data.get("attributes"))
            return (200 if existed else 201), new
        if method == "POST" and path.startswith("/api/services/"):
            service, _, query = path[len("/api/services/"):].partition("?")
            handler = self._handlers.get(service)
            if handler is None:
                return 400, {"message": f"Service {service} not found"}
            domain, name = service.split("/", 1)
            response = handler({"domain": domain, "service": name,
                                "service_data": json.loads(body or b"{}"),
                                "return_response": "return_response" in query})
            if "return_response" in query:
                return 200, {"changed_states": [], "service_response": response}
            return 200, []
        return 404, {"message": "not found"}
