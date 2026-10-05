"""
Home Assistant add-on entry point.

``/data/options.json`` (the add-on options, validated by the Supervisor against
``addon/battery_optimizer/config.yaml``) -> the args dict the optimizer has
always read from ``apps.yaml`` -> ``BatteryOptimizerConfig.from_args``.

The connection is not configured: inside an add-on with
``homeassistant_api: true`` the Supervisor injects ``SUPERVISOR_TOKEN`` and
proxies Core at ``http://supervisor/core``. That URL and token replace
``ha_url`` / ``ha_token`` for BOTH paths - the host's websocket and the price
service's direct REST call to ``nordpool/get_price_indices_for_date``.

Run inside the image as ``python3 -m battery_optimizer_lib.addon_main`` from
the directory holding ``battery_optimizer.py``.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
from typing import Any, Dict, Optional

from .ha_host import HAHost, ShadowPolicy, apply_process_timezone

OPTIONS_PATH = "/data/options.json"
SUPERVISOR_CORE_URL = "http://supervisor/core"
SUPERVISOR_WS_URL = "ws://supervisor/core/websocket"

# Options that configure the add-on itself, never handed to from_args.
HOST_OPTIONS = frozenset({"shadow_mode", "entity_suffix", "log_level", "worker_threads"})
# Connection keys the add-on owns: present in apps.yaml, never in the options.
CONNECTION_KEYS = frozenset({"ha_url", "ha_token"})
DEFAULT_ENTITY_SUFFIX = "_shadow"
DEFAULT_WORKER_THREADS = 4


def load_options(path: str = OPTIONS_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return data


def is_shadow(options: Dict[str, Any]) -> bool:
    return bool(options.get("shadow_mode", False))


def options_to_args(options: Dict[str, Any], *, supervisor_token: str,
                    core_url: str = SUPERVISOR_CORE_URL) -> Dict[str, Any]:
    """The apps.yaml-shaped args dict for ``BatteryOptimizerConfig.from_args``.

    Every optimizer option passes through unchanged - the Supervisor has
    already type-checked it, and from_args does its own conversion exactly as
    it did for apps.yaml. In shadow mode ``device_id`` is forced empty, which
    is DirectControl's dry run: the inverter is never commanded, whatever the
    options say. (The host additionally refuses every non-read service in
    shadow mode; this is the second, independent guard.)
    """
    args = {k: v for k, v in options.items()
            if k not in HOST_OPTIONS and k not in CONNECTION_KEYS}
    args["ha_url"] = core_url
    args["ha_token"] = supervisor_token
    if is_shadow(options):
        args["device_id"] = ""
    return args


def shadow_policy(options: Dict[str, Any]) -> Optional[ShadowPolicy]:
    if not is_shadow(options):
        return None
    suffix = str(options.get("entity_suffix") or DEFAULT_ENTITY_SUFFIX)
    return ShadowPolicy(entity_suffix=suffix)


def build_host(options: Dict[str, Any], token: str, **overrides) -> HAHost:
    kwargs = dict(
        ws_url=SUPERVISOR_WS_URL,
        rest_url=SUPERVISOR_CORE_URL,
        token=token,
        threads=int(options.get("worker_threads") or DEFAULT_WORKER_THREADS),
        log_level=str(options.get("log_level") or "info"),
        shadow=shadow_policy(options),
    )
    kwargs.update(overrides)
    return HAHost(**kwargs)


def main(argv=None) -> int:
    token = os.environ.get("SUPERVISOR_TOKEN", "")
    if not token:
        print("SUPERVISOR_TOKEN is not set - is homeassistant_api enabled?",
              file=sys.stderr)
        return 2
    options = load_options(os.environ.get("BATTERY_OPTIMIZER_OPTIONS", OPTIONS_PATH))
    host = build_host(options, token)
    host.log(
        f"Add-on starting: shadow_mode={is_shadow(options)} "
        f"suffix={shadow_policy(options).entity_suffix if is_shadow(options) else '-'}"
    )
    host.start()  # blocks until connected and the state cache is synced
    if apply_process_timezone(host.time_zone_name):
        host.log(f"Process time zone set to {host.time_zone_name}")

    import battery_optimizer  # the orchestrator, next to this package

    app = battery_optimizer.BatteryOptimizer(
        host=host,
        args=options_to_args(options, supervisor_token=token),
    )
    stop = threading.Event()

    def _on_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    try:
        app.initialize()
    except Exception:
        import traceback

        host.log(f"initialize() failed:\n{traceback.format_exc()}", level="ERROR")
        host.stop()
        return 1

    stop.wait()
    host.log("Stopping")
    try:
        app.terminate()
    finally:
        host.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover - container entry point
    sys.exit(main())
