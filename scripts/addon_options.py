#!/usr/bin/env python
"""
Convert an AppDaemon apps.yaml app block into add-on options.

The add-on reads Supervisor-validated options instead of apps.yaml. This is the
one conversion between the two, used by the smoke test, the deploy script and
the tests:

* AppDaemon-only keys (``module``, ``class``, ``pin_app``, ``pin_thread``) and
  the connection keys (``ha_url``, ``ha_token``) are dropped - inside the
  add-on the Supervisor provides the connection;
* every other key must exist in ``addon/battery_optimizer/config.yaml``'s
  schema, and its value is coerced the way the Supervisor will coerce it
  (``int`` would TRUNCATE 0.25, so a fractional value under an int type is an
  error here, not a silent change);
* ``shadow_mode`` / ``entity_suffix`` are added on request, and a shadow
  conversion blanks ``device_id`` (dry run).

Usage:
    uv run python scripts/addon_options.py <apps.yaml> [--live] [--out FILE]

Without ``--out`` the options are printed with secret-looking keys redacted;
``--out`` writes them unredacted (the file is for the Supervisor API).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
ADDON_CONFIG = REPO_ROOT / "addon" / "battery_optimizer" / "config.yaml"

APPDAEMON_ONLY_KEYS = frozenset({"module", "class", "pin_app", "pin_thread"})
CONNECTION_KEYS = frozenset({"ha_url", "ha_token"})
SECRET_HINTS = ("token", "key", "password", "secret", "passwd", "credential")


class OptionError(ValueError):
    pass


def load_manifest(path: Path = ADDON_CONFIG) -> Dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def supervisor_coerce(key: str, value: Any, typ: str) -> Any:
    """What the Supervisor stores for ``value`` under schema type ``typ``.

    Mirrors supervisor/addons/options.py: str/password -> str(value),
    int -> Coerce(int) with an optional range, float -> Coerce(float),
    bool -> strict boolean, list(a|b) -> one of the names.
    """
    typ = typ.rstrip("?")
    if typ.startswith(("str", "password")):
        return str(value)
    if typ.startswith("int"):
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            raise OptionError(f"{key}: {value!r} is not an integer")
        if float(value) != coerced:
            raise OptionError(f"{key}: {value!r} would be truncated to {coerced} by int")
        m = re.match(r"int\((-?\d*),(-?\d*)\)", typ)
        if m:
            lo, hi = m.groups()
            if (lo and coerced < int(lo)) or (hi and coerced > int(hi)):
                raise OptionError(f"{key}: {coerced} outside {typ}")
        return coerced
    if typ.startswith("float"):
        try:
            return float(value)
        except (TypeError, ValueError):
            raise OptionError(f"{key}: {value!r} is not a number")
    if typ == "bool":
        if not isinstance(value, bool):
            raise OptionError(f"{key}: {value!r} is not a boolean")
        return value
    if typ.startswith("list("):
        choices = typ[5:-1].split("|")
        if str(value) not in choices:
            raise OptionError(f"{key}: {value!r} not one of {choices}")
        return str(value)
    raise OptionError(f"{key}: unsupported schema type {typ!r}")


def find_app_block(raw: Dict[str, Any], app_name: Optional[str] = None) -> Dict[str, Any]:
    """The battery_optimizer entry of an apps.yaml, or the mapping itself."""
    if app_name:
        block = raw.get(app_name)
        if not isinstance(block, dict):
            raise OptionError(f"app '{app_name}' not found")
        return block
    for value in raw.values():
        if isinstance(value, dict) and value.get("module") == "battery_optimizer":
            return value
    return raw  # already a flat options mapping


def apps_yaml_to_options(app_args: Dict[str, Any], schema: Dict[str, str], *,
                         shadow: bool, entity_suffix: str = "_shadow") -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    unknown = []
    for key, value in app_args.items():
        if key in APPDAEMON_ONLY_KEYS or key in CONNECTION_KEYS:
            continue
        if key not in schema:
            unknown.append(key)
            continue
        if value is None:
            continue  # a bare YAML key means "default", like an absent option
        options[key] = supervisor_coerce(key, value, schema[key])
    if unknown:
        raise OptionError(f"keys not in the add-on schema: {sorted(unknown)}")
    options["shadow_mode"] = bool(shadow)
    if shadow:
        options["entity_suffix"] = entity_suffix
        # Dry run, stated in the options themselves. addon_main forces the
        # same whatever the options say; this makes the Supervisor UI agree.
        options["device_id"] = ""
    return options


def redacted(options: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: ("<set>" if v else "<empty>")
        if any(h in k.lower() for h in SECRET_HINTS) else v
        for k, v in options.items()
    }


def main(argv=None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("apps_yaml")
    parser.add_argument("--app-name", default=None)
    parser.add_argument("--live", action="store_true",
                        help="shadow_mode: false (default: shadow mode)")
    parser.add_argument("--suffix", default="_shadow")
    parser.add_argument("--out", default=None, help="write JSON here (unredacted)")
    args = parser.parse_args(argv)

    raw = yaml.safe_load(Path(args.apps_yaml).read_text(encoding="utf-8"))
    block = find_app_block(raw, args.app_name)
    schema = load_manifest()["schema"]
    try:
        options = apps_yaml_to_options(block, schema, shadow=not args.live,
                                       entity_suffix=args.suffix)
    except OptionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    if args.out:
        Path(args.out).write_text(json.dumps(options, indent=2, sort_keys=True),
                                  encoding="utf-8")
        print(f"wrote {len(options)} options to {args.out}")
    else:
        print(json.dumps(redacted(options), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
