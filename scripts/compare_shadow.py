#!/usr/bin/env python
"""
Compare the live plan with the shadow add-on's plan, side by side.

Reads ``sensor.battery_optimizer_schedule`` (the live instance) and
``sensor.battery_optimizer_schedule<suffix>`` (shadow add-on) plus both main
sensors from HA's REST API with the admin token (~/.ha_token, never printed),
and checks the S5 acceptance criteria:

* the current slot and the next 4 have the same mode (and WIT mode);
* over the slots both plans cover, each mode's total differs by at most one
  slot.

Exit code 0 when both hold. The table shows every slot of the common range;
``*`` marks a difference.

Usage:
    uv run python scripts/compare_shadow.py [--suffix _shadow] [--rows 200]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

HA_URL = os.environ.get("BO_HA_URL", "http://192.168.77.167:8123")
TOKEN_FILE = Path(os.environ.get("BO_HA_TOKEN_FILE", Path.home() / ".ha_token"))
SLOT = datetime.timedelta(minutes=15)
LOOKAHEAD = 4


def _parse(ts: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def plan_by_slot(entries: List[dict]) -> Dict[datetime.datetime, dict]:
    """``{aware UTC slot start: entry}`` - UTC keys, so DST cannot alias."""
    out = {}
    for e in entries or []:
        try:
            t = _parse(e["time"])
        except (KeyError, ValueError, TypeError):
            continue
        if t.tzinfo is None:
            continue
        out[t.astimezone(datetime.timezone.utc)] = e
    return out


def compare(live: List[dict], shadow: List[dict], now: datetime.datetime,
            lookahead: int = LOOKAHEAD) -> dict:
    a, b = plan_by_slot(live), plan_by_slot(shadow)
    now = now.astimezone(datetime.timezone.utc)
    common = sorted(set(a) & set(b))
    upcoming = [t for t in common if t + SLOT > now][: lookahead + 1]
    near = []
    for t in upcoming:
        la, sb = a[t], b[t]
        near.append({
            "time": t, "live": la.get("mode"), "shadow": sb.get("mode"),
            "live_wit": la.get("wit_mode"), "shadow_wit": sb.get("wit_mode"),
            "match": (la.get("mode"), la.get("wit_mode")) == (sb.get("mode"), sb.get("wit_mode")),
        })
    count_a = Counter(a[t].get("mode") for t in common)
    count_b = Counter(b[t].get("mode") for t in common)
    modes = sorted(set(count_a) | set(count_b))
    totals = {m: (count_a.get(m, 0), count_b.get(m, 0)) for m in modes}
    near_ok = len(near) == lookahead + 1 and all(r["match"] for r in near)
    totals_ok = all(abs(x - y) <= 1 for x, y in totals.values())
    return {
        "common": common, "near": near, "totals": totals,
        "near_ok": near_ok, "totals_ok": totals_ok,
        "live_only": len(set(a) - set(b)), "shadow_only": len(set(b) - set(a)),
        "ok": near_ok and totals_ok,
    }


def _get(entity: str, token: str) -> Optional[dict]:
    import requests

    r = requests.get(f"{HA_URL}/api/states/{entity}",
                     headers={"Authorization": f"Bearer {token}"}, timeout=20)
    if r.status_code == 404:
        return None
    if r.status_code == 401:
        raise SystemExit("401 from HA - not retrying")
    r.raise_for_status()
    return r.json()


def _summary(name: str, main: Optional[dict]) -> str:
    if not main:
        return f"{name}: (missing)"
    a = main.get("attributes", {})
    horizon = a.get("price_horizon") or {}
    return (f"{name}: state={main.get('state')} app_version={a.get('app_version')} "
            f"slots={a.get('schedule_slots')} horizon_ok={horizon.get('ok')} "
            f"last_optimization={a.get('last_optimization')} "
            f"last_updated={main.get('last_updated')}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suffix", default="_shadow")
    parser.add_argument("--rows", type=int, default=400)
    args = parser.parse_args(argv)

    token = TOKEN_FILE.read_text(encoding="utf-8").splitlines()[0].strip()
    live_main = _get("sensor.battery_optimizer", token)
    shadow_main = _get(f"sensor.battery_optimizer{args.suffix}", token)
    live_sched = _get("sensor.battery_optimizer_schedule", token) or {}
    shadow_sched = _get(f"sensor.battery_optimizer_schedule{args.suffix}", token) or {}
    print(_summary("live  ", live_main))
    print(_summary("shadow", shadow_main))

    now = datetime.datetime.now(datetime.timezone.utc)
    live = (live_sched.get("attributes") or {}).get("schedule") or []
    shadow = (shadow_sched.get("attributes") or {}).get("schedule") or []
    result = compare(live, shadow, now)
    a, b = plan_by_slot(live), plan_by_slot(shadow)

    print()
    print(f"{'slot (local)':<17} {'live':<10} {'live WIT':<18} {'shadow':<10} {'shadow WIT':<18}")
    for t in result["common"][: args.rows]:
        la, sb = a[t], b[t]
        same = (la.get("mode"), la.get("wit_mode")) == (sb.get("mode"), sb.get("wit_mode"))
        local = t.astimezone().strftime("%m-%d %H:%M")
        print(f"{local:<17} {la.get('mode', ''):<10} {str(la.get('wit_mode')):<18} "
              f"{sb.get('mode', ''):<10} {str(sb.get('wit_mode')):<18}{'' if same else ' *'}")

    print()
    print("current + next 4:")
    for r in result["near"]:
        print(f"  {r['time'].astimezone().strftime('%H:%M')} live={r['live']}/{r['live_wit']} "
              f"shadow={r['shadow']}/{r['shadow_wit']} {'OK' if r['match'] else 'DIFF'}")
    print("mode totals over the common range (slots live/shadow):")
    for m, (x, y) in result["totals"].items():
        print(f"  {m:<10} {x:>4} / {y:<4} {'OK' if abs(x - y) <= 1 else 'DIFF'}")
    print(f"common slots={len(result['common'])} live-only={result['live_only']} "
          f"shadow-only={result['shadow_only']}")
    print("RESULT:", "PASS" if result["ok"] else "FAIL",
          json.dumps({"near_ok": result["near_ok"], "totals_ok": result["totals_ok"]}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
