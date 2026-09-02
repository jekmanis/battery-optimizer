#!/usr/bin/env python
"""
Drop implausible observations from a battery_learning_data.json file.

Why this exists
---------------
Production 2026-09-02: ``cost_tracker`` computed a learning observation's
duration from ``_last_sig_soc_time``, which the SOC listener re-stamps
milliseconds before the energy-sensor callback records the charge. A genuine
0.1 kWh delta divided by a 10-40 ms "duration" produced charge-rate observations
of 34535 kW and 44653 kW. They landed in ``charge_rates_by_soc[_temp]``, and
``get_charge_rate_for_soc`` returned their raw median: **14308.71 kW** for the
0-25 %/>20 C bucket of a 4.5 kW battery.

``BatteryLearningEngine`` now rejects such samples at ingest and filters them out
of every median, so a poisoned file is already neutralised in memory on load.
This script is for INSPECTING what a file contains and for producing a cleaned
copy offline, using the SAME rule (``BatteryLearningEngine.sanitize_stats``) —
there is deliberately no second cleaning rule to drift from the engine's.

It NEVER writes to the input file. Give it an explicit ``--out``.

Usage:
    uv run python scripts/clean_learning_data.py <in.json> --out <out.json>
    uv run python scripts/clean_learning_data.py <in.json>            # report only
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "appdaemon" / "apps"))

from battery_optimizer_lib.learning_engine import (  # noqa: E402
    BatteryLearningEngine,
    thermal_coeffs_are_sane,
)


def _median(rates):
    return statistics.median(rates[-10:]) if rates else None


def _fmt(value):
    return "-" if value is None else f"{value:.3f}"


def _bucket_report(title, before, after, bound):
    print(f"\n{title}")
    print(f"  {'bucket':<18}{'n':>5}{'->':>4}{'n':>5}   "
          f"{'median(last10)':>16} -> {'median(last10)':>16}   dropped")
    for key in sorted(before):
        old = before[key]
        new = after.get(key, [])
        dropped = [r for r in old if r > bound]
        marker = "  <-- POISONED" if dropped else ""
        print(f"  {key:<18}{len(old):>5}{'->':>4}{len(new):>5}   "
              f"{_fmt(_median(old)):>16} -> {_fmt(_median(new)):>16}   "
              f"{len(dropped)}{marker}")
        for rate in dropped:
            print(f"      dropped {rate:.3f} kW")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="learning data JSON to inspect")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the cleaned copy (never the input)")
    parser.add_argument("--capacity", type=float, default=14.3,
                        help="battery_capacity_kwh (default 14.3)")
    parser.add_argument("--charge-rate", type=float, default=4.5,
                        help="nominal charge_rate_kw (default 4.5)")
    parser.add_argument("--discharge-rate", type=float, default=None,
                        help="nominal discharge_rate_kw (widens the bound)")
    parser.add_argument("--max-rate-factor", type=float, default=None,
                        help="override the plausibility factor (default 2.0x nominal)")
    parser.add_argument("--drop-thermal-fit", action="store_true",
                        help="also discard thermal_coeffs/thermal_samples entirely "
                             "(re-bootstrap the thermal model from scratch)")
    args = parser.parse_args()

    if args.out is not None and args.out.resolve() == args.input.resolve():
        print("Refusing to overwrite the input file; choose a different --out.")
        return 2

    raw = args.input.read_text(encoding="utf-8")

    kwargs = dict(
        battery_capacity_kwh=args.capacity,
        nominal_charge_rate_kw=args.charge_rate,
        nominal_discharge_rate_kw=args.discharge_rate,
        log_func=lambda *a, **k: None,
    )
    if args.max_rate_factor is not None:
        kwargs["max_rate_factor"] = args.max_rate_factor
    engine = BatteryLearningEngine(**kwargs)

    # Snapshot the raw contents BEFORE the engine sanitises them on load.
    original = json.loads(raw).get("stats", {})
    before_soc = {k: list(v) for k, v in original.get("charge_rates_by_soc", {}).items()}
    before_soc_temp = {
        f"{soc} / {temp}": list(rates)
        for soc, temp_data in original.get("charge_rates_by_soc_temp", {}).items()
        for temp, rates in temp_data.items()
    }
    before_thermal = list(original.get("thermal_samples", []))
    before_coeffs = dict(original.get("thermal_coeffs", {}) or {})

    if not engine.load_from_json(raw):
        print(f"Could not parse {args.input}")
        return 1

    bound = engine.max_plausible_rate_kw
    print(f"Input:  {args.input}")
    print(f"Bound:  {bound:.2f} kW "
          f"({engine.max_rate_factor:.1f}x nominal "
          f"{max(args.charge_rate, args.discharge_rate or 0):.2f} kW)")
    print(f"Floor:  {engine.min_observation_minutes:.2f} min per observation, "
          f"or {2 * engine.counter_resolution_kwh:.2f} kWh "
          f"(2x counter resolution) at any duration")

    after_soc = {k: list(v) for k, v in engine.stats.charge_rates_by_soc.items()}
    after_soc_temp = {
        f"{soc} / {temp}": list(rates)
        for soc, temp_data in engine.stats.charge_rates_by_soc_temp.items()
        for temp, rates in temp_data.items()
    }

    _bucket_report("charge_rates_by_soc", before_soc, after_soc, bound)
    _bucket_report("charge_rates_by_soc_temp", before_soc_temp, after_soc_temp, bound)

    if args.drop_thermal_fit:
        engine.reset_thermal_calibration()

    print("\nthermal")
    print(f"  samples      {len(before_thermal)} -> {len(engine.stats.thermal_samples)}")
    print(f"  coeffs       {before_coeffs or '{}'} -> {engine.stats.thermal_coeffs or '{}'}"
          f"{'' if thermal_coeffs_are_sane(before_coeffs) or not before_coeffs else '  <-- out of bounds'}")
    print(f"  k1 estimate  {engine.get_cooling_rate_estimate(engine.get_estimated_ambient_min_temp()):.4f}/min")
    print(f"  k2 estimate  {engine.get_heating_coefficient():.3f} C/kWh")

    print("\nresulting get_charge_rate_for_soc(soc, temp):")
    header = "".join(f"{'T=' + str(t):>10}" for t in (8, 12, 17, 22, 26)) + f"{'T=None':>10}"
    print(f"  {'soc':>5}{header}")
    for soc in (10, 30, 60, 80, 95):
        row = "".join(
            f"{engine.get_charge_rate_for_soc(soc, t):>10.2f}" for t in (8, 12, 17, 22, 26)
        )
        row += f"{engine.get_charge_rate_for_soc(soc, None):>10.2f}"
        print(f"  {soc:>5}{row}")

    if args.out is None:
        print("\nNo --out given: nothing written.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(engine.save_to_json(), encoding="utf-8")
    print(f"\nCleaned copy written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
