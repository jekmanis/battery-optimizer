#!/usr/bin/env python
"""
Reproduce and profile one production ``DPOptimizer.optimize`` solve.

The startup ``full_optimize`` of commit 09b4814 took 206 s on the HA machine
where the previous release took 9 s, with ``rate_refinement`` reporting
``{'branch': 'converged', 'passes': 2}`` -- so the thermal refinement is not
looping and each solve is simply expensive.  This script rebuilds that solve
locally from the LIVE persisted data files so the predictors handed to the DP
are the real ones (learned charge rates, load profile, PV profile, prediction
tracker, ambient service, temperature projector).

The share is only ever READ.  Copy the JSON files somewhere local first:

    mkdir -p "$LOCALAPPDATA/Temp/bo-profile"
    cp //192.168.77.167/addon_configs/a0d7b954_appdaemon/{battery_learning_data,load_profile,pv_profile,prediction_tracker}.json \
       "$LOCALAPPDATA/Temp/bo-profile/"

Usage:
    uv run --python 3.14 --extra dev --isolated python scripts/profile_dp.py
    uv run ... python scripts/profile_dp.py --profile          # cProfile
    uv run ... python scripts/profile_dp.py --counters         # predictor call counts
    uv run ... python scripts/profile_dp.py --save-baseline out.json
    uv run ... python scripts/profile_dp.py --compare out.json # identical plan?
"""

import argparse
import cProfile
import datetime
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
APPS_DIR = REPO_ROOT / "appdaemon" / "apps"

DEFAULT_DATA_DIR = Path(
    os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
) / "Temp" / "bo-profile"
DEFAULT_APPS_YAML = (
    "//192.168.77.167/addon_configs/a0d7b954_appdaemon/apps/apps.yaml"
)

# The live case being reproduced (see the module docstring).
LIVE_TZ = "Europe/Riga"
LIVE_START = datetime.datetime(2026, 9, 5, 16, 30)
LIVE_N_SLOTS = 130
LIVE_SOC = 100.0
LIVE_TEMP = 33.1
LIVE_MINUTES_INTO_SLOT = 2.0


def _install_mocks():
    sys.path.insert(0, str(TESTS_DIR))
    import conftest  # noqa: F401  installs the appdaemon.plugins.hass mock
    sys.path.insert(0, str(APPS_DIR))


def _load_config(apps_yaml: str, data_dir: Path):
    import yaml
    from battery_optimizer_lib.config import BatteryOptimizerConfig

    raw = yaml.safe_load(Path(apps_yaml).read_text(encoding="utf-8"))
    app_args = next(
        v for v in raw.values()
        if isinstance(v, dict) and v.get("module") == "battery_optimizer"
    )
    cfg = BatteryOptimizerConfig.from_args(app_args, log_func=lambda *a, **k: None)
    # Never read (or write) the share for the state files.
    cfg.learning_data_file = str(data_dir / "battery_learning_data.json")
    cfg.load_profile_file = str(data_dir / "load_profile.json")
    cfg.pv_profile_file = str(data_dir / "pv_profile.json")
    cfg.prediction_tracker_file = str(data_dir / "prediction_tracker.json")
    return cfg


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def build_context(cfg, quiet=True):
    """Mirror ``BatteryOptimizer.initialize``'s DP wiring, standalone."""
    from zoneinfo import ZoneInfo

    from battery_optimizer_lib.ambient_service import (
        AmbientServiceConfig,
        AmbientTemperatureService,
    )
    from battery_optimizer_lib.learning_engine import BatteryLearningEngine
    from battery_optimizer_lib.load_prediction_tracker import LoadPredictionTracker
    from battery_optimizer_lib.load_profile import LoadProfile
    from battery_optimizer_lib.pv_profile import PvProfile
    from battery_optimizer_lib.thermal_model import TemperatureProjector

    log = (lambda *a, **k: None) if quiet else (lambda m, level="INFO": print(f"[{level}] {m}"))
    tz = ZoneInfo(LIVE_TZ)
    now = LIVE_START.replace(tzinfo=tz) + datetime.timedelta(
        minutes=LIVE_MINUTES_INTO_SLOT
    )

    engine = BatteryLearningEngine(
        battery_capacity_kwh=cfg.battery_capacity,
        nominal_charge_rate_kw=cfg.charge_rate,
        nominal_discharge_rate_kw=cfg.discharge_rate,
        nominal_export_rate_kw=cfg.effective_export_discharge_rate,
        nominal_efficiency=cfg.efficiency,
        min_soc=cfg.default_min_soc,
        max_soc=cfg.default_max_soc,
        log_func=log,
    )
    engine.load_from_json(_read(cfg.learning_data_file))

    load_profile = LoadProfile(
        slot_minutes=cfg.slot_minutes,
        default_load_w=cfg.base_consumption,
        max_samples=cfg.load_profile_max_samples,
        min_samples=cfg.load_profile_min_samples,
        log_func=log,
    )
    load_profile.load_from_json(_read(cfg.load_profile_file))

    tracker = LoadPredictionTracker(slot_minutes=cfg.slot_minutes, log_func=log)
    tracker.load_from_json(_read(cfg.prediction_tracker_file))

    pv_profile = PvProfile(
        slot_minutes=cfg.slot_minutes,
        default_pv_w=0.0,
        max_samples=cfg.pv_profile_max_samples,
        min_samples=cfg.pv_profile_min_samples,
        log_func=log,
    )
    pv_profile.load_from_json(_read(cfg.pv_profile_file))

    ambient = AmbientTemperatureService(
        config=AmbientServiceConfig.from_main_config(cfg),
        get_state_func=None,          # no HA here: diurnal profile fallback
        call_service_func=None,
        get_datetime_func=lambda: now,
        get_timezone_func=lambda: tz,
        log_func=log,
        min_temp_provider=lambda: engine.get_estimated_ambient_min_temp(default=None),
    )
    projector = TemperatureProjector(
        learning_engine=engine,
        ambient_provider=ambient,
        log_func=log,
        default_cooling_rate=cfg.thermal_default_cooling_rate_per_min,
        default_heating_c_per_kwh=cfg.thermal_default_heating_c_per_kwh,
    )

    def predict_load_kw(dt):
        return load_profile.predict_kw(
            dt, cfg.load_quantile, tracker.get_correction_factor(dt)
        )

    def predict_pv_kw(dt):
        return pv_profile.predict_kw(dt)

    return {
        "tz": tz,
        "now": now,
        "engine": engine,
        "load_profile": load_profile,
        "pv_profile": pv_profile,
        "tracker": tracker,
        "projector": projector,
        "predict_load_kw": predict_load_kw,
        "predict_pv_kw": predict_pv_kw,
        "log": log,
    }


def build_prices(tz, n_slots=LIVE_N_SLOTS, slot_minutes=15):
    """130 synthetic 15-minute prices, 0.005-0.25 EUR/kWh with evening peaks.

    Real Nord Pool values for the day are not available offline; the shape only
    has to be plausible, because the runtime question is per-state work, not
    which action wins.
    """
    from battery_optimizer_lib.models import PricePoint

    start = LIVE_START.replace(tzinfo=tz)
    points = []
    for i in range(n_slots):
        t = start + datetime.timedelta(minutes=slot_minutes * i)
        h = t.hour + t.minute / 60.0
        if 17.0 <= h < 22.0:
            price = 0.18 + 0.07 * ((h - 17.0) / 5.0)
        elif 0.0 <= h < 6.0:
            price = 0.005 + 0.02 * (h / 6.0)
        elif 10.0 <= h < 15.0:
            price = 0.02 + 0.01 * ((h - 10.0) / 5.0)
        else:
            price = 0.09 + 0.03 * ((i % 7) / 7.0)
        points.append(PricePoint(time=t, price=round(price, 5)))
    return points


class CountingPredictor:
    """Wraps the charge-rate predictor and counts calls + total seconds."""

    def __init__(self, fn):
        self._fn = fn
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, soc, temp=None):
        self.calls += 1
        t0 = time.perf_counter()
        try:
            return self._fn(soc, temp)
        finally:
            self.seconds += time.perf_counter() - t0


def instrument_breakdown(optimizer, counter):
    """Attribute ``charge_rate_predictor`` calls to the method that caused them."""
    import functools

    from battery_optimizer_lib.dp_optimizer import DPOptimizer

    tally = {}

    def wrap(name):
        original = getattr(DPOptimizer, name)

        @functools.wraps(original)
        def wrapper(self, *a, **kw):
            before, t0 = counter.calls, time.perf_counter()
            try:
                return original(self, *a, **kw)
            finally:
                rec = tally.setdefault(name, [0, 0, 0.0])
                rec[0] += 1
                rec[1] += counter.calls - before
                rec[2] += time.perf_counter() - t0

        setattr(optimizer, name, wrapper.__get__(optimizer, DPOptimizer))

    for name in ("_run_dp", "_profiles_agree", "_replay_plan", "_idle_temp_profile"):
        wrap(name)
    return tally


def make_optimizer(cfg, ctx, rate_predictor=None):
    from battery_optimizer_lib.dp_optimizer import DPOptimizer, DPOptimizerConfig

    return DPOptimizer(
        config=DPOptimizerConfig.from_main_config(
            cfg, min_soc=cfg.default_min_soc, max_soc=cfg.default_max_soc
        ),
        load_predictor=ctx["predict_load_kw"],
        charge_rate_predictor=rate_predictor or ctx["engine"].get_charge_rate_for_soc,
        temp_after_charge_predictor=ctx["engine"].predict_temp_after_duration,
        temp_after_idle_predictor=ctx["engine"].predict_temp_after_idle,
        log_fn=ctx["log"],
        decision_log_level=0,
        pv_predictor=ctx["predict_pv_kw"],
        temp_projector=ctx["projector"],
        warn_degenerate_terminal=False,
    )


LAST_OPTIMIZER = []


def run_solve(cfg, ctx, prices, rate_predictor=None, breakdown=None):
    optimizer = make_optimizer(cfg, ctx, rate_predictor)
    LAST_OPTIMIZER[:] = [optimizer]
    if breakdown is not None:
        breakdown["tally"] = instrument_breakdown(optimizer, rate_predictor)
    return optimizer.optimize(
        prices=prices,
        current_slot=LIVE_START.replace(tzinfo=ctx["tz"]),
        current_soc=LIVE_SOC,
        current_temp=LIVE_TEMP,
        minutes_into_slot=LIVE_MINUTES_INTO_SLOT,
    )


def fingerprint(result):
    """Everything a consumer of the result can observe, as plain JSON."""
    def key(dt):
        return dt.isoformat()

    def by_time(item):
        return item[0].isoformat()

    return {
        "schedule": [
            [
                key(t),
                e.mode.name,
                e.reason,
                e.export_rate,
                e.ac_charge_mode,
                round(e.marginal_value_eur_kwh, 10)
                if e.marginal_value_eur_kwh is not None else None,
                e.value_basis,
                bool(e.energy_limited),
                e.price_source,
            ]
            for t, e in sorted(result.schedule.items(), key=by_time)
        ],
        "soc": [
            [key(t), round(v[0], 10), round(v[1], 10)]
            for t, v in sorted(result.soc_trajectory.items(), key=by_time)
        ],
        "temp": [
            [
                key(t),
                round(v[0], 10) if v[0] is not None else None,
                round(v[1], 10) if v[1] is not None else None,
            ]
            for t, v in sorted(result.temp_trajectory.items(), key=by_time)
        ],
        "counts": [
            result.charge_count, result.hold_count,
            result.export_slot_count, result.self_consume_slot_count,
        ],
        "terminal": result.terminal_value_eur_kwh,
        "refinement": [
            result.rate_refinement_passes,
            result.rate_refinement_branch,
            round(result.rate_refinement_shortfall_kwh, 10),
        ],
        "planning_temp": [
            [key(t), round(v, 10) if v is not None else None]
            for t, v in sorted(result.planning_temp_by_slot.items(), key=by_time)
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apps-yaml", default=DEFAULT_APPS_YAML)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--slots", type=int, default=LIVE_N_SLOTS)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--profile", action="store_true", help="run under cProfile")
    ap.add_argument("--counters", action="store_true",
                    help="count charge_rate_predictor calls and their cost")
    ap.add_argument("--breakdown", action="store_true",
                    help="attribute predictor calls to _run_dp / _profiles_agree / _replay_plan")
    ap.add_argument("--save-baseline", default=None)
    ap.add_argument("--compare", default=None,
                    help="assert the result is identical to this baseline")
    args = ap.parse_args(argv)

    _install_mocks()
    data_dir = Path(args.data_dir)
    cfg = _load_config(args.apps_yaml, data_dir)
    ctx = build_context(cfg)
    prices = build_prices(ctx["tz"], n_slots=args.slots)

    print(f"slots={len(prices)} soc_step={cfg.soc_step_percent}% "
          f"capacity={cfg.battery_capacity} min_soc={cfg.default_min_soc} "
          f"max_soc={cfg.default_max_soc} soc={LIVE_SOC} temp={LIVE_TEMP}")
    n_states = int(
        ((cfg.default_max_soc - cfg.default_min_soc) / 100 * cfg.battery_capacity)
        / ((cfg.soc_step_percent / 100) * cfg.battery_capacity) + 1e-9
    ) + 1
    print(f"n_states={n_states}")

    counter = None
    predictor = None
    if args.counters or args.breakdown:
        counter = CountingPredictor(ctx["engine"].get_charge_rate_for_soc)
        predictor = counter
    bd = {} if args.breakdown else None

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
        result = run_solve(cfg, ctx, prices, predictor)
        prof.disable()
        for sort in ("cumulative", "tottime"):
            s = io.StringIO()
            pstats.Stats(prof, stream=s).sort_stats(sort).print_stats(28)
            print(f"\n===== sorted by {sort} =====")
            print(s.getvalue())
    else:
        elapsed = []
        for _ in range(max(1, args.repeat)):
            t0 = time.perf_counter()
            result = run_solve(cfg, ctx, prices, predictor, bd)
            elapsed.append(time.perf_counter() - t0)
        for i, e in enumerate(elapsed):
            print(f"run {i + 1}: {e:.3f} s")
        print(f"best: {min(elapsed):.3f} s")

    print(f"branch={result.rate_refinement_branch} "
          f"passes={result.rate_refinement_passes} "
          f"charge={result.charge_count} hold={result.hold_count} "
          f"export={result.export_slot_count} "
          f"self_consume={result.self_consume_slot_count}")

    if counter is not None:
        print(f"charge_rate_predictor calls={counter.calls} "
              f"total={counter.seconds:.3f} s "
              f"per_call={counter.seconds / max(1, counter.calls) * 1e6:.1f} us")

    if LAST_OPTIMIZER:
        opt = LAST_OPTIMIZER[0]
        print(f"visited_state_socs={len(getattr(opt, '_visited_state_socs', ()))} "
              f"rate_cache_entries={len(getattr(opt, '_rate_cache', ()))}")

    if bd:
        print("breakdown (calls into the predictor, attributed to the caller):")
        for name, (n, calls, secs) in sorted(
            bd["tally"].items(), key=lambda kv: -kv[1][1]
        ):
            print(f"  {name:<20} invocations={n:<5} predictor_calls={calls:<10} "
                  f"wall={secs:.3f} s")

    fp = fingerprint(result)
    if args.save_baseline:
        Path(args.save_baseline).write_text(
            json.dumps(fp, indent=1, sort_keys=True), encoding="utf-8"
        )
        print(f"baseline written: {args.save_baseline}")
    if args.compare:
        want = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        if want == fp:
            print("IDENTICAL to baseline (schedule, SOC, temperature, counts, "
                  "refinement, planning temps)")
        else:
            print("DIFFERENT from baseline:")
            for k in want:
                if want[k] != fp.get(k):
                    print(f"  section {k} differs")
                    a, b = want[k], fp.get(k)
                    if isinstance(a, list) and isinstance(b, list):
                        for i, (x, y) in enumerate(zip(a, b)):
                            if x != y:
                                print(f"    [{i}] baseline={x}")
                                print(f"    [{i}] current ={y}")
                                break
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
