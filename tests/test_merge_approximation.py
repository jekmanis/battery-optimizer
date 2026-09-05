"""The bucket merge is an APPROXIMATION, and this is the counterexample.

The solver keeps one path per SOC bucket: the highest-valued one. That is not a
valid dominance rule. A higher-valued path does **not** dominate a lower-valued
path holding more energy -- the extra energy can be worth more later than the
value gap is worth now. Documentation used to claim the solver was "exact for
its discretized model"; it is not, and this file is why.

The maintainer's example, at the default 1 % step:

    10 kWh pack, min SOC 10 %, initial SOC 10.9 %
    two 15-minute slots, house load 0.2 kW
    prices 0.10 then 1.00 EUR/kWh
    storage and inverter efficiency 1.0
    no grid fee, no wear, no export, no terminal value

    the solver returns   DISCHARGE, DISCHARGE   costing 0.010 EUR
    exhaustive search finds HOLD, DISCHARGE     costing 0.005 EUR

Why. A slot serves 0.2 kW * 0.25 h = 0.05 kWh. Both paths land in bucket 0
after slot 1 (its 0.1 kWh span covers 1.00 to 1.10 kWh). DISCHARGE ends there
holding 1.04 kWh with value 0; HOLD ends holding 1.09 kWh with value -0.005,
having bought its 0.05 kWh at the cheap price. DISCHARGE wins the merge on
value, and the 0.05 kWh of usable energy the HOLD path was carrying -- worth
1.00 EUR/kWh in slot 2 -- is discarded with it.

This test PINS the approximation. It is not a wish: if a future change to the
merge rule makes the solver find 0.005 here, this test fails and should be
updated deliberately, with the new rule's cost measured. What must never happen
is the solver claiming a value the enumeration says is impossible -- that is a
recurrence error, and ``tests/test_dp_energy_conservation.py::
TestExhaustiveEnumeration`` is the test for it.

The exact alternative, and why it is not used, is in
``docs/scheduling-algorithm.md`` SS Conservative quantization.
"""

import datetime
import itertools
import re

import pytest

from battery_optimizer_lib import (
    BatteryMode,
    DPOptimizer,
    DPOptimizerConfig,
    PricePoint,
    ScheduleEntry,
)
from battery_optimizer_lib.slot_energy import SlotEnergyParams, simulate_slot


BASE = datetime.datetime(2026, 4, 6, 0, 0, 0)

CAPACITY = 10.0
MIN_SOC = 10.0
INITIAL_SOC = 10.9
LOAD_KW = 0.2
PRICES = [0.10, 1.00]


def _config() -> DPOptimizerConfig:
    return DPOptimizerConfig(
        battery_capacity=CAPACITY,
        min_soc=MIN_SOC,
        max_soc=100.0,
        efficiency=1.0,
        discharge_rate=4.0,
        slot_minutes=15,
        soc_step_percent=1.0,
        grid_fee=0.0,
        battery_wear_cost=0.0,
        grid_export_fee=0.0,
        export_rate_multiplier=0.0,
        inverter_efficiency=1.0,
        import_price_multiplier=1.0,
        terminal_energy_value_eur_kwh=0.0,
    )


def _prices():
    return [
        PricePoint(time=BASE + datetime.timedelta(minutes=15 * i), price=p)
        for i, p in enumerate(PRICES)
    ]


def _optimizer(cfg):
    return DPOptimizer(
        config=cfg,
        load_predictor=lambda dt: LOAD_KW,
        # No charging capability: CHARGE is physically identical to HOLD here,
        # so the counterexample is about the merge and nothing else.
        charge_rate_predictor=lambda soc, temp: 0.0,
        temp_after_charge_predictor=lambda t, m: t,
        temp_after_idle_predictor=lambda t, m: t,
        pv_predictor=lambda dt: 0.0,
    )


def _cost_of(cfg, modes, initial_soc: float = INITIAL_SOC) -> float:
    """Grid cost (EUR) of one action sequence, from the shared physical model.

    Independent of the DP: it walks ``simulate_slot`` and prices the import at
    the same tariff the DP optimizes. Nothing here reads the planner's answer.
    """
    params = SlotEnergyParams(
        battery_capacity=cfg.battery_capacity,
        efficiency=cfg.efficiency,
        discharge_rate=cfg.discharge_rate,
        inverter_efficiency=cfg.inverter_efficiency,
        min_soc=cfg.min_soc,
        max_soc=cfg.max_soc,
        slot_minutes=cfg.slot_minutes,
    )
    energy = initial_soc / 100.0 * cfg.battery_capacity
    cost = 0.0
    for mode, price in zip(modes, PRICES):
        outcome = simulate_slot(
            stored_energy_kwh=energy,
            mode=mode,
            params=params,
            charge_input_dc_kw=0.0,
            load_kw=LOAD_KW,
            pv_kw=0.0,
        )
        cost += price * outcome.grid_import_ac_kwh
        energy = outcome.energy_end_kwh
    return cost


def _exhaustive_best(cfg):
    """The cheapest of all 3^2 action sequences, and the sequence itself."""
    best = None
    for combo in itertools.product(
        [BatteryMode.HOLD, BatteryMode.CHARGE, BatteryMode.DISCHARGE], repeat=2
    ):
        cost = _cost_of(cfg, combo)
        if best is None or cost < best[0] - 1e-12:
            best = (cost, combo)
    return best


class TestTheMergeKeepsValueAndDiscardsEnergy:
    def test_the_arithmetic_of_the_counterexample(self):
        """Independent of the planner: 0.010 vs 0.005 EUR."""
        cfg = _config()
        discharge_twice = _cost_of(
            cfg, (BatteryMode.DISCHARGE, BatteryMode.DISCHARGE)
        )
        hold_then_discharge = _cost_of(
            cfg, (BatteryMode.HOLD, BatteryMode.DISCHARGE)
        )
        assert discharge_twice == pytest.approx(0.010, abs=1e-12)
        assert hold_then_discharge == pytest.approx(0.005, abs=1e-12)

    def test_exhaustive_search_finds_the_cheaper_plan(self):
        cfg = _config()
        cost, combo = _exhaustive_best(cfg)
        assert cost == pytest.approx(0.005, abs=1e-12)
        assert combo[1] == BatteryMode.DISCHARGE
        assert combo[0] in (BatteryMode.HOLD, BatteryMode.CHARGE)

    def test_the_solver_returns_the_more_expensive_plan(self):
        """PINNED, not wished for. The merge discards the higher-energy path."""
        cfg = _config()
        result = _optimizer(cfg).optimize(
            prices=_prices(),
            current_slot=BASE,
            current_soc=INITIAL_SOC,
        )
        modes = [result.schedule[p.time].mode for p in _prices()]
        assert modes == [BatteryMode.DISCHARGE, BatteryMode.DISCHARGE]
        assert _cost_of(cfg, modes) == pytest.approx(0.010, abs=1e-12)

    def test_the_gap_is_within_one_step_times_the_marginal_value(self):
        """The per-merge value bound the docs state: step x marginal value.

        The solver is CALLED. Hardcoding ``(DISCHARGE, DISCHARGE)`` here made
        this an arithmetic identity about two constants -- it would have gone on
        passing if the solver started returning anything at all, including the
        plan the enumeration prefers, and the bound it claims to check is a
        bound on the SOLVER's gap.
        """
        cfg = _config()
        best, _combo = _exhaustive_best(cfg)
        result = _optimizer(cfg).optimize(
            prices=_prices(),
            current_slot=BASE,
            current_soc=INITIAL_SOC,
        )
        modes = [result.schedule[p.time].mode for p in _prices()]
        solver = _cost_of(cfg, modes)
        step_kwh = cfg.soc_step_percent / 100.0 * cfg.battery_capacity
        marginal_value = max(PRICES)
        assert solver - best <= step_kwh * marginal_value + 1e-12

    def test_the_solver_never_beats_the_enumeration(self):
        """The merge loses value; it must never INVENT any.

        A solver returning a plan cheaper than every physically simulated
        sequence would be a recurrence error, not an approximation.
        """
        cfg = _config()
        best, _combo = _exhaustive_best(cfg)
        result = _optimizer(cfg).optimize(
            prices=_prices(),
            current_slot=BASE,
            current_soc=INITIAL_SOC,
        )
        modes = [result.schedule[p.time].mode for p in _prices()]
        assert _cost_of(cfg, modes) >= best - 1e-12

    def test_a_finer_grid_recovers_the_cheaper_plan(self):
        """The gap is the MERGE, not the recurrence.

        At a 0.1 % step the two paths no longer share a bucket and the solver
        finds the 0.005 EUR plan. Nothing else changes -- which is the evidence
        that what is lost is resolution of the merge, not correctness of the
        transition.

        It does NOT say that finer is always better -- see
        ``TestTheErrorIsNotMonotoneInTheStep``. On this case a 0.9 % step is
        already enough; 0.1 % is not the threshold, it is just a step small
        enough to be sure.
        """
        cfg = _config()
        cfg.soc_step_percent = 0.1
        result = _optimizer(cfg).optimize(
            prices=_prices(),
            current_slot=BASE,
            current_soc=INITIAL_SOC,
        )
        modes = [result.schedule[p.time].mode for p in _prices()]
        assert _cost_of(cfg, modes) == pytest.approx(0.005, abs=1e-12)

    def test_a_slightly_finer_grid_already_recovers_it(self):
        """0.9 % is enough: the pinned example is the mildest of its family."""
        cfg = _config()
        cfg.soc_step_percent = 0.9
        result = _optimizer(cfg).optimize(
            prices=_prices(),
            current_slot=BASE,
            current_soc=INITIAL_SOC,
        )
        modes = [result.schedule[p.time].mode for p in _prices()]
        assert _cost_of(cfg, modes) == pytest.approx(0.005, abs=1e-12)

    def test_a_lower_initial_soc_costs_more(self):
        """Same two slots from 10.5 %: a 0.045 EUR gap, not 0.005."""
        cfg = _config()
        result = _optimizer(cfg).optimize(
            prices=_prices(),
            current_slot=BASE,
            current_soc=10.5,
        )
        modes = [result.schedule[p.time].mode for p in _prices()]
        solver = _cost_of(cfg, modes, initial_soc=10.5)
        best = min(
            _cost_of(cfg, combo, initial_soc=10.5)
            for combo in itertools.product(
                [BatteryMode.HOLD, BatteryMode.CHARGE, BatteryMode.DISCHARGE],
                repeat=2,
            )
        )
        assert solver - best == pytest.approx(0.045, abs=1e-9)


# ---------------------------------------------------------------------------
# The error is not monotone in the step
# ---------------------------------------------------------------------------

NM_PRICES = [0.6450, 0.9446, 0.6896, 0.7114, 0.0915]
NM_LOAD_KW = 1.35
NM_RATE_KW = 1.0
NM_INITIAL_SOC = 14.75


def _nm_config(step: float) -> DPOptimizerConfig:
    cfg = _config()
    cfg.soc_step_percent = step
    return cfg


def _nm_cost(cfg, modes) -> float:
    params = SlotEnergyParams(
        battery_capacity=cfg.battery_capacity,
        efficiency=cfg.efficiency,
        discharge_rate=cfg.discharge_rate,
        inverter_efficiency=cfg.inverter_efficiency,
        min_soc=cfg.min_soc,
        max_soc=cfg.max_soc,
        slot_minutes=cfg.slot_minutes,
    )
    energy = NM_INITIAL_SOC / 100.0 * cfg.battery_capacity
    cost = 0.0
    for mode, price in zip(modes, NM_PRICES):
        outcome = simulate_slot(
            stored_energy_kwh=energy,
            mode=mode,
            params=params,
            charge_input_dc_kw=NM_RATE_KW,
            load_kw=NM_LOAD_KW,
            pv_kw=0.0,
        )
        cost += price * outcome.grid_import_ac_kwh
        energy = outcome.energy_end_kwh
    return cost


def _nm_gap(step: float) -> float:
    cfg = _nm_config(step)
    opt = DPOptimizer(
        config=cfg,
        load_predictor=lambda dt: NM_LOAD_KW,
        charge_rate_predictor=lambda soc, temp: NM_RATE_KW,
        temp_after_charge_predictor=lambda t, m: t,
        temp_after_idle_predictor=lambda t, m: t,
        pv_predictor=lambda dt: 0.0,
    )
    points = [
        PricePoint(time=BASE + datetime.timedelta(minutes=15 * i), price=p)
        for i, p in enumerate(NM_PRICES)
    ]
    result = opt.optimize(prices=points, current_slot=BASE, current_soc=NM_INITIAL_SOC)
    modes = [result.schedule[p.time].mode for p in points]
    solver = _nm_cost(cfg, modes)
    best = min(
        _nm_cost(cfg, combo)
        for combo in itertools.product(
            [BatteryMode.HOLD, BatteryMode.CHARGE, BatteryMode.DISCHARGE],
            repeat=len(NM_PRICES),
        )
    )
    return solver - best


class TestTheErrorIsNotMonotoneInTheStep:
    """A finer grid usually helps and sometimes hurts.

    `docs/scheduling-algorithm.md` and `docs/dp_optimization_parameters.md` both
    used to read as if shrinking `soc_step_percent` could only improve the
    answer, and the second recommends 0.25 % on that basis. It is not true, and
    this is the measurement that says so.
    """

    def test_one_percent_is_worse_than_two_percent_here(self):
        assert _nm_gap(2.0) == pytest.approx(0.0, abs=1e-9)
        assert _nm_gap(1.0) == pytest.approx(0.0092, abs=1e-3)
        assert _nm_gap(1.0) > _nm_gap(2.0) + 1e-9

    def test_and_finer_steps_recover(self):
        for step in (0.5, 0.25, 0.1):
            assert _nm_gap(step) == pytest.approx(0.0, abs=1e-9), step


# ---------------------------------------------------------------------------
# The claim guard
# ---------------------------------------------------------------------------
#
# The previous version of this guard passed any line containing "not ",
# "never ", "no " or "is not" -- which is most English prose, and certainly any
# sentence long enough to make an optimality claim. It also only looked at two
# phrases in four files. It was inert: it did not object to "globally optimal",
# "provably optimal", "Guaranteed minimum cost", "finds the cheapest schedule"
# or "exact for the discretized model" appearing anywhere.
#
# This is a real guard: an explicit list of forbidden claims, matched
# whole-phrase and case-insensitively over every source and document, with an
# explicit allowlist of the exact places a phrase may legitimately appear
# (a sentence that DENIES the claim, or the description of the exact
# Pareto-label alternative that is not what this solver does).

FORBIDDEN_CLAIMS = (
    r"globally\s+optimal",
    r"provably\s+optimal",
    r"guaranteed\s+minimum\s+cost",
    r"finds\s+the\s+cheapest\s+schedule",
    r"exact\s+for\s+(?:its|the)\s+discretized\s+model",
    # "optimal schedule" as a LABEL -- a Markdown heading, a diagram box, a
    # table cell: the phrase standing on its own as the name of the thing the
    # solver produces. Anchored to the start of the line through non-word
    # characters only (`#`, `|`, `-`, whitespace, box-drawing glyphs), so
    # prose that merely mentions the phrase mid-sentence is untouched --
    # including `dp_optimizer.py`'s own `Not "the optimal schedule": ...`,
    # which denies the claim and must keep working without an allowance.
    r"^[\s\W]*optimal\s+schedule\b",
)

# (path suffix, exact substring that must appear in the offending line).
# Narrow on purpose: an allowlist entry is a licence for ONE sentence in ONE
# file, and `test_every_allowance_is_still_needed` fails when one goes stale.
ALLOWED = (
    (
        "appdaemon/apps/battery_optimizer_lib/dp_optimizer.py",
        'This solver is therefore NOT "exact for its discretized model"',
    ),
    (
        "docs/scheduling-algorithm.md",
        "is exact for the discretized model, because that",
    ),
    (
        "CLAUDE.md",
        'Do not write "exact for its discretized model" anywhere',
    ),
)


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parent.parent


def _tracked_paths(root):
    """Paths git tracks, or None when git is unavailable.

    The guard is about what the repository CLAIMS, so an untracked scratch
    file, a review brief, or a local note that happens to quote a forbidden
    phrase must not fail the suite. Falling back to the glob keeps the guard
    alive in a source export without git metadata.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {root / p for p in out.decode("utf-8").split("\0") if p}


def _scanned_files(root=None):
    """Every tracked `.py`/`.md` outside `tests/`, or a glob when git is absent.

    The tracked set is the SOURCE of the list, not a filter over globs. It was
    a filter, which meant the four glob patterns still decided what could ever
    be scanned: `AGENTS.md`, `scripts/smoke_config.py` and 66 other tracked
    files were never opened, and a forbidden claim planted in any of them
    passed the guard silently. Narrowing a list of candidates by "is it
    tracked?" answers a different question from "what does this repository
    say?".

    `tests/` is excluded because this file itself has to spell the forbidden
    phrases out to test for them.
    """
    root = _repo_root() if root is None else root
    tracked = _tracked_paths(root)
    if tracked is None:
        # No git metadata (a source export). Fall back to the globs; partial
        # cover beats none, and `test_the_guard_scans_every_tracked_source`
        # skips rather than lying about it.
        paths = []
        for pattern in ("appdaemon/**/*.py", "docs/**/*.md"):
            paths.extend(
                p for p in root.glob(pattern) if "__pycache__" not in p.parts
            )
        paths.append(root / "README.md")
        paths.append(root / "CLAUDE.md")
        return sorted({p for p in paths if p.is_file()})
    return sorted(
        p
        for p in tracked
        if p.suffix in (".py", ".md")
        and not p.relative_to(root).as_posix().startswith("tests/")
        and "__pycache__" not in p.parts
        and p.is_file()
    )


def _is_allowed(root, path, line) -> bool:
    relative = path.relative_to(root).as_posix()
    return any(
        relative == suffix and permitted in line for suffix, permitted in ALLOWED
    )


def _offences(root, paths):
    """Every line in *paths* that makes a forbidden claim without an allowance."""
    found = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            for pattern in FORBIDDEN_CLAIMS:
                if re.search(pattern, line, re.IGNORECASE) and not _is_allowed(
                    root, path, line
                ):
                    found.append(
                        f"{path.relative_to(root).as_posix()}:{number}: "
                        f"{line.strip()}"
                    )
    return found


class TestNoSourceClaimsAnOptimalityItDoesNotHave:
    """The solver is an approximation; nothing may say otherwise."""

    def test_no_forbidden_claim_survives_anywhere(self):
        root = _repo_root()
        offences = _offences(root, _scanned_files())
        assert not offences, (
            "these lines claim an optimality the bucket merge does not have:\n"
            + "\n".join(offences)
        )

    def test_the_guard_actually_matches_the_claims_it_lists(self):
        """A typo in a pattern would make the guard silently vacuous."""
        samples = {
            r"globally\s+optimal": "the plan is globally optimal",
            r"provably\s+optimal": "Provably optimal over the horizon",
            r"guaranteed\s+minimum\s+cost": "Guaranteed minimum cost",
            r"finds\s+the\s+cheapest\s+schedule": "it finds the cheapest schedule",
            r"exact\s+for\s+(?:its|the)\s+discretized\s+model": (
                "exact for the discretized model"
            ),
            r"^[\s\W]*optimal\s+schedule\b": "## Optimal Schedule",
        }
        assert set(samples) == set(FORBIDDEN_CLAIMS)
        for pattern, sample in samples.items():
            assert re.search(pattern, sample, re.IGNORECASE), pattern

    def test_the_label_pattern_leaves_prose_alone(self):
        """The anchored pattern must not fire on a sentence that uses the
        phrase, only on one that titles something with it."""
        for label in (
            "## Optimal Schedule",
            "        Optimal Schedule (CHARGE / HOLD / DISCHARGE per slot)",
            "| Optimal schedule | per slot |",
            "  - Optimal Schedule",
        ):
            assert re.search(
                r"^[\s\W]*optimal\s+schedule\b", label, re.IGNORECASE
            ), label
        for prose in (
            'Not "the optimal schedule": the merge keeps one path per bucket',
            "The result is close to the optimal schedule but is not it.",
            "def find_optimal_schedule(self):",
            "This is no more the optimal schedule than a greedy pass is.",
        ):
            assert not re.search(
                r"^[\s\W]*optimal\s+schedule\b", prose, re.IGNORECASE
            ), prose

    def test_the_guard_covers_the_files_that_have_made_the_claim(self):
        root = _repo_root()
        scanned = {p.relative_to(root).as_posix() for p in _scanned_files()}
        for required in (
            "appdaemon/apps/battery_optimizer.py",
            "appdaemon/apps/battery_optimizer_lib/dp_optimizer.py",
            "appdaemon/apps/battery_optimizer_lib/schedule_formatter.py",
            "docs/scheduling-algorithm.md",
            "docs/dp_optimization_parameters.md",
            "README.md",
            "CLAUDE.md",
            # Tracked, and outside every glob the guard used to use. AGENTS.md
            # is the file an agent reads before touching the solver; a claim
            # planted there was invisible to this guard.
            "AGENTS.md",
            "scripts/smoke_config.py",
            "scripts/README.md",
        ):
            assert required in scanned, required

    def test_the_guard_scans_every_tracked_source_and_document(self):
        """No tracked .py/.md outside tests/ may fall outside the scan.

        The previous version globbed four patterns and then FILTERED by the
        tracked set, so "tracked" only ever narrowed the globs -- 68 of the 101
        tracked files were never opened.
        """
        root = _repo_root()
        tracked = _tracked_paths(root)
        if tracked is None:
            pytest.skip("git metadata unavailable")
        expected = {
            p.relative_to(root).as_posix()
            for p in tracked
            if p.suffix in (".py", ".md")
            and not p.relative_to(root).as_posix().startswith("tests/")
        }
        scanned = {p.relative_to(root).as_posix() for p in _scanned_files()}
        assert expected - scanned == set(), sorted(expected - scanned)

    def test_a_claim_planted_in_a_tracked_file_outside_the_globs_trips(
        self, tmp_path
    ):
        """The proof, not the argument.

        A throwaway git checkout containing exactly one file -- ``AGENTS.md``,
        which matches none of the old globs -- with one forbidden sentence in
        it. The guard must report it.
        """
        import subprocess

        def git(*args):
            subprocess.run(
                ["git", *args], cwd=tmp_path, check=True, capture_output=True
            )

        try:
            git("init", "-q")
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("git unavailable")
        git("config", "user.email", "guard@example.invalid")
        git("config", "user.name", "guard")
        (tmp_path / "AGENTS.md").write_text(
            "# Agent notes\n\nThe DP is globally optimal.\n", encoding="utf-8"
        )
        # A second tracked file with no claim, so a scan that returned
        # everything unconditionally would still have to pick the right line.
        (tmp_path / "innocent.md").write_text(
            "# Notes\n\nThe DP is an approximation.\n", encoding="utf-8"
        )
        git("add", "AGENTS.md", "innocent.md")
        git("commit", "-qm", "plant")

        scanned = _scanned_files(tmp_path)
        assert tmp_path / "AGENTS.md" in scanned, (
            "a tracked file outside the old globs is never opened"
        )
        offences = _offences(tmp_path, scanned)
        assert len(offences) == 1, offences
        assert offences[0].startswith("AGENTS.md:3:")

    def test_every_allowance_is_still_needed(self):
        """A stale allowlist entry is a hole nobody is watching."""
        root = _repo_root()
        for relative, permitted in ALLOWED:
            text = (root / relative).read_text(encoding="utf-8")
            assert permitted in text, (
                f"{relative} no longer contains the sentence this allowance was "
                f"written for; delete the allowance: {permitted!r}"
            )
