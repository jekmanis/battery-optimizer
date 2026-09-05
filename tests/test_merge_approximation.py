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


def _cost_of(cfg, modes) -> float:
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
    energy = INITIAL_SOC / 100.0 * cfg.battery_capacity
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

        One 1 % step is 0.1 kWh; the marginal value of a stored kWh in the
        expensive slot is 1.00 EUR/kWh. 0.005 EUR is comfortably inside 0.1.
        """
        cfg = _config()
        best, _combo = _exhaustive_best(cfg)
        solver = _cost_of(
            cfg, (BatteryMode.DISCHARGE, BatteryMode.DISCHARGE)
        )
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


class TestTheStateMergeIsDocumentedAsApproximate:
    """No source may claim the solver is exact for its discretized model."""

    FILES = [
        "appdaemon/apps/battery_optimizer_lib/dp_optimizer.py",
        "docs/scheduling-algorithm.md",
        "CLAUDE.md",
        "README.md",
    ]

    @pytest.mark.parametrize("relative", FILES)
    def test_no_exactness_claim_survives(self, relative):
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        text = (root / relative).read_text(encoding="utf-8")
        lowered = text.lower()
        for claim in (
            "exact for its discretized model",
            "globally optimal",
        ):
            for line in lowered.splitlines():
                if claim in line:
                    # Only a line that explicitly denies the claim may contain it.
                    assert any(
                        marker in line
                        for marker in ("not ", "never ", "no ", "is not")
                    ), f"{relative}: {line.strip()}"
