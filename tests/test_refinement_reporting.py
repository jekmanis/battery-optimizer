"""Group B: refinement and reporting residuals.

B1  ``_profiles_agree`` criterion 2 compared charge rates at the DP's GRID SOCs
    while ``_run_dp`` evaluates them at each path's exact ``dp_energy[idx]``. A
    curve that differs between two temperatures only at an energy strictly
    between grid points therefore read as "the same plan" at exactly the states
    the solve visits.

B2  ``ScheduleEntry.energy_limited`` was set from the final walk and never
    cleared, so a slot the DP flagged partial on its conservative planning
    rates kept the flag -- and the "(until depleted)" prose -- even when the
    final walk serves it in full.

B3  ``rate_refinement_branch`` / ``_shortfall_kwh`` / ``_degraded`` had no
    consumer at all.
"""

from __future__ import annotations

import datetime
import inspect

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import (
    BatteryMode,
    DPOptimizer,
    DPOptimizerConfig,
    PricePoint,
)
from battery_optimizer_lib.plan_validation import replay_plan


BASE = datetime.datetime(2026, 3, 2, 0, 0)
CAPACITY = 10.0


def _slots(n):
    return [BASE + datetime.timedelta(minutes=15 * i) for i in range(n)]


def _config(**kwargs) -> DPOptimizerConfig:
    base = dict(
        battery_capacity=CAPACITY,
        min_soc=10.0,
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
    base.update(kwargs)
    return DPOptimizerConfig(**base)


def _threshold_rate(soc, temp=None):
    """1 kW below 10 C, 4 kW at or above it."""
    if temp is None:
        return 4.0
    return 1.0 if temp < 10.0 else 4.0


def _late_load(dt):
    return 4.0 if dt == BASE + datetime.timedelta(minutes=30) else 0.0


def _oscillating_plan():
    """The refinement case that falls back to the conservative solve.

    The plan is built on the idle profile (0 C throughout, so 1 kW), while the
    walk of that plan warms the pack past the threshold and charges at 4 kW.
    """
    opt = DPOptimizer(
        config=_config(),
        load_predictor=_late_load,
        charge_rate_predictor=_threshold_rate,
        temp_after_charge_predictor=lambda t, m: t + m,
        temp_after_idle_predictor=lambda t, m: t,
        pv_predictor=lambda dt: 0.0,
    )
    result = opt.optimize(
        prices=[
            PricePoint(time=t, price=p)
            for t, p in zip(_slots(3), [0.60, 0.01, 1.00])
        ],
        current_slot=BASE,
        current_soc=10.0,
        current_temp=0.0,
    )
    assert result.rate_refinement_fallback, (
        "this scenario no longer reaches the conservative fallback"
    )
    return opt, result


# ---------------------------------------------------------------------------
# B1
# ---------------------------------------------------------------------------


class TestProfileAgreementUsesTheStatesTheSolveVisits:
    def test_criterion_two_is_evaluated_at_the_exact_path_energies(self, monkeypatch):
        """The SOCs handed to ``_profiles_agree`` are the ones ``_run_dp`` used.

        The DP's grid SOCs on this pack are exactly 10, 11, 12 ... %; a path
        that lands between them is what every transition is actually computed
        from. Comparing rates only at the grid was comparing a different set of
        states from the one the solve visits.
        """
        seen = []
        real = DPOptimizer._profiles_agree

        def spy(self, a, b, state_socs=None):
            if state_socs is not None:
                seen.append(list(state_socs))
            return real(self, a, b, state_socs)

        monkeypatch.setattr(DPOptimizer, "_profiles_agree", spy)
        _opt, _result = _oscillating_plan()

        assert seen, "_profiles_agree was never given a state set"
        socs = seen[0]
        assert socs, "criterion 2 has no states to evaluate at"
        grid = {round(10.0 + i, 6) for i in range(91)}
        off_grid = [s for s in socs if round(s, 6) not in grid]
        assert off_grid, (
            "every SOC offered to criterion 2 sits on the grid, so it is still "
            "the grid and not the energies the paths actually reach"
        )

    def test_the_visited_set_is_a_subset_of_what_the_dp_could_reach(self):
        opt, _result = _oscillating_plan()
        visited = opt.last_visited_state_socs()
        assert visited
        assert all(10.0 - 1e-9 <= s <= 100.0 + 1e-9 for s in visited)


# ---------------------------------------------------------------------------
# B2
# ---------------------------------------------------------------------------


class TestEnergyLimitedIsSetInBothDirections:
    def test_a_slot_the_final_walk_serves_in_full_is_not_flagged(self):
        _opt, result = _oscillating_plan()
        discharge_slot = _slots(3)[2]
        entry = result.schedule[discharge_slot]
        assert entry.mode == BatteryMode.DISCHARGE

        # The published trajectory IS the final walk. It shows the slot serving
        # the whole 4 kW x 0.25 h and ending above min SOC, so nothing was
        # truncated -- the flag came from the conservative planning pass, where
        # the pack held less.
        start_soc, end_soc = result.soc_trajectory[discharge_slot]
        served_kwh = (start_soc - end_soc) / 100.0 * CAPACITY
        assert served_kwh == pytest.approx(4.0 * 0.25, abs=1e-9), (
            "this scenario no longer serves the discharge slot in full"
        )
        assert end_soc > 10.0 + 1e-9
        assert entry.energy_limited is False
        assert "until depleted" not in entry.reason

    def test_a_slot_that_really_runs_dry_still_declares_it(self):
        """Clearing the flag must not stop it being set."""
        opt = DPOptimizer(
            config=_config(min_soc=10.0),
            load_predictor=lambda dt: 4.0,
            charge_rate_predictor=lambda soc, temp: 0.0,
            temp_after_charge_predictor=lambda t, m: t,
            temp_after_idle_predictor=lambda t, m: t,
            pv_predictor=lambda dt: 0.0,
        )
        result = opt.optimize(
            prices=[PricePoint(time=t, price=1.00) for t in _slots(3)],
            current_slot=BASE,
            current_soc=12.0,
            current_temp=20.0,
        )
        limited = [e for e in result.schedule.values() if e.energy_limited]
        assert limited, "a pack that runs dry must declare the slot it runs dry in"
        assert all("until depleted" in e.reason for e in limited)


# ---------------------------------------------------------------------------
# B3
# ---------------------------------------------------------------------------


class TestTheRefinementBranchIsReported:
    def test_the_optimizer_records_the_branch_from_the_result(self):
        """A real plan, through ``find_optimal_schedule``, reaches the sensor."""
        from tests.test_plan_publication_and_current_slot import PlannerApp

        app = PlannerApp(
            battery_capacity=CAPACITY,
            charge_rate=4.0,
            discharge_rate=4.0,
            efficiency=1.0,
            slot_minutes=15,
            battery_temp=0.0,
            decision_log_level=0,
            now=BASE,
        )
        app.load_by_slot = {_slots(3)[2]: 4.0}
        app.learning_engine.get_charge_rate_for_soc = _threshold_rate
        app.learning_engine.predict_temp_after_duration = lambda t, m, **kw: t + m
        app.learning_engine.predict_temp_after_idle = lambda t, m, **kw: t
        app.find_optimal_schedule(
            [
                PricePoint(time=t, price=p)
                for t, p in zip(_slots(3), [0.60, 0.01, 1.00])
            ],
            0,
            current_soc=10.0,
        )
        diag = app._rate_refinement_diagnostics()
        assert diag["branch"] in (
            "single_solve",
            "converged",
            "conservative_fallback",
            "degraded",
        )
        assert diag["degraded"] is False
        assert diag["shortfall_kwh"] == pytest.approx(0.0)
        assert diag["passes"] >= 1

    def test_the_schedule_sensor_publishes_the_refinement(self):
        source = inspect.getsource(bo.BatteryOptimizer._update_schedule_sensor)
        assert "_rate_refinement_diagnostics()" in source
        assert '"rate_refinement"' in source

    def test_diagnostics_are_empty_before_any_plan(self):
        app = object.__new__(bo.BatteryOptimizer)
        assert bo.BatteryOptimizer._rate_refinement_diagnostics(app) == {}
