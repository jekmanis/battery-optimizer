"""``get_charge_rate_for_soc`` is memoized, and the memo has one owner.

The DP evaluates the charge rate per candidate transition, and the thermal
refinement evaluates it at every state SOC the solve visited against every
temperature profile it compares. One live 130-slot horizon at
``soc_step_percent: 0.25`` asked the question 8.4 million times; recomputing a
plausibility filter over 50 observations plus a median every time made the
startup optimize take 206 s.

The answer depends on its arguments ONLY through the SOC bucket and the
temperature bucket, so it is memoized on that pair. These tests pin the two
halves of that: the memo must not change any answer, and it must be dropped
whenever the observations behind it change.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "appdaemon" / "apps"))

import pytest

from battery_optimizer_lib.learning_engine import BatteryLearningEngine


def _engine(**kwargs):
    kwargs.setdefault("battery_capacity_kwh", 10.0)
    kwargs.setdefault("nominal_charge_rate_kw", 4.0)
    kwargs.setdefault("nominal_efficiency", 0.95)
    kwargs.setdefault("log_func", lambda *a, **k: None)
    return BatteryLearningEngine(**kwargs)


class TestTheMemoNeverChangesAnAnswer:
    def test_repeated_lookups_agree_with_a_fresh_engine(self):
        engine = _engine()
        for _ in range(5):
            engine.record_charging(
                soc_start=40.0, soc_end=50.0, duration_minutes=15.0,
                energy_from_grid_kwh=1.2, battery_temp=22.0,
            )
        first = engine.get_charge_rate_for_soc(45.0, 22.0)
        repeated = [engine.get_charge_rate_for_soc(45.0, 22.0) for _ in range(3)]
        assert repeated == [first] * 3

        # Same observations, never asked twice: the uncached path.
        fresh = _engine()
        for _ in range(5):
            fresh.record_charging(
                soc_start=40.0, soc_end=50.0, duration_minutes=15.0,
                energy_from_grid_kwh=1.2, battery_temp=22.0,
            )
        assert fresh.get_charge_rate_for_soc(45.0, 22.0) == first

    def test_every_bucket_pair_keeps_its_own_answer(self):
        engine = _engine()
        for _ in range(5):
            engine.record_charging(
                soc_start=10.0, soc_end=20.0, duration_minutes=15.0,
                energy_from_grid_kwh=2.4, battery_temp=22.0,
            )
        for _ in range(5):
            engine.record_charging(
                soc_start=80.0, soc_end=85.0, duration_minutes=15.0,
                energy_from_grid_kwh=0.6, battery_temp=22.0,
            )
        low = engine.get_charge_rate_for_soc(15.0, 22.0)
        high = engine.get_charge_rate_for_soc(82.0, 22.0)
        assert low != high
        # Re-asking in the other order must not swap them.
        assert engine.get_charge_rate_for_soc(82.0, 22.0) == high
        assert engine.get_charge_rate_for_soc(15.0, 22.0) == low

    def test_temperature_none_is_its_own_key(self):
        engine = _engine()
        for _ in range(5):
            engine.record_charging(
                soc_start=40.0, soc_end=50.0, duration_minutes=15.0,
                energy_from_grid_kwh=1.2, battery_temp=22.0,
            )
        with_temp = engine.get_charge_rate_for_soc(45.0, 22.0)
        without = engine.get_charge_rate_for_soc(45.0, None)
        assert engine.get_charge_rate_for_soc(45.0, 22.0) == with_temp
        assert engine.get_charge_rate_for_soc(45.0, None) == without


class TestTheMemoIsDroppedWhenItsInputsChange:
    def test_a_new_observation_changes_the_answer(self):
        engine = _engine()
        nominal = engine.get_charge_rate_for_soc(45.0, 22.0)
        for _ in range(5):
            engine.record_charging(
                soc_start=40.0, soc_end=50.0, duration_minutes=15.0,
                energy_from_grid_kwh=1.2, battery_temp=22.0,
            )
        learned = engine.get_charge_rate_for_soc(45.0, 22.0)
        assert learned != nominal, (
            "the memo survived record_charging -- a learned rate would never "
            "reach the planner"
        )

    def test_loading_a_file_replaces_the_answers(self):
        source = _engine()
        for _ in range(5):
            source.record_charging(
                soc_start=40.0, soc_end=50.0, duration_minutes=15.0,
                energy_from_grid_kwh=1.2, battery_temp=22.0,
            )
        payload = source.save_to_json()

        target = _engine()
        stale = target.get_charge_rate_for_soc(45.0, 22.0)   # nominal, memoized
        assert target.load_from_json(payload)
        assert target.get_charge_rate_for_soc(45.0, 22.0) == pytest.approx(
            source.get_charge_rate_for_soc(45.0, 22.0)
        )
        assert target.get_charge_rate_for_soc(45.0, 22.0) != stale

    def test_sanitize_drops_the_answers_it_invalidates(self):
        engine = _engine()
        for _ in range(5):
            engine.record_charging(
                soc_start=40.0, soc_end=50.0, duration_minutes=15.0,
                energy_from_grid_kwh=1.2, battery_temp=22.0,
            )
        learned = engine.get_charge_rate_for_soc(45.0, 22.0)
        # Poison the bucket behind the memoized answer, then sanitize.
        engine.stats.charge_rates_by_soc["25-50"] = [99999.0] * 5
        engine.stats.charge_rates_by_soc_temp["25-50"] = {">20": [99999.0] * 5}
        engine.sanitize_stats()
        after = engine.get_charge_rate_for_soc(45.0, 22.0)
        assert after != learned
        assert after == pytest.approx(engine._nominal_input_dc_rate("25-50"))

    def test_explicit_invalidation_is_available_for_direct_mutation(self):
        """Reaching into ``stats`` is allowed; saying so is required."""
        engine = _engine()
        nominal = engine.get_charge_rate_for_soc(45.0, 22.0)
        engine.stats.charge_rates_by_soc_temp["25-50"] = {">20": [2.85] * 5}
        engine.invalidate_rate_cache()
        assert engine.get_charge_rate_for_soc(45.0, 22.0) != nominal


class TestTemperatureBucketsAreUnchanged:
    @pytest.mark.parametrize(
        "temp,expected",
        [(-3.0, "<5"), (4.999, "<5"), (5.0, "5-10"), (12.0, "10-15"),
         (17.5, "15-20"), (20.0, ">20"), (33.1, ">20")],
    )
    def test_default_boundaries(self, temp, expected):
        engine = _engine()
        assert engine._get_temp_range(temp) == expected
        # Second call comes from the memo and must not differ.
        assert engine._get_temp_range(temp) == expected

    def test_custom_boundaries(self):
        engine = _engine(temp_ranges=[0, 15, 30])
        assert engine._get_temp_range(-1.0) == "<0"
        assert engine._get_temp_range(7.0) == "0-15"
        assert engine._get_temp_range(20.0) == "15-30"
        assert engine._get_temp_range(40.0) == ">30"

    def test_no_boundaries(self):
        engine = _engine(temp_ranges=[])
        assert engine._get_temp_range(20.0) == "all"

    def test_replacing_the_boundaries_rebuilds_the_labels(self):
        engine = _engine()
        assert engine._get_temp_range(12.0) == "10-15"
        engine.temp_ranges = [0, 15, 30]
        engine._temp_range_by_temp.clear()  # the per-value memo is separate
        assert engine._get_temp_range(12.0) == "0-15"
