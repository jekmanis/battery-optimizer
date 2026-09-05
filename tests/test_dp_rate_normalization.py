"""The DP has ONE normalization of the injected charge-rate predictor.

``DPOptimizer._rate_uncached`` is it: ``max(0.0, float(pred(soc, temp) or 0.0))``
-- a raised ``None``, a negative rate and a ``0`` all become ``0.0``, and an
``int`` becomes a ``float``.

``_rates_over`` spells that expression out again inside its list comprehension
instead of calling ``_rate_uncached`` per element, because it is the inner loop
of ``_profiles_agree`` and runs millions of times in one live solve (measured:
8.5 M evaluations on the 130-slot, 0.25 %-step production horizon, which is why
the startup optimize took 206 s). A saved call frame is not worth a second
normalization drifting from the first, so the two are pinned here -- both by
value, on the awkward inputs, and by source, so that "fixing" one and not the
other fails.
"""

import datetime
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "appdaemon" / "apps"))

import pytest

from battery_optimizer_lib.dp_optimizer import DPOptimizer, DPOptimizerConfig


def _optimizer(predictor):
    return DPOptimizer(
        config=DPOptimizerConfig(
            battery_capacity=10.0,
            min_soc=10.0,
            max_soc=100.0,
            efficiency=0.95,
            discharge_rate=5.0,
            slot_minutes=15,
            soc_step_percent=1.0,
        ),
        load_predictor=lambda dt: 0.5,
        charge_rate_predictor=predictor,
        temp_after_charge_predictor=lambda *a, **k: 20.0,
        temp_after_idle_predictor=lambda *a, **k: 20.0,
    )


AWKWARD = [None, 0, 0.0, -0.0, -1.0, -1e-12, 1, 4, 4.5, 1e-15]


class TestOneNormalization:
    def test_rates_over_matches_rate_uncached_value_for_value(self):
        """Every awkward predictor answer normalizes the same way both ways."""
        for raw in AWKWARD:
            opt = _optimizer(lambda soc, temp, _r=raw: _r)
            socs = [10.0, 42.5, 99.9]
            one_by_one = [opt._rate_uncached(soc, 25.0) for soc in socs]
            vectorized = opt._rates_over(socs, 25.0)
            assert vectorized == one_by_one, f"raw={raw!r}"
            for value in vectorized:
                assert isinstance(value, float)
                assert value >= 0.0

    def test_rates_over_passes_soc_and_temp_through_unchanged(self):
        seen = []

        def predictor(soc, temp=None):
            seen.append((soc, temp))
            return soc / 10.0

        opt = _optimizer(predictor)
        socs = [11.0, 12.25, 13.5]
        assert opt._rates_over(socs, 31.4) == [1.1, 1.225, 1.35]
        assert seen == [(11.0, 31.4), (12.25, 31.4), (13.5, 31.4)]

    def test_rates_over_preserves_order(self):
        opt = _optimizer(lambda soc, temp=None: soc)
        socs = [90.0, 10.0, 50.0]
        assert opt._rates_over(socs, None) == [90.0, 10.0, 50.0]

    def test_the_two_spellings_are_the_same_expression(self):
        """Source-level guard: change one, change the other."""
        pattern = re.compile(
            r"max\(\s*0\.0\s*,\s*float\(\s*[\w.]+\(\s*soc\s*,\s*temp\s*\)"
            r"\s*or\s*0\.0\s*\)\s*\)"
        )
        for method in (DPOptimizer._rate_uncached, DPOptimizer._rates_over):
            source = inspect.getsource(method)
            # The docstring quotes the expression too; look at code only.
            code = source.split('"""')[-1]
            assert pattern.search(code), (
                f"{method.__name__} no longer normalizes the predictor with "
                "max(0.0, float(<predictor>(soc, temp) or 0.0)) -- the DP must "
                "keep exactly one normalization; see this module's docstring."
            )


class TestProfilesAgreeUsesThatNormalization:
    def test_a_negative_and_a_zero_rate_are_the_same_profile(self):
        """Normalization is what makes these two temperatures agree.

        Comparing the raw predictor answers would call -1.0 and 0.0 different
        and restart the refinement for a difference no transition can see.
        """
        def predictor(soc, temp=None):
            return -1.0 if temp is not None and temp > 25.0 else 0.0

        opt = _optimizer(predictor)
        assert opt._profiles_agree([30.0], [20.0], [10.0, 55.0, 99.0]) is True

    def test_a_real_rate_difference_is_still_a_disagreement(self):
        def predictor(soc, temp=None):
            return 4.0 if temp is not None and temp > 25.0 else 1.0

        opt = _optimizer(predictor)
        assert opt._profiles_agree([30.0], [20.0], [10.0, 55.0, 99.0]) is False
