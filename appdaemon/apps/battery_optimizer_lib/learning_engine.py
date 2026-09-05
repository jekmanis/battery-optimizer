"""
Self-learning battery performance tracking engine.

Learns actual charge rates, SOC-dependent behavior, temperature effects,
and round-trip efficiency from observed battery performance.

Charge-rate units -- THE contract
---------------------------------

Two different quantities were both called "the charge rate", and the confusion
cost a factor of ``efficiency`` on every learned observation::

    charge_input_dc_kw   DC power at the battery terminal, BEFORE retention.
                         This is what ``charge_rate_kw`` in apps.yaml means and
                         what the inverter is commanded to.
    stored_charge_kw     rate at which STORED energy grows
                         = charge_input_dc_kw * storage_efficiency

An observation is measured on the STORED side: a SOC delta, or the inverter's
"energy to battery" counter, divided by the interval. It is recorded and
persisted in exactly those units -- unchanged from every previous version of
this file, so no migration and no repeated division on reload.

:meth:`BatteryLearningEngine.get_charge_rate_for_soc` is the API boundary and
returns ``charge_input_dc_kw``, because that is what every consumer multiplies
by ``efficiency`` to obtain stored energy (the DP, ``soc_projection``,
``dp_optimizer``, ``cost_tracker``) and what the thermal model wants as
``|P_bat|``. The nominal fallback is already in those units; a learned
stored-side observation is divided by :attr:`storage_efficiency` on the way out.

Consequence, and the reason the defect was invisible: replaying an observation
of 40 % -> 50 % in 15 min on a 10 kWh pack used to predict 48.5 %, not 50 %.
``tests/test_charge_rate_units.py`` is that replay.
"""

import datetime
import json
import math
import statistics
from typing import Dict, List, Optional, Tuple

from .models import LearningStats
from .thermal_model import (
    DEFAULT_COOLING_RATE_PER_MIN,
    DEFAULT_HEATING_C_PER_KWH,
    MAX_BATTERY_TEMP_C,
    step_temperature,
)

# Rolling window of battery temperature observations used to estimate the
# ambient DAILY MINIMUM. 192 samples = 48 h at one sample per 15 min. The old
# 48-sample window covered only ~12 h once observations became event-driven,
# which is far too short to contain a diurnal minimum.
TEMP_OBSERVATION_WINDOW = 192

# Minimum number of raw thermal samples before k1/k2 are trusted.
MIN_THERMAL_SAMPLES = 20
MAX_THERMAL_SAMPLES = 300

# Persisted-state version. v7 fixed the |P_bat| units of CHARGING thermal
# samples (terminal power, not stored-energy growth); a file below it has its
# thermal samples and fitted coefficients dropped once on load. Charge-rate
# observations are untouched by that and have never changed representation.
THERMAL_UNITS_VERSION = 7

# ---------------------------------------------------------------------------
# Learned-rate plausibility bounds — THE one sanity bound
# ---------------------------------------------------------------------------
# Every consumer of a learned battery power (the DP through
# ``DPOptimizer._rate_for``, the expected-SOC trajectory through
# ``soc_projection._effective_charge_rate``, the deviation detector through
# ``_project_charge_completion`` / ``_calculate_extra_charge_slots``) reaches it
# via :meth:`BatteryLearningEngine.get_charge_rate_for_soc`. The bound therefore
# lives HERE and nowhere else — three ad-hoc clamps in three consumers is
# exactly the drift CLAUDE.md's "one model" rule exists to prevent.
#
# Why 2x nominal: the reference installation is configured at 4.5 kW nominal and
# the observation history contains a hard physical ceiling around 6.8 kW
# (1.5x nominal) — the inverter's warm-battery rate. 2x leaves headroom for a
# faster pack than configured while rejecting everything above what a 15-minute
# slot can physically deliver. A tighter 1.5x would clip the genuine 6.77-6.82 kW
# cluster.
DEFAULT_MAX_RATE_FACTOR = 2.0

# Quantization step of the energy source an observation is measured from. The
# inverter energy counters move in 0.1 kWh; the SOC sensor's 1% is coarser
# (0.143 kWh) but its samples are gated the same way.
DEFAULT_COUNTER_RESOLUTION_KWH = 0.1

# Absolute wall-time floor for an observation, used ONLY when the measured
# energy is too small to resolve a rate on its own (see
# ``_observation_is_resolvable``). 0.25 min = 15 s.
#
# It was 1.0 min, and that was wrong in a way the 2x rate bound was explicitly
# tuned against: ``cost_tracker`` re-stamps ``_last_sig_soc_time`` after EVERY
# accepted event, so a genuine interval is however long it takes the counter to
# advance one 0.1 kWh tick — 0.1/P hours, i.e. under a minute for any P above
# 6 kW. A 1-minute floor therefore rejected exactly the 6.77-6.82 kW warm-pack
# cluster the bound exists to keep, while a 0.1 kWh / 44 ms sample (~9000 kW) is
# caught by ``is_plausible_rate`` regardless of any wall-time rule.
MIN_OBSERVATION_MINUTES = 0.25

# Absolute ceiling for the self-heating coefficient k2 (C per kWh through the
# pack). Measured on the reference pack: 21.9C -> 25.8C while storing 1.716 kWh
# over one 15-minute slot = 2.27 C/kWh, so the previous 2.0 ceiling was itself
# binding on real data. 3.0 keeps a genuinely hot pack representable while still
# rejecting a fit driven by corrupted power figures.
MAX_HEATING_C_PER_KWH = 3.0

# Bounds for the relaxation coefficient k1 (per minute).
MIN_COOLING_RATE_PER_MIN = 0.001
MAX_COOLING_RATE_PER_MIN = 0.1


def _clamp_k1(k1: float) -> float:
    """Clamp the relaxation coefficient to its physical range."""
    return min(MAX_COOLING_RATE_PER_MIN, max(MIN_COOLING_RATE_PER_MIN, k1))


def _clamp_k2(k2: float) -> float:
    """Clamp the self-heating coefficient to its physical range."""
    return min(MAX_HEATING_C_PER_KWH, max(0.0, k2))


def thermal_coeffs_are_sane(coeffs: Optional[Dict]) -> bool:
    """Whether a persisted ``thermal_coeffs`` dict is inside the model's bounds.

    A fit produced from corrupted power figures lands outside these bounds (or
    carries a non-finite value); such a fit is discarded on load and the engine
    falls back to the bootstrap/default chain rather than feeding the DP a
    temperature model it can never satisfy.
    """
    if not coeffs:
        return False
    try:
        k1 = float(coeffs.get("k1"))
        k2 = float(coeffs.get("k2"))
        n = float(coeffs.get("n", 0))
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(k1) and math.isfinite(k2) and math.isfinite(n)):
        return False
    if not (MIN_COOLING_RATE_PER_MIN <= k1 <= MAX_COOLING_RATE_PER_MIN):
        return False
    return 0.0 <= k2 <= MAX_HEATING_C_PER_KWH


class BatteryLearningEngine:
    """
    Self-learning engine that adapts predictions based on actual battery performance.

    Learns:
    - Actual charge rate (may differ from configured)
    - SOC-dependent charge rate curve (batteries charge slower when full)
    - Round-trip efficiency
    - Provides confidence intervals for predictions
    """

    def __init__(
        self,
        battery_capacity_kwh: float = 14.3,
        nominal_charge_rate_kw: float = 4.5,
        nominal_efficiency: float = 0.85,
        min_soc: float = 10.0,
        max_soc: float = 100.0,
        log_func=None,
        temp_ranges: Optional[List[int]] = None,
        default_cooling_rate_per_min: float = DEFAULT_COOLING_RATE_PER_MIN,
        default_heating_c_per_kwh: float = DEFAULT_HEATING_C_PER_KWH,
        nominal_discharge_rate_kw: Optional[float] = None,
        nominal_export_rate_kw: Optional[float] = None,
        max_rate_factor: float = DEFAULT_MAX_RATE_FACTOR,
        min_observation_minutes: float = MIN_OBSERVATION_MINUTES,
        counter_resolution_kwh: float = DEFAULT_COUNTER_RESOLUTION_KWH,
    ):
        self.default_cooling_rate_per_min = default_cooling_rate_per_min
        self.default_heating_c_per_kwh = default_heating_c_per_kwh
        self.nominal_discharge_rate = nominal_discharge_rate_kw
        self.nominal_export_rate = nominal_export_rate_kw
        self.max_rate_factor = max(1.0, float(max_rate_factor))
        self.min_observation_minutes = max(0.0, float(min_observation_minutes))
        self.counter_resolution_kwh = max(0.0, float(counter_resolution_kwh))
        self._rejected_observations = 0
        self._thermal_samples_since_calibration = 0
        self.battery_capacity = battery_capacity_kwh
        self.nominal_charge_rate = nominal_charge_rate_kw
        self.nominal_efficiency = nominal_efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.log = log_func or print

        # Temperature range boundaries for bucketing (default: <5, 5-10, 10-15, 15-20, >20)
        self.temp_ranges = temp_ranges if temp_ranges is not None else [5, 10, 15, 20]

        # Learning data
        self.stats = LearningStats()

        # Exponential moving average factor (0.1 = slow learning, stable)
        self.ema_alpha = 0.1

        # Learned parameters (start with nominals)
        self.learned_efficiency = nominal_efficiency

        # SOC-dependent charge rate multipliers
        # NOTE: Default assumes flat curve - learning will discover actual behavior
        # Some batteries (like user's LiPO) charge SLOWER at low SOC and FASTER at high SOC
        # (opposite of typical CC-CV), due to BMS protection or inverter behavior
        self.soc_charge_multipliers = {
            "0-25": 1.0,    # Will be learned
            "25-50": 1.0,   # Will be learned
            "50-75": 1.0,   # Will be learned
            "75-90": 1.0,   # Will be learned
            "90-100": 1.0,  # Will be learned
        }

    # ------------------------------------------------------------------
    # Unit conversion boundary (see the module docstring)
    # ------------------------------------------------------------------

    @property
    def storage_efficiency(self) -> float:
        """Charge-retention factor used to convert stored <-> input DC rates.

        This is the CONFIGURED ``efficiency``, not :attr:`learned_efficiency`,
        and deliberately so: it is the same constant every consumer multiplies
        back in (``rate * efficiency * duration``), so an observation replayed
        through them reproduces itself exactly. Converting out with one factor
        and back in with another would reintroduce the very mismatch this
        boundary exists to remove.
        """
        eff = float(self.nominal_efficiency or 0.0)
        if not math.isfinite(eff) or eff <= 0.0:
            return 1.0
        return min(1.0, eff)

    def stored_to_input_dc_kw(self, stored_charge_kw: float) -> float:
        """``stored_charge_kw`` -> ``charge_input_dc_kw``."""
        return stored_charge_kw / self.storage_efficiency

    def input_dc_to_stored_kw(self, charge_input_dc_kw: float) -> float:
        """``charge_input_dc_kw`` -> ``stored_charge_kw``."""
        return charge_input_dc_kw * self.storage_efficiency

    # ------------------------------------------------------------------
    # The one plausibility bound on a learned battery power
    # ------------------------------------------------------------------

    @property
    def max_plausible_rate_kw(self) -> float:
        """Upper bound (kW) for any learned charge/discharge rate.

        Derived from the configured nominal rates, so an installation with a
        bigger battery does not need a different constant. Consumers must NOT
        re-derive their own bound — they call
        :meth:`get_charge_rate_for_soc`, which applies this one.

        All THREE configured powers feed the maximum. The export discharge rate
        (``config.effective_export_discharge_rate``) is the one the inverter
        actually runs during a ``discharge_to_grid`` / ``max_export`` slot, and
        it is routinely the largest of the three: judging those genuine samples
        against the (smaller) load-discharge rate would reject the very slots
        the DP plans the export around.

        UNITS: this bound is expressed in ``charge_input_dc_kw`` (the configured
        rates are terminal powers). The ingest filters compare it against
        stored-side observations, which is the LOOSER of the two comparisons --
        deliberately, so that reloading a persisted file never discards history
        an earlier version kept. The strict comparison happens once, at the API
        boundary, in :meth:`_bounded_input_dc_rate`.
        """
        nominal = max(
            float(self.nominal_charge_rate or 0.0),
            float(self.nominal_discharge_rate or 0.0),
            float(self.nominal_export_rate or 0.0),
        )
        if nominal <= 0:
            nominal = 4.5
        return nominal * self.max_rate_factor

    def is_plausible_rate(self, rate_kw: Optional[float]) -> bool:
        """Whether ``rate_kw`` could physically have been produced by the pack."""
        if rate_kw is None:
            return False
        try:
            rate = float(rate_kw)
        except (TypeError, ValueError):
            return False
        return math.isfinite(rate) and 0.0 < rate <= self.max_plausible_rate_kw

    def observation_is_resolvable(
        self, energy_kwh: float, duration_minutes: float
    ) -> Optional[str]:
        """Can this (energy, duration) pair carry a rate at all?

        Returns ``None`` when the sample is resolvable, otherwise the reason it
        is not (ready for :meth:`_reject_observation`).

        The question is quantization, not wall time. A sample resolves a rate
        when EITHER

        * the measured energy spans at least two counter ticks
          (``2 * counter_resolution_kwh``), so the quotient is dominated by real
          energy rather than by one tick of granularity — no matter how short
          the interval; OR
        * the interval is at least ``min_observation_minutes``, the absolute
          floor below which even a multi-tick delta is not worth trusting.

        A pure wall-time floor was wrong here: ``cost_tracker`` re-stamps the
        interval start after every accepted event, so at 6.8 kW the counter
        ticks 0.1 kWh every 53 s and a 1-minute floor rejected the real
        warm-pack cluster. The five-digit production samples (0.1 kWh over
        10-40 ms) fail BOTH conditions, and even if a bigger delta passed this
        gate, ``is_plausible_rate`` still stands behind it.
        """
        try:
            energy = float(energy_kwh)
            duration = float(duration_minutes)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return "non-numeric energy/duration"
        if duration >= self.min_observation_minutes:
            return None
        min_energy = 2.0 * self.counter_resolution_kwh
        if min_energy > 0 and energy >= min_energy:
            return None
        return (
            f"duration {duration:.4f} min < {self.min_observation_minutes:.2f} min "
            f"and energy {energy:.3f} kWh < {min_energy:.3f} kWh "
            f"(2x counter resolution)"
        )

    def clamp_learned_rate(self, rate_kw: Optional[float]) -> Optional[float]:
        """Clamp a learned rate to the plausibility bound (None stays None)."""
        if rate_kw is None:
            return None
        try:
            rate = float(rate_kw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(rate) or rate <= 0:
            return None
        return min(rate, self.max_plausible_rate_kw)

    def _plausible_rates(self, rates: Optional[List[float]]) -> List[float]:
        """Observations that pass the bound, most recent last.

        Applied BEFORE the ``[-10:]`` window so a burst of corrupted samples
        cannot push every usable observation out of the median window — which
        is precisely what happened to the live 0-25%/>20C bucket, where five
        millisecond-duration samples left a median of 14308.71 kW.
        """
        if not rates:
            return []
        return [float(r) for r in rates if self.is_plausible_rate(r)]

    def _reject_observation(self, kind: str, reason: str, detail: str) -> None:
        """Log a dropped learning observation (INFO: it is operationally useful)."""
        self._rejected_observations += 1
        self.log(
            f"Learning: rejected implausible {kind} observation ({reason}): {detail}"
        )

    def _get_soc_range(self, soc: float) -> str:
        """Get the SOC range bucket for a given SOC."""
        if soc < 25:
            return "0-25"
        elif soc < 50:
            return "25-50"
        elif soc < 75:
            return "50-75"
        elif soc < 90:
            return "75-90"
        else:
            return "90-100"

    def _get_temp_range(self, temp: float) -> str:
        """Get the temperature range bucket for a given temperature (Celsius).

        Uses configured temp_ranges boundaries. For default [5, 10, 15, 20]:
        - "<5" for temps below 5C
        - "5-10" for temps 5-10C
        - "10-15" for temps 10-15C
        - "15-20" for temps 15-20C
        - ">20" for temps above 20C
        """
        if not self.temp_ranges:
            return "all"

        for i, boundary in enumerate(self.temp_ranges):
            if temp < boundary:
                if i == 0:
                    return f"<{boundary}"
                else:
                    return f"{self.temp_ranges[i-1]}-{boundary}"

        # Above the highest boundary
        return f">{self.temp_ranges[-1]}"

    def record_charging(
        self,
        soc_start: float,
        soc_end: float,
        duration_minutes: float,
        energy_from_grid_kwh: Optional[float] = None,
        charge_price: float = 0.0,
        battery_temp: Optional[float] = None,
        battery_temp_start: Optional[float] = None,
        battery_temp_end: Optional[float] = None,
        energy_to_battery_kwh: Optional[float] = None,
        ambient_temp: Optional[float] = None,
    ):
        """
        Record a charging observation and update learned parameters.

        Args:
            soc_start: SOC at start of charging
            soc_end: SOC at end of charging
            duration_minutes: How long charging took
            energy_from_grid_kwh: INDEPENDENTLY MEASURED AC energy drawn from
                the grid for this charge interval (a meter reading). It is the
                only input from which efficiency can be learned. A value
                derived as ``stored / configured_efficiency`` is not a
                measurement -- it re-derives the configured constant -- and is
                rejected.
            charge_price: Price paid per kWh
            battery_temp: Battery temperature in Celsius (if available) - used for charge rate bucketing
            battery_temp_start: Battery temperature at start of charging (for warming rate tracking)
            battery_temp_end: Battery temperature at end of charging (for warming rate tracking)
            energy_to_battery_kwh: Actual energy stored in battery (from inverter sensor, more precise than SOC)
            ambient_temp: Ambient temperature during the interval (C).
                MUST come from the same source ``record_discharging`` uses (the
                ambient service), because both recorders feed ONE pooled
                regression in ``calibrate_thermal_coefficients`` over
                ``x1 = -(t_start - ambient)``. When only the discharge path
                supplied a real ambient and this one silently fell back to the
                rolling battery-temperature minimum, the regressor became
                correlated with the charge/discharge mode: a summer sample at
                t_start=33C gave x1=-3 (rolling min 30C) charging versus x1=-13
                (real ambient 20C) discharging for otherwise identical physics,
                so the fitted k1 absorbed the mode rather than the relaxation.
        """
        if duration_minutes <= 0 or soc_end <= soc_start:
            return

        # STORED-DC energy, both ways. The SOC delta is unambiguous by
        # definition; the inverter's "energy to battery" counter is the pack's
        # own accumulator and is likewise stored-side (it is the same quantity
        # `cost_tracker` weights the stored-energy cost basis by). Neither is a
        # terminal power: the conversion to `charge_input_dc_kw` happens once,
        # in `get_charge_rate_for_soc` (see the module docstring).
        if energy_to_battery_kwh is not None and energy_to_battery_kwh > 0:
            energy_added = energy_to_battery_kwh
            energy_source = "inverter"
        else:
            energy_added = (soc_end - soc_start) / 100 * self.battery_capacity
            energy_source = "soc"

        # A sample that resolves neither in energy nor in time cannot carry a
        # rate: the quotient is pure quantization noise. Rejecting it here is
        # the FIRST of the two lines of defence; ``get_charge_rate_for_soc`` is
        # the second (it also protects against a file poisoned before this
        # guard existed).
        unresolvable = self.observation_is_resolvable(energy_added, duration_minutes)
        if unresolvable is not None:
            self._reject_observation(
                "charge",
                unresolvable,
                f"{soc_start:.1f}%->{soc_end:.1f}% [{energy_source}]",
            )
            return

        # Observed STORED-side rate (stored_charge_kw). Persisted as such.
        charge_rate = energy_added / (duration_minutes / 60)

        if not self.is_plausible_rate(charge_rate):
            self._reject_observation(
                "charge",
                f"rate {charge_rate:.2f} kW > {self.max_plausible_rate_kw:.2f} kW "
                f"({self.max_rate_factor:.1f}x nominal)",
                f"{soc_start:.1f}%->{soc_end:.1f}% ({energy_added:.3f} kWh "
                f"[{energy_source}]) in {duration_minutes:.3f}min",
            )
            return

        # Efficiency is only learnable from an INDEPENDENT measurement of the
        # energy that entered the conversion chain. Passing
        # ``stored / configured_efficiency`` here is a tautology: it returns the
        # configured constant and would present it as an observation. The
        # caller (`cost_tracker`) therefore passes None until a real grid/AC
        # meter reading for the charge interval is available; the guard below
        # rejects a synthetic value that reproduces the configured factor.
        if energy_from_grid_kwh and energy_from_grid_kwh > 0:
            observed_efficiency = energy_added / energy_from_grid_kwh
            if abs(observed_efficiency - self.nominal_efficiency) < 1e-9:
                observed_efficiency = 0.0  # tautological: not a measurement
            if 0.5 < observed_efficiency < 1.0:
                self.learned_efficiency = (
                    self.ema_alpha * observed_efficiency +
                    (1 - self.ema_alpha) * self.learned_efficiency
                )

        # Update SOC-range specific charge rates (fallback when temperature unavailable)
        soc_range = self._get_soc_range((soc_start + soc_end) / 2)
        if soc_range not in self.stats.charge_rates_by_soc:
            self.stats.charge_rates_by_soc[soc_range] = []
        self.stats.charge_rates_by_soc[soc_range].append(charge_rate)

        # Keep last 50 observations per range
        if len(self.stats.charge_rates_by_soc[soc_range]) > 50:
            self.stats.charge_rates_by_soc[soc_range] = \
                self.stats.charge_rates_by_soc[soc_range][-50:]

        # Update temperature-aware charge rates (2D: SOC + temp)
        if battery_temp is not None:
            temp_range = self._get_temp_range(battery_temp)
            if soc_range not in self.stats.charge_rates_by_soc_temp:
                self.stats.charge_rates_by_soc_temp[soc_range] = {}
            if temp_range not in self.stats.charge_rates_by_soc_temp[soc_range]:
                self.stats.charge_rates_by_soc_temp[soc_range][temp_range] = []
            self.stats.charge_rates_by_soc_temp[soc_range][temp_range].append(charge_rate)

            # Keep last 50 observations per SOC+temp combination
            if len(self.stats.charge_rates_by_soc_temp[soc_range][temp_range]) > 50:
                self.stats.charge_rates_by_soc_temp[soc_range][temp_range] = \
                    self.stats.charge_rates_by_soc_temp[soc_range][temp_range][-50:]

        # Track temperature warming rate during charging
        # This helps predict when the inverter will switch to higher charge power
        if (battery_temp_start is not None and battery_temp_end is not None and
                duration_minutes >= 1.0):
            temp_change = battery_temp_end - battery_temp_start
            # Only record positive warming (battery heating up during charge)
            if temp_change > 0:
                warming_rate = temp_change / duration_minutes  # °C/minute
                start_temp_range = self._get_temp_range(battery_temp_start)

                if start_temp_range not in self.stats.temp_warming_rates:
                    self.stats.temp_warming_rates[start_temp_range] = []
                self.stats.temp_warming_rates[start_temp_range].append(warming_rate)

                # Keep last 50 observations per starting temp range
                if len(self.stats.temp_warming_rates[start_temp_range]) > 50:
                    self.stats.temp_warming_rates[start_temp_range] = \
                        self.stats.temp_warming_rates[start_temp_range][-50:]

        # Raw thermal sample for k1/k2 calibration. Charging samples are kept
        # alongside discharging ones: |P_bat| is the regressor, not the mode.
        #
        # UNITS: |P_bat| is the TERMINAL power, because k2 is Celsius per kWh
        # moved THROUGH the battery and because that is what every consumer of
        # the projector feeds it (`simulate_slot.battery_power_kw`,
        # `battery_power_for_entry`). `charge_rate` above is stored-side growth,
        # so it is converted here. Feeding the stored-side number made the
        # regressor low by `efficiency` on every charging sample, the fitted k2
        # high by `1/efficiency`, and then applied it to the larger power --
        # over-warming every projected charge. The discharge recorder needs no
        # conversion: its energy counter is already DC out of the pack.
        if battery_temp_start is not None and battery_temp_end is not None:
            self.record_thermal_observation(
                temp_start=battery_temp_start,
                temp_end=battery_temp_end,
                duration_minutes=duration_minutes,
                avg_power_kw=self.stored_to_input_dc_kw(charge_rate),
                ambient_temp=ambient_temp,
            )

        # Update totals
        self.stats.total_energy_charged_kwh += energy_added
        self.stats.total_charge_cost_eur += energy_added * charge_price

        # Update timestamps
        self.stats.last_observation = datetime.datetime.now().isoformat()

        temp_str = f", temp={battery_temp:.1f}C" if battery_temp is not None else ""
        # Get observation count for this bucket
        obs_count = len(self.stats.charge_rates_by_soc.get(soc_range, []))
        self.log(f"Learning: Recorded charge {soc_start:.1f}%->{soc_end:.1f}% ({energy_added:.3f} kWh [{energy_source}]) "
                 f"in {duration_minutes:.0f}min, rate={charge_rate:.2f}kW{temp_str}, "
                 f"bucket={soc_range} ({obs_count} obs)")

    def record_discharging(
        self,
        soc_start: float,
        soc_end: float,
        duration_minutes: float,
        energy_delivered_kwh: Optional[float] = None,
        price_eur_kwh: float = 0.0,
        battery_temp_start: Optional[float] = None,
        battery_temp_end: Optional[float] = None,
        ambient_temp: Optional[float] = None,
    ):
        """Record a discharging observation.

        ``battery_temp_start``/``battery_temp_end`` feed the thermal calibration.
        Before they existed the learning data contained ZERO discharge thermal
        observations, so the heating coefficient could not be fitted at all —
        which is exactly why discharging was modelled as thermally idle.
        """
        if duration_minutes <= 0 or soc_start <= soc_end:
            return

        # Use measured energy if available, otherwise calculate from SOC
        if energy_delivered_kwh is not None and energy_delivered_kwh > 0:
            energy_source = "inverter"
        else:
            energy_delivered_kwh = (soc_start - soc_end) / 100 * self.battery_capacity
            energy_source = "soc"

        # Same two guards as record_charging: the discharge path shares the
        # duration source (``cost_tracker._last_sig_soc_time``) and feeds the
        # SAME pooled thermal regression, so a corrupted |P_bat| here poisons
        # k1/k2 exactly as a corrupted charge sample would.
        unresolvable = self.observation_is_resolvable(
            energy_delivered_kwh, duration_minutes
        )
        if unresolvable is not None:
            self._reject_observation(
                "discharge",
                unresolvable,
                f"{soc_start:.1f}%->{soc_end:.1f}% [{energy_source}]",
            )
            return

        # Calculate observed discharge rate
        discharge_rate = energy_delivered_kwh / (duration_minutes / 60)

        if not self.is_plausible_rate(discharge_rate):
            self._reject_observation(
                "discharge",
                f"rate {discharge_rate:.2f} kW > {self.max_plausible_rate_kw:.2f} kW "
                f"({self.max_rate_factor:.1f}x nominal)",
                f"{soc_start:.1f}%->{soc_end:.1f}% ({energy_delivered_kwh:.3f} kWh "
                f"[{energy_source}]) in {duration_minutes:.3f}min",
            )
            return

        # Update totals
        self.stats.total_energy_discharged_kwh += energy_delivered_kwh
        self.stats.total_discharge_revenue_eur += energy_delivered_kwh * price_eur_kwh

        if battery_temp_start is not None and battery_temp_end is not None:
            self.record_thermal_observation(
                temp_start=battery_temp_start,
                temp_end=battery_temp_end,
                duration_minutes=duration_minutes,
                avg_power_kw=discharge_rate,
                ambient_temp=ambient_temp,
            )

        self.log(f"Learning: Recorded discharge {soc_start:.1f}%->{soc_end:.1f}% ({energy_delivered_kwh:.3f} kWh [{energy_source}]) "
                 f"in {duration_minutes:.0f}min, rate={discharge_rate:.2f}kW")

    def record_temperature_observation(self, temp: float):
        """
        Record a battery temperature observation for ambient temperature estimation.

        Keeps a rolling window of recent observations to estimate ambient temperature
        (minimum temperature the battery reaches when idle).

        Args:
            temp: Current battery temperature (°C)
        """
        if temp is None:
            return

        self.stats.recent_min_temps.append(temp)

        # Keep a 48 h window (192 samples at 15 min). Observations are event
        # driven as well as timer driven, so a short window can cover only a few
        # hours and would never contain an overnight trough.
        max_observations = TEMP_OBSERVATION_WINDOW
        if len(self.stats.recent_min_temps) > max_observations:
            self.stats.recent_min_temps = self.stats.recent_min_temps[-max_observations:]

    def get_estimated_ambient_min_temp(self, default: float = 10.0) -> float:
        """
        Rolling minimum battery temperature — an UPPER BOUND on ambient.

        The pack is self-heated and approaches ambient from above without ever
        falling below it, so ``min(T_bat)`` over the window bounds ambient from
        above; it is not the trough of the ambient swing.
        ``AmbientTemperatureService`` therefore anchors it as the daily MAXIMUM
        of its fallback diurnal profile — treating it as the minimum and adding
        the amplitude produced an "ambient" hotter than the battery itself.

        Args:
            default: Value to return when no observations are available (°C)
        """
        if not self.stats.recent_min_temps:
            return default
        return min(self.stats.recent_min_temps)

    def has_ambient_observations(self) -> bool:
        """Whether any battery temperature observation has been recorded."""
        return bool(self.stats.recent_min_temps)

    def get_estimated_ambient_temp(self, default: float = 10.0) -> float:
        """Backwards-compatible alias for :meth:`get_estimated_ambient_min_temp`."""
        return self.get_estimated_ambient_min_temp(default=default)

    # ------------------------------------------------------------------
    # Thermal model calibration (k1 relaxation, k2 self-heating)
    # ------------------------------------------------------------------

    def record_thermal_observation(
        self,
        temp_start: float,
        temp_end: float,
        duration_minutes: float,
        avg_power_kw: float,
        ambient_temp: Optional[float] = None,
    ) -> bool:
        """Store a raw thermal sample for k1/k2 least-squares calibration.

        Args:
            temp_start: Battery temperature at the start of the interval (°C)
            temp_end: Battery temperature at the end of the interval (°C)
            duration_minutes: Interval length (minutes)
            avg_power_kw: Mean magnitude of battery power over the interval (kW)
            ambient_temp: Ambient during the interval; estimated when omitted

        Returns:
            True when the sample was accepted.
        """
        if temp_start is None or temp_end is None:
            return False
        if duration_minutes is None or duration_minutes < max(1.0, self.min_observation_minutes):
            return False
        if avg_power_kw is None:
            return False

        power = abs(float(avg_power_kw))
        # |P_bat| is the k2 regressor, so an implausible power does not merely
        # add noise: it drags the whole pooled fit. Use the SAME bound the
        # charge-rate consumers use rather than a second magic constant.
        if power > self.max_plausible_rate_kw:
            return False
        if abs(float(temp_end) - float(temp_start)) >= 20.0:
            return False

        if ambient_temp is None:
            ambient_temp = self.get_estimated_ambient_min_temp(default=10.0)

        self.stats.thermal_samples.append([
            float(temp_start),
            float(temp_end),
            float(duration_minutes),
            power,
            float(ambient_temp),
        ])
        if len(self.stats.thermal_samples) > MAX_THERMAL_SAMPLES:
            self.stats.thermal_samples = self.stats.thermal_samples[-MAX_THERMAL_SAMPLES:]

        self._thermal_samples_since_calibration += 1
        if self._thermal_samples_since_calibration >= 10:
            self._thermal_samples_since_calibration = 0
            self.calibrate_thermal_coefficients()

        return True

    def calibrate_thermal_coefficients(self) -> Optional[Tuple[float, float]]:
        """Fit ``(k1, k2)`` of the SHARED thermal model to the raw samples.

        The model being fitted is exactly ``thermal_model.step_temperature``::

            T_end = Ta + (T_start - Ta) * exp(-k1*dt) + k2 * |P| * dt/60

        not its Euler linearisation ``(T_end-T_start)/dt = -k1*(T_start-Ta) +
        k2'*|P|``. The two are only equal as ``k1*dt -> 0``: fitting the Euler
        form to data generated by the exponential recovers a k1 that is low by
        roughly ``k1*dt/2`` — about 3 % at dt=5 min, 16 % at 30 min and 29 % at
        60 min for k1=0.012/min. Thermal samples come from whole charge and
        discharge sessions, so intervals of 20-40 min are the norm rather than
        the exception, and the calibrated k1 was therefore systematically
        biased against the very model it feeds.

        Solved in pure Python (numpy is not a dependency): the Euler normal
        equations provide the starting point, then damped Gauss-Newton
        iterations on the exact exponential residual
        ``r = d0*exp(-k1*dt) + k2*h - (T_end - Ta)`` refine it, where
        ``d0 = T_start - Ta`` and ``h = |P|*dt/60`` is the energy through the
        pack in kWh. ``k2`` is fitted directly in C per kWh, so the documented
        units of both coefficients are unchanged.

        Returns ``(k1, k2)`` or None when there is not enough data or the two
        regressors cannot be separated.
        """
        samples = self.stats.thermal_samples
        if len(samples) < MIN_THERMAL_SAMPLES:
            return None

        # (d0, dt, h, y) per usable sample, in the exponential model's terms.
        points: List[Tuple[float, float, float, float]] = []
        s11 = s12 = s22 = b1 = b2 = 0.0
        for sample in samples:
            if len(sample) < 5:
                continue
            t_start, t_end, dt_min, power, ambient = sample[:5]
            if dt_min <= 0:
                continue
            d0 = t_start - ambient
            h = power * dt_min / 60.0
            points.append((d0, dt_min, h, t_end - ambient))

            # Euler normal equations — used only as the starting point.
            y = (t_end - t_start) / dt_min
            x1 = -d0        # coefficient of k1
            x2 = power      # coefficient of k2' (C per min per kW)
            s11 += x1 * x1
            s12 += x1 * x2
            s22 += x2 * x2
            b1 += x1 * y
            b2 += x2 * y

        used = len(points)
        if used < MIN_THERMAL_SAMPLES:
            return None

        det = s11 * s22 - s12 * s12
        if abs(det) < 1e-12:
            # Regressors are collinear (e.g. every sample at the same power).
            return None

        k1 = (b1 * s22 - b2 * s12) / det
        k2 = (b2 * s11 - b1 * s12) / det * 60.0  # C per kWh through the battery

        k1 = _clamp_k1(k1)
        k2 = _clamp_k2(k2)

        refined = self._refine_thermal_coefficients(points, k1, k2)
        if refined is None:
            return None
        k1, k2 = refined

        self.stats.thermal_coeffs = {"k1": k1, "k2": k2, "n": float(used)}
        self.log(
            f"Learning: Calibrated thermal model k1={k1:.4f}/min, "
            f"k2={k2:.3f}C/kWh from {used} samples"
        )
        return k1, k2

    @staticmethod
    def _thermal_sse(
        points: List[Tuple[float, float, float, float]], k1: float, k2: float
    ) -> float:
        """Sum of squared residuals of ``step_temperature`` over the samples."""
        total = 0.0
        for d0, dt_min, h, y in points:
            r = d0 * math.exp(-k1 * dt_min) + k2 * h - y
            total += r * r
        return total

    def _refine_thermal_coefficients(
        self,
        points: List[Tuple[float, float, float, float]],
        k1: float,
        k2: float,
        max_iterations: int = 40,
    ) -> Optional[Tuple[float, float]]:
        """Damped Gauss-Newton refinement against the exponential model.

        Starting from the Euler estimate, this drives the residual of the
        actual ``step_temperature`` form to zero, so the coefficients that come
        out reproduce the projector on the observed samples instead of on its
        small-``k1*dt`` approximation.
        """
        sse = self._thermal_sse(points, k1, k2)

        for _ in range(max_iterations):
            j11 = j12 = j22 = g1 = g2 = 0.0
            for d0, dt_min, h, y in points:
                decay = math.exp(-k1 * dt_min)
                r = d0 * decay + k2 * h - y
                jac1 = -dt_min * d0 * decay   # dr/dk1
                jac2 = h                      # dr/dk2
                j11 += jac1 * jac1
                j12 += jac1 * jac2
                j22 += jac2 * jac2
                g1 += jac1 * r
                g2 += jac2 * r

            det = j11 * j22 - j12 * j12
            if abs(det) < 1e-18:
                # Cannot separate relaxation from self-heating at this point.
                break

            step1 = -(g1 * j22 - g2 * j12) / det
            step2 = -(g2 * j11 - g1 * j12) / det

            # Backtrack until the exact-model SSE actually improves.
            scale = 1.0
            improved = False
            for _ in range(20):
                cand1 = _clamp_k1(k1 + scale * step1)
                cand2 = _clamp_k2(k2 + scale * step2)
                cand_sse = self._thermal_sse(points, cand1, cand2)
                if cand_sse <= sse:
                    converged = (
                        abs(cand1 - k1) < 1e-9 and abs(cand2 - k2) < 1e-7
                    )
                    k1, k2, sse = cand1, cand2, cand_sse
                    improved = True
                    if converged:
                        return k1, k2
                    break
                scale *= 0.5

            if not improved:
                break

        return k1, k2

    def reset_thermal_calibration(self, drop_samples: bool = True) -> None:
        """Discard the fitted ``k1``/``k2`` (and optionally the raw samples).

        Used when a persisted fit is known to have been built from corrupted
        power figures: the engine falls back to the warming-rate bootstrap and
        the configured defaults, then re-bootstraps from freshly recorded
        samples. ``load_from_json`` calls it automatically for a fit outside
        :func:`thermal_coeffs_are_sane`.
        """
        self.stats.thermal_coeffs = {}
        if drop_samples:
            self.stats.thermal_samples = []
        self._thermal_samples_since_calibration = 0

    def sanitize_stats(self) -> Dict[str, int]:
        """Drop persisted observations that fail the plausibility bounds.

        Returns a dict of removal counts. Called on every load so that a file
        poisoned before the ingest guards existed (production 2026-09-02) is
        neutralised in memory immediately and written back clean on the next
        save — the offline cleanup script uses the SAME rule.
        """
        removed = {"charge_rates": 0, "thermal_samples": 0, "thermal_coeffs": 0}

        for soc_range, rates in list(self.stats.charge_rates_by_soc.items()):
            keep = self._plausible_rates(rates)
            removed["charge_rates"] += len(rates) - len(keep)
            self.stats.charge_rates_by_soc[soc_range] = keep

        for soc_range, temp_data in list(self.stats.charge_rates_by_soc_temp.items()):
            for temp_range, rates in list(temp_data.items()):
                keep = self._plausible_rates(rates)
                removed["charge_rates"] += len(rates) - len(keep)
                temp_data[temp_range] = keep

        kept_samples = []
        for sample in self.stats.thermal_samples:
            if len(sample) < 5:
                removed["thermal_samples"] += 1
                continue
            _t0, _t1, dt_min, power, _ambient = sample[:5]
            if (
                dt_min is None
                or dt_min < max(1.0, self.min_observation_minutes)
                or power is None
                or abs(float(power)) > self.max_plausible_rate_kw
            ):
                removed["thermal_samples"] += 1
                continue
            kept_samples.append(sample)
        self.stats.thermal_samples = kept_samples

        if self.stats.thermal_coeffs and not thermal_coeffs_are_sane(
            self.stats.thermal_coeffs
        ):
            removed["thermal_coeffs"] = 1
            self.stats.thermal_coeffs = {}

        return removed

    def get_heating_coefficient(
        self, default: Optional[float] = None
    ) -> float:
        """``k2`` in °C per kWh moved through the battery.

        Fallback chain:
        1. Calibrated value (>= ``MIN_THERMAL_SAMPLES`` raw samples)
        2. Bootstrap from the already-collected charge warming rates:
           ``median(°C/min) / nominal_charge_rate * 60``
        3. Configured default
        """
        if default is None:
            default = self.default_heating_c_per_kwh

        coeffs = self.stats.thermal_coeffs
        if coeffs and coeffs.get("n", 0) >= MIN_THERMAL_SAMPLES and "k2" in coeffs:
            return float(coeffs["k2"])

        all_rates: List[float] = []
        for rates in self.stats.temp_warming_rates.values():
            all_rates.extend(rates[-10:])
        if all_rates and self.nominal_charge_rate > 0:
            median_c_per_min = statistics.median(all_rates)
            bootstrap = median_c_per_min / self.nominal_charge_rate * 60.0
            return _clamp_k2(bootstrap)

        return default

    def get_cooling_rate_estimate(
        self, starting_temp: float, default: Optional[float] = None
    ) -> float:
        """``k1`` for a starting temperature, always returning a usable value."""
        if default is None:
            default = self.default_cooling_rate_per_min

        learned = self.get_cooling_rate(starting_temp)
        if learned is not None:
            return learned

        coeffs = self.stats.thermal_coeffs
        if coeffs and coeffs.get("n", 0) >= MIN_THERMAL_SAMPLES and "k1" in coeffs:
            return float(coeffs["k1"])

        return default

    def record_cooling(
        self,
        temp_start: float,
        temp_end: float,
        duration_minutes: float,
        ambient_temp: Optional[float] = None
    ):
        """
        Record a cooling observation during idle (HOLD/DISCHARGE) periods.

        Calculates the exponential decay rate and stores it by starting temperature range.

        Args:
            temp_start: Temperature at start of idle period (°C)
            temp_end: Temperature at end of idle period (°C)
            duration_minutes: Duration of idle period (minutes)
            ambient_temp: Ambient temperature estimate (°C), uses estimated if not provided
        """
        # Use estimated ambient if not provided
        if ambient_temp is None:
            ambient_temp = self.get_estimated_ambient_temp(default=10.0)

        # Only record if we have meaningful cooling (temp dropped toward ambient)
        if duration_minutes < 1.0:
            return
        if temp_start <= ambient_temp:
            return  # Already at or below ambient, no cooling to measure
        if temp_end >= temp_start:
            return  # Temperature didn't drop (maybe it even rose)
        if temp_end < ambient_temp:
            return  # Dropped below ambient - invalid data

        # Calculate the decay rate from the exponential decay formula:
        # T(t) = ambient + (start - ambient) * e^(-rate * t)
        # Solving for rate: rate = -ln((end - ambient) / (start - ambient)) / t
        import math
        temp_diff_start = temp_start - ambient_temp
        temp_diff_end = temp_end - ambient_temp

        if temp_diff_end <= 0 or temp_diff_start <= 0:
            return  # Avoid math errors

        ratio = temp_diff_end / temp_diff_start
        if ratio <= 0 or ratio >= 1:
            return  # Invalid ratio

        cooling_rate = -math.log(ratio) / duration_minutes

        # Sanity check: rate should be positive and reasonable (0.001 to 0.1 per minute)
        if cooling_rate <= 0.001 or cooling_rate > 0.1:
            return

        # Store by starting temperature range
        start_temp_range = self._get_temp_range(temp_start)

        if start_temp_range not in self.stats.temp_cooling_rates:
            self.stats.temp_cooling_rates[start_temp_range] = []
        self.stats.temp_cooling_rates[start_temp_range].append(cooling_rate)

        # Keep last 50 observations per starting temp range
        if len(self.stats.temp_cooling_rates[start_temp_range]) > 50:
            self.stats.temp_cooling_rates[start_temp_range] = \
                self.stats.temp_cooling_rates[start_temp_range][-50:]

        self.log(f"Learning: Recorded cooling {temp_start:.1f}C->{temp_end:.1f}C "
                 f"in {duration_minutes:.0f}min, rate={cooling_rate:.4f}/min, "
                 f"bucket={start_temp_range}")

    def get_cooling_rate(self, starting_temp: float) -> Optional[float]:
        """
        Get predicted cooling rate for a given starting temperature.

        Args:
            starting_temp: Battery temperature at start of idle period (°C)

        Returns:
            Cooling rate (decay per minute), or None if insufficient data
        """
        temp_range = self._get_temp_range(starting_temp)

        if temp_range in self.stats.temp_cooling_rates:
            rates = self.stats.temp_cooling_rates[temp_range]
            if len(rates) >= 3:
                return statistics.median(rates[-10:])

        # Try adjacent temperature ranges if exact match not found
        for try_temp in [starting_temp - 5, starting_temp + 5]:
            try_range = self._get_temp_range(try_temp)
            if try_range in self.stats.temp_cooling_rates:
                rates = self.stats.temp_cooling_rates[try_range]
                if len(rates) >= 3:
                    return statistics.median(rates[-10:])

        return None

    def get_charge_rate_for_soc(self, soc: float, battery_temp: Optional[float] = None) -> float:
        """
        Predicted ``charge_input_dc_kw`` for a SOC level and optional temperature.

        UNITS -- this is the API boundary of the module docstring's contract.
        Learned observations are stored-side (``stored_charge_kw``); they are
        divided by :attr:`storage_efficiency` here so that every consumer can
        keep doing ``rate * efficiency * duration``. The nominal fallback is
        already a terminal power and is returned unconverted, which is what
        makes a nominal rate and an equivalent learned observation predict
        identical physics.

        Uses learned data with fallback chain:
        1. Exact SOC+temp match (>=3 plausible observations) -> median of last 10
        2. SOC match, aggregate all temps
        3. SOC-only data (when temperature unavailable)
        4. Nominal rate

        THIS IS THE ONE GATE every consumer of a learned charge rate passes
        through — the DP (via ``DPOptimizer._rate_for``), the expected-SOC
        trajectory (via ``soc_projection._effective_charge_rate``) and the SOC
        deviation detector. Implausible observations are filtered out of every
        median (see :meth:`_plausible_rates`) and the result is clamped to
        :attr:`max_plausible_rate_kw`, so a persisted file poisoned before the
        ingest guards existed cannot leak a five-digit kW rate into any of them.
        Consumers must NOT add their own clamp.

        Args:
            soc: Current state of charge (%)
            battery_temp: Current battery temperature in Celsius (optional)

        Returns:
            Predicted charge rate in kW
        """
        soc_range = self._get_soc_range(soc)

        # Fallback 1: Try temperature-aware lookup if temp is available
        if battery_temp is not None:
            temp_range = self._get_temp_range(battery_temp)

            # Check for exact SOC+temp match
            if soc_range in self.stats.charge_rates_by_soc_temp:
                temp_data = self.stats.charge_rates_by_soc_temp[soc_range]
                exact = self._plausible_rates(temp_data.get(temp_range))
                if len(exact) >= 3:
                    return self._bounded_input_dc_rate(exact[-10:], soc_range)

                # Fallback 2: Aggregate all temps for this SOC range
                all_rates = []
                for rates in temp_data.values():
                    # Last 10 usable observations from each temp bucket
                    all_rates.extend(self._plausible_rates(rates)[-10:])
                if len(all_rates) >= 3:
                    return self._bounded_input_dc_rate(all_rates, soc_range)

        # Fallback 3: Use SOC-only data (when temperature unavailable)
        if soc_range in self.stats.charge_rates_by_soc:
            observations = self._plausible_rates(self.stats.charge_rates_by_soc[soc_range])
            if len(observations) >= 3:
                return self._bounded_input_dc_rate(observations[-10:], soc_range)

        # Fallback 4: Use configured nominal charge rate (already input DC)
        return self._nominal_input_dc_rate(soc_range)

    def get_stored_charge_rate_for_soc(
        self, soc: float, battery_temp: Optional[float] = None
    ) -> float:
        """``stored_charge_kw`` for a SOC level -- the observed-side view.

        Reporting/diagnostics only. Planning consumers want the terminal power
        and must call :meth:`get_charge_rate_for_soc`.
        """
        return self.input_dc_to_stored_kw(
            self.get_charge_rate_for_soc(soc, battery_temp)
        )

    def _nominal_input_dc_rate(self, soc_range: str) -> float:
        multiplier = self.soc_charge_multipliers.get(soc_range, 1.0)
        return self.nominal_charge_rate * multiplier

    def _bounded_input_dc_rate(self, stored_rates: List[float], soc_range: str) -> float:
        """Median of stored-side observations, converted and bounded.

        The conversion is the module docstring's contract; the clamp is applied
        AFTER it, because :attr:`max_plausible_rate_kw` is a terminal power.
        """
        median_stored = statistics.median(stored_rates)
        clamped = self.clamp_learned_rate(self.stored_to_input_dc_kw(median_stored))
        if clamped is None:
            return self._nominal_input_dc_rate(soc_range)
        return clamped

    def get_warming_rate(self, starting_temp: float) -> Optional[float]:
        """
        Get predicted battery warming rate during charging for a given starting temperature.

        Args:
            starting_temp: Battery temperature at start of charging (°C)

        Returns:
            Warming rate in °C/minute, or None if insufficient data
        """
        temp_range = self._get_temp_range(starting_temp)

        if temp_range in self.stats.temp_warming_rates:
            rates = self.stats.temp_warming_rates[temp_range]
            if len(rates) >= 3:
                return statistics.median(rates[-10:])

        # Try adjacent temperature ranges if exact match not found
        for try_temp in [starting_temp - 5, starting_temp + 5]:
            try_range = self._get_temp_range(try_temp)
            if try_range in self.stats.temp_warming_rates:
                rates = self.stats.temp_warming_rates[try_range]
                if len(rates) >= 3:
                    return statistics.median(rates[-10:])

        return None

    def predict_temp_after_duration(
        self,
        start_temp: float,
        duration_minutes: float,
        ambient_temp: Optional[float] = None,
        battery_power_kw: Optional[float] = None,
    ) -> float:
        """
        Predict battery temperature after charging for a given duration.

        When a warming rate has been learned for this temperature bucket it is
        used directly: it already nets self-heating against relaxation at that
        temperature. Otherwise the shared thermal model is used (relaxation
        toward ambient + ``k2 * |P|``), which is far better behaved than the old
        flat ``+0.1 °C/min``.

        The result is capped at ``MAX_BATTERY_TEMP_C``. Without that cap,
        the historical charge-rate precomputation projected 132 slots of
        unbounded linear warming and reached ~230 °C by the end of a 33 h
        horizon.

        Args:
            start_temp: Starting battery temperature (°C)
            duration_minutes: Charging duration in minutes
            ambient_temp: Ambient to relax toward (estimated when omitted)
            battery_power_kw: Charge power (nominal rate when omitted)

        Returns:
            Predicted temperature after charging
        """
        warming_rate = self.get_warming_rate(start_temp)
        if warming_rate is not None:
            return min(MAX_BATTERY_TEMP_C, start_temp + (warming_rate * duration_minutes))

        if ambient_temp is None:
            ambient_temp = self.get_estimated_ambient_min_temp(default=10.0)
        if battery_power_kw is None:
            battery_power_kw = self.nominal_charge_rate

        predicted = step_temperature(
            start_temp=start_temp,
            ambient_temp=ambient_temp,
            duration_minutes=duration_minutes,
            battery_power_kw=battery_power_kw,
            cooling_rate_per_min=self.get_cooling_rate_estimate(start_temp),
            heating_c_per_kwh=self.get_heating_coefficient(),
        )
        return min(MAX_BATTERY_TEMP_C, predicted)

    def predict_temp_after_idle(
        self,
        start_temp: float,
        duration_minutes: float,
        ambient_temp: Optional[float] = None,
        default_cooling_rate: float = 0.012
    ) -> float:
        """
        Predict battery temperature after idle (not charging) for a given duration.

        Uses exponential decay toward ambient temperature. First tries to use
        learned cooling rate for the starting temperature, falls back to default.

        Args:
            start_temp: Starting battery temperature (°C)
            duration_minutes: Idle duration in minutes
            ambient_temp: Ambient temperature to decay toward (°C), uses estimated if not provided
            default_cooling_rate: Fallback rate if no learned data available
                         0.012 means battery loses ~50% of excess temp per hour
                         (e.g., 21°C → ~18°C after 1 hour with 15°C ambient)

        Returns:
            Predicted temperature after idle period
        """
        # Use estimated ambient if not provided
        if ambient_temp is None:
            ambient_temp = self.get_estimated_ambient_min_temp(default=10.0)

        if start_temp <= ambient_temp:
            # Already at or below ambient, won't cool further
            return start_temp

        # Try to use learned cooling rate, fall back to default
        cooling_rate = self.get_cooling_rate_estimate(
            start_temp, default=default_cooling_rate
        )

        # Zero battery power: the shared thermal model degenerates to the
        # exponential decay this method has always used.
        return step_temperature(
            start_temp=start_temp,
            ambient_temp=ambient_temp,
            duration_minutes=duration_minutes,
            battery_power_kw=0.0,
            cooling_rate_per_min=cooling_rate,
        )

    def get_time_to_reach_temp(
        self,
        start_temp: float,
        target_temp: float
    ) -> Optional[float]:
        """
        Predict how long it will take to reach a target temperature during charging.

        Args:
            start_temp: Starting battery temperature (°C)
            target_temp: Target temperature to reach (°C)

        Returns:
            Time in minutes to reach target temp, or None if won't warm up
        """
        if target_temp <= start_temp:
            return 0.0

        warming_rate = self.get_warming_rate(start_temp)
        if warming_rate is None or warming_rate <= 0:
            return None

        return (target_temp - start_temp) / warming_rate

    def predict_charge_input_dc_energy(
        self,
        current_soc: float,
        start_temp: float,
        duration_minutes: float,
        temp_threshold: float = 16.0
    ) -> Tuple[float, float]:
        """
        Predict charge energy accounting for temperature warming during charging.

        **DIAGNOSTIC ONLY. Not used for planning or projection.** This is the
        legacy two-phase (cold/warm) within-slot model, and it is a SECOND
        thermal model: it warms the pack with ``get_time_to_reach_temp`` /
        ``predict_temp_after_duration`` rather than through
        ``thermal_model.TemperatureProjector``, and it changes the charge rate
        inside a slot, which the DP does not. On a 10 kWh pack at 10 % SOC with
        one 15-minute CHARGE crossing 1 kW -> 4 kW halfway it answered 16.25 %
        where the planner answered 12.5 %, and the expected-SOC trajectory that
        used to call it produced SOC-shortfall events on a battery that was
        following the plan exactly.

        The one within-slot model is a constant ``charge_input_dc_kw`` at the
        start-of-slot temperature -- ``soc_projection.project_slot_soc``,
        ``slot_energy.simulate_slot``, ``plan_validation.replay_plan`` and
        ``DPOptimizer`` all use it. Do not reintroduce this method into any of
        them; see ``docs/scheduling-algorithm.md`` SS Within-slot charge model.

        The inverter may charge faster once the battery warms above a threshold.
        This method calculates total energy by splitting the duration into
        cold and warm periods.

        UNITS: the returned energy is ``charge_input_dc_kwh`` -- DC energy at
        the battery terminal, BEFORE storage retention, because it is the time
        integral of :meth:`get_charge_rate_for_soc`. Callers obtain stored
        energy by multiplying by ``efficiency``. It was previously named
        ``predict_charge_energy_with_warming`` and its result was bound to a
        variable called ``energy_ac``, which it never was.

        Args:
            current_soc: Current state of charge (%)
            start_temp: Starting battery temperature (°C)
            duration_minutes: Total charging duration (minutes)
            temp_threshold: Temperature above which faster charging occurs (°C)

        Returns:
            Tuple of (charge_input_dc_kwh, end_temperature)
        """
        total_energy = 0.0
        remaining_minutes = duration_minutes
        current_temp = start_temp

        # If already warm, use warm charge rate for entire duration
        if start_temp >= temp_threshold:
            charge_rate = self.get_charge_rate_for_soc(current_soc, start_temp)
            total_energy = charge_rate * (duration_minutes / 60)
            # Predict end temperature
            end_temp = self.predict_temp_after_duration(start_temp, duration_minutes)
            return total_energy, end_temp

        # Calculate time to reach threshold temperature
        time_to_warm = self.get_time_to_reach_temp(start_temp, temp_threshold)

        if time_to_warm is not None and time_to_warm < duration_minutes:
            # Phase 1: Cold charging until reaching threshold
            cold_rate = self.get_charge_rate_for_soc(current_soc, start_temp)
            cold_energy = cold_rate * (time_to_warm / 60)
            total_energy += cold_energy

            # Phase 2: Warm charging for remaining time
            warm_minutes = remaining_minutes - time_to_warm
            warm_rate = self.get_charge_rate_for_soc(current_soc, temp_threshold)
            warm_energy = warm_rate * (warm_minutes / 60)
            total_energy += warm_energy

            # End temperature continues warming
            end_temp = self.predict_temp_after_duration(temp_threshold, warm_minutes)
        else:
            # Won't reach threshold during this duration - use cold rate throughout
            cold_rate = self.get_charge_rate_for_soc(current_soc, start_temp)
            total_energy = cold_rate * (duration_minutes / 60)
            end_temp = self.predict_temp_after_duration(start_temp, duration_minutes)

        return total_energy, end_temp

    def get_learning_summary(self) -> Dict:
        """Get summary of learned parameters."""
        if self.stats.total_energy_charged_kwh > 0:
            overall_efficiency = (
                self.stats.total_energy_discharged_kwh /
                self.stats.total_energy_charged_kwh
            )
        else:
            overall_efficiency = self.nominal_efficiency

        total_profit = (
            self.stats.total_discharge_revenue_eur -
            self.stats.total_charge_cost_eur
        )

        # Build temperature-aware rates summary with per-bucket confidence
        # The reported medians must be the ones the DP actually sees, so they go
        # through the same plausibility filter as get_charge_rate_for_soc.
        temp_aware_rates = {}
        for soc_range, temp_data in self.stats.charge_rates_by_soc_temp.items():
            temp_aware_rates[soc_range] = {}
            for temp_range, raw_rates in temp_data.items():
                rates = self._plausible_rates(raw_rates)
                if rates:
                    count = len(rates)
                    # Confidence: 0.7 base + up to 0.3 based on count (max at 10 obs)
                    confidence = 0.7 + min(0.3, (count - 3) / 7 * 0.3) if count >= 3 else 0.0
                    temp_aware_rates[soc_range][temp_range] = {
                        "median_kw": round(statistics.median(rates[-10:]), 2),
                        "observations": count,
                        "confidence": round(confidence, 2)
                    }

        # Build SOC-only rates with confidence
        soc_charge_rates = {}
        for soc_range, raw_rates in self.stats.charge_rates_by_soc.items():
            rates = self._plausible_rates(raw_rates)
            if rates:
                count = len(rates)
                # Confidence: 0.3 base + up to 0.2 based on count (max at 10 obs)
                confidence = 0.3 + min(0.2, (count - 3) / 7 * 0.2) if count >= 3 else 0.0
                soc_charge_rates[soc_range] = {
                    "median_kw": round(statistics.median(rates[-10:]), 2),
                    "observations": count,
                    "confidence": round(confidence, 2)
                }

        # Calculate total observations across all buckets
        total_observations = sum(
            len(self._plausible_rates(v)) for v in self.stats.charge_rates_by_soc.values()
        )

        # Build warming rates summary
        warming_rates_summary = {}
        for temp_range, rates in self.stats.temp_warming_rates.items():
            if rates:
                count = len(rates)
                warming_rates_summary[temp_range] = {
                    "median_c_per_min": round(statistics.median(rates[-10:]), 3),
                    "observations": count
                }

        # Build cooling rates summary
        cooling_rates_summary = {}
        for temp_range, rates in self.stats.temp_cooling_rates.items():
            if rates:
                count = len(rates)
                cooling_rates_summary[temp_range] = {
                    "median_rate_per_min": round(statistics.median(rates[-10:]), 4),
                    "observations": count
                }

        # Estimated ambient daily minimum
        estimated_ambient = self.get_estimated_ambient_min_temp(default=10.0)

        return {
            "learned_efficiency": round(self.learned_efficiency, 3),
            "total_energy_charged_kwh": round(self.stats.total_energy_charged_kwh, 1),
            "total_energy_discharged_kwh": round(self.stats.total_energy_discharged_kwh, 1),
            "overall_efficiency": round(overall_efficiency, 3),
            "total_profit_eur": round(total_profit, 2),
            "total_observations": total_observations,
            "soc_charge_rates": soc_charge_rates,
            "temp_aware_rates": temp_aware_rates,
            "temp_warming_rates": warming_rates_summary,
            "temp_cooling_rates": cooling_rates_summary,
            "estimated_ambient_temp": round(estimated_ambient, 1),
            "thermal_samples": len(self.stats.thermal_samples),
            "thermal_k1_per_min": round(self.get_cooling_rate_estimate(estimated_ambient), 4),
            "thermal_k2_c_per_kwh": round(self.get_heating_coefficient(), 3),
            "rejected_observations": self._rejected_observations,
            "max_plausible_rate_kw": round(self.max_plausible_rate_kw, 2),
            "thermal_calibrated": bool(
                self.stats.thermal_coeffs
                and self.stats.thermal_coeffs.get("n", 0) >= MIN_THERMAL_SAMPLES
            ),
        }

    def _drop_pre_v7_thermal_data(self) -> None:
        """One-off: discard thermal samples recorded in the wrong |P_bat| units.

        Charging samples before v7 stored stored-energy growth instead of
        terminal power. They cannot be converted: the sample list carries no
        mode flag, so a charge sample cannot be told from a discharge one and a
        blanket ``/efficiency`` would corrupt the discharge half.

        They are DERIVED, re-learnable data -- unlike the charge-rate history,
        which is preserved untouched. Roughly ten charge/discharge events
        re-accumulate them, and ``MIN_THERMAL_SAMPLES`` keeps the shared
        thermal model on its defaults until then. Idempotent by construction:
        the next save writes v7, so this runs at most once per file.
        """
        n_samples = len(self.stats.thermal_samples or [])
        had_fit = bool(self.stats.thermal_coeffs)
        if not n_samples and not had_fit:
            return
        self.stats.thermal_samples = []
        self.stats.thermal_coeffs = {}
        self._thermal_samples_since_calibration = 0
        self.log(
            "Learning: dropped pre-v7 thermal calibration data "
            f"({n_samples} sample(s)"
            + (", 1 fit" if had_fit else "")
            + "). Charging samples recorded stored-energy growth where the "
            "shared thermal model wants terminal power, so k2 was high by "
            "1/efficiency. Charge-rate history is unaffected; k1/k2 fall back "
            "to their defaults until ~10 events have re-accumulated."
        )

    def save_to_json(self) -> str:
        """Serialize learning state for persistence."""
        data = {
            # v6 added thermal_samples / thermal_coeffs. v7 fixes their |P_bat|
            # units on the charging side: terminal power, not stored-energy
            # growth. See load_from_json for what happens to a v6 file.
            "version": THERMAL_UNITS_VERSION,
            "learned_efficiency": self.learned_efficiency,
            "stats": self.stats.to_dict(),
        }
        return json.dumps(data)

    def load_from_json(self, json_str: str) -> bool:
        """Load learning state from JSON. Returns True if successful."""
        try:
            data = json.loads(json_str)
            if data.get("version", 0) >= 1:
                # Note: learned_charge_rate removed in v4, global confidence removed in v5
                self.learned_efficiency = data.get("learned_efficiency", self.nominal_efficiency)
                if "stats" in data:
                    self.stats = LearningStats.from_dict(data["stats"])
                    if data.get("version", 0) < THERMAL_UNITS_VERSION:
                        self._drop_pre_v7_thermal_data()
                    removed = self.sanitize_stats()
                    if any(removed.values()):
                        self.log(
                            "Learning: dropped implausible persisted data on load "
                            f"({removed['charge_rates']} charge-rate observations "
                            f"> {self.max_plausible_rate_kw:.2f} kW, "
                            f"{removed['thermal_samples']} thermal samples, "
                            f"{removed['thermal_coeffs']} thermal fits)"
                        )
                return True
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.log(f"Could not load learning data: {e}")
        return False
