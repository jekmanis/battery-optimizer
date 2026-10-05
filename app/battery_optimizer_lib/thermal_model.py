"""
Shared battery thermal model.

This module owns the ONE physics model for "given a battery temperature at the
start of an interval, an ambient temperature and the power actually flowing
through the battery, what is the temperature at the end of the interval?".

Historically four places modelled this independently and all of them made
warming a function of the scheduled MODE rather than of the power:

* ``DPOptimizer._build_temp_trajectory`` — warmed only in ``CHARGE`` slots
* ``BatteryOptimizer.calculate_expected_soc_schedule`` (via
  ``soc_projection._idle_temp``) — cooled in ``DISCHARGE``/``HOLD``
* ``ScheduleFormatter._format_*_trajectory`` — same split again
* ``charge_rate_utils.compute_charge_rates_per_slot`` — unbounded linear
  warming (that module is gone; the DP's temperature profile now comes from
  ``DPOptimizer._idle_temp_profile`` and ``_replay_plan_temps``)

Because the optimizer scheduled zero CHARGE slots over a 33 h production window,
the resulting trajectory was monotonically non-increasing and, with the ambient
estimate stuck at ~the current battery temperature, practically flat: 5763 of
5891 logged trajectory rows were of the form ``(33C->33C)``.

The model
---------

    T(t+dt) = Ta(t) + (T(t) - Ta(t)) * exp(-k1*dt)  +  k2 * |P_bat| * dt/60

* ``k1`` (per minute) is the Newtonian relaxation rate toward ambient. The
  exponential (rather than the Euler form ``T + k1*(Ta-T)*dt``) is deliberate:
  the already-learned ``LearningStats.temp_cooling_rates`` are decay-per-minute
  values derived as ``-ln(ratio)/duration``, so the existing learned data stays
  valid.
* ``k2`` (Celsius per kWh moved through the battery) is the ohmic/BMS self
  heating. It depends on |P_bat| only — charging and discharging both heat the
  pack. A 5.9 kW discharge is not thermally idle.

``battery_power_for_entry`` is the single place that derives |P_bat| from a
scheduled mode, so every consumer of the model feeds it the same number.
"""

import datetime
import math
from typing import List, Optional, Sequence, Tuple

from .models import BatteryMode

# Fallback Newtonian relaxation rate (per minute) when nothing has been learned.
# 0.012/min ~= the battery sheds half of its excess temperature per hour.
DEFAULT_COOLING_RATE_PER_MIN = 0.012

# Fallback self-heating coefficient, Celsius per kWh moved through the pack.
DEFAULT_HEATING_C_PER_KWH = 0.35

# Hard ceiling for a *projected* temperature. Not a physical limit — it stops a
# runaway linear warming projection from poisoning charge-rate lookups.
MAX_BATTERY_TEMP_C = 55.0

# A projection may undershoot the ambient/starting temperature by at most this
# much; anything beyond that is a modelling artifact.
PROJECTION_UNDERSHOOT_C = 2.0


def step_temperature(
    start_temp: float,
    ambient_temp: float,
    duration_minutes: float,
    battery_power_kw: float = 0.0,
    cooling_rate_per_min: float = DEFAULT_COOLING_RATE_PER_MIN,
    heating_c_per_kwh: float = DEFAULT_HEATING_C_PER_KWH,
) -> float:
    """One thermal step: relaxation toward ambient plus self-heating.

    Args:
        start_temp: Battery temperature at the start of the interval (C).
        ambient_temp: Ambient temperature during the interval (C).
        duration_minutes: Interval length in minutes.
        battery_power_kw: Magnitude of the power through the battery (kW).
            Sign is ignored — charging and discharging both heat the pack.
        cooling_rate_per_min: ``k1``, Newtonian relaxation rate per minute.
        heating_c_per_kwh: ``k2``, Celsius per kWh moved through the battery.

    Returns:
        Predicted temperature at the end of the interval (C). No clamping is
        applied here — see ``TemperatureProjector.project``.
    """
    if duration_minutes <= 0:
        return start_temp

    k1 = max(0.0, float(cooling_rate_per_min))
    k2 = max(0.0, float(heating_c_per_kwh))

    relaxed = ambient_temp + (start_temp - ambient_temp) * math.exp(-k1 * duration_minutes)
    heating = k2 * abs(float(battery_power_kw)) * (duration_minutes / 60.0)
    return relaxed + heating


def battery_power_for_entry(
    mode: BatteryMode,
    *,
    charge_rate_kw: float = 0.0,
    load_kw: float = 0.0,
    pv_kw: float = 0.0,
    discharge_rate_kw: float = 0.0,
    export_discharge_rate_kw: float = 0.0,
    export_rate: Optional[float] = None,
    inverter_efficiency: float = 1.0,
) -> float:
    """Magnitude of the DC power a scheduled slot REQUESTS from the battery (kW).

    **This is the requested flow, not the actual one.** It takes no SOC and no
    capacity limits, so it reports a full slot of charging power for a pack that
    is already at ``max_soc`` and cannot take a joule. Anything that needs the
    energy that really moved must use ``slot_energy.simulate_slot``'s
    ``battery_power_kw``, which is capped by headroom and by available energy;
    ``soc_projection.project_slot_soc`` now does, and every consumer of the
    shared projection with it. Warming an idle pack because the schedule asked
    it to do something impossible is exactly how imaginary power manufactures
    future charging capability.

    The mode split below mirrors ``project_slot_soc``'s energy split so the two
    never disagree about what the battery is being ASKED to do:

    * ``CHARGE``            -> the (learned) charge rate
    * ``DISCHARGE`` export  -> the export discharge rate, DC side
    * ``DISCHARGE`` self    -> ``min(max(0, load-pv), discharge_rate)``, DC side,
      **or**, when ``pv >= load``, the same PV-surplus charging power ``HOLD``
      reports. ``project_slot_soc``'s self-consumption branch charges the pack
      from ``min(max(0, pv-load), charge_rate)`` in exactly that regime, so
      returning 0 kW here modelled a pack whose SOC was rising as thermally
      idle. The orchestrator's cloud-safe HOLD -> ``discharge_to_load``
      conversion makes midday ``DISCHARGE`` slots with ``pv > load`` routine,
      so this was the common case, not an edge case — and a ``mode``-keyed
      special case of exactly the kind the one-thermal-model invariant forbids.
    * ``HOLD``              -> ``min(max(0, pv-load), charge_rate)`` (PV surplus)
    """
    inv_eff = inverter_efficiency if inverter_efficiency and inverter_efficiency > 0 else 1.0
    net_load_kw = max(0.0, load_kw - pv_kw)
    pv_surplus_kw = max(0.0, pv_kw - load_kw)
    charge_rate_kw = max(0.0, charge_rate_kw)

    if mode == BatteryMode.CHARGE:
        return charge_rate_kw

    if mode == BatteryMode.DISCHARGE:
        if export_rate is not None and export_rate > 0:
            edr = (
                export_discharge_rate_kw
                if export_discharge_rate_kw > 0
                else discharge_rate_kw
            )
            return max(0.0, edr) / inv_eff
        # Self-consumption. ``net_load_kw`` and ``pv_surplus_kw`` are mutually
        # exclusive by construction, so this is the sum of the two branches
        # ``project_slot_soc`` applies: DC drawn to serve the net load, plus DC
        # stored from PV surplus. With ``pv >= load`` it reduces to the HOLD
        # expression below — the pack is charging, not idling.
        discharge_kw = min(net_load_kw, max(0.0, discharge_rate_kw)) / inv_eff
        pv_charge_kw = min(pv_surplus_kw, charge_rate_kw)
        return discharge_kw + pv_charge_kw

    # HOLD — the battery only sees PV surplus, if any.
    return min(pv_surplus_kw, charge_rate_kw)


class TemperatureProjector:
    """Projects battery temperature forward using the shared thermal model.

    One instance is shared by the DP trajectory, the expected-SOC trajectory,
    the schedule formatter and the charge-rate pre-computation, so the log can
    never show two different temperature models depending on the code path.

    Args:
        learning_engine: Optional ``BatteryLearningEngine`` supplying learned
            ``k1`` (cooling rate) and ``k2`` (heating coefficient).
        ambient_provider: Optional object with ``predict_c(dt) -> Optional[float]``
            (an ``AmbientTemperatureService``). When it returns a value, ambient
            becomes a function of TIME across the whole horizon.
        fallback_ambient_c: Ambient used when neither source knows anything.
    """

    def __init__(
        self,
        learning_engine=None,
        ambient_provider=None,
        log_func=None,
        default_cooling_rate: float = DEFAULT_COOLING_RATE_PER_MIN,
        default_heating_c_per_kwh: float = DEFAULT_HEATING_C_PER_KWH,
        fallback_ambient_c: float = 10.0,
        max_temp_c: float = MAX_BATTERY_TEMP_C,
    ):
        self.learning_engine = learning_engine
        self.ambient_provider = ambient_provider
        self.log = log_func
        self.default_cooling_rate = default_cooling_rate
        self.default_heating_c_per_kwh = default_heating_c_per_kwh
        self.fallback_ambient_c = fallback_ambient_c
        self.max_temp_c = max_temp_c

    # ------------------------------------------------------------------
    # Coefficient / ambient sourcing
    # ------------------------------------------------------------------

    def ambient_at(self, slot_time: Optional[datetime.datetime] = None) -> float:
        """Ambient temperature (C) at a point in time.

        Source chain: external ambient provider (weather forecast / outdoor
        sensor / diurnal profile) -> learning engine's rolling minimum ->
        configured fallback.
        """
        if self.ambient_provider is not None:
            try:
                value = self.ambient_provider.predict_c(slot_time)
            except Exception:  # pragma: no cover - provider must never break projection
                value = None
            if value is not None:
                return float(value)

        if self.learning_engine is not None:
            try:
                return float(
                    self.learning_engine.get_estimated_ambient_min_temp(
                        default=self.fallback_ambient_c
                    )
                )
            except AttributeError:
                return float(
                    self.learning_engine.get_estimated_ambient_temp(
                        default=self.fallback_ambient_c
                    )
                )

        return self.fallback_ambient_c

    def cooling_rate(self, start_temp: float) -> float:
        """``k1`` for the given starting temperature (per minute)."""
        if self.learning_engine is not None:
            getter = getattr(self.learning_engine, "get_cooling_rate_estimate", None)
            if getter is not None:
                return float(getter(start_temp, default=self.default_cooling_rate))
            learned = self.learning_engine.get_cooling_rate(start_temp)
            if learned is not None:
                return float(learned)
        return self.default_cooling_rate

    def heating_coefficient(self) -> float:
        """``k2`` in Celsius per kWh moved through the battery."""
        if self.learning_engine is not None:
            getter = getattr(self.learning_engine, "get_heating_coefficient", None)
            if getter is not None:
                return float(getter(default=self.default_heating_c_per_kwh))
        return self.default_heating_c_per_kwh

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(
        self,
        start_temp: float,
        slot_time: Optional[datetime.datetime] = None,
        duration_minutes: float = 0.0,
        battery_power_kw: float = 0.0,
    ) -> float:
        """Project the temperature over one interval starting at ``slot_time``."""
        if start_temp is None:
            return start_temp
        if duration_minutes <= 0:
            return start_temp

        ambient = self.ambient_at(slot_time)
        projected = step_temperature(
            start_temp=start_temp,
            ambient_temp=ambient,
            duration_minutes=duration_minutes,
            battery_power_kw=battery_power_kw,
            cooling_rate_per_min=self.cooling_rate(start_temp),
            heating_c_per_kwh=self.heating_coefficient(),
        )

        # A step can never take the pack below both its own starting point and
        # ambient; anything lower is numerical noise, not physics.
        floor = min(start_temp, ambient) - PROJECTION_UNDERSHOOT_C
        return min(self.max_temp_c, max(floor, projected))

    def project_trajectory(
        self,
        start_temp: float,
        slot_times: Sequence[Optional[datetime.datetime]],
        durations_minutes: Sequence[float],
        powers_kw: Sequence[float],
    ) -> List[Tuple[float, float]]:
        """Project a whole horizon, returning ``(start, end)`` per slot."""
        trajectory: List[Tuple[float, float]] = []
        temp = start_temp
        for i, slot_time in enumerate(slot_times):
            duration = durations_minutes[i] if i < len(durations_minutes) else 0.0
            power = powers_kw[i] if i < len(powers_kw) else 0.0
            slot_start = temp
            temp = self.project(temp, slot_time, duration, power)
            trajectory.append((slot_start, temp))
        return trajectory
