"""
Tests for the shared battery thermal model (thermal_model.py).

Regression coverage for DEFECT 6 — "warming happens only in CHARGE mode".
The optimizer scheduled zero CHARGE slots over a 33 h production window, so the
old model made the temperature trajectory monotonically non-increasing by
construction: a 5.9 kW discharge was treated as thermally idle.
"""

import datetime
import math

import pytest

from battery_optimizer_lib import BatteryMode
from battery_optimizer_lib.thermal_model import (
    DEFAULT_COOLING_RATE_PER_MIN,
    DEFAULT_HEATING_C_PER_KWH,
    MAX_BATTERY_TEMP_C,
    TemperatureProjector,
    battery_power_for_entry,
    step_temperature,
)


class TestStepTemperature:
    """The single thermal step."""

    def test_step_temperature_pure_cooling(self):
        """P=0 degenerates to the historical exponential decay toward ambient."""
        result = step_temperature(
            start_temp=33.0,
            ambient_temp=25.0,
            duration_minutes=60.0,
            battery_power_kw=0.0,
            cooling_rate_per_min=0.012,
            heating_c_per_kwh=0.35,
        )
        expected = 25.0 + 8.0 * math.exp(-0.012 * 60.0)
        assert abs(result - expected) < 0.01
        assert 28.8 < result < 29.0

    def test_step_temperature_discharge_heats(self):
        """Identical inputs but 5.9 kW through the pack must warm it.

        This is the direct DEFECT 6 regression: the old model produced the
        pure-cooling number for ANY non-CHARGE slot regardless of power.
        """
        cooling_only = step_temperature(
            start_temp=33.0,
            ambient_temp=25.0,
            duration_minutes=60.0,
            battery_power_kw=0.0,
            cooling_rate_per_min=0.012,
            heating_c_per_kwh=0.35,
        )
        discharging = step_temperature(
            start_temp=33.0,
            ambient_temp=25.0,
            duration_minutes=60.0,
            battery_power_kw=5.9,
            cooling_rate_per_min=0.012,
            heating_c_per_kwh=0.35,
        )

        assert discharging > cooling_only
        # k2 * |P| * hours = 0.35 * 5.9 * 1.0
        assert abs((discharging - cooling_only) - 0.35 * 5.9) < 0.001

    def test_heating_is_sign_agnostic(self):
        """Charging and discharging of equal magnitude heat identically."""
        charging = step_temperature(30.0, 25.0, 15.0, battery_power_kw=4.5)
        discharging = step_temperature(30.0, 25.0, 15.0, battery_power_kw=-4.5)
        assert abs(charging - discharging) < 1e-9

    def test_zero_duration_is_a_noop(self):
        assert step_temperature(31.3, 20.0, 0.0, battery_power_kw=5.0) == 31.3

    def test_warms_toward_ambient_when_below_it(self):
        """A cold pack in a warm room warms up even with no power."""
        result = step_temperature(10.0, 27.0, 60.0, battery_power_kw=0.0)
        assert 10.0 < result < 27.0


class TestBatteryPowerForEntry:
    """|P_bat| derivation must match soc_projection's energy split."""

    def test_charge_uses_charge_rate(self):
        assert battery_power_for_entry(
            BatteryMode.CHARGE, charge_rate_kw=4.5, load_kw=1.0, pv_kw=0.0
        ) == pytest.approx(4.5)

    def test_export_discharge_uses_export_rate_dc_side(self):
        power = battery_power_for_entry(
            BatteryMode.DISCHARGE,
            charge_rate_kw=4.5,
            load_kw=0.5,
            pv_kw=0.0,
            discharge_rate_kw=4.5,
            export_discharge_rate_kw=5.9,
            export_rate=100,
            inverter_efficiency=0.97,
        )
        assert power == pytest.approx(5.9 / 0.97)

    def test_export_discharge_falls_back_to_discharge_rate(self):
        power = battery_power_for_entry(
            BatteryMode.DISCHARGE,
            discharge_rate_kw=4.5,
            export_discharge_rate_kw=0.0,
            export_rate=100,
        )
        assert power == pytest.approx(4.5)

    def test_self_consumption_discharge_uses_net_load(self):
        power = battery_power_for_entry(
            BatteryMode.DISCHARGE,
            charge_rate_kw=4.5,
            load_kw=3.5,
            pv_kw=0.5,
            discharge_rate_kw=4.5,
            export_rate=0,
            inverter_efficiency=0.97,
        )
        assert power == pytest.approx(3.0 / 0.97)

    def test_self_consumption_discharge_capped_by_discharge_rate(self):
        power = battery_power_for_entry(
            BatteryMode.DISCHARGE, load_kw=9.0, pv_kw=0.0, discharge_rate_kw=4.5
        )
        assert power == pytest.approx(4.5)

    def test_hold_uses_pv_surplus(self):
        power = battery_power_for_entry(
            BatteryMode.HOLD, charge_rate_kw=4.5, load_kw=1.0, pv_kw=3.0
        )
        assert power == pytest.approx(2.0)

    def test_hold_without_pv_is_idle(self):
        power = battery_power_for_entry(
            BatteryMode.HOLD, charge_rate_kw=4.5, load_kw=1.0, pv_kw=0.0
        )
        assert power == pytest.approx(0.0)


class _StubAmbient:
    """Ambient provider returning a per-hour value."""

    def __init__(self, by_hour, default=None):
        self.by_hour = by_hour
        self.default = default

    def predict_c(self, dt):
        if dt is None:
            return self.default
        return self.by_hour.get(dt.hour, self.default)


class TestTemperatureProjector:
    def test_ambient_provider_wins_over_learning_engine(self, learning_engine):
        learning_engine.record_temperature_observation(30.5)
        projector = TemperatureProjector(
            learning_engine=learning_engine,
            ambient_provider=_StubAmbient({8: 27.4, 15: 33.0}),
        )
        assert projector.ambient_at(datetime.datetime(2026, 7, 27, 8)) == pytest.approx(27.4)
        assert projector.ambient_at(datetime.datetime(2026, 7, 27, 15)) == pytest.approx(33.0)

    def test_falls_back_to_learning_engine_minimum(self, learning_engine):
        learning_engine.record_temperature_observation(31.0)
        learning_engine.record_temperature_observation(27.4)
        projector = TemperatureProjector(learning_engine=learning_engine)
        assert projector.ambient_at(datetime.datetime(2026, 7, 27, 8)) == pytest.approx(27.4)

    def test_falls_back_to_configured_default(self):
        projector = TemperatureProjector(fallback_ambient_c=12.0)
        assert projector.ambient_at(None) == pytest.approx(12.0)

    def test_discharge_slot_warms_relative_to_idle(self):
        """Same slot, same ambient — only the power differs."""
        projector = TemperatureProjector(fallback_ambient_c=25.0)
        slot = datetime.datetime(2026, 7, 27, 20, 0)

        idle = projector.project(33.0, slot, 15.0, 0.0)
        discharging = projector.project(33.0, slot, 15.0, 5.9)

        assert discharging > idle
        assert discharging - idle == pytest.approx(
            DEFAULT_HEATING_C_PER_KWH * 5.9 * 0.25, abs=1e-6
        )

    def test_projection_is_capped(self):
        projector = TemperatureProjector(fallback_ambient_c=25.0)
        result = projector.project(54.0, None, 600.0, 15.0)
        assert result <= MAX_BATTERY_TEMP_C

    def test_projection_does_not_undershoot(self):
        projector = TemperatureProjector(fallback_ambient_c=27.0)
        result = projector.project(27.0, None, 60.0, 0.0)
        assert result >= 27.0 - 2.0

    def test_project_trajectory_follows_ambient_over_time(self):
        """A time-varying ambient must produce a time-varying trajectory."""
        base = datetime.datetime(2026, 7, 27, 16, 0)
        ambient = _StubAmbient(
            {h: 27.0 + 6.0 * math.cos(2 * math.pi * (h - 15) / 24) for h in range(24)},
            default=27.0,
        )
        projector = TemperatureProjector(ambient_provider=ambient)

        slots = [base + datetime.timedelta(hours=i) for i in range(20)]
        trajectory = projector.project_trajectory(
            start_temp=34.0,
            slot_times=slots,
            durations_minutes=[60.0] * len(slots),
            powers_kw=[0.0] * len(slots),
        )

        ends = [round(end, 1) for _, end in trajectory]
        # The old model produced a single repeated value here.
        assert len(set(ends)) >= 5
        assert max(ends) - min(ends) > 1.0

    def test_uses_learned_cooling_rate(self, learning_engine):
        for _ in range(5):
            learning_engine.record_cooling(
                temp_start=21.0, temp_end=18.0, duration_minutes=60, ambient_temp=15.0
            )
        projector = TemperatureProjector(
            learning_engine=learning_engine, fallback_ambient_c=15.0
        )
        # Learned rate ~0.0116/min reproduces the observed 21 -> 18 in one hour.
        assert projector.cooling_rate(21.0) == pytest.approx(0.0116, abs=0.002)
        assert projector.project(21.0, None, 60.0, 0.0) == pytest.approx(18.0, abs=0.5)

    def test_default_cooling_rate_without_data(self, learning_engine):
        projector = TemperatureProjector(learning_engine=learning_engine)
        assert projector.cooling_rate(33.0) == pytest.approx(DEFAULT_COOLING_RATE_PER_MIN)
