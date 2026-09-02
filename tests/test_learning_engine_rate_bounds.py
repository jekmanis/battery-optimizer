"""
Plausibility bounds on learned battery rates.

Production defect (33 h AppDaemon log, 2026-09-02, lines 510-548): during the
05:00 CHARGE slot the SOC deviation detector logged TWELVE times

    SOC behind by 90.0% during CHARGE (actual=10.0%, expected=100.0%), but
    projected to reach 21894.1% with remaining charge hours - skipping
    recalculation

Root cause chain:

* ``cost_tracker.process_soc_change`` re-stamps ``_last_sig_soc_time`` a few
  MILLISECONDS before the energy-sensor callback calls ``record_charging``, so a
  genuine 0.1 kWh delta divided by a 10-40 ms "duration" produced charge-rate
  observations of 34535 kW and 44653 kW.
* ``record_charging`` only guarded ``duration_minutes <= 0``.
* ``get_charge_rate_for_soc`` returned the raw median: the live file's
  0-25%/>20C bucket held ``[2.806, 34535.687, 44653.932, 14308.71, 5.959]`` and
  therefore answered **14308.71 kW** for a 4.5 kW battery.
* Every consumer took that number unbounded.

These tests pin both lines of defence: reject at ingest, and never serve an
implausible rate even from an already-poisoned persisted file.
"""

import json

import pytest

from battery_optimizer_lib import BatteryLearningEngine
from battery_optimizer_lib.learning_engine import (
    MAX_HEATING_C_PER_KWH,
    MIN_OBSERVATION_MINUTES,
    thermal_coeffs_are_sane,
)

CAPACITY = 14.3
NOMINAL = 4.5

# Verbatim from the live //192.168.33.167 learning file (2026-09-02 08:02).
LIVE_0_25_ABOVE_20 = [2.806, 34535.687, 44653.932, 14308.71, 5.959]


def _engine(**kwargs) -> BatteryLearningEngine:
    base = dict(
        battery_capacity_kwh=CAPACITY,
        nominal_charge_rate_kw=NOMINAL,
        nominal_efficiency=0.95,
        min_soc=10.0,
        max_soc=100.0,
        log_func=lambda *a, **k: None,
    )
    base.update(kwargs)
    return BatteryLearningEngine(**base)


class TestPlausibilityBound:
    def test_bound_is_two_times_nominal(self):
        assert _engine().max_plausible_rate_kw == pytest.approx(2.0 * NOMINAL)

    def test_bound_follows_the_larger_nominal_rate(self):
        engine = _engine(nominal_discharge_rate_kw=5.9)
        assert engine.max_plausible_rate_kw == pytest.approx(2.0 * 5.9)

    def test_real_warm_battery_rate_is_inside_the_bound(self):
        """6.8 kW is the observed warm-pack ceiling — it must stay learnable."""
        assert _engine().is_plausible_rate(6.82)

    def test_five_digit_rate_is_outside_the_bound(self):
        engine = _engine()
        for rate in LIVE_0_25_ABOVE_20[1:4]:
            assert not engine.is_plausible_rate(rate)


class TestRecordChargingGuards:
    def test_sub_minute_duration_is_rejected(self):
        engine = _engine()
        # 0.1 kWh over 40 ms — the exact production shape. One counter tick, so
        # the energy does not resolve a rate either.
        engine.record_charging(
            soc_start=10.0,
            soc_end=11.0,
            duration_minutes=0.00073,
            energy_to_battery_kwh=0.1,
        )
        assert engine.stats.charge_rates_by_soc == {}
        assert engine.stats.total_energy_charged_kwh == 0
        assert engine._rejected_observations == 1

    def test_duration_floor_is_configurable(self):
        engine = _engine(min_observation_minutes=5.0)
        engine.record_charging(soc_start=50.0, soc_end=51.0, duration_minutes=3.0)
        assert engine.stats.charge_rates_by_soc == {}
        assert MIN_OBSERVATION_MINUTES == 0.25

    def test_implausible_rate_is_rejected_even_with_a_long_duration(self):
        engine = _engine()
        # 30 kWh into a 14.3 kWh pack in 10 minutes = 180 kW.
        engine.record_charging(
            soc_start=10.0,
            soc_end=90.0,
            duration_minutes=10.0,
            energy_to_battery_kwh=30.0,
        )
        assert engine.stats.charge_rates_by_soc == {}
        assert engine._rejected_observations == 1

    def test_plausible_observation_is_still_recorded(self):
        engine = _engine()
        # 12% of 14.3 kWh in 15 min = 6.86 kW — the real 05:00 slot.
        engine.record_charging(soc_start=9.0, soc_end=21.0, duration_minutes=15.0)
        assert engine.stats.charge_rates_by_soc["0-25"] == pytest.approx([6.864], rel=1e-3)
        assert engine._rejected_observations == 0

    def test_rejection_is_logged(self):
        messages = []
        engine = _engine(log_func=lambda msg, *a, **k: messages.append(msg))
        engine.record_charging(
            soc_start=10.0, soc_end=11.0, duration_minutes=0.001,
            energy_to_battery_kwh=0.1,
        )
        assert any("rejected implausible charge" in m for m in messages)

    def test_rejected_sample_does_not_move_efficiency(self):
        engine = _engine()
        before = engine.learned_efficiency
        engine.record_charging(
            soc_start=10.0, soc_end=11.0, duration_minutes=0.001,
            energy_to_battery_kwh=0.1, energy_from_grid_kwh=0.15,
        )
        assert engine.learned_efficiency == before

    def test_rejected_sample_does_not_create_a_thermal_sample(self):
        engine = _engine()
        engine.record_charging(
            soc_start=10.0, soc_end=11.0, duration_minutes=0.001,
            energy_to_battery_kwh=0.1,
            battery_temp_start=21.9, battery_temp_end=22.0,
        )
        assert engine.stats.thermal_samples == []


class TestRecordDischargingGuards:
    def test_sub_minute_duration_is_rejected(self):
        engine = _engine()
        engine.record_discharging(
            soc_start=50.0, soc_end=49.0, duration_minutes=0.002,
            energy_delivered_kwh=0.1,
        )
        assert engine.stats.total_energy_discharged_kwh == 0
        assert engine.stats.thermal_samples == []

    def test_implausible_rate_is_rejected(self):
        engine = _engine()
        engine.record_discharging(
            soc_start=90.0, soc_end=10.0, duration_minutes=5.0,
            energy_delivered_kwh=11.4,   # 136 kW
        )
        assert engine.stats.total_energy_discharged_kwh == 0

    def test_normal_discharge_is_recorded(self):
        engine = _engine(nominal_discharge_rate_kw=5.9)
        engine.record_discharging(
            soc_start=50.0, soc_end=40.0, duration_minutes=15.0,
        )
        assert engine.stats.total_energy_discharged_kwh == pytest.approx(1.43)


class TestQuantizationAwareFloor:
    """The floor is about counter granularity, not about wall time.

    ``cost_tracker`` re-stamps ``_last_sig_soc_time`` after every accepted
    event, so a genuine interval lasts only as long as the counter needs to
    advance one 0.1 kWh tick: 53 s at 6.8 kW. The old flat 1-minute floor
    therefore rejected precisely the 6.77-6.82 kW warm-pack cluster the 2x rate
    bound was tuned to keep.
    """

    def test_a_53_second_warm_pack_tick_is_accepted(self):
        engine = _engine()          # nominal 4.5 kW -> bound 9.0 kW
        # 0.1 kWh in 53 s = 6.79 kW, the real warm-battery rate.
        engine.record_charging(
            soc_start=30.0,
            soc_end=31.0,
            duration_minutes=53.0 / 60.0,
            energy_to_battery_kwh=0.1,
        )
        assert engine.stats.charge_rates_by_soc["25-50"] == pytest.approx(
            [6.792], rel=1e-3
        )
        assert engine._rejected_observations == 0

    def test_the_millisecond_production_sample_is_still_rejected(self):
        engine = _engine()
        # 0.1 kWh over 0.04 min = 150 kW. One counter tick, so it fails the
        # quantization gate; the rate bound stands behind it either way.
        engine.record_charging(
            soc_start=10.0,
            soc_end=11.0,
            duration_minutes=0.04,
            energy_to_battery_kwh=0.1,
        )
        assert engine.stats.charge_rates_by_soc == {}
        assert engine._rejected_observations == 1

    def test_a_multi_tick_delta_over_milliseconds_dies_on_the_rate_bound(self):
        """Passing the quantization gate is not passing verification."""
        messages = []
        engine = _engine(log_func=lambda msg, *a, **k: messages.append(msg))
        # 0.2 kWh = 2 counter ticks, so the quantization gate lets it through;
        # 0.2 kWh / 0.04 min = 300 kW must still be rejected.
        engine.record_charging(
            soc_start=10.0,
            soc_end=12.0,
            duration_minutes=0.04,
            energy_to_battery_kwh=0.2,
        )
        assert engine.stats.charge_rates_by_soc == {}
        assert engine._rejected_observations == 1
        assert any("rate 300.00 kW" in m for m in messages)

    def test_the_same_rule_applies_to_discharge(self):
        engine = _engine(nominal_discharge_rate_kw=5.9)
        engine.record_discharging(
            soc_start=50.0,
            soc_end=49.0,
            duration_minutes=53.0 / 60.0,
            energy_delivered_kwh=0.1,
        )
        assert engine.stats.total_energy_discharged_kwh == pytest.approx(0.1)
        assert engine._rejected_observations == 0

    def test_counter_resolution_is_configurable(self):
        # A 1 kWh-granular counter needs a 2 kWh delta to resolve a rate.
        engine = _engine(counter_resolution_kwh=1.0)
        engine.record_charging(
            soc_start=30.0,
            soc_end=31.0,
            duration_minutes=0.1,
            energy_to_battery_kwh=0.5,
        )
        assert engine.stats.charge_rates_by_soc == {}
        assert engine._rejected_observations == 1

    def test_resolvability_helper_states_its_verdict(self):
        engine = _engine()
        assert engine.observation_is_resolvable(0.1, 1.0) is None
        assert engine.observation_is_resolvable(0.2, 0.01) is None
        reason = engine.observation_is_resolvable(0.1, 0.01)
        assert reason is not None
        assert "counter resolution" in reason


class TestExportRateIsPartOfTheBound:
    """Export slots run at the export discharge rate, often the largest power."""

    def test_export_rate_raises_the_bound(self):
        engine = _engine(nominal_discharge_rate_kw=5.0, nominal_export_rate_kw=8.0)
        assert engine.max_plausible_rate_kw == pytest.approx(16.0)

    def test_a_genuine_export_slot_sample_survives(self):
        # 2.0 kWh out in 15 min = 8.0 kW: a max_export slot. Against the 5.0 kW
        # load-discharge rate alone (bound 10.0 kW) it would still fit, but at
        # the 1.5x warm-pack margin above it (12 kW) it would not.
        engine = _engine(nominal_discharge_rate_kw=5.0, nominal_export_rate_kw=8.0)
        engine.record_discharging(
            soc_start=50.0, soc_end=36.0, duration_minutes=15.0,
            energy_delivered_kwh=3.0,   # 12 kW
        )
        assert engine.stats.total_energy_discharged_kwh == pytest.approx(3.0)

    def test_without_the_export_rate_the_same_sample_is_rejected(self):
        engine = _engine(nominal_discharge_rate_kw=5.0)
        engine.record_discharging(
            soc_start=50.0, soc_end=36.0, duration_minutes=15.0,
            energy_delivered_kwh=3.0,   # 12 kW > 2 x 5.0
        )
        assert engine.stats.total_energy_discharged_kwh == 0

    def test_the_orchestrator_passes_the_effective_export_rate(self):
        """`effective_export_discharge_rate` is the property, not the raw field."""
        import inspect

        import battery_optimizer as bo

        source = inspect.getsource(bo.BatteryOptimizer.initialize)
        assert (
            "nominal_export_rate_kw=self.config.effective_export_discharge_rate"
            in source
        )


class TestPoisonedPersistedFile:
    """A file written before the ingest guards existed must not leak through."""

    @staticmethod
    def _poisoned_json() -> str:
        return json.dumps({
            "version": 6,
            "learned_efficiency": 0.948,
            "stats": {
                "charge_rates_by_soc": {"0-25": [2.7, 2.8, 2.9] + LIVE_0_25_ABOVE_20},
                "charge_rates_by_soc_temp": {
                    "0-25": {">20": list(LIVE_0_25_ABOVE_20)}
                },
                "thermal_coeffs": {"k1": 0.5, "k2": 9.9, "n": 40},
                "thermal_samples": [[30.0, 31.0, 0.01, 34535.0, 21.0]],
            },
        })

    def test_load_drops_the_implausible_observations(self):
        engine = _engine()
        assert engine.load_from_json(self._poisoned_json())
        assert engine.stats.charge_rates_by_soc_temp["0-25"][">20"] == [2.806, 5.959]
        assert all(
            r <= engine.max_plausible_rate_kw
            for r in engine.stats.charge_rates_by_soc["0-25"]
        )

    def test_load_drops_an_insane_thermal_fit_and_its_samples(self):
        engine = _engine()
        engine.load_from_json(self._poisoned_json())
        assert engine.stats.thermal_coeffs == {}
        assert engine.stats.thermal_samples == []

    def test_charge_rate_lookup_never_serves_the_poison(self):
        engine = _engine()
        engine.load_from_json(self._poisoned_json())
        # >20C bucket now has only 2 usable observations -> falls through to the
        # SOC-only fallback (median of [2.7, 2.8, 2.9, 2.806, 5.959]) instead of
        # answering 14308.71 kW.
        rate = engine.get_charge_rate_for_soc(10.0, 21.9)
        assert rate == pytest.approx(2.806)
        assert rate <= engine.max_plausible_rate_kw

    def test_bound_holds_even_without_the_load_time_filter(self):
        """Defence in depth: poison injected straight into the in-memory stats."""
        engine = _engine()
        engine.stats.charge_rates_by_soc_temp["0-25"] = {">20": list(LIVE_0_25_ABOVE_20)}
        engine.stats.charge_rates_by_soc["0-25"] = list(LIVE_0_25_ABOVE_20)
        rate = engine.get_charge_rate_for_soc(10.0, 21.9)
        assert rate <= engine.max_plausible_rate_kw

    def test_summary_reports_the_filtered_medians(self):
        engine = _engine()
        engine.load_from_json(self._poisoned_json())
        summary = engine.get_learning_summary()
        assert summary["soc_charge_rates"]["0-25"]["median_kw"] <= 9.0
        assert summary["max_plausible_rate_kw"] == pytest.approx(9.0)


class TestThermalGuards:
    def test_implausible_power_sample_is_rejected(self):
        engine = _engine()
        assert not engine.record_thermal_observation(
            temp_start=30.0, temp_end=31.0, duration_minutes=15.0,
            avg_power_kw=14308.71, ambient_temp=21.0,
        )
        assert engine.stats.thermal_samples == []

    def test_zero_power_sample_is_still_accepted(self):
        """Idle relaxation samples carry the k1 information — keep them."""
        engine = _engine()
        assert engine.record_thermal_observation(
            temp_start=30.0, temp_end=29.5, duration_minutes=15.0,
            avg_power_kw=0.0, ambient_temp=21.0,
        )

    def test_reset_thermal_calibration(self):
        engine = _engine()
        engine.stats.thermal_coeffs = {"k1": 0.01, "k2": 0.4, "n": 40}
        engine.stats.thermal_samples = [[30.0, 30.5, 15.0, 4.0, 25.0]]
        engine.reset_thermal_calibration()
        assert engine.stats.thermal_coeffs == {}
        assert engine.stats.thermal_samples == []

    def test_sane_fit_survives(self):
        assert thermal_coeffs_are_sane({"k1": 0.012, "k2": 2.27, "n": 40})
        assert not thermal_coeffs_are_sane({"k1": 0.5, "k2": 9.9, "n": 40})
        assert not thermal_coeffs_are_sane({})

    def test_measured_pack_heating_is_representable(self):
        """The reference pack measures 2.27 C/kWh (21.9C -> 25.8C over 1.716 kWh).

        The old ceiling of 2.0 clipped it, so the model could not reproduce the
        very slot it was calibrated against.
        """
        assert MAX_HEATING_C_PER_KWH > 2.27
