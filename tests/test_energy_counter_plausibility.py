"""Regression tests for the 2026-09-07 inverter energy-counter glitch.

Production log (add-on, 2026-09-07)::

    19:38:50.979 Initialized energy sensors: charge=18.80, discharge=18.40 kWh
    ... normal -0.100 kWh discharges every ~6 min, SOC 59 % -> 55 % ...
    20:18:29.593 Resyncing stored-energy accumulator 6.307 -> 25.535 kWh (drift)
    20:18:29.595 Battery discharged: -19.100 kWh [inverter]
    20:18:29.621 Resyncing stored-energy accumulator 6.435 -> 0.000 kWh (depleted)
    20:18:29.624 Battery charged: +18.800 kWh [inverter, no-pv-grid] at stored-energy
                 cost 0.3138 EUR/kWh, new avg cost: 0.3138 EUR/kWh

Both daily counters read 0 for one poll and came straight back.  The dip was a
negative delta, silently discarded by the ``energy_delta < 0.05`` noise floor;
the RETURN was then a ``+19.1`` / ``+18.8`` kWh forward delta and was booked as
real energy at a constant 55 % SOC on a pack with 12.87 kWh of usable capacity.
The landed cost basis was overwritten from a non-event.

The guard (`battery_optimizer_lib/energy_counter.py`) rejects both readings and
re-anchors on each, so the sequence books nothing in either direction while a
genuine reset and a genuine multi-slot catch-up still work.
"""

import datetime
import pathlib

import pytest

from battery_optimizer_lib import (
    BatteryCostConfig,
    BatteryCostTracker,
    CounterVerdict,
    EnergyCounterGuard,
    is_daily_counter_reset,
)

# The reference installation, as of the incident.
CAPACITY_KWH = 14.3          # usable at min_soc=10 %: 12.87 kWh
MIN_SOC = 10.0
MAX_SOC = 100.0
CHARGE_SENSOR = "sensor.growatt_battery_battery_charge_today"
DISCHARGE_SENSOR = "sensor.growatt_battery_battery_discharge_today"

# max(2.0, 0.25 * 14.3) -- the floor the tracker gives its guards.
EXPECTED_FLOOR_KWH = 3.575


class _DummyLearningEngine:
    def __init__(self):
        self.charge_calls = []
        self.discharge_calls = []

    def record_charging(self, **kwargs):
        self.charge_calls.append(kwargs)

    def record_discharging(self, **kwargs):
        self.discharge_calls.append(kwargs)

    def record_temperature_observation(self, temp):
        pass

    def record_cooling(self, **kwargs):
        pass

    def get_charge_rate_for_soc(self, soc, temp=None):
        return 4.5


class _Harness:
    """A BatteryCostTracker wired to mutable clock / SOC / sensor state."""

    def __init__(self, start=datetime.datetime(2026, 9, 7, 19, 38, 50), soc=55.0):
        self.now = start
        self.soc = soc
        self.state = {
            CHARGE_SENSOR: "18.80",
            DISCHARGE_SENSOR: "18.40",
        }
        self.logs = []
        self.services = []
        self.engine = _DummyLearningEngine()

        config = BatteryCostConfig(
            battery_charge_sensor=CHARGE_SENSOR,
            battery_discharge_sensor=DISCHARGE_SENSOR,
            use_inverter_energy_sensors=True,
            battery_capacity=CAPACITY_KWH,
            charge_rate=4.5,
            discharge_rate=4.5,
            efficiency=0.85,
            inverter_efficiency=0.95,
            slot_minutes=15,
            default_cost=0.0248,
        )

        self.tracker = BatteryCostTracker(
            config=config,
            get_state_func=lambda e: self.state.get(e),
            call_service_func=lambda *a, **k: self.services.append((a, k)),
            get_datetime_func=lambda: self.now,
            get_timezone_func=lambda: None,
            align_to_slot_func=lambda dt: dt.replace(
                minute=(dt.minute // 15) * 15, second=0, microsecond=0
            ),
            get_min_soc_func=lambda: MIN_SOC,
            get_max_soc_func=lambda: MAX_SOC,
            get_current_soc_func=lambda: self.soc,
            get_battery_temp_func=lambda: 22.0,
            learning_engine=self.engine,
            get_cached_prices_func=lambda: [],
            save_learning_data_func=lambda: None,
            update_learning_sensor_func=lambda: None,
            log_func=self.log,
        )
        self.tracker.initialize()
        self.tracker._avg_cost = 0.0248
        self.tracker._cost_from_fallback = False
        # SOC 55 % on a 14.3 kWh pack above a 10 % floor.
        self.tracker._stored_energy_kwh = self.tracker._soc_to_energy_kwh(soc)

    # -- helpers -----------------------------------------------------------

    def log(self, msg, level="INFO"):
        self.logs.append((level, msg))

    def advance(self, minutes):
        self.now += datetime.timedelta(minutes=minutes)

    def report(self, entity, value):
        """Deliver one HA state change for `entity`, exactly as the app sees it."""
        old = self.state.get(entity)
        self.state[entity] = f"{value:.2f}"
        self.tracker.on_energy_sensor_change(entity, old, self.state[entity])

    def warnings(self):
        return [m for lvl, m in self.logs if lvl == "WARNING"]

    def messages(self):
        return [m for _, m in self.logs]

    def booked(self):
        return [
            m
            for m in self.messages()
            if m.startswith("Battery charged:") or m.startswith("Battery discharged:")
        ]

    def resyncs(self):
        return [m for m in self.messages() if m.startswith("Resyncing")]

    def clear_log(self):
        self.logs.clear()


def _drain_normal_discharges(h, steps=3):
    """The `-0.100 kWh` steps every ~6 min that precede the glitch."""
    value = 18.40
    for _ in range(steps):
        h.advance(6)
        value += 0.10
        h.soc -= 1.0
        h.report(DISCHARGE_SENSOR, value)
    return value


# =========================================================================
# The production sequence
# =========================================================================


class TestProductionGlitch:
    def test_normal_steps_are_still_booked(self):
        h = _Harness()
        _drain_normal_discharges(h, steps=3)
        assert h.booked() == ["Battery discharged: -0.100 kWh [inverter]"] * 3
        assert h.warnings() == []

    def test_value_zero_value_books_nothing_on_the_discharge_counter(self):
        h = _Harness()
        value = _drain_normal_discharges(h, steps=7)  # 18.40 -> 19.10
        assert value == pytest.approx(19.10)
        stored_before = h.tracker._stored_energy_kwh
        cost_before = h.tracker.avg_cost
        h.clear_log()
        h.engine.discharge_calls.clear()

        # One bad poll reads 0, the next reads the real total back.
        h.advance(3)
        h.report(DISCHARGE_SENSOR, 0.0)
        h.advance(0.001)
        h.report(DISCHARGE_SENSOR, 19.10)

        assert h.booked() == []
        assert h.resyncs() == []
        assert h.tracker.avg_cost == cost_before
        assert h.tracker._stored_energy_kwh == pytest.approx(stored_before)
        assert h.engine.discharge_calls == []
        # One WARNING per implausible reading: the drop, then the return.
        warnings = h.warnings()
        assert len(warnings) == 2
        assert "went BACKWARDS" in warnings[0]
        assert "JUMPED" in warnings[1]
        assert all(DISCHARGE_SENSOR in w for w in warnings)

    def test_value_zero_value_books_nothing_on_the_charge_counter(self):
        h = _Harness()
        _drain_normal_discharges(h, steps=7)
        cost_before = h.tracker.avg_cost
        stored_before = h.tracker._stored_energy_kwh
        h.clear_log()

        # The charge counter had not moved since the 19:38 restart.
        h.advance(3)
        h.report(CHARGE_SENSOR, 0.0)
        h.advance(0.001)
        h.report(CHARGE_SENSOR, 18.80)

        assert h.booked() == []
        # This is the number that reached input_number.battery_avg_cost.
        assert h.tracker.avg_cost == cost_before
        assert h.tracker._stored_energy_kwh == pytest.approx(stored_before)
        assert h.services == []  # save_to_ha() never ran
        assert h.engine.charge_calls == []
        assert len(h.warnings()) == 2

    def test_the_whole_incident_replayed(self):
        """Both counters glitch within the same 30 ms, as they did in production."""
        h = _Harness()
        _drain_normal_discharges(h, steps=7)
        cost_before = h.tracker.avg_cost
        stored_before = h.tracker._stored_energy_kwh
        h.clear_log()

        h.advance(3)
        h.report(DISCHARGE_SENSOR, 0.0)
        h.report(CHARGE_SENSOR, 0.0)
        h.report(DISCHARGE_SENSOR, 19.10)
        h.report(CHARGE_SENSOR, 18.80)

        assert h.booked() == []
        assert h.resyncs() == []
        assert h.tracker.avg_cost == cost_before
        assert h.tracker._stored_energy_kwh == pytest.approx(stored_before)
        assert len(h.warnings()) == 4

        # ... and the very next genuine 0.100 kWh step is booked normally.
        h.clear_log()
        h.advance(6)
        h.soc -= 1.0
        h.report(DISCHARGE_SENSOR, 19.20)
        assert h.booked() == ["Battery discharged: -0.100 kWh [inverter]"]
        assert h.warnings() == []
        assert h.tracker._stored_energy_kwh == pytest.approx(stored_before - 0.1)

    def test_doubling_then_returning_books_nothing(self):
        """value -> 2*value -> value, the other plausible sensor failure shape."""
        h = _Harness()
        _drain_normal_discharges(h, steps=3)
        cost_before = h.tracker.avg_cost
        stored_before = h.tracker._stored_energy_kwh
        h.clear_log()

        h.advance(3)
        h.report(CHARGE_SENSOR, 37.60)
        h.advance(0.001)
        h.report(CHARGE_SENSOR, 18.80)

        assert h.booked() == []
        assert h.tracker.avg_cost == cost_before
        assert h.tracker._stored_energy_kwh == pytest.approx(stored_before)
        assert h.services == []
        warnings = h.warnings()
        assert len(warnings) == 2
        assert "JUMPED" in warnings[0]
        assert "went BACKWARDS" in warnings[1]

    def test_warning_names_the_values_the_delta_and_the_bound(self):
        h = _Harness()
        _drain_normal_discharges(h, steps=7)
        h.clear_log()
        h.advance(3)
        h.report(DISCHARGE_SENSOR, 0.0)
        h.advance(0.001)
        h.report(DISCHARGE_SENSOR, 19.10)

        jump = h.warnings()[1]
        assert "0.000" in jump and "19.100" in jump      # raw values
        assert "+19.100" in jump                          # the delta
        assert "bound" in jump                            # and what it violated
        assert f"{EXPECTED_FLOOR_KWH:.3f}" in jump

    def test_a_persistent_offset_costs_exactly_one_rejection(self):
        """If the counter jumps and STAYS, deltas resume from the new level."""
        h = _Harness()
        _drain_normal_discharges(h, steps=3)
        h.clear_log()

        h.advance(3)
        h.report(DISCHARGE_SENSOR, 118.70)   # +100 kWh, rejected
        assert len(h.warnings()) == 1
        assert h.booked() == []

        h.clear_log()
        h.advance(6)
        h.soc -= 1.0
        h.report(DISCHARGE_SENSOR, 118.80)   # +0.100 off the NEW baseline
        assert h.warnings() == []
        assert h.booked() == ["Battery discharged: -0.100 kWh [inverter]"]


# =========================================================================
# What must keep working
# =========================================================================


class TestLegitimateReadings:
    def test_midnight_reset_still_works(self):
        h = _Harness(start=datetime.datetime(2026, 9, 7, 23, 50, 0))
        h.advance(6)
        h.soc -= 1.0
        h.report(DISCHARGE_SENSOR, 18.50)
        h.clear_log()

        # 00:02 local, counter rolls over to 0.
        h.advance(6)
        assert h.now.hour == 0 and h.now.minute == 2
        h.report(DISCHARGE_SENSOR, 0.0)

        assert any(m.startswith("Midnight reset on") for m in h.messages())
        assert h.warnings() == []
        assert h.booked() == []

        # The new day's first real step is measured from 0.
        h.clear_log()
        h.advance(6)
        h.soc -= 1.0
        h.report(DISCHARGE_SENSOR, 0.10)
        assert h.booked() == ["Battery discharged: -0.100 kWh [inverter]"]

    def test_a_2kwh_catchup_after_30_minutes_is_accepted(self):
        """A stalled sensor that catches up several slots at once is real energy."""
        h = _Harness()
        h.advance(30)
        h.soc -= 14.0
        h.report(DISCHARGE_SENSOR, 20.40)   # +2.000 kWh

        assert h.booked() == ["Battery discharged: -2.000 kWh [inverter]"]
        assert h.warnings() == []

    def test_a_1kwh_catchup_after_one_slot_is_accepted(self):
        h = _Harness()
        h.advance(15)
        h.soc -= 7.0
        h.report(CHARGE_SENSOR, 19.80)      # +1.000 kWh

        assert any(m.startswith("Battery charged:") for m in h.messages())
        assert h.warnings() == []

    def test_sub_resolution_rounding_backwards_is_not_a_warning(self):
        h = _Harness()
        h.advance(6)
        h.report(DISCHARGE_SENSOR, 18.38)   # -0.02 kWh, counter rounding
        assert h.warnings() == []
        assert h.booked() == []

    def test_sensor_recovery_reanchors_without_booking(self):
        """A long unavailable window is not a charge, however far the counter moved."""
        h = _Harness()
        h.advance(5)
        h.tracker.on_energy_sensor_change(CHARGE_SENSOR, "18.80", "unavailable")
        assert h.tracker.is_energy_sensor_available is False

        h.advance(120)
        h.state[CHARGE_SENSOR] = "30.00"
        h.tracker.on_energy_sensor_change(CHARGE_SENSOR, "unavailable", "30.00")

        assert h.booked() == []
        assert h.warnings() == []
        assert h.tracker._charge_counter.last_value == pytest.approx(30.0)

        # ... and deltas resume from the recovered value.
        h.clear_log()
        h.advance(6)
        h.soc += 1.0
        h.report(CHARGE_SENSOR, 30.20)
        assert any(m.startswith("Battery charged:") for m in h.messages())


# =========================================================================
# The guard itself
# =========================================================================


class TestEnergyCounterGuard:
    def _guard(self, **kw):
        kw.setdefault("max_rate_kw", 4.5)
        kw.setdefault("min_bound_kwh", EXPECTED_FLOOR_KWH)
        return EnergyCounterGuard(**kw)

    def test_bound_is_the_max_of_floor_and_rate_times_elapsed(self):
        g = self._guard()
        assert g.bound_for(0.0) == pytest.approx(EXPECTED_FLOOR_KWH)
        assert g.bound_for(0.05) == pytest.approx(EXPECTED_FLOOR_KWH)   # 0.45 < floor
        assert g.bound_for(0.5) == pytest.approx(4.5)                   # 4.5 kW * .5h * 2
        assert g.bound_for(2.0) == pytest.approx(18.0)

    def test_first_reading_only_sets_the_baseline(self):
        g = self._guard()
        v = g.observe(18.8, datetime.datetime(2026, 9, 7, 19, 38))
        assert v.kind == CounterVerdict.FIRST
        assert not v.accepted and not v.is_glitch
        assert v.delta_kwh == 0.0
        assert g.last_value == 18.8

    def test_daily_counter_may_exceed_pack_capacity_over_a_long_gap(self):
        """A DAILY total legitimately outruns the pack: 4.5 kW for 5 h = 22.5 kWh."""
        g = self._guard()
        t = datetime.datetime(2026, 9, 7, 8, 0)
        g.observe(0.0, t)
        v = g.observe(20.0, t + datetime.timedelta(hours=5))
        assert v.accepted
        assert v.delta_kwh == pytest.approx(20.0)

    def test_rejection_reanchors_in_both_directions(self):
        g = self._guard()
        t = datetime.datetime(2026, 9, 7, 20, 15)
        g.observe(19.1, t)

        drop = g.observe(0.0, t + datetime.timedelta(minutes=3))
        assert drop.kind == CounterVerdict.IMPLAUSIBLE_DROP
        assert g.last_value == 0.0

        back = g.observe(19.1, t + datetime.timedelta(minutes=3, seconds=1))
        assert back.kind == CounterVerdict.IMPLAUSIBLE_JUMP
        assert g.last_value == 19.1
        assert back.delta_kwh == 0.0

        ok = g.observe(19.2, t + datetime.timedelta(minutes=9))
        assert ok.accepted and ok.delta_kwh == pytest.approx(0.1)

    def test_naive_and_aware_timestamps_do_not_raise(self):
        g = self._guard()
        aware = datetime.datetime(2026, 9, 7, 20, 15, tzinfo=datetime.timezone.utc)
        g.observe(1.0, aware)
        v = g.observe(1.1, datetime.datetime(2026, 9, 7, 20, 21))
        assert v.accepted

    def test_a_clock_that_goes_backwards_falls_back_to_the_floor(self):
        g = self._guard()
        t = datetime.datetime(2026, 9, 7, 20, 15)
        g.observe(1.0, t)
        v = g.observe(2.0, t - datetime.timedelta(minutes=30))
        assert v.elapsed_hours == 0.0
        assert v.bound_kwh == pytest.approx(EXPECTED_FLOOR_KWH)
        assert v.accepted


class TestMidnightRule:
    def test_reset_needs_both_the_window_and_a_small_value(self):
        near = datetime.datetime(2026, 9, 8, 0, 2)
        assert is_daily_counter_reset(0.0, 19.1, near) is True
        assert is_daily_counter_reset(5.0, 19.1, near) is False  # too large
        assert is_daily_counter_reset(0.0, 19.1, near.replace(hour=20)) is False
        # 23:58 is inside the window on the other side.
        assert is_daily_counter_reset(0.0, 19.1, datetime.datetime(2026, 9, 7, 23, 58))
        # A rise is never a reset.
        assert is_daily_counter_reset(19.1, 0.0, near) is False


# =========================================================================
# Wiring: the orchestrator is not unit-tested, so scan the source
# =========================================================================

_APPS = pathlib.Path(__file__).resolve().parents[1] / "appdaemon" / "apps"


class TestWiring:
    def test_cost_tracker_derives_its_delta_from_the_guard(self):
        src = (_APPS / "battery_optimizer_lib" / "cost_tracker.py").read_text(
            encoding="utf-8"
        )
        assert "from .energy_counter import" in src
        assert "self._counter_for(entity).observe(" in src
        assert "verdict.is_glitch" in src
        assert "energy_delta = verdict.delta_kwh" in src
        # The old, unguarded delta must not come back.
        assert "current_value - old_value" not in src

    def test_orchestrator_delegates_the_listener_to_the_tracker(self):
        src = (_APPS / "battery_optimizer.py").read_text(encoding="utf-8")
        assert "self._cost_tracker.on_energy_sensor_change(entity, old, new)" in src
        # And it must not derive an energy delta of its own.
        assert "float(new) - float(old)" not in src
