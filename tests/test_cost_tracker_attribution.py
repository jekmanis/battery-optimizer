"""Source attribution for measured battery charging.

Production defect (2026-09-02, log lines 511/550/552): the app sent
``grid_charge ... duration=20min`` at 05:00:05, transitioned to HOLD at
05:15:05, and five seconds later — an hour before sunrise — booked a +0.1 kWh
delta as ``[inverter, pv]`` at 0.0253 EUR/kWh, dragging the landed basis from
0.1261 to 0.1199 EUR/kWh.

``_observed_charge_cost`` attributed purely from ``self._current_mode``: any
charge measured under HOLD/DISCHARGE was "pv". The mode the app believes it is
in is not the mode the inverter is executing, and "not commanded to grid-charge"
is not evidence of sunshine. Two guards now stand between them, and both fail
toward the (conservative) grid cost.
"""

import datetime

import pytest

from battery_optimizer_lib import (
    BatteryCostConfig,
    BatteryCostTracker,
    BatteryLearningEngine,
    BatteryMode,
)
from battery_optimizer_lib.config import BatteryOptimizerConfig


SPOT = 0.10
EFFICIENCY = 0.95
INVERTER_EFFICIENCY = 0.97
GRID_FEE = 0.052
EXPORT_FEE = 0.02


def _make_tracker(pv_w=None, grid_charge_active=None, **config_overrides):
    """Cost tracker with injectable PV power and grid-charge window."""
    config = BatteryCostConfig(
        battery_capacity=14.3,
        efficiency=EFFICIENCY,
        inverter_efficiency=INVERTER_EFFICIENCY,
        grid_fee=GRID_FEE,
        grid_export_fee=EXPORT_FEE,
        slot_minutes=15,
        **config_overrides,
    )
    state = {"pv_w": pv_w, "grid": grid_charge_active}

    tracker = BatteryCostTracker(
        config=config,
        get_state_func=lambda e: None,
        call_service_func=lambda *a, **k: None,
        get_datetime_func=lambda: datetime.datetime(2026, 9, 2, 5, 15),
        get_timezone_func=lambda: None,
        align_to_slot_func=lambda dt: dt.replace(minute=0, second=0, microsecond=0),
        get_min_soc_func=lambda: 10.0,
        get_max_soc_func=lambda: 100.0,
        get_current_soc_func=lambda: 11.0,
        get_battery_temp_func=lambda: 22.0,
        learning_engine=BatteryLearningEngine(
            battery_capacity_kwh=14.3,
            nominal_charge_rate_kw=4.5,
            nominal_efficiency=EFFICIENCY,
        ),
        get_cached_prices_func=lambda: [],
        save_learning_data_func=lambda: None,
        update_learning_sensor_func=lambda: None,
        log_func=lambda *a, **k: None,
        get_pv_power_w_func=(
            None if pv_w is None and "pv_w" not in state else (lambda: state["pv_w"])
        ),
        grid_charge_active_func=(
            None if grid_charge_active is None else (lambda: state["grid"])
        ),
    )
    return tracker, state


def _grid_cost():
    return (SPOT + GRID_FEE) / (EFFICIENCY * INVERTER_EFFICIENCY)


def _pv_cost():
    return (SPOT - EXPORT_FEE) / EFFICIENCY


class TestPvRequiresSun:
    def test_charge_in_hold_before_sunrise_is_not_pv(self):
        """The reported defect: 0 W measured PV, yet booked at PV cost."""
        tracker, _ = _make_tracker(pv_w=0.0)
        tracker._current_mode = BatteryMode.HOLD

        cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "no-pv-grid"
        assert cost == pytest.approx(_grid_cost())

    def test_charge_in_discharge_mode_below_the_floor_is_not_pv(self):
        tracker, _ = _make_tracker(pv_w=60.0, pv_attribution_min_w=100.0)
        tracker._current_mode = BatteryMode.DISCHARGE

        _cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "no-pv-grid"

    def test_charge_with_real_pv_production_is_still_pv(self):
        tracker, _ = _make_tracker(pv_w=2400.0)
        tracker._current_mode = BatteryMode.DISCHARGE

        cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "pv"
        assert cost == pytest.approx(_pv_cost())

    def test_pv_exactly_at_the_floor_counts_as_pv(self):
        tracker, _ = _make_tracker(pv_w=100.0, pv_attribution_min_w=100.0)
        tracker._current_mode = BatteryMode.HOLD

        _cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "pv"

    def test_unavailable_pv_sensor_keeps_the_legacy_attribution(self):
        """A momentarily dead sensor must not invent a grid charge."""
        tracker, state = _make_tracker(pv_w=0.0)
        state["pv_w"] = None
        tracker._current_mode = BatteryMode.HOLD

        cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "pv"
        assert cost == pytest.approx(_pv_cost())

    def test_no_pv_provider_injected_keeps_the_legacy_attribution(self):
        tracker, _ = _make_tracker()
        tracker._current_mode = BatteryMode.HOLD

        _cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "pv"

    def test_a_raising_pv_provider_is_treated_as_unavailable(self):
        def boom():
            raise RuntimeError("sensor exploded")

        tracker, _ = _make_tracker(pv_w=0.0)
        tracker._get_pv_power_w = boom
        tracker._current_mode = BatteryMode.HOLD

        _cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "pv"


class TestGridChargeStillInForce:
    def test_charge_after_the_transition_to_hold_is_still_grid(self):
        """grid_charge runs for 20 min; the app moved to HOLD after 15."""
        tracker, _ = _make_tracker(pv_w=0.0, grid_charge_active=True)
        tracker._current_mode = BatteryMode.HOLD

        cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "grid-command"
        assert cost == pytest.approx(_grid_cost())

    def test_measured_pv_beats_the_window(self):
        """The window is a TIME bound on a superseded command, not evidence.

        Two minutes after a midday CHARGE -> HOLD transition the window is
        still open, but 4 kW of measured PV is what is actually charging the
        battery. Booking that at the grid price overstated the basis exactly as
        badly as the pre-dawn case understated it.
        """
        tracker, _ = _make_tracker(pv_w=4000.0, grid_charge_active=True)
        tracker._current_mode = BatteryMode.HOLD

        cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "pv"
        assert cost == pytest.approx(_pv_cost())

    def test_the_window_holds_when_the_sun_is_down(self):
        """Same window, PV 0 W: nothing but the grid can explain the energy."""
        tracker, _ = _make_tracker(pv_w=0.0, grid_charge_active=True)
        tracker._current_mode = BatteryMode.HOLD

        cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "grid-command"
        assert cost == pytest.approx(_grid_cost())

    def test_the_window_holds_below_the_pv_floor(self):
        tracker, _ = _make_tracker(
            pv_w=99.0, grid_charge_active=True, pv_attribution_min_w=100.0
        )
        tracker._current_mode = BatteryMode.DISCHARGE

        _cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "grid-command"

    def test_the_window_holds_when_no_pv_reading_exists(self):
        """No PV provider injected: the window is the only evidence left."""
        tracker, _ = _make_tracker(grid_charge_active=True)
        tracker._current_mode = BatteryMode.DISCHARGE

        _cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "grid-command"

    def test_once_the_window_expires_pv_attribution_resumes(self):
        tracker, state = _make_tracker(pv_w=3000.0, grid_charge_active=True)
        tracker._current_mode = BatteryMode.DISCHARGE
        state["grid"] = False

        _cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "pv"

    def test_commanded_charge_still_reports_plain_grid(self):
        tracker, _ = _make_tracker(pv_w=3000.0, grid_charge_active=True)
        tracker._current_mode = BatteryMode.CHARGE

        _cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "grid"

    def test_unknown_mode_is_still_conservative_grid(self):
        tracker, _ = _make_tracker(pv_w=3000.0)
        tracker._current_mode = None

        cost, source = tracker._observed_charge_cost(SPOT)

        assert source == "unknown-grid"
        assert cost == pytest.approx(_grid_cost())


class TestEndToEndBasisMovement:
    """The defect as it appeared in the log: the basis must not fall."""

    def test_pre_dawn_charge_no_longer_drags_the_basis_down(self):
        tracker, _ = _make_tracker(pv_w=0.0, grid_charge_active=True)
        tracker._current_mode = BatteryMode.HOLD
        tracker._avg_cost = 0.1261
        tracker._stored_energy_kwh = 1.9
        tracker._last_soc = 20.0
        tracker._last_price_slot = None  # forces the "price unavailable" branch

        # With a price available the attributed cost must be the grid cost,
        # which is ABOVE the current basis, so the average cannot fall.
        cost, source = tracker._observed_charge_cost(SPOT)
        assert source == "grid-command"
        assert cost > tracker._avg_cost

    def test_learning_engine_is_offered_no_synthetic_grid_energy(self):
        """A grid charge must NOT hand the engine a derived AC figure.

        This used to pass ``energy_kwh / config.efficiency``, whose quotient
        with ``energy_kwh`` is the configured efficiency by construction. The
        engine then "learned" the constant it was configured with and reported
        it as an observation. There is no independent AC meter reading for the
        charge interval, so nothing is offered.
        """
        recorded = {}
        tracker, _ = _make_tracker(pv_w=0.0)
        tracker._current_mode = BatteryMode.HOLD
        tracker._stored_energy_kwh = 1.0
        tracker._last_soc = 20.0

        def fake_record_charging(**kwargs):
            recorded.update(kwargs)

        tracker._learning_engine.record_charging = fake_record_charging
        tracker._get_price_for_slot = lambda slot: SPOT
        tracker._last_price_slot = datetime.datetime(2026, 9, 2, 5, 0)

        tracker._process_energy_change(
            energy_kwh=0.1,
            is_charge=True,
            current_soc=21.0,
            now=datetime.datetime(2026, 9, 2, 5, 15, 10),
        )

        assert recorded["energy_from_grid_kwh"] is None
        # The stored-side measurement is still handed over: that one IS a
        # measurement, and it is what the charge-rate curve is learned from.
        assert recorded["energy_to_battery_kwh"] == pytest.approx(0.1)


class TestLearningBaselineOwnership:
    """`_last_sig_*` belongs to the energy path when energy sensors are live.

    HA delivers the SOC state change BEFORE the energy-counter change: on
    2026-09-02 the SOC listener ran at 05:02:13.724 and the energy callback
    44 ms later at 05:02:13.768. `process_soc_change` used to re-stamp the
    learning baseline in that gap, so `record_charging` measured a
    44-millisecond charge with `soc_start == soc_end` — the live learning file
    holds 34 535 kW and 44 653 kW observations because of it.
    """

    def _tracker_with_clock(self, clock):
        tracker, _ = _make_tracker(pv_w=0.0)
        tracker._get_datetime = lambda: clock["now"]
        tracker._energy_sensor_available = True
        tracker._current_mode = BatteryMode.CHARGE
        tracker._stored_energy_kwh = 1.0
        return tracker

    def test_soc_listener_does_not_reset_the_learning_baseline(self):
        clock = {"now": datetime.datetime(2026, 9, 2, 5, 2, 13, 724000)}
        tracker = self._tracker_with_clock(clock)
        baseline_time = datetime.datetime(2026, 9, 2, 5, 1, 30)
        tracker._last_soc = 10.0
        tracker._last_sig_soc = 9.0
        tracker._last_sig_soc_time = baseline_time
        tracker._last_sig_temp = 21.9

        tracker.process_soc_change(11.0)

        assert tracker._last_sig_soc == 9.0
        assert tracker._last_sig_soc_time == baseline_time
        assert tracker._last_sig_temp == 21.9
        # The cheap per-observation state IS updated.
        assert tracker._last_soc == 11.0

    def test_duration_is_measured_from_the_previous_energy_event(self):
        clock = {"now": datetime.datetime(2026, 9, 2, 5, 2, 13, 724000)}
        tracker = self._tracker_with_clock(clock)
        tracker._last_soc = 10.0
        tracker._last_sig_soc = 9.0
        tracker._last_sig_soc_time = datetime.datetime(2026, 9, 2, 5, 1, 30)
        tracker._last_sig_temp = 21.9

        recorded = {}
        tracker._learning_engine.record_charging = lambda **kw: recorded.update(kw)
        tracker._get_price_for_slot = lambda slot: SPOT
        tracker._last_price_slot = datetime.datetime(2026, 9, 2, 5, 0)

        # 1. SOC listener fires first ...
        tracker.process_soc_change(11.0)
        # 2. ... the energy counter follows 44 ms later.
        clock["now"] = datetime.datetime(2026, 9, 2, 5, 2, 13, 768000)
        tracker._process_energy_change(
            energy_kwh=0.1, is_charge=True, current_soc=11.0, now=clock["now"]
        )

        assert recorded["duration_minutes"] == pytest.approx(43.768 / 60, abs=1e-3)
        assert recorded["soc_start"] == 9.0
        assert recorded["soc_end"] == 11.0
        assert recorded["battery_temp_start"] == 21.9


class TestConfigPlumbing:
    def test_main_config_defaults_reach_the_cost_config(self):
        cfg = BatteryCostConfig.from_main_config(BatteryOptimizerConfig())

        assert cfg.pv_attribution_min_w == 100.0
        assert cfg.grid_charge_grace_seconds == 120

    def test_main_config_overrides_reach_the_cost_config(self):
        main = BatteryOptimizerConfig.from_args(
            {
                "cost_pv_attribution_min_w": 250,
                "cost_grid_charge_grace_seconds": 300,
            }
        )
        cfg = BatteryCostConfig.from_main_config(main)

        assert cfg.pv_attribution_min_w == 250.0
        assert cfg.grid_charge_grace_seconds == 300


class TestResyncMessageNamesTheInTransitEnergy:
    """Two resyncs at the same SOC are not a contradiction — say why."""

    def test_charge_event_message_states_the_delta(self):
        messages = []
        tracker, _ = _make_tracker(pv_w=0.0)
        tracker._log = lambda msg, level="INFO": messages.append(msg)
        tracker._stored_energy_kwh = 0.143

        tracker._resync_stored_energy(11.0, energy_in_transit_kwh=0.1)

        assert len(messages) == 1
        assert "0.143 -> 0.043 kWh" in messages[0]
        assert "SOC 11.0% = 0.143 kWh" in messages[0]
        assert "less the 0.100 kWh charged in this event" in messages[0]

    def test_discharge_event_message_states_the_delta(self):
        messages = []
        tracker, _ = _make_tracker(pv_w=0.0)
        tracker._log = lambda msg, level="INFO": messages.append(msg)
        # Far enough from the SOC-derived value to trip the gross-drift net.
        tracker._stored_energy_kwh = 6.0

        tracker._resync_stored_energy(10.5, energy_in_transit_kwh=-0.1)

        assert "plus the 0.100 kWh discharged in this event" in messages[0]
        assert "(drift)" in messages[0]

    def test_plain_soc_observation_has_no_transit_clause(self):
        messages = []
        tracker, _ = _make_tracker(pv_w=0.0)
        tracker._log = lambda msg, level="INFO": messages.append(msg)
        tracker._stored_energy_kwh = 0.1

        tracker._resync_stored_energy(11.0)

        assert "in this event" not in messages[0]
        assert "SOC 11.0% = 0.143 kWh" in messages[0]
