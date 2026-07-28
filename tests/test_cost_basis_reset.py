"""Regressions for DEFECT 9(b): the cost basis survived a full depletion.

Production evidence (33h AppDaemon window): the tracked battery cost decayed
0.0892 -> 0.0682 -> 0.0090 -> 0.0017 -> 0.0009 -> 0.0000 and then stuck. On
2026-07-28 11:12 the battery genuinely hit bottom ("Safety: HOLD (battery
depleted at 10.0%)") yet the basis stayed at 0.0009 — a depleted battery must
take the price of whatever is charged into it next, with no memory.

The weighted-average formula itself is correct; the accumulator feeding it was
not. `_stored_energy_kwh` is only seeded at initialize()/sensor-recovery and is
otherwise a pure running total: deltas below 0.05 kWh are dropped as noise, the
midnight counter reset skips a delta, and conversion losses are unmodelled. So
at SOC == min_soc it could still claim several kWh of phantom stored energy,
against which the next charge was averaged.
"""

import datetime

import pytest

from battery_optimizer_lib import BatteryCostTracker, BatteryMode

from tests.test_inverter_energy_tracking import make_cost_tracker


# ---------------------------------------------------------------------------
# The formula (unchanged — documented so a future refactor can't silently
# reintroduce the bug at this level instead).
# ---------------------------------------------------------------------------

def test_weighted_avg_resets_when_old_energy_is_zero():
    """With no energy left, the new average IS the new energy's price."""
    result = BatteryCostTracker._compute_weighted_avg_cost(
        old_energy=0.0, old_avg_cost=0.0009, added_energy=2.0, added_price=0.1500
    )
    assert result == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# The accumulator (the actual defect)
# ---------------------------------------------------------------------------

def test_depleted_battery_resets_cost_basis_despite_stale_accumulator():
    """A charge after a real depletion must not blend in the degenerate basis.

    Reproduces the logged numbers: min_soc = 10%, a degenerate basis of 0.0009
    EUR/kWh, and an accumulator that still claims 3 kWh after the battery hit
    bottom. Before the fix the new average came out at ~0.0009 + a sliver;
    afterwards it is exactly the charge's landed cost.
    """
    tracker, _state, _soc = make_cost_tracker(
        battery_capacity=14.3, min_soc=10.0, efficiency=1.0, initial_cost=0.0009
    )
    tracker._avg_cost = 0.0009
    tracker._stored_energy_kwh = 3.0          # drifted accumulator
    tracker._current_mode = BatteryMode.CHARGE

    # Battery was at min_soc (10%) and 1.0 kWh has just been charged into it,
    # so the SOC sensor now reads 10% + 1.0/14.3 -> ~17.0%.
    soc_after = 10.0 + (1.0 / 14.3) * 100
    tracker._process_energy_change(
        energy_kwh=1.0,
        is_charge=True,
        current_soc=soc_after,
        now=datetime.datetime(2026, 7, 28, 12, 0),
    )

    # No cached prices -> the charge is booked at the preserved average, which
    # after the reset is the ONLY contributor. The point is that the phantom
    # 3 kWh no longer participates: stored energy equals the charged amount.
    assert tracker._stored_energy_kwh == pytest.approx(1.0, abs=0.05)


def test_depleted_battery_takes_the_new_charge_price():
    """The reset is visible in the average itself, not just the accumulator."""
    tracker, _state, _soc = make_cost_tracker(
        battery_capacity=14.3, min_soc=10.0, efficiency=1.0, initial_cost=0.0009
    )
    tracker._avg_cost = 0.0009
    tracker._stored_energy_kwh = 3.0
    tracker._current_mode = BatteryMode.CHARGE

    slot = datetime.datetime(2026, 7, 28, 12, 0)
    # Grid charging at a 0.10 EUR/kWh spot price. BatteryCostConfig in the
    # shared harness uses grid_fee=0.05 and efficiency 1.0.
    landed = tracker._grid_landed_cost(0.10)
    tracker._get_price_for_slot = lambda _slot: 0.10
    tracker._last_price_slot = slot

    tracker._process_energy_change(
        energy_kwh=1.0,
        is_charge=True,
        current_soc=10.0 + (1.0 / 14.3) * 100,
        now=slot,
    )

    assert tracker.avg_cost == pytest.approx(landed, abs=1e-4)
    # Sanity: without the reset the 3 phantom kWh at 0.0009 would have pulled
    # this down to roughly a quarter of the landed cost.
    assert tracker.avg_cost > landed * 0.9


def test_soc_observation_at_min_soc_clears_the_accumulator():
    """Depletion is caught by the SOC listener, before any later charge."""
    tracker, _state, soc_holder = make_cost_tracker(
        battery_capacity=14.3, min_soc=10.0, efficiency=1.0
    )
    tracker._stored_energy_kwh = 3.0
    tracker._last_soc = 25.0
    soc_holder[0] = 10.0

    tracker.process_soc_change(10.0)

    assert tracker._stored_energy_kwh == pytest.approx(0.0)


def test_accumulator_within_tolerance_is_not_resynced():
    """Normal operation must keep using measured inverter energy, not SOC.

    The accumulator is the better weighting signal (measured DC energy, not a
    1%-granular SOC sensor), so ordinary slot-to-slot mismatch must NOT pull it
    around. Guards test_avg_cost_accumulates_with_multiple_energy_deltas.
    """
    tracker, _state, _soc = make_cost_tracker(
        battery_capacity=10.0, min_soc=0.0, efficiency=1.0
    )
    tracker._stored_energy_kwh = 5.0
    tracker._current_mode = BatteryMode.CHARGE

    tracker._process_energy_change(
        energy_kwh=1.0,
        is_charge=True,
        current_soc=50.0,          # SOC says 5.0 kWh, accumulator says 5.0+1.0
        now=datetime.datetime(2024, 1, 1, 1, 0),
    )

    assert tracker._stored_energy_kwh == pytest.approx(6.0)


def test_gross_drift_is_resynced_as_a_safety_net():
    """An absurdly desynced accumulator is still corrected."""
    tracker, _state, _soc = make_cost_tracker(
        battery_capacity=10.0, min_soc=0.0, efficiency=1.0
    )
    tracker._stored_energy_kwh = 9.5      # SOC 50% -> 5.0 kWh, drift 4.5 kWh
    tracker._current_mode = BatteryMode.CHARGE

    tracker._process_energy_change(
        energy_kwh=0.5,
        is_charge=True,
        current_soc=50.0,
        now=datetime.datetime(2024, 1, 1, 1, 0),
    )

    # Re-anchored to the pre-event SOC energy (5.0 - 0.5), then the delta added.
    assert tracker._stored_energy_kwh == pytest.approx(5.0)


def test_discharge_transit_is_signed_correctly():
    """A discharge event re-anchors to SOC energy BEFORE the discharge."""
    tracker, _state, _soc = make_cost_tracker(
        battery_capacity=10.0, min_soc=0.0, efficiency=1.0
    )
    tracker._stored_energy_kwh = 9.9      # gross drift, forces the resync path
    tracker._current_mode = BatteryMode.DISCHARGE

    tracker._process_energy_change(
        energy_kwh=1.0,
        is_charge=False,
        current_soc=40.0,                 # 4.0 kWh now, so 5.0 kWh before
        now=datetime.datetime(2024, 1, 1, 1, 0),
    )

    assert tracker._stored_energy_kwh == pytest.approx(4.0)
