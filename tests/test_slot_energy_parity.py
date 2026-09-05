"""``slot_energy.simulate_slot`` is the shared physical transition.

Two consumers must agree with it over the whole reachable parameter space:

* ``soc_projection.project_slot_soc`` -- the SOC view used by the expected
  trajectory, the deviation detector, the schedule formatter and the projected
  cost column;
* ``DPOptimizer`` -- which keeps an inlined transition fused with the value
  recursion. That parity is proven in
  ``test_dp_energy_conservation.py::TestPrefixConservationAcrossConditions::
  test_no_prefix_creates_energy``, which replays 210 selected plans through
  ``simulate_slot`` and requires the DP's own SOC trajectory to match.

Agreement is checked by sweeping the parameters, not by re-deriving the formula
in the test: a test that copies the implementation reproduces its mistakes.
"""

import itertools

import pytest

from battery_optimizer_lib.models import BatteryMode
from battery_optimizer_lib.slot_energy import (
    SlotEnergyParams,
    params_from_soc_projection,
    simulate_slot,
)
from battery_optimizer_lib.soc_projection import (
    SocProjectionParams,
    project_slot_soc,
)


CAPACITY = 10.0


def _soc_params(efficiency, inverter_efficiency, charge_rate):
    return SocProjectionParams(
        battery_capacity=CAPACITY,
        efficiency=efficiency,
        charge_rate=charge_rate,
        discharge_rate=4.0,
        export_discharge_rate=5.0,
        inverter_efficiency=inverter_efficiency,
        min_soc=10.0,
        max_soc=100.0,
        slot_minutes=15,
    )


SOCS = [10.0, 10.4, 33.3, 50.0, 99.6, 100.0]
LOADS = [0.0, 0.56, 2.0, 6.0]
PVS = [0.0, 1.0, 5.0]
FRACTIONS = [1.0, 0.4, 0.07]
MODES = [
    (BatteryMode.HOLD, None),
    (BatteryMode.CHARGE, None),
    (BatteryMode.DISCHARGE, 0),
    (BatteryMode.DISCHARGE, 100),
]
EFFS = [(1.0, 1.0), (0.85, 0.97)]


@pytest.mark.parametrize("efficiency,inv_eff", EFFS)
def test_soc_projection_and_simulate_slot_agree(efficiency, inv_eff):
    charge_rate = 4.0
    sp = _soc_params(efficiency, inv_eff, charge_rate)
    ep = params_from_soc_projection(sp)

    checked = 0
    for soc, load, pv, fraction, (mode, export_rate) in itertools.product(
        SOCS, LOADS, PVS, FRACTIONS, MODES
    ):
        transition = project_slot_soc(
            soc_start=soc,
            mode=mode,
            params=sp,
            load_kw=load,
            pv_kw=pv,
            fraction=fraction,
            export_rate=export_rate,
        )
        result = simulate_slot(
            stored_energy_kwh=sp.battery_capacity * soc / 100.0,
            mode=mode,
            params=ep,
            charge_input_dc_kw=charge_rate,
            load_kw=load,
            pv_kw=pv,
            fraction=fraction,
            is_export=bool(export_rate),
        )
        assert ep.soc_of(result.energy_end_kwh) == pytest.approx(
            transition.soc_end, abs=1e-9
        ), (soc, load, pv, fraction, mode, export_rate)
        assert result.stored_dc_in_kwh == pytest.approx(
            transition.dc_energy_in_kwh, abs=1e-9
        ), (soc, load, pv, fraction, mode, export_rate)
        checked += 1

    assert checked == len(SOCS) * len(LOADS) * len(PVS) * len(FRACTIONS) * len(MODES)


class TestConservation:
    """Analytical energy balance, computed independently of the code."""

    @pytest.mark.parametrize("efficiency,inv_eff", EFFS)
    def test_grid_import_covers_exactly_what_the_battery_did_not(
        self, efficiency, inv_eff
    ):
        params = SlotEnergyParams(
            battery_capacity=CAPACITY,
            efficiency=efficiency,
            discharge_rate=4.0,
            inverter_efficiency=inv_eff,
            min_soc=10.0,
            max_soc=100.0,
            slot_minutes=60,
        )
        # 2 kW load for an hour from a pack holding only 0.5 kWh usable.
        result = simulate_slot(
            stored_energy_kwh=1.5,
            mode=BatteryMode.DISCHARGE,
            params=params,
            load_kw=2.0,
        )
        assert result.stored_dc_out_kwh == pytest.approx(0.5)
        assert result.battery_ac_served_kwh == pytest.approx(0.5 * inv_eff)
        assert result.grid_import_ac_kwh == pytest.approx(2.0 - 0.5 * inv_eff)
        assert result.unmet_battery_ac_kwh == pytest.approx(2.0 - 0.5 * inv_eff)
        assert result.energy_end_kwh == pytest.approx(1.0)
        assert result.energy_limited

    def test_a_full_battery_absorbs_nothing_and_reports_the_request(self):
        params = SlotEnergyParams(
            battery_capacity=CAPACITY,
            efficiency=0.85,
            discharge_rate=4.0,
            inverter_efficiency=0.97,
            min_soc=10.0,
            max_soc=100.0,
            slot_minutes=60,
        )
        result = simulate_slot(
            stored_energy_kwh=10.0,
            mode=BatteryMode.CHARGE,
            params=params,
            charge_input_dc_kw=4.0,
        )
        assert result.stored_dc_in_kwh == pytest.approx(0.0)
        assert result.charge_input_dc_kwh == pytest.approx(0.0)
        assert result.requested_charge_input_dc_kwh == pytest.approx(4.0)
        assert result.grid_charge_ac_kwh == pytest.approx(0.0)
        assert result.battery_power_kw == pytest.approx(0.0)
        assert result.energy_limited

    def test_partial_slot_scales_every_flow(self):
        params = SlotEnergyParams(
            battery_capacity=CAPACITY,
            efficiency=0.85,
            discharge_rate=4.0,
            inverter_efficiency=0.97,
            min_soc=10.0,
            max_soc=100.0,
            slot_minutes=60,
        )
        full = simulate_slot(
            stored_energy_kwh=5.0,
            mode=BatteryMode.CHARGE,
            params=params,
            charge_input_dc_kw=4.0,
            load_kw=1.0,
        )
        half = simulate_slot(
            stored_energy_kwh=5.0,
            mode=BatteryMode.CHARGE,
            params=params,
            charge_input_dc_kw=4.0,
            load_kw=1.0,
            fraction=0.5,
        )
        assert half.stored_dc_in_kwh == pytest.approx(full.stored_dc_in_kwh / 2)
        assert half.grid_import_ac_kwh == pytest.approx(full.grid_import_ac_kwh / 2)
        # Power is a rate: it does NOT halve.
        assert half.battery_power_kw == pytest.approx(full.battery_power_kw)
