"""Task 2: one unit contract for nominal and learned charge rates.

The contract (see ``docs/scheduling-algorithm.md`` SS Charge-rate units):

* ``BatteryLearningEngine.get_charge_rate_for_soc`` returns
  ``charge_input_dc_kw`` -- DC power delivered to the battery terminal, BEFORE
  storage retention.
* Observations are recorded (and persisted) as ``stored_charge_kw`` -- the rate
  at which stored energy grows. The conversion happens at the API boundary.
* ``stored_charge_kw = charge_input_dc_kw * storage_efficiency``
* ``grid_charge_ac_kwh = charge_input_dc_kwh / inverter_efficiency``

Every test here is an *independent* check: it replays a physical observation or
computes the expected energy analytically, never by copying the implementation's
formula.
"""

import json

import pytest

from battery_optimizer_lib import BatteryLearningEngine
from battery_optimizer_lib.models import BatteryMode
from battery_optimizer_lib.soc_projection import (
    SocProjectionParams,
    project_slot_soc,
)


CAPACITY = 10.0
NOMINAL_RATE = 4.0


def _engine(efficiency=0.85, nominal_rate=NOMINAL_RATE, capacity=CAPACITY):
    return BatteryLearningEngine(
        battery_capacity_kwh=capacity,
        nominal_charge_rate_kw=nominal_rate,
        nominal_efficiency=efficiency,
        min_soc=10.0,
        max_soc=100.0,
        log_func=lambda *a, **k: None,
    )


def _params(efficiency=0.85, inverter_efficiency=1.0, capacity=CAPACITY):
    return SocProjectionParams(
        battery_capacity=capacity,
        efficiency=efficiency,
        charge_rate=NOMINAL_RATE,
        discharge_rate=4.0,
        inverter_efficiency=inverter_efficiency,
        min_soc=10.0,
        max_soc=100.0,
        slot_minutes=15,
    )


def _record_soc_observations(engine, soc_start, soc_end, minutes, count=3, temp=None):
    """Record ``count`` identical SOC-derived charge observations."""
    for _ in range(count):
        engine.record_charging(
            soc_start=soc_start,
            soc_end=soc_end,
            duration_minutes=minutes,
            battery_temp=temp,
        )


class TestObservationReplay:
    """A learned observation must reproduce its own physical SOC gain."""

    def test_replaying_a_soc_observation_predicts_its_own_end_soc(self):
        """40% -> 50% in 15 min, replayed, must land on 50% (was 48.5%).

        The observation is unambiguous: 10 kWh capacity, +10 SOC points is
        +1.0 kWh of STORED energy in 0.25 h. Projecting the very same interval
        must therefore end at 50%. The reviewed implementation returned 48.5%
        because the storage-retention factor was applied to a rate that already
        described stored-energy growth.
        """
        engine = _engine(efficiency=0.85)
        _record_soc_observations(engine, 40.0, 50.0, 15.0)

        transition = project_slot_soc(
            soc_start=40.0,
            mode=BatteryMode.CHARGE,
            params=_params(efficiency=0.85),
            learning_engine=engine,
        )

        assert transition.soc_end == pytest.approx(50.0, abs=1e-6)
        assert transition.dc_energy_in_kwh == pytest.approx(1.0, abs=1e-9)

    def test_replay_holds_at_unit_efficiency(self):
        engine = _engine(efficiency=1.0)
        _record_soc_observations(engine, 40.0, 50.0, 15.0)

        transition = project_slot_soc(
            soc_start=40.0,
            mode=BatteryMode.CHARGE,
            params=_params(efficiency=1.0),
            learning_engine=engine,
        )
        assert transition.soc_end == pytest.approx(50.0, abs=1e-6)

    def test_replay_holds_with_temperature_data(self):
        """The warming-aware path must replay the observation too."""
        engine = _engine(efficiency=0.85)
        _record_soc_observations(engine, 40.0, 50.0, 15.0, temp=20.0)

        transition = project_slot_soc(
            soc_start=40.0,
            mode=BatteryMode.CHARGE,
            params=_params(efficiency=0.85),
            temp_start=20.0,
            learning_engine=engine,
        )
        assert transition.soc_end == pytest.approx(50.0, abs=1e-6)

    def test_replay_holds_with_measured_stored_energy_observations(self):
        """An inverter ``energy_to_battery`` observation is stored-DC too."""
        engine = _engine(efficiency=0.85)
        for _ in range(3):
            engine.record_charging(
                soc_start=40.0,
                soc_end=50.0,
                duration_minutes=15.0,
                energy_to_battery_kwh=1.0,
            )
        transition = project_slot_soc(
            soc_start=40.0,
            mode=BatteryMode.CHARGE,
            params=_params(efficiency=0.85),
            learning_engine=engine,
        )
        assert transition.soc_end == pytest.approx(50.0, abs=1e-6)


class TestNominalAndLearnedAgree:
    """Metamorphic: the source of a rate must not change the physics."""

    @pytest.mark.parametrize("efficiency", [1.0, 0.85])
    @pytest.mark.parametrize("temp", [None, 20.0])
    def test_nominal_and_equivalent_learned_rate_predict_the_same_soc(
        self, efficiency, temp
    ):
        """Nominal 4 kW input DC == a learned 4*eff kW stored observation."""
        params = _params(efficiency=efficiency)

        no_learning = project_slot_soc(
            soc_start=40.0,
            mode=BatteryMode.CHARGE,
            params=params,
            temp_start=temp,
        )

        # An observation that is exactly the nominal rate physically: the
        # nominal 4 kW arrives at the terminal, so stored growth is 4*eff kW,
        # i.e. 4*eff*0.25 kWh over a 15-minute slot -> that many SOC points.
        stored_kwh = NOMINAL_RATE * efficiency * 0.25
        soc_gain = stored_kwh / CAPACITY * 100
        engine = _engine(efficiency=efficiency)
        _record_soc_observations(engine, 40.0, 40.0 + soc_gain, 15.0, temp=temp)

        learned = project_slot_soc(
            soc_start=40.0,
            mode=BatteryMode.CHARGE,
            params=params,
            temp_start=temp,
            learning_engine=engine,
        )

        assert learned.soc_end == pytest.approx(no_learning.soc_end, abs=1e-6)

    def test_get_charge_rate_returns_input_dc_units(self):
        """The documented boundary: learned stored kW / storage efficiency."""
        engine = _engine(efficiency=0.85)
        _record_soc_observations(engine, 40.0, 50.0, 15.0)
        # Stored growth was 4.0 kW; the terminal therefore delivered 4/0.85.
        assert engine.get_charge_rate_for_soc(45.0) == pytest.approx(
            4.0 / 0.85, rel=1e-9
        )

    def test_nominal_fallback_is_already_input_dc(self):
        engine = _engine(efficiency=0.85)
        assert engine.get_charge_rate_for_soc(45.0) == pytest.approx(NOMINAL_RATE)


class TestSourceAccounting:
    """Grid / PV / mixed charging conserve energy and bill only AC imports."""

    def test_grid_only_charge_imports_stored_over_both_efficiencies(self):
        """1 kWh stored at eff 0.85 and inverter 0.97 imports ~1.21286 kWh AC."""
        from battery_optimizer_lib.slot_energy import simulate_slot, SlotEnergyParams

        params = SlotEnergyParams(
            battery_capacity=CAPACITY,
            efficiency=0.85,
            inverter_efficiency=0.97,
            discharge_rate=4.0,
            min_soc=10.0,
            max_soc=100.0,
            slot_minutes=60,
        )
        # Charge for a full hour at the input-DC rate that stores exactly 1 kWh.
        charge_input_dc_kw = 1.0 / 0.85
        result = simulate_slot(
            stored_energy_kwh=5.0,
            mode=BatteryMode.CHARGE,
            params=params,
            charge_input_dc_kw=charge_input_dc_kw,
            load_kw=0.0,
            pv_kw=0.0,
        )

        assert result.stored_dc_in_kwh == pytest.approx(1.0, abs=1e-9)
        assert result.grid_charge_ac_kwh == pytest.approx(1.21286, abs=1e-5)
        assert result.pv_charge_dc_kwh == pytest.approx(0.0, abs=1e-12)

    def test_pv_only_charge_bills_nothing_and_skips_the_inverter_loss(self):
        """DC-coupled PV surplus must not be charged the grid inverter loss."""
        from battery_optimizer_lib.slot_energy import simulate_slot, SlotEnergyParams

        params = SlotEnergyParams(
            battery_capacity=CAPACITY,
            efficiency=0.85,
            inverter_efficiency=0.97,
            discharge_rate=4.0,
            min_soc=10.0,
            max_soc=100.0,
            slot_minutes=60,
        )
        result = simulate_slot(
            stored_energy_kwh=5.0,
            mode=BatteryMode.HOLD,
            params=params,
            charge_input_dc_kw=4.0,
            load_kw=0.0,
            pv_kw=2.0,
        )
        assert result.grid_charge_ac_kwh == pytest.approx(0.0, abs=1e-12)
        assert result.grid_import_ac_kwh == pytest.approx(0.0, abs=1e-12)
        assert result.pv_charge_dc_kwh == pytest.approx(2.0, abs=1e-9)
        assert result.stored_dc_in_kwh == pytest.approx(2.0 * 0.85, abs=1e-9)

    def test_mixed_source_charge_bills_only_the_grid_share(self):
        from battery_optimizer_lib.slot_energy import simulate_slot, SlotEnergyParams

        params = SlotEnergyParams(
            battery_capacity=CAPACITY,
            efficiency=0.85,
            inverter_efficiency=0.97,
            discharge_rate=4.0,
            min_soc=10.0,
            max_soc=100.0,
            slot_minutes=60,
        )
        # 4 kW terminal charge, 1 kW of it from PV surplus.
        result = simulate_slot(
            stored_energy_kwh=1.0,
            mode=BatteryMode.CHARGE,
            params=params,
            charge_input_dc_kw=4.0,
            load_kw=1.0,
            pv_kw=2.0,
        )
        assert result.pv_charge_dc_kwh == pytest.approx(1.0, abs=1e-9)
        # Grid supplies 3 kWh DC -> 3/0.97 AC, and nothing for the load
        # (PV covers it exactly).
        assert result.grid_charge_ac_kwh == pytest.approx(3.0 / 0.97, abs=1e-9)
        assert result.stored_dc_in_kwh == pytest.approx(4.0 * 0.85, abs=1e-9)

    def test_discharge_accounting_keeps_dc_to_ac_units(self):
        """Wear is per discharged DC kWh; served load is AC after inverter."""
        from battery_optimizer_lib.slot_energy import simulate_slot, SlotEnergyParams

        params = SlotEnergyParams(
            battery_capacity=CAPACITY,
            efficiency=0.85,
            inverter_efficiency=0.97,
            discharge_rate=4.0,
            min_soc=10.0,
            max_soc=100.0,
            slot_minutes=60,
        )
        result = simulate_slot(
            stored_energy_kwh=5.0,
            mode=BatteryMode.DISCHARGE,
            params=params,
            charge_input_dc_kw=0.0,
            load_kw=2.0,
            pv_kw=0.0,
        )
        assert result.battery_ac_served_kwh == pytest.approx(2.0, abs=1e-9)
        assert result.stored_dc_out_kwh == pytest.approx(2.0 / 0.97, abs=1e-9)
        assert result.grid_import_ac_kwh == pytest.approx(0.0, abs=1e-12)


class TestPersistedLearningData:
    """Persisted observations keep their original (stored-DC) units."""

    def test_save_load_reload_does_not_convert_twice(self):
        engine = _engine(efficiency=0.85)
        _record_soc_observations(engine, 40.0, 50.0, 15.0)
        first = engine.get_charge_rate_for_soc(45.0)

        blob = engine.save_to_json()
        reloaded = _engine(efficiency=0.85)
        assert reloaded.load_from_json(blob)
        second = reloaded.get_charge_rate_for_soc(45.0)

        blob2 = reloaded.save_to_json()
        again = _engine(efficiency=0.85)
        assert again.load_from_json(blob2)
        third = again.get_charge_rate_for_soc(45.0)

        assert first == pytest.approx(second, rel=1e-12)
        assert second == pytest.approx(third, rel=1e-12)
        # And the persisted numbers are the raw stored-DC observations.
        assert json.loads(blob2)["stats"]["charge_rates_by_soc"]["25-50"] == [
            pytest.approx(4.0),
            pytest.approx(4.0),
            pytest.approx(4.0),
        ]

    def test_a_file_written_by_an_older_version_still_replays(self):
        """v6 files hold stored-DC rates; loading must not rescale them."""
        blob = json.dumps(
            {
                "version": 6,
                "learned_efficiency": 0.85,
                "stats": {"charge_rates_by_soc": {"25-50": [4.0, 4.0, 4.0]}},
            }
        )
        engine = _engine(efficiency=0.85)
        assert engine.load_from_json(blob)
        transition = project_slot_soc(
            soc_start=40.0,
            mode=BatteryMode.CHARGE,
            params=_params(efficiency=0.85),
            learning_engine=engine,
        )
        assert transition.soc_end == pytest.approx(50.0, abs=1e-6)


class TestEfficiencyLearningHonesty:
    """A synthetic grid figure is not an efficiency measurement."""

    def test_efficiency_only_learns_from_independent_grid_energy(self):
        engine = _engine(efficiency=0.85)
        before = engine.learned_efficiency
        # Synthetic input: stored / configured efficiency. Learning from it
        # would just re-derive the configured constant.
        engine.record_charging(
            soc_start=40.0,
            soc_end=50.0,
            duration_minutes=15.0,
            energy_from_grid_kwh=1.0 / 0.85,
        )
        assert engine.learned_efficiency == pytest.approx(before)

    def test_a_real_independent_measurement_still_teaches_efficiency(self):
        """An AC meter figure that differs from the configured factor is used."""
        engine = _engine(efficiency=0.85)
        before = engine.learned_efficiency
        engine.record_charging(
            soc_start=40.0,
            soc_end=50.0,
            duration_minutes=15.0,
            energy_from_grid_kwh=1.0 / 0.78,  # measured: worse than configured
        )
        assert engine.learned_efficiency < before
