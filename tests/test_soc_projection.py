"""
Tests for the shared slot-SOC transition model (battery_optimizer_lib.soc_projection).

Background — the defects these tests guard against (measured in a real 33-hour
AppDaemon window, 2026-07-27..28):

1. Partial first slot: ``calculate_expected_soc_schedule`` projected a FULL slot
   for the current slot even when only one minute of it was left, while the DP
   applied ``slot_fractions``. After the 11:59 recalculation the expected SOC at
   12:00 was 33.1% while the DP's own trajectory said 27.2% — a guaranteed false
   "SOC behind plan" at the next slot boundary (8 such events in the log).

2. Two opposite physics models for DISCHARGE: the deviation detector never read
   PV and always projected SOC falling, while the orchestrator/DP treat
   ``pv >= load`` as PV charging. 78 false "SOC ahead ... favorable deviation"
   events in the log; at 11:59 the same 27.0% was "+9.1% ahead" and 47 seconds
   later "6.1% behind".
"""

import datetime

import pytest

from battery_optimizer import BatteryMode, BatteryOptimizer, PricePoint, ScheduleEntry
from battery_optimizer_lib import (
    BatteryOptimizerConfig,
    DPOptimizer,
    DPOptimizerConfig,
    SocProjectionParams,
    project_slot_soc,
)

CAPACITY = 14.3


def _params(**overrides) -> SocProjectionParams:
    base = dict(
        battery_capacity=CAPACITY,
        efficiency=0.95,
        charge_rate=4.5,
        discharge_rate=4.5,
        export_discharge_rate=0.0,
        inverter_efficiency=1.0,
        min_soc=10.0,
        max_soc=100.0,
        slot_minutes=15,
    )
    base.update(overrides)
    return SocProjectionParams(**base)


class TestProjectSlotSocCharge:
    def test_partial_slot_scales_linearly(self):
        params = _params()
        full = project_slot_soc(soc_start=50.0, mode=BatteryMode.CHARGE, params=params)
        half = project_slot_soc(
            soc_start=50.0, mode=BatteryMode.CHARGE, params=params, fraction=0.5
        )
        assert full.soc_end - 50.0 == pytest.approx(
            4.5 * 0.95 * 0.25 / CAPACITY * 100
        )
        assert half.soc_end - 50.0 == pytest.approx((full.soc_end - 50.0) / 2)

    def test_zero_fraction_is_a_no_op(self):
        """minutes_into_slot == slot_minutes must leave the SOC untouched."""
        params = _params()
        t = project_slot_soc(
            soc_start=50.0, mode=BatteryMode.CHARGE, params=params, fraction=0.0
        )
        assert t.soc_end == 50.0
        assert t.dc_energy_in_kwh == 0.0

    def test_charge_clamped_at_max_soc(self):
        params = _params(max_soc=90.0)
        t = project_slot_soc(soc_start=89.5, mode=BatteryMode.CHARGE, params=params)
        assert t.soc_end == 90.0


class TestProjectSlotSocDischarge:
    def test_discharge_without_pv_falls(self):
        params = _params()
        t = project_slot_soc(
            soc_start=50.0, mode=BatteryMode.DISCHARGE, params=params, load_kw=2.0
        )
        assert t.soc_end == pytest.approx(50.0 - (2.0 * 0.25 / CAPACITY) * 100)

    def test_discharge_with_pv_surplus_rises(self):
        """The 27.07 11:59 log scenario: pv=4.46 kW, load=0.80 kW -> SOC RISES.

        The old detector modelled this as a discharge and reported the battery
        as "ahead of plan"; the DP and the orchestrator both charge here.
        """
        params = _params()
        t = project_slot_soc(
            soc_start=27.0,
            mode=BatteryMode.DISCHARGE,
            params=params,
            load_kw=0.80,
            pv_kw=4.46,
        )
        expected_gain = (4.46 - 0.80) * 0.95 * 0.25 / CAPACITY * 100
        assert t.soc_end > 27.0
        assert t.soc_end == pytest.approx(27.0 + expected_gain)
        assert t.dc_energy_out_kwh == 0.0

    def test_discharge_with_pv_equal_to_load_is_flat(self):
        params = _params()
        t = project_slot_soc(
            soc_start=60.0,
            mode=BatteryMode.DISCHARGE,
            params=params,
            load_kw=1.5,
            pv_kw=1.5,
        )
        assert t.soc_end == pytest.approx(60.0)

    def test_pv_charge_capped_by_charge_rate(self):
        params = _params(charge_rate=2.0)
        t = project_slot_soc(
            soc_start=50.0,
            mode=BatteryMode.DISCHARGE,
            params=params,
            load_kw=0.0,
            pv_kw=10.0,
        )
        assert t.soc_end == pytest.approx(50.0 + (2.0 * 0.95 * 0.25 / CAPACITY) * 100)

    def test_export_slot_uses_export_discharge_rate(self):
        """An export slot drains at the export rate, not at the load rate."""
        params = _params(export_discharge_rate=9.0)
        t = project_slot_soc(
            soc_start=80.0,
            mode=BatteryMode.DISCHARGE,
            params=params,
            load_kw=0.5,
            pv_kw=0.0,
            export_rate=100,
        )
        assert t.soc_end == pytest.approx(80.0 - (9.0 * 0.25 / CAPACITY) * 100)

    def test_inverter_efficiency_increases_dc_drain(self):
        """AC energy served costs more DC energy when the inverter loses power."""
        params = _params(inverter_efficiency=0.95)
        t = project_slot_soc(
            soc_start=50.0, mode=BatteryMode.DISCHARGE, params=params, load_kw=2.0
        )
        ac_kwh = 2.0 * 0.25
        assert t.dc_energy_out_kwh == pytest.approx(ac_kwh / 0.95)
        assert t.soc_end == pytest.approx(50.0 - (ac_kwh / 0.95 / CAPACITY) * 100)

    def test_discharge_clamped_at_min_soc(self):
        params = _params(min_soc=10.0)
        t = project_slot_soc(
            soc_start=10.5, mode=BatteryMode.DISCHARGE, params=params, load_kw=4.5
        )
        assert t.soc_end == 10.0


class TestProjectSlotSocHold:
    def test_hold_with_pv_surplus_charges(self):
        params = _params()
        t = project_slot_soc(
            soc_start=40.0,
            mode=BatteryMode.HOLD,
            params=params,
            load_kw=0.5,
            pv_kw=3.5,
        )
        assert t.soc_end == pytest.approx(40.0 + (3.0 * 0.95 * 0.25 / CAPACITY) * 100)

    def test_hold_without_pv_is_flat(self):
        params = _params()
        t = project_slot_soc(
            soc_start=40.0, mode=BatteryMode.HOLD, params=params, load_kw=1.2, pv_kw=0.0
        )
        assert t.soc_end == 40.0


class TestSharedProjectionMatchesDp:
    """The shared model must agree with the DP's internal slot transition.

    TOLERANCE: the DP discretizes stored energy on a ``soc_step_percent`` grid
    (``_energy_to_index`` floors after charging, ``_discharge_index`` rounds to
    nearest with a no-free-lunch guard), so a single slot may land up to one
    grid step away from the continuous projection. The comparison is therefore
    done PER SLOT, always restarting from the DP's own slot-start SOC — never
    cumulatively, which would let the quantization error accumulate.
    """

    TOLERANCE = 0.1 + 1e-6  # == soc_step_percent used below

    @staticmethod
    def _build(load_kw: float = 0.80):
        base = datetime.datetime(2026, 7, 27, 11, 45)
        prices = [0.05, 0.06, 0.20, 0.22, 0.04, 0.03, 0.25, 0.26, 0.05, 0.05, 0.24, 0.23]
        slots = [
            PricePoint(time=base + datetime.timedelta(minutes=15 * i), price=p)
            for i, p in enumerate(prices)
        ]

        # Sunny in the middle: some slots have pv > load, some pv < load.
        pv_by_index = [4.46, 4.20, 0.0, 0.0, 3.9, 4.6, 0.0, 0.0, 2.0, 0.4, 0.0, 0.0]

        def load_predictor(_dt):
            return load_kw

        def pv_predictor(dt):
            idx = int((dt - base).total_seconds() // 900)
            return pv_by_index[idx] if 0 <= idx < len(pv_by_index) else 0.0

        config = DPOptimizerConfig(
            battery_capacity=CAPACITY,
            min_soc=10.0,
            max_soc=100.0,
            efficiency=0.95,
            discharge_rate=4.5,
            export_discharge_rate=0.0,
            slot_minutes=15,
            soc_step_percent=0.1,
            grid_fee=0.05,
            battery_wear_cost=0.0,
            grid_export_fee=0.02,
            export_rate_multiplier=0.0,  # export disabled: compare base transitions
            inverter_efficiency=0.97,
            import_price_multiplier=1.0,
            terminal_energy_value_eur_kwh=0.0,
        )

        optimizer = DPOptimizer(
            config=config,
            load_predictor=load_predictor,
            charge_rate_predictor=lambda soc, temp=None: 4.5,
            temp_after_charge_predictor=lambda t, d: t,
            temp_after_idle_predictor=lambda t, d: t,
            pv_predictor=pv_predictor,
        )
        return base, slots, config, optimizer, load_predictor, pv_predictor

    def _assert_matches(
        self,
        minutes_into_slot: float,
        load_kw: float = 0.80,
        current_soc: float = 50.0,
    ):
        base, slots, config, optimizer, load_predictor, pv_predictor = self._build(
            load_kw=load_kw
        )

        result = optimizer.optimize(
            prices=slots,
            current_slot=base,
            current_soc=current_soc,
            current_temp=None,
            minutes_into_slot=minutes_into_slot,
        )

        params = SocProjectionParams(
            battery_capacity=config.battery_capacity,
            efficiency=config.efficiency,
            charge_rate=4.5,
            discharge_rate=config.discharge_rate,
            export_discharge_rate=config.export_discharge_rate,
            inverter_efficiency=config.inverter_efficiency,
            min_soc=config.min_soc,
            max_soc=config.max_soc,
            slot_minutes=config.slot_minutes,
        )
        first_fraction = min(1.0, max(0.0, (15 - minutes_into_slot) / 15))

        assert result.soc_trajectory, "DP produced no trajectory"
        for i, point in enumerate(slots):
            hour = point.time
            start_soc, dp_end_soc = result.soc_trajectory[hour]
            entry = result.schedule[hour]
            projected = project_slot_soc(
                soc_start=start_soc,
                mode=entry.mode,
                params=params,
                load_kw=load_predictor(hour),
                pv_kw=pv_predictor(hour),
                fraction=first_fraction if i == 0 else 1.0,
                export_rate=entry.export_rate,
            )
            assert projected.soc_end == pytest.approx(dp_end_soc, abs=self.TOLERANCE), (
                f"slot {hour:%H:%M} mode={entry.mode.name} "
                f"start={start_soc:.2f}% dp_end={dp_end_soc:.2f}% "
                f"shared={projected.soc_end:.2f}%"
            )
        return result

    def test_shared_projection_matches_dp_trajectory_full_first_slot(self):
        result = self._assert_matches(minutes_into_slot=0.0)
        modes = {e.mode for e in result.schedule.values()}
        # The scenario must exercise both PV-charging HOLD and net-load DISCHARGE.
        assert BatteryMode.HOLD in modes
        assert BatteryMode.DISCHARGE in modes

    def test_shared_projection_matches_dp_trajectory_partial_first_slot(self):
        # 14 minutes elapsed -> only 1 of 15 minutes left, the 11:59 log case.
        self._assert_matches(minutes_into_slot=14.0)

    def test_shared_projection_matches_dp_trajectory_with_charge_slots(self):
        """High load + low SOC makes the DP buy energy — covers CHARGE too."""
        result = self._assert_matches(
            minutes_into_slot=7.0, load_kw=4.0, current_soc=20.0
        )
        modes = {e.mode for e in result.schedule.values()}
        assert BatteryMode.CHARGE in modes


class _PartialSlotSocOptimizer:
    """Minimal host for the real ``calculate_expected_soc_schedule``."""

    def __init__(self, pv_kw: float, load_kw: float):
        self.config = BatteryOptimizerConfig(
            battery_capacity=CAPACITY,
            charge_rate=4.5,
            discharge_rate=4.5,
            efficiency=0.95,
            slot_minutes=15,
            default_min_soc=10.0,
            default_max_soc=100.0,
            decision_log_level=0,
        )
        self.min_soc = 10.0
        self.max_soc = 100.0
        self.learning_engine = None
        self._pv_kw = pv_kw
        self._load_kw = load_kw

    def _get_local_timezone(self):
        return None

    def _predict_load_kw(self, dt):
        return self._load_kw

    def _predict_pv_kw(self, dt):
        return self._pv_kw


_PartialSlotSocOptimizer.calculate_expected_soc_schedule = (
    BatteryOptimizer.calculate_expected_soc_schedule
)


class TestExpectedSocSchedulePartialFirstSlot:
    """Regression for defect 1, expressed in the numbers from the log."""

    @staticmethod
    def _schedule(base):
        return {
            base
            + datetime.timedelta(minutes=15 * i): ScheduleEntry(
                time=base + datetime.timedelta(minutes=15 * i),
                mode=BatteryMode.DISCHARGE,
                reason="[cloud-safe]",
            )
            for i in range(3)
        }

    def test_one_minute_left_projects_one_fifteenth_of_the_pv_gain(self):
        base = datetime.datetime(2026, 7, 27, 11, 45)
        opt = _PartialSlotSocOptimizer(pv_kw=4.46, load_kw=0.80)
        schedule = self._schedule(base)

        expected_soc, _ = opt.calculate_expected_soc_schedule(
            schedule,
            starting_soc=27.0,
            current_slot=base,
            minutes_into_slot=14.0,
        )

        full_slot_gain = (4.46 - 0.80) * 0.95 * 0.25 / CAPACITY * 100  # ~6.08%
        next_slot = base + datetime.timedelta(minutes=15)

        assert expected_soc[base] == 27.0
        gain = expected_soc[next_slot] - expected_soc[base]
        # Only 1 of 15 minutes remained: ~0.41%, not the ~6.08% that produced
        # the bogus "expected 33.1%" at 12:00 in the log.
        assert gain == pytest.approx(full_slot_gain / 15, abs=1e-6)
        assert gain < 1.0
        assert expected_soc[next_slot] < 27.5

    def test_without_partial_context_full_slot_is_projected(self):
        """Legacy callers (no current_slot) keep the full-slot behaviour."""
        base = datetime.datetime(2026, 7, 27, 11, 45)
        opt = _PartialSlotSocOptimizer(pv_kw=4.46, load_kw=0.80)
        schedule = self._schedule(base)

        expected_soc, _ = opt.calculate_expected_soc_schedule(schedule, starting_soc=27.0)

        full_slot_gain = (4.46 - 0.80) * 0.95 * 0.25 / CAPACITY * 100
        next_slot = base + datetime.timedelta(minutes=15)
        assert expected_soc[next_slot] - expected_soc[base] == pytest.approx(
            full_slot_gain
        )

    def test_partial_fraction_applied_only_once(self):
        base = datetime.datetime(2026, 7, 27, 11, 45)
        opt = _PartialSlotSocOptimizer(pv_kw=4.46, load_kw=0.80)
        schedule = self._schedule(base)

        expected_soc, _ = opt.calculate_expected_soc_schedule(
            schedule,
            starting_soc=27.0,
            current_slot=base,
            minutes_into_slot=14.0,
        )

        full_slot_gain = (4.46 - 0.80) * 0.95 * 0.25 / CAPACITY * 100
        slot_1 = base + datetime.timedelta(minutes=15)
        slot_2 = base + datetime.timedelta(minutes=30)
        # Second slot is a full slot again.
        assert expected_soc[slot_2] - expected_soc[slot_1] == pytest.approx(
            full_slot_gain
        )

    def test_expected_soc_schedule_matches_shared_projection_for_discharge(self):
        """The orchestrator must not re-implement the transition locally."""
        base = datetime.datetime(2026, 7, 27, 11, 45)
        opt = _PartialSlotSocOptimizer(pv_kw=0.0, load_kw=2.0)
        schedule = self._schedule(base)

        expected_soc, _ = opt.calculate_expected_soc_schedule(schedule, starting_soc=60.0)

        params = _params()
        soc = 60.0
        for i in range(3):
            slot = base + datetime.timedelta(minutes=15 * i)
            assert expected_soc[slot] == pytest.approx(soc)
            soc = project_slot_soc(
                soc_start=soc,
                mode=BatteryMode.DISCHARGE,
                params=params,
                load_kw=2.0,
                pv_kw=0.0,
            ).soc_end


class TestScheduleFormatterUsesTheSharedModel:
    """The log fallback path was a THIRD transition model.

    ``_format_hold_trajectory`` printed ``end_soc = start_soc`` (no PV surplus
    charging at all) and ``_format_discharge_trajectory`` drained
    ``min(load, discharge_rate)`` from raw load without subtracting PV or
    dividing by the inverter efficiency. On a sunny slot that contradicted the
    trajectory the SOC deviation detector compares against — in the same log.
    """

    SUNNY = dict(pv_kw=4.0, load_kw=0.8, start_soc=50.0)

    def _formatter(self, **overrides):
        from battery_optimizer_lib import ScheduleFormatter, ScheduleFormatterConfig

        cfg = dict(
            slot_minutes=15,
            slot_hours=0.25,
            battery_capacity=CAPACITY,
            charge_rate=4.5,
            discharge_rate=4.5,
            export_discharge_rate=0.0,
            efficiency=0.95,
            battery_wear_cost=0.0,
            decision_log_level=1,
            inverter_efficiency=1.0,
        )
        cfg.update(overrides)
        messages = []
        formatter = ScheduleFormatter(
            config=ScheduleFormatterConfig(**cfg),
            log_func=lambda msg, level="INFO": messages.append(msg),
            learning_engine=None,
        )
        return formatter, messages

    @staticmethod
    def _logged_end_soc(messages, mode_name):
        for msg in messages:
            if mode_name in msg and "->" in msg:
                return float(msg.split("->")[-1].strip().split("%")[0])
        raise AssertionError(f"no {mode_name} line in {messages}")

    @pytest.mark.parametrize(
        "mode", [BatteryMode.HOLD, BatteryMode.DISCHARGE, BatteryMode.CHARGE]
    )
    def test_formatter_matches_project_slot_soc(self, mode):
        hour = datetime.datetime(2026, 7, 27, 12, 0)
        formatter, messages = self._formatter()

        formatter.log_schedule(
            schedule={hour: ScheduleEntry(time=hour, mode=mode, reason="0.10 EUR/kWh")},
            expected_soc={hour: self.SUNNY["start_soc"]},
            expected_temp=None,
            local_tz=None,
            predict_load_kw=lambda h: self.SUNNY["load_kw"],
            predict_pv_kw=lambda h: self.SUNNY["pv_kw"],
            min_soc=10.0,
            max_soc=100.0,
        )

        expected = project_slot_soc(
            soc_start=self.SUNNY["start_soc"],
            mode=mode,
            params=_params(),
            load_kw=self.SUNNY["load_kw"],
            pv_kw=self.SUNNY["pv_kw"],
        ).soc_end
        assert self._logged_end_soc(messages, mode.name) == pytest.approx(
            expected, abs=0.05
        )

    def test_sunny_hold_charges_instead_of_standing_still(self):
        """Concrete repro: the old HOLD line said 50.0%->50.0%."""
        hour = datetime.datetime(2026, 7, 27, 12, 0)
        formatter, messages = self._formatter()

        formatter.log_schedule(
            schedule={
                hour: ScheduleEntry(time=hour, mode=BatteryMode.HOLD, reason="0.10 EUR/kWh")
            },
            expected_soc={hour: 50.0},
            expected_temp=None,
            local_tz=None,
            predict_load_kw=lambda h: 0.8,
            predict_pv_kw=lambda h: 4.0,
            min_soc=10.0,
            max_soc=100.0,
        )

        assert self._logged_end_soc(messages, "HOLD") > 54.0

    def test_sunny_discharge_does_not_drain(self):
        """Concrete repro: the old DISCHARGE line said 50.0%->48.6%."""
        hour = datetime.datetime(2026, 7, 27, 12, 0)
        formatter, messages = self._formatter()

        formatter.log_schedule(
            schedule={
                hour: ScheduleEntry(
                    time=hour, mode=BatteryMode.DISCHARGE, reason="0.10 EUR/kWh"
                )
            },
            expected_soc={hour: 50.0},
            expected_temp=None,
            local_tz=None,
            predict_load_kw=lambda h: 0.8,
            predict_pv_kw=lambda h: 4.0,
            min_soc=10.0,
            max_soc=100.0,
        )

        assert self._logged_end_soc(messages, "DISCHARGE") > 54.0

    def test_export_discharge_uses_the_export_rate(self):
        hour = datetime.datetime(2026, 7, 27, 19, 0)
        formatter, messages = self._formatter(export_discharge_rate=7.0)

        entry = ScheduleEntry(
            time=hour,
            mode=BatteryMode.DISCHARGE,
            reason="0.30 EUR/kWh",
            export_rate=100,
        )
        formatter.log_schedule(
            schedule={hour: entry},
            expected_soc={hour: 80.0},
            expected_temp=None,
            local_tz=None,
            predict_load_kw=lambda h: 0.8,
            predict_pv_kw=lambda h: 0.0,
            min_soc=10.0,
            max_soc=100.0,
        )

        expected = project_slot_soc(
            soc_start=80.0,
            mode=BatteryMode.DISCHARGE,
            params=_params(export_discharge_rate=7.0),
            load_kw=0.8,
            pv_kw=0.0,
            export_rate=100,
        ).soc_end
        assert self._logged_end_soc(messages, "DISCHARGE") == pytest.approx(
            expected, abs=0.05
        )

    def test_inverter_efficiency_is_honoured(self):
        """The old discharge model never divided by inverter_efficiency."""
        hour = datetime.datetime(2026, 7, 27, 19, 0)
        formatter, messages = self._formatter(inverter_efficiency=0.9)

        formatter.log_schedule(
            schedule={
                hour: ScheduleEntry(
                    time=hour, mode=BatteryMode.DISCHARGE, reason="0.30 EUR/kWh"
                )
            },
            expected_soc={hour: 80.0},
            expected_temp=None,
            local_tz=None,
            predict_load_kw=lambda h: 3.0,
            predict_pv_kw=lambda h: 0.0,
            min_soc=10.0,
            max_soc=100.0,
        )

        expected = project_slot_soc(
            soc_start=80.0,
            mode=BatteryMode.DISCHARGE,
            params=_params(inverter_efficiency=0.9),
            load_kw=3.0,
            pv_kw=0.0,
        ).soc_end
        logged = self._logged_end_soc(messages, "DISCHARGE")
        assert logged == pytest.approx(expected, abs=0.05)
        # 3 kW AC for 15 min at 90% = 0.833 kWh DC, not 0.75.
        assert logged < 80.0 - (3.0 * 0.25 / CAPACITY) * 100
