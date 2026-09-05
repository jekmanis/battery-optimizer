from __future__ import annotations

import datetime

from battery_optimizer import (
    BatteryLearningEngine,
    BatteryMode,
    BatteryOptimizer,
    LoadProfile,
    PricePoint,
)
from battery_optimizer_lib import BatteryCostTracker, BatteryCostConfig, BatteryOptimizerConfig, PvForecastService, PvForecastServiceConfig


class PartialSlotOptimizer:
    """Minimal optimizer mock for partial-slot regression tests."""

    def __init__(
        self,
        now: datetime.datetime,
        soc_step_percent: float = 1.0,
        base_consumption_w: float = 850.0,
    ):
        # Create config object
        self.config = BatteryOptimizerConfig(
            battery_capacity=14.3,
            charge_rate=4.5,
            discharge_rate=4.5,
            efficiency=0.95,
            grid_fee=0.0,
            slot_minutes=60,
            base_consumption=base_consumption_w,
            load_quantile=0.75,
            soc_step_percent=soc_step_percent,
            default_min_soc=10.0,
            default_max_soc=100.0,
            decision_log_level=0,
            battery_wear_cost=0.0,
            export_rate_multiplier=0.0,  # Disable export to test partial slot logic only
        )

        # Legacy attributes for tests
        self.battery_avg_cost = 0.1046
        self.min_soc = 10.0
        self.max_soc = 100.0
        self._current_time = now
        self._last_projected_costs = {}
        self._last_min_charge_slots = 0
        self._last_charge_slots = []

        self.learning_engine = BatteryLearningEngine(
            battery_capacity_kwh=self.config.battery_capacity,
            nominal_charge_rate_kw=self.config.charge_rate,
            nominal_efficiency=self.config.efficiency,
        )
        self.load_profile = LoadProfile(
            slot_minutes=self.config.slot_minutes,
            default_load_w=self.config.base_consumption,
        )

        # Create cost tracker for project_costs method
        self._cost_tracker = BatteryCostTracker(
            config=BatteryCostConfig(
                battery_capacity=self.config.battery_capacity,
                efficiency=self.config.efficiency,
                slot_minutes=self.config.slot_minutes,
                charge_rate=self.config.charge_rate,
                discharge_rate=self.config.discharge_rate,
                grid_fee=self.config.grid_fee,
                battery_wear_cost=self.config.battery_wear_cost,
            ),
            get_state_func=lambda e: None,
            call_service_func=lambda *a, **k: None,
            get_datetime_func=self.datetime,
            get_timezone_func=self._get_local_timezone,
            align_to_slot_func=self._align_to_slot,
            get_min_soc_func=lambda: self.min_soc,
            get_max_soc_func=lambda: self.max_soc,
            get_current_soc_func=lambda: 50.0,
            get_battery_temp_func=lambda: 20.0,
            learning_engine=self.learning_engine,
            get_cached_prices_func=lambda: [],
            save_learning_data_func=lambda: None,
            update_learning_sensor_func=lambda: None,
            log_func=self.log,
        )

        # PV forecast service (empty — no forecast data in tests)
        self._pv_forecast_service = PvForecastService(
            config=PvForecastServiceConfig(slot_minutes=self.config.slot_minutes),
            get_state_func=lambda e, **kw: None,
            get_datetime_func=self.datetime,
            get_timezone_func=self._get_local_timezone,
            log_func=self.log,
        )

    def datetime(self):
        return self._current_time

    def log(self, *args, **kwargs):
        pass

    def _align_to_slot(self, dt: datetime.datetime) -> datetime.datetime:
        minutes = dt.hour * 60 + dt.minute
        slot_start = (minutes // self.config.slot_minutes) * self.config.slot_minutes
        return dt.replace(
            hour=slot_start // 60,
            minute=slot_start % 60,
            second=0,
            microsecond=0,
        )

    def _get_local_timezone(self):
        return None

    def _get_battery_temp(self) -> float:
        return 20.0

    def _predict_load_kw(self, dt: datetime.datetime) -> float:
        return self.load_profile.predict_kw(dt, self.config.load_quantile)

    def _predict_pv_kw(self, dt: datetime.datetime) -> float:
        return 0.0

    def _get_prices_for_date(self, date, tz):
        return []

    def _get_discharge_threshold(self) -> float:
        return (self.battery_avg_cost / self.config.efficiency) + self.config.grid_fee + self.config.battery_wear_cost

    def _log_schedule_decision_context(self, *args, **kwargs):
        pass

    @property
    def _price_service(self):
        """Mock price service with get_prices_for_date method."""
        class MockPriceService:
            def get_prices_for_date(self, date, tz):
                return []
        return MockPriceService()


PartialSlotOptimizer.find_optimal_schedule = BatteryOptimizer.find_optimal_schedule
PartialSlotOptimizer._compute_slot_fractions = BatteryOptimizer._compute_slot_fractions
# An unpriced current slot is resolved BEFORE the solve: the retained action (or
# the HOLD fallback) advances the SOC across the rest of the slot, so the DP
# plans the next interval from where the pack will actually be.
PartialSlotOptimizer._entry_has_real_price = BatteryOptimizer._entry_has_real_price
PartialSlotOptimizer._resolve_unpriced_current_slot = (
    BatteryOptimizer._resolve_unpriced_current_slot
)
PartialSlotOptimizer._advance_current_slot = BatteryOptimizer._advance_current_slot
PartialSlotOptimizer._note_current_slot_state = (
    BatteryOptimizer._note_current_slot_state
)
# The resolved entry joins the plan before the hedge and the validation, and
# the published trajectory is rebuilt from the MEASURED SOC across it.
PartialSlotOptimizer.project_schedule_trajectory = (
    BatteryOptimizer.project_schedule_trajectory
)


def _make_price_points():
    base = datetime.datetime(2026, 1, 24, 16, 0, 0)
    prices = [
        0.1421, 0.1425, 0.1485, 0.1418, 0.1303, 0.1197, 0.1162, 0.1081,
        0.0963, 0.0951, 0.0984, 0.0971, 0.0961, 0.0953, 0.0941, 0.0909,
        0.0949, 0.0974, 0.1018, 0.1099, 0.1153, 0.1250, 0.1224, 0.1206,
    ]
    return base, [
        PricePoint(time=base + datetime.timedelta(hours=i), price=price)
        for i, price in enumerate(prices)
    ]


def test_full_slot_discharge_with_min_charge_slots():
    base, prices = _make_price_points()
    optimizer = PartialSlotOptimizer(now=base, soc_step_percent=1.0)
    schedule = optimizer.find_optimal_schedule(prices, 4, current_soc=36.0)
    assert schedule[base].mode == BatteryMode.DISCHARGE


def test_partial_slot_discharges_correctly():
    base, prices = _make_price_points()
    now = base + datetime.timedelta(minutes=53)
    optimizer = PartialSlotOptimizer(now=now, soc_step_percent=1.0)
    schedule = optimizer.find_optimal_schedule(prices, 4, current_soc=36.0)
    assert schedule[base].mode == BatteryMode.DISCHARGE


def test_partial_slot_supports_configured_finer_soc_steps():
    base, prices = _make_price_points()
    now = base + datetime.timedelta(minutes=53)
    optimizer = PartialSlotOptimizer(now=now, soc_step_percent=0.5)
    schedule = optimizer.find_optimal_schedule(prices, 4, current_soc=36.0)
    # The mode decision must not depend on the SOC grid resolution: this
    # scenario discharges at the default step and must still discharge at a
    # finer configured step (guards against rounding-direction bias).
    assert base in schedule
    assert schedule[base].mode == BatteryMode.DISCHARGE


def test_missing_current_slot_is_not_planned_on_an_invented_price():
    """The current slot is never given a price the source did not publish.

    This used to assert the opposite: `_ensure_current_slot_price` copied
    yesterday's same-clock price (0.071 here) into the current slot and the DP
    planned on it. The DP is now not given the interval at all; the slot gets
    the pre-solve HOLD/`no_price` fallback (there is nothing real-priced to
    retain here), which carries no provenance and so cannot be executed as
    anything but HOLD.
    """
    base, prices = _make_price_points()
    current = base
    future = [p for p in prices if p.time > current]
    optimizer = PartialSlotOptimizer(now=current, soc_step_percent=1.0)

    schedule = optimizer.find_optimal_schedule(future, 0, current_soc=36.0)

    entry = schedule[current]
    assert entry.mode == BatteryMode.HOLD
    assert entry.reason == "no_price"
    assert entry.price_source is None
    assert len(schedule) > 1, "the rest of the horizon is still planned"
    assert min(p.time for p in future) in schedule
