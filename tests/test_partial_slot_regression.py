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

    def _get_discharge_threshold_for_cost(self, avg_cost: float) -> float:
        return (avg_cost / self.config.efficiency) + self.config.grid_fee + self.config.battery_wear_cost

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
PartialSlotOptimizer._ensure_current_slot_price = BatteryOptimizer._ensure_current_slot_price
PartialSlotOptimizer._compute_slot_fractions = BatteryOptimizer._compute_slot_fractions
PartialSlotOptimizer._compute_charge_rates_per_slot = BatteryOptimizer._compute_charge_rates_per_slot


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


def test_partial_slot_discharges_with_finer_soc_steps():
    base, prices = _make_price_points()
    now = base + datetime.timedelta(minutes=53)
    optimizer = PartialSlotOptimizer(now=now, soc_step_percent=0.5)
    schedule = optimizer.find_optimal_schedule(prices, 4, current_soc=36.0)
    assert schedule[base].mode == BatteryMode.DISCHARGE
