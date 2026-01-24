import datetime

from battery_optimizer import (
    BatteryLearningEngine,
    BatteryMode,
    BatteryOptimizer,
    LoadProfile,
    PricePoint,
)


class PartialSlotOptimizer:
    """Minimal optimizer mock for partial-slot regression tests."""

    def __init__(
        self,
        now: datetime.datetime,
        soc_step_percent: float = 1.0,
        base_consumption_w: float = 850.0,
    ):
        self.battery_capacity = 14.3
        self.charge_rate = 4.5
        self.discharge_rate = 4.5
        self.efficiency = 0.95
        self.grid_fee = 0.0
        self.slot_minutes = 60
        self.slot_hours = 1.0
        self.battery_avg_cost = 0.1046
        self.base_consumption = base_consumption_w
        self.load_quantile = 0.75
        self.soc_step_percent = soc_step_percent
        self.decision_log_level = 0
        self.min_soc = 10.0
        self.max_soc = 100.0
        self._current_time = now
        self.battery_wear_cost = 0.0
        self._last_projected_costs = {}
        self._last_min_charge_slots = 0
        self._last_charge_slots = []

        self.learning_engine = BatteryLearningEngine(
            battery_capacity_kwh=self.battery_capacity,
            nominal_charge_rate_kw=self.charge_rate,
            nominal_efficiency=self.efficiency,
        )
        self.load_profile = LoadProfile(
            slot_minutes=self.slot_minutes,
            default_load_w=base_consumption_w,
        )

    def datetime(self):
        return self._current_time

    def log(self, *args, **kwargs):
        pass

    def _align_to_slot(self, dt: datetime.datetime) -> datetime.datetime:
        minutes = dt.hour * 60 + dt.minute
        slot_start = (minutes // self.slot_minutes) * self.slot_minutes
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
        return self.load_profile.predict_kw(dt, self.load_quantile)

    def _get_prices_for_date(self, date, tz):
        return []

    def _get_discharge_threshold(self) -> float:
        return (self.battery_avg_cost / self.efficiency) + self.grid_fee + self.battery_wear_cost

    def _get_discharge_threshold_for_cost(self, avg_cost: float) -> float:
        return (avg_cost / self.efficiency) + self.grid_fee + self.battery_wear_cost

    def _log_schedule_decision_context(self, *args, **kwargs):
        pass


PartialSlotOptimizer.find_optimal_schedule = BatteryOptimizer.find_optimal_schedule
PartialSlotOptimizer._project_battery_costs = BatteryOptimizer._project_battery_costs


def _make_price_points():
    base = datetime.datetime(2026, 1, 24, 16, 0, 0)
    prices = [
        0.1421, 0.1425, 0.1485, 0.1418, 0.1303, 0.1197, 0.1162, 0.1081,
        0.0963, 0.0951, 0.0984, 0.0971, 0.0961, 0.0953, 0.0941, 0.0909,
        0.0949, 0.0974, 0.1018, 0.1099, 0.1153, 0.1250, 0.1224, 0.1206,
    ]
    return base, [
        PricePoint(hour=base + datetime.timedelta(hours=i), price=price)
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
