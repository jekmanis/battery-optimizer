"""The current slot is never planned on a price nobody published.

Defect (pre-fix)
----------------

`PriceHorizonMonitor` reports `missing_current_interval` and the orchestrator
arms a bounded retry - and then plans the slot anyway.
`find_optimal_schedule` called `_ensure_current_slot_price`, which substituted,
in order, yesterday's same-clock price, the most recent past price, or the next
price.  The DP then planned the current slot on that number and
`execute_scheduled_mode` sent the resulting command.

The maintainer's reproduction: the current price is absent, yesterday's
corresponding price is 0.01 EUR/kWh and the next slot is 1.00 EUR/kWh.  The
planner produced a CHARGE for the current slot priced at 0.01 and the inverter
was told to charge.  A scheduled retry does not prevent that command.

Policy under test
-----------------

1. Nothing is synthesized.  Both fetch paths request WHOLE DAYS
   (`_get_prices_via_service` calls `get_price_indices_for_date` per date;
   `_get_prices_via_sensor` reads `raw_today` / `raw_tomorrow`), so an absent
   current interval means the data is genuinely missing, not that the source
   trimmed the past.
2. A current slot with no validated price either RETAINS a previously planned
   entry that was itself built from a real price, or falls back to
   HOLD / `no_price`.  Future slots are still planned from the next validated
   interval.
3. `execute_scheduled_mode` refuses to send any non-HOLD current-slot entry
   that carries no real-price provenance.

Everything here is deterministic: the settable clock and AppDaemon double from
`test_price_recovery`, the REAL planner, and the real `NordPoolPriceService`
driven from scripted HA responses.  No AppDaemon, no HA, no network.
"""

from __future__ import annotations

import datetime

import pytest

import battery_optimizer as bo
from battery_optimizer_lib import (
    BatteryCostConfig,
    BatteryCostTracker,
    BatteryLearningEngine,
    BatteryMode,
    LoadProfile,
    NordPoolPriceService,
    PricePoint,
    ScheduleEntry,
)
from battery_optimizer_lib.models import PRICE_SOURCE_MARKET

from tests.test_price_recovery import (
    TZ,
    UTC,
    RecoveryOptimizer,
    day_start,
    full_day_prices,
    make_prices,
    slots_between,
)


SLOT = 15


# ---------------------------------------------------------------------------
# A double with the REAL planner
# ---------------------------------------------------------------------------

class PlanningOptimizer(RecoveryOptimizer):
    """`RecoveryOptimizer` with `find_optimal_schedule` NOT stubbed.

    The reproduction is about a price, so the DP has to actually run: a stub
    planner can never produce a CHARGE at 0.01 EUR/kWh, and a test that asserts
    it does not would pass for the wrong reason.
    """

    find_optimal_schedule = bo.BatteryOptimizer.find_optimal_schedule
    calculate_min_charge_slots_for_horizon = (
        bo.BatteryOptimizer.calculate_min_charge_slots_for_horizon
    )

    def __init__(self, now, prices=None, **kwargs):
        kwargs.setdefault("battery_capacity", 14.3)
        kwargs.setdefault("charge_rate", 4.5)
        kwargs.setdefault("discharge_rate", 4.5)
        kwargs.setdefault("efficiency", 0.85)
        kwargs.setdefault("grid_fee", 0.05)
        kwargs.setdefault("battery_wear_cost", 0.0)
        kwargs.setdefault("soc_step_percent", 1.0)
        kwargs.setdefault("base_consumption", 500.0)
        super().__init__(now, prices=prices, **kwargs)

        self.learning_engine = BatteryLearningEngine(
            battery_capacity_kwh=self.config.battery_capacity,
            nominal_charge_rate_kw=self.config.charge_rate,
            nominal_efficiency=self.config.efficiency,
            log_func=lambda *a, **k: None,
        )
        self.load_profile = LoadProfile(
            slot_minutes=self.config.slot_minutes,
            default_load_w=self.config.base_consumption,
            log_func=lambda *a, **k: None,
        )
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
            get_state_func=lambda e, **kw: None,
            call_service_func=lambda *a, **k: None,
            get_datetime_func=self.datetime,
            get_timezone_func=self._get_local_timezone,
            align_to_slot_func=self._align_to_slot,
            get_min_soc_func=lambda: self.min_soc,
            get_max_soc_func=lambda: self.max_soc,
            get_current_soc_func=lambda: self._soc,
            get_battery_temp_func=lambda: 20.0,
            learning_engine=self.learning_engine,
            get_cached_prices_func=lambda: [],
            save_learning_data_func=lambda: None,
            update_learning_sensor_func=lambda: None,
            log_func=lambda *a, **k: None,
        )

        # The real planner refreshes the forecast caches before solving.
        self._pv_forecast_service = _RefreshablePvForecast()

    def _predict_load_kw(self, dt):
        return 0.5

    def _predict_pv_kw(self, dt):
        return 0.0


class _RefreshablePvForecast:
    has_forecast = True

    def refresh(self):
        return None


# ---------------------------------------------------------------------------
# Price sources: the two REAL fetch paths, driven from scripted HA responses
# ---------------------------------------------------------------------------

def _service_payload(points, area="LV"):
    """`nordpool.get_price_indices_for_date` response shape (EUR/MWh)."""
    return {
        area: [
            {
                "start": p.time.astimezone(UTC).isoformat(),
                "end": (p.time.astimezone(UTC)
                        + datetime.timedelta(minutes=SLOT)).isoformat(),
                "price": p.price * 1000.0,
            }
            for p in points
        ]
    }


def _sensor_payload(points):
    """HACS Nord Pool `raw_today` / `raw_tomorrow` shape (EUR/kWh)."""
    return [
        {
            "start": p.time.isoformat(),
            "end": (p.time.astimezone(UTC)
                    + datetime.timedelta(minutes=SLOT)).astimezone(TZ).isoformat(),
            "value": p.price,
        }
        for p in points
    ]


def attach_service_path(app, points_by_date):
    """Wire the REAL price service to its built-in-integration path."""
    def call_service(service, **kwargs):
        if not service.endswith("get_price_indices_for_date"):
            return None
        date = datetime.date.fromisoformat(kwargs["date"])
        points = points_by_date.get(date, [])
        return _service_payload(points) if points else None

    app._price_service = NordPoolPriceService(
        nordpool_config_entry="cfg",
        nordpool_area="LV",
        nordpool_sensor="",
        ha_url="",           # forces the call_service fallback, no REST
        ha_token="",
        tomorrow_prices_hour=app.config.tomorrow_prices_hour,
        slot_minutes=app.config.slot_minutes,
        get_state_func=lambda *a, **k: None,
        call_service_func=call_service,
        get_datetime_func=app.datetime,
        get_date_func=lambda: app.datetime().date(),
        get_timezone_func=app._get_local_timezone,
        log_func=lambda *a, **k: None,
    )
    return app._price_service


def attach_sensor_path(app, today_points, tomorrow_points=()):
    """Wire the REAL price service to its HACS-sensor path."""
    def get_state(entity, attribute=None):
        return {
            "attributes": {
                "raw_today": _sensor_payload(today_points),
                "raw_tomorrow": _sensor_payload(tomorrow_points),
            }
        }

    app._price_service = NordPoolPriceService(
        nordpool_config_entry="",     # no built-in integration -> sensor path
        nordpool_area="LV",
        nordpool_sensor="sensor.nordpool",
        ha_url="",
        ha_token="",
        tomorrow_prices_hour=app.config.tomorrow_prices_hour,
        slot_minutes=app.config.slot_minutes,
        get_state_func=get_state,
        call_service_func=lambda *a, **k: None,
        get_datetime_func=app.datetime,
        get_date_func=lambda: app.datetime().date(),
        get_timezone_func=app._get_local_timezone,
        log_func=lambda *a, **k: None,
    )
    return app._price_service


# ---------------------------------------------------------------------------
# The maintainer's scenario
# ---------------------------------------------------------------------------

CHEAP = 0.01      # yesterday's same-clock price
EXPENSIVE = 1.00  # the next (real) interval


def maintainer_scenario(now):
    """Prices for today with the CURRENT interval missing.

    Yesterday's same-clock interval is 0.01 EUR/kWh; every real interval from
    the next slot on is 1.00.  Charging the current slot at 0.01 and
    discharging into a 1.00 slot is overwhelmingly profitable, so a planner
    that accepts the invented number WILL choose CHARGE.
    """
    current_slot = now.replace(second=0, microsecond=0)
    current_slot = current_slot.replace(
        minute=(current_slot.minute // SLOT) * SLOT
    )
    today = day_start(now)
    tomorrow = day_start(now + datetime.timedelta(days=1))

    today_points = [
        p for p in make_prices(today, slots_between(today, tomorrow), price=EXPENSIVE)
        if p.time.astimezone(UTC) != current_slot.astimezone(UTC)
    ]
    yesterday = day_start(now - datetime.timedelta(days=1))
    yesterday_points = make_prices(
        yesterday, slots_between(yesterday, today), price=CHEAP
    )
    return current_slot, today_points, yesterday_points


@pytest.fixture
def scenario_now():
    """Mid-slot on a plain day, before the tomorrow-price publication window."""
    return datetime.datetime(2024, 1, 15, 10, 7, tzinfo=TZ)


def current_entry(app, slot=None):
    slot = slot if slot is not None else app._align_to_slot(app.datetime())
    return bo.lookup_by_time(app.schedule, slot, app._get_local_timezone())


# ===========================================================================
# Reproduction: no CHARGE at an invented price, on BOTH fetch paths
# ===========================================================================

class TestNoInventedCurrentSlotPrice:
    def test_service_path_does_not_charge_at_yesterdays_price(self, scenario_now):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot, today_points, yesterday_points = maintainer_scenario(scenario_now)
        attach_service_path(app, {
            scenario_now.date(): today_points,
            (scenario_now - datetime.timedelta(days=1)).date(): yesterday_points,
        })

        app.full_optimize(None)

        entry = current_entry(app, current_slot)
        assert entry is None or entry.mode != BatteryMode.CHARGE, (
            "the current slot must not be planned on yesterday's 0.01 price"
        )
        assert app.applied, "the slot is still executed"
        assert app.applied[-1].mode != BatteryMode.CHARGE, (
            "no CHARGE command may be sent for an unpriced slot"
        )

    def test_sensor_path_does_not_price_the_current_slot_from_the_next_one(
        self, scenario_now
    ):
        """The HACS sensor path has no per-date accessor, so the old code fell
        through to the NEXT interval's price. Borrowing 1.00 for a slot nobody
        priced is the same defect wearing a different number: the slot is
        planned, logged and executed on a price that was never published."""
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot, today_points, _ = maintainer_scenario(scenario_now)
        attach_sensor_path(app, today_points)

        app.full_optimize(None)

        entry = current_entry(app, current_slot)
        assert entry is None or entry.price_source is None
        assert app.applied and app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "no_price"

    def test_the_previous_price_is_not_substituted_either(self, scenario_now):
        """The old code's second fallback: the most recent PAST price.

        `full_optimize` filters the past out before it calls the planner, so
        this branch is only reachable by a caller that passes an unfiltered
        list - which is exactly why it must not exist rather than merely be
        unreachable from one call site.
        """
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot, today_points, _ = maintainer_scenario(scenario_now)
        past = current_slot - datetime.timedelta(minutes=SLOT)
        unfiltered = [
            p for p in today_points if p.time.astimezone(UTC) != past.astimezone(UTC)
        ] + [PricePoint(time=past, price=CHEAP)]

        schedule = app.find_optimal_schedule(unfiltered, 0, current_soc=50.0)

        entry = bo.lookup_by_time(schedule, current_slot, app._get_local_timezone())
        assert entry is None, (
            "an unpriced current slot must not appear in a generated schedule"
        )

    def test_the_planner_never_sees_a_synthetic_current_price(self, scenario_now):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot, today_points, yesterday_points = maintainer_scenario(scenario_now)
        attach_service_path(app, {
            scenario_now.date(): today_points,
            (scenario_now - datetime.timedelta(days=1)).date(): yesterday_points,
        })

        app.full_optimize(None)

        assert not any(
            f"{CHEAP:.4f}" in message for message, _ in app.logs
        ), "no log line may report a synthesized current-slot price"

    def test_future_slots_are_still_planned(self, scenario_now):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.full_optimize(None)

        later = [
            key for key in app.schedule
            if key.astimezone(UTC) > current_slot.astimezone(UTC)
        ]
        assert len(later) > 4, (
            "an unpriced current slot must not cost the rest of the horizon"
        )

    def test_the_retry_stays_armed(self, scenario_now):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.full_optimize(None)

        assert len(app.pending_retries()) == 1
        assert app._price_horizon.last_health.reason == "missing_current_interval"


# ===========================================================================
# What the current slot does instead
# ===========================================================================

class TestFallbackAndRetention:
    def _app_with_a_real_priced_plan(self, scenario_now, mode=BatteryMode.DISCHARGE):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot = app._align_to_slot(scenario_now)
        app.schedule = {
            bo.canonical_slot_key(current_slot): ScheduleEntry(
                time=current_slot,
                mode=mode,
                reason="0.9000 EUR/kWh load~0.50kW",
                price_source=PRICE_SOURCE_MARKET,
            )
        }
        return app, current_slot

    def test_no_previous_entry_holds_with_reason_no_price(self, scenario_now):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.full_optimize(None)

        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "no_price"

    def test_a_real_priced_entry_is_retained(self, scenario_now):
        app, current_slot = self._app_with_a_real_priced_plan(scenario_now)
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.full_optimize(None)

        entry = current_entry(app, current_slot)
        assert entry is not None, "the real-priced entry must survive the rebuild"
        assert entry.mode == BatteryMode.DISCHARGE
        assert app.applied[-1].mode == BatteryMode.DISCHARGE

    def test_a_retained_entry_still_respects_the_min_soc_guard(self, scenario_now):
        app, current_slot = self._app_with_a_real_priced_plan(scenario_now)
        app._soc = app.min_soc
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.full_optimize(None)

        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "safety_min_soc"

    def test_a_retained_entry_still_respects_the_max_soc_guard(self, scenario_now):
        """The retention path is how a CHARGE reaches a full pack.

        The safety HOLD makes the current interval unpriced from the plan's
        point of view; the rebuild then retains the previous real-priced CHARGE
        entry and re-sends it. With the SOC pinned at max no SOC event fires to
        undo it, so the execution guard is the only thing that can.
        """
        app, current_slot = self._app_with_a_real_priced_plan(
            scenario_now, mode=BatteryMode.CHARGE
        )
        app._soc = app.max_soc
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.full_optimize(None)

        assert current_entry(app, current_slot) is not None, (
            "the real-priced entry is still retained"
        )
        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "safety_max_soc"

    def test_a_retained_charge_below_max_soc_is_still_sent(self, scenario_now):
        app, current_slot = self._app_with_a_real_priced_plan(
            scenario_now, mode=BatteryMode.CHARGE
        )
        app._soc = app.max_soc - 1.0
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.full_optimize(None)

        assert app.applied[-1].mode == BatteryMode.CHARGE

    def test_a_retained_entry_sends_nothing_during_a_manual_override(
        self, scenario_now
    ):
        app, current_slot = self._app_with_a_real_priced_plan(scenario_now)
        app.override = True
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.full_optimize(None)

        assert app.applied == [], "manual override sends nothing"
        assert current_entry(app, current_slot) is not None, (
            "the plan is still refreshed during an override"
        )

    def test_a_disabled_optimizer_sends_nothing(self, scenario_now):
        app, _ = self._app_with_a_real_priced_plan(scenario_now)
        app.enabled = False
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.full_optimize(None)

        assert app.applied == []
        assert app.pending_retries() == []

    def test_an_entry_without_provenance_is_not_executed(self, scenario_now):
        """Guard for step 3: nothing should reach here after step 1, but a
        current-slot entry that cannot prove a real price executes as HOLD."""
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot = app._align_to_slot(scenario_now)
        app.schedule = {
            bo.canonical_slot_key(current_slot): ScheduleEntry(
                time=current_slot,
                mode=BatteryMode.CHARGE,
                reason="0.0100 EUR/kWh load~0.50kW",
                price_source=None,
            )
        }
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.execute_scheduled_mode(None)

        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "unpriced_slot"
        assert any("unpriced" in message.lower() for message, _ in app.logs)

    def test_a_priced_current_slot_is_planned_normally(self, scenario_now):
        """Control: with the interval present nothing is retained or held."""
        app = PlanningOptimizer(scenario_now, soc=50.0)
        today = day_start(scenario_now)
        tomorrow = day_start(scenario_now + datetime.timedelta(days=1))
        attach_service_path(app, {
            scenario_now.date(): make_prices(
                today, slots_between(today, tomorrow), price=0.10
            )
        })

        app.full_optimize(None)

        entry = current_entry(app)
        assert entry is not None
        assert entry.price_source == PRICE_SOURCE_MARKET
        assert entry.reason not in ("no_price", "no_schedule")


# ===========================================================================
# Recovery: the real price arrives
# ===========================================================================

class TestRecoveryRebuildsAndApplies:
    def test_the_plan_is_rebuilt_and_applied_when_the_price_arrives(
        self, scenario_now
    ):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot, today_points, _ = maintainer_scenario(scenario_now)
        by_date = {scenario_now.date(): today_points}
        attach_service_path(app, by_date)

        app.full_optimize(None)
        assert app.applied[-1].reason == "no_price"

        # The source publishes the missing interval at a REAL price.
        by_date[scenario_now.date()] = today_points + [
            PricePoint(time=current_slot, price=EXPENSIVE)
        ]
        app.advance(30)
        app.fire_latest_retry()

        entry = current_entry(app, current_slot)
        assert entry is not None
        assert entry.price_source == PRICE_SOURCE_MARKET
        assert app.applied[-1].reason != "no_price"
        assert app._price_horizon.attempts == 0
        assert not app._price_retry_pending(), "recovery disarms the retry"

    def test_recovery_during_an_override_refreshes_without_sending(
        self, scenario_now
    ):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot, today_points, _ = maintainer_scenario(scenario_now)
        by_date = {scenario_now.date(): today_points}
        attach_service_path(app, by_date)
        app.full_optimize(None)
        sent_before = len(app.applied)

        app.override = True
        by_date[scenario_now.date()] = today_points + [
            PricePoint(time=current_slot, price=EXPENSIVE)
        ]
        app.advance(30)
        app.fire_latest_retry()

        assert len(app.applied) == sent_before, "override sends nothing"
        assert current_entry(app, current_slot) is not None


# ===========================================================================
# Instants, not clock faces
# ===========================================================================

class TestSlotMatchingOnInstants:
    def test_the_repeated_autumn_03_00_is_not_reported_missing(
        self, riga_timezone
    ):
        """Both 03:00 intervals exist on 2024-10-27; the SECOND one is current.

        Matching on the local clock face finds the first occurrence and calls
        the interval present when the app is living in the second (or the other
        way round). Only the UTC instant separates them.
        """
        second_03 = datetime.datetime(
            2024, 10, 27, 3, 0, tzinfo=riga_timezone, fold=1
        )
        # NOT `second_03 + timedelta(minutes=7)`: datetime arithmetic zeroes
        # `fold`, which would silently put "now" in the FIRST 03:00 and make
        # this test pass without testing anything.
        now = datetime.datetime(2024, 10, 27, 3, 7, tzinfo=riga_timezone, fold=1)
        app = PlanningOptimizer(now, tz=riga_timezone, soc=50.0)
        start = datetime.datetime(2024, 10, 27, 0, 0, tzinfo=riga_timezone)
        end = datetime.datetime(2024, 10, 28, 0, 0, tzinfo=riga_timezone)
        full = make_prices(start, slots_between(start, end), tz=riga_timezone,
                           price=0.10)
        assert len(full) == 100, "an autumn Riga day is 100 quarter-hours"

        health = app._price_horizon.evaluate(full, app.datetime())
        assert health.has_current, "the second 03:00 interval IS present"

        # Removing only the FIRST 03:00 must not make the second look present.
        first_03 = datetime.datetime(
            2024, 10, 27, 3, 0, tzinfo=riga_timezone, fold=0
        )
        without_second = [
            p for p in full
            if p.time.astimezone(UTC) != second_03.astimezone(UTC)
        ]
        assert any(
            p.time.astimezone(UTC) == first_03.astimezone(UTC)
            for p in without_second
        ), "the first 03:00 is still there"
        health = app._price_horizon.evaluate(without_second, app.datetime())
        assert not health.has_current, (
            "the first 03:00 must not stand in for the second"
        )

    def test_midnight_rollover_does_not_borrow_yesterdays_first_interval(self):
        """00:00 is missing from the new day while yesterday is complete.

        This is where the yesterday-same-clock substitution was most dangerous:
        the borrowed interval is a night price, so the plan would charge at a
        number nobody published for a day that has barely started.
        """
        now = datetime.datetime(2024, 1, 16, 0, 3, tzinfo=TZ)
        today = day_start(now)
        tomorrow = day_start(now + datetime.timedelta(days=1))
        yesterday = day_start(now - datetime.timedelta(days=1))
        current_slot = today

        app = PlanningOptimizer(now, soc=50.0)
        today_points = [
            p for p in make_prices(
                today, slots_between(today, tomorrow), price=EXPENSIVE
            )
            if p.time.astimezone(UTC) != current_slot.astimezone(UTC)
        ]
        attach_service_path(app, {
            now.date(): today_points,
            (now - datetime.timedelta(days=1)).date(): make_prices(
                yesterday, slots_between(yesterday, today), price=CHEAP
            ),
        })

        app.full_optimize(None)

        assert app._price_horizon.last_health.reason == "missing_current_interval"
        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "no_price"
        assert len(app.pending_retries()) == 1
        assert app.schedule, "the rest of the new day is still planned"

    def test_the_last_interval_of_the_day_is_priced_normally(self):
        """Control for the rollover: 23:45 exists, tomorrow does not.

        Coverage is short (`tomorrow_missing`), but the CURRENT interval is
        published, so nothing is retained or held.
        """
        now = datetime.datetime(2024, 1, 15, 23, 50, tzinfo=TZ)
        today = day_start(now)
        tomorrow = day_start(now + datetime.timedelta(days=1))
        app = PlanningOptimizer(now, soc=50.0)
        attach_service_path(app, {
            now.date(): make_prices(
                today, slots_between(today, tomorrow), price=0.10
            )
        })

        app.full_optimize(None)

        entry = current_entry(app)
        assert entry is not None
        assert entry.price_source == PRICE_SOURCE_MARKET
        assert app.applied[-1].reason not in ("no_price", "no_schedule")


# ===========================================================================
# Diagnostics
# ===========================================================================

class TestDiagnostics:
    def test_the_payload_reports_an_unpriced_current_slot(self, scenario_now):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.full_optimize(None)

        diag = app._price_horizon_diagnostics()
        assert diag["current_slot_priced"] is False
        assert diag["current_slot_entry"] == "fallback"

    def test_the_payload_reports_a_retained_entry(self, scenario_now):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot = app._align_to_slot(scenario_now)
        app.schedule = {
            bo.canonical_slot_key(current_slot): ScheduleEntry(
                time=current_slot,
                mode=BatteryMode.HOLD,
                reason="0.9000 EUR/kWh load~0.50kW",
                price_source=PRICE_SOURCE_MARKET,
            )
        }
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app.full_optimize(None)

        diag = app._price_horizon_diagnostics()
        assert diag["current_slot_priced"] is False
        assert diag["current_slot_entry"] == "retained"

    def test_the_payload_does_not_report_a_stale_retention(self, scenario_now):
        """"retained" describes ONE interval; the next slot must not inherit
        it just because nothing has rebuilt since."""
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot = app._align_to_slot(scenario_now)
        app.schedule = {
            bo.canonical_slot_key(current_slot): ScheduleEntry(
                time=current_slot,
                mode=BatteryMode.HOLD,
                reason="0.9000 EUR/kWh load~0.50kW",
                price_source=PRICE_SOURCE_MARKET,
            )
        }
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})
        app.full_optimize(None)
        assert app._price_horizon_diagnostics()["current_slot_entry"] == "retained"

        app.advance(SLOT * 60)

        diag = app._price_horizon_diagnostics()
        assert diag["current_slot_entry"] is None
        assert diag["current_slot_priced"] is None

    def test_the_payload_reports_a_normally_planned_slot(self, scenario_now):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        today = day_start(scenario_now)
        tomorrow = day_start(scenario_now + datetime.timedelta(days=1))
        attach_service_path(app, {
            scenario_now.date(): make_prices(
                today, slots_between(today, tomorrow), price=0.10
            )
        })

        app.full_optimize(None)

        diag = app._price_horizon_diagnostics()
        assert diag["current_slot_priced"] is True
        assert diag["current_slot_entry"] == "planned"


# ===========================================================================
# Every planning path, not just the daily one
# ===========================================================================

class TestEveryPlanningPath:
    def test_recalculate_remaining_schedule_retains_and_falls_back(
        self, scenario_now
    ):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot = app._align_to_slot(scenario_now)
        app.schedule = {
            bo.canonical_slot_key(current_slot): ScheduleEntry(
                time=current_slot,
                mode=BatteryMode.DISCHARGE,
                reason="0.9000 EUR/kWh load~0.50kW",
                price_source=PRICE_SOURCE_MARKET,
            )
        }
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app._recalculate_remaining_schedule(app._soc)

        entry = current_entry(app, current_slot)
        assert entry is not None and entry.mode == BatteryMode.DISCHARGE
        assert app.applied[-1].mode == BatteryMode.DISCHARGE

    def test_recalculate_without_a_previous_entry_holds(self, scenario_now):
        app = PlanningOptimizer(scenario_now, soc=50.0)
        _, today_points, _ = maintainer_scenario(scenario_now)
        attach_service_path(app, {scenario_now.date(): today_points})

        app._recalculate_remaining_schedule(app._soc)

        assert app.applied[-1].mode == BatteryMode.HOLD
        assert app.applied[-1].reason == "no_price"

    def test_the_adaptive_horizon_extension_does_not_invent_a_price(
        self, scenario_now
    ):
        """`_check_price_horizon_health` rebuilds when the plan ran out. With
        the current interval unpriced the rebuild must not produce one."""
        app = PlanningOptimizer(scenario_now, soc=50.0)
        current_slot, today_points, yesterday_points = maintainer_scenario(scenario_now)
        attach_service_path(app, {
            scenario_now.date(): today_points,
            (scenario_now - datetime.timedelta(days=1)).date(): yesterday_points,
        })
        app.get_prices()          # populate the monitor's retained snapshot

        app.adaptive_optimize(None)

        entry = current_entry(app, current_slot)
        assert entry is None or entry.mode != BatteryMode.CHARGE
        assert all(e.mode != BatteryMode.CHARGE for e in app.applied)
