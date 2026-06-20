"""
Tests for PvForecastService — Solcast and Forecast.Solar integration.
"""

import datetime
from unittest.mock import patch, MagicMock

import pytest

from battery_optimizer_lib.pv_forecast_service import (
    PvForecastService,
    PvForecastServiceConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TZ_PLUS2 = datetime.timezone(datetime.timedelta(hours=2))


def _make_service(
    config: PvForecastServiceConfig = None,
    states: dict = None,
    now: datetime.datetime = None,
) -> PvForecastService:
    """Create a PvForecastService with mock callbacks."""
    if config is None:
        config = PvForecastServiceConfig(slot_minutes=15)
    if now is None:
        now = datetime.datetime(2026, 3, 24, 12, 0, tzinfo=TZ_PLUS2)

    state_store = states or {}

    def mock_get_state(entity_id, attribute=None):
        val = state_store.get(entity_id)
        if attribute == "all" and isinstance(val, dict):
            return val
        if attribute and isinstance(val, dict):
            return val.get("attributes", {}).get(attribute)
        return val

    return PvForecastService(
        config=config,
        get_state_func=mock_get_state,
        get_datetime_func=lambda: now,
        get_timezone_func=lambda: TZ_PLUS2,
        log_func=lambda msg, **kw: None,
    )


def _solcast_entry(hour, minute, kw, kw10=None, kw90=None):
    """Build a single Solcast detailedForecast entry."""
    dt = datetime.datetime(2026, 3, 24, hour, minute, tzinfo=TZ_PLUS2)
    entry = {
        "period_start": dt.isoformat(),
        "pv_estimate": kw,
    }
    if kw10 is not None:
        entry["pv_estimate10"] = kw10
    if kw90 is not None:
        entry["pv_estimate90"] = kw90
    return entry


# ---------------------------------------------------------------------------
# Solcast Parsing
# ---------------------------------------------------------------------------

class TestSolcastFetch:

    def test_parse_today_detailed_forecast(self):
        """Solcast detailedForecast attribute is parsed into per-slot kW."""
        detailed = [
            _solcast_entry(8, 0, 0.5),   # 08:00-08:30 -> 0.5 kW
            _solcast_entry(8, 30, 1.2),   # 08:30-09:00 -> 1.2 kW
            _solcast_entry(9, 0, 2.0),    # 09:00-09:30 -> 2.0 kW
        ]

        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": detailed},
            }
        }

        svc = _make_service(config=config, states=states)
        svc.refresh()

        assert svc.has_forecast
        assert svc.last_source == "solcast"

        # 08:00 30-min period -> two 15-min slots at 08:00 and 08:15
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 8, 0, tzinfo=TZ_PLUS2)) == 0.5
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 8, 15, tzinfo=TZ_PLUS2)) == 0.5

        # 08:30 period -> 08:30 and 08:45
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 8, 30, tzinfo=TZ_PLUS2)) == 1.2
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 8, 45, tzinfo=TZ_PLUS2)) == 1.2

        # 09:00 period -> 09:00 and 09:15
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 9, 0, tzinfo=TZ_PLUS2)) == 2.0

    def test_30min_to_15min_expansion(self):
        """Each 30-min Solcast entry should produce exactly 2 x 15-min slots."""
        detailed = [_solcast_entry(10, 0, 3.0)]

        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "10.0",
                "attributes": {"detailedForecast": detailed},
            }
        }

        svc = _make_service(config=config, states=states)
        svc.refresh()

        # Should have exactly 2 slots
        assert len(svc._cache) == 2
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 0, tzinfo=TZ_PLUS2)) == 3.0
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 15, tzinfo=TZ_PLUS2)) == 3.0

    def test_estimate_field_selection_pv_estimate10(self):
        """Conservative estimate (pv_estimate10) should be used when configured."""
        detailed = [_solcast_entry(10, 0, kw=3.0, kw10=1.5, kw90=4.5)]

        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            solcast_estimate_field="pv_estimate10",
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": detailed},
            }
        }

        svc = _make_service(config=config, states=states)
        svc.refresh()

        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 0, tzinfo=TZ_PLUS2)) == 1.5

    def test_estimate_field_selection_pv_estimate90(self):
        """Optimistic estimate (pv_estimate90) should be used when configured."""
        detailed = [_solcast_entry(10, 0, kw=3.0, kw10=1.5, kw90=4.5)]

        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            solcast_estimate_field="pv_estimate90",
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": detailed},
            }
        }

        svc = _make_service(config=config, states=states)
        svc.refresh()

        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 0, tzinfo=TZ_PLUS2)) == 4.5

    def test_today_and_tomorrow_combined(self):
        """Forecast from both today and tomorrow entities should be merged."""
        today_detailed = [_solcast_entry(14, 0, 2.0)]
        # Tomorrow entry at same hour but different day
        tomorrow_dt = datetime.datetime(2026, 3, 25, 10, 0, tzinfo=TZ_PLUS2)
        tomorrow_detailed = [{
            "period_start": tomorrow_dt.isoformat(),
            "pv_estimate": 1.5,
        }]

        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            solcast_tomorrow_entity="sensor.solcast_tomorrow",
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": today_detailed},
            },
            "sensor.solcast_tomorrow": {
                "state": "4.0",
                "attributes": {"detailedForecast": tomorrow_detailed},
            },
        }

        svc = _make_service(config=config, states=states)
        svc.refresh()

        # Today's slot
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 14, 0, tzinfo=TZ_PLUS2)) == 2.0
        # Tomorrow's slot
        assert svc.predict_kw(datetime.datetime(2026, 3, 25, 10, 0, tzinfo=TZ_PLUS2)) == 1.5

    def test_negative_values_clamped_to_zero(self):
        """Negative PV values should be treated as zero."""
        detailed = [_solcast_entry(10, 0, -0.5)]

        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "1.0",
                "attributes": {"detailedForecast": detailed},
            },
        }

        svc = _make_service(config=config, states=states)
        svc.refresh()

        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 0, tzinfo=TZ_PLUS2)) == 0.0

    def test_missing_attribute_returns_empty(self):
        """If detailedForecast attribute is missing, should return empty."""
        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {},  # no detailedForecast
            },
        }

        svc = _make_service(config=config, states=states)
        result = svc.refresh()

        assert not svc.has_forecast

    def test_entity_unavailable_returns_empty(self):
        """If Solcast entity doesn't exist, should gracefully fall through."""
        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_nonexistent",
            slot_minutes=15,
        )

        svc = _make_service(config=config, states={})
        svc.refresh()

        assert not svc.has_forecast

    def test_slot_minutes_30(self):
        """With 30-min slots, each 30-min Solcast entry maps to exactly 1 slot."""
        detailed = [_solcast_entry(10, 0, 3.0), _solcast_entry(10, 30, 2.5)]

        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            slot_minutes=30,
        )
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": detailed},
            },
        }

        svc = _make_service(config=config, states=states)
        svc.refresh()

        assert len(svc._cache) == 2
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 0, tzinfo=TZ_PLUS2)) == 3.0
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 30, tzinfo=TZ_PLUS2)) == 2.5


# ---------------------------------------------------------------------------
# Forecast.Solar Parsing
# ---------------------------------------------------------------------------

class TestForecastSolarFetch:

    @patch("battery_optimizer_lib.pv_forecast_service.requests")
    def test_parse_wh_to_kw(self, mock_requests):
        """watt_hours_period Wh values should be correctly converted to kW."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "watt_hours_period": {
                    "2026-03-24 08:00:00": 500,   # 500 Wh in 1h = 0.5 kW avg
                    "2026-03-24 09:00:00": 1000,   # 1000 Wh in 1h = 1.0 kW avg
                    "2026-03-24 10:00:00": 2000,   # 2000 Wh in 1h = 2.0 kW avg
                }
            }
        }
        mock_requests.get.return_value = mock_response

        config = PvForecastServiceConfig(
            forecast_solar_lat=56.9,
            forecast_solar_lon=24.1,
            forecast_solar_declination=30,
            forecast_solar_azimuth=180,
            forecast_solar_kwp=5.0,
            slot_minutes=15,
        )

        svc = _make_service(config=config)
        svc.refresh()

        assert svc.has_forecast
        assert svc.last_source == "forecast_solar"

        # 500 Wh / 1h / 1000 = 0.5 kW, expanded to 4 x 15-min slots
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 8, 0, tzinfo=TZ_PLUS2)) == pytest.approx(0.5)
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 8, 15, tzinfo=TZ_PLUS2)) == pytest.approx(0.5)
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 8, 30, tzinfo=TZ_PLUS2)) == pytest.approx(0.5)
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 8, 45, tzinfo=TZ_PLUS2)) == pytest.approx(0.5)

        # 1000 Wh / 1h / 1000 = 1.0 kW
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 9, 0, tzinfo=TZ_PLUS2)) == pytest.approx(1.0)

        # 2000 Wh / 1h / 1000 = 2.0 kW
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 0, tzinfo=TZ_PLUS2)) == pytest.approx(2.0)

    @patch("battery_optimizer_lib.pv_forecast_service.requests")
    def test_api_url_without_key(self, mock_requests):
        """Free API URL should not include API key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"watt_hours_period": {}}}
        mock_requests.get.return_value = mock_response

        config = PvForecastServiceConfig(
            forecast_solar_lat=56.9,
            forecast_solar_lon=24.1,
            forecast_solar_declination=30,
            forecast_solar_azimuth=180,
            forecast_solar_kwp=5.0,
        )

        svc = _make_service(config=config)
        svc.refresh()

        call_url = mock_requests.get.call_args[0][0]
        assert call_url == "https://api.forecast.solar/estimate/56.9/24.1/30/180/5.0"

    @patch("battery_optimizer_lib.pv_forecast_service.requests")
    def test_api_url_with_key(self, mock_requests):
        """Paid API URL should include the API key."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"watt_hours_period": {}}}
        mock_requests.get.return_value = mock_response

        config = PvForecastServiceConfig(
            forecast_solar_lat=56.9,
            forecast_solar_lon=24.1,
            forecast_solar_declination=30,
            forecast_solar_azimuth=180,
            forecast_solar_kwp=5.0,
            forecast_solar_api_key="mykey123",
        )

        svc = _make_service(config=config)
        svc.refresh()

        call_url = mock_requests.get.call_args[0][0]
        assert call_url == "https://api.forecast.solar/mykey123/estimate/56.9/24.1/30/180/5.0"

    @patch("battery_optimizer_lib.pv_forecast_service.requests")
    def test_hourly_expansion_to_15min(self, mock_requests):
        """Hourly data should expand to 4 x 15-min slots."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "watt_hours_period": {
                    "2026-03-24 10:00:00": 2000,
                    "2026-03-24 11:00:00": 3000,
                }
            }
        }
        mock_requests.get.return_value = mock_response

        config = PvForecastServiceConfig(
            forecast_solar_lat=56.9,
            forecast_solar_lon=24.1,
            forecast_solar_declination=30,
            forecast_solar_azimuth=180,
            forecast_solar_kwp=5.0,
            slot_minutes=15,
        )

        svc = _make_service(config=config)
        svc.refresh()

        # 2000 Wh / 1h = 2.0 kW -> 4 slots
        assert len([k for k in svc._cache
                     if k.hour == 10]) == 4
        for m in [0, 15, 30, 45]:
            assert svc.predict_kw(
                datetime.datetime(2026, 3, 24, 10, m, tzinfo=TZ_PLUS2)
            ) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:

    def test_fresh_cache_not_refetched(self):
        """Refresh should no-op if cache is still fresh."""
        detailed = [_solcast_entry(10, 0, 3.0)]
        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            pv_forecast_cache_minutes=60,
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": detailed},
            },
        }

        svc = _make_service(config=config, states=states)

        # First refresh should fetch
        assert svc.refresh() is True
        assert svc.has_forecast

        # Second refresh should no-op (cache is fresh)
        assert svc.refresh() is False

    def test_stale_cache_triggers_refetch(self):
        """After cache_minutes elapse, refresh should re-fetch."""
        detailed = [_solcast_entry(10, 0, 3.0)]
        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            pv_forecast_cache_minutes=60,
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": detailed},
            },
        }

        now = datetime.datetime(2026, 3, 24, 12, 0, tzinfo=TZ_PLUS2)
        svc = _make_service(config=config, states=states, now=now)
        svc.refresh()

        # Simulate time passing beyond cache duration
        later = now + datetime.timedelta(minutes=61)
        svc.datetime = lambda: later

        assert svc.refresh() is True

    def test_stale_cache_kept_on_fetch_failure(self):
        """If sources fail, stale cache should be retained (within 3x limit)."""
        detailed = [_solcast_entry(10, 0, 3.0)]
        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            pv_forecast_cache_minutes=60,
            slot_minutes=15,
        )

        now = datetime.datetime(2026, 3, 24, 12, 0, tzinfo=TZ_PLUS2)

        # First: populate cache with good data
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": detailed},
            },
        }
        svc = _make_service(config=config, states=states, now=now)
        svc.refresh()
        assert svc.has_forecast

        # Now make sensor unavailable
        svc.get_state = lambda entity_id, attribute=None: None

        # 90 min later (within 3x60=180 limit), stale cache should be kept
        later = now + datetime.timedelta(minutes=90)
        svc.datetime = lambda: later
        svc.refresh()

        assert svc.has_forecast  # stale cache still present
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 0, tzinfo=TZ_PLUS2)) == 3.0

    def test_very_old_cache_cleared(self):
        """Cache older than 3x cache_minutes should be discarded."""
        detailed = [_solcast_entry(10, 0, 3.0)]
        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            pv_forecast_cache_minutes=60,
            slot_minutes=15,
        )

        now = datetime.datetime(2026, 3, 24, 12, 0, tzinfo=TZ_PLUS2)
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": detailed},
            },
        }
        svc = _make_service(config=config, states=states, now=now)
        svc.refresh()

        # Make sensor unavailable
        svc.get_state = lambda entity_id, attribute=None: None

        # 200 min later (beyond 3x60=180 limit)
        later = now + datetime.timedelta(minutes=200)
        svc.datetime = lambda: later
        svc.refresh()

        assert not svc.has_forecast  # cache cleared


# ---------------------------------------------------------------------------
# Fallback Chain
# ---------------------------------------------------------------------------

class TestFallbackChain:

    @patch("battery_optimizer_lib.pv_forecast_service.requests")
    def test_solcast_primary_over_forecast_solar(self, mock_requests):
        """When Solcast is available, Forecast.Solar should not be called."""
        detailed = [_solcast_entry(10, 0, 3.0)]
        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            forecast_solar_kwp=5.0,
            forecast_solar_lat=56.9,
            forecast_solar_lon=24.1,
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": detailed},
            },
        }

        svc = _make_service(config=config, states=states)
        svc.refresh()

        assert svc.last_source == "solcast"
        mock_requests.get.assert_not_called()

    @patch("battery_optimizer_lib.pv_forecast_service.requests")
    def test_forecast_solar_used_when_solcast_fails(self, mock_requests):
        """Forecast.Solar should be used if Solcast entity is unavailable."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {
                "watt_hours_period": {
                    "2026-03-24 10:00:00": 2000,
                }
            }
        }
        mock_requests.get.return_value = mock_response

        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_nonexistent",
            forecast_solar_lat=56.9,
            forecast_solar_lon=24.1,
            forecast_solar_declination=30,
            forecast_solar_azimuth=180,
            forecast_solar_kwp=5.0,
            slot_minutes=15,
        )

        svc = _make_service(config=config, states={})
        svc.refresh()

        assert svc.last_source == "forecast_solar"
        assert svc.has_forecast

    def test_empty_when_nothing_configured(self):
        """With no sources configured, service should have no data."""
        config = PvForecastServiceConfig(slot_minutes=15)

        svc = _make_service(config=config)
        svc.refresh()

        assert not svc.has_forecast
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 0, tzinfo=TZ_PLUS2)) == 0.0


# ---------------------------------------------------------------------------
# predict_kw Lookup
# ---------------------------------------------------------------------------

class TestPredictKw:

    def test_missing_slot_returns_zero(self):
        """Slots not covered by forecast should return 0.0."""
        detailed = [_solcast_entry(10, 0, 3.0)]
        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": detailed},
            },
        }

        svc = _make_service(config=config, states=states)
        svc.refresh()

        # Slot not in forecast (nighttime)
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 22, 0, tzinfo=TZ_PLUS2)) == 0.0

    def test_empty_cache_returns_zero(self):
        """predict_kw on empty cache should return 0.0."""
        svc = _make_service()
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 0, tzinfo=TZ_PLUS2)) == 0.0

    def test_naive_datetime_lookup(self):
        """predict_kw should work with naive datetimes too."""
        detailed = [_solcast_entry(10, 0, 3.0)]
        config = PvForecastServiceConfig(
            solcast_today_entity="sensor.solcast_today",
            slot_minutes=15,
        )
        states = {
            "sensor.solcast_today": {
                "state": "5.0",
                "attributes": {"detailedForecast": detailed},
            },
        }

        svc = _make_service(config=config, states=states)
        svc.refresh()

        # Naive datetime should also find the slot
        assert svc.predict_kw(datetime.datetime(2026, 3, 24, 10, 0)) == 3.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_solcast_configured(self):
        cfg = PvForecastServiceConfig(solcast_today_entity="sensor.x")
        assert cfg.solcast_configured is True

    def test_solcast_not_configured(self):
        cfg = PvForecastServiceConfig()
        assert cfg.solcast_configured is False

    def test_forecast_solar_configured(self):
        cfg = PvForecastServiceConfig(forecast_solar_kwp=5.0)
        assert cfg.forecast_solar_configured is True

    def test_forecast_solar_not_configured(self):
        cfg = PvForecastServiceConfig()
        assert cfg.forecast_solar_configured is False

    def test_from_main_config(self):
        """from_main_config should extract all relevant fields."""
        from battery_optimizer_lib.config import BatteryOptimizerConfig

        main_cfg = BatteryOptimizerConfig(
            solcast_today_entity="sensor.solcast_today",
            solcast_tomorrow_entity="sensor.solcast_tomorrow",
            solcast_estimate_field="pv_estimate10",
            forecast_solar_lat=56.9,
            forecast_solar_lon=24.1,
            forecast_solar_declination=30,
            forecast_solar_azimuth=180,
            forecast_solar_kwp=5.0,
            forecast_solar_api_key="key123",
            pv_forecast_cache_minutes=30,
            slot_minutes=15,
        )

        cfg = PvForecastServiceConfig.from_main_config(main_cfg)

        assert cfg.solcast_today_entity == "sensor.solcast_today"
        assert cfg.solcast_tomorrow_entity == "sensor.solcast_tomorrow"
        assert cfg.solcast_estimate_field == "pv_estimate10"
        assert cfg.forecast_solar_lat == 56.9
        assert cfg.forecast_solar_lon == 24.1
        assert cfg.forecast_solar_declination == 30
        assert cfg.forecast_solar_azimuth == 180
        assert cfg.forecast_solar_kwp == 5.0
        assert cfg.forecast_solar_api_key == "key123"
        assert cfg.pv_forecast_cache_minutes == 30
        assert cfg.slot_minutes == 15
