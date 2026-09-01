"""
Tests for AmbientTemperatureService.

Regression coverage for DEFECT 5 and DEFECT 7:

* DEFECT 5 — ambient came only from ``min(recent battery temperatures)``. In the
  33 h production window the real values were 27.4C (07-27 08:00) -> 33.0C
  (15:00) -> 30.5C (07-28 11:14) -> 35C (16:00), but the min-window returned
  ~30.5C against an actual 33C, i.e. a 2.5C delta.
* DEFECT 7 — that estimate was a single SCALAR for the whole horizon. The last
  logged schedule showed 34C for 1.5 h and then 33C through the night AND the
  next day's noon.
"""

import datetime

import pytest

from battery_optimizer_lib.ambient_service import (
    AmbientServiceConfig,
    AmbientTemperatureService,
)

BASE = datetime.datetime(2026, 7, 27, 8, 0)

# Real hourly outdoor temperatures from the analysed window.
REAL_TEMPS = [27.4, 28.0, 29.5, 31.0, 32.0, 33.0, 32.0, 30.0]


def _forecast_entries(start, temps):
    return [
        {
            "datetime": (start + datetime.timedelta(hours=i)).isoformat(),
            "temperature": t,
        }
        for i, t in enumerate(temps)
    ]


def _service(config=None, **kwargs):
    cfg = config or AmbientServiceConfig(slot_minutes=15)
    kwargs.setdefault("get_datetime_func", lambda: BASE)
    kwargs.setdefault("get_timezone_func", lambda: None)
    return AmbientTemperatureService(config=cfg, **kwargs)


class TestWeatherForecastSource:
    def test_weather_forecast_gives_time_varying_ambient(self):
        """A real hourly forecast makes ambient a function of time.

        Before the fix ambient was one constant for the whole 33 h horizon,
        so tomorrow's ~27C morning slots were modelled at today's 33C.
        """
        calls = []

        def call_service(service, **kwargs):
            calls.append((service, kwargs))
            return {
                "weather.home": {
                    "forecast": _forecast_entries(BASE, REAL_TEMPS)
                }
            }

        svc = _service(
            AmbientServiceConfig(weather_entity="weather.home", slot_minutes=15),
            call_service_func=call_service,
        )
        assert svc.refresh() is True
        assert svc.has_forecast is True
        assert svc.source == "forecast"
        assert calls[0][0] == "weather/get_forecasts"
        assert calls[0][1]["type"] == "hourly"

        values = [
            svc.predict_c(BASE + datetime.timedelta(hours=i))
            for i in range(len(REAL_TEMPS))
        ]
        assert values == pytest.approx(REAL_TEMPS)
        assert len(set(values)) > 1

    def test_hourly_forecast_is_expanded_to_slots(self):
        svc = _service(
            AmbientServiceConfig(weather_entity="weather.home", slot_minutes=15),
            call_service_func=lambda s, **k: {
                "forecast": _forecast_entries(BASE, REAL_TEMPS)
            },
        )
        svc.refresh()
        for minute in (0, 15, 30, 45):
            got = svc.predict_c(BASE + datetime.timedelta(minutes=minute))
            assert got == pytest.approx(27.4)

    def test_legacy_forecast_attribute_fallback(self):
        """Older HA installs only expose the `forecast` state attribute."""

        def call_service(service, **kwargs):
            raise RuntimeError("service not found")

        def get_state(entity, attribute=None):
            if attribute == "forecast":
                return _forecast_entries(BASE, REAL_TEMPS)
            return None

        svc = _service(
            AmbientServiceConfig(weather_entity="weather.home", slot_minutes=15),
            call_service_func=call_service,
            get_state_func=get_state,
        )
        assert svc.refresh() is True
        assert svc.has_forecast is True
        assert svc.predict_c(BASE) == pytest.approx(27.4)

    def test_broken_provider_does_not_raise(self):
        def call_service(service, **kwargs):
            raise RuntimeError("boom")

        svc = _service(
            AmbientServiceConfig(weather_entity="weather.home"),
            call_service_func=call_service,
            get_state_func=lambda e, attribute=None: None,
        )
        svc.refresh()
        assert svc.has_forecast is False
        assert svc.predict_c(BASE) is None


class TestOutdoorSensorSource:
    def test_sensor_reading_is_extended_by_diurnal_profile(self):
        cfg = AmbientServiceConfig(
            outdoor_temp_sensor="sensor.outdoor",
            diurnal_amplitude_c=4.0,
            diurnal_peak_hour=15.0,
        )
        svc = _service(
            cfg,
            get_state_func=lambda e, **kw: "33.0",
            get_datetime_func=lambda: datetime.datetime(2026, 7, 27, 15, 0),
        )
        svc.refresh()
        assert svc.source == "sensor" or svc.predict_c(BASE) is not None

        # Profile passes through the observed reading at the reading's hour.
        at_reading = svc.predict_c(datetime.datetime(2026, 7, 27, 15, 0))
        assert at_reading == pytest.approx(33.0, abs=0.01)

        # ...and drops overnight instead of staying at 33C.
        at_night = svc.predict_c(datetime.datetime(2026, 7, 28, 3, 0))
        assert at_night == pytest.approx(25.0, abs=0.01)
        assert at_reading - at_night > 7.0

    def test_invalid_sensor_value_is_ignored(self):
        cfg = AmbientServiceConfig(outdoor_temp_sensor="sensor.outdoor")
        svc = _service(cfg, get_state_func=lambda e, **kw: "unavailable")
        svc.refresh()
        assert svc.predict_c(BASE) is None


class TestMinWindowFallback:
    def test_fallback_diurnal_is_not_constant(self):
        """Even the heuristic fallback must vary across the horizon.

        The rolling battery minimum anchors the daily MAXIMUM of a diurnal
        profile (the pack is self-heated, so its minimum is a CEILING on
        ambient) instead of being the ambient at every hour of a 33 h horizon.
        """
        cfg = AmbientServiceConfig(
            diurnal_amplitude_c=4.0, diurnal_peak_hour=15.0, slot_minutes=15
        )
        svc = _service(cfg, min_temp_provider=lambda: 27.0)
        svc.refresh()

        horizon = [
            svc.predict_c(BASE + datetime.timedelta(minutes=15 * i))
            for i in range(33 * 4)
        ]
        assert all(v is not None for v in horizon)

        assert min(horizon) == pytest.approx(19.0, abs=0.05)
        assert max(horizon) == pytest.approx(27.0, abs=0.05)
        assert max(horizon) - min(horizon) >= 7.0

        # Peak near 15:00, trough near 03:00
        assert svc.predict_c(datetime.datetime(2026, 7, 27, 15, 0)) == pytest.approx(27.0, abs=0.05)
        assert svc.predict_c(datetime.datetime(2026, 7, 28, 3, 0)) == pytest.approx(19.0, abs=0.05)

    def test_fallback_never_exceeds_the_rolling_battery_minimum(self):
        """Regression: the profile peaked at min + 2A and warmed idle packs.

        Summer data (pack 27..33 C) produced ambient(15:00) = 35.0 C, i.e. above
        the battery's own temperature, so TemperatureProjector heated a pack
        sitting at 0 kW.
        """
        cfg = AmbientServiceConfig(
            diurnal_amplitude_c=4.0, diurnal_peak_hour=15.0, slot_minutes=15
        )
        svc = _service(cfg, min_temp_provider=lambda: 27.0)
        svc.refresh()

        horizon = [
            svc.predict_c(BASE + datetime.timedelta(minutes=15 * i))
            for i in range(33 * 4)
        ]
        assert max(horizon) <= 27.0 + 1e-6

    def test_idle_pack_cools_instead_of_warming(self):
        """End-to-end: the shared thermal model must not heat a 0 kW pack."""
        from battery_optimizer_lib.thermal_model import TemperatureProjector

        cfg = AmbientServiceConfig(
            diurnal_amplitude_c=4.0, diurnal_peak_hour=15.0, slot_minutes=15
        )
        svc = _service(cfg, min_temp_provider=lambda: 27.0)
        svc.refresh()

        projector = TemperatureProjector(learning_engine=None, ambient_provider=svc)
        temp = 33.0
        at = datetime.datetime(2026, 7, 27, 12, 0)
        for _ in range(12):  # three hours at 0 kW
            temp = projector.project(temp, at, 15.0, 0.0)
            at += datetime.timedelta(minutes=15)

        assert temp < 33.0

    def test_tomorrow_morning_is_colder_than_today_afternoon(self):
        """The concrete DEFECT 7 symptom: same value through night and noon."""
        cfg = AmbientServiceConfig(diurnal_amplitude_c=4.0, diurnal_peak_hour=15.0)
        svc = _service(cfg, min_temp_provider=lambda: 27.0)

        today_afternoon = svc.predict_c(datetime.datetime(2026, 7, 27, 16, 0))
        tomorrow_morning = svc.predict_c(datetime.datetime(2026, 7, 28, 6, 0))
        assert today_afternoon - tomorrow_morning > 5.0

    def test_no_source_returns_none(self):
        svc = _service()
        svc.refresh()
        assert svc.predict_c(BASE) is None
        assert svc.has_forecast is False

    def test_forecast_gap_falls_through_to_profile(self):
        """A forecast that does not reach the end of the horizon still varies."""
        cfg = AmbientServiceConfig(
            weather_entity="weather.home",
            diurnal_amplitude_c=4.0,
            diurnal_peak_hour=15.0,
        )
        svc = _service(
            cfg,
            call_service_func=lambda s, **k: {
                "forecast": _forecast_entries(BASE, REAL_TEMPS)
            },
            min_temp_provider=lambda: 27.0,
        )
        svc.refresh()

        beyond = svc.predict_c(BASE + datetime.timedelta(hours=30))
        assert beyond is not None
        assert beyond != svc.predict_c(BASE + datetime.timedelta(hours=42))


class TestCaching:
    def test_cache_is_reused_until_ttl(self):
        now = {"t": BASE}
        fetches = {"n": 0}

        def call_service(service, **kwargs):
            fetches["n"] += 1
            return {"forecast": _forecast_entries(BASE, REAL_TEMPS)}

        cfg = AmbientServiceConfig(weather_entity="weather.home", cache_minutes=60)
        svc = _service(
            cfg, call_service_func=call_service, get_datetime_func=lambda: now["t"]
        )
        svc.refresh()
        assert fetches["n"] == 1

        now["t"] = BASE + datetime.timedelta(minutes=30)
        svc.refresh()
        assert fetches["n"] == 1  # still fresh

        now["t"] = BASE + datetime.timedelta(minutes=90)
        svc.refresh()
        assert fetches["n"] == 2


class TestFailedFetchBackOff:
    """A failing weather entity must not be re-fetched on every refresh().

    ``_cache_timestamp`` was assigned only in the success branch, so while
    ``_fetch_weather_forecast`` returned {} the cache-age guard never engaged.
    ``refresh()`` is called by every full optimize, every adaptive and
    PV-shortfall recalculation and the 15-min ambient observation timer, and
    each one then re-issued a blocking ``weather/get_forecasts`` call_service on
    an AppDaemon callback thread. Both failure paths logged at DEBUG only, so
    the cost was invisible.
    """

    def test_empty_response_backs_off_for_the_failure_retry_interval(self):
        now = {"t": BASE}
        fetches = {"n": 0}

        def call_service(service, **kwargs):
            fetches["n"] += 1
            return {}

        cfg = AmbientServiceConfig(
            weather_entity="weather.home", cache_minutes=60, failure_retry_minutes=10
        )
        svc = _service(
            cfg, call_service_func=call_service, get_datetime_func=lambda: now["t"]
        )
        assert svc.refresh() is False
        assert fetches["n"] == 1

        for minutes in (1, 5, 9):
            now["t"] = BASE + datetime.timedelta(minutes=minutes)
            svc.refresh()
        assert fetches["n"] == 1, "a failing entity was retried on every refresh"

        now["t"] = BASE + datetime.timedelta(minutes=11)
        svc.refresh()
        assert fetches["n"] == 2

    def test_raising_service_backs_off_too(self):
        now = {"t": BASE}
        fetches = {"n": 0}

        def call_service(service, **kwargs):
            fetches["n"] += 1
            raise RuntimeError("entity not found")

        cfg = AmbientServiceConfig(
            weather_entity="weather.home", cache_minutes=60, failure_retry_minutes=10
        )
        svc = _service(
            cfg, call_service_func=call_service, get_datetime_func=lambda: now["t"]
        )
        svc.refresh()
        now["t"] = BASE + datetime.timedelta(minutes=5)
        svc.refresh()
        assert fetches["n"] == 1

    def test_first_failure_recovers_well_before_the_cache_interval(self):
        """A restart that races the HA weather integration must self-heal fast.

        The back-off used to key off ``_last_fetch_attempt`` alone, which is set
        unconditionally, so the FIRST-ever failure — typically one "entity not
        found" seconds after an AppDaemon restart, before the weather
        integration has loaded — suppressed every retry for a full
        ``cache_minutes`` (60). No caller passes force=True, so T_ambient(t) sat
        on the outdoor-sensor / diurnal fallback for an hour and degraded the
        DP-facing charge rates.
        """
        now = {"t": BASE}
        fetches = {"n": 0}

        def call_service(service, **kwargs):
            fetches["n"] += 1
            if fetches["n"] == 1:
                raise RuntimeError("Entity weather.home not found")
            return {"forecast": _forecast_entries(BASE, REAL_TEMPS)}

        cfg = AmbientServiceConfig(
            weather_entity="weather.home", cache_minutes=60, failure_retry_minutes=10
        )
        svc = _service(
            cfg, call_service_func=call_service, get_datetime_func=lambda: now["t"]
        )
        assert svc.refresh() is False
        assert svc.has_forecast is False

        # Immediately after the failure: still backing off.
        now["t"] = BASE + datetime.timedelta(minutes=2)
        assert svc.refresh() is False
        assert fetches["n"] == 1

        # Past failure_retry_minutes but far inside cache_minutes: retried,
        # succeeds, and the real forecast is used.
        now["t"] = BASE + datetime.timedelta(minutes=12)
        assert svc.refresh() is True
        assert fetches["n"] == 2
        assert svc.source == "forecast"
        assert svc.predict_c(BASE) == pytest.approx(REAL_TEMPS[0])

    def test_success_is_cached_for_the_full_cache_interval(self):
        """The short retry applies to FAILURES only, not to a good forecast."""
        now = {"t": BASE}
        fetches = {"n": 0}

        def call_service(service, **kwargs):
            fetches["n"] += 1
            return {"forecast": _forecast_entries(BASE, REAL_TEMPS)}

        cfg = AmbientServiceConfig(
            weather_entity="weather.home", cache_minutes=60, failure_retry_minutes=10
        )
        svc = _service(
            cfg, call_service_func=call_service, get_datetime_func=lambda: now["t"]
        )
        assert svc.refresh() is True

        for minutes in (11, 30, 59):
            now["t"] = BASE + datetime.timedelta(minutes=minutes)
            svc.refresh()
        assert fetches["n"] == 1, "a fresh forecast was re-fetched at the retry cadence"

        now["t"] = BASE + datetime.timedelta(minutes=61)
        svc.refresh()
        assert fetches["n"] == 2

    def test_force_still_bypasses_the_back_off(self):
        fetches = {"n": 0}

        def call_service(service, **kwargs):
            fetches["n"] += 1
            return {}

        cfg = AmbientServiceConfig(weather_entity="weather.home", cache_minutes=60)
        svc = _service(cfg, call_service_func=call_service)
        svc.refresh()
        svc.refresh(force=True)
        assert fetches["n"] == 2

    def test_outdoor_sensor_is_still_read_while_backing_off(self):
        """The back-off must not starve the cheap, non-blocking fallback."""
        now = {"t": BASE}
        reads = {"n": 0}

        def get_state(entity, **kwargs):
            reads["n"] += 1
            return "21.5"

        cfg = AmbientServiceConfig(
            weather_entity="weather.home",
            outdoor_temp_sensor="sensor.outdoor",
            cache_minutes=60,
        )
        svc = _service(
            cfg,
            call_service_func=lambda s, **k: {},
            get_state_func=get_state,
            get_datetime_func=lambda: now["t"],
        )
        svc.refresh()
        before = reads["n"]
        # Inside the failure retry window, so the forecast fetch is skipped.
        now["t"] = BASE + datetime.timedelta(minutes=5)
        svc.refresh()
        assert reads["n"] > before
        assert svc.predict_c(BASE) is not None


class TestForecastCallIsBounded:
    def test_call_service_passes_a_hass_timeout(self):
        """The call is synchronous on an AppDaemon callback thread."""
        calls = []

        def call_service(service, **kwargs):
            calls.append(kwargs)
            return {"forecast": _forecast_entries(BASE, REAL_TEMPS)}

        cfg = AmbientServiceConfig(weather_entity="weather.home")
        svc = _service(cfg, call_service_func=call_service)
        svc.refresh()

        assert calls
        assert calls[0]["hass_timeout"] == pytest.approx(
            cfg.forecast_timeout_seconds
        )
        assert cfg.forecast_timeout_seconds > 0

    def test_timeout_is_configurable(self):
        calls = []
        cfg = AmbientServiceConfig(
            weather_entity="weather.home", forecast_timeout_seconds=3.0
        )
        svc = _service(
            cfg,
            call_service_func=lambda s, **k: (calls.append(k), {})[1],
        )
        svc.refresh()
        assert calls[0]["hass_timeout"] == pytest.approx(3.0)


class TestFetchFailureLogging:
    """First failure and every subsequent state change warn; the rest are DEBUG."""

    @staticmethod
    def _capture():
        records = []

        def log(msg, level="INFO"):
            records.append((level, msg))

        return records, log

    @staticmethod
    def _warnings(records):
        return [m for lvl, m in records if lvl == "WARNING"]

    def test_first_failure_warns_once_then_stays_quiet(self):
        now = {"t": BASE}
        records, log = self._capture()
        cfg = AmbientServiceConfig(weather_entity="weather.home", cache_minutes=60)
        svc = _service(
            cfg,
            call_service_func=lambda s, **k: {},
            get_datetime_func=lambda: now["t"],
            log_func=log,
        )
        svc.refresh()
        assert len(self._warnings(records)) == 1
        assert "weather.home" in self._warnings(records)[0]

        for i in range(1, 6):
            now["t"] = BASE + datetime.timedelta(minutes=61 * i)
            svc.refresh()
        assert len(self._warnings(records)) == 1, "warned on every retry"

    def test_recovery_is_reported_and_re_arms_the_warning(self):
        now = {"t": BASE}
        state = {"ok": False}
        records, log = self._capture()

        def call_service(service, **kwargs):
            if state["ok"]:
                return {"forecast": _forecast_entries(BASE, REAL_TEMPS)}
            return {}

        cfg = AmbientServiceConfig(weather_entity="weather.home", cache_minutes=60)
        svc = _service(
            cfg,
            call_service_func=call_service,
            get_datetime_func=lambda: now["t"],
            log_func=log,
        )
        svc.refresh()
        assert len(self._warnings(records)) == 1

        state["ok"] = True
        now["t"] = BASE + datetime.timedelta(minutes=61)
        assert svc.refresh() is True
        assert any("recovered" in m for _, m in records)

        # Failing again is a NEW state change and must warn again.
        state["ok"] = False
        now["t"] = BASE + datetime.timedelta(minutes=122)
        svc.refresh()
        assert len(self._warnings(records)) == 2
