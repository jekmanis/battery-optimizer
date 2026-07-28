"""
Ambient temperature service.

Supplies ``T_ambient(t)`` — a temperature that varies across the whole planning
horizon — with automatic fallback:

1. Home Assistant **weather forecast** entity (hourly forecast, real values for
   tomorrow morning as well as tonight)
2. An **outdoor temperature sensor** reading, extended over the horizon with a
   synthetic diurnal profile
3. The learning engine's rolling **minimum battery temperature**, used as an
   upper bound (the daily *maximum*) of a synthetic diurnal profile

Why this exists
---------------

``BatteryLearningEngine.get_estimated_ambient_temp`` returns
``min(recent battery temperatures)``. In summer the pack never drops below
~26-27 C, so "ambient" ends up being roughly the current battery temperature and
the exponential relaxation produces ~zero change. Worse, it is a single scalar
for a 33 h horizon, so tomorrow morning (really ~27 C) is modelled with today's
33 C. Both effects are fixed here: the value becomes time-dependent, and the
fallback anchors a diurnal profile to it.

**Which end of the profile the rolling minimum anchors is a sign question, and
getting it wrong warms an idle pack.** The battery is self-heated, so
``T_bat(t) >= T_ambient(t)`` at all times; therefore ``min(T_bat)`` over the
window is an *upper bound* on ambient, never its trough. Anchoring it as the
daily minimum and adding the amplitude on top produced a profile peaking at
``min + 2A`` (default +8 C) — e.g. a pack observed at 27..33 C got an "ambient"
of 35 C at 15:00, and ``TemperatureProjector`` then heated an idle pack from
33.0 to 34.6 C over three hours at 0 kW. The rolling minimum is therefore used
as the profile's daily MAXIMUM: the fallback ambient spans
``[min - 2A, min]`` and can never exceed a temperature the pack has actually
been seen at.

The diurnal profile is ``T(h) = mean + A*cos(2*pi*(h - peak_hour)/24)`` — maximum
at ``peak_hour``, minimum 12 h later.
"""

import datetime
import math
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Optional, TYPE_CHECKING

from .timezone_utils import align_to_slot, canonical_slot_key, ensure_local_tz, instant_key

if TYPE_CHECKING:
    from .config import BatteryOptimizerConfig


@dataclass
class AmbientServiceConfig:
    """Configuration for :class:`AmbientTemperatureService`."""

    # HA weather entity providing an hourly forecast (e.g. weather.forecast_home)
    weather_entity: str = ""
    # Plain outdoor / room temperature sensor (preferred if the battery is indoors)
    outdoor_temp_sensor: str = ""

    # Synthetic diurnal profile used when no hourly forecast is available.
    diurnal_amplitude_c: float = 4.0   # half of the peak-to-peak swing
    diurnal_peak_hour: float = 15.0    # local hour of the daily maximum

    cache_minutes: int = 60
    slot_minutes: int = 15

    # Used when literally nothing is known.
    fallback_ambient_c: float = 10.0

    @property
    def configured(self) -> bool:
        return bool(self.weather_entity or self.outdoor_temp_sensor)

    @classmethod
    def from_main_config(cls, cfg: "BatteryOptimizerConfig") -> "AmbientServiceConfig":
        return cls(
            weather_entity=cfg.ambient_weather_entity,
            outdoor_temp_sensor=cfg.outdoor_temp_sensor,
            diurnal_amplitude_c=cfg.ambient_diurnal_amplitude_c,
            diurnal_peak_hour=cfg.ambient_diurnal_peak_hour,
            cache_minutes=cfg.ambient_forecast_cache_minutes,
            slot_minutes=cfg.slot_minutes,
        )


class AmbientTemperatureService:
    """Time-varying ambient temperature with a documented degradation chain."""

    def __init__(
        self,
        config: AmbientServiceConfig,
        get_state_func: Optional[Callable] = None,
        call_service_func: Optional[Callable] = None,
        get_datetime_func: Optional[Callable] = None,
        get_timezone_func: Optional[Callable] = None,
        log_func: Optional[Callable] = None,
        min_temp_provider: Optional[Callable[[], Optional[float]]] = None,
    ):
        self._config = config
        self.get_state = get_state_func
        self.call_service = call_service_func
        self.datetime = get_datetime_func or datetime.datetime.now
        self.get_timezone = get_timezone_func or (lambda: None)
        self.log = log_func or (lambda *a, **k: None)
        # Callable returning the learning engine's rolling minimum battery
        # temperature (or None when it has no observations yet).
        self.min_temp_provider = min_temp_provider

        # Hourly forecast cache: canonical slot key -> ambient C
        self._cache: Dict[datetime.datetime, float] = {}
        self._cache_timestamp: Optional[datetime.datetime] = None

        # Last outdoor sensor reading and when it was taken.
        self._sensor_temp: Optional[float] = None
        self._sensor_time: Optional[datetime.datetime] = None

        self._source: str = "none"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_forecast(self) -> bool:
        """Whether a real hourly forecast is cached."""
        return len(self._cache) > 0

    @property
    def source(self) -> str:
        """Which source last produced data: forecast / sensor / min_window / none."""
        return self._source

    def refresh(self, force: bool = False) -> bool:
        """Refresh the forecast cache / sensor reading if stale.

        Returns True when new data was obtained.
        """
        now = self.datetime()

        if not force and self._cache_timestamp is not None:
            age_minutes = (now - self._cache_timestamp).total_seconds() / 60.0
            if age_minutes < self._config.cache_minutes:
                self._read_outdoor_sensor(now)
                return False

        updated = False

        if self._config.weather_entity:
            try:
                data = self._fetch_weather_forecast()
                if data:
                    self._cache = data
                    self._cache_timestamp = now
                    self._source = "forecast"
                    self.log(
                        f"Ambient forecast updated from {self._config.weather_entity}: "
                        f"{len(data)} slots, {min(data.values()):.1f}..{max(data.values()):.1f}C"
                    )
                    updated = True
            except Exception as e:  # pragma: no cover - defensive
                self.log(f"Ambient weather fetch failed: {e}", level="WARNING")
                self.log(traceback.format_exc(), level="DEBUG")

        if self._read_outdoor_sensor(now):
            updated = True

        if not updated and not self._cache and self._sensor_temp is None:
            self._source = "min_window" if self.min_temp_provider else "none"

        return updated

    def predict_c(self, dt: Optional[datetime.datetime] = None) -> Optional[float]:
        """Ambient temperature (C) for the slot containing ``dt``.

        Returns None when no source is available at all — callers then keep
        their own fallback (e.g. the learning engine's estimate).
        """
        if dt is None:
            dt = self.datetime()

        if self._cache:
            value = self._cache.get(self._slot_key(dt))
            if value is not None:
                return value
            # Forecast exists but does not cover this slot -> diurnal profile.

        mean = self._profile_mean()
        if mean is None:
            return None
        return self._diurnal(dt, mean)

    # ------------------------------------------------------------------
    # Diurnal profile
    # ------------------------------------------------------------------

    def _profile_mean(self) -> Optional[float]:
        """Daily mean of the synthetic profile, or None if unknown."""
        cfg = self._config
        amplitude = max(0.0, cfg.diurnal_amplitude_c)

        if self._sensor_temp is not None and self._sensor_time is not None:
            # Solve so the profile passes through the observed reading now.
            phase = self._phase(self._sensor_time)
            self._source = "sensor"
            return self._sensor_temp - amplitude * math.cos(phase)

        if self.min_temp_provider is not None:
            try:
                min_temp = self.min_temp_provider()
            except Exception:  # pragma: no cover - defensive
                min_temp = None
            if min_temp is not None:
                # The pack never sits below ambient, so its rolling minimum is
                # a CEILING on ambient, not the trough of a diurnal swing.
                # Anchor it as the profile's daily MAXIMUM: mean = min - A,
                # giving a profile that spans [min - 2A, min]. Adding the
                # amplitude instead put the peak at min + 2A, above the pack's
                # own temperature, and warmed idle batteries.
                self._source = "min_window"
                return float(min_temp) - amplitude

        return None

    def _phase(self, dt: datetime.datetime) -> float:
        cfg = self._config
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        return 2.0 * math.pi * (hour - cfg.diurnal_peak_hour) / 24.0

    def _diurnal(self, dt: datetime.datetime, mean: float) -> float:
        amplitude = max(0.0, self._config.diurnal_amplitude_c)
        return mean + amplitude * math.cos(self._phase(dt))

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def _read_outdoor_sensor(self, now: datetime.datetime) -> bool:
        """Read the configured outdoor temperature sensor. Returns True on success."""
        if not self._config.outdoor_temp_sensor or self.get_state is None:
            return False
        try:
            raw = self.get_state(self._config.outdoor_temp_sensor)
        except Exception:  # pragma: no cover - defensive
            return False
        if raw is None:
            return False
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return False
        if not -60.0 <= value <= 70.0:
            return False
        self._sensor_temp = value
        self._sensor_time = now
        return True

    def _fetch_weather_forecast(self) -> Dict[datetime.datetime, float]:
        """Fetch the hourly forecast from a HA weather entity.

        Two paths, because the HA API changed: the modern
        ``weather.get_forecasts`` service and the legacy ``forecast`` attribute.
        Both are wrapped defensively — AppDaemon's ``call_service`` return value
        differs between versions and may raise.
        """
        entries = None

        if self.call_service is not None:
            try:
                response = self.call_service(
                    "weather/get_forecasts",
                    entity_id=self._config.weather_entity,
                    type="hourly",
                    return_result=True,
                )
                entries = self._extract_forecast_entries(response)
            except Exception as e:
                self.log(f"weather/get_forecasts unavailable: {e}", level="DEBUG")
                self.log(traceback.format_exc(), level="DEBUG")

        if not entries and self.get_state is not None:
            try:
                legacy = self.get_state(
                    self._config.weather_entity, attribute="forecast"
                )
                entries = self._extract_forecast_entries(legacy)
            except Exception as e:  # pragma: no cover - defensive
                self.log(f"Legacy weather forecast attribute failed: {e}", level="DEBUG")

        if not entries:
            return {}

        tz = self.get_timezone()
        result: Dict[datetime.datetime, float] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            temp = entry.get("temperature")
            when = entry.get("datetime") or entry.get("date_time")
            if temp is None or when is None:
                continue
            try:
                temp = float(temp)
            except (TypeError, ValueError):
                continue
            dt = self._parse_datetime(when)
            if dt is None:
                continue
            result.update(self._expand_period_to_slots(dt, 60, temp, tz))

        return result

    @staticmethod
    def _extract_forecast_entries(response):
        """Normalise the many shapes a HA forecast response can take."""
        if response is None:
            return None
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            if "forecast" in response and isinstance(response["forecast"], list):
                return response["forecast"]
            # {entity_id: {"forecast": [...]}}
            for value in response.values():
                if isinstance(value, dict) and isinstance(value.get("forecast"), list):
                    return value["forecast"]
                if isinstance(value, list):
                    return value
        return None

    @staticmethod
    def _parse_datetime(value) -> Optional[datetime.datetime]:
        if isinstance(value, datetime.datetime):
            return value
        if not isinstance(value, str):
            return None
        try:
            return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Slot helpers (identical semantics to PvForecastService)
    # ------------------------------------------------------------------

    def _slot_key(self, dt: datetime.datetime) -> datetime.datetime:
        tz = self.get_timezone()
        return canonical_slot_key(align_to_slot(dt, self._config.slot_minutes, tz))

    def _expand_period_to_slots(
        self,
        period_start: datetime.datetime,
        period_minutes: int,
        value: float,
        tz: Optional[datetime.tzinfo],
    ) -> Dict[datetime.datetime, float]:
        slot_min = self._config.slot_minutes
        result: Dict[datetime.datetime, float] = {}

        dt = ensure_local_tz(period_start, tz)
        if tz is not None and dt.tzinfo is not None:
            dt = dt.astimezone(tz)
        slot_dt = align_to_slot(dt, slot_min, tz)
        n_slots = max(1, period_minutes // slot_min)

        for i in range(n_slots):
            if slot_dt.tzinfo is not None:
                expanded = (
                    instant_key(slot_dt) + datetime.timedelta(minutes=i * slot_min)
                ).astimezone(slot_dt.tzinfo)
            else:
                expanded = slot_dt + datetime.timedelta(minutes=i * slot_min)
            result[canonical_slot_key(expanded)] = value

        return result
