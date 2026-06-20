"""
PV forecast service for fetching solar production forecasts.

Supports multiple forecast sources with automatic fallback:
1. Solcast (HACS integration) — reads detailedForecast sensor attributes
2. Forecast.Solar (built-in) — direct REST API call
3. Falls back to empty (caller uses PvProfile statistical history)
"""

import datetime
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Optional, TYPE_CHECKING

from .timezone_utils import align_to_slot, ensure_local_tz

if TYPE_CHECKING:
    from .config import BatteryOptimizerConfig

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class PvForecastServiceConfig:
    """Configuration for PvForecastService."""

    # Solcast HACS integration entities
    solcast_today_entity: str = ""  # e.g. sensor.solcast_pv_forecast_forecast_today
    solcast_tomorrow_entity: str = ""  # e.g. sensor.solcast_pv_forecast_forecast_tomorrow
    solcast_estimate_field: str = "pv_estimate"  # pv_estimate, pv_estimate10, pv_estimate90

    # Forecast.Solar direct API parameters
    forecast_solar_lat: float = 0.0
    forecast_solar_lon: float = 0.0
    forecast_solar_declination: int = 0  # panel tilt degrees
    forecast_solar_azimuth: int = 0  # 0=north, 90=east, 180=south, 270=west
    forecast_solar_kwp: float = 0.0  # peak kW capacity
    forecast_solar_api_key: str = ""  # optional paid API key

    # Cache settings
    pv_forecast_cache_minutes: int = 60

    # Slot resolution
    slot_minutes: int = 15

    @property
    def solcast_configured(self) -> bool:
        return bool(self.solcast_today_entity)

    @property
    def forecast_solar_configured(self) -> bool:
        return self.forecast_solar_kwp > 0

    @classmethod
    def from_main_config(cls, cfg: "BatteryOptimizerConfig") -> "PvForecastServiceConfig":
        """Create from the central BatteryOptimizerConfig."""
        return cls(
            solcast_today_entity=cfg.solcast_today_entity,
            solcast_tomorrow_entity=cfg.solcast_tomorrow_entity,
            solcast_estimate_field=cfg.solcast_estimate_field,
            forecast_solar_lat=cfg.forecast_solar_lat,
            forecast_solar_lon=cfg.forecast_solar_lon,
            forecast_solar_declination=cfg.forecast_solar_declination,
            forecast_solar_azimuth=cfg.forecast_solar_azimuth,
            forecast_solar_kwp=cfg.forecast_solar_kwp,
            forecast_solar_api_key=cfg.forecast_solar_api_key,
            pv_forecast_cache_minutes=cfg.pv_forecast_cache_minutes,
            slot_minutes=cfg.slot_minutes,
        )


class PvForecastService:
    """
    Service for fetching PV production forecasts from external providers.

    Tries Solcast first (via HA sensor attributes), then Forecast.Solar
    (via direct REST API). Caches results and provides per-slot kW lookup.
    """

    def __init__(
        self,
        config: PvForecastServiceConfig,
        get_state_func: Callable,
        get_datetime_func: Callable,
        get_timezone_func: Callable,
        log_func: Callable,
    ):
        self._config = config
        self.get_state = get_state_func
        self.datetime = get_datetime_func
        self.get_timezone = get_timezone_func
        self.log = log_func

        # Cache: slot-aligned naive local datetime -> kW
        self._cache: Dict[datetime.datetime, float] = {}
        self._cache_timestamp: Optional[datetime.datetime] = None
        self._last_source: str = ""  # "solcast", "forecast_solar", or ""

    @property
    def has_forecast(self) -> bool:
        """Whether the cache contains any forecast data."""
        return len(self._cache) > 0

    @property
    def last_source(self) -> str:
        """Name of the last successful forecast source."""
        return self._last_source

    def _slot_key(self, dt: datetime.datetime) -> datetime.datetime:
        """Convert a datetime to a naive local-time cache key."""
        tz = self.get_timezone()
        slot_dt = align_to_slot(dt, self._config.slot_minutes, tz)
        return slot_dt.replace(tzinfo=None)

    def has_slot(self, dt: datetime.datetime) -> bool:
        """Whether the cache contains data for this specific slot."""
        if not self._cache:
            return False
        return self._slot_key(dt) in self._cache

    def predict_kw(self, dt: datetime.datetime) -> float:
        """
        Look up forecasted PV production (kW) for the given time slot.

        Returns 0.0 if no forecast data is available for the slot.
        This method only reads the cache — call refresh() to update it.
        """
        if not self._cache:
            return 0.0

        val = self._cache.get(self._slot_key(dt))
        return val if val is not None else 0.0

    def refresh(self) -> bool:
        """
        Fetch fresh forecast data if cache is stale.

        Returns True if cache was updated, False if still using existing cache.
        """
        now = self.datetime()

        # Check cache freshness
        if self._cache_timestamp is not None:
            age_minutes = (now - self._cache_timestamp).total_seconds() / 60.0
            if age_minutes < self._config.pv_forecast_cache_minutes:
                return False  # Cache is fresh

        if not self._config.solcast_configured and not self._config.forecast_solar_configured:
            return False

        # Try Solcast first
        if self._config.solcast_configured:
            try:
                data = self._fetch_solcast()
                if data:
                    self._cache = data
                    self._cache_timestamp = now
                    self._last_source = "solcast"
                    self.log(
                        f"PV forecast updated from Solcast: {len(data)} slots, "
                        f"peak {max(data.values()):.2f} kW"
                    )
                    return True
            except Exception as e:
                self.log(f"Solcast fetch failed: {e}", level="WARNING")
                self.log(traceback.format_exc(), level="DEBUG")

        # Try Forecast.Solar
        if self._config.forecast_solar_configured and REQUESTS_AVAILABLE:
            try:
                data = self._fetch_forecast_solar()
                if data:
                    self._cache = data
                    self._cache_timestamp = now
                    self._last_source = "forecast_solar"
                    self.log(
                        f"PV forecast updated from Forecast.Solar: {len(data)} slots, "
                        f"peak {max(data.values()):.2f} kW"
                    )
                    return True
            except Exception as e:
                self.log(f"Forecast.Solar fetch failed: {e}", level="WARNING")
                self.log(traceback.format_exc(), level="DEBUG")

        # Both sources failed — check if stale cache is still usable
        if self._cache_timestamp is not None:
            stale_limit = self._config.pv_forecast_cache_minutes * 3
            age_minutes = (now - self._cache_timestamp).total_seconds() / 60.0
            if age_minutes < stale_limit:
                self.log(
                    f"PV forecast sources unavailable, using stale cache "
                    f"({age_minutes:.0f} min old)",
                    level="WARNING",
                )
                return False
            else:
                self.log(
                    f"PV forecast cache too old ({age_minutes:.0f} min), clearing",
                    level="WARNING",
                )
                self._cache.clear()
                self._cache_timestamp = None
                self._last_source = ""

        return False

    # ------------------------------------------------------------------
    # Solcast
    # ------------------------------------------------------------------

    def _fetch_solcast(self) -> Dict[datetime.datetime, float]:
        """Fetch forecast from Solcast HACS integration sensor attributes."""
        result: Dict[datetime.datetime, float] = {}
        tz = self.get_timezone()
        field = self._config.solcast_estimate_field

        for entity in [self._config.solcast_today_entity,
                       self._config.solcast_tomorrow_entity]:
            if not entity:
                continue

            state = self.get_state(entity, attribute="all")
            if not state or not isinstance(state, dict):
                continue

            attrs = state.get("attributes", {})
            detailed = attrs.get("detailedForecast", [])
            if not detailed:
                # Try alternate attribute name
                detailed = attrs.get("detailed_forecast", [])

            if not isinstance(detailed, list):
                self.log(
                    f"Solcast {entity}: detailedForecast is not a list "
                    f"(type={type(detailed).__name__})",
                    level="WARNING",
                )
                continue

            for entry in detailed:
                if not isinstance(entry, dict):
                    continue
                period_start = entry.get("period_start")
                kw_value = entry.get(field)
                if period_start is None or kw_value is None:
                    continue

                try:
                    kw_value = float(kw_value)
                except (ValueError, TypeError):
                    continue

                if kw_value < 0:
                    kw_value = 0.0

                # Parse ISO datetime
                dt = self._parse_iso_datetime(period_start)
                if dt is None:
                    continue

                # Expand 30-min Solcast period to slot-aligned entries
                slots = self._expand_period_to_slots(dt, 30, kw_value, tz)
                result.update(slots)

        return result

    def _parse_iso_datetime(self, iso_str) -> Optional[datetime.datetime]:
        """Parse an ISO 8601 datetime string."""
        if isinstance(iso_str, datetime.datetime):
            return iso_str
        if not isinstance(iso_str, str):
            return None
        try:
            return datetime.datetime.fromisoformat(iso_str)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Forecast.Solar
    # ------------------------------------------------------------------

    def _fetch_forecast_solar(self) -> Dict[datetime.datetime, float]:
        """Fetch forecast from Forecast.Solar REST API."""
        cfg = self._config
        lat = cfg.forecast_solar_lat
        lon = cfg.forecast_solar_lon
        dec = cfg.forecast_solar_declination
        az = cfg.forecast_solar_azimuth
        kwp = cfg.forecast_solar_kwp

        if cfg.forecast_solar_api_key:
            url = (
                f"https://api.forecast.solar/{cfg.forecast_solar_api_key}"
                f"/estimate/{lat}/{lon}/{dec}/{az}/{kwp}"
            )
        else:
            url = f"https://api.forecast.solar/estimate/{lat}/{lon}/{dec}/{az}/{kwp}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        result_data = data.get("result", {})
        wh_period = result_data.get("watt_hours_period", {})

        if not wh_period:
            self.log("Forecast.Solar: empty watt_hours_period", level="WARNING")
            return {}

        tz = self.get_timezone()
        result: Dict[datetime.datetime, float] = {}

        # Sort timestamps to detect period duration
        timestamps = sorted(wh_period.keys())
        for i, ts_str in enumerate(timestamps):
            try:
                wh = float(wh_period[ts_str])
            except (ValueError, TypeError):
                continue

            # Parse timestamp (Forecast.Solar uses "YYYY-MM-DD HH:MM:SS" in local tz)
            dt = self._parse_forecast_solar_timestamp(ts_str, tz)
            if dt is None:
                continue

            # Detect period duration from gap to next timestamp
            period_minutes = 60  # default assumption
            if i + 1 < len(timestamps):
                next_dt = self._parse_forecast_solar_timestamp(timestamps[i + 1], tz)
                if next_dt is not None:
                    gap = (next_dt - dt).total_seconds() / 60.0
                    if 0 < gap <= 120:
                        period_minutes = int(gap)

            # Convert Wh to average kW: kW = Wh / (period_hours * 1000)
            period_hours = period_minutes / 60.0
            kw = (wh / 1000.0) / period_hours if period_hours > 0 else 0.0
            if kw < 0:
                kw = 0.0

            # Expand to slot-aligned entries
            slots = self._expand_period_to_slots(dt, period_minutes, kw, tz)
            result.update(slots)

        return result

    def _parse_forecast_solar_timestamp(
        self, ts_str: str, tz: Optional[datetime.tzinfo]
    ) -> Optional[datetime.datetime]:
        """Parse Forecast.Solar timestamp (local time without offset)."""
        try:
            dt = datetime.datetime.fromisoformat(ts_str)
            # Forecast.Solar returns local time based on lat/lon — add tz if naive
            if dt.tzinfo is None and tz is not None:
                dt = dt.replace(tzinfo=tz)
            return dt
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Slot expansion helpers
    # ------------------------------------------------------------------

    def _expand_period_to_slots(
        self,
        period_start: datetime.datetime,
        period_minutes: int,
        kw_value: float,
        tz: Optional[datetime.tzinfo],
    ) -> Dict[datetime.datetime, float]:
        """
        Expand a forecast period into slot-aligned entries.

        If the source period is coarser than our slot resolution (e.g., 30-min
        Solcast data with 15-min slots), the same kW value is duplicated to
        each sub-slot within the period.

        Returns dict of naive local datetime -> kW.
        """
        slot_min = self._config.slot_minutes
        result: Dict[datetime.datetime, float] = {}

        # Ensure local tz
        dt = ensure_local_tz(period_start, tz)
        # Align to slot boundary
        slot_dt = align_to_slot(dt, slot_min, tz)

        # How many of our slots fit in this source period?
        n_slots = max(1, period_minutes // slot_min)

        for i in range(n_slots):
            key = (slot_dt + datetime.timedelta(minutes=i * slot_min)).replace(tzinfo=None)
            result[key] = kw_value

        return result
