"""
Nord Pool price fetching service.

Handles fetching electricity prices from Nord Pool via:
- Built-in Home Assistant Nord Pool integration (service call / REST API)
- HACS custom component (sensor attributes)
"""

import datetime
import json
import traceback
from typing import Callable, Dict, List, Optional

from .models import PricePoint

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class NordPoolPriceService:
    """
    Service for fetching electricity prices from Nord Pool.

    Supports both the built-in Home Assistant Nord Pool integration (via service call)
    and the HACS custom component (via sensor attributes).
    """

    def __init__(
        self,
        nordpool_config_entry: str,
        nordpool_area: str,
        nordpool_sensor: str,
        ha_url: str,
        ha_token: str,
        tomorrow_prices_hour: int,
        slot_minutes: int,
        get_state_func: Callable,
        call_service_func: Callable,
        get_datetime_func: Callable,
        get_date_func: Callable,
        get_timezone_func: Callable,
        log_func: Callable,
    ):
        """
        Initialize the price service.

        Args:
            nordpool_config_entry: Nord Pool config entry ID for built-in integration
            nordpool_area: Area code (e.g., "LV")
            nordpool_sensor: Sensor entity ID for HACS component fallback
            ha_url: Home Assistant URL for REST API calls
            ha_token: Long-lived access token for HA API
            tomorrow_prices_hour: Hour when tomorrow's prices become available (local time)
            slot_minutes: Schedule slot size in minutes
            get_state_func: Callback to get entity state
            call_service_func: Callback to call HA services
            get_datetime_func: Callback to get current datetime
            get_date_func: Callback to get current date
            get_timezone_func: Callback to get local timezone
            log_func: Callback for logging
        """
        self.nordpool_config_entry = nordpool_config_entry
        self.nordpool_area = nordpool_area
        self.nordpool_sensor = nordpool_sensor
        self.ha_url = ha_url.rstrip("/") if ha_url else ""
        self.ha_token = ha_token
        self.tomorrow_prices_hour = tomorrow_prices_hour
        self.slot_minutes = slot_minutes

        self.get_state = get_state_func
        self.call_service = call_service_func
        self.datetime = get_datetime_func
        self.date = get_date_func
        self.get_timezone = get_timezone_func
        self.log = log_func

        # Cache
        self.cached_prices: List[PricePoint] = []
        self.cached_prices_date: Optional[datetime.date] = None

    def get_prices(self) -> List[PricePoint]:
        """
        Fetch prices from Nord Pool.
        Returns combined today + tomorrow prices when available.
        """
        self.log(f"get_prices called. Config: nordpool_config_entry='{self.nordpool_config_entry}', "
                 f"area='{self.nordpool_area}', ha_url='{self.ha_url[:30] if self.ha_url else 'NOT SET'}...'")

        prices = []
        today = self.date()
        tomorrow = today + datetime.timedelta(days=1)
        tz = self.get_timezone()

        # Try built-in HA Nord Pool integration first (uses service call)
        if self.nordpool_config_entry:
            prices = self._get_prices_via_service(today, tomorrow, tz)
            if prices:
                prices = self._normalize_prices(prices)
                self.cached_prices = prices
                self.cached_prices_date = today
                self.log(f"Fetched {len(prices)} price points via service call")
                return prices

        # Fall back to HACS custom component (uses sensor attributes)
        prices = self._get_prices_via_sensor(today, tomorrow, tz)
        if prices:
            prices = self._normalize_prices(prices)
            self.cached_prices = prices
            self.cached_prices_date = today
            self.log(f"Fetched {len(prices)} price points via sensor attributes")
            return prices

        # Validate cached prices are not stale before using
        if self.cached_prices and self.cached_prices_date:
            # Cache is valid if it was fetched today or yesterday (may contain tomorrow's prices)
            cache_age_days = (today - self.cached_prices_date).days
            if cache_age_days <= 1:
                # Additional check: ensure we have prices for today
                has_today_prices = any(p.time.date() == today for p in self.cached_prices)
                if has_today_prices:
                    self.log(f"Using cached prices (cached {cache_age_days} day(s) ago)", level="WARNING")
                    return self.cached_prices
                else:
                    self.log("Cached prices don't contain today's prices, clearing stale cache", level="WARNING")
                    self.cached_prices = []
                    self.cached_prices_date = None
            else:
                self.log(f"Cached prices are {cache_age_days} days old, clearing stale cache", level="WARNING")
                self.cached_prices = []
                self.cached_prices_date = None

        self.log("No price data available from any source", level="WARNING")
        return []

    def _normalize_prices(self, prices: List[PricePoint]) -> List[PricePoint]:
        """
        Normalize price points to the configured slot size.
        Aggregates higher-resolution data or expands lower-resolution data.
        """
        if not prices:
            return []

        sorted_prices = sorted(prices, key=lambda p: p.time)
        deltas = []
        for i in range(1, len(sorted_prices)):
            delta_min = (sorted_prices[i].time - sorted_prices[i - 1].time).total_seconds() / 60
            if delta_min > 0:
                deltas.append(delta_min)
        min_delta = min(deltas) if deltas else self.slot_minutes

        # Already at desired resolution
        if abs(min_delta - self.slot_minutes) < 0.1:
            return sorted_prices

        # Aggregate finer data into slot buckets
        if min_delta < self.slot_minutes:
            buckets: Dict[datetime.datetime, List[float]] = {}
            for p in sorted_prices:
                bucket = self._align_to_slot(p.time)
                buckets.setdefault(bucket, []).append(p.price)
            normalized = [
                PricePoint(time=dt, price=sum(vals) / len(vals))
                for dt, vals in buckets.items()
            ]
            return sorted(normalized, key=lambda p: p.time)

        # Expand coarser data into multiple slots
        factor = int(round(min_delta / self.slot_minutes))
        if factor <= 1 or abs(min_delta - factor * self.slot_minutes) > 0.1:
            # Fallback to bucketing if resolution is irregular
            buckets: Dict[datetime.datetime, List[float]] = {}
            for p in sorted_prices:
                bucket = self._align_to_slot(p.time)
                buckets.setdefault(bucket, []).append(p.price)
            normalized = [
                PricePoint(time=dt, price=sum(vals) / len(vals))
                for dt, vals in buckets.items()
            ]
            return sorted(normalized, key=lambda p: p.time)

        expanded: List[PricePoint] = []
        for p in sorted_prices:
            for i in range(factor):
                expanded.append(PricePoint(
                    time=p.time + datetime.timedelta(minutes=i * self.slot_minutes),
                    price=p.price
                ))
        return sorted(expanded, key=lambda p: p.time)

    def _align_to_slot(self, dt: datetime.datetime) -> datetime.datetime:
        """Floor datetime to the start of the current time slot."""
        tz = self.get_timezone()
        if dt.tzinfo is not None and tz is not None:
            dt = dt.astimezone(tz)
        elif tz is not None and dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        minutes = dt.hour * 60 + dt.minute
        slot_start = (minutes // self.slot_minutes) * self.slot_minutes
        return dt.replace(
            hour=int(slot_start // 60),
            minute=int(slot_start % 60),
            second=0,
            microsecond=0
        )

    def _get_prices_via_service(self, today, tomorrow, tz) -> List[PricePoint]:
        """
        Fetch prices using the built-in HA Nord Pool integration service.
        Uses nordpool.get_price_indices_for_date action.
        """
        prices = []

        if not self.nordpool_config_entry:
            self.log("nordpool_config_entry not configured, skipping service call", level="DEBUG")
            return prices

        self.log(f"Fetching prices via service for config_entry={self.nordpool_config_entry}, "
                 f"area={self.nordpool_area}")

        try:
            # Fetch today's prices
            self.log(f"Fetching today's prices ({today.isoformat()})...")
            today_data = self._call_nordpool_service(today.isoformat())
            self.log(f"Today data received: {today_data is not None}, type={type(today_data)}")
            if today_data:
                today_prices = self._parse_service_response(today_data, tz)
                self.log(f"Parsed {len(today_prices)} prices for today")
                prices.extend(today_prices)

            # Fetch tomorrow's prices (available after ~13:00 CET)
            current_hour = self.datetime().hour
            self.log(f"Current hour: {current_hour}, will fetch tomorrow: {current_hour >= self.tomorrow_prices_hour}")
            if current_hour >= self.tomorrow_prices_hour:
                try:
                    self.log(f"Fetching tomorrow's prices ({tomorrow.isoformat()})...")
                    tomorrow_data = self._call_nordpool_service(tomorrow.isoformat())
                    if tomorrow_data:
                        tomorrow_prices = self._parse_service_response(tomorrow_data, tz)
                        self.log(f"Parsed {len(tomorrow_prices)} prices for tomorrow")
                        prices.extend(tomorrow_prices)
                except Exception as e:
                    self.log(f"Tomorrow's prices not yet available: {e}", level="DEBUG")

        except Exception as e:
            self.log(f"Error fetching prices via service: {e}", level="WARNING")
            self.log(traceback.format_exc(), level="WARNING")

        self.log(f"Service method returned {len(prices)} total prices")
        return prices

    def get_prices_for_date(self, date_obj, tz) -> List[PricePoint]:
        """Fetch and normalize prices for a specific date via Nord Pool service."""
        if not self.nordpool_config_entry:
            return []

        try:
            data = self._call_nordpool_service(date_obj.isoformat())
            if not data:
                return []
            parsed = self._parse_service_response(data, tz)
            if not parsed:
                return []
            return self._normalize_prices(parsed)
        except Exception as e:
            self.log(f"Error fetching prices for {date_obj.isoformat()}: {e}", level="DEBUG")
            return []

    def _call_nordpool_service(self, date_str: str) -> Optional[Dict]:
        """
        Call the nordpool.get_price_indices_for_date service.
        Tries REST API approach first, falls back to AppDaemon call_service.
        """
        # Try REST API approach (more reliable for response actions)
        result = self._call_nordpool_rest_api(date_str)
        if result:
            return result

        # Fallback to AppDaemon call_service (may not return response data)
        try:
            result = self.call_service(
                "nordpool/get_price_indices_for_date",
                config_entry=self.nordpool_config_entry,
                date=date_str,
                areas=self.nordpool_area,
                resolution=self.slot_minutes,
                return_result=True
            )
            self.log(f"Nord Pool service response for {date_str}: {type(result)}")
            return result
        except Exception as e:
            self.log(f"Service call failed for {date_str}: {e}", level="WARNING")
            return None

    def _call_nordpool_rest_api(self, date_str: str) -> Optional[Dict]:
        """
        Call Nord Pool service via Home Assistant REST API.
        This is more reliable for response-returning actions.
        """
        if not REQUESTS_AVAILABLE:
            self.log("requests module not available for REST API", level="WARNING")
            return None

        try:
            if not self.ha_url or not self.ha_token:
                self.log("ha_url and ha_token not configured - needed for Nord Pool service calls",
                        level="WARNING")
                return None

            # Add return_response for HA 2023.7+ response-returning actions
            url = f"{self.ha_url}/api/services/nordpool/get_price_indices_for_date?return_response"
            headers = {
                "Authorization": f"Bearer {self.ha_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "config_entry": self.nordpool_config_entry,
                "date": date_str,
                "areas": self.nordpool_area,
                "resolution": self.slot_minutes,
            }

            self.log(f"Calling Nord Pool API: POST {url}")
            self.log(f"Payload: {json.dumps(payload)}")
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            self.log(f"REST API response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                self.log(f"REST API response for {date_str}: type={type(data)}, "
                        f"keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}")

                # Response is wrapped in service_response for HA 2023.7+ actions
                if isinstance(data, dict):
                    if "service_response" in data:
                        self.log("Found service_response wrapper")
                        return data["service_response"]
                    if "response" in data:
                        return data["response"]
                    if self.nordpool_area in data or self.nordpool_area.lower() in data:
                        return data
                    self.log(f"Response keys: {list(data.keys())[:10]}", level="DEBUG")
                    return data
                elif isinstance(data, list) and data:
                    self.log(f"Response is list with {len(data)} items", level="DEBUG")
                    return {self.nordpool_area: data}
                return data
            else:
                self.log(f"REST API returned status {response.status_code}: {response.text[:200]}",
                        level="WARNING")
                return None

        except requests.exceptions.RequestException as e:
            self.log(f"REST API request failed: {e}", level="WARNING")
            return None
        except Exception as e:
            self.log(f"REST API call failed: {e}", level="WARNING")
            self.log(traceback.format_exc(), level="DEBUG")
            return None

    def _parse_service_response(self, data: Dict, tz) -> List[PricePoint]:
        """
        Parse the response from nordpool.get_price_indices_for_date service.
        Response format: {area: [{start, end, price}, ...]}
        Prices are in EUR/MWh, need to convert to EUR/kWh.
        """
        prices = []

        if not data:
            self.log("No data to parse", level="DEBUG")
            return prices

        self.log(f"Parsing response data type: {type(data)}, "
                f"keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

        # Handle different response formats
        area_prices = []

        if isinstance(data, dict):
            # Try direct area lookup
            area_prices = data.get(self.nordpool_area, [])
            if not area_prices:
                area_prices = data.get(self.nordpool_area.lower(), [])

            # Try nested in 'prices' key
            if not area_prices and "prices" in data:
                area_prices = data["prices"]

            # Try the first key if it contains a list
            if not area_prices:
                for key, value in data.items():
                    if isinstance(value, list) and value:
                        self.log(f"Found price list under key '{key}' with {len(value)} entries")
                        area_prices = value
                        break

        elif isinstance(data, list):
            area_prices = data

        self.log(f"Parsing {len(area_prices)} price entries for area {self.nordpool_area}")

        for entry in area_prices:
            if not isinstance(entry, dict):
                continue

            start_str = entry.get("start", "")
            price_mwh = entry.get("price")

            if price_mwh is None:
                continue

            # Parse start time (UTC format: 2026-01-21T00:00:00+00:00)
            try:
                if isinstance(start_str, str):
                    start_str = start_str.replace("Z", "+00:00")
                    start_dt = datetime.datetime.fromisoformat(start_str)
                else:
                    continue

                # Convert to local timezone
                if tz and start_dt.tzinfo:
                    start_dt = start_dt.astimezone(tz)
                elif tz:
                    start_dt = start_dt.replace(tzinfo=tz)

                # Convert EUR/MWh to EUR/kWh
                price_kwh = float(price_mwh) / 1000.0
                prices.append(PricePoint(time=start_dt, price=price_kwh))

            except (ValueError, TypeError) as e:
                self.log(f"Error parsing price entry {entry}: {e}", level="DEBUG")
                continue

        return prices

    def _get_prices_via_sensor(self, today, tomorrow, tz) -> List[PricePoint]:
        """
        Fetch prices from HACS custom component sensor attributes.
        """
        try:
            state = self.get_state(self.nordpool_sensor, attribute="all")
            if not state:
                return []

            prices = []
            attrs = state.get("attributes", {})

            # HACS custom component uses raw_today/raw_tomorrow or today/tomorrow
            today_prices = attrs.get("raw_today") or attrs.get("today") or []
            tomorrow_prices = attrs.get("raw_tomorrow") or attrs.get("tomorrow") or []

            # Process today's prices
            if today_prices:
                prices.extend(self._parse_sensor_prices(today_prices, today, tz))

            # Process tomorrow's prices
            if tomorrow_prices:
                prices.extend(self._parse_sensor_prices(tomorrow_prices, tomorrow, tz))

            return prices

        except Exception as e:
            self.log(f"Error fetching prices via sensor: {e}", level="WARNING")
            return []

    def _parse_sensor_prices(self, price_data: List, date_for_simple: datetime.date,
                            tz) -> List[PricePoint]:
        """Parse price data from sensor attributes."""
        prices = []

        if not isinstance(price_data, list) or not price_data:
            return prices

        if isinstance(price_data[0], dict):
            # raw format: list of {start, end, value}
            for entry in price_data:
                price = entry.get("value")
                start = entry.get("start")
                if price is not None and start:
                    try:
                        if isinstance(start, str):
                            dt = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
                        else:
                            dt = start
                        if tz and dt.tzinfo:
                            dt = dt.astimezone(tz)
                        elif tz:
                            dt = dt.replace(tzinfo=tz)
                        prices.append(PricePoint(time=dt, price=float(price)))
                    except (ValueError, TypeError):
                        pass
        else:
            # Simple list of prices (infer step size)
            step_minutes = 60
            if len(price_data) > 0 and 1440 % len(price_data) == 0:
                step_minutes = int(1440 / len(price_data))
            for idx, price in enumerate(price_data):
                if price is not None:
                    minutes = idx * step_minutes
                    dt = datetime.datetime.combine(
                        date_for_simple,
                        datetime.time(minutes // 60, minutes % 60)
                    )
                    if tz:
                        dt = dt.replace(tzinfo=tz)
                    prices.append(PricePoint(time=dt, price=float(price)))

        return prices
