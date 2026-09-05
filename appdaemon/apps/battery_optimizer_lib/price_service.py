"""
Nord Pool price fetching service.

Handles fetching electricity prices from Nord Pool via:
- Built-in Home Assistant Nord Pool integration (service call / REST API)
- HACS custom component (sensor attributes)
"""

import datetime
import json
import traceback
from typing import Callable, Dict, List, Optional, Tuple

from .models import PricePoint
from .timezone_utils import (
    ensure_local_tz,
    instant_key,
    is_aware,
    normalize_tz_pair,
)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# How often a corrupt price record may be reported. See
# `NordPoolPriceService._warn_malformed_record`.
MALFORMED_WARNING_INTERVAL_S = 3600


class _MalformedEnd:
    """Sentinel: the source PUBLISHED an ``end`` and it cannot be used.

    `_parse_interval_end` used to answer ``None`` both for a record that
    carries no ``end`` and for one whose ``end`` is ``"not-a-timestamp"``.
    Those are opposite statements. An absent end is a normal case with a
    documented answer - one `slot_minutes` slot - and collapsing the second
    into the first handed that slot of coverage to a record whose own width
    field is garbage. A cheap current record with an unreadable end therefore
    reported `has_current=True` and was planned as CHARGE carrying
    `PRICE_SOURCE_MARKET`.

    Three states, so the caller can tell them apart: a datetime, ``None`` for
    absent, this for published-and-unusable.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "<malformed interval end>"


MALFORMED_END = _MalformedEnd()

# The longest span a SINGLE published record may declare.
#
# Nord Pool publishes 15- or 60-minute intervals and a day-ahead block never
# exceeds a day, so anything longer is a corrupt timestamp rather than a market
# that changed its mind about resolution.
#
# It bounds the WORK a record can ask for as much as its plausibility.
# `_normalize_prices` allocates a bucket for every slot inside a declared
# interval, and the only guard on that used to be `modeled_horizon`'s 168-hour
# budget -- which runs on the RESULT, long after every bucket has been built.
# One record declaring a year produced 35,040 quarter-hour buckets, silently,
# inside the app lock and on a callback that has to finish in milliseconds.
#
# A 25-hour local day is not a counterexample: the day is published as hourly or
# quarter-hourly records, never as one record spanning it.
MAX_RECORD_SPAN_HOURS = 24.0

# The widest window `_normalize_prices` will map onto the slot grid, measured
# from the EARLIEST record in the reply.
#
# The same week as `battery_optimizer.MODELED_HORIZON_MAX_HOURS`, which takes
# its value from this name so the two cannot drift: anything past it is dropped
# by the planner anyway, so expanding it here only spends memory to produce
# slots that are then discarded. `price_retain_max_age_hours` is clamped to the
# same week.
#
# It is the second half of the bound. `MAX_RECORD_SPAN_HOURS` stops ONE record
# from being unbounded; this stops a reply of ten thousand plausible records
# from being unbounded in aggregate, and it is what keeps the list handed to
# `PriceHorizonMonitor.merge_with_retained` -- and therefore the retained set --
# finite.
MAX_NORMALIZED_WINDOW_HOURS = 168.0

Span = Tuple[datetime.datetime, datetime.datetime]


def _unclaimed(
    start: datetime.datetime,
    end: datetime.datetime,
    claimed: List[Span],
) -> List[Span]:
    """The parts of ``[start, end)`` no more specific record has taken.

    ``claimed`` is disjoint and sorted (see :func:`_merge_claims`). Attributing
    each minute to exactly one record is what keeps overlaps from being SUMMED
    into coverage that was never published.
    """
    pieces: List[Span] = []
    cursor = start
    for taken_start, taken_end in claimed:
        if taken_end <= cursor:
            continue
        if taken_start >= end:
            break
        if taken_start > cursor:
            pieces.append((cursor, min(taken_start, end)))
        cursor = max(cursor, taken_end)
        if cursor >= end:
            return pieces
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def _merge_claims(
    claimed: List[Span], start: datetime.datetime, end: datetime.datetime
) -> List[Span]:
    """``claimed`` with ``[start, end)`` added, kept disjoint and sorted."""
    merged: List[Span] = []
    for span_start, span_end in sorted(claimed + [(start, end)]):
        if merged and span_start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span_end))
        else:
            merged.append((span_start, span_end))
    return merged


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

        # When a corrupt record was last reported (see
        # `_warn_malformed_record`).
        self._last_malformed_warning: Optional[datetime.datetime] = None

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

    def _shift(self, dt: datetime.datetime, minutes: float) -> datetime.datetime:
        """Add elapsed time to *dt*, in UTC when it identifies an instant.

        Wall-clock addition is an hour wrong across a DST transition, and both
        Riga transitions fall inside a published day.
        """
        delta = datetime.timedelta(minutes=minutes)
        if dt.tzinfo is not None and dt.utcoffset() is not None:
            return (dt.astimezone(datetime.timezone.utc) + delta).astimezone(dt.tzinfo)
        return dt + delta

    def _warn_malformed_record(self, message: str) -> None:
        """WARN about a corrupt record at most once an hour.

        Prices are re-fetched every slot and again on every bounded retry, so a
        source that keeps sending the same bad record would otherwise produce a
        line per poll -- and be scrolled past exactly like the WARNINGs that
        need reading. Rate-limited is not silenced: it says so again an hour
        later, for as long as the source keeps saying it.
        """
        now = None
        try:
            now = self.datetime()
        except Exception:
            now = None
        last = self._last_malformed_warning
        if now is not None and last is not None:
            cmp_last, cmp_now = normalize_tz_pair(last, now)
            if (cmp_now - cmp_last).total_seconds() < MALFORMED_WARNING_INTERVAL_S:
                return
        self._last_malformed_warning = now
        self.log(message, level="WARNING")

    def _match_awareness(
        self, end: datetime.datetime, start: datetime.datetime
    ) -> datetime.datetime:
        """*end* expressed with the same awareness as its own *start*.

        A record whose two fields disagree is not exotic: with no timezone
        configured in AppDaemon, both parsers leave `start` and `end` with
        whatever awareness their own ISO strings carried, so a source
        publishing one bare and the other with an offset produces exactly this
        pair. Comparing them raised `TypeError` out of `_normalize_prices` and
        out of `get_prices`, which does not catch it -- one malformed record
        cost the whole reply and the fetch that would have noticed.

        The conversion goes through the local zone when there is one, so an end
        published as 08:00Z against a +03:00 clock is read as 11:00 local, not
        as a span that runs backwards.
        """
        if is_aware(end) == is_aware(start):
            return end
        tz = self.get_timezone()
        if is_aware(start):
            # Naive end: local wall time, like every other naive value here.
            return end.replace(tzinfo=tz or start.tzinfo)
        # Aware end against a naive start: convert to local time, then drop the
        # offset so the two describe the same clock.
        if tz is not None:
            end = end.astimezone(tz)
        return end.replace(tzinfo=None)

    def _reject_malformed_end(self, start, raw) -> None:
        """Say, once an hour, that a record declared an unusable ``end``."""
        self._warn_malformed_record(
            f"Price record for {start} declares an end of {raw!r}, which is "
            f"not a usable timestamp; dropping it. The interval it claimed is "
            f"a gap until a source publishes it properly."
        )

    def _interval_end(
        self,
        start: datetime.datetime,
        end: Optional[datetime.datetime],
    ) -> Optional[datetime.datetime]:
        """Exclusive end of the interval a record actually covers, or None.

        The source's own ``end`` when it published one. Otherwise exactly ONE
        `slot_minutes` slot -- never a width guessed from how far away the next
        surviving timestamp happens to be.

        ``None`` means the record is unusable and the caller must DROP it. An
        ABSENT end is a normal case with a documented answer; an end that does
        not come after its start is not a width to guess around, it is evidence
        the record is corrupt. Falling back to one slot for it published a
        price for that interval on the strength of a field saying the opposite,
        and coverage is what the schedule, the horizon monitor and the
        provenance guard are all built on.

        An end that was PUBLISHED but cannot be read (`MALFORMED_END`, or any
        value that is not a datetime) is the same kind of evidence and gets the
        same answer. Both parsers already drop such a record where they build
        it; this is the second gate, for a `PricePoint` assembled anywhere else.
        """
        if end is None:
            return self._shift(start, self.slot_minutes)
        if not isinstance(end, datetime.datetime):
            self._reject_malformed_end(start, end)
            return None
        end = self._match_awareness(end, start)
        if instant_key(end) > instant_key(start):
            return end
        self._warn_malformed_record(
            f"Price record for {start} declares an end of {end}, which "
            f"is not after its start; dropping it. The interval it claimed is "
            f"a gap until a source publishes it properly."
        )
        return None

    def _slots_in(self, span: datetime.timedelta) -> int:
        """How many slots a span would expand to, WITHOUT expanding it."""
        minutes = span.total_seconds() / 60.0
        if minutes <= 0:
            return 0
        slots = int(minutes // self.slot_minutes)
        if minutes % self.slot_minutes:
            slots += 1
        return slots

    def _bounded_to_window(self, records: List[Tuple]) -> List[Tuple]:
        """*records* clipped to `MAX_NORMALIZED_WINDOW_HOURS` of coverage.

        Measured from the EARLIEST record, because that is the only anchor a
        reply carries: the app clock says nothing about how far ahead a source
        chose to publish, and a reply that legitimately starts in the past must
        still reach a week forward from its own beginning.

        A record that STRADDLES the bound is truncated rather than dropped -
        the part inside the window was published like any other. A record
        entirely past it is dropped. One WARNING says how many slots went, so a
        horizon shortened here is a stated fact and not, as it would be
        downstream, a sequence that simply ends.

        Arithmetic only: nothing here iterates a span, which is the whole point
        (see `MAX_RECORD_SPAN_HOURS`).
        """
        if not records:
            return records
        window_end = (
            min(r[2] for r in records)
            + datetime.timedelta(hours=MAX_NORMALIZED_WINDOW_HOURS)
        )

        bounded: List[Tuple] = []
        dropped_slots = 0
        dropped_records = 0
        first_dropped = None
        for span, rank, start_key, end_key, start_dt, price in records:
            if start_key >= window_end:
                dropped_slots += self._slots_in(end_key - start_key)
                dropped_records += 1
                if first_dropped is None or start_key < first_dropped:
                    first_dropped = start_key
                continue
            if end_key > window_end:
                dropped_slots += self._slots_in(end_key - window_end)
                dropped_records += 1
                if first_dropped is None or window_end < first_dropped:
                    first_dropped = window_end
                end_key = window_end
                span = end_key - start_key
            bounded.append((span, rank, start_key, end_key, start_dt, price))

        if dropped_slots:
            self._warn_malformed_record(
                f"Price data reaches beyond the "
                f"{MAX_NORMALIZED_WINDOW_HOURS:.0f}h normalization window from "
                f"{records[0][4] if records else '?'}: dropping {dropped_slots} "
                f"slot(s) across {dropped_records} record(s), from "
                f"{first_dropped} on. Nothing past that window is planned, so "
                f"expanding it would only spend memory on slots that are then "
                f"discarded."
            )
        return bounded

    def _normalize_prices(self, prices: List[PricePoint]) -> List[PricePoint]:
        """Map published intervals onto the configured slot grid.

        A price covers `[start, end)` and NOTHING else. Coarser intervals are
        expanded within their own span; finer ones are aggregated, and only
        into a slot they FILL. A slot no source interval covers completely
        stays absent -- it is a gap, and every consumer downstream is built to
        see one (`PriceHorizonMonitor.evaluate` reports `gap` or
        `missing_current_interval`, the planner models the interval as a forced
        HOLD, `execute_scheduled_mode` refuses to send anything else).

        There is deliberately no inference from timestamp SPACING. It was the
        whole mechanism here, and it cannot distinguish "these are 30-minute
        intervals" from "these are 15-minute intervals and the record between
        them is missing" -- so a reply holding 10:00-10:15 at 0.01 and
        10:30-10:45 at 1.00 published 10:15 at 0.01, and the planner charged
        the live slot on it.

        Where two records overlap, the MOST SPECIFIC one owns the shared
        minutes: the narrower span wins, and equal spans are broken by the
        later position in the reply. Every minute is then attributed to exactly
        ONE record.

        That is the documented rule made real. It used to say "publication
        order - a later interval contributes only the minutes an earlier one
        did not already cover" while the code sorted by instant (discarding
        input order) and clipped to `covered_until`, which is EARLIEST START
        WINS. The two differ exactly where a correction lives: 10:00-11:00 at
        0.10 followed by 10:15-10:30 at 0.90 came out as four quarter hours at
        0.10, the correction discarded in silence. Publication order cannot be
        recovered from a reply - the records arrive as a list, with no
        publication timestamps - but specificity is a property of the record
        itself, and a narrower interval nested inside a wider one is a more
        specific statement about the minutes they share.

        Attribution, never SUMMING: summing raw overlaps would let two
        intervals that both start inside a slot add up to its full width and
        mark it covered when the beginning of the slot never was.
        """
        if not prices:
            return []

        slot_minutes = self.slot_minutes
        slot = datetime.timedelta(minutes=slot_minutes)
        # UTC instant of a slot start -> [local slot start, covered minutes,
        # sum(price * minutes)].
        buckets: Dict[datetime.datetime, List] = {}

        max_span = datetime.timedelta(hours=MAX_RECORD_SPAN_HOURS)

        records = []
        for index, point in enumerate(prices):
            # ONE awareness for the whole record, before anything is compared.
            # `_align_to_slot` reads a naive value as local wall time, so a
            # naive `start` at a site that HAS a timezone produced an aware
            # cursor against naive keys and raised `TypeError` in the walk
            # below - the same defect `_match_awareness` fixes one field
            # earlier, and the fetch has no catch for either.
            start_dt = ensure_local_tz(point.time, self.get_timezone())
            interval_end = self._interval_end(
                start_dt, getattr(point, "end", None)
            )
            if interval_end is None:
                # Corrupt: an end at or before its own start, or one that was
                # published unreadably. `_interval_end` has already said so,
                # once an hour.
                continue
            start_key = instant_key(start_dt)
            end_key = instant_key(interval_end)
            span = end_key - start_key
            if span > max_span:
                # Rejected on the DECLARED span, before a single bucket exists
                # for it. See `MAX_RECORD_SPAN_HOURS`.
                self._warn_malformed_record(
                    f"Price record for {start_dt} declares a span of "
                    f"{span.total_seconds() / 3600.0:.1f}h, longer than the "
                    f"{MAX_RECORD_SPAN_HOURS:.0f}h a single published interval "
                    f"can be; dropping it. The interval it claimed is a gap "
                    f"until a source publishes it properly."
                )
                continue
            records.append((span, -index, start_key, end_key,
                            start_dt, point.price))

        records = self._bounded_to_window(records)

        # Narrowest span first, then the later record of an equal-span pair.
        # `-index` sorts second so the key is total and the outcome cannot
        # depend on the sort being stable.
        claimed: List[Tuple[datetime.datetime, datetime.datetime]] = []
        for _span, _rank, start_key, end_key, start_dt, price in sorted(
            records, key=lambda r: (r[0], r[1])
        ):
            for piece_start, piece_end in _unclaimed(
                start_key, end_key, claimed
            ):
                offset = (piece_start - start_key).total_seconds() / 60.0
                cursor = self._align_to_slot(self._shift(start_dt, offset))
                while True:
                    cursor_key = instant_key(cursor)
                    if cursor_key >= piece_end:
                        break
                    overlap_start = max(cursor_key, piece_start)
                    overlap_end = min(cursor_key + slot, piece_end)
                    minutes = (
                        overlap_end - overlap_start
                    ).total_seconds() / 60.0
                    if minutes > 0:
                        bucket = buckets.setdefault(
                            cursor_key, [cursor, 0.0, 0.0]
                        )
                        bucket[1] += minutes
                        bucket[2] += price * minutes
                    cursor = self._shift(cursor, slot_minutes)
            claimed = _merge_claims(claimed, start_key, end_key)

        normalized: List[PricePoint] = []
        for cursor_key in sorted(buckets):
            slot_start, minutes, weighted = buckets[cursor_key]
            if minutes < slot_minutes - 1e-6:
                # Partially covered: a fraction of a slot is not a price for
                # the slot.
                continue
            normalized.append(PricePoint(
                time=slot_start,
                price=weighted / minutes,
                end=self._shift(slot_start, slot_minutes),
            ))
        return normalized

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

                # How far this price REACHES, as published. Dropping it left
                # `_normalize_prices` guessing the width from the spacing of
                # whatever records survived, which prices the gaps.
                interval_end = self._parse_interval_end(entry.get("end"), tz)
                if interval_end is MALFORMED_END:
                    # Published and unreadable: the record is corrupt, and one
                    # slot of coverage for it is a price nobody stated.
                    self._reject_malformed_end(start_dt, entry.get("end"))
                    continue

                # Convert EUR/MWh to EUR/kWh
                price_kwh = float(price_mwh) / 1000.0
                prices.append(PricePoint(
                    time=start_dt,
                    price=price_kwh,
                    end=interval_end,
                ))

            except (ValueError, TypeError) as e:
                self.log(f"Error parsing price entry {entry}: {e}", level="DEBUG")
                continue

        return prices

    def _parse_interval_end(self, value, tz):
        """The source's ``end`` field, TRI-STATE.

        * a local datetime -- the source said how far this price reaches;
        * ``None`` -- the source said NOTHING, and the record then covers
          exactly one slot (`_interval_end`);
        * `MALFORMED_END` -- the source published an ``end`` and it is not a
          timestamp: an empty string, a bare number, a list, garbage.

        The third case used to answer ``None``, which is the second case's
        answer, so a record whose own width field was unreadable was granted a
        slot of coverage anyway. Both are "no usable end", but only one of them
        is a record saying so on purpose. See `_MalformedEnd`.
        """
        if value is None:
            return None
        try:
            if isinstance(value, datetime.datetime):
                end_dt = value
            elif isinstance(value, str) and value.strip():
                end_dt = datetime.datetime.fromisoformat(
                    value.strip().replace("Z", "+00:00")
                )
            else:
                # A bool, an int, an empty string, a list: published, unusable.
                return MALFORMED_END
        except (ValueError, TypeError):
            return MALFORMED_END
        if tz and end_dt.tzinfo:
            return end_dt.astimezone(tz)
        if tz:
            return end_dt.replace(tzinfo=tz)
        return end_dt

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
                        # The raw attributes publish an `end`; keeping it is
                        # what stops a missing record from being filled in from
                        # its neighbour's spacing.
                        interval_end = self._parse_interval_end(
                            entry.get("end"), tz
                        )
                        if interval_end is MALFORMED_END:
                            # Published and unreadable: drop it, exactly as the
                            # service path does.
                            self._reject_malformed_end(dt, entry.get("end"))
                            continue
                        prices.append(PricePoint(
                            time=dt,
                            price=float(price),
                            end=interval_end,
                        ))
                    except (ValueError, TypeError):
                        pass
        else:
            # Simple list of prices. This format carries no ends, but its own
            # RESOLUTION is explicit coverage: 24 values for a local day are
            # hourly intervals, 96 are quarter-hours, and the list is the whole
            # day by construction. That is a statement by the format, not an
            # inference from the spacing of surviving records, so each value is
            # given the step as its end.
            step_minutes = 60
            day_minutes = 1440
            local_midnight = datetime.datetime.combine(date_for_simple, datetime.time(), tzinfo=tz)
            if tz is not None:
                next_midnight = datetime.datetime.combine(
                    date_for_simple + datetime.timedelta(days=1), datetime.time(), tzinfo=tz
                )
                day_minutes = int((
                    next_midnight.astimezone(datetime.timezone.utc)
                    - local_midnight.astimezone(datetime.timezone.utc)
                ).total_seconds() / 60)
            if len(price_data) > 0 and day_minutes % len(price_data) == 0:
                step_minutes = int(day_minutes / len(price_data))
            elif len(price_data) > 0:
                self.log(
                    f"Price list length {len(price_data)} does not divide the "
                    f"{day_minutes}-minute local day; assuming hourly steps. "
                    f"Times may be wrong if the source uses a different resolution.",
                    level="WARNING",
                )
            for idx, price in enumerate(price_data):
                if price is not None:
                    if tz:
                        dt = (
                            local_midnight.astimezone(datetime.timezone.utc)
                            + datetime.timedelta(minutes=idx * step_minutes)
                        ).astimezone(tz)
                    else:
                        dt = local_midnight + datetime.timedelta(minutes=idx * step_minutes)
                    prices.append(PricePoint(
                        time=dt,
                        price=float(price),
                        end=self._shift(dt, step_minutes),
                    ))

        return prices
