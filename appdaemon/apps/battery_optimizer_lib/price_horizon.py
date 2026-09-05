"""Price-coverage health and bounded price-recovery scheduling.

One owner for two questions the orchestrator used to answer nowhere:

1. **Is the price horizon usable right now?**  Not "did a fetch succeed", not
   "is the cache non-empty", not "when did we last recalculate" - but: does the
   price data actually contain the interval we are living in, does it continue
   without gaps, and does it reach as far forward as the current publication
   window says it should?

2. **When do we ask again after it does not?**  A transient empty response or a
   response that arrives without tomorrow's intervals used to leave the app on
   an old or absent plan until the next daily optimization.  Recovery is a
   bounded backoff with at most one pending retry, not an unthrottled loop.

Everything here is deterministic and pure with respect to time: the caller
passes `now`.  Timer registration stays in the orchestrator, because that is
where AppDaemon's `run_in` and the app lock live.

Canonical instants, never slot counts
-------------------------------------
A local day is 23, 24 or 25 hours at the Europe/Riga transitions, so "96 slots"
is not a coverage test.  Coverage is measured between UTC instants: intervals
are keyed by `instant_key`, stepped by `slot_minutes` of elapsed time, and the
required horizon end is a *local midnight* converted to its UTC instant.  A
23-hour day therefore needs 92 fifteen-minute intervals and a 25-hour day needs
100, without either number appearing anywhere.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .models import PricePoint
from .timezone_utils import align_to_slot, ensure_local_tz, instant_key


# Reasons reported by :meth:`PriceHorizonMonitor.evaluate`.  They are part of
# the diagnostics payload, so keep them stable.
REASON_OK = "ok"
REASON_NO_PRICES = "no_prices"
REASON_MISSING_CURRENT = "missing_current_interval"
REASON_GAP = "gap"
REASON_TOMORROW_MISSING = "tomorrow_missing"
REASON_SHORT_HORIZON = "short_horizon"

# Failures that a fresh, usable price snapshot actually resolves. Anything else
# ("no_schedule", "soc_unavailable") is outstanding work that a healthy price
# verdict must NOT disarm: it says nothing about an empty schedule or an
# unreadable battery.
COVERAGE_REASONS = frozenset({
    REASON_NO_PRICES,
    REASON_MISSING_CURRENT,
    REASON_GAP,
    REASON_TOMORROW_MISSING,
    REASON_SHORT_HORIZON,
})


def is_coverage_reason(reason: Optional[str]) -> bool:
    """True when `reason` is a price-coverage failure (not e.g. no_schedule)."""
    return reason in COVERAGE_REASONS


@dataclass
class PriceHorizonConfig:
    """Knobs for coverage evaluation and the recovery backoff."""

    slot_minutes: int = 15
    tomorrow_prices_hour: int = 14
    retry_enabled: bool = True
    # 30 s, 2 min, 5 min, then `retry_max_seconds` forever.
    retry_delays_seconds: Tuple[int, ...] = (30, 120, 300)
    retry_max_seconds: int = 900
    # How long the retained set survives WITHOUT a non-empty reply. It is not
    # a per-interval age: see `merge_with_retained`. Only FUTURE intervals are
    # ever retained, so this is the backstop for a source that went silent.
    retain_max_age_hours: float = 36.0
    # Rate limit for the repeated "still incomplete" WARNING.
    log_interval_minutes: int = 30

    @classmethod
    def from_main_config(cls, config) -> "PriceHorizonConfig":
        return cls(
            slot_minutes=config.slot_minutes,
            tomorrow_prices_hour=config.tomorrow_prices_hour,
            retry_enabled=config.price_retry_enabled,
            retry_delays_seconds=tuple(config.price_retry_delays_seconds),
            retry_max_seconds=config.price_retry_max_seconds,
            retain_max_age_hours=config.price_retain_max_age_hours,
        )


@dataclass(frozen=True)
class HorizonHealth:
    """Verdict on one price snapshot, at one instant."""

    ok: bool
    reason: str
    has_current: bool
    interval_count: int
    expects_tomorrow: bool
    required_end: Optional[datetime.datetime] = None
    horizon_end: Optional[datetime.datetime] = None

    @property
    def short_by(self) -> Optional[datetime.timedelta]:
        """How much forward coverage is missing (None when unknown/ok)."""
        if self.horizon_end is None or self.required_end is None:
            return None
        missing = self.required_end - self.horizon_end
        return missing if missing > datetime.timedelta(0) else None


class PriceHorizonMonitor:
    """Coverage verdicts, retained intervals, and the recovery backoff state.

    The monitor does not own timers.  The orchestrator asks it *what* to do
    (`record_failure` returns the delay for the next attempt) and does the
    `run_in` itself, so the app lock and the generation guard stay in one place.
    """

    def __init__(
        self,
        config: PriceHorizonConfig,
        get_timezone_func: Optional[Callable] = None,
        log_func: Optional[Callable] = None,
        get_zone_func: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            get_timezone_func: the app's local timezone. Used to interpret a
                naive clock and to align slots. A fixed offset is fine here:
                everything it feeds is an instant comparison.
            get_zone_func: a timezone WITH DST RULES, used only for local
                midnight arithmetic. `BatteryOptimizer._get_local_timezone()`
                degrades to `datetime.now().astimezone().tzinfo` - a fixed
                `datetime.timezone` - whenever AppDaemon hands the app a naive
                `self.datetime()`, and combining a date with a fixed offset puts
                the midnight on the far side of a DST transition an hour wrong.
                Defaults to `get_timezone_func`, with a one-shot warning when
                the result carries no DST information.
        """
        self.config = config
        self._get_timezone = get_timezone_func or (lambda: None)
        self._get_zone = get_zone_func or self._get_timezone
        self._log = log_func
        self._warned_fixed_offset = False

        self.attempts: int = 0
        self.last_health: Optional[HorizonHealth] = None
        self.last_success_time: Optional[datetime.datetime] = None
        self.last_success_horizon_end: Optional[datetime.datetime] = None
        self.last_failure_reason: Optional[str] = None

        self._retained: Dict[datetime.datetime, PricePoint] = {}
        self._retained_at: Optional[datetime.datetime] = None
        self._log_times: Dict[str, datetime.datetime] = {}

    # ------------------------------------------------------------------
    # time helpers
    # ------------------------------------------------------------------

    def _tz(self):
        try:
            return self._get_timezone()
        except Exception:
            return None

    def _local(self, dt: datetime.datetime) -> datetime.datetime:
        """Interpret a possibly naive orchestrator datetime as local time."""
        return ensure_local_tz(dt, self._tz())

    def _key(self, dt: datetime.datetime) -> datetime.datetime:
        """Canonical comparison key: the UTC instant (naive stays naive)."""
        return instant_key(self._local(dt))

    def _slot_key(self, now: datetime.datetime) -> datetime.datetime:
        tz = self._tz()
        return instant_key(
            align_to_slot(self._local(now), self.config.slot_minutes, tz)
        )

    @property
    def _step(self) -> datetime.timedelta:
        return datetime.timedelta(minutes=self.config.slot_minutes)

    def _boundary_zone(self):
        """Timezone used for local-midnight arithmetic (DST rules preferred)."""
        try:
            zone = self._get_zone()
        except Exception:
            zone = None
        if isinstance(zone, datetime.tzinfo):
            return zone
        return self._tz()

    def _warn_once_without_dst_rules(self, zone, local_now) -> None:
        """Say so when the midnight boundary is computed on a fixed offset.

        A fixed offset gets the boundary exactly right on every ordinary day
        and an hour wrong on the two DST days a year - too late in spring (a
        complete horizon reads as incomplete, so the app retries all evening)
        and too early in autumn (an incomplete horizon reads as complete).
        Harmless where the site genuinely has no DST; worth one line otherwise.
        """
        if zone is None or self._warned_fixed_offset:
            return
        try:
            has_dst_rules = zone.dst(local_now.replace(tzinfo=None)) is not None
        except Exception:
            has_dst_rules = False
        if has_dst_rules:
            return
        self._warned_fixed_offset = True
        if self._log is not None:
            self._log(
                f"Price horizon boundary uses a timezone without DST rules "
                f"({zone}); the required horizon end can be an hour off on the "
                f"two DST transition days. Set a timezone in AppDaemon so "
                f"get_timezone() reports a region name.",
                level="WARNING",
            )

    def required_horizon_end(
        self, now: datetime.datetime
    ) -> Tuple[datetime.datetime, bool]:
        """(required end instant, whether tomorrow is expected to be published).

        Before the configured publication hour only the rest of today is
        required - tomorrow's intervals are legitimately unavailable and must
        not drive a retry loop.  From that hour on, the horizon must reach the
        end of tomorrow.

        The boundary is a LOCAL midnight converted to an instant, so a 23- or
        25-hour DST day needs exactly its own number of intervals.  It is
        computed in the ZONE (see `get_zone_func`), not in whatever fixed
        offset happens to be in force right now: `combine(2024-04-01, 00:00,
        +02:00)` is an hour later than Europe/Riga's actual midnight that day.
        """
        local_now = self._local(now)
        zone = self._boundary_zone()
        if zone is not None and local_now.tzinfo is not None:
            try:
                local_now = local_now.astimezone(zone)
            except Exception:
                zone = local_now.tzinfo
        else:
            zone = local_now.tzinfo
        self._warn_once_without_dst_rules(zone, local_now)

        expects_tomorrow = local_now.hour >= self.config.tomorrow_prices_hour
        target_date = local_now.date() + datetime.timedelta(
            days=2 if expects_tomorrow else 1
        )
        boundary = datetime.datetime.combine(
            target_date, datetime.time(0, 0), tzinfo=zone
        )
        return instant_key(boundary), expects_tomorrow

    # ------------------------------------------------------------------
    # coverage evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, prices: Optional[Sequence[PricePoint]], now: datetime.datetime
    ) -> HorizonHealth:
        """Judge one price snapshot for usable coverage from `now` forward."""
        required_end, expects_tomorrow = self.required_horizon_end(now)
        keys = sorted({self._key(p.time) for p in (prices or [])})

        if not keys:
            health = HorizonHealth(
                ok=False,
                reason=REASON_NO_PRICES,
                has_current=False,
                interval_count=0,
                expects_tomorrow=expects_tomorrow,
                required_end=required_end,
            )
            self.last_health = health
            return health

        slot_key = self._slot_key(now)
        if slot_key not in keys:
            health = HorizonHealth(
                ok=False,
                reason=REASON_MISSING_CURRENT,
                has_current=False,
                interval_count=len(keys),
                expects_tomorrow=expects_tomorrow,
                required_end=required_end,
            )
            self.last_health = health
            return health

        index = keys.index(slot_key)
        step = self._step
        last = index
        while last + 1 < len(keys) and keys[last + 1] == keys[last] + step:
            last += 1
        horizon_end = keys[last] + step
        has_more_beyond = last + 1 < len(keys)

        if horizon_end >= required_end:
            reason = REASON_OK
            ok = True
        else:
            ok = False
            if has_more_beyond:
                # Data exists past the break: this is a hole, not a pending
                # publication.
                reason = REASON_GAP
            elif expects_tomorrow:
                reason = REASON_TOMORROW_MISSING
            else:
                reason = REASON_SHORT_HORIZON

        health = HorizonHealth(
            ok=ok,
            reason=reason,
            has_current=True,
            interval_count=len(keys),
            expects_tomorrow=expects_tomorrow,
            required_end=required_end,
            horizon_end=horizon_end,
        )
        self.last_health = health
        return health

    # ------------------------------------------------------------------
    # retained intervals
    # ------------------------------------------------------------------

    @property
    def retained_prices(self) -> List[PricePoint]:
        """Last merged snapshot, chronologically ordered (may be empty)."""
        return [self._retained[k] for k in sorted(self._retained)]

    def merge_with_retained(
        self, prices: Optional[Sequence[PricePoint]], now: datetime.datetime
    ) -> List[PricePoint]:
        """Fill gaps in `prices` with still-valid intervals fetched earlier.

        Rules, in order of precedence:

        * A fresh interval always wins over a retained one with the same
          instant.  Nothing is invented and nothing is extrapolated.
        * Only FUTURE intervals (>= the current slot) are ever retained, so a
          date rollover cannot resurrect yesterday.
        * The whole retained set is dropped once the last NON-EMPTY reply is
          older than `retain_max_age_hours`.

        This exists because a service reply containing today only used to
        replace a cache that already held tomorrow, silently shortening the
        horizon straight after the daily optimization.

        Two properties of that rule are deliberate and worth stating, because
        the obvious reading of "retain_max_age_hours" is not what it measures:

        * The age is measured from the last non-empty reply, not from when each
          interval was fetched.  A steady stream of successful (even if short)
          replies therefore keeps an older interval alive indefinitely.  That is
          bounded in practice, not by this timer: retained intervals are pruned
          to the future on every merge, so an interval can only survive until
          its own instant passes - at most the length of the horizon.  The knob
          is the backstop for a source that goes permanently silent.
        * An interval a later reply OMITS is retained, even when the reply spans
          it.  A partial response and a genuine withdrawal look identical from
          here, and dropping a known price because one response was short is the
          failure this method exists to prevent.  Nord Pool does not withdraw a
          published interval; if a source ever does, its correction will only
          take effect for instants the reply actually contains.
        """
        incoming = {self._key(p.time): p for p in (prices or [])}
        slot_key = self._slot_key(now)
        now_key = self._key(now)

        retained = self._valid_retained(now_key, slot_key)

        merged = dict(incoming)
        added = 0
        for key, point in retained.items():
            if key not in merged:
                merged[key] = point
                added += 1

        if added and self._log is not None:
            self._log(
                f"Price refresh returned {len(incoming)} intervals; kept {added} "
                f"still-valid cached intervals to preserve the known horizon",
                level="DEBUG",
            )

        self._retained = {k: v for k, v in merged.items() if k >= slot_key}
        if incoming:
            self._retained_at = now_key

        return [merged[k] for k in sorted(merged)]

    def _valid_retained(
        self, now_key: datetime.datetime, slot_key: datetime.datetime
    ) -> Dict[datetime.datetime, PricePoint]:
        if not self._retained:
            return {}
        if self._retained_at is None:
            return {}
        age_hours = (now_key - self._retained_at).total_seconds() / 3600.0
        if age_hours > self.config.retain_max_age_hours:
            if self._log is not None:
                self._log(
                    f"Discarding retained price intervals: last successful fetch "
                    f"was {age_hours:.1f}h ago",
                    level="WARNING",
                )
            self._retained = {}
            self._retained_at = None
            return {}
        return {k: v for k, v in self._retained.items() if k >= slot_key}

    def forget_retained(self) -> None:
        """Drop retained intervals (used when the app is disabled/terminated)."""
        self._retained = {}
        self._retained_at = None

    # ------------------------------------------------------------------
    # backoff state
    # ------------------------------------------------------------------

    def delay_for_attempt(self, attempt: int) -> int:
        """Backoff delay in seconds for a 1-based attempt number."""
        delays = self.config.retry_delays_seconds or ()
        if attempt <= 0:
            attempt = 1
        if attempt <= len(delays):
            return int(delays[attempt - 1])
        return int(self.config.retry_max_seconds)

    @property
    def next_delay_seconds(self) -> int:
        """Delay the next failure would use."""
        return self.delay_for_attempt(self.attempts + 1)

    def record_failure(
        self, reason: str, now: datetime.datetime, health: Optional[HorizonHealth] = None
    ) -> int:
        """Count one coverage failure and return the delay before retrying."""
        self.attempts += 1
        self.last_failure_reason = reason
        if health is not None:
            self.last_health = health
        return self.delay_for_attempt(self.attempts)

    def note_coverage_ok(
        self, health: Optional[HorizonHealth], now: datetime.datetime
    ) -> None:
        """Record a usable verdict WITHOUT touching the backoff.

        Used when coverage is fine but something else is still outstanding (an
        empty schedule, an unreadable SOC): the retry that is going to fix that
        must keep its place in the backoff, and its reason must keep naming the
        real problem.
        """
        self.last_success_time = self._key(now)
        if health is not None:
            self.last_health = health
            self.last_success_horizon_end = health.horizon_end

    def record_success(
        self, health: Optional[HorizonHealth], now: datetime.datetime
    ) -> None:
        """Reset the backoff after coverage was confirmed usable."""
        self.attempts = 0
        self.last_failure_reason = None
        self.note_coverage_ok(health, now)

    def reset_backoff(self) -> None:
        self.attempts = 0

    # ------------------------------------------------------------------
    # logging + diagnostics
    # ------------------------------------------------------------------

    def should_log(self, key: str, now: datetime.datetime) -> bool:
        """Rate limit repeated identical messages to one per log interval."""
        now_key = self._key(now)
        last = self._log_times.get(key)
        if last is not None:
            elapsed = (now_key - last).total_seconds()
            if elapsed < self.config.log_interval_minutes * 60:
                return False
        self._log_times[key] = now_key
        return True

    def clear_log_gate(self, key: str) -> None:
        self._log_times.pop(key, None)

    @staticmethod
    def _iso(dt: Optional[datetime.datetime]) -> Optional[str]:
        return dt.isoformat() if dt is not None else None

    def diagnostics(
        self,
        retry_pending: bool,
        current_slot_priced: Optional[bool] = None,
        current_slot_entry: Optional[str] = None,
    ) -> Dict:
        """Payload for `sensor.battery_optimizer`.

        Args:
            current_slot_priced: whether a published interval covers the slot
                being executed right now. `ok` does not answer this: a horizon
                can be short (`tomorrow_missing`) with the current interval
                perfectly well priced, and `missing_current_interval` can clear
                for the NEXT slot while the one on the inverter never had a
                price at all.
            current_slot_entry: how that slot got its entry - "planned",
                "retained" (a decision made earlier on a real price) or
                "fallback" (HOLD/no_price). Without it, a HOLD on the dashboard
                cannot be told from a plan.
        """
        health = self.last_health
        return {
            "current_slot_priced": current_slot_priced,
            "current_slot_entry": current_slot_entry,
            "ok": bool(health.ok) if health is not None else None,
            "reason": health.reason if health is not None else None,
            "horizon_end": self._iso(health.horizon_end) if health else None,
            "required_end": self._iso(health.required_end) if health else None,
            "intervals": health.interval_count if health else 0,
            "expects_tomorrow": bool(health.expects_tomorrow) if health else None,
            # Why the LAST retry was armed. It is not always `reason`: an empty
            # current slot ("no_schedule") is a coverage failure even when the
            # price data itself validates.
            "last_failure_reason": self.last_failure_reason,
            "last_success_horizon_end": self._iso(self.last_success_horizon_end),
            "last_success_time": self._iso(self.last_success_time),
            "retry_pending": bool(retry_pending),
            "retry_attempts": self.attempts,
            "retry_next_seconds": (
                self.next_delay_seconds if self.attempts or not (health and health.ok)
                else None
            ),
            "retained_intervals": len(self._retained),
        }
