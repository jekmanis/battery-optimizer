"""
Sliding PV forecast bias estimation and slot-energy PV sampling.

Two production defects motivated this module:

1. The reactive "PV below forecast" check compared a *single instantaneous*
   power reading, taken 5 seconds after a slot boundary, against the slot
   *average* forecast.  Morning/evening ramps and passing clouds made that
   comparison meaningless — 43 recalculations were triggered in 33 hours.
2. The correction derived from that comparison was applied to the current slot
   only (``PvForecastService.refresh_for_shortfall``), so every rerun still
   planned the remaining horizon on the same 3-5x optimistic forecast.

``PvBiasTracker`` fixes both:

* ``add_sample`` accumulates PV power readings inside the open slot.  A closed
  slot therefore reports its *mean* power (= slot energy / slot hours), built
  from many samples, not one boundary reading.
* ``get_factor`` returns a clamped, slowly-relaxing median of
  ``actual / forecast`` over a sliding window.  The caller multiplies the whole
  remaining horizon by it.

The module is deliberately free of Home Assistant / AppDaemon dependencies so
it can be unit tested directly (``battery_optimizer.py`` is not unit tested).
"""

import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .timezone_utils import canonical_slot_key, instant_key

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import BatteryOptimizerConfig


def _median(values: List[float]) -> float:
    """Return the median of *values* (0.0 for an empty list)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass
class PvBiasConfig:
    """Configuration for :class:`PvBiasTracker`."""

    enabled: bool = True
    slot_minutes: int = 15
    window_minutes: int = 120
    min_slots: int = 2
    min_forecast_kw: float = 0.2
    min_factor: float = 0.2
    max_factor: float = 1.5
    decay_slots: int = 8
    min_samples_per_slot: int = 3
    shortfall_threshold: float = 0.5
    next_day_weight: float = 0.5
    next_day_min_factor: float = 0.7

    @classmethod
    def from_main_config(cls, cfg: "BatteryOptimizerConfig") -> "PvBiasConfig":
        """Create from the central :class:`BatteryOptimizerConfig`."""
        return cls(
            enabled=cfg.pv_bias_enabled,
            slot_minutes=cfg.slot_minutes,
            window_minutes=cfg.pv_bias_window_minutes,
            min_slots=cfg.pv_bias_min_slots,
            min_forecast_kw=cfg.pv_reactive_min_forecast_w / 1000.0,
            min_factor=cfg.pv_bias_min_factor,
            max_factor=cfg.pv_bias_max_factor,
            decay_slots=cfg.pv_bias_decay_slots,
            min_samples_per_slot=cfg.pv_reactive_min_samples,
            shortfall_threshold=cfg.pv_reactive_threshold,
            next_day_weight=cfg.pv_bias_next_day_weight,
            next_day_min_factor=cfg.pv_bias_next_day_min_factor,
        )


@dataclass
class ClosedSlot:
    """A completed slot with its forecast snapshot and measured mean power."""

    slot: datetime.datetime
    forecast_kw: float
    actual_kw: float
    samples: int
    ratio: float


class PvBiasTracker:
    """
    Accumulates PV power samples per slot and derives a sliding forecast bias.

    Usage from the orchestrator:

    * every ``pv_sample_seconds``: ``close_slots_before(now_slot)`` then
      ``ensure_slot_forecast(now_slot, raw_forecast_kw)`` then
      ``add_sample(now, pv_kw)``
    * before generating a schedule: ``get_factor(now)`` and multiply the raw
      PV prediction of every current/future slot by it.
    """

    MAX_CLOSED = 8

    def __init__(
        self,
        config: PvBiasConfig,
        align_to_slot_func: Callable[[datetime.datetime], datetime.datetime],
        log_func: Optional[Callable] = None,
    ):
        self._config = config
        self._align_to_slot = align_to_slot_func
        self._log = log_func

        # Open (still accumulating) slots: key -> [sum_kw, count]
        self._open: Dict[datetime.datetime, List[float]] = {}
        # Forecast snapshot taken the first time a slot is seen.
        self._forecast: Dict[datetime.datetime, float] = {}
        # Recently closed slots (bounded).
        self._closed: Dict[datetime.datetime, ClosedSlot] = {}
        # Sliding (slot_start, ratio) history used for the bias factor.
        self._ratios: List[Tuple[datetime.datetime, float]] = []
        self._shortfall_streak: int = 0
        # Last timezone discovered from *align_to_slot_func*.  Kept only as a
        # fallback for a transiently naive alignment function — it is re-derived
        # on every localization, see :meth:`_key_tz`.
        self._tz: Optional[datetime.tzinfo] = None

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def _key_tz(self, sample: datetime.datetime) -> Optional[datetime.tzinfo]:
        """Return the timezone the alignment function produces *right now*.

        This must NOT be cached across the horizon.  The orchestrator's
        ``_get_local_timezone`` returns a **fixed-offset** ``datetime.timezone``
        recomputed from the current wall clock (AppDaemon's ``self.datetime()``
        is naive in production, so the value falls back to
        ``datetime.now().astimezone().tzinfo``).  That object is ``UTC+03:00``
        in summer and ``UTC+02:00`` in winter.

        A permanently cached ``+03:00`` therefore survived the autumn DST
        transition: ``_localize`` kept stamping naive input (``add_sample`` with
        ``self.datetime()``) with the stale offset while the already-aware
        ``now_slot`` used by ``ensure_slot_forecast`` / ``close_slots_before``
        moved to ``+02:00``.  The same wall-clock instant then produced two
        different slot keys — one ClosedSlot with a forecast and no samples,
        another with samples and no forecast, both rejected by
        ``_register_closed`` — so the sliding bias factor and the reactive
        shortfall detector went silently dead until the next restart.

        Re-deriving is a single ``align_to_slot`` call, which is cheap.  The
        last successful discovery is kept only as a fallback for an alignment
        function that transiently raises or returns naive.
        """
        try:
            aligned = self._align_to_slot(sample)
        except (TypeError, ValueError):
            return self._tz
        if aligned.tzinfo is not None and aligned.utcoffset() is not None:
            self._tz = aligned.tzinfo
        return self._tz

    def _localize(self, dt: datetime.datetime) -> datetime.datetime:
        """Bring an externally supplied datetime into the tracker's key space.

        AppDaemon's ``self.datetime()`` is **naive local time** in production
        while every stored slot key is tz-aware (``align_to_slot`` attaches the
        HA local timezone).  Subtracting one from the other raises ``TypeError``,
        and that exception used to be swallowed: ``_prune`` then kept every
        entry forever and ``get_factor``/``describe`` reported an age of zero, so
        the sliding window and the decay-to-1.0 relaxation were silently dead —
        a stale 0.2 factor survived for days with no fresh observations.

        A naive value is therefore interpreted as local wall time and stamped
        with the alignment function's timezone.  If that function is itself
        naive (no local timezone configured) the value is left alone, so a fully
        naive tracker stays internally consistent.
        """
        if dt.tzinfo is not None and dt.utcoffset() is not None:
            return dt
        tz = self._key_tz(dt)
        if tz is None:
            return dt
        return dt.replace(tzinfo=tz)

    def _key(self, dt: datetime.datetime) -> datetime.datetime:
        """Slot-align *dt* and return a DST-safe dictionary key."""
        return canonical_slot_key(self._align_to_slot(self._localize(dt)))

    def _order(self, dt: datetime.datetime) -> datetime.datetime:
        """Return a value safe for chronological ordering of slot keys.

        Naive input is localized first — mixing a naive ``now`` with aware slot
        keys is the defect this method exists to prevent.
        """
        return instant_key(self._localize(dt))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def add_sample(self, dt: datetime.datetime, pv_kw: float) -> None:
        """Add a PV power sample (kW) to the slot containing *dt*.

        Negative readings are ignored — an inverter reporting a negative PV
        power is reporting a fault, not production.
        """
        if pv_kw is None:
            return
        try:
            value = float(pv_kw)
        except (TypeError, ValueError):
            return
        if value < 0:
            return
        key = self._key(dt)
        bucket = self._open.get(key)
        if bucket is None:
            self._open[key] = [value, 1]
        else:
            bucket[0] += value
            bucket[1] += 1

    def ensure_slot_forecast(
        self, slot_dt: datetime.datetime, forecast_kw: float
    ) -> float:
        """Record the forecast snapshot for *slot_dt* — first write wins.

        This must never be overwritten: ``PvForecastService.refresh_for_shortfall``
        caps the cached forecast of the current slot at the observed production,
        so a later read would report ``ratio ~ 1.0`` and the bias signal would
        erase itself.
        """
        key = self._key(slot_dt)
        if key in self._forecast:
            return self._forecast[key]
        try:
            value = max(0.0, float(forecast_kw))
        except (TypeError, ValueError):
            value = 0.0
        self._forecast[key] = value
        return value

    def slot_mean_kw(self, slot_dt: datetime.datetime) -> Optional[float]:
        """Mean measured PV power (kW) of an open slot, or None if no samples."""
        bucket = self._open.get(self._key(slot_dt))
        if not bucket or bucket[1] <= 0:
            return None
        return bucket[0] / bucket[1]

    def slot_sample_count(self, slot_dt: datetime.datetime) -> int:
        """Number of samples accumulated in an open slot."""
        bucket = self._open.get(self._key(slot_dt))
        return int(bucket[1]) if bucket else 0

    def get_slot_forecast(self, slot_dt: datetime.datetime) -> Optional[float]:
        """The forecast snapshot recorded for a slot, if any."""
        return self._forecast.get(self._key(slot_dt))

    # ------------------------------------------------------------------
    # Slot closing
    # ------------------------------------------------------------------

    def close_slots_before(
        self, now_slot_dt: datetime.datetime
    ) -> List[ClosedSlot]:
        """Close every slot strictly older than *now_slot_dt*.

        Idempotent: each slot is returned exactly once.  Safe to call from both
        the sampling timer and the adaptive callback.
        """
        boundary = self._order(self._key(now_slot_dt))

        candidates = set(self._open.keys()) | set(self._forecast.keys())
        due = sorted(
            (k for k in candidates if self._order(k) < boundary),
            key=self._order,
        )

        closed: List[ClosedSlot] = []
        for key in due:
            bucket = self._open.pop(key, None)
            forecast_kw = self._forecast.pop(key, 0.0)
            if key in self._closed:
                continue
            samples = int(bucket[1]) if bucket else 0
            actual_kw = (bucket[0] / bucket[1]) if (bucket and bucket[1] > 0) else 0.0
            ratio = (actual_kw / forecast_kw) if forecast_kw > 0 else 0.0
            entry = ClosedSlot(
                slot=key,
                forecast_kw=forecast_kw,
                actual_kw=actual_kw,
                samples=samples,
                ratio=ratio,
            )
            self._register_closed(entry)
            closed.append(entry)

        return closed

    def _register_closed(self, entry: ClosedSlot) -> None:
        self._closed[entry.slot] = entry
        if len(self._closed) > self.MAX_CLOSED:
            oldest = sorted(self._closed.keys(), key=self._order)[
                : len(self._closed) - self.MAX_CLOSED
            ]
            for key in oldest:
                del self._closed[key]

        trustworthy = entry.samples >= self._config.min_samples_per_slot
        usable_forecast = entry.forecast_kw >= self._config.min_forecast_kw

        if trustworthy and usable_forecast:
            self._ratios.append((entry.slot, entry.ratio))

        # Shortfall streak: only a trustworthy measurement may change it.
        if not trustworthy:
            return
        if not usable_forecast:
            # Night / negligible forecast — nothing to be short of.
            self._shortfall_streak = 0
        elif entry.ratio < self._config.shortfall_threshold:
            self._shortfall_streak += 1
        else:
            self._shortfall_streak = 0

    def get_closed(self, slot_dt: datetime.datetime) -> Optional[ClosedSlot]:
        """The :class:`ClosedSlot` for *slot_dt*, if it has been closed."""
        return self._closed.get(self._key(slot_dt))

    @property
    def shortfall_streak(self) -> int:
        """Consecutive closed slots measured below ``shortfall_threshold`` since
        the last recalculation (see :meth:`reset_shortfall_streak`)."""
        return self._shortfall_streak

    def reset_shortfall_streak(self) -> None:
        """Clear the streak after the caller has acted on it.

        The streak is otherwise only cleared by a GOOD slot, so under persistent
        cloud cover it grows 2, 3, 4, 5 ... and the
        ``streak < pv_reactive_consecutive_slots`` guard can never hold again —
        every subsequent slot pays for a full recalculation, which is exactly
        the 43-recalculations-in-33-hours behaviour the threshold was added to
        stop. Resetting on trigger makes the guard mean "N consecutive
        shortfalls SINCE THE LAST RECALC", so the cadence is bounded at one
        recalculation per ``pv_reactive_consecutive_slots`` slots.
        """
        self._shortfall_streak = 0

    # ------------------------------------------------------------------
    # Bias factor
    # ------------------------------------------------------------------

    def _prune(self, now: datetime.datetime) -> None:
        """Drop ratio observations older than ``window_minutes``.

        Every value is normalized through :meth:`_order` first, so a naive
        ``now`` (AppDaemon's ``self.datetime()``) is comparable with the aware
        slot keys.  An entry that is *still* incomparable after normalization is
        a genuine bug: it is logged and DISCARDED rather than kept, because
        keeping it is what disabled the window in production.
        """
        if not self._ratios:
            return
        now_key = self._order(now)
        window = datetime.timedelta(minutes=self._config.window_minutes)
        kept: List[Tuple[datetime.datetime, float]] = []
        for slot, ratio in self._ratios:
            try:
                age = now_key - self._order(slot)
            except TypeError:
                self._log_error(
                    f"PV bias: discarding unusable ratio observation {slot!r} "
                    f"(not comparable with now={now!r})"
                )
                continue
            if age <= window:
                kept.append((slot, ratio))
        self._ratios = kept

    def _log_error(self, message: str) -> None:
        if self._log is None:
            return
        try:
            self._log(message, level="ERROR")
        except TypeError:  # pragma: no cover - log callable without level kwarg
            self._log(message)

    def _age_minutes(
        self, now: datetime.datetime, slot: datetime.datetime
    ) -> float:
        """Minutes between the newest observation *slot* and *now*."""
        return (self._order(now) - self._order(slot)).total_seconds() / 60.0

    def ratio_count(self, now: datetime.datetime) -> int:
        """Number of in-window ratio observations."""
        self._prune(now)
        return len(self._ratios)

    def get_factor(self, now: datetime.datetime) -> float:
        """Return the PV forecast bias multiplier for the remaining horizon.

        1.0 means "trust the provider".  Values below 1.0 scale the forecast
        down.  With no fresh observations the factor relaxes back to 1.0 over
        ``decay_slots`` slots.
        """
        if not self._config.enabled:
            return 1.0

        self._prune(now)
        if len(self._ratios) < max(1, self._config.min_slots):
            return 1.0

        factor = _median([r for _, r in self._ratios])
        factor = max(self._config.min_factor, min(self._config.max_factor, factor))

        newest = max((slot for slot, _ in self._ratios), key=self._order)
        age_minutes = self._age_minutes(now, newest)
        age_slots = age_minutes / max(1, self._config.slot_minutes)
        if age_slots > 1.0:
            t = min(1.0, (age_slots - 1.0) / max(1, self._config.decay_slots))
            factor = factor + (1.0 - factor) * t

        return round(factor, 3)

    @staticmethod
    def _day_distance(now: datetime.datetime, slot: datetime.datetime) -> int:
        """Whole local days from *now*'s date to *slot*'s date (0 = same day)."""
        try:
            if now.tzinfo is not None and slot.tzinfo is not None:
                slot = slot.astimezone(now.tzinfo)
            elif now.tzinfo is None and slot.tzinfo is not None:
                slot = slot.replace(tzinfo=None)
            elif now.tzinfo is not None and slot.tzinfo is None:
                now = now.replace(tzinfo=None)
            return max(0, (slot.date() - now.date()).days)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return 0

    def factor_for_slot(
        self,
        factor: float,
        now: datetime.datetime,
        slot: datetime.datetime,
    ) -> float:
        """Attenuate the horizon bias *factor* for a slot on a later local day.

        The measured ``actual / forecast`` ratio conflates two things: a
        systematic calibration error of the provider's model (which does carry
        over to tomorrow) and today's actual weather (which does not). Applying
        the raw factor to every future slot meant that two cloudy hours before
        the 13:15 daily optimization scaled the WHOLE of tomorrow down to the
        ``min_factor`` clamp — a 5x under-forecast that systematically pushes
        the DP toward expensive grid charging. The decay in :meth:`get_factor`
        cannot help: it only starts once observations go stale, which does not
        happen within the same day.

        Each day beyond today keeps ``next_day_weight ** days`` of the deviation
        from 1.0, and the result is floored at ``next_day_min_factor`` — a
        separate, looser clamp than the same-day ``min_factor``.
        """
        if factor == 1.0:
            return 1.0
        days = self._day_distance(self._localize(now), self._localize(slot))
        if days <= 0:
            return factor

        weight = self._config.next_day_weight ** days
        attenuated = 1.0 + (factor - 1.0) * weight
        if attenuated < self._config.next_day_min_factor:
            attenuated = self._config.next_day_min_factor
        return round(attenuated, 3)

    def describe(self, now: datetime.datetime) -> str:
        """Human readable summary of the current bias evidence."""
        if not self._config.enabled:
            return "disabled"
        self._prune(now)
        if not self._ratios:
            return "no observations"
        newest = max((slot for slot, _ in self._ratios), key=self._order)
        age_minutes = self._age_minutes(now, newest)
        return (
            f"median of {len(self._ratios)} slots over "
            f"{self._config.window_minutes}min, newest {age_minutes:.0f}min ago"
        )
