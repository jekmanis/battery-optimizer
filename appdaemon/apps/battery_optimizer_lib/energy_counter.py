"""Physical-plausibility guard for the inverter's daily energy counters.

`sensor.growatt_battery_battery_charge_today` /
`..._discharge_today` are monotonically rising daily kWh totals that reset at
local midnight.  Every charge/discharge the cost tracker books comes from the
*difference* between two consecutive readings of one of them, so a single bad
poll is indistinguishable from real energy unless something checks the number
against physics.

Nothing did, and on 2026-09-07 20:18:29 both counters dipped to 0 for one poll
and came straight back:

    discharge 19.1 -> 0.0 -> 19.1     charge 18.8 -> 0.0 -> 18.8

The dip produced a negative delta, which the old code discarded as "noise"
(`energy_delta < 0.05`), and the *return* produced ``+19.1`` and ``+18.8`` kWh
deltas that were booked as a real discharge and a real grid charge within 30 ms
of each other, at a constant SOC of 55 %, on a pack whose usable capacity is
12.87 kWh.  The landed cost basis was overwritten with 0.3138 EUR/kWh from a
non-event and the stored-energy accumulator was slammed to capacity.

The guard is the single owner of "what does this reading mean".  It keeps the
last *accepted* value and the time it was accepted, and classifies each new
reading as one of:

``ACCEPTED``
    a forward delta within the plausible bound; the caller may book it (after
    its own noise floor).
``MIDNIGHT_RESET``
    the daily counter rolled over near local midnight — the historical
    behaviour, preserved exactly.
``IMPLAUSIBLE_JUMP``
    a forward delta larger than the pack could physically have moved.
``IMPLAUSIBLE_DROP``
    a backwards step on a monotonic counter that is not a midnight reset.
``NOISE_DROP``
    a backwards step smaller than the counter's own resolution (rounding).
``FIRST``
    no baseline yet; the reading becomes the baseline.

**Both directions re-anchor.**  A rejected reading updates the baseline to the
value that was rejected.  That is what makes the guard idempotent against both
failure shapes:

* a counter that jumps and *stays* jumped (a genuine offset) costs exactly one
  rejection, after which deltas are measured from the new level;
* a one-poll dip (``v -> 0 -> v``) books nothing in either direction: the dip is
  an ``IMPLAUSIBLE_DROP`` (re-anchored to 0), and the return is then a ``+v``
  ``IMPLAUSIBLE_JUMP`` off that anchor (re-anchored back to v).  Had the drop
  been ignored instead of re-anchored, the return would have read as a delta of
  0 — also harmless — but a genuine reset would then never be picked up.  Had
  the drop re-anchored *silently*, the operator would see one unexplained
  rejection instead of the pair that describes the glitch.
* a doubling (``v -> 2v -> v``) is symmetric: the doubling is a jump, the return
  is a drop.

The bound
---------

``max_energy_per_poll = max(min_bound_kwh, max_rate_kw * elapsed_hours *
safety_factor)``

The rate term is the physical one: a daily counter can only have advanced by
what actually flowed since the last accepted reading, and ``elapsed`` is
measured from that reading, so a sensor that stalls for several slots and then
catches up is covered automatically.  ``safety_factor`` (2.0) absorbs a battery
that briefly exceeds its configured rate and clock jitter.

Pack capacity is deliberately **not** an absolute ceiling: a *daily* counter can
legitimately exceed the pack size — 4.5 kW for five hours is 22.5 kWh into a
14.3 kWh battery.  Capacity only enters through ``min_bound_kwh``, the floor
that keeps the guard from rejecting real catch-up deltas when ``elapsed`` is
small or mismeasured; the cost tracker uses ``max(2 kWh, 25 % of capacity)``,
the same shape as the accumulator's drift tolerance.

Against the incident: the discharge counter's last accepted reading was 3 min
old, giving ``max(3.575, 4.5 * 0.05 * 2 = 0.45) = 3.575`` kWh — the 19.1 kWh
delta violates it by 5x.  The charge counter had not moved since the 19:38
restart, giving ``max(3.575, 4.5 * 0.66 * 2 = 5.94) = 5.94`` kWh — the 18.8 kWh
delta violates that by 3x.  A genuine 2 kWh catch-up after 30 min is bounded by
``4.5 * 0.5 * 2 = 4.5`` kWh and is accepted.
"""

import datetime
from dataclasses import dataclass
from typing import Optional

# Historical midnight-reset detection, unchanged: within this many minutes of
# local midnight AND the new value is small enough to be a fresh daily total.
MIDNIGHT_WINDOW_MINUTES = 5
MIDNIGHT_RESET_MAX_KWH = 1.0

# A backwards step at or below this is the counter's own rounding, not a glitch.
DEFAULT_DROP_TOLERANCE_KWH = 0.05

DEFAULT_SAFETY_FACTOR = 2.0


def is_daily_counter_reset(
    current: float,
    previous: float,
    now: datetime.datetime,
    window_minutes: int = MIDNIGHT_WINDOW_MINUTES,
    max_value_kwh: float = MIDNIGHT_RESET_MAX_KWH,
) -> bool:
    """Return True if a drop is the inverter's daily counter rolling over.

    ``now`` must be LOCAL time (HA's configured timezone), which is what the
    inverter resets on.
    """
    if current >= previous:
        return False
    minutes_since_midnight = now.hour * 60 + now.minute
    return (
        minutes_since_midnight < window_minutes
        or minutes_since_midnight > (1440 - window_minutes)
    ) and current < max_value_kwh


@dataclass(frozen=True)
class CounterVerdict:
    """What one counter reading means, and why."""

    FIRST = "first"
    ACCEPTED = "accepted"
    NOISE_DROP = "noise-drop"
    MIDNIGHT_RESET = "midnight-reset"
    IMPLAUSIBLE_JUMP = "implausible-jump"
    IMPLAUSIBLE_DROP = "implausible-drop"

    kind: str
    value_kwh: float
    previous_kwh: Optional[float]
    delta_kwh: float
    bound_kwh: float
    elapsed_hours: float
    message: str = ""

    @property
    def accepted(self) -> bool:
        """True only when ``delta_kwh`` is real energy the caller may book."""
        return self.kind == self.ACCEPTED

    @property
    def is_glitch(self) -> bool:
        """True when the reading was rejected as physically implausible."""
        return self.kind in (self.IMPLAUSIBLE_JUMP, self.IMPLAUSIBLE_DROP)


class EnergyCounterGuard:
    """Derives a plausible delta from one monotonic daily energy counter.

    Pure and side-effect free apart from its own two-field baseline, so it is
    unit-testable without AppDaemon or Home Assistant.
    """

    def __init__(
        self,
        max_rate_kw: float,
        min_bound_kwh: float,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
        drop_tolerance_kwh: float = DEFAULT_DROP_TOLERANCE_KWH,
        name: str = "energy counter",
    ):
        self._max_rate_kw = max(0.0, float(max_rate_kw))
        self._min_bound_kwh = max(0.0, float(min_bound_kwh))
        self._safety_factor = max(1.0, float(safety_factor))
        self._drop_tolerance_kwh = max(0.0, float(drop_tolerance_kwh))
        self._name = name
        self._last_value: Optional[float] = None
        self._last_time: Optional[datetime.datetime] = None

    # -- state -------------------------------------------------------------

    @property
    def last_value(self) -> Optional[float]:
        """Last ACCEPTED or re-anchored counter value, kWh."""
        return self._last_value

    @property
    def last_time(self) -> Optional[datetime.datetime]:
        """When the baseline was set."""
        return self._last_time

    def reset(self, value: Optional[float], now: datetime.datetime) -> None:
        """Adopt ``value`` as the baseline without producing a delta.

        Used at startup and whenever the sensors recover from unavailable: the
        counter may have moved a long way while nobody was reading it, and that
        gap is not a booking.
        """
        if value is None:
            self._last_value = None
            self._last_time = None
            return
        self._last_value = float(value)
        self._last_time = now

    # -- bound -------------------------------------------------------------

    def bound_for(self, elapsed_hours: float) -> float:
        """Largest delta this counter could plausibly show over ``elapsed_hours``."""
        rate_term = self._max_rate_kw * max(0.0, elapsed_hours) * self._safety_factor
        return max(self._min_bound_kwh, rate_term)

    def _elapsed_hours(self, now: datetime.datetime) -> float:
        last = self._last_time
        if last is None:
            return 0.0
        if now.tzinfo is not None and last.tzinfo is None:
            last = last.replace(tzinfo=now.tzinfo)
        elif now.tzinfo is None and last.tzinfo is not None:
            last = last.replace(tzinfo=None)
        return max(0.0, (now - last).total_seconds() / 3600.0)

    # -- classification ----------------------------------------------------

    def observe(self, value: float, now: datetime.datetime) -> CounterVerdict:
        """Classify a new reading and advance the baseline to it.

        The baseline is advanced for EVERY outcome — accepted, reset or
        rejected — so a persistent counter offset costs one rejection and not a
        rejection per poll.
        """
        value = float(value)
        previous = self._last_value
        elapsed = self._elapsed_hours(now)
        bound = self.bound_for(elapsed)

        if previous is None or self._last_time is None:
            self._last_value = value
            self._last_time = now
            return CounterVerdict(
                kind=CounterVerdict.FIRST,
                value_kwh=value,
                previous_kwh=None,
                delta_kwh=0.0,
                bound_kwh=bound,
                elapsed_hours=elapsed,
                message=f"{self._name} baseline set to {value:.3f} kWh",
            )

        delta = value - previous
        self._last_value = value
        self._last_time = now

        if delta < 0.0:
            if is_daily_counter_reset(value, previous, now):
                return CounterVerdict(
                    kind=CounterVerdict.MIDNIGHT_RESET,
                    value_kwh=value,
                    previous_kwh=previous,
                    delta_kwh=0.0,
                    bound_kwh=bound,
                    elapsed_hours=elapsed,
                    message=(
                        f"{self._name} daily reset {previous:.2f} -> {value:.2f} kWh"
                    ),
                )
            if -delta <= self._drop_tolerance_kwh:
                return CounterVerdict(
                    kind=CounterVerdict.NOISE_DROP,
                    value_kwh=value,
                    previous_kwh=previous,
                    delta_kwh=0.0,
                    bound_kwh=bound,
                    elapsed_hours=elapsed,
                    message=(
                        f"{self._name} rounded back {previous:.3f} -> {value:.3f} kWh"
                    ),
                )
            return CounterVerdict(
                kind=CounterVerdict.IMPLAUSIBLE_DROP,
                value_kwh=value,
                previous_kwh=previous,
                delta_kwh=0.0,
                bound_kwh=bound,
                elapsed_hours=elapsed,
                message=(
                    f"{self._name} went BACKWARDS {previous:.3f} -> {value:.3f} kWh "
                    f"(delta {delta:+.3f} kWh) after {elapsed * 60:.1f} min at "
                    f"{now.strftime('%H:%M')} local, which is not a midnight reset "
                    f"(reset needs < {MIDNIGHT_WINDOW_MINUTES} min from midnight and "
                    f"a value < {MIDNIGHT_RESET_MAX_KWH:.1f} kWh); nothing booked, "
                    f"baseline re-anchored to {value:.3f} kWh"
                ),
            )

        if delta > bound:
            return CounterVerdict(
                kind=CounterVerdict.IMPLAUSIBLE_JUMP,
                value_kwh=value,
                previous_kwh=previous,
                delta_kwh=0.0,
                bound_kwh=bound,
                elapsed_hours=elapsed,
                message=(
                    f"{self._name} JUMPED {previous:.3f} -> {value:.3f} kWh "
                    f"(delta {delta:+.3f} kWh) in {elapsed * 60:.1f} min, above the "
                    f"plausible bound {bound:.3f} kWh "
                    f"(max {self._max_rate_kw:.2f} kW x {elapsed:.3f} h x "
                    f"{self._safety_factor:.1f} safety, floor "
                    f"{self._min_bound_kwh:.3f} kWh); nothing booked, baseline "
                    f"re-anchored to {value:.3f} kWh"
                ),
            )

        return CounterVerdict(
            kind=CounterVerdict.ACCEPTED,
            value_kwh=value,
            previous_kwh=previous,
            delta_kwh=delta,
            bound_kwh=bound,
            elapsed_hours=elapsed,
            message=(
                f"{self._name} {previous:.3f} -> {value:.3f} kWh "
                f"(delta {delta:+.3f} kWh)"
            ),
        )
