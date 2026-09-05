"""
Data models for the Battery Optimizer.

Contains pure data structures with minimal dependencies - enums and dataclasses
used throughout the battery optimizer system.
"""

import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional


class BatteryMode(Enum):
    """Battery operating modes."""
    HOLD = 0
    CHARGE = 1
    DISCHARGE = 2


@dataclass
class PricePoint:
    """Represents a single time-slot price data point."""
    time: datetime.datetime
    price: float

    end: Optional[datetime.datetime] = None
    # EXCLUSIVE end of the interval this price covers, as the SOURCE published
    # it. Both fetch paths carry one (`{start, end, price}` from
    # `nordpool.get_price_indices_for_date`, `{start, end, value}` from the
    # HACS `raw_today` / `raw_tomorrow` attributes) and both used to discard
    # it, leaving `NordPoolPriceService._normalize_prices` to infer the
    # interval width from the minimum SPACING between the timestamps that
    # survived - and then expand every point by that factor. A reply holding
    # only 10:00-10:15 and 10:30-10:45 was thereby "expanded" into four
    # quarter hours, publishing 10:15 at the 10:00 price. Spacing is not
    # coverage: it cannot tell a 30-minute interval from a 15-minute one with
    # the next record missing.
    #
    # `None` means the source stated no end, and then the point covers exactly
    # ONE `slot_minutes` slot. It is deliberately optional and last, so the
    # many `PricePoint(time, price)` call sites (the DP's canonicalization,
    # every fixture) keep working - none of them describe a source reply.

    @property
    def has_declared_end(self) -> bool:
        """Whether the source stated how far this price reaches."""
        return self.end is not None


# Provenance marker for `ScheduleEntry.price_source`. "market" means the slot
# was planned on an interval that a price source actually published and that
# `PriceHorizonMonitor` accepted as coverage. There is deliberately no marker
# for anything else: the optimizer no longer manufactures a price, so an entry
# with no marker is an entry whose price cannot be vouched for.
PRICE_SOURCE_MARKET = "market"

# The reason a slot carries when nobody published a price for its interval.
# ONE spelling, because three things test it: the orchestrator's
# `_is_no_price_fallback` (the stand-in HOLD for an unpriced CURRENT interval),
# the DP's forced-HOLD entry for an unpriced interval INSIDE the horizon, and
# `execute_scheduled_mode`, which must treat the two identically once the
# second becomes the first.
NO_PRICE_REASON = "no_price"


@dataclass
class ScheduleEntry:
    """Represents a scheduled battery mode for a specific time slot."""
    time: datetime.datetime
    mode: BatteryMode
    reason: str = ""

    # --- Direct control fields ---
    export_rate: Optional[int] = None
    # None = use mode default, 0 = zero export, 100 = full export

    ac_charge_mode: Optional[str] = None
    # None = auto-detect, "disabled" / "pv_priority" / "ac_priority"

    price_source: Optional[str] = None
    # Provenance of the price this slot was planned on. `PRICE_SOURCE_MARKET`
    # when the interval came from a published, coverage-validated price;
    # None when nothing can vouch for it (a fallback HOLD, a restored entry,
    # a hand-built safety entry).
    #
    # This is the difference between "the planner chose CHARGE" and "the
    # planner chose CHARGE at a number we made up", which the schedule used to
    # be unable to state. `execute_scheduled_mode` refuses to send a non-HOLD
    # CURRENT-slot entry that carries no marker, and a rebuild that finds the
    # current interval unpriced retains the previous entry only if it has one.
    # Never infer it from the reason string -- that is prose.

    # --- Reporting fields (never read by the DP objective) ---
    marginal_value_eur_kwh: Optional[float] = None
    # EUR per battery DC kWh that THIS slot's decision is worth, using the same
    # tariff arithmetic the DP scores the slot with. Reported so the schedule
    # log explains the decision even when the tracked stored-energy cost basis
    # has legitimately degenerated to 0.0000 (PV booked at a zero export floor).

    value_basis: Optional[str] = None
    # Which economics the number above describes: "avoided-import", "export",
    # "landed-charge" or "kept" — plus "kept (cloud-safe)" for a slot the
    # orchestrator's cloud-safe hedge turned into discharge_to_load. That one
    # keeps the DP's HOLD number on purpose: the hedge only fires where the
    # modeled energy flow is unchanged, so "kept" is still the right basis, and
    # the suffix says why a DISCHARGE row carries it.

    energy_limited: bool = False
    # The plan DECLARES that the pack runs dry inside this slot and that the
    # grid was priced to cover the remainder ("until depleted" in the reason).
    # It is the plan's own statement about its physics, and it is what makes
    # `plan_validation`'s conservation check falsifiable: a slot the continuous
    # replay cannot serve in full is either declared here, or the plan credited
    # the battery with service it does not have. Never infer it from the reason
    # string -- that is prose.


@dataclass
class LearningStats:
    """Aggregated learning statistics for battery performance."""
    # Charging rates by SOC range (kW observed at different SOC levels)
    charge_rates_by_soc: Dict[str, List[float]] = field(default_factory=dict)
    # Totals
    total_energy_charged_kwh: float = 0.0
    total_energy_discharged_kwh: float = 0.0
    total_charge_cost_eur: float = 0.0
    total_discharge_revenue_eur: float = 0.0
    # Timestamps
    last_observation: Optional[str] = None
    # Temperature-aware charge rates: {"25-50": {"5-10": [3.1, 3.2], "10-15": [4.2, 4.5]}}
    charge_rates_by_soc_temp: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    # Temperature warming rates during charging: {"10-15": [0.3, 0.4, 0.35]} (°C/minute by starting temp range)
    temp_warming_rates: Dict[str, List[float]] = field(default_factory=dict)
    # Temperature cooling rates during idle: {">20": [0.015, 0.012]} (decay rate per minute by starting temp range)
    temp_cooling_rates: Dict[str, List[float]] = field(default_factory=dict)
    # Recent minimum battery temperatures for ambient estimation (last ~48h of observations)
    recent_min_temps: List[float] = field(default_factory=list)
    # Raw thermal observations for k1/k2 calibration. Each sample is
    # [T_start, T_end, duration_minutes, avg_battery_power_kw, ambient_temp].
    # The aggregated temp_warming_rates / temp_cooling_rates cannot be used for
    # this: they are already averaged and carry no power information.
    thermal_samples: List[List[float]] = field(default_factory=list)
    # Calibrated thermal coefficients: {"k1": per-minute, "k2": C/kWh, "n": samples}
    thermal_coeffs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'LearningStats':
        # Ignore unknown fields when loading from JSON
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


@dataclass
class LoadProfileStats:
    """Aggregated load observations per time slot."""
    samples_by_slot: Dict[str, List[float]] = field(default_factory=dict)  # slot -> W samples
    observation_count: int = 0
    last_observation: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'LoadProfileStats':
        return cls(**data)


@dataclass
class PvProfileStats:
    """Aggregated PV production observations per time slot."""
    samples_by_slot: Dict[str, List[float]] = field(default_factory=dict)  # slot -> W samples
    observation_count: int = 0
    last_observation: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PvProfileStats':
        return cls(**data)


@dataclass
class PredictionAccuracyStats:
    """Tracks predicted vs actual load ratios per time-of-day slot."""
    ratios_by_slot: Dict[str, List[float]] = field(default_factory=dict)
    global_ratios: List[float] = field(default_factory=list)
    total_comparisons: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'PredictionAccuracyStats':
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


@dataclass(frozen=True)
class ScheduleModeCounts:
    """Mode census of a schedule, counted from the entries themselves.

    The DP reports its own counts, but the orchestrator mutates the schedule
    afterwards (the cloud-safe HOLD -> DISCHARGE(to load) conversion). Reporting
    the DP's pre-conversion counts printed two contradictory censuses back to
    back on every sunny run: "Schedule generated: ... 30 hold" followed a few
    lines later by the schedule log's own "Total: ... 6 hold". Counts must be
    derived from the schedule that will actually execute.
    """

    charge: int = 0
    hold: int = 0
    export: int = 0          # DISCHARGE with export_rate > 0
    self_consume: int = 0    # DISCHARGE to load

    @property
    def discharge(self) -> int:
        return self.export + self.self_consume

    def summary_parts(self) -> List[str]:
        """The shared wording of the "N charge, M discharge(self), ..." line."""
        parts = [f"{self.charge} charge"]
        if self.self_consume:
            parts.append(f"{self.self_consume} discharge(self)")
        if self.export:
            parts.append(f"{self.export} discharge(export)")
        parts.append(f"{self.hold} hold")
        return parts


def count_schedule_modes(schedule: Dict[datetime.datetime, ScheduleEntry]
                         ) -> ScheduleModeCounts:
    """Count modes in a schedule as it stands (post any conversion)."""
    charge = hold = export = self_consume = 0
    for entry in schedule.values():
        if entry.mode == BatteryMode.CHARGE:
            charge += 1
        elif entry.mode == BatteryMode.HOLD:
            hold += 1
        elif entry.mode == BatteryMode.DISCHARGE:
            if entry.export_rate is not None and entry.export_rate > 0:
                export += 1
            else:
                self_consume += 1
    return ScheduleModeCounts(
        charge=charge, hold=hold, export=export, self_consume=self_consume
    )
