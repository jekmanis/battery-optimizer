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

    # --- Reporting fields (never read by the DP objective) ---
    marginal_value_eur_kwh: Optional[float] = None
    # EUR per battery DC kWh that THIS slot's decision is worth, using the same
    # tariff arithmetic the DP scores the slot with. Reported so the schedule
    # log explains the decision even when the tracked stored-energy cost basis
    # has legitimately degenerated to 0.0000 (PV booked at a zero export floor).

    value_basis: Optional[str] = None
    # Which economics the number above describes: "avoided-import", "export",
    # "landed-charge" or "kept".


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
