"""
Battery Optimizer package.

This package contains helper modules for the main BatteryOptimizer AppDaemon app:
- models: Data classes and enums (BatteryMode, PricePoint, ScheduleEntry, etc.)
- learning_engine: Self-learning battery performance tracking
- load_profile: Statistical load forecasting by time-of-day
- price_service: Nord Pool electricity price fetching
- tou_sync: TOU schedule sync and inverter control via Modbus
- timezone_utils: Timezone-aware datetime comparison and slot alignment
- ha_helpers: Home Assistant state reading helpers
"""

from .models import (
    BatteryMode,
    PricePoint,
    ScheduleEntry,
    TouPeriod,
    LearningStats,
    LoadProfileStats,
)
from .learning_engine import BatteryLearningEngine
from .load_profile import LoadProfile, _quantile
from .price_service import NordPoolPriceService
from .tou_sync import TouSyncManager
from .timezone_utils import (
    normalize_tz_pair,
    datetimes_match_slot,
    dt_ge,
    dt_gt,
    dt_lt,
    ensure_local_tz,
    align_to_slot,
    next_slot_time,
    next_interval_time,
    lookup_by_hour,
    duration_minutes,
)
from .ha_helpers import (
    is_state_valid,
    get_float_state,
    get_bool_state,
    get_string_state,
    SensorReader,
)

__all__ = [
    # Models
    "BatteryMode",
    "PricePoint",
    "ScheduleEntry",
    "TouPeriod",
    "LearningStats",
    "LoadProfileStats",
    # Classes
    "BatteryLearningEngine",
    "LoadProfile",
    "NordPoolPriceService",
    "TouSyncManager",
    "SensorReader",
    # Timezone utilities
    "normalize_tz_pair",
    "datetimes_match_slot",
    "dt_ge",
    "dt_gt",
    "dt_lt",
    "ensure_local_tz",
    "align_to_slot",
    "next_slot_time",
    "next_interval_time",
    "lookup_by_hour",
    "duration_minutes",
    # HA helpers
    "is_state_valid",
    "get_float_state",
    "get_bool_state",
    "get_string_state",
    # Functions
    "_quantile",
]
