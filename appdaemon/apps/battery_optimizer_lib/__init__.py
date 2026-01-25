"""
Battery Optimizer package.

This package contains helper modules for the main BatteryOptimizer AppDaemon app:
- models: Data classes and enums (BatteryMode, PricePoint, ScheduleEntry, etc.)
- learning_engine: Self-learning battery performance tracking
- load_profile: Statistical load forecasting by time-of-day
- price_service: Nord Pool electricity price fetching
- tou_sync: TOU schedule sync and inverter control via Modbus
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
    # Functions
    "_quantile",
]
