"""
Battery Optimizer package.

This package contains helper modules for the main BatteryOptimizer AppDaemon app:
- config: Configuration dataclass with typed fields and validation
- models: Data classes and enums (BatteryMode, PricePoint, ScheduleEntry, etc.)
- learning_engine: Self-learning battery performance tracking
- load_profile: Statistical load forecasting by time-of-day
- price_service: Nord Pool electricity price fetching
- direct_control: Direct inverter control via set_wit_mode service
- timezone_utils: Timezone-aware datetime comparison and slot alignment
- ha_helpers: Home Assistant state reading helpers
- cost_tracker: Battery cost tracking with weighted average calculations
- charge_rate_utils: Temperature-aware charge rate computation
- schedule_formatter: Schedule logging and formatting for display
- pv_forecast_service: PV forecast fetching (Solcast / Forecast.Solar)
"""

from .config import BatteryOptimizerConfig
from .models import (
    BatteryMode,
    PricePoint,
    ScheduleEntry,
    LearningStats,
    LoadProfileStats,
    PredictionAccuracyStats,
)
from .pv_profile import PvProfile
from .learning_engine import BatteryLearningEngine
from .load_profile import LoadProfile, _quantile
from .price_service import NordPoolPriceService
from .direct_control import DirectControl
from .dp_optimizer import DPOptimizer, DPOptimizerConfig, DPOptimizerResult
from .timezone_utils import (
    normalize_tz_pair,
    datetimes_match_slot,
    instant_key,
    canonical_slot_key,
    dt_ge,
    ensure_local_tz,
    align_to_slot,
    next_slot_time,
    next_interval_time,
)
from .ha_helpers import SensorReader
from .cost_tracker import BatteryCostTracker, BatteryCostConfig
from .charge_rate_utils import compute_charge_rates_per_slot
from .soc_deviation import SocDeviationDetector, SocDeviationConfig
from .schedule_formatter import ScheduleFormatter, ScheduleFormatterConfig
from .load_prediction_tracker import LoadPredictionTracker
from .pv_forecast_service import PvForecastService, PvForecastServiceConfig

__all__ = [
    # Config
    "BatteryOptimizerConfig",
    # Models
    "BatteryMode",
    "PricePoint",
    "ScheduleEntry",
    "LearningStats",
    "LoadProfileStats",
    "PredictionAccuracyStats",
    "PvProfile",
    # Classes
    "BatteryLearningEngine",
    "LoadProfile",
    "NordPoolPriceService",
    "DirectControl",
    "SensorReader",
    # DP Optimizer
    "DPOptimizer",
    "DPOptimizerConfig",
    "DPOptimizerResult",
    # Timezone utilities
    "normalize_tz_pair",
    "datetimes_match_slot",
    "instant_key",
    "canonical_slot_key",
    "dt_ge",
    "ensure_local_tz",
    "align_to_slot",
    "next_slot_time",
    "next_interval_time",
    # Functions
    "_quantile",
    # Cost tracker
    "BatteryCostTracker",
    "BatteryCostConfig",
    # Charge rate utilities
    "compute_charge_rates_per_slot",
    # SOC deviation detection
    "SocDeviationDetector",
    "SocDeviationConfig",
    # Schedule formatting
    "ScheduleFormatter",
    "ScheduleFormatterConfig",
    # Prediction tracker
    "LoadPredictionTracker",
    # PV forecast service
    "PvForecastService",
    "PvForecastServiceConfig",
]
