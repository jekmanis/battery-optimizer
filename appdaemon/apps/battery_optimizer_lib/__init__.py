"""
Battery Optimizer package.

This package contains helper modules for the main BatteryOptimizer AppDaemon app:
- config: Configuration dataclass with typed fields and validation
- models: Data classes and enums (BatteryMode, PricePoint, ScheduleEntry, etc.)
- learning_engine: Self-learning battery performance tracking
- load_profile: Statistical load forecasting by time-of-day
- price_service: Nord Pool electricity price fetching
- price_horizon: Price coverage health and bounded price-recovery backoff
- direct_control: Direct inverter control via set_wit_mode service
- timezone_utils: Timezone-aware datetime comparison and slot alignment
- ha_helpers: Home Assistant state reading helpers
- cost_tracker: Battery cost tracking with weighted average calculations
- schedule_formatter: Schedule logging and formatting for display
- pv_forecast_service: PV forecast fetching (Solcast / Forecast.Solar)
- pv_bias_tracker: Sliding PV forecast bias estimation and slot-energy sampling
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
from .price_horizon import (
    COVERAGE_REASONS,
    HorizonHealth,
    PriceHorizonConfig,
    PriceHorizonMonitor,
    is_coverage_reason,
)
from .direct_control import (
    ApplyOutcome,
    DirectControl,
    ModeSensorVerifier,
    RegisterVerifier,
    VerificationOutcome,
    VerificationResult,
)
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
    prev_slot_time,
    slot_offset,
    next_interval_time,
    lookup_by_time,
)
from .ha_helpers import SensorReader
from .cost_tracker import BatteryCostTracker, BatteryCostConfig
from .soc_projection import SocProjectionParams, SocTransition, project_slot_soc
from .slot_energy import (
    SlotEnergyParams,
    SlotEnergyResult,
    simulate_slot,
    params_from_soc_projection,
)
from .soc_deviation import SocDeviationDetector, SocDeviationConfig
from .schedule_formatter import ScheduleFormatter, ScheduleFormatterConfig
from .load_prediction_tracker import LoadPredictionTracker
from .pv_forecast_service import PvForecastService, PvForecastServiceConfig
from .pv_bias_tracker import PvBiasTracker, PvBiasConfig, ClosedSlot
from .thermal_model import (
    TemperatureProjector,
    battery_power_for_entry,
    step_temperature,
    DEFAULT_COOLING_RATE_PER_MIN,
    DEFAULT_HEATING_C_PER_KWH,
    MAX_BATTERY_TEMP_C,
)
from .ambient_service import AmbientTemperatureService, AmbientServiceConfig
from .callback_lock import CallbackLock

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
    # Price horizon health / recovery
    "COVERAGE_REASONS",
    "HorizonHealth",
    "PriceHorizonConfig",
    "PriceHorizonMonitor",
    "is_coverage_reason",
    "DirectControl",
    "ApplyOutcome",
    "ModeSensorVerifier",
    "RegisterVerifier",
    "VerificationOutcome",
    "VerificationResult",
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
    "prev_slot_time",
    "slot_offset",
    "next_interval_time",
    "lookup_by_time",
    # Functions
    "_quantile",
    # Cost tracker
    "BatteryCostTracker",
    "BatteryCostConfig",
    # SOC projection (shared slot transition model)
    "SocProjectionParams",
    "SocTransition",
    "project_slot_soc",
    # Slot energy flows (shared pure transition, named units)
    "SlotEnergyParams",
    "SlotEnergyResult",
    "simulate_slot",
    "params_from_soc_projection",
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
    # PV forecast bias
    "PvBiasTracker",
    "PvBiasConfig",
    "ClosedSlot",
    # Thermal model (shared temperature projection)
    "TemperatureProjector",
    "battery_power_for_entry",
    "step_temperature",
    "DEFAULT_COOLING_RATE_PER_MIN",
    "DEFAULT_HEATING_C_PER_KWH",
    "MAX_BATTERY_TEMP_C",
    # Ambient temperature service
    "AmbientTemperatureService",
    "AmbientServiceConfig",
    # Thread safety
    "CallbackLock",
]
