"""
Configuration dataclass for BatteryOptimizer.

Centralizes all configuration with type hints, defaults, and validation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BatteryOptimizerConfig:
    """
    Configuration for the Battery Optimizer AppDaemon app.

    All fields have sensible defaults matching the original apps.yaml defaults.
    Use `from_args()` to load from AppDaemon's args dict.
    """

    # =========================================================================
    # Nord Pool Configuration
    # =========================================================================
    nordpool_config_entry: str = ""  # For built-in HA integration (from diagnostics)
    nordpool_area: str = "LV"
    nordpool_sensor: str = "sensor.nord_pool_lv_current_price"  # For HACS component
    tomorrow_prices_hour: int = 14  # Hour when tomorrow's prices become available (local time)

    # =========================================================================
    # Home Assistant Connection
    # =========================================================================
    ha_url: str = ""
    ha_token: str = ""

    # =========================================================================
    # Sensor Entities
    # =========================================================================
    soc_sensor: str = "sensor.growatt_battery_soc"
    pv_power_sensor: str = "sensor.growatt_pv_power"
    battery_temp_sensor: str = ""
    battery_charge_sensor: str = "sensor.growatt_battery_charge_today"
    battery_discharge_sensor: str = "sensor.growatt_battery_discharge_today"
    use_inverter_energy_sensors: bool = True
    load_power_sensor: str = ""

    # =========================================================================
    # Device Control
    # =========================================================================
    device_id: str = ""  # Empty = dry-run mode (no inverter control)

    # =========================================================================
    # Direct Control Settings
    # =========================================================================
    direct_control_buffer_minutes: int = 5
    # Buffer added to slot_minutes for override duration.
    # slot=15 + buffer=5 = 20 min override. If optimizer misses a refresh,
    # inverter reverts to safe base mode after 20 min.

    default_power_percent: int = 100
    # Default charge/discharge power when not specified per-slot.

    # =========================================================================
    # Battery Parameters
    # =========================================================================
    battery_capacity: float = 14.3  # kWh
    charge_rate: float = 4.5  # kW
    discharge_rate: float = 4.5  # kW (from_args defaults to charge_rate if not specified)
    export_discharge_rate: float = 0.0  # kW — discharge rate during grid export (0 = use discharge_rate)
    efficiency: float = 0.85
    base_consumption: float = 500.0  # W (fallback when no load profile)

    # =========================================================================
    # Scheduling Resolution
    # =========================================================================
    slot_minutes: int = 15
    adaptive_recalc_minutes: int = 15
    load_observation_minutes: int = 15
    soc_step_percent: float = 0.25  # DP resolution for SOC (must be < load/slot in kWh)

    # =========================================================================
    # Load Profile
    # =========================================================================
    load_quantile: float = 0.75
    load_profile_entity: str = "input_text.battery_load_profile"
    load_profile_max_samples: int = 60
    load_profile_min_samples: int = 6
    load_zero_floor_w: float = 450.0
    load_profile_file: str = "/config/load_profile.json"
    prediction_tracker_file: str = "/config/prediction_tracker.json"
    load_profile_last_obs_entity: str = "sensor.load_profile_last_observation"
    load_profile_count_entity: str = "sensor.load_profile_observation_count"

    # =========================================================================
    # PV Profile
    # =========================================================================
    pv_profile_file: str = "/config/pv_profile.json"
    pv_profile_max_samples: int = 60
    pv_profile_min_samples: int = 6
    pv_quantile: float = 0.5
    pv_forecast_sensor: str = ""  # Optional external PV forecast sensor (e.g., Solcast)
    pv_forecast_unit: str = "W"  # Unit of pv_forecast_sensor: "W" or "kW"
    inverter_mode_sensor: str = ""  # Inverter mode sensor for monitoring (e.g., sensor.growatt_wit_inverter_mode)

    # =========================================================================
    # PV Forecast Service (Solcast / Forecast.Solar)
    # =========================================================================
    solcast_today_entity: str = ""  # e.g. sensor.solcast_pv_forecast_forecast_today
    solcast_tomorrow_entity: str = ""  # e.g. sensor.solcast_pv_forecast_forecast_tomorrow
    solcast_estimate_field: str = "pv_estimate"  # pv_estimate, pv_estimate10, pv_estimate90

    forecast_solar_lat: float = 0.0
    forecast_solar_lon: float = 0.0
    forecast_solar_declination: int = 0  # panel tilt degrees
    forecast_solar_azimuth: int = 0  # 0=north, 90=east, 180=south, 270=west
    forecast_solar_kwp: float = 0.0  # peak kW (0 = disabled)
    forecast_solar_api_key: str = ""  # optional paid API key

    pv_forecast_cache_minutes: int = 60  # how often to refresh forecast

    # =========================================================================
    # SOC Limits (defaults - can be overridden by HA entities at runtime)
    # =========================================================================
    default_min_soc: float = 10.0
    default_max_soc: float = 100.0
    default_pv_threshold: float = 500.0  # W
    soc_deviation_threshold: float = 10.0  # % deviation to trigger recalc
    soc_shortfall_recalc_threshold: float = 3.0  # % SOC shortfall to trigger pre-execution recalc

    # =========================================================================
    # Pricing
    # =========================================================================
    grid_fee: float = 0.052  # EUR/kWh — trading margin + distribution fee on purchases
    grid_export_fee: float = 0.02  # EUR/kWh — fixed deduction from spot price when selling
    battery_wear_cost: float = 0.0  # EUR/kWh
    export_rate_multiplier: float = 1.0  # 1.0 = no percentage reduction (deduction is fixed)

    # =========================================================================
    # HA Entities for Dynamic Config
    # =========================================================================
    min_soc_entity: str = "input_number.battery_min_soc"
    max_soc_entity: str = "input_number.battery_max_soc"
    pv_threshold_entity: str = "input_number.battery_pv_threshold"
    battery_cost_entity: str = "input_number.battery_avg_cost"

    # =========================================================================
    # Control Entities
    # =========================================================================
    enabled_entity: str = "input_boolean.battery_optimizer_enabled"
    override_entity: str = "input_boolean.battery_optimizer_override"
    manual_mode_entity: str = "input_select.battery_manual_mode"

    # =========================================================================
    # Persistence
    # =========================================================================
    learning_data_file: str = ""

    # =========================================================================
    # Logging
    # =========================================================================
    decision_log_level: int = 1  # 0=minimal, 1=summary, 2=verbose

    # =========================================================================
    # Derived Properties (computed after initialization)
    # =========================================================================
    slot_hours: float = field(init=False)

    def __post_init__(self):
        """Validate and compute derived values."""
        # Validate slot_minutes
        if self.slot_minutes <= 0 or 1440 % self.slot_minutes != 0:
            self.slot_minutes = 15

        # Validate adaptive_recalc_minutes
        if self.adaptive_recalc_minutes <= 0 or 1440 % self.adaptive_recalc_minutes != 0:
            self.adaptive_recalc_minutes = 15

        # Validate load_observation_minutes
        if self.load_observation_minutes <= 0 or 1440 % self.load_observation_minutes != 0:
            self.load_observation_minutes = 15

        # Validate soc_step_percent
        if self.soc_step_percent <= 0:
            self.soc_step_percent = 1.0

        # Clamp load_quantile to valid range
        self.load_quantile = min(1.0, max(0.0, self.load_quantile))

        # Compute derived values
        self.slot_hours = self.slot_minutes / 60.0

    @property
    def effective_export_discharge_rate(self) -> float:
        """Discharge rate during grid export (kW). Falls back to discharge_rate if not set."""
        return self.export_discharge_rate if self.export_discharge_rate > 0 else self.discharge_rate

    @classmethod
    def from_args(cls, args: dict, log_func=None) -> "BatteryOptimizerConfig":
        """
        Load configuration from AppDaemon args dictionary.

        Args:
            args: The args dict from AppDaemon's apps.yaml
            log_func: Optional logging function for warnings

        Returns:
            Configured BatteryOptimizerConfig instance
        """
        def log_warn(msg):
            if log_func:
                log_func(msg, level="WARNING")

        # Extract discharge_rate with fallback to charge_rate
        charge_rate = float(args.get("charge_rate_kw", 4.5))
        discharge_rate = float(args.get("discharge_rate_kw", charge_rate))
        export_discharge_rate = float(args.get("export_discharge_rate_kw", 0))

        # Extract slot_minutes with validation warning
        slot_minutes = int(args.get("slot_minutes", 15))
        if slot_minutes <= 0 or 1440 % slot_minutes != 0:
            log_warn(f"Invalid slot_minutes={slot_minutes}, falling back to 15")

        # Extract adaptive_recalc_minutes with validation warning
        adaptive_recalc_minutes = int(args.get("adaptive_recalc_minutes", 15))
        if adaptive_recalc_minutes <= 0 or 1440 % adaptive_recalc_minutes != 0:
            log_warn(f"Invalid adaptive_recalc_minutes={adaptive_recalc_minutes}, falling back to 15")

        # Extract load_observation_minutes with validation warning
        load_observation_minutes = int(args.get("load_observation_minutes", adaptive_recalc_minutes))
        if load_observation_minutes <= 0 or 1440 % load_observation_minutes != 0:
            log_warn(f"Invalid load_observation_minutes={load_observation_minutes}, falling back to 15")

        return cls(
            # Nord Pool
            nordpool_config_entry=args.get("nordpool_config_entry", ""),
            nordpool_area=args.get("nordpool_area", "LV"),
            nordpool_sensor=args.get("nordpool_sensor", "sensor.nord_pool_lv_current_price"),
            tomorrow_prices_hour=int(args.get("tomorrow_prices_hour", 14)),

            # HA Connection
            ha_url=args.get("ha_url", ""),
            ha_token=args.get("ha_token", ""),

            # Sensors
            soc_sensor=args.get("soc_sensor", "sensor.growatt_battery_soc"),
            pv_power_sensor=args.get("pv_power_sensor", "sensor.growatt_pv_power"),
            battery_temp_sensor=args.get("battery_temp_sensor", ""),
            battery_charge_sensor=args.get("battery_charge_sensor", "sensor.growatt_battery_charge_today"),
            battery_discharge_sensor=args.get("battery_discharge_sensor", "sensor.growatt_battery_discharge_today"),
            use_inverter_energy_sensors=args.get("use_inverter_energy_sensors", True),
            load_power_sensor=args.get("load_power_sensor", ""),

            # Device Control
            device_id=args.get("device_id", ""),

            # Direct Control
            direct_control_buffer_minutes=int(args.get("direct_control_buffer_minutes", 5)),
            default_power_percent=int(args.get("default_power_percent", 100)),

            # Battery Parameters
            battery_capacity=float(args.get("battery_capacity_kwh", 14.3)),
            charge_rate=charge_rate,
            discharge_rate=discharge_rate,
            export_discharge_rate=export_discharge_rate,
            efficiency=float(args.get("efficiency", 0.85)),
            base_consumption=float(args.get("base_consumption_w", 500)),

            # Scheduling
            slot_minutes=slot_minutes,
            adaptive_recalc_minutes=adaptive_recalc_minutes,
            load_observation_minutes=load_observation_minutes,
            soc_step_percent=float(args.get("soc_step_percent", 1.0)),

            # Load Profile
            load_quantile=float(args.get("load_quantile", 0.75)),
            load_profile_entity=args.get("load_profile_entity", "input_text.battery_load_profile"),
            load_profile_max_samples=int(args.get("load_profile_max_samples", 60)),
            load_profile_min_samples=int(args.get("load_profile_min_samples", 6)),
            load_zero_floor_w=float(args.get("load_zero_floor_w", 450)),
            load_profile_file=args.get("load_profile_file", "/config/load_profile.json"),
            prediction_tracker_file=args.get("prediction_tracker_file", "/config/prediction_tracker.json"),
            load_profile_last_obs_entity=args.get(
                "load_profile_last_observation_entity",
                "sensor.load_profile_last_observation"
            ),
            load_profile_count_entity=args.get(
                "load_profile_observation_count_entity",
                "sensor.load_profile_observation_count"
            ),

            # PV Profile
            pv_profile_file=args.get("pv_profile_file", "/config/pv_profile.json"),
            pv_profile_max_samples=int(args.get("pv_profile_max_samples", 60)),
            pv_profile_min_samples=int(args.get("pv_profile_min_samples", 6)),
            pv_quantile=float(args.get("pv_quantile", 0.5)),
            pv_forecast_sensor=args.get("pv_forecast_sensor", ""),
            pv_forecast_unit=args.get("pv_forecast_unit", "W"),
            inverter_mode_sensor=args.get("inverter_mode_sensor", ""),

            # PV Forecast Service
            solcast_today_entity=args.get("solcast_today_entity", ""),
            solcast_tomorrow_entity=args.get("solcast_tomorrow_entity", ""),
            solcast_estimate_field=args.get("solcast_estimate_field", "pv_estimate"),
            forecast_solar_lat=float(args.get("forecast_solar_lat", 0.0)),
            forecast_solar_lon=float(args.get("forecast_solar_lon", 0.0)),
            forecast_solar_declination=int(args.get("forecast_solar_declination", 0)),
            forecast_solar_azimuth=int(args.get("forecast_solar_azimuth", 0)),
            forecast_solar_kwp=float(args.get("forecast_solar_kwp", 0.0)),
            forecast_solar_api_key=args.get("forecast_solar_api_key", ""),
            pv_forecast_cache_minutes=int(args.get("pv_forecast_cache_minutes", 60)),

            # SOC Limits
            default_min_soc=float(args.get("min_soc", 10)),
            default_max_soc=float(args.get("max_soc", 100)),
            default_pv_threshold=float(args.get("pv_threshold_w", 500)),
            soc_deviation_threshold=float(args.get("soc_deviation_threshold", 10)),
            soc_shortfall_recalc_threshold=float(args.get("soc_shortfall_recalc_threshold", 3.0)),

            # Pricing
            grid_fee=float(args.get("grid_fee_eur_kwh", 0.052)),
            grid_export_fee=float(args.get("grid_export_fee_eur_kwh", 0.02)),
            battery_wear_cost=float(args.get("battery_wear_cost_eur_kwh", 0.0)),
            export_rate_multiplier=float(args.get("export_rate_multiplier", 1.0)),

            # HA Entities
            min_soc_entity=args.get("min_soc_entity", "input_number.battery_min_soc"),
            max_soc_entity=args.get("max_soc_entity", "input_number.battery_max_soc"),
            pv_threshold_entity=args.get("pv_threshold_entity", "input_number.battery_pv_threshold"),
            battery_cost_entity=args.get("battery_cost_entity", "input_number.battery_avg_cost"),

            # Control Entities
            enabled_entity=args.get("enabled_entity", "input_boolean.battery_optimizer_enabled"),
            override_entity=args.get("override_entity", "input_boolean.battery_optimizer_override"),
            manual_mode_entity=args.get("manual_mode_entity", "input_select.battery_manual_mode"),

            # Persistence
            learning_data_file=args.get("learning_data_file", ""),

            # Logging
            decision_log_level=int(args.get("decision_log_level", 1)),
        )

    def log_summary(self, log_func) -> None:
        """Log a summary of the configuration."""
        log_func(
            f"Nord Pool config: config_entry='{self.nordpool_config_entry}', "
            f"area='{self.nordpool_area}', sensor='{self.nordpool_sensor}'"
        )
        log_func(
            f"HA connection: ha_url='{self.ha_url or 'NOT SET'}', "
            f"ha_token={'SET' if self.ha_token else 'NOT SET'}"
        )
        if self.device_id:
            log_func(f"Direct control enabled via growatt_modbus/set_wit_mode (device: {self.device_id})")
        log_func(f"Loaded grid_fee: {self.grid_fee} EUR/kWh")
        log_func(
            f"Config loaded: capacity={self.battery_capacity}kWh, "
            f"charge_rate={self.charge_rate}kW, discharge_rate={self.discharge_rate}kW, "
            f"export_discharge_rate={self.effective_export_discharge_rate}kW, "
            f"efficiency={self.efficiency}, slot={self.slot_minutes}min"
        )
        pv_sources = []
        if self.solcast_today_entity:
            pv_sources.append(f"Solcast({self.solcast_estimate_field})")
        if self.forecast_solar_kwp > 0:
            pv_sources.append(f"Forecast.Solar({self.forecast_solar_kwp}kWp)")
        if pv_sources:
            log_func(f"PV forecast: {' + '.join(pv_sources)}")
