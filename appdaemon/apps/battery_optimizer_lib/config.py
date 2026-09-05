"""
Configuration dataclass for BatteryOptimizer.

Centralizes all configuration with type hints, defaults, and validation.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


# Emitted at startup (config load) and by the DP whenever the deployed
# configuration pins the end-of-horizon value to zero. Kept as one constant so
# the config warning, the DP warning and the tests all assert the same text.
TERMINAL_VALUE_ZERO_NOTICE = (
    "terminal_energy_value_eur_kwh=0 is no-salvage mode: energy still in the "
    "battery at the end of the price horizon is valued at zero, so the last "
    "slots spend it (EXPORT/DISCHARGE until depleted). This is harmless as long "
    "as the daily re-optimization extends the horizon before those slots "
    "execute. The alternative, \"auto\", derives a salvage value from the median "
    "forecast import price and instead risks stranding charge at the horizon "
    "edge and skipping evening slots priced below that median. Neither is "
    "universally correct — pick per installation."
)


# Accepted values for `verify_source`. Anything else falls back to "auto"
# during validation — an unknown value must never quietly turn verification off
# or quietly select the mode sensor.
VERIFY_SOURCES = ("auto", "registers", "mode_sensor", "none")


def _arg(args: dict, key: str, default):
    """``args.get`` that treats a *present but empty* YAML key as absent.

    YAML maps a bare ``verify_source:`` (or a commented-out value) to ``None``,
    not to a missing key, so ``args.get("verify_source", "auto")`` returns
    ``None``. The old call chains then turned that into silence rather than a
    default: ``str(None).strip().lower()`` is ``"none"``, which is a VALID
    ``verify_source`` meaning "never verify", and ``bool(None)`` is ``False``,
    which is the master switch off. A blank line in apps.yaml disabled
    verification with no warning anywhere.

    Numeric knobs fail loudly instead (``float(None)`` raises), but they are
    routed through here too so a blank key behaves the same everywhere: it means
    "use the default".
    """
    value = args.get(key, default)
    return default if value is None else value


_FALSE_STRINGS = frozenset({"", "0", "false", "no", "off", "none", "null"})


def _bool_arg(args: dict, key: str, default: bool) -> bool:
    """``_arg`` for a boolean, honouring the strings YAML can produce.

    ``bool("false")`` is ``True``. AppDaemon's apps.yaml is hand-edited, quoting
    happens, and a key written as ``price_retry_enabled: "false"`` must not turn
    the feature ON. Real booleans and numbers pass through unchanged.
    """
    value = _arg(args, key, default)
    if isinstance(value, str):
        return value.strip().lower() not in _FALSE_STRINGS
    return bool(value)


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

    # Price recovery: a transient fetch failure, or a reply without tomorrow's
    # intervals after `tomorrow_prices_hour`, used to leave the app on an old or
    # absent plan until the next daily optimization. Recovery is a bounded
    # backoff with at most ONE pending retry (see price_horizon.py).
    price_retry_enabled: bool = True
    # Delays for the 1st, 2nd, 3rd attempt; every further attempt waits
    # `price_retry_max_seconds`.
    price_retry_delays_seconds: Tuple[int, ...] = (30, 120, 300)
    price_retry_max_seconds: int = 900
    # How long already-fetched FUTURE intervals may be reused to fill a
    # shortened or failed refresh.
    price_retain_max_age_hours: float = 36.0

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
    # Battery cost source attribution
    # =========================================================================
    # Charging measured while HOLD/DISCHARGE is commanded is normally surplus
    # PV. It is only PV if the sun is actually up: at 05:15 on 2026-09-02, five
    # seconds after a CHARGE -> HOLD transition, the tail of a still-running
    # 20-minute grid_charge command was booked "[inverter, pv]" at 0.0253
    # EUR/kWh and pulled the basis 0.1261 -> 0.1199. Measured PV must clear
    # this floor before a charge may be attributed to PV.
    cost_pv_attribution_min_w: float = 100.0
    # A grid_charge command runs for slot_minutes + direct_control_buffer_minutes
    # at the inverter, so it can still be charging after the app has moved on to
    # the next slot's mode. Measured charging stays attributed to the grid while
    # that command is in force, plus this grace period after a different mode
    # supersedes it (the charge counter lags the command).
    cost_grid_charge_grace_seconds: int = 120

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

    # --- Verify-after-set timing -------------------------------------------
    # The integration recomputes its Inverter Mode sensor on each coordinator
    # poll (~30-60s), so the sensor LAGS a write. At a fixed delay a lagging
    # sensor is indistinguishable from a lost command, which is why both the
    # first check and the post-resend re-check are configurable.
    verify_delay_seconds: int = 90       # first check after a mode was sent
    verify_recheck_seconds: int = 60     # single re-check after a resend
    # Per-call websocket timeout for set_wit_mode. This call is SYNCHRONOUS on
    # the AppDaemon callback thread: every second here blocks every other
    # callback of this app. Keep it just above the handler's normal duration.
    set_wit_mode_timeout_seconds: int = 15
    # Master switch for verify-after-set. Off means no check is scheduled at
    # all, whatever `verify_source` says.
    verify_enabled: bool = True
    # Which source answers "did the mode take?".
    #   "registers"   read 30407-30410 / 30200-30201 back through
    #                 growatt_modbus/get_register_data. RECOMMENDED.
    #   "mode_sensor" compare sensor.growatt_inverter_mode (inverter_mode_sensor).
    #                 Only correct where that entity is not frozen by the
    #                 integration's never-cleared _failed_optional_holding_addrs
    #                 blacklist — on 2026-09-01T03:46:34Z one transient read
    #                 failure froze it at "Passthrough" indefinitely, and the
    #                 2026-09-02 log then carried 73/73 false mismatches, each
    #                 costing a blocking ~10 s resend plus a re-check on the
    #                 single AppDaemon thread.
    #   "none"        no verification.
    #   "auto"        registers when device_id is set, otherwise none. It
    #                 deliberately never falls back to the mode sensor: that
    #                 choice must be explicit.
    verify_source: str = "auto"

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
    pv_reactive_threshold: float = 0.5  # Recalc if actual PV < this fraction of forecast
    # Gate for BOTH the reactive shortfall check and the sliding bias window
    # (`PvBiasConfig.min_forecast_kw`). Sunrise/sunset ramp slots must stay out
    # of it: on 2026-09-02 two dawn slots forecast at 292 W measured 0 W and
    # their ratio-0.0 pair alone slammed the horizon bias onto the 0.20 clamp.
    # A few minutes of ramp-timing error is a ~100 % relative error there while
    # being economically meaningless — below the site's baseline load the DP's
    # net load `max(0, load - pv)` barely moves. 600 W is above the ramp and
    # roughly one baseline house load, yet still ~12 % of a 5 kW array's peak,
    # so genuine daytime cloud events are unaffected.
    pv_reactive_min_forecast_w: float = 600.0  # Only check PV shortfall when forecast > this (W)
    # A shortfall is measured on COMPLETED slots from the mean of many samples
    # (i.e. slot energy), never from a single instantaneous reading.
    pv_reactive_consecutive_slots: int = 2  # Consecutive shortfall slots before a full recalc
    pv_reactive_min_samples: int = 3  # Min samples in a slot before its mean is trusted
    pv_sample_seconds: int = 60  # PV power sampling interval (s)
    inverter_mode_sensor: str = ""  # Integration "Inverter Mode" sensor. The
    # entity id depends on the config entry name (slugified "<entry> Inverter
    # Mode"): e.g. sensor.growatt_inverter_mode (entry "Growatt") or
    # sensor.growatt_wit_inverter_mode (entry "Growatt WIT").
    #
    # TWO INDEPENDENT CONSUMERS, and only one of them is verification:
    #
    #   1. MODE-COMPLIANCE HISTORY. The orchestrator's `_get_inverter_mode`
    #      reads this entity and passes it to
    #      `SlotOutcomeTracker.record_slot_end(actual_mode=...)`, which is the
    #      only record of what the inverter actually did per slot. Leaving this
    #      empty silently loses that history — `mode_compliance` degrades to
    #      "unknown" for every slot — even when verification is working
    #      perfectly through the registers.
    #   2. VERIFY-AFTER-SET, but ONLY when `verify_source: mode_sensor` is
    #      selected. `verify_source` decides the verification strategy; this
    #      entity does not. With `verify_source: registers` (the recommendation,
    #      and what `auto` picks whenever device_id is set) verification never
    #      touches this sensor, so setting it costs nothing and buys the
    #      compliance history.
    #
    # Empty disables BOTH: there is no fallback entity any more. The old default
    # guessed sensor.growatt_inverter_mode and fed it to a mode-sensor
    # verification that was not opt-in, which on the reference installation
    # reported the inverter's base work mode rather than the active override:
    # 73/73 verifications logged a 'Passthrough' mismatch while the battery
    # followed every command, each one paying for a blocking resend. That was a
    # defect of the verification default, not of the entity — keep the entity
    # set for monitoring and leave verification on the registers.

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
    # Retry interval after a FAILED provider fetch. Much shorter than the cache
    # TTL: the cache-age guard keys off the last SUCCESS, so a provider that is
    # down would otherwise be re-tried by every optimize / adaptive /
    # PV-shortfall pass (each a blocking HTTP call on a callback thread), while
    # a transient failure must still recover within minutes.
    pv_forecast_failure_retry_minutes: int = 10

    # =========================================================================
    # PV Forecast Bias
    # =========================================================================
    # Sliding median of measured/forecast PV over the last window, clamped and
    # applied to the CURRENT AND REMAINING horizon (not just one slot).
    pv_bias_enabled: bool = True
    pv_bias_window_minutes: int = 120
    pv_bias_min_slots: int = 2
    pv_bias_min_factor: float = 0.2
    pv_bias_max_factor: float = 1.5
    pv_bias_decay_slots: int = 8  # slots without fresh data to relax back to 1.0
    # Across a local-day boundary the bias is attenuated: today's cloud cover is
    # weather, not a calibration error of the provider's model for tomorrow.
    # Each further day keeps `weight ** days` of the deviation from 1.0, and a
    # separate (looser) clamp floors the result for those slots.
    pv_bias_next_day_weight: float = 0.5
    pv_bias_next_day_min_factor: float = 0.7

    # =========================================================================
    # Ambient Temperature / Thermal Model
    # =========================================================================
    # Ambient must be a function of TIME across the horizon. Without an external
    # source it was estimated as min(recent battery temps) — in summer that is
    # ~the current battery temperature, so the projected trajectory was flat.
    ambient_weather_entity: str = ""   # e.g. weather.forecast_home (hourly forecast)
    outdoor_temp_sensor: str = ""      # preferred if the battery is indoors
    ambient_diurnal_amplitude_c: float = 4.0   # half the peak-to-peak daily swing
    ambient_diurnal_peak_hour: float = 15.0    # local hour of the daily maximum
    ambient_forecast_cache_minutes: int = 60
    # Retry interval after a FAILED weather-forecast fetch (same semantics as
    # pv_forecast_failure_retry_minutes). Backing a failure off for the full
    # cache interval left T_ambient(t) on the diurnal fallback for an hour after
    # a restart that raced the HA weather integration.
    ambient_forecast_failure_retry_minutes: int = 10
    # Thermal model fallbacks (used until enough samples are collected)
    thermal_default_cooling_rate_per_min: float = 0.012
    thermal_default_heating_c_per_kwh: float = 0.35

    # =========================================================================
    # SOC Limits (defaults - can be overridden by HA entities at runtime)
    # =========================================================================
    default_min_soc: float = 10.0
    default_max_soc: float = 100.0
    default_pv_threshold: float = 500.0  # W
    soc_deviation_threshold: float = 10.0  # % deviation to trigger recalc
    soc_shortfall_recalc_threshold: float = 3.0  # % SOC shortfall to trigger pre-execution recalc
    # Reaching min_soc is only "unexpected" when the plan did not ask for it.
    # An "(until depleted)" EXPORT/DISCHARGE slot ends AT min_soc by design, so
    # a planned end SOC within this margin of min_soc suppresses the depletion
    # re-optimization (the safety HOLD is still applied).
    planned_depletion_margin_percent: float = 1.0

    # =========================================================================
    # Execution
    # =========================================================================
    # A recalculation applies the current slot itself; the quarter-hour timer
    # then fired seconds later and applied the identical entry again (production
    # 2026-09-02 07:30:06 -> 07:30:12). DirectControl's duplicate suppression
    # absorbed it, but each repeat is another blocking set_wit_mode on the
    # single AppDaemon thread. The TIMER call is skipped when the same slot was
    # already applied this recently; internal calls (recalc, override resume,
    # manual "Auto") are never skipped. 0 disables.
    execute_dedupe_seconds: int = 60

    # =========================================================================
    # Pricing
    # =========================================================================
    grid_fee: float = 0.052  # EUR/kWh — trading margin + distribution fee on purchases
    grid_export_fee: float = 0.02  # EUR/kWh — fixed deduction from spot price when selling
    battery_wear_cost: float = 0.0  # EUR/kWh
    export_rate_multiplier: float = 1.0  # 1.0 = no percentage reduction (deduction is fixed)
    inverter_efficiency: float = 1.0  # AC↔DC conversion efficiency (e.g., 0.97 for 97%)
    import_price_multiplier: float = 1.0  # e.g. 1.21 when spot + variable fees exclude Latvian VAT
    # None derives a terminal value from the median forecast import price. A
    # numeric value is an explicit EUR/kWh value for energy left in the battery.
    terminal_energy_value_eur_kwh: Optional[float] = None
    # Set by from_args() when it has already emitted TERMINAL_VALUE_ZERO_NOTICE,
    # so log_summary() does not repeat the same 600-character paragraph 4 ms
    # later during the SAME initialization (production 2026-09-02 01:59:31.084
    # and .088). CLAUDE.md requires the mode to be STATED once at config load —
    # once, not twice. A config built directly (tests, embedders) still gets the
    # notice from log_summary, because nothing announced it earlier.
    # init=False: not a configuration knob, and it must never appear in
    # from_args()'s keyword list.
    terminal_zero_notice_emitted: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    # =========================================================================
    # HA Entities for Dynamic Config
    # =========================================================================
    min_soc_entity: str = "input_number.battery_min_soc"
    max_soc_entity: str = "input_number.battery_max_soc"
    pv_threshold_entity: str = "input_number.battery_pv_threshold"
    battery_cost_entity: str = "input_number.battery_avg_cost"
    battery_cost_basis_version_entity: str = "input_number.battery_cost_basis_version"

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
    # AppDaemon serializes an app's callbacks on its worker thread(s) and warns
    # at 10s by default ("Excessive time spent in callback"). We measure the
    # same thing locally so the log names the offending callback and can point
    # at total_threads.
    callback_warn_seconds: float = 10.0

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

        # Execution / cost attribution guards
        self.execute_dedupe_seconds = max(
            0, min(int(self.execute_dedupe_seconds), self.slot_minutes * 30)
        )
        self.planned_depletion_margin_percent = max(
            0.0, float(self.planned_depletion_margin_percent)
        )
        self.cost_pv_attribution_min_w = max(0.0, float(self.cost_pv_attribution_min_w))
        self.cost_grid_charge_grace_seconds = max(
            0, int(self.cost_grid_charge_grace_seconds)
        )

        # Reactive PV shortfall detection
        self.pv_reactive_min_forecast_w = max(0.0, float(self.pv_reactive_min_forecast_w))
        self.pv_reactive_consecutive_slots = max(1, int(self.pv_reactive_consecutive_slots))
        self.pv_reactive_min_samples = max(1, int(self.pv_reactive_min_samples))
        # A sample interval longer than a slot could never produce a usable mean
        self.pv_sample_seconds = max(
            10, min(self.slot_minutes * 60, int(self.pv_sample_seconds))
        )

        # PV forecast bias
        self.pv_bias_window_minutes = max(self.slot_minutes, int(self.pv_bias_window_minutes))
        self.pv_bias_min_slots = max(1, int(self.pv_bias_min_slots))
        self.pv_bias_min_factor = max(0.0, min(1.0, float(self.pv_bias_min_factor)))
        self.pv_bias_max_factor = max(
            self.pv_bias_min_factor + 0.01, float(self.pv_bias_max_factor)
        )
        self.pv_bias_decay_slots = max(1, int(self.pv_bias_decay_slots))
        self.pv_bias_next_day_weight = max(
            0.0, min(1.0, float(self.pv_bias_next_day_weight))
        )
        # Never tighter than the same-day clamp — attenuation must move the
        # factor TOWARD 1.0, never further away from it.
        self.pv_bias_next_day_min_factor = max(
            self.pv_bias_min_factor,
            min(1.0, float(self.pv_bias_next_day_min_factor)),
        )

        # Ambient / thermal model
        self.ambient_diurnal_amplitude_c = max(0.0, float(self.ambient_diurnal_amplitude_c))
        self.ambient_diurnal_peak_hour = float(self.ambient_diurnal_peak_hour) % 24.0
        self.ambient_forecast_cache_minutes = max(1, int(self.ambient_forecast_cache_minutes))
        # A retry interval above the cache TTL would be a no-op back-off.
        self.pv_forecast_cache_minutes = max(1, int(self.pv_forecast_cache_minutes))
        self.pv_forecast_failure_retry_minutes = max(
            1,
            min(
                int(self.pv_forecast_failure_retry_minutes),
                self.pv_forecast_cache_minutes,
            ),
        )
        self.ambient_forecast_failure_retry_minutes = max(
            1,
            min(
                int(self.ambient_forecast_failure_retry_minutes),
                self.ambient_forecast_cache_minutes,
            ),
        )
        self.thermal_default_cooling_rate_per_min = min(
            0.1, max(0.001, float(self.thermal_default_cooling_rate_per_min))
        )
        self.thermal_default_heating_c_per_kwh = min(
            2.0, max(0.0, float(self.thermal_default_heating_c_per_kwh))
        )

        # Inverter control timing / blocking
        self.verify_delay_seconds = max(5, min(600, int(self.verify_delay_seconds)))
        self.verify_recheck_seconds = max(5, min(600, int(self.verify_recheck_seconds)))
        self.set_wit_mode_timeout_seconds = max(
            5, min(120, int(self.set_wit_mode_timeout_seconds))
        )
        # An unrecognised verify_source must not silently disable verification
        # (nor silently pick the mode sensor). Fall back to "auto", which reads
        # the registers whenever a device_id exists.
        source = str(self.verify_source or "auto").strip().lower()
        if source not in VERIFY_SOURCES:
            source = "auto"
        self.verify_source = source
        self.callback_warn_seconds = max(1.0, min(60.0, float(self.callback_warn_seconds)))

        # Price recovery backoff. A zero/negative first delay would turn the
        # bounded retry into a busy loop against the price service, so every
        # step is floored at 5 s and capped by `price_retry_max_seconds`.
        self.price_retry_enabled = bool(self.price_retry_enabled)
        self.price_retry_max_seconds = max(
            30, min(3600, int(self.price_retry_max_seconds))
        )
        delays = tuple(
            max(5, min(self.price_retry_max_seconds, int(d)))
            for d in (self.price_retry_delays_seconds or ())
        )
        self.price_retry_delays_seconds = delays or (
            min(30, self.price_retry_max_seconds),
        )
        self.price_retain_max_age_hours = max(
            0.0, min(168.0, float(self.price_retain_max_age_hours))
        )

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

        def log_info(msg):
            # log_func is documented as optional, so every call site must be
            # guarded — not just the warnings.
            if log_func:
                log_func(msg)

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

        # Price-recovery backoff. Accepts a YAML list, a comma-separated string
        # or a single number; a blank key means "use the default".
        def parse_delays(raw, default):
            if raw is None:
                return default
            if isinstance(raw, (list, tuple)):
                values = list(raw)
            elif isinstance(raw, str):
                values = [part for part in raw.replace(";", ",").split(",") if part.strip()]
            else:
                values = [raw]
            parsed = []
            for value in values:
                try:
                    parsed.append(int(float(str(value).strip())))
                except (TypeError, ValueError):
                    log_warn(f"Ignoring invalid price_retry_delays_seconds entry: {value!r}")
            if not parsed:
                log_warn(
                    "price_retry_delays_seconds had no usable values, falling back "
                    f"to {list(default)}"
                )
                return default
            return tuple(parsed)

        price_retry_delays = parse_delays(
            args.get("price_retry_delays_seconds"), (30, 120, 300)
        )

        terminal_zero_notice_emitted = False
        terminal_value_raw = args.get("terminal_energy_value_eur_kwh", "auto")
        if terminal_value_raw is None or str(terminal_value_raw).strip().lower() == "auto":
            terminal_energy_value = None
        else:
            terminal_energy_value = max(0.0, float(terminal_value_raw))
            if terminal_energy_value == 0.0:
                log_info(TERMINAL_VALUE_ZERO_NOTICE)
                # Only a notice that was actually PRINTED suppresses the one in
                # log_summary. With no logger nothing was stated, so the summary
                # must still state it.
                terminal_zero_notice_emitted = log_func is not None

        config = cls(
            # Nord Pool
            nordpool_config_entry=args.get("nordpool_config_entry", ""),
            nordpool_area=args.get("nordpool_area", "LV"),
            nordpool_sensor=args.get("nordpool_sensor", "sensor.nord_pool_lv_current_price"),
            tomorrow_prices_hour=int(args.get("tomorrow_prices_hour", 14)),
            price_retry_enabled=_bool_arg(args, "price_retry_enabled", True),
            price_retry_delays_seconds=price_retry_delays,
            price_retry_max_seconds=int(_arg(args, "price_retry_max_seconds", 900)),
            price_retain_max_age_hours=float(
                _arg(args, "price_retain_max_age_hours", 36.0)
            ),

            # HA Connection
            ha_url=args.get("ha_url", ""),
            ha_token=args.get("ha_token", ""),

            # Sensors
            soc_sensor=args.get("soc_sensor", "sensor.growatt_battery_soc"),
            pv_power_sensor=args.get("pv_power_sensor", "sensor.growatt_pv_power"),
            battery_temp_sensor=args.get("battery_temp_sensor", ""),
            battery_charge_sensor=args.get("battery_charge_sensor", "sensor.growatt_battery_charge_today"),
            battery_discharge_sensor=args.get("battery_discharge_sensor", "sensor.growatt_battery_discharge_today"),
            use_inverter_energy_sensors=bool(
                _arg(args, "use_inverter_energy_sensors", True)
            ),
            cost_pv_attribution_min_w=float(
                _arg(args, "cost_pv_attribution_min_w", 100.0)
            ),
            cost_grid_charge_grace_seconds=int(
                _arg(args, "cost_grid_charge_grace_seconds", 120)
            ),
            load_power_sensor=args.get("load_power_sensor", ""),

            # Device Control
            device_id=args.get("device_id", ""),

            # Direct Control
            direct_control_buffer_minutes=int(args.get("direct_control_buffer_minutes", 5)),
            default_power_percent=int(args.get("default_power_percent", 100)),
            verify_delay_seconds=int(_arg(args, "verify_delay_seconds", 90)),
            verify_recheck_seconds=int(_arg(args, "verify_recheck_seconds", 60)),
            # A blank `verify_enabled:` must mean "default on", not off, and a
            # blank `verify_source:` must mean "auto", not the literal "none"
            # that str(None) produces. See `_arg`.
            verify_enabled=bool(_arg(args, "verify_enabled", True)),
            verify_source=str(_arg(args, "verify_source", "auto")).strip().lower(),
            set_wit_mode_timeout_seconds=int(
                _arg(args, "set_wit_mode_timeout_seconds", 15)
            ),

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
            pv_reactive_threshold=float(args.get("pv_reactive_threshold", 0.5)),
            pv_reactive_min_forecast_w=float(
                _arg(args, "pv_reactive_min_forecast_w", 600.0)
            ),
            pv_reactive_consecutive_slots=max(
                1, int(args.get("pv_reactive_consecutive_slots", 2))
            ),
            pv_reactive_min_samples=max(1, int(args.get("pv_reactive_min_samples", 3))),
            pv_sample_seconds=int(args.get("pv_sample_seconds", 60)),
            inverter_mode_sensor=_arg(args, "inverter_mode_sensor", ""),

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
            pv_forecast_failure_retry_minutes=int(
                args.get("pv_forecast_failure_retry_minutes", 10)
            ),

            # PV Forecast Bias
            pv_bias_enabled=bool(args.get("pv_bias_enabled", True)),
            pv_bias_window_minutes=int(args.get("pv_bias_window_minutes", 120)),
            pv_bias_min_slots=int(args.get("pv_bias_min_slots", 2)),
            pv_bias_min_factor=float(args.get("pv_bias_min_factor", 0.2)),
            pv_bias_max_factor=float(args.get("pv_bias_max_factor", 1.5)),
            pv_bias_decay_slots=int(args.get("pv_bias_decay_slots", 8)),
            pv_bias_next_day_weight=float(args.get("pv_bias_next_day_weight", 0.5)),
            pv_bias_next_day_min_factor=float(
                args.get("pv_bias_next_day_min_factor", 0.7)
            ),

            # Ambient / thermal model
            ambient_weather_entity=args.get("ambient_weather_entity", ""),
            outdoor_temp_sensor=args.get("outdoor_temp_sensor", ""),
            ambient_diurnal_amplitude_c=float(args.get("ambient_diurnal_amplitude_c", 4.0)),
            ambient_diurnal_peak_hour=float(args.get("ambient_diurnal_peak_hour", 15.0)),
            ambient_forecast_cache_minutes=int(
                args.get("ambient_forecast_cache_minutes", 60)
            ),
            ambient_forecast_failure_retry_minutes=int(
                args.get("ambient_forecast_failure_retry_minutes", 10)
            ),
            thermal_default_cooling_rate_per_min=float(
                args.get("thermal_default_cooling_rate_per_min", 0.012)
            ),
            thermal_default_heating_c_per_kwh=float(
                args.get("thermal_default_heating_c_per_kwh", 0.35)
            ),

            # SOC Limits
            default_min_soc=float(args.get("min_soc", 10)),
            default_max_soc=float(args.get("max_soc", 100)),
            default_pv_threshold=float(args.get("pv_threshold_w", 500)),
            soc_deviation_threshold=float(args.get("soc_deviation_threshold", 10)),
            soc_shortfall_recalc_threshold=float(args.get("soc_shortfall_recalc_threshold", 3.0)),
            planned_depletion_margin_percent=float(
                _arg(args, "planned_depletion_margin_percent", 1.0)
            ),
            execute_dedupe_seconds=int(_arg(args, "execute_dedupe_seconds", 60)),

            # Pricing
            grid_fee=float(args.get("grid_fee_eur_kwh", 0.052)),
            grid_export_fee=float(args.get("grid_export_fee_eur_kwh", 0.02)),
            battery_wear_cost=float(args.get("battery_wear_cost_eur_kwh", 0.0)),
            export_rate_multiplier=float(args.get("export_rate_multiplier", 1.0)),
            inverter_efficiency=float(args.get("inverter_efficiency", 1.0)),
            import_price_multiplier=float(args.get("import_price_multiplier", 1.0)),
            terminal_energy_value_eur_kwh=terminal_energy_value,

            # HA Entities
            min_soc_entity=args.get("min_soc_entity", "input_number.battery_min_soc"),
            max_soc_entity=args.get("max_soc_entity", "input_number.battery_max_soc"),
            pv_threshold_entity=args.get("pv_threshold_entity", "input_number.battery_pv_threshold"),
            battery_cost_entity=args.get("battery_cost_entity", "input_number.battery_avg_cost"),
            battery_cost_basis_version_entity=args.get(
                "battery_cost_basis_version_entity",
                "input_number.battery_cost_basis_version",
            ),

            # Control Entities
            enabled_entity=args.get("enabled_entity", "input_boolean.battery_optimizer_enabled"),
            override_entity=args.get("override_entity", "input_boolean.battery_optimizer_override"),
            manual_mode_entity=args.get("manual_mode_entity", "input_select.battery_manual_mode"),

            # Persistence
            learning_data_file=args.get("learning_data_file", ""),

            # Logging
            decision_log_level=int(args.get("decision_log_level", 1)),
            callback_warn_seconds=float(_arg(args, "callback_warn_seconds", 10.0)),
        )
        config.terminal_zero_notice_emitted = terminal_zero_notice_emitted
        return config

    def _verification_summary(self) -> str:
        """How verify-after-set will actually behave, in one clause.

        Named for the SOURCE, not just on/off: the 2026-09-02 log showed 73
        mismatches with no way to tell from the startup lines WHAT was being
        compared. "registers" and "mode_sensor" fail in completely different
        ways, so the log has to say which one is armed.
        """
        if not self.verify_enabled:
            return " - verification DISABLED"
        source = self.verify_source
        if source == "none":
            return " - verification DISABLED"
        if source == "mode_sensor":
            if not self.inverter_mode_sensor:
                return " - verification DISABLED (no inverter_mode_sensor)"
            return f" - verification via {self.inverter_mode_sensor}"
        if source == "registers" or (source == "auto" and self.device_id):
            return (
                " - verification via registers 30407-30410/30200-30201"
                if self.device_id
                else " - verification DISABLED (dry run, no device_id)"
            )
        return " - verification DISABLED (dry run, no device_id)"

    def log_summary(self, log_func, warn_func=None) -> None:
        """Log a summary of the configuration.

        Args:
            log_func: INFO-level logger.
            warn_func: Optional WARNING-level logger. Falls back to log_func so
                existing callers keep working; the degenerate terminal-value
                notice is then prefixed with "WARNING:" instead.
        """
        def warn(msg):
            if warn_func is not None:
                warn_func(msg)
            else:
                log_func(f"WARNING: {msg}")

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
        log_func(
            f"Inverter control timing: set_wit_mode timeout="
            f"{self.set_wit_mode_timeout_seconds}s (blocks the callback thread), "
            f"verify after {self.verify_delay_seconds}s, "
            f"re-check {self.verify_recheck_seconds}s after a resend; "
            f"slow-callback warning at {self.callback_warn_seconds:.0f}s"
            f"; verification source={self.verify_source}"
            f"{self._verification_summary()}"
        )
        if self.price_retry_enabled:
            log_func(
                "Price recovery: tomorrow expected from "
                f"{self.tomorrow_prices_hour:02d}:00 local, retry backoff "
                f"{list(self.price_retry_delays_seconds)}s then "
                f"{self.price_retry_max_seconds}s, cached intervals reused for up "
                f"to {self.price_retain_max_age_hours:.0f}h"
            )
        else:
            warn(
                "Price recovery DISABLED (price_retry_enabled=false): a failed or "
                "incomplete price fetch will not be retried before the next daily "
                "optimization"
            )
        log_func(f"Loaded grid_fee: {self.grid_fee} EUR/kWh")
        if self.terminal_energy_value_eur_kwh is None:
            log_func(
                "Terminal energy value: auto (derived from the median forecast "
                "import price, discharge conversion and wear)"
            )
        elif self.terminal_energy_value_eur_kwh == 0.0:
            # STATED once per initialization, not twice: from_args() already
            # emitted it a few milliseconds ago when it parsed the value.
            if self.terminal_zero_notice_emitted:
                log_func(
                    "Terminal energy value: 0 EUR/kWh (no-salvage mode, see the "
                    "notice above)"
                )
            else:
                log_func(TERMINAL_VALUE_ZERO_NOTICE)
                self.terminal_zero_notice_emitted = True
        else:
            log_func(
                f"Terminal energy value: "
                f"{self.terminal_energy_value_eur_kwh:.4f} EUR/kWh (configured)"
            )
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
        log_func(
            f"PV bias: enabled={self.pv_bias_enabled}, "
            f"window={self.pv_bias_window_minutes}min, "
            f"clamp=[{self.pv_bias_min_factor}, {self.pv_bias_max_factor}], "
            f"min_slots={self.pv_bias_min_slots}, decay={self.pv_bias_decay_slots} slots, "
            f"next-day weight={self.pv_bias_next_day_weight} "
            f"floor={self.pv_bias_next_day_min_factor}; "
            f"reactive: threshold={self.pv_reactive_threshold}, "
            f"consecutive_slots={self.pv_reactive_consecutive_slots}, "
            f"min_samples={self.pv_reactive_min_samples}, "
            f"min_forecast={self.pv_reactive_min_forecast_w:.0f}W, "
            f"sample_every={self.pv_sample_seconds}s"
        )
        log_func(
            f"Cost attribution: charging counts as PV only above "
            f"{self.cost_pv_attribution_min_w:.0f}W measured PV; a grid_charge "
            f"command keeps grid attribution for "
            f"{self.cost_grid_charge_grace_seconds}s after it is superseded. "
            f"Slot re-execution deduped within {self.execute_dedupe_seconds}s; "
            f"planned depletion margin {self.planned_depletion_margin_percent:.1f}%"
        )
        if self.ambient_weather_entity:
            ambient_source = f"weather forecast ({self.ambient_weather_entity})"
        elif self.outdoor_temp_sensor:
            ambient_source = f"outdoor sensor ({self.outdoor_temp_sensor})"
        else:
            ambient_source = "battery min-window heuristic + diurnal profile"
        log_func(
            f"Ambient: source={ambient_source}, "
            f"diurnal amplitude=+-{self.ambient_diurnal_amplitude_c}C "
            f"peaking at {self.ambient_diurnal_peak_hour:.0f}:00; "
            f"thermal defaults k1={self.thermal_default_cooling_rate_per_min}/min, "
            f"k2={self.thermal_default_heating_c_per_kwh}C/kWh"
        )
