"""
Battery Cost Tracker - Tracks weighted average cost of energy in the battery.

Handles:
- Weighted average cost calculations during charge/discharge
- Dual-source energy tracking (inverter sensors vs SOC-based fallback)
- Midnight reset detection for daily energy counters
- Discharge threshold calculations
- Cost projection through a schedule
"""

import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import BatteryOptimizerConfig
    from .models import BatteryMode, PricePoint, ScheduleEntry
    from .learning_engine import BatteryLearningEngine


@dataclass
class BatteryCostConfig:
    """Configuration for BatteryCostTracker."""

    # HA entity for persisting battery cost
    battery_cost_entity: str = "input_number.battery_avg_cost"

    # Inverter energy sensors
    battery_charge_sensor: str = "sensor.growatt_battery_charge_today"
    battery_discharge_sensor: str = "sensor.growatt_battery_discharge_today"
    use_inverter_energy_sensors: bool = True

    # Battery parameters
    battery_capacity: float = 14.3  # kWh
    efficiency: float = 0.85
    slot_minutes: int = 15
    charge_rate: float = 4.5  # kW
    discharge_rate: float = 4.5  # kW
    export_discharge_rate: float = 0.0  # kW — discharge rate during grid export (0 = use discharge_rate)

    # Pricing
    grid_fee: float = 0.052  # EUR/kWh — trading margin + distribution on purchases
    battery_wear_cost: float = 0.0  # EUR/kWh

    # Default fallback cost when HA entity unavailable
    default_cost: float = 0.10  # EUR/kWh

    @property
    def effective_export_discharge_rate(self) -> float:
        """Discharge rate during grid export (kW). Falls back to discharge_rate if not set."""
        return self.export_discharge_rate if self.export_discharge_rate > 0 else self.discharge_rate

    @classmethod
    def from_main_config(cls, cfg: "BatteryOptimizerConfig") -> "BatteryCostConfig":
        """Create from the central BatteryOptimizerConfig."""
        return cls(
            battery_cost_entity=cfg.battery_cost_entity,
            battery_charge_sensor=cfg.battery_charge_sensor,
            battery_discharge_sensor=cfg.battery_discharge_sensor,
            use_inverter_energy_sensors=cfg.use_inverter_energy_sensors,
            battery_capacity=cfg.battery_capacity,
            efficiency=cfg.efficiency,
            slot_minutes=cfg.slot_minutes,
            charge_rate=cfg.charge_rate,
            discharge_rate=cfg.discharge_rate,
            export_discharge_rate=cfg.export_discharge_rate,
            grid_fee=cfg.grid_fee,
            battery_wear_cost=cfg.battery_wear_cost,
        )


class BatteryCostTracker:
    """
    Tracks the weighted average cost of energy stored in the battery.

    Uses a dual-source approach:
    1. Primary: Inverter energy sensors (more accurate)
    2. Fallback: SOC-based calculations (when sensors unavailable)

    The weighted average cost is updated on every charge event and persisted
    to a Home Assistant entity for survival across restarts.
    """

    def __init__(
        self,
        config: BatteryCostConfig,
        # HA integration functions
        get_state_func: Callable[[str], Optional[str]],
        call_service_func: Callable[..., None],
        # Time functions
        get_datetime_func: Callable[[], datetime.datetime],
        get_timezone_func: Callable[[], Optional[datetime.tzinfo]],
        align_to_slot_func: Callable[[datetime.datetime], datetime.datetime],
        # Dynamic property getters (min/max SOC can change at runtime)
        get_min_soc_func: Callable[[], float],
        get_max_soc_func: Callable[[], float],
        # Sensor reading functions
        get_current_soc_func: Callable[[], Optional[float]],
        get_battery_temp_func: Callable[[], Optional[float]],
        # Learning engine for recording observations
        learning_engine: "BatteryLearningEngine",
        # Price lookup
        get_cached_prices_func: Callable[[], List["PricePoint"]],
        # Callbacks after learning updates
        save_learning_data_func: Callable[[], None],
        update_learning_sensor_func: Callable[[], None],
        # Logging
        log_func: Callable[..., None],
    ):
        self._config = config
        self._get_state = get_state_func
        self._call_service = call_service_func
        self._get_datetime = get_datetime_func
        self._get_timezone = get_timezone_func
        self._align_to_slot = align_to_slot_func
        self._get_min_soc = get_min_soc_func
        self._get_max_soc = get_max_soc_func
        self._get_current_soc = get_current_soc_func
        self._get_battery_temp = get_battery_temp_func
        self._learning_engine = learning_engine
        self._get_cached_prices = get_cached_prices_func
        self._save_learning_data = save_learning_data_func
        self._update_learning_sensor = update_learning_sensor_func
        self._log = log_func

        # Cost tracking state
        self._avg_cost: float = config.default_cost
        self._cost_from_fallback: bool = True

        # SOC tracking state
        self._last_soc: Optional[float] = None
        self._last_soc_time: Optional[datetime.datetime] = None
        self._last_sig_soc: Optional[float] = None
        self._last_sig_soc_time: Optional[datetime.datetime] = None
        self._last_sig_temp: Optional[float] = None
        self._last_price_slot: Optional[datetime.datetime] = None

        # Idle tracking (for cooling rate learning)
        self._idle_start_time: Optional[datetime.datetime] = None
        self._idle_start_temp: Optional[float] = None

        # Inverter energy sensor tracking
        self._last_charge_today_kwh: Optional[float] = None
        self._last_discharge_today_kwh: Optional[float] = None
        self._energy_sensor_available: bool = False
        self._stored_energy_kwh: Optional[float] = None

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def avg_cost(self) -> float:
        """Current weighted average cost of energy in the battery (EUR/kWh)."""
        return self._avg_cost

    @property
    def is_energy_sensor_available(self) -> bool:
        """True if using inverter energy sensors, False if using SOC fallback."""
        return self._energy_sensor_available

    @property
    def last_soc(self) -> Optional[float]:
        """Last recorded SOC value."""
        return self._last_soc

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _soc_to_energy_kwh(self, soc: float) -> float:
        """
        Convert SOC percentage to usable energy in kWh (above min_soc).

        Args:
            soc: State of charge percentage (0-100)

        Returns:
            Usable energy in kWh above the minimum SOC threshold
        """
        return (soc - self._get_min_soc()) / 100 * self._config.battery_capacity

    @staticmethod
    def _compute_weighted_avg_cost(
        old_energy: float, old_avg_cost: float, added_energy: float, added_price: float
    ) -> float:
        """
        Compute the new weighted average cost when adding energy to the battery.

        Calculates: new_avg = (old_energy * old_avg + added_energy * added_price) / total_energy

        Args:
            old_energy: Energy already in battery (kWh)
            old_avg_cost: Current average cost (EUR/kWh)
            added_energy: Energy being added (kWh)
            added_price: Price paid for added energy (EUR/kWh)

        Returns:
            New weighted average cost (EUR/kWh), or old_avg_cost if total energy is 0
        """
        new_total_energy = old_energy + added_energy
        if new_total_energy > 0:
            old_total_cost = old_energy * old_avg_cost
            new_total_cost = old_total_cost + (added_energy * added_price)
            return new_total_cost / new_total_energy
        return old_avg_cost

    def _update_weighted_avg_cost(self, old_energy: float, added_energy: float, added_price: float) -> None:
        """
        Update the weighted average cost when adding energy to the battery.

        Calculates: new_avg = (old_energy * old_avg + added_energy * added_price) / total_energy

        Args:
            old_energy: Energy already in battery (kWh)
            added_energy: Energy being added (kWh)
            added_price: Price paid for added energy (EUR/kWh)
        """
        self._avg_cost = self._compute_weighted_avg_cost(
            old_energy, self._avg_cost, added_energy, added_price
        )

    # =========================================================================
    # Initialization
    # =========================================================================

    def initialize(self) -> bool:
        """
        Initialize the cost tracker.

        Returns True if HA is ready and cost was loaded, False if waiting for HA start.
        """
        # Initialize SOC tracking
        current_soc = self._get_current_soc()
        self._last_soc = current_soc
        self._last_soc_time = self._get_datetime()
        self._last_sig_soc = current_soc
        self._last_sig_soc_time = self._last_soc_time
        self._last_sig_temp = self._get_battery_temp()
        self._last_price_slot = self._align_to_slot(self._get_datetime())

        # Initialize stored energy tracking
        if current_soc is not None:
            self._stored_energy_kwh = max(0.0, self._soc_to_energy_kwh(current_soc))

        # Initialize inverter energy sensor readings
        charge_today, discharge_today = self._get_inverter_energy_readings()
        if charge_today is not None:
            self._last_charge_today_kwh = charge_today
            self._last_discharge_today_kwh = discharge_today
            self._energy_sensor_available = True
            # Recalculate stored energy with current SOC
            if current_soc is not None:
                self._stored_energy_kwh = max(0.0, self._soc_to_energy_kwh(current_soc))
            self._log(f"Initialized energy sensors: charge={charge_today:.2f}, discharge={discharge_today:.2f} kWh")
        elif self._config.use_inverter_energy_sensors:
            self._log("Inverter energy sensors unavailable, will use SOC-based calculation", level="WARNING")

        return True

    def load_from_ha(self) -> bool:
        """
        Load battery cost from HA entity.

        Returns True if loaded successfully, False if using fallback.
        """
        try:
            state = self._get_state(self._config.battery_cost_entity)
            if state and state not in ("unknown", "unavailable"):
                self._avg_cost = float(state)
                self._cost_from_fallback = False
                self._log(f"Loaded battery avg cost: {self._avg_cost:.4f} EUR/kWh")
                return True
        except (ValueError, TypeError) as e:
            self._log(f"Could not parse battery cost: {e}", level="WARNING")

        self._avg_cost = self._config.default_cost
        self._cost_from_fallback = True
        self._log(f"Using default battery cost: {self._avg_cost:.4f} EUR/kWh", level="WARNING")
        return False

    def save_to_ha(self):
        """Persist battery cost to Home Assistant entity."""
        # Don't save if we're still using the fallback value (not computed from real charging)
        if self._cost_from_fallback:
            self._log(f"Skipping save: battery cost {self._avg_cost:.4f} is from fallback, not computed", level="DEBUG")
            return

        try:
            self._call_service(
                "input_number/set_value",
                entity_id=self._config.battery_cost_entity,
                value=round(self._avg_cost, 4)
            )
            self._log(f"Saved battery avg cost to HA: {self._avg_cost:.4f} EUR/kWh", level="DEBUG")
        except Exception as e:
            self._log(f"Could not save battery cost to {self._config.battery_cost_entity}: {e}", level="WARNING")

    # =========================================================================
    # Energy Sensor Handling
    # =========================================================================

    def _get_inverter_energy_readings(self) -> Tuple[Optional[float], Optional[float]]:
        """Read current values from inverter energy sensors."""
        if not self._config.use_inverter_energy_sensors:
            return None, None
        try:
            charge_state = self._get_state(self._config.battery_charge_sensor)
            discharge_state = self._get_state(self._config.battery_discharge_sensor)
            if charge_state in ("unknown", "unavailable", None) or \
               discharge_state in ("unknown", "unavailable", None):
                return None, None
            return float(charge_state), float(discharge_state)
        except (ValueError, TypeError):
            return None, None

    def _is_midnight_reset(self, current: float, previous: float, now: datetime.datetime) -> bool:
        """
        Detect if value drop is due to midnight reset.

        Note: `now` comes from get_datetime() which returns HA's configured timezone
        (local time), matching the inverter's midnight reset behavior.
        """
        if current >= previous:
            return False
        minutes_since_midnight = now.hour * 60 + now.minute
        # Within 5 min of local midnight and current value is small (post-reset)
        return (minutes_since_midnight < 5 or minutes_since_midnight > 1435) and current < 1.0

    def on_energy_sensor_change(
        self,
        entity: str,
        old: Optional[str],
        new: Optional[str]
    ):
        """
        Handle changes to inverter energy sensors.

        This is the PRIMARY trigger for cost tracking and learning when energy sensors are enabled.
        Should be called by the main app's state listener.
        """
        if new in ("unknown", "unavailable", None):
            if self._energy_sensor_available:
                self._log(f"Energy sensor {entity} became unavailable, falling back to SOC-based tracking")
                self._energy_sensor_available = False
                self._stored_energy_kwh = None
            return

        if old in ("unknown", "unavailable", None):
            # Sensor just became available - check if both sensors are now available
            charge_today, discharge_today = self._get_inverter_energy_readings()
            if charge_today is not None:
                self._last_charge_today_kwh = charge_today
                self._last_discharge_today_kwh = discharge_today
                if not self._energy_sensor_available:
                    self._energy_sensor_available = True
                    current_soc = self._get_current_soc()
                    if current_soc is not None:
                        self._stored_energy_kwh = max(0.0, self._soc_to_energy_kwh(current_soc))
                    self._log(f"Energy sensors recovered: charge={charge_today:.2f}, discharge={discharge_today:.2f} kWh")
            return

        try:
            current_value = float(new)
            old_value = float(old)
        except (ValueError, TypeError):
            return

        if not self._energy_sensor_available:
            # Avoid double-counting: SOC fallback handles cost/learning when sensors are unavailable.
            return

        now = self._get_datetime()

        # Detect midnight reset
        if self._is_midnight_reset(current_value, old_value, now):
            self._log(f"Midnight reset on {entity}: {old_value:.2f} -> {current_value:.2f} kWh")
            # Reset tracking for new day
            if entity == self._config.battery_charge_sensor:
                self._last_charge_today_kwh = current_value
            else:
                self._last_discharge_today_kwh = current_value
            return

        # Calculate energy delta
        energy_delta = current_value - old_value
        if energy_delta < 0.05:  # Ignore tiny changes (noise)
            return

        # Determine if this is charge or discharge
        is_charge = (entity == self._config.battery_charge_sensor)

        # Get current SOC for context
        current_soc = self._get_current_soc()
        if current_soc is None:
            return

        # Process the energy change
        self._process_energy_change(
            energy_kwh=energy_delta,
            is_charge=is_charge,
            current_soc=current_soc,
            now=now
        )

    def _process_energy_change(
        self,
        energy_kwh: float,
        is_charge: bool,
        current_soc: float,
        now: datetime.datetime
    ):
        """
        Process an energy change event from inverter sensors.
        Updates battery cost tracking and learning engine.
        """
        current_slot = self._align_to_slot(now)

        # Calculate duration since last significant event
        if self._last_sig_soc_time:
            last_time = self._last_sig_soc_time
            if now.tzinfo is not None and last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=now.tzinfo)
            duration_minutes = (now - last_time).total_seconds() / 60
        else:
            duration_minutes = 1.0

        if self._stored_energy_kwh is None:
            base_soc = self._last_soc if self._last_soc is not None else current_soc
            self._stored_energy_kwh = max(0.0, self._soc_to_energy_kwh(base_soc))

        if is_charge:
            # Get price for charging period
            charge_price = self._get_price_for_slot(self._last_price_slot) if self._last_price_slot else None
            if charge_price is None:
                charge_price = self._avg_cost

            # Update weighted average cost
            # Use stored-energy accumulator to keep base in sync with inverter deltas
            old_energy = self._stored_energy_kwh or 0.0
            self._update_weighted_avg_cost(old_energy, energy_kwh, charge_price)
            new_total_energy = old_energy + energy_kwh
            # Cap stored energy at max usable capacity (max_soc - min_soc)
            max_usable_energy = self._soc_to_energy_kwh(self._get_max_soc())
            self._stored_energy_kwh = max(0.0, min(new_total_energy, max_usable_energy))

            self._log(f"Battery charged: +{energy_kwh:.3f} kWh [inverter] at {charge_price:.4f} EUR/kWh, "
                     f"new avg cost: {self._avg_cost:.4f} EUR/kWh")
            self._cost_from_fallback = False  # We computed a real value
            self.save_to_ha()

            # Feed learning engine with actual measured energy
            battery_temp = self._get_battery_temp()
            self._learning_engine.record_charging(
                soc_start=self._last_soc if self._last_soc else current_soc,
                soc_end=current_soc,
                duration_minutes=duration_minutes,
                energy_from_grid_kwh=energy_kwh / self._config.efficiency,  # Estimate grid energy
                charge_price=charge_price,
                battery_temp=battery_temp,
                battery_temp_start=self._last_sig_temp,
                battery_temp_end=battery_temp,
                energy_to_battery_kwh=energy_kwh  # Actual measured energy from inverter
            )
        else:
            # Discharge
            discharge_price = self._get_price_for_slot(self._last_price_slot) if self._last_price_slot else 0.0
            self._log(f"Battery discharged: -{energy_kwh:.3f} kWh [inverter]")
            if self._stored_energy_kwh is not None:
                self._stored_energy_kwh = max(
                    0.0,
                    self._stored_energy_kwh - energy_kwh
                )

            self._learning_engine.record_discharging(
                soc_start=self._last_soc if self._last_soc else current_soc,
                soc_end=current_soc,
                duration_minutes=duration_minutes,
                energy_delivered_kwh=energy_kwh,
                price_eur_kwh=discharge_price or 0.0
            )

        self._save_learning_data()
        self._update_learning_sensor()

        # Update tracking state
        self._last_soc = current_soc
        self._last_soc_time = now
        self._last_sig_soc = current_soc
        self._last_sig_soc_time = now
        self._last_sig_temp = self._get_battery_temp()
        self._last_price_slot = current_slot

    # =========================================================================
    # SOC-based Tracking (Fallback)
    # =========================================================================

    def process_soc_change(self, current_soc: float):
        """
        Process SOC change for battery cost tracking and learning.

        This is the FALLBACK method when inverter energy sensors are unavailable.
        Should be called by the main app's SOC state listener.
        """
        now = self._get_datetime()
        current_slot = self._align_to_slot(now)

        # Record temperature observation for ambient estimation
        current_temp = self._get_battery_temp()
        if current_temp is not None and self._learning_engine:
            self._learning_engine.record_temperature_observation(current_temp)

        if self._last_soc is None:
            self._last_soc = current_soc
            self._last_soc_time = now
            self._last_sig_soc = current_soc
            self._last_sig_soc_time = now
            self._last_price_slot = current_slot
            return

        soc_change = current_soc - self._last_soc

        # Only process significant changes (> 1%)
        if abs(soc_change) < 1.0:
            self._last_soc = current_soc
            self._last_soc_time = now
            self._last_price_slot = current_slot  # Always update slot to prevent stale pricing
            return

        # If energy sensors are enabled and available, they handle cost/learning processing
        # We only do SOC-based processing as fallback
        if self._energy_sensor_available:
            # Update tracking state, but skip cost/learning (handled by energy sensor listener)
            self._last_soc = current_soc
            self._last_soc_time = now
            self._last_sig_soc = current_soc
            self._last_sig_soc_time = now
            self._last_sig_temp = self._get_battery_temp()
            self._last_price_slot = current_slot
            return

        # Fallback: SOC-based energy calculation (when energy sensors unavailable)
        energy_change_kwh = abs(soc_change) / 100 * self._config.battery_capacity

        # Calculate time since last observation
        if self._last_sig_soc_time:
            # Ensure consistent timezone handling to avoid naive/aware mismatch
            last_time = self._last_sig_soc_time
            compare_now = now
            if compare_now.tzinfo is not None and last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=compare_now.tzinfo)
            elif compare_now.tzinfo is None and last_time.tzinfo is not None:
                compare_now = compare_now.replace(tzinfo=last_time.tzinfo)
            duration_minutes = (compare_now - last_time).total_seconds() / 60
        else:
            duration_minutes = 1.0  # Fallback

        if soc_change > 0:
            # Battery charged - get price for the charging period
            charge_price = self._get_price_for_slot(self._last_price_slot) if self._last_price_slot else None
            if charge_price is None:
                charge_price = self._avg_cost  # Fallback to current avg

            # Calculate energy BEFORE this charge (at old SOC)
            old_energy = max(0, self._soc_to_energy_kwh(self._last_soc))

            # Update weighted average cost
            self._update_weighted_avg_cost(old_energy, energy_change_kwh, charge_price)

            self._log(f"Battery charged: +{soc_change:.1f}% (+{energy_change_kwh:.2f} kWh) at {charge_price:.4f} EUR/kWh, "
                     f"new avg cost: {self._avg_cost:.4f} EUR/kWh")

            self._cost_from_fallback = False  # We computed a real value
            self.save_to_ha()

            # Feed learning engine with charging observation (include battery temp if available)
            battery_temp_end = self._get_battery_temp()
            self._learning_engine.record_charging(
                soc_start=self._last_soc,
                soc_end=current_soc,
                duration_minutes=duration_minutes,
                charge_price=charge_price,
                battery_temp=battery_temp_end,
                battery_temp_start=self._last_sig_temp,
                battery_temp_end=battery_temp_end
            )
            self._save_learning_data()
            self._update_learning_sensor()

        elif soc_change < 0:
            # Battery discharged - cost per kWh stays same, just less energy
            discharge_price = self._get_price_for_slot(self._last_price_slot) if self._last_price_slot else 0.0

            self._log(f"Battery discharged: {soc_change:.1f}% ({energy_change_kwh:.2f} kWh)")

            # Feed learning engine with discharging observation
            self._learning_engine.record_discharging(
                soc_start=self._last_soc,
                soc_end=current_soc,
                duration_minutes=duration_minutes,
                price_eur_kwh=discharge_price or 0.0
            )
            self._save_learning_data()
            self._update_learning_sensor()

        self._last_soc = current_soc
        self._last_soc_time = now
        self._last_sig_soc = current_soc
        self._last_sig_soc_time = now
        self._last_sig_temp = self._get_battery_temp()  # Track temp at start of next session
        self._last_price_slot = current_slot

    # =========================================================================
    # Mode Transition Handling
    # =========================================================================

    def on_mode_transition(
        self,
        old_mode: "BatteryMode",
        new_mode: "BatteryMode",
        current_soc: Optional[float],
        charge_kwh: Optional[float] = None,
        discharge_kwh: Optional[float] = None
    ):
        """
        Handle mode transition for tracking.

        When transitioning to CHARGE or DISCHARGE, resets the tracking baseline.
        When transitioning from CHARGE to HOLD/DISCHARGE, records cooling observation.

        Args:
            old_mode: Previous battery mode
            new_mode: New battery mode
            current_soc: Current SOC (optional)
            charge_kwh: Current charge_today reading (if energy sensors available)
            discharge_kwh: Current discharge_today reading (if energy sensors available)
        """
        from .models import BatteryMode

        now = self._get_datetime()
        current_temp = self._get_battery_temp()

        # Record temperature observation for ambient estimation
        if current_temp is not None and self._learning_engine:
            self._learning_engine.record_temperature_observation(current_temp)

        # Transitioning TO CHARGE: record cooling if we were idle, then reset baseline
        if new_mode == BatteryMode.CHARGE and old_mode != BatteryMode.CHARGE:
            # Record cooling observation if we have idle tracking data
            if (self._idle_start_time is not None and
                self._idle_start_temp is not None and
                current_temp is not None and
                self._learning_engine):
                duration_minutes = (now - self._idle_start_time).total_seconds() / 60
                if duration_minutes >= 10:  # Only record if idle for at least 10 minutes
                    self._learning_engine.record_cooling(
                        temp_start=self._idle_start_temp,
                        temp_end=current_temp,
                        duration_minutes=duration_minutes
                    )
                    self._save_learning_data()

            # Clear idle tracking
            self._idle_start_time = None
            self._idle_start_temp = None

            # Reset tracking baseline for charging
            if current_soc is not None:
                self._last_sig_soc = current_soc
                self._last_sig_soc_time = now
                self._last_sig_temp = current_temp
                temp_str = f", temp={current_temp:.1f}C" if current_temp is not None else ""
                # Include energy sensor values when available
                energy_str = ""
                if self._energy_sensor_available and charge_kwh is not None:
                    self._last_charge_today_kwh = charge_kwh
                    self._last_discharge_today_kwh = discharge_kwh
                    energy_str = f" (charge: {charge_kwh:.2f} kWh, discharge: {discharge_kwh:.2f} kWh)"
                self._log(f"Mode transition to CHARGE: reset learning baseline to {current_soc:.1f}%{energy_str}{temp_str}")

        # Transitioning TO HOLD or DISCHARGE from CHARGE: start idle tracking
        elif new_mode in (BatteryMode.HOLD, BatteryMode.DISCHARGE) and old_mode == BatteryMode.CHARGE:
            # Start tracking idle period for cooling rate learning
            self._idle_start_time = now
            self._idle_start_temp = current_temp
            temp_str = f", temp={current_temp:.1f}C" if current_temp is not None else ""
            self._log(f"Mode transition to {new_mode.name}: started idle tracking{temp_str}")

            # Also reset tracking baseline for discharge tracking
            if new_mode == BatteryMode.DISCHARGE and current_soc is not None:
                self._last_sig_soc = current_soc
                self._last_sig_soc_time = now
                self._last_sig_temp = current_temp
                energy_str = ""
                if self._energy_sensor_available and charge_kwh is not None:
                    self._last_charge_today_kwh = charge_kwh
                    self._last_discharge_today_kwh = discharge_kwh
                    energy_str = f" (charge: {charge_kwh:.2f} kWh, discharge: {discharge_kwh:.2f} kWh)"
                self._log(f"Mode transition to DISCHARGE: reset learning baseline to {current_soc:.1f}%{energy_str}{temp_str}")

        # Transitioning from HOLD to DISCHARGE: update discharge baseline
        elif new_mode == BatteryMode.DISCHARGE and old_mode == BatteryMode.HOLD:
            if current_soc is not None:
                self._last_sig_soc = current_soc
                self._last_sig_soc_time = now
                self._last_sig_temp = current_temp
                temp_str = f", temp={current_temp:.1f}C" if current_temp is not None else ""
                energy_str = ""
                if self._energy_sensor_available and charge_kwh is not None:
                    self._last_charge_today_kwh = charge_kwh
                    self._last_discharge_today_kwh = discharge_kwh
                    energy_str = f" (charge: {charge_kwh:.2f} kWh, discharge: {discharge_kwh:.2f} kWh)"
                self._log(f"Mode transition to DISCHARGE: reset learning baseline to {current_soc:.1f}%{energy_str}{temp_str}")

    # =========================================================================
    # Threshold Calculations
    # =========================================================================

    def get_discharge_threshold(self) -> float:
        """
        Calculate discharge threshold based on actual battery cost.

        Threshold = (what we paid + grid fees) / efficiency + wear cost
        Only discharge if we can "sell" above this price.
        """
        return self.get_discharge_threshold_for_cost(self._avg_cost)

    def get_discharge_threshold_for_cost(self, avg_cost: float) -> float:
        """Calculate discharge threshold for a given battery average cost."""
        return ((avg_cost + self._config.grid_fee) / self._config.efficiency) + self._config.battery_wear_cost

    # =========================================================================
    # Price Lookup
    # =========================================================================

    def _get_price_for_slot(self, slot: datetime.datetime) -> Optional[float]:
        """Get the electricity price for a specific slot from price service cache."""
        from .timezone_utils import datetimes_match_slot

        local_tz = self._get_timezone()
        for price_point in self._get_cached_prices():
            if datetimes_match_slot(price_point.time, slot, local_tz):
                return price_point.price
        return None

    # =========================================================================
    # Cost Projection
    # =========================================================================

    def project_costs(
        self,
        schedule: Dict[datetime.datetime, "ScheduleEntry"],
        starting_soc: float,
        starting_cost: float,
        prices_by_slot: Dict[datetime.datetime, float],
        predict_load_func: Callable[[datetime.datetime], float],
        charge_rates_by_slot: Optional[Dict[datetime.datetime, float]] = None,
        slot_fractions_by_slot: Optional[Dict[datetime.datetime, float]] = None,
        predict_pv_func: Optional[Callable[[datetime.datetime], float]] = None,
    ) -> Tuple[Dict[datetime.datetime, float], float]:
        """
        Project battery avg cost evolution through a schedule.

        Args:
            schedule: Dict mapping datetime to ScheduleEntry with mode
            starting_soc: Initial SOC percentage
            starting_cost: Initial avg cost (EUR/kWh)
            prices_by_slot: Dict mapping datetime to electricity price
            predict_load_func: Function that takes datetime and returns predicted load in kW
            charge_rates_by_slot: Optional dict mapping datetime to charge rate (kW)
            slot_fractions_by_slot: Optional dict mapping datetime to slot fraction (0-1)

        Returns:
            Tuple of (dict mapping slot -> projected cost at START of that slot, final avg cost)
        """
        from .models import BatteryMode

        slot_hours = self._config.slot_minutes / 60.0
        projected_costs: Dict[datetime.datetime, float] = {}
        current_soc = starting_soc
        current_cost = starting_cost

        for hour in sorted(schedule.keys()):
            projected_costs[hour] = current_cost
            entry = schedule[hour]

            if entry.mode == BatteryMode.CHARGE:
                charge_price = prices_by_slot.get(hour, current_cost)
                old_energy = max(0, self._soc_to_energy_kwh(current_soc))
                slot_charge_rate = (
                    charge_rates_by_slot.get(hour, self._config.charge_rate)
                    if charge_rates_by_slot is not None
                    else self._config.charge_rate
                )
                fraction = (
                    slot_fractions_by_slot.get(hour, 1.0)
                    if slot_fractions_by_slot is not None
                    else 1.0
                )
                energy_added = slot_charge_rate * self._config.efficiency * slot_hours * fraction
                # Calculate headroom: (max_soc - current_soc) in kWh
                headroom_kwh = max(0.0, self._soc_to_energy_kwh(self._get_max_soc()) - self._soc_to_energy_kwh(current_soc))
                energy_added = max(0.0, min(energy_added, headroom_kwh))

                current_cost = self._compute_weighted_avg_cost(
                    old_energy, current_cost, energy_added, charge_price
                )

                current_soc = min(self._get_max_soc(), current_soc + (energy_added / self._config.battery_capacity) * 100)

            elif entry.mode == BatteryMode.DISCHARGE:
                fraction = (
                    slot_fractions_by_slot.get(hour, 1.0)
                    if slot_fractions_by_slot is not None
                    else 1.0
                )
                if entry.export_rate is not None and entry.export_rate > 0:
                    # Export: drain at full export discharge rate
                    edr = self._config.effective_export_discharge_rate
                    energy_removed = edr * slot_hours * fraction
                else:
                    # Self-consumption: drain at load rate
                    load_kw = predict_load_func(hour)
                    energy_removed = min(load_kw, self._config.discharge_rate) * slot_hours * fraction
                current_soc = max(self._get_min_soc(), current_soc - (energy_removed / self._config.battery_capacity) * 100)

        return projected_costs, current_cost
