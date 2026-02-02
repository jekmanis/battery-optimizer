"""
Home Assistant helper functions for the Battery Optimizer.

Provides functions for safely reading and parsing HA entity states.
These are pure functions that take a get_state callable to remain
testable and decoupled from AppDaemon.
"""

from typing import Callable, Optional, Any


# Type alias for the get_state function signature
GetStateFunc = Callable[[str], Any]


def is_state_valid(state: Any) -> bool:
    """
    Check if a Home Assistant state value is valid (not unknown/unavailable/None).

    Args:
        state: The state value from get_state()

    Returns:
        True if state is usable, False otherwise
    """
    return state is not None and state not in ("unknown", "unavailable", "")


def get_float_state(
    get_state: GetStateFunc,
    entity_id: str,
    default: Optional[float] = None,
    log_func: Optional[Callable] = None,
    log_errors: bool = False
) -> Optional[float]:
    """
    Safely get a float value from a Home Assistant entity.

    Args:
        get_state: Function to get entity state (e.g., self.get_state)
        entity_id: The entity ID to read
        default: Default value if state is invalid or parse fails
        log_func: Optional logging function for errors
        log_errors: Whether to log parse errors

    Returns:
        The float value, or default if unavailable/invalid
    """
    try:
        state = get_state(entity_id)
        if is_state_valid(state):
            return float(state)
    except (ValueError, TypeError) as e:
        if log_errors and log_func:
            log_func(f"Error reading {entity_id}: {e}", level="WARNING")
    except Exception:
        pass
    return default


def get_bool_state(
    get_state: GetStateFunc,
    entity_id: str,
    default: bool = False
) -> bool:
    """
    Safely get a boolean value from a Home Assistant entity.

    Interprets "on" as True, everything else as False.

    Args:
        get_state: Function to get entity state
        entity_id: The entity ID to read
        default: Default value if entity doesn't exist or state is invalid

    Returns:
        True if state is "on", False otherwise (or default on error)
    """
    try:
        state = get_state(entity_id)
        if not is_state_valid(state):
            return default
        if isinstance(state, bool):
            return state
        return str(state).lower() == "on"
    except Exception:
        return default


def get_string_state(
    get_state: GetStateFunc,
    entity_id: str,
    default: str = ""
) -> str:
    """
    Safely get a string value from a Home Assistant entity.

    Args:
        get_state: Function to get entity state
        entity_id: The entity ID to read
        default: Default value if state is invalid

    Returns:
        The state as string, or default if unavailable
    """
    try:
        state = get_state(entity_id)
        if is_state_valid(state):
            return str(state)
    except Exception:
        pass
    return default


class SensorReader:
    """
    Helper class for reading sensor values with consistent error handling.

    Wraps get_state and log functions to provide a clean interface for
    reading various sensor types.
    """

    def __init__(
        self,
        get_state: GetStateFunc,
        log_func: Optional[Callable] = None
    ):
        """
        Initialize the sensor reader.

        Args:
            get_state: Function to get entity state (e.g., self.get_state)
            log_func: Optional logging function for errors
        """
        self._get_state = get_state
        self._log = log_func

    def get_float(
        self,
        entity_id: str,
        default: Optional[float] = None,
        log_errors: bool = False
    ) -> Optional[float]:
        """Get a float value from an entity."""
        return get_float_state(
            self._get_state, entity_id, default,
            self._log, log_errors
        )

    def get_bool(self, entity_id: str, default: bool = False) -> bool:
        """Get a boolean value from an entity (True if "on")."""
        return get_bool_state(self._get_state, entity_id, default)

    def get_string(self, entity_id: str, default: str = "") -> str:
        """Get a string value from an entity."""
        return get_string_state(self._get_state, entity_id, default)

    def get_soc(self, soc_sensor: str) -> Optional[float]:
        """
        Get current battery SOC.

        Args:
            soc_sensor: Entity ID of the SOC sensor

        Returns:
            SOC as percentage (0-100), or None if unavailable
        """
        return self.get_float(soc_sensor, log_errors=True)

    def get_power(self, power_sensor: str, default: float = 0.0) -> float:
        """
        Get current power reading.

        Args:
            power_sensor: Entity ID of the power sensor
            default: Default value if unavailable

        Returns:
            Power in Watts (or default)
        """
        value = self.get_float(power_sensor, default=default)
        return default if value is None else value

    def get_temperature(self, temp_sensor: str) -> Optional[float]:
        """
        Get current temperature reading.

        Args:
            temp_sensor: Entity ID of the temperature sensor (can be empty)

        Returns:
            Temperature in Celsius, or None if sensor not configured/unavailable
        """
        if not temp_sensor:
            return None
        return self.get_float(temp_sensor)

    def is_on(self, entity_id: str, default: bool = True) -> bool:
        """
        Check if a boolean entity is "on".

        Args:
            entity_id: Entity ID to check
            default: Default if entity doesn't exist

        Returns:
            True if entity state is "on"
        """
        return self.get_bool(entity_id, default=default)
