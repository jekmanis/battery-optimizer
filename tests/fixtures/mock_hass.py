"""
Mock AppDaemon/Home Assistant framework for testing BatteryOptimizer.

This module provides lightweight mocks that allow testing the battery optimizer
without requiring a running Home Assistant instance or AppDaemon.
"""

import datetime
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock


class MockHass:
    """
    Mock of appdaemon.plugins.hass.hassapi.Hass base class.

    Provides the minimal interface needed for BatteryOptimizer testing.
    """

    def __init__(self):
        # Simulated entity states: entity_id -> {"state": value, "attributes": {}}
        self._states: Dict[str, Dict[str, Any]] = {}

        # Service call log: list of (domain, service, kwargs)
        self._service_calls: List[tuple] = []

        # Scheduled callbacks: list of (callback, kwargs)
        self._scheduled: List[tuple] = []

        # Current simulated time
        self._current_time: datetime.datetime = datetime.datetime.now()
        self._current_date: datetime.date = datetime.date.today()

        # Timezone
        self._timezone: Optional[datetime.timezone] = datetime.timezone.utc

        # Logged messages
        self._logs: List[tuple] = []

        # Args (simulating apps.yaml config)
        self.args: Dict[str, Any] = {}

    # --- Time methods ---

    def datetime(self, **kwargs) -> datetime.datetime:
        """Return current simulated datetime."""
        if kwargs:
            return datetime.datetime(**kwargs)
        return self._current_time

    def date(self) -> datetime.date:
        """Return current simulated date."""
        return self._current_date

    def time(self) -> datetime.time:
        """Return current simulated time."""
        return self._current_time.time()

    def set_datetime(self, dt: datetime.datetime):
        """Set simulated current time (test helper)."""
        self._current_time = dt
        self._current_date = dt.date()

    # --- State methods ---

    def get_state(self, entity_id: str, attribute: str = None, default: Any = None) -> Any:
        """Get entity state or attribute."""
        if entity_id not in self._states:
            return default

        state_data = self._states[entity_id]

        if attribute is None:
            return state_data.get("state", default)
        elif attribute == "all":
            return state_data
        else:
            return state_data.get("attributes", {}).get(attribute, default)

    def set_state(self, entity_id: str, state: Any = None, attributes: Dict = None, **kwargs):
        """Set entity state."""
        if entity_id not in self._states:
            self._states[entity_id] = {"state": None, "attributes": {}}

        if state is not None:
            self._states[entity_id]["state"] = state

        if attributes is not None:
            self._states[entity_id]["attributes"].update(attributes)

    def setup_state(self, entity_id: str, state: Any, attributes: Dict = None):
        """Test helper: set up an entity state."""
        self._states[entity_id] = {
            "state": state,
            "attributes": attributes or {}
        }

    # --- Service methods ---

    def call_service(self, service: str, **kwargs) -> Any:
        """Call a Home Assistant service."""
        # Parse domain/service
        if "/" in service:
            domain, svc = service.split("/", 1)
        else:
            domain = "homeassistant"
            svc = service

        self._service_calls.append((domain, svc, kwargs))
        return None

    def get_service_calls(self, domain: str = None) -> List[tuple]:
        """Test helper: get logged service calls."""
        if domain is None:
            return self._service_calls
        return [(d, s, k) for d, s, k in self._service_calls if d == domain]

    def clear_service_calls(self):
        """Test helper: clear service call log."""
        self._service_calls.clear()

    # --- Scheduling methods ---

    def run_daily(self, callback: Callable, time: datetime.time, **kwargs):
        """Schedule daily callback."""
        self._scheduled.append(("daily", callback, time, kwargs))
        return MagicMock()

    def run_every(self, callback: Callable, start: datetime.datetime, interval: int, **kwargs):
        """Schedule recurring callback."""
        self._scheduled.append(("every", callback, start, interval, kwargs))
        return MagicMock()

    def run_in(self, callback: Callable, delay: int, **kwargs):
        """Schedule callback after delay."""
        self._scheduled.append(("in", callback, delay, kwargs))
        return MagicMock()

    def run_at(self, callback: Callable, time: datetime.datetime, **kwargs):
        """Schedule callback at specific time."""
        self._scheduled.append(("at", callback, time, kwargs))
        return MagicMock()

    # --- State listener methods ---

    def listen_state(self, callback: Callable, entity_id: str, **kwargs):
        """Listen for state changes."""
        self._scheduled.append(("listen", callback, entity_id, kwargs))
        return MagicMock()

    # --- Logging methods ---

    def log(self, message: str, level: str = "INFO", **kwargs):
        """Log a message."""
        self._logs.append((level, message, kwargs))

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log(message, level="ERROR", **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log(message, level="WARNING", **kwargs)

    def get_logs(self, level: str = None) -> List[str]:
        """Test helper: get logged messages."""
        if level is None:
            return [msg for _, msg, _ in self._logs]
        return [msg for lvl, msg, _ in self._logs if lvl == level]

    def clear_logs(self):
        """Test helper: clear log messages."""
        self._logs.clear()


def create_mock_optimizer(
    config: Dict[str, Any] = None,
    current_time: datetime.datetime = None,
    states: Dict[str, Dict[str, Any]] = None,
) -> "BatteryOptimizer":
    """
    Factory function to create a BatteryOptimizer with mocked dependencies.

    Args:
        config: apps.yaml-style configuration dict
        current_time: Simulated current time
        states: Initial entity states

    Returns:
        BatteryOptimizer instance with MockHass as base
    """
    import sys
    from pathlib import Path

    # Add apps directory to path if needed
    apps_path = Path(__file__).parent.parent.parent / "appdaemon" / "apps"
    if str(apps_path) not in sys.path:
        sys.path.insert(0, str(apps_path))

    # Create mock module for appdaemon
    mock_hass_module = type(sys)("appdaemon.plugins.hass.hassapi")
    mock_hass_module.Hass = MockHass
    sys.modules["appdaemon"] = type(sys)("appdaemon")
    sys.modules["appdaemon.plugins"] = type(sys)("appdaemon.plugins")
    sys.modules["appdaemon.plugins.hass"] = type(sys)("appdaemon.plugins.hass")
    sys.modules["appdaemon.plugins.hass.hassapi"] = mock_hass_module

    # Import the optimizer
    from battery_optimizer import BatteryOptimizer

    # Create instance (bypassing normal __init__)
    optimizer = object.__new__(BatteryOptimizer)
    MockHass.__init__(optimizer)

    # Set default config
    default_config = {
        "battery_capacity_kwh": 14.3,
        "charge_rate_kw": 4.5,
        "discharge_rate_kw": 4.5,
        "efficiency": 0.85,
        "min_soc": 10,
        "max_soc": 100,
        "grid_fee": 0.05,
        "slot_minutes": 60,
        "base_consumption_w": 500,
        "nordpool_sensor": "sensor.nordpool",
        "soc_sensor": "sensor.battery_soc",
        "pv_sensor": "sensor.pv_power",
        "battery_temp_sensor": "",
        "load_sensor": "sensor.house_load",
        "device_id": "",  # Dry run mode
        "load_quantile": 0.75,
        "soc_step_percent": 1.0,
        "tomorrow_prices_hour": 14,
        "adaptive_recalc_minutes": 30,
        "load_observation_minutes": 10,
        "load_profile_max_samples": 60,
        "load_profile_min_samples": 6,
        "load_profile_file": "",
        "learning_data_file": "",
        "decision_log_level": 0,
    }

    if config:
        default_config.update(config)

    optimizer.args = default_config

    # Set current time
    if current_time:
        optimizer.set_datetime(current_time)
    else:
        optimizer.set_datetime(datetime.datetime(2024, 1, 15, 10, 0, 0))

    # Set up initial states
    if states:
        for entity_id, state_data in states.items():
            optimizer.setup_state(
                entity_id,
                state_data.get("state"),
                state_data.get("attributes", {})
            )

    # Set up default SOC sensor
    if "sensor.battery_soc" not in (states or {}):
        optimizer.setup_state("sensor.battery_soc", "50")

    return optimizer
