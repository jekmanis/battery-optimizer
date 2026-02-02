"""Tests for ha_helpers module."""

import pytest
from battery_optimizer_lib.ha_helpers import (
    is_state_valid,
    get_float_state,
    get_bool_state,
    get_string_state,
    SensorReader,
)


class TestIsStateValid:
    """Tests for is_state_valid function."""

    def test_none_invalid(self):
        assert is_state_valid(None) is False

    def test_unknown_invalid(self):
        assert is_state_valid("unknown") is False

    def test_unavailable_invalid(self):
        assert is_state_valid("unavailable") is False

    def test_empty_string_invalid(self):
        assert is_state_valid("") is False

    def test_valid_string(self):
        assert is_state_valid("on") is True

    def test_valid_number(self):
        assert is_state_valid("42.5") is True

    def test_valid_zero(self):
        assert is_state_valid("0") is True

    def test_valid_numeric(self):
        assert is_state_valid(100) is True


class TestGetFloatState:
    """Tests for get_float_state function."""

    def test_valid_float(self):
        def get_state(entity_id):
            return "42.5"
        result = get_float_state(get_state, "sensor.test")
        assert result == 42.5

    def test_valid_integer_string(self):
        def get_state(entity_id):
            return "100"
        result = get_float_state(get_state, "sensor.test")
        assert result == 100.0

    def test_unavailable_returns_default(self):
        def get_state(entity_id):
            return "unavailable"
        result = get_float_state(get_state, "sensor.test", default=0.0)
        assert result == 0.0

    def test_unknown_returns_default(self):
        def get_state(entity_id):
            return "unknown"
        result = get_float_state(get_state, "sensor.test", default=-1.0)
        assert result == -1.0

    def test_none_returns_default(self):
        def get_state(entity_id):
            return None
        result = get_float_state(get_state, "sensor.test", default=None)
        assert result is None

    def test_invalid_string_returns_default(self):
        def get_state(entity_id):
            return "not a number"
        result = get_float_state(get_state, "sensor.test", default=0.0)
        assert result == 0.0

    def test_exception_returns_default(self):
        def get_state(entity_id):
            raise Exception("Connection error")
        result = get_float_state(get_state, "sensor.test", default=99.9)
        assert result == 99.9


class TestGetBoolState:
    """Tests for get_bool_state function."""

    def test_on_returns_true(self):
        def get_state(entity_id):
            return "on"
        result = get_bool_state(get_state, "input_boolean.test")
        assert result is True

    def test_off_returns_false(self):
        def get_state(entity_id):
            return "off"
        result = get_bool_state(get_state, "input_boolean.test")
        assert result is False

    def test_unavailable_returns_false(self):
        def get_state(entity_id):
            return "unavailable"
        result = get_bool_state(get_state, "input_boolean.test")
        assert result is False

    def test_exception_returns_default_true(self):
        def get_state(entity_id):
            raise Exception("Entity not found")
        result = get_bool_state(get_state, "input_boolean.test", default=True)
        assert result is True

    def test_exception_returns_default_false(self):
        def get_state(entity_id):
            raise Exception("Entity not found")
        result = get_bool_state(get_state, "input_boolean.test", default=False)
        assert result is False


class TestGetStringState:
    """Tests for get_string_state function."""

    def test_valid_string(self):
        def get_state(entity_id):
            return "Hello"
        result = get_string_state(get_state, "sensor.test")
        assert result == "Hello"

    def test_number_converted_to_string(self):
        def get_state(entity_id):
            return 42
        result = get_string_state(get_state, "sensor.test")
        assert result == "42"

    def test_unavailable_returns_default(self):
        def get_state(entity_id):
            return "unavailable"
        result = get_string_state(get_state, "sensor.test", default="N/A")
        assert result == "N/A"

    def test_exception_returns_default(self):
        def get_state(entity_id):
            raise Exception("Error")
        result = get_string_state(get_state, "sensor.test", default="error")
        assert result == "error"


class TestSensorReader:
    """Tests for SensorReader class."""

    def test_get_float(self):
        def get_state(entity_id):
            if entity_id == "sensor.soc":
                return "85.5"
            return None
        reader = SensorReader(get_state)
        assert reader.get_float("sensor.soc") == 85.5

    def test_get_bool(self):
        def get_state(entity_id):
            if entity_id == "input_boolean.enabled":
                return "on"
            return "off"
        reader = SensorReader(get_state)
        assert reader.get_bool("input_boolean.enabled") is True
        assert reader.get_bool("other") is False

    def test_get_string(self):
        def get_state(entity_id):
            return "test_value"
        reader = SensorReader(get_state)
        assert reader.get_string("sensor.test") == "test_value"

    def test_get_soc(self):
        def get_state(entity_id):
            return "75.0"
        reader = SensorReader(get_state)
        result = reader.get_soc("sensor.battery_soc")
        assert result == 75.0

    def test_get_soc_unavailable(self):
        def get_state(entity_id):
            return "unavailable"
        reader = SensorReader(get_state)
        result = reader.get_soc("sensor.battery_soc")
        assert result is None

    def test_get_power(self):
        def get_state(entity_id):
            return "1500"
        reader = SensorReader(get_state)
        result = reader.get_power("sensor.pv_power", default=0.0)
        assert result == 1500.0

    def test_get_power_unavailable_uses_default(self):
        def get_state(entity_id):
            return "unknown"
        reader = SensorReader(get_state)
        result = reader.get_power("sensor.pv_power", default=0.0)
        assert result == 0.0

    def test_get_temperature(self):
        def get_state(entity_id):
            return "22.5"
        reader = SensorReader(get_state)
        result = reader.get_temperature("sensor.battery_temp")
        assert result == 22.5

    def test_get_temperature_empty_sensor(self):
        def get_state(entity_id):
            return "25.0"
        reader = SensorReader(get_state)
        result = reader.get_temperature("")
        assert result is None

    def test_is_on(self):
        def get_state(entity_id):
            if entity_id == "input_boolean.enabled":
                return "on"
            return "off"
        reader = SensorReader(get_state)
        assert reader.is_on("input_boolean.enabled") is True
        assert reader.is_on("input_boolean.other") is False

    def test_is_on_default_true(self):
        def get_state(entity_id):
            raise Exception("Not found")
        reader = SensorReader(get_state)
        assert reader.is_on("missing", default=True) is True
        assert reader.is_on("missing", default=False) is False
