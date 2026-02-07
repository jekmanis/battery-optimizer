"""
Pytest configuration and shared fixtures for battery optimizer tests.
"""

import datetime
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

# Add the apps directory to path for imports
APPS_DIR = Path(__file__).parent.parent / "appdaemon" / "apps"
sys.path.insert(0, str(APPS_DIR))

# Create mock appdaemon module before importing battery_optimizer
mock_hass_module = type(sys)("appdaemon.plugins.hass.hassapi")


class MockHassBase:
    """Minimal mock for Hass base class."""
    pass


mock_hass_module.Hass = MockHassBase
sys.modules["appdaemon"] = type(sys)("appdaemon")
sys.modules["appdaemon.plugins"] = type(sys)("appdaemon.plugins")
sys.modules["appdaemon.plugins.hass"] = type(sys)("appdaemon.plugins.hass")
sys.modules["appdaemon.plugins.hass.hassapi"] = mock_hass_module


# Now we can import the module components
from battery_optimizer_lib import (
    BatteryLearningEngine,
    BatteryMode,
    LearningStats,
    LoadProfile,
    LoadProfileStats,
    PricePoint,
    ScheduleEntry,
    TouPeriod,
    _quantile,
)


@pytest.fixture
def learning_engine():
    """Create a fresh BatteryLearningEngine for testing."""
    return BatteryLearningEngine(
        battery_capacity_kwh=14.3,
        nominal_charge_rate_kw=4.5,
        nominal_efficiency=0.85,
        min_soc=10.0,
        max_soc=100.0,
        log_func=lambda msg: None,  # Silent logging
    )


@pytest.fixture
def learning_engine_with_data(learning_engine):
    """Learning engine pre-populated with training data."""
    # Record some charging observations across SOC ranges
    learning_engine.record_charging(10, 25, 60, charge_price=0.05, battery_temp=15.0)
    learning_engine.record_charging(25, 50, 60, charge_price=0.06, battery_temp=16.0)
    learning_engine.record_charging(50, 75, 60, charge_price=0.07, battery_temp=17.0)
    learning_engine.record_charging(75, 90, 60, charge_price=0.08, battery_temp=18.0)

    # Add more observations for confidence
    for _ in range(5):
        learning_engine.record_charging(30, 45, 60, charge_price=0.06, battery_temp=15.0)

    return learning_engine


@pytest.fixture
def learning_engine_with_warming_data(learning_engine):
    """Learning engine with temperature warming rate data for cold->warm transitions.

    Simulates a battery that charges slowly when cold (~3kW at 10-15°C) and
    faster when warm (~6kW at >16°C), with warming rate data.
    """
    # Record cold charge rates (10-15°C -> ~3kW)
    # These represent slow charging when battery is cold
    for _ in range(5):
        learning_engine.record_charging(
            soc_start=30, soc_end=50, duration_minutes=60,
            battery_temp=12.0,
            battery_temp_start=10.0, battery_temp_end=14.0
        )

    # Record warm charge rates (>16°C -> ~6kW)
    # These represent faster charging when battery is warm
    for _ in range(5):
        learning_engine.record_charging(
            soc_start=50, soc_end=80, duration_minutes=60,
            battery_temp=18.0,
            battery_temp_start=16.0, battery_temp_end=20.0
        )

    return learning_engine


@pytest.fixture
def load_profile():
    """Create a fresh LoadProfile for testing."""
    return LoadProfile(
        slot_minutes=60,
        default_load_w=500.0,
        max_samples=60,
        min_samples=6,
        log_func=lambda msg: None,
    )


@pytest.fixture
def load_profile_with_data(load_profile):
    """LoadProfile pre-populated with sample data."""
    base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)

    # Record observations for different hours
    # Morning: low load
    for day in range(7):
        dt = base_time + datetime.timedelta(days=day, hours=6)
        load_profile.record(dt, 300.0 + day * 10)

    # Midday: medium load
    for day in range(7):
        dt = base_time + datetime.timedelta(days=day, hours=12)
        load_profile.record(dt, 600.0 + day * 20)

    # Evening: high load
    for day in range(7):
        dt = base_time + datetime.timedelta(days=day, hours=18)
        load_profile.record(dt, 1200.0 + day * 30)

    # Night: minimal load
    for day in range(7):
        dt = base_time + datetime.timedelta(days=day, hours=2)
        load_profile.record(dt, 150.0 + day * 5)

    return load_profile


@pytest.fixture
def sample_prices() -> List[PricePoint]:
    """Sample Nord Pool prices for a 24-hour period."""
    base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)

    # Realistic price pattern: cheap at night, expensive during peak hours
    prices_cents = [
        3.5,   # 00:00 - night low
        3.2,   # 01:00
        3.0,   # 02:00 - lowest
        3.1,   # 03:00
        3.5,   # 04:00
        5.0,   # 05:00 - morning rise
        8.0,   # 06:00
        12.0,  # 07:00 - morning peak
        14.0,  # 08:00
        10.0,  # 09:00
        8.0,   # 10:00
        7.5,   # 11:00
        8.0,   # 12:00
        7.0,   # 13:00
        6.5,   # 14:00
        7.0,   # 15:00
        9.0,   # 16:00
        15.0,  # 17:00 - evening peak
        18.0,  # 18:00 - highest
        16.0,  # 19:00
        12.0,  # 20:00
        8.0,   # 21:00
        5.0,   # 22:00
        4.0,   # 23:00
    ]

    return [
        PricePoint(
            time=base_time + datetime.timedelta(hours=i),
            price=price / 100  # Convert cents to EUR
        )
        for i, price in enumerate(prices_cents)
    ]


@pytest.fixture
def sample_prices_tomorrow() -> List[PricePoint]:
    """Sample prices for tomorrow (for multi-day scheduling tests)."""
    base_time = datetime.datetime(2024, 1, 16, 0, 0, 0)

    # Different pattern for tomorrow
    prices_cents = [
        4.0, 3.5, 3.2, 3.0, 3.5, 6.0,
        10.0, 15.0, 16.0, 12.0, 9.0, 8.0,
        7.5, 7.0, 6.5, 7.5, 10.0, 17.0,
        20.0, 18.0, 14.0, 10.0, 6.0, 4.5,
    ]

    return [
        PricePoint(
            time=base_time + datetime.timedelta(hours=i),
            price=price / 100
        )
        for i, price in enumerate(prices_cents)
    ]


@pytest.fixture
def extreme_prices() -> List[PricePoint]:
    """Extreme price scenario for edge case testing."""
    base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)

    # Extreme scenario: negative prices, very high peaks
    prices_cents = [
        -2.0,  # 00:00 - negative!
        -1.0,  # 01:00
        0.0,   # 02:00
        0.5,   # 03:00
        1.0,   # 04:00
        2.0,   # 05:00
        5.0,   # 06:00
        25.0,  # 07:00 - spike
        50.0,  # 08:00 - extreme spike
        30.0,  # 09:00
        15.0,  # 10:00
        10.0,  # 11:00
        8.0,   # 12:00
        7.0,   # 13:00
        6.0,   # 14:00
        8.0,   # 15:00
        15.0,  # 16:00
        40.0,  # 17:00 - evening spike
        60.0,  # 18:00 - extreme peak
        45.0,  # 19:00
        25.0,  # 20:00
        12.0,  # 21:00
        5.0,   # 22:00
        2.0,   # 23:00
    ]

    return [
        PricePoint(
            time=base_time + datetime.timedelta(hours=i),
            price=price / 100
        )
        for i, price in enumerate(prices_cents)
    ]


@pytest.fixture
def flat_prices() -> List[PricePoint]:
    """Flat price scenario (no optimization opportunity)."""
    base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)
    flat_price = 0.10  # 10 cents

    return [
        PricePoint(time=base_time + datetime.timedelta(hours=i), price=flat_price)
        for i in range(24)
    ]
