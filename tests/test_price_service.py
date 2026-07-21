"""
Tests for NordPoolPriceService, focusing on 15-minute slot resolution.

Tests cover:
- Price normalization from hourly to 15-min slots
- 15-min passthrough (no change needed)
- 30-min to 15-min expansion
- Service call includes resolution parameter
- REST API call includes resolution parameter
"""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from battery_optimizer_lib import NordPoolPriceService, PricePoint


def _make_price_service(slot_minutes: int = 15, ha_url: str = "", ha_token: str = "") -> NordPoolPriceService:
    """Create a NordPoolPriceService with mock dependencies."""
    return NordPoolPriceService(
        nordpool_config_entry="test-config-entry",
        nordpool_area="LV",
        nordpool_sensor="sensor.nordpool",
        ha_url=ha_url,
        ha_token=ha_token,
        tomorrow_prices_hour=13,
        slot_minutes=slot_minutes,
        get_state_func=MagicMock(return_value=None),
        call_service_func=MagicMock(return_value=None),
        get_datetime_func=lambda: datetime.datetime(2024, 1, 15, 14, 0, 0),
        get_date_func=lambda: datetime.date(2024, 1, 15),
        get_timezone_func=lambda: None,
        log_func=lambda msg, **kwargs: None,
    )


class TestNormalizePrices:
    """Tests for _normalize_prices with various input resolutions."""

    def test_normalize_hourly_to_15min(self):
        """Hourly prices (24 points) should expand to 96 fifteen-minute points."""
        service = _make_price_service(slot_minutes=15)
        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)

        # Create 24 hourly PricePoints with distinct prices
        hourly_prices = [
            PricePoint(time=base_time + datetime.timedelta(hours=h), price=0.10 + h * 0.01)
            for h in range(24)
        ]

        result = service._normalize_prices(hourly_prices)

        # Should produce 96 fifteen-minute slots
        assert len(result) == 96

        # Each hour's price should appear exactly 4 times
        for h in range(24):
            expected_price = 0.10 + h * 0.01
            slots_for_hour = [
                p for p in result
                if p.time >= base_time + datetime.timedelta(hours=h)
                and p.time < base_time + datetime.timedelta(hours=h + 1)
            ]
            assert len(slots_for_hour) == 4, (
                f"Hour {h} should have 4 slots, got {len(slots_for_hour)}"
            )
            for slot in slots_for_hour:
                assert abs(slot.price - expected_price) < 1e-9, (
                    f"Hour {h} slot price {slot.price} != expected {expected_price}"
                )

    def test_normalize_15min_passthrough(self):
        """96 fifteen-minute prices should pass through unchanged."""
        service = _make_price_service(slot_minutes=15)
        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)

        # Create 96 fifteen-minute PricePoints
        prices_15min = [
            PricePoint(
                time=base_time + datetime.timedelta(minutes=15 * i),
                price=0.05 + i * 0.001
            )
            for i in range(96)
        ]

        result = service._normalize_prices(prices_15min)

        # Should be identical count
        assert len(result) == 96

        # Should have the same prices in order
        for original, normalized in zip(prices_15min, result):
            assert original.time == normalized.time
            assert abs(original.price - normalized.price) < 1e-9

    def test_normalize_30min_to_15min(self):
        """48 thirty-minute prices should expand to 96 fifteen-minute points."""
        service = _make_price_service(slot_minutes=15)
        base_time = datetime.datetime(2024, 1, 15, 0, 0, 0)

        # Create 48 thirty-minute PricePoints
        prices_30min = [
            PricePoint(
                time=base_time + datetime.timedelta(minutes=30 * i),
                price=0.08 + i * 0.002
            )
            for i in range(48)
        ]

        result = service._normalize_prices(prices_30min)

        # Should produce 96 fifteen-minute slots
        assert len(result) == 96

        # Each 30-min price should be repeated 2 times
        for i in range(48):
            expected_price = 0.08 + i * 0.002
            half_hour_start = base_time + datetime.timedelta(minutes=30 * i)
            half_hour_end = half_hour_start + datetime.timedelta(minutes=30)
            slots_for_half_hour = [
                p for p in result
                if p.time >= half_hour_start and p.time < half_hour_end
            ]
            assert len(slots_for_half_hour) == 2, (
                f"Half-hour slot {i} should have 2 sub-slots, got {len(slots_for_half_hour)}"
            )
            for slot in slots_for_half_hour:
                assert abs(slot.price - expected_price) < 1e-9

    def test_fall_dst_repeated_hour_is_preserved(self, riga_timezone):
        """Both physical 03:00 occurrences must survive normalization."""
        riga = riga_timezone
        service = _make_price_service(slot_minutes=15)
        service.get_timezone = lambda: riga
        utc_start = datetime.datetime(2024, 10, 26, 21, tzinfo=datetime.timezone.utc)
        hourly = [
            PricePoint(
                time=(utc_start + datetime.timedelta(hours=i)).astimezone(riga),
                price=float(i),
            )
            for i in range(25)
        ]

        result = service._normalize_prices(hourly)

        assert len(result) == 100
        repeated = [p for p in result if p.time.hour == 3 and p.time.minute == 0]
        assert len(repeated) == 2
        assert repeated[0].time.astimezone(datetime.timezone.utc) != repeated[1].time.astimezone(datetime.timezone.utc)

    def test_simple_price_list_supports_25_hour_dst_day(self, riga_timezone):
        riga = riga_timezone
        service = _make_price_service(slot_minutes=60)
        result = service._parse_sensor_prices(
            list(range(25)), datetime.date(2024, 10, 27), riga
        )
        assert len(result) == 25
        repeated = [p for p in result if p.time.hour == 3]
        assert len(repeated) == 2
        assert {p.time.fold for p in repeated} == {0, 1}


class TestServiceCallResolution:
    """Tests verifying resolution parameter in service/API calls."""

    def test_service_call_includes_resolution(self):
        """call_service should include resolution=15 in kwargs."""
        mock_call_service = MagicMock(return_value=None)
        service = _make_price_service(slot_minutes=15)
        service.call_service = mock_call_service

        # Ensure REST API path does not succeed so we fall through to call_service
        service.ha_url = ""
        service.ha_token = ""

        service._call_nordpool_service("2024-01-15")

        # call_service should have been called
        mock_call_service.assert_called_once()

        # Verify the call arguments
        args, kwargs = mock_call_service.call_args
        assert args[0] == "nordpool/get_price_indices_for_date"
        assert kwargs.get("resolution") == 15

    def test_rest_api_includes_resolution(self):
        """REST API call should include resolution=15 in the payload."""
        service = _make_price_service(slot_minutes=15, ha_url="http://localhost:8123", ha_token="test-token")

        with patch("battery_optimizer_lib.price_service.requests") as mock_requests:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"LV": []}
            mock_requests.post.return_value = mock_response
            # Ensure REQUESTS_AVAILABLE is True
            with patch("battery_optimizer_lib.price_service.REQUESTS_AVAILABLE", True):
                service._call_nordpool_rest_api("2024-01-15")

            # Verify requests.post was called
            mock_requests.post.assert_called_once()

            # Verify URL contains get_price_indices_for_date
            call_args = mock_requests.post.call_args
            url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
            assert "get_price_indices_for_date" in url

            # Verify payload includes resolution=15
            payload = call_args[1].get("json", {}) if call_args[1] else {}
            assert payload.get("resolution") == 15
