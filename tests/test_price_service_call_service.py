"""The AppDaemon ``call_service`` fallback in ``NordPoolPriceService``.

Every case here is a production failure from 2026-09-08. The fallback sent
``return_result=True``, and AppDaemon 4.5.13 consumes only three of its own
keyword arguments by name -- ``hass_timeout``, ``return_response`` and
``suppress_log_messages`` -- sweeping everything else into the websocket
request's ``service_data`` (``HassPlugin.call_plugin_service``)::

    if return_response is not None:
        req["return_response"] = return_response

    service_data = data.pop("service_data", {})
    service_data.update(data)

So Home Assistant received ``return_result`` as a service PARAMETER and refused
the call with ``invalid_format: not a valid option at 'return_result'``. The
``return_response: True`` that appears at the top level of that same logged
request was AppDaemon's own doing -- the match block just below forces it when
HA's service definition declares a response -- so the flag the call needed was
already present and it still failed on the junk parameter beside it.

The second half of the failure was the response handling: the error envelope
AppDaemon returns
(``{'id', 'type', 'success', 'error', 'ad_status', 'ad_duration'}``) was handed
straight to ``_parse_service_response``, which found no list under any area key
and logged "Parsing 0 price entries" at INFO. A refused call and a day nobody
published prices for produced the same line, and the caller fell through to the
cached-price path with no reason recorded anywhere.
"""

import datetime
from unittest.mock import MagicMock, patch

import pytest

from battery_optimizer_lib import price_service

from tests.test_price_service import _make_price_service

UTC = datetime.timezone.utc


def _no_rest(service):
    """Guarantee the REST path cannot answer, so the fallback runs."""
    service.ha_url = ""
    service.ha_token = ""
    return service


def _capture_log(service):
    """Collect ``(level, message)`` pairs from the service's logger."""
    logged = []
    service.log = lambda msg, **kw: logged.append((kw.get("level", "INFO"), msg))
    return logged


def _service_entries(day="2026-09-06", hours=3):
    """``get_price_indices_for_date`` entries: EUR/MWh with published ends."""
    return [
        {
            "start": f"{day}T{h:02d}:00:00+00:00",
            "end": f"{day}T{h + 1:02d}:00:00+00:00",
            "price": 100.0 + h,
        }
        for h in range(hours)
    ]


def _ad_success_envelope(response):
    """AppDaemon's shape for a successful response-returning service call.

    ``HassPlugin.websocket_send_json`` returns the whole websocket envelope and
    stamps its own bookkeeping onto it::

        result.update({"ad_status": ad_status.name, "ad_duration": travel_time})

    and ``ADAPI.call_service``'s own docstring shows where the payload sits::

        events = self.call_service("calendar/get_events", ...)
                    ["result"]["response"]["calendar.home"]["events"]
    """
    return {
        "id": 136,
        "type": "result",
        "success": True,
        "result": {"context": {"id": "01KB8MP9Q5"}, "response": response},
        "ad_status": "OK",
        "ad_duration": 0.42,
    }


class TestFallbackKwargs:
    """What the fallback actually asks AppDaemon for."""

    def test_passes_return_response_and_never_return_result(self):
        mock_call_service = MagicMock(return_value=None)
        service = _no_rest(_make_price_service(slot_minutes=15))
        service.call_service = mock_call_service

        service._call_nordpool_service("2026-09-06")

        _args, kwargs = mock_call_service.call_args
        assert kwargs.get("return_response") is True
        # The whole bug: anything AppDaemon does not consume by name lands in
        # service_data, and Home Assistant rejects the call for it.
        assert "return_result" not in kwargs

    def test_passes_a_hass_timeout(self):
        """AppDaemon's ``ws_timeout`` default is 10s; a day-ahead fetch is slower.

        ``call_plugin_service`` forwards ``hass_timeout`` as
        ``websocket_send_json(timeout=hass_timeout, ...)``, and the REST path
        already allows 30s, so the two transports agree on what "too slow" is.
        """
        mock_call_service = MagicMock(return_value=None)
        service = _no_rest(_make_price_service(slot_minutes=15))
        service.call_service = mock_call_service

        service._call_nordpool_service("2026-09-06")

        _args, kwargs = mock_call_service.call_args
        assert kwargs.get("hass_timeout") == price_service.SERVICE_CALL_TIMEOUT_S

    def test_still_sends_the_service_parameters(self):
        """Fixing the flag must not drop what the service is actually being asked."""
        mock_call_service = MagicMock(return_value=None)
        service = _no_rest(_make_price_service(slot_minutes=15))
        service.call_service = mock_call_service

        service._call_nordpool_service("2026-09-06")

        args, kwargs = mock_call_service.call_args
        assert args[0] == "nordpool/get_price_indices_for_date"
        assert kwargs["config_entry"] == "test-config-entry"
        assert kwargs["date"] == "2026-09-06"
        assert kwargs["areas"] == "LV"
        assert kwargs["resolution"] == 15


class TestFallbackFailureEnvelopes:
    """A refused or unanswered call says so and returns None."""

    def test_error_envelope_returns_none_and_warns(self):
        """The exact envelope logged in production on 2026-09-08."""
        service = _no_rest(_make_price_service(slot_minutes=15))
        logged = _capture_log(service)
        service.call_service = MagicMock(return_value={
            "id": 136,
            "type": "result",
            "success": False,
            "error": {
                "code": "invalid_format",
                "message": "not a valid option at 'return_result'",
            },
            "ad_status": "OK",
            "ad_duration": 0.02,
        })

        assert service._call_nordpool_service("2026-09-06") is None

        warnings = [m for level, m in logged if level == "WARNING"]
        assert warnings, logged
        assert any("invalid_format" in m for m in warnings), logged
        assert any("2026-09-06" in m for m in warnings), logged

    def test_success_false_without_an_error_key_returns_none(self):
        service = _no_rest(_make_price_service(slot_minutes=15))
        logged = _capture_log(service)
        service.call_service = MagicMock(return_value={
            "id": 1, "type": "result", "success": False, "ad_status": "OK",
        })

        assert service._call_nordpool_service("2026-09-06") is None
        assert any(level == "WARNING" for level, _ in logged), logged

    def test_timeout_envelope_returns_none_and_warns(self):
        """``ad_status: TIMEOUT`` means AppDaemon stopped waiting.

        ``websocket_send_json`` synthesises ``result = {"success": False}`` on
        ``asyncio.TimeoutError`` and stamps the status onto it, so the payload
        carries no evidence at all about what Home Assistant did.
        """
        service = _no_rest(_make_price_service(slot_minutes=15))
        logged = _capture_log(service)
        service.call_service = MagicMock(return_value={
            "success": False,
            "ad_status": "TIMEOUT",
            "ad_duration": 30.0,
        })

        assert service._call_nordpool_service("2026-09-06") is None
        assert any("TIMEOUT" in m for level, m in logged
                   if level == "WARNING"), logged

    def test_nested_terminating_envelope_returns_none(self):
        """The stamp lands under ``result`` on some AD versions; both are checked."""
        service = _no_rest(_make_price_service(slot_minutes=15))
        logged = _capture_log(service)
        service.call_service = MagicMock(return_value={
            "id": 7,
            "type": "result",
            "result": {"ad_status": "TERMINATING", "success": False},
        })

        assert service._call_nordpool_service("2026-09-06") is None
        assert any("TERMINATING" in m for level, m in logged
                   if level == "WARNING"), logged

    def test_none_response_returns_none_and_warns(self):
        service = _no_rest(_make_price_service(slot_minutes=15))
        logged = _capture_log(service)
        service.call_service = MagicMock(return_value=None)

        assert service._call_nordpool_service("2026-09-06") is None
        assert any(level == "WARNING" for level, _ in logged), logged

    def test_success_envelope_without_a_response_warns(self):
        """``return_response`` not honoured: say so, do not parse the context."""
        service = _no_rest(_make_price_service(slot_minutes=15))
        logged = _capture_log(service)
        service.call_service = MagicMock(return_value={
            "id": 3, "type": "result", "success": True,
            "result": None, "ad_status": "OK",
        })

        assert service._call_nordpool_service("2026-09-06") is None
        assert any("return_response" in m for level, m in logged
                   if level == "WARNING"), logged

    def test_a_failed_fallback_yields_no_prices_for_the_date(self):
        """The whole point: `get_prices_for_date` must not report 0 parsed prices."""
        service = _no_rest(_make_price_service(slot_minutes=15))
        _capture_log(service)
        service.call_service = MagicMock(return_value={
            "id": 136, "type": "result", "success": False,
            "error": {"code": "invalid_format", "message": "nope"},
            "ad_status": "OK", "ad_duration": 0.02,
        })

        assert service.get_prices_for_date(datetime.date(2026, 9, 6), UTC) == []


class TestFallbackSuccessEnvelope:
    """A well-formed envelope must parse exactly like the REST path."""

    def test_success_envelope_parses_like_the_rest_path(self):
        """Same day, two transports, identical PricePoints.

        The REST path unwraps ``service_response``; the fallback unwraps
        ``result.response``. If the two did not converge on ``{area: [...]}``
        the corrected keyword would still leave the fallback useless.
        """
        payload = {"LV": _service_entries()}

        rest_service = _make_price_service(
            slot_minutes=15, ha_url="http://ha:8123", ha_token="tok")
        with patch("battery_optimizer_lib.price_service.requests") as mock_requests, \
                patch("battery_optimizer_lib.price_service.REQUESTS_AVAILABLE", True):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"service_response": payload}
            mock_requests.post.return_value = mock_response
            rest_data = rest_service._call_nordpool_service("2026-09-06")

        ad_service = _no_rest(_make_price_service(slot_minutes=15))
        ad_service.call_service = MagicMock(
            return_value=_ad_success_envelope(payload))
        ad_data = ad_service._call_nordpool_service("2026-09-06")

        assert ad_data == rest_data == payload

        rest_prices = rest_service._parse_service_response(rest_data, UTC)
        ad_prices = ad_service._parse_service_response(ad_data, UTC)

        assert len(ad_prices) == 3
        assert [(p.time, p.price, p.end) for p in ad_prices] == \
               [(p.time, p.price, p.end) for p in rest_prices]
        # EUR/MWh -> EUR/kWh, and the end the source published is kept.
        assert ad_prices[0].price == pytest.approx(0.100)
        assert ad_prices[0].time == datetime.datetime(2026, 9, 6, 0, 0, tzinfo=UTC)
        assert ad_prices[0].end == datetime.datetime(2026, 9, 6, 1, 0, tzinfo=UTC)

    def test_success_envelope_normalizes_onto_the_slot_grid(self):
        """End to end: three published hours become twelve quarter hours."""
        service = _no_rest(_make_price_service(slot_minutes=15))
        service.call_service = MagicMock(
            return_value=_ad_success_envelope({"LV": _service_entries()}))

        prices = service.get_prices_for_date(datetime.date(2026, 9, 6), UTC)

        assert len(prices) == 12
        assert prices[0].price == pytest.approx(0.100)
        assert prices[4].price == pytest.approx(0.101)
        assert prices[8].price == pytest.approx(0.102)

    def test_bare_payload_is_passed_through_untouched(self):
        """A plugin (or a test double) may answer with ``{area: [...]}`` directly.

        Unwrapping must key off envelope markers, not off the presence of a
        ``result`` key, or the shape several existing tests inject would be
        thrown away.
        """
        payload = {"LV": _service_entries(hours=1)}
        service = _no_rest(_make_price_service(slot_minutes=15))
        service.call_service = MagicMock(return_value=payload)

        assert service._call_nordpool_service("2026-09-06") == payload

    def test_payload_directly_under_result_is_accepted(self):
        """Some AD/HA combinations skip the ``response`` wrapper."""
        payload = {"LV": _service_entries(hours=1)}
        service = _no_rest(_make_price_service(slot_minutes=15))
        service.call_service = MagicMock(return_value={
            "id": 9, "type": "result", "success": True,
            "result": payload, "ad_status": "OK",
        })

        assert service._call_nordpool_service("2026-09-06") == payload

    def test_rest_path_wins_and_the_fallback_is_not_called(self):
        """The fallback is a fallback: a good REST answer must short-circuit it."""
        payload = {"LV": _service_entries(hours=1)}
        mock_call_service = MagicMock(return_value=None)
        service = _make_price_service(
            slot_minutes=15, ha_url="http://ha:8123", ha_token="tok")
        service.call_service = mock_call_service

        with patch("battery_optimizer_lib.price_service.requests") as mock_requests, \
                patch("battery_optimizer_lib.price_service.REQUESTS_AVAILABLE", True):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"service_response": payload}
            mock_requests.post.return_value = mock_response
            assert service._call_nordpool_service("2026-09-06") == payload

        mock_call_service.assert_not_called()


class TestNoStaleKeywordInSource:
    """``return_result`` must not come back.

    It is a plausible-looking name (it was AppDaemon's own, years ago) and the
    failure it produces is silent: a WARNING from HASS's websocket handler, not
    an exception, and a parser that reports zero prices.
    """

    def test_return_result_is_absent_from_the_price_service(self):
        import inspect

        source = inspect.getsource(price_service)
        offenders = [
            line.strip()
            for line in source.splitlines()
            if "return_result" in line and not line.lstrip().startswith("#")
            and "``return_result``" not in line
        ]
        assert not offenders, offenders
