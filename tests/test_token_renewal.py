"""Tests for Noonlight API token renewal."""

import asyncio
from datetime import timedelta
from unittest.mock import patch

import aiohttp
import pytest
from freezegun.api import FrozenDateTimeFactory
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_ID
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_fire_time_changed,
)
from pytest_homeassistant_custom_component.test_util.aiohttp import (
    AiohttpClientMocker,
)

from custom_components.noonlight import NoonlightIntegration
from custom_components.noonlight.const import (
    CONF_API_ENDPOINT,
    CONF_SECRET,
    CONF_TOKEN_ENDPOINT,
    DOMAIN,
)

TOKEN_URL = "https://noonlight.example/ha/token"
API_URL = "https://api.noonlight.example/dispatch/v1/alarms"
CONFIG = {
    CONF_ID: "test-id",
    CONF_SECRET: "test-secret",
    CONF_API_ENDPOINT: API_URL,
    CONF_TOKEN_ENDPOINT: TOKEN_URL,
}

RETRY_INTERVAL = timedelta(minutes=3)


def _token_body() -> dict:
    """Return a valid token response that expires well in the future."""
    return {
        "token": "a-token",
        "expires": (dt_util.utcnow() + timedelta(hours=24)).isoformat(),
    }


async def _advance(hass: HomeAssistant, freezer: FrozenDateTimeFactory, delta):
    freezer.tick(delta)
    async_fire_time_changed(hass)
    await hass.async_block_till_done()


async def _setup_entry(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> MockConfigEntry:
    """Set up the entry; the renewal task runs its first check right away."""
    entry = MockConfigEntry(domain=DOMAIN, data=CONFIG)
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry


@pytest.mark.parametrize(
    "exc",
    [
        aiohttp.ClientConnectionError("Timeout while contacting DNS servers"),
        aiohttp.ClientError("generic client error"),
        TimeoutError(),
    ],
)
async def test_check_api_token_network_error_returns_false(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, exc
) -> None:
    """A network error during renewal is a failed renewal, not an exception."""
    aioclient_mock.post(TOKEN_URL, exc=exc)
    integration = NoonlightIntegration(hass, CONFIG)

    assert await integration.check_api_token() is False
    assert integration.access_token is None


async def test_check_api_token_non_json_response_returns_false(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A non-JSON error page from the token endpoint is a failed renewal."""
    aioclient_mock.post(
        TOKEN_URL,
        status=502,
        text="<html>Bad Gateway</html>",
        headers={"Content-Type": "text/html"},
    )
    integration = NoonlightIntegration(hass, CONFIG)

    assert await integration.check_api_token() is False


async def test_renewal_retries_after_network_error(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    freezer: FrozenDateTimeFactory,
) -> None:
    """Renewal keeps retrying after a network error and eventually recovers."""
    aioclient_mock.post(TOKEN_URL, exc=aiohttp.ClientConnectionError("DNS timeout"))
    with patch(
        "custom_components.noonlight.persistent_notification.async_create"
    ) as mock_notify:
        await _setup_entry(hass, freezer)
    assert aioclient_mock.call_count == 1

    # A failure notification is shown instead of failing silently.
    mock_notify.assert_called_once()
    assert mock_notify.call_args.args[2] == "Noonlight Token Renewal Failure"

    # It retries again after the short retry interval.
    await _advance(hass, freezer, RETRY_INTERVAL)
    assert aioclient_mock.call_count == 2

    # Once the network is back, the token is renewed.
    aioclient_mock.clear_requests()
    aioclient_mock.post(TOKEN_URL, json=_token_body())
    await _advance(hass, freezer, RETRY_INTERVAL)
    assert aioclient_mock.call_count == 1
    integration = next(iter(hass.data[DOMAIN].values()))
    assert integration.access_token == "a-token"


async def test_renewal_reschedules_after_unexpected_exception(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """Any exception from the check must not stop future renewal attempts."""
    with patch.object(
        NoonlightIntegration,
        "check_api_token",
        side_effect=RuntimeError("boom"),
        autospec=True,
    ) as mock_check:
        await _setup_entry(hass, freezer)
        assert mock_check.call_count == 1

        await _advance(hass, freezer, RETRY_INTERVAL)
        assert mock_check.call_count == 2

        await _advance(hass, freezer, delta=RETRY_INTERVAL)
        assert mock_check.call_count == 3


async def test_unload_stops_renewal(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """Unloading the entry cancels the scheduled renewal check."""
    with patch.object(
        NoonlightIntegration, "check_api_token", return_value=False, autospec=True
    ) as mock_check:
        entry = await _setup_entry(hass, freezer)
        assert mock_check.call_count == 1

        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
        assert entry.state is ConfigEntryState.NOT_LOADED

        await _advance(hass, freezer, timedelta(hours=1))
        assert mock_check.call_count == 1


async def test_unload_cancels_in_flight_renewal(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """Unloading mid-request cancels the renewal; it doesn't finish or retry."""
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def hang(self, force_renew=False):
        started.set()
        try:
            await asyncio.Event().wait()  # a request that never returns
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with (
        patch.object(
            NoonlightIntegration, "check_api_token", side_effect=hang, autospec=True
        ) as mock_check,
        patch(
            "custom_components.noonlight.persistent_notification.async_create"
        ) as mock_notify,
    ):
        entry = await _setup_entry(hass, freezer)
        await started.wait()

        assert await hass.config_entries.async_unload(entry.entry_id)
        assert cancelled.is_set()

        await _advance(hass, freezer, timedelta(hours=1))
        assert mock_check.call_count == 1
        mock_notify.assert_not_called()


async def test_recovery_restores_normal_check_interval(
    hass: HomeAssistant, freezer: FrozenDateTimeFactory
) -> None:
    """Successful recovery switches from three-minute to fifteen-minute checks."""
    with patch.object(
        NoonlightIntegration,
        "check_api_token",
        side_effect=[False, True, True],
        autospec=True,
    ) as mock_check:
        # first check_api_token fails, so will retry in 3 minutes
        await _setup_entry(hass, freezer)
        # advance 3 minutes and succeeds, reset to 15 minute checks
        await _advance(hass, freezer, RETRY_INTERVAL)
        assert mock_check.call_count == 2

        # advance another 3 minutes, but the retry is 15 minutes, so still 2
        await _advance(hass, freezer, RETRY_INTERVAL)
        assert mock_check.call_count == 2

        # advance to the remainder of the 15 minutes, and see a third call
        await _advance(hass, freezer, timedelta(minutes=12))
        assert mock_check.call_count == 3
