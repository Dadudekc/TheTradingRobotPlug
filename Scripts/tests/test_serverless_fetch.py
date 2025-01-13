from Scripts.Utilities.serverless_utils import ServerlessFetcher  # Adjust import path as needed
import aiohttp
from unittest.mock import patch, AsyncMock
import pytest

@pytest.mark.asyncio
async def test_fetch_success():
    """
    Test successful fetch from the serverless function.
    """
    fetcher = ServerlessFetcher()
    mock_response_data = {"status": "success", "data": {"symbol": "AAPL", "price": 150.0}}

    # Mock environment variable
    with patch("os.getenv", return_value="https://example-serverless-api.com/fetch"):
        # Mock aiohttp.ClientSession
        with patch("aiohttp.ClientSession") as MockClientSession:
            # Mock response object
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)

            # Mock session and its post method
            mock_session = AsyncMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            MockClientSession.return_value.__aenter__.return_value = mock_session

            # Execute the fetch method
            result = await fetcher.fetch(symbol="AAPL", data_sources=["Alpha Vantage"])

            # Assert the response matches the expected mock response
            assert result == mock_response_data



@pytest.mark.asyncio
async def test_fetch_failure():
    """
    Test failure response from the serverless function.
    """
    fetcher = ServerlessFetcher()
    serverless_url = "https://example-serverless-api.com/fetch"

    # Mock environment variable
    with patch("os.getenv", return_value=serverless_url):
        # Mock the aiohttp session to return a 500 error
        with patch("aiohttp.ClientSession") as MockClientSession:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")

            mock_session = AsyncMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            MockClientSession.return_value.__aenter__.return_value = mock_session

            result = await fetcher.fetch(symbol="AAPL", data_sources=["Alpha Vantage"])
            assert result == {}


@pytest.mark.asyncio
async def test_fetch_exception():
    """
    Test exception handling in the serverless function.
    """
    fetcher = ServerlessFetcher()
    serverless_url = "https://example-serverless-api.com/fetch"

    # Mock environment variable
    with patch("os.getenv", return_value=serverless_url):
        # Mock the aiohttp session to raise an exception
        with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("Connection error")

            result = await fetcher.fetch(symbol="AAPL", data_sources=["Alpha Vantage"])
            assert result == {}


@pytest.mark.asyncio
async def test_fetch_missing_env():
    """
    Test behavior when the serverless endpoint environment variable is missing.
    """
    fetcher = ServerlessFetcher()

    # Mock environment variable to return None
    with patch("os.getenv", return_value=None):
        result = await fetcher.fetch(symbol="AAPL", data_sources=["Alpha Vantage"])
        assert result == {}
