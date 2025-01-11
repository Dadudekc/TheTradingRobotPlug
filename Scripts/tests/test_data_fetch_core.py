import pytest
from unittest.mock import patch
import pandas as pd
from aioresponses import aioresponses
import aiohttp
import re


def test_initialize_alpaca_invalid_config(monkeypatch):
    from Scripts.Utilities.data_fetch_utils import initialize_alpaca

    # Missing keys
    monkeypatch.delenv('ALPACA_API_KEY', raising=False)
    monkeypatch.delenv('ALPACA_SECRET_KEY', raising=False)
    assert initialize_alpaca() is None, "Expected None with missing Alpaca keys."

    # Invalid URL
    monkeypatch.setenv('ALPACA_API_KEY', 'dummy_key')
    monkeypatch.setenv('ALPACA_SECRET_KEY', 'dummy_secret')
    monkeypatch.setenv('ALPACA_BASE_URL', 'http://invalid_url')
    try:
        initialize_alpaca()
    except ValueError as e:
        assert "invalid_url" in str(e), "Expected ValueError for invalid base URL."


def test_parse_polygon_data_invalid_timestamps(data_fetch_utils_fixture):
    data = {
        "results": [
            {
                "t": "invalid_timestamp",
                "o": 130.0,
                "h": 132.0,
                "l": 129.0,
                "c": 131.0,
                "v": 1000000
            }
        ]
    }
    try:
        result = data_fetch_utils_fixture._parse_polygon_data(data)
        assert result.empty, "Expected an empty DataFrame for invalid timestamps."
    except Exception as e:
        assert isinstance(e, TypeError), f"Expected TypeError, got {type(e).__name__}"


@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_overlap(data_fetch_utils_fixture):
    symbols = ['AAPL']
    data_sources = ["Alpha Vantage", "Polygon"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    with aioresponses() as mocked:
        # Mock responses for Alpha Vantage
        mocked.get(re.compile(r"alphavantage"), payload={"Time Series (Daily)": {}}, status=200)
        # Mock responses for Polygon
        mocked.get(re.compile(r"polygon"), payload={"results": [{"o": 130.0, "h": 132.0, "l": 129.0, "c": 131.0, "v": 1000000}]}, status=200)

        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols, data_sources, start_date, end_date, interval
        )

    assert 'AAPL' in result, "Expected symbol AAPL in result."
    assert 'Alpha Vantage' in result['AAPL'], "Expected Alpha Vantage data."
    assert 'Polygon' in result['AAPL'], "Expected Polygon data."


@pytest.mark.asyncio
async def test_fetch_polygon_data_unexpected_keys(data_fetch_utils_fixture):
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    mock_response = {
        "results": [
            {
                "unexpected_key": 12345,
                "o": 130.0,
                "h": 132.0,
                "l": 129.0,
                "c": 131.0,
                "v": 1000000
            }
        ]
    }

    with aioresponses() as mocked:
        url_pattern = re.compile(
            rf"https://api\.polygon\.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}\?adjusted=true&apiKey=test_polygonio_key"
        )
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_polygon_data(symbol, session, start_date, end_date)

    assert not result.empty, "Expected non-empty DataFrame."
    assert 'unexpected_key' not in result.columns, "Unexpected key should not appear in the DataFrame."
