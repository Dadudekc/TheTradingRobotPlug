"""
test_alpha_vantage.py

Contains tests specifically for Alpha Vantage data fetching
and parsing logic, split from the main test_data_fetch_utils.py.
"""

import pytest
import re
import aiohttp
import pandas as pd
from aioresponses import aioresponses
from unittest.mock import patch, MagicMock
from src.Utilities.data_fetch_utils import DataFetchUtils
from pathlib import Path

@pytest.mark.asyncio
async def test_fetch_alphavantage_data_success(data_fetch_utils_fixture):
    """
    Test fetching data successfully from Alpha Vantage.
    """
    symbol = "AAPL"
    mock_response = {
        "Time Series (Daily)": {
            "2023-01-03": {
                "1. open": "130.0",
                "2. high": "132.0",
                "3. low": "129.0",
                "4. close": "131.0",
                "5. volume": "1000000"
            }
        }
    }

    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey=test_alphavantage_key"
    with aioresponses() as mocked:
        mocked.get(url, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_alphavantage_data(symbol, session, '2023-01-01', '2023-01-31')

    expected_df = pd.DataFrame([
        {
            'open': 130.0,
            'high': 132.0,
            'low': 129.0,
            'close': 131.0,
            'volume': 1000000
        }
    ], index=pd.to_datetime(['2023-01-03'])).rename_axis('date')

    assert not result.empty, "Expected non-empty DataFrame."
    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_alphavantage_data_failure(data_fetch_utils_fixture):
    """
    Test fetching data from Alpha Vantage with an API failure.
    """
    symbol = "AAPL"
    url_pattern = re.compile(
        rf"https://www\.alphavantage\.co/query\?function=TIME_SERIES_DAILY&symbol={symbol}&apikey=test_alphavantage_key.*"
    )
    with aioresponses() as mocked:
        mocked.get(url_pattern, status=500)  # Simulate server error

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_alphavantage_data(symbol, session, '2023-01-01', '2023-01-31')

    assert result.empty, "Expected an empty DataFrame for API failure"

@pytest.mark.asyncio
async def test_fetch_alphavantage_data_missing_timeseries(data_fetch_utils_fixture):
    """
    Test fetching data from Alpha Vantage with missing 'Time Series (Daily)'.
    """
    symbol = "AAPL"
    mock_response = {}  # Missing time series data
    url_pattern = re.compile(
        rf"https://www\.alphavantage\.co/query\?function=TIME_SERIES_DAILY&symbol={symbol}&apikey=test_alphavantage_key.*"
    )
    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_alphavantage_data(symbol, session, '2023-01-01', '2023-01-31')

    assert result.empty, "Expected an empty DataFrame for missing 'Time Series (Daily)'"

@pytest.mark.asyncio
async def test_fetch_alphavantage_data_empty_time_series(data_fetch_utils_fixture):
    """
    Test fetching data from Alpha Vantage where 'Time Series (Daily)' is empty.
    """
    symbol = "AAPL"
    mock_response = {
        "Time Series (Daily)": {}
    }
    url_pattern = re.compile(
        rf"https://www\.alphavantage\.co/query\?function=TIME_SERIES_DAILY&symbol={symbol}&apikey=test_alphavantage_key.*"
    )
    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_alphavantage_data(symbol, session, '2023-01-01', '2023-01-31')

    assert result.empty, "Expected an empty DataFrame when time series is empty."

@pytest.mark.asyncio
async def test_fetch_alphavantage_data_invalid_date_range(data_fetch_utils_fixture):
    """
    Test fetch_alphavantage_data with an invalid or reversed date range.
    """
    symbol = 'AAPL'

    # Mock the behavior of fetch_alphavantage_data to raise a ValueError for invalid dates
    with patch.object(data_fetch_utils_fixture, 'fetch_alphavantage_data', side_effect=ValueError("Invalid date range")):
        async with aiohttp.ClientSession() as session:
            # Expect the ValueError to be raised due to invalid date range
            with pytest.raises(ValueError, match="Invalid date range"):
                await data_fetch_utils_fixture.fetch_alphavantage_data(symbol, session, '2023-12-31', '2023-01-01')


def test_parse_alphavantage_data_valid(data_fetch_utils_fixture):
    """
    Test parsing for valid Alpha Vantage response data.
    """
    data = {
        "Time Series (Daily)": {
            "2023-01-03": {
                "1. open": "130.0",
                "2. high": "132.0",
                "3. low": "129.0",
                "4. close": "131.0",
                "5. volume": "1000000"
            },
            "2023-01-04": {
                "1. open": "131.0",
                "2. high": "133.0",
                "3. low": "130.0",
                "4. close": "132.0",
                "5. volume": "1500000"
            }
        }
    }
    expected_df = pd.DataFrame([
        {
            'date': pd.to_datetime('2023-01-03'),
            'open': 130.0,
            'high': 132.0,
            'low': 129.0,
            'close': 131.0,
            'volume': 1000000
        },
        {
            'date': pd.to_datetime('2023-01-04'),
            'open': 131.0,
            'high': 133.0,
            'low': 130.0,
            'close': 132.0,
            'volume': 1500000
        }
    ]).set_index('date')

    parsed_df = data_fetch_utils_fixture._parse_alphavantage_data(data)
    pd.testing.assert_frame_equal(parsed_df, expected_df)

def test_parse_alphavantage_data_missing_keys(data_fetch_utils_fixture):
    """
    Test _parse_alphavantage_data with missing keys in the response.
    """
    data = {
        "Time Series (Daily)": {
            "2023-01-03": {
                "1. open": "130.0"
                # Missing '2. high', '3. low', '4. close', '5. volume'
            }
        }
    }
    parsed_df = data_fetch_utils_fixture._parse_alphavantage_data(data)
    expected_df = pd.DataFrame([
        {
            'date': pd.to_datetime('2023-01-03'),
            'open': 130.0,
            'high': 0.0,
            'low': 0.0,
            'close': 0.0,
            'volume': 0
        }
    ]).set_index('date')
    pd.testing.assert_frame_equal(parsed_df, expected_df)
