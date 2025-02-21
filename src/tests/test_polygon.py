"""
test_polygon.py

Contains tests specifically for the Polygon API, including
data fetching, malformed responses, and timestamp parsing.
"""
from pathlib import Path
import pytest
import re
import aiohttp
import pandas as pd
from aioresponses import aioresponses
from unittest.mock import MagicMock
from Utilities.data_fetchers.main_data_fetcher import MainDataFetcher

@pytest.mark.asyncio
async def test_fetch_polygon_data_success(data_fetch_utils_fixture):
    """
    Test fetching historical data from Polygon API successfully.
    """
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    mock_response = {
        "results": [
            {
                "t": 1672531200000,  # Timestamp in milliseconds (2023-01-01)
                "o": 130.0,
                "h": 132.0,
                "l": 129.0,
                "c": 131.0,
                "v": 1000000
            },
            {
                "t": 1672617600000,  # Timestamp in milliseconds (2023-01-02)
                "o": 131.0,
                "h": 133.0,
                "l": 130.0,
                "c": 132.0,
                "v": 1500000
            }
        ]
    }

    expected_df = pd.DataFrame({
        'open': [130.0, 131.0],
        'high': [132.0, 133.0],
        'low': [129.0, 130.0],
        'close': [131.0, 132.0],
        'volume': [1000000, 1500000]
    }, index=pd.to_datetime(['2023-01-01', '2023-01-02']).date).rename_axis('date')

    with aioresponses() as mocked:
        url_pattern = re.compile(
            rf"https://api\.polygon\.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}\?adjusted=true&apiKey=test_polygonio_key"
        )
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_polygon_data(symbol, session, start_date, end_date)

    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_polygon_data_unexpected_keys(data_fetch_utils_fixture):
    """
    Test fetching from Polygon when the response contains unexpected fields.
    """
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    mock_response = {
        "results": [
            {
                "unexpected_key": 12345,
                "t": 1672531200000,
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

def test_parse_polygon_data_invalid_timestamps(data_fetch_utils_fixture):
    """
    Test parsing Polygon data containing invalid timestamps.
    """
    data = {
        "results": [
            {
                "t": "invalid_timestamp",  # Invalid
                "o": 130.0,
                "h": 132.0,
                "l": 129.0,
                "c": 131.0,
                "v": 1000000
            }
        ]
    }
    with pytest.raises(TypeError, match="unsupported operand type"):
        data_fetch_utils_fixture._parse_polygon_data(data)

@pytest.mark.asyncio
async def test_fetch_polygon_data_malformed_response(data_fetch_utils_fixture):
    """
    Test fetch_polygon_data with a malformed Polygon API response.
    """
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    mock_response = {
        "results": [
            {
                "o": 130.0,
                "h": 132.0,
                # Missing 't', 'l', 'c', 'v'
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

    assert isinstance(result, pd.DataFrame), "Expected a DataFrame even with malformed data."
    assert result.empty, "Expected an empty DataFrame when response is malformed."
