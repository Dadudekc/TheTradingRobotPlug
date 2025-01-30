"""
test_finnhub.py

Contains tests specifically for Finnhub data fetching,
including quotes, metrics, and related error cases.
"""
from pathlib import Path
import pytest
import re
import aiohttp
import pandas as pd
from aioresponses import aioresponses
from unittest.mock import patch, MagicMock
from src.Utilities.data_fetch_utils import DataFetchUtils

@pytest.mark.asyncio
async def test_fetch_finnhub_quote_success(data_fetch_utils_fixture):
    """
    Test fetching a successful Finnhub quote.
    """
    symbol = 'AAPL'
    url_pattern = re.compile(rf"https://finnhub\.io/api/v1/quote\?symbol={symbol}&token=test_finnhub_key.*")
    mock_response = {
        'c': 150.0,
        'd': 2.5,
        'dp': 1.7,
        'h': 151.0,
        'l': 149.0,
        'o': 148.5,
        'pc': 147.5,
        't': 1618308000
    }

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)
        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_quote(symbol, session)

    expected_df = pd.DataFrame([{
        'date': pd.to_datetime(1618308000, unit='s'),
        'current_price': 150.0,
        'change': 2.5,
        'percent_change': 1.7,
        'high': 151.0,
        'low': 149.0,
        'open': 148.5,
        'previous_close': 147.5
    }]).set_index('date')
    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_finnhub_quote_failure(data_fetch_utils_fixture):
    """
    Test fetching a Finnhub quote with server error response.
    """
    symbol = 'AAPL'
    url_pattern = re.compile(rf"https://finnhub\.io/api/v1/quote\?symbol={symbol}&token=test_finnhub_key.*")
    with aioresponses() as mocked:
        mocked.get(url_pattern, status=500)
        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_quote(symbol, session)

    assert isinstance(result, pd.DataFrame)
    assert result.empty, "Expected an empty DataFrame on failure"

@pytest.mark.asyncio
async def test_fetch_finnhub_quote_unexpected_exception(data_fetch_utils_fixture):
    """
    Test fetch_finnhub_quote handling of unexpected exceptions.
    """
    symbol = 'AAPL'
    with aioresponses() as mocked:
        url_pattern = re.compile(
            rf"https://finnhub\.io/api/v1/quote\?symbol={symbol}&token=test_finnhub_key.*"
        )
        mocked.get(url_pattern, exception=Exception("Unexpected Error"))

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_quote(symbol, session)

    assert isinstance(result, pd.DataFrame), "Expected a DataFrame even on exception."
    assert result.empty, "Expected an empty DataFrame when an exception occurs."

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_success(data_fetch_utils_fixture):
    """
    Test fetching Finnhub metrics with a valid response.
    """
    symbol = "TSLA"
    mock_response = {
        "metric": {
            "52WeekHigh": 123,
            "52WeekLow": 456,
            "MarketCapitalization": 789,
            "P/E": 25.5
        }
    }
    fixed_time = pd.Timestamp('2025-01-10 03:45:03', tz="UTC").floor("s")
    url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token=test_finnhub_key"

    with patch("pandas.Timestamp.utcnow", return_value=fixed_time):
        with aioresponses() as mocked:
            mocked.get(url, payload=mock_response, status=200)
            async with aiohttp.ClientSession() as session:
                result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    expected_df = pd.DataFrame([{
        "52WeekHigh": 123,
        "52WeekLow": 456,
        "MarketCapitalization": 789,
        "P/E": 25.5,
        "date_fetched": fixed_time
    }]).set_index("date_fetched")

    # Normalize for timezone differences
    result = result.reset_index()
    expected_df = expected_df.reset_index()
    result["date_fetched"] = pd.to_datetime(result["date_fetched"]).dt.tz_localize(None)
    expected_df["date_fetched"] = pd.to_datetime(expected_df["date_fetched"]).dt.tz_localize(None)

    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_missing_metric(data_fetch_utils_fixture):
    """
    Test Finnhub metrics response missing the 'metric' key.
    """
    symbol = "AAPL"
    mock_response = {}
    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )
    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)
        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    assert result.empty, "Expected an empty DataFrame when 'metric' is missing."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' in columns."

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_malformed_data(data_fetch_utils_fixture):
    """
    Test Finnhub metrics response with malformed data under 'metric'.
    """
    symbol = "AAPL"
    mock_response = {"metric": "invalid_data"}
    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )
    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)
        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    assert result.empty, "Expected an empty DataFrame when data is malformed."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' column in the DataFrame."

def test_parse_finnhub_metrics_data_valid(data_fetch_utils_fixture):
    """
    Test _parse_finnhub_metrics_data with valid data.
    """
    data = {
        "metric": {
            "52WeekHigh": 123,
            "52WeekLow": 456,
            "MarketCapitalization": 789,
            "P/E": 25.5
        }
    }
    parsed_df = data_fetch_utils_fixture._parse_finnhub_metrics_data(data)
    assert not parsed_df.empty, "Expected a non-empty DataFrame."
    assert "52WeekHigh" in parsed_df.columns, "Missing '52WeekHigh' column."
    assert parsed_df.index.name == "date_fetched", "Index name should be 'date_fetched'."

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_additional_fields(data_fetch_utils_fixture):
    """
    Test fetch_finnhub_metrics with additional unexpected fields in the 'metric' dict.
    """
    symbol = "AAPL"
    mock_response = {
        "metric": {
            "52WeekHigh": 123,
            "52WeekLow": 456,
            "MarketCapitalization": 789,
            "P/E": 25.5,
            "ExtraMetric": 999  # unexpected
        }
    }
    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )
    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)
        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    assert not result.empty, "Expected non-empty DataFrame despite extra fields."
    assert "52WeekHigh" in result.columns, "Expected standard fields to remain."
    assert "ExtraMetric" not in result.columns, "Unexpected fields should typically not appear."
