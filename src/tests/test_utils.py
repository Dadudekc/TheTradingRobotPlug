"""
test_utils.py

Contains tests for:
- Alpaca data fetching (single source),
- yfinance-based fetches (fetch_stock_data_async),
- Retries (fetch_with_retries),
- Concurrency or large symbol handling (when not mixing multiple data sources).
"""

import pytest
import pandas as pd
import re
import aiohttp
from aioresponses import aioresponses
from unittest.mock import patch, MagicMock
from datetime import date
from Utilities.main_data_fetcher import DataFetchUtils


@pytest.mark.asyncio
async def test_fetch_stock_data_async(data_fetch_utils_fixture):
    """
    Test yfinance-based stock data fetch for a single ticker.
    """
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    expected_index = pd.Index([date(2023, 1, 3), date(2023, 1, 4)], name='date')
    expected_data = pd.DataFrame({
        'open': [130.0, 131.0],
        'high': [132.0, 133.0],
        'low': [129.0, 130.0],
        'close': [131.0, 132.0],
        'volume': [1000000, 1500000],
        'adj_close': [130.5, 131.5],
        'symbol': ['AAPL', 'AAPL']
    }, index=expected_index)

    with patch('yfinance.download') as mock_yf_download:
        mock_yf_download.return_value = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-03', '2023-01-04']),
            'Open': [130.0, 131.0],
            'High': [132.0, 133.0],
            'Low': [129.0, 130.0],
            'Close': [131.0, 132.0],
            'Volume': [1000000, 1500000],
            'Adj Close': [130.5, 131.5]
        }).set_index('Date')

        result = await data_fetch_utils_fixture.fetch_stock_data_async(ticker, start_date, end_date, interval)

    pd.testing.assert_frame_equal(result, expected_data)


@pytest.mark.asyncio
async def test_fetch_alpaca_data_async_success(data_fetch_utils_fixture):
    """
    Test successful Alpaca data fetch for a single symbol.
    """
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1Day'
    mock_bars_df = pd.DataFrame({
        'timestamp': [pd.Timestamp('2023-01-03'), pd.Timestamp('2023-01-04')],
        'open': [130.0, 131.0],
        'high': [132.0, 133.0],
        'low': [129.0, 130.0],
        'close': [131.0, 132.0],
        'trade_count': [1000000, 1500000],
        'symbol': ['AAPL', 'AAPL']
    })

    with patch.object(data_fetch_utils_fixture.alpaca_api, 'get_bars', return_value=MagicMock(df=mock_bars_df.copy())):
        result = await data_fetch_utils_fixture.fetch_alpaca_data_async(symbol, start_date, end_date, interval)

    expected_df = mock_bars_df.rename(columns={'timestamp': 'date', 'trade_count': 'volume'}).copy()
    expected_df['date'] = pd.to_datetime(expected_df['date']).dt.date
    expected_df.set_index('date', inplace=True)
    expected_df.drop(columns=['date'], errors='ignore', inplace=True)

    assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume', 'symbol']
    pd.testing.assert_frame_equal(result, expected_df)


@pytest.mark.asyncio
async def test_fetch_alpaca_data_async_failure(data_fetch_utils_fixture):
    """
    Test that an Alpaca fetch failure returns an empty DataFrame.
    """
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1Day'

    with patch.object(data_fetch_utils_fixture.alpaca_api, 'get_bars', side_effect=Exception("API Error")):
        result = await data_fetch_utils_fixture.fetch_alpaca_data_async(symbol, start_date, end_date, interval)

    assert isinstance(result, pd.DataFrame)
    assert result.empty, "Expected an empty DataFrame on API failure."


@pytest.mark.asyncio
async def test_fetch_alpaca_data_async_invalid_date_range(data_fetch_utils_fixture):
    """
    Test behavior of fetch_alpaca_data_async with invalid or reversed date range.
    """
    symbol = 'AAPL'
    start_date = '2023-02-30'  # invalid date
    end_date = '2023-01-31'
    interval = '1Day'

    with patch.object(data_fetch_utils_fixture.alpaca_api, 'get_bars', side_effect=ValueError("Invalid date range")):
        result = await data_fetch_utils_fixture.fetch_alpaca_data_async(symbol, start_date, end_date, interval)

    assert result.empty, "Expected an empty DataFrame for invalid date range."


@pytest.mark.asyncio
async def test_fetch_alpaca_no_key(data_fetch_utils_fixture):
    """
    Test fetch_alpaca_data_async when the Alpaca API key is missing or the client is None.
    """
    data_fetch_utils_fixture.alpaca_api = None
    result = await data_fetch_utils_fixture.fetch_alpaca_data_async(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2023-01-31",
        interval="1Day"
    )
    assert result.empty, "Expected an empty DataFrame when Alpaca API key/client is uninitialized."


@pytest.mark.asyncio
async def test_fetch_with_retries_success(data_fetch_utils_fixture):
    """
    Test fetch_with_retries with a successful request.
    """
    url = "https://example.com/api/data"
    headers = {}
    retries = 3
    mock_response = {"data": "success"}

    with aioresponses() as mocked:
        mocked.get(url, payload=mock_response, status=200)
        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, headers, session, retries=retries)

    assert result == mock_response, "Expected successful response after retries"


@pytest.mark.asyncio
async def test_fetch_with_retries_failure(data_fetch_utils_fixture):
    """
    Test fetch_with_retries when all retries fail.
    """
    url = "https://example.com/api/failure"
    headers = {}
    retries = 3

    with aioresponses() as mocked:
        mocked.get(url, status=500, repeat=True)
        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, headers, session, retries=retries)

    assert result is None, "Expected None after exceeding retry attempts"


@pytest.mark.asyncio
async def test_fetch_with_retries_mixed_failures(data_fetch_utils_fixture):
    """
    Test fetch_with_retries where initial attempts fail and a subsequent attempt succeeds.
    """
    url = "https://example.com/api/mixed"
    headers = {}
    retries = 3
    mock_response = {"data": "eventual success"}

    with aioresponses() as mocked:
        # First two attempts fail with 500, third attempt succeeds
        mocked.get(url, status=500)
        mocked.get(url, status=500)
        mocked.get(url, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, headers, session, retries=retries)

    assert result == mock_response, "Expected successful response on the third attempt."


@pytest.mark.asyncio
async def test_fetch_with_retries_empty_response(data_fetch_utils_fixture):
    """
    Test fetch_with_retries with an empty or None API response.
    """
    url = "https://example.com/empty-response"
    with patch.object(aiohttp.ClientSession, "get", return_value=None):
        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, {}, session, retries=3)

    assert result is None, "Expected None for empty or no response from the server."


@pytest.mark.asyncio
async def test_concurrency_fetch_many_symbols(data_fetch_utils_fixture):
    """
    Test concurrency or large symbol list for a single data source (Alpha Vantage).
    (If your code actually merges multiple sources here, move this to integration.)
    """
    symbols = [f'SYM{i}' for i in range(100)]
    data_sources = ["Alpha Vantage"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    with aioresponses() as mocked:
        for symbol in symbols:
            url_pattern = re.compile(
                rf"https://(?:www\.)?alphavantage\.co/query\?(?=.*function=TIME_SERIES_DAILY)(?=.*symbol={symbol})(?=.*apikey=test_alphavantage_key).*"
            )
            mocked.get(
                url_pattern,
                payload={
                    "Time Series (Daily)": {
                        "2023-01-03": {
                            "1. open": "100.0",
                            "2. high": "105.0",
                            "3. low": "95.0",
                            "4. close": "102.0",
                            "5. volume": "1000000"
                        }
                    }
                },
                status=200
            )

        # This calls fetch_data_for_multiple_symbols on a single data source or an equivalent concurrency approach
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols, data_sources, start_date, end_date, interval
        )

    for symbol in symbols:
        assert symbol in result, f"{symbol} not in result."
        alpha_vantage_df = result[symbol]["Alpha Vantage"]
        assert not alpha_vantage_df.empty, f"Expected data for {symbol}."
