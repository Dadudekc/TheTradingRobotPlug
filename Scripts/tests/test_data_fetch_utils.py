# Scripts/tests/test_data_fetch_core.py

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, date
import re
from aioresponses import aioresponses  
import aiohttp
import numpy as np
import os

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols(data_fetch_utils_fixture):
    symbols = ['AAPL', 'GOOGL']
    data_sources = ["Alpha Vantage"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock Alpha Vantage data with DatetimeIndex
    mock_alpha_vantage_df_aapl = pd.DataFrame([
        {
            'open': 130.0,
            'high': 132.0,
            'low': 129.0,
            'close': 131.0,
            'volume': 1000000
        }
    ], index=pd.to_datetime(['2023-01-03'])).rename_axis('date')

    mock_alpha_vantage_df_googl = pd.DataFrame([
        {
            'open': 140.0,
            'high': 142.0,
            'low': 139.0,
            'close': 141.0,
            'volume': 2000000
        }
    ], index=pd.to_datetime(['2023-01-03'])).rename_axis('date')

    with aioresponses(assert_all_requests_are_fired=False) as mocked:
        for symbol, df in [('AAPL', mock_alpha_vantage_df_aapl), ('GOOGL', mock_alpha_vantage_df_googl)]:
            url_pattern = re.compile(
                rf"https://(?:www\.)?alphavantage\.co/query\?(?=.*function=TIME_SERIES_DAILY)(?=.*symbol={symbol})(?=.*apikey=test_alphavantage_key).*"
            )
            mocked.get(
                url_pattern,
                payload={
                    "Time Series (Daily)": {
                        "2023-01-03": {
                            "1. open": str(df.iloc[0]['open']),
                            "2. high": str(df.iloc[0]['high']),
                            "3. low": str(df.iloc[0]['low']),
                            "4. close": str(df.iloc[0]['close']),
                            "5. volume": str(int(df.iloc[0]['volume']))
                        }
                    }
                },
                status=200,
                repeat=True
            )

        # Define a side effect function to return different DataFrames based on the symbol
        def get_bars_side_effect(symbol, *args, **kwargs):
            if symbol == 'AAPL':
                return MagicMock(df=pd.DataFrame())  # Assuming no Alpaca data for simplicity
            elif symbol == 'GOOGL':
                return MagicMock(df=pd.DataFrame())  # Assuming no Alpaca data for simplicity
            else:
                return MagicMock(df=pd.DataFrame())

        with patch.object(data_fetch_utils_fixture.alpaca_api, 'get_bars', side_effect=get_bars_side_effect):
            result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
                symbols,
                data_sources,
                start_date,
                end_date,
                interval
            )

    # Normalize the index types to DatetimeIndex
    result['AAPL']['Alpha Vantage'].index = pd.to_datetime(result['AAPL']['Alpha Vantage'].index)
    mock_alpha_vantage_df_aapl.index = pd.to_datetime(mock_alpha_vantage_df_aapl.index)

    # Perform the assertion
    pd.testing.assert_frame_equal(result['AAPL']['Alpha Vantage'], mock_alpha_vantage_df_aapl)
    pd.testing.assert_frame_equal(result['GOOGL']['Alpha Vantage'], mock_alpha_vantage_df_googl)

@pytest.mark.asyncio
async def test_fetch_news_data_async_success(data_fetch_utils_fixture):
    symbol = 'AAPL'
    mock_response = {
        'status': 'ok',
        'totalResults': 1,
        'articles': [
            {
                'publishedAt': '2023-04-14T12:34:56Z',
                'title': 'Apple Releases New Product',
                'description': 'Apple has released a new product...',
                'source': {'name': 'TechCrunch'},
                'url': 'https://techcrunch.com/apple-new-product'
            }
        ]
    }

    with patch('requests.get') as mock_get, patch('Scripts.Utilities.data_fetch_utils.TextBlob') as mock_textblob:
        # Mocking the HTTP request
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_get.return_value = mock_resp

        # Mocking sentiment analysis to return the same value as production
        mock_blob_instance = MagicMock()
        mock_blob_instance.sentiment.polarity = 0.13636363636363635
        mock_textblob.return_value = mock_blob_instance

        result = await data_fetch_utils_fixture.fetch_news_data_async(symbol, page_size=1)

    expected_df = pd.DataFrame([
        {
            'date': pd.to_datetime('2023-04-14').date(),
            'headline': 'Apple Releases New Product',
            'content': 'Apple has released a new product...',
            'symbol': 'AAPL',
            'source': 'TechCrunch',
            'url': 'https://techcrunch.com/apple-new-product',
            'sentiment': 0.13636363636363635  # Match the mocked sentiment
        }
    ]).set_index('date')
    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_finnhub_quote_success(data_fetch_utils_fixture):
    symbol = 'AAPL'
    # Regex pattern to match the exact URL, allow additional parameters after apiKey
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
    symbol = 'AAPL'
    url_pattern = re.compile(rf"https://finnhub\.io/api/v1/quote\?symbol={symbol}&token=test_finnhub_key.*")
    
    with aioresponses() as mocked:
        mocked.get(url_pattern, status=500)  # Simulate server error
        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_quote(symbol, session)
    # Ensure result is a DataFrame even on failure
    assert isinstance(result, pd.DataFrame)
    assert result.empty

@pytest.mark.asyncio
async def test_fetch_stock_data_async(data_fetch_utils_fixture):
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Expected index with date objects
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
        })
        result = await data_fetch_utils_fixture.fetch_stock_data_async(ticker, start_date, end_date, interval)

    pd.testing.assert_frame_equal(result, expected_data)

@pytest.mark.asyncio
async def test_fetch_alpaca_data_async_success(data_fetch_utils_fixture):
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
    
    # Build expected DataFrame independently
    expected_df = mock_bars_df.rename(columns={'timestamp': 'date', 'trade_count': 'volume'}).copy()
    expected_df['date'] = pd.to_datetime(expected_df['date']).dt.date
    expected_df.set_index('date', inplace=True)
    # No need to drop 'date' column since it's already set as index
    expected_df = expected_df.drop(columns=['date'], errors='ignore')
    
    # Ensure 'symbol' is part of the columns and 'date' is the index
    expected_columns = ['open', 'high', 'low', 'close', 'volume', 'symbol']
    assert list(result.columns) == expected_columns, f"Expected columns {list(expected_columns)}, got {list(result.columns)}"
    
    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_alpaca_data_async_failure(data_fetch_utils_fixture):
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1Day'

    with patch.object(data_fetch_utils_fixture.alpaca_api, 'get_bars', side_effect=Exception("API Error")):
        result = await data_fetch_utils_fixture.fetch_alpaca_data_async(symbol, start_date, end_date, interval)

    # Ensure result is a DataFrame even on failure
    assert isinstance(result, pd.DataFrame)
    assert result.empty

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_invalid_symbol(data_fetch_utils_fixture):
    symbols = ['INVALID']
    data_sources = ["Alpha Vantage"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Create an empty DataFrame for invalid symbol
    mock_alpha_vantage_df_invalid = pd.DataFrame()

    with aioresponses() as mocked:
        # No valid results; let's assume the response is empty
        url_pattern = re.compile(
            rf"https://(?:www\.)?alphavantage\.co/query\?(?=.*function=TIME_SERIES_DAILY)(?=.*symbol=INVALID)(?=.*apikey=test_alphavantage_key).*"
        )
        mocked.get(
            url_pattern,
            payload={"Time Series (Daily)": {}},  # Empty time series
            status=200
        )

        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols,
            data_sources,
            start_date,
            end_date,
            interval
        )

    # Assertions
    assert 'INVALID' in result, "Symbol 'INVALID' should still appear in the result."
    assert 'Alpha Vantage' in result['INVALID'], "'Alpha Vantage' data source not found in the result."
    # The returned DataFrame should be empty since no data
    returned_df = result['INVALID']['Alpha Vantage']
    assert returned_df.empty, "Expected an empty DataFrame for an invalid symbol."

@pytest.mark.asyncio
async def test_fetch_data_for_mixed_symbols(data_fetch_utils_fixture):
    symbols = ['AAPL', 'INVALID']
    data_sources = ["Alpha Vantage"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock Alpha Vantage data for 'AAPL'
    mock_alpha_vantage_df_aapl = pd.DataFrame([{
        'open': 130.0,
        'high': 132.0,
        'low': 129.0,
        'close': 131.0,
        'volume': 1000000
    }], index=pd.to_datetime(['2023-01-03'])).rename_axis('date')

    with aioresponses() as mocked:
        # Mock data for valid symbol
        url_pattern_aapl = re.compile(
            rf"https://(?:www\.)?alphavantage\.co/query\?(?=.*function=TIME_SERIES_DAILY)(?=.*symbol=AAPL)(?=.*apikey=test_alphavantage_key).*"
        )
        mocked.get(
            url_pattern_aapl,
            payload={
                "Time Series (Daily)": {
                    "2023-01-03": {
                        "1. open": "130.0",
                        "2. high": "132.0",
                        "3. low": "129.0",
                        "4. close": "131.0",
                        "5. volume": "1000000"
                    }
                }
            },
            status=200
        )

        # Mock data for invalid symbol
        url_pattern_invalid = re.compile(
            rf"https://(?:www\.)?alphavantage\.co/query\?(?=.*function=TIME_SERIES_DAILY)(?=.*symbol=INVALID)(?=.*apikey=test_alphavantage_key).*"
        )
        mocked.get(
            url_pattern_invalid,
            payload={"Time Series (Daily)": {}},  # Empty response
            status=200
        )

        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols,
            data_sources,
            start_date,
            end_date,
            interval
        )

    # Normalize the index types to ensure compatibility
    result['AAPL']['Alpha Vantage'].index = pd.to_datetime(result['AAPL']['Alpha Vantage'].index)
    mock_alpha_vantage_df_aapl.index = pd.to_datetime(mock_alpha_vantage_df_aapl.index)

    # Assertions for AAPL
    assert 'AAPL' in result, "AAPL data should be in the result."
    assert 'Alpha Vantage' in result['AAPL'], "Alpha Vantage data source should be present for AAPL."
    pd.testing.assert_frame_equal(result['AAPL']['Alpha Vantage'], mock_alpha_vantage_df_aapl)

    # Assertions for INVALID
    assert 'INVALID' in result, "INVALID symbol should still appear in the result."
    assert 'Alpha Vantage' in result['INVALID'], "Alpha Vantage data source should be present for INVALID."
    assert result['INVALID']['Alpha Vantage'].empty, "Expected an empty DataFrame for INVALID symbol."

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

    # Verify the result after retries
    assert result == mock_response, "Expected successful response after retries."

@pytest.mark.asyncio
async def test_fetch_with_retries_failure(data_fetch_utils_fixture):
    """
    Test fetch_with_retries when all retries fail.
    """
    url = "https://example.com/api/data"
    headers = {}
    retries = 3

    with aioresponses() as mocked:
        # All attempts return 500 (server error)
        mocked.get(url, status=500, repeat=True)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, headers, session, retries=retries)

    # Verify the result is None after exceeding retries
    assert result is None, "Expected None when all retries fail."

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_missing_metric(data_fetch_utils_fixture):
    symbol = "AAPL"
    mock_response = {}  # Missing "metric"

    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    # Expect an empty DataFrame with 'date_fetched' column
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame when 'metric' is missing."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' column in the DataFrame."

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_malformed_data(data_fetch_utils_fixture):
    symbol = "AAPL"
    mock_response = {
        "metric": "invalid_data"
    }

    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    # Expect an empty DataFrame due to malformed data
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame when data is malformed."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' column in the DataFrame."

@pytest.mark.asyncio
async def test_fetch_alphavantage_data_empty_time_series(data_fetch_utils_fixture):
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

    # Ensure result is an empty DataFrame
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame."

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

    # Validate the result is an empty DataFrame
    assert result.empty, "Expected an empty DataFrame for missing 'Time Series (Daily)'"

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

    # Validate the result is an empty DataFrame
    assert result.empty, "Expected an empty DataFrame for API failure"

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

    # Verify the result after retries
    assert result == mock_response, "Expected successful response after retries"

@pytest.mark.asyncio
async def test_fetch_with_retries_failure(data_fetch_utils_fixture):
    """
    Test fetch_with_retries when all retries fail.
    """
    url = "https://example.com/api/data"
    headers = {}
    retries = 3

    with aioresponses() as mocked:
        # All attempts return 500 (server error)
        mocked.get(url, status=500, repeat=True)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, headers, session, retries=retries)

    # Verify the result is None after exceeding retries
    assert result is None, "Expected None when all retries fail"

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_missing_metric(data_fetch_utils_fixture):
    symbol = "AAPL"
    mock_response = {}  # Missing "metric"

    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    # Expect an empty DataFrame with 'date_fetched' column
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame when 'metric' is missing."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' column in the DataFrame."

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_malformed_data(data_fetch_utils_fixture):
    symbol = "AAPL"
    mock_response = {
        "metric": "invalid_data"
    }

    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    # Expect an empty DataFrame due to malformed data
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame when data is malformed."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' column in the DataFrame."

@pytest.mark.asyncio
async def test_fetch_alphavantage_data_empty_time_series(data_fetch_utils_fixture):
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

    # Ensure result is an empty DataFrame
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame."

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

    # Validate the result is an empty DataFrame
    assert result.empty, "Expected an empty DataFrame for missing 'Time Series (Daily)'"

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

    # Validate the result is an empty DataFrame
    assert result.empty, "Expected an empty DataFrame for API failure"

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

    # Verify the result after retries
    assert result == mock_response, "Expected successful response after retries"

@pytest.mark.asyncio
async def test_fetch_with_retries_failure(data_fetch_utils_fixture):
    """
    Test fetch_with_retries when all retries fail.
    """
    url = "https://example.com/api/data"
    headers = {}
    retries = 3

    with aioresponses() as mocked:
        # All attempts return 500 (server error)
        mocked.get(url, status=500, repeat=True)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, headers, session, retries=retries)

    # Verify the result is None after exceeding retries
    assert result is None, "Expected None when all retries fail"

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_missing_metric(data_fetch_utils_fixture):
    symbol = "AAPL"
    mock_response = {}  # Missing "metric"

    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    # Expect an empty DataFrame with 'date_fetched' column
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame when 'metric' is missing."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' column in the DataFrame."

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_malformed_data(data_fetch_utils_fixture):
    symbol = "AAPL"
    mock_response = {
        "metric": "invalid_data"
    }

    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    # Expect an empty DataFrame due to malformed data
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame when data is malformed."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' column in the DataFrame."

@pytest.mark.asyncio
async def test_fetch_alphavantage_data_empty_time_series(data_fetch_utils_fixture):
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

    # Ensure result is an empty DataFrame
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame."

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

    # Validate the result is an empty DataFrame
    assert result.empty, "Expected an empty DataFrame for missing 'Time Series (Daily)'"

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

    # Validate the result is an empty DataFrame
    assert result.empty, "Expected an empty DataFrame for API failure"

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

    # Verify the result after retries
    assert result == mock_response, "Expected successful response after retries"

@pytest.mark.asyncio
async def test_fetch_with_retries_failure(data_fetch_utils_fixture):
    """
    Test fetch_with_retries when all retries fail.
    """
    url = "https://example.com/api/data"
    headers = {}
    retries = 3

    with aioresponses() as mocked:
        # All attempts return 500 (server error)
        mocked.get(url, status=500, repeat=True)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, headers, session, retries=retries)

    # Verify the result is None after exceeding retries
    assert result is None, "Expected None when all retries fail"

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_missing_metric(data_fetch_utils_fixture):
    symbol = "AAPL"
    mock_response = {}  # Missing "metric"

    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    # Expect an empty DataFrame with 'date_fetched' column
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame when 'metric' is missing."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' column in the DataFrame."

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_malformed_data(data_fetch_utils_fixture):
    symbol = "AAPL"
    mock_response = {
        "metric": "invalid_data"
    }

    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    # Expect an empty DataFrame due to malformed data
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame when data is malformed."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' column in the DataFrame."

@pytest.mark.asyncio
async def test_fetch_alphavantage_data_empty_time_series(data_fetch_utils_fixture):
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

    # Ensure result is an empty DataFrame
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame."

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

    # Validate the result is an empty DataFrame
    assert result.empty, "Expected an empty DataFrame for missing 'Time Series (Daily)'"

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

    # Validate the result is an empty DataFrame
    assert result.empty, "Expected an empty DataFrame for API failure"

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

    # Verify the result after retries
    assert result == mock_response, "Expected successful response after retries"

@pytest.mark.asyncio
async def test_fetch_with_retries_failure(data_fetch_utils_fixture):
    """
    Test fetch_with_retries when all retries fail.
    """
    url = "https://example.com/api/data"
    headers = {}
    retries = 3

    with aioresponses() as mocked:
        # All attempts return 500 (server error)
        mocked.get(url, status=500, repeat=True)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, headers, session, retries=retries)

    # Verify the result is None after exceeding retries
    assert result is None, "Expected None when all retries fail"

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_missing_metric(data_fetch_utils_fixture):
    symbol = "AAPL"
    mock_response = {}  # Missing "metric"

    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    # Expect an empty DataFrame with 'date_fetched' column
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame when 'metric' is missing."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' column in the DataFrame."

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_malformed_data(data_fetch_utils_fixture):
    symbol = "AAPL"
    mock_response = {
        "metric": "invalid_data"
    }

    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    # Expect an empty DataFrame due to malformed data
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame when data is malformed."
    assert "date_fetched" in result.columns, "Expected 'date_fetched' column in the DataFrame."

@pytest.mark.asyncio
async def test_fetch_alphavantage_data_empty_time_series(data_fetch_utils_fixture):
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

    # Ensure result is an empty DataFrame
    assert isinstance(result, pd.DataFrame), "Expected a DataFrame."
    assert result.empty, "Expected an empty DataFrame."

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

    # Validate the result is an empty DataFrame
    assert result.empty, "Expected an empty DataFrame for missing 'Time Series (Daily)'"

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

    # Validate the result is an empty DataFrame
    assert result.empty, "Expected an empty DataFrame for API failure"

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

    # Verify the result after retries
    assert result == mock_response, "Expected successful response after retries"

@pytest.mark.asyncio
async def test_fetch_with_retries_failure(data_fetch_utils_fixture):
    """
    Test fetch_with_retries when all retries fail.
    """
    url = "https://example.com/api/data"
    headers = {}
    retries = 3

    with aioresponses() as mocked:
        # All attempts return 500 (server error)
        mocked.get(url, status=500, repeat=True)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, headers, session, retries=retries)

    # Verify the result is None after exceeding retries
    assert result is None, "Expected None when all retries fail"

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_success(data_fetch_utils_fixture):
    symbol = "TSLA"
    mock_response = {
        "metric": {
            "52WeekHigh": 123,
            "52WeekLow": 456,
            "MarketCapitalization": 789,
            "P/E": 25.5
        }
    }

    # Exact URL without regex
    url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token=test_finnhub_key"

    # Fixed timestamp for consistency
    fixed_time = pd.Timestamp('2025-01-10 03:45:03', tz="UTC").floor("s")

    with patch("pandas.Timestamp.utcnow", return_value=fixed_time):
        with aioresponses() as mocked:
            mocked.get(url, payload=mock_response, status=200)
            print(f"Mocking exact URL: {url}")

            async with aiohttp.ClientSession() as session:
                result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)
                print(f"Result from fetch_finnhub_metrics: {result}")

    # Define the expected DataFrame
    expected_df = pd.DataFrame([{
        "52WeekHigh": 123,
        "52WeekLow": 456,
        "MarketCapitalization": 789,
        "P/E": 25.5,
        "date_fetched": fixed_time
    }]).set_index("date_fetched")

    # Normalize timezones to naive datetime64[ns]
    result = result.reset_index()
    expected_df = expected_df.reset_index()
    result["date_fetched"] = pd.to_datetime(result["date_fetched"]).dt.tz_localize(None)
    expected_df["date_fetched"] = pd.to_datetime(expected_df["date_fetched"]).dt.tz_localize(None)

    # Compare DataFrames
    assert not result.empty, "Expected a non-empty DataFrame."
    pd.testing.assert_frame_equal(result, expected_df)

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

    # Exact URL without regex
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey=test_alphavantage_key"

    with aioresponses() as mocked:
        mocked.get(url, payload=mock_response, status=200)
        print(f"Mocking exact URL: {url}")

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_alphavantage_data(symbol, session, '2023-01-01', '2023-01-31')
            print(f"Result from fetch_alphavantage_data: {result}")

    # Define the expected DataFrame
    expected_df = pd.DataFrame([
        {
            'open': 130.0,
            'high': 132.0,
            'low': 129.0,
            'close': 131.0,
            'volume': 1000000
        }
    ], index=pd.to_datetime(['2023-01-03'])).rename_axis('date')

    # Validate the result
    assert not result.empty, "Expected non-empty DataFrame."
    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_parse_alphavantage_data_valid(data_fetch_utils_fixture):
    # Directly test the _parse_alphavantage_data method
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

    # Access the protected method
    parsed_df = data_fetch_utils_fixture._parse_alphavantage_data(data)
    pd.testing.assert_frame_equal(parsed_df, expected_df)


@pytest.mark.asyncio
async def test_fetch_news_data_async_failure(data_fetch_utils_fixture):
    symbol = 'AAPL'
    # Use lookaheads to match required params in any order:
    url_pattern = re.compile(
        rf"https://newsapi\.org/v2/everything\?(?=.*q={symbol})(?=.*pageSize=5)(?=.*apiKey=test_newsapi_key).*"
    )

    with aioresponses(assert_all_requests_are_fired=True) as mocked:
        # Simulate rate limiting
        mocked.get(url_pattern, status=429)

        result = await data_fetch_utils_fixture.fetch_news_data_async(symbol, page_size=5)

    assert isinstance(result, pd.DataFrame)
    assert result.empty, "Expected an empty DataFrame on rate limiting."



@pytest.mark.asyncio
async def test_fetch_polygon_data_success(data_fetch_utils_fixture):
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
async def test_logging_setup(data_fetch_utils_fixture):
    """
    Test that logging is properly set up and logs messages as expected.
    """
    with patch.object(data_fetch_utils_fixture.logger, 'info') as mock_info:
        data_fetch_utils_fixture.logger.info("Test log message.")
        mock_info.assert_called_with("Test log message.")

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
async def test_fetch_stock_data_async_invalid_interval(data_fetch_utils_fixture):
    """
    Test fetch_stock_data_async with an invalid interval.
    """
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '5d'  # Assuming '5d' is unsupported

    with patch('yfinance.download', side_effect=ValueError("Unsupported interval")):
        result = await data_fetch_utils_fixture.fetch_stock_data_async(ticker, start_date, end_date, interval)

    assert isinstance(result, pd.DataFrame), "Expected a DataFrame even on failure."
    assert result.empty, "Expected an empty DataFrame when an invalid interval is used."

@pytest.mark.asyncio
async def test_fetch_alpaca_data_async_invalid_date_range(data_fetch_utils_fixture):
    """
    Test fetch_alpaca_data_async with an invalid date range.
    """
    symbol = 'AAPL'
    start_date = '2023-02-30'  # Invalid date
    end_date = '2023-01-31'    # End date before start date
    interval = '1Day'

    with patch.object(data_fetch_utils_fixture.alpaca_api, 'get_bars', side_effect=ValueError("Invalid date range")):
        result = await data_fetch_utils_fixture.fetch_alpaca_data_async(symbol, start_date, end_date, interval)

    assert isinstance(result, pd.DataFrame), "Expected a DataFrame even on failure."
    assert result.empty, "Expected an empty DataFrame for invalid date ranges."

@pytest.mark.asyncio
async def test_fetch_news_data_async_missing_fields(data_fetch_utils_fixture):
    """
    Test fetch_news_data_async with articles missing some fields.
    """
    symbol = 'AAPL'
    mock_response = {
        'status': 'ok',
        'totalResults': 1,
        'articles': [
            {
                'publishedAt': '2023-04-14T12:34:56Z',
                'title': 'Apple Releases New Product',
                # Missing 'description', 'source', 'url'
            }
        ]
    }

    with patch('requests.get') as mock_get, patch('Scripts.Utilities.data_fetch_utils.TextBlob') as mock_textblob:
        # Mocking the HTTP request
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_get.return_value = mock_resp

        # Mocking sentiment analysis to handle missing description
        mock_blob_instance = MagicMock()
        mock_blob_instance.sentiment.polarity = 0.0
        mock_textblob.return_value = mock_blob_instance

        result = await data_fetch_utils_fixture.fetch_news_data_async(symbol, page_size=1)

    expected_df = pd.DataFrame([
        {
            'date': pd.to_datetime('2023-04-14').date(),
            'headline': 'Apple Releases New Product',
            'content': '',  # Default to empty string
            'symbol': 'AAPL',
            'source': '',    # Default to empty string
            'url': '',       # Default to empty string
            'sentiment': 0.0
        }
    ]).set_index('date')

    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_finnhub_metrics_additional_fields(data_fetch_utils_fixture):
    """
    Test fetch_finnhub_metrics with additional unexpected fields in the metrics.
    """
    symbol = "AAPL"
    mock_response = {
        "metric": {
            "52WeekHigh": 123,
            "52WeekLow": 456,
            "MarketCapitalization": 789,
            "P/E": 25.5,
            "ExtraMetric": 999  # Unexpected field
        }
    }

    url_pattern = re.compile(
        rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol={symbol})(?=.*metric=all)(?=.*token=test_finnhub_key).*"
    )

    with aioresponses() as mocked:
        mocked.get(url_pattern, payload=mock_response, status=200)
        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_metrics(symbol, session)

    expected_df = pd.DataFrame([{
        "52WeekHigh": 123,
        "52WeekLow": 456,
        "MarketCapitalization": 789,
        "P/E": 25.5,
        "date_fetched": pd.Timestamp.utcnow().floor("s")
    }]).set_index("date_fetched")

    # Normalize timezones to naive datetime64[ns]
    result = result.reset_index()
    expected_df = expected_df.reset_index()
    result["date_fetched"] = pd.to_datetime(result["date_fetched"]).dt.tz_localize(None)
    expected_df["date_fetched"] = pd.to_datetime(expected_df["date_fetched"]).dt.tz_localize(None)

    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_mixed_symbols(data_fetch_utils_fixture):
    symbols = ['AAPL', 'INVALID']
    data_sources = ["Alpha Vantage", "Finnhub"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock Alpha Vantage for AAPL
    mock_alpha_vantage_df_aapl = pd.DataFrame([
        {
            'open': 130.0,
            'high': 132.0,
            'low': 129.0,
            'close': 131.0,
            'volume': 1000000
        }
    ], index=pd.to_datetime(['2023-01-03'])).rename_axis('date')

    # Mock Alpha Vantage response for INVALID symbol (empty)
    mock_alpha_vantage_response_invalid = {
        "Time Series (Daily)": {}
    }

    # Mock Finnhub metrics for AAPL
    mock_finnhub_response_aapl = {
        "metric": {
            "52WeekHigh": 123,
            "52WeekLow": 456,
            "MarketCapitalization": 789,
            "P/E": 25.5
        }
    }

    # Mock Finnhub metrics for INVALID symbol (missing metric)
    mock_finnhub_response_invalid = {}

    with aioresponses() as mocked:
        # Mock Alpha Vantage for AAPL
        url_pattern_aapl = re.compile(
            rf"https://(?:www\.)?alphavantage\.co/query\?(?=.*function=TIME_SERIES_DAILY)(?=.*symbol=AAPL)(?=.*apikey=test_alphavantage_key).*"
        )
        mocked.get(
            url_pattern_aapl,
            payload={
                "Time Series (Daily)": {
                    "2023-01-03": {
                        "1. open": "130.0",
                        "2. high": "132.0",
                        "3. low": "129.0",
                        "4. close": "131.0",
                        "5. volume": "1000000"
                    }
                }
            },
            status=200
        )

        # Mock Alpha Vantage for INVALID
        url_pattern_invalid_av = re.compile(
            rf"https://(?:www\.)?alphavantage\.co/query\?(?=.*function=TIME_SERIES_DAILY)(?=.*symbol=INVALID)(?=.*apikey=test_alphavantage_key).*"
        )
        mocked.get(
            url_pattern_invalid_av,
            payload=mock_alpha_vantage_response_invalid,
            status=200
        )

        # Mock Finnhub for AAPL
        url_pattern_finnhub_aapl = re.compile(
            rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol=AAPL)(?=.*metric=all)(?=.*token=test_finnhub_key).*"
        )
        mocked.get(
            url_pattern_finnhub_aapl,
            payload=mock_finnhub_response_aapl,
            status=200
        )

        # Mock Finnhub for INVALID
        url_pattern_finnhub_invalid = re.compile(
            rf"https://finnhub\.io/api/v1/stock/metric\?(?=.*symbol=INVALID)(?=.*metric=all)(?=.*token=test_finnhub_key).*"
        )
        mocked.get(
            url_pattern_finnhub_invalid,
            payload=mock_finnhub_response_invalid,
            status=200
        )

        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols,
            data_sources,
            start_date,
            end_date,
            interval
        )

    # Assertions for AAPL
    assert 'AAPL' in result, "AAPL should be in the result."
    assert 'Alpha Vantage' in result['AAPL'], "Alpha Vantage data for AAPL missing."
    pd.testing.assert_frame_equal(result['AAPL']['Alpha Vantage'], mock_alpha_vantage_df_aapl)
    assert 'Finnhub' in result['AAPL'], "Finnhub data for AAPL missing."
    assert not result['AAPL']['Finnhub'].empty, "Finnhub data for AAPL should not be empty."

    # Assertions for INVALID
    assert 'INVALID' in result, "INVALID should be in the result."
    assert 'Alpha Vantage' in result['INVALID'], "Alpha Vantage data for INVALID missing."
    assert result['INVALID']['Alpha Vantage'].empty, "Alpha Vantage data for INVALID should be empty."
    assert 'Finnhub' in result['INVALID'], "Finnhub data for INVALID missing."
    assert result['INVALID']['Finnhub'].empty, "Finnhub data for INVALID should be empty."

@pytest.mark.asyncio
async def test_parse_alphavantage_data_missing_keys(data_fetch_utils_fixture):
    """
    Test _parse_alphavantage_data with missing keys in the response.
    """
    data = {
        "Time Series (Daily)": {
            "2023-01-03": {
                "1. open": "130.0",
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

    assert isinstance(parsed_df, pd.DataFrame), "Expected a DataFrame."
    assert not parsed_df.empty, "Expected a non-empty DataFrame."
    assert "52WeekHigh" in parsed_df.columns, "Missing '52WeekHigh' column."
    assert "52WeekLow" in parsed_df.columns, "Missing '52WeekLow' column."
    assert "MarketCapitalization" in parsed_df.columns, "Missing 'MarketCapitalization' column."
    assert "P/E" in parsed_df.columns, "Missing 'P/E' column."
    assert parsed_df.index.name == "date_fetched", "Missing or incorrect 'date_fetched' index."


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



def test_get_project_root_failure(monkeypatch):
    from Scripts.Utilities.data_fetch_utils import get_project_root

    # Simulate no .env file found
    monkeypatch.setattr('Scripts.Utilities.data_fetch_utils.Path.exists', lambda _: False)

    with pytest.raises(RuntimeError):
        get_project_root()


@pytest.mark.asyncio
async def test_fetch_with_retries_unexpected_status(data_fetch_utils_fixture):
    url = "https://example.com/api/data"
    headers = {}
    retries = 3

    with aioresponses() as mocked:
        # Simulate unexpected status code 403
        mocked.get(url, status=403, repeat=True)

        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, headers, session, retries=retries)

    assert result is None, "Expected None for unexpected status codes."


@pytest.mark.asyncio
async def test_fetch_alpaca_data_async_uninitialized_client(data_fetch_utils_fixture):
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1Day'

    # Set Alpaca client to None
    data_fetch_utils_fixture.alpaca_api = None

    result = await data_fetch_utils_fixture.fetch_alpaca_data_async(symbol, start_date, end_date, interval)

    assert result.empty, "Expected an empty DataFrame when Alpaca client is uninitialized."



@pytest.mark.asyncio
async def test_fetch_news_data_async_unexpected_fields(data_fetch_utils_fixture):
    symbol = 'AAPL'
    mock_response = {
        'status': 'ok',
        'totalResults': 1,
        'articles': [
            {
                'publishedAt': '2023-04-14T12:34:56Z',
                'title': 'Apple Releases New Product',
                'unexpected_field': 'unexpected_value'
            }
        ]
    }

    with patch('requests.get') as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_get.return_value = mock_resp

        result = await data_fetch_utils_fixture.fetch_news_data_async(symbol, page_size=1)

    assert 'unexpected_field' not in result.columns, "Unexpected field should not appear in the DataFrame."

@pytest.mark.asyncio
async def test_fetch_polygon_data_unexpected_keys(data_fetch_utils_fixture):
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
    data = {
        "results": [
            {
                "t": "invalid_timestamp",  # Invalid timestamp
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


def test_initialize_alpaca_invalid_config(monkeypatch):
    from Scripts.Utilities.data_fetch_utils import initialize_alpaca

    # Missing keys
    monkeypatch.delenv('ALPACA_API_KEY', raising=False)
    monkeypatch.delenv('ALPACA_SECRET_KEY', raising=False)
    assert initialize_alpaca() is None, "Expected None with missing Alpaca keys."

    # Invalid URL
    monkeypatch.setenv('ALPACA_API_KEY', 'dummy_key')
    monkeypatch.setenv('ALPACA_SECRET_KEY', 'dummy_secret')
    monkeypatch.setenv('ALPACA_BASE_URL', 'invalid_url')
    with pytest.raises(ValueError, match="Invalid URL"):
        initialize_alpaca()

    # Valid configuration
    monkeypatch.setenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    client = initialize_alpaca()
    assert client is not None, "Expected a valid Alpaca client with correct configuration."


@pytest.mark.asyncio
async def test_fetch_data_large_symbol_list(data_fetch_utils_fixture):
    """
    Test fetch_data_for_multiple_symbols with a large number of symbols.
    """
    symbols = [f'SYM{i}' for i in range(100)]  # Generate 100 mock symbols
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    data_sources = ["Alpha Vantage"]

    result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
        symbols,
        data_sources,
        start_date,
        end_date
    )
    assert len(result) == len(symbols), "Expected results for all symbols."
    for symbol in symbols:
        assert symbol in result, f"Missing result for symbol {symbol}."



@pytest.mark.asyncio
async def test_fetch_alpaca_no_key(data_fetch_utils_fixture):
    """
    Test fetch_alpaca_data_async when Alpaca API key is missing.
    """
    data_fetch_utils_fixture.alpaca_api = None
    result = await data_fetch_utils_fixture.fetch_alpaca_data_async(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2023-01-31",
        interval="1Day"
    )
    assert result.empty, "Expected an empty DataFrame when Alpaca API key is missing."


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

    assert isinstance(parsed_df, pd.DataFrame), "Expected a DataFrame."
    assert not parsed_df.empty, "Expected a non-empty DataFrame."
    assert "52WeekHigh" in parsed_df.columns, "Missing '52WeekHigh' column."
    assert "52WeekLow" in parsed_df.columns, "Missing '52WeekLow' column."
    assert "MarketCapitalization" in parsed_df.columns, "Missing 'MarketCapitalization' column."
    assert "P/E" in parsed_df.columns, "Missing 'P/E' column."
    assert parsed_df.index.name == "date_fetched", "Missing or incorrect 'date_fetched' index."


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


def test_parse_polygon_data_invalid_timestamps(data_fetch_utils_fixture):
    data = {
        "results": [
            {
                "t": "invalid_timestamp",  # Invalid timestamp
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
async def test_fetch_polygon_data_unexpected_keys(data_fetch_utils_fixture):
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
    data = {
        "results": [
            {
                "t": "invalid_timestamp",  # Invalid timestamp
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


def test_initialize_alpaca_invalid_config(monkeypatch):
    from Scripts.Utilities.data_fetch_utils import initialize_alpaca

    # Missing keys
    monkeypatch.delenv('ALPACA_API_KEY', raising=False)
    monkeypatch.delenv('ALPACA_SECRET_KEY', raising=False)
    assert initialize_alpaca() is None, "Expected None with missing Alpaca keys."

    # Invalid URL
    monkeypatch.setenv('ALPACA_API_KEY', 'dummy_key')
    monkeypatch.setenv('ALPACA_SECRET_KEY', 'dummy_secret')
    monkeypatch.setenv('ALPACA_BASE_URL', 'invalid_url')
    with pytest.raises(ValueError, match="Invalid URL"):
        initialize_alpaca()

    # Valid configuration
    monkeypatch.setenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    # Mock tradeapi.REST
    with patch('alpaca_trade_api.REST') as mock_tradeapi:
        mock_client = MagicMock()
        mock_tradeapi.return_value = mock_client
        client = initialize_alpaca()
        assert client == mock_client, "Expected a valid Alpaca client with correct configuration."


@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_empty_data_sources(data_fetch_utils_fixture):
    """
    Test fetching data with empty data_sources list.
    """
    symbols = ['AAPL']
    data_sources = []
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Call the function with empty data_sources
    result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
        symbols,
        data_sources,
        start_date,
        end_date,
        interval
    )

    # Assertions
    assert 'AAPL' in result, "AAPL should be in the result."
    assert not result['AAPL'], "AAPL data should be empty when no data sources are provided."


@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_success(data_fetch_utils_fixture):
    """
    Test fetching data for multiple symbols from multiple sources successfully.
    """
    symbols = ['AAPL', 'GOOGL']
    data_sources = ["Alpha Vantage", "Polygon"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock Alpha Vantage data
    mock_alpha_vantage_data = {
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

    # Mock Polygon data
    mock_polygon_data = {
        "results": [
            {
                "t": 1672531200000,  # 2023-01-01
                "o": 130.0,
                "h": 132.0,
                "l": 129.0,
                "c": 131.0,
                "v": 1000000
            }
        ]
    }

    # Expected DataFrames
    expected_alpha_vantage_df_aapl = pd.DataFrame([{
        'open': 130.0,
        'high': 132.0,
        'low': 129.0,
        'close': 131.0,
        'volume': 1000000
    }], index=pd.to_datetime(['2023-01-03'])).rename_axis('date')

    expected_alpha_vantage_df_googl = pd.DataFrame([{
        'open': 130.0,
        'high': 132.0,
        'low': 129.0,
        'close': 131.0,
        'volume': 1000000
    }], index=pd.to_datetime(['2023-01-03'])).rename_axis('date')

    expected_polygon_df_aapl = pd.DataFrame({
        'open': [130.0],
        'high': [132.0],
        'low': [129.0],
        'close': [131.0],
        'volume': [1000000]
    }, index=pd.to_datetime(['2023-01-01']).date).rename_axis('date')

    expected_polygon_df_googl = pd.DataFrame({
        'open': [130.0],
        'high': [132.0],
        'low': [129.0],
        'close': [131.0],
        'volume': [1000000]
    }, index=pd.to_datetime(['2023-01-01']).date).rename_axis('date')

    with aioresponses() as mocked:
        # Mock Alpha Vantage for AAPL and GOOGL
        for symbol in symbols:
            url_pattern_av = re.compile(
                rf"https://(?:www\.)?alphavantage\.co/query\?(?=.*function=TIME_SERIES_DAILY)(?=.*symbol={symbol})(?=.*apikey=test_alphavantage_key).*"
            )
            mocked.get(
                url_pattern_av,
                payload=mock_alpha_vantage_data,
                status=200
            )

            # Mock Polygon for AAPL and GOOGL
            url_pattern_polygon = re.compile(
                rf"https://api\.polygon\.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}\?adjusted=true&apiKey=test_polygonio_key"
            )
            mocked.get(
                url_pattern_polygon,
                payload=mock_polygon_data,
                status=200
            )

        # Call the function under test
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols, data_sources, start_date, end_date, interval
        )

    # Assertions for AAPL
    assert 'AAPL' in result, "AAPL should be in the result."
    assert 'Alpha Vantage' in result['AAPL'], "Alpha Vantage data for AAPL missing."
    pd.testing.assert_frame_equal(result['AAPL']['Alpha Vantage'], expected_alpha_vantage_df_aapl)
    assert 'Polygon' in result['AAPL'], "Polygon data for AAPL missing."
    pd.testing.assert_frame_equal(result['AAPL']['Polygon'], expected_polygon_df_aapl)

    # Assertions for GOOGL
    assert 'GOOGL' in result, "GOOGL should be in the result."
    assert 'Alpha Vantage' in result['GOOGL'], "Alpha Vantage data for GOOGL missing."
    pd.testing.assert_frame_equal(result['GOOGL']['Alpha Vantage'], expected_alpha_vantage_df_googl)
    assert 'Polygon' in result['GOOGL'], "Polygon data for GOOGL missing."
    pd.testing.assert_frame_equal(result['GOOGL']['Polygon'], expected_polygon_df_googl)


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
async def test_concurrency_fetch_many_symbols(data_fetch_utils_fixture):
    """
    Test fetching data concurrently for a large number of symbols.
    """
    symbols = [f'SYM{i}' for i in range(100)]  # 100 symbols
    data_sources = ["Alpha Vantage"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock Alpha Vantage data for each symbol
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
        
        # Fetch data
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols,
            data_sources,
            start_date,
            end_date,
            interval
        )
    
    # Assertions
    for symbol in symbols:
        assert symbol in result, f"{symbol} not in result."
        assert "Alpha Vantage" in result[symbol], f"Alpha Vantage data missing for {symbol}."
        df = result[symbol]["Alpha Vantage"]
        assert not df.empty, f"Expected data for {symbol}."
        # Check specific data
        expected_df = pd.DataFrame([{
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000000
        }], index=pd.to_datetime(['2023-01-03'])).rename_axis('date')
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)

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

    assert isinstance(parsed_df, pd.DataFrame), "Expected a DataFrame."
    assert not parsed_df.empty, "Expected a non-empty DataFrame."
    assert "52WeekHigh" in parsed_df.columns, "Missing '52WeekHigh' column."
    assert "52WeekLow" in parsed_df.columns, "Missing '52WeekLow' column."
    assert "MarketCapitalization" in parsed_df.columns, "Missing 'MarketCapitalization' column."
    assert "P/E" in parsed_df.columns, "Missing 'P/E' column."
    assert parsed_df.index.name == "date_fetched", "Missing or incorrect 'date_fetched' index."


# -----------------------------
# Additional Test Functions
# -----------------------------

@pytest.mark.asyncio
async def test_fetch_yahoo_finance_data_success(data_fetch_utils_fixture):
    """
    Test fetching data successfully from Yahoo Finance (yfinance).
    """
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock yfinance.download to return expected DataFrame
    mock_yf_data = pd.DataFrame({
        'Date': pd.to_datetime(['2023-01-03', '2023-01-04']),
        'Open': [130.0, 131.0],
        'High': [132.0, 133.0],
        'Low': [129.0, 130.0],
        'Close': [131.0, 132.0],
        'Volume': [1000000, 1500000],
        'Adj Close': [130.5, 131.5]
    }).set_index('Date')

    expected_df = pd.DataFrame({
        'open': [130.0, 131.0],
        'high': [132.0, 133.0],
        'low': [129.0, 130.0],
        'close': [131.0, 132.0],
        'volume': [1000000, 1500000],
        'adj_close': [130.5, 131.5],
        'symbol': ['AAPL', 'AAPL']
    }, index=pd.to_datetime(['2023-01-03', '2023-01-04']).date).rename_axis('date')

    with patch('yfinance.download', return_value=mock_yf_data):
        result = await data_fetch_utils_fixture.fetch_stock_data_async(ticker, start_date, end_date, interval)

    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_yahoo_finance_data_unsupported_interval(data_fetch_utils_fixture):
    """
    Test fetching data from Yahoo Finance with unsupported interval.
    """
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '5d'  # Unsupported

    with patch('yfinance.download', side_effect=ValueError("Unsupported interval")):
        result = await data_fetch_utils_fixture.fetch_stock_data_async(ticker, start_date, end_date, interval)

    assert isinstance(result, pd.DataFrame), "Expected a DataFrame even on failure."
    assert result.empty, "Expected an empty DataFrame when an invalid interval is used."


