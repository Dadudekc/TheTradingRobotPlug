# Scripts/tests/test_data_fetch_core.py
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, date
import re
from aioresponses import aioresponses  
import aiohttp  # Ensure aiohttp is imported
from aiohttp import ClientSession  # Import ClientSession
import numpy as np
import os
from pathlib import Path
from Scripts.Utilities.data_fetch_utils import DataFetchUtils


@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_overlap(data_fetch_utils_fixture):
    symbols = ['AAPL']
    data_sources = ["Alpha Vantage", "Polygon"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock data for Alpha Vantage
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

    # Mock data for Polygon
    mock_polygon_data = {
        "results": [
            {
                "t": 1672531200000,  # Corresponds to 2023-01-01 in milliseconds
                "o": 130.0,
                "h": 132.0,
                "l": 129.0,
                "c": 131.0,
                "v": 1000000
            }
        ]
    }

    expected_polygon_df = pd.DataFrame({
        'date': [pd.to_datetime('2023-01-01')],
        'open': [130.0],
        'high': [132.0],
        'low': [129.0],
        'close': [131.0],
        'volume': [1000000]
    }).set_index('date')

    with aioresponses() as mocked:
        # Mock Alpha Vantage API response
        mocked.get(
            re.compile(r"https://www\.alphavantage\.co/query\?.*"),
            payload=mock_alpha_vantage_data,
            status=200
        )

        # Mock Polygon API response
        mocked.get(
            re.compile(r"https://api\.polygon\.io/v2/aggs/ticker/.*"),
            payload=mock_polygon_data,
            status=200
        )

        # Call the function under test
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols, data_sources, start_date, end_date, interval
        )

        # Debugging Output
        print("Test Result: ", result)

    # Convert result['AAPL']['Polygon'].index to DatetimeIndex
    result['AAPL']['Polygon'].index = pd.to_datetime(result['AAPL']['Polygon'].index)

    # Assertions
    assert 'AAPL' in result, "Expected symbol AAPL in result."
    assert 'Alpha Vantage' in result['AAPL'], "Expected Alpha Vantage data for AAPL."
    assert 'Polygon' in result['AAPL'], "Expected Polygon data for AAPL."

    # Validate Polygon DataFrame
    pd.testing.assert_frame_equal(
        result['AAPL']['Polygon'], expected_polygon_df, check_dtype=False
    )


@pytest.mark.asyncio
async def test_fetch_with_retries_empty_response(data_fetch_utils_fixture):
    """
    Test fetch_with_retries with an empty API response.
    """
    url = "https://example.com/empty-response"
    with patch.object(ClientSession, "get", return_value=None):
        async with ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_with_retries(url, {}, session, retries=3)
    assert result is None, "Expected None for empty API response."

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_overlap(data_fetch_utils_fixture):
    symbols = ['AAPL']
    data_sources = ["Alpha Vantage", "Polygon"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock data for Alpha Vantage
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

    # Mock data for Polygon
    mock_polygon_data = {
        "results": [
            {
                "t": 1672531200000,  # Corresponds to 2023-01-01 in milliseconds
                "o": 130.0,
                "h": 132.0,
                "l": 129.0,
                "c": 131.0,
                "v": 1000000
            }
        ]
    }

    expected_polygon_df = pd.DataFrame({
        'open': [130.0],
        'high': [132.0],
        'low': [129.0],
        'close': [131.0],
        'volume': [1000000]
    }, index=pd.to_datetime(['2023-01-01']).date).rename_axis('date')

    with aioresponses() as mocked:
        # Mock Alpha Vantage API response
        mocked.get(
            re.compile(r"https://www\.alphavantage\.co/query\?.*"),
            payload=mock_alpha_vantage_data,
            status=200
        )

        # Mock Polygon API response
        mocked.get(
            re.compile(r"https://api\.polygon\.io/v2/aggs/ticker/.*"),
            payload=mock_polygon_data,
            status=200
        )

        # Call the function under test
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols, data_sources, start_date, end_date, interval
        )

        # Debugging Output
        print("Test Result: ", result)

    # Convert result['AAPL']['Polygon'].index to DatetimeIndex
    result['AAPL']['Polygon'].index = pd.to_datetime(result['AAPL']['Polygon'].index)

    # Assertions
    assert 'AAPL' in result, "Expected symbol AAPL in result."
    assert 'Alpha Vantage' in result['AAPL'], "Expected Alpha Vantage data for AAPL."
    assert 'Polygon' in result['AAPL'], "Expected Polygon data for AAPL."

    # Validate Polygon DataFrame
    pd.testing.assert_frame_equal(
        result['AAPL']['Polygon'], expected_polygon_df, check_dtype=False
    )

@pytest.mark.parametrize("missing_key,expect_error", [
    ("FINNHUB_API_KEY", False),   # Expect a warning only
    ("NEWSAPI_API_KEY", True),    # Expect an EnvironmentError
])
def test_data_fetch_utils_init_missing_keys(monkeypatch, missing_key, expect_error):
    from Scripts.Utilities.data_fetch_utils import DataFetchUtils

    # Set up environment variables
    env_vars = {"NEWSAPI_API_KEY": "test_newsapi_key"}
    if missing_key != "NEWSAPI_API_KEY":
        env_vars.pop("NEWSAPI_API_KEY", None)

    with patch.dict(os.environ, env_vars, clear=True):
        mock_logger = MagicMock()
        if expect_error:
            with pytest.raises(EnvironmentError):
                DataFetchUtils(logger=mock_logger)
            mock_logger.error.assert_called_with(
                "NEWSAPI_API_KEY is not set in environment variables."
            )
        else:
            data_fetch_utils = DataFetchUtils(logger=mock_logger)
            mock_logger.warning.assert_called_with(
                "FINNHUB_API_KEY is not set in environment variables. Finnhub features may fail."
            )
            mock_logger.error.assert_called_with(
                "Alpaca API credentials are not fully set in environment variables."
            )
            assert data_fetch_utils.alpaca_api is None, "Alpaca API client should be None when keys are missing."

@pytest.mark.asyncio
async def test_fetch_data_invalid_date_format(data_fetch_utils_fixture):
    """
    Test fetch_data_for_multiple_symbols with invalid date format.
    """
    symbols = ['AAPL']
    start_date = 'invalid_date'
    end_date = '2023-01-31'
    data_sources = ["Alpha Vantage"]

    with pytest.raises(ValueError, match="Invalid date format"):
        await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols,
            data_sources,
            start_date,
            end_date
        )

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_integration(data_fetch_utils_fixture):
    """
    Integration test fetching data from multiple sources for multiple symbols.
    """
    symbols = ['AAPL', 'GOOGL']
    data_sources = ["Alpha Vantage", "Polygon", "NewsAPI"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock Alpha Vantage
    with aioresponses() as mocked:
        for symbol in symbols:
            url_pattern_av = re.compile(
                rf"https://(?:www\.)?alphavantage\.co/query\?(?=.*function=TIME_SERIES_DAILY)(?=.*symbol={symbol})(?=.*apikey=test_alphavantage_key).*"
            )
            mocked.get(
                url_pattern_av,
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

        # Mock Polygon
        for symbol in symbols:
            url_pattern_polygon = re.compile(
                rf"https://api\.polygon\.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}\?adjusted=true&apiKey=test_polygonio_key"
            )
            mocked.get(
                url_pattern_polygon,
                payload={
                    "results": [
                        {
                            "t": 1672531200000,  # Corresponds to 2023-01-01 in milliseconds
                            "o": 130.0,
                            "h": 132.0,
                            "l": 129.0,
                            "c": 131.0,
                            "v": 1000000
                        }
                    ]
                },
                status=200
            )

        # Mock NewsAPI
        with patch('requests.get') as mock_get:
            mock_response = {
                'status': 'ok',
                'totalResults': 2,
                'articles': [
                    {
                        'publishedAt': '2023-04-14T12:34:56Z',
                        'title': 'Apple Releases New Product',
                        'description': 'Apple has released a new product...',
                        'source': {'name': 'TechCrunch'},
                        'url': 'https://techcrunch.com/apple-new-product'
                    },
                    {
                        'publishedAt': '2023-04-15T08:20:00Z',
                        'title': 'Google Announces Update',
                        'description': 'Google has announced a new update...',
                        'source': {'name': 'The Verge'},
                        'url': 'https://www.theverge.com/google-update'
                    }
                ]
            }
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response
            mock_get.return_value = mock_resp

            # Mock TextBlob
            with patch('Scripts.Utilities.data_fetch_utils.TextBlob') as mock_textblob:
                mock_blob_instance = MagicMock()
                mock_blob_instance.sentiment.polarity = 0.1
                mock_textblob.return_value = mock_blob_instance

                # Call fetch_data_for_multiple_symbols
                result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
                    symbols,
                    data_sources,
                    start_date,
                    end_date,
                    interval
                )

    # Assertions
    for symbol in symbols:
        assert symbol in result, f"{symbol} should be in the result."

        # Alpha Vantage
        assert "Alpha Vantage" in result[symbol], f"Alpha Vantage data for {symbol} missing."
        df_av = result[symbol]["Alpha Vantage"]
        expected_av_df = pd.DataFrame([{
            'open': 130.0,
            'high': 132.0,
            'low': 129.0,
            'close': 131.0,
            'volume': 1000000
        }], index=pd.to_datetime(['2023-01-03'])).rename_axis('date')
        pd.testing.assert_frame_equal(df_av, expected_av_df)

        # Polygon
        assert "Polygon" in result[symbol], f"Polygon data for {symbol} missing."
        df_polygon = result[symbol]["Polygon"]
        expected_polygon_df = pd.DataFrame([{
            'open': 130.0,
            'high': 132.0,
            'low': 129.0,
            'close': 131.0,
            'volume': 1000000
        }], index=pd.to_datetime(['2023-01-01']).date).rename_axis('date')
        pd.testing.assert_frame_equal(df_polygon, expected_polygon_df)

        # NewsAPI
        assert "NewsAPI" in result[symbol], f"NewsAPI data for {symbol} missing."
        df_news = result[symbol]["NewsAPI"]
        expected_news_df = pd.DataFrame([
            {
                'date': pd.to_datetime('2023-04-14').date(),
                'headline': 'Apple Releases New Product',
                'content': 'Apple has released a new product...',
                'symbol': 'AAPL',
                'source': 'TechCrunch',
                'url': 'https://techcrunch.com/apple-new-product',
                'sentiment': 0.1
            },
            {
                'date': pd.to_datetime('2023-04-15').date(),
                'headline': 'Google Announces Update',
                'content': 'Google has announced a new update...',
                'symbol': 'AAPL',
                'source': 'The Verge',
                'url': 'https://www.theverge.com/google-update',
                'sentiment': 0.1
            }
        ]).set_index('date')

        pd.testing.assert_frame_equal(df_news, expected_news_df)

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_empty_symbols(data_fetch_utils_fixture):
    # Empty symbols list
    symbols = []
    data_sources = ["Alpha Vantage"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Call the function with empty symbols
    result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
        symbols,
        data_sources,
        start_date,
        end_date,
        interval
    )

    # Assertions
    assert isinstance(result, dict), "Result should be a dictionary."
    assert not result, "Result should be an empty dictionary when no symbols are provided."

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_overlap(data_fetch_utils_fixture):
    symbols = ['AAPL']
    data_sources = ["Alpha Vantage", "Polygon"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock data for Alpha Vantage
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

    # Mock data for Polygon
    mock_polygon_data = {
        "results": [
            {
                "t": 1672531200000,  # Corresponds to 2023-01-01 in milliseconds
                "o": 130.0,
                "h": 132.0,
                "l": 129.0,
                "c": 131.0,
                "v": 1000000
            }
        ]
    }

    expected_polygon_df = pd.DataFrame({
        'open': [130.0],
        'high': [132.0],
        'low': [129.0],
        'close': [131.0],
        'volume': [1000000]
    }, index=pd.to_datetime(['2023-01-01'])).rename_axis('date')  # Use Timestamp index

    with aioresponses() as mocked:
        # Mock Alpha Vantage API response
        mocked.get(
            re.compile(r"https://www\.alphavantage\.co/query\?.*"),
            payload=mock_alpha_vantage_data,
            status=200
        )

        # Mock Polygon API response
        mocked.get(
            re.compile(r"https://api\.polygon\.io/v2/aggs/ticker/.*"),
            payload=mock_polygon_data,
            status=200
        )

        # Call the function under test
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols, data_sources, start_date, end_date, interval
        )

        # Debugging Output
        print("Test Result: ", result)

    # Convert result['AAPL']['Polygon'].index to DatetimeIndex
    result['AAPL']['Polygon'].index = pd.to_datetime(result['AAPL']['Polygon'].index)

    # Assertions
    assert 'AAPL' in result, "Expected symbol AAPL in result."
    assert 'Alpha Vantage' in result['AAPL'], "Expected Alpha Vantage data for AAPL."
    assert 'Polygon' in result['AAPL'], "Expected Polygon data for AAPL."

    # Validate Polygon DataFrame
    pd.testing.assert_frame_equal(
        result['AAPL']['Polygon'], expected_polygon_df, check_dtype=False
    )
