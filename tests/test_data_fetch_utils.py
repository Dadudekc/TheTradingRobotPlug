# tests/test_data_fetch_utils.py

import pytest
import asyncio
from unittest.mock import patch, MagicMock
import pandas as pd
from aioresponses import aioresponses
from .data_fetch_utils import DataFetchUtils

@pytest.fixture
def data_fetch_utils():
    with patch('data_fetch_utils.setup_logging') as mock_logging_setup:
        mock_logger = MagicMock()
        mock_logging_setup.return_value = mock_logger
        # Mock environment variables
        with patch.dict('os.environ', {
            'NEWSAPI_API_KEY': 'test_newsapi_key',
            'FINNHUB_API_KEY': 'test_finnhub_key',
            'ALPACA_API_KEY': 'test_alpaca_key',
            'ALPACA_SECRET_KEY': 'test_alpaca_secret',
            'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets'
        }):
            yield DataFetchUtils()

@pytest.mark.asyncio
async def test_fetch_finnhub_quote_success(data_fetch_utils):
    symbol = 'AAPL'
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token=test_finnhub_key"
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
        mocked.get(url, payload=mock_response, status=200)
        result = await data_fetch_utils.fetch_finnhub_quote(symbol, MagicMock())
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
async def test_fetch_finnhub_quote_failure(data_fetch_utils):
    symbol = 'AAPL'
    url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token=test_finnhub_key"

    with aioresponses() as mocked:
        mocked.get(url, status=500)
        result = await data_fetch_utils.fetch_finnhub_quote(symbol, MagicMock())
        assert result.empty

@pytest.mark.asyncio
async def test_fetch_news_data_async_success(data_fetch_utils):
    symbol = 'AAPL'
    url = f'https://newsapi.org/v2/everything?q={symbol}&pageSize=5&apiKey=test_newsapi_key'
    mock_response = {
        'articles': [
            {
                'publishedAt': '2023-04-14T12:34:56Z',
                'title': 'Apple Releases New Product',
                'description': 'Apple has released a new product...',
                'source': {'name': 'TechCrunch'},
                'url': 'https://techcrunch.com/apple-new-product'
            },
            # Add more articles if needed
        ]
    }

    with aioresponses() as mocked:
        mocked.get(url, payload=mock_response, status=200)
        result = await data_fetch_utils.fetch_news_data_async(symbol, page_size=1)
        expected_df = pd.DataFrame([
            {
                'date': pd.to_datetime('2023-04-14').date(),
                'headline': 'Apple Releases New Product',
                'content': 'Apple has released a new product...',
                'symbol': 'AAPL',
                'source': 'TechCrunch',
                'url': 'https://techcrunch.com/apple-new-product',
                'sentiment': 0.0  # Assuming neutral sentiment; adjust if TextBlob returns differently
            }
        ]).set_index('date')
        pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_news_data_async_failure(data_fetch_utils):
    symbol = 'AAPL'
    url = f'https://newsapi.org/v2/everything?q={symbol}&pageSize=5&apiKey=test_newsapi_key'

    with aioresponses() as mocked:
        mocked.get(url, status=429)  # Simulate rate limiting
        result = await data_fetch_utils.fetch_news_data_async(symbol, page_size=5)
        assert result.empty

@pytest.mark.asyncio
async def test_fetch_stock_data_async(data_fetch_utils):
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'
    expected_data = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-03', '2023-01-04']),
        'open': [130.0, 131.0],
        'high': [132.0, 133.0],
        'low': [129.0, 130.0],
        'close': [131.0, 132.0],
        'volume': [1000000, 1500000],
        'symbol': ['AAPL', 'AAPL']
    }).set_index('date')

    with patch('yfinance.download') as mock_yf_download:
        mock_yf_download.return_value = expected_data.reset_index()
        result = await data_fetch_utils.fetch_stock_data_async(ticker, start_date, end_date, interval)
        pd.testing.assert_frame_equal(result, expected_data)

@pytest.mark.asyncio
async def test_fetch_alpaca_data_async_success(data_fetch_utils):
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1Day'
    mock_bars_df = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-03'), pd.Timestamp('2023-01-04')],
        'open': [130.0, 131.0],
        'high': [132.0, 133.0],
        'low': [129.0, 130.0],
        'close': [131.0, 132.0],
        'volume': [1000000, 1500000],
        'symbol': ['AAPL', 'AAPL']
    })

    with patch.object(data_fetch_utils.alpaca_api, 'get_bars', return_value=MagicMock(df=mock_bars_df)):
        result = await data_fetch_utils.fetch_alpaca_data_async(symbol, start_date, end_date, interval)
        expected_df = mock_bars_df.set_index('date')
        pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_alpaca_data_async_failure(data_fetch_utils):
    symbol = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1Day'

    with patch.object(data_fetch_utils.alpaca_api, 'get_bars', side_effect=Exception("API Error")):
        result = await data_fetch_utils.fetch_alpaca_data_async(symbol, start_date, end_date, interval)
        assert result.empty

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols(data_fetch_utils):
    symbols = ['AAPL', 'GOOGL']
    data_sources = ["Alpaca", "Alpha Vantage"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock Alpaca data
    mock_alpaca_df = pd.DataFrame({
        'date': [pd.Timestamp('2023-01-03')],
        'open': [130.0],
        'high': [132.0],
        'low': [129.0],
        'close': [131.0],
        'volume': [1000000],
        'symbol': ['AAPL']
    }).set_index('date')

    # Mock Alpha Vantage data
    mock_alpha_vantage_df = pd.DataFrame([
        {
            'date': pd.to_datetime('2023-01-03'),
            'open': 130.0,
            'high': 132.0,
            'low': 129.0,
            'close': 131.0,
            'volume': 1000000
        }
    ]).set_index('date')

    with aioresponses() as mocked:
        # Mock Alpha Vantage API responses
        for symbol in symbols:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey=test_alphavantage_key"
            mocked.get(url, payload={
                "Time Series (Daily)": {
                    "2023-01-03": {
                        "1. open": "130.0",
                        "2. high": "132.0",
                        "3. low": "129.0",
                        "4. close": "131.0",
                        "5. volume": "1000000"
                    }
                }
            }, status=200)

        # Mock Alpaca API responses
        for symbol in symbols:
            with patch.object(data_fetch_utils.alpaca_api, 'get_bars', return_value=MagicMock(df=mock_alpaca_df)):
                pass  # Already mocked above

        result = await data_fetch_utils.fetch_data_for_multiple_symbols(
            symbols,
            data_sources,
            start_date,
            end_date,
            interval
        )

        assert 'AAPL' in result
        assert 'Alpaca' in result['AAPL']
        assert 'Alpha Vantage' in result['AAPL']
        pd.testing.assert_frame_equal(result['AAPL']['Alpaca'], mock_alpaca_df)
        pd.testing.assert_frame_equal(result['AAPL']['Alpha Vantage'], mock_alpha_vantage_df)
        
        # Similarly, you can add assertions for 'GOOGL'

