import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime, date
import re
from aioresponses import aioresponses  
import aiohttp

@pytest.fixture
def data_fetch_utils_fixture(monkeypatch):
    # Patch environment variables
    monkeypatch.setenv('NEWSAPI_API_KEY', 'test_newsapi_key')
    monkeypatch.setenv('FINNHUB_API_KEY', 'test_finnhub_key')
    monkeypatch.setenv('ALPACA_API_KEY', 'test_alpaca_key')
    monkeypatch.setenv('ALPACA_SECRET_KEY', 'test_alpaca_secret')
    monkeypatch.setenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    monkeypatch.setenv('ALPHAVANTAGE_API_KEY', 'test_alphavantage_key')
    monkeypatch.setenv('POLYGONIO_API_KEY', 'test_polygonio_key')
    
    # Mock the logging setup
    with patch('Scripts.Utilities.config_manager.setup_logging') as mock_logging_setup:
        mock_logger = MagicMock()
        mock_logging_setup.return_value = mock_logger
        
        # Import DataFetchUtils **after** patching environment variables
        from Scripts.Utilities.data_fetch_utils import DataFetchUtils
        yield DataFetchUtils()

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
        mocked.get(url_pattern, status=500)
        async with aiohttp.ClientSession() as session:
            result = await data_fetch_utils_fixture.fetch_finnhub_quote(symbol, session)
    # Ensure result is a DataFrame even on failure
    assert isinstance(result, pd.DataFrame)
    assert result.empty

@pytest.mark.asyncio
async def test_fetch_news_data_async_failure(data_fetch_utils_fixture):
    symbol = 'AAPL'
    url_pattern = re.compile(rf"https://newsapi\.org/v2/everything\?q={symbol}&pageSize=5&apiKey=test_newsapi_key.*")
    
    with aioresponses() as mocked:
        mocked.get(url_pattern, status=429)  # Simulate rate limiting
        result = await data_fetch_utils_fixture.fetch_news_data_async(symbol, page_size=5)
    
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
