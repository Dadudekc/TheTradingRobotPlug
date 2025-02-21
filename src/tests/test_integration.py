"""
test_integration.py

Contains integration tests involving multiple data sources
(Alpha Vantage, Polygon, Finnhub, NewsAPI) and/or complex
multi-symbol logic in a single call.
"""
from pathlib import Path
import pytest
import re
import pandas as pd
from aioresponses import aioresponses
from unittest.mock import patch, MagicMock
from Utilities.data_fetchers.main_data_fetcher import MainDataFetcher


@pytest.mark.asyncio
async def test_fetch_data_invalid_date_format(data_fetch_utils_fixture):
    """
    Test fetch_data_for_multiple_symbols with an invalid date format.
    """
    symbols = ['AAPL']
    start_date = 'invalid_date'
    end_date = '2023-01-31'
    data_sources = ["Alpha Vantage"]

    with pytest.raises(ValueError, match="Invalid date format"):
        await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols, data_sources, start_date, end_date, interval='1d'
        )

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_integration(data_fetch_utils_fixture):
    """
    Integration test for multiple data sources (Alpha Vantage, Polygon, NewsAPI).
    """
    symbols = ['AAPL', 'GOOGL']
    data_sources = ["Alpha Vantage", "Polygon", "NewsAPI"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    with aioresponses() as mocked:
        for symbol in symbols:
            # Mock Alpha Vantage
            mocked.get(
                re.compile(rf"https://.*alphavantage.*symbol={symbol}.*"),
                payload={"Time Series (Daily)": {"2023-01-03": {"1. open": "130.0"}}},
                status=200
            )
            # Mock Polygon
            mocked.get(
                re.compile(rf"https://.*polygon.*{symbol}.*"),
                payload={"results": [{"t": 1672531200000, "o": 130.0}]},
                status=200
            )

        # Mock NewsAPI (using requests.get)
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = {
                'status': 'ok',
                'articles': [
                    {'publishedAt': '2023-04-14T12:34:56Z', 'title': 'Title 1', 'description': 'Desc 1'},
                    {'publishedAt': '2023-04-15T08:20:00Z', 'title': 'Title 2', 'description': 'Desc 2'}
                ]
            }

            result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
                symbols, data_sources, start_date, end_date, interval
            )

    # Basic assertion: ensure NewsAPI results exist for AAPL
    assert len(result['AAPL']['NewsAPI']) == 2, "Expected 2 news articles for AAPL."

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols(data_fetch_utils_fixture):
    """
    Tests multiple symbols from a single data source (Alpha Vantage).
    """
    symbols = ['AAPL', 'GOOGL']
    data_sources = ["Alpha Vantage"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Mock DataFrames
    mock_df_aapl = pd.DataFrame([{'open':130.0}], index=pd.to_datetime(['2023-01-03']))
    mock_df_googl = pd.DataFrame([{'open':140.0}], index=pd.to_datetime(['2023-01-03']))

    with aioresponses(assert_all_requests_are_fired=False) as mocked:
        # Provide a small payload for each symbol
        for symbol, df in [('AAPL', mock_df_aapl), ('GOOGL', mock_df_googl)]:
            url_pattern = re.compile(
                rf"https://(?:www\.)?alphavantage\.co/query\?(?=.*symbol={symbol}).*"
            )
            mocked.get(
                url_pattern,
                payload={
                    "Time Series (Daily)": {
                        "2023-01-03": {
                            "1. open": str(df.iloc[0]['open']),
                            "2. high": "999.0",
                            "3. low": "999.0",
                            "4. close": "999.0",
                            "5. volume": "999999"
                        }
                    }
                },
                status=200
            )
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols, data_sources, start_date, end_date, interval
        )

    assert 'AAPL' in result and 'GOOGL' in result, "Missing results for one or more symbols."

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_invalid_symbol(data_fetch_utils_fixture):
    """
    Test invalid symbol returning an empty DataFrame.
    """
    symbols = ['INVALID']
    data_sources = ["Alpha Vantage"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    with aioresponses() as mocked:
        mocked.get(
            re.compile(r"https://(?:www\.)?alphavantage\.co/query\?.*symbol=INVALID.*"),
            payload={"Time Series (Daily)": {}},
            status=200
        )
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols, data_sources, start_date, end_date, interval
        )

    assert 'INVALID' in result, "Symbol INVALID should appear in the result"
    assert result['INVALID']['Alpha Vantage'].empty, "Should be empty for invalid symbol"

@pytest.mark.asyncio
async def test_fetch_data_for_mixed_symbols(data_fetch_utils_fixture):
    """
    Test partial valid + partial invalid symbol scenario.
    """
    symbols = ['AAPL', 'INVALID']
    data_sources = ["Alpha Vantage"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # One valid, one invalid
    with aioresponses() as mocked:
        mocked.get(
            re.compile(r".*symbol=AAPL.*"),
            payload={"Time Series (Daily)": {"2023-01-03": {"1. open":"130.0"}}},
            status=200
        )
        mocked.get(
            re.compile(r".*symbol=INVALID.*"),
            payload={"Time Series (Daily)": {}},
            status=200
        )
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols, data_sources, start_date, end_date, interval
        )

    assert 'AAPL' in result
    assert not result['AAPL']['Alpha Vantage'].empty, "AAPL should have data"
    assert result['INVALID']['Alpha Vantage'].empty, "INVALID should have empty data"

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_empty_data_sources(data_fetch_utils_fixture):
    """
    Test multiple symbols with an empty data_sources list.
    """
    symbols = ['AAPL']
    data_sources = []
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
        symbols, data_sources, start_date, end_date, interval
    )

    assert 'AAPL' in result, "AAPL should appear even if data sources are empty"
    assert not result['AAPL'], "Should be no data frames if no data sources"

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_success(data_fetch_utils_fixture):
    """
    Multiple data sources for multiple symbols with a successful response.
    """
    symbols = ['AAPL', 'GOOGL']
    data_sources = ["Alpha Vantage", "Polygon"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    mock_alpha_vantage_data = {
        "Time Series (Daily)": {
            "2023-01-03": {
                "1. open": "130.0", "2. high": "132.0",
                "3. low": "129.0", "4. close": "131.0",
                "5. volume": "1000000"
            }
        }
    }
    mock_polygon_data = {
        "results": [
            {
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
        for symbol in symbols:
            mocked.get(
                re.compile(rf".*alphavantage.*symbol={symbol}.*"),
                payload=mock_alpha_vantage_data,
                status=200
            )
            mocked.get(
                re.compile(rf".*polygon.*{symbol}.*"),
                payload=mock_polygon_data,
                status=200
            )
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(symbols, data_sources, start_date, end_date, interval)

    # Check for presence of data
    assert 'AAPL' in result
    assert 'Alpha Vantage' in result['AAPL']
    assert 'Polygon' in result['AAPL']
    assert not result['AAPL']['Alpha Vantage'].empty, "Expected AAPL Alpha Vantage data"
    assert not result['AAPL']['Polygon'].empty, "Expected AAPL Polygon data"

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_mixed_symbols(data_fetch_utils_fixture):
    """
    Test multiple data sources with partial valid & partial invalid symbols.
    """
    symbols = ['AAPL', 'INVALID']
    data_sources = ["Alpha Vantage", "Finnhub"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    # Simplified mock responses
    with aioresponses() as mocked:
        # AAPL Alpha Vantage
        mocked.get(
            re.compile(r".*alphavantage.*symbol=AAPL.*"),
            payload={"Time Series (Daily)": {"2023-01-03": {"1. open": "130.0"}}},
            status=200
        )
        # INVALID Alpha Vantage
        mocked.get(
            re.compile(r".*alphavantage.*symbol=INVALID.*"),
            payload={"Time Series (Daily)": {}},
            status=200
        )
        # AAPL Finnhub
        mocked.get(
            re.compile(r".*finnhub.*symbol=AAPL.*"),
            payload={"metric":{"52WeekHigh":123}},
            status=200
        )
        # INVALID Finnhub
        mocked.get(
            re.compile(r".*finnhub.*symbol=INVALID.*"),
            payload={},
            status=200
        )

        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(symbols, data_sources, start_date, end_date, interval)

    assert 'AAPL' in result and 'INVALID' in result
    assert 'Alpha Vantage' in result['AAPL']
    assert not result['INVALID']['Alpha Vantage'].empty is False, "INVALID alpha vantage is empty"
    assert 'Finnhub' in result['AAPL']
    assert not result['AAPL']['Finnhub'].empty, "AAPL Finnhub not empty"
    assert result['INVALID']['Finnhub'].empty, "INVALID Finnhub is empty"

@pytest.mark.asyncio
async def test_fetch_data_for_multiple_symbols_overlap(data_fetch_utils_fixture):
    """
    Test partial date overlap scenario (Alpha Vantage + Polygon).
    """
    symbols = ['AAPL']
    data_sources = ["Alpha Vantage", "Polygon"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

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
    mock_polygon_data = {
        "results": [
            {
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
        mocked.get(
            re.compile(r".*alphavantage.*"),
            payload=mock_alpha_vantage_data,
            status=200
        )
        mocked.get(
            re.compile(r".*polygon.*"),
            payload=mock_polygon_data,
            status=200
        )
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(symbols, data_sources, start_date, end_date, interval)

    # Convert index if needed
    result['AAPL']['Polygon'].index = pd.to_datetime(result['AAPL']['Polygon'].index)
    assert 'AAPL' in result
    assert 'Alpha Vantage' in result['AAPL']
    assert 'Polygon' in result['AAPL']
    assert not result['AAPL']['Polygon'].empty, "Polygon data is not empty"

@pytest.mark.asyncio
async def test_fetch_data_large_symbol_list(data_fetch_utils_fixture):
    """
    Test fetch_data_for_multiple_symbols with large number of symbols across multiple sources.
    """
    symbols = [f'SYM{i}' for i in range(100)]
    data_sources = ["Alpha Vantage"]
    start_date = '2023-01-01'
    end_date = '2023-01-31'
    interval = '1d'

    with aioresponses() as mocked:
        for symbol in symbols:
            url_pattern = re.compile(
                rf".*alphavantage.*symbol={symbol}.*"
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
        result = await data_fetch_utils_fixture.fetch_data_for_multiple_symbols(
            symbols, data_sources, start_date, end_date, interval
        )

    assert len(result) == len(symbols), "Expected entries for all 100 symbols."
    for symbol in symbols:
        assert symbol in result, f"{symbol} missing from result"

