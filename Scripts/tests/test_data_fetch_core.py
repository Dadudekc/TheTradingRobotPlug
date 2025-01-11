import pytest
from unittest.mock import patch
import pandas as pd
from aioresponses import aioresponses
import aiohttp
import re


import pytest
import pandas as pd
from aioresponses import aioresponses
import re


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

    # Assertions
    assert 'AAPL' in result, "Expected symbol AAPL in result."
    assert 'Alpha Vantage' in result['AAPL'], "Expected Alpha Vantage data for AAPL."
    assert 'Polygon' in result['AAPL'], "Expected Polygon data for AAPL."

    # Validate Polygon DataFrame
    pd.testing.assert_frame_equal(
        result['AAPL']['Polygon'], expected_polygon_df, check_dtype=False,
        err_msg="Polygon data does not match expected DataFrame."
    )
