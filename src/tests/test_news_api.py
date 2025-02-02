"""
test_news_api.py

Contains tests for fetching and parsing news data via NewsAPI,
including sentiment analysis and edge cases.
"""
from pathlib import Path
import pytest
import re
import os
from unittest.mock import patch, MagicMock
import pandas as pd
from aioresponses import aioresponses
from Utilities.main_data_fetcher import DataFetchUtils

@pytest.mark.asyncio
async def test_fetch_news_data_async_success(data_fetch_utils_fixture):
    """
    Test fetching news data from NewsAPI successfully.
    """
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

    with patch('requests.get') as mock_get, patch('src.Utilities.data_fetch_utils.TextBlob') as mock_textblob:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_get.return_value = mock_resp

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
            'sentiment': 0.13636363636363635
        }
    ]).set_index('date')

    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_news_data_async_failure(data_fetch_utils_fixture):
    """
    Test NewsAPI returning a failure status like 429 (rate limit).
    """
    symbol = 'AAPL'
    url_pattern = re.compile(
        rf"https://newsapi\.org/v2/everything\?(?=.*q={symbol})(?=.*apiKey=test_newsapi_key).*"
    )
    with aioresponses(assert_all_requests_are_fired=True) as mocked:
        mocked.get(url_pattern, status=429)
        result = await data_fetch_utils_fixture.fetch_news_data_async(symbol, page_size=5)

    assert isinstance(result, pd.DataFrame)
    assert result.empty, "Expected an empty DataFrame on rate limiting."

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
                'title': 'Apple Releases New Product'
                # Missing 'description', 'source', 'url'
            }
        ]
    }

    with patch('requests.get') as mock_get, patch('src.Utilities.data_fetch_utils.TextBlob') as mock_textblob:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_get.return_value = mock_resp

        mock_blob_instance = MagicMock()
        mock_blob_instance.sentiment.polarity = 0.0
        mock_textblob.return_value = mock_blob_instance

        result = await data_fetch_utils_fixture.fetch_news_data_async(symbol, page_size=1)

    expected_df = pd.DataFrame([
        {
            'date': pd.to_datetime('2023-04-14').date(),
            'headline': 'Apple Releases New Product',
            'content': '',
            'symbol': 'AAPL',
            'source': '',
            'url': '',
            'sentiment': 0.0
        }
    ]).set_index('date')

    pd.testing.assert_frame_equal(result, expected_df)

@pytest.mark.asyncio
async def test_fetch_news_data_async_unexpected_fields(data_fetch_utils_fixture):
    """
    Test fetch_news_data_async with unexpected fields in the articles.
    """
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
