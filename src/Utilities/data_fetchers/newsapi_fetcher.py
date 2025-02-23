# -------------------------------------------------------------------
# File: newsapi_fetcher.py
# Location: src/Utilities/fetchers
# Description: Fetches news data from NewsAPI.
# -------------------------------------------------------------------

import os
import requests
import pandas as pd
from textblob import TextBlob
from typing import Optional
import logging
import asyncio

class NewsAPIFetcher:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.api_key = os.getenv('NEWSAPI_API_KEY')
        if not self.api_key:
            self.logger.error("NEWSAPI_API_KEY is not set in environment variables.")

    def get_news_data(self, ticker: str, page_size: int = 5) -> pd.DataFrame:
        """
        Fetches news articles for a given stock ticker.

        Args:
            ticker (str): Stock symbol.
            page_size (int, optional): Number of articles to fetch. Defaults to 5.

        Returns:
            pd.DataFrame: News articles data.
        """
        if not self.api_key:
            self.logger.error("NewsAPI key is missing. Cannot fetch news.")
            return pd.DataFrame()

        url = f"https://newsapi.org/v2/everything?q={ticker}&pageSize={page_size}&apiKey={self.api_key}"
        self.logger.debug(f"Fetching NewsAPI data for {ticker} from URL: {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            news_data = []
            for article in articles:
                published_at = article.get('publishedAt')
                if published_at:
                    date_val = pd.to_datetime(published_at).date()
                else:
                    date_val = pd.Timestamp.utcnow().date()

                news_data.append({
                    'date': date_val,
                    'headline': article.get('title', ''),
                    'content': article.get('description', '') or '',
                    'symbol': ticker,
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ''),
                    'sentiment': TextBlob(article.get('description', '') or '').sentiment.polarity
                })
            self.logger.info(f"Fetched {len(news_data)} news articles for {ticker} from NewsAPI.")
            df = pd.DataFrame(news_data)
            if not df.empty:
                df.set_index('date', inplace=True)
            return df
        except requests.HTTPError as http_err:
            self.logger.error(f"HTTP error fetching news for {ticker}: {http_err}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching news for {ticker}: {e}")

        return pd.DataFrame()

    async def fetch_news_data_async(self, ticker: str, page_size: int = 5) -> pd.DataFrame:
        """
        Async wrapper for get_news_data.

        Args:
            ticker (str): Stock symbol.
            page_size (int, optional): Number of articles to fetch. Defaults to 5.

        Returns:
            pd.DataFrame: News articles data.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_news_data, ticker, page_size)
