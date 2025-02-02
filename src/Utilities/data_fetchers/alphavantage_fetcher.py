# File: alphavantage_fetcher.py
# Location: src/Utilities/fetchers
# Description: Fetches data from Alpha Vantage API.

import os
import aiohttp
import pandas as pd
from typing import Optional
import logging

class AlphaVantageFetcher:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if not self.api_key:
            self.logger.error("ALPHAVANTAGE_API_KEY is not set in environment variables.")

    async def fetch_stock_data_async(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical stock data from Alpha Vantage asynchronously.

        Args:
            symbol (str): Stock symbol.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: Historical stock data.
        """
        if not self.api_key:
            self.logger.error("Alpha Vantage API key is missing. Cannot fetch stock data.")
            return pd.DataFrame()

        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={self.api_key}"
        self.logger.debug(f"Fetching Alpha Vantage data for {symbol} from URL: {url}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        time_series = data.get("Time Series (Daily)", {})
                        records = []
                        for date_str, metrics in time_series.items():
                            date = pd.to_datetime(date_str).date()
                            if start_date <= str(date) <= end_date:
                                records.append({
                                    'date': date,
                                    'open': float(metrics.get("1. open", 0)),
                                    'high': float(metrics.get("2. high", 0)),
                                    'low': float(metrics.get("3. low", 0)),
                                    'close': float(metrics.get("4. close", 0)),
                                    'volume': int(metrics.get("6. volume", 0))
                                })
                        df = pd.DataFrame(records)
                        if not df.empty:
                            df.set_index('date', inplace=True)
                            self.logger.info(f"Fetched {len(df)} records for {symbol} from Alpha Vantage.")
                        return df
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to fetch Alpha Vantage data for {symbol}: {response.status} - {error_text}")
        except Exception as e:
            self.logger.error(f"Exception while fetching Alpha Vantage data for {symbol}: {e}")

        return pd.DataFrame()
