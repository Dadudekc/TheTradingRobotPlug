# File: finnhub_fetcher.py
# Location: src/Utilities/fetchers
# Description: Fetches data from Finnhub API.

import os
import aiohttp
import pandas as pd
from typing import Optional
from datetime import datetime
import logging

class FinnhubFetcher:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.api_key = os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            self.logger.error("FINNHUB_API_KEY is not set in environment variables.")

    async def fetch_quote(self, symbol: str, session: aiohttp.ClientSession) -> pd.DataFrame:
        """
        Fetches real-time quote data from Finnhub.

        Args:
            symbol (str): Stock symbol.
            session (aiohttp.ClientSession): Active aiohttp session.

        Returns:
            pd.DataFrame: Real-time quote data.
        """
        if not self.api_key:
            self.logger.error("Finnhub API key is missing. Cannot fetch quote.")
            return pd.DataFrame()

        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.api_key}"
        self.logger.debug(f"Fetching Finnhub quote for {symbol} from URL: {url}")

        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    df = pd.DataFrame([{
                        'date': pd.to_datetime(data.get('t', 0), unit='s'),
                        'current_price': data.get('c'),
                        'change': data.get('d'),
                        'percent_change': data.get('dp'),
                        'high': data.get('h'),
                        'low': data.get('l'),
                        'open': data.get('o'),
                        'previous_close': data.get('pc')
                    }]).set_index('date')
                    self.logger.info(f"Fetched Finnhub quote for {symbol}.")
                    return df
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to fetch Finnhub quote for {symbol}: {response.status} - {error_text}")
        except Exception as e:
            self.logger.error(f"Exception while fetching Finnhub quote for {symbol}: {e}")

        return pd.DataFrame()

    async def fetch_financial_metrics(self, symbol: str, session: aiohttp.ClientSession) -> pd.DataFrame:
        """
        Fetches financial metrics from Finnhub.

        Args:
            symbol (str): Stock symbol.
            session (aiohttp.ClientSession): Active aiohttp session.

        Returns:
            pd.DataFrame: Financial metrics data.
        """
        if not self.api_key:
            self.logger.error("Finnhub API key is missing. Cannot fetch financial metrics.")
            return pd.DataFrame()

        url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={self.api_key}"
        self.logger.debug(f"Fetching Finnhub financial metrics for {symbol} from URL: {url}")

        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    if not data or not isinstance(data.get("metric"), dict):
                        self.logger.error(f"No valid financial metrics found for {symbol}.")
                        return pd.DataFrame()

                    metrics = data["metric"]
                    df = pd.DataFrame([{
                        "52WeekHigh": metrics.get("52WeekHigh"),
                        "52WeekLow": metrics.get("52WeekLow"),
                        "MarketCapitalization": metrics.get("MarketCapitalization"),
                        "P/E": metrics.get("P/E"),
                        "date_fetched": pd.Timestamp.utcnow().floor("s")
                    }]).set_index("date_fetched")

                    self.logger.info(f"Fetched Finnhub financial metrics for {symbol}.")
                    return df
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to fetch Finnhub financial metrics for {symbol}: {response.status} - {error_text}")
        except Exception as e:
            self.logger.error(f"Exception while fetching Finnhub financial metrics for {symbol}: {e}")

        return pd.DataFrame()

# File: finnhub_fetcher.py
# Location: src/Utilities/fetchers
# Description: Fetches data from Finnhub API.

import os
import aiohttp
import pandas as pd
from typing import Optional
import logging

class FinnhubFetcher:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.api_key = os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            self.logger.error("FINNHUB_API_KEY is not set in environment variables.")

    async def fetch_stock_data_async(self, symbol: str, session: Optional[aiohttp.ClientSession] = None) -> pd.DataFrame:
        """
        Fetches stock quote data from Finnhub asynchronously.

        Args:
            symbol (str): Stock symbol.
            session (aiohttp.ClientSession, optional): Optional existing aiohttp session.

        Returns:
            pd.DataFrame: Stock quote data.
        """
        if not self.api_key:
            self.logger.error("Finnhub API key is missing. Cannot fetch stock data.")
            return pd.DataFrame()

        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.api_key}"
        self.logger.debug(f"Fetching Finnhub stock data for {symbol} from URL: {url}")

        async_session = session or aiohttp.ClientSession()
        close_session = session is None  # Close session only if created here

        try:
            async with async_session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    df = pd.DataFrame([{
                        'date': pd.to_datetime(data.get('t', 0), unit='s'),
                        'open': data.get('o'),
                        'high': data.get('h'),
                        'low': data.get('l'),
                        'close': data.get('c'),
                        'previous_close': data.get('pc'),
                        'change': data.get('d'),
                        'percent_change': data.get('dp'),
                        'volume': data.get('v', 0)
                    }]).set_index('date')

                    self.logger.info(f"Fetched Finnhub stock data for {symbol}.")
                    return df
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to fetch Finnhub stock data for {symbol}: {response.status} - {error_text}")
        except Exception as e:
            self.logger.error(f"Exception while fetching Finnhub stock data for {symbol}: {e}")
        finally:
            if close_session:
                await async_session.close()

        return pd.DataFrame()
