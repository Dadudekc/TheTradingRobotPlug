# File: alphavantage_fetcher.py
# Location: src/Utilities/data_fetchers
# Description: Fetches data from Alpha Vantage API.

import os
import aiohttp
import pandas as pd
import logging
from typing import Optional
from Utilities.shared_utils import setup_logging


class AlphaVantageFetcher:
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initializes AlphaVantage Fetcher.

        Args:
            logger (Optional[logging.Logger]): Logger instance. Defaults to None.
        """
        self.logger = logger or setup_logging(script_name="AlphaVantageFetcher")
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"

        if not self.api_key:
            self.logger.error("AlphaVantage API key is missing.")

    async def fetch_stock_data_async(
        self, symbol: str, start_date: str, end_date: str, session: Optional[aiohttp.ClientSession] = None
    ) -> pd.DataFrame:
        """
        Fetches stock data from AlphaVantage asynchronously.

        Args:
            symbol (str): Stock symbol.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            session (Optional[aiohttp.ClientSession]): Optional shared session.

        Returns:
            pd.DataFrame: Stock data.
        """
        url = f"{self.base_url}?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={self.api_key}&outputsize=full"
        self.logger.debug(f"Fetching AlphaVantage data for {symbol} from {url}")

        session_created = False
        if session is None:
            session = aiohttp.ClientSession()
            session_created = True

        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    time_series = data.get("Time Series (Daily)", {})

                    if not time_series:
                        self.logger.warning(f"No data returned from AlphaVantage for {symbol}.")
                        return pd.DataFrame()

                    df = pd.DataFrame.from_dict(time_series, orient="index")
                    df.index = pd.to_datetime(df.index)
                    df.rename(
                        columns={
                            "1. open": "open",
                            "2. high": "high",
                            "3. low": "low",
                            "4. close": "close",
                            "6. volume": "volume",
                        },
                        inplace=True,
                    )
                    df.sort_index(inplace=True)

                    self.logger.info(f"Fetched {len(df)} records for {symbol} from AlphaVantage.")
                    return df
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to fetch AlphaVantage data for {symbol}: {response.status} - {error_text}")

        except Exception as e:
            self.logger.error(f"Exception while fetching AlphaVantage data for {symbol}: {e}")

        finally:
            if session_created:
                await session.close()

        return pd.DataFrame()
