# -------------------------------------------------------------------
# File: yahoo_finance_fetcher.py
# Location: src/Utilities/fetchers
# Description: Fetches stock data from Yahoo Finance.
# -------------------------------------------------------------------

import asyncio
import yfinance as yf
import pandas as pd
from typing import Optional
import logging

class YahooFinanceFetcher:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical stock data using yfinance.

        Args:
            ticker (str): Stock symbol.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            interval (str, optional): Data interval. Defaults to "1d".

        Returns:
            pd.DataFrame: Historical stock data.
        """
        self.logger.info(f"Fetching {ticker} data from Yahoo Finance.")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            
            # Check if the data is a tuple and extract the DataFrame
            if isinstance(data, tuple):
                data = data[0]

            if isinstance(data.index, pd.DatetimeIndex):
                data.reset_index(inplace=True)

            # Rename columns to lowercase and replace spaces with underscores
            data.rename(columns={col: col.lower().replace(' ', '_') for col in data.columns}, inplace=True)

            # Convert 'date' to datetime objects
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date']).dt.date
            else:
                raise ValueError("Stock data does not contain 'date' column after reset_index.")

            data['symbol'] = ticker

            # Set 'date' as index
            data.set_index('date', inplace=True)

            self.logger.info(f"Fetched {len(data)} records for {ticker} from Yahoo Finance.")
            return data
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {ticker} from Yahoo Finance: {e}")
            return pd.DataFrame()

    async def fetch_stock_data_async(self, ticker: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
        """
        Asynchronously fetches historical stock data using yfinance.

        Args:
            ticker (str): Stock symbol.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            interval (str, optional): Data interval. Defaults to "1d".

        Returns:
            pd.DataFrame: Historical stock data.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fetch_stock_data, ticker, start_date, end_date, interval)
