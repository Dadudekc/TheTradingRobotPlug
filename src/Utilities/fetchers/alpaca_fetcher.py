# -------------------------------------------------------------------
# File: alpaca_fetcher.py
# Location: src/Utilities/fetchers
# Description: Fetches stock data from the Alpaca API.
# -------------------------------------------------------------------

import os
import pandas as pd
import alpaca_trade_api as tradeapi
from typing import Optional
from Utilities.shared_utils import setup_logging  # Import the global logger setup

class AlpacaDataFetcher:
    """Handles fetching stock data from the Alpaca API asynchronously."""

    def __init__(self):
        """Initializes Alpaca API client and logging."""
        self.logger = setup_logging("alpaca_fetcher")  # Use global logging setup
        self.logger.info("Initializing AlpacaDataFetcher...")

        # Load API Credentials
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not self.api_key or not self.secret_key:
            self.logger.error("Alpaca API credentials are missing.")
            self.alpaca_api = None
            return

        try:
            self.alpaca_api = tradeapi.REST(
                self.api_key, self.secret_key, self.base_url, api_version='v2'
            )
            self.logger.info("Alpaca API client initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca API client: {e}")
            self.alpaca_api = None

    async def fetch_stock_data_async(
        self, symbol: str, start_date: str, end_date: str, interval: str = "1Day"
    ) -> pd.DataFrame:
        """
        Fetches historical stock data asynchronously from Alpaca.

        Args:
            symbol (str): Stock symbol.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            interval (str, optional): Data interval. Defaults to "1Day".

        Returns:
            pd.DataFrame: Historical stock data.
        """
        if not self.alpaca_api:
            self.logger.error("Alpaca API client is not initialized.")
            return pd.DataFrame()

        try:
            bars = self.alpaca_api.get_bars(
                symbol,
                timeframe=interval,
                start=pd.to_datetime(start_date).strftime('%Y-%m-%d'),
                end=pd.to_datetime(end_date).strftime('%Y-%m-%d')
            ).df

            if bars.empty:
                self.logger.warning(f"No data returned from Alpaca for {symbol}.")
                return pd.DataFrame()

            # Process data
            bars.reset_index(drop=True, inplace=True)
            bars.rename(columns={'timestamp': 'date', 'trade_count': 'volume'}, inplace=True)
            bars['date'] = pd.to_datetime(bars['date']).dt.date
            bars.set_index('date', inplace=True)
            bars['symbol'] = symbol  # Add symbol column

            self.logger.info(f"Fetched {len(bars)} records for {symbol} from Alpaca.")
            return bars

        except Exception as e:
            self.logger.error(f"Error fetching Alpaca data for {symbol}: {e}")
            return pd.DataFrame()
