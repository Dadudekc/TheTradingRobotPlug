# -------------------------------------------------------------------
# File: polygon_fetcher.py
# Location: src/Utilities/fetchers
# Description: Fetches historical stock data from Polygon.io API.
# -------------------------------------------------------------------

import os
import requests
import pandas as pd
import logging
from datetime import datetime
from typing import Optional
from Utilities.shared_utils import setup_logging  # Centralized logging setup

class PolygonDataFetcher:
    """
    Fetches historical stock data from Polygon.io.
    """

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initializes the PolygonDataFetcher with API credentials and logging.
        
        Args:
            log_dir (Optional[str]): Directory to store log files.
        """
        self.logger = setup_logging(script_name="polygon_fetcher", log_dir=log_dir)
        self.api_key = os.getenv("POLYGON_API_KEY")

        if not self.api_key:
            self.logger.error("Polygon API key is missing. Set POLYGON_API_KEY in your environment.")
        
        self.base_url = "https://api.polygon.io"
        self.logger.info("PolygonDataFetcher initialized.")

    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str, interval: str = "day") -> pd.DataFrame:
        """
        Fetches historical stock data from Polygon.io.
        
        Args:
            symbol (str): Stock ticker symbol.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            interval (str, optional): Data interval. Defaults to "day".

        Returns:
            pd.DataFrame: Historical stock data, or empty DataFrame if request fails.
        """
        if not self.api_key:
            self.logger.error("Polygon API key is missing. Cannot fetch data.")
            return pd.DataFrame()

        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/{interval}/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={self.api_key}"
        self.logger.info(f"Requesting Polygon.io data for {symbol} from {start_date} to {end_date}...")

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "results" not in data:
                self.logger.warning(f"No data found for {symbol} from {start_date} to {end_date}.")
                return pd.DataFrame()

            df = pd.DataFrame(data["results"])
            df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["symbol"] = symbol
            df.set_index("timestamp", inplace=True)

            self.logger.info(f"Successfully fetched {len(df)} records for {symbol} from Polygon.io.")
            return df

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data from Polygon.io for {symbol}: {e}")
            return pd.DataFrame()
