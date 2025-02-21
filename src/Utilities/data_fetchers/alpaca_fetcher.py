import os
import aiohttp
import pandas as pd
import logging
from typing import Optional
from Utilities.shared_utils import setup_logging

class AlpacaDataFetcher:
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initializes Alpaca Data Fetcher.
        Args:
            logger (Optional[logging.Logger]): Logger instance. Defaults to None.
        """
        self.logger = logger or setup_logging(script_name="AlpacaFetcher")
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = "https://paper-api.alpaca.markets"
        if not self.api_key or not self.api_secret:
            self.logger.error("Alpaca API credentials are missing.")

    async def fetch_stock_data_async(
        self, symbol: str, start_date: str, end_date: str, session: Optional[aiohttp.ClientSession] = None
    ) -> pd.DataFrame:
        """
        Fetches stock data from Alpaca asynchronously.
        Args:
            symbol (str): Stock symbol.
            start_date (str): Start date (YYYY-MM-DD).
            end_date (str): End date (YYYY-MM-DD).
            session (Optional[aiohttp.ClientSession]): Optional shared session.
        Returns:
            pd.DataFrame: Stock data.
        """
        url = f"{self.base_url}/v2/stocks/{symbol}/bars?start={start_date}&end={end_date}&timeframe=1Day"
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

        self.logger.debug(f"Fetching Alpaca data for {symbol} from URL: {url}")

        session_created = False
        if session is None:
            session = aiohttp.ClientSession()
            session_created = True

        try:
            async with session.get(url, headers=headers, timeout=30) as response:
                self.logger.debug(f"Received response with status {response.status} for {symbol}.")
                if response.status == 200:
                    data = await response.json()
                    self.logger.debug(f"Alpaca raw response type: {type(data)}")
                    bars = data.get("bars", [])
                    if not bars:
                        self.logger.warning(f"No data returned from Alpaca for {symbol}.")
                        return pd.DataFrame()
                    df = pd.DataFrame(bars)
                    df["date"] = pd.to_datetime(df["t"]).dt.date  # Convert timestamp to date
                    df.rename(
                        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"},
                        inplace=True,
                    )
                    df.set_index("date", inplace=True)
                    self.logger.info(f"Fetched {len(df)} records for {symbol} from Alpaca.")
                    return df
                else:
                    error_text = await response.text()
                    self.logger.error(f"Failed to fetch Alpaca data for {symbol}: {response.status} - {error_text}")
        except Exception as e:
            self.logger.error(f"Exception while fetching Alpaca data for {symbol}: {e}", exc_info=True)
        finally:
            if session_created:
                await session.close()

        return pd.DataFrame()
