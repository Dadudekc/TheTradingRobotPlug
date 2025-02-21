# File: stock_data_agent.py
# Location: src/Utilities
# Description: Agent interface for fetching stock data from multiple sources.

import asyncio
import logging
import aiohttp
import pandas as pd
from typing import Dict, Any

class StockDataAgent:
    """
    Agent interface for fetching stock data from multiple sources.
    """

    def __init__(self, fetcher) -> None:
        """
        Initialize the StockDataAgent with a provided fetcher instance.
        
        Args:
            fetcher: An instance that provides sub-components like finnhub, alpaca, alphavantage, newsapi, and async_fetcher.
        """
        self.logger = logging.getLogger("StockDataAgent")
        self.fetcher = fetcher

    async def get_real_time_quote(self, symbol: str) -> dict:
        self.logger.info(f"Fetching real-time quote for {symbol}...")
        async with aiohttp.ClientSession() as session:
            try:
                return await self.fetcher.finnhub.fetch_quote(symbol, session)
            except Exception as e:
                self.logger.error(f"Failed to fetch real-time quote for {symbol}: {e}")
                return {}

    async def get_historical_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1Day") -> pd.DataFrame:
        self.logger.info(f"Fetching historical data for {symbol} from Alpaca...")
        try:
            return await self.fetcher.alpaca.fetch_stock_data_async(symbol, start_date, end_date, interval=interval)
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def get_historical_data_alpha_vantage(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        self.logger.info(f"Fetching historical data for {symbol} from Alpha Vantage...")
        async with aiohttp.ClientSession() as session:
            try:
                return await self.fetcher.alphavantage.fetch_stock_data_async(symbol, start_date, end_date)
            except Exception as e:
                self.logger.error(f"Failed to fetch historical data for {symbol} from Alpha Vantage: {e}")
                return pd.DataFrame()

    async def get_news(self, symbol: str, page_size: int = 3) -> pd.DataFrame:
        self.logger.info(f"Fetching recent news articles for {symbol}...")
        try:
            return await self.fetcher.newsapi.fetch_news_data_async(symbol, page_size=page_size)
        except Exception as e:
            self.logger.error(f"Failed to fetch news for {symbol}: {e}")
            return pd.DataFrame()

    async def get_combined_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> Dict[str, Any]:
        self.logger.info(f"Fetching combined data for {symbol} from multiple sources...")
        try:
            # Create a session using the async_fetcher sub-component
            session = self.fetcher.async_fetcher.create_session()
            return await self.fetcher.fetch_symbol_data(symbol, start_date, end_date, interval, session)
        except Exception as e:
            self.logger.error(f"Failed to fetch combined data for {symbol}: {e}")
            return {}
