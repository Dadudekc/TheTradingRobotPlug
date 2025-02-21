"""
File: main_data_fetcher.py
Location: src/Utilities

Description:
    Orchestrates fetching stock data from multiple sources (Alpaca, AlphaVantage, Finnhub, Yahoo, NewsAPI),
    applies technical indicators using AllIndicatorsUnifier, and stores data in PostgreSQL.
    Includes debug logs for verifying missing columns or technical indicators.
"""

import asyncio
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
from aiohttp import ClientSession
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# --------------------------------------------------------------------------------
# Data Fetchers (Adjust imports if your modules differ)
# --------------------------------------------------------------------------------
from Utilities.data_fetchers.alpaca_fetcher import AlpacaDataFetcher
from Utilities.data_fetchers.alphavantage_fetcher import AlphaVantageFetcher
from Utilities.data_fetchers.finnhub_fetcher import FinnhubFetcher
from Utilities.data_fetchers.yahoo_finance_fetcher import YahooFinanceFetcher
from Utilities.data_fetchers.newsapi_fetcher import NewsAPIFetcher

# --------------------------------------------------------------------------------
# Technical Indicators
# --------------------------------------------------------------------------------
from Utilities.data_processing.Technical_Indicators.indicator_aggregator import AllIndicatorsUnifier

# --------------------------------------------------------------------------------
# Shared Utilities
# --------------------------------------------------------------------------------
from Utilities.shared_utils import setup_logging

# --------------------------------------------------------------------------------
# SQLAlchemy Models
# --------------------------------------------------------------------------------
from Utilities.data.models.alpha_vantage_daily import AlphaVantageDaily
from Utilities.data.models.news_article import NewsArticle
from Utilities.data.models.finnhub_quote import FinnhubQuote
from Utilities.data.models.finnhub_metrics import FinnhubMetrics
from Utilities.data.models.alpaca_data import AlpacaData
from Utilities.data.models.alpha_vantage_data import AlphaVantageData
from Utilities.data.models.base import Base

# --------------------------------------------------------------------------------
# Environment & DB Setup
# --------------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
env_file = project_root / ".env"
if not env_file.exists():
    raise FileNotFoundError(f"⚠️ .env file not found at {env_file}")

load_dotenv(env_file)

DB_NAME = os.getenv("POSTGRES_DBNAME", "trading_robot_plug")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "your_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5434")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

# --------------------------------------------------------------------------------
# Logging Setup
# --------------------------------------------------------------------------------
logger = setup_logging(script_name="main_data_fetcher", log_dir=project_root / "logs")
logger.info("Logger initialized for main_data_fetcher.")

# --------------------------------------------------------------------------------
# Data Orchestrator
# --------------------------------------------------------------------------------
class DataOrchestrator:
    """
    Orchestrates data fetching from multiple sources, applies technical indicators, 
    and stores data in PostgreSQL.
    """

    def __init__(self):
        self.logger = logger
        self.logger.info("Initializing DataOrchestrator...")

        # Initialize fetchers with logger
        self.alpaca = AlpacaDataFetcher(self.logger)
        self.alphavantage = AlphaVantageFetcher(self.logger)
        self.finnhub = FinnhubFetcher(self.logger)
        self.yahoo_finance = YahooFinanceFetcher(self.logger)
        self.newsapi = NewsAPIFetcher(self.logger)

        # Initialize the aggregator for technical indicators
        self.indicator_aggregator = AllIndicatorsUnifier(
            config_manager=None,
            logger=self.logger,
            use_csv=False
        )

    async def fetch_data_for_symbol(
        self, 
        symbol: str, 
        start_date: str = "2023-01-01", 
        end_date: Optional[str] = "2023-12-31", 
        interval: str = "1d",
        session: Optional[ClientSession] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches stock data from each fetcher for one symbol, returning a dict of DataFrames keyed by the source name.
        """
        self.logger.info(f"Fetching data for '{symbol}' from multiple sources.")
        results = {}
        
        async_session = session or ClientSession()
        close_session = session is None

        try:
            tasks = [
                self.alpaca.fetch_stock_data_async(symbol, start_date, end_date, session=async_session),
                self.alphavantage.fetch_stock_data_async(symbol, start_date, end_date, session=async_session),
                self.finnhub.fetch_stock_data_async(symbol, session=async_session),
            ]

            fetched = await asyncio.gather(*tasks, return_exceptions=True)

            results["Alpaca"] = fetched[0] if not isinstance(fetched[0], Exception) else pd.DataFrame()
            results["AlphaVantage"] = fetched[1] if not isinstance(fetched[1], Exception) else pd.DataFrame()
            results["Finnhub"] = fetched[2] if not isinstance(fetched[2], Exception) else pd.DataFrame()

            # Yahoo data (synchronous)
            try:
                yahoo_df = self.yahoo_finance.fetch_data_sync(symbol, start_date, end_date, interval)
                results["YahooFinance"] = yahoo_df if isinstance(yahoo_df, pd.DataFrame) else pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Yahoo fetch failed for {symbol}: {e}")
                results["YahooFinance"] = pd.DataFrame()

            # NewsAPI data (async)
            try:
                news_data = await self.newsapi.fetch_news_data_async(symbol, page_size=5, session=async_session)
                results["NewsAPI"] = news_data if isinstance(news_data, pd.DataFrame) else pd.DataFrame()
            except Exception as e:
                self.logger.error(f"NewsAPI fetch failed for {symbol}: {e}")
                results["NewsAPI"] = pd.DataFrame()

        finally:
            if close_session:
                await async_session.close()

        # Log summary of fetched data
        for source, df in results.items():
            if df.empty:
                self.logger.warning(f"No data from {source} for {symbol}.")
            else:
                self.logger.info(f"Fetched {len(df)} records from {source} for {symbol}.")

        return results

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the technical indicators from AllIndicatorsUnifier.
        """
        if df.empty:
            self.logger.warning("apply_indicators(): DataFrame is empty, skipping.")
            return df

        self.logger.debug(f"Columns BEFORE applying indicators: {df.columns.tolist()}")
        indicated_df = self.indicator_aggregator.apply_all_indicators(df)
        self.logger.debug(f"Columns AFTER applying indicators: {indicated_df.columns.tolist()}")
        return indicated_df

    def store_data_in_db(self, symbol: str, data: Dict[str, pd.DataFrame]):
        """
        Stores fetched data (with optional indicators) into PostgreSQL.
        """
        session = SessionLocal()
        self.logger.info(f"Storing data in DB for '{symbol}'...")

        try:
            # Example: storing YahooFinance data
            if "YahooFinance" in data and not data["YahooFinance"].empty:
                yahoo_df = data["YahooFinance"].copy()
                self.logger.debug(f"YahooFinance columns pre-indicators: {yahoo_df.columns.tolist()}")

                # Apply technical indicators
                yahoo_df = self.apply_indicators(yahoo_df)
                self.logger.debug(f"YahooFinance columns post-indicators: {yahoo_df.columns.tolist()}")

                yahoo_df.reset_index(drop=True, inplace=True)
                records = yahoo_df.to_dict(orient="records")
                yf_objects = [
                    AlphaVantageDaily(
                        symbol=row.get("symbol"),
                        date=row["date"].date() if isinstance(row["date"], pd.Timestamp) else row["date"],
                        open=row.get("open"),
                        high=row.get("high"),
                        low=row.get("low"),
                        close=row.get("close"),
                        volume=row.get("volume"),
                        sma_50=row.get("SMA_50"),
                        sma_200=row.get("SMA_200"),
                    )
                    for row in records
                ]
                session.bulk_save_objects(yf_objects, update_existing=True)
                self.logger.info(f"Inserted/Updated YahooFinance data for {symbol}.")

            # Repeat similar steps for other data sources: Alpaca, AlphaVantage, Finnhub, NewsAPI...
            # For example: data["Alpaca"], data["AlphaVantage"], etc.

            session.commit()
            self.logger.info(f"Successfully stored data for '{symbol}' in DB.")

        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"DB error storing {symbol}: {e}", exc_info=True)
        except Exception as e:
            session.rollback()
            self.logger.error(f"Unexpected error storing {symbol}: {e}", exc_info=True)
        finally:
            session.close()

    async def fetch_all_data(
        self,
        symbols: List[str],
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d",
        max_concurrent_tasks: int = 5
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        High-level function: For each symbol, fetch data from sources, store in DB.
        Returns a nested dict: { symbol: { "Alpaca": df, "AlphaVantage": df, ... } }
        """
        self.logger.info("Starting data fetch for multiple symbols.")
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = [
            self._run_with_semaphore(
                self.fetch_data_for_symbol(symbol, start_date, end_date, interval), 
                semaphore
            )
            for symbol in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        final_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching data for {symbol}: {result}")
                final_data[symbol] = {}
            else:
                final_data[symbol] = result
                self.store_data_in_db(symbol, result)

        self.logger.info("Completed data fetch for all symbols.")
        return final_data

    async def _run_with_semaphore(self, task, semaphore):
        """Runs a task with concurrency control."""
        async with semaphore:
            try:
                return await task
            except Exception as e:
                self.logger.error(f"Task encountered an error: {e}", exc_info=True)
                return e

# --------------------------------------------------------------------------------
# Main Entry & Data Cleanup
# --------------------------------------------------------------------------------
from src.Utilities.db.fix_stock_data import clean_stock_data  # Import to avoid circular dependency

def finalize_data():
    """
    Final step in the data pipeline: Cleans stock data after all fetching & storing.
    """
    try:
        print("Running stock data cleanup...")
        clean_stock_data()
        print("Stock Data Cleanup Completed Successfully!")
    except Exception as e:
        print(f"Stock data cleanup failed: {e}")

async def main():
    orchestrator = DataOrchestrator()
    
    # Example usage: define symbols and parameters
    symbols = ["AAPL", "TSLA"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    interval = "1d"

    try:
        results = await orchestrator.fetch_all_data(symbols, start_date, end_date, interval)
        orchestrator.logger.info("Data Fetching & Storage Completed Successfully!\n")
        print("Data Fetching & Storage Completed Successfully!\n")
    except Exception as e:
        orchestrator.logger.error(f"Fatal error during data fetching: {e}", exc_info=True)
        print(f"Fatal error during data fetching: {e}")

if __name__ == "__main__":
    asyncio.run(main())
    finalize_data()
