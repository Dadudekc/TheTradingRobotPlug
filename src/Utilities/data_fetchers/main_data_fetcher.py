"""
File: main_data_fetcher.py
Location: src/Utilities/data_fetchers

Description:
    Orchestrates fetching stock data from multiple sources (Alpaca, AlphaVantage, Finnhub, Yahoo, NewsAPI),
    applies technical indicators using AllIndicatorsUnifier, and stores data in PostgreSQL.
    Includes detailed debug logs for verifying missing columns or technical indicators.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
import yfinance as yf
from aiohttp import ClientSession
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

# -------------------------------------------------------------------------
# Lazy Imports for modules from src.Utilities to avoid circular imports
# -------------------------------------------------------------------------
def get_setup_logging():
    from src.Utilities.config_manager import setup_logging
    return setup_logging

def get_indicator_unifier(logger):
    from src.Utilities.data_processing.Technical_Indicators.indicator_unifier import AllIndicatorsUnifier
    return AllIndicatorsUnifier(
        config_manager=None,
        logger=logger,
        use_csv=False
    )

def get_clean_stock_data():
    from src.Utilities.db.fix_stock_data import clean_stock_data
    return clean_stock_data

# -------------------------------------------------------------------------
# Data Fetchers (direct imports from Utilities are typically fine)
# -------------------------------------------------------------------------
from Utilities.data_fetchers.yahoo_finance_fetcher import YahooFinanceFetcher
from Utilities.data_fetchers.finnhub_fetcher import FinnhubFetcher
from Utilities.data_fetchers.alpaca_fetcher import AlpacaDataFetcher  # or AlpacaFetcher if different
from Utilities.data_fetchers.alphavantage_fetcher import AlphaVantageFetcher
from Utilities.data_fetchers.newsapi_fetcher import NewsAPIFetcher

# -------------------------------------------------------------------------
# SQLAlchemy Models
# -------------------------------------------------------------------------
from Utilities.data.models.alpaca_data import AlpacaData
from Utilities.data.models.alpha_vantage_data import AlphaVantageData
from Utilities.data.models.alpha_vantage_daily import AlphaVantageDaily
from Utilities.data.models.finnhub_quote import FinnhubQuote
from Utilities.data.models.finnhub_metrics import FinnhubMetrics
from Utilities.data.models.news_article import NewsArticle
from Utilities.data.models.base import Base

# -------------------------------------------------------------------------
# Environment & Database Setup
# -------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[3]
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

# -------------------------------------------------------------------------
# Logging Setup (using our lazy import helper)
# -------------------------------------------------------------------------
try:
    setup_logging = get_setup_logging()
    logger = setup_logging(
        script_name="main_data_fetcher",
        log_file=os.path.join("logs", "data_fetch_utils.log"),
        level=logging.INFO
    )
except TypeError:
    try:
        setup_logging = get_setup_logging()
        logger = setup_logging(
            script_name="main_data_fetcher",
            log_file=os.path.join("logs", "data_fetch_utils.log")
        )
    except TypeError:
        setup_logging = get_setup_logging()
        logger = setup_logging(script_name="main_data_fetcher")

logger.info("Logger initialized for main_data_fetcher.")


class DataOrchestrator:
    """
    Orchestrates data fetching from multiple sources, applies technical indicators, and stores data in PostgreSQL.
    """
    def __init__(self):
        self.logger = logger
        self.logger.info("Initializing DataOrchestrator...")

        # -------------------- Data Fetchers -------------------- #
        self.yahoo_finance = YahooFinanceFetcher(self.logger)
        self.finnhub = FinnhubFetcher(self.logger)
        self.alpaca = AlpacaDataFetcher(self.logger)
        self.alphavantage = AlphaVantageFetcher(self.logger)
        self.newsapi = NewsAPIFetcher(self.logger)

        # For backward compatibility:
        self.alpaca_api = self.alpaca

        # -------------------- Indicator Aggregator -------------------- #
        self.indicator_unifier = get_indicator_unifier(self.logger)

    # ----------------------------------------------------------------
    # Yahoo Finance fetch
    # ----------------------------------------------------------------
    async def fetch_stock_data_async(self, ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        self.logger.info(f"Fetching {ticker} data from Yahoo Finance...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            df.index.name = 'date'
            # Flatten columns if MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                self.logger.debug(f"MultiIndex detected for {ticker}: {df.columns}")
                df.columns = ["_".join(map(str, col)).lower().replace(" ", "_") for col in df.columns.values]
            else:
                df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            return df
        except Exception as e:
            self.logger.error(f"Error fetching {ticker} via Yahoo Finance: {e}", exc_info=True)
            return pd.DataFrame()

    # ----------------------------------------------------------------
    # Alpaca fetch
    # ----------------------------------------------------------------
    async def fetch_alpaca_data_async(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        if not self.alpaca_api:
            self.logger.warning("Alpaca API client not configured.")
            return pd.DataFrame()
        self.logger.info(f"Fetching {symbol} data from Alpaca...")
        try:
            bars = self.alpaca_api.get_bars(symbol, start_date, end_date, interval)
            self.logger.debug(f"Alpaca get_bars returned type: {type(bars)}")
            if isinstance(bars, str):
                self.logger.error(f"Alpaca API returned a string instead of expected object: {bars}")
                return pd.DataFrame()
            df = bars.df.copy()  # Expecting an object with .df attribute
            df.rename(columns={'timestamp': 'date', 'trade_count': 'volume'}, inplace=True)
            df['date'] = pd.to_datetime(df['date']).dt.date
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching Alpaca data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    # ----------------------------------------------------------------
    # Generic fetch with retries
    # ----------------------------------------------------------------
    async def fetch_with_retries(self, url: str, headers: dict, session: ClientSession, retries: int):
        for attempt in range(retries):
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
            except Exception as e:
                self.logger.error(f"Attempt {attempt+1} failed for {url}: {e}", exc_info=True)
            await asyncio.sleep(0.5)
        return None

    # ----------------------------------------------------------------
    # Data Gathering for Multiple Symbols
    # ----------------------------------------------------------------
    async def fetch_data_for_multiple_symbols(self, symbols: list, data_sources: list, start_date: str, end_date: str, interval: str) -> dict:
        results = {}
        for sym in symbols:
            df = await self.fetch_stock_data_async(sym, start_date, end_date, interval)
            results[sym] = {"Alpha Vantage": df}
        return results

    # ----------------------------------------------------------------
    # Full Data Orchestration
    # ----------------------------------------------------------------
    async def fetch_all_data(
        self,
        symbols: List[str],
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d",
        max_concurrent_tasks: int = 5
    ) -> dict:
        self.logger.info("Starting to fetch data for multiple symbols.")
        async with ClientSession() as session:
            tasks = [self._fetch_symbol_data(sym, start_date, end_date, interval, session) for sym in symbols]
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            sem_tasks = [self._run_with_semaphore(task, semaphore) for task in tasks]
            results = await asyncio.gather(*sem_tasks, return_exceptions=True)
        all_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching data for {symbol}: {result}", exc_info=True)
                all_data[symbol] = None
            else:
                all_data[symbol] = result
                self.store_data_in_db(symbol, result)
        self.logger.info("Completed fetching data for all symbols.")
        return all_data

    async def _run_with_semaphore(self, task, semaphore: asyncio.Semaphore):
        async with semaphore:
            return await task

    async def _fetch_symbol_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str],
        interval: str,
        session: ClientSession
    ) -> Dict[str, pd.DataFrame]:
        self.logger.info(f"Fetching data for {symbol}.")
        data = {}

        # Yahoo Finance fetch
        try:
            yf_data = await self.yahoo_finance.fetch_stock_data_async(symbol, start_date, end_date, interval)
            if yf_data is not None and not yf_data.empty:
                data["Yahoo Finance"] = yf_data
            else:
                self.logger.error(f"Yahoo Finance fetch returned empty for {symbol}.")
        except Exception as e:
            self.logger.error(f"Yahoo Finance fetch failed for {symbol}: {e}", exc_info=True)

        # Alpaca fetch
        try:
            alpaca_df = await self.alpaca.fetch_stock_data_async(symbol, start_date, end_date, interval)
            if alpaca_df is not None and not alpaca_df.empty:
                data["Alpaca"] = alpaca_df
            else:
                self.logger.error(f"Alpaca fetch returned empty for {symbol}.")
        except Exception as e:
            self.logger.error(f"Alpaca fetch failed for {symbol}: {e}", exc_info=True)

        # Finnhub fetch
        try:
            if hasattr(self.finnhub, "fetch_quote"):
                finnhub_quote = await self.finnhub.fetch_quote(symbol, session)
            else:
                self.logger.error(f"FinnhubFetcher does not have fetch_quote for {symbol}.")
                finnhub_quote = pd.DataFrame()
            if hasattr(self.finnhub, "fetch_financial_metrics"):
                finnhub_metrics = await self.finnhub.fetch_financial_metrics(symbol, session)
            else:
                self.logger.error(f"FinnhubFetcher does not have fetch_financial_metrics for {symbol}.")
                finnhub_metrics = pd.DataFrame()
            if finnhub_quote is not None and not finnhub_quote.empty:
                data["Finnhub Quote"] = finnhub_quote
            else:
                self.logger.error(f"Finnhub quote fetch returned empty for {symbol}.")
            if finnhub_metrics is not None and not finnhub_metrics.empty:
                data["Finnhub Metrics"] = finnhub_metrics
            else:
                self.logger.error(f"Finnhub metrics fetch returned empty for {symbol}.")
        except Exception as e:
            self.logger.error(f"Finnhub fetch failed for {symbol}: {e}", exc_info=True)

        # AlphaVantage fetch
        try:
            alpha_df = await self.alphavantage.fetch_stock_data_async(symbol, start_date, end_date, interval)
            if alpha_df is not None and not alpha_df.empty:
                data["Alpha Vantage"] = alpha_df
            else:
                self.logger.error(f"Alpha Vantage fetch returned empty for {symbol}.")
        except Exception as e:
            self.logger.error(f"Alpha Vantage fetch failed for {symbol}: {e}", exc_info=True)

        # NewsAPI fetch
        try:
            news_data = await self.newsapi.fetch_news_data_async(symbol, page_size=5)
            if news_data is not None and not news_data.empty:
                data["NewsAPI"] = news_data
            else:
                self.logger.error(f"NewsAPI fetch returned empty for {symbol}.")
        except Exception as e:
            self.logger.error(f"NewsAPI fetch failed for {symbol}: {e}", exc_info=True)
            # Ensure that in newsapi_fetcher.py, 'import asyncio' is present.
        self.logger.info(f"Data fetching completed for {symbol}.")
        return data

    def store_data_in_db(self, symbol: str, data: Dict[str, pd.DataFrame]):
        self.logger.info(f"Storing data for {symbol} in the database...")
        session: Session = SessionLocal()
        try:
            if "Yahoo Finance" in data and not data["Yahoo Finance"].empty:
                df_yf = data["Yahoo Finance"].copy()
                df_yf = self.indicator_unifier.apply_all_indicators(df_yf)
                df_yf.reset_index(inplace=True)
                records_yf = df_yf.to_dict(orient='records')
                yahoo_objects = []
                for row in records_yf:
                    yahoo_objects.append(
                        AlphaVantageDaily(
                            symbol=row.get('symbol', symbol),
                            date=row.get('date'),
                            open=row.get('open'),
                            high=row.get('high'),
                            low=row.get('low'),
                            close=row.get('close'),
                            volume=row.get('volume')
                        )
                    )
                session.bulk_save_objects(yahoo_objects, update_existing=True)
                self.logger.info(f"Inserted/Updated Yahoo Finance data for {symbol}.")
            # Additional storage logic for other data sources...
            session.commit()
            self.logger.info(f"✅ Successfully stored data for {symbol} in the database.")
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Database error while storing data for {symbol}: {e}")
        except Exception as e:
            session.rollback()
            self.logger.error(f"Unexpected error while storing data for {symbol}: {e}")
        finally:
            session.close()

    def initialize_database(self, symbol: str):
        pass


def finalize_data():
    try:
        print("Running stock data cleanup...")
        clean_stock_data = get_clean_stock_data()
        clean_stock_data()
        print("Stock Data Cleanup Completed Successfully!")
    except Exception as e:
        print(f"Stock data cleanup failed: {e}")


async def main():
    orchestrator = DataOrchestrator()
    symbols = ["AAPL", "TSLA"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    interval = "1d"
    try:
        results = await orchestrator.fetch_all_data(symbols, start_date, end_date, interval)
        orchestrator.logger.info("\nData Fetching & Storage Completed Successfully!\n")
        print("\nData Fetching & Storage Completed Successfully!\n")
    except Exception as e:
        orchestrator.logger.error(f"Fatal error during data fetching: {e}")
        print(f"Fatal error during data fetching: {e}")


if __name__ == "__main__":
    asyncio.run(main())
    finalize_data()
