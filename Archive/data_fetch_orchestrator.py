# -------------------------------------------------------------------
# File Path: src/Utilities/data_fetch_utils.py
# Description: Orchestrates fetching stock data from multiple sources,
#              applies technical indicators using AllIndicatorsUnifier,
#              and stores data in PostgreSQL.
# -------------------------------------------------------------------

import asyncio
import logging
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine

from aiohttp import ClientSession
import os
from dotenv import load_dotenv

import pandas as pd

# Import fetchers
from Utilities.data_fetchers.finnhub_fetcher import FinnhubFetcher
from Utilities.data_fetchers.alpaca_fetcher import AlpacaDataFetcher
from Utilities.data_fetchers.alphavantage_fetcher import AlphaVantageFetcher
from Utilities.data_fetchers.newsapi_fetcher import NewsAPIFetcher


# Import indicators aggregator
def get_indicator_aggregator(logger):
    from Utilities.data_processing.Technical_Indicators.indicator_aggregator import AllIndicatorsUnifier
    return AllIndicatorsUnifier(
        config_manager=None,  # Provide ConfigManager instance if required
        logger=logger,
        use_csv=False
    )


# Shared utilities
from Utilities.shared_utils import setup_logging

# Import SQLAlchemy models
from Utilities.data.models.alpha_vantage_daily import AlphaVantageDaily
from Utilities.data.models.news_article import NewsArticle
from Utilities.data.models.finnhub_quote import FinnhubQuote
from Utilities.data.models.finnhub_metrics import FinnhubMetrics

from Utilities.data.models.alpaca_data import AlpacaData
from Utilities.data.models.alpha_vantage_data import AlphaVantageData
from Utilities.data.models.base import Base


# Load environment variables
load_dotenv()

# Database Connection Parameters
DB_NAME = os.getenv("POSTGRES_DBNAME", "trading_robot_plug")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "your_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5434")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Setup SQLAlchemy Engine and Session
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = Session(bind=engine)

# Create all tables
Base.metadata.create_all(bind=engine)


class MainDataFetcher:
    def __init__(self):
        """Initializes fetchers, indicators aggregator, and logging."""
        self.logger = setup_logging(
            script_name="MainDataFetcher",
            log_file=os.path.join("logs", "data_fetch_utils.log"),
            level=logging.INFO
        )

        self.logger.info("Initializing MainDataFetcher...")

        # Initialize fetchers
        self.yahoo_finance = YahooFinanceFetcher(self.logger)
        self.finnhub = FinnhubFetcher(self.logger)
        self.alpaca = AlpacaFetcher(self.logger)
        self.alphavantage = AlphaVantageFetcher(self.logger)
        self.newsapi = NewsAPIFetcher(self.logger)

        # Initialize indicators aggregator
        self.indicator_aggregator = AllIndicatorsUnifier(
            config_manager=None,  # Provide ConfigManager instance if required
            logger=self.logger,
            use_csv=False
        )

    async def fetch_all_data(
        self,
        symbols: List[str],
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d",
        max_concurrent_tasks: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """Fetches stock data for multiple symbols from various sources."""
        self.logger.info("Starting to fetch data for multiple symbols.")

        async with ClientSession() as session:
            tasks = [
                self._fetch_symbol_data(symbol, start_date, end_date, interval, session)
                for symbol in symbols
            ]
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            sem_tasks = [self._run_with_semaphore(task, semaphore) for task in tasks]
            results = await asyncio.gather(*sem_tasks, return_exceptions=True)

        all_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error fetching data for {symbol}: {result}")
                all_data[symbol] = None
            else:
                all_data[symbol] = result
                self.store_data_in_db(symbol, result)

        self.logger.info("Completed fetching data for all symbols.")
        return all_data

    async def _run_with_semaphore(self, task, semaphore):
        """Limits concurrent tasks."""
        async with semaphore:
            return await task

    async def _fetch_symbol_data(
        self,
        symbol: str,
        start_date: str,
        end_date: Optional[str],
        interval: str,
        session: ClientSession
    ) -> Dict[str, Any]:
        """Fetches stock data from multiple sources for a single symbol."""
        self.logger.info(f"Fetching data for {symbol}.")
        data = {}

        # Yahoo Finance
        try:
            yf_data = await self.yahoo_finance.fetch_stock_data_async(symbol, start_date, end_date, interval)
            if yf_data is not None and not yf_data.empty:
                data["Yahoo Finance"] = yf_data
        except Exception as e:
            self.logger.error(f"Yahoo Finance fetch failed for {symbol}: {e}")

        # Finnhub
        try:
            finnhub_quote = await self.finnhub.fetch_quote(symbol, session)
            finnhub_metrics = await self.finnhub.fetch_financial_metrics(symbol, session)
            if finnhub_quote is not None:
                data["Finnhub Quote"] = finnhub_quote
            if finnhub_metrics is not None:
                data["Finnhub Metrics"] = finnhub_metrics
        except Exception as e:
            self.logger.error(f"Finnhub fetch failed for {symbol}: {e}")

        # Alpaca
        try:
            alpaca_data = await self.alpaca.fetch_stock_data_async(symbol, start_date, end_date, interval)
            if alpaca_data is not None and not alpaca_data.empty:
                data["Alpaca"] = alpaca_data
        except Exception as e:
            self.logger.error(f"Alpaca fetch failed for {symbol}: {e}")

        # Alpha Vantage
        try:
            alphavantage_data = await self.alphavantage.fetch_stock_data_async(symbol, start_date, end_date, interval)
            if alphavantage_data is not None and not alphavantage_data.empty:
                data["Alpha Vantage"] = alphavantage_data
        except Exception as e:
            self.logger.error(f"Alpha Vantage fetch failed for {symbol}: {e}")

        # NewsAPI
        try:
            news_data = await self.newsapi.fetch_news_data_async(symbol, page_size=5)
            if news_data is not None and not news_data.empty:
                data["NewsAPI"] = news_data
        except Exception as e:
            self.logger.error(f"NewsAPI fetch failed for {symbol}: {e}")

        self.logger.info(f"Data fetching completed for {symbol}.")
        return data

    def store_data_in_db(self, symbol: str, data: Dict[str, Any]):
        """Stores fetched stock data into PostgreSQL using SQLAlchemy."""
        self.logger.info(f"Storing data for {symbol} in the database...")

        session: Session = SessionLocal()
        try:
            # Apply and store Yahoo Finance data
            if "Yahoo Finance" in data and not data["Yahoo Finance"].empty:
                yf_df = data["Yahoo Finance"].copy()
                yf_df = self.indicator_aggregator.apply_all_indicators(yf_df)
                yf_df.reset_index(inplace=True)  # 'Date' becomes a column

                # Prepare data for bulk insert
                yf_records = yf_df.to_dict(orient='records')
                yf_objects = [
                    AlphaVantageDaily(
                        symbol=row['symbol'],
                        date=row['Date'].date(),
                        open=row.get('Open'),
                        high=row.get('High'),
                        low=row.get('Low'),
                        close=row.get('Close'),
                        volume=row.get('Volume'),
                        sma_50=row.get('SMA_50'),
                        sma_200=row.get('SMA_200')
                    )
                    for row in yf_records
                ]

                # Bulk save
                session.bulk_save_objects(yf_objects, update_existing=True)
                self.logger.info(f"Inserted/Updated Yahoo Finance data for {symbol}.")

            # Apply and store NewsAPI data
            if "NewsAPI" in data and not data["NewsAPI"].empty:
                news_df = data["NewsAPI"].copy()
                news_df.reset_index(inplace=True)
                news_records = news_df.to_dict(orient='records')
                news_objects = [
                    NewsArticle(
                        title=row.get('headline'),
                        description=row.get('description'),
                        url=row.get('url'),
                        published_at=row.get('published_at'),
                        source=row.get('source'),
                        content=row.get('content')
                    )
                    for row in news_records
                ]

                # Bulk save
                session.bulk_save_objects(news_objects, update_existing=True)
                self.logger.info(f"Inserted/Updated NewsAPI data for {symbol}.")

            # Apply and store Finnhub Quote data
            if "Finnhub Quote" in data and not data["Finnhub Quote"].empty:
                fin_quote_df = data["Finnhub Quote"].copy()
                fin_quote_df.reset_index(inplace=True)
                fin_quote_records = fin_quote_df.to_dict(orient='records')
                fin_quote_objects = [
                    FinnhubQuote(
                        symbol=symbol,
                        date=row['date'].date(),
                        current_price=row.get('current_price'),
                        change=row.get('change'),
                        percent_change=row.get('percent_change'),
                        high=row.get('high'),
                        low=row.get('low'),
                        open=row.get('open'),
                        previous_close=row.get('previous_close')
                    )
                    for row in fin_quote_records
                ]

                # Bulk save
                session.bulk_save_objects(fin_quote_objects, update_existing=True)
                self.logger.info(f"Inserted/Updated Finnhub Quote data for {symbol}.")

            # Apply and store Finnhub Metrics data
            if "Finnhub Metrics" in data and not data["Finnhub Metrics"].empty:
                fin_metrics_df = data["Finnhub Metrics"].copy()
                fin_metrics_df.reset_index(inplace=True)
                fin_metrics_records = fin_metrics_df.to_dict(orient='records')
                fin_metrics_objects = [
                    FinnhubMetrics(
                        symbol=symbol,
                        date_fetched=row['date_fetched'],
                        week_52_high=row.get('52WeekHigh'),
                        week_52_low=row.get('52WeekLow'),
                        market_cap=row.get('MarketCapitalization'),
                        pe_ratio=row.get('P/E')
                    )
                    for row in fin_metrics_records
                ]

                # Bulk save
                session.bulk_save_objects(fin_metrics_objects, update_existing=True)
                self.logger.info(f"Inserted/Updated Finnhub Metrics data for {symbol}.")

            # Apply and store Alpaca data
            if "Alpaca" in data and not data["Alpaca"].empty:
                alpaca_df = data["Alpaca"].copy()
                alpaca_df.reset_index(inplace=True)
                alpaca_records = alpaca_df.to_dict(orient='records')
                alpaca_objects = [
                    AlpacaData(
                        symbol=row['symbol'],
                        date=row['Date'].date(),
                        open=row.get('Open'),
                        high=row.get('High'),
                        low=row.get('Low'),
                        close=row.get('Close'),
                        volume=row.get('Volume')
                    )
                    for row in alpaca_records
                ]

                # Bulk save
                session.bulk_save_objects(alpaca_objects, update_existing=True)
                self.logger.info(f"Inserted/Updated Alpaca data for {symbol}.")

            # Apply and store Alpha Vantage data
            if "Alpha Vantage" in data and not data["Alpha Vantage"].empty:
                alpha_df = data["Alpha Vantage"].copy()
                alpha_df.reset_index(inplace=True)
                alpha_records = alpha_df.to_dict(orient='records')
                alpha_objects = [
                    AlphaVantageData(
                        symbol=row['symbol'],
                        date=row['Date'].date(),
                        open=row.get('Open'),
                        high=row.get('High'),
                        low=row.get('Low'),
                        close=row.get('Close'),
                        volume=row.get('Volume')
                    )
                    for row in alpha_records
                ]

                # Bulk save
                session.bulk_save_objects(alpha_objects, update_existing=True)
                self.logger.info(f"Inserted/Updated Alpha Vantage data for {symbol}.")

            # Commit all changes
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
        """Initializes database tables for a given symbol if they don't exist."""
        # Not needed anymore as SQLAlchemy models handle table creation
        pass


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
async def main():
    """Fetch stock data, apply indicators, and store in PostgreSQL."""
    data_fetcher = MainDataFetcher()

    # Define symbols and parameters
    symbols = ["AAPL", "TSLA", "AMZN"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    interval = "1d"

    try:
        # Fetch and store data
        await data_fetcher.fetch_all_data(symbols, start_date=start_date, end_date=end_date, interval=interval)
        data_fetcher.logger.info("\nData Fetching & Storage Completed Successfully!\n")
        print("\nData Fetching & Storage Completed Successfully!\n")
    except Exception as e:
        data_fetcher.logger.error(f"Fatal error during data fetching: {e}")
        print(f"Fatal error during data fetching: {e}")


if __name__ == "__main__":
    asyncio.run(main())
