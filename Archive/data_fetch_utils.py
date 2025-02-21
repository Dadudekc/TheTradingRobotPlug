# -------------------------------------------------------------------
# File: data_fetch_utils.py
# Location: src/Utilities
# Description: Fetches stock data & stores it in PostgreSQL.
# -------------------------------------------------------------------

import asyncio
import logging
import pandas as pd
import psycopg2
from typing import List, Dict, Any, Optional
from aiohttp import ClientSession
from Utilities.data_fetchers.yahoo_finance_fetcher import YahooFinanceFetcher
from Utilities.data_fetchers.finnhub_fetcher import FinnhubFetcher
from Utilities.data_fetchers.alpaca_fetcher import AlpacaFetcher
from Utilities.data_fetchers.newsapi_fetcher import NewsAPIFetcher
from Utilities.data_fetchers.async_fetcher import AsyncFetcher
from Utilities.shared_utils import setup_logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database Connection
DB_NAME = os.getenv("POSTGRES_DBNAME", "trading_robot_plug")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "your_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5434")


class MainDataFetcher:
    def __init__(self):
        """ Initializes fetchers and logging. """
        self.logger = setup_logging(script_name="MainDataFetcher", log_level=logging.INFO)
        self.logger.info("Initializing MainDataFetcher...")

        # Initialize fetchers
        self.yahoo_finance = YahooFinanceFetcher(self.logger)
        self.finnhub = FinnhubFetcher(self.logger)
        self.alpaca = AlpacaFetcher(self.logger)
        self.newsapi = NewsAPIFetcher(self.logger)
        self.async_fetcher = AsyncFetcher(self.logger)

    async def fetch_all_data(
        self,
        symbols: List[str],
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d",
        max_concurrent_tasks: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """ Fetches stock data for multiple symbols and stores it in the database. """
        self.logger.info("Starting to fetch data for multiple symbols.")
        
        async with await self.async_fetcher.create_session() as session:
            tasks = [self._fetch_symbol_data(symbol, start_date, end_date, interval, session) for symbol in symbols]
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
        """ Limits concurrent tasks. """
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
        """ Fetches stock data from multiple sources for a single symbol. """
        self.logger.info(f"Fetching data for {symbol}.")

        data = {}

        try:
            yf_data = await self.yahoo_finance.fetch_stock_data_async(symbol, start_date, end_date, interval)
            data["Yahoo Finance"] = yf_data
        except Exception as e:
            self.logger.error(f"Yahoo Finance fetch failed for {symbol}: {e}")

        try:
            finnhub_quote = await self.finnhub.fetch_quote(symbol, session)
            finnhub_metrics = await self.finnhub.fetch_financial_metrics(symbol, session)
            data["Finnhub Quote"] = finnhub_quote
            data["Finnhub Metrics"] = finnhub_metrics
        except Exception as e:
            self.logger.error(f"Finnhub fetch failed for {symbol}: {e}")

        try:
            alpaca_data = await self.alpaca.fetch_stock_data_async(symbol, start_date, end_date, interval)
            data["Alpaca"] = alpaca_data
        except Exception as e:
            self.logger.error(f"Alpaca fetch failed for {symbol}: {e}")

        try:
            news_data = await self.newsapi.fetch_news_data_async(symbol, page_size=5)
            data["NewsAPI"] = news_data
        except Exception as e:
            self.logger.error(f"NewsAPI fetch failed for {symbol}: {e}")

        self.logger.info(f"Data fetching completed for {symbol}.")
        return data

    def store_data_in_db(self, symbol: str, data: Dict[str, Any]):
        """ Stores fetched stock data into PostgreSQL. """
        self.logger.info(f"Storing data for {symbol} in the database...")

        # Connect to the database
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
            )
            cursor = conn.cursor()

            # Ensure table exists
            self.ensure_table_exists(cursor, symbol)

            # Insert Yahoo Finance Data
            if "Yahoo Finance" in data and data["Yahoo Finance"] is not None:
                yf_df = pd.DataFrame(data["Yahoo Finance"])
                for _, row in yf_df.iterrows():
                    query = f"""
                    INSERT INTO {symbol} (symbol, date, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, date) DO UPDATE 
                    SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, 
                        close = EXCLUDED.close, volume = EXCLUDED.volume;
                    """
                    cursor.execute(query, (symbol, row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']))

            conn.commit()
            self.logger.info(f"✅ Successfully stored data for {symbol} in the database.")

        except Exception as e:
            self.logger.error(f"⚠️ Database error storing data for {symbol}: {e}")

        finally:
            cursor.close()
            conn.close()

    def ensure_table_exists(self, cursor, symbol: str):
        """ Ensures the stock table exists in the database before inserting data. """
        query = f"""
        CREATE TABLE IF NOT EXISTS {symbol} (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT,
            UNIQUE(symbol, date)
        );
        """
        cursor.execute(query)
        self.logger.info(f"✅ Ensured table {symbol} exists.")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
async def main():
    """ Fetch stock data & store it in PostgreSQL. """
    fetcher = MainDataFetcher()
    
    symbols = ["AAPL", "TSLA", "AMZN"]
    start_date = "2023-01-01"
    interval = "1d"

    try:
        await fetcher.fetch_all_data(symbols, start_date=start_date, interval=interval)
        print("\n✅ Data Fetching & Storage Completed Successfully!\n")

    except Exception as e:
        fetcher.logger.error(f"Fatal error during data fetching: {e}")

if __name__ == "__main__":
    asyncio.run(main())
