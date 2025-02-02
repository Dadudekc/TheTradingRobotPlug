# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/src/Utilities/db/sql_data_handler.py
# Description:
#   A unified, class-based module that handles database interactions,
#   synchronous/asynchronous data saving, real-time updates from various APIs,
#   plus table/column management. Called SQLDataHandler for clarity and
#   long-term modular expansion.
# -------------------------------------------------------------------

import sys
import os
import io
import logging
import asyncio
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker, AsyncEngine
)

import alpaca_trade_api as tradeapi
import yfinance as yf
from bs4 import BeautifulSoup

# Local utilities
from Utilities.config_manager import ConfigManager
from Utilities.shared_utils import setup_logging
# If you want aggregator logic, uncomment or adapt
# from Utilities.data_processing.indicator_unifier import AllIndicatorsUnifier

# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]  # Adjust as needed
dotenv_path = project_root / '.env'

Base = declarative_base()

logger = setup_logging(script_name="sql_data_handler")
logger.info("Logger initialized for SQL Data Handler.")

# -------------------------------------------------------------------
# Database Models
# -------------------------------------------------------------------
class StockData(Base):
    """
    Example ORM model representing your 'stock_data' table.
    """
    __tablename__ = 'stock_data'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, name="Date", nullable=True)
    open = Column(Float, name="Open", nullable=True)
    high = Column(Float, name="High", nullable=True)
    low = Column(Float, name="Low", nullable=True)
    close = Column(Float, name="Close", nullable=True)
    volume = Column(Integer, name="Volume", nullable=True)
    symbol = Column(String, name="symbol", nullable=True)
    # Add other columns if necessary


class AIReport(Base):
    """
    Example ORM model representing your 'ai_reports' table.
    """
    __tablename__ = 'ai_reports'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    report = Column(String, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)


class Symbol(Base):
    """
    Example ORM model representing your 'symbols' table.
    """
    __tablename__ = 'symbols'
    symbol = Column(String, primary_key=True, nullable=False)
    name = Column(String, nullable=True)
    exchange = Column(String, nullable=True)
    asset_type = Column(String, nullable=True)
    date_added = Column(DateTime, default=datetime.utcnow)


# -------------------------------------------------------------------
# The SQLDataHandler class
# -------------------------------------------------------------------
class SQLDataHandler:
    """
    A single class that handles:
     - Database setup for sync/async engines
     - Table/column creation and checking
     - Data fetching from yfinance, Alpaca, or any external API
     - Sync saving and partial async upsert
     - Additional utility methods for modifications and expansions
    """

    def __init__(self):
        load_dotenv(dotenv_path)  # Load environment variables
        self.logger = logger
        self.config_manager = self._initialize_config()
        self.engine, self.SessionLocal = self._create_sync_engine()
        self.async_engine, self.AsyncSessionLocal = self._create_async_engine()
        # If you want aggregator logic, you can set self.indicator_unifier = AllIndicatorsUnifier(...) here
        self.alpaca_api = self._initialize_alpaca()

        # Create DB tables
        Base.metadata.create_all(self.engine)
        self.logger.info("All ORM models created or verified in the sync database.")

    def _initialize_config(self) -> ConfigManager:
        required_keys = [
            'POSTGRES_HOST', 'POSTGRES_DBNAME', 'POSTGRES_USER',
            'POSTGRES_PASSWORD', 'POSTGRES_PORT',
            'ALPHAVANTAGE_API_KEY', 'ALPHAVANTAGE_BASE_URL',
            'ALPACA_API_KEY', 'ALPACA_SECRET_KEY',
            'ALPACA_BASE_URL', 'DATABASE_URL'
        ]
        try:
            return ConfigManager(env_file=dotenv_path, required_keys=required_keys, logger=self.logger)
        except KeyError as e:
            self.logger.error(f"Missing required config keys: {e}")
            sys.exit(1)

    def _create_sync_engine(self):
        """
        Creates a synchronous SQLAlchemy engine and sessionmaker.
        """
        database_url = self.config_manager.get('DATABASE_URL')
        if not database_url:
            # Build from components if not found
            database_url = self.config_manager.get_db_url()
        self.logger.info(f"Using sync database URL: {database_url}")

        try:
            engine = create_engine(database_url, echo=False)
            session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            return engine, session_local
        except SQLAlchemyError as e:
            self.logger.error(f"Error setting up sync database: {e}", exc_info=True)
            sys.exit(1)

    def _create_async_engine(self):
        """
        Creates an asynchronous SQLAlchemy engine and sessionmaker.
        """
        database_url = self.config_manager.get('DATABASE_URL')
        if not database_url:
            database_url = self.config_manager.get_db_url()

        # Convert 'postgresql://', 'postgresql+psycopg2' to 'postgresql+asyncpg'
        if 'postgresql+psycopg2' in database_url:
            async_database_url = database_url.replace('postgresql+psycopg2', 'postgresql+asyncpg')
        elif database_url.startswith('postgresql://'):
            async_database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
        else:
            async_database_url = database_url

        self.logger.info(f"Using async database URL: {async_database_url}")
        try:
            async_engine = create_async_engine(async_database_url, echo=False)
            async_session_local = async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)
            return async_engine, async_session_local
        except SQLAlchemyError as e:
            self.logger.error(f"Error setting up async database: {e}", exc_info=True)
            sys.exit(1)

    def _initialize_alpaca(self):
        """
        Initializes the Alpaca API client from environment variables.
        """
        try:
            api_key = self.config_manager.get('ALPACA_API_KEY')
            secret_key = self.config_manager.get('ALPACA_SECRET_KEY')
            base_url = self.config_manager.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

            if not all([api_key, secret_key, base_url]):
                self.logger.error("Missing Alpaca credentials in environment variables.")
                raise ValueError("Missing Alpaca credentials.")
            self.logger.info("Alpaca API credentials loaded.")
            return tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        except Exception as e:
            self.logger.error(f"Error initializing Alpaca API: {e}", exc_info=True)
            return None

    # --------------
    # Sync Database
    # --------------
    def get_session(self) -> Session:
        return self.SessionLocal()

    def create_or_update_tables(self):
        """
        If you want to do custom DDL beyond just Base.metadata.create_all(),
        place it here. Right now, it's effectively a placeholder.
        """
        inspector = inspect(self.engine)
        self.logger.info("Tables checked or updated as needed.")

    def alter_table_add_columns(self):
        """
        Example: Add missing columns to your tables if they do not exist.
        """
        with self.engine.connect() as conn:
            inspector = inspect(conn)
            if inspector.has_table('stock_data'):
                existing_cols = [col['name'] for col in inspector.get_columns('stock_data')]
                required_cols = {
                    'symbol': 'VARCHAR(50)',
                    'timestamp': 'TIMESTAMP',
                    'open': 'FLOAT',
                    'high': 'FLOAT',
                    'low': 'FLOAT',
                    'close': 'FLOAT',
                    'volume': 'INT'
                }
                for col_name, col_type in required_cols.items():
                    if col_name not in existing_cols:
                        try:
                            conn.execute(text(f"ALTER TABLE stock_data ADD COLUMN {col_name} {col_type}"))
                            self.logger.info(f"Added missing column '{col_name}' to 'stock_data'.")
                        except Exception as e:
                            self.logger.error(f"Error adding column '{col_name}': {e}", exc_info=True)
            else:
                self.logger.warning("Table 'stock_data' does not exist.")

    def save_stock_data(self, symbol: str, data: pd.DataFrame):
        """
        Example method for saving DataFrame rows into 'stock_data'.
        """
        session = self.get_session()
        try:
            # Ensure table
            if not inspect(self.engine).has_table('stock_data'):
                Base.metadata.create_all(self.engine)

            # Convert date column if needed
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

            # Insert rows
            records = []
            for _, row in data.iterrows():
                sd = StockData(
                    date=row.get('Date'),
                    open=row.get('Open'),
                    high=row.get('High'),
                    low=row.get('Low'),
                    close=row.get('Close'),
                    volume=row.get('Volume'),
                    symbol=symbol
                )
                records.append(sd)

            session.bulk_save_objects(records)
            session.commit()
            self.logger.info(f"Saved {len(records)} rows for {symbol} into stock_data.")
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"SQLAlchemy error while saving stock data for {symbol}: {e}", exc_info=True)
        finally:
            session.close()

    # --------------
    # Async Database
    # --------------
    def get_async_session(self) -> AsyncSession:
        return self.AsyncSessionLocal()

    async def upsert_data_async(self, data: pd.DataFrame, table_name: str):
        """
        Asynchronously upsert data into a given table using a naive approach.
        Adjust for your real table schema / PK constraints.
        """
        async with self.get_async_session() as session:
            try:
                records = data.to_dict(orient='records')
                if not records:
                    self.logger.warning("No records to upsert.")
                    return

                # Example naive approach: Insert each row, ignoring duplicates
                for record in records:
                    # Example usage of raw SQL. 
                    # You'll want to adapt this to match your table schema:
                    # E.g., 
                    #   "INSERT INTO tablename (col1, col2) VALUES (:col1, :col2)
                    #    ON CONFLICT(col1) DO UPDATE SET col2=EXCLUDED.col2"

                    placeholders = ', '.join([f":{k}" for k in record.keys()])
                    columns = ', '.join(record.keys())
                    sql = f"""
                    INSERT INTO {table_name} ({columns})
                    VALUES ({placeholders})
                    """
                    await session.execute(text(sql), record)
                await session.commit()
                self.logger.info(f"Upserted {len(records)} records into '{table_name}'.")
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Error during async upsert to {table_name}: {e}", exc_info=True)

    # --------------
    # Data Fetching Methods
    # --------------
    def fetch_from_yfinance(self, symbol: str, period: str='1y', interval: str='1d') -> pd.DataFrame:
        """
        Simple wrapper for yfinance data retrieval.
        """
        try:
            yf_ticker = yf.Ticker(symbol)
            df = yf_ticker.history(period=period, interval=interval)
            df.reset_index(inplace=True)
            df['symbol'] = symbol
            self.logger.info(f"Fetched {len(df)} rows for {symbol} from yfinance.")
            return df
        except Exception as e:
            self.logger.error(f"fetch_from_yfinance error for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def fetch_from_alpaca(self, symbol: str, timeframe: str='1Day', start_date: Optional[str]=None, end_date: Optional[str]=None) -> pd.DataFrame:
        """
        Example fetch from Alpaca. 
        """
        try:
            if not self.alpaca_api:
                raise ValueError("Alpaca API client not initialized.")
            # Convert timeframe
            # ...
            # For brevity, just return an empty df
            self.logger.info(f"Fetched data from Alpaca for {symbol} timeframe {timeframe}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching Alpaca data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    # --------------
    # Additional Utility
    # --------------
    def get_top_150_symbols(self) -> List[str]:
        """
        Reuse your code logic for top symbols. 
        """
        mandatory_symbols = {"AAPL", "TSLA", "QQQ", "SPY", "AMZN"}
        top_150_symbols = list(mandatory_symbols)  # Start with mandatory
        # ...
        self.logger.info("Fetched top 150 symbols (placeholder).")
        return top_150_symbols

    def unify_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        If you want aggregator logic from AllIndicatorsUnifier, adapt here.
        """
        # e.g.:
        # df = self.indicator_unifier.apply_all_indicators(df)
        return df

# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    # Initialize the SQLDataHandler
    handler = SQLDataHandler()

    # Example usage: fetch from yfinance and save to DB
    symbol = "AAPL"
    df = handler.fetch_from_yfinance(symbol, period='1mo', interval='1d')
    if not df.empty:
        handler.save_stock_data(symbol, df)
    else:
        handler.logger.info(f"No data to save for {symbol} from yfinance.")

    # Example usage: create or update your table columns
    handler.alter_table_add_columns()

    # Example usage: async upsert
    # We won't run an actual event loop here, but demonstration:
    # import asyncio
    # data_for_upsert = pd.DataFrame([...])
    # asyncio.run(handler.upsert_data_async(data_for_upsert, 'some_table'))

    handler.logger.info("Done with main logic in sql_data_handler.")

if __name__ == "__main__":
    main()
