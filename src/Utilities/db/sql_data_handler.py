# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/src/Utilities/db/sql_data_handler.py
# Description: Merged and improved module that handles database interactions,
#              data fetching from various APIs, and supports both synchronous
#              and asynchronous operations. It integrates utility functions
#              from database_utils.py and ensures consistent configuration
#              and logging.
# -------------------------------------------------------------------

import sys
import os
import io
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, func, inspect, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, AsyncEngine
from sqlalchemy.ext.asyncio import create_async_engine
from dotenv import load_dotenv

import alpaca_trade_api as tradeapi
import yfinance as yf
from aiohttp import ClientSession, ClientTimeout
import requests
from bs4 import BeautifulSoup

# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------

# Dynamically set the project root based on the current file's location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]  # Adjusted for the project structure
utilities_dir = project_root / 'src' / 'Utilities'
model_dir = project_root / 'SavedModels'
model_utils = project_root / 'src' / 'model_training' / 'utils'
data_processing_dir = project_root / 'src' / 'Data_Processing'
utils_dir = utilities_dir / 'utils'  # Path for indicator_utils.py

# Add the necessary directories to the Python path
sys.path.extend([
    str(utilities_dir),
    str(model_dir),
    str(model_utils),
    str(data_processing_dir),
    str(utils_dir)
])

# -------------------------------------------------------------------
# Import ConfigManager and Logging Setup Using Absolute Imports
# -------------------------------------------------------------------
from config_handling.config_manager import ConfigManager
from config_handling.logging_setup import setup_logging
from Utilities.utils.indicator_utils import apply_all_indicators

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------

# Initialize logger
logger = setup_logging(script_name="sql_data_handler")
logger.info("Logger initialized for SQL Data Handler.")

# -------------------------------------------------------------------
# Configuration Setup
# -------------------------------------------------------------------
dotenv_path = project_root / '.env'

# Load environment variables
load_dotenv(dotenv_path)
try:
    config_manager = ConfigManager(env_file=dotenv_path, required_keys=[
        'POSTGRES_HOST',
        'POSTGRES_DBNAME',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'POSTGRES_PORT',
        'ALPHAVANTAGE_API_KEY',
        'ALPHAVANTAGE_BASE_URL',
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY',
        'ALPACA_BASE_URL',
        'DATABASE_URL'  # Added DATABASE_URL for consistency
    ], logger=logger)
except KeyError as e:
    logger.error(f"Missing required configuration keys: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Database Configuration and Engine Setup
# -------------------------------------------------------------------

DATABASE_URL = config_manager.get('DATABASE_URL')
if not DATABASE_URL:
    # Build DATABASE_URL from individual components if not provided
    DATABASE_URL = config_manager.get_db_url()
logger.info(f"Using DATABASE_URL: {DATABASE_URL}")

# Replace 'postgresql+psycopg2://' with 'postgresql+asyncpg://' for async support
if 'postgresql+psycopg2' in DATABASE_URL:
    ASYNC_DATABASE_URL = DATABASE_URL.replace('postgresql+psycopg2', 'postgresql+asyncpg')
    logger.info(f"Replaced 'postgresql+psycopg2' with 'postgresql+asyncpg'.")
elif DATABASE_URL.startswith('postgresql://'):
    ASYNC_DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://')
    logger.info("Replaced 'postgresql://' with 'postgresql+asyncpg://'.")
else:
    ASYNC_DATABASE_URL = DATABASE_URL
    logger.warning("DATABASE_URL does not contain 'postgresql+psycopg2' or start with 'postgresql://', using as is for async.")

# Log the final ASYNC_DATABASE_URL to ensure it's correct
logger.info(f"Final ASYNC_DATABASE_URL: {ASYNC_DATABASE_URL}")

# Create synchronous engine
try:
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    Base.metadata.create_all(bind=engine)
    logger.info("Synchronous Database engine created and models initialized.")
except SQLAlchemyError as e:
    logger.error(f"Error setting up the synchronous database: {e}", exc_info=True)
    sys.exit(1)

# Create asynchronous engine
try:
    async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
    AsyncSessionLocal = sessionmaker(
        async_engine, expire_on_commit=False, class_=AsyncSession
    )
    logger.info("Asynchronous Database engine created.")
except SQLAlchemyError as e:
    logger.error(f"Error setting up the asynchronous database: {e}", exc_info=True)
    sys.exit(1)

# -------------------------------------------------------------------
# Database Models
# -------------------------------------------------------------------

class StockData(Base):
    __tablename__ = 'stock_data'

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, name="Date", nullable=True)
    open = Column(Float, name="Open", nullable=True)
    high = Column(Float, name="High", nullable=True)
    low = Column(Float, name="Low", nullable=True)
    close = Column(Float, name="Close", nullable=True)
    volume = Column(Integer, name="Volume", nullable=True)
    symbol = Column(String, name="symbol", nullable=True)
    custom_ma_memory = Column(Float, name="Custom_MA_Memory", nullable=True)
    custom_bollinger_upper = Column(Float, name="Custom_Bollinger_Upper", nullable=True)
    custom_bollinger_lower = Column(Float, name="Custom_Bollinger_Lower", nullable=True)
    timestamp = Column(DateTime, name="timestamp", nullable=True)
    # Add other columns if necessary

class AIReport(Base):
    __tablename__ = 'ai_reports'

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    report = Column(String, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)

class Symbol(Base):
    __tablename__ = 'symbols'
    symbol = Column(String, primary_key=True, nullable=False)
    name = Column(String, nullable=True)
    exchange = Column(String, nullable=True)
    asset_type = Column(String, nullable=True)
    date_added = Column(DateTime, default=datetime.utcnow)

# -------------------------------------------------------------------
# DataHandler Class
# -------------------------------------------------------------------

class DataHandler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.engine = engine
        self.async_engine = async_engine
        self.SessionLocal = SessionLocal
        self.AsyncSessionLocal = AsyncSessionLocal
        # Initialize Alpaca API client
        self.alpaca_api = self.initialize_alpaca()

    def get_news_data(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetches recent news articles for the given stock symbol using NewsAPI or another source.

        Args:
            symbol (str): The stock symbol for which to fetch news articles.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing news articles.
        """
        # NewsAPI configuration
        api_key = config_manager.get('NEWS_API_KEY')
        if not api_key:
            self.logger.error("Missing NewsAPI key in environment variables.")
            return []

        # URL for NewsAPI endpoint (or replace with your preferred news source)
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            news_data = response.json()
            
            if news_data.get("status") == "ok":
                articles = news_data.get("articles", [])
                self.logger.info(f"Fetched {len(articles)} news articles for {symbol}.")
                return articles
            else:
                self.logger.error(f"Error in NewsAPI response: {news_data.get('message')}")
                return []

        except Exception as e:
            self.logger.error(f"Error fetching news data for {symbol}: {e}", exc_info=True)
            return []

    def get_session(self) -> Session:
        return self.SessionLocal()

    def get_async_session(self) -> AsyncSession:
        return self.AsyncSessionLocal()

    def initialize_alpaca(self):
        """
        Initializes Alpaca API client using credentials from environment variables.
        """
        api_key = config_manager.get('ALPACA_API_KEY')
        secret_key = config_manager.get('ALPACA_SECRET_KEY')
        base_url = config_manager.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

        if not all([api_key, secret_key, base_url]):
            self.logger.error("Missing Alpaca API credentials in environment variables.")
            raise ValueError("Missing Alpaca API credentials in environment variables.")

        self.logger.info("Initializing Alpaca API with provided credentials.")
        try:
            return tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        except Exception as e:
            self.logger.error(f"Error initializing Alpaca API: {str(e)}", exc_info=True)
            raise

    # -------------------------------------------------------------------
    # Synchronous Database Operations
    # -------------------------------------------------------------------

    def save_data(self, symbol: str, data: pd.DataFrame):
        """
        Saves stock data into the SQL database.

        Args:
            symbol (str): Stock symbol.
            data (pd.DataFrame): DataFrame containing stock data.
        """
        with self.get_session() as session:
            try:
                # Check if the table exists
                if not inspect(self.engine).has_table(StockData.__tablename__):
                    Base.metadata.create_all(bind=self.engine)
                    self.logger.info(f"Created table '{StockData.__tablename__}'.")

                # Iterate over DataFrame rows and create StockData instances
                stock_data_list = [
                    StockData(
                        symbol=row['symbol'],
                        timestamp=row['timestamp'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        # Add other fields as necessary
                    )
                    for _, row in data.iterrows()
                ]

                # Bulk save objects
                session.bulk_save_objects(stock_data_list)
                session.commit()
                self.logger.info(f"Successfully saved {len(stock_data_list)} records for {symbol} to the database.")
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"SQLAlchemy error while saving data for {symbol}: {e}", exc_info=True)
            except Exception as e:
                session.rollback()
                self.logger.error(f"Unexpected error while saving data for {symbol}: {e}", exc_info=True)

    def save_ai_report(self, symbol: str, report: str):
        """
        Saves the AI-generated report into the SQL database.

        Args:
            symbol (str): Stock symbol.
            report (str): AI-generated report text.
        """
        with self.get_session() as session:
            try:
                ai_report_entry = AIReport(
                    symbol=symbol,
                    report=report
                )
                session.add(ai_report_entry)
                session.commit()
                self.logger.info(f"AI report saved for {symbol}.")
            except SQLAlchemyError as e:
                session.rollback()
                self.logger.error(f"SQLAlchemy error while saving AI report for {symbol}: {e}", exc_info=True)
            except Exception as e:
                session.rollback()
                self.logger.error(f"Unexpected error while saving AI report for {symbol}: {e}", exc_info=True)

    def fetch_and_save_to_sql(self, session: Session, symbol: str, data: pd.DataFrame, source: str, date_col: str):
        """
        Saves the provided stock data into the SQL database.

        Args:
            session (Session): SQLAlchemy session.
            symbol (str): Stock symbol for the data.
            data (pd.DataFrame): DataFrame containing stock data.
            source (str): Data source.
            date_col (str): Name of the date column in the DataFrame.
        """
        try:
            if data.empty:
                self.logger.warning(f"No data to save for {symbol} from {source}.")
                return

            # Ensure the date column exists
            if date_col not in data.columns:
                self.logger.error(f"'{date_col}' column missing in data for {symbol} from {source}.")
                return

            # Add 'symbol' column if not present
            if 'symbol' not in data.columns:
                data['symbol'] = symbol

            # Rename columns to match the ORM model
            column_map = {
                date_col: 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }

            # Check and log the columns before renaming
            self.logger.debug(f"Data columns before renaming: {data.columns.tolist()}")
            data = data.rename(columns=column_map)
            self.logger.debug(f"Data columns after renaming: {data.columns.tolist()}")

            # Verify required columns
            required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns {missing_columns} in data for {symbol} from {source}.")
                return

            # Convert 'timestamp' to datetime if not already
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

            # Drop rows with invalid timestamps
            before_drop = len(data)
            data = data.dropna(subset=['timestamp'])
            after_drop = len(data)
            if before_drop != after_drop:
                self.logger.warning(f"Dropped {before_drop - after_drop} rows with invalid timestamps for {symbol} from {source}.")

            # Remove duplicates
            data = data.drop_duplicates(subset=['symbol', 'timestamp'])

            if data.empty:
                self.logger.info(f"No new data to save for {symbol} from {source}.")
                return

            # Check if data already exists in the database to avoid duplicates
            existing_records = session.query(StockData.symbol, StockData.timestamp).filter(
                StockData.symbol == symbol,
                StockData.timestamp.in_(data['timestamp'].tolist())
            ).all()

            existing_timestamps = {(record.symbol, record.timestamp) for record in existing_records}
            new_data = data[~data.apply(lambda row: (row['symbol'], row['timestamp']) in existing_timestamps, axis=1)]
            if new_data.empty:
                self.logger.info(f"All data for {symbol} from {source} already exists in the database.")
                return

            # Create list of StockData objects
            stock_data_list = [
                StockData(
                    symbol=row['symbol'],
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                for _, row in new_data.iterrows()
            ]

            # Bulk save to the database
            session.bulk_save_objects(stock_data_list)
            session.commit()
            self.logger.info(f"Successfully saved {len(stock_data_list)} records for {symbol} from {source} to the database.")

        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"SQLAlchemy error while saving data for {symbol} from {source}: {e}", exc_info=True)
        except Exception as e:
            session.rollback()
            self.logger.error(f"Unexpected error while saving data for {symbol} from {source}: {e}", exc_info=True)

    # -------------------------------------------------------------------
    # Asynchronous Database Operations
    # -------------------------------------------------------------------

    async def upsert_data_async(data: pd.DataFrame, table_name: str, async_engine: AsyncEngine):
        """
        Asynchronously upserts data into a specified database table.

        Args:
            data (pd.DataFrame): Data to be inserted or updated.
            table_name (str): Target table name in the database.
            async_engine: SQLAlchemy async engine for database connection.
        """
        async_session = async_sessionmaker(bind=async_engine, expire_on_commit=False, class_=AsyncSession)
        async with async_session() as session:
            try:
                records = data.to_dict(orient='records')
                if not records:
                    logger.warning("No records to insert.")
                    return

                for record in records:
                    # Adjust the query to reflect your table schema
                    query = f"""
                    INSERT INTO {table_name} (source, author, title, description, url, urlToImage, publishedAt, content)
                    VALUES (:source, :author, :title, :description, :url, :urlToImage, :publishedAt, :content)
                    ON CONFLICT (url) DO UPDATE 
                    SET source = EXCLUDED.source,
                        author = EXCLUDED.author,
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        urlToImage = EXCLUDED.urlToImage,
                        publishedAt = EXCLUDED.publishedAt,
                        content = EXCLUDED.content;
                    """
                    await session.execute(text(query), record)
                await session.commit()
                logger.info(f"Data upserted into table '{table_name}'")
            except Exception as e:
                logger.error(f"Error during async upsert to {table_name}: {e}")
                await session.rollback()

    # -------------------------------------------------------------------
    # Data Fetching Methods
    # -------------------------------------------------------------------

    def fetch_from_alpaca(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetches stock data from Alpaca API.

        Args:
            symbol (str): Stock symbol.
            timeframe (str): Timeframe for data (e.g., '1D', '1H').
            start (str): Start date in 'YYYY-MM-DD' format.
            end (str): End date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame containing stock data.
        """
        from alpaca_trade_api.rest import TimeFrame, TimeFrameUnit
        try:
            self.logger.info(f"Fetching data for {symbol} from {start} to {end} with timeframe {timeframe}")

            # Normalize timeframe string to handle case-insensitivity
            timeframe_normalized = timeframe.upper()

            # Map timeframe string to Alpaca TimeFrame object
            timeframe_map = {
                "1MIN": TimeFrame(1, TimeFrameUnit.Minute),
                "5MIN": TimeFrame(5, TimeFrameUnit.Minute),
                "15MIN": TimeFrame(15, TimeFrameUnit.Minute),
                "1H": TimeFrame(1, TimeFrameUnit.Hour),
                "1D": TimeFrame(1, TimeFrameUnit.Day),
            }

            alpaca_timeframe = timeframe_map.get(timeframe_normalized)
            if not alpaca_timeframe:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                raise ValueError(f"Invalid timeframe: {timeframe}")

            # Adjust end date to avoid querying recent SIP data
            today = datetime.today()
            end_date_dt = pd.to_datetime(end)

            # Set the end date to be at least 5 days before today to avoid SIP data restrictions
            if (today - end_date_dt).days < 5:
                end = (today - timedelta(days=5)).strftime('%Y-%m-%d')
                self.logger.info(f"Adjusted end date to {end} to avoid recent SIP data restrictions.")

            bars = self.alpaca_api.get_bars(
                symbol,
                alpaca_timeframe,
                start=start,
                end=end,
                adjustment='raw',
                limit=10000  # Set appropriate limit
            ).df

            if bars.empty:
                self.logger.warning(f"No data returned for {symbol} between {start} and {end}.")
                return pd.DataFrame()

            bars.reset_index(inplace=True)  # Ensure 'timestamp' is a column
            return bars

        except Exception as e:
            self.logger.error(f"Error fetching data from Alpaca for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    async def fetch_from_yfinance_async(self, symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """
        Asynchronously fetches stock data using yfinance.

        Args:
            symbol (str): Stock symbol.
            period (str): Period to fetch data for (e.g., '1y').
            interval (str): Data interval (e.g., '1d').

        Returns:
            pd.DataFrame: DataFrame containing stock data.
        """
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self.fetch_from_yfinance, symbol, period, interval)
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data from yfinance for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def fetch_from_yfinance(self, symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """
        Fetches stock data using yfinance.

        Args:
            symbol (str): Stock symbol.
            period (str): Period to fetch data for (e.g., '1y').
            interval (str): Data interval (e.g., '1d').

        Returns:
            pd.DataFrame: DataFrame containing stock data.
        """
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            if df.empty:
                self.logger.warning(f"No data returned for {symbol} using yfinance.")
                return pd.DataFrame()

            df.reset_index(inplace=True)
            df['symbol'] = symbol

            # Ensure that the date column is named 'Date'
            if 'Date' not in df.columns:
                # Attempt to find the correct date column
                possible_date_cols = ['date', 'Datetime', 'datetime']
                for col in possible_date_cols:
                    if col in df.columns:
                        df.rename(columns={col: 'Date'}, inplace=True)
                        self.logger.info(f"Renamed '{col}' column to 'Date' for {symbol}.")
                        break
                else:
                    self.logger.error(f"No date column found in data for {symbol}. Available columns: {df.columns.tolist()}")
                    return pd.DataFrame()

            return df

        except Exception as e:
            self.logger.error(f"Error fetching data from yfinance for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    # -------------------------------------------------------------------
    # Data Update Methods
    # -------------------------------------------------------------------

    async def update_stock_data_async(self, symbol: str, source: str = 'yfinance',
                                      start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                                      interval: str = '1d'):
        """
        Asynchronously updates stock data for a given symbol.

        Args:
            symbol (str): Stock symbol.
            source (str): Data source ('yfinance' or 'alpaca').
            start_date (Optional[datetime]): Start date for fetching data.
            end_date (Optional[datetime]): End date for fetching data.
            interval (str): Data interval (e.g., '1d').
        """
        try:
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=365 * 5)  # Default to last 5 years

            if source == 'yfinance':
                df = await self.fetch_from_yfinance_async(symbol, period='max', interval=interval)
                if df.empty:
                    self.logger.warning(f"No data returned for {symbol} using yfinance.")
                    return

                # Identify the date column
                date_col = 'Date' if 'Date' in df.columns else None
                if not date_col:
                    self.logger.error(f"No 'Date' column found in data for {symbol}. Available columns: {df.columns.tolist()}")
                    return

                # Convert date column to datetime and remove timezone
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df[date_col] = df[date_col].dt.tz_localize(None)

                # Drop rows with invalid dates
                df = df.dropna(subset=[date_col])

                # Filter data within the specified date range
                df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

                if df.empty:
                    self.logger.info(f"No new data to update for {symbol}.")
                    return

                # Limit data fetched to prevent overloading the database
                if len(df) > 1000:
                    self.logger.info(f"Data for {symbol} has {len(df)} rows, limiting to the most recent 1000 records.")
                    df = df.tail(1000)

                # Apply technical indicators
                df_with_indicators = apply_all_indicators(df, db_handler=self, logger=self.logger)
                self.logger.info(f"Technical indicators applied for {symbol}.")

                # Save data asynchronously
                await self.upsert_data_async(df_with_indicators, 'stock_data')

            else:
                self.logger.error(f"Unsupported source: {source}")

        except Exception as e:
            self.logger.error(f"Error updating stock data for {symbol} asynchronously: {e}", exc_info=True)

    async def update_all_stocks_async(self, symbols: List[str], source: str = 'yfinance',
                                      start_date: Optional[datetime] = None, end_date: Optional[datetime] = None,
                                      interval: str = '1d'):
        """
        Asynchronously updates stock data for all specified symbols in the list.

        Args:
            symbols (List[str]): List of stock symbols.
            source (str): Data source ('yfinance' or 'alpaca').
            start_date (Optional[datetime]): Start date for fetching data.
            end_date (Optional[datetime]): End date for fetching data.
            interval (str): Data interval (e.g., '1d').
        """
        async def update_symbol(symbol: str):
            # Skip symbols with suffixes indicating delisting or special status
            if '-' in symbol:
                self.logger.info(f"Skipping delisted or special symbol: {symbol}")
                return

            self.logger.info(f"Updating data for {symbol}")
            try:
                await self.update_stock_data_async(symbol, source=source,
                                                   start_date=start_date, end_date=end_date, interval=interval)
                self.logger.info(f"Successfully updated data for {symbol}")
            except Exception as e:
                self.logger.error(f"Failed to update data for {symbol}: {e}", exc_info=True)

        tasks = [update_symbol(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)

    # -------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------

    def alter_table_add_columns(self):
        """
        Alters the 'stock_data' and 'ai_reports' tables to add missing columns if they do not exist.
        """
        inspector = inspect(self.engine)

        # For stock_data table
        if inspector.has_table('stock_data'):
            columns_stock = inspector.get_columns('stock_data')
            existing_columns_stock = [column['name'] for column in columns_stock]

            required_columns_stock = {
                'open': Float,
                'high': Float,
                'low': Float,
                'close': Float,
                'volume': Integer,
                'symbol': String,
                'timestamp': DateTime,
                # Add other columns as necessary
            }

            with self.engine.connect() as conn:
                for column_name, column_type in required_columns_stock.items():
                    if column_name not in existing_columns_stock:
                        try:
                            if column_type == Float:
                                sql_type = 'FLOAT'
                            elif column_type == Integer:
                                sql_type = 'INTEGER'
                            elif column_type == String:
                                sql_type = 'VARCHAR(255)'
                            elif column_type == DateTime:
                                sql_type = 'TIMESTAMP'
                            else:
                                sql_type = 'VARCHAR(255)'

                            alter_stmt = f'ALTER TABLE stock_data ADD COLUMN {column_name} {sql_type};'
                            conn.execute(text(alter_stmt))
                            self.logger.info(f"Added missing column '{column_name}' to 'stock_data' table.")
                        except Exception as e:
                            self.logger.error(f"Error adding column '{column_name}' to 'stock_data': {e}", exc_info=True)
        else:
            self.logger.error("Table 'stock_data' does not exist.")

        # For ai_reports table
        if not inspector.has_table('ai_reports'):
            try:
                AIReport.__table__.create(bind=self.engine)
                self.logger.info("Created table 'ai_reports'.")
            except Exception as e:
                self.logger.error(f"Error creating table 'ai_reports': {e}", exc_info=True)
        else:
            self.logger.info("Table 'ai_reports' already exists.")

    def get_top_150_symbols(self) -> List[str]:
        """
        Fetches the top 150 symbols from the NASDAQ-100 and NASDAQ Composite.
        Ensures that specified symbols like AAPL, TSLA, etc., are included.

        Returns:
            List[str]: List of top 150 stock symbols.
        """
        # Start with mandatory symbols
        mandatory_symbols = {"AAPL", "TSLA", "QQQ", "SPY", "AMZN", "RYCEY"}  # Rolls-Royce (RYCEY) as a placeholder
        top_150_symbols = []

        # Fetch NASDAQ-100 symbols from Wikipedia
        try:
            url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable sortable'})

            nasdaq_100 = []
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cols = row.find_all('td')
                    if cols:
                        symbol = cols[1].text.strip().upper()
                        nasdaq_100.append(symbol)
                self.logger.info(f"Fetched {len(nasdaq_100)} NASDAQ-100 symbols.")
            else:
                self.logger.error("NASDAQ-100 table not found.")

            # Include NASDAQ-100 symbols in top_150_symbols
            top_150_symbols.extend(nasdaq_100)

        except Exception as e:
            self.logger.error(f"Error fetching NASDAQ-100 symbols: {e}", exc_info=True)

        # Ensure mandatory symbols are included and count towards the 150
        top_150_symbols = list(mandatory_symbols | set(top_150_symbols))[:150]

        # Fetch additional top NASDAQ symbols if needed
        if len(top_150_symbols) < 150:
            additional_needed = 150 - len(top_150_symbols)
            try:
                url = "https://finance.yahoo.com/quote/%5EIXIC/components?p=%5EIXIC"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                tables = pd.read_html(response.text)
                if tables:
                    composite_symbols = tables[0]['Symbol'].tolist()
                    composite_symbols = [s.upper() for s in composite_symbols if isinstance(s, str)]
                    composite_symbols = [s for s in composite_symbols if s not in top_150_symbols]
                    top_150_symbols.extend(composite_symbols[:additional_needed])
                else:
                    self.logger.error("No tables found on NASDAQ Composite page.")
            except Exception as e:
                self.logger.error(f"Error fetching additional NASDAQ symbols: {e}", exc_info=True)

        # Ensure exactly 150 symbols
        top_150_symbols = top_150_symbols[:150]
        self.logger.info(f"Total top symbols fetched: {len(top_150_symbols)}")
        return top_150_symbols

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError

async def create_async_engine_connection(async_database_url: str):
    """
    Creates and returns an asynchronous SQLAlchemy engine.

    Args:
        async_database_url (str): The async URL for the database connection.

    Returns:
        AsyncEngine: A configured SQLAlchemy asynchronous engine.
    """
    try:
        # Create the async engine
        async_engine = create_async_engine(async_database_url, echo=False)
        
        # Log success
        logger.info("Asynchronous database engine created successfully.")
        
        return async_engine  # Return the AsyncEngine instance

    except SQLAlchemyError as e:
        logger.error(f"Error creating asynchronous database engine: {e}", exc_info=True)
        raise

# -------------------------------------------------------------------
# Main Function with Summary Log
# -------------------------------------------------------------------

def main():
    data_handler = DataHandler(logger)
    summary_log = {
        "database_setup": "Pending",
        "top_150_symbols_fetched": "Pending",
        "stock_data_updates": [],
        "missing_columns_added": "Pending"
    }

    # Step 1: Ensure 'stock_data' and 'ai_reports' tables have required columns
    try:
        data_handler.alter_table_add_columns()
        summary_log["missing_columns_added"] = "Success"
    except Exception as e:
        summary_log["missing_columns_added"] = f"Failed: {e}"

    # Step 2: Fetch top 150 symbols with custom inclusions
    try:
        top_150_symbols = data_handler.get_top_150_symbols()
        summary_log["top_150_symbols_fetched"] = f"Fetched {len(top_150_symbols)} symbols"
    except Exception as e:
        summary_log["top_150_symbols_fetched"] = f"Failed: {e}"
        top_150_symbols = []

    # Step 3: Update stock data for fetched symbols asynchronously
    try:
        asyncio.run(data_handler.update_all_stocks_async(
            symbols=top_150_symbols,
            source='yfinance',
            start_date=datetime.now() - timedelta(days=365*5),  # Last 5 years
            end_date=datetime.now(),
            interval='1d'  # Daily data
        ))
        summary_log["stock_data_updates"].append({"status": "Success"})
    except Exception as e:
        summary_log["stock_data_updates"].append({"status": f"Failed: {e}"})

    # Final Summary Report
    logger.info("Summary Report:")
    logger.info(f"Database setup: {summary_log['database_setup']}")
    logger.info(f"Missing columns added: {summary_log['missing_columns_added']}")
    logger.info(f"Top 150 symbols fetched: {summary_log['top_150_symbols_fetched']}")

    logger.info("Processing completed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Script interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        logger.info("Processing completed.")
