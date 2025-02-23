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
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from sqlalchemy import (
    create_engine, inspect, text,
    Column, Integer, String, Float, DateTime
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Local utilities
from Utilities.config_manager import ConfigManager
from Utilities.shared_utils import setup_logging

# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]  # Adjust if needed
dotenv_path = project_root / ".env"

# SQLAlchemy base for ORM models
Base = declarative_base()

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = setup_logging(script_name="sql_data_handler")
logger.info("Logger initialized for SQL Data Handler.")

# -------------------------------------------------------------------
# Database Models
# -------------------------------------------------------------------
class StockData(Base):
    """
    Example ORM model representing your 'stock_data' table.
    """
    __tablename__ = "stock_data"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, name="Date", nullable=True)
    open = Column(Float, name="Open", nullable=True)
    high = Column(Float, name="High", nullable=True)
    low = Column(Float, name="Low", nullable=True)
    close = Column(Float, name="Close", nullable=True)
    volume = Column(Integer, name="Volume", nullable=True)
    symbol = Column(String, name="symbol", nullable=True)


class AIReport(Base):
    """
    Example ORM model representing your 'ai_reports' table.
    """
    __tablename__ = "ai_reports"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    report = Column(String, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)


class Symbol(Base):
    """
    Example ORM model representing your 'symbols' table.
    """
    __tablename__ = "symbols"
    symbol = Column(String, primary_key=True, nullable=False)
    name = Column(String, nullable=True)
    exchange = Column(String, nullable=True)
    asset_type = Column(String, nullable=True)
    date_added = Column(DateTime, default=datetime.utcnow)


class TradeHistory(Base):
    """
    ORM model for the trade history table.
    """
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    action = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


# -------------------------------------------------------------------
# SQLDataHandler
# -------------------------------------------------------------------
class SQLDataHandler:
    """
    A unified database handler with sync/async support.
    """

    def __init__(self):
        load_dotenv(dotenv_path)  # Load environment variables from .env
        self.logger = logger

        # 1) Initialize ConfigManager
        self.config_manager = self._initialize_config()

        # 2) Create sync engine & session
        self.engine, self.SessionLocal = self._create_sync_engine()

        # 3) Optionally create an async engine if needed
        #    (If you do want async usage, define the next line & _create_async_engine method)
        # self.async_engine, self.AsyncSessionLocal = self._create_async_engine()

        # 4) Ensure tables exist
        Base.metadata.create_all(self.engine)
        self.logger.info("✅ Database tables initialized.")

        # If you want aggregator logic, uncomment or adapt:
        # self.indicator_unifier = AllIndicatorsUnifier(...)

        # Optionally initialize Alpaca
        # self.alpaca_api = self._initialize_alpaca()

    def _initialize_config(self) -> ConfigManager:
        """
        Load config from environment, ensuring required keys exist.
        """
        required_keys = [
            "POSTGRES_HOST", "POSTGRES_DBNAME", "POSTGRES_USER",
            "POSTGRES_PASSWORD", "POSTGRES_PORT",
            "ALPHAVANTAGE_API_KEY", "ALPHAVANTAGE_BASE_URL",
            "ALPACA_API_KEY", "ALPACA_SECRET_KEY",
            "ALPACA_BASE_URL", "DATABASE_URL"
        ]
        try:
            cfg = ConfigManager(env_file=dotenv_path, required_keys=required_keys, logger=self.logger)
            return cfg
        except KeyError as e:
            self.logger.error(f"❌ Missing required config keys: {e}")
            sys.exit(1)

    def _create_sync_engine(self):
        """
        Creates a synchronous SQLAlchemy engine and sessionmaker.
        """
        database_url = self.config_manager.get("DATABASE_URL")
        if not database_url:
            self.logger.error("❌ DATABASE_URL is missing or not set.")
            sys.exit(1)

        try:
            engine = create_engine(database_url, echo=False)
            session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            return engine, session_local
        except SQLAlchemyError as e:
            self.logger.error(f"❌ Error setting up sync database: {e}", exc_info=True)
            raise

    # If you do want an async engine, define a method like:
    """
    def _create_async_engine(self):
        database_url = self.config_manager.get("DATABASE_URL")
        if not database_url:
            ...
        # convert if needed
        async_engine = create_async_engine(async_database_url, echo=False)
        async_session_local = async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)
        return async_engine, async_session_local
    """

    def get_session(self) -> Session:
        """
        Returns a synchronous session.
        """
        return self.SessionLocal()

    def save_trade(self, trade_details: dict):
        """
        Saves a trade record into the `trades` table.
        """
        session = self.get_session()
        try:
            trade_entry = TradeHistory(
                symbol=trade_details["symbol"],
                action=trade_details["action"],
                quantity=trade_details["quantity"],
                price=trade_details["price"],
                timestamp=trade_details["timestamp"]
            )
            session.add(trade_entry)
            session.commit()
            self.logger.info(f"📊 Trade saved to DB: {trade_details}")
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"❌ Failed to save trade: {e}", exc_info=True)
        finally:
            session.close()

    def alter_table_add_columns(self):
        """
        Example: Add missing columns to the 'stock_data' table if they do not exist.
        """
        inspector = inspect(self.engine)
        if inspector.has_table("stock_data"):
            existing_cols = [col["name"] for col in inspector.get_columns("stock_data")]
            required_cols = {
                "symbol": "VARCHAR(50)",
                "timestamp": "TIMESTAMP",
                "open": "FLOAT",
                "high": "FLOAT",
                "low": "FLOAT",
                "close": "FLOAT",
                "volume": "INT"
            }
            with self.engine.connect() as conn:
                for col_name, col_type in required_cols.items():
                    if col_name not in existing_cols:
                        try:
                            conn.execute(text(f"ALTER TABLE stock_data ADD COLUMN {col_name} {col_type}"))
                            self.logger.info(f"✅ Added missing column '{col_name}' to 'stock_data'.")
                        except Exception as e:
                            self.logger.error(f"❌ Error adding column '{col_name}': {e}", exc_info=True)
        else:
            self.logger.warning("⚠️ Table 'stock_data' does not exist.")

    # Example method for saving stock data from a DataFrame into 'stock_data' table
    def save_stock_data(self, symbol: str, data: pd.DataFrame):
        """
        Saves rows from a DataFrame into 'stock_data' table.
        """
        session = self.get_session()
        try:
            # Ensure table
            if not inspect(self.engine).has_table("stock_data"):
                Base.metadata.create_all(self.engine)

            # Convert date column if needed
            if "Date" in data.columns:
                data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

            # Insert rows
            records = []
            for _, row in data.iterrows():
                sd = StockData(
                    date=row.get("Date"),
                    open=row.get("Open"),
                    high=row.get("High"),
                    low=row.get("Low"),
                    close=row.get("Close"),
                    volume=row.get("Volume"),
                    symbol=symbol
                )
                records.append(sd)

            session.bulk_save_objects(records)
            session.commit()
            self.logger.info(f"✅ Saved {len(records)} rows for {symbol} into stock_data.")
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"❌ SQLAlchemy error while saving stock data for {symbol}: {e}", exc_info=True)
        finally:
            session.close()

    def fetch_from_yfinance(self, symbol: str, period: str="1y", interval: str="1d") -> pd.DataFrame:
        """
        Simple wrapper for yfinance data retrieval.
        """
        try:
            yf_ticker = yf.Ticker(symbol)
            df = yf_ticker.history(period=period, interval=interval)
            df.reset_index(inplace=True)
            df["symbol"] = symbol
            self.logger.info(f"📈 Fetched {len(df)} rows for {symbol} from yfinance.")
            return df
        except Exception as e:
            self.logger.error(f"❌ fetch_from_yfinance error for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def create_or_update_tables(self):
        """
        Placeholder for further custom DDL operations if needed.
        """
        inspector = inspect(self.engine)
        self.logger.info("Tables checked or updated as needed.")

    # If you want to integrate Alpaca or other fetch methods:
    """
    def _initialize_alpaca(self):
        try:
            api_key = self.config_manager.get('ALPACA_API_KEY')
            secret_key = self.config_manager.get('ALPACA_SECRET_KEY')
            base_url = self.config_manager.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

            if not all([api_key, secret_key, base_url]):
                raise ValueError("Missing Alpaca credentials.")
            self.logger.info("Alpaca API credentials loaded.")
            return tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        except Exception as e:
            self.logger.error(f"❌ Error initializing Alpaca API: {e}", exc_info=True)
            return None

    def fetch_from_alpaca(self, symbol: str, timeframe: str='1Day', start_date: Optional[str]=None, end_date: Optional[str]=None):
        ...
    """

# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    # Example usage
    handler = SQLDataHandler()

    # Create/update any table columns if needed
    handler.alter_table_add_columns()

    # Example: fetch data from yfinance and save
    symbol = "AAPL"
    df = handler.fetch_from_yfinance(symbol, period="1mo", interval="1d")
    if not df.empty:
        handler.save_stock_data(symbol, df)
    else:
        handler.logger.info(f"No data to save for {symbol} from yfinance.")

    handler.logger.info("Done with main logic in sql_data_handler.")

if __name__ == "__main__":
    main()
