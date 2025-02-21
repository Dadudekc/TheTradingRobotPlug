"""
File: db_inspect_and_update.py
Path: src/Utilities/db/db_inspect_and_update.py

Description:
    Handles database inspection, schema modifications,
    fetching stock data from APIs, and migration to PostgreSQL.
"""

import os
import sys
import asyncio
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import logging

# -------------------------------------------------------------------
# 🛠️ Project Path Setup
# -------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
utilities_dir = project_root / 'src' / 'Utilities'

sys.path.append(str(utilities_dir))

# -------------------------------------------------------------------
# 🛠️ Import Core Modules (Lazy Fetchers Are Handled Separately)
# -------------------------------------------------------------------
try:
    from Utilities.config_manager import ConfigManager
    from Utilities.shared_utils import setup_logging
    from Utilities.data_processing.Technical_Indicators.indicator_unifier import AllIndicatorsUnifier
except ModuleNotFoundError as e:
    print(f"❌ Error importing core modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# 📝 Environment & Logging Setup
# -------------------------------------------------------------------
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Logging Setup
log_dir = project_root / 'logs' / 'Utilities'
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logging(str(log_dir / 'db_inspect_and_update'))


# -------------------------------------------------------------------
# 🛠️ Lazy Import Fetchers (To Avoid Circular Imports)
# -------------------------------------------------------------------
def get_fetchers():
    from Utilities.data_fetchers.alpaca_fetcher import AlpacaDataFetcher
    from Utilities.data_fetchers.polygon_fetcher import PolygonDataFetcher
    from Utilities.data_fetchers.alphavantage_fetcher import AlphaVantageFetcher
    from Utilities.data_fetchers.yahoo_finance_fetcher import YahooFinanceFetcher
    
    return {
        "alpaca": AlpacaDataFetcher(),
        "polygon": PolygonDataFetcher(),
        "alpha_vantage": AlphaVantageFetcher(),
        "yahoo": YahooFinanceFetcher(),
    }


# -------------------------------------------------------------------
# 🚀 **DBInspectAndUpdate Class**
# -------------------------------------------------------------------
class DBInspectAndUpdate:
    """
    Handles database inspection, schema modifications,
    data fetching, and migration.
    """

    def __init__(self):
        """Initialize database paths, connections, and data fetchers lazily."""
        self.config_manager = ConfigManager()
        self.sqlite_db_path = self.config_manager.get(
            'SQLITE_DB_PATH', str(project_root / 'data/databases/trading_data.db')
        )

        if not os.path.exists(self.sqlite_db_path):
            logger.error(f"⚠️ SQLite Database not found at {self.sqlite_db_path}")
            raise FileNotFoundError(f"Database not found at {self.sqlite_db_path}")

        self.sqlite_engine = create_engine(f"sqlite:///{self.sqlite_db_path}")
        self.indicator_aggregator = AllIndicatorsUnifier(logger=logger)

        # Lazily initialize fetchers
        self.fetchers = get_fetchers()

        logger.info("✅ DBInspectAndUpdate initialized successfully.")

    def inspect_table_schema(self, table_name: str = 'stock_data') -> Optional[pd.DataFrame]:
        """Inspect the schema of the specified SQLite table."""
        query = f"PRAGMA table_info({table_name});"
        try:
            df_schema = pd.read_sql(query, self.sqlite_engine)
            logger.info(f"🔍 Schema for table '{table_name}':\n{df_schema.to_string(index=False)}")
            return df_schema
        except Exception as e:
            logger.error(f"⚠️ Error inspecting table schema for {table_name}: {e}", exc_info=True)
            return None

    def query_sample_data(self, table_name: str = 'stock_data', limit: int = 5) -> Optional[pd.DataFrame]:
        """Query sample records from the SQLite table."""
        query = f"SELECT * FROM {table_name} LIMIT {limit};"
        try:
            df_sample = pd.read_sql(query, self.sqlite_engine)
            logger.info(f"📊 Sample data from '{table_name}':\n{df_sample.to_string(index=False)}")
            return df_sample
        except Exception as e:
            logger.error(f"⚠️ Error querying sample data from {table_name}: {e}", exc_info=True)
            return None

    def add_technical_indicators_columns(self, table_name: str = 'stock_data'):
        """Add necessary technical indicator columns to a given SQLite table."""
        columns_to_add = [
            'stochastic REAL', 'rsi REAL', 'williams_r REAL',
            'rate_of_change REAL', 'macd REAL', 'bollinger_bands REAL',
            'proprietary_prediction REAL'
        ]
        with self.sqlite_engine.connect() as connection:
            for column in columns_to_add:
                try:
                    query = text(f"ALTER TABLE {table_name} ADD COLUMN {column};")
                    connection.execute(query)
                    logger.info(f"✅ Added column: {column}")
                except SQLAlchemyError as e:
                    if 'duplicate column name' in str(e).lower():
                        logger.warning(f"⚠️ Column '{column.split()[0]}' already exists. Skipping.")
                    else:
                        logger.error(f"⚠️ Could not add column '{column}'. Error: {e}", exc_info=True)

    async def fetch_and_store_data(self, symbol: str, start_date: str, end_date: str):
        """Fetch stock data from APIs and apply technical indicators."""
        logger.info(f"🚀 Fetching data for {symbol} from APIs...")

        fetch_tasks = [
            self.fetchers["alpaca"].fetch_and_store_data(symbol, start=start_date, end=end_date),
            self.fetchers["polygon"].fetch_data_with_date_range(symbol, start_date, end_date),
            self.fetchers["alpha_vantage"].fetch_data_for_symbol(symbol, start_date, end_date),
        ]

        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"⚠️ API Fetch Error: {result}")
            elif result is not None:
                self.indicator_aggregator.apply_all_indicators(result)

        try:
            yfinance_data = self.fetchers["yahoo"].fetch_data_from_yfinance(symbol, start_date, end_date)
            if yfinance_data is not None:
                self.indicator_aggregator.apply_all_indicators(yfinance_data)
        except Exception as e:
            logger.error(f"⚠️ Error fetching Yahoo Finance data: {e}", exc_info=True)

    def execute_update(self):
        """Execute all steps for inspecting, modifying, fetching, and migrating data."""
        try:
            logger.info("🔍 Inspecting table schema...")
            self.inspect_table_schema()

            logger.info("📊 Querying sample data...")
            self.query_sample_data(limit=10)

            logger.info("🛠️ Adding technical indicator columns...")
            self.add_technical_indicators_columns()

            logger.info("🚀 Fetching stock data asynchronously...")
            asyncio.run(self.fetch_and_store_data("AAPL", "2023-01-01", "2023-12-31"))

            logger.info("✅ Database update process completed successfully!")
        except Exception as e:
            logger.error(f"❌ Fatal error during database update: {e}", exc_info=True)


# -------------------------------------------------------------------
# 🔧 Run Script
# -------------------------------------------------------------------
if __name__ == "__main__":
    db_update = DBInspectAndUpdate()
    db_update.execute_update()
