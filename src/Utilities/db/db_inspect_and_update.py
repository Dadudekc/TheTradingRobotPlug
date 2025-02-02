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
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import logging

# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
utilities_dir = project_root / 'src' / 'Utilities'

sys.path.append(str(utilities_dir))

# -------------------------------------------------------------------
# Import Required Modules
# -------------------------------------------------------------------
try:
    from Utilities.config_manager import ConfigManager
    from Utilities.shared_utils import setup_logging
    from Utilities.data_fetchers.alpaca_fetcher import AlpacaDataFetcher
    from Utilities.data_fetchers.polygon_fetcher import PolygonDataFetcher
    from Utilities.data_fetchers.alphavantage_fetcher import AlphaVantageFetcher
    from Utilities.data_fetchers.yahoo_finance_fetcher import YahooFinanceFetcher
    from Utilities.data_processing.indicator_unifier import AllIndicatorsUnifier
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# Logging Setup
log_dir = project_root / 'logs' / 'Utilities'
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logging(str(log_dir / 'db_inspect_and_update'))

# -------------------------------------------------------------------
# DBInspectAndUpdate Class
# -------------------------------------------------------------------
class DBInspectAndUpdate:
    """
    Handles database inspection, schema modifications,
    data fetching, and migration.
    """

    def __init__(self):
        """Initialize database paths, connections, and data fetchers."""
        self.config_manager = ConfigManager()
        self.sqlite_db_path = self.config_manager.get('SQLITE_DB_PATH', str(project_root / 'data/databases/trading_data.db'))

        if not os.path.exists(self.sqlite_db_path):
            logger.error(f"⚠️ SQLite Database not found at {self.sqlite_db_path}")
            raise FileNotFoundError(f"Database not found at {self.sqlite_db_path}")

        self.sqlite_engine = create_engine(f"sqlite:///{self.sqlite_db_path}")
        self.alpaca_fetcher = AlpacaDataFetcher()
        self.polygon_fetcher = PolygonDataFetcher()
        self.alpha_vantage_fetcher = AlphaVantageFetcher()
        self.yfinance_fetcher = YahooFinanceFetcher()
        self.indicator_aggregator = AllIndicatorsUnifier(logger=logger)

    def inspect_table_schema(self, table_name: str = 'stock_data') -> Optional[pd.DataFrame]:
        """
        Inspect the schema of the specified SQLite table.

        :param table_name: Table name to inspect.
        :return: DataFrame with schema details.
        """
        query = f"PRAGMA table_info({table_name});"
        try:
            df_schema = pd.read_sql(query, self.sqlite_engine)
            logger.info(f"Schema for table '{table_name}':\n{df_schema.to_string(index=False)}")
            return df_schema
        except Exception as e:
            logger.error(f"⚠️ Error inspecting table schema for {table_name}: {e}")
            return None

    def query_sample_data(self, table_name: str = 'stock_data', limit: int = 5) -> Optional[pd.DataFrame]:
        """
        Query sample records from the SQLite table.

        :param table_name: Table name.
        :param limit: Number of rows to fetch.
        :return: DataFrame with sample data.
        """
        query = f"SELECT * FROM {table_name} LIMIT {limit};"
        try:
            df_sample = pd.read_sql(query, self.sqlite_engine)
            logger.info(f"Sample data from table '{table_name}':\n{df_sample.to_string(index=False)}")
            return df_sample
        except Exception as e:
            logger.error(f"⚠️ Error querying sample data from {table_name}: {e}")
            return None

    def add_technical_indicators_columns(self, table_name: str = 'stock_data'):
        """
        Add necessary technical indicator columns to a given SQLite table.

        :param table_name: Table name.
        """
        columns_to_add = [
            'stochastic REAL',
            'rsi REAL',
            'williams_r REAL',
            'rate_of_change REAL',
            'macd REAL',
            'bollinger_bands REAL',
            'proprietary_prediction REAL'
        ]
        with self.sqlite_engine.connect() as connection:
            for column in columns_to_add:
                try:
                    query = text(f"ALTER TABLE {table_name} ADD COLUMN {column};")
                    connection.execute(query)
                    logger.info(f"✅ Successfully added column: {column}")
                except SQLAlchemyError as e:
                    if 'duplicate column name' in str(e).lower():
                        logger.warning(f"⚠️ Column '{column.split()[0]}' already exists. Skipping.")
                    else:
                        logger.error(f"⚠️ Could not add column '{column}'. Error: {e}")

    def validate_modifications(self, table_name: str = 'stock_data'):
        """
        Validate schema modifications by inspecting schema and sample data.

        :param table_name: Table name.
        """
        self.query_sample_data(table_name)
        self.inspect_table_schema(table_name)

    async def fetch_and_store_data(self, symbol: str, start_date: str, end_date: str):
        """
        Fetches stock data from multiple APIs and applies technical indicators.

        :param symbol: Stock symbol.
        :param start_date: Start date.
        :param end_date: End date.
        """
        logger.info(f"Fetching data for symbol: {symbol}")
        fetchers = [
            self.alpaca_fetcher.fetch_and_store_data(symbol, start=start_date, end=end_date),
            self.polygon_fetcher.fetch_data_with_date_range(symbol, start_date, end_date),
            self.alpha_vantage_fetcher.fetch_data_for_symbol(symbol, start_date, end_date)
        ]

        results = await asyncio.gather(*fetchers, return_exceptions=True)

        for data in results:
            if isinstance(data, Exception):
                logger.error(f"⚠️ Error fetching data: {data}")
            elif data is not None:
                self.indicator_aggregator.apply_all_indicators(data)

        # Fetch Yahoo Finance data separately (as it may not be async)
        try:
            yfinance_data = self.yfinance_fetcher.fetch_data_from_yfinance(symbol, start_date, end_date)
            if yfinance_data is not None:
                self.indicator_aggregator.apply_all_indicators(yfinance_data)
        except Exception as e:
            logger.error(f"⚠️ Error fetching Yahoo Finance data: {e}")

    def execute_update(self):
        """
        Execute all steps for inspecting, modifying, fetching, and migrating data.
        """
        try:
            logger.info("🔍 Inspecting table schema...")
            self.inspect_table_schema()

            logger.info("📊 Querying sample data...")
            self.query_sample_data(limit=10)

            logger.info("🛠️ Adding technical indicator columns...")
            self.add_technical_indicators_columns()

            logger.info("✅ Validating modifications...")
            self.validate_modifications()

            logger.info("🚀 Fetching stock data...")
            asyncio.run(self.fetch_and_store_data("AAPL", "2023-01-01", "2023-12-31"))

            logger.info("✅ Database update process completed successfully!")

        except Exception as e:
            logger.error(f"⚠️ Error during database update: {e}")

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    db_update = DBInspectAndUpdate()
    db_update.execute_update()
