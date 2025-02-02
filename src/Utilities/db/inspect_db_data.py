"""
File: inspect_db_data.py
Description:
    Connects to a PostgreSQL database using SQLAlchemy,
    retrieves stock data for a given symbol from the 'stock_data' table,
    and fetches a configurable number of records sorted by date.
"""

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import os

from Utilities.shared_utils import setup_logging

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------
logger = setup_logging("inspect_db_data")


# -------------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------------
env_path = Path(__file__).resolve().parents[3] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info("Environment variables loaded successfully.")
else:
    logger.warning(".env file not found. Ensure database credentials are set.")

# -------------------------------------------------------------------
# Database Configuration
# -------------------------------------------------------------------
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5434")
DB_NAME = os.getenv("POSTGRES_DBNAME", "trading_robot_plug")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


class InspectDBData:
    """
    Handles fetching stock data from the PostgreSQL database.
    """


    def __init__(self):
        """
        Initialize the database connection.
        """
        try:
            self.engine = create_engine(DATABASE_URL)
            logger.info("Successfully connected to the PostgreSQL database.")
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to the database: {e}", exc_info=True)
            self.engine = None

    def inspect_db_data(self, symbol: str = "AAPL", limit: int = 10) -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a given symbol, sorted by date.


        Args:
            symbol (str): Stock symbol to fetch data for (default: 'AAPL').
            limit (int): Number of records to fetch (default: 10).

        Returns:
            pd.DataFrame: DataFrame containing fetched stock data, or None on error.
        """
        if not self.engine:
            logger.error("No database connection available.")
            return None


        query = text(
            f"""
            SELECT * FROM stock_data
            WHERE symbol = :symbol
            ORDER BY "Date" ASC
            LIMIT :limit;
            """
        )

        try:
            with self.engine.connect() as connection:
                df = pd.read_sql_query(query, con=connection, params={"symbol": symbol, "limit": limit})
                logger.info(f"Successfully fetched {len(df)} records for {symbol}.")
                return df
        except SQLAlchemyError as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}", exc_info=True)
            return None


    def close_connection(self):
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed.")



# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    fetcher = InspectDBData()
    df_aapl = fetcher.inspect_db_data(symbol="AAPL", limit=10)


    if df_aapl is not None and not df_aapl.empty:
        print(df_aapl)
    else:
        print("No data found or failed to fetch data.")


    fetcher.close_connection()
