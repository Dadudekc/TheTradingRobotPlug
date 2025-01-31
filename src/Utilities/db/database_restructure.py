# -------------------------------------------------------------------
# File Path: src/Utilities/db/database_restructure.py
# Description: Handles database restructuring tasks, including:
#              - Ensuring required columns exist
#              - Creating indexes for efficient queries
#              - Cleaning up old data
# -------------------------------------------------------------------

import psycopg2
import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# -------------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------------
env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DatabaseRestructure")

# -------------------------------------------------------------------
# DatabaseRestructure Class
# -------------------------------------------------------------------
class DatabaseRestructure:
    """Handles all database restructuring tasks such as column validation, indexing, and cleanup."""

    def __init__(self):
        """Initialize database connection parameters."""
        self.db_name = os.getenv("POSTGRES_DBNAME", "trading_robot_plug")
        self.db_user = os.getenv("POSTGRES_USER", "postgres")
        self.db_password = os.getenv("POSTGRES_PASSWORD", "your_password")
        self.db_host = os.getenv("POSTGRES_HOST", "localhost")
        self.db_port = os.getenv("POSTGRES_PORT", "5434")

        self.stock_tables = ["AAPL", "TSLA", "AMZN"]
        self.required_columns = [
            "macd_line", "macd_signal", "macd_histogram", "rsi", "bollinger_width"
        ]

    def connect_db(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            conn = psycopg2.connect(
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            return conn
        except Exception as e:
            logger.error(f"⚠️ Error connecting to the database: {e}")
            raise

    def execute_query(self, cursor, conn, query, success_msg, error_msg):
        """Executes a query and commits changes, with rollback on failure."""
        try:
            cursor.execute(query)
            conn.commit()
            logger.info(success_msg)
        except Exception as e:
            conn.rollback()
            logger.error(f"{error_msg}: {e}")

    def clean_old_data(self):
        """Removes stock data older than 2 years."""
        conn = self.connect_db()
        cursor = conn.cursor()

        for table in self.stock_tables:
            query = f"""
            DELETE FROM {table} 
            WHERE TO_TIMESTAMP(date::TEXT, 'YYYY-MM-DD HH24:MI:SS') < NOW() - INTERVAL '2 years';
            """
            self.execute_query(cursor, conn, query, f"🧹 Cleaned old data in {table}", f"⚠️ Error cleaning old data in {table}")

        cursor.close()
        conn.close()

    def ensure_required_columns(self):
        """Ensures required technical indicator columns exist in stock tables."""
        conn = self.connect_db()
        cursor = conn.cursor()

        for table in self.stock_tables:
            for col in self.required_columns:
                query = f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION;"
                self.execute_query(cursor, conn, query, f"✅ Ensured column {col} exists in {table}", f"⚠️ Error adding column {col} in {table}")

        cursor.close()
        conn.close()

    def ensure_indexes(self):
        """Ensures indexes exist for faster queries."""
        conn = self.connect_db()
        cursor = conn.cursor()

        for table in self.stock_tables:
            query = f"CREATE INDEX IF NOT EXISTS idx_{table.lower()}_symbol_date ON {table}(symbol, date);"
            self.execute_query(cursor, conn, query, f"✅ Index created on {table} (symbol, date)", f"⚠️ Error creating index on {table}")

        cursor.close()
        conn.close()

    def restructure_database(self):
        """Executes all database restructuring steps."""
        try:
            logger.info("🔄 Starting database restructuring...")
            self.ensure_required_columns()
            self.ensure_indexes()
            self.clean_old_data()
            logger.info("✅ Database restructuring complete!")
        except Exception as e:
            logger.error(f"⚠️ Error during database restructuring: {e}")

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    db_restructure = DatabaseRestructure()
    db_restructure.restructure_database()
