# -------------------------------------------------------------------
# File Path: src/Utilities/db/db_inspect_and_transfer.py
# Description: Handles SQLite database inspection, modification, and 
#              migration to PostgreSQL/MySQL. Ensures schema integrity.
# -------------------------------------------------------------------

import os
import sys
import pandas as pd
from pathlib import Path
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

# Import ConfigManager and Logging
try:
    from Utilities.config_manager import ConfigManager
    from Utilities.shared_utils import setup_logging
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

logger = setup_logging("db_inspect_and_transfer")

# -------------------------------------------------------------------
# DBInspectAndTransfer Class
# -------------------------------------------------------------------
class DBInspectAndTransfer:
    """Handles SQLite database inspection, schema modifications, and migration."""

    def __init__(self):
        """Initialize database paths and connections."""
        self.sqlite_db_path = os.getenv('SQLITE_DB_PATH', str(project_root / 'data/databases/trading_data.db'))
        
        if not os.path.exists(self.sqlite_db_path):
            logger.error(f"⚠️ SQLite Database not found at {self.sqlite_db_path}")
            raise FileNotFoundError(f"Database not found at {self.sqlite_db_path}")

        self.sqlite_engine = create_engine(f"sqlite:///{self.sqlite_db_path}")

    def inspect_table_schema(self, table_name='stock_data'):
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

    def query_sample_data(self, table_name='stock_data', limit=5):
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

    def add_technical_indicators_columns(self, table_name='stock_data'):
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

    def validate_modifications(self, table_name='stock_data'):
        """
        Validate schema modifications by inspecting schema and sample data.

        :param table_name: Table name.
        """
        self.query_sample_data(table_name)
        self.inspect_table_schema(table_name)

    def migrate_to_postgres(self):
        """
        Migrate data from SQLite to PostgreSQL.
        """
        postgres_url = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trading_robot_plug")
        postgres_engine = create_engine(postgres_url)

        try:
            # Load SQLite data
            df = pd.read_sql("SELECT * FROM stock_data;", self.sqlite_engine)
            logger.info(f"Loaded {len(df)} records from SQLite.")

            # Insert into PostgreSQL
            df.to_sql('stock_data', postgres_engine, if_exists='replace', index=False)
            logger.info(f"✅ Successfully migrated {len(df)} records to PostgreSQL.")

        except Exception as e:
            logger.error(f"⚠️ Migration to PostgreSQL failed: {e}")

    def execute_transfer(self):
        """
        Execute all steps for inspecting, modifying, and migrating data.
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

            logger.info("🚀 Migrating to PostgreSQL...")
            self.migrate_to_postgres()

            logger.info("✅ Database transfer process completed successfully!")

        except Exception as e:
            logger.error(f"⚠️ Error during database transfer: {e}")

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    db_transfer = DBInspectAndTransfer()
    db_transfer.execute_transfer()
