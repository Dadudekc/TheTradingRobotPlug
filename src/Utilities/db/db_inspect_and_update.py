# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/src/Utilities/db_inspect_and_update.py
# Description: Script to inspect and update the database, integrating technical indicators 
#              and fetching data from various APIs. It handles database connections, 
#              schema modifications, data migration, and data fetching from multiple sources.
#              Utilizes SQLAlchemy for database interactions and pandas for data manipulation.
#              Logging is set up to track the migration and update processes.
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import logging
import pandas as pd
from pathlib import Path
from typing import Optional
import asyncio
import os
import sys
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError, OperationalError

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
data_fetching_dir = project_root / 'src' / 'Data_Fetchers' / 'Stock_Data'

# Add the necessary directories to the Python path
sys.path.extend([
    str(data_processing_dir),
    str(utilities_dir),
    str(data_fetching_dir),
    str(model_dir),
    str(model_utils)
])

# -------------------------------------------------------------------
# Import ConfigManager and Logging Setup
# -------------------------------------------------------------------
try:
    from config_handling.config_manager import ConfigManager
    from config_handling.logging_setup import setup_logging
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------
# Load environment variables from .env file
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------
log_dir = project_root / 'logs' / 'Utilities'
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logging(str(log_dir / 'db_inspect_and_update'))
logger.info("Logging initialized for db_inspect_and_update.py")

# -------------------------------------------------------------------
# SQLAlchemy Base and Models
# -------------------------------------------------------------------
Base = declarative_base()

class StockData(Base):
    __tablename__ = 'stock_data'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    stochastic = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    williams_r = Column(Float, nullable=True)
    rate_of_change = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    bollinger_bands = Column(Float, nullable=True)
    proprietary_prediction = Column(Float, nullable=True)

class ModelData(Base):
    __tablename__ = 'model_data'
    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False, index=True)
    model_type = Column(String, nullable=False)
    training_date = Column(DateTime, nullable=False)
    hyperparameters = Column(String)  # JSON string or similar format
    metrics = Column(String)  # JSON string to store model metrics like accuracy, loss, etc.
    description = Column(String)  # Optional description of the model

# -------------------------------------------------------------------
# Function Definitions
# -------------------------------------------------------------------

def inspect_table_schema(engine, table_name='stock_data') -> Optional[pd.DataFrame]:
    """
    Inspects the schema of the provided database table and logs the information.

    Args:
        engine: SQLAlchemy engine connected to the database.
        table_name (str): Name of the table to inspect.

    Returns:
        pd.DataFrame: DataFrame containing the schema information.
    """
    query = f"PRAGMA table_info({table_name});"
    try:
        df_schema = pd.read_sql(query, engine)
        print(f"Schema for table '{table_name}':\n{df_schema}")
        logger.info(f"Schema for table '{table_name}':\n{df_schema.to_string(index=False)}")
        return df_schema
    except Exception as e:
        logger.error(f"Error inspecting table schema for {table_name}: {e}")
        print(f"Error inspecting table schema for {table_name}: {e}")
        return None

def query_sample_data(engine, table_name='stock_data', limit=10) -> Optional[pd.DataFrame]:
    """
    Queries a sample of the data from the specified table.

    Args:
        engine: SQLAlchemy engine connected to the database.
        table_name (str): Name of the table to query.
        limit (int): Number of rows to retrieve.

    Returns:
        pd.DataFrame: DataFrame containing the sample data.
    """
    query = f"SELECT * FROM {table_name} LIMIT {limit};"
    try:
        df_sample = pd.read_sql(query, engine)
        print(f"Sample data from table '{table_name}' (Limited to {limit} rows):\n{df_sample}")
        logger.info(f"Sample data from table '{table_name}':\n{df_sample.to_string(index=False)}")
        return df_sample
    except Exception as e:
        logger.error(f"Error querying sample data from {table_name}: {e}")
        print(f"Error querying sample data from {table_name}: {e}")
        return None

def add_technical_indicators_columns(engine, table_name='stock_data'):
    """
    Adds columns for technical indicators to the specified table.

    Args:
        engine: SQLAlchemy engine connected to the database.
        table_name (str): Name of the table to modify.
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

    with engine.connect() as connection:
        for column in columns_to_add:
            try:
                query = text(f"ALTER TABLE {table_name} ADD COLUMN {column};")
                connection.execute(query)
                print(f"Successfully added column: {column}")
                logger.info(f"Successfully added column: {column}")
            except SQLAlchemyError as e:
                if 'duplicate column name' in str(e).lower():
                    column_name = column.split()[0]
                    print(f"Column '{column_name}' already exists. Skipping.")
                    logger.warning(f"Column '{column_name}' already exists. Skipping.")
                else:
                    print(f"Warning: Could not add column '{column}'. Error: {e}")
                    logger.error(f"Could not add column '{column}'. Error: {e}")

def check_modifications(engine, table_name='stock_data'):
    """
    Validates the modifications to the table by querying sample data and inspecting schema.

    Args:
        engine: SQLAlchemy engine connected to the database.
        table_name (str): Name of the table to validate.
    """
    query_sample_data(engine, table_name, limit=10)
    inspect_table_schema(engine, table_name)

def migrate_stockdata_to_postgresql(sqlite_engine, pg_engine, table_name='stock_data'):
    """
    Migrates StockData from SQLite to PostgreSQL.

    Args:
        sqlite_engine: SQLAlchemy engine connected to the SQLite database.
        pg_engine: SQLAlchemy engine connected to the PostgreSQL database.
        table_name (str): Name of the table to migrate.
    """
    try:
        # Read data from SQLite
        stockdata_df = pd.read_sql_table(table_name, sqlite_engine)
        print(f"Read {len(stockdata_df)} rows from SQLite table '{table_name}'.")
        logger.info(f"Read {len(stockdata_df)} rows from SQLite table '{table_name}'.")

        # Handle missing technical indicator columns by filling with None
        technical_columns = ['stochastic', 'rsi', 'williams_r', 'rate_of_change', 'macd', 'bollinger_bands', 'proprietary_prediction']
        for col in technical_columns:
            if col not in stockdata_df.columns:
                stockdata_df[col] = None
                logger.warning(f"Column '{col}' not found in SQLite data. Filled with None.")

        # Write to PostgreSQL using SQLAlchemy
        stockdata_df.to_sql(table_name, pg_engine, if_exists='append', index=False)
        print(f"Data migrated to PostgreSQL table '{table_name}' successfully.")
        logger.info(f"Data migrated to PostgreSQL table '{table_name}' successfully.")
    except Exception as e:
        logger.error(f"Error migrating StockData to PostgreSQL: {e}")
        print(f"Error migrating StockData to PostgreSQL: {e}")

def create_modeldata_table_postgresql(pg_engine, table_name='model_data'):
    """
    Creates the ModelData table in PostgreSQL if it does not exist.

    Args:
        pg_engine: SQLAlchemy engine connected to the PostgreSQL database.
        table_name (str): Name of the ModelData table.
    """
    try:
        Base.metadata.create_all(pg_engine, tables=[ModelData.__table__])
        print(f"Table '{table_name}' created or already exists in PostgreSQL.")
        logger.info(f"Table '{table_name}' created or already exists in PostgreSQL.")
    except Exception as e:
        logger.error(f"Error creating ModelData table in PostgreSQL: {e}")
        print(f"Error creating ModelData table in PostgreSQL: {e}")

def migrate_modeldata_to_postgresql(sqlite_engine, pg_engine, table_name='model_data'):
    """
    Migrates ModelData from SQLite to PostgreSQL if data exists.

    Args:
        sqlite_engine: SQLAlchemy engine connected to the SQLite database.
        pg_engine: SQLAlchemy engine connected to the PostgreSQL database.
        table_name (str): Name of the table to migrate.
    """
    try:
        # Check if ModelData table exists in SQLite
        tables = inspect_table_schema(sqlite_engine)
        if tables is None or table_name not in tables['name'].tolist():
            print(f"No data migration for '{table_name}' as it does not exist in SQLite.")
            logger.info(f"No data migration for '{table_name}' as it does not exist in SQLite.")
            return

        # Read data from SQLite
        modeldata_df = pd.read_sql_table(table_name, sqlite_engine)
        if modeldata_df.empty:
            print(f"No data found in '{table_name}' to migrate.")
            logger.info(f"No data found in '{table_name}' to migrate.")
            return

        # Write to PostgreSQL using SQLAlchemy
        modeldata_df.to_sql(table_name, pg_engine, if_exists='append', index=False)
        print(f"Data migrated to PostgreSQL table '{table_name}' successfully.")
        logger.info(f"Data migrated to PostgreSQL table '{table_name}' successfully.")
    except Exception as e:
        logger.error(f"Error migrating ModelData to PostgreSQL: {e}")
        print(f"Error migrating ModelData to PostgreSQL: {e}")

# -------------------------------------------------------------------
# Function to Fetch and Store Data from Various APIs
# -------------------------------------------------------------------
async def fetch_and_store_data(symbol: str, start_date: str, end_date: str):
    """
    Fetches historical stock data from multiple APIs and stores it in separate tables.

    Args:
        symbol (str): Stock symbol to fetch data for.
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).
    """
    logger.info(f"Fetching data for symbol: {symbol} from multiple sources.")
    print(f"Fetching data for symbol: {symbol} from multiple sources.")

    # Initialize data fetchers
    alpaca_fetcher = AlpacaDataFetcher()
    polygon_fetcher = PolygonDataFetcher()
    alpha_vantage_fetcher = AlphaVantageDataFetcher(ticker=symbol)
    yfinance_fetcher = YFinanceDataFetcher() 

    # Fetch data from Alpaca, Polygon, and Alpha Vantage
    alpaca_data = await alpaca_fetcher.fetch_and_store_data(symbol, start=start_date, end=end_date)
    
    # Ensure polygon fetcher has the correct method name or handle missing method gracefully
    try:
        polygon_data = await polygon_fetcher.fetch_data_with_date_range(symbol, start_date, end_date)
    except AttributeError as e:
        logger.error(f"Method fetch_data_with_date_range missing: {e}")
        print(f"Method fetch_data_with_date_range missing: {e}")
        polygon_data = None  # Handle missing method gracefully

    alpha_vantage_data = await alpha_vantage_fetcher.fetch_data_for_symbol(symbol, start_date, end_date)

    # Fetch data from yfinance
    yfinance_data = yfinance_fetcher.fetch_data_from_yfinance(symbol, start_date, end_date)

    # Apply technical indicators if data exists
    if alpaca_data is not None:
        apply_all_indicators(alpaca_data)
    if polygon_data is not None:
        apply_all_indicators(polygon_data)
    if alpha_vantage_data is not None:
        apply_all_indicators(alpha_vantage_data)
    if not yfinance_data.empty:
        apply_all_indicators(yfinance_data)
        # Save yfinance data to CSV and SQL database
        yfinance_fetcher.save_to_csv_and_sql(yfinance_data, symbol)

# -------------------------------------------------------------------
# Function to Inspect Database Schema and Tables
# -------------------------------------------------------------------
def inspect_schema(engine, table_name='stock_data') -> Optional[pd.DataFrame]:
    """
    Inspects the schema of the provided database table and logs the information.

    Args:
        engine: SQLAlchemy engine connected to the database.
        table_name (str): Name of the table to inspect.

    Returns:
        pd.DataFrame: DataFrame containing the schema information.
    """
    query = f"PRAGMA table_info({table_name});"
    try:
        df_schema = pd.read_sql(query, engine)
        print(f"Schema for table '{table_name}':\n{df_schema}")
        logger.info(f"Schema for table '{table_name}':\n{df_schema.to_string(index=False)}")
        return df_schema
    except Exception as e:
        logger.error(f"Error inspecting table schema for {table_name}: {e}")
        print(f"Error inspecting table schema for {table_name}: {e}")
        return None

# -------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------
def main():
    """
    Main function to inspect the database, update schema, fetch data, and migrate to target databases.
    """
    # Initialize ConfigManager
    config_manager = ConfigManager()

    # SQLite database path
    sqlite_db_path = config_manager.get('SQLITE_DB_PATH', 'C:/Projects/TradingRobotPlug/data/databases/trading_data.db')

    # Check if the SQLite database exists before proceeding
    if not os.path.exists(sqlite_db_path):
        logger.error(f"SQLite Database not found at {sqlite_db_path}")
        print(f"Database not found at {sqlite_db_path}")
        sys.exit(1)

    # Connect to the SQLite database using SQLAlchemy
    sqlite_engine = create_engine(f"sqlite:///{sqlite_db_path}")
    print("Inspecting current schema...")
    logger.info("Inspecting current schema of SQLite database.")
    inspect_table_schema(sqlite_engine)

    # Query a sample of the data
    print("\nQuerying sample data...")
    logger.info("Querying sample data from SQLite database.")
    query_sample_data(sqlite_engine, limit=10)

    # Add new columns for the technical indicators
    print("\nAdding technical indicator columns...")
    logger.info("Adding technical indicator columns to SQLite database.")
    add_technical_indicators_columns(sqlite_engine)

    # Check if the modifications were successful
    print("\nChecking modifications to the database...")
    logger.info("Checking modifications to the SQLite database.")
    check_modifications(sqlite_engine)

    # Migration options
    migration_option = input("\nChoose migration target (postgresql/mysql/none): ").strip().lower()

    if migration_option == 'postgresql':
        # PostgreSQL connection parameters from environment variables
        POSTGRES_HOST = config_manager.get('POSTGRES_HOST', 'localhost')
        POSTGRES_DBNAME = config_manager.get('POSTGRES_DBNAME', 'trading_robot_plug')
        POSTGRES_USER = config_manager.get('POSTGRES_USER', 'postgres')
        POSTGRES_PASSWORD = config_manager.get('POSTGRES_PASSWORD', 'password')
        POSTGRES_PORT = config_manager.get('POSTGRES_PORT', '5432')

        # Construct PostgreSQL connection string
        postgres_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DBNAME}"

        # Create PostgreSQL engine
        try:
            pg_engine = create_engine(postgres_url, pool_size=10, max_overflow=20)
            Base.metadata.create_all(pg_engine)  # Ensure all tables are created
            logger.info("Connected to PostgreSQL successfully.")
            print("Connected to PostgreSQL successfully.")
        except SQLAlchemyError as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            print(f"Error connecting to PostgreSQL: {e}")
            sys.exit(1)

        # Migrate StockData
        print("\nMigrating StockData to PostgreSQL...")
        logger.info("Migrating StockData to PostgreSQL.")
        migrate_stockdata_to_postgresql(sqlite_engine, pg_engine)

        # Create ModelData table in PostgreSQL
        print("\nCreating ModelData table in PostgreSQL...")
        logger.info("Creating ModelData table in PostgreSQL.")
        create_modeldata_table_postgresql(pg_engine)

        # Migrate ModelData (if any data exists)
        print("\nMigrating ModelData to PostgreSQL...")
        logger.info("Migrating ModelData to PostgreSQL.")
        migrate_modeldata_to_postgresql(sqlite_engine, pg_engine)

    elif migration_option == 'mysql':
        # MySQL migration logic can be implemented similarly
        print("MySQL migration is not implemented yet.")
        logger.info("MySQL migration option selected but not implemented.")
    else:
        print("No migration performed.")
        logger.info("Migration option selected as 'none'. No migration performed.")

    # Close the SQLite connection
    sqlite_engine.dispose()
    print("\nMigration process completed.")
    logger.info("Migration process completed.")

# -------------------------------------------------------------------
# Future Improvements
# -------------------------------------------------------------------
"""
Future Improvements:
1. **Implement MySQL Migration**:
    - Add functionality to migrate data to MySQL databases with comprehensive error handling.

2. **Automated Data Validation**:
    - Implement automated checks to validate data integrity post-migration.

3. **Enhanced Logging**:
    - Integrate more granular logging levels and handlers (e.g., file logging, remote logging services).
    - Include timestamps and context-specific information in logs for better traceability.

4. **User Interface for Selection**:
    - Develop a user interface (CLI or GUI) to allow users to select specific tables and columns for migration.

5. **Incremental Migrations**:
    - Introduce support for incremental data migrations to avoid duplicating entries and handle updates efficiently.

6. **Connection Pooling Enhancements**:
    - Implement more sophisticated connection pooling strategies to optimize performance.
    - Monitor and dynamically adjust pool sizes based on load.

7. **Error Handling Enhancements**:
    - Implement retries for transient connection errors.
    - Provide more granular error messages for different failure scenarios.

8. **Support for Additional Databases**:
    - Extend the module to support other databases like MongoDB or Oracle based on configuration.
    - Implement factory patterns to create engines dynamically based on the target database.

9. **Unit and Integration Testing**:
    - Develop comprehensive tests for each function to ensure reliability and ease of maintenance.

10. **Documentation and Code Comments**:
    - Expand docstrings and inline comments to cover all functions and complex logic.
    - Generate API documentation using tools like Sphinx for better maintainability.

11. **Secure Credential Handling**:
    - Implement secure methods for handling sensitive credentials, such as using secrets managers or encrypted storage.

12. **Asynchronous Data Fetching**:
    - Optimize data fetching by implementing asynchronous operations where applicable to improve performance.

13. **Modularize the Codebase**:
    - Split the script into smaller, reusable modules or classes to enhance readability and maintainability.

14. **Automated Deployment Scripts**:
    - Create scripts to automate the deployment and setup of databases and related services.

15. **Performance Monitoring**:
    - Integrate performance monitoring tools to track query performance and optimize as needed.
"""

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
# To test the script for runtime errors, simply run this file:
#     python db_inspect_and_update.py
# The script will guide you through inspecting the SQLite database,
# adding technical indicators, fetching data, and migrating data based on your input.

if __name__ == "__main__":
    main()
