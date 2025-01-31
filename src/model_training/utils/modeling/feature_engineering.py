# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/Scripts/model_training/utils/modeling/feature_engineering.py
# Description: Unified feature engineering module that applies technical indicators,
#              feature engineering transformations, and saves enriched data for model training.
# -------------------------------------------------------------------

import pandas as pd
import logging
from pathlib import Path
import sys
from dotenv import load_dotenv
import asyncio

# Load Project-Specific Utilities
from Scripts.Utilities.data.data_store import DataStore
from Scripts.Utilities.db.db_handler import DatabaseHandler
from Scripts.Utilities.config_handling.config_manager import ConfigManager
from Scripts.Utilities.config_handling.logging_setup import setup_logging
from Scripts.Data_Processing.main_indicators import apply_all_indicators, IndicatorProcessor

# -------------------------------------------------------------------
# Project Path Setup and Environment Loading
# -------------------------------------------------------------------

project_root = Path("C:/Projects/TradingRobotPlug")
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print("Environment variables loaded from .env")
else:
    print("Warning: .env file not found. Ensure environment variables are set.")

# Define directories
directories = {
    'config': project_root / 'config',
    'data': project_root / 'data',
    'database': project_root / 'database',
    'logs': project_root / 'logs',
    'results': project_root / 'results',
    'scripts': project_root / 'Scripts',
}

# Add directories to sys.path for imports
for path in directories.values():
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    sys.path.append(str(path.resolve()))

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------

log_dir = directories['logs'] / 'feature_engineering'
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logging(script_name="feature_engineering", log_dir=log_dir)
logger.info("Logger initialized for feature engineering script.")

# -------------------------------------------------------------------
# Config and Database Setup
# -------------------------------------------------------------------

required_keys = [
    'POSTGRES_HOST', 'POSTGRES_DBNAME', 'POSTGRES_USER',
    'POSTGRES_PASSWORD', 'POSTGRES_PORT', 'ALPHAVANTAGE_API_KEY'
]

config_manager = ConfigManager(env_file=env_path, required_keys=required_keys, logger=logger)
data_store = DataStore(config=config_manager, logger=logger, use_csv=False)
db_handler = DatabaseHandler(logger=logger)

# -------------------------------------------------------------------
# Feature Engineering and Indicator Application Function
# -------------------------------------------------------------------

def apply_features_and_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies both technical indicators and additional feature engineering to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing stock data.

    Returns:
        pd.DataFrame: The DataFrame enriched with technical indicators and engineered features.
    """
    # Apply technical indicators
    logger.info("Applying technical indicators...")
    df = apply_all_indicators(df, logger=logger, db_handler=db_handler, config=config_manager)

    # Additional Feature Engineering (e.g., Lag Features, Moving Averages, RSI, Bollinger Bands)
    logger.info("Applying additional feature engineering transformations...")
    df['lag_1'] = df['close'].shift(1)
    df['lag_5'] = df['close'].shift(5)
    df['rolling_mean_5'] = df['close'].rolling(window=5).mean()
    df['rolling_std_5'] = df['close'].rolling(window=5).std()

    # Relative Strength Index (RSI)
    df = add_rsi(df)

    # Bollinger Bands
    df = add_bollinger_bands(df)

    # Drop any rows with NaN values resulting from shifting/rolling calculations
    df = df.dropna().reset_index(drop=True)
    logger.info("Feature engineering and indicator application completed.")
    
    return df

def add_rsi(df: pd.DataFrame, column='close', window=14) -> pd.DataFrame:
    """
    Add a Relative Strength Index (RSI) column to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): The column name on which to calculate the RSI.
        window (int): The size of the window to calculate RSI.

    Returns:
        pd.DataFrame: DataFrame with the RSI column added.
    """
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_bollinger_bands(df: pd.DataFrame, column='close', window=20) -> pd.DataFrame:
    """
    Add Bollinger Bands columns to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): The column name on which to calculate Bollinger Bands.
        window (int): The size of the window to calculate Bollinger Bands.

    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands columns added.
    """
    sma = df[column].rolling(window=window).mean()
    std = df[column].rolling(window=window).std()
    df['BollingerUpper'] = sma + (std * 2)
    df['BollingerLower'] = sma - (std * 2)
    return df

# -------------------------------------------------------------------
# Main Execution Function for Processing Symbols
# -------------------------------------------------------------------

async def process_symbols(symbols):
    """
    Process feature engineering and indicator application for each symbol.

    Args:
        symbols (list): List of stock symbols to process.
    """
    for symbol in symbols:
        logger.info(f"Processing symbol: {symbol}")
        try:
            raw_df = data_store.load_data(symbol=symbol)
            if raw_df is None or raw_df.empty:
                logger.warning(f"No data available for {symbol}. Skipping.")
                continue

            # Apply indicators and features
            processed_df = apply_features_and_indicators(raw_df)

            # Ensure 'date' column exists and is in datetime format
            if 'date' not in processed_df.columns:
                logger.warning(f"'date' column missing in processed data for {symbol}. Attempting to create it.")
                if 'Date' in processed_df.columns:
                    processed_df.rename(columns={'Date': 'date'}, inplace=True)
                else:
                    logger.error(f"Unable to create 'date' column for {symbol}. Skipping.")
                    continue

            processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
            processed_df = processed_df.dropna(subset=['date'])

            # Save processed data
            if not processed_df.empty:
                data_store.save_data(processed_df, symbol=symbol, overwrite=True)
                logger.info(f"Data for {symbol} processed and saved successfully.")
            else:
                logger.warning(f"No valid data to save for {symbol}. Skipping.")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    logger.info("Data processing completed for all symbols.")

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'GOOG']  # Define symbols to process
    try:
        asyncio.run(process_symbols(symbols))
    except Exception as e:
        logger.error(f"Runtime error: {e}", exc_info=True)
