# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/src/Data_Processing/main_indicators.py
# Description: Main file to apply all technical indicators to datasets from cleaned tables.
# -------------------------------------------------------------------

import os
import pandas as pd
import logging
import asyncio
from pathlib import Path
import sys
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Project Path Setup and Environment Loading
# -------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
env_path = project_root / '.env'

if env_path.exists():
    load_dotenv(env_path)
    print("Environment variables loaded from .env")
else:
    print("Warning: .env file not found at project root. Ensure environment variables are set.")

# Dynamically define config directory and file
# Reads from an environment variable, or defaults to "config" under project root
CONFIG_DIR = os.getenv("TRADINGBOT_CONFIG_DIR", str(project_root / "config"))
CONFIG_FILE = Path(CONFIG_DIR) / "config.ini"

# Log or warn if config file is missing
if not CONFIG_FILE.exists():
    print(f"Warning: config file does not exist at: {CONFIG_FILE}")
else:
    print(f"Config file found at: {CONFIG_FILE}")

directories = {
    'src': project_root / 'src',
    'utilities': project_root / 'src' / 'Utilities',
    'logs': project_root / 'logs',
    'data': project_root / 'data',
}

# Create directories if they don't exist
for name, path in directories.items():
    path.mkdir(parents=True, exist_ok=True)

# Add directories to sys.path
sys.path.extend([str(path.resolve()) for path in directories.values()])

print("Current sys.path entries:")
for p in sys.path:
    print(p)

# -------------------------------------------------------------------
# Importing Indicator Classes and Utilities
# -------------------------------------------------------------------
try:
    from Utilities.data.data_store import DataStore
    from Utilities.db.db_handler import DatabaseHandler
    from Utilities.config_manager import ConfigManager, setup_logging
    from Data_Processing.Technical_Indicators.momentum_indicators import MomentumIndicators
    from Data_Processing.Technical_Indicators.trend_indicators import TrendIndicators
    from Data_Processing.Technical_Indicators.volume_indicators import VolumeIndicators
    from Data_Processing.Technical_Indicators.volatility_indicators import VolatilityIndicators
    from Data_Processing.Technical_Indicators.machine_learning_indicators import MachineLearningIndicators
    from Data_Processing.Technical_Indicators.custom_indicators import CustomIndicators
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Initialize Logger
# -------------------------------------------------------------------
logger = setup_logging(
    script_name="main_indicators",
    log_dir=directories['logs'] / 'technical_indicators'
)
logger.info("Logger initialized for main_indicators.")

# -------------------------------------------------------------------
# Configuration Setup
# -------------------------------------------------------------------
required_keys = [
    'POSTGRES_HOST', 'POSTGRES_DBNAME', 'POSTGRES_USER',
    'POSTGRES_PASSWORD', 'POSTGRES_PORT', 'ALPHAVANTAGE_API_KEY',
    'ALPHAVANTAGE_BASE_URL',
    'ML_MODEL_PATH'  # Ensure this key is added to .env
]

try:
    config_manager = ConfigManager(env_file=env_path, required_keys=required_keys, logger=logger)
    logger.info("ConfigManager initialized successfully.")
except KeyError as e:
    logger.error(f"Missing required configuration keys: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Utility Function for Standardizing Column Names
# -------------------------------------------------------------------
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all column names in the DataFrame to lowercase.
    
    Args:
        df (pd.DataFrame): The original DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with standardized column names.
    """
    df.columns = [col.lower() for col in df.columns]
    return df

# -------------------------------------------------------------------
# Utility Function for Applying Indicators
# -------------------------------------------------------------------
def apply_all_indicators(df: pd.DataFrame, logger: logging.Logger, db_handler=None, config=None) -> pd.DataFrame:
    """
    Applies all available technical indicators to the provided DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing stock data.
        logger (logging.Logger): Logger instance for logging.
        db_handler: The database handler for database interactions (optional).
        config: Configuration manager instance (optional).

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    logger.info("Initializing technical indicators classes...")

    # Initialize indicator classes with the logger and optional parameters
    indicators = {
        "Momentum Indicators": MomentumIndicators(logger=logger),
        "Trend Indicators": TrendIndicators(logger=logger),
        "Volume Indicators": VolumeIndicators(logger=logger),
        "Volatility Indicators": VolatilityIndicators(logger=logger),
        "Machine Learning Indicators": MachineLearningIndicators(db_handler=db_handler, config=config, logger=logger),
        "Custom Indicators": CustomIndicators(db_handler=db_handler, config_manager=config, logger=logger),
    }

    def safe_apply(indicator_instance, df, indicator_name):
        """
        Applies indicators safely by checking if the 'apply_indicators' method exists.

        Args:
            indicator_instance: The indicator class instance.
            df (pd.DataFrame): DataFrame to apply the indicator to.
            indicator_name (str): The name of the indicator for logging.

        Returns:
            pd.DataFrame: The DataFrame with applied indicators.
        """
        if hasattr(indicator_instance, 'apply_indicators') and callable(getattr(indicator_instance, 'apply_indicators')):
            try:
                logger.info(f"Applying {indicator_name}...")
                return indicator_instance.apply_indicators(df)
            except Exception as e:
                logger.error(f"Error applying {indicator_name}: {e}", exc_info=True)
                return df
        else:
            logger.warning(f"{indicator_name} does not have an 'apply_indicators' method. Skipping...")
            return df

    # Apply each type of indicator using the safe_apply function
    for name, instance in indicators.items():
        df = safe_apply(instance, df, name)

    logger.info("All technical indicators applied successfully.")
    return df

# -------------------------------------------------------------------
# IndicatorProcessor Class
# -------------------------------------------------------------------
class IndicatorProcessor:
    def __init__(self, data_store: DataStore, db_handler: DatabaseHandler, config: ConfigManager):
        self.data_store = data_store
        self.db_handler = db_handler
        self.config = config

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all technical indicators to the provided DataFrame.
        """
        logger.info("Standardizing column names to lowercase...")
        df = standardize_column_names(df)
        logger.debug(f"DataFrame columns after standardization: {df.columns.tolist()}")

        # Verify required columns
        required_columns = ['close', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}. Skipping indicators that depend on them.")
            # Depending on your application logic, you might choose to skip processing or continue without certain indicators
            # For now, we'll proceed to apply indicators that can handle missing columns

        logger.info("Applying all technical indicators...")
        df = apply_all_indicators(df, logger=logger, db_handler=self.db_handler, config=self.config)
        logger.debug(f"DataFrame preview after applying indicators:\n{df.head()}")
        return df

# -------------------------------------------------------------------
# Async Main Function
# -------------------------------------------------------------------
async def main():
    symbols = ['AAPL', 'MSFT', 'GOOG']
    try:
        data_store = DataStore(config=config_manager, logger=logger, use_csv=False)
        db_handler = DatabaseHandler(logger=logger)
        indicator_processor = IndicatorProcessor(data_store=data_store, db_handler=db_handler, config=config_manager)
    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        sys.exit(1)

    # Process data for each symbol
    for symbol in symbols:
        logger.info(f"Processing indicators for {symbol}")
        try:
            raw_df = data_store.load_data(symbol=symbol)
            if raw_df is None or raw_df.empty:
                logger.warning(f"No data for {symbol}. Skipping.")
                continue

            processed_df = indicator_processor.process_data(raw_df)

            # Ensure 'date' column exists and is in datetime format
            if 'date' not in processed_df.columns:
                logger.warning(f"'date' column missing in processed data for {symbol}. Attempting to create it.")
                if 'date' in processed_df.columns:
                    pass  # Already exists
                elif 'date' in processed_df.columns:
                    processed_df.rename(columns={'date': 'date'}, inplace=True)
                elif 'Date' in processed_df.columns:
                    processed_df.rename(columns={'Date': 'date'}, inplace=True)
                else:
                    logger.error(f"Unable to create 'date' column for {symbol}. Skipping.")
                    continue

            processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
            processed_df = processed_df.dropna(subset=['date'])

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
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Runtime error: {e}", exc_info=True)
