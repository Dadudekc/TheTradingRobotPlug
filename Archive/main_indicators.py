# -------------------------------------------------------------------
# File: main_indicators.py
# Location: D:\TradingRobotPlug2\src\Data_Processing\main_indicators.py
# Description: Composable file to apply all technical indicators from
#              multiple modules to datasets from cleaned tables.
#              Extended to handle streaming data with a sliding window.
# -------------------------------------------------------------------

import os
import sys
import asyncio
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
import pandas_ta as ta

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

# Reads from an environment variable, or defaults to "config" under project root
CONFIG_DIR = os.getenv("TRADINGBOT_CONFIG_DIR", str(project_root / "config"))
CONFIG_FILE = Path(CONFIG_DIR) / "config.ini"

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

# Ensure required directories exist
for name, path in directories.items():
    path.mkdir(parents=True, exist_ok=True)

# Add directories to sys.path
sys.path.extend([str(path.resolve()) for path in directories.values()])

print("Current sys.path entries:")
for p in sys.path:
    print(p)

# -------------------------------------------------------------------
# Attempting Imports of Indicators, Utilities, ColumnUtils
# -------------------------------------------------------------------
try:
    from Utilities.data.data_store import DataStore
    from Utilities.db.db_handler import DatabaseHandler
    from Utilities.config_manager import ConfigManager, setup_logging
    from Utilities.column_utils import ColumnUtils

    from Data_Processing.Technical_Indicators.momentum_indicators import MomentumIndicators
    from Data_Processing.Technical_Indicators.trend_indicators import TrendIndicators
    from Data_Processing.Technical_Indicators.volume_indicators import VolumeIndicators
    from Data_Processing.Technical_Indicators.volatility_indicators import VolatilityIndicators
    from Data_Processing.Technical_Indicators.machine_learning_indicators import MachineLearningIndicators
    from Data_Processing.Technical_Indicators.custom_indicators import CustomIndicators

    print("[main_indicators.py] Successfully imported all modules.")
except ImportError as e:
    print(f"[main_indicators.py] Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Initialize Logger
# -------------------------------------------------------------------
logger = setup_logging(
    script_name="main_indicators",
    log_dir=directories['logs'] / 'technical_indicators'
)
logger.info("[main_indicators.py] Logger initialized for main_indicators.")

# -------------------------------------------------------------------
# Configuration Setup
# -------------------------------------------------------------------
required_keys = [
    'POSTGRES_HOST', 'POSTGRES_DBNAME', 'POSTGRES_USER',
    'POSTGRES_PASSWORD', 'POSTGRES_PORT', 'ALPHAVANTAGE_API_KEY',
    'ALPHAVANTAGE_BASE_URL',
    'ML_MODEL_PATH'  # Ensure this key is present in .env or config
]

try:
    config_manager = ConfigManager(env_file=env_path, required_keys=required_keys, logger=logger)
    logger.info("[main_indicators.py] ConfigManager initialized successfully.")
except KeyError as e:
    logger.error(f"[main_indicators.py] Missing required configuration keys: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Composable Function to Apply All Indicators
# -------------------------------------------------------------------
def apply_all_indicators(
    df: pd.DataFrame,
    logger: logging.Logger,
    db_handler: Optional[DatabaseHandler],
    config: ConfigManager,
    data_store: DataStore
) -> pd.DataFrame:
    """
    Applies all available technical indicators from various modules
    (Momentum, Trend, Volume, Volatility, Machine Learning, and Custom)
    to the provided DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing stock data.
        logger (logging.Logger): Logger instance for logging.
        db_handler (Optional[DatabaseHandler]): Database handler.
        config (ConfigManager): Configuration manager.
        data_store (DataStore): Data store for data operations.

    Returns:
        pd.DataFrame: DataFrame with applied indicators.
    """
    logger.info("[main_indicators.py] Initializing technical indicator classes...")

    # Initialize indicator classes with logger, db_handler, and config
    indicators_map = {
        "Momentum Indicators": MomentumIndicators(logger=logger, data_store=data_store),
        "Trend Indicators": TrendIndicators(logger=logger, data_store=data_store),
        "Volume Indicators": VolumeIndicators(logger=logger),
        "Volatility Indicators": VolatilityIndicators(logger=logger),
        "Machine Learning Indicators": MachineLearningIndicators(
            data_store=data_store, 
            db_handler=db_handler, 
            config=config, 
            logger=logger
        ),
        "Custom Indicators": CustomIndicators(
            db_handler=db_handler, 
            logger=logger
        ),
    }

    def safe_apply(instance, df_in, name):
        """
        Safely applies an indicator's 'apply_indicators' method if available.
        """
        if hasattr(instance, 'apply_indicators') and callable(getattr(instance, 'apply_indicators')):
            try:
                logger.info(f"[main_indicators.py] Applying {name}...")
                return instance.apply_indicators(df_in)
            except Exception as exc:
                logger.error(f"[main_indicators.py] Error applying {name}: {exc}", exc_info=True)
                return df_in
        else:
            logger.warning(f"[main_indicators.py] {name} has no 'apply_indicators' method. Skipping...")
            return df_in

    # Sequentially apply each set of indicators
    for name, instance in indicators_map.items():
        df = safe_apply(instance, df, name)

    logger.info("[main_indicators.py] All technical indicators applied successfully.")
    return df

# -------------------------------------------------------------------
# IndicatorProcessor Class
# -------------------------------------------------------------------
class IndicatorProcessor:
    """
    A composable class to unify all indicator applications.
    """
    def __init__(self, data_store: DataStore, db_handler: DatabaseHandler, config: ConfigManager):
        self.data_store = data_store
        self.db_handler = db_handler
        self.config = config
        self.logger = logger

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes columns with ColumnUtils, then applies all indicators.
        """
        self.logger.info("[main_indicators.py] Standardizing columns with ColumnUtils...")
        column_config_path = project_root / 'src' / 'Utilities' / 'column_config.json'
        try:
            df = ColumnUtils.process_dataframe(
                df=df,
                config_path=column_config_path,
                required_columns=self._required_columns(),
                logger=self.logger
            )
            self.logger.info("[main_indicators.py] DataFrame processed with ColumnUtils.")
        except Exception as e:
            self.logger.error(f"[main_indicators.py] Column standardization error: {e}", exc_info=True)
            # Depending on the use-case, you might skip further processing or return the df as is
            return df

        # Ensure 'Date' column exists
        if 'date' in df.columns and 'Date' not in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)
            self.logger.info("Renamed 'date' column to 'Date'")
        elif 'Date' not in df.columns and 'date' not in df.columns:
            self.logger.error("Neither 'date' nor 'Date' column exists in DataFrame.")
            raise KeyError("Date column missing")

        # Check for minimal columns needed
        essential_cols = ["close", "high", "low", "volume", "Date"]
        missing_cols = [col for col in essential_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"[main_indicators.py] Missing essential columns {missing_cols}. Some indicators may fail.")

        self.logger.info("[main_indicators.py] Applying all technical indicators...")
        df = apply_all_indicators(
            df,
            logger=self.logger,
            db_handler=self.db_handler,
            config=self.config,
            data_store=self.data_store
        )
        return df

    def _required_columns(self) -> list:
        """
        Returns a set of columns likely needed across all indicators.
        """
        return [
            "close", "high", "low", "volume",
            "macd_line", "macd_signal", "macd_histogram",
            "rsi", "bollinger_width", "Date"
        ]

# -------------------------------------------------------------------
# Async Main Function
# -------------------------------------------------------------------
async def main():
    symbols = ["AAPL", "MSFT", "GOOG"]
    try:
        data_store = DataStore(config=config_manager, logger=logger, use_csv=False)
        db_handler = DatabaseHandler(logger=logger)
        processor = IndicatorProcessor(data_store, db_handler, config_manager)
    except Exception as e:
        logger.error(f"[main_indicators.py] Initialization error: {e}", exc_info=True)
        sys.exit(1)

    # Process data for each symbol
    for symbol in symbols:
        logger.info(f"[main_indicators.py] Processing indicators for {symbol}")
        try:
            raw_df = data_store.load_data(symbol=symbol)
            if raw_df is None or raw_df.empty:
                logger.warning(f"[main_indicators.py] No data for {symbol}. Skipping.")
                continue

            processed_df = processor.process_dataframe(raw_df)

            # Validate 'Date' column
            if "Date" not in processed_df.columns:
                logger.warning(f"[main_indicators.py] 'Date' column missing in processed data for {symbol}. Attempting to create it.")
                # Attempt to create 'Date' from index if possible
                if processed_df.index.name == 'Date':
                    processed_df.reset_index(inplace=True)
                else:
                    # If 'Date' can be inferred from another column, handle accordingly
                    if 'date' in processed_df.columns:
                        processed_df['Date'] = pd.to_datetime(processed_df['date'], errors='coerce')
                        processed_df.drop(columns=['date'], inplace=True)
                        logger.info("Created 'Date' column from 'date' column.")
                    else:
                        logger.error("Unable to create 'Date' column. Skipping this symbol.")
                        continue

            # Ensure 'Date' is datetime
            processed_df["Date"] = pd.to_datetime(processed_df["Date"], errors="coerce")
            processed_df.dropna(subset=["Date"], inplace=True)

            if processed_df.empty or processed_df.isna().all().all():
                logger.warning(f"[main_indicators.py] No valid data to save for {symbol}. Skipping.")
            else:
                data_store.save_data(processed_df, symbol=symbol, overwrite=True)
                logger.info(f"[main_indicators.py] Data for {symbol} processed and saved successfully.")
        except Exception as ex:
            logger.error(f"[main_indicators.py] Error processing {symbol}: {ex}", exc_info=True)

    logger.info("[main_indicators.py] Data processing completed for all symbols.")

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"[main_indicators.py] Runtime error: {e}", exc_info=True)

def calculate_indicators(df):
    # Ensure 'close' column exists and has enough data
    if 'close' not in df.columns or len(df['close']) < 26:
        print("Error: 'close' column is missing or not enough data to calculate MACD.")
        return df

    # Fill or drop missing values in 'close' column
    df['close'] = df['close'].ffill().bfill()  # Forward and backward fill

    # Debugging: Check the 'close' column after filling
    print("Close column after filling missing values:")
    print(df['close'].head(30))  # Print first 30 values for inspection

    # Calculate MACD
    macd = ta.macd(df['close'])
    if macd is not None:
        macd = macd.dropna()  # Drop NaN values before assigning columns
        df['macd_line'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_histogram'] = macd['MACDh_12_26_9']
    else:
        print("Error: MACD calculation returned None.")

    # Calculate RSI
    df['rsi'] = ta.rsi(df['close'], length=14)

    # Calculate Bollinger Bands Width
    bb = ta.bbands(df['close'], length=20, std=2)
    if bb is not None:
        df['bollinger_width'] = bb['BBL_20_2.0']
    else:
        print("Error: Bollinger Bands calculation returned None.")

    return df

# Example usage
# Ensure the DataFrame has at least 26 data points
df = pd.DataFrame({
    'close': [100, 102, 101, 105, 107, 110, 108, 107, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
})

df = calculate_indicators(df)
print(df)
