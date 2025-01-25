# -------------------------------------------------------------------
# File Path: C:\Projects\TradingRobotPlug\src\Data_Processing\Technical_Indicators\volume_indicators.py
# Description: Provides volume indicators such as MFI, OBV, VWAP, 
#              Accumulation/Distribution Line, Chaikin Money Flow, and Volume Oscillator.
# -------------------------------------------------------------------

import sys
import pandas as pd
import logging
from pathlib import Path
from time import perf_counter as timer
import numpy as np
import pandas_ta as ta

# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------
# Dynamically adjust the Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]  # Adjust based on the project structure
utilities_dir = project_root / 'src' / 'Utilities'

# Add directories to sys.path for importing modules
sys.path.append(str(utilities_dir))

# -------------------------------------------------------------------
# Import the logging setup, config handling, and DataStore
# -------------------------------------------------------------------
try:

    from Utilities.config_manager import ConfigManager, setup_logging
    from Utilities.data.data_store import DataStore
    from Utilities.db.db_handler import DatabaseHandler
except ImportError:
    # Fallback to alternative import paths if the first import fails
    try:
        from src.Utilities.config_manager import ConfigManager, setup_logging
        from src.Utilities.data.data_store import DataStore
        from src.Utilities.db.db_handler import DatabaseHandler
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------
log_dir = project_root / 'logs' / 'technical_indicators'
log_dir.mkdir(parents=True, exist_ok=True)

# Initialize logger using the setup_logging function
logger = setup_logging(script_name="volume_indicators", log_dir=log_dir)
logger.info("Logger initialized for volume indicators.")

# -------------------------------------------------------------------
# Configuration Setup
# -------------------------------------------------------------------
dotenv_path = project_root / '.env'
required_keys = [
    'POSTGRES_HOST', 'POSTGRES_DBNAME', 'POSTGRES_USER', 
    'POSTGRES_PASSWORD', 'POSTGRES_PORT'
]

config_manager = ConfigManager(env_file=dotenv_path, required_keys=required_keys, logger=logger)

# -------------------------------------------------------------------
# VolumeIndicators Class Definition
# -------------------------------------------------------------------
class VolumeIndicators:
    def __init__(self, logger: logging.Logger = None):
        """
        Initializes the VolumeIndicators class with an optional logger.
        
        Args:
            logger (logging.Logger): Logger instance for logging. Defaults to None.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("VolumeIndicators initialized.")
    
    def add_money_flow_index(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Adds Money Flow Index (MFI) to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
            window (int): The period for MFI calculation. Defaults to 14.
        
        Returns:
            pd.DataFrame: DataFrame with 'MFI' column added.
        """
        self.logger.info("Adding Money Flow Index (MFI)")
        required_columns = ['high', 'low', 'close', 'volume']
        self._validate_columns(df, required_columns)

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0).rolling(window=window).sum()
        negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0).rolling(window=window).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1)))
        df['MFI'] = mfi.fillna(0)
        
        self.logger.info("Successfully added MFI")
        return df

    def add_on_balance_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds On-Balance Volume (OBV) to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
        
        Returns:
            pd.DataFrame: DataFrame with 'OBV' column added.
        """
        self.logger.info("Adding On-Balance Volume (OBV)")
        self._validate_columns(df, ['volume', 'close'])

        obv_change = df['volume'] * np.sign(df['close'].diff()).fillna(0)
        df['OBV'] = obv_change.cumsum()
        
        self.logger.info("Successfully added OBV")
        return df

    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds Volume Weighted Average Price (VWAP) to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
        
        Returns:
            pd.DataFrame: DataFrame with 'VWAP' column added.
        """
        self.logger.info("Adding Volume Weighted Average Price (VWAP)")
        start_time = timer()
        self._validate_columns(df, ['high', 'low', 'close', 'volume'])

        prices = (df['high'] + df['low'] + df['close']) / 3
        vwap = (prices * df['volume']).cumsum() / df['volume'].cumsum()
        df['VWAP'] = vwap
        
        execution_time = timer() - start_time
        self.logger.info(f"VWAP calculation completed in {execution_time:.2f} seconds.")
        return df

    def add_accumulation_distribution_line(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the Accumulation/Distribution Line (ADL) to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
        
        Returns:
            pd.DataFrame: DataFrame with 'ADL' column added.
        """
        self.logger.info("Adding Accumulation/Distribution Line (ADL)")
        self._validate_columns(df, ['close', 'low', 'high', 'volume'])

        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 1)
        df['ADL'] = (clv * df['volume']).cumsum()
        
        self.logger.info("Successfully added ADL")
        return df

    def add_chaikin_money_flow(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Adds Chaikin Money Flow (CMF) to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
            window (int): The period for CMF calculation. Defaults to 14.
        
        Returns:
            pd.DataFrame: DataFrame with 'CMF' column added.
        """
        self.logger.info("Adding Chaikin Money Flow (CMF)")
        self._validate_columns(df, ['high', 'low', 'close', 'volume'])

        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']).replace(0, 1)
        money_flow_volume = clv * df['volume']
        cmf = money_flow_volume.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        df['CMF'] = cmf.fillna(0)
        
        self.logger.info("Successfully added CMF")
        return df

    def add_volume_oscillator(self, df: pd.DataFrame, short_window: int = 12, long_window: int = 26) -> pd.DataFrame:
        """
        Adds Volume Oscillator to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
            short_window (int): Short period for EMA calculation.
            long_window (int): Long period for EMA calculation.
        
        Returns:
            pd.DataFrame: DataFrame with 'Volume_Oscillator' column added.
        """
        self.logger.info("Adding Volume Oscillator")
        self._validate_columns(df, ['volume'])

        short_vol_ema = df['volume'].ewm(span=short_window, adjust=False).mean()
        long_vol_ema = df['volume'].ewm(span=long_window, adjust=False).mean()
        df['Volume_Oscillator'] = short_vol_ema - long_vol_ema

        self.logger.info("Successfully added Volume Oscillator")
        return df

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required_columns: list):
        """
        Validates the presence of required columns in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
            required_columns (list): List of columns required for the calculation.
        
        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies volume-based technical indicators to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing stock data with 'close' and 'volume' columns.

        Returns:
            pd.DataFrame: DataFrame with added volume-based indicators.
        """
        self.logger.info("Applying Volume Indicators...")
        
        if 'volume' not in df.columns:
            self.logger.error("Column 'volume' not found in DataFrame.")
            return df

        # Example Volume Indicators
        try:
            # On-Balance Volume (OBV)
            df['OBV'] = ta.obv(df['close'], df['volume'])

            # Volume Moving Average (VMA)
            df['VMA_20'] = ta.sma(df['volume'], length=20)

            # Chaikin Money Flow (CMF)
            df['CMF'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)

            self.logger.info("Successfully applied Volume Indicators.")
        except Exception as e:
            self.logger.error(f"Error applying Volume Indicators: {e}")

        return df

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize DatabaseHandler and DataStore
    db_handler = DatabaseHandler(use_sqlite=False, logger=logger)
    store = DataStore(config=config_manager, logger=logger, use_csv=False)

    # Load stock data from SQL for a specific symbol (e.g., 'AAPL')
    symbol = "AAPL"
    stock_data = store.load_data(symbol=symbol)

    # Corrected Data Saving
    if stock_data is not None:
        # Ensure column names are standardized
        stock_data.columns = stock_data.columns.str.lower()
        logger.info(f"Standardized DataFrame columns: {list(stock_data.columns)}")

        # Initialize and apply indicators with logger
        indicators = VolumeIndicators(logger=logger)
        stock_data = indicators.add_money_flow_index(stock_data)
        stock_data = indicators.add_on_balance_volume(stock_data)
        stock_data = indicators.add_vwap(stock_data)
        stock_data = indicators.add_accumulation_distribution_line(stock_data)
        stock_data = indicators.add_chaikin_money_flow(stock_data)
        stock_data = indicators.add_volume_oscillator(stock_data)

        # Display the updated data
        print("Stock Data with Volume Indicators:\n", stock_data.head())

        # Save the modified data back to the SQL database using standardized column names
        store.save_data(stock_data, symbol=symbol, overwrite=True)
    else:
        logger.error(f"No data loaded from SQL for symbol '{symbol}'.")
