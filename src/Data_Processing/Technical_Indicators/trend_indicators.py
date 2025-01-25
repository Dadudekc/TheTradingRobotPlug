# -------------------------------------------------------------------
# File Path: D:/TradingRobotPlug2/src/Data_Processing/Technical_Indicators/trend_indicators.py
# Description: Provides trend indicators such as Moving Averages, MACD, ADX, 
# Ichimoku Cloud, and Parabolic SAR. This script integrates with the project’s 
# ConfigManager, DatabaseHandler, and logging setup.
# -------------------------------------------------------------------

import os
import sys
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Optional
from ta.trend import MACD, ADXIndicator, IchimokuIndicator, PSARIndicator
import logging  # Added import for logging

# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------

# Dynamically set the project root based on the current file's location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]  # Adjust based on the project structure
utilities_dir = project_root / 'src' / 'Utilities'
model_dir = project_root / 'SavedModels'
model_utils = project_root / 'src' / 'model_training' / 'utils'
data_processing_dir = project_root / 'src' / 'Data_Processing'

# Add directories to sys.path for importing modules
sys.path.extend([
    str(utilities_dir.resolve()), 
    str(model_dir.resolve()), 
    str(model_utils.resolve()), 
    str(data_processing_dir.resolve())
])

# -------------------------------------------------------------------
# Importing ConfigManager, Logging Setup, and Database Session
# -------------------------------------------------------------------
try:
    from Utilities.config_manager import ConfigManager, setup_logging
    from Utilities.db.db_connection import Session
    from Utilities.db.db_handler import DatabaseHandler  # Ensure this import is correct
    from Utilities.data.data_store import DataStore  # Import DataStore correctly
except ImportError:
    # Fallback to alternative path if the first import fails
    try:
        from src.Utilities.config_manager import ConfigManager, setup_logging
        from src.Utilities.db.db_connection import Session
        from src.Utilities.db.db_handler import DatabaseHandler
        from src.Utilities.data.data_store import DataStore
    except ImportError as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------
# Initialize logger using a consistent setup method
logger = setup_logging(script_name="trend_indicators", log_dir=project_root / 'logs' / 'technical_indicators')
logger.info("Logger setup successfully for Trend Indicators")

# -------------------------------------------------------------------
# Configuration Setup
# -------------------------------------------------------------------
dotenv_path = project_root / '.env'

# Initialize ConfigManager with required keys
required_keys = [
    'POSTGRES_HOST',
    'POSTGRES_DBNAME',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_PORT',
    'ALPHAVANTAGE_API_KEY',
    'ALPHAVANTAGE_BASE_URL'
    # Add other required keys as needed
]

try:
    config_manager = ConfigManager(env_file=dotenv_path, required_keys=required_keys, logger=logger)
    logger.info("ConfigManager initialized successfully.")
except KeyError as e:
    logger.error(f"Missing required configuration keys: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Email Alert Function
# -------------------------------------------------------------------
import smtplib
from email.mime.text import MIMEText

def send_email_alert(subject: str, body: str, to_addresses: List[str], logger: logging.Logger):
    """
    Sends an email alert.
    
    Args:
        subject (str): Subject of the email.
        body (str): Body of the email.
        to_addresses (List[str]): List of recipient email addresses.
        logger (logging.Logger): Logger instance for logging.
    """
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'your_email@example.com'
    msg['To'] = ', '.join(to_addresses)
    
    try:
        with smtplib.SMTP('smtp.example.com', 587) as server:
            server.starttls()
            server.login('your_email@example.com', 'your_password')
            server.sendmail('your_email@example.com', to_addresses, msg.as_string())
        logger.info("Sent email alert successfully.")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}", exc_info=True)

# -------------------------------------------------------------------
# Indicator Base Class
# -------------------------------------------------------------------
class Indicator(ABC):
    """
    Abstract base class for all indicators.
    """
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the indicator to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
        
        Returns:
            pd.DataFrame: Updated DataFrame with the indicator applied.
        """
        pass

# -------------------------------------------------------------------
# Specific Indicator Classes
# -------------------------------------------------------------------
class MovingAverageIndicator(Indicator):
    def __init__(self, window_size: int = 10, ma_type: str = "SMA", handle_nans: str = "warn", column: str = "close", logger: logging.Logger = None):
        """
        Initializes the MovingAverageIndicator with specified parameters.
        
        Args:
            window_size (int): The window size for the moving average.
            ma_type (str): Type of moving average ('SMA' or 'EMA').
            handle_nans (str): Strategy to handle NaN values ('drop', 'ffill', 'warn').
            column (str): Column name on which to apply the moving average.
            logger (logging.Logger): Logger instance for logging.
        """
        self.window_size = window_size
        self.ma_type = ma_type.upper()
        self.handle_nans = handle_nans
        self.column = column
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("MovingAverageIndicator initialized.")
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the Moving Average indicator to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
        
        Returns:
            pd.DataFrame: Updated DataFrame with the moving average added.
        """
        self.logger.info(f"Adding {self.ma_type} Moving Average with window size {self.window_size}")
        self.logger.debug(f"Parameters - window_size: {self.window_size}, ma_type: {self.ma_type}, handle_nans: {self.handle_nans}, column: {self.column}")

        if self.column not in df.columns:
            self.logger.error(f"Column '{self.column}' not found in DataFrame for Moving Average")
            raise ValueError(f"Missing '{self.column}' column for Moving Average")

        if self.ma_type == "SMA":
            ma_column = f"sma_{self.window_size}"
            df[ma_column] = df[self.column].rolling(window=self.window_size, min_periods=1).mean()
        elif self.ma_type == "EMA":
            ma_column = f"ema_{self.window_size}"
            df[ma_column] = df[self.column].ewm(span=self.window_size, adjust=False).mean()
        else:
            self.logger.error(f"Moving average type '{self.ma_type}' is not supported.")
            raise ValueError(f"Unsupported moving average type '{self.ma_type}'")

        self.logger.info(f"Successfully added {self.ma_type} Moving Average as '{ma_column}'")

        # Handle NaN values
        nan_count = df[ma_column].isna().sum()
        if nan_count > 0:
            self.logger.warning(f"'{ma_column}' contains {nan_count} NaN values after computation.")
            if self.handle_nans == 'drop':
                df.dropna(subset=[ma_column], inplace=True)
                self.logger.info(f"Dropped rows with NaN values in '{ma_column}'.")
            elif self.handle_nans == 'ffill':
                df[ma_column] = df[ma_column].ffill()
                filled_nans = df[ma_column].isna().sum()
                self.logger.info(f"Filled NaN values in '{ma_column}' using forward fill. Remaining NaNs: {filled_nans}")
            elif self.handle_nans == 'warn':
                self.logger.warning(f"'{ma_column}' still contains NaN values after computation.")
            else:
                self.logger.warning(f"Unknown NaN handling strategy '{self.handle_nans}'. No action taken.")

        return df

class MACDIndicatorClass(Indicator):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, price_column: str = "close", logger: logging.Logger = None):
        """
        Initializes the MACDIndicatorClass with specified parameters.
        
        Args:
            fast_period (int): Fast EMA period.
            slow_period (int): Slow EMA period.
            signal_period (int): Signal line period.
            price_column (str): Column name to calculate MACD on.
            logger (logging.Logger): Logger instance for logging.
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_column = price_column
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("MACDIndicatorClass initialized.")
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the MACD indicator to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
        
        Returns:
            pd.DataFrame: Updated DataFrame with MACD components added.
        """
        self.logger.info("Calculating MACD components")
        self.logger.debug(f"Parameters - fast_period: {self.fast_period}, slow_period: {self.slow_period}, signal_period: {self.signal_period}, price_column: {self.price_column}")

        if self.price_column not in df.columns:
            self.logger.error(f"Column '{self.price_column}' not found in DataFrame for MACD")
            raise ValueError(f"Missing '{self.price_column}' column for MACD")

        macd = MACD(close=df[self.price_column], window_slow=self.slow_period, window_fast=self.fast_period, window_sign=self.signal_period)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        self.logger.info("Successfully calculated MACD components")

        return df

class ADXIndicatorClass(Indicator):
    def __init__(self, window: int = 14, handle_nans: str = "warn", logger: logging.Logger = None):
        """
        Initializes the ADXIndicatorClass with specified parameters.
        
        Args:
            window (int): Window size for ADX calculation.
            handle_nans (str): Strategy to handle NaN values ('drop', 'ffill', 'warn').
            logger (logging.Logger): Logger instance for logging.
        """
        self.window = window
        self.handle_nans = handle_nans
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("ADXIndicatorClass initialized.")
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the ADX indicator to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
        
        Returns:
            pd.DataFrame: Updated DataFrame with ADX components added.
        """
        self.logger.info(f"Adding ADX with window {self.window}")
        self.logger.debug(f"Parameters - window: {self.window}, handle_nans: {self.handle_nans}")

        required_columns = ["high", "low", "close"]
        if not all(column in df.columns for column in required_columns):
            self.logger.error(f"DataFrame must contain {required_columns} columns for ADX")
            raise ValueError(f"Missing columns for ADX: {', '.join(required_columns)}")

        adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=self.window)
        df["adx"] = adx.adx()
        df["+di"] = adx.adx_pos()
        df["-di"] = adx.adx_neg()

        self.logger.info("Successfully added ADX components")

        # Handle NaN values
        adx_columns = ["adx", "+di", "-di"]
        for column in adx_columns:
            nan_count = df[column].isna().sum()
            if nan_count > 0:
                self.logger.warning(f"'{column}' contains {nan_count} NaN values after computation.")
                if self.handle_nans == 'drop':
                    df.dropna(subset=[column], inplace=True)
                    self.logger.info(f"Dropped rows with NaN values in '{column}'.")
                elif self.handle_nans == 'ffill':
                    df[column] = df[column].ffill()
                    filled_nans = df[column].isna().sum()
                    self.logger.info(f"Filled NaN values in '{column}' using forward fill. Remaining NaNs: {filled_nans}")
                elif self.handle_nans == 'warn':
                    self.logger.warning(f"'{column}' still contains NaN values after computation.")
                else:
                    self.logger.warning(f"Unknown NaN handling strategy '{self.handle_nans}'. No action taken.")

        return df

class IchimokuCloudIndicatorClass(Indicator):
    def __init__(self, nine_period: int = 9, twenty_six_period: int = 26, fifty_two_period: int = 52, handle_nans: str = "warn", logger: logging.Logger = None):
        """
        Initializes the IchimokuCloudIndicatorClass with specified parameters.
        
        Args:
            nine_period (int): Conversion line period.
            twenty_six_period (int): Base line period.
            fifty_two_period (int): Leading span B period.
            handle_nans (str): Strategy to handle NaN values ('drop', 'ffill', 'warn').
            logger (logging.Logger): Logger instance for logging.
        """
        self.nine_period = nine_period
        self.twenty_six_period = twenty_six_period
        self.fifty_two_period = fifty_two_period
        self.handle_nans = handle_nans
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("IchimokuCloudIndicatorClass initialized.")
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the Ichimoku Cloud indicator to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
        
        Returns:
            pd.DataFrame: Updated DataFrame with Ichimoku Cloud components added.
        """
        self.logger.info("Adding Ichimoku Cloud components.")
        self.logger.debug(f"Parameters - nine_period: {self.nine_period}, twenty_six_period: {self.twenty_six_period}, fifty_two_period: {self.fifty_two_period}, handle_nans: {self.handle_nans}")

        required_columns = ["high", "low", "close"]
        if not all(column in df.columns for column in required_columns):
            self.logger.error(f"DataFrame must contain {required_columns} columns for Ichimoku Cloud")
            raise ValueError(f"Missing columns for Ichimoku Cloud: {', '.join(required_columns)}")

        ichimoku = IchimokuIndicator(high=df["high"], low=df["low"],
                                     window1=self.nine_period, window2=self.twenty_six_period, window3=self.fifty_two_period)
        df["ichimoku_conversion_line"] = ichimoku.ichimoku_conversion_line()
        df["ichimoku_base_line"] = ichimoku.ichimoku_base_line()
        df["ichimoku_leading_span_a"] = ichimoku.ichimoku_a()
        df["ichimoku_leading_span_b"] = ichimoku.ichimoku_b()
        df["ichimoku_lagging_span"] = df["close"].shift(-self.twenty_six_period)

        self.logger.info("Successfully added Ichimoku Cloud components")

        # Handle NaN values
        ichimoku_columns = [
            "ichimoku_conversion_line", 
            "ichimoku_base_line", 
            "ichimoku_leading_span_a", 
            "ichimoku_leading_span_b", 
            "ichimoku_lagging_span"
        ]
        for column in ichimoku_columns:
            nan_count = df[column].isna().sum()
            if nan_count > 0:
                self.logger.warning(f"'{column}' contains {nan_count} NaN values after computation.")
                if self.handle_nans == 'drop':
                    df.dropna(subset=[column], inplace=True)
                    self.logger.info(f"Dropped rows with NaN values in '{column}'.")
                elif self.handle_nans == 'ffill':
                    df[column] = df[column].ffill()
                    filled_nans = df[column].isna().sum()
                    self.logger.info(f"Filled NaN values in '{column}' using forward fill. Remaining NaNs: {filled_nans}")
                elif self.handle_nans == 'warn':
                    self.logger.warning(f"'{column}' still contains NaN values after computation.")
                else:
                    self.logger.warning(f"Unknown NaN handling strategy '{self.handle_nans}'. No action taken.")

        return df

class PSARIndicatorClass(Indicator):
    def __init__(self, step: float = 0.02, max_step: float = 0.2, handle_nans: str = "warn", logger: logging.Logger = None):
        """
        Initializes the PSARIndicatorClass with specified parameters.
        
        Args:
            step (float): Step value for Parabolic SAR.
            max_step (float): Maximum step value for Parabolic SAR.
            handle_nans (str): Strategy to handle NaN values ('drop', 'ffill', 'warn').
            logger (logging.Logger): Logger instance for logging.
        """
        self.step = step
        self.max_step = max_step
        self.handle_nans = handle_nans
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("PSARIndicatorClass initialized.")
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the Parabolic SAR indicator to the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing stock data.
        
        Returns:
            pd.DataFrame: Updated DataFrame with Parabolic SAR added.
        """
        self.logger.info("Adding Parabolic SAR")
        self.logger.debug(f"Parameters - step: {self.step}, max_step: {self.max_step}, handle_nans: {self.handle_nans}")

        required_columns = ["high", "low", "close"]
        if not all(column in df.columns for column in required_columns):
            self.logger.error(f"DataFrame must contain {required_columns} columns for Parabolic SAR")
            raise ValueError(f"Missing columns for Parabolic SAR: {', '.join(required_columns)}")

        psar = PSARIndicator(high=df["high"], low=df["low"], close=df["close"], step=self.step, max_step=self.max_step)
        df["psar"] = psar.psar()

        self.logger.info("Successfully added Parabolic SAR")

        # Handle NaN values
        psar_column = "psar"
        nan_count = df[psar_column].isna().sum()
        if nan_count > 0:
            self.logger.warning(f"'{psar_column}' contains {nan_count} NaN values after computation.")
            if self.handle_nans == 'drop':
                df.dropna(subset=[psar_column], inplace=True)
                self.logger.info(f"Dropped rows with NaN values in '{psar_column}'.")
            elif self.handle_nans == 'ffill':
                df[psar_column] = df[psar_column].ffill()
                filled_nans = df[psar_column].isna().sum()
                self.logger.info(f"Filled NaN values in '{psar_column}' using forward fill. Remaining NaNs: {filled_nans}")
            elif self.handle_nans == 'warn':
                self.logger.warning(f"'{psar_column}' still contains NaN values after computation.")
            else:
                self.logger.warning(f"Unknown NaN handling strategy '{self.handle_nans}'. No action taken.")

        return df

# -------------------------------------------------------------------
# Indicator Pipeline
# -------------------------------------------------------------------
class IndicatorPipeline:
    """
    Manages a sequence of indicators to apply to a DataFrame.
    """
    def __init__(self, indicators: Optional[List[Indicator]] = None):
        self.indicators = indicators or []
    
    def add_indicator(self, indicator: Indicator):
        self.indicators.append(indicator)
        logger.info(f"Added indicator: {indicator.__class__.__name__}")
    
    def remove_indicator(self, indicator_cls):
        before_count = len(self.indicators)
        self.indicators = [ind for ind in self.indicators if not isinstance(ind, indicator_cls)]
        after_count = len(self.indicators)
        logger.info(f"Removed indicator: {indicator_cls.__name__}. Indicators count: {before_count} -> {after_count}")
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for indicator in self.indicators:
            df = indicator.apply(df)
        return df

# -------------------------------------------------------------------
# TrendIndicators Class Definition
# -------------------------------------------------------------------
class TrendIndicators:
    """
    Encapsulates all trend indicators and provides a composable interface to apply them.
    """
    def __init__(self, pipeline: Optional[IndicatorPipeline] = None, logger: logging.Logger = None):
        """
        Initializes the TrendIndicators class with an optional pipeline and logger.
        
        Args:
            pipeline (Optional[IndicatorPipeline]): An existing indicator pipeline.
            logger (logging.Logger): Logger instance for logging.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.pipeline = pipeline or IndicatorPipeline()
        self.logger.info("TrendIndicators initialized.")

    def add_indicator(self, indicator: Indicator):
        self.pipeline.add_indicator(indicator)
        self.logger.info(f"Added indicator: {indicator.__class__.__name__}")

    def remove_indicator(self, indicator_cls):
        self.pipeline.remove_indicator(indicator_cls)
        self.logger.info(f"Removed indicator: {indicator_cls.__name__}")

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting to apply trend indicators pipeline")
        df = self.pipeline.apply(df)
        self.logger.info("Completed applying trend indicators pipeline")
        return df

# -------------------------------------------------------------------
# Example Usage of TrendIndicators with Real Data from DataStore
# -------------------------------------------------------------------

def main():
    # Initialize Database Handler
    db_handler = DatabaseHandler(logger=logger)  # Initialize your DatabaseHandler appropriately
    logger.info("Database handler initialized.")

    try:
        # Initialize DataStore with the required arguments directly
        data_store = DataStore(config=config_manager, logger=logger, use_csv=False)
        logger.info("DataStore initialized with config and logger.")

        # Load data for a specific symbol
        symbol = 'AAPL'
        df = data_store.load_data(symbol)

        if df is None or df.empty:
            logger.error(f"No data found for symbol: {symbol}")
            return
        else:
            logger.info(f"Data for symbol {symbol} loaded successfully.")

        # Preprocess data (handle 'date' column and standardize)
        df = data_store.preprocess_data(df)

        # Optionally, attempt to correct date formats if original strings are available
        # df = data_store.correct_date_formats(df)

        # Validate dates before import (placeholder for any additional checks)
        df = data_store.validate_dates_before_import(df, symbol)

        # Initialize TrendIndicators with the logger
        trend_indicators = TrendIndicators(logger=logger)

        # Add Moving Average (SMA) with forward fill
        trend_indicators.add_indicator(MovingAverageIndicator(window_size=10, ma_type="SMA", handle_nans='ffill', logger=logger))
        # Add Moving Average (EMA) with forward fill
        trend_indicators.add_indicator(MovingAverageIndicator(window_size=20, ma_type="EMA", handle_nans='ffill', logger=logger))

        # Calculate MACD components
        trend_indicators.add_indicator(MACDIndicatorClass(fast_period=12, slow_period=26, signal_period=9, price_column="close", logger=logger))

        # Add ADX with forward fill
        trend_indicators.add_indicator(ADXIndicatorClass(window=14, handle_nans='ffill', logger=logger))

        # Add Ichimoku Cloud components with forward fill
        trend_indicators.add_indicator(IchimokuCloudIndicatorClass(nine_period=9, twenty_six_period=26, fifty_two_period=52, handle_nans='ffill', logger=logger))

        # Add Parabolic SAR with forward fill
        trend_indicators.add_indicator(PSARIndicatorClass(step=0.02, max_step=0.2, handle_nans='ffill', logger=logger))

        # Apply all indicators
        df = trend_indicators.apply_indicators(df)

        # Reset index to include 'date' as a column for saving
        df.reset_index(inplace=True)

        # Rename 'date' back to 'Date' if DataStore expects 'Date' with uppercase 'D'
        if 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)
            logger.info("Renamed 'date' column back to 'Date' for saving.")

        # Save data back to the database
        data_store.save_data(df, symbol=symbol, overwrite=True)
        logger.info(f"Processed data saved successfully for {symbol}.")

        # Display the first 15 rows of applied indicators for verification
        print("Moving Average (SMA):\n", df[["sma_10"]].head(15))
        print("\nMoving Average (EMA):\n", df[["ema_20"]].head(15))
        print("\nMACD:\n", df[["macd", "macd_signal", "macd_hist"]].head(15))
        print("\nADX:\n", df[["+di", "-di", "adx"]].head(15))
        print("\nIchimoku Cloud:\n", df[[
            "ichimoku_conversion_line", 
            "ichimoku_base_line", 
            "ichimoku_leading_span_a", 
            "ichimoku_leading_span_b", 
            "ichimoku_lagging_span"
        ]].head(15))
        print("\nParabolic SAR:\n", df[["psar"]].head(15))

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        db_handler.close()
        logger.info("Database handler closed.")

if __name__ == "__main__":
    main()
