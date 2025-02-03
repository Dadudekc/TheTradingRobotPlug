# -------------------------------------------------------------------
# File: volatility_indicators.py
# Location: src/Data_Processing/Technical_Indicators
# Description: Provides volatility indicators using pandas-ta instead
#              of TA-Lib, including Bollinger Bands, Standard Deviation,
#              Historical Volatility, Chandelier Exit, Keltner Channel,
#              and Moving Average Envelope.
# 
# Enhancements:
#  - Logging references the script name for clarity.
#  - Data type downcasting for memory efficiency.
#  - Chunked processing for large datasets with multiprocessing.
#  - Potential streaming data approach with a sliding window.
#  - Integration with ColumnUtils for column standardization.
# -------------------------------------------------------------------

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from time import perf_counter as timer
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import deque

# For indicators, we use pandas-ta
import pandas_ta as ta

# -------------------------------------------------------------------
# Identify Current Script for Logging
# -------------------------------------------------------------------
script_file = Path(__file__).resolve()
script_name = script_file.name  # e.g., "volatility_indicators.py"
project_root = script_file.parents[4]

# Ensure the needed directories are on sys.path
utilities_dir = project_root / "src" / "Utilities"
if str(utilities_dir.resolve()) not in sys.path:
    sys.path.append(str(utilities_dir.resolve()))

# -------------------------------------------------------------------
# Import Utility Scripts
# -------------------------------------------------------------------
try:
    from Utilities.config_manager import ConfigManager, setup_logging
    from Utilities.data.data_store import DataStore
    from Utilities.db.db_handler import DBHandler
    from Utilities.column_utils import ColumnUtils

    print(f"[{script_name}] Imported config_manager, db_handler, data_store, column_utils successfully.")
except ImportError as e:
    print(f"[{script_name}] Error importing utility modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Configuration and Logger Setup
# -------------------------------------------------------------------
dotenv_path = project_root / '.env'
required_keys = [
    'POSTGRES_HOST', 'POSTGRES_DBNAME', 'POSTGRES_USER',
    'POSTGRES_PASSWORD', 'POSTGRES_PORT'
]
try:
    config_manager = ConfigManager(env_file=dotenv_path, required_keys=required_keys, logger=None)
except KeyError as e:
    print(f"[{script_name}] Missing required configuration keys: {e}")
    sys.exit(1)

log_dir = project_root / 'logs' / 'technical_indicators'
log_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logging(
    script_name=script_name,
    log_dir=log_dir,
    max_log_size=5 * 1024 * 1024,  # 5 MB
    backup_count=3,
    console_log_level=logging.INFO,
    file_log_level=logging.DEBUG,
    feedback_loop_enabled=True
)
logger.info(f"[{script_name}] Logger initialized for volatility indicators (pandas-ta).")

# -------------------------------------------------------------------
# ColumnUtils Integration
# -------------------------------------------------------------------
# Define the required columns for volatility indicators
required_columns = [
    "close",
    "high",
    "low",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "rsi",
    "bollinger_width",
    "date"
]

# -------------------------------------------------------------------
# VolatilityIndicators Class (pandas-ta)
# -------------------------------------------------------------------
class VolatilityIndicators:
    def __init__(self, logger: logging.Logger = None, data_store: DataStore = None):
        """
        Initializes the VolatilityIndicators class with a logger and data store.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info(f"[{script_name}] VolatilityIndicators (pandas-ta) initialized.")
        self.data_store = data_store
        self.column_utils = ColumnUtils()
        self.config_manager = config_manager
        self.logger.info(f"[{script_name}] VolatilityIndicators (pandas-ta) initialized.")

    def downcast_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcasts numerical columns to reduce memory usage.

        """
        self.logger.info(f"[{script_name}] Downcasting DataFrame to optimize memory usage.")
        float_cols = df.select_dtypes(include=['float64']).columns
        int_cols = df.select_dtypes(include=['int64']).columns

        df[float_cols] = df[float_cols].astype('float32')
        df[int_cols] = df[int_cols].astype('int32')

        self.logger.info(f"[{script_name}] Downcasting completed.")
        return df
    
    def downcast_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcasts numerical columns to reduce memory usage.
        """
        self.logger.info(f"[{script_name}] Downcasting DataFrame to optimize memory usage.")
        float_cols = df.select_dtypes(include=['float64']).columns
        int_cols = df.select_dtypes(include=['int64']).columns

        df[float_cols] = df[float_cols].astype('float32')
        df[int_cols] = df[int_cols].astype('int32')

        self.logger.info(f"[{script_name}] Downcasting completed.")
        return df

    def set_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sets the 'date' or 'Date' column as the DatetimeIndex and sorts the DataFrame.
        """
        self.logger.info(f"[{script_name}] Setting 'Date' as DatetimeIndex.")
        
        try:
            if 'date' in df.columns:
                # Convert to datetime with coercion and handle invalid dates
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                invalid_dates = df['date'].isna().sum()
                if invalid_dates > 0:
                    self.logger.warning(f"Found {invalid_dates} invalid dates in 'date' column")
                
                # Drop NaT values and duplicates
                original_len = len(df)
                df = df.dropna(subset=['date'])
                df = df.drop_duplicates(subset=['date'])
                dropped_rows = original_len - len(df)
                if dropped_rows > 0:
                    self.logger.warning(f"Dropped {dropped_rows} rows with invalid/duplicate dates")
                
                df = df.set_index('date').sort_index()
                df.rename_axis('Date', inplace=True)
                
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                invalid_dates = df['Date'].isna().sum()
                if invalid_dates > 0:
                    self.logger.warning(f"Found {invalid_dates} invalid dates in 'Date' column")
                
                original_len = len(df)
                df = df.dropna(subset=['Date'])
                df = df.drop_duplicates(subset=['Date'])
                dropped_rows = original_len - len(df)
                if dropped_rows > 0:
                    self.logger.warning(f"Dropped {dropped_rows} rows with invalid/duplicate dates")
                
                df = df.set_index('Date').sort_index()
            else:
                self.logger.error("No 'date' or 'Date' column found in DataFrame.")
                raise KeyError("Date column missing")
            
            self.logger.info(f"[{script_name}] 'Date' set as DatetimeIndex. {len(df)} records remain after cleaning.")
            return df
        
        except Exception as e:
            self.logger.error(f"Error setting datetime index: {e}", exc_info=True)
            raise

    def add_bollinger_bands(self, df, window_size=20, std_multiplier=2, user_defined_window=None):
        """
        Adds Bollinger Bands (Bollinger_Low, Bollinger_Mid, Bollinger_High) using pandas-ta.
        """
        self.logger.info(f"[{script_name}] Adding Bollinger Bands (pandas-ta)")
        if 'close' not in df.columns:
            raise ValueError("[volatility_indicators.py] Column 'close' not found in DataFrame")

        length = user_defined_window if user_defined_window is not None else window_size
        bb = ta.bbands(df['close'], length=length, std=std_multiplier)

        if bb is None or bb.empty:
            raise ValueError("[volatility_indicators.py] ta.bbands returned empty DataFrame")

        # Rename columns to Bollinger_Low, Bollinger_Mid, Bollinger_High
        bb_cols = list(bb.columns)  # e.g., [BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0]
        if len(bb_cols) < 3:
            raise ValueError("[volatility_indicators.py] Unexpected columns from ta.bbands")

        df['bollinger_low'] = bb[bb_cols[0]].astype('float32')
        df['bollinger_mid'] = bb[bb_cols[1]].astype('float32')
        df['bollinger_high'] = bb[bb_cols[2]].astype('float32')

        self.logger.info(f"[{script_name}] Successfully added Bollinger Bands.")
        return df

    def add_standard_deviation(self, df, window_size=20, user_defined_window=None):
        """
        Adds Standard Deviation to the DataFrame using pandas-ta stdev.
        """
        self.logger.info(f"[{script_name}] Adding Standard Deviation (pandas-ta)")
        if 'close' not in df.columns:
            raise ValueError("[volatility_indicators.py] Column 'close' not found in DataFrame")

        length = user_defined_window if user_defined_window is not None else window_size
        std_series = ta.stdev(df['close'], length=length)

        if std_series is None or std_series.empty:
            raise ValueError("[volatility_indicators.py] ta.stdev returned empty DataFrame")

        df['standard_deviation'] = std_series.astype('float32')
        self.logger.info(f"[{script_name}] Successfully added Standard Deviation.")
        return df

    def add_historical_volatility(self, df, window=20, user_defined_window=None):
        """
        Adds Historical Volatility using log returns and rolling std.
        """
        self.logger.info(f"[{script_name}] Adding Historical Volatility (custom)")
        if 'close' not in df.columns:
            raise ValueError("[volatility_indicators.py] DataFrame must contain 'close'")

        length = user_defined_window if user_defined_window is not None else window

        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        hv_series = df['log_returns'].rolling(window=length, min_periods=1).std() * np.sqrt(252)
        if hv_series is None or hv_series.empty:
            raise ValueError("[volatility_indicators.py] Rolling std of log returns is empty")

        df['historical_volatility'] = hv_series.astype('float32')
        df.drop(['log_returns'], axis=1, inplace=True)
        self.logger.info(f"[{script_name}] Successfully added Historical Volatility.")
        return df

    def add_chandelier_exit(self, df, window=22, multiplier=3, user_defined_window=None, user_defined_multiplier=None):
        """
        Adds Chandelier Exit. Requires 'high', 'low', 'close'.
        """
        self.logger.info(f"[{script_name}] Adding Chandelier Exit")
        required_columns = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("[volatility_indicators.py] Missing required columns for Chandelier Exit")

        length = user_defined_window if user_defined_window is not None else window
        atr_mult = user_defined_multiplier if user_defined_multiplier is not None else multiplier

        df['highest_high'] = df['high'].rolling(window=length, min_periods=1).max()

        df['true_range'] = df[['high', 'low', 'close']].apply(
            lambda row: max(
                row['high'] - row['low'],
                abs(row['high'] - row['close']),
                abs(row['low'] - row['close'])
            ), axis=1
        )
        df['atr'] = df['true_range'].rolling(window=length, min_periods=1).mean()
        df['chandelier_exit_long'] = df['highest_high'] - (atr_mult * df['atr'])

        # Drop intermediate columns
        df.drop(['highest_high', 'true_range', 'atr'], axis=1, inplace=True)
        self.logger.info(f"[{script_name}] Successfully added Chandelier Exit.")
        return df

    def add_keltner_channel(self, df, window=20, multiplier=2, user_defined_window=None, user_defined_multiplier=None):
        """
        Adds Keltner Channel using pandas-ta (kc).
        """
        self.logger.info(f"[{script_name}] Adding Keltner Channel (pandas-ta)")
        required_columns = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("[volatility_indicators.py] Missing columns for Keltner Channel")

        length = user_defined_window if user_defined_window is not None else window
        scalar = user_defined_multiplier if user_defined_multiplier is not None else multiplier

        kc = ta.kc(high=df['high'], low=df['low'], close=df['close'], length=length, scalar=scalar)
        if kc is None or kc.empty:
            raise ValueError("[volatility_indicators.py] ta.kc returned empty DataFrame")

        kc_cols = list(kc.columns)  # e.g., [KCL_20_2.0, KCB_20_2.0, KCU_20_2.0]
        if len(kc_cols) < 3:
            raise ValueError("[volatility_indicators.py] Keltner Channel columns unexpected")

        df['keltner_channel_low'] = kc[kc_cols[0]].astype('float32')
        df['keltner_channel_basis'] = kc[kc_cols[1]].astype('float32')
        df['keltner_channel_high'] = kc[kc_cols[2]].astype('float32')

        self.logger.info(f"[{script_name}] Successfully added Keltner Channel.")
        return df

    def add_moving_average_envelope(self, df, window_size=10, percentage=0.025, user_defined_window=None, user_defined_percentage=None):
        """
        Adds a simple MA Envelope: Upper/Lower based on SMA and a percentage.
        """
        self.logger.info(f"[{script_name}] Adding Moving Average Envelope")
        if 'close' not in df.columns:
            raise ValueError("[volatility_indicators.py] 'close' not found for MA Envelope")

        length = user_defined_window if user_defined_window is not None else window_size
        env_pct = user_defined_percentage if user_defined_percentage is not None else percentage

        sma_series = ta.sma(df['close'], length=length)
        if sma_series is None or sma_series.empty:
            raise ValueError("[volatility_indicators.py] ta.sma returned empty for Envelope")

        df['mae_upper'] = sma_series * (1 + env_pct)
        df['mae_lower'] = sma_series * (1 - env_pct)

        self.logger.info(f"[{script_name}] Successfully added Moving Average Envelope.")
        return df

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all volatility indicators to the DataFrame.
        """
        self.logger.info(f"[{script_name}] Applying Volatility Indicators...")
        
        # Log initial shape
        self.logger.info(f"Initial DataFrame shape: {df.shape}")
        
        required_columns = ['close', 'high', 'low']
        if not all(col in df.columns for col in required_columns):
            self.logger.error(f"[{script_name}] DataFrame missing columns: {required_columns}")
            return df

        try:
            # Set datetime index with enhanced validation
            df = self.set_datetime_index(df)
            
            # Apply indicators
            df = self.add_bollinger_bands(df)
            df = self.add_standard_deviation(df)
            df = self.add_historical_volatility(df)
            df = self.add_chandelier_exit(df)
            df = self.add_keltner_channel(df)
            df = self.add_moving_average_envelope(df)
            
            # Reset the index to move 'Date' back to a column before saving
            df = df.reset_index()
            
            # Log final shape
            self.logger.info(f"Final DataFrame shape after applying indicators: {df.shape}")
            
            self.logger.info(f"[{script_name}] Successfully applied all volatility indicators.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Error applying Volatility Indicators: {e}", exc_info=True)

        return df

    # -------------------------------------------------------------------
    # Optimized Methods for Large / Streaming Data
    # -------------------------------------------------------------------
    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a single chunk by applying indicators.
        """
        self.logger.debug(f"[{script_name}] Processing chunk of size {len(chunk)}")
        chunk = self.downcast_dataframe(chunk)
        try:
            chunk = self.set_datetime_index(chunk)
            chunk = self.apply_indicators(chunk)
        except Exception as e:
            self.logger.error(f"[{script_name}] Error processing chunk: {e}", exc_info=True)
        return chunk

    def process_large_dataset(self, file_path: str, chunksize=50000) -> None:
        """
        Processes a large CSV in chunks and saves the result with indicators.
        """
        self.logger.info(f"[{script_name}] Starting chunked processing for {file_path} (chunksize={chunksize})")
        start_time = timer()

        try:
            pool = Pool(cpu_count())
            reader = pd.read_csv(
                file_path, 
                chunksize=chunksize, 
                dtype={'close': 'float32', 'high': 'float32', 'low': 'float32', 'volume': 'int32'}
            )
            processed_chunks = pool.map(self.process_chunk, reader)
            pool.close()
            pool.join()

            processed_df = pd.concat(processed_chunks, ignore_index=True)
            output_path = file_path.replace('.csv', '_processed.csv')
            processed_df.to_csv(output_path, index=False)
            self.logger.info(f"[{script_name}] Chunked processing complete. Saved to {output_path}")
        except Exception as e:
            self.logger.error(f"[{script_name}] Error during chunked processing: {e}", exc_info=True)
        finally:
            total_time = timer() - start_time
            self.logger.info(f"[{script_name}] Total processing time: {total_time:.2f} s")

    def process_streaming_data(self, data_stream, window_size=20):
        """
        Processes streaming data with a sliding window approach.
        """
        self.logger.info(f"[{script_name}] Starting streaming data processing.")
        buffer = deque(maxlen=window_size)

        for data_point in data_stream:
            buffer.append(data_point)
            if len(buffer) < window_size:
                self.logger.debug(f"[{script_name}] Not enough data for window_size={window_size}")
                continue

            df = pd.DataFrame(list(buffer))
            df = self.downcast_dataframe(df)
            try:
                df = self.set_datetime_index(df)
                df = self.apply_indicators(df)
                latest = df.iloc[-1].to_dict()
                self.logger.info(f"[{script_name}] Processed streaming data point: {latest}")
            except Exception as e:
                self.logger.error(f"[{script_name}] Error processing streaming data point: {e}", exc_info=True)

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
def main():
    logger.info(f"[{script_name}] Entering main()")
    try:
        # Initialize DBHandler
        db_handler = DBHandler()
        logger.info(f"[{script_name}] DatabaseHandler initialized.")

        # Initialize DataStore
        data_store = DataStore(config=config_manager, logger=logger, use_csv=False)
        logger.info(f"[{script_name}] DataStore initialized.")

        # Load data for a sample symbol
        symbol = "AAPL"
        df = data_store.load_data(symbol)

        if df is None or df.empty:
            logger.error(f"[{script_name}] No data found for symbol '{symbol}'. Exiting.")
            return

        # Process the DataFrame using ColumnUtils
        try:
            # Initialize ColumnUtils instance
            column_utils = ColumnUtils()
            # Process the DataFrame
            df = column_utils.process_dataframe(df, stage="pre")
            logger.info(f"[{script_name}] DataFrame processed with ColumnUtils.")
        except (KeyError, FileNotFoundError, ValueError) as ve:
            logger.error(f"[{script_name}] Data processing failed: {ve}")
            return

        # Initialize VolatilityIndicators and apply indicators
        volatility_indicators = VolatilityIndicators(logger=logger)

        # Apply indicators
        df = volatility_indicators.apply_indicators(df)

        # Save the DataFrame back to the database
        data_store.save_data(df, symbol, overwrite=True)
        logger.info(f"[{script_name}] Data saved with new volatility indicators for symbol '{symbol}'.")

        # Optional: Log or print sample data
        expected_cols = [
            "bollinger_low",
            "bollinger_mid",
            "bollinger_high",
            "standard_deviation",
            "historical_volatility",
            "chandelier_exit_long",
            "keltner_channel_low",
            "keltner_channel_basis",
            "keltner_channel_high",
            "mae_upper",
            "mae_lower"
        ]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"[{script_name}] Missing volatility indicator columns: {missing_cols}")
        else:
            logger.info(f"[{script_name}] All volatility indicator columns are present.")
            print(f"\n[{script_name}] Sample volatility indicators:\n", df[expected_cols].tail())

    except Exception as e:
        logger.error(f"[{script_name}] Unexpected error in main(): {e}", exc_info=True)
    finally:
        if 'db_handler' in locals() and db_handler:
            try:
                db_handler.close()
                logger.info(f"[{script_name}] DatabaseHandler closed.")
            except Exception as ex:
                logger.error(f"[{script_name}] Error closing DatabaseHandler: {ex}", exc_info=True)

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
