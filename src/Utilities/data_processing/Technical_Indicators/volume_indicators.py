# -------------------------------------------------------------------
# File Path: C:\Projects\TradingRobotPlug\src\Data_Processing\Technical_Indicators\volume_indicators.py
# Description: Provides volume indicators such as MFI, OBV, VWAP,
#              Accumulation/Distribution Line, Chaikin Money Flow, 
#              and Volume Oscillator, integrating with ColumnUtils 
#              for column standardization and logging setup.
# -------------------------------------------------------------------

import sys
import pandas as pd
import logging
from pathlib import Path
from time import perf_counter as timer
import numpy as np
import pandas_ta as ta
from multiprocessing import Pool, cpu_count
from collections import deque
from functools import partial
from Utilities.data_processing.base_indicators import BaseIndicator

# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------
script_dir = Path(__file__).resolve()
project_root = script_dir.parents[4]  # Adjust based on your project structure
utilities_dir = project_root / 'src' / 'Utilities'

# Add directories to sys.path for importing modules
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
    print("[volume_indicators.py] Imported config_manager, db_handler, data_store, column_utils successfully.")
except ImportError as e:
    print(f"[volume_indicators.py] Error importing utility modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Configuration and Logger Setup
# -------------------------------------------------------------------
dotenv_path = project_root / '.env'
required_keys = [
    'POSTGRES_HOST', 
    'POSTGRES_DBNAME', 
    'POSTGRES_USER', 
    'POSTGRES_PASSWORD', 
    'POSTGRES_PORT'
]

try:
    config_manager = ConfigManager(env_file=dotenv_path, required_keys=required_keys, logger=None)
except KeyError as e:
    print(f"[volume_indicators.py] Missing required configuration keys: {e}")
    sys.exit(1)

log_dir = project_root / 'logs' / 'technical_indicators'
log_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logging(
    script_name="volume_indicators.py",
    log_dir=log_dir,
    max_log_size=5 * 1024 * 1024,  # 5 MB
    backup_count=3,
    console_log_level=logging.INFO,
    file_log_level=logging.DEBUG,
    feedback_loop_enabled=True
)
logger.info("Logger initialized for volume indicators.")

# -------------------------------------------------------------------
# ColumnUtils Integration
# -------------------------------------------------------------------
# Only the columns needed for volume indicators
required_columns = [
    "close",
    "high",
    "low",
    "volume",
    "date"
]

# -------------------------------------------------------------------
# VolumeIndicators Class Definition
# -------------------------------------------------------------------
class VolumeIndicators(BaseIndicator):
    """
    Applies volume-based technical indicators (MFI, OBV, VWAP, ADL, CMF, Volume Oscillator).
    """

    def __init__(self, logger=None):
        """
        Initializes the VolumeIndicators class with an optional logger.
        
        Args:
            logger (logging.Logger, optional): Logger instance for logging.
        """
        super().__init__(logger)
        self.logger.info("VolumeIndicators initialized.")
    
    def downcast_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcasts numerical columns to reduce memory usage.
        """
        self.logger.info("Downcasting DataFrame to optimize memory usage.")
        float_cols = df.select_dtypes(include=['float64']).columns
        int_cols = df.select_dtypes(include=['int64']).columns

        df[float_cols] = df[float_cols].astype('float32')
        df[int_cols] = df[int_cols].astype('int32')

        self.logger.info("Downcasting completed.")
        return df
    
    def set_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sets the 'date' or 'Date' column as the DatetimeIndex and sorts the DataFrame.
        """
        self.logger.info("Setting 'Date' as DatetimeIndex.")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.set_index('date').sort_index()
            df.rename_axis('Date', inplace=True)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.set_index('Date').sort_index()
        else:
            self.logger.error("No 'date' or 'Date' column found in DataFrame.")
            raise KeyError("Date column missing")

        self.logger.info("'Date' set as DatetimeIndex.")
        return df

    def add_money_flow_index(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Adds Money Flow Index (MFI) to the DataFrame.
        """
        self.logger.info("Adding Money Flow Index (MFI)")
        required = ['high', 'low', 'close', 'volume']
        self._validate_columns(df, required)

        try:
            mfi = ta.mfi(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], length=window)
            df['mfi'] = mfi.fillna(0).astype('float32')
            self.logger.info("Successfully added MFI")
        except Exception as e:
            self.logger.error(f"Failed to calculate MFI: {e}", exc_info=True)
            df['mfi'] = 0.0  # Assign default value or handle as needed

        return df

    def add_on_balance_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds On-Balance Volume (OBV) to the DataFrame.
        """
        self.logger.info("Adding On-Balance Volume (OBV)")
        self._validate_columns(df, ['volume', 'close'])

        try:
            obv = ta.obv(close=df['close'], volume=df['volume'])
            df['obv'] = obv.fillna(0).astype('float32')
            self.logger.info("Successfully added OBV")
        except Exception as e:
            self.logger.error(f"Failed to calculate OBV: {e}", exc_info=True)
            df['obv'] = 0.0

        return df

    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds Volume Weighted Average Price (VWAP) to the DataFrame.
        """
        self.logger.info("Adding Volume Weighted Average Price (VWAP)")
        self._validate_columns(df, ['high', 'low', 'close', 'volume'])

        try:
            vwap = ta.vwap(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
            df['vwap'] = vwap.fillna(method='ffill').fillna(0).astype('float32')
            self.logger.info("Successfully added VWAP")
        except Exception as e:
            self.logger.error(f"Failed to calculate VWAP: {e}", exc_info=True)
            df['vwap'] = 0.0

        return df

    def add_accumulation_distribution_line(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds the Accumulation/Distribution Line (ADL) to the DataFrame.
        """
        self.logger.info("Adding Accumulation/Distribution Line (ADL)")
        self._validate_columns(df, ['close', 'low', 'high', 'volume'])

        try:
            adl = ta.adl(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
            df['adl'] = adl.fillna(method='ffill').fillna(0).astype('float32')
            self.logger.info("Successfully added ADL")
        except Exception as e:
            self.logger.error(f"Failed to calculate ADL: {e}", exc_info=True)
            df['adl'] = 0.0

        return df

    def add_chaikin_money_flow(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Adds Chaikin Money Flow (CMF) to the DataFrame.
        """
        self.logger.info("Adding Chaikin Money Flow (CMF)")
        self._validate_columns(df, ['high', 'low', 'close', 'volume'])

        try:
            cmf = ta.cmf(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], length=window)
            df['cmf'] = cmf.fillna(0).astype('float32')
            self.logger.info("Successfully added CMF")
        except Exception as e:
            self.logger.error(f"Failed to calculate CMF: {e}", exc_info=True)
            df['cmf'] = 0.0

        return df

    def add_volume_oscillator(self, df: pd.DataFrame, short_window: int = 12, long_window: int = 26) -> pd.DataFrame:
        """
        Adds Volume Oscillator to the DataFrame.
        """
        self.logger.info("Adding Volume Oscillator")
        self._validate_columns(df, ['volume'])

        try:
            short_vol_ema = ta.ema(df['volume'], length=short_window)
            long_vol_ema = ta.ema(df['volume'], length=long_window)
            volume_osc = short_vol_ema - long_vol_ema
            df['volume_oscillator'] = volume_osc.fillna(0).astype('float32')
            self.logger.info("Successfully added Volume Oscillator")
        except Exception as e:
            self.logger.error(f"Failed to calculate Volume Oscillator: {e}", exc_info=True)
            df['volume_oscillator'] = 0.0

        return df

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required_columns: list):
        """
        Validates the presence of required columns in the DataFrame.
        """
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies volume-based technical indicators to the DataFrame.
        """
        self.logger.info("Applying Volume Indicators...")

        try:
            df = self.set_datetime_index(df)
            df = self.downcast_dataframe(df)
            df = self.add_money_flow_index(df)
            df = self.add_on_balance_volume(df)
            df = self.add_vwap(df)
            df = self.add_accumulation_distribution_line(df)
            df = self.add_chaikin_money_flow(df)
            df = self.add_volume_oscillator(df)
            self.logger.info("Successfully applied all Volume Indicators.")
        except Exception as e:
            self.logger.error(f"Error applying Volume Indicators: {e}", exc_info=True)

        return df

    # -------------------------------------------------------------------
    # Optimized Methods for Large / Streaming Data
    # -------------------------------------------------------------------
    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a single chunk by applying indicators.
        """
        self.logger.debug(f"Processing chunk of size {len(chunk)}")
        chunk = self.downcast_dataframe(chunk)
        try:
            chunk = self.set_datetime_index(chunk)
            chunk = self.apply_indicators(chunk)
        except Exception as e:
            self.logger.error(f"Error processing chunk: {e}", exc_info=True)
        return chunk

    def process_large_dataset(self, file_path: str, chunksize: int = 50000) -> None:
        """
        Processes a large CSV in chunks and saves the result with indicators.
        """
        self.logger.info(f"Starting chunked processing for {file_path} (chunksize={chunksize})")
        start_time = timer()

        try:
            pool = Pool(cpu_count())
            reader = pd.read_csv(
                file_path,
                chunksize=chunksize,
                dtype={
                    'close': 'float32',
                    'high': 'float32',
                    'low': 'float32',
                    'volume': 'int32'
                }
            )
            processed_chunks = pool.map(self.process_chunk, reader)
            pool.close()
            pool.join()

            processed_df = pd.concat(processed_chunks, ignore_index=True)
            output_path = file_path.replace('.csv', '_processed.csv')
            processed_df.to_csv(output_path, index=False)
            self.logger.info(f"Chunked processing complete. Saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error during chunked processing: {e}", exc_info=True)
        finally:
            total_time = timer() - start_time
            self.logger.info(f"Total processing time: {total_time:.2f} seconds.")

    def process_streaming_data(self, data_stream, window_size: int = 20):
        """
        Processes streaming data with a sliding window approach.
        """
        self.logger.info("Starting streaming data processing.")
        buffer = deque(maxlen=window_size)

        for data_point in data_stream:
            buffer.append(data_point)
            if len(buffer) < window_size:
                self.logger.debug(f"Not enough data for window_size={window_size}")
                continue

            df = pd.DataFrame(list(buffer))
            df = self.downcast_dataframe(df)
            try:
                df = self.set_datetime_index(df)
                df = self.apply_indicators(df)
                latest = df.iloc[-1].to_dict()
                self.logger.info(f"Processed streaming data point: {latest}")
            except Exception as e:
                self.logger.error(f"Error processing streaming data point: {e}", exc_info=True)

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
def main():
    logger.info("Entering main() in volume_indicators.py")
    try:
        # Initialize DatabaseHandler
        db_handler = DBHandler(logger=logger)
        logger.info("DatabaseHandler initialized.")

        # Initialize DataStore
        data_store = DataStore(config=config_manager, logger=logger, use_csv=False)
        logger.info("DataStore initialized.")

        # Load data for a sample symbol
        symbol = "AAPL"
        df = data_store.load_data(symbol=symbol)

        if df is None or df.empty:
            logger.error(f"No data found for symbol '{symbol}'. Exiting.")
            return

        # Process the DataFrame using ColumnUtils
        # (only requiring the volume-related columns for this module)
        try:
            column_config_path = project_root / 'src' / 'Utilities' / 'column_config.json'
            df = ColumnUtils.process_dataframe(
                df,
                config_path=column_config_path,
                required_columns=required_columns,  # Only volume-based columns
                logger=logger
            )
            logger.info("DataFrame processed with ColumnUtils.")
        except (KeyError, FileNotFoundError, ValueError) as ve:
            logger.error(f"Data processing failed: {ve}")
            return

        # Initialize VolumeIndicators and apply them
        volume_indicators = VolumeIndicators(logger=logger)
        df = volume_indicators.apply_indicators(df)

        # Display the updated data
        expected_cols = [
            "mfi", "obv", "vwap", "adl",
            "cmf", "volume_oscillator"
        ]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing volume indicator columns: {missing_cols}")
        else:
            logger.info("All volume indicator columns are present.")
            print(f"\n[volume_indicators.py] Sample Volume Indicators:\n", df[expected_cols].tail())

        # Save the modified data back to the SQL database
        data_store.save_data(df, symbol=symbol, overwrite=True)
        logger.info(f"Data saved with new volume indicators for symbol '{symbol}'.")

    except Exception as e:
        logger.error(f"Unexpected error in main(): {e}", exc_info=True)
    finally:
        if 'db_handler' in locals() and db_handler:
            try:
                db_handler.close()
                logger.info("DatabaseHandler closed.")
            except Exception as ex:
                logger.error(f"Error closing DatabaseHandler: {ex}", exc_info=True)

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
