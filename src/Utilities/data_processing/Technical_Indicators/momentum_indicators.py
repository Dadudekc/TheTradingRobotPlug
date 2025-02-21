# -------------------------------------------------------------------
# File Path: D:\TradingRobotPlug2\src\Data_Processing\Technical_Indicators\momentum_indicators.py
# Description:
#     Provides composable momentum indicators such as Stochastic Oscillator, RSI,
#     Williams %R, ROC, and TRIX. Integrates with ConfigManager, DatabaseHandler,
#     ColumnUtils for standardized column handling, and logging setup.
# -------------------------------------------------------------------

import pandas as pd
import logging
from pathlib import Path
import sys
from abc import ABC, abstractmethod
from typing import List, Optional
from collections import deque
from timeit import default_timer as timer

from ta.momentum import StochasticOscillator, RSIIndicator, WilliamsRIndicator, ROCIndicator
from ta.trend import TRIXIndicator
from dotenv import load_dotenv

import numpy as np  # for example usage
from multiprocessing import Pool, cpu_count

# -------------------------------------------------------------------
# Identify Script Name and Project Root
# -------------------------------------------------------------------
script_file = Path(__file__).resolve()
script_name = script_file.name  # "momentum_indicators.py"
project_root = script_file.parents[4]  # Adjusted for the project structure

print(f"[{script_name}] Current script path: {script_file}")
print(f"[{script_name}] Project root: {project_root}")

# -------------------------------------------------------------------
# Ensure project_root is in sys.path
# -------------------------------------------------------------------
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# -------------------------------------------------------------------
# Load Environment Variables from the .env File
# -------------------------------------------------------------------
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"[{script_name}] Environment variables loaded from {env_path}")
else:
    print(f"[{script_name}] Warning: .env file not found at {env_path}")

# -------------------------------------------------------------------
# Additional Directories Setup
# -------------------------------------------------------------------
utilities_dir = project_root / 'src' / 'Utilities'
scripts_dir = project_root / 'src'
data_processing_dir = scripts_dir / 'Data_Processing'

sys.path.extend([
    str(utilities_dir.resolve()),
    str(scripts_dir.resolve()),
    str(data_processing_dir.resolve())
])

# -------------------------------------------------------------------
# Importing ConfigManager, Logging Setup, DatabaseHandler, DataStore, ColumnUtils
# -------------------------------------------------------------------
try:
    from Utilities.config_manager import ConfigManager, setup_logging
    from Utilities.data.data_store import DataStore
    from Utilities.column_utils import ColumnUtils
    print(f"[{script_name}] Successfully imported config_manager, db_handler, data_store, column_utils.")

except ImportError as e:
    print(f"[{script_name}] Error importing modules: {e}")
    sys.exit(1)

def get_db_handler():
    from Utilities.db.db_handler import DBHandler
    return DBHandler
# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
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
logger.info(f"[{script_name}] Logger setup successfully for Momentum Indicators")

# -------------------------------------------------------------------
# Configuration Setup
# -------------------------------------------------------------------
required_keys = [
    'POSTGRES_HOST',
    'POSTGRES_DBNAME',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_PORT',
    'ALPHAVANTAGE_API_KEY',
    'ALPHAVANTAGE_BASE_URL',
    'ML_FEATURE_COLUMNS',        # Added for machine learning
    'ML_TARGET_COLUMN',          # Added for machine learning
    'ML_MODEL_PATH',             # Added for machine learning
    'ML_MIN_ROWS'                # Added for machine learning
]

try:
    config_manager = ConfigManager(env_file=env_path, required_keys=required_keys, logger=logger)
    logger.info(f"[{script_name}] ConfigManager initialized successfully.")
except KeyError as e:
    logger.error(f"[{script_name}] Missing required configuration keys: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Abstract Indicator Class
# -------------------------------------------------------------------
class Indicator(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# -------------------------------------------------------------------
# Specific Momentum Indicator Classes
# -------------------------------------------------------------------
class StochasticOscillatorIndicator(Indicator):
    def __init__(self, window: int = 14, smooth_window: int = 3, logger: Optional[logging.Logger] = None):
        self.window = window
        self.smooth_window = smooth_window
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"[{script_name}] StochasticOscillatorIndicator initialized: window={self.window}, smooth={self.smooth_window}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Applying Stochastic Oscillator (window={self.window}, smooth={self.smooth_window})")
        required_cols = ['high', 'low', 'close']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.logger.error(f"[{script_name}] Missing columns for Stochastic: {missing}")
            raise ValueError(f"Missing columns for StochasticOscillator: {missing}")

        try:
            indicator = StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.window,
                smooth_window=self.smooth_window
            )
            df['stochastic'] = indicator.stoch()
            df['stochastic_signal'] = indicator.stoch_signal()
            self.logger.info(f"[{script_name}] Stochastic Oscillator applied successfully.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate Stochastic Oscillator: {e}", exc_info=True)
            df['stochastic'] = 0.0
            df['stochastic_signal'] = 0.0

        return df

class RSIIndicatorClass(Indicator):
    def __init__(self, window: int = 14, logger: Optional[logging.Logger] = None):
        self.window = window
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"[{script_name}] RSIIndicatorClass initialized with window={self.window}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Applying RSI (window={self.window})")

        if 'close' not in df.columns:
            self.logger.error(f"[{script_name}] 'close' column missing for RSI. Cannot compute RSI.")
            raise ValueError("Missing 'close' column for RSI")

        try:
            indicator = RSIIndicator(close=df['close'], window=self.window)
            df['rsi'] = indicator.rsi()
            self.logger.info(f"[{script_name}] RSI applied successfully.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate RSI: {e}", exc_info=True)
            df['rsi'] = 0.0

        return df

class WilliamsRIndicatorClass(Indicator):
    def __init__(self, lbp: int = 14, logger: Optional[logging.Logger] = None):
        self.lbp = lbp
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"[{script_name}] WilliamsRIndicatorClass initialized with lbp={self.lbp}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Applying Williams %R (lbp={self.lbp})")
        required_cols = ['high', 'low', 'close']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.logger.error(f"[{script_name}] Missing columns for Williams %R: {missing}")
            raise ValueError(f"Missing columns for Williams %R: {missing}")

        try:
            indicator = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=self.lbp)
            df['williams_r'] = indicator.williams_r()
            self.logger.info(f"[{script_name}] Williams %R applied successfully.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate Williams %R: {e}", exc_info=True)
            df['williams_r'] = 0.0

        return df

class ROCIndicatorClass(Indicator):
    def __init__(self, window: int = 12, logger: Optional[logging.Logger] = None):
        self.window = window
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"[{script_name}] ROCIndicatorClass initialized with window={self.window}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Applying ROC (window={self.window})")
        if 'close' not in df.columns:
            self.logger.error(f"[{script_name}] 'close' column missing for ROC")
            raise ValueError("Missing 'close' column for ROC")

        try:
            indicator = ROCIndicator(close=df['close'], window=self.window)
            df['roc'] = indicator.roc()
            self.logger.info(f"[{script_name}] ROC applied successfully.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate ROC: {e}", exc_info=True)
            df['roc'] = 0.0

        return df
class TRIXIndicatorClass(Indicator):
    def __init__(self, window: int = 15, signal_window: int = 9, logger: Optional[logging.Logger] = None):
        self.window = window
        self.signal_window = signal_window
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"[{script_name}] TRIXIndicatorClass initialized: window={self.window}, signal_window={self.signal_window}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Applying TRIX (window={self.window}, signal={self.signal_window})")
        if 'close' not in df.columns:
            self.logger.error(f"[{script_name}] 'close' column missing for TRIX")
            raise ValueError("Missing 'close' column for TRIX")

        try:
            # Calculate TRIX
            trix_ind = TRIXIndicator(close=df['close'], window=self.window, fillna=False)
            df['trix'] = trix_ind.trix()

            # ✅ FIX: Manually compute TRIX Signal Line using EMA
            df['trix_signal'] = df['trix'].ewm(span=self.signal_window, adjust=False).mean()

            self.logger.info(f"[{script_name}] TRIX applied successfully.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate TRIX: {e}", exc_info=True)
            df['trix'] = 0.0
            df['trix_signal'] = 0.0

        return df

# -------------------------------------------------------------------
# Indicator Pipeline
# -------------------------------------------------------------------
class IndicatorPipeline:
    def __init__(self, indicators: Optional[List[Indicator]] = None, logger: Optional[logging.Logger] = None):
        self.indicators = indicators or []
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"[{script_name}] Initialized IndicatorPipeline with {len(self.indicators)} indicators")

    def add_indicator(self, indicator: Indicator):
        self.indicators.append(indicator)
        self.logger.info(f"[{script_name}] Added {indicator.__class__.__name__} to the pipeline")

    def remove_indicator(self, indicator_cls):
        before = len(self.indicators)
        self.indicators = [i for i in self.indicators if not isinstance(i, indicator_cls)]
        after = len(self.indicators)
        self.logger.info(f"[{script_name}] Removed {indicator_cls.__name__}: {before}->{after}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for indicator in self.indicators:
            self.logger.info(f"[{script_name}] Applying {indicator.__class__.__name__}")
            try:
                df = indicator.apply(df)
            except Exception as e:
                self.logger.error(f"[{script_name}] Failed to apply {indicator.__class__.__name__}: {e}", exc_info=True)
        return df

# -------------------------------------------------------------------
# MomentumIndicators Class Definition
# -------------------------------------------------------------------

class MomentumIndicators:
    """
    Encapsulates all momentum indicators and provides a composable interface to apply them.
    """

    def __init__(self, logger: logging.Logger, data_store: DataStore, pipeline: Optional[IndicatorPipeline] = None):
        """
        Initializes the MomentumIndicators class.
        
        :param logger: Logger instance for logging operations.
        :param data_store: DataStore instance to handle data operations.
        :param pipeline: Optional IndicatorPipeline instance. If None, a new pipeline is created.
        """
        self.logger = logger
        self.pipeline = pipeline or IndicatorPipeline(logger=self.logger)
        self.data_store = data_store

        self.logger.info("[momentum_indicators.py] MomentumIndicators instance created.")
        self.initialize_indicators()

    def initialize_indicators(self):
        """
        Initializes and adds all momentum indicators to the pipeline.
        """
        self.logger.info("[momentum_indicators.py] Initializing momentum indicators.")

        try:
            self.add_indicator(RSIIndicatorClass(window=14, logger=self.logger))
            self.add_indicator(StochasticOscillatorIndicator(window=14, smooth_window=3, logger=self.logger))
            self.add_indicator(WilliamsRIndicatorClass(lbp=14, logger=self.logger))
            self.add_indicator(ROCIndicatorClass(window=12, logger=self.logger))
            self.add_indicator(TRIXIndicatorClass(window=15, signal_window=9, logger=self.logger))
            self.logger.info(f"[momentum_indicators.py] All {len(self.pipeline.indicators)} momentum indicators added.")
        except Exception as e:
            self.logger.error(f"[momentum_indicators.py] Error initializing indicators: {e}", exc_info=True)

    def add_indicator(self, indicator):
        """
        Adds an indicator to the pipeline.
        
        :param indicator: Indicator instance to add.
        """
        self.pipeline.add_indicator(indicator)
        self.logger.info(f"[momentum_indicators.py] Added {indicator.__class__.__name__} to the pipeline.")

    def remove_indicator(self, indicator_cls):
        """
        Removes an indicator from the pipeline.
        
        :param indicator_cls: The class of the indicator to remove.
        """
        self.pipeline.remove_indicator(indicator_cls)
        self.logger.info(f"[momentum_indicators.py] Removed {indicator_cls.__name__} from the pipeline.")

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("[momentum_indicators.py] Applying momentum indicators.")

        if not self.pipeline.indicators:
            self.logger.error("[momentum_indicators.py] No indicators in the pipeline. Ensure indicators are initialized.")
            return df

        try:
            # Validate required columns
            required_cols = {"date", "open", "high", "low", "close", "volume"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                self.logger.warning(f"[momentum_indicators.py] Missing required columns: {missing_cols}. Some indicators may not work.")

            column_utils = ColumnUtils()
            df = column_utils.process_dataframe(df, stage="pre")

            df = self.pipeline.apply(df)

            df = column_utils.process_dataframe(df, stage="post")

            # ✅ FIX: Ensure TRIX columns exist before saving
            expected_cols = ["stochastic", "stochastic_signal", "rsi", "williams_r", "roc", "trix", "trix_signal"]
            for col in expected_cols:
                if col not in df.columns:
                    self.logger.warning(f"[momentum_indicators.py] Column '{col}' is missing. Initializing with default values.")
                    df[col] = 0.0

            self.logger.info("[momentum_indicators.py] Successfully applied all momentum indicators.")
        except Exception as e:
            self.logger.error(f"[momentum_indicators.py] Error while applying indicators: {e}", exc_info=True)

        return df



    def load_data_from_sql(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Loads data for a given symbol from the SQL database.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Optional[pd.DataFrame]: The loaded DataFrame or None if failed.
        """
        try:
            self.logger.info(f"[{script_name}] Loading data for {symbol} from SQL.")
            df = self.data_store.load_data(symbol=symbol)
            if df is None or df.empty:
                self.logger.warning(f"[{script_name}] No data for {symbol}.")
                return None
            self.logger.info(f"[{script_name}] Loaded {len(df)} records for {symbol}.")
            return df
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to load data for {symbol}: {e}", exc_info=True)
            return None

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

        Args:
            file_path (str): Path to the input CSV file.
            chunksize (int, optional): Number of rows per chunk. Defaults to 50000.
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

        Args:
            data_stream (iterable): An iterable stream of data points.
            window_size (int, optional): The size of the sliding window. Defaults to 20.
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

    def downcast_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcasts numerical columns to optimize memory usage.
        """
        self.logger.debug("Downcasting DataFrame to optimize memory usage.")
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        return df

    def set_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sets 'Date' as the DatetimeIndex and removes duplicates.
        """
        self.logger.info(f"[{script_name}] Setting 'Date' as DatetimeIndex.")
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.set_index('Date').sort_index()

            # Remove duplicate indices
            if df.index.duplicated().any():
                self.logger.warning("Duplicate 'Date' entries found. Removing duplicates.")
                df = df[~df.index.duplicated(keep='last')]
                self.logger.info("Duplicate 'Date' entries removed.")

            self.logger.info(f"[{script_name}] 'Date' set as DatetimeIndex.")
            return df
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to set 'Date' as DatetimeIndex: {e}", exc_info=True)
            raise

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------

if __name__ == "__main__":
    def main():
        # Lazy imports to prevent circular dependencies
        from Utilities.db.db_handler import DBHandler
        from Utilities.config_manager import ConfigManager
        from Utilities.data.data_store import DataStore
        from Utilities.data_processing.Technical_Indicators.momentum_indicators import MomentumIndicators
        import pandas as pd

        logger.info("Entering main() in momentum_indicators.py")
        try:
            # Initialize ConfigManager
            config_manager = ConfigManager()

            # Initialize DBHandler (REMOVE 'config' ARGUMENT)
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

            # Initialize MomentumIndicators and apply indicators
            momentum_indicators = MomentumIndicators(logger=logger, data_store=data_store)

            # Apply indicators
            df = momentum_indicators.apply_indicators(df)

            # Display the updated data
            expected_cols = [
                "stochastic",
                "stochastic_signal",
                "rsi",
                "williams_r",
                "roc",
                "trix",
                "trix_signal"
            ]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing momentum indicator columns: {missing_cols}")
            else:
                logger.info("All momentum indicator columns are present.")
                print(f"\n[momentum_indicators.py] Sample Momentum Indicators:\n", df[expected_cols].tail())

            # Save the modified data back to the SQL database
            data_store.save_data(df, symbol=symbol, overwrite=True)
            logger.info(f"Data saved with new momentum indicators for symbol '{symbol}'.")

        except Exception as e:
            logger.error(f"Unexpected error in main(): {e}", exc_info=True)
        finally:
            if 'db_handler' in locals() and db_handler:
                try:
                    db_handler.close()
                    logger.info("DatabaseHandler closed.")
                except Exception as ex:
                    logger.error(f"Error closing DatabaseHandler: {ex}", exc_info=True)

    main()

