# -------------------------------------------------------------------
# File Path: D:\TradingRobotPlug2\src\Data_Processing\Technical_Indicators\custom_indicators.py
# Description: Provides custom technical indicators with caching support for trading algorithms,
#              integrated with PostgreSQL database. All DataFrame columns are standardized to lowercase.
# -------------------------------------------------------------------
    
import os
import sys
from pathlib import Path
import logging
import pandas as pd
import joblib
from typing import Callable, Any, Dict, Tuple, List, Optional
from time import time as timer
from abc import ABC, abstractmethod

# Import ta library indicators
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import TRIXIndicator

# -------------------------------------------------------------------
# Identify Script Name for Logging/Print
# -------------------------------------------------------------------
script_file = Path(__file__).resolve()
script_name = script_file.name  # "custom_indicators.py"
project_root = script_file.parents[4]

print(f"[{script_name}] Script path: {script_file}")
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
    from dotenv import load_dotenv
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
    from Utilities.db.db_handler import DatabaseHandler
    from Utilities.data.data_store import DataStore
    from Utilities.column_utils import ColumnUtils
    print(f"[{script_name}] Successfully imported config_manager, db_handler, data_store, column_utils.")
except ImportError as e:
    print(f"[{script_name}] Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------
logger = setup_logging(
    script_name=script_name,
    log_dir=project_root / 'logs' / 'technical_indicators',
    max_log_size=5 * 1024 * 1024,  # 5 MB
    backup_count=3,
    console_log_level=logging.INFO,
    file_log_level=logging.DEBUG,
    feedback_loop_enabled=True
)
logger.info(f"[{script_name}] Logger setup successfully for Custom Indicators")

# -------------------------------------------------------------------
# Configuration Setup
# -------------------------------------------------------------------
required_keys = [
    'POSTGRES_HOST',
    'POSTGRES_DBNAME',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_PORT',
    'CACHE_STRATEGY',
    'CACHE_DIRECTORY',
    # Add other required keys as needed
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
    """
    Abstract base class for all indicators.
    """
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# -------------------------------------------------------------------
# Specific Custom Indicator Classes
# -------------------------------------------------------------------
class CustomRSIIndicator(Indicator):
    def __init__(self, window: int = 14, handle_nans: str = "warn", column: str = "close", logger: Optional[logging.Logger] = None):
        self.window = window
        self.handle_nans = handle_nans
        self.column = column
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info(f"[{script_name}] Initialized RSI Indicator with window={self.window}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Adding RSI with window={self.window}")
        if self.column not in df.columns:
            self.logger.error(f"[{script_name}] Column '{self.column}' not found in DataFrame for RSI")
            raise ValueError(f"Missing '{self.column}' column for RSI")

        try:
            indicator = RSIIndicator(close=df[self.column], window=self.window)
            df['rsi'] = indicator.rsi()
            self.logger.info(f"[{script_name}] RSI applied successfully.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate RSI: {e}", exc_info=True)

        self._handle_nans(df, 'rsi')
        return df

    def _handle_nans(self, df: pd.DataFrame, column: str):
        nan_count = df[column].isna().sum()
        if nan_count > 0:
            self.logger.warning(f"[{script_name}] Column '{column}' has {nan_count} NaN values.")
            if self.handle_nans == 'drop':
                df.dropna(subset=[column], inplace=True)
                self.logger.info(f"[{script_name}] Dropped NaNs in '{column}'.")
            elif self.handle_nans == 'ffill':
                df[column].fillna(method='ffill', inplace=True)
                self.logger.info(f"[{script_name}] Forward filled NaNs in '{column}'.")
            elif self.handle_nans == 'warn':
                self.logger.warning(f"[{script_name}] '{column}' still has NaNs.")
            else:
                self.logger.warning(f"[{script_name}] Unknown NaN strategy '{self.handle_nans}'. No action taken.")

class CustomBollingerBandsIndicator(Indicator):
    def __init__(self, window: int = 20, window_dev: int = 2, handle_nans: str = "warn", column: str = "close", logger: Optional[logging.Logger] = None):
        self.window = window
        self.window_dev = window_dev
        self.handle_nans = handle_nans
        self.column = column
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info(f"[{script_name}] Initialized Bollinger with window={self.window}, dev={self.window_dev}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Adding Bollinger Bands window={self.window}, dev={self.window_dev}")
        if self.column not in df.columns:
            self.logger.error(f"[{script_name}] Column '{self.column}' not in DataFrame for Bollinger Bands")
            raise ValueError(f"Missing '{self.column}' for Bollinger Bands")

        try:
            rolling_mean = df[self.column].rolling(window=self.window, min_periods=1).mean()
            rolling_std = df[self.column].rolling(window=self.window, min_periods=1).std()

            df["bollinger_upper"] = rolling_mean + (rolling_std * self.window_dev)
            df["bollinger_lower"] = rolling_mean - (rolling_std * self.window_dev)
            df["bollinger_mid"] = rolling_mean

            self.logger.info(f"[{script_name}] Bollinger Bands added successfully.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate Bollinger Bands: {e}", exc_info=True)

        # Handle NaNs in Bollinger Bands
        for col in ["bollinger_upper", "bollinger_lower", "bollinger_mid"]:
            self._handle_nans(df, col)

        return df

    def _handle_nans(self, df: pd.DataFrame, column: str):
        nan_count = df[column].isna().sum()
        if nan_count > 0:
            self.logger.warning(f"[{script_name}] Column '{column}' has {nan_count} NaN values.")
            if self.handle_nans == 'drop':
                df.dropna(subset=[column], inplace=True)
                self.logger.info(f"[{script_name}] Dropped NaNs in '{column}'.")
            elif self.handle_nans == 'ffill':
                df[column].fillna(method='ffill', inplace=True)
                self.logger.info(f"[{script_name}] Forward filled NaNs in '{column}'.")
            elif self.handle_nans == 'warn':
                self.logger.warning(f"[{script_name}] '{column}' still has NaNs.")
            else:
                self.logger.warning(f"[{script_name}] Unknown NaN strategy '{self.handle_nans}'. No action taken.")

# You can add more custom indicators following the above pattern

# -------------------------------------------------------------------
# Indicator Pipeline
# -------------------------------------------------------------------
class IndicatorPipeline:
    def __init__(self, indicators: Optional[List[Indicator]] = None, logger: Optional[logging.Logger] = None):
        self.indicators = indicators or []
        self.logger = logger or logging.getLogger(__class__.__name__)
        self.logger.info(f"[{script_name}] Initialized IndicatorPipeline with {len(self.indicators)} indicators.")

    def add_indicator(self, indicator: Indicator):
        self.indicators.append(indicator)
        self.logger.info(f"[{script_name}] Added indicator: {indicator.__class__.__name__}")

    def remove_indicator(self, indicator_cls):
        before_count = len(self.indicators)
        self.indicators = [ind for ind in self.indicators if not isinstance(ind, indicator_cls)]
        after_count = len(self.indicators)
        self.logger.info(f"[{script_name}] Removed {indicator_cls.__name__} from pipeline. {before_count}->{after_count}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for indicator in self.indicators:
            self.logger.info(f"[{script_name}] Applying {indicator.__class__.__name__}")
            try:
                df = indicator.apply(df)
            except Exception as e:
                self.logger.error(f"[{script_name}] Failed to apply {indicator.__class__.__name__}: {e}", exc_info=True)
        return df

# -------------------------------------------------------------------
# CustomIndicators Class Definition
# -------------------------------------------------------------------
class CustomIndicators:
    """
    Encapsulates all custom indicators and provides a composable interface to apply them.
    """
    _cache: Dict[str, pd.Series] = {}

    def __init__(self, db_handler: Optional[DatabaseHandler] = None, config_manager: Optional[ConfigManager] = None, logger: Optional[logging.Logger] = None):
        self.db_handler = db_handler
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        if self.config_manager and self.db_handler:
            self.data_store = DataStore(config=self.config_manager, logger=self.logger)
            cache_dir = Path(self.config_manager.get('CACHE_DIRECTORY', project_root / 'data' / 'cache'))
            self.cache_path = cache_dir
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"[{script_name}] Initialized for SQL mode. Cache: {self.cache_path}")
        else:
            self.data_store = None
            self.cache_path = project_root / 'data' / 'cache'
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"[{script_name}] Initialized without DataStore integration. Cache: {self.cache_path}")

    @staticmethod
    def get_cache_key(df: pd.DataFrame, function_name: str, args: Tuple, kwargs: Dict) -> str:
        df_index_hash = joblib.hash(tuple(df.index))
        args_hash = joblib.hash((args, frozenset(kwargs.items())))
        return f"{function_name}_{df_index_hash}_{args_hash}"

    def memory_cache(self, key: str, function: Callable, *args: Any, **kwargs: Any) -> pd.Series:
        if key not in self._cache:
            self._cache[key] = function(*args, **kwargs)
            self.logger.debug(f"[{script_name}] Cached result in memory for key={key}")
        else:
            self.logger.debug(f"[{script_name}] Using cached in-memory result for key={key}")
        return self._cache[key]

    def file_cache(self, key: str, function: Callable, *args: Any, **kwargs: Any) -> pd.Series:
        cache_file = self.cache_path / f"{key}.pkl"
        if cache_file.exists():
            self.logger.info(f"[{script_name}] Loading from file cache: {cache_file}")
            return joblib.load(cache_file)

        result = function(*args, **kwargs)
        joblib.dump(result, cache_file)
        self.logger.info(f"[{script_name}] Cached result to file: {cache_file}")
        return result

    def cached_indicator_function(self, df: pd.DataFrame, indicator_function: Callable, *args: Any, cache_strategy: str = 'memory', **kwargs: Any) -> pd.Series:
        cache_key = self.get_cache_key(df, indicator_function.__name__, args, kwargs)
        self.logger.debug(f"[{script_name}] Cache key={cache_key}")

        # Parse cache_strategy if it contains additional parameters after '#'
        if '#' in cache_strategy:
            cache_strategy = cache_strategy.split('#')[0].strip().lower()
        else:
            cache_strategy = cache_strategy.strip().lower()

        if cache_strategy == 'memory':
            return self.memory_cache(cache_key, indicator_function, df, *args, **kwargs)
        elif cache_strategy == 'file':
            return self.file_cache(cache_key, indicator_function, df, *args, **kwargs)
        else:
            self.logger.error(f"[{script_name}] Unknown cache strategy: {cache_strategy}")
            raise ValueError(f"Unknown cache strategy: {cache_strategy}")

    def add_custom_indicator(self, df: pd.DataFrame, indicator_name: str, indicator_function: Callable, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame.")
        if not callable(indicator_function):
            raise ValueError("'indicator_function' must be callable.")
        if not indicator_name or not isinstance(indicator_name, str):
            raise ValueError("'indicator_name' must be a non-empty string.")

        self.logger.info(f"[{script_name}] Adding custom indicator '{indicator_name}'")
        start_time = timer()

        try:
            if self.config_manager:
                indicator_params = self.config_manager.get('INDICATORS', {}).get(indicator_name, {})
                if isinstance(indicator_params, str):
                    try:
                        indicator_params = eval(indicator_params)
                    except Exception as e:
                        self.logger.error(f"[{script_name}] Could not eval indicator params for '{indicator_name}': {e}")
                        raise ValueError(f"Invalid indicator params for '{indicator_name}'")
                kwargs.update(indicator_params)
                cache_strategy = kwargs.pop('cache_strategy', self.config_manager.get('CACHE_STRATEGY', 'memory'))
            else:
                cache_strategy = 'memory'

            df[indicator_name] = self.cached_indicator_function(df, indicator_function, *args, cache_strategy=cache_strategy, **kwargs)
        except Exception as e:
            self.logger.error(f"[{script_name}] Error in custom indicator '{indicator_name}': {e}")
            raise RuntimeError(f"Error in custom indicator '{indicator_name}': {e}")

        end_time = timer()
        self.logger.info(f"[{script_name}] Added '{indicator_name}' in {end_time - start_time:.2f}s")
        return df

    def process_and_save_indicators(self, symbol: str, indicators: Dict[str, Callable], **kwargs: Any) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Loading data for {symbol} from SQL.")
        if not self.data_store:
            self.logger.error(f"[{script_name}] DataStore not initialized. Cannot load data.")
            raise ValueError("DataStore is not initialized.")

        df = self.data_store.load_data(symbol)
        if df is None or df.empty:
            self.logger.error(f"[{script_name}] No data for {symbol}. Cannot compute indicators.")
            raise ValueError(f"No data available for {symbol}.")

        # **Process** the DataFrame using ColumnUtils
        try:
            column_config_path = project_root / 'src' / 'Utilities' / 'column_config.json'
            # Define required columns for custom indicators
            required_columns = [
                'close',
                'high',
                'low',
                'macd_line',
                'macd_signal',
                'macd_histogram',
                'rsi',
                'bollinger_width',
                'date'
            ]
            df = ColumnUtils.process_dataframe(
                df,
                config_path=column_config_path,
                required_columns=required_columns,
                logger=self.logger
            )
        except (KeyError, FileNotFoundError, ValueError) as ve:
            self.logger.error(f"[{script_name}] Data processing failed: {ve}")
            return df

        self.logger.info(f"[{script_name}] Loaded {len(df)} records for {symbol}. Applying custom indicators...")

        for name, func in indicators.items():
            df = self.add_custom_indicator(df, name, func, **kwargs)

        self.logger.debug(f"[{script_name}] 'date' dtype after custom indicators: {df['date'].dtype}")
        self.logger.debug(f"[{script_name}] Sample 'date' values:\n{df['date'].head()}")

        self.data_store.save_data(df, symbol, overwrite=True)
        self.logger.info(f"[{script_name}] Indicators processed/saved for {symbol}.")
        return df

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Applying custom indicators to DataFrame...")

        # **Process** the DataFrame using ColumnUtils
        try:
            column_config_path = project_root / 'src' / 'Utilities' / 'column_config.json'
            # Define required columns for custom indicators
            required_columns = [
                'close',
                'high',
                'low',
                'macd_line',
                'macd_signal',
                'macd_histogram',
                'rsi',
                'bollinger_width',
                'date'
            ]
            df = ColumnUtils.process_dataframe(
                df,
                config_path=column_config_path,
                required_columns=required_columns,
                logger=self.logger
            )
        except (KeyError, FileNotFoundError, ValueError) as ve:
            self.logger.error(f"[{script_name}] Data processing failed: {ve}")
            return df

        # Initialize Indicator Pipeline
        pipeline = IndicatorPipeline(logger=self.logger)
        # Add desired custom indicators to the pipeline
        pipeline.add_indicator(CustomRSIIndicator(window=14, handle_nans='ffill', logger=self.logger))
        pipeline.add_indicator(CustomBollingerBandsIndicator(window=20, window_dev=2, handle_nans='ffill', logger=self.logger))
        # Add more indicators as needed

        # Apply the pipeline
        df = pipeline.apply(df)

        # Ensure all required indicator columns are present
        required_indicator_cols = [
            'rsi',
            'bollinger_upper',
            'bollinger_lower',
            'bollinger_mid',
            # Add other required indicator columns here
        ]
        missing_indicators = [col for col in required_indicator_cols if col not in df.columns]
        if missing_indicators:
            self.logger.error(f"[{script_name}] Missing indicator columns after pipeline: {missing_indicators}")
        else:
            self.logger.info(f"[{script_name}] All required indicator columns are present.")

        self.logger.info(f"[{script_name}] Completed applying custom indicators pipeline.")
        return df

    # Below are the same custom indicator functions (unchanged), but logs now prefixed with script_name
    def custom_moving_average(self, df: pd.DataFrame, window: int = 10) -> pd.Series:
        self.logger.debug(f"[{script_name}] Calculating custom MA with window={window}")
        col = 'close' if 'close' in df.columns else 'close'  # Ensuring lowercase
        return df[col].rolling(window=window, min_periods=1).mean()

    def custom_bollinger_upper(self, df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> pd.Series:
        self.logger.debug(f"[{script_name}] Calculating custom Bollinger Upper with window={window}, dev={window_dev}")
        col = 'close' if 'close' in df.columns else 'close'  # Ensuring lowercase
        rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[col].rolling(window=window, min_periods=1).std()
        return rolling_mean + (rolling_std * window_dev)

    def custom_bollinger_lower(self, df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> pd.Series:
        self.logger.debug(f"[{script_name}] Calculating custom Bollinger Lower with window={window}, dev={window_dev}")
        col = 'close' if 'close' in df.columns else 'close'  # Ensuring lowercase
        rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[col].rolling(window=window, min_periods=1).std()
        return rolling_mean - (rolling_std * window_dev)

    def custom_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        self.logger.debug(f"[{script_name}] Calculating custom RSI with window={window}")
        col = 'close' if 'close' in df.columns else 'close'  # Ensuring lowercase
        try:
            indicator = RSIIndicator(close=df[col], window=window)
            rsi = indicator.rsi()
            return rsi
        except Exception as e:
            self.logger.error(f"[{script_name}] Error calculating custom RSI: {e}", exc_info=True)
            return pd.Series([np.nan]*len(df), index=df.index)

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
def main():
    print(f"[{script_name}] Entering main() in {script_file}")
    try:
        # Initialize DatabaseHandler (Assuming it requires ConfigManager and logger)
        db_handler = DatabaseHandler(config=config_manager, logger=logger)

        # Initialize CustomIndicators
        custom_indicators = CustomIndicators(db_handler=db_handler, config_manager=config_manager, logger=logger)
        logger.info(f"[{script_name}] CustomIndicators initialized for demonstration.")

        # Define indicators to add (name: function)
        indicators = {
            'custom_rsi': custom_indicators.custom_rsi,
            'custom_bollinger_upper': custom_indicators.custom_bollinger_upper,
            'custom_bollinger_lower': custom_indicators.custom_bollinger_lower,
            'custom_ma_memory': custom_indicators.custom_moving_average  # Example of another custom indicator
        }

        # Process and save indicators for a sample symbol
        symbol = "AAPL"
        df_with_indicators = custom_indicators.process_and_save_indicators(symbol, indicators, window=14, window_dev=2)

        if df_with_indicators is not None and not df_with_indicators.empty:
            expected_cols = ['rsi', 'bollinger_upper', 'bollinger_lower', 'bollinger_mid', 'custom_ma_memory']
            missing_cols = [c for c in expected_cols if c not in df_with_indicators.columns]
            if missing_cols:
                logger.error(f"[{script_name}] Missing columns: {missing_cols}")
            else:
                print(f"\n[{script_name}] Sample custom indicators:\n", df_with_indicators[expected_cols].tail())

    except Exception as e:
        logger.error(f"[{script_name}] Error in main(): {e}", exc_info=True)
    finally:
        if 'db_handler' in locals() and db_handler:
            try:
                db_handler.close()
                logger.info(f"[{script_name}] Database connection closed.")
            except Exception as ex:
                logger.error(f"[{script_name}] Unexpected error closing DB: {ex}")

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
