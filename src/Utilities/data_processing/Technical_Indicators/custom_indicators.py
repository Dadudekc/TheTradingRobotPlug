"""
File: custom_indicators.py
Path: D:/TradingRobotPlug2/src/Utilities/data_processing/Technical_Indicators/custom_indicators.py

Description:
    Provides custom technical indicators with caching support for trading algorithms,
    integrated with a PostgreSQL database. All DataFrame columns are standardized to lowercase.
"""

import os
import sys
from pathlib import Path
import logging
import pandas as pd
import joblib
from time import time as timer
from typing import Callable, Any, Dict, Tuple, List, Optional
from abc import ABC, abstractmethod

# Import BaseIndicator from the relative location.
from Utilities.data_processing.base_indicators import BaseIndicator

# Import technical analysis library indicators.
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import TRIXIndicator

# -------------------------------------------------------------------
# Script Setup
# -------------------------------------------------------------------
script_file = Path(__file__).resolve()
script_name = script_file.name  # "custom_indicators.py"
project_root = script_file.parents[4]

print(f"[{script_name}] Script path: {script_file}")
print(f"[{script_name}] Project root: {project_root}")

# Ensure project_root is in sys.path.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# (Optionally) Extend sys.path for Utilities and Data_Processing directories.
for subdir in [project_root / 'src' / 'Utilities', project_root / 'src', project_root / 'src' / 'Data_Processing']:
    resolved = str(subdir.resolve())
    if resolved not in sys.path:
        sys.path.append(resolved)

# Load environment variables from the .env file.
env_path = project_root / '.env'
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)
    print(f"[{script_name}] Environment variables loaded from {env_path}")
else:
    print(f"[{script_name}] Warning: .env file not found at {env_path}")

# -------------------------------------------------------------------
# Import Required Modules from Utilities
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
# Logger and Configuration Setup
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

required_keys = [
    'POSTGRES_HOST',
    'POSTGRES_DBNAME',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_PORT',
    'CACHE_STRATEGY',
    'CACHE_DIRECTORY',
    # Add additional required keys as needed.
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
    """Abstract base class for all indicators."""
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# -------------------------------------------------------------------
# Specific Custom Indicator Classes
# -------------------------------------------------------------------
class CustomRSIIndicator(Indicator):
    def __init__(self, window: int = 14, handle_nans: str = "warn", column: str = "close",
                 logger: Optional[logging.Logger] = None):
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
    def __init__(self, window: int = 20, window_dev: int = 2, handle_nans: str = "warn", column: str = "close",
                 logger: Optional[logging.Logger] = None):
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
        # Handle NaNs for each Bollinger column.
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

# -------------------------------------------------------------------
# Indicator Pipeline
# -------------------------------------------------------------------
class IndicatorPipeline:
    def __init__(self, indicators: Optional[List[Indicator]] = None, logger: Optional[logging.Logger] = None):
        self.indicators = indicators or []
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"[{script_name}] Initialized IndicatorPipeline with {len(self.indicators)} indicators.")

    def add_indicator(self, indicator: Indicator):
        self.indicators.append(indicator)
        self.logger.info(f"[{script_name}] Added indicator: {indicator.__class__.__name__}")

    def remove_indicator(self, indicator_cls):
        before_count = len(self.indicators)
        self.indicators = [ind for ind in self.indicators if not isinstance(ind, indicator_cls)]
        after_count = len(self.indicators)
        self.logger.info(f"[{script_name}] Removed {indicator_cls.__name__} from pipeline. {before_count} -> {after_count}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for indicator in self.indicators:
            self.logger.info(f"[{script_name}] Applying {indicator.__class__.__name__}")
            try:
                df = indicator.apply(df)
            except Exception as e:
                self.logger.error(f"[{script_name}] Failed to apply {indicator.__class__.__name__}: {e}", exc_info=True)
        return df

# -------------------------------------------------------------------
# CustomIndicators Class
# -------------------------------------------------------------------
class CustomIndicators(BaseIndicator):
    """
    Implements custom technical indicators with caching support.
    """
    def __init__(self, data_store: DataStore, config_manager: ConfigManager, logger: logging.Logger):
        """
        Initialize CustomIndicators with data_store, config_manager, and logger.
        """
        self.data_store = data_store
        self.config_manager = config_manager
        self.logger = logger
        
        try:
            # Initialize ColumnUtils
            self.column_utils = ColumnUtils()
            self.logger.info(f"[{script_name}] CustomIndicators initialized for demonstration.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Error initializing CustomIndicators: {e}", exc_info=True)
            raise

    def process_and_save_indicators(self, symbol: str, indicators: dict, window: int = 14, window_dev: float = 2.0) -> pd.DataFrame:
        """
        Process and save custom indicators for a given symbol.
        """
        self.logger.info(f"[{script_name}] Loading data for {symbol} from SQL.")
        try:
            # Load data using DataStore
            df = self.data_store.load_data(symbol)
            if df is None or df.empty:
                self.logger.error(f"[{script_name}] No data found for symbol {symbol}")
                return None

            # Process DataFrame using ColumnUtils
            try:
                df = self.column_utils.process_dataframe(df, stage="pre")
                self.logger.info(f"[{script_name}] DataFrame processed with ColumnUtils.")
            except Exception as e:
                self.logger.error(f"[{script_name}] Data processing failed: {e}")
                return None

            # Apply each indicator
            for indicator_name, indicator_func in indicators.items():
                try:
                    df = self.add_custom_indicator(
                        df, 
                        indicator_name=indicator_name,
                        indicator_function=indicator_func,
                        window=window,
                        window_dev=window_dev
                    )
                except Exception as e:
                    self.logger.error(f"[{script_name}] Failed to add indicator {indicator_name}: {e}")

            # Save processed data
            self.data_store.save_data(df, symbol, overwrite=True)
            self.logger.info(f"[{script_name}] Successfully saved processed data for {symbol}")

            return df

        except Exception as e:
            self.logger.error(f"[{script_name}] Error processing indicators: {e}", exc_info=True)
            return None

    def close(self):
        """
        Clean up resources.
        """
        try:
            # DataStore doesn't need explicit closing as it handles connections internally
            self.logger.info(f"[{script_name}] CustomIndicators cleanup completed.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Error during cleanup: {e}", exc_info=True)

    def custom_rsi(self, df: pd.DataFrame, window: int = 14, **kwargs) -> pd.Series:
        """
        Calculate custom RSI.
        """
        try:
            indicator = CustomRSIIndicator(window=window, logger=self.logger)
            df = indicator.apply(df)
            return df['rsi']
        except Exception as e:
            self.logger.error(f"[{script_name}] Error calculating custom RSI: {e}")
            return pd.Series(index=df.index)

    def custom_bollinger_bands(self, df: pd.DataFrame, window: int = 20, window_dev: int = 2, **kwargs) -> pd.DataFrame:
        """
        Calculate custom Bollinger Bands.
        """
        try:
            indicator = CustomBollingerBandsIndicator(
                window=window, 
                window_dev=window_dev, 
                logger=self.logger
            )
            return indicator.apply(df)
        except Exception as e:
            self.logger.error(f"[{script_name}] Error calculating Bollinger Bands: {e}")
            return df

    def custom_moving_average(self, df: pd.DataFrame, window: int = 10, **kwargs) -> pd.Series:
        """
        Calculate custom moving average.
        """
        self.logger.debug(f"[{script_name}] Calculating custom MA with window={window}")
        try:
            return df['close'].rolling(window=window, min_periods=1).mean().astype('float32')
        except Exception as e:
            self.logger.error(f"[{script_name}] Error calculating Moving Average: {e}")
            return pd.Series(index=df.index)

    # --- Caching Utilities ---
    _cache: Dict[str, pd.Series] = {}

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
        # Define the cache directory from config.
        cache_dir = Path(self.config_manager.get('CACHE_DIRECTORY', 'cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{key}.pkl"
        if cache_file.exists():
            self.logger.info(f"[{script_name}] Loading from file cache: {cache_file}")
            return joblib.load(cache_file)
        result = function(*args, **kwargs)
        joblib.dump(result, cache_file)
        self.logger.info(f"[{script_name}] Cached result to file: {cache_file}")
        return result

    def cached_indicator_function(self, df: pd.DataFrame, indicator_function: Callable, *args: Any,
                                    cache_strategy: str = 'memory', **kwargs: Any) -> pd.Series:
        cache_key = self.get_cache_key(df, indicator_function.__name__, args, kwargs)
        self.logger.debug(f"[{script_name}] Cache key={cache_key}")
        cache_strategy = cache_strategy.split('#')[0].strip().lower() if '#' in cache_strategy else cache_strategy.strip().lower()
        if cache_strategy == 'memory':
            return self.memory_cache(cache_key, indicator_function, df, *args, **kwargs)
        elif cache_strategy == 'file':
            return self.file_cache(cache_key, indicator_function, df, *args, **kwargs)
        else:
            self.logger.error(f"[{script_name}] Unknown cache strategy: {cache_strategy}")
            raise ValueError(f"Unknown cache strategy: {cache_strategy}")

    def add_custom_indicator(self, df: pd.DataFrame, indicator_name: str, indicator_function: Callable,
                             *args: Any, **kwargs: Any) -> pd.DataFrame:
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

            df[indicator_name] = self.cached_indicator_function(df, indicator_function, *args,
                                                                  cache_strategy=cache_strategy, **kwargs)
        except Exception as e:
            self.logger.error(f"[{script_name}] Error in custom indicator '{indicator_name}': {e}")
            raise RuntimeError(f"Error in custom indicator '{indicator_name}': {e}")

        end_time = timer()
        self.logger.info(f"[{script_name}] Added '{indicator_name}' in {end_time - start_time:.2f}s")
        return df

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
def main():
    print(f"[{script_name}] Entering main() in {script_file}")
    try:
        # Initialize CustomIndicators
        custom_indicators = CustomIndicators(data_store=data_store, config_manager=config_manager, logger=logger)
        logger.info(f"[{script_name}] CustomIndicators initialized for demonstration.")

        # Define indicators to add
        indicators = {
            'custom_rsi': custom_indicators.custom_rsi,
            'custom_bollinger_bands': custom_indicators.custom_bollinger_bands,
            'ma_10': custom_indicators.custom_moving_average  # Renamed for clarity
        }

        # Process and save indicators for a sample symbol
        symbol = "AAPL"
        df_with_indicators = custom_indicators.process_and_save_indicators(
            symbol=symbol,
            indicators=indicators,
            window=14,
            window_dev=2
        )

        if df_with_indicators is not None and not df_with_indicators.empty:
            # Update expected columns to match new indicator names
            expected_cols = ['rsi', 'bollinger_upper', 'bollinger_lower', 'bollinger_mid', 'ma_10']
            missing_cols = [c for c in expected_cols if c not in df_with_indicators.columns]
            if missing_cols:
                logger.error(f"[{script_name}] Missing columns: {missing_cols}")
            else:
                print(f"\n[{script_name}] Sample custom indicators:\n", df_with_indicators[expected_cols].tail())
    except Exception as e:
        logger.error(f"[{script_name}] Error in main(): {e}", exc_info=True)
    finally:
        if 'custom_indicators' in locals() and custom_indicators:
            custom_indicators.close()

if __name__ == "__main__":
    main()
