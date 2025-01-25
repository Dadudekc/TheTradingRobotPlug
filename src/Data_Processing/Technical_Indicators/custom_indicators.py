# -------------------------------------------------------------------
# File Path: D:\TradingRobotPlug2\src\Data_Processing\Technical_Indicators\custom_indicators.py
# Description: Provides custom technical indicators with caching support for trading algorithms,
#              integrated with PostgreSQL database.
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

# -------------------------------------------------------------------
# Identify Script Name for Logging/Print
# -------------------------------------------------------------------
script_file = Path(__file__).resolve()
script_name = script_file.name  # "custom_indicators.py"
project_root = script_file.parents[3]

print(f"[{script_name}] Script path: {script_file}")
print(f"[{script_name}] Project root: {project_root}")

# Ensure project root is in sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# -------------------------------------------------------------------
# Attempt Imports
# -------------------------------------------------------------------
try:
    from src.Utilities.config_manager import ConfigManager, setup_logging
    from src.Utilities.db.db_handler import DatabaseHandler
    from src.Utilities.data.data_store import DataStore
    from src.Utilities.db.db_connection import Session
    from src.Utilities.data.data_store import DataStore, DatabaseHandler
    print(f"[{script_name}] Imported config_manager, db_handler, data_store successfully.")
except ImportError as e:
    print(f"[{script_name}] Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------
logger = setup_logging(script_name=script_name)
logger.info(f"[{script_name}] Logger setup successfully for Custom Indicators")

# -------------------------------------------------------------------
# Configuration Setup
# -------------------------------------------------------------------
dotenv_path = project_root / '.env'
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
    config_manager = ConfigManager(env_file=dotenv_path, required_keys=required_keys, logger=logger)
    logger.info(f"[{script_name}] Loaded environment variables from {dotenv_path}")
except Exception as e:
    logger.error(f"[{script_name}] Failed to initialize ConfigManager: {e}")
    config_manager = None

# -------------------------------------------------------------------
# DataFrame Standardization Helper
# -------------------------------------------------------------------
def standardize_dataframe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    1. Flatten multi-index columns if present.
    2. Rename columns to lowercase except 'Date'.
    3. Ensure 'Date' is a column (not an index).
    4. Convert 'Date' to datetime, dropping rows with invalid or missing dates.
    """
    # 1) Flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        logger.warning(f"[{script_name}] Flattening MultiIndex columns.")
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]

    # 2) Rename columns to lowercase except 'Date'
    def transform_col(col: str) -> str:
        return 'Date' if col.lower() == 'date' else col.lower()

    original_cols = df.columns.tolist()
    df.columns = [transform_col(c) for c in df.columns]
    logger.debug(f"[{script_name}] Renamed columns from {original_cols} to {df.columns.tolist()}")

    # 3) Ensure 'Date' is a column (not an index)
    if df.index.name and df.index.name.lower() == 'date':
        logger.info(f"[{script_name}] Resetting index to move '{df.index.name}' into a column 'Date'")
        df.reset_index(inplace=True)

    # If we have 'date' but not 'Date', rename it
    if 'date' in df.columns and 'Date' not in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
        logger.debug(f"[{script_name}] Renamed 'date' -> 'Date'")

    if 'Date' not in df.columns:
        logger.warning(f"[{script_name}] No 'Date' column found after standardization.")
        return df

    # 4) Convert 'Date' to datetime and drop invalid
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    invalid_dates = df['Date'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"[{script_name}] Dropping {invalid_dates} rows with invalid 'Date'.")
        df.dropna(subset=['Date'], inplace=True)

    return df

# -------------------------------------------------------------------
# Indicator Base Class
# -------------------------------------------------------------------
class Indicator(ABC):
    """
    Abstract base class for all indicators.
    """
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# -------------------------------------------------------------------
# Specific Indicator Classes (RSI, Bollinger, etc.)
# [unchanged from your original content, but logs now prefixed with script_name]
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

        df[self.column] = pd.to_numeric(df[self.column], errors='coerce')
        initial_row_count = len(df)
        df.dropna(subset=[self.column], inplace=True)
        if len(df) < initial_row_count:
            self.logger.warning(f"[{script_name}] Dropped {initial_row_count - len(df)} rows with invalid '{self.column}' for RSI.")

        delta = df[self.column].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.window, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        self.logger.info(f"[{script_name}] Successfully added RSI")
        self._handle_nans(df, 'RSI')
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

        rolling_mean = df[self.column].rolling(window=self.window, min_periods=1).mean()
        rolling_std = df[self.column].rolling(window=self.window, min_periods=1).std()

        df["Bollinger_High"] = rolling_mean + (rolling_std * self.window_dev)
        df["Bollinger_Low"]  = rolling_mean - (rolling_std * self.window_dev)
        df["Bollinger_Mid"]  = rolling_mean

        self.logger.info(f"[{script_name}] Bollinger Bands added")
        for col in ["Bollinger_High", "Bollinger_Low", "Bollinger_Mid"]:
            self._handle_nans(df, col)
        return df

    def _handle_nans(self, df: pd.DataFrame, column: str):
        nan_count = df[column].isna().sum()
        if nan_count > 0:
            self.logger.warning(f"[{script_name}] '{column}' has {nan_count} NaNs.")
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
# IndicatorPipeline, CustomIndicators, etc. (unchanged)
# But we add calls to `standardize_dataframe()` in process_and_save_indicators() & apply_indicators()
# -------------------------------------------------------------------
class IndicatorPipeline:
    def __init__(self, indicators: Optional[List[Indicator]] = None, logger: Optional[logging.Logger] = None):
        self.indicators = indicators or []
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info(f"[{script_name}] Initialized IndicatorPipeline")

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
            df = indicator.apply(df)
        return df

class CustomIndicators:
    _cache: Dict[str, pd.Series] = {}

    def __init__(self, db_handler: Optional[DatabaseHandler] = None, config_manager: Optional[ConfigManager] = None, logger: Optional[logging.Logger] = None):
        self.db_handler = db_handler
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        if self.config_manager and self.db_handler:
            self.data_store = DataStore(config=self.config_manager, logger=self.logger)
            self.cache_path = Path(self.config_manager.get('CACHE_DIRECTORY', project_root / 'data' / 'cache'))
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

        # **Standardize** the DataFrame before indicators
        df = standardize_dataframe(df, self.logger)

        self.logger.info(f"[{script_name}] Loaded {len(df)} records for {symbol}. Applying custom indicators...")

        for name, func in indicators.items():
            df = self.add_custom_indicator(df, name, func, **kwargs)

        self.logger.debug(f"[{script_name}] 'Date' dtype after custom indicators: {df['Date'].dtype}")
        self.logger.debug(f"[{script_name}] Sample 'Date' values:\n{df['Date'].head()}")

        self.data_store.save_data(df, symbol, overwrite=True)
        self.logger.info(f"[{script_name}] Indicators processed/saved for {symbol}.")
        return df

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Applying custom indicators to DataFrame...")

        # **Standardize** first
        df = standardize_dataframe(df, self.logger)

        # Example indicators (as in your original code)
        indicators = {
            'Custom_MA_Memory': self.custom_moving_average,
            'Custom_Bollinger_Upper': self.custom_bollinger_upper,
            'Custom_Bollinger_Lower': self.custom_bollinger_lower,
            'Custom_RSI': self.custom_rsi
        }
        for name, func in indicators.items():
            df = self.add_custom_indicator(df, name, func)
        self.logger.info(f"[{script_name}] All custom indicators applied.")
        return df

    # Below are the same custom indicator functions (unchanged), but logs now have a [custom_indicators.py] prefix.
    def custom_moving_average(self, df: pd.DataFrame, window: int = 10) -> pd.Series:
        self.logger.debug(f"[{script_name}] Calculating custom MA with window={window}")
        col = 'close' if 'close' in df.columns else 'Close'
        return df[col].rolling(window=window, min_periods=1).mean()

    def custom_bollinger_upper(self, df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> pd.Series:
        self.logger.debug(f"[{script_name}] Calculating custom Bollinger Upper with window={window}, dev={window_dev}")
        col = 'close' if 'close' in df.columns else 'Close'
        mean_ = df[col].rolling(window=window, min_periods=1).mean()
        std_ = df[col].rolling(window=window, min_periods=1).std()
        return mean_ + (std_ * window_dev)

    def custom_bollinger_lower(self, df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> pd.Series:
        self.logger.debug(f"[{script_name}] Calculating custom Bollinger Lower with window={window}, dev={window_dev}")
        col = 'close' if 'close' in df.columns else 'Close'
        mean_ = df[col].rolling(window=window, min_periods=1).mean()
        std_ = df[col].rolling(window=window, min_periods=1).std()
        return mean_ - (std_ * window_dev)

    def custom_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        self.logger.debug(f"[{script_name}] Calculating custom RSI with window={window}")
        col = 'close' if 'close' in df.columns else 'Close'
        delta = df[col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
def main():
    print(f"[{script_name}] Entering main() in {script_file}")
    try:
        db_handler = DatabaseHandler(logger=logger)
        indicators = CustomIndicators(db_handler=db_handler, config_manager=config_manager, logger=logger)
        logger.info(f"[{script_name}] CustomIndicators initialized for demonstration.")

        # Simple usage with a test symbol
        symbol = "AAPL"
        custom_indicators_dict = {
            'Custom_MA_Memory': indicators.custom_moving_average,
            'Custom_Bollinger_Upper': indicators.custom_bollinger_upper,
            'Custom_Bollinger_Lower': indicators.custom_bollinger_lower,
            'Custom_RSI': indicators.custom_rsi
        }
        df_with_indicators = indicators.process_and_save_indicators(symbol, custom_indicators_dict)

        if df_with_indicators is not None and not df_with_indicators.empty:
            expected_cols = ['Custom_MA_Memory', 'Custom_Bollinger_Upper', 'Custom_Bollinger_Lower', 'Custom_RSI']
            missing_cols = [c for c in expected_cols if c not in df_with_indicators.columns]
            if missing_cols:
                logger.error(f"[{script_name}] Missing columns: {missing_cols}")
            else:
                print(f"\n[{script_name}] Sample custom indicators:\n", df_with_indicators[expected_cols].head(10))

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
