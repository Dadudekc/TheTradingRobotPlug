# -------------------------------------------------------------------
# File Path: D:/TradingRobotPlug2/src/Data_Processing/Technical_Indicators/trend_indicators.py
# Description: Provides trend indicators (Moving Averages, MACD, ADX,
# Ichimoku Cloud, Parabolic SAR, and Bollinger Width). Integrates with
# the project’s ConfigManager, DatabaseHandler, ColumnUtils, and logging setup.
# All DataFrame columns are standardized to lowercase.
# -------------------------------------------------------------------

import os
import sys
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Optional
import logging

# External libs
from ta.trend import MACD, ADXIndicator, IchimokuIndicator, PSARIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# -------------------------------------------------------------------
# Identify Current Script for Logging
# -------------------------------------------------------------------
script_file = Path(__file__).resolve()
script_name = script_file.name  # e.g., "trend_indicators.py"
project_root = script_file.parents[1]

# Ensure the needed directories are on sys.path
utilities_dir = project_root / 'src' / 'Utilities'
data_processing_dir = project_root / 'src' / 'Data_Processing'
sys.path.extend([
    str(utilities_dir.resolve()),
    str(data_processing_dir.resolve()),
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
logger.info(f"[{script_name}] Logger setup successfully.")

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
    'ALPHAVANTAGE_API_KEY',
    'ALPHAVANTAGE_BASE_URL'
]

try:
    config_manager = ConfigManager(env_file=dotenv_path, required_keys=required_keys, logger=logger)
    logger.info(f"[{script_name}] ConfigManager initialized successfully.")
except KeyError as e:
    logger.error(f"[{script_name}] Missing required config keys: {e}")
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
# Specific Indicator Classes
# -------------------------------------------------------------------
class MovingAverageIndicator(Indicator):
    def __init__(self, window_size=10, ma_type="SMA", handle_nans="warn",
                 column="close", logger: Optional[logging.Logger] = None):
        self.window_size = window_size
        self.ma_type = ma_type.upper()
        self.handle_nans = handle_nans
        self.column = column
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info(f"[{script_name}] MovingAverageIndicator initialized: type={self.ma_type}, window={self.window_size}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Adding {self.ma_type} Moving Average (window={self.window_size})")
        required_cols = [self.column]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            self.logger.error(f"[{script_name}] Missing columns for Moving Average: {missing}")
            raise ValueError(f"Missing columns for Moving Average: {missing}")

        try:
            if self.ma_type == "SMA":
                ma_col = f"sma_{self.window_size}"
                df[ma_col] = df[self.column].rolling(window=self.window_size, min_periods=1).mean()
            elif self.ma_type == "EMA":
                ma_col = f"ema_{self.window_size}"
                df[ma_col] = df[self.column].ewm(span=self.window_size, adjust=False).mean()
            else:
                self.logger.error(f"[{script_name}] Unsupported moving average type: {self.ma_type}")
                raise ValueError(f"Unsupported MA type '{self.ma_type}'")

            # Handle NaNs
            nan_count = df[ma_col].isna().sum()
            if nan_count > 0:
                self.logger.warning(f"[{script_name}] {ma_col} has {nan_count} NaN values.")
                if self.handle_nans == 'drop':
                    df.dropna(subset=[ma_col], inplace=True)
                    self.logger.info(f"[{script_name}] Dropped NaNs in '{ma_col}'.")
                elif self.handle_nans == 'ffill':
                    df[ma_col].fillna(method='ffill', inplace=True)
                    self.logger.info(f"[{script_name}] Forward filled NaNs in '{ma_col}'.")
                elif self.handle_nans == 'warn':
                    self.logger.warning(f"[{script_name}] '{ma_col}' still has NaNs.")
                else:
                    self.logger.warning(f"[{script_name}] Unknown NaN strategy '{self.handle_nans}'. No action taken.")

        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate Moving Average: {e}", exc_info=True)
            df[ma_col] = 0.0

        return df

class MACDIndicatorClass(Indicator):
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, price_column="close", logger: Optional[logging.Logger] = None):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_column = price_column
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info(f"[{script_name}] MACDIndicatorClass initialized with fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Calculating MACD components")

        if self.price_column not in df.columns:
            self.logger.error(f"[{script_name}] Column '{self.price_column}' not found for MACD")
            raise ValueError(f"Missing '{self.price_column}' column for MACD")

        try:
            macd = MACD(
                close=df[self.price_column],
                window_slow=self.slow_period,
                window_fast=self.fast_period,
                window_sign=self.signal_period
            )
            df["macd"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_hist"] = macd.macd_diff()
            self.logger.info(f"[{script_name}] MACD columns added: macd, macd_signal, macd_hist")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate MACD: {e}", exc_info=True)

        return df

class ADXIndicatorClass(Indicator):
    def __init__(self, window=14, handle_nans="warn", logger: Optional[logging.Logger] = None):
        self.window = window
        self.handle_nans = handle_nans
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info(f"[{script_name}] ADXIndicatorClass initialized with window={self.window}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Adding ADX (window={self.window})")
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in df.columns:
                self.logger.error(f"[{script_name}] Missing column '{col}' for ADX")
                raise ValueError(f"Missing column '{col}' for ADX")

        try:
            adx = ADXIndicator(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=self.window
            )
            df["adx"] = adx.adx()
            df["+di"] = adx.adx_pos()
            df["-di"] = adx.adx_neg()
            self.logger.info(f"[{script_name}] ADX columns added: adx, +di, -di")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate ADX: {e}", exc_info=True)

        # Handle NaNs in ADX
        self._handle_nans(df, "adx")
        self._handle_nans(df, "+di")
        self._handle_nans(df, "-di")

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

class IchimokuCloudIndicatorClass(Indicator):
    def __init__(self, nine_period=9, twenty_six_period=26, fifty_two_period=52,
                 handle_nans="warn", logger: Optional[logging.Logger] = None):
        self.nine_period = nine_period
        self.twenty_six_period = twenty_six_period
        self.fifty_two_period = fifty_two_period
        self.handle_nans = handle_nans
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info(f"[{script_name}] IchimokuCloudIndicatorClass initialized with periods: 9={self.nine_period}, 26={self.twenty_six_period}, 52={self.fifty_two_period}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Adding Ichimoku Cloud")
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in df.columns:
                self.logger.error(f"[{script_name}] Missing column '{col}' for Ichimoku Cloud")
                raise ValueError(f"Missing column '{col}' for Ichimoku Cloud")

        try:
            ichimoku = IchimokuIndicator(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window1=self.nine_period,
                window2=self.twenty_six_period,
                window3=self.fifty_two_period
            )
            df["ichimoku_conversion_line"] = ichimoku.ichimoku_conversion_line()
            df["ichimoku_base_line"] = ichimoku.ichimoku_base_line()
            df["ichimoku_leading_span_a"] = ichimoku.ichimoku_a()
            df["ichimoku_leading_span_b"] = ichimoku.ichimoku_b()
            df["ichimoku_lagging_span"] = df["close"].shift(-self.twenty_six_period)
            self.logger.info(f"[{script_name}] Ichimoku Cloud columns added: ichimoku_conversion_line, ichimoku_base_line, ichimoku_leading_span_a, ichimoku_leading_span_b, ichimoku_lagging_span")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate Ichimoku Cloud: {e}", exc_info=True)

        # Handle NaNs in Ichimoku
        self._handle_nans(df, "ichimoku_conversion_line")
        self._handle_nans(df, "ichimoku_base_line")
        self._handle_nans(df, "ichimoku_leading_span_a")
        self._handle_nans(df, "ichimoku_leading_span_b")
        self._handle_nans(df, "ichimoku_lagging_span")

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

class PSARIndicatorClass(Indicator):
    def __init__(self, step=0.02, max_step=0.2, handle_nans="warn", logger: Optional[logging.Logger] = None):
        self.step = step
        self.max_step = max_step
        self.handle_nans = handle_nans
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info(f"[{script_name}] PSARIndicatorClass initialized with step={self.step}, max_step={self.max_step}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Adding Parabolic SAR")
        required_columns = ["high", "low", "close"]
        for col in required_columns:
            if col not in df.columns:
                self.logger.error(f"[{script_name}] Missing column '{col}' for Parabolic SAR")
                raise ValueError(f"Missing column '{col}' for Parabolic SAR")

        try:
            psar = PSARIndicator(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                step=self.step,
                max_step=self.max_step
            )
            df["psar"] = psar.psar()
            self.logger.info(f"[{script_name}] Parabolic SAR column added: psar")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate Parabolic SAR: {e}", exc_info=True)

        # Handle NaNs in PSAR
        self._handle_nans(df, "psar")

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

class BollingerWidthIndicator(Indicator):
    """
    Creates 'bollinger_width' = bollinger_hband - bollinger_lband
    using ta.volatility.BollingerBands
    """
    def __init__(self, window=20, window_dev=2.0, price_column="close", handle_nans="warn", logger: Optional[logging.Logger] = None):
        self.window = window
        self.window_dev = window_dev
        self.price_column = price_column
        self.handle_nans = handle_nans
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info(f"[{script_name}] BollingerWidthIndicator initialized with window={self.window}, dev={self.window_dev}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Adding Bollinger Bands and Bollinger Width")
        if self.price_column not in df.columns:
            self.logger.error(f"[{script_name}] Missing column '{self.price_column}' for BollingerWidthIndicator")
            raise ValueError(f"Missing column '{self.price_column}' for BollingerWidthIndicator")

        try:
            bb = BollingerBands(
                close=df[self.price_column],
                window=self.window,
                window_dev=self.window_dev
            )
            df["bollinger_hband"] = bb.bollinger_hband()
            df["bollinger_lband"] = bb.bollinger_lband()
            df["bollinger_mid"] = bb.bollinger_mavg()
            df["bollinger_width"] = df["bollinger_hband"] - df["bollinger_lband"]
            self.logger.info(f"[{script_name}] Bollinger Bands and Bollinger Width columns added.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to calculate Bollinger Bands: {e}", exc_info=True)

        # Handle NaNs in Bollinger Width
        self._handle_nans(df, "bollinger_width")

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
# TrendIndicators Class
# -------------------------------------------------------------------
class TrendIndicators:
    """
    Encapsulates all trend indicators and provides a composable interface to apply them.
    """
    def __init__(self, data_store: DataStore, pipeline: Optional[IndicatorPipeline] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__class__.__name__)
        self.pipeline = pipeline or IndicatorPipeline(logger=self.logger)
        self.data_store = data_store
        self.logger.info(f"[{script_name}] TrendIndicators instance created with data_store.")

        # Initialize and add indicators to the pipeline
        self.initialize_indicators()

    def initialize_indicators(self):
        """Initialize and add all desired trend indicators to the pipeline."""
        self.logger.info(f"[{script_name}] Initializing trend indicators.")

        try:
            self.add_indicator(MovingAverageIndicator(window_size=10, ma_type="SMA", handle_nans='ffill', logger=self.logger))
            self.add_indicator(MovingAverageIndicator(window_size=20, ma_type="EMA", handle_nans='ffill', logger=self.logger))
            self.add_indicator(MACDIndicatorClass(logger=self.logger))
            self.add_indicator(ADXIndicatorClass(window=14, handle_nans='ffill', logger=self.logger))
            self.add_indicator(IchimokuCloudIndicatorClass(nine_period=9, twenty_six_period=26, fifty_two_period=52, handle_nans='ffill', logger=self.logger))
            self.add_indicator(PSARIndicatorClass(step=0.02, max_step=0.2, handle_nans='ffill', logger=self.logger))
            self.add_indicator(BollingerWidthIndicator(window=20, window_dev=2.0, price_column="close", handle_nans='ffill', logger=self.logger))
            self.logger.info(f"[{script_name}] All trend indicators added to the pipeline.")
        except Exception as e:
            self.logger.error(f"[{script_name}] Error initializing indicators: {e}", exc_info=True)

    def add_indicator(self, indicator: Indicator):
        self.pipeline.add_indicator(indicator)

    def remove_indicator(self, indicator_cls):
        self.pipeline.remove_indicator(indicator_cls)

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Starting to apply TrendIndicators pipeline.")

        # 1) Process the DataFrame using ColumnUtils
        try:
            column_config_path = project_root / 'src' / 'Utilities' / 'column_config.json'
            # Define required columns for trend indicators
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
            self.logger.info("DataFrame processed with ColumnUtils.")
        except (KeyError, FileNotFoundError, ValueError) as ve:
            self.logger.error(f"[{script_name}] Data processing failed: {ve}")
            return df

        # 2) Apply pipeline of indicators
        df = self.pipeline.apply(df)

        # 3) Ensure MACD is calculated correctly (if not already handled by ColumnUtils)
        if 'macd_line' not in df.columns:
            self.logger.warning(f"[{script_name}] 'macd_line' missing after applying indicators. Attempting manual calculation.")
            df['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()

            if 'macd_line' not in df.columns:
                self.logger.error(f"[{script_name}] Failed to calculate 'macd_line' manually.")

        # Similarly, ensure other required columns are present or calculate manually if needed
        # (This step can be expanded based on your requirements)

        self.logger.info(f"[{script_name}] Completed applying TrendIndicators pipeline.")
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
