# -------------------------------------------------------------------
# File Path: D:\TradingRobotPlug2\src\Data_Processing\Technical_Indicators\momentum_indicators.py
# Description:
#     Provides composable momentum indicators such as Stochastic Oscillator, RSI,
#     Williams %R, ROC, and TRIX. Integrates with ConfigManager, DatabaseHandler,
#     and logging setup.
# -------------------------------------------------------------------

import pandas as pd
import logging
from pathlib import Path
import sys
from abc import ABC, abstractmethod
from typing import List, Optional
from ta.momentum import StochasticOscillator, RSIIndicator, WilliamsRIndicator, ROCIndicator
from ta.trend import TRIXIndicator
from dotenv import load_dotenv
import numpy as np  # for example usage

# -------------------------------------------------------------------
# Identify Script Name and Project Root
# -------------------------------------------------------------------
script_file = Path(__file__).resolve()
script_name = script_file.name  # "momentum_indicators.py"
project_root = script_file.parents[3]

print(f"[{script_name}] Current script path: {script_file}")
print(f"[{script_name}] Project root: {project_root}")

# Ensure project_root is in sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# -------------------------------------------------------------------
# Environment Variables
# -------------------------------------------------------------------
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"[{script_name}] Environment variables loaded from {env_path}")
else:
    print(f"[{script_name}] Warning: .env file not found at {env_path}")

# Add necessary directories to sys.path (if needed)
utilities_dir = project_root / 'src' / 'Utilities'
scripts_dir = project_root / 'src'
data_processing_dir = scripts_dir / 'Data_Processing'

sys.path.extend([
    str(utilities_dir.resolve()),
    str(scripts_dir.resolve()),
    str(data_processing_dir.resolve())
])

# -------------------------------------------------------------------
# Attempt Imports
# -------------------------------------------------------------------
try:
    from Utilities.db.db_connection import Session
    from Utilities.config_manager import ConfigManager, setup_logging
    print(f"[{script_name}] Successfully imported config_manager and db_connection.")
except ImportError as e:
    print(f"[{script_name}] Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
log_dir = project_root / 'logs' / 'technical_indicators'
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logging(script_name=script_name, log_dir=log_dir)
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
    'ALPHAVANTAGE_BASE_URL'
]
try:
    config_manager = ConfigManager(env_file=env_path, required_keys=required_keys, logger=logger)
    logger.info(f"[{script_name}] ConfigManager initialized successfully.")
except KeyError as e:
    logger.error(f"[{script_name}] Missing required configuration keys: {e}")
    sys.exit(1)

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
    # 1) Flatten multi-index
    if isinstance(df.columns, pd.MultiIndex):
        logger.warning(f"[{script_name}] Flattening MultiIndex columns.")
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]

    # 2) Rename columns
    def transform_col(col: str) -> str:
        return 'Date' if col.lower() == 'date' else col.lower()

    original_cols = df.columns.tolist()
    df.columns = [transform_col(c) for c in df.columns]
    logger.debug(f"[{script_name}] Renamed columns from {original_cols} to {df.columns.tolist()}")

    # 3) Ensure 'Date' is a column
    if df.index.name and df.index.name.lower() == 'date':
        logger.info(f"[{script_name}] Resetting index to make '{df.index.name}' a real column 'Date'")
        df.reset_index(inplace=True)

    # If we still have 'date' but not 'Date', rename
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
        return df

class RSIIndicatorClass(Indicator):
    def __init__(self, window: int = 14, logger: Optional[logging.Logger] = None):
        self.window = window
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"[{script_name}] RSIIndicatorClass initialized with window={self.window}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Applying RSI (window={self.window})")
        if 'close' not in df.columns:
            self.logger.error(f"[{script_name}] 'close' column missing for RSI")
            raise ValueError("Missing 'close' column for RSI")

        indicator = RSIIndicator(close=df['close'], window=self.window)
        df['rsi'] = indicator.rsi()
        self.logger.info(f"[{script_name}] RSI applied successfully.")
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

        indicator = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=self.lbp)
        df['williams_r'] = indicator.williams_r()
        self.logger.info(f"[{script_name}] Williams %R applied successfully.")
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

        indicator = ROCIndicator(close=df['close'], window=self.window)
        df['roc'] = indicator.roc()
        self.logger.info(f"[{script_name}] ROC applied successfully.")
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

        trix_ind = TRIXIndicator(close=df['close'], window=self.window, fillna=False)
        df['trix'] = trix_ind.trix()

        # If ta.trend.TRIXIndicator has trix_signal() method
        if hasattr(trix_ind, 'trix_signal'):
            df['trix_signal'] = trix_ind.trix_signal()
            self.logger.info(f"[{script_name}] TRIX_Signal applied from built-in 'trix_signal()'")
        else:
            # fallback if the version doesn't have trix_signal
            self.logger.warning(f"[{script_name}] 'trix_signal()' not found, computing manually via EMA")
            df['trix_signal'] = df['trix'].ewm(span=self.signal_window, adjust=False).mean()

        self.logger.info(f"[{script_name}] TRIX applied successfully.")
        return df

# -------------------------------------------------------------------
# Indicator Pipeline
# -------------------------------------------------------------------
class IndicatorPipeline:
    def __init__(self, indicators: Optional[List[Indicator]] = None, logger: Optional[logging.Logger] = None):
        self.indicators = indicators or []
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"[{script_name}] Initialized IndicatorPipeline")

    def add_indicator(self, indicator: Indicator):
        self.indicators.append(indicator)
        self.logger.info(f"[{script_name}] Added {indicator.__class__.__name__}")

    def remove_indicator(self, indicator_cls):
        before = len(self.indicators)
        self.indicators = [i for i in self.indicators if not isinstance(i, indicator_cls)]
        after = len(self.indicators)
        self.logger.info(f"[{script_name}] Removed {indicator_cls.__name__}: {before}->{after}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        for indicator in self.indicators:
            self.logger.info(f"[{script_name}] Applying {indicator.__class__.__name__}")
            df = indicator.apply(df)
        return df

# -------------------------------------------------------------------
# MomentumIndicators Class Definition
# -------------------------------------------------------------------
class MomentumIndicators:
    """
    Encapsulates all momentum indicators and provides a composable interface to apply them.
    """

    def __init__(self, logger: logging.Logger, pipeline: Optional[IndicatorPipeline] = None):
        self.logger = logger
        self.pipeline = pipeline or IndicatorPipeline(logger=self.logger)
        self.logger.info(f"[{script_name}] MomentumIndicators instance created")

    def add_indicator(self, indicator: Indicator):
        self.pipeline.add_indicator(indicator)

    def remove_indicator(self, indicator_cls):
        self.pipeline.remove_indicator(indicator_cls)

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Starting to apply momentum indicators pipeline")

        # 1) Standardize the DataFrame
        df = standardize_dataframe(df, self.logger)

        # 2) Apply pipeline of indicators
        df = self.pipeline.apply(df)

        self.logger.info(f"[{script_name}] Completed applying momentum indicators pipeline")
        return df

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Create a sample DataFrame
        df_sample = pd.DataFrame({
            'Date': pd.date_range("2020-01-01", periods=100, freq='D'),
            'Close': np.random.uniform(100, 200, size=100),
            'High': np.random.uniform(105, 205, size=100),
            'Low': np.random.uniform(95, 195, size=100)
        })
        # Optionally set 'Date' as the index to simulate real data
        df_sample.set_index('Date', inplace=True)

        # Initialize MomentumIndicators with some default indicators
        momentum = MomentumIndicators(logger=logger)
        momentum.add_indicator(StochasticOscillatorIndicator(window=14, smooth_window=3, logger=logger))
        momentum.add_indicator(RSIIndicatorClass(window=14, logger=logger))
        momentum.add_indicator(WilliamsRIndicatorClass(lbp=14, logger=logger))
        momentum.add_indicator(ROCIndicatorClass(window=12, logger=logger))
        momentum.add_indicator(TRIXIndicatorClass(window=15, signal_window=9, logger=logger))

        # Apply the pipeline
        df_with_momentum = momentum.apply_indicators(df_sample)

        print(f"\n[{script_name}] Final DF with momentum indicators:\n", df_with_momentum.tail())

    except Exception as e:
        logger.error(f"[{script_name}] Error applying momentum indicators: {e}", exc_info=True)
