# -------------------------------------------------------------------
# File: volatility_indicators.py
# Location: src/Data_Processing/Technical_Indicators
# Description: Provides volatility indicators using pandas-ta instead
#              of TA-Lib, including Bollinger Bands, Standard Deviation,
#              Historical Volatility, Chandelier Exit, Keltner Channel,
#              and Moving Average Envelope.
# Optimizations:
#              - Data type downcasting for memory efficiency
#              - Chunked processing for large datasets
#              - Streaming data processing with sliding window
#              - Parallel processing using multiprocessing
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

# Replace TA-Lib with pandas-ta
import pandas_ta as ta

# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------
# Add project root to the Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
utilities_dir = project_root / "src" / "Utilities"
sys.path.append(str(utilities_dir))

# -------------------------------------------------------------------
# Import Utility Scripts
# -------------------------------------------------------------------
try:
    from config_manager import ConfigManager, setup_logging
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Configuration and Utilities Initialization
# -------------------------------------------------------------------
dotenv_path = project_root / '.env'
required_keys = [
    'POSTGRES_HOST', 'POSTGRES_DBNAME', 'POSTGRES_USER',
    'POSTGRES_PASSWORD', 'POSTGRES_PORT'
]
config_manager = ConfigManager(env_file=dotenv_path, required_keys=required_keys)

log_dir = project_root / 'logs' / 'technical_indicators'
log_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logging(script_name="volatility_indicators_pandas_ta", log_dir=log_dir)
logger.info("Logger initialized for volatility indicators (pandas-ta).")

# -------------------------------------------------------------------
# VolatilityIndicators Class Definition (Using pandas-ta)
# -------------------------------------------------------------------
class VolatilityIndicators:
    def __init__(self, logger: logging.Logger = None):
        """
        Initializes the VolatilityIndicators class with a logger.

        Args:
            logger (logging.Logger): Logger instance for logging. Defaults to None.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("VolatilityIndicators (pandas-ta) initialized.")

    def downcast_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downcasts numerical columns to reduce memory usage.

        Args:
            df (pd.DataFrame): The original DataFrame.

        Returns:
            pd.DataFrame: The downcasted DataFrame.
        """
        self.logger.info("Downcasting DataFrame to optimize memory usage.")
        float_cols = df.select_dtypes(include=['float64']).columns
        int_cols = df.select_dtypes(include=['int64']).columns

        df[float_cols] = df[float_cols].astype('float32')
        df[int_cols] = df[int_cols].astype('int32')

        self.logger.info("Downcasting completed.")
        return df

    def add_bollinger_bands(self, df, window_size=20, std_multiplier=2, user_defined_window=None):
        """
        Adds Bollinger Bands (BBL, BBM, BBU) to the DataFrame using pandas-ta,
        then renames columns to fixed names for consistent access.
        """
        self.logger.info("Adding Bollinger Bands via pandas-ta")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        length = user_defined_window if user_defined_window is not None else window_size

        # Calculate Bollinger Bands with pandas-ta
        bb = ta.bbands(df['close'], length=length, std=std_multiplier)

        if bb is None or bb.empty:
            raise ValueError("pandas-ta did not return Bollinger Band columns. Check your data or parameters.")

        # Rename columns to BBL, BBM, BBU
        bb_cols = list(bb.columns)  # e.g., ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0']
        if len(bb_cols) >= 3:
            bb_renamed = bb.rename(columns={
                bb_cols[0]: "BBL",
                bb_cols[1]: "BBM",
                bb_cols[2]: "BBU"
            })
        else:
            raise ValueError("Unexpected number of columns returned by ta.bbands")

        # Assign final columns to the main DataFrame
        df['Bollinger_Low'] = bb_renamed['BBL']
        df['Bollinger_Mid'] = bb_renamed['BBM']
        df['Bollinger_High'] = bb_renamed['BBU']

        self.logger.info("Successfully added Bollinger Bands (pandas-ta)")
        return df

    def add_standard_deviation(self, df, window_size=20, user_defined_window=None):
        """
        Adds Standard Deviation to the DataFrame using pandas-ta.
        This indicator returns a single-column Series, so we assign it directly.
        """
        self.logger.info("Adding Standard Deviation via pandas-ta")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        length = user_defined_window if user_defined_window is not None else window_size
        std_series = ta.stdev(df['close'], length=length)

        if std_series is None or std_series.empty:
            raise ValueError("pandas-ta did not return a Standard Deviation result. Check your data or parameters.")

        df['Standard_Deviation'] = std_series

        self.logger.info("Successfully added Standard Deviation (pandas-ta)")
        return df

    def add_historical_volatility(self, df, window=20, user_defined_window=None):
        """
        Adds Historical Volatility to the DataFrame.
        Calculated as the rolling standard deviation of logarithmic returns.

        Args:
            df (pd.DataFrame): DataFrame with 'close' prices.
            window (int): Rolling window size.
            user_defined_window (int, optional): Custom window size.

        Returns:
            pd.DataFrame: DataFrame with 'Historical_Volatility' column added.
        """
        self.logger.info("Adding Historical Volatility via custom implementation")
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        length = user_defined_window if user_defined_window is not None else window

        # Calculate logarithmic returns
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))

        # Calculate rolling standard deviation of log returns
        hv_series = df['Log_Returns'].rolling(window=length, min_periods=1).std() * np.sqrt(252)  # Annualized

        if hv_series is None or hv_series.empty:
            raise ValueError("Failed to calculate Historical Volatility. Check your data or parameters.")

        df['Historical_Volatility'] = hv_series

        # Drop the 'Log_Returns' column as it's intermediate
        df.drop(['Log_Returns'], axis=1, inplace=True)

        self.logger.info("Successfully added Historical Volatility (custom implementation)")
        return df

    def add_chandelier_exit(self, df, window=22, multiplier=3, user_defined_window=None, user_defined_multiplier=None):
        """
        Adds Chandelier Exit to the DataFrame.
        The Chandelier Exit is calculated using the highest high and ATR:
            Long Exit = Highest High - ATR * multiplier
        """
        self.logger.info("Adding Chandelier Exit (manual implementation)")
        required_columns = ['high', 'low', 'close']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain: {', '.join(required_columns)}")

        # Use user-defined parameters if provided
        length = user_defined_window if user_defined_window is not None else window
        atr_mult = user_defined_multiplier if user_defined_multiplier is not None else multiplier

        # Calculate Highest High
        df['Highest_High'] = df['high'].rolling(window=length, min_periods=1).max()

        # Calculate ATR (Average True Range)
        df['True_Range'] = df[['high', 'low', 'close']].apply(
            lambda row: max(
                row['high'] - row['low'],  # High - Low
                abs(row['high'] - row['close']),  # High - Previous Close
                abs(row['low'] - row['close'])  # Low - Previous Close
            ), axis=1
        )
        df['ATR'] = df['True_Range'].rolling(window=length, min_periods=1).mean()

        # Calculate Chandelier Exit (Long)
        df['Chandelier_Exit_Long'] = df['Highest_High'] - (atr_mult * df['ATR'])

        # Drop intermediate columns if desired
        df.drop(['Highest_High', 'True_Range', 'ATR'], axis=1, inplace=True)

        self.logger.info("Successfully added Chandelier Exit (manual implementation)")
        return df

    def add_keltner_channel(self, df, window=20, multiplier=2, user_defined_window=None, user_defined_multiplier=None):
        """
        Adds Keltner Channel to the DataFrame using pandas-ta (kc),
        then renames them to consistent columns.
        """
        self.logger.info("Adding Keltner Channel via pandas-ta")
        required_columns = ['high', 'low', 'close']
        if not all(column in df.columns for column in required_columns):
            raise ValueError(f"DataFrame must contain: {', '.join(required_columns)}")

        length = user_defined_window if user_defined_window is not None else window
        scalar = user_defined_multiplier if user_defined_multiplier is not None else multiplier

        kc = ta.kc(
            high=df['high'], low=df['low'], close=df['close'],
            length=length, scalar=scalar
        )
        if kc is None or kc.empty:
            raise ValueError("pandas-ta did not return Keltner Channel columns. Check your data or parameters.")

        # Typically returns columns like KCL_20_2.0, KCB_20_2.0, KCU_20_2.0
        kc_cols = list(kc.columns)
        kc_renamed = kc.rename(columns={
            kc_cols[0]: "KCL",
            kc_cols[1]: "KCB",
            kc_cols[2]: "KCU"
        })

        df['Keltner_Channel_Low'] = kc_renamed['KCL']
        df['Keltner_Channel_Basis'] = kc_renamed['KCB']
        df['Keltner_Channel_High'] = kc_renamed['KCU']

        self.logger.info("Successfully added Keltner Channel (pandas-ta)")
        return df

    def add_moving_average_envelope(self, df, window_size=10, percentage=0.025, user_defined_window=None, user_defined_percentage=None):
        """
        Adds a Moving Average Envelope (MAE) using pandas-ta's SMA.
        MAE consists of an upper and lower band around the SMA, based on the given percentage.
        """
        self.logger.info("Adding Moving Average Envelope (custom + pandas-ta SMA)")
        if 'close' not in df.columns:
            raise ValueError("Column 'close' not found in DataFrame")

        # Use user-defined parameters if provided
        length = user_defined_window if user_defined_window is not None else window_size
        env_pct = user_defined_percentage if user_defined_percentage is not None else percentage

        # Calculate the Simple Moving Average (SMA)
        sma_series = ta.sma(df['close'], length=length)

        if sma_series is None or sma_series.empty:
            raise ValueError("pandas-ta did not return an SMA result. Check your data or parameters.")

        # Create Moving Average Envelope upper and lower bands
        df['MAE_Upper'] = sma_series * (1 + env_pct)
        df['MAE_Lower'] = sma_series * (1 - env_pct)

        self.logger.info("Successfully added Moving Average Envelope (custom + pandas-ta SMA)")
        return df

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies volatility indicators to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing stock data with necessary columns.

        Returns:
            pd.DataFrame: DataFrame with added volatility indicators.
        """
        self.logger.info("Applying Volatility Indicators...")

        required_columns = ['close', 'high', 'low']
        if not all(column in df.columns for column in required_columns):
            self.logger.error(f"DataFrame must contain columns: {', '.join(required_columns)}.")
            return df

        try:
            # Bollinger Bands
            df = self.add_bollinger_bands(df)

            # Standard Deviation
            df = self.add_standard_deviation(df)

            # Historical Volatility
            df = self.add_historical_volatility(df)

            # Chandelier Exit
            df = self.add_chandelier_exit(df)

            # Keltner Channel
            df = self.add_keltner_channel(df)

            # Moving Average Envelope
            df = self.add_moving_average_envelope(df)

            self.logger.info("Successfully applied Volatility Indicators.")
        except Exception as e:
            self.logger.error(f"Error applying Volatility Indicators: {e}")

        return df

    # -------------------------------------------------------------------
    # Optimized Methods for Large Datasets and Streaming Data
    # -------------------------------------------------------------------
    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a single chunk of data by applying indicators.

        Args:
            chunk (pd.DataFrame): A chunk of the original DataFrame.

        Returns:
            pd.DataFrame: Processed chunk with indicators.
        """
        self.logger.debug(f"Processing a chunk of size: {len(chunk)}")
        chunk = self.downcast_dataframe(chunk)
        chunk = self.apply_indicators(chunk)
        return chunk

    def process_large_dataset(self, file_path: str, chunksize: int = 50000) -> None:
        """
        Processes a large CSV file in chunks and saves the results.

        Args:
            file_path (str): Path to the large CSV file.
            chunksize (int): Number of rows per chunk.
        """
        self.logger.info(f"Starting chunked processing for file: {file_path} with chunksize: {chunksize}")
        start_time = timer()

        try:
            pool = Pool(cpu_count())
            reader = pd.read_csv(file_path, chunksize=chunksize, dtype={'close': 'float32', 'high': 'float32', 'low': 'float32', 'volume': 'float32'})
            processed_chunks = pool.map(self.process_chunk, reader)
            pool.close()
            pool.join()

            # Concatenate all processed chunks
            processed_df = pd.concat(processed_chunks, ignore_index=True)

            # Save to a new CSV or database
            output_path = file_path.replace('.csv', '_processed.csv')
            processed_df.to_csv(output_path, index=False)
            self.logger.info(f"Chunked processing completed. Output saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error during chunked processing: {e}")
        finally:
            end_time = timer()
            self.logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

    def process_streaming_data(self, data_stream, window_size=20):
        """
        Processes streaming data using a sliding window approach.

        Args:
            data_stream (iterable): An iterable that yields new data points as dictionaries.
            window_size (int): The size of the sliding window.
        """
        self.logger.info("Starting streaming data processing.")
        buffer = deque(maxlen=window_size)

        for data_point in data_stream:
            buffer.append(data_point)
            if len(buffer) < window_size:
                self.logger.debug("Not enough data to process yet.")
                continue

            df = pd.DataFrame(list(buffer))
            df = self.downcast_dataframe(df)
            df = self.apply_indicators(df)
            latest_data = df.iloc[-1]
            self.logger.info(f"Processed latest data point: {latest_data.to_dict()}")

# -------------------------------------------------------------------
# Example Usage of VolatilityIndicators (pandas-ta)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize VolatilityIndicators with logger
    indicators = VolatilityIndicators(logger=logger)

    # Create a sample DataFrame with manageable periods to avoid OutOfBoundsDatetime
    try:
        periods = 50000  # Adjusted to stay within pandas' datetime limits
        start_date = '2022-01-01'
        end_date = pd.Timestamp(start_date) + pd.Timedelta(days=periods)
        if end_date > pd.Timestamp.max:
            periods = (pd.Timestamp.max - pd.Timestamp(start_date)).days - 1
            logger.warning(f"Adjusted periods to {periods} to stay within datetime bounds.")

        data = {
            'date': pd.date_range(start=start_date, periods=periods),
            'high': pd.Series(np.random.uniform(100, 200, periods)),
            'low': pd.Series(np.random.uniform(50, 100, periods)),
            'close': pd.Series(np.random.uniform(75, 175, periods)),
            'volume': pd.Series(np.random.randint(1000, 10000, periods))
        }
        df = pd.DataFrame(data)

        # Optimize memory usage
        df = indicators.downcast_dataframe(df)

        # Apply indicators
        df = indicators.apply_indicators(df)

        # Display some output
        print("\nBollinger Bands (head):")
        print(df[['Bollinger_Low', 'Bollinger_Mid', 'Bollinger_High']].head())

        print("\nStandard Deviation (head):")
        print(df[['Standard_Deviation']].head())

        print("\nHistorical Volatility (head):")
        print(df[['Historical_Volatility']].head())

        print("\nChandelier Exit (head):")
        print(df[['Chandelier_Exit_Long']].head())

        print("\nKeltner Channel (head):")
        print(df[['Keltner_Channel_Low', 'Keltner_Channel_Basis', 'Keltner_Channel_High']].head())

        print("\nMoving Average Envelope (head):")
        print(df[['MAE_Upper', 'MAE_Lower']].head())

        # Example of chunked processing
        # indicators.process_large_dataset('path_to_large_csv_file.csv', chunksize=50000)

        # Example of streaming data processing
        # Simulate a data stream
        # def generate_stream(n):
        #     for i in range(n):
        #         yield {
        #             'close': np.random.uniform(75, 175),
        #             'high': np.random.uniform(100, 200),
        #             'low': np.random.uniform(50, 100),
        #             'volume': np.random.randint(1000, 10000)
        #         }
        # indicators.process_streaming_data(generate_stream(1000), window_size=20)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
