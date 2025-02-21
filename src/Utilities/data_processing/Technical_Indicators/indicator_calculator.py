"""
File: indicator_calculator.py
Location: D:\\TradingRobotPlug2\\src\\Utilities\\data_processing\\indicator_calculator.py

Description:
    A "best-in-class" indicator calculator module that provides a wide range of
    technical indicators—both classic and advanced—along with robust logging, 
    error handling, and customizable parameters.

Indicators Covered:
  1. Relative Strength Index (RSI)
  2. Moving Average Convergence Divergence (MACD)
  3. Simple Moving Average (SMA)
  4. Exponential Moving Average (EMA)
  5. Bollinger Bands
  6. Average True Range (ATR)
  7. Stochastic Oscillator
  8. Average Directional Index (ADX)
  9. Keltner Channels
  10. Heiken Ashi Conversion
  11. Pivot Points (classic floor pivots)
  12. Fibonacci Retracements & Extensions
  13. Commodity Channel Index (CCI)

Key Features:
  - Advanced logging for each computation
  - Configurable parameters (window sizes, methods, fill behaviors)
  - Resilient checks for missing columns or invalid data
  - Single entry point for flexible usage in scripts, strategies, or unifiers
  - Clear docstrings with example usage
"""

import logging
from typing import Optional, Tuple, List, Union
import numpy as np
import pandas as pd


class IndicatorCalculator:
    """
    Provides a comprehensive suite of technical indicator computations 
    with robust configuration, logging, and error handling.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Args:
            logger (logging.Logger, optional): A logger instance for debug/info messages.
                If none provided, a basic stream logger is created.
        """
        self.logger = logger or self._setup_default_logger()

    # ---------------------------------------------------------------------
    # Logging Setup
    # ---------------------------------------------------------------------
    def _setup_default_logger(self) -> logging.Logger:
        logger = logging.getLogger("IndicatorCalculator")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(handler)
        return logger

    # ---------------------------------------------------------------------
    # 1) RSI
    # ---------------------------------------------------------------------
    def compute_rsi(
        self,
        series: pd.Series,
        window: int = 14,
        method: str = "classic",
        fillna_value: float = 50.0
    ) -> pd.Series:
        """
        Computes Relative Strength Index (RSI).

        Args:
            series (pd.Series): Price series, typically 'Close'.
            window (int): RSI lookback period (default 14).
            method (str): Calculation method, one of ['classic', 'ema'].
                - 'classic': Uses simple moving averages for up/down moves.
                - 'ema': Uses exponential moving averages (Wilder's smoothing).
            fillna_value (float): Value to fill NaNs with (default 50).

        Returns:
            pd.Series: RSI values (0-100 scale).
        """
        if series.empty:
            self.logger.warning("RSI computation: input series is empty.")
            return pd.Series(dtype=np.float64)

        delta = series.diff()

        if method.lower() == "ema":
            # Wilder's smoothing via EWMA
            gain = delta.where(delta > 0, 0).ewm(span=window, adjust=False).mean()
            loss = -delta.where(delta < 0, 0).ewm(span=window, adjust=False).mean()
        else:
            # Classic RSI
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=window).mean()

        rs = gain / loss.replace({0: np.nan})
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.fillna(fillna_value)  # fill remaining NaNs

        self.logger.debug(f"Computed RSI - window={window}, method={method}.")
        return rsi

    # ---------------------------------------------------------------------
    # 2) MACD
    # ---------------------------------------------------------------------
    def compute_macd(
        self,
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Computes MACD (Moving Average Convergence Divergence) plus signal line.

        Args:
            series (pd.Series): Price series, typically 'Close'.
            fast (int): Fast EMA period (default=12).
            slow (int): Slow EMA period (default=26).
            signal (int): Signal line EMA period (default=9).

        Returns:
            (pd.Series, pd.Series): macd_line, signal_line
        """
        if series.empty:
            self.logger.warning("MACD computation: input series is empty.")
            return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)

        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        self.logger.debug(f"Computed MACD - fast={fast}, slow={slow}, signal={signal}.")
        return macd_line, signal_line

    # ---------------------------------------------------------------------
    # 3) SMA
    # ---------------------------------------------------------------------
    def compute_sma(
        self,
        series: pd.Series,
        window: int = 50,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """
        Computes Simple Moving Average (SMA).

        Args:
            series (pd.Series): Price series, typically 'Close'.
            window (int): Rolling window (default=50).
            min_periods (int): Minimum periods for rolling calc (default=window).

        Returns:
            pd.Series: SMA values.
        """
        if series.empty:
            self.logger.warning("SMA computation: input series is empty.")
            return pd.Series(dtype=np.float64)

        sma = series.rolling(window=window, min_periods=min_periods or window).mean()
        self.logger.debug(f"Computed SMA - window={window}.")
        return sma

    # ---------------------------------------------------------------------
    # 4) EMA
    # ---------------------------------------------------------------------
    def compute_ema(
        self,
        series: pd.Series,
        window: int = 50,
        adjust: bool = False
    ) -> pd.Series:
        """
        Computes Exponential Moving Average (EMA).

        Args:
            series (pd.Series): Price series.
            window (int): Rolling window for EMA calculation.
            adjust (bool): Whether to adjust weights recursively (default=False).

        Returns:
            pd.Series: EMA values.
        """
        if series.empty:
            self.logger.warning("EMA computation: input series is empty.")
            return pd.Series(dtype=np.float64)

        ema = series.ewm(span=window, adjust=adjust).mean()
        self.logger.debug(f"Computed EMA - window={window}, adjust={adjust}.")
        return ema

    # ---------------------------------------------------------------------
    # 5) Bollinger Bands
    # ---------------------------------------------------------------------
    def compute_bollinger_bands(
        self,
        series: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Computes Bollinger Bands: upper, middle (SMA), lower.

        Args:
            series (pd.Series): Price series.
            window (int): Rolling window for SMA.
            num_std (float): Number of standard deviations (default=2.0).

        Returns:
            (pd.Series, pd.Series, pd.Series): upper_band, middle_band, lower_band
        """
        if series.empty:
            self.logger.warning("Bollinger Bands: input series is empty.")
            return (pd.Series(dtype=np.float64),)*3

        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        self.logger.debug(f"Computed Bollinger Bands - window={window}, num_std={num_std}.")
        return upper_band, sma, lower_band

    # ---------------------------------------------------------------------
    # 6) ATR
    # ---------------------------------------------------------------------
    def compute_atr(
        self,
        df: pd.DataFrame,
        window: int = 14
    ) -> pd.Series:
        """
        Computes Average True Range (ATR) for volatility measurement.

        Args:
            df (pd.DataFrame): Must contain columns 'High', 'Low', 'Close'.
            window (int): Rolling window (default=14).

        Returns:
            pd.Series: ATR values.
        """
        required_cols = {"High", "Low", "Close"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            self.logger.error(f"ATR computation: missing columns {missing}")
            return pd.Series(dtype=np.float64)

        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()

        self.logger.debug(f"Computed ATR - window={window}.")
        return atr

    # ---------------------------------------------------------------------
    # 7) Stochastic Oscillator
    # ---------------------------------------------------------------------
    def compute_stochastic(
        self,
        df: pd.DataFrame,
        k_window: int = 14,
        d_window: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Computes Stochastic Oscillator %K and %D lines.

        Args:
            df (pd.DataFrame): Must have columns 'High', 'Low', 'Close'.
            k_window (int): Lookback for %K (default=14).
            d_window (int): Lookback for %D (default=3).

        Returns:
            (pd.Series, pd.Series): k_line, d_line
        """
        required_cols = {"High", "Low", "Close"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            self.logger.error(f"Stochastic computation: missing columns {missing}")
            return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)

        low_min = df["Low"].rolling(k_window).min()
        high_max = df["High"].rolling(k_window).max()

        k_line = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
        d_line = k_line.rolling(d_window).mean()

        self.logger.debug(f"Computed Stochastic - k_window={k_window}, d_window={d_window}.")
        return k_line.fillna(50.0), d_line.fillna(50.0)

    # ---------------------------------------------------------------------
    # 8) ADX (Average Directional Index)
    # ---------------------------------------------------------------------
    def compute_adx(
        self,
        df: pd.DataFrame,
        window: int = 14
    ) -> pd.Series:
        """
        Computes the Average Directional Index (ADX).

        Args:
            df (pd.DataFrame): Must have columns 'High', 'Low', 'Close'.
            window (int): Lookback for smoothing (default=14).

        Returns:
            pd.Series: ADX values.
        """
        required_cols = {"High", "Low", "Close"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            self.logger.error(f"ADX computation: missing columns {missing}")
            return pd.Series(dtype=np.float64)

        # True Range for single period
        df["_TR"] = (df["High"] - df["Low"]).abs()
        df["_TR"] = df[["_TR", (df["High"] - df["Close"].shift(1)).abs(), (df["Low"] - df["Close"].shift(1)).abs()]].max(axis=1)

        # +DM, -DM
        df["_+DM"] = df["High"].diff()
        df["_-DM"] = df["Low"].diff() * -1

        df["_+DM"] = np.where((df["_+DM"] > df["_-DM"]) & (df["_+DM"] > 0), df["_+DM"], 0.0)
        df["_-DM"] = np.where((df["_-DM"] > df["_+DM"]) & (df["_-DM"] > 0), df["_-DM"], 0.0)

        # Exponential smoothing or simple smoothing
        atr_series = df["_TR"].rolling(window=window).mean()
        plus_dm = df["_+DM"].rolling(window=window).mean()
        minus_dm = df["_-DM"].rolling(window=window).mean()

        df["_+DI"] = 100 * (plus_dm / atr_series.replace(0, np.nan))
        df["_-DI"] = 100 * (minus_dm / atr_series.replace(0, np.nan))

        df["_DX"] = 100 * (df["_+DI"] - df["_-DI"]).abs() / (df["_+DI"] + df["_-DI"]).replace(0, np.nan)
        adx = df["_DX"].rolling(window=window).mean()

        # Clean up
        df.drop(["_TR", "_+DM", "_-DM", "_+DI", "_-DI", "_DX"], axis=1, inplace=True)

        self.logger.debug(f"Computed ADX - window={window}.")
        return adx.fillna(method="bfill")

    # ---------------------------------------------------------------------
    # 9) Keltner Channels
    # ---------------------------------------------------------------------
    def compute_keltner_channels(
        self,
        df: pd.DataFrame,
        ema_window: int = 20,
        atr_window: int = 10,
        multiplier: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Computes Keltner Channels (upper, mid, lower).

        Args:
            df (pd.DataFrame): Must have 'High', 'Low', 'Close'.
            ema_window (int): EMA lookback for mid line (default=20).
            atr_window (int): ATR lookback (default=10).
            multiplier (float): ATR multiplier (default=2.0).

        Returns:
            (pd.Series, pd.Series, pd.Series): upper_channel, mid_line, lower_channel
        """
        required_cols = {"High", "Low", "Close"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            self.logger.error(f"Keltner Channels: missing columns {missing}")
            return (pd.Series(dtype=np.float64),)*3

        mid_line = df["Close"].ewm(span=ema_window, adjust=False).mean()
        atr_values = self.compute_atr(df, window=atr_window)

        upper_channel = mid_line + (atr_values * multiplier)
        lower_channel = mid_line - (atr_values * multiplier)

        self.logger.debug(
            f"Computed Keltner Channels - ema_window={ema_window}, atr_window={atr_window}, multiplier={multiplier}."
        )
        return upper_channel, mid_line, lower_channel

    # ---------------------------------------------------------------------
    # 10) Heiken Ashi Conversion
    # ---------------------------------------------------------------------
    def compute_heiken_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts standard candlestick data to Heiken Ashi candlesticks.

        Args:
            df (pd.DataFrame): Must have columns ['Open', 'High', 'Low', 'Close'].

        Returns:
            pd.DataFrame: DataFrame with columns [HA_Open, HA_High, HA_Low, HA_Close].
        """
        required_cols = {"Open", "High", "Low", "Close"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            self.logger.error(f"Heiken Ashi: missing columns {missing}")
            return pd.DataFrame()

        ha_df = df.copy()

        # Heiken Ashi close is average of O,H,L,C
        ha_close = (ha_df["Open"] + ha_df["High"] + ha_df["Low"] + ha_df["Close"]) / 4

        # Heiken Ashi open is average of previous HA_Open and previous HA_Close
        ha_open = [ha_df["Open"].iloc[0]]  # first bar's open is the actual open
        for i in range(1, len(ha_df)):
            prev_ha_open = ha_open[-1]
            prev_ha_close = ha_close.iloc[i-1]
            ha_open.append((prev_ha_open + prev_ha_close) / 2)

        # High and low
        ha_high = pd.DataFrame(
            [ha_open, ha_close, ha_df["High"]]
        ).T.max(axis=1)
        ha_low = pd.DataFrame(
            [ha_open, ha_close, ha_df["Low"]]
        ).T.min(axis=1)

        # Construct final
        ha_result = pd.DataFrame({
            "HA_Open": ha_open,
            "HA_Close": ha_close,
            "HA_High": ha_high,
            "HA_Low": ha_low
        }, index=ha_df.index)

        self.logger.debug("Computed Heiken Ashi candlesticks.")
        return ha_result

    # ---------------------------------------------------------------------
    # 11) Pivot Points (Classic Floor Pivots)
    # ---------------------------------------------------------------------
    def compute_pivot_points(
        self,
        df: pd.DataFrame,
        offset: int = 0
    ) -> pd.DataFrame:
        """
        Computes Classic Floor Pivot Points (daily) with R1, R2, R3, S1, S2, S3.

        Args:
            df (pd.DataFrame): Must contain 'High', 'Low', 'Close' columns.
            offset (int): Use offset for pivot calculations, e.g. previous day data.

        Returns:
            pd.DataFrame: A DataFrame with columns [Pivot, R1, R2, R3, S1, S2, S3].
        """
        required_cols = {"High", "Low", "Close"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            self.logger.error(f"Pivot Points: missing columns {missing}")
            return pd.DataFrame()

        # Shift by offset if using previous day's data
        H = df["High"].shift(offset)
        L = df["Low"].shift(offset)
        C = df["Close"].shift(offset)

        pivot = (H + L + C) / 3
        r1 = (2 * pivot) - L
        s1 = (2 * pivot) - H
        r2 = pivot + (H - L)
        s2 = pivot - (H - L)
        r3 = H + 2 * (pivot - L)
        s3 = L - 2 * (H - pivot)

        pivot_df = pd.DataFrame({
            "Pivot": pivot,
            "R1": r1, "R2": r2, "R3": r3,
            "S1": s1, "S2": s2, "S3": s3
        }, index=df.index)

        self.logger.debug(f"Computed Pivot Points with offset={offset}.")
        return pivot_df

    # ---------------------------------------------------------------------
    # 12) Fibonacci Retracements & Extensions
    # ---------------------------------------------------------------------
    def compute_fibonacci_levels(
        self,
        high: Union[float, pd.Series],
        low: Union[float, pd.Series],
        levels: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Computes Fibonacci retracement levels based on given high and low prices.

        Args:
            high (float | pd.Series): The highest price in the range.
            low (float | pd.Series): The lowest price in the range.
            levels (List[float], optional): Custom Fibonacci retracement levels.

        Returns:
            pd.DataFrame: Fibonacci retracement levels as columns.
        """
        if levels is None:
            levels = [0.0, 0.236, 0.382, 0.5, 0.618, 1.0]

        diff = high - low
        fib_data = {f"Fib_{ratio}": high - (diff * ratio) for ratio in levels}

        if isinstance(high, (int, float)) and isinstance(low, (int, float)):
            return pd.DataFrame([fib_data])

        if isinstance(high, pd.Series) and isinstance(low, pd.Series):
            return pd.DataFrame(fib_data, index=high.index)

        raise ValueError("Invalid input types. 'high' and 'low' must be float values or Pandas Series.")

    def compute_fibonacci_extensions(
        self,
        high: Union[float, pd.Series],
        low: Union[float, pd.Series],
        levels: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Computes Fibonacci extension levels for price projections.

        Args:
            high (float | pd.Series): The highest price in the range.
            low (float | pd.Series): The lowest price in the range.
            levels (List[float], optional): Custom Fibonacci extension levels.

        Returns:
            pd.DataFrame: Fibonacci extension levels as columns.
        """
        if levels is None:
            levels = [1.272, 1.414, 1.618, 2.0, 2.618, 3.618]

        diff = high - low
        ext_data = {f"FibExt_{ratio}": high + (diff * ratio) for ratio in levels}

        if isinstance(high, (int, float)) and isinstance(low, (int, float)):
            return pd.DataFrame([ext_data])

        if isinstance(high, pd.Series) and isinstance(low, pd.Series):
            return pd.DataFrame(ext_data, index=high.index)

        raise ValueError("Invalid input types. 'high' and 'low' must be float values or Pandas Series.")

    def apply_fibonacci_to_dataframe(
        self,
        df: pd.DataFrame,
        high_col: str = "High",
        low_col: str = "Low",
        levels: Optional[List[float]] = None,
        extensions: bool = False
    ) -> pd.DataFrame:
        """
        Applies Fibonacci retracement and (optionally) extension levels to a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing high and low price columns.
            high_col (str): Column name for high prices.
            low_col (str): Column name for low prices.
            levels (List[float], optional): Custom Fibonacci retracement levels.
            extensions (bool): If True, includes Fibonacci extensions.

        Returns:
            pd.DataFrame: DataFrame with added Fibonacci retracement and extension columns.
        """
        if high_col not in df.columns or low_col not in df.columns:
            self.logger.warning(f"Columns {high_col} or {low_col} not found in DataFrame. Skipping Fibonacci computation.")
            return df

        fib_df = self.compute_fibonacci_levels(df[high_col], df[low_col], levels)
        if extensions:
            fib_ext_df = self.compute_fibonacci_extensions(df[high_col], df[low_col])
            fib_df = pd.concat([fib_df, fib_ext_df], axis=1)

        return pd.concat([df, fib_df], axis=1)

    # ---------------------------------------------------------------------
    # 13) Commodity Channel Index (CCI)
    # ---------------------------------------------------------------------
    def compute_cci(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Computes the Commodity Channel Index (CCI).

        Args:
            df (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns.
            window (int): Lookback period.

        Returns:
            pd.Series: CCI values.
        """
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        sma = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci

# ---------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Sample DataFrame
    data = {
        "Open": [100, 102, 103, 104, 103],
        "High": [101, 103, 105, 105, 104],
        "Low": [99, 101, 102, 102, 101],
        "Close": [100, 102, 104, 103, 102]
    }
    df_sample = pd.DataFrame(data)

    # Instantiate calculator
    calc = IndicatorCalculator()

    # Apply Fibonacci levels (retracements + extensions)
    df_fib = calc.apply_fibonacci_to_dataframe(df_sample, extensions=True)
    print("\nFibonacci Retracement & Extension Levels:\n", df_fib)

    # Compute Pivot Points
    df_pivot = calc.compute_pivot_points(df_sample)
    print("\nPivot Points:\n", df_pivot)

    # Compute CCI
    df_sample["CCI"] = calc.compute_cci(df_sample)
    print("\nCCI:\n", df_sample[["CCI"]])
