# -------------------------------------------------------------------
# File: moving_averages.py
# Location: src/Utilities/strategies
# Description: Implements various moving average calculations.
# -------------------------------------------------------------------

import pandas as pd
import numpy as np
from typing import Optional
import logging

class MovingAverages:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def sma(self, data: pd.Series, window: int) -> pd.Series:
        """
        Calculates Simple Moving Average.

        Args:
            data (pd.Series): Price data.
            window (int): Lookback period.

        Returns:
            pd.Series: SMA values.
        """
        self.logger.debug(f"Calculating SMA with window={window}")
        return data.rolling(window=window).mean()

    def ema(self, data: pd.Series, span: int) -> pd.Series:
        """
        Calculates Exponential Moving Average.

        Args:
            data (pd.Series): Price data.
            span (int): Lookback period.

        Returns:
            pd.Series: EMA values.
        """
        self.logger.debug(f"Calculating EMA with span={span}")
        return data.ewm(span=span, adjust=False).mean()

    def lma(self, data: pd.Series, window: int) -> pd.Series:
        """
        Calculates Linear Moving Average.

        Args:
            data (pd.Series): Price data.
            window (int): Lookback period.

        Returns:
            pd.Series: LMA values.
        """
        self.logger.debug(f"Calculating LMA with window={window}")
        weights = np.linspace(1, window, window)
        return data.rolling(window).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)

    def rema(self, data: pd.Series, span: int) -> pd.Series:
        """
        Calculates Reverse Exponential Moving Average.

        Args:
            data (pd.Series): Price data.
            span (int): Lookback period.

        Returns:
            pd.Series: REMA values.
        """
        self.logger.debug(f"Calculating REMA with span={span}")
        # Reverse the data, calculate EMA, then reverse back
        reversed_data = data[::-1]
        ema_reversed = reversed_data.ewm(span=span, adjust=False).mean()
        return ema_reversed[::-1]
