# -------------------------------------------------------------------
# File: trading_strategies.py
# Location: src/Utilities/strategies
# Description: Implements trading strategies based on moving averages.
# -------------------------------------------------------------------

import pandas as pd
from typing import Optional
import logging

class TradingStrategies:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def momentum_rule(self, data: pd.Series, k: int = 1) -> pd.Series:
        """
        Implements the Momentum Rule.

        Args:
            data (pd.Series): Price data.
            k (int, optional): Lookback period in months. Defaults to 1.

        Returns:
            pd.Series: Trade signals (1 for buy, -1 for sell, 0 otherwise).
        """
        self.logger.debug(f"Applying Momentum Rule with k={k}")
        momentum = data.diff(k)
        signals = pd.Series(0, index=data.index)
        signals[momentum > 0] = 1
        signals[momentum < 0] = -1
        return signals

    def price_minus_ma_rule(self, price: pd.Series, ma: pd.Series) -> pd.Series:
        """
        Implements the Price-Minus-MA Rule.

        Args:
            price (pd.Series): Price data.
            ma (pd.Series): Moving average data.

        Returns:
            pd.Series: Trade signals (1 for buy, -1 for sell, 0 otherwise).
        """
        self.logger.debug("Applying Price-Minus-MA Rule")
        signals = pd.Series(0, index=price.index)
        signals[price > ma] = 1
        signals[price < ma] = -1
        return signals

    def double_crossover_rule(self, short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
        """
        Implements the Double Crossover Rule.

        Args:
            short_ma (pd.Series): Short-term moving average.
            long_ma (pd.Series): Long-term moving average.

        Returns:
            pd.Series: Trade signals (1 for buy, -1 for sell, 0 otherwise).
        """
        self.logger.debug("Applying Double Crossover Rule")
        signals = pd.Series(0, index=short_ma.index)
        crossover = short_ma > long_ma
        crossover_shifted = crossover.shift(1)
        signals[(crossover) & (~crossover_shifted)] = 1  # Buy signal
        signals[(~crossover) & (crossover_shifted)] = -1  # Sell signal
        return signals

    def ma_change_of_direction_rule(self, ma: pd.Series) -> pd.Series:
        """
        Implements the MA Change of Direction Rule.

        Args:
            ma (pd.Series): Moving average data.

        Returns:
            pd.Series: Trade signals (1 for buy, -1 for sell, 0 otherwise).
        """
        self.logger.debug("Applying MA Change of Direction Rule")
        slope = ma.diff()
        slope_sign = slope.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        signals = slope_sign.diff()
        signals = signals.fillna(0)
        signals = signals.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        return signals
