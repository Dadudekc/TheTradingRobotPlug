"""
File: rsi_macd_strategy.py
Location: src/Utilities/strategies

Description:
    A trading strategy that combines the Relative Strength Index (RSI)
    and Moving Average Convergence Divergence (MACD) to generate 
    buy and sell signals.

    - RSI identifies overbought and oversold conditions.
    - MACD detects trend changes and momentum.
    - Buy when RSI is oversold AND MACD line crosses above Signal line.
    - Sell when RSI is overbought OR MACD line crosses below Signal line.
"""

import pandas as pd
import numpy as np
import logging
from src.Utilities.indicators.indicator_calculator import IndicatorCalculator


class RSIMACDStrategy:
    """
    Implements an RSI & MACD-based trading strategy.
    """

    def __init__(self, df: pd.DataFrame, rsi_window: int = 14, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 rsi_buy_threshold: int = 30, rsi_sell_threshold: int = 70, logger: logging.Logger = None):
        """
        Initializes the strategy.

        Args:
            df (pd.DataFrame): DataFrame containing stock price data with 'Close' prices.
            rsi_window (int): RSI lookback period.
            macd_fast (int): Fast period for MACD.
            macd_slow (int): Slow period for MACD.
            macd_signal (int): Signal line period for MACD.
            rsi_buy_threshold (int): RSI level considered oversold (default 30).
            rsi_sell_threshold (int): RSI level considered overbought (default 70).
            logger (logging.Logger): Optional logger instance.
        """
        self.df = df.copy()
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.logger = logger or self._setup_logger()

        self.logger.info("RSIMACDStrategy initialized.")

    def _setup_logger(self):
        """Sets up logging for this strategy."""
        logger = logging.getLogger("RSIMACDStrategy")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def generate_signals(self) -> pd.DataFrame:
        """
        Generates buy/sell signals based on RSI and MACD conditions.

        Returns:
            pd.DataFrame: DataFrame with Buy_Signal and Sell_Signal columns.
        """
        self.logger.info("Generating RSI and MACD indicators...")
        
        # Compute RSI
        self.df["RSI"] = IndicatorCalculator.compute_rsi(self.df["Close"], window=self.rsi_window)
        
        # Compute MACD
        self.df["MACD"], self.df["MACD_Signal"] = IndicatorCalculator.compute_macd(
            self.df["Close"], self.macd_fast, self.macd_slow, self.macd_signal
        )

        self.logger.info("Applying buy/sell conditions...")
        
        # Buy conditions
        self.df["Buy_Signal"] = (self.df["RSI"] < self.rsi_buy_threshold) & (self.df["MACD"] > self.df["MACD_Signal"])
        
        # Sell conditions
        self.df["Sell_Signal"] = (self.df["RSI"] > self.rsi_sell_threshold) | (self.df["MACD"] < self.df["MACD_Signal"])

        self.logger.info(f"Generated {self.df['Buy_Signal'].sum()} buy signals and {self.df['Sell_Signal'].sum()} sell signals.")

        return self.df

