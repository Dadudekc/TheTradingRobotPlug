# -------------------------------------------------------------------
# File: backtester.py
# Location: src/Utilities/strategies
# Description: Implements backtesting for trading strategies.
# -------------------------------------------------------------------
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import logging

class Backtester:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def backtest(self, price_data: pd.Series, signals: pd.Series) -> pd.DataFrame:
        """
        Backtests a trading strategy based on generated signals.

        Args:
            price_data (pd.Series): Price data.
            signals (pd.Series): Trade signals (1 for buy, -1 for sell, 0 otherwise).

        Returns:
            pd.DataFrame: Portfolio performance metrics.
        """
        self.logger.debug("Starting backtest.")
        # Calculate daily returns
        returns = price_data.pct_change().fillna(0)

        # Shift signals to represent next day execution
        positions = signals.shift(1).fillna(0)

        # Calculate strategy returns
        strategy_returns = positions * returns

        # Create a DataFrame to hold results
        performance = pd.DataFrame({
            'Price': price_data,
            'Returns': returns,
            'Position': positions,
            'Strategy_Returns': strategy_returns
        })

        # Calculate cumulative returns
        performance['Cumulative_Returns'] = (1 + performance['Returns']).cumprod()
        performance['Cumulative_Strategy_Returns'] = (1 + performance['Strategy_Returns']).cumprod()

        self.logger.debug("Backtest completed.")
        return performance

    def calculate_performance_metrics(self, performance: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates performance metrics for the backtest.

        Args:
            performance (pd.DataFrame): Backtest performance data.

        Returns:
            Dict[str, float]: Performance metrics.
        """
        self.logger.debug("Calculating performance metrics.")

        # Calculate daily strategy returns
        strategy_returns = performance['Strategy_Returns']

        # Sharpe Ratio
        sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)  # Annualized

        # CAGR
        cagr = (performance['Cumulative_Strategy_Returns'].iloc[-1] ** (252 / len(strategy_returns))) - 1

        # Max Drawdown
        cumulative = performance['Cumulative_Strategy_Returns']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        metrics = {
            "Sharpe Ratio": sharpe_ratio,
            "CAGR": cagr,
            "Max Drawdown": max_drawdown
        }

        self.logger.info(f"Performance Metrics: {metrics}")
        return metrics
