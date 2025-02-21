"""
File: backtester.py
Location: src/Utilities/strategies

Description:
    A modular backtesting engine for trading strategies.

    - Fetches data via DataOrchestrator (no inline data fetch).
    - Optionally applies all technical indicators via AllIndicatorsUnifier.
    - Supports any strategy class registered in STRATEGY_REGISTRY.
    - Provides a simple ClassicBacktester for row-by-row trading logic.
    - Can be imported and tested without a main entry block.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
from typing import Dict, List

# Updated imports reflecting new project structure
from Utilities.data_fetchers.main_data_fetcher import DataOrchestrator
from Utilities.data_processing.Technical_Indicators.indicator_unifier import AllIndicatorsUnifier
from Utilities.strategies.registry import STRATEGY_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BaseStrategy(bt.Strategy):
    """
    Base class for any trading strategy in Backtrader.
    
    Subclasses must implement `next()` for the main trading logic.
    """
    def __init__(self):
        pass

    def next(self):
        raise NotImplementedError("Subclasses must implement the 'next()' method.")


class ClassicBacktester:
    """
    A simple row-by-row backtester that uses 'Buy_Signal'/'Sell_Signal'
    columns from a DataFrame to log trades.
    """

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.trade_log: List[tuple] = []

    def run(self):
        """
        Simulates a backtest by scanning row-by-row for Buy_Signal or Sell_Signal.
        """
        for i in range(len(self.df)):
            price = self.df["Close"].iloc[i]
            date = self.df["Date"].iloc[i]
            if self.df.get("Buy_Signal", pd.Series([False]*len(self.df))).iloc[i]:
                shares = int(self.balance // price)
                if shares > 0:
                    self.balance -= shares * price
                    self.position += shares
                    self.trade_log.append(("BUY", date, price, shares))
                    logging.info(f"BUY on {date}: {shares} shares at {price}")
            elif self.df.get("Sell_Signal", pd.Series([False]*len(self.df))).iloc[i] and self.position > 0:
                self.balance += self.position * price
                self.trade_log.append(("SELL", date, price, self.position))
                logging.info(f"SELL on {date}: {self.position} shares at {price}")
                self.position = 0

    def final_portfolio_value(self) -> float:
        """
        Returns the final portfolio value = cash + (shares * latest close).
        """
        return self.balance + (self.position * self.df["Close"].iloc[-1])

    def performance_metrics(self) -> dict:
        """
        Basic performance metrics:
         - final portfolio value
         - total return in percentage
        """
        final_value = self.final_portfolio_value()
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100
        return {"final_value": final_value, "total_return_pct": total_return}


class BacktestRunner:
    """
    Main backtesting engine that:
    - Fetches data from DataOrchestrator (async).
    - (Optionally) applies AllIndicatorsUnifier for full indicator coverage.
    - Runs a registered strategy in Backtrader.
    - Returns performance metrics using either Backtrader logs or the ClassicBacktester.
    """

    def __init__(
        self,
        strategy_name: str,
        initial_balance: float = 10000.0,
        apply_unifier: bool = False
    ):
        """
        Args:
            strategy_name (str): The name of a registered strategy.
            initial_balance (float): Starting capital for the performance evaluation.
            apply_unifier (bool): If True, applies AllIndicatorsUnifier to the DataFrame.
        """
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(f"Strategy '{strategy_name}' not found in registry.")
        self.strategy_class = STRATEGY_REGISTRY[strategy_name]
        self.initial_balance = initial_balance
        self.apply_unifier = apply_unifier
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_orchestrator = DataOrchestrator()

    async def fetch_data(self, symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Fetches and unifies data for a single symbol using DataOrchestrator.
        
        Returns:
            A single combined DataFrame sorted by "Date".
        """
        result_map = await self.data_orchestrator.fetch_all_data([symbol], start_date, end_date, interval)
        symbol_map = result_map.get(symbol, {})
        sources = [df for df in symbol_map.values() if not df.empty]
        if not sources:
            self.logger.warning(f"No data sources available for {symbol}.")
            return pd.DataFrame()

        combined = pd.concat(sources, ignore_index=True)
        combined.drop_duplicates(subset=["Date"], inplace=True)
        combined.sort_values(by="Date", inplace=True)
        return combined

    def run(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str
    ) -> dict:
        """
        Runs the backtest for a single symbol and returns performance metrics.

        Steps:
          1) Asynchronously fetch data.
          2) Optionally apply AllIndicatorsUnifier.
          3) Feed data into Backtrader with the chosen strategy.
          4) Evaluate performance using ClassicBacktester.
        """
        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(self.fetch_data(symbol, start_date, end_date, interval))

        if df.empty:
            self.logger.error(f"Empty DataFrame for symbol {symbol}. Cannot backtest.")
            return {"error": "no_data"}

        # Optionally apply unifier for full indicator coverage
        if self.apply_unifier:
            unifier = AllIndicatorsUnifier(config_manager=None, logger=self.logger, use_csv=False)
            df = unifier.apply_all_indicators(df)

        # Prepare DataFrame for Backtrader: ensure 'Date' and 'Close' columns are set up correctly
        df["datetime"] = pd.to_datetime(df["Date"])
        df.set_index("datetime", inplace=True, drop=True)
        df.dropna(subset=["Close"], inplace=True)

        datafeed = bt.feeds.PandasData(dataname=df)

        # Backtrader setup: add strategy and datafeed to Cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(self.strategy_class)
        cerebro.adddata(datafeed)
        cerebro.run()  # Execute the strategy

        # Evaluate performance using a classic row-by-row approach
        backtester = ClassicBacktester(df, initial_balance=self.initial_balance)
        backtester.run()
        metrics = backtester.performance_metrics()
        self.logger.info(f"Performance for {symbol}: {metrics}")
        return metrics


# Example usage:
# To use this module, first register your strategy in the registry (e.g., via a decorator)
# Then, you can run a backtest as follows:
#
# from src.Utilities.strategies.backtester import BacktestRunner
# runner = BacktestRunner(strategy_name="RSI_MACD", apply_unifier=True)
# results = runner.run(symbol="AAPL", start_date="2020-01-01", end_date="2020-12-31", interval="1d")
# print(results)
