"""
File: backtrader_engine.py
Location: src/Utilities/backtest_engine

Description:
    An advanced Backtrader-specific backtesting engine.
    Provides a flexible interface for running strategies with multiple data feeds,
    customizable broker settings, and advanced performance analyzers.
    
Features:
    - Dynamic DataFeed creation from Pandas DataFrames.
    - Customizable broker configuration (initial cash, commission, slippage).
    - Integration of multiple analyzers: Sharpe Ratio, Drawdown, TradeAnalyzer.
    - Detailed logging of strategy execution and errors.
    - Support for additional optional configurations (e.g., multi-data feeds).
    - Performance metrics extraction and plotting support.
"""

import logging
import backtrader as bt
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List

# Setup a module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class BacktraderEngine:
    """
    A Backtrader-specific engine for executing backtests with advanced configuration.
    """
    def __init__(self,
                 strategy: type,
                 data: Optional[pd.DataFrame] = None,
                 initial_cash: float = 10000.0,
                 commission: float = 0.001,  # 0.1%
                 slippage_perc: float = 0.0,
                 tradehistory: bool = True,
                 strategy_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the engine.

        Args:
            strategy (type): A Backtrader Strategy class.
            data (pd.DataFrame, optional): Pandas DataFrame containing historical data.
            initial_cash (float): Initial cash in the broker.
            commission (float): Commission rate.
            slippage_perc (float): Percentage slippage to simulate.
            tradehistory (bool): Whether to store detailed trade history.
            strategy_params (dict, optional): Extra parameters for the strategy.
        """
        self.strategy = strategy
        self.data = data  # Expecting a DataFrame
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage_perc = slippage_perc
        self.tradehistory = tradehistory
        self.strategy_params = strategy_params or {}
        self.cerebro = bt.Cerebro(stdstats=False)
        self.analyzers = {}
        self._setup_engine()

    def _setup_engine(self):
        """
        Configures the Cerebro engine, including broker settings, slippage, and analyzers.
        """
        logger.info("Setting up Backtrader Cerebro engine.")
        # Set broker cash and commission
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)
        # Configure slippage if applicable.
        if self.slippage_perc > 0:
            self.cerebro.broker.set_slippage_fixed(self.slippage_perc)
            logger.info(f"Setting fixed slippage: {self.slippage_perc}")

        # Add the strategy with any provided parameters.
        self.cerebro.addstrategy(self.strategy, **self.strategy_params)
        logger.info(f"Added strategy: {self.strategy.__name__} with params {self.strategy_params}")

        # Add common analyzers
        self._add_analyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days)
        self._add_analyzer(bt.analyzers.DrawDown, _name="drawdown")
        self._add_analyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        logger.info("Analyzers added to Cerebro engine.")

    def _add_analyzer(self, analyzer_cls, _name: str, **kwargs):
        """
        Helper method to add an analyzer to Cerebro.
        """
        self.cerebro.addanalyzer(analyzer_cls, _name=_name, **kwargs)
        self.analyzers[_name] = None

    def _prepare_datafeed(self, df: pd.DataFrame, timeframe: bt.TimeFrame = bt.TimeFrame.Days) -> bt.feeds.PandasData:
        """
        Prepares a Backtrader data feed from a Pandas DataFrame.

        The DataFrame must have at least 'Date' and 'Close' columns.
        """
        logger.info("Preparing data feed from DataFrame.")
        # Ensure the DataFrame is sorted by date.
        df = df.copy()
        if "Date" in df.columns:
            df["datetime"] = pd.to_datetime(df["Date"])
            df.set_index("datetime", inplace=True, drop=True)
        else:
            raise ValueError("DataFrame must contain a 'Date' column.")

        # Drop rows without a valid 'Close' price.
        df.dropna(subset=["Close"], inplace=True)

        # Optional: Add any extra columns that Backtrader may expect.
        # Backtrader expects: Open, High, Low, Close, Volume (if available).
        # Fill missing values with reasonable defaults.
        if "Open" not in df.columns:
            df["Open"] = df["Close"]
        if "High" not in df.columns:
            df["High"] = df["Close"]
        if "Low" not in df.columns:
            df["Low"] = df["Close"]
        if "Volume" not in df.columns:
            df["Volume"] = 0

        # Create the datafeed.
        datafeed = bt.feeds.PandasData(dataname=df, timeframe=timeframe)
        logger.info(f"Data feed prepared with {len(df)} records.")
        return datafeed

    def add_data(self, df: pd.DataFrame, name: Optional[str] = None):
        """
        Adds a Pandas DataFrame as a data feed to Cerebro.
        """
        try:
            datafeed = self._prepare_datafeed(df)
            self.cerebro.adddata(datafeed, name=name)
            logger.info(f"Data feed '{name or 'default'}' added to Cerebro.")
        except Exception as e:
            logger.error(f"Error adding data feed: {e}", exc_info=True)
            raise

    def run_backtest(self) -> Dict[str, Any]:
        """
        Runs the backtest and returns performance metrics and analyzer data.
        """
        logger.info("Starting backtest run.")
        if self.data is not None:
            self.add_data(self.data)
        else:
            logger.error("No data provided to BacktraderEngine.")
            raise ValueError("Data must be provided to run backtest.")

        # Run Cerebro engine and capture results
        try:
            results = self.cerebro.run()
            self.results = results[0]
            logger.info("Backtest run completed successfully.")
        except Exception as e:
            logger.error("Backtest run failed.", exc_info=True)
            raise e

        # Extract analyzer results
        analyzer_results = {}
        for analyzer_name in self.analyzers.keys():
            analyzer = self.results.analyzers.get(analyzer_name)
            if analyzer:
                analyzer_results[analyzer_name] = analyzer.get_analysis()
                logger.info(f"Analyzer '{analyzer_name}' results: {analyzer_results[analyzer_name]}")
            else:
                analyzer_results[analyzer_name] = None
                logger.warning(f"Analyzer '{analyzer_name}' not found in results.")

        # Get final portfolio value
        final_value = self.cerebro.broker.getvalue()
        logger.info(f"Final portfolio value: {final_value}")

        # Optionally, store trade history if enabled.
        trade_history = None
        if self.tradehistory:
            trade_history = self._extract_trade_history()

        performance = {
            "final_value": final_value,
            "analyzers": analyzer_results,
            "trade_history": trade_history
        }
        return performance

    def _extract_trade_history(self) -> List[Dict[str, Any]]:
        """
        Extracts trade history from the strategy's observers or internal logs.
        Note: Backtrader does not provide a built-in trade history list,
        so this function may require custom strategy logging.
        """
        logger.info("Extracting trade history (if available).")
        trades = []
        try:
            # Attempt to access trade data from TradeAnalyzer if available.
            trade_analyzer = self.results.analyzers.get("trades")
            if trade_analyzer:
                analysis = trade_analyzer.get_analysis()
                # Example processing: iterate over closed trades.
                if "total" in analysis:
                    trades.append(analysis)
            else:
                logger.warning("TradeAnalyzer not found; trade history extraction skipped.")
        except Exception as e:
            logger.error(f"Error extracting trade history: {e}", exc_info=True)
        return trades

    def plot_results(self, plot_title: Optional[str] = None):
        """
        Plots the backtest results using Backtrader's built-in plot functionality.
        """
        try:
            logger.info("Plotting backtest results.")
            self.cerebro.plot(style="bar", title=plot_title)
        except Exception as e:
            logger.error(f"Error plotting results: {e}", exc_info=True)

    def run_and_report(self) -> Dict[str, Any]:
        """
        Convenience method to run the backtest, report performance metrics,
        and optionally plot the results.
        """
        performance = self.run_backtest()
        logger.info("Backtest performance metrics:")
        for key, value in performance.items():
            logger.info(f"{key}: {value}")
        return performance

# Example usage (for testing purposes):
if __name__ == "__main__":
    # Import a sample strategy for testing.
    from Utilities.strategies.example_strategy import ExampleStrategy

    # Load a sample CSV file into a DataFrame as an example.
    try:
        sample_df = pd.read_csv("sample_data.csv")
    except Exception as e:
        logger.error(f"Failed to load sample data: {e}")
        sample_df = None

    # Initialize the engine with the strategy and sample data.
    engine = BacktraderEngine(
        strategy=ExampleStrategy,
        data=sample_df,
        initial_cash=100000.0,
        commission=0.001,
        slippage_perc=0.01,
        strategy_params={"param1": 10, "param2": 20}
    )

    # Run the backtest and print performance.
    performance = engine.run_and_report()
    print("Performance Metrics:", performance)
    # Optionally plot results.
    engine.plot_results("Advanced Backtest Results")
