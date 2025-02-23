"""
File: portfolio_backtester.py
Location: src/Utilities/portfolio_management

Description:
    A multi-asset portfolio backtester that runs individual backtests for
    each asset and aggregates results to simulate a diversified portfolio.
    Features include:
      - Allocating initial capital across assets (equal or custom weights).
      - Running separate backtests (via BacktraderEngine) for each asset.
      - Aggregating final portfolio value and calculating overall returns.
      - Computing daily returns and correlation matrix for the assets.
      - Detailed logging and error handling.
      - Reporting individual asset performance and portfolio-level metrics.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Import our Backtrader engine.
from Utilities.backtesting.backtrader_engine import BacktraderEngine

# Setup a module-level logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class PortfolioBacktester:
    """
    Executes backtests across multiple assets and aggregates performance metrics
    to simulate a diversified portfolio.
    """
    def __init__(self,
                 symbol_data: Dict[str, pd.DataFrame],
                 strategy: type,
                 initial_capital: float = 100000.0,
                 allocation: Optional[Dict[str, float]] = None,
                 engine_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the portfolio backtester.

        Args:
            symbol_data (Dict[str, pd.DataFrame]): Mapping from symbol to its historical data.
            strategy (type): Backtrader Strategy class to use for each asset.
            initial_capital (float): Total initial capital for the portfolio.
            allocation (Dict[str, float], optional): Allocation weights per symbol.
                If None, equal weight is assumed.
            engine_params (Dict[str, Any], optional): Extra parameters to pass to each engine.
        """
        self.symbol_data = symbol_data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.allocation = allocation or self._equal_allocation()
        self.engine_params = engine_params or {}
        self.asset_performance: Dict[str, Dict[str, Any]] = {}
        self.daily_returns: Dict[str, pd.Series] = {}
        self.trade_histories: Dict[str, Any] = {}
        logger.info("PortfolioBacktester initialized.")

    def _equal_allocation(self) -> Dict[str, float]:
        """
        Creates an equal allocation dictionary if no custom allocation is provided.
        """
        symbols = list(self.symbol_data.keys())
        equal_weight = 1 / len(symbols)
        allocation = {sym: equal_weight for sym in symbols}
        logger.info(f"Equal allocation set: {allocation}")
        return allocation

    def run_asset_backtest(self, symbol: str, df: pd.DataFrame, cash: float) -> Tuple[Dict[str, Any], pd.Series]:
        """
        Runs the backtest for a single asset using BacktraderEngine.
        
        Args:
            symbol (str): Asset symbol.
            df (pd.DataFrame): Historical data for the asset.
            cash (float): Initial cash allocation for this asset.
        
        Returns:
            Tuple containing performance metrics and daily returns as a Pandas Series.
        """
        logger.info(f"Running backtest for {symbol} with allocated cash: {cash}")
        try:
            engine = BacktraderEngine(
                strategy=self.strategy,
                data=df,
                initial_cash=cash,
                commission=self.engine_params.get("commission", 0.001),
                slippage_perc=self.engine_params.get("slippage_perc", 0.0),
                strategy_params=self.engine_params.get("strategy_params", {})
            )
            # Run the backtest.
            performance = engine.run_and_report()
            # Optionally, plot results if desired (commented out for batch runs).
            # engine.plot_results(f"{symbol} Backtest")
            
            # Attempt to extract daily returns from the engine's trade history or price series.
            # For demonstration, assume we compute returns from the provided data.
            df = df.copy()
            if "Date" in df.columns:
                df["datetime"] = pd.to_datetime(df["Date"])
                df.set_index("datetime", inplace=True, drop=True)
            df.sort_index(inplace=True)
            daily_returns = df["Close"].pct_change().fillna(0)
            logger.info(f"Completed backtest for {symbol}. Final Value: {performance.get('final_value')}")
            return performance, daily_returns
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}", exc_info=True)
            return {"error": str(e)}, pd.Series(dtype=float)

    def run(self) -> Dict[str, Any]:
        """
        Runs backtests for all assets, aggregates portfolio performance, and computes portfolio-level metrics.
        
        Returns:
            A dictionary containing:
              - individual asset performance metrics,
              - aggregated portfolio final value,
              - overall portfolio return percentage,
              - correlation matrix of asset returns.
        """
        logger.info("Starting portfolio backtest...")
        portfolio_final_value = 0.0
        asset_returns_list = []
        symbols = list(self.symbol_data.keys())

        # Run individual asset backtests.
        for symbol in symbols:
            df = self.symbol_data[symbol]
            allocated_cash = self.initial_capital * self.allocation.get(symbol, 0)
            performance, daily_ret = self.run_asset_backtest(symbol, df, allocated_cash)
            self.asset_performance[symbol] = performance
            self.daily_returns[symbol] = daily_ret
            asset_final_value = performance.get("final_value", 0)
            portfolio_final_value += asset_final_value
            asset_returns_list.append(daily_ret.rename(symbol))
            self.trade_histories[symbol] = performance.get("trade_history", None)

        # Create a DataFrame of daily returns for correlation analysis.
        if asset_returns_list:
            returns_df = pd.concat(asset_returns_list, axis=1).fillna(0)
            correlation_matrix = returns_df.corr()
        else:
            correlation_matrix = pd.DataFrame()

        overall_return_pct = ((portfolio_final_value - self.initial_capital) / self.initial_capital) * 100
        logger.info(f"Portfolio final value: {portfolio_final_value}")
        logger.info(f"Overall portfolio return: {overall_return_pct:.2f}%")

        # Prepare aggregated results.
        results = {
            "asset_performance": self.asset_performance,
            "portfolio_final_value": portfolio_final_value,
            "overall_return_pct": overall_return_pct,
            "correlation_matrix": correlation_matrix,
            "trade_histories": self.trade_histories
        }
        logger.info("Portfolio backtest completed successfully.")
        return results

    def report(self) -> None:
        """
        Prints a summary report of the portfolio performance.
        """
        results = self.run()
        print("----- Portfolio Backtest Report -----")
        print(f"Total Initial Capital: {self.initial_capital}")
        print(f"Final Portfolio Value: {results['portfolio_final_value']:.2f}")
        print(f"Overall Return: {results['overall_return_pct']:.2f}%\n")
        print("Individual Asset Performance:")
        for symbol, perf in results["asset_performance"].items():
            final_val = perf.get("final_value", "N/A")
            print(f"  {symbol}: Final Value = {final_val}")
        print("\nAsset Return Correlation Matrix:")
        print(results["correlation_matrix"])
        print("---------------------------------------")

# -------------------------------------------------------------------------------------------------
# Example Usage:
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # For demonstration, load sample CSV files for different symbols.
    # In practice, these would come from your data fetcher or database.
    try:
        data_aapl = pd.read_csv("sample_data_AAPL.csv")
        data_msft = pd.read_csv("sample_data_MSFT.csv")
        data_tsla = pd.read_csv("sample_data_TSLA.csv")
    except Exception as e:
        logger.error(f"Error loading sample CSVs: {e}")
        data_aapl, data_msft, data_tsla = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Dictionary mapping symbols to their respective DataFrames.
    symbol_data = {
        "AAPL": data_aapl,
        "MSFT": data_msft,
        "TSLA": data_tsla
    }

    # Optional: Define custom allocation if desired. Otherwise, equal allocation is used.
    allocation = {
        "AAPL": 0.4,
        "MSFT": 0.3,
        "TSLA": 0.3
    }

    # Import a sample strategy (must be implemented in your strategies module).
    try:
        from Utilities.strategies.example_strategy import ExampleStrategy
    except ImportError:
        logger.error("ExampleStrategy not found. Please implement a strategy.")
        ExampleStrategy = None

    if ExampleStrategy is not None:
        portfolio_backtester = PortfolioBacktester(
            symbol_data=symbol_data,
            strategy=ExampleStrategy,
            initial_capital=150000.0,
            allocation=allocation,
            engine_params={
                "commission": 0.001,
                "slippage_perc": 0.005,
                "strategy_params": {"param1": 15, "param2": 30}
            }
        )

        # Run the portfolio backtest and display the report.
        portfolio_results = portfolio_backtester.run()
        portfolio_backtester.report()
    else:
        logger.error("No valid strategy available for portfolio backtesting.")
