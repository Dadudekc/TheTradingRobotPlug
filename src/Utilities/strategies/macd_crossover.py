"""
File: macd_crossover.py
Location: src/Utilities/strategies

Description:
    1. Fetches historical data through DataOrchestrator (no inline fetch).
    2. Uses AllIndicatorsUnifier to compute indicators (RSI, MACD, etc.).
    3. Defines MACD crossover strategy in Backtrader.
    4. Provides a run_backtest(...) function that can be extended for
       multiple strategies (just adapt the chosen strategy class).
"""

import argparse
import logging
import os
import asyncio
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
from typing import List, Dict, Any, Optional

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------------------------------------------
# 1) Import DataOrchestrator & Indicator Aggregator
# -------------------------------------------------------------------
from src.Utilities.data_fetchers.main_data_fetcher import DataOrchestrator
from Utilities.data_processing.Technical_Indicators.indicator_aggregator import AllIndicatorsUnifier

# -------------------------------------------------------------------
# 2) Extra modules for performance metrics & plotting
# -------------------------------------------------------------------
from evaluation.metrics import calculate_performance
from evaluation.visualization import plot_backtest_results


class MACDStrategy(bt.Strategy):
    """
    A simple MACD crossover strategy for Backtrader.
    Buys when MACD crosses above its signal line, sells otherwise.
    """

    params = (("fast", 12), ("slow", 26), ("signal", 9))

    def __init__(self):
        """
        If you have a custom MACD aggregator function from your AllIndicatorsUnifier,
        you can integrate it below. This example uses a hypothetical aggregator function
        `MACD(...)` from `indicator_aggregator`.
        """
        # Suppose you have a function MACD(...) to compute indicator lines
        from Utilities.data_processing.Technical_Indicators.indicator_aggregator import MACD

        self.macd, self.signal, _ = MACD(self.data.close,
                                         self.params.fast,
                                         self.params.slow,
                                         self.params.signal)
        self.crossover = bt.indicators.CrossOver(self.macd, self.signal)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        else:
            if self.crossover < 0:
                self.close()


async def _fetch_symbol_data(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """
    Asynchronously fetches data for a single symbol from DataOrchestrator,
    unifies all sources, sorts them by 'Date', and returns one combined DataFrame.

    Args:
        symbol (str): Stock ticker symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        interval (str): e.g., "1d", "1Day".

    Returns:
        pd.DataFrame: Combined historical data for the symbol.
    """
    orchestrator = DataOrchestrator()
    symbol_map = await orchestrator.fetch_all_data([symbol], start_date, end_date, interval=interval)
    # symbol_map: { "AAPL": { "Yahoo Finance": df, "Alpaca": df, ... } }
    data_dict = symbol_map.get(symbol, {})
    # Combine the DataFrames
    sources = [df for df in data_dict.values() if not df.empty]
    if not sources:
        logging.error(f"No data fetched for {symbol} from any source.")
        return pd.DataFrame()

    combined = pd.concat(sources, ignore_index=True)
    combined.drop_duplicates(subset=["Date"], inplace=True)
    combined.sort_values(by="Date", inplace=True)
    return combined


def run_backtest(symbol: str, start_date: str, end_date: str, timeframe: str, output_file: str):
    """
    1) Fetch data from DataOrchestrator.
    2) Optionally apply AllIndicatorsUnifier if you want to compute RSI etc. in the DataFrame.
    3) Convert data to Backtrader feed & run the MACDStrategy.
    4) Evaluate performance & visualize results.

    Args:
        symbol (str): Stock ticker symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        timeframe (str): e.g. "1d"
        output_file (str): Output path to cache CSV data.
    """
    # 1) Check if data is cached
    if os.path.exists(output_file):
        logging.info(f"Loading cached data from {output_file}")
        df = pd.read_csv(output_file)
    else:
        logging.info(f"Fetching data for {symbol} from {start_date} to {end_date} ...")
        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(_fetch_symbol_data(symbol, start_date, end_date, timeframe))
        if df.empty:
            logging.error("Data fetch failed, cannot proceed with backtest.")
            return
        df.to_csv(output_file, index=False)
        logging.info(f"Data saved to {output_file}")

    # Convert "Date" column to datetime, if present
    if "Date" in df.columns:
        df["datetime"] = pd.to_datetime(df["Date"])
    else:
        logging.warning("No 'Date' column found; expecting 'datetime'.")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Set index for Backtrader
    df.set_index("datetime", inplace=True)
    df.dropna(subset=["Close"], inplace=True)

    # 2) Convert DataFrame to Backtrader feed
    data_feed = bt.feeds.PandasData(dataname=df)

    # 3) Run backtest with MACDStrategy
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MACDStrategy)
    cerebro.adddata(data_feed)
    cerebro.run()

    # 4) Evaluate performance
    # Provide minimal or custom logic from evaluation.metrics
    perf = calculate_performance(df)
    logging.info(f"Performance: {perf}")

    # 5) Visualize
    plot_backtest_results(df, symbol, timeframe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MACD Crossover Backtest using DataOrchestrator + Backtrader.")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol (e.g. AAPL)")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=datetime.today().strftime('%Y-%m-%d'), help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="1d", help="Data timeframe (e.g. 1d, 1Day)")
    parser.add_argument("--output_file", type=str, default="data.csv", help="Where to cache the fetched CSV data")

    args = parser.parse_args()
    run_backtest(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        output_file=args.output_file
    )
