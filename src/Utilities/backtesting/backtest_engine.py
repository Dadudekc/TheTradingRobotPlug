import argparse
import asyncio
import os
import pandas as pd
import backtrader as bt

# Data Orchestrator from your existing setup
from src.Utilities.data_fetchers.main_data_fetcher import DataOrchestrator

# MACD aggregator from your existing technical indicators
from src.Utilities.data_processing.Technical_Indicators.indicator_aggregator import MACD

# Performance & Visualization (adjust the import path based on your folder structure)
from src.Utilities.strategies.evaluation.metrics import calculate_performance
from src.Utilities.strategies.evaluation.visualization import plot_backtest_results


class MACDStrategy(bt.Strategy):
    params = (
        ("fast", 12),
        ("slow", 26),
        ("signal", 9),
    )

    def __init__(self):
        # Build MACD indicators from the aggregator
        self.macd, self.signal, _ = MACD(
            self.data.close, 
            self.params.fast, 
            self.params.slow, 
            self.params.signal
        )
        self.crossover = bt.indicators.CrossOver(self.macd, self.signal)

    def next(self):
        # If we have an open position, check for exit
        if self.position:
            # If MACD crosses below signal, exit
            if self.crossover < 0:
                self.close()
        else:
            # If MACD crosses above signal, go long
            if self.crossover > 0:
                self.buy()


async def fetch_and_cache_data(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Uses DataOrchestrator to fetch data asynchronously.
    Caches the data to a CSV in `data/` so we don't fetch repeatedly.
    """
    data_path = f"data/{symbol}_{timeframe}.csv"
    if os.path.exists(data_path):
        print(f"Loading cached data for {symbol} from {data_path}...")
        df = pd.read_csv(data_path)
        return df

    print(f"Fetching fresh data for {symbol} via DataOrchestrator...")
    orchestrator = DataOrchestrator()

    # The main_data_fetcher uses Yahoo / Alpaca / etc. in parallel,
    # but we only need one set of OHLC data for backtesting.
    # This fetches from Yahoo by default in your code. Adjust as needed.
    df = await orchestrator.fetch_stock_data_async(
        ticker=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=timeframe
    )

    # Ensure we have standard columns for Backtrader
    # e.g. date/time -> 'datetime', rename columns to 'open', 'high', 'low', 'close', 'volume'
    if not df.empty:
        # Make sure we have a datetime column
        if 'datetime' not in df.columns:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'datetime'}, inplace=True)

        # If your columns differ, rename them accordingly
        rename_map = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            # Adjust if your DataFrame has different col names
        }
        df.rename(columns=rename_map, inplace=True)
        df.to_csv(data_path, index=False)
    else:
        print(f"⚠️ No data returned for {symbol}. Please check data fetcher logs.")
    
    return df


def run_backtest(symbol: str, strategy: str, timeframe: str, start_date: str, end_date: str):
    """
    1) Fetch or load data for the given symbol & timeframe.
    2) Run Backtrader with the specified strategy.
    3) Calculate & print performance metrics.
    4) Plot the backtest results (candles + signals).
    """

    # Use asyncio to fetch data (DataOrchestrator is async)
    df = asyncio.run(fetch_and_cache_data(symbol, timeframe, start_date, end_date))

    if df.empty:
        print(f"❌ No data available for symbol {symbol}. Aborting backtest.")
        return

    # Convert 'datetime' to actual datetime type for Backtrader
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    # Create Backtrader data feed
    bt_data = bt.feeds.PandasData(dataname=df)

    # Set up Backtrader
    cerebro = bt.Cerebro()
    # You could add more analyzers or broker configurations here
    # e.g. cerebro.broker.set_cash(10000)

    # Add your strategy
    if strategy == "MACD":
        cerebro.addstrategy(MACDStrategy)
    else:
        print(f"❌ Strategy '{strategy}' not recognized.")
        return

    cerebro.adddata(bt_data)
    print(f"Starting Backtest for {symbol} [{strategy} | {timeframe}]...")
    results = cerebro.run()  # run the strategy
    cerebro.plot()           # optional to see chart

    # Evaluate performance
    # (Ensure your `calculate_performance` expects or references the price data in `df`)
    performance = calculate_performance(df)
    print("Performance Report:", performance)

    # Plot or log further results
    plot_backtest_results(df, symbol, timeframe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol for backtesting")
    parser.add_argument("--strategy", type=str, required=True, choices=["MACD"], help="Trading strategy to use")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe for backtesting (e.g. '1d', '1h', '30m')")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date for data fetching")
    parser.add_argument("--end-date", type=str, default=None, help="End date for data fetching")

    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol,
        strategy=args.strategy,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
    )
