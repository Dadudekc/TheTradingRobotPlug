# File: src/Utilities/backtesting/run_macd_backtest.py

import os
import argparse
import asyncio
import pandas as pd

# The advanced engine
from Utilities.backtesting.backtrader_engine import BacktraderEngine
# Our newly created MACD Strategy
from src.Utilities.strategies.macd_crossover import MACDStrategy
# The orchestrator for data fetching
from src.Utilities.data_fetchers.main_data_fetcher import DataOrchestrator

# (Optional) if you want to combine with your custom metrics
# from src.Utilities.strategies.evaluation.metrics import calculate_performance
# from src.Utilities.strategies.evaluation.visualization import plot_backtest_results


async def fetch_data(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """
    Fetches historical data for 'symbol' using DataOrchestrator
    and returns a Pandas DataFrame with columns suitable for Backtrader.
    """
    orchestrator = DataOrchestrator()
    
    print(f"Fetching data for {symbol} from {start_date} to {end_date} with interval={interval}...")
    df = await orchestrator.fetch_stock_data_async(
        ticker=symbol,
        start_date=start_date,
        end_date=end_date,
        interval=interval
    )

    if df.empty:
        print("⚠️  No data returned.")
        return df
    
    # If DataFrame does not have a 'Date' column, rename the index or 'datetime'
    if "datetime" in df.columns:
        df.rename(columns={"datetime": "Date"}, inplace=True)
    elif df.index.name == "datetime":
        df.reset_index(inplace=True)
        df.rename(columns={"datetime": "Date"}, inplace=True)

    # Backtrader expects these columns at minimum:
    # Date, Open, High, Low, Close, Volume
    # So rename if necessary.
    # Example: If orchestrator returned all-lower columns, rename them:
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }
    for old_col, new_col in rename_map.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    # Make sure 'Date' is converted to datetime and sort
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    return df


async def main(symbol: str, start_date: str, end_date: str, interval: str, initial_cash: float):
    # 1) Fetch data
    df = await fetch_data(symbol, start_date, end_date, interval)
    if df.empty:
        return

    # 2) Instantiate the advanced BacktraderEngine
    engine = BacktraderEngine(
        strategy=MACDStrategy,
        data=df,
        initial_cash=initial_cash,
        commission=0.001,
        slippage_perc=0.0,
        strategy_params={"fast": 12, "slow": 26, "signal": 9}
    )

    # 3) Run backtest & get results
    performance = engine.run_and_report()
    print("Backtest Performance:", performance)

    # 4) (Optional) Plot using Backtrader’s built-in or your own methods
    engine.plot_results(plot_title=f"{symbol} MACD Backtest")
    
    # 5) (Optional) If you want to also use your custom evaluation scripts:
    # if not df.empty:
    #     from src.Utilities.strategies.evaluation.metrics import calculate_performance
    #     from src.Utilities.strategies.evaluation.visualization import plot_backtest_results
    #
    #     perf_metrics = calculate_performance(df)  # or pass the final portfolio/time-series
    #     print("Custom Performance Metrics:", perf_metrics)
    #     plot_backtest_results(df, symbol, interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="TSLA", help="Stock symbol (e.g. TSLA, AAPL)")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-03-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (1d, 1h, etc.)")
    parser.add_argument("--initial-cash", type=float, default=10000.0, help="Initial broker cash")
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(
        main(
            args.symbol,
            args.start_date,
            args.end_date,
            args.interval,
            args.initial_cash
        )
    )
