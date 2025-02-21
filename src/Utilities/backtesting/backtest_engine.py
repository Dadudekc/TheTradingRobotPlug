import argparse
import pandas as pd
import numpy as np
import backtrader as bt
import os
from Utilities.datamain_data_fetcher import DataOrchestrator
from Utilities.data_processing.Technical_Indicators.indicator_aggregator import MACD
from evaluation.metrics import calculate_performance
from evaluation.visualization import plot_backtest_results

class MACDStrategy(bt.Strategy):
    params = (
        ('fast', 12),
        ('slow', 26),
        ('signal', 9),
    )

    def __init__(self):
        self.macd, self.signal, _ = MACD(self.data.close, self.params.fast, self.params.slow, self.params.signal)
        self.crossover = bt.indicators.CrossOver(self.macd, self.signal)

    def next(self):
        if self.position:
            if self.crossover < 0:  # MACD crosses below signal
                self.close()
        else:
            if self.crossover > 0:  # MACD crosses above signal
                self.buy()


def run_backtest(symbol, strategy, timeframe):
    data_path = f"data/{symbol}_{timeframe}.csv"
    
    if not os.path.exists(data_path):
        print(f"Fetching data for {symbol}...")
        df = fetch_historical_data(symbol, timeframe)
        df.to_csv(data_path, index=False)
    else:
        print(f"Loading cached data for {symbol}...")
        df = pd.read_csv(data_path)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(MACDStrategy if strategy == 'MACD' else None)
    cerebro.adddata(data)
    cerebro.run()

    performance = calculate_performance(df)
    print("Performance Report:", performance)

    plot_backtest_results(df, symbol, timeframe)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol for backtesting")
    parser.add_argument("--strategy", type=str, required=True, choices=['MACD'], help="Trading strategy to use")
    parser.add_argument("--timeframe", type=str, required=True, help="Timeframe for backtesting")
    
    args = parser.parse_args()
    run_backtest(args.symbol, args.strategy, args.timeframe)
