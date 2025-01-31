# -------------------------------------------------------------------
# File: example_usage.py
# Location: TradingTool/
# Description: Example script to demonstrate usage of DataFetchUtils and trading strategies.
# -------------------------------------------------------------------

import asyncio
import logging
from aiohttp import ClientSession
from Utilities.data_fetch_utils import DataFetchUtils
from Utilities.strategies.moving_averages import MovingAverages
from Utilities.strategies.trading_strategies import TradingStrategies
from Utilities.strategies.backtester import Backtester
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create an instance of DataFetchUtils
data_fetcher = DataFetchUtils()

# Initialize strategies
logger = logging.getLogger("TradingTool")
moving_averages = MovingAverages(logger)
trading_strategies = TradingStrategies(logger)
backtester = Backtester(logger)

async def main():
    # 1. Fetch data for TSLA
    symbol = "TSLA"
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    print(f"\n🔹 Fetching data for {symbol}...\n")
    async with ClientSession() as session:
        stock_data = await data_fetcher.fetch_all_data(
            symbols=["TSLA"],
            data_sources=["Yahoo Finance", "Finnhub", "Alpaca", "NewsAPI"],
            start_date="2024-01-01",
            end_date="2024-06-01",
            interval="1d"
        )

    
    tsla_data = stock_data[symbol].get("Yahoo Finance")
    finnhub_quote = stock_data[symbol].get("Finnhub Quote")
    finnhub_metrics = stock_data[symbol].get("Finnhub Metrics")
    
    if tsla_data.empty:
        print("No data fetched for TSLA.")
        return

    # 2. Calculate Moving Averages
    tsla_data['SMA_50'] = moving_averages.sma(tsla_data['close'], window=50)
    tsla_data['EMA_20'] = moving_averages.ema(tsla_data['close'], span=20)
    tsla_data['LMA_30'] = moving_averages.lma(tsla_data['close'], window=30)
    tsla_data['REMA_10'] = moving_averages.rema(tsla_data['close'], span=10)

    # 3. Apply Trading Strategies
    # Example: Price-Minus-MA Rule using SMA_50
    tsla_data['Signal_Price_Minus_SMA50'] = trading_strategies.price_minus_ma_rule(tsla_data['close'], tsla_data['SMA_50'])

    # Example: Double Crossover Rule using EMA_20 and SMA_50
    tsla_data['Signal_Double_Crossover'] = trading_strategies.double_crossover_rule(tsla_data['EMA_20'], tsla_data['SMA_50'])

    # 4. Backtest Strategies
    # Example: Backtest Price-Minus-MA Rule
    performance_pmma = backtester.backtest(tsla_data['close'], tsla_data['Signal_Price_Minus_SMA50'])
    metrics_pmma = backtester.calculate_performance_metrics(performance_pmma)

    # Example: Backtest Double Crossover Rule
    performance_dc = backtester.backtest(tsla_data['close'], tsla_data['Signal_Double_Crossover'])
    metrics_dc = backtester.calculate_performance_metrics(performance_dc)

    # 5. Plot Results
    plt.figure(figsize=(14, 7))
    plt.plot(tsla_data['close'], label='Close Price', alpha=0.5)
    plt.plot(tsla_data['SMA_50'], label='SMA 50', alpha=0.75)
    plt.plot(tsla_data['EMA_20'], label='EMA 20', alpha=0.75)
    plt.title(f"{symbol} Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Plot Strategy Performance
    plt.figure(figsize=(14, 7))
    plt.plot(performance_pmma['Cumulative_Returns'], label='Buy and Hold')
    plt.plot(performance_pmma['Cumulative_Strategy_Returns'], label='Price-Minus-MA Strategy')
    plt.title(f"{symbol} Strategy Backtest: Price-Minus-MA vs Buy and Hold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(performance_dc['Cumulative_Returns'], label='Buy and Hold')
    plt.plot(performance_dc['Cumulative_Strategy_Returns'], label='Double Crossover Strategy')
    plt.title(f"{symbol} Strategy Backtest: Double Crossover vs Buy and Hold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.show()

    # 6. Display Performance Metrics
    print(f"\n📊 Performance Metrics for Price-Minus-MA Strategy:")
    for metric, value in metrics_pmma.items():
        print(f"{metric}: {value:.4f}")

    print(f"\n📊 Performance Metrics for Double Crossover Strategy:")
    for metric, value in metrics_dc.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
