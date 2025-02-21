# evaluation/visualization.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_backtest_results(df: pd.DataFrame, symbol: str, timeframe: str):
    plt.figure(figsize=(12, 6))
    plt.title(f"Backtest Results for {symbol} ({timeframe})")
    plt.plot(df["Close"], label="Close")
    plt.legend()
    plt.show()
