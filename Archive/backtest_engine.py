import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import logging
from typing import Optional, List

# Optional: Set up logging for detailed reporting
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# API keys configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


# -------------------- Data Fetching -------------------- #
class DataFetcher:
    @staticmethod
    def fetch_yahoo_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            df.reset_index(inplace=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df['Source'] = 'Yahoo'
            logging.info("Fetched Yahoo Finance data successfully.")
            return df
        except Exception as e:
            logging.error(f"Yahoo Finance Error: {e}")
            return None

    @staticmethod
    def fetch_alpaca_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            logging.warning("Alpaca API keys not set.")
            return None

        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars?start={start_date}&end={end_date}&timeframe=1Day"
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json().get('bars', [])
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['t'], unit='s')
            df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df['Source'] = 'Alpaca'
            logging.info("Fetched Alpaca data successfully.")
            return df
        logging.error(f"Error fetching Alpaca data: {response.json()}")
        return None

    @staticmethod
    def fetch_polygon_data(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        if not POLYGON_API_KEY:
            logging.warning("Polygon API key not set.")
            return None

        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get('results', [])
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df['Source'] = 'Polygon'
            logging.info("Fetched Polygon data successfully.")
            return df
        logging.error("Error fetching Polygon data")
        return None

    @staticmethod
    def fetch_alpha_vantage_data(ticker: str) -> Optional[pd.DataFrame]:
        if not ALPHA_VANTAGE_API_KEY:
            logging.warning("AlphaVantage API key not set.")
            return None

        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=compact"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get("Time Series (Daily)", {})
            df = pd.DataFrame.from_dict(data, orient='index')
            df.index = pd.to_datetime(df.index)
            df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'}, inplace=True)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Date'}, inplace=True)
            df['Source'] = 'AlphaVantage'
            logging.info("Fetched AlphaVantage data successfully.")
            return df
        logging.error("Error fetching AlphaVantage data")
        return None

    @classmethod
    def fetch_all_data(cls, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        data_sources = [
            cls.fetch_yahoo_data(ticker, start_date, end_date),
            cls.fetch_alpaca_data(ticker, start_date, end_date),
            cls.fetch_polygon_data(ticker, start_date, end_date),
            cls.fetch_alpha_vantage_data(ticker)
        ]
        # Only combine available data sources
        dfs = [df for df in data_sources if df is not None]
        if not dfs:
            raise ValueError("No data could be fetched from any source.")
        combined_data = pd.concat(dfs, ignore_index=True)
        # Deduplicate based on Date & source if needed
        combined_data.drop_duplicates(subset=['Date', 'Source'], inplace=True)
        combined_data.sort_values(by='Date', inplace=True)
        return combined_data


# -------------------- Indicator Calculations -------------------- #
class IndicatorCalculator:
    @staticmethod
    def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def compute_macd(series: pd.Series, short_window: int = 12, long_window: int = 26, signal_window: int = 9):
        short_ema = series.ewm(span=short_window, adjust=False).mean()
        long_ema = series.ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal


# -------------------- Strategy Definitions -------------------- #
class BaseStrategy:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def generate_signals(self) -> pd.DataFrame:
        raise NotImplementedError("Strategy must implement generate_signals()")


class RSIMACDStrategy(BaseStrategy):
    def generate_signals(self) -> pd.DataFrame:
        # Calculate indicators
        self.df['RSI'] = IndicatorCalculator.compute_rsi(self.df['Close'])
        self.df['MACD'], self.df['Signal_Line'] = IndicatorCalculator.compute_macd(self.df['Close'])
        # Define signals
        self.df['Buy_Signal'] = (self.df['RSI'] < 30) & (self.df['MACD'] > self.df['Signal_Line'])
        self.df['Sell_Signal'] = (self.df['RSI'] > 70) & (self.df['MACD'] < self.df['Signal_Line'])
        return self.df


# -------------------- Backtesting Engine -------------------- #
class Backtester:
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0
        self.trade_log: List[tuple] = []

    def run(self):
        for i in range(len(self.df)):
            price = self.df['Close'].iloc[i]
            date = self.df['Date'].iloc[i]
            # Buy condition
            if self.df['Buy_Signal'].iloc[i]:
                shares = int(self.balance // price)
                if shares > 0:
                    self.balance -= shares * price
                    self.position += shares
                    self.trade_log.append(('BUY', date, price, shares))
                    logging.info(f"BUY on {date.date()}: {shares} shares at {price}")
            # Sell condition
            elif self.df['Sell_Signal'].iloc[i] and self.position > 0:
                self.balance += self.position * price
                self.trade_log.append(('SELL', date, price, self.position))
                logging.info(f"SELL on {date.date()}: {self.position} shares at {price}")
                self.position = 0

    def final_portfolio_value(self) -> float:
        return self.balance + self.position * self.df['Close'].iloc[-1]

    def performance_metrics(self) -> dict:
        final_value = self.final_portfolio_value()
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100

        # Win rate calculation based on executed SELL trades
        sell_trades = [trade for trade in self.trade_log if trade[0] == 'SELL']
        wins = 0
        for trade in sell_trades:
            trade_date = trade[1]
            # Find corresponding BUY trade (naively matching by order)
            buy_trades = [t for t in self.trade_log if t[0] == 'BUY' and t[1] < trade_date]
            if buy_trades:
                avg_buy_price = np.mean([t[2] for t in buy_trades])
                if trade[2] > avg_buy_price:
                    wins += 1
        win_rate = (wins / len(sell_trades)) * 100 if sell_trades else 0

        # Calculate max drawdown (a simple version)
        running_max = self.df['Close'].cummax()
        drawdown = (running_max - self.df['Close']) / running_max * 100
        max_drawdown = drawdown.max()

        return {
            "final_value": final_value,
            "total_return_pct": total_return,
            "win_rate_pct": win_rate,
            "max_drawdown_pct": max_drawdown
        }

    def plot_results(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.df['Date'], self.df['Close'], label="Close Price", alpha=0.5)
        plt.scatter(self.df['Date'][self.df['Buy_Signal']], self.df['Close'][self.df['Buy_Signal']],
                    label="Buy Signal", marker="^", color="green", s=100)
        plt.scatter(self.df['Date'][self.df['Sell_Signal']], self.df['Close'][self.df['Sell_Signal']],
                    label="Sell Signal", marker="v", color="red", s=100)
        plt.title("Trading Strategy: RSI & MACD Signals")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()


# -------------------- Main Execution -------------------- #
def run_backtester(ticker: str, start_date: str, end_date: str, output_file: str):
    # Fetch data and save to CSV cache
    df = DataFetcher.fetch_all_data(ticker, start_date, end_date)
    df.to_csv(output_file, index=False)
    logging.info(f"Data saved to {output_file}")

    # Convert Date column to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)

    # Apply strategy signals
    strategy = RSIMACDStrategy(df)
    df_signals = strategy.generate_signals()

    # Run backtest simulation
    backtester = Backtester(df_signals)
    backtester.run()
    metrics = backtester.performance_metrics()
    logging.info("Performance Metrics: %s", metrics)

    # Plot results
    backtester.plot_results()

    # Optionally, display trade log
    trade_df = pd.DataFrame(backtester.trade_log, columns=["Type", "Date", "Price", "Shares"])
    print(trade_df)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultimate Backtester")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol for backtesting")
    parser.add_argument("--start_date", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=datetime.today().strftime('%Y-%m-%d'), help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_file", type=str, default="data.csv", help="Output CSV file path for data caching")
    
    args = parser.parse_args()
    run_backtester(args.symbol, args.start_date, args.end_date, args.output_file)
