"""
File: test_backtester.py
Location: tests/

Description:
    Unit tests for data fetching, technical indicators, strategy logic, 
    and backtesting execution using pytest and unittest.mock.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime
from src.Utilities.data_fetchers.main_data_fetcher import DataOrchestrator
from Utilities.data_processing.Technical_Indicators.indicator_unifier import AllIndicatorsUnifier
from src.Utilities.indicators.indicator_calculator import IndicatorCalculator
from src.Utilities.strategies.rsi_macd_strategy import RSIMACDStrategy
from src.Utilities.backtest_engine.backtrader_engine import Backtester

# --------------------- TEST DATA --------------------- #

@pytest.fixture
def sample_stock_data():
    """Generate sample stock price data for testing"""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    data = {
        "Date": dates,
        "Open": np.linspace(100, 110, 10),
        "High": np.linspace(102, 112, 10),
        "Low": np.linspace(98, 108, 10),
        "Close": np.linspace(99, 109, 10),
        "Volume": np.random.randint(1000, 5000, 10),
    }
    return pd.DataFrame(data)


# --------------------- TEST DataFetcher --------------------- #

@patch("yfinance.Ticker")
def test_fetch_yahoo_data(mock_ticker):
    """Test Yahoo Finance data fetching"""
    mock_stock = MagicMock()
    mock_stock.history.return_value = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5),
        "Open": [100, 101, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [95, 96, 97, 98, 99],
        "Close": [102, 103, 104, 105, 106],
        "Volume": [1000, 1500, 1200, 1800, 1300],
    })
    mock_ticker.return_value = mock_stock

    df = DataOrchestrator().fetch_yahoo_data("AAPL", "2024-01-01", "2024-01-05")
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Close" in df.columns
    assert df.iloc[0]["Close"] == 102


@patch("requests.get")
def test_fetch_alpaca_data(mock_get):
    """Test Alpaca data fetching with mock API response"""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "bars": [
            {"t": 1704067200, "o": 100, "h": 105, "l": 95, "c": 102, "v": 1500},
            {"t": 1704153600, "o": 101, "h": 106, "l": 96, "c": 103, "v": 1600},
        ]
    }

    df = DataOrchestrator().fetch_alpaca_data("AAPL", "2024-01-01", "2024-01-05")
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Close" in df.columns
    assert df.iloc[0]["Close"] == 102


# --------------------- TEST IndicatorCalculator --------------------- #

def test_compute_rsi(sample_stock_data):
    """Test RSI calculation"""
    sample_stock_data["RSI"] = IndicatorCalculator.compute_rsi(sample_stock_data["Close"], window=5)
    
    assert "RSI" in sample_stock_data.columns
    assert sample_stock_data["RSI"].isnull().sum() == 4  # First few values should be NaN
    assert sample_stock_data["RSI"].dtype == np.float64


def test_compute_macd(sample_stock_data):
    """Test MACD calculation"""
    macd, signal = IndicatorCalculator.compute_macd(sample_stock_data["Close"])
    
    assert isinstance(macd, pd.Series)
    assert isinstance(signal, pd.Series)
    assert len(macd) == len(sample_stock_data)
    assert len(signal) == len(sample_stock_data)
    assert macd.dtype == np.float64


# --------------------- TEST RSIMACDStrategy --------------------- #

def test_rsi_macd_strategy_signals(sample_stock_data):
    """Test that the strategy correctly assigns buy/sell signals"""
    strategy = RSIMACDStrategy(sample_stock_data)
    df_signals = strategy.generate_signals()
    
    assert "Buy_Signal" in df_signals.columns
    assert "Sell_Signal" in df_signals.columns
    assert df_signals["Buy_Signal"].dtype == bool
    assert df_signals["Sell_Signal"].dtype == bool


# --------------------- TEST Backtester --------------------- #

def test_backtester_run(sample_stock_data):
    """Test backtest execution with generated signals"""
    sample_stock_data["Buy_Signal"] = [False, True, False, False, True] + [False] * 5
    sample_stock_data["Sell_Signal"] = [False, False, True, False, False] + [False] * 5

    backtester = Backtester(sample_stock_data, initial_balance=10000)
    backtester.run()

    assert len(backtester.trade_log) > 0
    assert backtester.final_portfolio_value() >= 10000  # Should not be negative


def test_performance_metrics(sample_stock_data):
    """Test calculation of performance metrics"""
    sample_stock_data["Buy_Signal"] = [False, True, False, False, True] + [False] * 5
    sample_stock_data["Sell_Signal"] = [False, False, True, False, False] + [False] * 5

    backtester = Backtester(sample_stock_data, initial_balance=10000)
    backtester.run()
    metrics = backtester.performance_metrics()

    assert "final_value" in metrics
    assert "total_return_pct" in metrics
    assert "win_rate_pct" in metrics
    assert "max_drawdown_pct" in metrics
    assert metrics["total_return_pct"] >= -100  # Sanity check


def test_trade_log_structure(sample_stock_data):
    """Test trade log structure and execution"""
    sample_stock_data["Buy_Signal"] = [False, True, False, False, True] + [False] * 5
    sample_stock_data["Sell_Signal"] = [False, False, True, False, False] + [False] * 5

    backtester = Backtester(sample_stock_data, initial_balance=10000)
    backtester.run()

    assert isinstance(backtester.trade_log, list)
    assert len(backtester.trade_log) > 0
    assert isinstance(backtester.trade_log[0], tuple)
    assert len(backtester.trade_log[0]) == 4  # (Type, Date, Price, Shares)


# --------------------- RUNNING TESTS --------------------- #

if __name__ == "__main__":
    pytest.main()
