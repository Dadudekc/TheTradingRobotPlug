"""
File: test_backtester.py
Location: tests/

Test suite for the backtester.py module.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.Utilities.strategies.backtester import (
    BacktestRunner,
    BaseStrategy,
    ClassicBacktester
)

# A sample strategy for testing
class TestStrategy(BaseStrategy):
    """
    A minimalistic strategy that does no trades.
    Just for testing the pipeline.
    """
    def next(self):
        pass  # does nothing


@pytest.fixture
def sample_df():
    """
    Returns a small sample DataFrame with 'Date' and 'Close' for testing.
    """
    data = {
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Close": [100.0, 105.0, 102.0],
    }
    df = pd.DataFrame(data)
    return df


@patch("src.Utilities.strategies.backtester.DataOrchestrator")
def test_backtest_runner_no_data(mock_orchestrator):
    """
    Test that BacktestRunner returns an error dict if no data is fetched.
    """
    # Setup mock to return an empty DataFrame
    instance = mock_orchestrator.return_value
    instance.fetch_all_data.return_value = {
        "FAKE": {}  # no sources
    }

    runner = BacktestRunner(TestStrategy)
    results = runner.run("FAKE", "2023-01-01", "2023-01-10", "1d")
    assert "error" in results, "Should return an error key when no data is available."


@patch("src.Utilities.strategies.backtester.DataOrchestrator")
def test_backtest_runner_with_data(mock_orchestrator, sample_df):
    """
    Test that BacktestRunner handles sample data properly.
    """
    instance = mock_orchestrator.return_value
    # Return a dict with one source that has sample data
    instance.fetch_all_data.return_value = {
        "FAKE": {
            "Yahoo Finance": sample_df
        }
    }

    runner = BacktestRunner(TestStrategy, initial_balance=10000.0)
    results = runner.run("FAKE", "2023-01-01", "2023-01-10", "1d")

    # Confirm it returns a metric dict (e.g., final_value, total_return_pct)
    assert "final_value" in results
    assert "total_return_pct" in results
    assert results["final_value"] == 10000.0, "No trades should have been made, final_value remains 10000."
    assert results["total_return_pct"] == 0.0, "No trades => 0% return."


def test_classic_backtester(sample_df):
    """
    Test the ClassicBacktester logic in isolation.
    """
    # Add Buy_Signal and Sell_Signal to sample_df
    sample_df["Buy_Signal"] = [True, False, False]
    sample_df["Sell_Signal"] = [False, False, True]
    sample_df["Close"] = [100.0, 110.0, 120.0]  # Force a known upward movement

    backtester = ClassicBacktester(sample_df, initial_balance=1000.0)
    backtester.run()
    metrics = backtester.performance_metrics()

    # Expect 1 buy trade on first row, then 1 sell trade on last row
    assert len(backtester.trade_log) == 2
    buy_trade, sell_trade = backtester.trade_log
    assert buy_trade[0] == "BUY"
    assert sell_trade[0] == "SELL"

    # final portfolio value should reflect the difference in price
    assert "final_value" in metrics
    assert metrics["final_value"] > 1000.0, "Should have gained some profit due to sell at higher price."


@pytest.mark.asyncio
async def test_fetch_data_async(sample_df):
    """
    (Optional) Test the async fetch_data method in isolation with a real DataOrchestrator 
    or a partial mock, ensuring it merges frames properly.
    """
    # This could require partial mocking of DataOrchestrator, or an integration test
    pass
