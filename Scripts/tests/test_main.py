import pytest
from unittest.mock import AsyncMock, patch
import pandas as pd
from Scripts.main import StockDataAgent

@pytest.mark.asyncio
async def test_main():
    # Mock the DataFetchUtils methods called in StockDataAgent
    with patch("Scripts.main.DataFetchUtils") as MockFetcher:  # Ensure the patch path matches the import in StockDataAgent
        # Create mock methods for the fetcher
        mock_fetcher = MockFetcher.return_value
        mock_fetcher.fetch_finnhub_quote = AsyncMock(return_value=pd.DataFrame([{
            "current_price": 233.72,
            "change": -3.13,
            "percent_change": -1.3215,
            "high": 234.2,
            "low": 229.72,
            "open": 233.53,
            "previous_close": 233.72
        }]))
        mock_fetcher.fetch_alpaca_data_async = AsyncMock(return_value=pd.DataFrame([{"date": "2023-01-01", "price": 140.0}]))
        mock_fetcher.fetch_alphavantage_data = AsyncMock(return_value=pd.DataFrame([{"date": "2023-01-01", "price": 141.0}]))
        mock_fetcher.fetch_news_data_async = AsyncMock(return_value=[{"headline": "Stock news"}])
        mock_fetcher.fetch_data_for_multiple_symbols = AsyncMock(return_value={"symbol": "AAPL", "data": "combined_data"})

        # Initialize the StockDataAgent
        agent = StockDataAgent()

        # Test real-time quote
        real_time_quote = await agent.get_real_time_quote("AAPL")

        # Convert DataFrame to comparable format
        expected_quote = [{
            "current_price": 233.72,
            "change": -3.13,
            "percent_change": -1.3215,
            "high": 234.2,
            "low": 229.72,
            "open": 233.53,
            "previous_close": 233.72
        }]
        assert real_time_quote.to_dict(orient="records") == expected_quote

        # Test historical data
        historical_data = await agent.get_historical_data("AAPL", "2023-01-01", "2023-12-31")
        assert historical_data.to_dict(orient="records") == [{"date": "2023-01-01", "price": 140.0}]

        # Test historical data from Alpha Vantage
        historical_data_alpha = await agent.get_historical_data_alpha_vantage("AAPL", "2023-01-01", "2023-12-31")
        assert historical_data_alpha.to_dict(orient="records") == [{"date": "2023-01-01", "price": 141.0}]
