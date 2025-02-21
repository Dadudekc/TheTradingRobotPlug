import pytest
from unittest.mock import AsyncMock, patch
import pandas as pd
from src.main import showcase_stock_data, StockDataAgent
from pathlib import Path

@pytest.mark.asyncio
async def test_main_function():
    # Mock StockDataAgent methods
    with patch("src.main.StockDataAgent") as MockAgent:
        mock_agent = MockAgent.return_value
        mock_agent.get_real_time_quote = AsyncMock(return_value={"current_price": 150.0})
        mock_agent.get_historical_data = AsyncMock(return_value=[{"date": "2023-01-01", "price": 140.0}])
        mock_agent.get_historical_data_alpha_vantage = AsyncMock(return_value=[{"date": "2023-01-01", "price": 141.0}])
        mock_agent.get_news = AsyncMock(return_value=[{"headline": "Stock news"}])
        mock_agent.get_combined_data = AsyncMock(return_value={"symbol": "AAPL", "data": "combined_data"})

        # Run the showcase function
        await showcase_stock_data(symbol="AAPL")

        # Verify all mocked methods were called
        mock_agent.get_real_time_quote.assert_called_once_with("AAPL")
        mock_agent.get_historical_data.assert_called_once_with("AAPL", "2023-01-01", "2023-12-31", "1Day")
        mock_agent.get_historical_data_alpha_vantage.assert_called_once_with("AAPL", "2023-01-01", "2023-12-31")
        mock_agent.get_news.assert_called_once_with("AAPL", page_size=3)  # Ensure `page_size` is included
        mock_agent.get_combined_data.assert_called_once_with("AAPL", "2023-01-01", "2023-12-31", "1Day")

@pytest.mark.asyncio
async def test_main():
    with patch("src.main.MainDataFetcher") as MockFetcher:
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

        agent = StockDataAgent()

        real_time_quote = await agent.get_real_time_quote("AAPL")
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

@pytest.mark.asyncio
async def test_get_real_time_quote_exception():
    with patch("src.main.MainDataFetcher") as MockFetcher:
        mock_fetcher = MockFetcher.return_value
        mock_fetcher.fetch_finnhub_quote = AsyncMock(side_effect=Exception("API error"))

        agent = StockDataAgent()
        result = await agent.get_real_time_quote("AAPL")

        assert result == {}
