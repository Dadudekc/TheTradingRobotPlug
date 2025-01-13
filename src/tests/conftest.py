# src/tests/conftest.py
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def data_fetch_utils_fixture(monkeypatch):
    """
    Fixture to provide a mocked DataFetchUtils instance.
    """
    # Mock environment variables for all required APIs
    monkeypatch.setenv("FINNHUB_API_KEY", "test_finnhub_key")
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test_alphavantage_key")
    monkeypatch.setenv("POLYGONIO_API_KEY", "test_polygonio_key")
    monkeypatch.setenv("NEWSAPI_API_KEY", "test_newsapi_key")

    # Mock logging
    with patch("src.Utilities.config_manager.setup_logging") as mock_logging:
        mock_logger = MagicMock()
        mock_logging.return_value = mock_logger

        # Import DataFetchUtils **after** mocking
        from src.Utilities.data_fetch_utils import DataFetchUtils
        return DataFetchUtils(logger=mock_logger)
