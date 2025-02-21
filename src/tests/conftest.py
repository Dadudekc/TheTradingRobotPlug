# File: src/tests/conftest.py

import pytest
from unittest.mock import MagicMock

@pytest.fixture
def data_fetch_utils_fixture(monkeypatch):
    """
    Fixture to provide a MainDataFetcher instance for testing.
    """
    # Set required environment variables
    monkeypatch.setenv("FINNHUB_API_KEY", "test_finnhub_key")
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "test_alphavantage_key")
    monkeypatch.setenv("POLYGONIO_API_KEY", "test_polygonio_key")
    monkeypatch.setenv("NEWSAPI_API_KEY", "test_newsapi_key")

    # Override setup_logging to accept any arguments (including log_file)
    from unittest.mock import MagicMock
    import src.Utilities.config_manager as config_manager
    config_manager.setup_logging = lambda *args, **kwargs: MagicMock()
    # Now import MainDataFetcher using the correct module path
    from src.Utilities.data_fetchers.main_data_fetcher import MainDataFetcher
    return MainDataFetcher()
