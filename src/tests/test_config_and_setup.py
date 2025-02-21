"""
test_config_and_setup.py

Contains tests for configuration, environment variables,
logger setup, Alpaca initialization, and related setup logic.
"""
from pathlib import Path
import pytest
import os
from unittest.mock import patch, MagicMock
from Utilities.data_fetchers.main_data_fetcher import MainDataFetcher

def test_initialize_alpaca_invalid_config(monkeypatch):
    """
    Test behavior of initialize_alpaca when config is invalid
    or missing environment variables.
    """
    # Missing keys
    monkeypatch.delenv('ALPACA_API_KEY', raising=False)
    monkeypatch.delenv('ALPACA_SECRET_KEY', raising=False)
    assert initialize_alpaca() is None, "Expected None with missing Alpaca keys."

    # Invalid URL
    monkeypatch.setenv('ALPACA_API_KEY', 'dummy_key')
    monkeypatch.setenv('ALPACA_SECRET_KEY', 'dummy_secret')
    monkeypatch.setenv('ALPACA_BASE_URL', 'invalid_url')
    with pytest.raises(ValueError, match="Invalid URL"):
        initialize_alpaca()

    # Valid configuration
    monkeypatch.setenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    with patch('alpaca_trade_api.REST') as mock_tradeapi:
        mock_client = MagicMock()
        mock_tradeapi.return_value = mock_client
        client = initialize_alpaca()
        assert client == mock_tradeapi.return_value, "Expected a valid Alpaca client with correct configuration."

def test_get_project_root_failure(monkeypatch):
    """
    Test get_project_root when .env does not exist at any expected location.
    """
    from pathlib import Path

    # Simulate no .env file found
    monkeypatch.setattr('src.Utilities.data_fetch_utils.Path.exists', lambda _: False)

    with pytest.raises(RuntimeError, match="Project root not found. Ensure .env file exists at the project root."):
        get_project_root()

@pytest.mark.asyncio
async def test_logging_setup(data_fetch_utils_fixture):
    """
    Test that logging is properly set up and logs messages as expected.
    """
    with patch.object(data_fetch_utils_fixture.logger, 'info') as mock_info:
        data_fetch_utils_fixture.logger.info("Test log message.")
        mock_info.assert_called_with("Test log message.")

@pytest.mark.parametrize("missing_key,expect_error,expected_warning", [
    ("FINNHUB_API_KEY", False,
     "FINNHUB_API_KEY is not set in environment variables. Finnhub features may fail."),
    ("NEWSAPI_API_KEY", True, "Missing NewsAPI key."),
])
def test_data_fetch_utils_init_missing_keys(monkeypatch, missing_key, expect_error, expected_warning):
    """
    Test MainDataFetcher initialization with or without missing environment keys.
    """
    env_vars = {"NEWSAPI_API_KEY": "test_newsapi_key", "FINNHUB_API_KEY": "test_finnhub_key"}
    if missing_key:
        env_vars.pop(missing_key, None)

    with patch.dict(os.environ, env_vars, clear=True):
        mock_logger = MagicMock()
        if expect_error:
            with pytest.raises(EnvironmentError, match="Missing NewsAPI key."):
                MainDataFetcher(logger=mock_logger)
            mock_logger.error.assert_called_with("NEWSAPI_API_KEY is not set in environment variables.")
        else:
            MainDataFetcher(logger=mock_logger)
            mock_logger.warning.assert_any_call(expected_warning)
