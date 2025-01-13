import os
import aiohttp
import asyncio
from typing import List, Dict

class ServerlessFetcher:
    def __init__(self, logger=None):
        """
        Initialize the ServerlessFetcher class.

        Args:
            logger (Optional): Logger instance for logging messages.
        """
        self.logger = logger
        self.serverless_endpoint = os.getenv("SERVERLESS_API_URL")

    async def fetch(self, symbol: str, data_sources: List[str] = None, retries: int = 3, **kwargs) -> dict:
        """
        Fetch data from a serverless function for one or more data sources.

        Args:
            symbol (str): Stock symbol.
            data_sources (List[str]): List of data sources to fetch data from.
            retries (int): Number of retry attempts for the request.
            kwargs: Additional parameters to include in the request.

        Returns:
            dict: A dictionary with responses for each requested data source.
        """
        if not self.serverless_endpoint:
            if self.logger:
                self.logger.error("Serverless endpoint URL not found in environment variables.")
            return {}

        data_sources = data_sources or [
            "Alpaca",
            "Alpha Vantage",
            "Polygon",
            "Yahoo Finance",
            "Finnhub",
            "NewsAPI",
        ]

        # Supported sources
        supported_sources = {
            "Alpaca",
            "Alpha Vantage",
            "Polygon",
            "Yahoo Finance",
            "Finnhub",
            "NewsAPI",
        }

        # Validate requested sources
        invalid_sources = [source for source in data_sources if source not in supported_sources]
        if invalid_sources:
            if self.logger:
                self.logger.error(f"Invalid data sources requested: {invalid_sources}. Supported sources: {supported_sources}")
            return {}

        # Define allowed kwargs for the payload
        allowed_kwargs = ["start_date", "end_date", "interval", "outputsize", "apikey"]
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in allowed_kwargs}

        # Prepare the payload
        payload = {
            "symbol": symbol,
            "data_sources": data_sources,
            "kwargs": filtered_kwargs,
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for attempt in range(retries):
                try:
                    if self.logger:
                        self.logger.debug(f"Sending request to {self.serverless_endpoint} with payload: {payload}")
                    async with session.post(self.serverless_endpoint, json=payload) as response:
                        if self.logger:
                            self.logger.debug(f"Received response with status {response.status}")
                        if response.status == 200:
                            data = await response.json()
                            if self.logger:
                                self.logger.debug(f"Response JSON: {data}")
                            return data
                        else:
                            if self.logger:
                                self.logger.error(f"Failed with status {response.status}")
                            break
                except Exception as e:
                    if attempt < retries - 1:
                        if self.logger:
                            self.logger.warning(f"Attempt {attempt + 1} failed. Retrying...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        if self.logger:
                            self.logger.error(f"All retries failed. Error: {e}")
                        return {}

        return {}

# Example usage
async def test_fetch_interactively():
    """
    Test the fetch method interactively.
    """
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("ServerlessFetcher")

    fetcher = ServerlessFetcher(logger=logger)
    result = await fetcher.fetch(
        symbol="AAPL",
        data_sources=["Alpha Vantage", "Yahoo Finance"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        interval="1d",
        outputsize="full",
    )
    print("Result:", result)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    asyncio.run(test_fetch_interactively())
