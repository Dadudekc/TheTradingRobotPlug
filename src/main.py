import asyncio
import logging
from Utilities.data_fetch_utils import DataFetchUtils
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StockDataFetcher")


class StockDataAgent:
    """
    Agent interface for fetching stock data from multiple sources.
    """

    def __init__(self):
        self.logger = logging.getLogger("StockDataAgent")
        self.fetcher = DataFetchUtils()

    async def get_real_time_quote(self, symbol: str) -> dict:
        self.logger.info(f"Fetching real-time quote for {symbol}...")
        async with aiohttp.ClientSession() as session:
            try:
                return await self.fetcher.fetch_finnhub_quote(symbol, session)
            except Exception as e:
                self.logger.error(f"Failed to fetch real-time quote for {symbol}: {e}")
                return {}

    async def get_historical_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1Day") -> list:
        self.logger.info(f"Fetching historical data for {symbol} from Alpaca...")
        try:
            return await self.fetcher.fetch_alpaca_data_async(symbol, start_date, end_date, interval=interval)
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return []

    async def get_historical_data_alpha_vantage(self, symbol: str, start_date: str, end_date: str) -> list:
        self.logger.info(f"Fetching historical data for {symbol} from Alpha Vantage...")
        async with aiohttp.ClientSession() as session:
            try:
                return await self.fetcher.fetch_alphavantage_data(symbol, session, start_date, end_date)
            except Exception as e:
                self.logger.error(f"Failed to fetch historical data for {symbol} from Alpha Vantage: {e}")
                return []

    async def get_news(self, symbol: str, page_size: int = 3) -> list:
        self.logger.info(f"Fetching recent news articles for {symbol}...")
        try:
            return await self.fetcher.fetch_news_data_async(symbol, page_size=page_size)
        except Exception as e:
            self.logger.error(f"Failed to fetch news for {symbol}: {e}")
            return []

    async def get_combined_data(self, symbol: str, start_date: str, end_date: str, interval: str = "1d") -> dict:
        self.logger.info(f"Fetching combined data for {symbol} from multiple sources...")
        async with aiohttp.ClientSession() as session:
            try:
                return await self.fetcher.fetch_data_for_multiple_symbols(
                    symbols=[symbol],
                    data_sources=["Alpaca", "Alpha Vantage", "Finnhub", "NewsAPI"],
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                )
            except Exception as e:
                self.logger.error(f"Failed to fetch combined data for {symbol}: {e}")
                return {}


async def showcase_stock_data(symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31", interval="1Day"):
    """
    Demonstrates the full capabilities of the StockDataAgent.
    """
    agent = StockDataAgent()

    # Showcase data fetching capabilities
    try:
        real_time_quote = await agent.get_real_time_quote(symbol)
        logger.info(f"Showcase Real-Time Quote: {real_time_quote}")

        historical_data = await agent.get_historical_data(symbol, start_date, end_date, interval)
        logger.info(f"Showcase Historical Data from Alpaca: {historical_data}")

        historical_data_alpha = await agent.get_historical_data_alpha_vantage(symbol, start_date, end_date)
        logger.info(f"Showcase Historical Data from Alpha Vantage: {historical_data_alpha}")

        news = await agent.get_news(symbol, page_size=3)
        logger.info(f"Showcase Recent News: {news}")

        # Include `interval` explicitly
        combined_data = await agent.get_combined_data(symbol, start_date, end_date, interval)
        logger.info(f"Showcase Combined Data: {combined_data}")
    except Exception as e:
        logger.error(f"An error occurred during the showcase: {e}")


def run():
    """
    Entry point for the script, wrapping the asyncio event loop.
    """
    asyncio.run(showcase_stock_data())


if __name__ == "__main__":
    run()
