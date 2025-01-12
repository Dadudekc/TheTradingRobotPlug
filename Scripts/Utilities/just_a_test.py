import asyncio
import logging
from data_fetch_utils import DataFetchUtils
import aiohttp
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Stock Data Fetcher")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fetch stock data from multiple sources.")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol to fetch data for.")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date for historical data.")
    parser.add_argument("--end-date", type=str, default="2023-12-31", help="End date for historical data.")
    parser.add_argument("--interval", type=str, default="1Day", help="Data interval (e.g., '1Day', '1Min').")
    return parser.parse_args()

async def main():
    args = parse_arguments()
    symbol = args.symbol
    start_date = args.start_date
    end_date = args.end_date
    interval = args.interval

    fetcher = DataFetchUtils()

    try:
        async with aiohttp.ClientSession() as session:
            logger.info(f"Fetching real-time quote for {symbol}...")
            quote = await fetcher.fetch_finnhub_quote(symbol, session)
            logger.info(f"Real-Time Quote: {quote}")

        logger.info(f"\nFetching historical data for {symbol} from Alpaca...")
        alpaca_data = await fetcher.fetch_alpaca_data_async(symbol, start_date, end_date, interval=interval)
        logger.info(f"Alpaca Historical Data: {alpaca_data}")

        async with aiohttp.ClientSession() as session:
            logger.info(f"\nFetching historical data for {symbol} from Alpha Vantage...")
            alpha_vantage_data = await fetcher.fetch_alphavantage_data(symbol, session, start_date, end_date)
            logger.info(f"Alpha Vantage Historical Data: {alpha_vantage_data}")

        logger.info(f"\nFetching recent news articles for {symbol}...")
        news_data = await fetcher.fetch_news_data_async(symbol, page_size=3)
        logger.info(f"Recent News Data: {news_data}")

        async with aiohttp.ClientSession() as session:
            logger.info(f"\nFetching data for {symbol} from multiple sources...")
            all_data = await fetcher.fetch_data_for_multiple_symbols(
                symbols=[symbol],
                data_sources=["Alpaca", "Alpha Vantage", "Finnhub", "NewsAPI"],
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            logger.info(f"All Data Combined: {all_data}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
