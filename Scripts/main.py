import asyncio
from Utilities.data_fetch_utils import DataFetchUtils
import os

async def test_fetch_alpaca_data():
    """
    Test fetching Alpaca data to validate Docker setup.
    """
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-01-05"
    interval = "1Day"

    try:
        fetcher = DataFetchUtils()
        print(f"Testing Alpaca data fetch for symbol: {symbol}")
        data = await fetcher.fetch_alpaca_data_async(symbol, start_date, end_date, interval)
        if not data.empty:
            print(f"Alpaca data fetched successfully for {symbol}:")
            print(data.head())
        else:
            print(f"No data fetched for {symbol}. Check Alpaca credentials and configurations.")
    except Exception as e:
        print(f"Error fetching Alpaca data: {e}")


async def test_fetch_news():
    """
    Test fetching news data to validate Docker setup.
    """
    ticker = "AAPL"
    page_size = 5

    try:
        fetcher = DataFetchUtils()
        print(f"Testing NewsAPI data fetch for ticker: {ticker}")
        data = await fetcher.fetch_news_data_async(ticker, page_size)
        if not data.empty:
            print(f"NewsAPI data fetched successfully for {ticker}:")
            print(data.head())
        else:
            print(f"No news data fetched for {ticker}. Check NewsAPI key.")
    except Exception as e:
        print(f"Error fetching news data: {e}")


def main():
    """
    Main function to run tests.
    """
    print("Starting Docker setup test for TradingRobotPlug...")
    print(f"Environment variables loaded successfully: {os.getenv('ALPACA_API_KEY') is not None}")

    # Run asynchronous tests
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(
        test_fetch_alpaca_data(),
        test_fetch_news()
    ))
    print("Tests completed.")


if __name__ == "__main__":
    main()
