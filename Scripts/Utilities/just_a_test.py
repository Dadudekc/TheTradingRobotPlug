import asyncio
from data_fetch_utils import DataFetchUtils
import aiohttp

async def main():
    # Initialize the utility class
    fetcher = DataFetchUtils()

    # Specify stock symbol and date range
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    # 1. Fetch real-time quote from Finnhub
    async with aiohttp.ClientSession() as session:
        print(f"Fetching real-time quote for {symbol}...")
        quote = await fetcher.fetch_finnhub_quote(symbol, session)
        print("\nReal-Time Quote:")
        print(quote)

    # 2. Fetch historical data from Alpaca
    print(f"\nFetching historical data for {symbol} from Alpaca...")
    alpaca_data = await fetcher.fetch_alpaca_data_async(symbol, start_date, end_date, interval="1Day")
    print("\nAlpaca Historical Data:")
    print(alpaca_data)

    # 3. Fetch historical data from Alpha Vantage
    async with aiohttp.ClientSession() as session:
        print(f"\nFetching historical data for {symbol} from Alpha Vantage...")
        alpha_vantage_data = await fetcher.fetch_alphavantage_data(symbol, session, start_date, end_date)
        print("\nAlpha Vantage Historical Data:")
        print(alpha_vantage_data)

    # 4. Fetch recent news articles from NewsAPI
    print(f"\nFetching recent news articles for {symbol}...")
    news_data = await fetcher.fetch_news_data_async(symbol, page_size=3)
    print("\nRecent News Data:")
    print(news_data)

    # 5. Fetch data from multiple sources
    print(f"\nFetching data for {symbol} from multiple sources...")
    async with aiohttp.ClientSession() as session:
        all_data = await fetcher.fetch_data_for_multiple_symbols(
            symbols=[symbol],
            data_sources=["Alpaca", "Alpha Vantage", "Finnhub", "NewsAPI"],
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        print("\nAll Data Combined:")
        print(all_data)

# Run the example
asyncio.run(main())
