# File: main.py
# Location: src/
# Description: Entry point for fetching data, applying indicators, and storing in PostgreSQL.

import asyncio
import logging
import pandas as pd
import psycopg2
from Utilities.stock_data_agent import StockDataAgent
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StockDataMain")

# Database Connection Parameters
DB_NAME = os.getenv("POSTGRES_DBNAME", "trading_robot_plug")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "your_password")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5434")

async def showcase_stock_data(agent: StockDataAgent, symbol: str, start_date: str, end_date: str, interval: str = "1Day"):
    """
    Demonstrates the full capabilities of the StockDataAgent.
    """
    logger.info(f"Starting showcase for {symbol}.")

    # Showcase data fetching capabilities
    try:
        real_time_quote = await agent.get_real_time_quote(symbol)
        logger.info(f"Showcase Real-Time Quote for {symbol}: {real_time_quote}")

        historical_data = await agent.get_historical_data(symbol, start_date, end_date, interval)
        logger.info(f"Showcase Historical Data from Alpaca for {symbol}:\n{historical_data.head()}")

        historical_data_alpha = await agent.get_historical_data_alpha_vantage(symbol, start_date, end_date)
        logger.info(f"Showcase Historical Data from Alpha Vantage for {symbol}:\n{historical_data_alpha.head()}")

        news = await agent.get_news(symbol, page_size=3)
        logger.info(f"Showcase Recent News for {symbol}:\n{news.head()}")

        # Combined Data
        combined_data = await agent.get_combined_data(symbol, start_date, end_date, interval)
        logger.info(f"Showcase Combined Data for {symbol}:\n{combined_data}")

    except Exception as e:
        logger.error(f"An error occurred during the showcase for {symbol}: {e}")

async def main():
    """ Main function to fetch data and store in PostgreSQL. """
    agent = StockDataAgent()
    
    symbols = ["AAPL", "TSLA", "AMZN"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    interval = "1d"

    try:
        # Fetch and store raw data
        await agent.fetch_all_data(symbols, start_date=start_date, end_date=end_date, interval=interval)
        logger.info("\nData Fetching & Storage Completed Successfully!\n")

        # Showcase data capabilities
        for symbol in symbols:
            await showcase_stock_data(agent, symbol, start_date, end_date, interval)
    
    except Exception as e:
        logger.error(f"Fatal error during data fetching: {e}")

if __name__ == "__main__":
    asyncio.run(main())
