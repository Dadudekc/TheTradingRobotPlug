import logging
import asyncio
import random
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# ✅ Import stock data fetcher
from Utilities.data_fetchers.main_data_fetcher import DataOrchestrator

# ✅ Import sentiment analysis components
from Utilities.data_processing.Sentiment_Analysis.stocktwits_sentiment_analyzer import StocktwitsSentimentAnalyzer
from Utilities.data_processing.Sentiment_Analysis.sentiment_analysis import SentimentAnalyzer

# ✅ Database Setup
DATABASE_URL = "postgresql://postgres:your_password@localhost:5434/trading_robot_plug"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class SentimentIntegration:
    """
    Manages integration between sentiment data and stock prices.
    - Fetches sentiment from Stocktwits
    - (Future) Integrates Twitter, NewsAPI, other sources
    - Fetches stock prices from PostgreSQL (via DataOrchestrator)
    - Merges & stores final dataset in PostgreSQL
    """

    def __init__(self):
        logging.info("🔎 Sentiment Integration Initialized")
        self.stocktwits_analyzer = StocktwitsSentimentAnalyzer(
            db_name="trade_analyzer.db",
            cookie_file="stocktwits_cookies.json",
            max_scrolls=50,
            scroll_pause_time=2
        )
        self.sentiment_processor = SentimentAnalyzer()
        self.data_orchestrator = DataOrchestrator()  # ✅ Use existing stock price fetcher

    async def get_sentiment_score(self, symbol: str) -> float:
        """
        Fetches a combined sentiment score for a stock symbol.
        - Uses real Stocktwits data.
        - (Future) Can integrate Twitter, NewsAPI, etc.
        """
        stocktwits_score = await self.stocktwits_analyzer.get_stocktwits_sentiment_async(symbol)
        # Placeholder for future sources
        random_placeholder = round(random.uniform(-1.0, 1.0), 2)

        final_score = (stocktwits_score + random_placeholder) / 2
        logging.info(f"✅ Final Sentiment Score for {symbol}: {final_score:.2f}")
        return final_score

    async def merge_sentiment_with_stock_data(self, symbol: str):
        """
        Fetches stock data from PostgreSQL and merges it with sentiment scores.
        Stores the final dataset back into PostgreSQL.
        """
        logging.info(f"🔄 Merging sentiment & stock data for {symbol}...")

        # ✅ Fetch latest sentiment scores
        sentiment_df = await self.stocktwits_analyzer.get_sentiment_dataframe(symbol)

        if sentiment_df.empty:
            logging.warning(f"⚠ No sentiment data found for {symbol}. Skipping merge.")
            return

        # ✅ Fetch stock data using DataOrchestrator
        stock_data = await self.data_orchestrator.fetch_stock_data_async(
            ticker=symbol, start_date="2023-01-01", end_date=None, interval="1h"
        )

        if stock_data.empty:
            logging.warning(f"⚠ No stock price data found for {symbol}. Skipping merge.")
            return

        # ✅ Ensure timestamps are aligned before merging
        sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"])
        stock_data["timestamp"] = pd.to_datetime(stock_data.index)

        # ✅ Merge Sentiment Data with Stock Prices
        merged_df = pd.merge_asof(
            sentiment_df.sort_values("timestamp"),
            stock_data.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        # ✅ Store merged data into PostgreSQL
        session = SessionLocal()
        try:
            merged_df.to_sql(f"{symbol.lower()}_sentiment_analysis", con=engine, if_exists="replace", index=False)
            logging.info(f"✅ Merged sentiment & stock data stored in DB for {symbol}.")
        except Exception as e:
            logging.error(f"❌ Error storing merged data for {symbol}: {e}")
        finally:
            session.close()
