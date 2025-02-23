#!/usr/bin/env python3
import os
import json
import logging
import asyncio
import sqlite3
import pandas as pd
import re
import time
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException

# Sentiment Analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Trading & Data Imports
import yfinance as yf

# Additional Utilities (ensure these modules are in your PYTHONPATH)
from Utilities.data_processing.Sentiment_Analysis.sentiment_integration import SentimentIntegration
from Utilities.db.sql_data_handler import SQLDataHandler  # For database operations
from Utilities.config_manager import ConfigManager

# ------------------------------------------------------------------------------
# GLOBAL CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Strategy Constants
BASE_BUY_DROP = 0.998        # 0.2% drop triggers buy
BASE_SELL_RISE = 1.002       # 0.2% rise triggers sell
SENTIMENT_MULTIPLIER = 0.015 # Sentiment impact scaling factor
STOP_LOSS_THRESHOLD = 0.97   # 3% drop from buy price triggers auto-sell
TAKE_PROFIT_THRESHOLD = 1.05 # 5% rise from buy price triggers auto-sell

# Instantiate configuration and DB handler utilities
config = ConfigManager()
DB_HANDLER = SQLDataHandler()

# ------------------------------------------------------------------------------
# CLASS: StocktwitsSentimentAnalyzer
# ------------------------------------------------------------------------------
class StocktwitsSentimentAnalyzer:
    def __init__(self, db_name: str = "trade_analyzer.db", cookie_file: str = "stocktwits_cookies.json",
                 max_scrolls: int = 50, scroll_pause_time: int = 2):
        """
        Scrapes Stocktwits for a given ticker, extracts messages, computes sentiment
        (using both TextBlob and VADER) and logs results into a SQLite database.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_name = db_name
        self.cookie_file = cookie_file
        self.max_scrolls = max_scrolls
        self.scroll_pause_time = scroll_pause_time

        # Optionally refresh DB schema
        if os.path.exists(self.db_name):
            self.logger.info(f"Removing existing DB '{self.db_name}' to refresh schema.")
            os.remove(self.db_name)
        self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS SentimentData")
        cursor.execute('''
            CREATE TABLE SentimentData (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                timestamp TEXT,
                content TEXT,
                textblob_sentiment REAL,
                vader_sentiment REAL,
                UNIQUE(ticker, timestamp, content)
            )
        ''')
        conn.commit()
        conn.close()
        self.logger.info("Database setup complete.")

    def load_cookies(self, driver):
        """
        Loads cookies from a JSON file and applies them to the browser session.
        """
        if not os.path.exists(self.cookie_file):
            self.logger.warning(f"Cookie file '{self.cookie_file}' not found.")
            return False
        try:
            with open(self.cookie_file, 'r') as f:
                cookies = json.load(f)
            for cookie in cookies:
                if "sameSite" in cookie:
                    del cookie["sameSite"]
                driver.add_cookie(cookie)
            self.logger.info("Cookies loaded and applied.")
            return True
        except Exception as e:
            self.logger.error(f"Error loading cookies: {e}")
            return False

    def save_cookies(self, driver):
        """
        Saves current session cookies to a JSON file.
        """
        try:
            cookies = driver.get_cookies()
            with open(self.cookie_file, 'w') as f:
                json.dump(cookies, f, indent=4)
            self.logger.info(f"Cookies saved to '{self.cookie_file}'.")
        except Exception as e:
            self.logger.error(f"Error saving cookies: {e}")

    def is_logged_in(self, driver, symbol: str):
        """
        Checks if the browser is currently at the Stocktwits symbol page.
        """
        expected_url = f"https://stocktwits.com/symbol/{symbol}"
        return driver.current_url.lower() == expected_url.lower()

    def fetch_html_content(self, url: str, driver):
        """
        Navigates to the URL, scrolls to load more messages, and returns the HTML content.
        """
        self.logger.info(f"Navigating to URL: {url}")
        driver.get(url)
        time.sleep(5)

        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(self.max_scrolls):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(self.scroll_pause_time)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        self.logger.info("Completed scrolling. Extracting page source.")
        return driver.page_source

    def extract_messages(self, html_content: str) -> pd.DataFrame:
        soup = BeautifulSoup(html_content, 'html.parser')
        messages = []
        for msg_div in soup.find_all('div', class_='RichTextMessage_body__4qUeP'):
            try:
                timestamp_elem = msg_div.find_previous('time')
                content = msg_div.get_text(strip=True)
                if timestamp_elem and content:
                    timestamp = timestamp_elem.get("datetime")
                    messages.append({"timestamp": timestamp, "content": content})
            except Exception as e:
                self.logger.warning(f"Failed to parse a message: {e}")

        self.logger.info(f"Found {len(messages)} Stocktwits messages.")
        return pd.DataFrame(messages)

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def clean_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty:
            df["clean_content"] = df["content"].apply(self.clean_text)
        return df

    @staticmethod
    def analyze_sentiments(text: str):
        textblob_score = TextBlob(text).sentiment.polarity
        vader_score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
        return textblob_score, vader_score

    async def async_analyze_sentiments(self, text: str, executor=None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.analyze_sentiments, text)

    def insert_sentiment_data(self, ticker: str, df: pd.DataFrame):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        for _, row in df.iterrows():
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO SentimentData 
                    (ticker, timestamp, content, textblob_sentiment, vader_sentiment)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    ticker,
                    row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(row['timestamp']) else None,
                    row['content'],
                    row['textblob_sentiment'],
                    row['vader_sentiment']
                ))
            except sqlite3.IntegrityError as e:
                self.logger.warning(f"Duplicate entry skipped: {e}")
        conn.commit()
        conn.close()
        self.logger.info("Sentiment data inserted into DB.")

    async def get_stocktwits_sentiment_async(self, symbol: str) -> float:
        """
        Asynchronously retrieves sentiment from Stocktwits by launching a browser,
        handling login (with cookies), scrolling for messages, and computing sentiment.
        """
        chrome_options = Options()
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--remote-debugging-port=9222")
        chrome_options.add_argument("log-level=3")

        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        wait = WebDriverWait(driver, 60)

        sentiment_score = 0.0
        try:
            login_url = f"https://stocktwits.com/symbol/{symbol}"
            driver.get(login_url)
            time.sleep(3)

            if self.load_cookies(driver):
                driver.refresh()
                time.sleep(3)

            if not self.is_logged_in(driver, symbol):
                self.logger.info("Cookies invalid or not logged in. Proceeding with manual login flow.")
                driver.get(f"https://stocktwits.com/signin?next=/symbol/{symbol}")
                print("🚀 Please log in to Stocktwits manually... waiting for success.")
                wait.until(lambda d: self.is_logged_in(d, symbol))
                print(f"✅ Login successful. Current URL: {driver.current_url}")
                self.save_cookies(driver)
            else:
                self.logger.info("Logged in via cookies. No manual login needed.")

            html_content = self.fetch_html_content(login_url, driver)
            messages_df = self.extract_messages(html_content)
            messages_df = self.clean_messages(messages_df)

            if messages_df.empty:
                print("⚠️ No sentiment data found.")
                return 0.0

            executor = ThreadPoolExecutor(max_workers=8)
            tasks = [self.async_analyze_sentiments(row["content"], executor=executor)
                     for _, row in messages_df.iterrows()]
            results = await asyncio.gather(*tasks)

            messages_df["timestamp"] = pd.to_datetime(messages_df["timestamp"], errors='coerce')
            messages_df["textblob_sentiment"] = [r[0] for r in results]
            messages_df["vader_sentiment"] = [r[1] for r in results]

            avg_textblob = messages_df["textblob_sentiment"].mean()
            avg_vader = messages_df["vader_sentiment"].mean()
            sentiment_score = (avg_textblob + avg_vader) / 2

            print(f"✅ Stocktwits Sentiment Score for {symbol}: {sentiment_score:.2f}")

            self.insert_sentiment_data(symbol, messages_df)
            csv_name = f"{symbol}_sentiment.csv"
            messages_df.to_csv(csv_name, index=False)
            self.logger.info(f"Saved messages and sentiment data to '{csv_name}'.")
        except TimeoutException:
            self.logger.error("Login timed out. Please try again.")
        except Exception as e:
            self.logger.error(f"Error in get_stocktwits_sentiment_async: {e}")
        finally:
            driver.quit()

        return sentiment_score

# ------------------------------------------------------------------------------
# CLASS: SimulatedTradeEngine (Improved with DB integration and buy-price tracking)
# ------------------------------------------------------------------------------
class SimulatedTradeEngine:
    def __init__(self, starting_balance=10000):
        self.balance = starting_balance
        self.trade_history = []
        self.positions = {}
        self.buy_prices = {}  # Track the price at which positions were bought
        logging.info("✅ Simulated Trading Engine Initialized.")

    def execute_trade(self, symbol, action, quantity, price):
        cost = quantity * price
        trade_details = None

        if action == "BUY":
            if self.balance >= cost:
                self.balance -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                self.buy_prices[symbol] = price  # Record buy price for future reference
                logging.info(f"📈 BUY {quantity} {symbol} @ {price:.2f}, Remaining Balance: {self.balance:.2f}")
            else:
                logging.warning("❌ Insufficient funds for BUY trade.")
                return None

        elif action == "SELL":
            if self.positions.get(symbol, 0) >= quantity:
                self.balance += cost
                self.positions[symbol] -= quantity
                logging.info(f"📉 SELL {quantity} {symbol} @ {price:.2f}, New Balance: {self.balance:.2f}")
                if self.positions[symbol] == 0:
                    del self.buy_prices[symbol]
            else:
                logging.warning(f"❌ Attempted to SELL {quantity} {symbol}, but only {self.positions.get(symbol, 0)} available.")
                return None

        trade_details = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.now()
        }
        self.trade_history.append(trade_details)
        DB_HANDLER.save_trade(trade_details)  # Persist trade in the database
        return trade_details

    def get_trade_history(self):
        return pd.DataFrame(self.trade_history)

# ------------------------------------------------------------------------------
# CLASS: AITradeJournal
# ------------------------------------------------------------------------------
class AITradeJournal:
    def __init__(self):
        self.entries = []
        logging.info("📝 AI Trade Journal Initialized.")

    def log_trade(self, trade_details):
        if trade_details:
            self.entries.append(trade_details)
            logging.info(f"📝 Trade Logged: {trade_details}")

    def analyze_performance(self):
        df = pd.DataFrame(self.entries)
        if df.empty:
            logging.info("📊 No trades to analyze yet.")
            return None

        # For BUY trades, subtract cost; for SELL, add proceeds
        df["profit"] = df.apply(lambda row: row["quantity"] * row["price"] if row["action"] == "SELL"
                                  else -row["quantity"] * row["price"], axis=1)
        realized_profit = df[df["action"] == "SELL"]["profit"].sum()
        logging.info(f"💰 Realized Profit/Loss (Closed Trades): {realized_profit:.2f}")
        return df

# ------------------------------------------------------------------------------
# CLASS: SentimentDrivenTradingBot (Orchestrator)
# ------------------------------------------------------------------------------
class SentimentDrivenTradingBot:
    def __init__(self):
        self.analyzer = StocktwitsSentimentAnalyzer()
        self.trade_engine = SimulatedTradeEngine()
        self.journal = AITradeJournal()
        self.sentiment_integration = SentimentIntegration()

    def run(self):
        # Prompt for stock symbol (default AAPL if none provided)
        symbol = input("Enter the stock ticker symbol (e.g. AAPL): ").strip().upper()
        if not symbol:
            symbol = "AAPL"

        # 1. Get Stocktwits sentiment asynchronously
        sentiment_score = asyncio.run(self.analyzer.get_stocktwits_sentiment_async(symbol))
        logger.info(f"Overall Stocktwits Sentiment for {symbol}: {sentiment_score:.2f}\n")

        # 2. Download historical data using yfinance
        data = yf.download(symbol, period="30d", interval="1d", auto_adjust=True)
        if data.empty or "Close" not in data.columns:
            logger.warning(f"❌ No valid data retrieved for {symbol}. Exiting.")
            return
        logger.info(f"📊 Retrieved {len(data)} days of data for {symbol}.\n")
        logger.info(f"📊 Latest Stock Data:\n{data[['Close']].tail(5)}\n")

        # 3. Calculate rolling volatility (using 5-day window)
        rolling_volatility = data["Close"].pct_change().rolling(5).std().fillna(0.005)

        # 4. Force an initial BUY on Day 0
        initial_price = data["Close"].iloc[0].item()  # Updated for FutureWarning
        forced_trade = self.trade_engine.execute_trade(symbol, "BUY", 10, initial_price)
        self.journal.log_trade(forced_trade)

        # 5. Simulate trading over the historical period
        for i in range(1, len(data)):
            price = data["Close"].iloc[i].item()      # Updated for FutureWarning
            prev_price = data["Close"].iloc[i - 1].item()  # Updated for FutureWarning

            logger.info(f"🔎 Day {i}: Prev Close={prev_price:.2f}, Current Price={price:.2f}")

            # Retrieve additional sentiment from the integration module and combine
            # Await the sentiment integration since it is a coroutine
            ext_sentiment_score = asyncio.run(self.sentiment_integration.get_sentiment_score(symbol))
            combined_sentiment = (sentiment_score + ext_sentiment_score) / 2

            volatility = rolling_volatility.iloc[i]
            adaptive_multiplier = SENTIMENT_MULTIPLIER * (1 + volatility * 10)

            buy_threshold = BASE_BUY_DROP - (combined_sentiment * adaptive_multiplier)
            sell_threshold = BASE_SELL_RISE + (combined_sentiment * adaptive_multiplier)

            logger.info(f"🤖 Adjusted thresholds - BUY: {buy_threshold:.4f}, SELL: {sell_threshold:.4f}")

            # Check stop-loss / take-profit conditions if a buy price exists
            if symbol in self.trade_engine.buy_prices:
                buy_price = self.trade_engine.buy_prices[symbol]
                if price <= buy_price * STOP_LOSS_THRESHOLD:
                    logging.warning(f"🚨 STOP-LOSS Triggered! Selling {symbol} at {price:.2f}")
                    trade = self.trade_engine.execute_trade(symbol, "SELL", 10, price)
                    self.journal.log_trade(trade)
                    continue
                if price >= buy_price * TAKE_PROFIT_THRESHOLD:
                    logging.info(f"💰 TAKE-PROFIT Triggered! Selling {symbol} at {price:.2f}")
                    trade = self.trade_engine.execute_trade(symbol, "SELL", 10, price)
                    self.journal.log_trade(trade)
                    continue

            # BUY logic: if price drops below previous day adjusted by threshold
            if price < prev_price * buy_threshold and self.trade_engine.balance >= price * 10:
                logging.info("📉 BUY Condition Met! Executing Trade.")
                trade = self.trade_engine.execute_trade(symbol, "BUY", 10, price)
                self.journal.log_trade(trade)

            # SELL logic: if price rises above previous day adjusted by threshold
            elif price > prev_price * sell_threshold and self.trade_engine.positions.get(symbol, 0) > 0:
                logging.info("📈 SELL Condition Met! Executing Trade.")
                trade = self.trade_engine.execute_trade(symbol, "SELL", 10, price)
                self.journal.log_trade(trade)

        # 6. Analyze overall performance
        logger.info("\n🔎 Analyzing trading performance:")
        df_performance = self.journal.analyze_performance()
        if df_performance is not None:
            logger.info(f"\n📊 Trade Details:\n{df_performance}\n")

# ------------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    bot = SentimentDrivenTradingBot()
    bot.run()