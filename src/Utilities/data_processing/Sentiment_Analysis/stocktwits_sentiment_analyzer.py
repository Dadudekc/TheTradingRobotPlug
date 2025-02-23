import os
import time
import json
import logging
import sqlite3
import pandas as pd
import re
import subprocess
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException

# Sentiment Analysis Imports
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# --------------------------------------------------------------------------
# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StocktwitsSentimentAnalyzer:
    """
    A class-based system to:
      - Manage database setup (for storing sentiment data)
      - Handle Stocktwits login via Selenium + cookies
      - Scrape and scroll through messages
      - Clean and run sentiment analysis (TextBlob & VADER)
      - Insert results into SQLite
    """

    def __init__(
        self,
        db_name="trade_analyzer.db",
        cookie_file="stocktwits_cookies.json",
        max_scrolls=10,
        scroll_pause_time=2,
    ):
        """
        :param db_name: Name (and path) of the SQLite database file.
        :param cookie_file: JSON file where Stocktwits login cookies are saved/loaded.
        :param max_scrolls: How many times to scroll down the page to load more messages.
        :param scroll_pause_time: How many seconds to wait after each scroll.
        """
        self.db_name = db_name
        self.cookie_file = cookie_file
        self.max_scrolls = max_scrolls
        self.scroll_pause_time = scroll_pause_time

        # Setup DB at init (optional). If you always want a fresh table, drop it here.
        self._setup_database()

    # ----------------------------------------------------------------------
    # Database Setup
    def _setup_database(self):
        """
        Creates a SentimentData table with textblob & vader sentiment columns.
        Drops the old table if schema mismatch is suspected.
        (Currently always drops to ensure a fresh table each run.)
        """
        if os.path.exists(self.db_name):
            logger.info(f"Existing database '{self.db_name}' found. Removing it to update schema.")
            os.remove(self.db_name)

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS SentimentData (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                timestamp TEXT,
                content TEXT,
                textblob_sentiment REAL,
                vader_sentiment REAL,
                UNIQUE(ticker, timestamp, content)
            )
            """
        )
        conn.commit()
        conn.close()
        logger.info(f"Database '{self.db_name}' setup complete.")

    # ----------------------------------------------------------------------
    # Cookie Handling
    def load_cookies(self, driver):
        """
        Loads cookies from a JSON file and applies them to the browser session.
        Returns True if cookies loaded successfully, False otherwise.
        """
        if not os.path.exists(self.cookie_file):
            logger.warning(f"Cookie file '{self.cookie_file}' not found.")
            return False
        try:
            with open(self.cookie_file, "r") as f:
                cookies = json.load(f)
            for cookie in cookies:
                # 'sameSite' is not always recognized by Selenium
                cookie.pop("sameSite", None)
                driver.add_cookie(cookie)
            logger.info("Cookies loaded from file and added to the driver.")
            return True
        except Exception as e:
            logger.error(f"Failed to load cookies from {self.cookie_file}. Error: {e}")
            return False

    def save_cookies(self, driver):
        """
        Saves cookies from the current Selenium session to a JSON file.
        """
        try:
            cookies = driver.get_cookies()
            with open(self.cookie_file, "w") as f:
                json.dump(cookies, f, indent=4)
            logger.info(f"Cookies saved to {self.cookie_file}.")
        except Exception as e:
            logger.error(f"Failed to save cookies. Error: {e}")

    # ----------------------------------------------------------------------
    # Check if Logged In
    def is_logged_in(self, driver, symbol):
        """
        Checks if we're on https://stocktwits.com/symbol/{symbol}.
        """
        current_url = driver.current_url
        expected_url = f"https://stocktwits.com/symbol/{symbol}"
        return current_url.lower() == expected_url.lower()

    # ----------------------------------------------------------------------
    # Scrolling & HTML Fetch
    def fetch_html_content(self, url, driver):
        """
        Navigates to the URL, scrolls down to load more messages, and returns the final HTML content.
        """
        logger.info(f"Navigating to URL: {url}")
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

        logger.info("Completed scrolling. Collecting HTML content.")
        return driver.page_source

    # ----------------------------------------------------------------------
    # Extract Messages
    def extract_messages(self, html_content):
        """
        Extracts message text and timestamps from Stocktwits.
        Returns a DataFrame with columns: timestamp, content.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        messages = []

        for message_div in soup.find_all("div", class_="RichTextMessage_body__4qUeP"):
            try:
                timestamp_elem = message_div.find_previous("time")
                content = message_div.get_text(strip=True)
                if timestamp_elem and content:
                    timestamp = timestamp_elem.get("datetime")
                    messages.append({"timestamp": timestamp, "content": content})
            except Exception as e:
                logger.warning(f"Failed to extract a message: {e}")

        logger.info(f"Extracted {len(messages)} messages from HTML.")
        return pd.DataFrame(messages)

    # ----------------------------------------------------------------------
    # Clean Text
    def clean_text(self, text):
        """
        Removes URLs, special characters, and unnecessary whitespace.
        """
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean_messages(self, df):
        """
        Cleans the content column in the DataFrame.
        """
        if not df.empty:
            df["clean_content"] = df["content"].apply(self.clean_text)
        return df

    # ----------------------------------------------------------------------
    # Sentiment Analysis (TextBlob + VADER)
    def analyze_sentiments(self, text):
        """
        Uses TextBlob & VADER to analyze a given text.
        Returns a tuple (textblob_sentiment, vader_sentiment).
        """
        textblob_score = TextBlob(text).sentiment.polarity
        vader_analyzer = SentimentIntensityAnalyzer()
        vader_score = vader_analyzer.polarity_scores(text)["compound"]
        return textblob_score, vader_score

    async def async_analyze_sentiments(self, text, executor=None):
        """
        Wraps analyze_sentiments() in an async call to leverage concurrency.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.analyze_sentiments, text)

    # ----------------------------------------------------------------------
    # Insert Data into Database
    def insert_sentiment_data(self, ticker, df):
        """
        Inserts the analyzed sentiments into the database.
        """
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        for _, row in df.iterrows():
            try:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO SentimentData 
                    (ticker, timestamp, content, textblob_sentiment, vader_sentiment)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                        if not pd.isna(row["timestamp"])
                        else None,
                        row["content"],
                        row["textblob_sentiment"],
                        row["vader_sentiment"],
                    ),
                )
            except sqlite3.IntegrityError as e:
                logger.warning(f"Duplicate entry skipped: {e}")

        conn.commit()
        conn.close()
        logger.info("Sentiment data inserted into database.")

    # ----------------------------------------------------------------------
    # Core Async Method: Get Stocktwits Sentiment
    async def get_stocktwits_sentiment_async(self, symbol):
        """
        Main logic: 
         - Launches Selenium
         - Loads cookies (if available)
         - Navigates & logs in if needed
         - Fetches messages & runs sentiment analysis
         - Inserts results into DB
         - Returns an average sentiment score
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

        login_url = f"https://stocktwits.com/symbol/{symbol}"

        try:
            # Attempt to load cookies
            driver.get(login_url)
            time.sleep(3)
            if self.load_cookies(driver):
                driver.refresh()
                time.sleep(3)

            if not self.is_logged_in(driver, symbol):
                logger.info("Cookies invalid or not logged in. Proceeding with manual login flow.")
                driver.get(f"https://stocktwits.com/signin?next=/symbol/{symbol}")
                print("🚀 Please log in to Stocktwits manually... Waiting for login to complete.")

                # Wait until the URL changes to indicate successful login
                wait.until(lambda d: self.is_logged_in(d, symbol))
                print(f"✅ Login detected! URL changed to: {driver.current_url}")
                self.save_cookies(driver)
            else:
                logger.info("Successfully logged in via cookies. No manual login required.")

            # Fetch the content
            html_content = self.fetch_html_content(login_url, driver)
            messages_df = self.extract_messages(html_content)
            messages_df = self.clean_messages(messages_df)

            if messages_df.empty:
                print("⚠️ No sentiment data found. Check page structure.")
                return None

            # Run parallel sentiment analysis
            executor = ThreadPoolExecutor(max_workers=8)
            tasks = [
                self.async_analyze_sentiments(row["content"], executor=executor)
                for _, row in messages_df.iterrows()
            ]
            results = await asyncio.gather(*tasks)

            # Convert timestamp to datetime
            messages_df["timestamp"] = pd.to_datetime(messages_df["timestamp"], errors="coerce")
            messages_df["textblob_sentiment"] = [r[0] for r in results]
            messages_df["vader_sentiment"] = [r[1] for r in results]

            # Calculate an average "combined" sentiment
            avg_textblob = messages_df["textblob_sentiment"].mean()
            avg_vader = messages_df["vader_sentiment"].mean()
            sentiment_score = (avg_textblob + avg_vader) / 2

            print(f"✅ Stocktwits Sentiment Score for {symbol}: {sentiment_score:.2f}")

            # Insert into DB
            self.insert_sentiment_data(symbol, messages_df)

            # Save to CSV
            csv_name = f"{symbol}_sentiment.csv"
            messages_df.to_csv(csv_name, index=False)
            logger.info(f"Saved data to {csv_name}")

            return sentiment_score

        except TimeoutException:
            logger.error("Login timeout. Possibly need more time or corrected URL.")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            driver.quit()

    # ----------------------------------------------------------------------
    # Synchronous Entry Point
    def run(self):
        """
        For direct CLI usage:
         1) Prompt user for a symbol
         2) Run the async sentiment fetch
         3) Print or return the final result
        """
        symbol = input("Enter the stock ticker symbol (e.g., TSLA): ").strip().upper()
        final_score = asyncio.run(self.get_stocktwits_sentiment_async(symbol))
        if final_score is not None:
            logger.info(f"Final Sentiment Score for {symbol}: {final_score:.2f}")
        else:
            logger.warning("No sentiment score could be calculated.")
        return final_score


# --------------------------------------------------------------------------
# Execution from CLI
if __name__ == "__main__":
    analyzer = StocktwitsSentimentAnalyzer(
        db_name="trade_analyzer.db",
        cookie_file="stocktwits_cookies.json",
        max_scrolls=50,
        scroll_pause_time=2,
    )
    analyzer.run()
