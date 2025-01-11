# Scripts/Utilities/data_fetch_utils.py

import os
import pandas as pd
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from aiohttp import ClientSession, ClientConnectionError, ContentTypeError
from pathlib import Path
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import logging
import requests
from textblob import TextBlob
import yfinance as yf
from datetime import datetime
import numpy as np
from urllib.parse import urlparse
from datetime import timezone
try:
    from config_manager import setup_logging  # When running the script directly
except ImportError:
    from Scripts.Utilities.config_manager import setup_logging  # When the script is imported as a module








# -------------------------------------------------------------------
# Locate and load environment
# -------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Dynamically locate the project root based on the presence of a .env file.
    """
    script_dir = Path(__file__).resolve().parent
    for parent in script_dir.parents:
        if (parent / ".env").exists():
            return parent
    raise RuntimeError("Project root not found. Ensure .env file exists at the project root.")

PROJECT_ROOT = get_project_root()

ENV_PATH = PROJECT_ROOT / '.env'
if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found at {ENV_PATH}")
load_dotenv(dotenv_path=ENV_PATH)

# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------

LOGGER = setup_logging(
    script_name="data_fetch_utils",
    log_dir=PROJECT_ROOT / 'logs',
    max_log_size=5 * 1024 * 1024,
    backup_count=3
)

# -------------------------------------------------------------------
# Initialize Alpaca
# -------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)

def initialize_alpaca() -> Optional[tradeapi.REST]:
    """
    Initializes Alpaca API client using credentials from environment variables.
    Validates the URL format before initializing.
    """
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    if not api_key or not secret_key:
        LOGGER.error("Alpaca API credentials are not fully set in environment variables.")
        return None

    # Validate the base URL
    parsed_url = urlparse(base_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: {base_url}")

    LOGGER.info("Initializing Alpaca API with provided credentials.")
    return tradeapi.REST(api_key, secret_key, base_url, api_version='v2')

# Initialize the Alpaca client
try:
    ALPACA_CLIENT = initialize_alpaca()
except ValueError as e:
    LOGGER.error(f"Failed to initialize Alpaca client: {e}")
    ALPACA_CLIENT = None


# -------------------------------------------------------------------
# DataFetchUtils Class
# -------------------------------------------------------------------

class DataFetchUtils:
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the DataFetchUtils class.
        Reads environment variables for:
        - Finnhub (finnhub_api_key)
        - NewsAPI (news_api_key)
        - Alpaca (handled via initialize_alpaca)
        """
        self.logger = logger or logging.getLogger('DataFetchUtils')

        # Finnhub API key
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        if not self.finnhub_api_key:
            self.logger.warning("FINNHUB_API_KEY is not set in environment variables. Finnhub features may fail.")

        # NewsAPI key
        self.news_api_key = os.getenv('NEWSAPI_API_KEY')
        if not self.news_api_key:
            self.logger.error("NEWSAPI_API_KEY is not set in environment variables.")
            raise EnvironmentError("Missing NewsAPI key.")

        # Alpaca API client
        try:
            self.alpaca_api = initialize_alpaca()
            if self.alpaca_api:
                self.logger.info("Alpaca API client initialized successfully.")
            else:
                self.logger.warning("Alpaca API client could not be initialized.")
        except EnvironmentError as e:
            self.logger.error(f"Alpaca API initialization failed: {e}")
            self.alpaca_api = None

    # -------------------------------------------------------------------
    # Fetch with Retries
    # -------------------------------------------------------------------

    async def fetch_with_retries(
        self, url: str, headers: Dict[str, str], session: ClientSession, retries: int = 3
    ) -> Optional[Any]:
        """
        Fetches data from the provided URL with retries on failure.
        Implements exponential backoff.
        """
        for attempt in range(retries):
            try:
                self.logger.debug(f"Attempt {attempt + 1}/{retries} - Fetching URL: {url}")
                async with session.get(url, headers=headers, timeout=30) as response:
                    self.logger.debug(f"Response status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"Successful response data: {data}")
                        return data
                    elif response.status == 429:
                        self.logger.warning(
                            f"Rate limit reached (429). Retrying after delay. Attempt {attempt + 1}/{retries}"
                        )
                        await asyncio.sleep(60)  # Wait before retrying
                    else:
                        error_message = await response.text()
                        self.logger.error(
                            f"HTTP Error from {url} - Status: {response.status}, Message: {error_message}"
                        )
            except ClientConnectionError as e:
                self.logger.error(f"ClientConnectionError on URL {url}: {e}")
            except ContentTypeError as e:
                self.logger.error(f"ContentTypeError on URL {url}: {e}")
            except asyncio.TimeoutError:
                self.logger.error(
                    f"Timeout error while fetching {url}. Retrying attempt {attempt + 1}/{retries}..."
                )
            except Exception as e:
                self.logger.error(f"Unexpected error during fetch from {url}: {e}")

            await asyncio.sleep(2 ** attempt)  # Exponential backoff delay

        self.logger.error(f"Failed to fetch data after {retries} attempts: {url}")
        return None

    # -------------------------------------------------------------------
    # Fetch Finnhub Quote
    # -------------------------------------------------------------------

    async def fetch_finnhub_quote(self, symbol: str, session: ClientSession) -> pd.DataFrame:
        """
        Fetches real-time quote data from Finnhub.
        """
        if not self.finnhub_api_key:
            self.logger.error("No Finnhub API key set; cannot fetch quote.")
            return pd.DataFrame()

        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.finnhub_api_key}"
        try:
            data = await self.fetch_with_retries(url, headers={}, session=session)
            if data:
                df = pd.DataFrame([{
                    'date': pd.to_datetime(data.get('t', 0), unit='s'),
                    'current_price': data.get('c'),
                    'change': data.get('d'),
                    'percent_change': data.get('dp'),
                    'high': data.get('h'),
                    'low': data.get('l'),
                    'open': data.get('o'),
                    'previous_close': data.get('pc')
                }]).set_index('date')
                return df
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to fetch Finnhub quote for {symbol}: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------
    # Stock Data via yfinance
    # -------------------------------------------------------------------

    def get_stock_data(self, 
                    ticker: str, 
                    start_date: str = "2022-01-01", 
                    end_date: Optional[str] = None, 
                    interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical stock data using yfinance, expecting a lowercase 'date' after reset_index().
        """
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            # Guarantee 'date' is a column
            if isinstance(data.index, pd.DatetimeIndex):
                data.reset_index(inplace=True)

            # Rename all relevant columns to lowercase and replace spaces with underscores
            rename_dict = {col: col.lower().replace(' ', '_') for col in data.columns}
            data.rename(columns=rename_dict, inplace=True)

            # Convert 'date' to date objects
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date']).dt.date
            else:
                raise ValueError("Stock data does not contain 'date' column after reset_index.")

            data['symbol'] = ticker

            # Set 'date' as index to match test expectations
            data.set_index('date', inplace=True)

            self.logger.info(f"Fetched {len(data)} records for {ticker}")
            return data
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()


    async def fetch_stock_data_async(self, 
                                    ticker: str, 
                                    start_date: str = "2022-01-01", 
                                    end_date: Optional[str] = None, 
                                    interval: str = "1d") -> pd.DataFrame:
        """
        Async wrapper for get_stock_data.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_stock_data, ticker, start_date, end_date, interval)

    # -------------------------------------------------------------------
    # Fetch Alpaca Data
    # -------------------------------------------------------------------

    async def fetch_alpaca_data_async(self, symbol: str, 
                                    start_date: str, 
                                    end_date: str, 
                                    interval: str = "1Day") -> pd.DataFrame:
        """
        Fetches historical data from Alpaca, ensuring 'date' is the index.
        """
        if not self.alpaca_api:
            self.logger.error("Alpaca API client is not initialized.")
            return pd.DataFrame()

        try:
            bars = self.alpaca_api.get_bars(
                symbol, timeframe=interval,
                start=pd.to_datetime(start_date).strftime('%Y-%m-%d'),
                end=pd.to_datetime(end_date).strftime('%Y-%m-%d')
            ).df

            if bars.empty:
                self.logger.warning(f"No data returned from Alpaca for {symbol}.")
                return pd.DataFrame()

            # Reset index, rename columns, set 'date' as index
            bars.reset_index(drop=True, inplace=True)
            bars.rename(columns={'timestamp': 'date', 'trade_count': 'volume'}, inplace=True)
            bars['date'] = pd.to_datetime(bars['date']).dt.date  # Ensure date is properly formatted
            bars.set_index('date', inplace=True)  # Set 'date' as index
            bars['symbol'] = symbol  # Add the symbol column

            return bars

        except Exception as e:
            self.logger.error(f"Error fetching Alpaca data for {symbol}: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------
    # Fetch News Data via NewsAPI
    # -------------------------------------------------------------------

    def get_news_data(self, ticker: str, page_size: int = 5) -> pd.DataFrame:
        """
        Fetches news articles using the NEWSAPI_API_KEY from environment.
        Expects a final DataFrame with 'date' as index.
        """
        if not self.news_api_key:
            self.logger.error("No NewsAPI key set; cannot fetch news.")
            return pd.DataFrame()

        url = (f"https://newsapi.org/v2/everything?q={ticker}&pageSize={page_size}"
               f"&apiKey={self.news_api_key}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            news_data = []
            for article in articles:
                date_val = pd.to_datetime(article.get('publishedAt')).date()
                news_data.append({
                    'date': date_val,
                    'headline': article.get('title', ''),
                    'content': article.get('description', '') or '',
                    'symbol': ticker,
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ''),
                    'sentiment': TextBlob(article.get('description', '') or '').sentiment.polarity
                })
            self.logger.info(f"Fetched {len(news_data)} news articles for {ticker}")
            df = pd.DataFrame(news_data)
            if not df.empty:
                df.set_index('date', inplace=True)
            return df
        except requests.HTTPError as http_err:
            self.logger.error(f"HTTP error fetching news for {ticker}: {http_err}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching news for {ticker}: {e}")
        return pd.DataFrame()

    async def fetch_news_data_async(self, ticker: str, page_size: int = 5) -> pd.DataFrame:
        """
        Async wrapper for get_news_data.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_news_data, ticker, page_size)

    # -------------------------------------------------------------------
    # Fetch Data for Multiple Symbols
    # -------------------------------------------------------------------

    async def fetch_data_for_multiple_symbols(
        self,
        symbols: List[str],
        data_sources: List[str] = ["Alpaca", "Alpha Vantage", "Polygon", "Yahoo Finance", "Finnhub", "NewsAPI"],
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetches data for multiple stock symbols from various data sources.
        """
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        all_data = {}
        alpaca_interval = "1Day" if interval == "1d" else interval

        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                tasks.append(self._fetch_data_for_symbol(symbol, data_sources, start_date, end_date, alpaca_interval, session))
            results = await asyncio.gather(*tasks)
            for symbol, symbol_data in zip(symbols, results):
                all_data[symbol] = symbol_data

        return all_data

    async def _fetch_data_for_symbol(self, symbol: str, data_sources: List[str], start_date: str, end_date: str, interval: str, session: ClientSession) -> Dict[str, Any]:
        """
        Helper method to fetch data for a single symbol.
        """
        self.logger.info(f"Fetching data for symbol: {symbol}")
        symbol_data = {}

        if "Alpaca" in data_sources:
            try:
                alpaca_data = await self.fetch_alpaca_data_async(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )
                symbol_data["Alpaca"] = alpaca_data
            except Exception as e:
                self.logger.error(f"Alpaca data fetch failed for {symbol}: {e}")

        if "Alpha Vantage" in data_sources:
            try:
                alphavantage_data = await self.fetch_alphavantage_data(symbol, session, start_date, end_date)
                symbol_data["Alpha Vantage"] = alphavantage_data
            except Exception as e:
                self.logger.error(f"Alpha Vantage data fetch failed for {symbol}: {e}")

        if "Polygon" in data_sources:
            try:
                polygon_data = await self.fetch_polygon_data(symbol, session, start_date, end_date)
                symbol_data["Polygon"] = polygon_data
            except Exception as e:
                self.logger.error(f"Polygon data fetch failed for {symbol}: {e}")

        if "Yahoo Finance" in data_sources:
            try:
                yfinance_data = await self.fetch_stock_data_async(symbol, start_date, end_date, interval)
                symbol_data["Yahoo Finance"] = yfinance_data
            except Exception as e:
                self.logger.error(f"Yahoo Finance data fetch failed for {symbol}: {e}")

        if "Finnhub" in data_sources:
            try:
                finnhub_data = await self.fetch_finnhub_metrics(symbol, session)
                symbol_data["Finnhub"] = finnhub_data
            except Exception as e:
                self.logger.error(f"Finnhub data fetch failed for {symbol}: {e}")

        if "NewsAPI" in data_sources:
            try:
                newsapi_data = await self.fetch_news_data_async(symbol)
                symbol_data["NewsAPI"] = newsapi_data
            except Exception as e:
                self.logger.error(f"NewsAPI data fetch failed for {symbol}: {e}")

        return symbol_data

    # -------------------------------------------------------------------
    # Fetch Alpha Vantage Data
    # -------------------------------------------------------------------

    async def fetch_alphavantage_data(
        self, symbol: str, session: ClientSession, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetches historical data for a symbol from Alpha Vantage.

        Args:
            symbol (str): The stock symbol to fetch data for.
            session (ClientSession): The aiohttp session for making requests.
            start_date (str): Start date for filtering data (not used in Alpha Vantage API directly).
            end_date (str): End date for filtering data (not used in Alpha Vantage API directly).

        Returns:
            pd.DataFrame: A DataFrame containing the parsed historical data.
        """
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            self.logger.error("Alpha Vantage API key is missing.")
            return pd.DataFrame()

        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
        self.logger.debug(f"Attempting to fetch Alpha Vantage data for {symbol} from URL: {url}")

        try:
            # Fetch raw data with retries
            data = await self.fetch_with_retries(url, headers={}, session=session)

            # Validate response
            if not data:
                self.logger.error(f"No data received for {symbol} from Alpha Vantage.")
                return pd.DataFrame()

            # Parse the fetched data
            parsed_data = self._parse_alphavantage_data(data)

            if parsed_data.empty:
                self.logger.warning(f"No valid time series data found for {symbol}. Parsed DataFrame is empty.")
            else:
                self.logger.info(f"Successfully fetched and parsed Alpha Vantage data for {symbol}.")

            return parsed_data

        except Exception as e:
            self.logger.error(f"Unexpected error while fetching Alpha Vantage data for {symbol}: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------
    # Fetch Polygon Data
    # -------------------------------------------------------------------

    async def fetch_polygon_data(self, symbol: str, session: ClientSession, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for a symbol from Polygon.
        """
        api_key = os.getenv('POLYGONIO_API_KEY')
        if not api_key:
            self.logger.error("Polygon.io API key is not set in environment variables.")
            return pd.DataFrame()

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start_date}/{end_date}?adjusted=true&apiKey={api_key}"
        )
        data = await self.fetch_with_retries(url, headers={}, session=session)
        return self._parse_polygon_data(data)

    # -------------------------------------------------------------------
    # Parse Alpha Vantage Data
    # -------------------------------------------------------------------

    def _parse_alphavantage_data(self, data: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """
        Parses Alpha Vantage data into a pandas DataFrame.
        """
        if not data:
            self.logger.error("No data received for parsing.")
            return pd.DataFrame()

        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            self.logger.error("No 'Time Series (Daily)' found in Alpha Vantage response.")
            return pd.DataFrame()

        results = [
            {
                'date': pd.to_datetime(date),
                'open': float(values.get('1. open', 0)),
                'high': float(values.get('2. high', 0)),
                'low': float(values.get('3. low', 0)),
                'close': float(values.get('4. close', 0)),
                'volume': int(values.get('5. volume', 0))
            }
            for date, values in time_series.items()
        ]

        df = pd.DataFrame(results)
        df.set_index('date', inplace=True)
        return df

    # -------------------------------------------------------------------
    # Parse Polygon Data
    # -------------------------------------------------------------------

    def _parse_polygon_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Parses Polygon.io data into a pandas DataFrame.
        """
        results = data.get("results", [])
        if not results:
            self.logger.error("No data found in Polygon response.")
            return pd.DataFrame()

        parsed_results = []
        for item in results:
            try:
                parsed_results.append({
                    'date': datetime.fromtimestamp(item['t'] / 1000, tz=timezone.utc).date(),
                    'open': item['o'],
                    'high': item['h'],
                    'low': item['l'],
                    'close': item['c'],
                    'volume': item['v']
                })
            except KeyError as e:
                self.logger.warning(f"Missing key {e} in Polygon response item: {item}. Skipping entry.")
                continue

        if not parsed_results:
            self.logger.warning("No valid data entries parsed from Polygon response.")
            return pd.DataFrame()

        df = pd.DataFrame(parsed_results)
        df.set_index('date', inplace=True)
        return df


    # -------------------------------------------------------------------
    # Fetch Finnhub Metrics
    # -------------------------------------------------------------------

    async def fetch_finnhub_metrics(self, symbol: str, session: ClientSession) -> pd.DataFrame:
        """
        Fetches financial metrics from Finnhub.
        """
        if not self.finnhub_api_key:
            self.logger.error("Finnhub API key is not set. Cannot fetch metrics.")
            return pd.DataFrame()

        url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={self.finnhub_api_key}"
        self.logger.info(f"Fetching Finnhub financial metrics for symbol: {symbol}")
        data = await self.fetch_with_retries(url, headers={}, session=session)

        if not data or not isinstance(data.get("metric"), dict):
            self.logger.error("No valid metric data found in Finnhub response.")
            return pd.DataFrame(columns=["date_fetched"])  # Return an empty DataFrame with the required column

        return self._parse_finnhub_metrics_data(data)

    def _parse_finnhub_metrics_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Parses Finnhub metrics data into a pandas DataFrame.
        """
        self.logger.debug(f"Raw Finnhub data received: {data}")

        if "metric" not in data or not isinstance(data["metric"], dict):
            self.logger.error("Invalid or missing 'metric' key in Finnhub response.")
            return pd.DataFrame()

        # Extract metrics
        metrics = data["metric"]
        result = pd.DataFrame([{
            "52WeekHigh": metrics.get("52WeekHigh"),
            "52WeekLow": metrics.get("52WeekLow"),
            "MarketCapitalization": metrics.get("MarketCapitalization"),
            "P/E": metrics.get("P/E"),
            "date_fetched": pd.Timestamp.utcnow().floor("s")
        }]).set_index("date_fetched")

        self.logger.debug(f"Parsed Finnhub DataFrame: {result}")
        return result

    # -------------------------------------------------------------------
    # Additional Helper Methods
    # -------------------------------------------------------------------

    # Add any additional helper methods as needed, such as parsing methods for other APIs.
