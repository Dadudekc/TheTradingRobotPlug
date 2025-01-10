# TRP2/Utilities/data_fetch_utils.py
# -------------------------------------------------------------------
# File Path: TRP2/Utilities/data_fetch_utils.py
# Description: A utility module for fetching and processing data from 
#              financial APIs (e.g., Alpaca, Polygon, Yahoo Finance) 
#              and news sources (e.g., NewsAPI) with integrated logging,
#              retry logic, and dynamic project path management. 
#              Sensitive configurations are securely loaded via .env.
# -------------------------------------------------------------------

import os
import sys
import pandas as pd
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List, Callable
from aiohttp import ClientSession, ClientConnectionError, ContentTypeError
from pathlib import Path
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import logging
import requests
from textblob import TextBlob
import inspect
import yfinance as yf
from datetime import datetime

# Import setup_logging from external logging module
from Scripts.Utilities.config_manager import setup_logging

# -------------------------------------------------------------------
# Dynamic Project Root Setup
# -------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Dynamically locate the project root based on the presence of a .env file.

    :return: Path to the project root.
    :raises RuntimeError: If .env file is not found in the parent directories.
    """
    script_dir = Path(__file__).resolve().parent
    for parent in script_dir.parents:
        if (parent / ".env").exists():
            return parent
    raise RuntimeError("Project root not found. Ensure .env file exists at the project root.")

# Initialize project root
PROJECT_ROOT = get_project_root()

# Define directories dynamically
DIRECTORIES = {
    'config': PROJECT_ROOT / 'config',
    'logs': PROJECT_ROOT / 'logs',
    'data': PROJECT_ROOT / 'data',
    'utilities': PROJECT_ROOT / 'Scripts' / 'Utilities',
}

# Add directories to sys.path for imports
for path in DIRECTORIES.values():
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    sys.path.append(str(path))

# Load environment variables
ENV_PATH = PROJECT_ROOT / '.env'
if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found at {ENV_PATH}")
load_dotenv(dotenv_path=ENV_PATH)

# -------------------------------------------------------------------
# Logging Setup
# -------------------------------------------------------------------

# Initialize logger using the external setup_logging function
LOGGER = setup_logging(
    script_name="data_fetch_utils",
    log_dir=DIRECTORIES['logs'],
    max_log_size=5 * 1024 * 1024,
    backup_count=3
)

# -------------------------------------------------------------------
# Alpaca API Initialization
# -------------------------------------------------------------------

def initialize_alpaca() -> Optional[tradeapi.REST]:
    """
    Initializes Alpaca API client using credentials from environment variables.

    :return: Initialized Alpaca API client or None if credentials are missing.
    """
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

    if not all([api_key, secret_key]):
        LOGGER.error("Alpaca API credentials are not fully set in the environment variables.")
        return None

    LOGGER.info("Initializing Alpaca API with provided credentials.")
    return tradeapi.REST(api_key, secret_key, base_url, api_version='v2')

# Initialize Alpaca client
ALPACA_CLIENT = initialize_alpaca()

# -------------------------------------------------------------------
# DataFetchUtils Class Definition
# -------------------------------------------------------------------

class DataFetchUtils:
    def __init__(self, logger=None):
        """
        Initialize the DataFetchUtils class.
        All configurations are loaded via environment variables.
        """
        self.logger = logger or logging.getLogger('DataFetchUtils')
        self.news_api_key = os.getenv('NEWSAPI_API_KEY')
        if not self.news_api_key:
            self.logger.error("NewsAPI key is not set in environment variables.")
            raise EnvironmentError("Missing NewsAPI key.")

        # Initialize Alpaca API client
        try:
            self.alpaca_api = initialize_alpaca()
            if self.alpaca_api:
                self.logger.info("Alpaca API client initialized successfully.")
            else:
                self.logger.warning("Alpaca API client could not be initialized.")
        except EnvironmentError as e:
            self.logger.error(f"Alpaca API initialization failed: {e}")
            self.alpaca_api = None  # Continue without Alpaca if initialization fails

    # -------------------------------------------------------------------
    # Fetching Data with Retry Logic
    # -------------------------------------------------------------------

    async def fetch_data(self, source_params: Dict) -> pd.DataFrame:
        """
        Fetches data from the specified source based on provided parameters.
        Dynamically selects the appropriate fetch method, handles errors, and logs progress.

        Parameters:
            source_params (dict): Includes keys like 'data_source', 'ticker', 'start_date', 'end_date', 'interval'.

        Returns:
            pd.DataFrame: Fetched data as a DataFrame or an empty DataFrame on failure.
        """
        data_source = source_params.get('data_source')
        ticker = source_params.get('ticker')
        start_date = source_params.get('start_date', '2023-01-01')
        end_date = source_params.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        interval = source_params.get('interval', '1d')

        # Map each source to its fetch method
        fetch_methods: Dict[str, Callable] = {
            "Yahoo Finance": self.fetch_stock_data_async,
            "Alpaca": self.fetch_alpaca_data_async,
            "Alpha Vantage": self.fetch_alphavantage_data,
            "Polygon": self.fetch_polygon_data,
            "Finnhub": self.fetch_finnhub_data,
            "NewsAPI": self.fetch_news_data_async,
        }

        fetch_method = fetch_methods.get(data_source)
        if not fetch_method:
            self.logger.error(f"Data source '{data_source}' is not supported.")
            return pd.DataFrame()  # Return empty DataFrame if unsupported source

        async def execute_fetch(method: Callable, *args, **kwargs) -> pd.DataFrame:
            """ Helper to handle retries and logging for fetch execution. """
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data = await method(*args, **kwargs)
                    if not data.empty:
                        self.logger.info(f"Data successfully fetched from {data_source} for {ticker}")
                    return data  # Return on success
                except Exception as e:
                    self.logger.error(f"Attempt {attempt + 1}/{max_retries} failed for {data_source}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            self.logger.error(f"Failed to fetch data from {data_source} after {max_retries} attempts.")
            return pd.DataFrame()  # Return empty on repeated failure

        # Dynamically gather arguments for the fetch method
        args = [ticker]  # Base args for all methods
        kwargs = {'start_date': start_date, 'end_date': end_date, 'interval': interval}
        
        # Conditionally add session for sources that need it
        async with aiohttp.ClientSession() as session:
            if 'session' in inspect.signature(fetch_method).parameters:
                kwargs['session'] = session

            # Execute fetch with error handling and retries
            return await execute_fetch(
                fetch_method, 
                *args, 
                **{k: v for k, v in kwargs.items() if k in inspect.signature(fetch_method).parameters}
            )

    async def fetch_with_retries(self, url: str, headers: Dict[str, str], session: ClientSession, retries: int = 3) -> Optional[Any]:
        """
        Fetches data from the provided URL with retries on failure.

        :param url: The API URL to fetch.
        :param headers: Headers to include in the request (e.g., API keys).
        :param session: An active ClientSession.
        :param retries: Number of retry attempts in case of failure.
        :return: Parsed JSON data as a dictionary or list, or None if failed.
        """
        for attempt in range(retries):
            try:
                self.logger.debug(f"Fetching data from URL: {url}, Attempt: {attempt + 1}")
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"Fetched data: {data}")
                        return data
                    elif response.status == 429:
                        self.logger.warning(f"Rate limit reached. Waiting before retry. Attempt {attempt + 1}/{retries}")
                        await asyncio.sleep(60)  # Wait for 60 seconds before retrying
                    else:
                        error_message = await response.text()
                        self.logger.error(f"Failed to fetch data from {url}. Status: {response.status}, Message: {error_message}")
            except ClientConnectionError as e:
                self.logger.error(f"ClientConnectionError: {e}")
            except ContentTypeError as e:
                self.logger.error(f"ContentTypeError: {e}")
            except asyncio.TimeoutError:
                self.logger.error(f"Timeout error fetching data from {url}. Retrying ({attempt + 1}/{retries})...")
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")

            await asyncio.sleep(2 ** attempt)  # Exponential backoff for retries

        self.logger.error(f"Failed to fetch data after {retries} attempts: {url}")
        return None

    # -------------------------------------------------------------------
    # Fetching Real-Time Stock Quotes
    # -------------------------------------------------------------------

    async def fetch_finnhub_quote(self, symbol, session):
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={self.finnhub_api_key}"
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()

                # Debug: Log the response data
                self.logger.debug(f"Finnhub response for {symbol}: {data}")

                if data:
                    df = pd.DataFrame([{
                        'date': pd.to_datetime(data['t'], unit='s'),
                        'current_price': data['c'],
                        'change': data['d'],
                        'percent_change': data['dp'],
                        'high': data['h'],
                        'low': data['l'],
                        'open': data['o'],
                        'previous_close': data['pc']
                    }]).set_index('date')
                    return df
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to fetch data from {url}. Error: {e}")
            return pd.DataFrame()


    # -------------------------------------------------------------------
    # Fetching Basic Financial Metrics
    # -------------------------------------------------------------------

    async def fetch_finnhub_metrics(self, symbol: str, session: ClientSession) -> Optional[Dict[str, Any]]:
        """
        Fetches basic financial metrics for a given symbol from Finnhub.

        :param symbol: Stock symbol (e.g., 'AAPL').
        :param session: An active aiohttp ClientSession.
        :return: Parsed JSON data as a dictionary or None if failed.
        """
        finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        if not finnhub_api_key:
            self.logger.error("Finnhub API key is not set in environment variables.")
            return None

        url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={finnhub_api_key}"
        self.logger.info(f"Fetching Finnhub financial metrics for symbol: {symbol}")

        return await self.fetch_with_retries(url, headers={}, session=session)

    # -------------------------------------------------------------------
    # Fetching Stock Symbols for an Exchange
    # -------------------------------------------------------------------

    async def fetch_finnhub_symbols(self, exchange: str, session: ClientSession) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches available stock symbols for a given exchange from Finnhub.

        :param exchange: The stock exchange (e.g., 'US' for United States).
        :param session: An active aiohttp ClientSession.
        :return: Parsed JSON data as a list of dictionaries or None if failed.
        """
        finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        if not finnhub_api_key:
            self.logger.error("Finnhub API key is not set in environment variables.")
            return None

        url = f"https://finnhub.io/api/v1/stock/symbol?exchange={exchange}&token={finnhub_api_key}"
        self.logger.info(f"Fetching Finnhub stock symbols for exchange: {exchange}")

        return await self.fetch_with_retries(url, headers={}, session=session)

    # -------------------------------------------------------------------
    # Fetching Company News
    # -------------------------------------------------------------------

    async def fetch_company_news(self, symbol: str, from_date: str, to_date: str, session: ClientSession) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches recent news articles for a given symbol from Finnhub.

        :param symbol: Stock symbol (e.g., 'AAPL').
        :param from_date: Start date in YYYY-MM-DD format.
        :param to_date: End date in YYYY-MM-DD format.
        :param session: An active aiohttp ClientSession.
        :return: Parsed JSON data as a list of dictionaries or None if failed.
        """
        finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        if not finnhub_api_key:
            self.logger.error("Finnhub API key is not set in environment variables.")
            return None

        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={finnhub_api_key}"
        self.logger.info(f"Fetching Finnhub company news for symbol: {symbol} from {from_date} to {to_date}")

        return await self.fetch_with_retries(url, headers={}, session=session)

    # -------------------------------------------------------------------
    # Fetching Stock Data Using yfinance
    # -------------------------------------------------------------------

    def get_stock_data(self, ticker: str, start_date: str = "2022-01-01", end_date: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
        """
        Fetches historical stock data for a given ticker using yfinance.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date for fetching data (YYYY-MM-DD).
            end_date (str): End date for fetching data (YYYY-MM-DD).
            interval (str): Data interval (e.g., '1d', '1h').

        Returns:
            pd.DataFrame: DataFrame containing stock data.
        """
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        try:
            # Fetch stock data
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

            # Ensure 'Date' is a column, not an index
            if isinstance(data.index, pd.DatetimeIndex):
                data.reset_index(inplace=True)

            # Validate the presence of 'Date'
            if 'Date' not in data.columns:
                raise ValueError("Stock data does not contain 'Date' column after reset_index.")

            data['Date'] = pd.to_datetime(data['Date']).dt.date
            data['symbol'] = ticker
            data.rename(columns={'Date': 'date', 'Adj Close': 'adj_close', 'Volume': 'volume'}, inplace=True)
            self.logger.info(f"Fetched {len(data)} records for ticker {ticker}")
            return data
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()

    async def fetch_stock_data_async(self, ticker: str, start_date: str = "2022-01-01", end_date: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
        """
        Asynchronously fetches stock data for a given ticker.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (str): Start date for fetching data (YYYY-MM-DD).
            end_date (str): End date for fetching data (YYYY-MM-DD).
            interval (str): Data interval (e.g., '1d').

        Returns:
            pd.DataFrame: DataFrame containing stock data.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_stock_data, ticker, start_date, end_date, interval)

    # -------------------------------------------------------------------
    # Fetching News Data Using NewsAPI
    # -------------------------------------------------------------------

    def get_news_data(self, ticker: str, page_size: int = 5) -> pd.DataFrame:
        """
        Fetches news articles related to a stock ticker using News API.

        Args:
            ticker (str): Stock ticker symbol.
            page_size (int): Number of articles to fetch.

        Returns:
            pd.DataFrame: DataFrame containing news articles.
        """
        url = f'https://newsapi.org/v2/everything?q={ticker}&pageSize={page_size}&apiKey={self.news_api_key}'
        try:
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            news_data = [
                {
                    'date': pd.to_datetime(article['publishedAt']).date(),
                    'headline': article['title'],
                    'content': article.get('description', '') or '',
                    'symbol': ticker,
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ''),
                    'sentiment': TextBlob(article.get('description', '') or '').sentiment.polarity
                }
                for article in articles
            ]
            self.logger.info(f"Fetched {len(news_data)} news articles for ticker {ticker}")
            return pd.DataFrame(news_data)
        except requests.HTTPError as http_err:
            self.logger.error(f"HTTP error fetching news for {ticker}: {http_err}")
        except requests.RequestException as req_err:
            self.logger.error(f"Request exception fetching news data for {ticker}: {req_err}")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching news data for {ticker}: {e}")
        return pd.DataFrame()

    async def fetch_news_data_async(self, ticker: str, page_size: int = 5) -> pd.DataFrame:
        """
        Asynchronously fetches news articles for a given ticker.

        Args:
            ticker (str): Stock ticker symbol.
            page_size (int): Number of articles to fetch.

        Returns:
            pd.DataFrame: DataFrame containing news articles.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_news_data, ticker, page_size)

    # -------------------------------------------------------------------
    # Fetching Alpaca Data
    # -------------------------------------------------------------------

    async def fetch_alpaca_data_async(self, symbol: str, start_date: str, end_date: str, interval: str = "1Day") -> pd.DataFrame:
        """
        Fetches historical stock data for a symbol from Alpaca asynchronously.

        :param symbol: Stock symbol.
        :param start_date: Start date for fetching data (YYYY-MM-DD).
        :param end_date: End date for fetching data (YYYY-MM-DD).
        :param interval: Data interval (e.g., '1Day').

        :return: DataFrame containing the fetched data.
        """
        if not self.alpaca_api:
            self.logger.error("Alpaca API client is not initialized.")
            return pd.DataFrame()

        # Ensure dates are provided and format them to exclude time
        if not start_date or not end_date:
            self.logger.error("Both start_date and end_date are required.")
            return pd.DataFrame()

        try:
            # Format dates as 'YYYY-MM-DD' without the 'T00:00:00' time component
            start_dt = pd.to_datetime(start_date).strftime('%Y-%m-%d')
            end_dt = pd.to_datetime(end_date).strftime('%Y-%m-%d')

            # Fetch data from Alpaca
            bars = self.alpaca_api.get_bars(
                symbol,
                timeframe=interval,
                start=start_dt,
                end=end_dt
            ).df

            if bars.empty:
                self.logger.warning(f"No data found in Alpaca response for {symbol}. Returning empty DataFrame.")
                return pd.DataFrame()

            # Adjust the DataFrame and set up necessary columns
            bars = bars.reset_index()
            bars.rename(columns={'timestamp': 'date', 'trade_count': 'volume'}, inplace=True)
            bars['symbol'] = symbol
            self.logger.info(f"Fetched Alpaca data for {symbol} from {start_date} to {end_date}")
            return bars

        except Exception as e:
            self.logger.error(f"Error fetching Alpaca data for {symbol}: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------
    # Converting JSON to DataFrame
    # -------------------------------------------------------------------

    def convert_json_to_dataframe(self, data: Any, api_source: str) -> pd.DataFrame:
        """
        Converts JSON data to a pandas DataFrame for multiple APIs.

        :param data: Fetched JSON data from an API.
        :param api_source: Source API to determine how to parse the data.
        :return: A pandas DataFrame.
        """
        self.logger.debug(f"Converting JSON data to DataFrame for API: {api_source}")

        if api_source == 'alphavantage':
            return self._parse_alphavantage_data(data)
        elif api_source == 'polygon':
            return self._parse_polygon_data(data)
        elif api_source == 'yfinance':
            return self._parse_yfinance_data(data)
        elif api_source == 'finnhub_quote':
            return self._parse_finnhub_quote_data(data)
        elif api_source == 'finnhub_metrics':
            return self._parse_finnhub_metrics_data(data)
        elif api_source == 'finnhub_symbols':
            return self._parse_finnhub_symbols_data(data)
        elif api_source == 'company_news':
            return self._parse_company_news_data(data)
        elif api_source == 'alpaca':
            return self._parse_alpaca_data(data)
        elif api_source == 'newsapi':
            return self._parse_newsapi_data(data)
        else:
            self.logger.error(f"Unknown API source: {api_source}")
            raise ValueError(f"Unknown API source: {api_source}")

    # -------------------------------------------------------------------
    # Data Parsing for Different APIs
    # -------------------------------------------------------------------

    def _parse_alphavantage_data(self, data: dict) -> pd.DataFrame:
        """Parses Alpha Vantage data into a pandas DataFrame."""
        time_series_key = "Time Series (Daily)"
        time_series = data.get(time_series_key, {})

        if not time_series:
            self.logger.error(f"No data found for key: {time_series_key}")
            return pd.DataFrame()

        results = [
            {
                'date': date,
                'open': float(values["1. open"]),
                'high': float(values["2. high"]),
                'low': float(values["3. low"]),
                'close': float(values["4. close"]),
                'volume': int(values["5. volume"])
            }
            for date, values in time_series.items()
        ]

        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def _parse_polygon_data(self, data: dict) -> pd.DataFrame:
        """Parses Polygon.io data into a pandas DataFrame."""
        results = data.get("results", [])

        if not results:
            self.logger.error("No data found in Polygon response.")
            return pd.DataFrame()

        parsed_results = [
            {
                'date': datetime.utcfromtimestamp(item['t'] / 1000).strftime('%Y-%m-%d'),
                'open': item['o'],
                'high': item['h'],
                'low': item['l'],
                'close': item['c'],
                'volume': item['v']
            }
            for item in results
        ]

        df = pd.DataFrame(parsed_results)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def _parse_yfinance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Parses Yahoo Finance data into a pandas DataFrame."""
        if data.empty:
            self.logger.error("Yahoo Finance data is empty.")
            return pd.DataFrame()

        # Assuming data is already in DataFrame format from get_stock_data
        data.set_index('date', inplace=True)
        return data

    def _parse_finnhub_quote_data(self, data: dict) -> pd.DataFrame:
        """Parses Finnhub quote data into a pandas DataFrame."""
        required_keys = ['c', 'd', 'dp', 'h', 'l', 'o', 'pc', 't']
        if not all(key in data for key in required_keys):
            self.logger.error("Missing quote data in Finnhub response.")
            return pd.DataFrame()

        df = pd.DataFrame([{
            'date': datetime.utcfromtimestamp(data['t']),
            'current_price': data['c'],
            'change': data['d'],
            'percent_change': data['dp'],
            'high': data['h'],
            'low': data['l'],
            'open': data['o'],
            'previous_close': data['pc']
        }])

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def _parse_finnhub_metrics_data(self, data: dict) -> pd.DataFrame:
        """Parses Finnhub financial metrics data into a pandas DataFrame."""
        metrics = data.get("metric", {})
        if not metrics:
            self.logger.error("No metric data found in Finnhub response.")
            return pd.DataFrame()

        df = pd.DataFrame([metrics])
        df['date_fetched'] = datetime.utcnow()
        df.set_index('date_fetched', inplace=True)
        return df

    def _parse_finnhub_symbols_data(self, data: list) -> pd.DataFrame:
        """Parses Finnhub stock symbols data into a pandas DataFrame."""
        if not isinstance(data, list) or not data:
            self.logger.error("No symbol data found in Finnhub response.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        return df

    def _parse_company_news_data(self, data: list) -> pd.DataFrame:
        """Parses Finnhub company news data into a pandas DataFrame."""
        if not isinstance(data, list):
            self.logger.error("Company news data is not in expected list format.")
            return pd.DataFrame()
        
        if not data:
            self.logger.warning("No company news found for the given date range.")
            return pd.DataFrame()  # Return empty DataFrame

        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        df.set_index('datetime', inplace=True)
        return df

    def _parse_alpaca_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Parses Alpaca data into a pandas DataFrame."""
        if data.empty:
            self.logger.warning("Received empty DataFrame from Alpaca.")
            return data

        # Assuming 'date' is already in datetime format
        data.set_index('date', inplace=True)
        return data

    def _parse_newsapi_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Parses NewsAPI data into a pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            self.logger.error("NewsAPI data is not in expected DataFrame format.")
            return pd.DataFrame()

        # Assuming data is already a DataFrame from get_news_data
        data.set_index('date', inplace=True)
        return data

    # -------------------------------------------------------------------
    # Fetching Data for Multiple Symbols and Projects
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

        :param symbols: List of stock symbols to fetch data for.
        :param data_sources: List of data sources to fetch data from.
        :param start_date: Start date for fetching data (YYYY-MM-DD).
        :param end_date: End date for fetching data (defaults to today if None).
        :param interval: Data interval (e.g., '1d', '1h').
        :return: A dictionary with symbols as keys, each containing a dictionary of DataFrames by data source.
        """
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
        all_data = {}

        # Define Alpaca-specific interval conversion if needed
        alpaca_interval = "1Day" if interval == "1d" else interval

        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                self.logger.info(f"Fetching data for symbol: {symbol}")
                symbol_data = {}

                # Fetch data from each specified source
                if "Alpaca" in data_sources:
                    try:
                        alpaca_data = await self.fetch_alpaca_data_async(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            interval=alpaca_interval
                        )
                        symbol_data["Alpaca"] = self._parse_alpaca_data(alpaca_data)
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
                        symbol_data["Yahoo Finance"] = self._parse_yfinance_data(yfinance_data)
                    except Exception as e:
                        self.logger.error(f"Yahoo Finance data fetch failed for {symbol}: {e}")

                if "Finnhub" in data_sources:
                    try:
                        finnhub_data = await self.fetch_finnhub_data(symbol, session, start_date, end_date)
                        symbol_data["Finnhub"] = finnhub_data
                    except Exception as e:
                        self.logger.error(f"Finnhub data fetch failed for {symbol}: {e}")

                if "NewsAPI" in data_sources:
                    try:
                        newsapi_data = await self.fetch_news_data_async(symbol)
                        symbol_data["NewsAPI"] = self._parse_newsapi_data(newsapi_data)
                    except Exception as e:
                        self.logger.error(f"NewsAPI data fetch failed for {symbol}: {e}")

                # Add fetched data for the current symbol to the main dictionary
                all_data[symbol] = symbol_data

        return all_data

    async def fetch_alphavantage_data(self, symbol: str, session: ClientSession, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for a symbol from Alpha Vantage.

        :param symbol: Stock symbol.
        :param session: An active aiohttp ClientSession.
        :param start_date: Start date for fetching data (YYYY-MM-DD).
        :param end_date: End date for fetching data (YYYY-MM-DD).
        :return: A DataFrame containing the fetched data.
        """
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        if not api_key:
            self.logger.error("Alpha Vantage API key is not set in environment variables.")
            return pd.DataFrame()

        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
        data = await self.fetch_with_retries(url, headers={}, session=session)
        return self.convert_json_to_dataframe(data, "alphavantage")

    async def fetch_polygon_data(self, symbol: str, session: ClientSession, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for a symbol from Polygon.

        :param symbol: Stock symbol.
        :param session: An active aiohttp ClientSession.
        :param start_date: Start date for fetching data (YYYY-MM-DD).
        :param end_date: End date for fetching data (YYYY-MM-DD).
        :return: A DataFrame containing the fetched data.
        """
        api_key = os.getenv('POLYGONIO_API_KEY')
        if not api_key:
            self.logger.error("Polygon.io API key is not set in environment variables.")
            return pd.DataFrame()

        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&apiKey={api_key}"
        data = await self.fetch_with_retries(url, headers={}, session=session)
        return self.convert_json_to_dataframe(data, "polygon")

    async def fetch_finnhub_data(self, symbol: str, session: aiohttp.ClientSession, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetches financial and quote data from Finnhub.

        :param symbol: Stock symbol.
        :param session: An active aiohttp ClientSession.
        :param start_date: Start date for fetching data (YYYY-MM-DD).
        :param end_date: End date for fetching data (YYYY-MM-DD).
        :return: Dictionary containing DataFrames for Finnhub data types.
        """
        self.logger.info(f"Fetching Finnhub data for symbol: {symbol}")
        quote_data = await self.fetch_finnhub_quote(symbol, session)
        metrics_data = await self.fetch_finnhub_metrics(symbol, session)
        return {
            "quote": self.convert_json_to_dataframe(quote_data, "finnhub_quote") if quote_data else pd.DataFrame(),
            "metrics": self.convert_json_to_dataframe(metrics_data, "finnhub_metrics") if metrics_data else pd.DataFrame(),
        }
