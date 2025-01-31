from Utilities.fetchers.polygon_fetcher import PolygonDataFetcher
import logging

# Setup logging
logger = logging.getLogger("PolygonDataFetcher")
logging.basicConfig(level=logging.INFO)

# Initialize fetcher
polygon_fetcher = PolygonDataFetcher(logger=logger)

# Fetch data
symbol = "AAPL"
start_date = "2024-01-01"
end_date = "2024-01-10"
df = polygon_fetcher.fetch_stock_data(symbol, start_date, end_date)

# Display results
print(df.head())
