import os
import requests
import yfinance as yf

class SystemCheck:
    def __init__(self):
        """Initialize API keys and URLs for system check."""
        self.api_keys = {
            "Alpaca": os.getenv("ALPACA_API_KEY"),
            "Finnhub": os.getenv("FINNHUB_API_KEY"),
            "NewsAPI": os.getenv("NEWSAPI_API_KEY"),
        }
        self.urls = {
            "Alpaca": "https://paper-api.alpaca.markets/v2/account",
            "Finnhub": f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={self.api_keys['Finnhub']}",
            "NewsAPI": f"https://newsapi.org/v2/top-headlines?category=business&apiKey={self.api_keys['NewsAPI']}",
        }

    def check_connectivity(self):
        """Check API accessibility."""
        for service, url in self.urls.items():
            print(f"🔍 Checking {service} API...")
            
            # Skip check if API key is missing
            if self.api_keys.get(service) is None:
                print(f"⚠️ {service} API key is missing! Update your .env file.")
                continue

            try:
                headers = {}
                if service == "Alpaca":  # Alpaca requires authentication headers
                    headers = {
                        "APCA-API-KEY-ID": self.api_keys["Alpaca"],
                        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY"),
                    }
                
                response = requests.get(url, headers=headers if headers else None, timeout=5)
                
                if response.status_code == 200:
                    print(f"✅ {service} API is accessible.")
                else:
                    print(f"❌ {service} API returned error {response.status_code}: {response.text[:100]}")  # Print only first 100 chars
                
            except requests.RequestException as e:
                print(f"❌ {service} API connection failed: {e}")

    def check_yfinance(self, symbol="AAPL"):
        """Check stock data retrieval from Yahoo Finance using yfinance."""
        print(f"🔍 Checking Yahoo Finance via yfinance for {symbol}...")
        try:
            stock = yf.Ticker(symbol)
            # Use a reliable period
            data = stock.history(period="1d")  # Fetch one day of data

            if not data.empty:
                print(f"✅ yfinance is working! Latest {symbol} close price: {data['Close'].iloc[-1]:.2f}")
            else:
                print(f"❌ yfinance failed to retrieve data for {symbol}. No data returned.")

        except Exception as e:
            print(f"❌ yfinance encountered an error: {e}")

# Run system check
if __name__ == "__main__":
    checker = SystemCheck()
    checker.check_connectivity()
    checker.check_yfinance()  # Check Yahoo Finance via yfinance
