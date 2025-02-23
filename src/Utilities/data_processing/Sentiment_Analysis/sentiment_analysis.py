import pandas as pd
import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

# ✅ Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class SentimentAnalyzer:
    """
    Handles sentiment processing for stock market discussions.
    It cleans text, extracts trading terms, incorporates market data, 
    and computes sentiment scores using VADER and TextBlob.
    """
    def __init__(self, input_file: str = None, output_file: str = None):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None

        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vader = SentimentIntensityAnalyzer()

        # Trading terms with their associated sentiment weights.
        self.trading_terms = {
            "bullish": 1.2, "bearish": -1.2, "buy": 1.5, "sell": -1.5, "short": -2.0, "long": 2.0, 
            "breakout": 1.7, "support": 0.8, "resistance": -0.8, "rally": 1.3, "dump": -1.5,
            "fomo": 1.4, "hodl": 1.1, "dip": 1.5, "volume": 1.2, "trend": 1.0, "earnings": 2.0,
            "market": 1.0, "profit": 1.5, "loss": -1.5, "entry": 1.2, "exit": -1.2, 
            "consolidation": 0.7, "reversal": 1.5, "contract": 1.0, "break": 1.0, "gap": 1.2, 
            "pivot": 1.3, "downtrend": -1.3, "uptrend": 1.3, "bear flag": -1.8, "bull flag": 1.8, 
            "divergence": 1.4, "double top": -1.5, "double bottom": 1.5
        }

        # Market-based factors for sentiment weighting.
        self.market_volume_factor, self.market_momentum_factor = self.get_market_factors()

    def get_market_factors(self) -> (float, float):
        """
        Fetches real-time TSLA stock data to adjust sentiment weighting.
        Returns:
            market_volume_factor (float): Adjusts weight based on trading volume.
            market_momentum_factor (float): Adjusts weight based on price change.
        """
        try:
            tsla = yf.Ticker("TSLA")
            hist = tsla.history(period="1d")
            if hist.empty:
                logging.warning("⚠️ Market data unavailable. Using default weighting.")
                return 1.0, 1.0
            # Using volume and percentage price change for weighting adjustments.
            volume = hist['Volume'].iloc[-1]
            price_change = hist['Close'].pct_change().iloc[-1]
            market_volume_factor = volume / 1e7  # normalization factor for volume
            market_momentum_factor = abs(price_change) * 100  # percentage change as momentum
            return market_volume_factor, market_momentum_factor
        except Exception as e:
            logging.error(f"❌ Error fetching market data: {e}")
            return 1.0, 1.0

    def preprocess_text(self, text: str) -> str:
        """
        Cleans and tokenizes input text.
        - Converts text to lowercase.
        - Removes URLs and special characters.
        - Applies tokenization, stopword removal, and lemmatization.
        """
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)      # Remove special characters
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def extract_trading_terms(self, text: str) -> (list, float):
        """
        Identifies trading-specific terms in the text and calculates a weighted sentiment.
        Returns:
            found_terms (list): List of trading terms found.
            weighted_sentiment (float): Sum of sentiment weights (adjusted by market momentum).
        """
        words = text.split()
        found_terms = [word for word in words if word in self.trading_terms]
        weighted_sentiment = sum(self.trading_terms[word] for word in found_terms) * self.market_momentum_factor
        return found_terms, weighted_sentiment

    def analyze_text(self, text: str) -> dict:
        """
        Analyzes a single piece of text and returns sentiment results.
        Combines VADER and TextBlob sentiment scores along with trading term adjustments.
        """
        cleaned_text = self.preprocess_text(text)
        trading_terms, weighted_sentiment = self.extract_trading_terms(cleaned_text)

        vader_score = self.vader.polarity_scores(cleaned_text)['compound']
        textblob_score = TextBlob(cleaned_text).sentiment.polarity

        # Final sentiment weighted by volume factor.
        final_sentiment = (
            vader_score * 0.4 +
            textblob_score * 0.3 +
            weighted_sentiment * 0.3 * self.market_volume_factor
        )

        return {
            "cleaned_text": cleaned_text,
            "trading_terms": trading_terms,
            "vader_sentiment": vader_score,
            "textblob_sentiment": textblob_score,
            "weighted_sentiment": weighted_sentiment,
            "final_sentiment": final_sentiment
        }

    def load_data(self):
        """
        Loads data from CSV into a DataFrame.
        The CSV is expected to have a column named 'text'.
        """
        try:
            self.df = pd.read_csv(self.input_file)
            logging.info("✅ Sentiment data loaded successfully.")
        except Exception as e:
            logging.error(f"❌ Error loading data from {self.input_file}: {e}")
            raise

    def analyze_sentiment(self):
        """
        Applies sentiment analysis across the DataFrame.
        Creates new columns for cleaned text, extracted trading terms,
        individual sentiment scores, and the final sentiment score.
        """
        if self.df is None:
            logging.error("Dataframe is empty. Please load the data first.")
            return

        self.df['cleaned_text'] = self.df['text'].astype(str).apply(self.preprocess_text)
        sentiment_results = self.df['cleaned_text'].apply(lambda x: pd.Series(self.extract_trading_terms(x),
                                                                               index=['trading_terms', 'weighted_sentiment']))
        self.df = pd.concat([self.df, sentiment_results], axis=1)
        self.df['trading_term_count'] = self.df['trading_terms'].apply(len)

        self.df['vader_sentiment'] = self.df['cleaned_text'].apply(lambda x: self.vader.polarity_scores(x)['compound'])
        self.df['textblob_sentiment'] = self.df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

        self.df['final_sentiment_score'] = (
            self.df['vader_sentiment'] * 0.4 +
            self.df['textblob_sentiment'] * 0.3 +
            self.df['weighted_sentiment'] * 0.3 * self.market_volume_factor
        )

    def save_results(self):
        """
        Saves the processed sentiment data to CSV.
        """
        try:
            self.df.to_csv(self.output_file, index=False)
            logging.info(f"✅ Processed sentiment data saved at {self.output_file}.")
        except Exception as e:
            logging.error(f"❌ Error saving data to {self.output_file}: {e}")

    def run_analysis(self):
        """
        Executes the full sentiment analysis pipeline:
        Loads data, processes sentiment, and saves the results.
        """
        logging.info("🚀 Running Sentiment Analysis Pipeline...")
        self.load_data()
        self.analyze_sentiment()
        self.save_results()
        logging.info("✅ Analysis Complete.")


# ✅ Usage Example
if __name__ == "__main__":
    # File-based processing (batch analysis)
    input_csv = "D:/TradingRobotPlug2/TSLA_sentiment.csv"
    output_csv = "D:/TradingRobotPlug2/processed_TSLA_sentiment.csv"
    analyzer = SentimentAnalyzer(input_file=input_csv, output_file=output_csv)
    analyzer.run_analysis()

    # Example of single text analysis
    sample_text = "Bullish sentiment as TSLA shows a breakout rally. Investors are buying on the dip!"
    result = analyzer.analyze_text(sample_text)
    logging.info("Single text analysis result:")
    logging.info(result)

