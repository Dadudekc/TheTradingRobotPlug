"""
File: main_indicators.py
Location: D:\TradingRobotPlug2\src\Utilities\data_processing

Description:
    A high-level orchestration file for applying indicators in a composable fashion.
    - Demonstrates how to use IndicatorCalculator for individual indicators.
    - Shows how to use MainIndicatorsAggregator (which internally calls AllIndicatorsUnifier)
      to apply a full suite of indicators at once.
    - Illustrates advanced usage (e.g., partial indicator application, custom pipelines).
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# --------------------------------------------------------------------------------
# Project Imports
# --------------------------------------------------------------------------------

# 1) Add Utilities path to sys.path if not already present.
project_root = Path(__file__).resolve().parents[2]
utilities_path = project_root / "src" / "Utilities"
if str(utilities_path) not in sys.path:
    sys.path.append(str(utilities_path))

# 2) Import your custom modules
from Utilities.config_manager import ConfigManager, setup_logging
from Utilities.db.db_handler import DBHandler
from Utilities.data.data_store import DataStore

from Utilities.data_processing.Technical_Indicators.indicator_unifier import AllIndicatorsUnifier
from Utilities.data_processing.Technical_Indicators.indicator_calculator import IndicatorCalculator
from Utilities.data_processing.Technical_Indicators.indicator_aggregator import MainIndicatorsAggregator


# --------------------------------------------------------------------------------
# Main Indicators Orchestrator
# --------------------------------------------------------------------------------
class MainIndicatorsOrchestrator:
    """
    Demonstrates multiple ways to compose your indicators:
     - Directly using IndicatorCalculator for granular indicator application.
     - Using MainIndicatorsAggregator (calls AllIndicatorsUnifier) to apply
       a full suite of indicators in one go.
    """

    def __init__(self, config_manager: ConfigManager, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.config_manager = config_manager
        self.logger.info("Initializing MainIndicatorsOrchestrator...")

        # Initialize aggregator (which uses AllIndicatorsUnifier under the hood).
        self.aggregator = MainIndicatorsAggregator(config_manager, logger=self.logger, use_csv=False)

    def apply_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Example of manually applying RSI & MACD using IndicatorCalculator.
        """
        self.logger.info("Manually applying RSI & MACD indicators...")
        df = df.copy()

        # Compute RSI
        df["RSI"] = IndicatorCalculator.compute_rsi(df["Close"], window=14)

        # Compute MACD and Signal
        df["MACD"], df["MACD_Signal"] = IndicatorCalculator.compute_macd(df["Close"], fast=12, slow=26, signal=9)

        # Additional indicators can be appended here
        self.logger.info("RSI & MACD manually applied.")
        return df

    def apply_full_indicator_suite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Example of applying the full set of indicators using
        the MainIndicatorsAggregator, which calls AllIndicatorsUnifier.
        """
        self.logger.info("Applying full indicator suite via aggregator...")
        df_with_indicators = self.aggregator.apply_all_indicators(df)
        self.logger.info("AllIndicatorsUnifier has applied the entire indicator pipeline.")
        return df_with_indicators


# --------------------------------------------------------------------------------
# Main Entry
# --------------------------------------------------------------------------------
def main():
    """
    Demonstration of how to orchestrate indicator usage in multiple ways.
    """
    # Load environment variables and config
    dotenv_path = project_root / ".env"
    required_keys = [
        "POSTGRES_HOST",
        "POSTGRES_DBNAME",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_PORT",
        # ... plus any other keys your project needs
    ]
    try:
        config_manager = ConfigManager(env_file=dotenv_path, required_keys=required_keys)
    except KeyError as e:
        print(f"Missing config key: {e}")
        sys.exit(1)

    # Set up logging
    log_dir = project_root / "logs" / "main_indicators"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        script_name="main_indicators.py",
        log_dir=log_dir,
        max_log_size=5 * 1024 * 1024,  # 5 MB
        backup_count=3,
        console_log_level=logging.INFO,
        file_log_level=logging.DEBUG
    )
    logger.info("Logger initialized for main_indicators.")

    # Initialize data store
    data_store = DataStore(config=config_manager, logger=logger, use_csv=False)

    # Load sample data from DB or CSV
    symbol = "AAPL"
    df = data_store.load_data(symbol)
    if df is None or df.empty:
        logger.error(f"No data found for symbol '{symbol}'. Exiting.")
        return

    # 1) Example: Manually apply a couple of indicators using IndicatorCalculator
    orchestrator = MainIndicatorsOrchestrator(config_manager, logger=logger)
    df_basic = orchestrator.apply_basic_indicators(df)
    data_store.save_data(df_basic, symbol=f"{symbol}_basic_indicators", overwrite=True)
    logger.info(f"Saved data with manually applied basic indicators for {symbol}.")

    # 2) Example: Apply the full suite via AllIndicatorsUnifier
    df_full = orchestrator.apply_full_indicator_suite(df)
    data_store.save_data(df_full, symbol=f"{symbol}_full_indicators", overwrite=True)
    logger.info(f"Saved data with FULL indicator suite for {symbol}.")

    # Show sample rows in the logs
    logger.info("Sample of df_basic:\n" + df_basic.tail(5).to_string())
    logger.info("Sample of df_full:\n" + df_full.tail(5).to_string())

    logger.info("All indicator processing complete.")


if __name__ == "__main__":
    main()
