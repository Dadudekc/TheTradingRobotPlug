"""
File: indicator_aggregator.py
Location: src/Utilities

Description:
    Aggregates all indicators by referencing AllIndicatorsUnifier.
    Uses lazy import for AllIndicatorsUnifier to avoid circular import issues.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
from typing import Optional

# Define project root (assumed to be 3 directories up).
project_root = Path(__file__).resolve().parents[2]

# Append the Utilities directory to sys.path if not already present.
utilities_path = project_root / "src" / "Utilities"
if str(utilities_path) not in sys.path:
    sys.path.append(str(utilities_path))

# Lazy Import Helper for AllIndicatorsUnifier to prevent circular import issues
def get_all_indicators_unifier():
    from Utilities.data_processing.Technical_Indicators.indicator_unifier import AllIndicatorsUnifier
    return AllIndicatorsUnifier

# Import required modules
from Utilities.config_manager import ConfigManager, setup_logging
from Utilities.db.db_handler import DBHandler
from Utilities.data.data_store import DataStore


class MainIndicatorsAggregator:
    """
    Example usage of the AllIndicatorsUnifier class.
    Uses lazy loading to avoid import issues.
    """

    def __init__(self, config_manager: ConfigManager, logger: Optional[logging.Logger] = None, use_csv: bool = False):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing MainIndicatorsAggregator")

        # Lazy instantiation of AllIndicatorsUnifier
        self.unifier = get_all_indicators_unifier()(config_manager, self.logger, use_csv=use_csv)

    def apply_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.unifier.apply_all_indicators(df)


def main():
    dotenv_path = project_root / ".env"
    required_keys = [
        "POSTGRES_HOST",
        "POSTGRES_DBNAME",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_PORT",
        "ALPHAVANTAGE_API_KEY",
        "ALPHAVANTAGE_BASE_URL",
        "ML_FEATURE_COLUMNS",
        "ML_TARGET_COLUMN",
        "ML_MODEL_PATH",
        "ML_MIN_ROWS"
    ]

    try:
        config_manager = ConfigManager(env_file=dotenv_path, required_keys=required_keys)
    except KeyError as e:
        print(f"Missing config key: {e}")
        sys.exit(1)

    # Set up logging.
    log_dir = project_root / "logs" / "technical_indicators"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        script_name="main_indicators.py",
        log_dir=log_dir,
        max_log_size=5 * 1024 * 1024,  # 5 MB
        backup_count=3,
        console_log_level=logging.INFO,
        file_log_level=logging.DEBUG
    )
    logger.info("Logger initialized.")

    # Initialize the aggregator.
    aggregator = MainIndicatorsAggregator(config_manager=config_manager, logger=logger, use_csv=False)
    data_store = DataStore(config=config_manager, logger=logger, use_csv=False)
    symbol = "AAPL"
    df = data_store.load_data(symbol)

    if df is None or df.empty:
        logger.error(f"No data found for symbol '{symbol}'. Exiting.")
        return

    # Apply all indicators.
    df_with_indicators = aggregator.apply_all_indicators(df)
    data_store.save_data(df_with_indicators, symbol=symbol, overwrite=True)
    logger.info(f"Data with all indicators saved for symbol {symbol}.")
    logger.info(df_with_indicators.tail(5).to_string())


if __name__ == "__main__":
    main()
