# Filename: main_indicators.py
# Location: src/Utilities/data_processing
# Description:
#     Aggregates all indicators (Volume, Volatility, Trend, Momentum, 
#     Machine Learning, and Custom) into a single pipeline, applying 
#     them one after another to a DataFrame. This script demonstrates 
#     how to unify them and provides an example usage.


import sys
import logging
from pathlib import Path
import pandas as pd
from typing import Optional

# -------------------------------------------------------------------
# Ensure project paths are in sys.path if needed
# (Adjust the paths below if your directory structure is different)
# -------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root / "src" / "Utilities"))
sys.path.append(str(project_root / "src" / "Data_Processing"))

# -------------------------------------------------------------------
# Importing Various Indicator Classes
# -------------------------------------------------------------------
try:
    from Utilities.data_processing.Technical_Indicators.volume_indicators import VolumeIndicators
    from Utilities.data_processing.Technical_Indicators.volatility_indicators import VolatilityIndicators
    from Utilities.data_processing.Technical_Indicators.trend_indicators import TrendIndicators
    from Utilities.data_processing.Technical_Indicators.momentum_indicators import MomentumIndicators
    from Utilities.data_processing.Technical_Indicators.machine_learning_indicators import MachineLearningIndicators
    from Utilities.data_processing.Technical_Indicators.custom_indicators import CustomIndicators
    from Utilities.config_manager import ConfigManager, setup_logging
    from Utilities.db.db_handler import DatabaseHandler
    from Utilities.data.data_store import DataStore
except ImportError as e:
    print(f"[main_indicators.py] Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Main Code
# -------------------------------------------------------------------
class AllIndicatorsUnifier:
    """
    Applies all technical indicators by sequentially invoking each
    indicators class. This aggregator expects the underlying classes
    to handle their own column requirements and transformations.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Optional[logging.Logger] = None,
        use_csv: bool = False
    ):
        """
        Initializes references to each indicator set and the DataStore.

        Args:
            config_manager (ConfigManager): Configuration manager for environment variables.
            logger (logging.Logger, optional): Logger instance. Defaults to None.
            use_csv (bool): Whether DataStore should load/save CSV or SQL. Defaults to False.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info("[main_indicators.py] Initializing AllIndicatorsUnifier")


        # Setup DataStore for data retrieval and saving
        self.data_store = DataStore(config=config_manager, logger=self.logger, use_csv=use_csv)

        # Initialize each indicators class
        self.volume_indicators = VolumeIndicators(logger=self.logger)
        self.volatility_indicators = VolatilityIndicators(logger=self.logger)
        self.trend_indicators = TrendIndicators(
            data_store=self.data_store,
            logger=self.logger
        )
        self.momentum_indicators = MomentumIndicators(
            logger=self.logger,
            data_store=self.data_store
        )
        self.ml_indicators = MachineLearningIndicators(
            data_store=self.data_store,
            db_handler=None,      # Provide a real DB handler if needed
            config=config_manager,
            logger=self.logger
        )
        self.custom_indicators = CustomIndicators(
            db_handler=None,      # Provide a real DB handler if needed
            config_manager=config_manager,
            logger=self.logger
        )
        self.logger.info("[main_indicators.py] AllIndicatorsUnifier initialization complete.")

    def apply_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sequentially applies all sets of indicators to the DataFrame.
        Each method is expected to handle its own column naming and
        transformations.

        Args:
            df (pd.DataFrame): The DataFrame with price/volume data.

        Returns:
            pd.DataFrame: DataFrame enriched with all the technical indicators.
        """
        self.logger.info("[main_indicators.py] Applying Volume Indicators...")
        df = self.volume_indicators.apply_indicators(df)

        self.logger.info("[main_indicators.py] Applying Volatility Indicators...")
        df = self.volatility_indicators.apply_indicators(df)

        self.logger.info("[main_indicators.py] Applying Trend Indicators...")
        df = self.trend_indicators.apply_indicators(df)

        self.logger.info("[main_indicators.py] Applying Momentum Indicators...")
        df = self.momentum_indicators.apply_indicators(df)

        self.logger.info("[main_indicators.py] Applying Machine Learning Indicators...")
        df = self.ml_indicators.apply_indicators(df)

        self.logger.info("[main_indicators.py] Applying Custom Indicators...")
        df = self.custom_indicators.apply_indicators(df)

        self.logger.info("[main_indicators.py] All indicators applied successfully.")
        return df

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
def main():
    # 1) Setup config and logging
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
        print(f"[main_indicators.py] Missing config key: {e}")
        sys.exit(1)

    # Setup logging
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
    logger.info("[main_indicators.py] Logger initialized.")

    # 2) Create an instance of the aggregator
    aggregator = AllIndicatorsUnifier(config_manager=config_manager, logger=logger, use_csv=False)

    # 3) Load data for a chosen symbol
    symbol = "AAPL"
    df = aggregator.data_store.load_data(symbol=symbol)
    if df is None or df.empty:
        logger.error(f"[main_indicators.py] No data found for symbol '{symbol}'. Exiting.")
        return

    # 4) Apply all indicators
    df_with_indicators = aggregator.apply_all_indicators(df)

    # 5) Save or display result
    aggregator.data_store.save_data(df_with_indicators, symbol=symbol, overwrite=True)
    logger.info(f"[main_indicators.py] Data with all indicators saved for symbol {symbol}.")

    # Optional: Show the last few rows of the final DataFrame
    logger.info(df_with_indicators.tail(5).to_string())

if __name__ == "__main__":
    main()
