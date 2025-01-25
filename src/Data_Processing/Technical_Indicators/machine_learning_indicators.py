# -------------------------------------------------------------------
# File: machine_learning_indicators.py
# Location: D:\TradingRobotPlug2\src\Data_Processing\Technical_Indicators
# Description: Provides machine learning models and functions for trading algorithms.
# -------------------------------------------------------------------

import pandas as pd
import logging
from pathlib import Path
import sys
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from typing import List, Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import sklearn

# -------------------------------------------------------------------
# Dynamically Identify Current File Name
# -------------------------------------------------------------------
script_file = Path(__file__).resolve()
script_name = script_file.name  # e.g., "machine_learning_indicators.py"
project_root = script_file.parents[3]  # Adjusted for the project structure

# -------------------------------------------------------------------
# Print Where We're Working From
# -------------------------------------------------------------------
print(f"[{script_name}] Current script path: {script_file}")
print(f"[{script_name}] Project root: {project_root}")

# Ensure project_root is in sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables from the .env file
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"[{script_name}] Environment variables loaded from {env_path}")
else:
    print(f"[{script_name}] Warning: .env file not found at {env_path}")

# Additional directories
utilities_dir = project_root / 'src' / 'Utilities'
scripts_dir = project_root / 'src'
data_processing_dir = scripts_dir / 'Data_Processing'

sys.path.extend([
    str(utilities_dir.resolve()),
    str(scripts_dir.resolve()),
    str(data_processing_dir.resolve())
])

# -------------------------------------------------------------------
# Importing ConfigManager, Logging Setup, DatabaseHandler, DataStore
# -------------------------------------------------------------------
try:
    from Utilities.config_manager import ConfigManager, setup_logging
    from Utilities.db.db_handler import DatabaseHandler
    from Utilities.data.data_store import DataStore
    print(f"[{script_name}] Imported config_manager, db_handler, data_store successfully.")
except ImportError as e:
    print(f"[{script_name}] Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
log_dir = project_root / 'logs' / 'technical_indicators'
log_dir.mkdir(parents=True, exist_ok=True)

# Include the script name in the logger to make it obvious where logs come from
logger = setup_logging(
    script_name=script_name,
    log_dir=log_dir,
    max_log_size=5 * 1024 * 1024,  # 5 MB
    backup_count=3,
    console_log_level=logging.INFO,
    file_log_level=logging.DEBUG,
    feedback_loop_enabled=True
)

logger.info(f"[{script_name}] Logger initialized.")

# -------------------------------------------------------------------
# Configuration Setup
# -------------------------------------------------------------------
required_keys = [
    'POSTGRES_HOST',
    'POSTGRES_DBNAME',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_PORT',
    'CACHE_STRATEGY',
    'CACHE_DIRECTORY',
    'ALPHAVANTAGE_API_KEY',
    'ALPHAVANTAGE_BASE_URL'
]

try:
    config_manager = ConfigManager(env_file=env_path, required_keys=required_keys, logger=logger)
    logger.info(f"[{script_name}] ConfigManager initialized successfully.")
except KeyError as e:
    logger.error(f"[{script_name}] Missing required configuration keys: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Abstract Base Class for Indicators
# -------------------------------------------------------------------
class Indicator(ABC):
    """
    Abstract base class for all machine learning indicators.
    """
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# -------------------------------------------------------------------
# Helper Function: Standardize DataFrame
# -------------------------------------------------------------------
def standardize_dataframe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    1. Flatten MultiIndex columns if present.
    2. Rename all columns to lowercase except 'Date'.
    3. Ensure 'Date' is a column (not just an index).
    4. Convert 'Date' to datetime, dropping rows with invalid or missing dates.
    """
    # Flatten multi-index if any
    if isinstance(df.columns, pd.MultiIndex):
        logger.warning(f"[{script_name}] Flattening MultiIndex columns.")
        df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]

    def transform_col(col: str) -> str:
        return 'Date' if col.lower() == 'date' else col.lower()

    old_cols = df.columns.tolist()
    df.columns = [transform_col(c) for c in df.columns]
    logger.debug(f"[{script_name}] Columns changed from {old_cols} to {df.columns.tolist()}")

    # If the index is named 'date', reset it
    if df.index.name and df.index.name.lower() == 'date':
        logger.info(f"[{script_name}] Converting index 'date' into a column 'Date'.")
        df.reset_index(inplace=True)

    # If we have 'date' but not 'Date', rename
    if 'date' in df.columns and 'Date' not in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
        logger.debug(f"[{script_name}] Renamed 'date' -> 'Date'")

    if 'Date' not in df.columns:
        logger.warning(f"[{script_name}] No 'Date' column found after standardization.")
        return df

    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    invalid_dates = df['Date'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"[{script_name}] Dropping {invalid_dates} rows with invalid 'Date'.")
        df.dropna(subset=['Date'], inplace=True)

    return df

# -------------------------------------------------------------------
# MachineLearningIndicators Class
# -------------------------------------------------------------------
class MachineLearningIndicators:
    """
    Class to handle machine learning indicators with caching and DataStore integration.
    """
    def __init__(
        self,
        db_handler: Optional[DatabaseHandler] = None,
        config: Optional[ConfigManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the class and set up the DataStore for SQL data handling.

        Args:
            db_handler (DatabaseHandler, optional): For DB interactions.
            config (ConfigManager, optional): For configurations.
            logger (logging.Logger, optional): Logger instance.
        """
        self.db_handler = db_handler
        self.config = config
        # If caller doesn't pass a logger, we use the module-level logger above
        self.logger = logger or globals()['logger']
        self.logger.info(f"[{script_name}] Initializing MachineLearningIndicators class...")

        if self.config and self.db_handler:
            self.data_store = DataStore(config=self.config, logger=self.logger, use_csv=False)
            cache_dir = self.config.get('CACHE_DIRECTORY', project_root / 'data' / 'cache')
            self.cache_path = Path(cache_dir)
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"[{script_name}] Initialized with DataStore; cache dir: {self.cache_path}")
        else:
            self.data_store = None
            self.cache_path = project_root / 'data' / 'cache'
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"[{script_name}] Initialized without DataStore; DB ops unavailable.")

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            raise ValueError(f"The DataFrame is missing required columns: {missing_columns}")

    def load_data_from_sql(self, symbol: str) -> Optional[pd.DataFrame]:
        if not self.data_store:
            self.logger.error(f"[{script_name}] DataStore not initialized. Cannot load from SQL.")
            return None
        try:
            self.logger.info(f"[{script_name}] Loading data for {symbol} from SQL.")
            df = self.data_store.load_data(symbol=symbol)
            if df is None or df.empty:
                self.logger.warning(f"[{script_name}] No data for {symbol}.")
                return None
            self.logger.info(f"[{script_name}] Loaded {len(df)} records for {symbol}.")
            return df
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to load data for {symbol}: {e}", exc_info=True)
            return None

    def train_proprietary_model(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        save_path: Optional[str] = None
    ) -> GradientBoostingRegressor:
        try:
            self.logger.info(f"[{script_name}] Training proprietary model with features={feature_columns}, target={target_column}.")
            self.validate_dataframe(df, required_columns=feature_columns + [target_column])

            X = df[feature_columns].values
            y = df[target_column].values

            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            model.fit(X, y)

            self.logger.info(f"[{script_name}] Model trained successfully.")
            if save_path:
                joblib.dump(model, save_path)
                self.logger.info(f"[{script_name}] Model saved to {save_path}")
                version_save_path = Path(save_path).with_suffix('.version.pkl')
                sklearn_version = sklearn.__version__
                joblib.dump(sklearn_version, version_save_path)
                self.logger.info(f"[{script_name}] scikit-learn version {sklearn_version} saved to {version_save_path}")

            return model
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to train model: {e}", exc_info=True)
            raise ValueError(f"[{script_name}] Error in training model: {e}")

    def apply_proprietary_prediction(
        self,
        df: pd.DataFrame,
        model: GradientBoostingRegressor,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        try:
            self.logger.info(f"[{script_name}] Applying proprietary model prediction.")
            self.validate_dataframe(df, required_columns=feature_columns)

            X = df[feature_columns].values
            df["proprietary_prediction"] = model.predict(X)
            self.logger.info(f"[{script_name}] Predictions added as 'proprietary_prediction'.")
            return df
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to apply prediction: {e}", exc_info=True)
            raise ValueError(f"[{script_name}] Error in applying prediction: {e}")

    def load_model(self, model_path: str) -> GradientBoostingRegressor:
        try:
            if not Path(model_path).exists():
                raise ValueError(f"[{script_name}] Model file not found: {model_path}")

            version_save_path = Path(model_path).with_suffix('.version.pkl')
            if version_save_path.exists():
                sklearn_version = joblib.load(version_save_path)
                self.logger.info(f"[{script_name}] Model trained with scikit-learn {sklearn_version}")
                current_version = sklearn.__version__
                if sklearn_version != current_version:
                    self.logger.warning(
                        f"[{script_name}] Version mismatch: trained on {sklearn_version}, current {current_version}"
                    )
            else:
                self.logger.warning(f"[{script_name}] No .version.pkl file found. Skipping version check.")

            with open(model_path, 'rb') as f:
                model = joblib.load(f)
            self.logger.info(f"[{script_name}] Model loaded from {model_path}")
            return model

        except Exception as e:
            self.logger.error(f"[{script_name}] Error loading model: {e}", exc_info=True)
            raise ValueError(f"[{script_name}] Unexpected error: {e}")

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"[{script_name}] Applying ML indicators...")

        # 1) Standardize
        df = standardize_dataframe(df, self.logger)

        # 2) Define features/target
        feature_columns = ['open', 'high', 'low']
        target_column = 'close'

        # Validate columns exist
        self.validate_dataframe(df, feature_columns + [target_column])

        # 3) Train or Load Model
        model_save_path = self.cache_path / 'proprietary_model.pkl'
        if model_save_path.exists():
            try:
                model = self.load_model(str(model_save_path))
            except ValueError as e:
                self.logger.warning(f"[{script_name}] Loading existing model failed: {e}. Retrying training.")
                model = self.train_proprietary_model(df, feature_columns, target_column, str(model_save_path))
        else:
            self.logger.info(f"[{script_name}] No model found at {model_save_path}, training a new one.")
            model = self.train_proprietary_model(df, feature_columns, target_column, str(model_save_path))

        # 4) Apply
        df = self.apply_proprietary_prediction(df, model, feature_columns)

        self.logger.info(f"[{script_name}] Machine learning indicators applied successfully.")
        return df

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    def main():
        print(f"[{script_name}] Entering main function for demonstration.")
        try:
            db_handler = DatabaseHandler(logger=logger)
            ml_indicators = MachineLearningIndicators(db_handler=db_handler, config=config_manager, logger=logger)

            symbol = "AAPL"
            df = ml_indicators.load_data_from_sql(symbol)
            if df is not None:
                df_with_indicators = ml_indicators.apply_indicators(df)

                # Save
                output_path = project_root / 'data' / f"{symbol}_with_ml_indicators.csv"
                df_with_indicators.to_csv(output_path, index=False)
                logger.info(f"[{script_name}] Data saved to {output_path}")

                # Check for 'proprietary_prediction'
                if 'proprietary_prediction' not in df_with_indicators.columns:
                    logger.error(f"[{script_name}] Column 'proprietary_prediction' missing.")
                else:
                    print(f"\n[{script_name}] 'proprietary_prediction' sample:\n",
                          df_with_indicators[['Date', 'proprietary_prediction']].head(10))

            else:
                logger.error(f"[{script_name}] No data returned for {symbol}")
        except Exception as e:
            logger.error(f"[{script_name}] An error occurred in main: {e}", exc_info=True)
        finally:
            if db_handler:
                try:
                    db_handler.close()
                    logger.info(f"[{script_name}] Database connection closed.")
                except Exception as e:
                    logger.error(f"[{script_name}] Error closing DB connection: {e}")

    main()
