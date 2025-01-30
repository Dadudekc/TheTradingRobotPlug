# -------------------------------------------------------------------
# File: machine_learning_indicators.py
# Location: src/Data_Processing/Technical_Indicators
# Description: Provides machine learning-based technical indicators.
# -------------------------------------------------------------------

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
from abc import ABC, abstractmethod

import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv
import sklearn

# -------------------------------------------------------------------
# Dynamically Identify Current File Name and Project Root
# -------------------------------------------------------------------
script_file = Path(__file__).resolve()
script_name = script_file.name  # e.g., "machine_learning_indicators.py"
project_root = script_file.parents[3]  # Adjusted for the project structure

# -------------------------------------------------------------------
# Ensure project_root is in sys.path
# -------------------------------------------------------------------
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# -------------------------------------------------------------------
# Load Environment Variables from the .env File
# -------------------------------------------------------------------
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"[{script_name}] Environment variables loaded from {env_path}")
else:
    print(f"[{script_name}] Warning: .env file not found at {env_path}")

# -------------------------------------------------------------------
# Additional Directories Setup
# -------------------------------------------------------------------
utilities_dir = project_root / 'src' / 'Utilities'
scripts_dir = project_root / 'src'
data_processing_dir = scripts_dir / 'Data_Processing'

sys.path.extend([
    str(utilities_dir.resolve()),
    str(scripts_dir.resolve()),
    str(data_processing_dir.resolve())
])

# -------------------------------------------------------------------
# Importing ConfigManager, Logging Setup, DatabaseHandler, DataStore, ColumnUtils
# -------------------------------------------------------------------
try:
    from Utilities.config_manager import ConfigManager, setup_logging
    from Utilities.db.db_handler import DatabaseHandler
    from Utilities.data.data_store import DataStore
    from Utilities.column_utils import ColumnUtils
    print(f"[{script_name}] Imported config_manager, db_handler, data_store, column_utils successfully.")
except ImportError as e:
    print(f"[{script_name}] Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
log_dir = project_root / 'logs' / 'technical_indicators'
log_dir.mkdir(parents=True, exist_ok=True)

# Initialize Logger
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
    'ALPHAVANTAGE_BASE_URL',
    'ML_FEATURE_COLUMNS',        # Added for machine learning
    'ML_TARGET_COLUMN',          # Added for machine learning
    'ML_MODEL_PATH',             # Added for machine learning
    'ML_MIN_ROWS'                # Added for machine learning
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
# MachineLearningIndicators Class
# -------------------------------------------------------------------
class MachineLearningIndicators(Indicator):
    """
    Class to handle machine learning indicators with caching and DataStore integration.
    """
    def __init__(
        self,
        data_store: Optional[DataStore] = None,
        db_handler: Optional[DatabaseHandler] = None,
        config: Optional[ConfigManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the class and set up the DataStore for SQL data handling.

        Args:
            data_store (DataStore, optional): For data operations.
            db_handler (DatabaseHandler, optional): For DB interactions.
            config (ConfigManager, optional): For configurations.
            logger (logging.Logger, optional): Logger instance.
        """
        self.data_store = data_store
        self.db_handler = db_handler
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"[{script_name}] Initializing MachineLearningIndicators class...")

        if self.config and self.data_store:
            cache_dir = Path(self.config.get('CACHE_DIRECTORY', project_root / 'data' / 'cache'))
            self.cache_path = cache_dir
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"[{script_name}] Initialized with DataStore; cache dir: {self.cache_path}")
        else:
            self.cache_path = project_root / 'data' / 'cache'
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"[{script_name}] Initialized without DataStore; DB operations unavailable.")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply machine learning indicators to the provided DataFrame.
        Delegates to the 'apply_indicators' method.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with added machine learning indicators.
        """
        return self.apply_indicators(df)
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str], logger: logging.Logger):
        """
        Validates that the DataFrame contains the required columns.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            required_columns (List[str]): List of required column names.
            logger (logging.Logger): Logger instance.

        Raises:
            ValueError: If the DataFrame is missing any required columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The input 'df' must be a pandas DataFrame.")
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            logger.error(f"[{script_name}] The DataFrame is missing required columns: {missing_columns}")
            raise ValueError(f"[{script_name}] Missing required columns: {missing_columns}")
        logger.debug(f"[{script_name}] DataFrame contains all required columns.")

    def load_data_from_sql(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Loads data for a given symbol from the SQL database.

        Args:
            symbol (str): The stock symbol.

        Returns:
            Optional[pd.DataFrame]: The loaded DataFrame or None if failed.
        """
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
        """
        Trains a proprietary machine learning model.

        Args:
            df (pd.DataFrame): The input DataFrame.
            feature_columns (List[str]): List of feature column names.
            target_column (str): The target column name.
            save_path (Optional[str]): Path to save the trained model.

        Returns:
            GradientBoostingRegressor: The trained model.
        """
        try:
            self.logger.info(f"[{script_name}] Training proprietary model with features={feature_columns}, target={target_column}.")
            self.validate_dataframe(df, required_columns=feature_columns + [target_column], logger=self.logger)

            # Prepare the feature and target arrays
            X = df[feature_columns].values
            y = df[target_column].values

            # Train the model
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            model.fit(X, y)
            self.logger.info(f"[{script_name}] Model trained successfully.")

            # Ensure the save directory exists
            if save_path:
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)

                # Save the model
                joblib.dump(model, save_path)
                self.logger.info(f"[{script_name}] Model saved to {save_path}")

                # Save scikit-learn version for compatibility checks
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
        """
        Applies the proprietary ML model to generate predictions.

        Args:
            df (pd.DataFrame): The input DataFrame.
            model (GradientBoostingRegressor): The trained ML model.
            feature_columns (List[str]): List of feature column names.

        Returns:
            pd.DataFrame: DataFrame with added 'proprietary_prediction' column.
        """
        try:
            self.logger.info(f"[{script_name}] Applying proprietary model prediction.")
            self.validate_dataframe(df, required_columns=feature_columns, logger=self.logger)

            X = df[feature_columns].values
            if X.shape[1] != model.n_features_in_:
                self.logger.error(f"[{script_name}] Feature mismatch: Model expects {model.n_features_in_} features, but got {X.shape[1]}.")
                raise ValueError(f"Feature mismatch: Model expects {model.n_features_in_} features, but got {X.shape[1]}.")

            if X.size == 0:
                self.logger.warning(f"[{script_name}] Feature set X is empty. Skipping prediction.")
                return df

            predictions = model.predict(X)
            df["proprietary_prediction"] = predictions
            self.logger.info(f"[{script_name}] Predictions added as 'proprietary_prediction'.")
            return df
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to apply prediction: {e}", exc_info=True)
            # Decide whether to raise the exception or return df without predictions
            raise ValueError(f"[{script_name}] Error in applying prediction: {e}")

    def load_model(self, model_path: str) -> GradientBoostingRegressor:
        """
        Loads a trained model from the specified path.

        Args:
            model_path (str): Path to the saved model file.

        Returns:
            GradientBoostingRegressor: The loaded model.
        """
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

            model = joblib.load(model_path)
            self.logger.info(f"[{script_name}] Model loaded from {model_path}")
            return model

        except Exception as e:
            self.logger.error(f"[{script_name}] Error loading model: {e}", exc_info=True)
            raise ValueError(f"[{script_name}] Unexpected error: {e}")

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies machine learning indicators to the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with added machine learning indicators.
        """
        self.logger.info(f"[{script_name}] Applying ML indicators...")

        # Process the DataFrame using ColumnUtils
        try:
            column_config_path = project_root / 'src' / 'Utilities' / 'column_config.json'
            required_columns = self.config.get('ML_REQUIRED_COLUMNS', [])
            df = ColumnUtils.process_dataframe(
                df=df, 
                config_path=column_config_path, 
                required_columns=required_columns, 
                logger=self.logger
            )
            self.logger.info(f"[{script_name}] DataFrame processed with ColumnUtils.")
        except (KeyError, FileNotFoundError, ValueError) as ve:
            self.logger.error(f"[{script_name}] Data processing failed: {ve}")
            return df

        # Define features and target
        feature_columns = [col.strip() for col in self.config.get('ML_FEATURE_COLUMNS', '').split(',')]
        target_column = self.config.get('ML_TARGET_COLUMN', 'close')

        # Validate required columns (already done in ColumnUtils)
        # Additional validation can be performed here if necessary

        # Check data sufficiency
        try:
            MIN_ROWS = int(self.config.get('ML_MIN_ROWS', 50))
        except ValueError:
            self.logger.error(f"[{script_name}] ML_MIN_ROWS must be an integer.")
            MIN_ROWS = 50

        if len(df) < MIN_ROWS:
            self.logger.warning(f"[{script_name}] Not enough data ({len(df)} rows) to train/apply model. Required: {MIN_ROWS}")
            return df

        # Train or load the model
        model_save_path = Path(self.config.get('ML_MODEL_PATH', self.cache_path / 'proprietary_model.pkl'))
        if model_save_path.exists():
            try:
                model = self.load_model(str(model_save_path))
            except ValueError as e:
                self.logger.warning(f"[{script_name}] Loading existing model failed: {e}. Retrying training.")
                model = self.train_proprietary_model(df, feature_columns, target_column, str(model_save_path))
        else:
            self.logger.info(f"[{script_name}] No model found at {model_save_path}, training a new one.")
            model = self.train_proprietary_model(df, feature_columns, target_column, str(model_save_path))

        # Apply predictions
        try:
            df = self.apply_proprietary_prediction(df, model, feature_columns)
        except ValueError as ve:
            self.logger.error(f"[{script_name}] Prediction application failed: {ve}")
            return df

        # Optional: Log evaluation metrics
        try:
            X = df[feature_columns].values
            y_true = df[target_column].values
            y_pred = df["proprietary_prediction"].values

            rmse = mean_squared_error(y_true, y_pred, squared=False)
            r2 = r2_score(y_true, y_pred)
            self.logger.info(f"[{script_name}] Model Evaluation Metrics - RMSE: {rmse}, R²: {r2}")
        except Exception as e:
            self.logger.error(f"[{script_name}] Failed to evaluate model: {e}", exc_info=True)

        self.logger.info(f"[{script_name}] Machine learning indicators applied successfully.")
        return df

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    def main():
        print(f"[{script_name}] Entering main function for demonstration.")
        try:
            db_handler = DatabaseHandler(config=config_manager, logger=logger)
            data_store = DataStore(config=config_manager, logger=logger, use_csv=False)
            ml_indicators = MachineLearningIndicators(
                data_store=data_store,
                db_handler=db_handler,
                config=config_manager,
                logger=logger
            )

            symbols = ["AAPL", "MSFT", "GOOG"]
            for symbol in symbols:
                logger.info(f"[{script_name}] Processing indicators for {symbol}")
                try:
                    df = ml_indicators.load_data_from_sql(symbol)
                    if df is not None and not df.empty:
                        df_with_indicators = ml_indicators.apply_indicators(df)

                        # Ensure 'date' column exists and is in datetime format
                        if 'date' not in df_with_indicators.columns and 'Date' in df_with_indicators.columns:
                            df_with_indicators['date'] = df_with_indicators.index
                            logger.info(f"[{script_name}] Created 'date' column from index for {symbol}.")
                        elif 'date' not in df_with_indicators.columns and 'Date' not in df_with_indicators.columns:
                            logger.error(f"[{script_name}] 'date' column missing after processing for {symbol}. Skipping.")
                            continue

                        # Save the processed data
                        output_path = project_root / 'data' / f"{symbol}_with_ml_indicators.csv"
                        df_with_indicators.to_csv(output_path, index=False)
                        logger.info(f"[{script_name}] Data for {symbol} saved to {output_path}")

                        # Verify 'proprietary_prediction' column
                        if 'proprietary_prediction' in df_with_indicators.columns:
                            logger.info(f"[{script_name}] 'proprietary_prediction' sample for {symbol}:\n{df_with_indicators[['date', 'proprietary_prediction']].head(10)}")
                        else:
                            logger.error(f"[{script_name}] Column 'proprietary_prediction' missing for {symbol}.")
                    else:
                        logger.warning(f"[{script_name}] No data returned for {symbol}. Skipping.")
                except Exception as e:
                    logger.error(f"[{script_name}] Error processing {symbol}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"[{script_name}] An error occurred in main: {e}", exc_info=True)
        finally:
            if 'db_handler' in locals() and db_handler:
                try:
                    db_handler.close()
                    logger.info(f"[{script_name}] Database connection closed.")
                except Exception as e:
                    logger.error(f"[{script_name}] Error closing DB connection: {e}")

    main()
