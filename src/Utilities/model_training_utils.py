# -------------------------------------------------------------------
# File Path: D:\TradingRobotPlug2\src\Utilities\model_training_utils.py
# Description:
#     Handles saving, loading, and managing versions of machine learning models.
#     Supports different model types (e.g., LSTM, Neural Networks) and ensures
#     proper storage of models, scalers, and metadata.
# -------------------------------------------------------------------
import sys
import os
import json
import joblib  # Added import for joblib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import datetime
import numpy as np  # Added import for numpy

from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------------------------------------------------------
# Section 1: Project Path Setup
# -------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# -------------------------------------------------------------------
# Section 2: Import ConfigManager and Logging
# -------------------------------------------------------------------
from Utilities.config_manager import ConfigManager, setup_logging
from Utilities.data.data_store_interface import DataStoreInterface


# -------------------------------------------------------------------
# Section 3: ModelManager Class
# -------------------------------------------------------------------
class ModelManager:
    def __init__(self, logger: logging.Logger, config_manager: Optional[ConfigManager] = None):
        """
        Initializes the ModelManager with logging and configuration.

        Args:
            logger (logging.Logger): Logger instance for logging.
            config_manager (ConfigManager, optional): Configuration manager instance.
        """
        self.logger = logger
        self.config_manager = config_manager
        self.model_directory = self._get_model_save_dir()
        self.logger.info(f"ModelManager initialized with model directory: {self.model_directory}")

    def _get_model_save_dir(self) -> Path:
        """
        Retrieves the model save directory from the configuration or uses a default path.

        Returns:
            Path: The root directory where models are saved.
        """
        if self.config_manager:
            model_save_path = self.config_manager.get("MODEL_SAVE_PATH", fallback="D:/TradingRobotPlug2/SavedModels")
            return Path(model_save_path)
        else:
            return Path("D:/TradingRobotPlug2/SavedModels")  # Default fallback path

    def get_model_directory(self, symbol: str, model_type: str) -> Path:
        """
        Retrieve or create the model directory for a specific symbol and model type.

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL').
            model_type (str): Type of model (e.g., 'lstm').

        Returns:
            Path: Directory where the model should be saved.
        """
        model_dir = self.model_directory / model_type / symbol
        model_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Model directory set to: {model_dir}")
        return model_dir

    def save_model(
        self,
        model,
        symbol: str,
        model_type: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        scaler: Optional[Any] = None
    ) -> Dict[str, Path]:
        """
        Save a model and associated information such as hyperparameters, metrics, and scaler.

        Args:
            model: The model to save.
            symbol (str): Stock symbol.
            model_type (str): Type of model, e.g., 'lstm'.
            hyperparameters (dict): Hyperparameters for the model.
            metrics (dict): Performance metrics for the model.
            scaler: Optional scaler used for data normalization.

        Returns:
            dict: Paths to saved model, scaler, and metadata files.
        """
        self.logger.info(f"Saving model for {symbol}, type {model_type}")
        model_dir = self.get_model_directory(symbol, model_type)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v_{timestamp}"
        version_dir = model_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Version directory created at: {version_dir}")

        # Define model file paths
        if model_type.lower() in ['lstm', 'neural_network']:
            model_file = version_dir / f"{symbol}_{model_type}_model.h5"
        else:
            model_file = version_dir / f"{symbol}_{model_type}_model.pkl"

        # Define scaler file path
        scaler_file = version_dir / f"{symbol}_scaler.pkl"

        # Save model
        try:
            if model_type.lower() in ['lstm', 'neural_network']:
                save_model(model, model_file)
                self.logger.debug(f"Keras model saved at: {model_file}")
            else:
                joblib.dump(model, model_file)
                self.logger.debug(f"Joblib model saved at: {model_file}")
        except Exception as e:
            self.logger.error(f"Failed to save model at {model_file}: {e}", exc_info=True)
            raise

        # Save scaler if available
        if scaler:
            try:
                joblib.dump(scaler, scaler_file)
                self.logger.debug(f"Scaler saved at: {scaler_file}")
            except Exception as e:
                self.logger.error(f"Failed to save scaler at {scaler_file}: {e}", exc_info=True)
                raise

        # Save metadata
        metadata = {
            "model_type": model_type,
            "symbol": symbol,
            "model_file": str(model_file),
            "version": version,
            "timestamp": timestamp,
            "hyperparameters": hyperparameters,
            "metrics": metrics
        }
        metadata_file = version_dir / "metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            self.logger.debug(f"Metadata saved at: {metadata_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metadata at {metadata_file}: {e}", exc_info=True)
            raise

        self.logger.info(f"Model, scaler, and metadata saved for {symbol} at {version_dir}")
        return {"model": model_file, "scaler": scaler_file, "metadata": metadata_file}

    def load_model(
        self,
        symbol: str,
        model_type: str,
        version: Optional[str] = None
    ) -> Optional[Any]:
        """
        Load a model given the symbol, type, and optional version.

        Args:
            symbol (str): Stock symbol.
            model_type (str): Type of model.
            version (str, optional): Specific version of the model to load. If None, loads the latest version.

        Returns:
            Loaded model object or None if not found.
        """
        self.logger.info(f"Loading model for {symbol}, type {model_type}, version {version if version else 'latest'}")
        model_path = self._get_model_path(symbol, model_type, version)
        if model_path and model_path.exists():
            try:
                if model_type.lower() in ['lstm', 'neural_network']:
                    model = load_model(model_path)
                    self.logger.debug(f"Keras model loaded from: {model_path}")
                else:
                    model = joblib.load(model_path)
                    self.logger.debug(f"Joblib model loaded from: {model_path}")
                return model
            except Exception as e:
                self.logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
                return None
        self.logger.error(f"No model found for {symbol}, type {model_type}, version {version}")
        return None

    def _get_model_path(
        self,
        symbol: str,
        model_type: str,
        version: Optional[str] = None
    ) -> Optional[Path]:
        """
        Retrieve the model file path for a given symbol, type, and optional version.

        Args:
            symbol (str): Stock symbol.
            model_type (str): Type of model.
            version (str, optional): Specific version to load. If None, loads the latest.

        Returns:
            Path: Path to the model file or None if not found.
        """
        model_dir = self.get_model_directory(symbol, model_type)
        if version:
            version_dir = model_dir / version
            if not version_dir.exists():
                self.logger.error(f"Specified version directory does not exist: {version_dir}")
                return None
            model_file = version_dir / f"{symbol}_{model_type}_model.h5" if model_type.lower() in ['lstm', 'neural_network'] else version_dir / f"{symbol}_{model_type}_model.pkl"
        else:
            versions = sorted(model_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if not versions:
                self.logger.error(f"No versions found in model directory: {model_dir}")
                return None
            latest_version_dir = versions[0]
            model_file = latest_version_dir / f"{symbol}_{model_type}_model.h5" if model_type.lower() in ['lstm', 'neural_network'] else latest_version_dir / f"{symbol}_{model_type}_model.pkl"

        self.logger.debug(f"Model file path resolved to: {model_file}")
        return model_file if model_file.exists() else None

    def validate_model(self, symbol: str, model_type: str, version: Optional[str] = None) -> bool:
        """
        Validate if a model exists and can be loaded.

        Args:
            symbol (str): Stock symbol.
            model_type (str): Type of model.
            version (str, optional): Specific version of the model to validate.

        Returns:
            bool: True if model is valid and loaded successfully, False otherwise.
        """
        self.logger.info(f"Validating model for {symbol}, type {model_type}, version {version if version else 'latest'}")
        model = self.load_model(symbol, model_type, version)
        if model:
            self.logger.info(f"Model validation successful for {symbol}, type {model_type}")
            return True
        else:
            self.logger.error(f"Model validation failed for {symbol}, type {model_type}")
            return False

    def load_metadata(
        self,
        symbol: str,
        model_type: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load metadata for a specific model.

        Args:
            symbol (str): Stock symbol.
            model_type (str): Type of model.
            version (str, optional): Specific version of the metadata to load.

        Returns:
            dict: Metadata information or None if not found.
        """
        self.logger.info(f"Loading metadata for {symbol}, type {model_type}, version {version if version else 'latest'}")
        metadata_path = self._get_metadata_path(symbol, model_type, version)
        if metadata_path and metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.logger.debug(f"Metadata loaded from: {metadata_path}")
                return metadata
            except Exception as e:
                self.logger.error(f"Failed to load metadata from {metadata_path}: {e}", exc_info=True)
                return None
        self.logger.error(f"Metadata not found for {symbol}, type {model_type}, version {version}")
        return None

    def _get_metadata_path(
        self,
        symbol: str,
        model_type: str,
        version: Optional[str] = None
    ) -> Optional[Path]:
        """
        Retrieve the metadata file path for a given symbol, type, and optional version.

        Args:
            symbol (str): Stock symbol.
            model_type (str): Type of model.
            version (str, optional): Specific version to load. If None, loads the latest.

        Returns:
            Path: Path to the metadata file or None if not found.
        """
        model_dir = self.get_model_directory(symbol, model_type)
        if version:
            version_dir = model_dir / version
            metadata_file = version_dir / "metadata.json"
        else:
            versions = sorted(model_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if not versions:
                self.logger.error(f"No versions found in model directory: {model_dir}")
                return None
            latest_version_dir = versions[0]
            metadata_file = latest_version_dir / "metadata.json"

        self.logger.debug(f"Metadata file path resolved to: {metadata_file}")
        return metadata_file if metadata_file.exists() else None


# -------------------------------------------------------------------
# Section 4: Example Usage with Model Definition
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Setup logging
    log_dir = Path("D:/TradingRobotPlug2/logs")  # Example log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("model_training_utils", log_dir=log_dir)

    # Define configuration manager
    try:
        config_manager = ConfigManager(required_keys=[
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY",
            "LOG_FOLDER",
            "MODEL_SAVE_PATH"
        ])
    except ImportError:
        config_manager = None  # Use a mock or default if ConfigManager is not available

    # Initialize ModelManager
    model_manager = ModelManager(logger, config_manager)

    # Example usage: Creating and saving an LSTM model
    symbol = "TSLA"
    model_type = "lstm"
    hyperparameters = {"lstm_units": 100, "dropout_rate": 0.3, "learning_rate": 0.001}
    metrics = {"mse": 0.02, "mae": 0.01, "r2": 0.85}

    def create_lstm_model(input_shape: tuple) -> Sequential:
        """
        Creates and compiles an LSTM model based on provided hyperparameters.

        Args:
            input_shape (tuple): Shape of the input data (timesteps, features).

        Returns:
            Sequential: Compiled LSTM model.
        """
        model = Sequential()
        model.add(LSTM(hyperparameters['lstm_units'], input_shape=input_shape))
        model.add(Dropout(hyperparameters['dropout_rate']))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    input_shape = (10, 5)  # Example input shape (timesteps, features)
    model = create_lstm_model(input_shape)

    # Example: Save the model
    try:
        saved_files = model_manager.save_model(
            model=model,
            symbol=symbol,
            model_type=model_type,
            hyperparameters=hyperparameters,
            metrics=metrics,
            scaler=None  # Assuming no scaler is used for this example
        )
        print(f"Model saved successfully at: {saved_files}")
    except Exception as e:
        logger.error(f"Failed to save the model: {e}", exc_info=True)

    # Example: Load the latest model
    try:
        loaded_model = model_manager.load_model(symbol, model_type)
        if loaded_model:
            print(f"Model loaded successfully from: {loaded_model}")
        else:
            print("Failed to load the model.")
    except Exception as e:
        logger.error(f"Failed to load the model: {e}", exc_info=True)

    # Example: Validate the model
    is_valid = model_manager.validate_model(symbol, model_type)
    if is_valid:
        logger.info(f"Model validation successful for {symbol}, type {model_type}")
    else:
        logger.error(f"Model validation failed for {symbol}, type {model_type}")

    # Example: Load metadata
    try:
        metadata = model_manager.load_metadata(symbol, model_type)
        if metadata:
            print(f"Metadata loaded: {json.dumps(metadata, indent=4)}")
        else:
            print("No metadata found.")
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}", exc_info=True)
