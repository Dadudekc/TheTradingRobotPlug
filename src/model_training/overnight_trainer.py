# overnight_trainer.py

import os
import joblib
import pickle
import shutil
import hashlib
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Tuple, Optional, Dict
from abc import ABC, abstractmethod

# Optional: Uncomment if using AWS S3 or GCS
# import boto3
# from botocore.exceptions import ClientError
# from google.cloud import storage
# from google.api_core.exceptions import NotFound

# Optional Encryption
from cryptography.fernet import Fernet

# Uncomment the following line if using dotenv for environment variables
# from dotenv import load_dotenv

# Uncomment and configure the following lines if using dotenv
# load_dotenv()

# ==================== Configuration Management ====================

class Config:
    """
    Configuration management using environment variables.
    """
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")  # options: local, s3, gcs
    S3_BUCKET: str = os.getenv("S3_BUCKET", "")
    GCS_BUCKET: str = os.getenv("GCS_BUCKET", "")
    AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY", "")
    AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "")
    GCS_CREDENTIALS_JSON: str = os.getenv("GCS_CREDENTIALS_JSON", "")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "")  # Optional: Must be 32 url-safe base64-encoded bytes
    SERIALIZATION_FORMAT: str = os.getenv("SERIALIZATION_FORMAT", "joblib")  # joblib, pickle
    TRAINING_DURATION_HOURS: int = int(os.getenv("TRAINING_DURATION_HOURS", 8))
    TRAINING_SLEEP_SECONDS: int = int(os.getenv("TRAINING_SLEEP_SECONDS", 300))
    TRAINING_MAX_ITERATIONS: Optional[int] = int(os.getenv("TRAINING_MAX_ITERATIONS", 0)) or None
    TRAINING_MODEL_TYPE: str = os.getenv("TRAINING_MODEL_TYPE", "SVM")  # e.g., SVM, LSTM

# ==================== Logging Setup ====================

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Configure and return a logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# ==================== Storage Backend Interfaces ====================

class BaseStorage(ABC):
    """
    Abstract base class for storage backends.
    """
    @abstractmethod
    def save(self, local_path: Path, remote_path: str) -> None:
        pass

    @abstractmethod
    def load(self, remote_path: str, local_path: Path) -> None:
        pass

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        pass

class LocalStorage(BaseStorage):
    """
    Local filesystem storage implementation.
    """
    def save(self, local_path: Path, remote_path: str) -> None:
        dest = Path(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, dest)

    def load(self, remote_path: str, local_path: Path) -> None:
        src = Path(remote_path)
        if not src.exists():
            raise FileNotFoundError(f"Remote path {remote_path} does not exist.")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, local_path)

    def exists(self, remote_path: str) -> bool:
        return Path(remote_path).exists()
        
    def _initialize_storage(self) -> BaseStorage:
        """
        Initializes the storage backend based on configuration.
        Returns:
            BaseStorage: Initialized storage backend.
        """
        backend = self.config_manager.get('Storage', 'backend').lower()
        if backend == "local":
            return LocalStorage()
        elif backend == "s3":
            s3_bucket = self.config_manager.get('Storage', 's3_bucket')
            aws_access_key = self.config_manager.get('Storage', 'aws_access_key')
            aws_secret_key = self.config_manager.get('Storage', 'aws_secret_key')
            aws_region = self.config_manager.get('Storage', 'aws_region')
            if not all([s3_bucket, aws_access_key, aws_secret_key, aws_region]):
                self.logger.error("Incomplete S3 configuration.")
                raise ValueError("Incomplete S3 configuration.")
            return S3Storage(
                bucket_name=s3_bucket,
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                region=aws_region
            )
        else:
            self.logger.error(f"Unsupported storage backend: {backend}")
            raise ValueError(f"Unsupported storage backend: {backend}")

# class GCSStorage(BaseStorage):
#     """
#     Google Cloud Storage implementation.
#     """
#     def __init__(self, bucket_name: str, credentials_json: str):
#         self.bucket_name = bucket_name
#         self.client = storage.Client.from_service_account_json(credentials_json)
#         self.bucket = self.client.bucket(bucket_name)

#     def save(self, local_path: Path, remote_path: str) -> None:
#         blob = self.bucket.blob(remote_path)
#         blob.upload_from_filename(str(local_path))

#     def load(self, remote_path: str, local_path: Path) -> None:
#         blob = self.bucket.blob(remote_path)
#         local_path.parent.mkdir(parents=True, exist_ok=True)
#         blob.download_to_filename(str(local_path))

#     def exists(self, remote_path: str) -> bool:
#         blob = self.bucket.blob(remote_path)
#         return blob.exists()

# ==================== Utility Functions ====================

def calculate_checksum(file_path: Path) -> str:
    """
    Calculates SHA-256 checksum for a file.
    """
    hash_sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def generate_versioned_path(storage: BaseStorage, base_name: str, ext: str) -> str:
    """
    Generates a versioned remote path to prevent overwriting.
    """
    version = 1
    while True:
        remote_path = f"{base_name}_v{version}.{ext}"
        if not storage.exists(remote_path):
            return remote_path
        version += 1

def encrypt_file(file_path: Path, key: bytes) -> None:
    """
    Encrypts a file using the provided key.
    """
    fernet = Fernet(key)
    with file_path.open("rb") as f:
        data = f.read()
    encrypted = fernet.encrypt(data)
    with file_path.open("wb") as f:
        f.write(encrypted)

def decrypt_file(file_path: Path, key: bytes) -> None:
    """
    Decrypts a file using the provided key.
    """
    fernet = Fernet(key)
    with file_path.open("rb") as f:
        data = f.read()
    decrypted = fernet.decrypt(data)
    with file_path.open("wb") as f:
        f.write(decrypted)

# ==================== Model Management ====================

class ModelRegistry:
    """Registry to track and manage best models during training."""
    def __init__(self, storage: BaseStorage, logger: logging.Logger):
        self.best_model = None
        self.best_mse = float('inf')
        self.storage = storage
        self.logger = logger

    def update_best_model(self, mse: float, trainer: Any, model_name: str):
        """Update best model if current MSE is better."""
        if mse < self.best_mse:
            self.best_mse = mse
            self.best_model = trainer
            self.logger.info(f"New best model with MSE: {mse} saved as {model_name}")
            trainer.save_model(model_name)

    def get_best_mse(self) -> float:
        """Get the best MSE recorded so far."""
        return self.best_mse

class TrainerFactory:
    """Factory class to dynamically select trainer model."""
    @staticmethod
    def get_trainer(config: Dict[str, Any], logger: logging.Logger, model_type: str):
        if model_type == "SVM":
            return SVMTrainer(config, logger)
        elif model_type == "LSTM":
            return LSTMTrainer(config, logger)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

# ==================== Data Handler ====================

class DataHandler:
    """
    Handles data loading and preprocessing.
    """
    def __init__(self, config_manager: Any, logger: logging.Logger):
        self.config = config_manager.config
        self.logger = logger

    async def load_data(self, symbol: str) -> Any:
        """
        Asynchronously load data for the given symbol.
        Args:
            symbol (str): Stock symbol.
        Returns:
            DataFrame: Loaded data.
        """
        try:
            data_path = Path(self.config['Paths']['data_dir']) / f"{symbol}.csv"
            self.logger.info(f"Loading data from {data_path}")
            if not data_path.exists():
                self.logger.error(f"Data file {data_path} does not exist.")
                return pd.DataFrame()
            data = pd.read_csv(data_path)
            return data
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()

    async def preprocess_data(self, data: Any, features: list, target: str) -> Tuple[Any, Any, Any]:
        """
        Asynchronously preprocess the data.
        Args:
            data (DataFrame): Raw data.
            features (list): List of feature column names.
            target (str): Target column name.
        Returns:
            Tuple: (X, y, scaler)
        """
        try:
            self.logger.info("Preprocessing data")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = data[features]
            y = data[target]
            X_scaled = scaler.fit_transform(X)
            return X_scaled, y, scaler
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            return None, None, None

# ==================== Model Trainer Base Class ====================

class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def train(self, X_train: Any, y_train: Any) -> None:
        pass

    @abstractmethod
    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        pass

    @abstractmethod
    def save_model(self, model_name: str) -> None:
        pass

# ==================== Specific Model Trainers ====================

class SVMTrainer(BaseTrainer):
    """
    Support Vector Machine Trainer.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        from sklearn.svm import SVR
        self.model = SVR(**self.config.get('SVM', {}))

    def train(self, X_train: Any, y_train: Any) -> None:
        self.logger.info("Training SVM model")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        self.logger.info("Evaluating SVM model")
        predictions = self.model.predict(X_test)
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test, predictions)
        return {'mse': mse}

    def save_model(self, model_name: str) -> None:
        model_path = Path(Config.MODEL_DIR) / "models" / f"{model_name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving SVM model to {model_path}")
        joblib.dump(self.model, model_path)

class LSTMTrainer(BaseTrainer):
    """
    Long Short-Term Memory (LSTM) Trainer.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        self.model = Sequential()
        self.model.add(LSTM(self.config.get('LSTM', {}).get('units', 50), input_shape=self.config.get('LSTM', {}).get('input_shape', (None, 1))))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X_train: Any, y_train: Any) -> None:
        self.logger.info("Training LSTM model")
        self.model.fit(X_train, y_train, epochs=self.config.get('LSTM', {}).get('epochs', 10), batch_size=self.config.get('LSTM', {}).get('batch_size', 32), verbose=0)

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        self.logger.info("Evaluating LSTM model")
        predictions = self.model.predict(X_test)
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_test, predictions)
        return {'mse': mse}

    def save_model(self, model_name: str) -> None:
        model_path = Path(Config.MODEL_DIR) / "models" / f"{model_name}.h5"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving LSTM model to {model_path}")
        self.model.save(model_path)

# ==================== Overnight Trainer ====================

class OvernightTrainer:
    """
    Orchestrates continuous training sessions, model selection, and tracking of best performance.
    """
    def __init__(self, config_file: str):
        """
        Initialize the OvernightTrainer with the provided configuration file.
        Args:
            config_file (str): Path to the configuration file.
        """
        self.logger = self._setup_logger()
        self.config_manager = self._load_config(config_file)
        self.paths = self._setup_paths()
        self.storage = self._initialize_storage()
        self.model_registry = ModelRegistry(self.storage, self.logger)
        self.data_handler = DataHandler(self.config_manager, self.logger)
        self.max_duration = timedelta(hours=self.config_manager.TRAINING_DURATION_HOURS)
        self.sleep_time = self.config_manager.TRAINING_SLEEP_SECONDS
        self.max_iterations = self.config_manager.TRAINING_MAX_ITERATIONS
        self.model_type = self.config_manager.TRAINING_MODEL_TYPE

    def _load_config(self, config_file: str) -> Any:
        """
        Load configuration from a JSON or YAML file.
        Args:
            config_file (str): Path to the configuration file.
        Returns:
            Dict: Configuration dictionary.
        """
        try:
            self._validate_config_file(config_file)
            if config_file.endswith('.json'):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            elif config_file.endswith(('.yml', '.yaml')):
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported configuration file format. Use JSON or YAML.")
            self.logger.info(f"Configuration loaded from {config_file}")
            return config
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise

    def _validate_config_file(self, config_file: str) -> None:
        """
        Validate if the configuration file exists.
        Args:
            config_file (str): Path to the configuration file.
        """
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Configuration file {config_file} does not exist.")

    def _setup_paths(self) -> Any:
        """
        Setup necessary paths based on configuration.
        Returns:
            Paths: Paths object.
        """
        paths = Paths(self.config_manager.get('Paths', 'project_root'))
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        return paths

    def _setup_logger(self) -> logging.Logger:
        """
        Setup logger based on configuration.
        Returns:
            Logger: Configured logger.
        """
        log_dir = Path("logs")  # Use a temporary default path
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'overnight_trainer.log'

        logger = logging.getLogger('OvernightTrainer')
        logger.setLevel(getattr(logging, self.config_manager.LOG_LEVEL.upper(), logging.INFO))
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)

        return logger

    def _initialize_storage(self) -> BaseStorage:
        """
        Initializes the storage backend based on configuration.
        Returns:
            BaseStorage: Initialized storage backend.
        """
        backend = self.config_manager.STORAGE_BACKEND.lower()
        if backend == "local":
            return LocalStorage()
        # Uncomment and configure the following blocks if using AWS S3 or GCS
        # elif backend == "s3":
        #     if not all([self.config_manager.S3_BUCKET, self.config_manager.AWS_ACCESS_KEY, self.config_manager.AWS_SECRET_KEY, self.config_manager.AWS_REGION]):
        #         self.logger.error("S3 configuration incomplete.")
        #         raise ValueError("S3 configuration incomplete.")
        #     return S3Storage(
        #         bucket_name=self.config_manager.S3_BUCKET,
        #         aws_access_key=self.config_manager.AWS_ACCESS_KEY,
        #         aws_secret_key=self.config_manager.AWS_SECRET_KEY,
        #         region=self.config_manager.AWS_REGION
        #     )
        # elif backend == "gcs":
        #     if not all([self.config_manager.GCS_BUCKET, self.config_manager.GCS_CREDENTIALS_JSON]):
        #         self.logger.error("GCS configuration incomplete.")
        #         raise ValueError("GCS configuration incomplete.")
        #     return GCSStorage(
        #         bucket_name=self.config_manager.GCS_BUCKET,
        #         credentials_json=self.config_manager.GCS_CREDENTIALS_JSON
        #     )
        else:
            self.logger.error(f"Unsupported storage backend: {backend}")
            raise ValueError(f"Unsupported storage backend: {backend}")

    async def _load_and_preprocess_data(self, symbol: str) -> Optional[Tuple[Any, Any, Any, Any, Any]]:
        """
        Load and preprocess data for the given symbol.
        Args:
            symbol (str): Stock symbol.
        Returns:
            Optional[Tuple]: (X_train, X_test, y_train, y_test, scaler) or None if failed.
        """
        data = await self.data_handler.load_data(symbol)
        if data is None or data.empty:
            self.logger.error(f"No data to train on for symbol: {symbol}")
            return None
        features = self.config_manager.get('Features', 'feature_list').split(',')
        target = self.config_manager.get('Features', 'target')
        X, y, scaler = await self.data_handler.preprocess_data(data, features, target)
        if X is None or y is None:
            self.logger.error(f"Data preprocessing failed for symbol: {symbol}")
            return None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config_manager.getfloat('Training', 'test_size', fallback=0.2), shuffle=False
        )
        return X_train, X_test, y_train, y_test, scaler

    async def _train_and_evaluate(self, X_train, y_train, X_test, y_test, iteration: int) -> Tuple[float, Any]:
        """
        Train the model and evaluate its performance.
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Testing features.
            y_test: Testing labels.
            iteration (int): Current iteration number.
        Returns:
            Tuple[float, Any]: MSE and trained model.
        """
        trainer = TrainerFactory.get_trainer(self.config_manager, self.logger, self.model_type)
        self.logger.info(f"Iteration {iteration}: Training started with model type {self.model_type}")
        try:
            trainer.train(X_train, y_train)
            metrics = trainer.evaluate(X_test, y_test)
            mse = metrics.get('mse', float('inf'))
            self.logger.info(f"Iteration {iteration}: MSE={mse}")
            return mse, trainer
        except Exception as e:
            self.logger.error(f"Iteration {iteration}: Training/Evaluation failed: {e}")
            return float('inf'), None

    async def _sleep_between_iterations(self):
        """Asynchronously sleep between training iterations."""
        self.logger.info(f"Sleeping for {self.sleep_time} seconds before next iteration.")
        await asyncio.sleep(self.sleep_time)

    async def run_training_loop(self, symbol: str):
        """
        Run the continuous training loop for a specific symbol.
        Args:
            symbol (str): Stock symbol.
        """
        start_time = datetime.now()
        iteration = 0
        self.logger.info(f"Starting training loop for symbol: {symbol} at {start_time}")
        
        while True:
            current_time = datetime.now()
            elapsed_time = current_time - start_time

            # Check for maximum duration
            if elapsed_time > self.max_duration:
                self.logger.info(f"Maximum training duration of {self.max_duration} reached. Exiting training loop.")
                break

            # Check for maximum iterations if set
            if self.max_iterations and iteration >= self.max_iterations:
                self.logger.info(f"Maximum iterations of {self.max_iterations} reached. Exiting training loop.")
                break

            iteration += 1
            self.logger.info(f"Starting iteration {iteration}")

            # Load and preprocess data
            data = await self._load_and_preprocess_data(symbol)
            if data is None:
                self.logger.warning(f"Skipping iteration {iteration} due to data issues.")
                await self._sleep_between_iterations()
                continue

            X_train, X_test, y_train, y_test, scaler = data

            # Train and evaluate
            mse, trainer = await self._train_and_evaluate(X_train, y_train, X_test, y_test, iteration)
            if trainer:
                model_name = f"{symbol}_{self.model_type}_v{iteration}_{mse:.4f}"
                self.model_registry.update_best_model(mse, trainer, model_name)

            # Sleep before next iteration
            await self._sleep_between_iterations()

        self.logger.info(f"Training loop for symbol: {symbol} completed. Best MSE: {self.model_registry.get_best_mse()}")

    async def run(self, symbol: str):
        """
        Public method to start the training process.
        Args:
            symbol (str): Stock symbol.
        """
        await self.run_training_loop(symbol)

# ==================== Paths Management ====================

class Paths:
    """
    Manages project-related paths.
    """
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / "logs"
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"

# ==================== Logger Manager ====================

class LoggerManager:
    """
    Manages logger setup.
    """
    def __init__(self, log_file: Path, log_level: str):
        self.logger = logging.getLogger('OvernightTrainer')
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s'
        )
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def get_logger(self) -> logging.Logger:
        return self.logger

# ==================== Config Manager ====================

class ConfigManager:
    """
    Manages configuration loading and retrieval.
    """
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from JSON or YAML.
        Args:
            config_file (str): Path to the configuration file.
        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        try:
            if config_file.endswith('.json'):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            elif config_file.endswith(('.yml', '.yaml')):
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported configuration file format. Use JSON or YAML.")
            return config
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """
        Retrieve a configuration value.
        Args:
            section (str): Section name.
            key (str): Key name.
            fallback (Any): Fallback value if key is not found.
        Returns:
            Any: Retrieved value or fallback.
        """
        return self.config.get(section, {}).get(key, fallback)

    def getint(self, section: str, key: str, fallback: Any = None) -> int:
        """
        Retrieve a configuration value as integer.
        """
        return int(self.get(section, key, fallback)) if self.get(section, key, fallback) else fallback

    def getfloat(self, section: str, key: str, fallback: Any = None) -> float:
        """
        Retrieve a configuration value as float.
        """
        return float(self.get(section, key, fallback)) if self.get(section, key, fallback) else fallback

# ==================== Main Execution ====================

if __name__ == "__main__":
    import argparse
    import pandas as pd  # Ensure pandas is imported for DataHandler

    parser = argparse.ArgumentParser(description="Overnight Trainer for TradingRobotPlug")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (JSON or YAML)')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to train on')
    args = parser.parse_args()

    trainer = OvernightTrainer(config_file=args.config)

    asyncio.run(trainer.run(args.symbol))