# model_manager.py

import os
import joblib
import pickle
import shutil
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Tuple, Optional, Dict
from abc import ABC, abstractmethod
from datetime import datetime

# Optional: Uncomment if using AWS S3 or GCS
# import boto3
# from botocore.exceptions import ClientError
# from google.cloud import storage
# from google.api_core.exceptions import NotFound

# Uncomment the following line if using dotenv for environment variables
# from dotenv import load_dotenv

# Optional Encryption
from cryptography.fernet import Fernet

# Load environment variables from .env file if present
# load_dotenv()


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


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Configure and return a logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(Config.LOG_LEVEL.upper())
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


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


# Uncomment and configure the following classes if using AWS S3 or GCS

# class S3Storage(BaseStorage):
#     """
#     AWS S3 storage implementation.
#     """
#     def __init__(self, bucket_name: str, aws_access_key: str, aws_secret_key: str, region: str):
#         self.bucket_name = bucket_name
#         self.s3_client = boto3.client(
#             's3',
#             aws_access_key_id=aws_access_key,
#             aws_secret_access_key=aws_secret_key,
#             region_name=region
#         )

#     def save(self, local_path: Path, remote_path: str) -> None:
#         try:
#             self.s3_client.upload_file(str(local_path), self.bucket_name, remote_path)
#         except ClientError as e:
#             raise e

#     def load(self, remote_path: str, local_path: Path) -> None:
#         try:
#             local_path.parent.mkdir(parents=True, exist_ok=True)
#             self.s3_client.download_file(self.bucket_name, remote_path, str(local_path))
#         except ClientError as e:
#             raise e

#     def exists(self, remote_path: str) -> bool:
#         try:
#             self.s3_client.head_object(Bucket=self.bucket_name, Key=remote_path)
#             return True
#         except ClientError:
#             return False


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


class ModelManager:
    """
    Comprehensive model management for saving and loading models and scalers.
    """

    def __init__(self):
        self.logger = get_logger("ModelManager")
        self.storage = self._initialize_storage()
        self.serialization_format = Config.SERIALIZATION_FORMAT.lower()
        if self.serialization_format not in ["joblib", "pickle"]:
            self.logger.error(f"Unsupported serialization format: {self.serialization_format}")
            raise ValueError(f"Unsupported serialization format: {self.serialization_format}")
        # Handle encryption key
        self.encryption_key = Config.ENCRYPTION_KEY.encode() if Config.ENCRYPTION_KEY else None

    def _initialize_storage(self) -> BaseStorage:
        """
        Initializes the storage backend based on configuration.
        """
        backend = Config.STORAGE_BACKEND.lower()
        if backend == "local":
            return LocalStorage()
        # Uncomment and configure the following blocks if using AWS S3 or GCS
        # elif backend == "s3":
        #     if not all([Config.S3_BUCKET, Config.AWS_ACCESS_KEY, Config.AWS_SECRET_KEY, Config.AWS_REGION]):
        #         self.logger.error("S3 configuration incomplete.")
        #         raise ValueError("S3 configuration incomplete.")
        #     return S3Storage(
        #         bucket_name=Config.S3_BUCKET,
        #         aws_access_key=Config.AWS_ACCESS_KEY,
        #         aws_secret_key=Config.AWS_SECRET_KEY,
        #         region=Config.AWS_REGION
        #     )
        # elif backend == "gcs":
        #     if not all([Config.GCS_BUCKET, Config.GCS_CREDENTIALS_JSON]):
        #         self.logger.error("GCS configuration incomplete.")
        #         raise ValueError("GCS configuration incomplete.")
        #     return GCSStorage(
        #         bucket_name=Config.GCS_BUCKET,
        #         credentials_json=Config.GCS_CREDENTIALS_JSON
        #     )
        else:
            self.logger.error(f"Unsupported storage backend: {backend}")
            raise ValueError(f"Unsupported storage backend: {backend}")

    def _serialize(self, obj: Any, file_path: Path) -> None:
        """
        Serializes an object to a file using the specified format.
        """
        if self.serialization_format == "joblib":
            joblib.dump(obj, file_path)
        elif self.serialization_format == "pickle":
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
        self.logger.debug(f"Serialized object to {file_path}")

    def _deserialize(self, file_path: Path) -> Any:
        """
        Deserializes an object from a file using the specified format.
        """
        if self.serialization_format == "joblib":
            return joblib.load(file_path)
        elif self.serialization_format == "pickle":
            with open(file_path, "rb") as f:
                return pickle.load(f)

    def save_model_and_scaler(
        self,
        model: Any,
        scaler: Any,
        model_name: str,
        metadata: Dict[str, Any] = None,
        validate: bool = True
    ) -> Tuple[str, str, Optional[str]]:
        """
        Save the model and scaler with versioning, optional encryption, and validation.

        Args:
            model (Any): Trained model object.
            scaler (Any): Scaler object used in training.
            model_name (str): Base name for saving model and scaler.
            metadata (Dict[str, Any], optional): Additional metadata to save.
            validate (bool): Whether to validate save with checksum.

        Returns:
            Tuple[str, str, Optional[str]]: Remote paths to the saved model, scaler, and metadata.
        """
        try:
            # Define base names
            model_base = f"{model_name}_model"
            scaler_base = f"{model_name}_scaler"

            # Determine file extension
            ext = "pkl" if self.serialization_format == "pickle" else "joblib"

            # Generate versioned remote paths
            model_remote_path = generate_versioned_path(self.storage, model_base, ext)
            scaler_remote_path = generate_versioned_path(self.storage, scaler_base, ext)

            # Temporary local paths
            temp_dir = Path(Config.MODEL_DIR) / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            model_temp_path = temp_dir / f"{model_remote_path}"
            scaler_temp_path = temp_dir / f"{scaler_remote_path}"

            # Serialize model and scaler
            self._serialize(model, model_temp_path)
            self._serialize(scaler, scaler_temp_path)

            # Encrypt if needed
            if self.encryption_key:
                encrypt_file(model_temp_path, self.encryption_key)
                encrypt_file(scaler_temp_path, self.encryption_key)

            # Upload to storage
            self.storage.save(model_temp_path, model_remote_path)
            self.storage.save(scaler_temp_path, scaler_remote_path)
            self.logger.info(f"Model saved to {model_remote_path}")
            self.logger.info(f"Scaler saved to {scaler_remote_path}")

            # Handle metadata
            metadata_remote_path = None
            if metadata:
                metadata_remote_path = self._save_metadata(model_name, metadata)

            # Optional validation
            if validate:
                model_checksum = calculate_checksum(model_temp_path)
                scaler_checksum = calculate_checksum(scaler_temp_path)
                self.logger.info(f"Model checksum: {model_checksum}")
                self.logger.info(f"Scaler checksum: {scaler_checksum}")

            # Clean up temp files
            model_temp_path.unlink(missing_ok=True)
            scaler_temp_path.unlink(missing_ok=True)

            return model_remote_path, scaler_remote_path, metadata_remote_path

        except Exception as e:
            self.logger.error(f"Failed to save model and scaler: {e}")
            raise e

    def load_model_and_scaler(
        self,
        model_name: str,
        version: Optional[int] = None,
        load_metadata: bool = False
    ) -> Tuple[Any, Any, Optional[Dict[str, Any]]]:
        """
        Load the latest or specified version of the model and scaler, optionally with metadata.

        Args:
            model_name (str): Base name of the model and scaler files.
            version (Optional[int]): Specific version to load. If None, loads the latest version.
            load_metadata (bool): Whether to load associated metadata.

        Returns:
            Tuple[Any, Any, Optional[Dict[str, Any]]]: Loaded model, scaler, and metadata objects.
        """
        try:
            # Define base names
            model_base = f"{model_name}_model"
            scaler_base = f"{model_name}_scaler"

            # Determine file extension
            ext = "pkl" if self.serialization_format == "pickle" else "joblib"

            # Determine remote paths
            if version:
                model_remote_path = f"{model_base}_v{version}.{ext}"
                scaler_remote_path = f"{scaler_base}_v{version}.{ext}"
            else:
                model_remote_path = self._find_latest_version(model_base, ext)
                scaler_remote_path = self._find_latest_version(scaler_base, ext)

            if not model_remote_path or not scaler_remote_path:
                self.logger.error(f"Model or scaler not found for '{model_name}'")
                return None, None, None

            # Temporary local paths
            temp_dir = Path(Config.MODEL_DIR) / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            model_temp_path = temp_dir / Path(model_remote_path).name
            scaler_temp_path = temp_dir / Path(scaler_remote_path).name

            # Download from storage
            self.storage.load(model_remote_path, model_temp_path)
            self.storage.load(scaler_remote_path, scaler_temp_path)
            self.logger.info(f"Model downloaded from {model_remote_path}")
            self.logger.info(f"Scaler downloaded from {scaler_remote_path}")

            # Decrypt if needed
            if self.encryption_key:
                decrypt_file(model_temp_path, self.encryption_key)
                decrypt_file(scaler_temp_path, self.encryption_key)

            # Deserialize
            model = self._deserialize(model_temp_path)
            scaler = self._deserialize(scaler_temp_path)
            self.logger.info(f"Model and scaler deserialized successfully.")

            # Clean up temp files
            model_temp_path.unlink(missing_ok=True)
            scaler_temp_path.unlink(missing_ok=True)

            # Load metadata if requested
            metadata = None
            if load_metadata:
                metadata = self._load_metadata(model_name, version)

            return model, scaler, metadata

        except Exception as e:
            self.logger.error(f"Failed to load model and scaler: {e}")
            raise e

    def _find_latest_version(self, base_name: str, ext: str) -> Optional[str]:
        """
        Finds the latest version of the file based on naming convention.

        Args:
            base_name (str): Base name of the file.
            ext (str): File extension.

        Returns:
            Optional[str]: Remote path to the latest version.
        """
        version = 1
        latest_path = None
        while True:
            remote_path = f"{base_name}_v{version}.{ext}"
            if self.storage.exists(remote_path):
                latest_path = remote_path
                version += 1
            else:
                break
        if latest_path:
            return latest_path
        return None

    def _save_metadata(self, model_name: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Saves metadata associated with the model.

        Args:
            model_name (str): Base name of the model.
            metadata (Dict[str, Any]): Metadata to save.

        Returns:
            Optional[str]: Remote path to the metadata file.
        """
        try:
            metadata["model_name"] = model_name
            metadata["saved_at"] = datetime.utcnow().isoformat()
            metadata_filename = f"{model_name}_metadata.json"

            # Determine versioned path
            metadata_remote_path = generate_versioned_path(self.storage, metadata_filename.replace('.json', ''), 'json')

            # Temporary local path
            temp_dir = Path(Config.MODEL_DIR) / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            metadata_temp_path = temp_dir / Path(metadata_remote_path).name

            # Save metadata as JSON
            with metadata_temp_path.open("w") as f:
                json.dump(metadata, f)
            self.logger.debug(f"Serialized metadata to {metadata_temp_path}")

            # Encrypt if needed
            if self.encryption_key:
                encrypt_file(metadata_temp_path, self.encryption_key)

            # Upload to storage
            self.storage.save(metadata_temp_path, metadata_remote_path)
            self.logger.info(f"Metadata saved to {metadata_remote_path}")

            # Clean up temp file
            metadata_temp_path.unlink(missing_ok=True)

            return metadata_remote_path

        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            return None

    def _load_metadata(self, model_name: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Loads metadata associated with the model.

        Args:
            model_name (str): Base name of the model.
            version (Optional[int]): Specific version to load. If None, loads the latest version.

        Returns:
            Optional[Dict[str, Any]]: Metadata dictionary or None if not found.
        """
        try:
            metadata_base = f"{model_name}_metadata"
            ext = "json"

            if version:
                metadata_remote_path = f"{metadata_base}_v{version}.{ext}"
            else:
                metadata_remote_path = self._find_latest_version(metadata_base, ext)

            if not metadata_remote_path:
                self.logger.error(f"Metadata not found for '{model_name}'")
                return None

            # Temporary local path
            temp_dir = Path(Config.MODEL_DIR) / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            metadata_temp_path = temp_dir / Path(metadata_remote_path).name

            # Download from storage
            self.storage.load(metadata_remote_path, metadata_temp_path)
            self.logger.info(f"Metadata downloaded from {metadata_remote_path}")

            # Decrypt if needed
            if self.encryption_key:
                decrypt_file(metadata_temp_path, self.encryption_key)

            # Load JSON
            with metadata_temp_path.open("r") as f:
                metadata = json.load(f)
            self.logger.info(f"Metadata deserialized successfully.")

            # Clean up temp file
            metadata_temp_path.unlink(missing_ok=True)

            return metadata

        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None


# Usage Example
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Initialize ModelManager
    manager = ModelManager()

    # Mock training data
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # Train scaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Define metadata
    metadata = {
        "training_data": "dataset_v1.csv",
        "accuracy": 0.95,
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    # Save model and scaler with metadata
    model_path, scaler_path, metadata_path = manager.save_model_and_scaler(
        model=model,
        scaler=scaler,
        model_name="rf_trading_model",
        metadata=metadata,
        validate=True
    )

    print(f"Model saved at: {model_path}")
    print(f"Scaler saved at: {scaler_path}")
    print(f"Metadata saved at: {metadata_path}")

    # Load model, scaler, and metadata
    loaded_model, loaded_scaler, loaded_metadata = manager.load_model_and_scaler(
        model_name="rf_trading_model",
        load_metadata=True
    )

    print("Loaded Metadata:", loaded_metadata)