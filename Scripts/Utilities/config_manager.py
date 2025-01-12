# Scripts/Utilities/config_manager.py
# -------------------------------------------------------------------
# Description: Manages configuration and environment variables for the 
#              TradingRobotPlug project. Supports loading from environment 
#              variables, .env files, YAML, JSON, and TOML files, and 
#              provides type casting, validation, and dynamic reloading.
# -------------------------------------------------------------------

import os
import yaml
import json
import toml
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
import logging
from typing import Any, Optional, List, Dict, Union, Type
import threading

def setup_logging(
    script_name: str,
    log_dir: Path,
    max_log_size: int = 5 * 1024 * 1024,
    backup_count: int = 3
) -> logging.Logger:
    """
    Sets up a logger for a given script_name, writing logs to both a file and console.

    Args:
        script_name (str): The name of the script for log identification.
        log_dir (Path): Directory to store log files.
        max_log_size (int): Maximum log file size in bytes (unused in this example).
        backup_count (int): Number of backup log files to keep (unused in this example).

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Ensure log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)

        # File handler
        log_file = log_dir / f"{script_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

class ConfigManager:
    """
    A manager that loads configurations from YAML, JSON, TOML files and .env files,
    allows environment variables to override config values, and provides methods for
    reloading, listing, and validating configuration keys.
    """

    def __init__(
        self,
        config_files: Optional[List[Path]] = None,
        env_file: Optional[Path] = None,
        required_keys: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        project_root: Optional[Path] = None
    ):
        """
        Initializes the ConfigManager and loads configurations from various sources.

        Args:
            config_files (List[Path], optional): A list of configuration files to load (YAML, JSON, TOML).
            env_file (Path, optional): Path to the .env file. If not provided, attempts to load from project root.
            required_keys (List[str], optional): A list of required configuration keys to check for.
            logger (logging.Logger, optional): Logger object to log messages (optional).
            project_root (Path, optional): Explicit project root path. If not provided, auto-detects.
        """
        self.config = defaultdict(dict)
        self.logger = logger or self.setup_logger("ConfigManager")
        self.cache = {}
        self.lock = threading.Lock()

        # Determine the project root
        if project_root:
            self.project_root = project_root.resolve()
        else:
            script_dir = Path(__file__).resolve().parent
            self.project_root = script_dir.parents[2]  # Adjust as needed

        # Load the .env file if present
        env_path = env_file or (self.project_root / '.env')
        if env_path.exists():
            self._load_env(env_path)
        else:
            self.logger.warning(
                f"No .env file found at {env_path}. Environment variables will be used as-is."
            )

        # Load configurations from specified config files
        if config_files:
            self.load_configurations(config_files)

        # Check for missing required keys if any are provided
        self.required_keys = required_keys or []
        self.check_missing_keys(self.required_keys)

    def load_configurations(self, config_files: List[Path]):
        """
        Loads configuration data from various file formats (YAML, JSON, TOML).

        Args:
            config_files (List[Path]): List of configuration file paths.
        """
        for config_file in config_files:
            if not config_file.exists():
                self.logger.warning(f"Config file does not exist: {config_file}")
                continue

            file_ext = config_file.suffix.lower()
            try:
                if file_ext in ['.yaml', '.yml']:
                    self._load_yaml(config_file)
                elif file_ext == '.json':
                    self._load_json(config_file)
                elif file_ext == '.toml':
                    self._load_toml(config_file)
                else:
                    self.logger.warning(f"Unsupported config file format: {config_file}")
            except Exception as e:
                self.logger.error(f"Error loading {config_file}: {e}")

    def get(
        self,
        key: str,
        default: Any = None,
        fallback: Any = None,
        required: bool = False,
        value_type: Optional[Type] = None
    ) -> Any:
        """
        Retrieves a configuration value based on the provided key, with optional type casting.

        Args:
            key (str): The key to retrieve.
            default (Any, optional): Default value if key not found.
            fallback (Any, optional): Another fallback if key not found in config or env.
            required (bool, optional): If True, raises an error if the key is missing.
            value_type (Type, optional): Type to cast the retrieved value to.

        Returns:
            Any: The retrieved configuration value, optionally cast to the specified type.

        Raises:
            ValueError: If the key is required but not found.
            TypeError: If type casting fails.
        """
        with self.lock:
            full_key = key.lower()
            # If it's already in the cache, return it
            if full_key in self.cache:
                return self.cache[full_key]

            # Attempt to retrieve from environment variables first
            env_key = full_key.upper().replace('.', '_')
            value = os.getenv(env_key)

            # If not found in env, fall back to the config dictionary
            if value is None:
                value = self.config.get(full_key)

            # If still None, use default or fallback
            if value is None:
                if default is not None:
                    value = default
                elif fallback is not None:
                    value = fallback
                elif required:
                    self.logger.error(f"Configuration for '{env_key}' is required but not provided.")
                    raise ValueError(f"Configuration for '{env_key}' is required but not provided.")
                else:
                    value = None

            # Type casting if needed
            if value is not None and value_type is not None:
                try:
                    if value_type == bool:
                        value = self._str_to_bool(value)
                    elif value_type == list:
                        if isinstance(value, str):
                            value = [item.strip() for item in value.split(',')]
                        else:
                            value = list(value)
                    elif value_type == dict:
                        if isinstance(value, str):
                            value = json.loads(value)
                    else:
                        value = value_type(value)
                except Exception as e:
                    self.logger.error(f"Failed to cast config key '{env_key}' to {value_type}: {e}")
                    raise TypeError(f"Failed to cast config key '{env_key}' to {value_type}: {e}")

            self.cache[full_key] = value
            return value

    def _str_to_bool(self, value: Union[str, bool, int]) -> bool:
        """
        Converts a string, boolean, or integer to a boolean value.

        Args:
            value (Union[str, bool, int]): The input value.

        Returns:
            bool: The converted boolean.

        Raises:
            ValueError: If the value cannot be converted to bool.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str):
            if value.lower() in ['true', '1', 'yes', 'on']:
                return True
            elif value.lower() in ['false', '0', 'no', 'off']:
                return False
        raise ValueError(f"Cannot convert {value} to bool.")

    def _load_yaml(self, config_file: Path):
        """Loads configuration settings from a YAML file."""
        with open(config_file, 'r') as file:
            data = yaml.safe_load(file)
            if data:
                self._flatten_dict(data)
        self.logger.info(f"Loaded YAML config from {config_file}")

    def _load_json(self, config_file: Path):
        """Loads configuration settings from a JSON file."""
        with open(config_file, 'r') as file:
            data = json.load(file)
            if data:
                self._flatten_dict(data)
        self.logger.info(f"Loaded JSON config from {config_file}")

    def _load_toml(self, config_file: Path):
        """Loads configuration settings from a TOML file."""
        with open(config_file, 'r') as file:
            data = toml.load(file)
            if data:
                self._flatten_dict(data)
        self.logger.info(f"Loaded TOML config from {config_file}")

    def _load_env(self, env_path: Path):
        """Loads environment variables from a .env file."""
        try:
            load_dotenv(dotenv_path=env_path, override=True)
            self.logger.info(f"Loaded environment variables from {env_path}")
        except Exception as e:
            self.logger.error(f"Failed to load environment variables from {env_path}: {e}")

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.'):
        """
        Flattens a nested dictionary for easier key-based retrieval.

        For example, a dict like:
          {
            "database": {
              "user": "test_user"
            }
          }
        becomes:
          {
            "database.user": "test_user"
          }

        Args:
            d (Dict[str, Any]): Nested dictionary.
            parent_key (str, optional): Base key string for recursion.
            sep (str, optional): Separator between levels of hierarchy.
        """
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                self._flatten_dict(v, new_key, sep=sep)
            else:
                self.config[new_key.lower()] = v

    def check_missing_keys(self, required_keys: List[str]):
        """
        Ensures that all necessary configuration keys are present.

        Args:
            required_keys (List[str]): A list of required configuration keys.

        Raises:
            KeyError: If any required key is missing.
        """
        missing = []
        for key in required_keys:
            if self.get(key) is None:
                missing.append(key.upper())

        if missing:
            keys_str = ', '.join(missing)
            self.logger.error(f"Missing required configuration keys: {keys_str}")
            raise KeyError(f"Missing required configuration keys: {keys_str}")

    def get_all(self) -> Dict[str, Any]:
        """
        Returns all configurations as a dictionary, combining data from config files and environment variables.

        Returns:
            Dict[str, Any]: A dictionary of all loaded configurations in lowercase keys.
        """
        all_configs = dict(self.config)
        for k, v in os.environ.items():
            all_configs[k.lower()] = v
        return all_configs

    def get_db_url(self) -> str:
        """
        Constructs and returns the database URL based on the configuration values.

        Returns:
            str: The constructed database URL.
        """
        user = self.get('database.user', required=True)
        password = self.get('database.password', required=True)
        host = self.get('database.host', required=True)
        port = self.get('database.port', required=True)
        dbname = self.get('database.dbname', required=True)
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    def get_async_db_url(self) -> str:
        """
        Constructs and returns the asynchronous database URL.

        Returns:
            str: The constructed async database URL.
        """
        user = self.get('database.user', required=True)
        password = self.get('database.password', required=True)
        host = self.get('database.host', required=True)
        port = self.get('database.port', required=True)
        dbname = self.get('database.dbname', required=True)
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{dbname}"

    def reload_configurations(
        self,
        config_files: Optional[List[Path]] = None,
        env_file: Optional[Path] = None
    ):
        """
        Reloads configurations from specified files and environment variables.

        Args:
            config_files (Optional[List[Path]]): Additional config file paths to reload.
            env_file (Optional[Path]): Path to the .env file to reload.
        """
        with self.lock:
            if config_files:
                self.load_configurations(config_files)
            if env_file:
                self._load_env(env_file)

            # Clear cache to force re-fetching from updated sources
            self.cache.clear()

            # Re-check required keys
            self.check_missing_keys(self.required_keys)
            self.logger.info("Configurations reloaded successfully.")

    def list_configurations(self, mask_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Lists all loaded configurations, optionally masking specified keys.

        Args:
            mask_keys (Optional[List[str]]): Keys to mask in the output.

        Returns:
            Dict[str, Any]: Dictionary of configurations with masked values where applicable.
        """
        configs = self.get_all()
        if mask_keys:
            for mk in mask_keys:
                mk_lower = mk.lower()
                if mk_lower in configs:
                    configs[mk_lower] = "*****"
        return configs

    @staticmethod
    def setup_logger(log_name: str) -> logging.Logger:
        """
        Sets up a logger that writes logs to both the console and a dedicated file.

        Args:
            log_name (str): Name of the logger.

        Returns:
            logging.Logger: The configured logger.
        """
        logger = logging.getLogger(log_name)
        # Prevent multiple handlers if logger is requested repeatedly
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Determine project root for log path
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parents[2]
            log_dir = project_root / 'logs' / 'Utilities'
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_dir / f"{log_name}.log")
            file_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger
