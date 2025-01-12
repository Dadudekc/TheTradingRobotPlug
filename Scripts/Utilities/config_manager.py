# Scripts/Utilities/config_manager.py
# -------------------------------------------------------------------
# File Path: Scripts/Utilities/config_manager.py
# Description: Manages configuration and environment variables for the TradingRobotPlug project.
#              Supports loading from environment variables, .env files, YAML, JSON, and TOML files.
#              Provides type casting, validation, and dynamic reloading capabilities.
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

class ConfigManager:
    def __init__(self, 
                 config_files: Optional[List[Path]] = None, 
                 env_file: Optional[Path] = None,
                 required_keys: Optional[List[str]] = None, 
                 logger: Optional[logging.Logger] = None,
                 project_root: Optional[Path] = None):
        """
        Initializes the ConfigManager and loads configurations from various sources.

        Args:
            config_files (List[Path], optional): A list of configuration files to load (YAML, JSON, TOML).
            env_file (Path, optional): Path to the .env file. If not provided, attempts to load from project root.
            required_keys (List[str], optional): A list of required configuration keys to check for.
            logger (logging.Logger, optional): Logger object to log messages (optional).
            project_root (Path, optional): Explicit project root path. If not provided, auto-detects.
        """
        self.config = defaultdict(dict)  # For flattened configs
        self.logger = logger or self.setup_logger("ConfigManager")
        self.cache = {}  # Cache for config values
        self.lock = threading.Lock()  # For thread-safe operations

        # Determine the project root
        if project_root:
            self.project_root = project_root.resolve()
        else:
            script_dir = Path(__file__).resolve().parent
            self.project_root = script_dir.parents[2]  # Adjust based on your project structure

        # Determine the .env file path
        env_path = env_file or (self.project_root / '.env')
        if env_path.exists():
            self._load_env(env_path)
        else:
            self.logger.warning(f"No .env file found at {env_path}. Environment variables will be used as-is.")

        # Load configurations from specified config files
        if config_files:
            self.load_configurations(config_files)

        # Check for missing required keys if any are provided
        self.required_keys = required_keys or []
        self.check_missing_keys(self.required_keys)

    def load_configurations(self, config_files: List[Path]):
        """
        Loads configuration data from various file formats.

        Args:
            config_files (List[Path]): List of configuration file paths.
        """
        for config_file in config_files:
            file_ext = config_file.suffix.lower()
            if not config_file.exists():
                self.logger.warning(f"Config file does not exist: {config_file}")
                continue
            if file_ext in ['.yaml', '.yml']:
                self._load_yaml(config_file)
            elif file_ext == '.json':
                self._load_json(config_file)
            elif file_ext == '.toml':
                self._load_toml(config_file)
            else:
                self.logger.warning(f"Unsupported config file format: {config_file}")

    def get(self, key: str, 
            default: Any = None, 
            fallback: Any = None,
            required: bool = False, 
            value_type: Optional[Type] = None) -> Any:
        """
        Retrieves a configuration value based on the provided key, with optional type casting.

        Args:
            key (str): The key to retrieve.
            default (Any, optional): Default value if the key is not found.
            fallback (Any, optional): Optional fallback if the key is not found in config or env.
            required (bool, optional): If True, raises an error if the key is missing.
            value_type (Type, optional): The expected type of the configuration value. If provided, casts the value.

        Returns:
            Any: The retrieved configuration value, optionally cast to the specified type.

        Raises:
            ValueError: If the key is missing and no default is provided.
            TypeError: If type casting fails.
        """
        with self.lock:
            full_key = key.lower()

            # Check the cache first
            if full_key in self.cache:
                return self.cache[full_key]

            # Attempt to retrieve the value from environment variables
            # Replace dots with underscores to match environment variable naming
            env_key = full_key.upper().replace('.', '_')
            value = os.getenv(env_key)

            # If still None, check the loaded config
            if value is None:
                value = self.config.get(full_key)

            # Apply default or fallback if value is None
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

            # Type casting if necessary
            if value is not None and value_type is not None:
                try:
                    if value_type == bool:
                        # Special handling for boolean types
                        value = self._str_to_bool(value)
                    elif value_type == list:
                        if isinstance(value, str):
                            value = [item.strip() for item in value.split(',')]
                        elif isinstance(value, list):
                            pass  # already a list
                        else:
                            raise TypeError(f"Cannot cast type {type(value)} to list.")
                    elif value_type == dict:
                        if isinstance(value, str):
                            value = json.loads(value)
                        elif isinstance(value, dict):
                            pass  # already a dict
                        else:
                            raise TypeError(f"Cannot cast type {type(value)} to dict.")
                    else:
                        value = value_type(value)
                except Exception as e:
                    self.logger.error(f"Failed to cast configuration key '{env_key}' to {value_type}: {e}")
                    raise TypeError(f"Failed to cast configuration key '{env_key}' to {value_type}: {e}")

            # Cache the result for future calls
            self.cache[full_key] = value
            return value

    def _str_to_bool(self, value: Union[str, bool, int]) -> bool:
        """
        Converts a string or integer to a boolean.

        Args:
            value (Union[str, bool, int]): The value to convert.

        Returns:
            bool: The converted boolean value.

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
        try:
            with open(config_file, 'r') as file:
                yaml_config = yaml.safe_load(file)
                if yaml_config:
                    self._flatten_dict(yaml_config)
            self.logger.info(f"Loaded YAML config from {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load YAML config from {config_file}: {e}")

    def _load_json(self, config_file: Path):
        """Loads configuration settings from a JSON file."""
        try:
            with open(config_file, 'r') as file:
                json_config = json.load(file)
                if json_config:
                    self._flatten_dict(json_config)
            self.logger.info(f"Loaded JSON config from {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load JSON config from {config_file}: {e}")

    def _load_toml(self, config_file: Path):
        """Loads configuration settings from a TOML file."""
        try:
            with open(config_file, 'r') as file:
                toml_config = toml.load(file)
                if toml_config:
                    self._flatten_dict(toml_config)
            self.logger.info(f"Loaded TOML config from {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load TOML config from {config_file}: {e}")

    def _load_env(self, env_path: Path):
        """Loads environment variables from a .env file."""
        try:
            load_dotenv(dotenv_path=env_path)
            self.logger.info(f"Loaded environment variables from {env_path}")
        except Exception as e:
            self.logger.error(f"Failed to load environment variables from {env_path}: {e}")

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flattens a nested dictionary for easier key-based retrieval.

        Args:
            d (Dict[str, Any]): The dictionary to flatten.
            parent_key (str, optional): The base key string.
            sep (str, optional): Separator between keys.

        Returns:
            Dict[str, Any]: The flattened dictionary.
        """
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            else:
                self.config[new_key.lower()] = v
        return items

    def check_missing_keys(self, required_keys: List[str]):
        """
        Ensures that all necessary configuration keys are present.

        Args:
            required_keys (List[str]): List of required configuration keys.

        Raises:
            KeyError: If any required key is missing.
        """
        missing_keys = []
        for key in required_keys:
            value = self.get(key)
            if value is None:
                missing_keys.append(key.upper())

        if missing_keys:
            missing_keys_str = ', '.join(missing_keys)
            self.logger.error(f"Missing required configuration keys: {missing_keys_str}")
            raise KeyError(f"Missing required configuration keys: {missing_keys_str}")

    def get_all(self) -> Dict[str, Any]:
        """
        Returns all configurations as a dictionary, combining data from config files and environment variables.

        Returns:
            Dict[str, Any]: Combined configuration data.
        """
        all_configs = dict(self.config)
        for key, value in os.environ.items():
            all_configs[key.lower()] = value
        return all_configs

    def get_db_url(self) -> str:
        """
        Constructs and returns the database URL based on the configuration values.

        Returns:
            str: The constructed database URL.
        """
        db_user = self.get('database.user', required=True)
        db_password = self.get('database.password', required=True)
        db_host = self.get('database.host', required=True)
        db_port = self.get('database.port', required=True)
        db_name = self.get('database.dbname', required=True)

        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        return db_url

    def get_async_db_url(self) -> str:
        """
        Constructs and returns the asynchronous database URL.

        Returns:
            str: The constructed asynchronous database URL.
        """
        db_user = self.get('database.user', required=True)
        db_password = self.get('database.password', required=True)
        db_host = self.get('database.host', required=True)
        db_port = self.get('database.port', required=True)
        db_name = self.get('database.dbname', required=True)

        async_db_url = f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        return async_db_url

    def reload_configurations(self, config_files: Optional[List[Path]] = None, env_file: Optional[Path] = None):
        """
        Reloads configurations from specified files and environment variables.

        Args:
            config_files (Optional[List[Path]]): List of configuration file paths to reload.
            env_file (Optional[Path]): Path to the .env file to reload.
        """
        with self.lock:
            if config_files:
                self.load_configurations(config_files)
            if env_file:
                self._load_env(env_file)
            self.cache.clear()
            self.check_missing_keys(self.required_keys)
            self.logger.info("Configurations reloaded successfully.")

    def list_configurations(self, mask_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Lists all loaded configurations, optionally masking sensitive keys.

        Args:
            mask_keys (Optional[List[str]]): List of keys to mask in the output.

        Returns:
            Dict[str, Any]: Dictionary of all configurations with masked values where applicable.
        """
        all_configs = self.get_all()
        if mask_keys:
            for key in mask_keys:
                key_lower = key.lower()
                if key_lower in all_configs:
                    all_configs[key_lower] = "*****"  # Mask sensitive information
        return all_configs

    @staticmethod
    def setup_logger(log_name: str) -> logging.Logger:
        """
        Sets up a logger that writes to both the console and a file.

        Args:
            log_name (str): Name of the logger.

        Returns:
            logging.Logger: Configured logger.
        """
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # Create file handler
            # Adjust the log directory as needed
            project_root = Path(__file__).resolve().parent.parents[2]
            log_dir = project_root / 'logs' / 'Utilities'
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{log_name}.log")
            file_handler.setLevel(logging.DEBUG)

            # Create formatters and add them to the handlers
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)

            # Add handlers to the logger
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        return logger
