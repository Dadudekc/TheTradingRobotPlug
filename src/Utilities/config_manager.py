# -------------------------------------------------------------------
# File Path: src/Utilities/config_manager.py
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
from typing import Any, Optional, List, Dict, Union, Type, Callable
import threading
from src.Utilities.shared_utils import setup_logging

class ConfigManager:
    """
    A manager that loads configurations from YAML, JSON, TOML files and .env files,
    allows environment variables to override config values, and provides methods for
    reloading, listing, and validating configuration keys.

    By default, environment variables override config dictionary values. However, if
    `config_overrides_env=True` is set, reloaded config keys take precedence after reload.

    By default, environment keys from .env or OS can fulfill 'required=True'. If
    `strict_keys=True` is set, required keys must exist in config dictionary or OS environment
    explicitly (i.e., ignoring .env).
    """

    # Mapping of file extensions to their respective loader functions
    _CONFIG_LOADERS: Dict[str, Callable[[Path], Dict[str, Any]]] = {
        '.yaml': lambda path: yaml.safe_load(path.read_text()) or {},
        '.yml': lambda path: yaml.safe_load(path.read_text()) or {},
        '.json': lambda path: json.loads(path.read_text()) or {},
        '.toml': lambda path: toml.loads(path.read_text()) or {},
    }

    def __init__(
        self,
        config_files: Optional[List[Path]] = None,
        env_file: Optional[Path] = None,
        required_keys: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        project_root: Optional[Path] = None,
        config_overrides_env: bool = False,
        strict_keys: bool = False
    ):
        """
        Initializes the ConfigManager and loads configurations from various sources.

        Args:
            config_files (List[Path], optional): A list of configuration files to load (YAML, JSON, TOML).
            env_file (Path, optional): Path to the .env file. If not provided, attempts to load from project root.
            required_keys (List[str], optional): A list of required configuration keys to check for.
            logger (logging.Logger, optional): Logger object to log messages (optional).
            project_root (Path, optional): Explicit project root path. If not provided, auto-detects.
            config_overrides_env (bool): If True, config dictionary overrides environment variables after reload.
            strict_keys (bool): If True, environment variables from .env are ignored when checking `required=True`.
        """
        self.config = defaultdict(dict)
        self.logger = logger or self.setup_logger("ConfigManager")
        self.cache = {}
        self.lock = threading.Lock()

        self.config_overrides_env = config_overrides_env
        self.strict_keys = strict_keys

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
            loader = self._CONFIG_LOADERS.get(file_ext)

            if not loader:
                self.logger.warning(f"Unsupported config file format: {config_file}")
                continue

            try:
                data = loader(config_file)
                if data:
                    flattened = self._flatten_dict(data)
                    self.config.update(flattened)
                self.logger.info(f"Loaded {file_ext.upper()} config from {config_file}")
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

            # Attempt to retrieve from environment (OS) first
            env_key = full_key.upper().replace('.', '_')
            os_env_value = os.environ.get(env_key)

            # If strict_keys=True, then we do NOT accept .env-based environment overrides for required checks
            # i.e., if the OS environment lacks the variable, we treat it as missing.
            # Otherwise, we also consider the loaded .env overrides in `os.environ`.
            env_has_key = (os_env_value is not None)

            # Retrieve from config dictionary
            config_value = self.config.get(full_key)

            # Decide final value
            if self.config_overrides_env:
                # If config has a value, override environment
                value = config_value if config_value is not None else os_env_value
            else:
                # Environment has priority
                value = os_env_value if os_env_value is not None else config_value

            # If still None, use default or fallback
            if value is None:
                if default is not None:
                    value = default
                elif fallback is not None:
                    value = fallback
                elif required:
                    # Handle strict_keys logic
                    if self.strict_keys:
                        # Check if key exists explicitly in the OS environment or configuration dictionary
                        if not env_has_key and config_value is None:
                            self.logger.error(f"Configuration for '{env_key}' is required but not provided.")
                            raise ValueError(f"Configuration for '{env_key}' is required but not provided.")
                    else:
                        # .env or OS environment fulfills the requirement
                        self.logger.error(f"Configuration for '{env_key}' is required but not provided.")
                        raise ValueError(f"Configuration for '{env_key}' is required but not provided.")
                else:
                    value = None

            # Type casting if needed
            if value is not None and value_type is not None:
                try:
                    value = self._cast_value(env_key, value, value_type)
                except Exception as e:
                    self.logger.error(f"Failed to cast config key '{env_key}' to {value_type}: {e}")
                    raise TypeError(f"Failed to cast config key '{env_key}' to {value_type}: {e}")

            self.cache[full_key] = value
            return value

    def _cast_value(self, env_key: str, value: Any, value_type: Type) -> Any:
        """
        Casts the value to the specified type.

        Args:
            env_key (str): The environment key for logging purposes.
            value (Any): The value to cast.
            value_type (Type): The target type.

        Returns:
            Any: The casted value.

        Raises:
            ValueError: If conversion to bool fails.
            TypeError: If other type conversions fail.
        """
        if value_type == bool:
            return self._str_to_bool(value)
        elif value_type == list:
            if isinstance(value, str):
                return [item.strip() for item in value.split(',')]
            elif isinstance(value, (list, tuple)):
                return list(value)
            else:
                raise TypeError(f"Cannot cast type {type(value)} to list.")
        elif value_type == dict:
            if isinstance(value, str):
                return json.loads(value)
            elif isinstance(value, dict):
                return value
            else:
                raise TypeError(f"Cannot cast type {type(value)} to dict.")
        else:
            return value_type(value)

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

    def _load_env(self, env_path: Path):
        """Loads environment variables from a .env file, overriding existing ones."""
        try:
            load_dotenv(dotenv_path=env_path, override=True)
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
                items[new_key.lower()] = v
        return items

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

        Environment variables take precedence unless `config_overrides_env=True`.

        Returns:
            Dict[str, Any]: Combined configuration data with environment variables or config dict taking precedence.
        """
        # Start with flattened config
        combined = dict(self.config)

        # Merge environment, but if config_overrides_env is True, only fill in missing
        for key, value in os.environ.items():
            dot_key = key.lower().replace('_', '.')
            if self.config_overrides_env:
                if dot_key not in combined:
                    combined[dot_key] = value
            else:
                combined[dot_key] = value

        return combined

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

    def reload_configurations(self, config_files: Optional[List[Path]] = None, env_file: Optional[Path] = None):
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

            self.cache.clear()
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
        all_configs = self.get_all()
        masked_configs = {}

        if mask_keys:
            mask_keys_lower = {key.lower() for key in mask_keys}
            for key, value in all_configs.items():
                if key in mask_keys_lower:
                    masked_configs[key] = "*****"
                else:
                    masked_configs[key] = value
        else:
            masked_configs = all_configs

        return masked_configs

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
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

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
