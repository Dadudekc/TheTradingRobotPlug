# File: column_utils.py
# Location: src/Utilities
# Description: Provides utilities for flattening, standardizing, and validating DataFrame columns.

import pandas as pd
import logging
import json
import os
from pathlib import Path
from typing import Optional

class ColumnUtils:
    """
    Utility class for handling column naming, flattening MultiIndex columns,
    and ensuring consistency across the trading pipeline.
    """
    
    DEFAULT_COLUMN_MAP = {
        'close_': 'close',
        'adj_close': 'adjclose',
        'macd_line': 'macd_line',
        'macd_signal': 'macd_signal',
        'macd_histogram': 'macd_histogram',
        'rsi': 'rsi',
        'bollinger_upper': 'bollinger_upper',
        'bollinger_lower': 'bollinger_lower',
        'bollinger_mid': 'bollinger_mid',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'volume': 'volume',
        'date': 'date'
    }

    CONFIG_PATH = Path(__file__).parent / 'column_config.json'

    @classmethod
    def load_column_mapping(cls, config_path):
        if not os.path.exists(config_path):
            # Log a warning and use default mappings
            print(f"Warning: Column configuration file not found at {config_path}. Using default mappings.")
            return {
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }
        
        with open(config_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flattens MultiIndex columns into single-level column names.

        Example:
        MultiIndex ('close', '') -> 'close'
        MultiIndex ('macd', 'signal') -> 'macd_signal'
        """
        if isinstance(df.columns, pd.MultiIndex):
            new_columns = ['_'.join(filter(None, map(str, col))).strip().lower() for col in df.columns.values]
            df.columns = new_columns
        else:
            df.columns = [col.lower() for col in df.columns]
        return df

    @staticmethod
    def standardize_columns(df: pd.DataFrame, column_map: dict, logger: logging.Logger) -> pd.DataFrame:
        """
        Standardizes column names based on a predefined mapping.

        Args:
            df (pd.DataFrame): The DataFrame whose columns need to be standardized.
            column_map (dict): Mapping from existing column names to standardized names.
            logger (logging.Logger): Logger instance for logging operations.

        Returns:
            pd.DataFrame: DataFrame with standardized column names.
        """
        df = ColumnUtils.flatten_columns(df)
        logger.debug(f"Columns after flattening: {df.columns.tolist()}")

        # Rename columns based on mapping
        df.rename(columns=column_map, inplace=True)
        logger.info(f"Columns after standardization: {df.columns.tolist()}")

        return df

    @staticmethod
    def validate_required_columns(df: pd.DataFrame, required_columns: list, logger: logging.Logger) -> bool:
        """
        Ensures the DataFrame contains the required columns.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            required_columns (list): List of required column names.
            logger (logging.Logger): Logger instance for logging operations.

        Returns:
            bool: True if all required columns exist, otherwise raises an error.
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise KeyError(f"Missing required columns: {missing_columns}")
        logger.debug("All required columns are present.")
        return True

    @classmethod
    def process_dataframe(
        cls, 
        df: pd.DataFrame, 
        config_path: Optional[Path] = None,
        required_columns: Optional[list] = None,
        logger: Optional[logging.Logger] = None
    ) -> pd.DataFrame:
        """
        Fully processes a DataFrame to:
        - Flatten MultiIndex columns
        - Standardize column names
        - Validate required columns

        Args:
            df (pd.DataFrame): The DataFrame to process.
            config_path (Path, optional): Path to the JSON config file.
            required_columns (list, optional): List of essential column names.
            logger (logging.Logger, optional): Logger instance.

        Returns:
            pd.DataFrame: Processed DataFrame with clean column names.
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        column_map = cls.load_column_mapping(config_path)
        df = cls.standardize_columns(df, column_map, logger)

        if required_columns:
            cls.validate_required_columns(df, required_columns, logger)

        return df
