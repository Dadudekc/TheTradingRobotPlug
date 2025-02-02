# -------------------------------------------------------------------
# File: column_utils.py
# Location: src/Utilities
# Description: Provides utilities for flattening, standardizing, and validating DataFrame columns.
# -------------------------------------------------------------------

import pandas as pd
import logging
import json
import os
from pathlib import Path
from typing import Optional, Dict, List
from Utilities.shared_utils import setup_logging  # Ensure logging is unified

class ColumnUtils:
    """
    Utility class for handling column naming, flattening MultiIndex columns,
    and ensuring consistency across the trading pipeline.
    """

    DEFAULT_CONFIG_PATH_PRE = Path(__file__).parent / 'column_utils_mappings.json'
    DEFAULT_CONFIG_PATH_POST = Path(__file__).parent / 'final_column_mappings.json'

    def __init__(self, config_path: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize ColumnUtils with a configuration path for column mappings.

        Args:
            config_path (Path, optional): Path to the JSON config file.
            logger (logging.Logger, optional): Logger instance.
        """
        self.config_path = config_path if config_path else self.DEFAULT_CONFIG_PATH_PRE
        self.logger = logger or setup_logging("column_utils")

        # Load configuration
        self.column_mappings = self._load_column_mapping()
        self.logger.info("ColumnUtils initialized.")

    def _load_column_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Loads the column mapping configuration file.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary containing 'standard_columns' and 'required_columns'.
        """
        if not self.config_path.exists():
            self.logger.warning(f"Column config file not found at {self.config_path}. Using default mappings.")
            return {
                "standard_columns": {
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume"
                },
                "required_columns": [
                    "date", "open", "high", "low", "close", "volume"
                ]
            }

        try:
            with open(self.config_path, 'r') as file:
                config = json.load(file)
                self.logger.info(f"Loaded column configuration from {self.config_path}")
                return config
        except Exception as e:
            self.logger.error(f"Error loading column configuration: {e}", exc_info=True)
            return {}

    def flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flattens MultiIndex columns into single-level column names.
        """
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(filter(None, map(str, col))).strip().lower() for col in df.columns.values]
        else:
            df.columns = [col.lower() for col in df.columns]
        
        self.logger.debug(f"Flattened columns: {df.columns.tolist()}")
        return df

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes column names based on the loaded configuration.
        """
        df = self.flatten_columns(df)
        column_map = self.column_mappings.get("standard_columns", {})
        
        # Rename columns based on mapping
        df.rename(columns=column_map, inplace=True)
        self.logger.info(f"Standardized columns: {df.columns.tolist()}")

        return df

    def validate_required_columns(self, df: pd.DataFrame) -> bool:
        """
        Ensures the DataFrame contains the required columns.
        """
        print(f"[column_utils] Existing DataFrame columns: {list(df.columns)}")
        required_columns = self.column_mappings.get("required_columns", [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            raise KeyError(f"Missing required columns: {missing_columns}")
        
        self.logger.debug("All required columns are present.")
        return True

    def process_dataframe(self, df: pd.DataFrame, stage: str = "pre") -> pd.DataFrame:
        """
        Fully processes a DataFrame by:
        - Flattening MultiIndex columns
        - Standardizing column names
        - Validating required columns

        Args:
            df (pd.DataFrame): The DataFrame to process.
            stage (str): "pre" for preprocessing or "post" after indicators are applied.

        Returns:
            pd.DataFrame: Processed DataFrame with clean column names.
        """
        if stage == "pre":
            self.config_path = self.DEFAULT_CONFIG_PATH_PRE
        else:
            self.config_path = self.DEFAULT_CONFIG_PATH_POST
        
        self.column_mappings = self._load_column_mapping()
        df = self.standardize_columns(df)
        df = self.flatten_columns(df)
        df = self.validate_required_columns(df)

        self.logger.info(f"DataFrame processing completed for {stage} stage.")
        return df

# Example Usage
column_utils = ColumnUtils()

df = pd.DataFrame({
    "Date": ["2024-02-01", "2024-02-02"],
    "Open": [150, 152],
    "High": [155, 157],
    "Low": [149, 151],
    "Close": [153, 156],
    "Volume": [100000, 120000]
})

processed_df = column_utils.process_dataframe(df, stage="pre")
print(processed_df)