# model_training/base/data_handler.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, SimpleImputer
from Utilities.shared_utils import setup_logging
from Utilities.config_manager import ConfigManager
from utils.paths import Paths
from Utilities.data.data_store import DataStore
from Utilities.data_processing.Technical_Indicators.indicator_calculator import IndicatorCalculator
import numpy as np
from typing import Any, Tuple

class DataHandler:
    """
    Handles data loading and preprocessing tasks.

    Attributes:
        config (ConfigManager): Configuration manager instance.
        logger (logging.Logger): Logger instance.
        data_store (DataStore): DataStore instance for data access.
        indicator_calculator (IndicatorCalculator): Indicator calculator instance.
    """

    def __init__(self, config: ConfigManager, logger: Any):
        """
        Initialize the DataHandler.

        Args:
            config (ConfigManager): Configuration manager instance.
            logger (logging.Logger): Logger instance.
        """
        self.config = config
        self.logger = logger
        self.data_store = DataStore(config, logger)
        self.indicator_calculator = IndicatorCalculator(logger=self.logger, data_store=self.data_store)

    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load data for a specific symbol.

        Args:
            symbol (str): Stock symbol.

        Returns:
            pd.DataFrame: Loaded data.
        """
        self.logger.info(f"Loading data for symbol: {symbol}")
        data = self.data_store.load_data(symbol)
        if data is None or data.empty:
            self.logger.error(f"No data found for symbol: {symbol}")
            return pd.DataFrame()
        return data

    def preprocess_data(self, data: pd.DataFrame, features: list, target: str) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Preprocess the data: apply indicators, handle missing values, scale features.

        Args:
            data (pd.DataFrame): Raw data.
            features (list): List of feature column names.
            target (str): Target column name.

        Returns:
            tuple: Scaled features, target, scaler instance.
        """
        if data is None or data.empty:
            self.logger.error("Received empty data for preprocessing.")
            return np.array([]), np.array([]), StandardScaler()

        self.logger.info("Applying technical indicators.")
        data = self.indicator_calculator.apply_all_indicators(data)
        
        self.logger.info("Handling missing values.")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.dropna(inplace=True)

        if any(col not in data.columns for col in features + [target]):
            self.logger.error("Some required features or target columns are missing after preprocessing.")
            return np.array([]), np.array([]), StandardScaler()

        self.logger.info("Splitting features and target.")
        X = data[features].values
        y = data[target].values

        self.logger.info("Scaling features.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y, scaler
