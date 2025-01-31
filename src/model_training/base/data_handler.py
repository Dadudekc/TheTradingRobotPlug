# model_training/base/data_handler.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, SimpleImputer
from utils.logger_manager import LoggerManager
from utils.config_manager import ConfigManager
from utils.paths import Paths
from data.data_store import DataStore
from data.indicators import apply_all_indicators
import numpy as np

class DataHandler:
    """
    Handles data loading and preprocessing tasks.

    Attributes:
        config (ConfigManager): Configuration manager instance.
        logger (logging.Logger): Logger instance.
        data_store (DataStore): DataStore instance for data access.
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

    def preprocess_data(self, data: pd.DataFrame, features: list, target: str):
        """
        Preprocess the data: apply indicators, handle missing values, scale features.

        Args:
            data (pd.DataFrame): Raw data.
            features (list): List of feature column names.
            target (str): Target column name.

        Returns:
            tuple: Scaled features, target, scaler instance.
        """
        self.logger.info("Applying technical indicators.")
        data = apply_all_indicators(data)
        
        self.logger.info("Handling missing values.")
        data = data.replace([np.inf, -np.inf], np.nan)
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)
        data.dropna(inplace=True)

        self.logger.info("Splitting features and target.")
        X = data[features]
        y = data[target]

        self.logger.info("Scaling features.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y.values, scaler
