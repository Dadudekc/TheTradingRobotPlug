# File: indicator_unifier.py
# D:\TradingRobotPlug2\src\Utilities\data_processing\indicator_unifier.py
"""
Provides the AllIndicatorsUnifier class, which aggregates all technical
indicators (Volume, Volatility, Trend, Momentum, Machine Learning, Custom)
into a single pipeline for a DataFrame.
"""

import logging
import pandas as pd
from typing import Optional

from Utilities.config_manager import ConfigManager
from Utilities.data.data_store import DataStore

# Import each indicator module.
from Utilities.data_processing.Technical_Indicators.volume_indicators import VolumeIndicators
from Utilities.data_processing.Technical_Indicators.volatility_indicators import VolatilityIndicators
from Utilities.data_processing.Technical_Indicators.trend_indicators import TrendIndicators
from Utilities.data_processing.Technical_Indicators.momentum_indicators import MomentumIndicators
from Utilities.data_processing.Technical_Indicators.machine_learning_indicators import MachineLearningIndicators
from Utilities.data_processing.Technical_Indicators.custom_indicators import CustomIndicators

class AllIndicatorsUnifier:
    """
    Applies all technical indicators by sequentially invoking each indicator class.
    """
    def __init__(self, config_manager: ConfigManager, logger: Optional[logging.Logger] = None, use_csv: bool = False):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing AllIndicatorsUnifier")

        # Setup DataStore for data retrieval/saving.
        self.data_store = DataStore(config=config_manager, logger=self.logger, use_csv=use_csv)

        # Initialize each indicators class.
        self.volume_indicators = VolumeIndicators(logger=self.logger, data_store=self.data_store)
        self.volatility_indicators = VolatilityIndicators(logger=self.logger, data_store=self.data_store)
        self.trend_indicators = TrendIndicators(data_store=self.data_store, logger=self.logger)
        self.momentum_indicators = MomentumIndicators(data_store=self.data_store, logger=self.logger)
        self.ml_indicators = MachineLearningIndicators(data_store=self.data_store, db_handler=None, config=config_manager, logger=self.logger)
        self.custom_indicators = CustomIndicators(data_store=self.data_store, config_manager=config_manager, logger=self.logger)

        self.logger.info("AllIndicatorsUnifier initialization complete.")

    def apply_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sequentially applies all indicator groups to the DataFrame.
        """
        self.logger.info("Applying Volume Indicators...")
        df = self.volume_indicators.apply_indicators(df)

        self.logger.info("Applying Volatility Indicators...")
        df = self.volatility_indicators.apply_indicators(df)

        self.logger.info("Applying Trend Indicators...")
        df = self.trend_indicators.apply_indicators(df)

        self.logger.info("Applying Momentum Indicators...")
        df = self.momentum_indicators.apply_indicators(df)

        self.logger.info("Applying Machine Learning Indicators...")
        df = self.ml_indicators.apply_indicators(df)

        self.logger.info("Applying Custom Indicators...")
        df = self.custom_indicators.apply_indicators(df)

        self.logger.info("All indicators applied successfully.")
        return df
