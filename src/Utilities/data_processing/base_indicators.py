# base_indicators.py
# D:\TradingRobotPlug2\src\Utilities\data_processing\base_indicators.py
"""
base_indicators.py
------------------
Provides base classes and shared functionalities for all indicators.
"""


import logging
from typing import Optional
import pandas as pd

class BaseIndicator:
    """
    Abstract base class for all technical indicators.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply indicators to the DataFrame.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement apply_indicators method")