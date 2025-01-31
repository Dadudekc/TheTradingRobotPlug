# -------------------------------------------------------------------
# File: __init__.py
# Location: src/Utilities/strategies
# Description: Initializes the strategies package.
# -------------------------------------------------------------------

from .moving_averages import MovingAverages
from .trading_strategies import TradingStrategies
from .backtester import Backtester

__all__ = [
    "MovingAverages",
    "TradingStrategies",
    "Backtester"
]
