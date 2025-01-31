# -------------------------------------------------------------------
# File Path: C:/TheTradingRobotPlug/Scripts/model_training/continuous_learning/risk_management.py
# Description: Risk management module to dynamically handle trading risk
#              parameters based on market conditions and real-time data.
# -------------------------------------------------------------------

import logging
from typing import Optional, Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import sys

# Adjust paths to include utilities
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
utilities_dir = project_root / 'Scripts' / 'Utilities'

# Add the Utilities directory to sys.path
if utilities_dir.exists() and str(utilities_dir) not in sys.path:
    sys.path.append(str(utilities_dir))

# Import from your utilities
from utils_main import logger, config_manager

class RiskManager:
    def __init__(self):
        """
        Initialize the RiskManager with risk parameters from the configuration.
        """
        # Load risk parameters from configuration
        if "RiskManagement" in config_manager.config:
            risk_config = config_manager.config["RiskManagement"]
            max_drawdown = float(risk_config.get("max_drawdown", fallback=0.2))
            stop_loss = float(risk_config.get("stop_loss", fallback=0.05))
            take_profit = float(risk_config.get("take_profit", fallback=0.1))
        else:
            # Default values if not specified in config
            max_drawdown = 0.2
            stop_loss = 0.05
            take_profit = 0.1

        self.set_risk_parameters(max_drawdown, stop_loss, take_profit)
        self.reset()
        logger.info("RiskManager instance created with parameters from configuration.")

    def initialize(self, portfolio_value: float, market_conditions: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize the RiskManager with the initial portfolio value and optional market conditions.

        Parameters:
        - portfolio_value (float): The initial portfolio value.
        - market_conditions (Optional[Dict[str, float]]): Optional market conditions to adjust parameters.
        """
        self.initial_balance = portfolio_value
        self.high_water_mark = portfolio_value
        self.low_water_mark = portfolio_value
        if market_conditions:
            self._adjust_parameters_for_market_conditions(market_conditions)
        logger.info(f"RiskManager initialized with portfolio value: {portfolio_value}")

    def update(self, current_portfolio_value: float, market_conditions: Optional[Dict[str, float]] = None) -> None:
        """
        Update the RiskManager with the current portfolio value and optional market conditions.

        Parameters:
        - current_portfolio_value (float): The current portfolio value.
        - market_conditions (Optional[Dict[str, float]]): Optional market conditions to adjust parameters.
        """
        self._ensure_initialized()
        self._update_water_marks(current_portfolio_value)

        if market_conditions:
            self._adjust_parameters_for_market_conditions(market_conditions)
        logger.info(f"RiskManager updated with portfolio value: {current_portfolio_value}")

    def check_risk(self, current_portfolio_value: float, market_conditions: Optional[Dict[str, float]] = None) -> str:
        """
        Check if the current trading conditions exceed the risk thresholds.

        Parameters:
        - current_portfolio_value (float): The current portfolio value.
        - market_conditions (Optional[Dict[str, float]]): Optional market conditions to adjust parameters.

        Returns:
        - str: Risk assessment: "STOP_TRADING", "STOP_LOSS", "TAKE_PROFIT", or "CONTINUE_TRADING".
        """
        self._ensure_initialized()

        if market_conditions:
            self._adjust_parameters_for_market_conditions(market_conditions)

        drawdown = self._calculate_drawdown(current_portfolio_value)
        if drawdown > self.max_drawdown:
            logger.warning("Drawdown exceeded max drawdown. STOP_TRADING.")
            return "STOP_TRADING"

        if drawdown >= self.stop_loss:
            logger.warning("Drawdown exceeded stop loss. STOP_LOSS.")
            return "STOP_LOSS"

        profit = self._calculate_profit(current_portfolio_value)
        if profit >= self.take_profit:
            logger.info("Profit target achieved. TAKE_PROFIT.")
            return "TAKE_PROFIT"

        logger.debug("Conditions normal. CONTINUE_TRADING.")
        return "CONTINUE_TRADING"

    def get_risk_parameters(self) -> Dict[str, float]:
        """
        Return the current risk parameters.

        Returns:
        - Dict[str, float]: A dictionary containing the current risk parameters.
        """
        return {
            "max_drawdown": self.max_drawdown,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit
        }

    def set_risk_parameters(self, max_drawdown: float, stop_loss: float, take_profit: float) -> None:
        """
        Set risk parameters for the trading strategy.

        Parameters:
        - max_drawdown (float): Maximum allowable drawdown.
        - stop_loss (float): Stop-loss threshold.
        - take_profit (float): Take-profit threshold.
        """
        self._validate_risk_parameters(max_drawdown, stop_loss, take_profit)
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        logger.info(f"Risk parameters set: max_drawdown={max_drawdown}, stop_loss={stop_loss}, take_profit={take_profit}")

    def reset(self) -> None:
        """
        Reset the RiskManager to its initial state.
        """
        self.initial_balance = None
        self.high_water_mark = None
        self.low_water_mark = None
        logger.info("RiskManager reset.")

    def optimize_parameters(self, historical_data: np.ndarray) -> None:
        """
        Optimize risk parameters based on historical data using linear regression.

        Parameters:
        - historical_data (np.ndarray): Array of historical balance data for analysis.
        """
        logger.info("Optimizing risk parameters using historical data...")
        X, y = self._prepare_historical_data(historical_data)
        predicted_trend = self._train_regression_model(X, y)
        self._set_optimized_parameters(predicted_trend)
        logger.info(f"Optimized risk parameters: max_drawdown={self.max_drawdown}, stop_loss={self.stop_loss}, take_profit={self.take_profit}")

    def _update_water_marks(self, current_portfolio_value: float) -> None:
        """
        Update the high and low water marks based on the current portfolio value.

        Parameters:
        - current_portfolio_value (float): The current portfolio value.
        """
        if current_portfolio_value > self.high_water_mark:
            self.high_water_mark = current_portfolio_value
        if current_portfolio_value < self.low_water_mark:
            self.low_water_mark = current_portfolio_value
        logger.debug(f"Updated water marks: High={self.high_water_mark}, Low={self.low_water_mark}")

    def _calculate_drawdown(self, current_portfolio_value: float) -> float:
        """
        Calculate the drawdown based on the current portfolio value.

        Parameters:
        - current_portfolio_value (float): The current portfolio value.

        Returns:
        - float: The calculated drawdown percentage.
        """
        drawdown = (self.high_water_mark - current_portfolio_value) / self.high_water_mark
        logger.debug(f"Calculated drawdown: {drawdown}")
        return drawdown

    def _calculate_profit(self, current_portfolio_value: float) -> float:
        """
        Calculate the profit based on the initial balance.

        Parameters:
        - current_portfolio_value (float): The current portfolio value.

        Returns:
        - float: The calculated profit percentage.
        """
        profit = (current_portfolio_value - self.initial_balance) / self.initial_balance
        logger.debug(f"Calculated profit: {profit}")
        return profit

    def _ensure_initialized(self) -> None:
        """
        Ensure that the RiskManager has been initialized before use.
        """
        if self.initial_balance is None or self.high_water_mark is None or self.low_water_mark is None:
            raise ValueError("RiskManager not initialized. Call 'initialize' with a portfolio value first.")

    def _adjust_parameters_for_market_conditions(self, market_conditions: Dict[str, float]) -> None:
        """
        Adjust risk parameters based on market conditions such as volatility and trend.

        Parameters:
        - market_conditions (Dict[str, float]): Market conditions affecting risk parameters.
        """
        volatility = market_conditions.get('volatility', 0.1)
        trend = market_conditions.get('trend', 0.1)
        self.max_drawdown = min(max(0.1, volatility * 0.2), 0.3)
        self.stop_loss = min(max(0.02, trend * 0.05), 0.1)
        self.take_profit = min(max(0.05, trend * 0.1), 0.2)
        logger.debug(f"Adjusted risk parameters for market conditions: max_drawdown={self.max_drawdown}, stop_loss={self.stop_loss}, take_profit={self.take_profit}")

    def _validate_risk_parameters(self, max_drawdown: float, stop_loss: float, take_profit: float) -> None:
        """
        Validate risk parameters to ensure they are within an acceptable range.

        Parameters:
        - max_drawdown (float): Maximum allowable drawdown.
        - stop_loss (float): Stop-loss threshold.
        - take_profit (float): Take-profit threshold.
        """
        if not (0 < max_drawdown < 1):
            raise ValueError("max_drawdown must be between 0 and 1.")
        if not (0 < stop_loss < 1):
            raise ValueError("stop_loss must be between 0 and 1.")
        if not (0 < take_profit < 1):
            raise ValueError("take_profit must be between 0 and 1.")

    def _prepare_historical_data(self, historical_data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Prepare historical data for regression analysis.

        Parameters:
        - historical_data (np.ndarray): Array of historical balance data.

        Returns:
        - (np.ndarray, np.ndarray): Tuple containing the features (X) and targets (y) for regression.
        """
        X = np.arange(len(historical_data)).reshape(-1, 1)
        y = historical_data
        return X, y

    def _train_regression_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Train a linear regression model on historical data.

        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target vector.

        Returns:
        - float: The predicted trend value from the regression model.
        """
        model = LinearRegression().fit(X, y)
        predicted_trend = model.coef_[0]
        logger.debug(f"Trained regression model with coefficient: {predicted_trend}")
        return predicted_trend

    def _set_optimized_parameters(self, predicted_trend: float) -> None:
        """
        Set risk parameters based on the predicted trend.

        Parameters:
        - predicted_trend (float): The trend predicted by the regression model.
        """
        self.max_drawdown = min(max(0.1, abs(predicted_trend) * 0.2), 0.3)
        self.stop_loss = min(max(0.02, abs(predicted_trend) * 0.05), 0.1)
        self.take_profit = min(max(0.05, abs(predicted_trend) * 0.1), 0.2)
        logger.debug(f"Optimized risk parameters based on predicted trend: max_drawdown={self.max_drawdown}, stop_loss={self.stop_loss}, take_profit={self.take_profit}")
