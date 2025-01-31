# -------------------------------------------------------------------
# File Path: C:/TheTradingRobotPlug/Scripts/model_training/continuous_learning/trading_env.py
# Description: Custom trading environment with integrated risk management, 
#              data handling for backtesting, and training reinforcement learning models.
# -------------------------------------------------------------------

import numpy as np
from typing import Optional, Tuple, Dict, Any
import pandas as pd
from pathlib import Path
import sys
import logging

# Adjust paths to include utilities
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
utilities_dir = project_root / 'Scripts' / 'Utilities'

# Add the Utilities directory to sys.path
if utilities_dir.exists() and str(utilities_dir) not in sys.path:
    sys.path.append(str(utilities_dir))

# Import from your utilities
from utils_main import (
    logger,
    config_manager,
    data_store,
    fetch_and_store_stock_data_parallel,
    generate_model_performance_report,
)
from risk_management import RiskManager

class TradingEnv:
    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        risk_manager: Optional[RiskManager] = None,
    ):
        """
        Initialize the TradingEnv with stock data, initial balance, transaction cost, and an optional RiskManager.
        Integrates with data fetching utilities and DataStore for data handling.

        Parameters:
        - symbol (str): Stock symbol for which data is to be fetched.
        - start_date (str): Start date for historical data.
        - end_date (str): End date for historical data.
        - initial_balance (float): Starting balance for trading.
        - transaction_cost (float): Transaction cost percentage per trade.
        - risk_manager (Optional[RiskManager]): Risk management component to evaluate risk during trading.
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.risk_manager = risk_manager

        # Fetch data using DataStore and data fetching utilities
        self.data = self.fetch_stock_data()

        self.reset()

    def fetch_stock_data(self) -> pd.DataFrame:
        """
        Fetch stock data using DataStore. If data is not found, fetch and store using the configured API.

        Returns:
        - pd.DataFrame: The stock data fetched from DataStore or API.
        """
        try:
            # Try to load data from DataStore
            data = data_store.load_data(symbol=self.symbol, apply_indicators=False)
            if data is None or data.empty:
                logger.info(f"No data found for {self.symbol} in DataStore. Fetching data.")
                # Fetch and store data
                fetch_and_store_stock_data_parallel([self.symbol], self.start_date, self.end_date, api, logger)
                # Load data again
                data = data_store.load_data(symbol=self.symbol, apply_indicators=False)
                if data is None or data.empty:
                    raise ValueError(f"Data for {self.symbol} could not be fetched.")
            # Ensure data is sorted by date
            data.sort_values('date', inplace=True)
            data.reset_index(drop=True, inplace=True)
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            raise

    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state and return the initial observation.

        Returns:
        - np.ndarray: The initial state observation.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_reward = 0.0
        self.done = False
        self.price = self.data['close'].iloc[self.current_step]

        # Calculate portfolio value
        current_portfolio_value = self.balance + self.shares_held * self.price

        if self.risk_manager:
            self.risk_manager.initialize(current_portfolio_value)

        logger.info("Environment reset.")
        return self._get_observation()

    def step(self, action: Tuple[int, float]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute a trading step based on the given action and return the new state, reward, done flag, and info.

        Parameters:
        - action (Tuple[int, float]): Action tuple where the first element is the action type (0: Buy, 1: Sell, 2: Hold)
                                      and the second element is the fraction of the position to act on.

        Returns:
        - np.ndarray: New state observation.
        - float: Reward from the action.
        - bool: Flag indicating if the episode is done.
        - Dict[str, Any]: Additional information about the current state.
        """
        if self.done:
            raise RuntimeError("Cannot step in a finished environment. Please reset the environment.")

        self._take_action(action)

        # Update the current step and check if we've reached the end of the data
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True
            logger.info("Reached the end of the data. Episode done.")

        # Update the price for the new step if not done
        if not self.done:
            self.price = self.data['close'].iloc[self.current_step]

        # Calculate reward and update total reward
        reward = self._calculate_reward()
        self.total_reward += reward

        # Evaluate risk and possibly update the done flag
        if self.risk_manager:
            current_portfolio_value = self.balance + self.shares_held * self.price
            self.risk_manager.update(current_portfolio_value)
            if self._evaluate_risk(current_portfolio_value):
                self.done = True

        logger.debug(
            f"Step: {self.current_step}, Action: {action}, Reward: {reward}, "
            f"Balance: {self.balance}, Shares held: {self.shares_held}"
        )
        return self._get_observation(), reward, self.done, self._get_info()

    def _take_action(self, action: Tuple[int, float]) -> None:
        """
        Perform the specified action: buy, sell, or hold.

        Parameters:
        - action (Tuple[int, float]): The action tuple (action_type, amount).
        """
        action_type, amount = action
        if action_type == 0:
            self._buy_shares(amount)
        elif action_type == 1:
            self._sell_shares(amount)
        elif action_type == 2:
            logger.info("Hold action taken.")
        else:
            logger.warning(f"Invalid action type: {action_type}. Action ignored.")

    def _buy_shares(self, amount: float) -> None:
        """
        Buy shares based on the specified amount.

        Parameters:
        - amount (float): Fraction of balance to use for buying shares.
        """
        if amount <= 0 or amount > 1:
            logger.warning(f"Invalid buy amount: {amount}. Amount must be between 0 and 1.")
            return
        available_balance = self.balance * amount
        max_possible_shares = int(available_balance / (self.price * (1 + self.transaction_cost)))
        if max_possible_shares <= 0:
            logger.info("Insufficient balance to buy shares.")
            return
        cost = max_possible_shares * self.price * (1 + self.transaction_cost)
        self.balance -= cost
        self.shares_held += max_possible_shares
        logger.info(f"Bought {max_possible_shares} shares at price {self.price}, total cost: {cost}.")

    def _sell_shares(self, amount: float) -> None:
        """
        Sell shares based on the specified amount.

        Parameters:
        - amount (float): Fraction of shares held to sell.
        """
        if amount <= 0 or amount > 1:
            logger.warning(f"Invalid sell amount: {amount}. Amount must be between 0 and 1.")
            return
        shares_to_sell = int(self.shares_held * amount)
        if shares_to_sell <= 0:
            logger.info("No shares to sell.")
            return
        proceeds = shares_to_sell * self.price * (1 - self.transaction_cost)
        self.balance += proceeds
        self.shares_held -= shares_to_sell
        logger.info(f"Sold {shares_to_sell} shares at price {self.price}, total proceeds: {proceeds}.")

    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on the change in portfolio value.

        Returns:
        - float: The calculated reward.
        """
        portfolio_value = self.balance + self.shares_held * self.price
        reward = portfolio_value - (self.initial_balance + self.total_reward)
        return reward

    def _evaluate_risk(self, current_portfolio_value: float) -> bool:
        """
        Evaluate risk using the RiskManager and determine if trading should stop.

        Returns:
        - bool: True if trading should stop, otherwise False.
        """
        risk_status = self.risk_manager.check_risk(current_portfolio_value)
        if risk_status in ["STOP_TRADING", "STOP_LOSS", "TAKE_PROFIT"]:
            self._handle_risk_event(risk_status)
            return True
        return False

    def _handle_risk_event(self, risk_status: str) -> None:
        """
        Handle the event where a risk condition triggers a stop trading, stop loss, or take profit action.

        Parameters:
        - risk_status (str): The risk status returned by the RiskManager.
        """
        if risk_status in ["STOP_LOSS", "TAKE_PROFIT"]:
            if self.shares_held > 0:
                self._sell_shares(1.0)
        logger.warning(f"Risk event triggered: {risk_status}. Trading halted.")

    def _get_observation(self) -> np.ndarray:
        """
        Return the current state as an observation.

        Returns:
        - np.ndarray: The current state observation (balance, shares_held, price).
        """
        return np.array([self.balance, self.shares_held, self.price], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """
        Return additional information about the current state.

        Returns:
        - Dict[str, Any]: Dictionary containing additional state information.
        """
        return {
            'balance': self.balance,
            'price': self.price,
            'shares_held': self.shares_held,
            'portfolio_value': self.balance + self.shares_held * self.price,
        }

# Example usage
if __name__ == "__main__":
    # Initialize the trading environment
    env = TradingEnv(
        symbol="AAPL",
        start_date="2020-01-01",
        end_date="2020-12-31",
        initial_balance=10000.0,
        transaction_cost=0.001,
        risk_manager=RiskManager(),  # Initialize your RiskManager accordingly
    )

    # Reset the environment
    observation = env.reset()
    done = False

    # Run through the environment
    while not done:
        # Example action: Buy 10% of the possible shares
        action = (0, 0.1)
        observation, reward, done, info = env.step(action)
        print(f"Observation: {observation}, Reward: {reward}, Done: {done}, Info: {info}")

    # Generate a performance report after trading
    mse, rmse, mae, r2 = 0.25, 0.5, 0.3, 0.85  # Example metrics
    generate_model_performance_report(env.symbol, mse, rmse, mae, r2)
