# -------------------------------------------------------------------
# File Name: trading_functions.py
# Location: C:/TheTradingRobotPlug/Scripts/model_training/continuous_learning/trading_functions.py
# Description: This module contains the core trading functions used for backtesting, model training,
#              evaluation, and performance analysis within the TradingRobotPlug project.
# -------------------------------------------------------------------

import os
import sys
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
from typing import Any, Dict, Tuple

# Workaround for OpenMP runtime issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingFunctions:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        model_path: str = 'ppo_trading_model',
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        stop_loss: float = 0.05,
        take_profit: float = 0.1,
        data_store=None,
        config_manager=None
    ):
        """
        Initialize the TradingFunctions class with the given parameters.

        Parameters:
        - ticker (str): Stock ticker symbol.
        - start_date (str): Start date for fetching historical data.
        - end_date (str): End date for fetching historical data.
        - model_path (str): Path where the trained model will be saved.
        - initial_balance (float): Starting balance for trading.
        - transaction_cost (float): Transaction cost per trade as a percentage.
        - stop_loss (float): Stop-loss threshold as a percentage.
        - take_profit (float): Take-profit threshold as a percentage.
        - data_store (DataStore): Instance of DataStore for managing data access.
        - config_manager (ConfigManager): Instance of ConfigManager for configuration management.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model_path = model_path
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.data_store = data_store
        self.config_manager = config_manager

        # Fetch the stock data
        self.data = self.fetch_stock_data()

    def fetch_stock_data(self) -> pd.DataFrame:
        """
        Fetches stock data for the specified ticker and date range.

        Returns:
        - pd.DataFrame: Historical stock data with technical indicators.
        """
        try:
            logger.info(f"Fetching stock data for {self.ticker} from {self.start_date} to {self.end_date}")
            data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            if data.empty:
                raise ValueError(f"No data fetched for {self.ticker}.")
            # Feature Engineering: Add technical indicators
            data['MA10'] = data['Close'].rolling(window=10).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            data = data.dropna().reset_index()
            logger.info(f"Successfully fetched and processed stock data for {self.ticker}")
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            raise

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculates the Relative Strength Index (RSI) for a series of prices.

        Parameters:
        - prices (pd.Series): Series of prices.
        - window (int): Window size for RSI calculation.

        Returns:
        - pd.Series: RSI values.
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def create_trading_environment(self) -> gym.Env:
        """
        Creates and returns a custom trading environment.

        Returns:
        - gym.Env: A custom trading environment for reinforcement learning.
        """
        class CustomTradingEnv(gym.Env):
            metadata = {'render.modes': ['human']}

            def __init__(self, df: pd.DataFrame, initial_balance: float, transaction_cost: float,
                         stop_loss: float, take_profit: float):
                super(CustomTradingEnv, self).__init__()
                self.df = df
                self.initial_balance = initial_balance
                self.transaction_cost = transaction_cost
                self.stop_loss = stop_loss
                self.take_profit = take_profit

                # Actions: 0 = Sell, 1 = Hold, 2 = Buy
                self.action_space = spaces.Discrete(3)

                # Observations: [Cash Balance, Stock Holdings, Price, MA10, MA50, RSI]
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6,),
                    dtype=np.float32
                )

                self.reset()

            def reset(self) -> np.ndarray:
                self.current_step = 0
                self.cash_balance = self.initial_balance
                self.stock_holdings = 0
                self.total_shares = 0
                self.max_portfolio_value = self.initial_balance
                self.done = False
                return self._next_observation()

            def _next_observation(self) -> np.ndarray:
                row = self.df.iloc[self.current_step]
                obs = np.array([
                    self.cash_balance,
                    self.stock_holdings,
                    row['Close'],
                    row['MA10'],
                    row['MA50'],
                    row['RSI']
                ], dtype=np.float32)
                return obs

            def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
                prev_portfolio_value = self._calculate_portfolio_value()
                self._take_action(action)
                self.current_step += 1

                if self.current_step >= len(self.df) - 1:
                    self.done = True

                current_portfolio_value = self._calculate_portfolio_value()
                reward = current_portfolio_value - prev_portfolio_value

                # Update max portfolio value
                if current_portfolio_value > self.max_portfolio_value:
                    self.max_portfolio_value = current_portfolio_value

                # Check for stop loss
                drawdown = (self.max_portfolio_value - current_portfolio_value) / self.max_portfolio_value
                if drawdown >= self.stop_loss:
                    logger.info("Stop loss triggered.")
                    self.done = True

                # Check for take profit
                profit = (current_portfolio_value - self.initial_balance) / self.initial_balance
                if profit >= self.take_profit:
                    logger.info("Take profit target achieved.")
                    self.done = True

                obs = self._next_observation()
                info = {
                    'portfolio_value': current_portfolio_value,
                    'cash_balance': self.cash_balance,
                    'stock_holdings': self.stock_holdings
                }

                return obs, reward, self.done, info

            def _take_action(self, action: int) -> None:
                current_price = self.df.iloc[self.current_step]['Close']
                if action == 0:  # Sell
                    if self.stock_holdings > 0:
                        sell_value = self.stock_holdings * current_price * (1 - self.transaction_cost)
                        self.cash_balance += sell_value
                        logger.debug(f"Selling {self.stock_holdings} shares at {current_price}, total: {sell_value}")
                        self.stock_holdings = 0
                elif action == 2:  # Buy
                    max_shares = self.cash_balance // (current_price * (1 + self.transaction_cost))
                    if max_shares > 0:
                        buy_cost = max_shares * current_price * (1 + self.transaction_cost)
                        self.cash_balance -= buy_cost
                        self.stock_holdings += max_shares
                        self.total_shares += max_shares
                        logger.debug(f"Buying {max_shares} shares at {current_price}, total: {buy_cost}")

                # Action 1 is Hold, do nothing

            def _calculate_portfolio_value(self) -> float:
                current_price = self.df.iloc[self.current_step]['Close']
                return self.cash_balance + self.stock_holdings * current_price

            def render(self, mode='human') -> None:
                portfolio_value = self._calculate_portfolio_value()
                print(f"Step: {self.current_step}")
                print(f"Portfolio Value: {portfolio_value}")
                print(f"Cash Balance: {self.cash_balance}")
                print(f"Stock Holdings: {self.stock_holdings}")

        logger.info("Creating a custom trading environment")
        env = CustomTradingEnv(
            df=self.data,
            initial_balance=self.initial_balance,
            transaction_cost=self.transaction_cost,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit
        )
        return env

    def train_model(self, total_timesteps: int = 10000) -> None:
        """
        Trains the trading model using PPO.

        Parameters:
        - total_timesteps (int): Number of timesteps to train the model for.
        """
        try:
            env = DummyVecEnv([self.create_trading_environment])
            model = PPO("MlpPolicy", env, verbose=1)
            logger.info("Starting model training")
            model.learn(total_timesteps=total_timesteps)
            model.save(self.model_path)
            logger.info(f"Model training complete and saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def backtest_model(self) -> Tuple[float, float, float, float]:
        """
        Perform backtesting using the trained model.

        Returns:
        - total_reward (float): Total reward earned during backtesting.
        - final_portfolio_value (float): Final portfolio value after backtesting.
        - mfe (float): Maximum Favorable Excursion (MFE).
        - mae (float): Maximum Adverse Excursion (MAE).
        """
        try:
            model = PPO.load(self.model_path)
            env = self.create_trading_environment()
            obs = env.reset()
            total_reward = 0
            done = False
            portfolio_values = []
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                portfolio_values.append(info['portfolio_value'])

            final_portfolio_value = portfolio_values[-1]
            mfe = max(portfolio_values) - self.initial_balance
            mae = min(portfolio_values) - self.initial_balance

            logger.info(f"Backtesting completed: Total Reward: {total_reward}, Final Portfolio Value: {final_portfolio_value}, MFE: {mfe}, MAE: {mae}")

            # Plot backtest results
            self.plot_backtest_results(portfolio_values)

            return total_reward, final_portfolio_value, mfe, mae
        except Exception as e:
            logger.error(f"Error during backtesting: {e}")
            raise

    def plot_backtest_results(self, portfolio_values: list) -> None:
        """
        Plots the backtest results, including portfolio value over time.

        Parameters:
        - portfolio_values (list): List of portfolio values over time.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values, label='Portfolio Value')
        plt.title('Backtest Portfolio Value Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Additional methods can be added as needed

# Example usage
if __name__ == "__main__":
    try:
        trading_model = TradingFunctions(
            ticker="AAPL",
            start_date="2022-01-01",
            end_date="2023-01-01",
            initial_balance=10000.0,
            transaction_cost=0.001,
            stop_loss=0.1,
            take_profit=0.2
        )
        trading_model.train_model(total_timesteps=5000)
        total_reward, final_portfolio_value, mfe, mae = trading_model.backtest_model()
        logger.info(f"Final Total Reward: {total_reward}, Final Portfolio Value: {final_portfolio_value}, MFE: {mfe}, MAE: {mae}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
