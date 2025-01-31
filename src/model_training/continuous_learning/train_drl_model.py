# File Name: train_drl_model.py
# Location: C:\TheTradingRobotPlug\Scripts\model_training\continuous_learning\train_drl_model.py
# Description: This module contains the class to train a DRL model using the PPO algorithm in the TradingRobotPlug project.

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import yfinance as yf
import gym
from gym import spaces

# C:\TheTradingRobotPlug\Scripts\model_training\continuous_learning\continuous_learning.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import yfinance as yf
import gym
from gym import spaces
import logging
import sys
from pathlib import Path  # Corrected import
import json

# Dynamic project structure setup
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
utilities_dir = project_root / 'Scripts' / 'Utilities'
datafetch_dir = project_root / 'Scripts' / 'Data_Fetchers'
technical_indicators_dir = project_root / 'Scripts' / 'Data_Processing' / 'Technical_Indicators'

sys.path.append(str(utilities_dir))
sys.path.append(str(datafetch_dir))
sys.path.append(str(technical_indicators_dir))

# Import utilities from model_training_utils
from model_training_utils import DataLoader, DataPreprocessor, ModelManager
from config_handling.config_manager import ConfigManager
from config_handling.logging_setup import setup_logging
from main_indicators import apply_all_indicators
from data_store import DataStore

# Setup logging
logger = setup_logging("train_drl_model")  # Initialize the logger

# Load configuration
config_files = [Path('C:/TheTradingRobotPlug/config/config.ini')]
config_manager = ConfigManager(config_files=config_files)

# Initialize DataLoader and DataPreprocessor
data_loader = DataLoader(logger)
data_preprocessor = DataPreprocessor(logger, config_manager)

# Workaround for OpenMP runtime issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainDRLModel:
    def __init__(self, ticker, start_date, end_date, data_store, model_path='ppo_trading_model', transaction_cost=0.001, config_file='config.ini'):
        """
        Initialize the TrainDRLModel class for training and backtesting.

        Parameters:
        - ticker (str): Stock ticker symbol.
        - start_date (str): Start date for fetching historical data.
        - end_date (str): End date for fetching historical data.
        - data_store (DataStore): Instance of DataStore for data management.
        - model_path (str): Path where the trained model will be saved.
        - transaction_cost (float): Transaction cost per trade as a percentage.
        - config_file (str): Path to the configuration file.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model_path = model_path
        self.transaction_cost = transaction_cost
        self.data_store = data_store

        # Initialize ConfigManager
        self.config_manager = ConfigManager([Path(config_file)])

        # Fetch historical stock data
        self.data = self.fetch_stock_data()

    def fetch_stock_data(self):
        """
        Fetches stock data for the specified ticker and date range.

        Returns:
        - pd.DataFrame: Historical stock data.
        """
        try:
            # Attempt to load data from DataStore
            data = self.data_store.load_data(self.ticker)
            if data is None:
                logger.info(f"Fetching stock data for {self.ticker} from {self.start_date} to {self.end_date}")
                data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
                self.data_store.save_data(data, self.ticker, processed=False)
            data.columns = [col.capitalize() for col in data.columns]
            return data
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            raise

    def train_model(self, total_timesteps=10000, learning_rate=0.0001, clip_range=0.2):
        """
        Trains the trading model using PPO with enhanced settings.

        Parameters:
        - total_timesteps (int): Number of timesteps to train the model for.
        - learning_rate (float): Learning rate for the optimizer.
        - clip_range (float): Clip range for PPO updates.
        """
        try:
            logger.info(f"Training model for {self.ticker} with {total_timesteps} timesteps, learning_rate={learning_rate}, clip_range={clip_range}")
            
            env = make_vec_env(lambda: self.TradingEnv(self.data, self.transaction_cost), n_envs=1)
            
            # Custom policy with a more complex architecture
            policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])  # Example of a more complex policy network
            
            # Adding a learning rate schedule
            lr_schedule = lambda f: f * learning_rate
            
            # Train using PPO with specified hyperparameters
            model = PPO('MlpPolicy', env, learning_rate=lr_schedule, clip_range=clip_range, policy_kwargs=policy_kwargs, verbose=1)
            model.learn(total_timesteps=total_timesteps)
            model.save(self.model_path)
            self.model = model
            
            logger.info(f"Model trained and saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise


    def backtest_model(self):
        """
        Backtests the trained model and evaluates its performance.

        Returns:
        - total_reward (float): Total reward earned during backtesting.
        - final_balance (float): Final account balance after backtesting.
        - mfe (float): Maximum Favorable Excursion (MFE).
        - mae (float): Maximum Adverse Excursion (MAE).
        """
        try:
            logger.info(f"Backtesting model for {self.ticker}")
            data = self.data.copy()
            data['Date'] = pd.to_datetime(data.index)
            data.set_index('Date', inplace=True)

            env = make_vec_env(lambda: self.TradingEnv(data, self.transaction_cost), n_envs=1)
            model = PPO.load(self.model_path)

            obs = env.reset()
            done = False
            total_reward = 0
            final_balance = 0
            prices = []
            rewards = []
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                final_balance = info[0]['balance']
                prices.append(info[0]['price'])
                rewards.append(total_reward)

            prices = np.array(prices)
            mfe = np.max(prices) - prices[0]
            mae = prices[0] - np.min(prices)

            results = {
                "total_reward": total_reward,
                "final_balance": final_balance,
                "mfe": mfe,
                "mae": mae,
                "sharpe_ratio": self.calculate_sharpe_ratio(rewards),
                "sortino_ratio": self.calculate_sortino_ratio(rewards),
                "max_drawdown": self.calculate_max_drawdown(prices),
                "step_rewards": rewards,
                "step_prices": prices.tolist()
            }

            self.save_backtest_results(results)
            self.plot_backtest_results(rewards, prices)

            logger.info(f"Backtesting completed with Total Reward: {total_reward}, Final Balance: {final_balance}, MFE: {mfe}, MAE: {mae}")
            return total_reward, final_balance, mfe, mae
        except Exception as e:
            logger.error(f"Error during model backtesting: {e}")
            raise

    def save_backtest_results(self, results):
        """
        Saves backtest results to a JSON file.

        Parameters:
        - results (dict): Backtest results to be saved.
        """
        results_file = os.path.join(os.path.dirname(__file__), f"{self.ticker}_backtest_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Backtest results saved to {results_file}")

    def plot_backtest_results(self, step_rewards, step_prices):
        """
        Plots the backtest results, including cumulative rewards and stock prices.

        Parameters:
        - step_rewards (np.ndarray): Array of cumulative rewards.
        - step_prices (np.ndarray): Array of stock prices.
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(step_rewards, label='Cumulative Reward')
        plt.title('Backtest Results')
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(step_prices, label='Stock Price', color='orange')
        plt.xlabel('Steps')
        plt.ylabel('Stock Price')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def calculate_sharpe_ratio(self, rewards, risk_free_rate=0.01):
        """
        Calculates the Sharpe ratio.

        Parameters:
        - rewards (np.ndarray): Array of rewards.
        - risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.

        Returns:
        - float: The calculated Sharpe ratio.
        """
        returns = np.diff(rewards) / rewards[:-1]
        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def calculate_sortino_ratio(self, rewards, risk_free_rate=0.01):
        """
        Calculates the Sortino ratio.

        Parameters:
        - rewards (np.ndarray): Array of rewards.
        - risk_free_rate (float): Risk-free rate for Sortino ratio calculation.

        Returns:
        - float: The calculated Sortino ratio.
        """
        returns = np.diff(rewards) / rewards[:-1]
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)

    def calculate_max_drawdown(self, prices):
        """
        Calculates the maximum drawdown.

        Parameters:
        - prices (np.ndarray): Array of stock prices.

        Returns:
        - float: The maximum drawdown.
        """
        peak = prices[0]
        max_drawdown = 0
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    class TradingEnv(gym.Env):
        """
        Custom trading environment for training and backtesting.

        Attributes:
        - data (pd.DataFrame): Historical stock data.
        - transaction_cost (float): Transaction cost per trade.
        """
        metadata = {'render.modes': ['human']}

        def __init__(self, data, transaction_cost=0.001):
            super().__init__()
            self.data = data
            self.current_step = 0
            self.initial_balance = 10000
            self.balance = self.initial_balance
            self.shares_held = 0
            self.total_reward = 0
            self.done = False
            self.price = self.data['Close'].iloc[self.current_step]
            self.transaction_cost = transaction_cost

            self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=0, high=np.inf, shape=(3,), dtype=np.float32
            )

        def seed(self, seed=None):
            np.random.seed(seed)

        def reset(self):
            self.current_step = 0
            self.balance = self.initial_balance
            self.shares_held = 0
            self.total_reward = 0
            self.done = False
            self.price = self.data['Close'].iloc[self.current_step]
            return self._get_observation()

        def step(self, action):
            self._take_action(action)
            self.current_step += 1
            self.price = self.data['Close'].iloc[self.current_step]
            reward = self._calculate_reward()
            self.total_reward += reward
            self.done = self.current_step >= len(self.data) - 1
            info = {'balance': self.balance, 'price': self.price}
            return self._get_observation(), reward, self.done, info

        def _take_action(self, action):
            action_type = int(action[0])
            amount = action[1]

            if action_type == 0:  # Buy
                total_possible = self.balance // self.price
                shares_bought = total_possible * amount
                cost = shares_bought * self.price * (1 + self.transaction_cost)
                self.balance -= cost
                self.shares_held += shares_bought
            elif action_type == 1:  # Sell
                shares_sold = self.shares_held * amount
                self.balance += shares_sold * self.price * (1 - self.transaction_cost)
                self.shares_held -= shares_sold

        def _calculate_reward(self):
            current_value = self.shares_held * self.price + self.balance
            reward = current_value - self.total_reward
            return reward

        def _get_observation(self):
            return np.array([self.balance, self.shares_held, self.price])


if __name__ == "__main__":
    # Load configuration
    config_files = [Path('C:/TheTradingRobotPlug/config/config.ini')]
    config_manager = ConfigManager(config_files=config_files)

    # Pass the initialized ConfigManager to DataStore
    data_store = DataStore(ticker="AAPL", config_manager=config_manager)

    # Initialize the model
    model = TrainDRLModel(ticker="AAPL", start_date="2020-01-01", end_date="2020-12-31", data_store=data_store)

    # Train the model and backtest
    model.train_model()
    total_reward, final_balance, mfe, mae = model.backtest_model()

    # Print results
    print(f"Total Reward: {total_reward}, Final Balance: {final_balance}, MFE: {mfe}, MAE: {mae}")