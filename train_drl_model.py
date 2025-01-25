# File Name: train_drl_model.py
# Description: Enhanced module to train a DRL model using the PPO algorithm for TSLA trading
#              with standardized column names ('custom_rsi' and 'macd') in the final DataFrame.

import os
import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import yfinance as yf
import gym
from gym import spaces
from typing import Optional
from dotenv import load_dotenv
import pandas as pd

# Ensure compatibility with OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# -------------------------------------------------------------------
# Project Setup
# -------------------------------------------------------------------

# Ensure this line sets 'script_dir' to the directory of the script
script_dir = Path(__file__).resolve()

# Access the parent directory correctly
project_root = script_dir.parent

utilities_dir = project_root / 'src' / 'Utilities'
datafetch_dir = project_root / 'src' / 'Data_Fetchers'
tech_indicators_dir = project_root / 'src' / 'Data_Processing' / 'Technical_Indicators'

sys.path.extend([str(utilities_dir), str(datafetch_dir), str(tech_indicators_dir)])

load_dotenv()
# Directly set CONFIG_PATH to the new format (e.g., JSON or YAML)
CONFIG_PATH = project_root / 'config' / 'config.json'  # Change to 'config.yaml' if using YAML

# -------------------------------------------------------------------
# Attempt to Import or Fallback
# -------------------------------------------------------------------
try:
    from Utilities.model_training_utils import ModelManager, DataLoader, DataPreprocessor
except ImportError:
    class DataLoader:
        def __init__(self, logger, config_manager=None):
            self.logger = logger
            self.config_manager = config_manager
            self.logger.info("DataLoader initialized (fallback).")

        def load_data(self, source: str) -> dict:
            self.logger.info(f"Loading data from: {source}")
            return {"source": source, "data": []}

    class DataPreprocessor:
        def __init__(self, logger, config_manager=None):
            self.logger = logger
            self.config_manager = config_manager
            self.logger.info("DataPreprocessor initialized (fallback).")

        def preprocess_data(self, data: dict) -> dict:
            self.logger.info("Preprocessing data (fallback).")
            return data

from Utilities.config_manager import ConfigManager, setup_logging
from src.Data_Processing.main_indicators import apply_all_indicators
from src.Utilities.data.data_store import DataStore

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logger = setup_logging("train_drl_model")

if not CONFIG_PATH.exists():
    logger.warning(f"Config file does not exist: {CONFIG_PATH}")
else:
    logger.info(f"Config file found at: {CONFIG_PATH}")

config_manager = ConfigManager(config_files=[CONFIG_PATH], logger=logger)

data_store = DataStore(config=config_manager, logger=logger)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def standardize_columns_lowercase(df: pd.DataFrame) -> pd.DataFrame:
    """Converts all column names to lowercase except 'Date'."""
    df.columns = [col.lower() if col.lower() != 'date' else 'Date' for col in df.columns]
    return df

def rename_custom_rsi_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames any variant of 'Custom_RSI' to 'custom_rsi'.
    """
    # Identify all columns that represent 'custom_rsi' regardless of case
    for col in df.columns:
        if col.lower() == 'custom_rsi' and col != 'custom_rsi':
            df.rename(columns={col: 'custom_rsi'}, inplace=True)
            logger.info(f"Renamed column '{col}' to 'custom_rsi'")
    return df

def rename_macd_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames any variant of 'macd' to 'macd'.
    Handles cases like 'macd_line', 'macd_signal', 'macd_histogram', etc.
    """
    # Define possible MACD related column names
    possible_macd_names = ['macd_line', 'macd_signal', 'macd_histogram', 'macdindicator', 'macd14']

    for col in possible_macd_names:
        if col.lower() in df.columns and col.lower() != 'macd':
            df.rename(columns={col.lower(): 'macd'}, inplace=True)
            logger.info(f"Renamed column '{col.lower()}' to 'macd'")

    # Additionally, handle any columns containing 'macd' as substring
    for col in df.columns:
        if 'macd' in col.lower() and col.lower() != 'macd':
            df.rename(columns={col: 'macd'}, inplace=True)
            logger.info(f"Renamed column '{col}' to 'macd'")
    return df

# -------------------------------------------------------------------
# Custom Trading Environment
# -------------------------------------------------------------------
class TradingEnv(gym.Env):
    """
    Custom trading environment for TSLA trading.
    Expects columns: ['close', 'custom_rsi', 'macd', 'bollinger_width'] in df.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data: pd.DataFrame, transaction_cost=0.001, initial_balance=10000.0):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance
        self.reset()

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observations: [balance, shares_held, close, custom_rsi, macd, bollinger_width]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.total_reward = 0
        self.done = False
        return self._get_observation()

    def step(self, action: int) -> tuple:
        if self.done:
            return self._get_observation(), 0, True, {}

        current_price = self.data.loc[self.current_step, 'close']
        custom_rsi = self.data.loc[self.current_step, 'custom_rsi']
        macd = self.data.loc[self.current_step, 'macd']
        bollinger_width = self.data.loc[self.current_step, 'bollinger_width']

        prev_total = self.balance + self.shares_held * current_price

        if action == 1:  # Buy
            shares_to_buy = self.balance // (current_price * (1 + self.transaction_cost))
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.shares_held += shares_to_buy
        elif action == 2:  # Sell
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.shares_held = 0

        current_total = self.balance + self.shares_held * current_price
        reward = current_total - prev_total
        self.total_reward += reward

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, {
            "balance": self.balance,
            "price": current_price
        }

    def _get_observation(self) -> np.ndarray:
        current_price = self.data.loc[self.current_step, 'close']
        custom_rsi = self.data.loc[self.current_step, 'custom_rsi']
        macd = self.data.loc[self.current_step, 'macd']
        bollinger_width = self.data.loc[self.current_step, 'bollinger_width']

        return np.array([
            self.balance,
            self.shares_held,
            current_price,
            custom_rsi,
            macd,
            bollinger_width
        ], dtype=np.float32)

    def render(self, mode: str = 'human'):
        profit = self.balance + self.shares_held * self.data.loc[self.current_step, 'close'] - self.initial_balance
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Close: {self.data.loc[self.current_step, 'close']:.2f}, Profit: {profit:.2f}")

# -------------------------------------------------------------------
# TrainDRLModel
# -------------------------------------------------------------------
class TrainDRLModel:
    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        data_store: DataStore,
        model_path: str = 'ppo_tsla_trading_model',
        transaction_cost: float = 0.001,
        config_file: Optional[str] = None
    ):
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.model_path = model_path
        self.transaction_cost = transaction_cost
        self.data_store = data_store

        if not config_file:
            config_file = str(CONFIG_PATH)
        self.config_manager = ConfigManager([Path(config_file)], logger=logger)

        # Load and preprocess the data
        self.data = self.fetch_and_preprocess_data()

    def fetch_and_preprocess_data(self) -> pd.DataFrame:
        try:
            # Load data
            df = self.data_store.load_data(self.ticker)
            if df is None or df.empty:
                logger.info(f"Fetching stock data for {self.ticker} from {self.start_date} to {self.end_date}")
                df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
                if 'Date' not in df.columns:
                    df.reset_index(inplace=True)  # Ensure 'Date' is a column
                logger.debug(f"DataFrame after fetching and resetting index:\n{df.head()}")
                self.data_store.save_data(df, self.ticker)

            # Standardize column names
            df = standardize_columns_lowercase(df)

            # Ensure 'Date' column is present and properly named
            if 'date' in df.columns and 'Date' not in df.columns:
                df.rename(columns={'date': 'Date'}, inplace=True)

            if 'Date' not in df.columns:
                logger.error("'Date' column is missing after standardizing columns.")
                raise KeyError("Date")

            # Apply indicators
            logger.debug(f"DataFrame before applying indicators: {df.head()}")
            df = apply_all_indicators(df, logger=logger, db_handler=None, config=self.config_manager)

            # Flatten MultiIndex columns if necessary
            if isinstance(df.columns, pd.MultiIndex):
                logger.warning("Flattening MultiIndex columns.")
                df.columns = ['_'.join(col).strip() for col in df.columns.values]

            # Validate required columns
            required_cols = ['Date', 'close', 'macd', 'bollinger_width']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing columns after preprocessing: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Save processed data
            self.data_store.save_data(df, self.ticker)
            logger.info(f"Data for {self.ticker} processed successfully.")
            return df.dropna().reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}", exc_info=True)
            raise

    def train_model(self, total_timesteps=100000, learning_rate=0.0001, clip_range=0.2):
        """
        Trains the PPO model using the environment that expects 'custom_rsi' and 'macd'.
        """
        try:
            logger.info(f"Starting training for {self.ticker} with {total_timesteps} timesteps.")
            env = make_vec_env(lambda: TradingEnv(self.data, self.transaction_cost), n_envs=1)

            policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
            lr_schedule = lambda progress: learning_rate * (1 - progress)

            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=lr_schedule,
                clip_range=clip_range,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(project_root / 'logs' / 'ppo_logs')
            )

            model.learn(total_timesteps=total_timesteps)
            model.save(self.model_path)
            logger.info(f"Model trained and saved at {self.model_path}")
            self.model = model

        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            raise

    def backtest_model(self) -> dict:
        """
        Backtests the trained model.
        """
        try:
            logger.info(f"Starting backtest for {self.ticker}")
            env = TradingEnv(self.data, self.transaction_cost)
            obs = env.reset()
            model = PPO.load(self.model_path, env=env)

            total_reward = 0
            rewards = []
            prices = []
            done = False

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                rewards.append(total_reward)
                prices.append(info['price'])

            metrics = self.calculate_performance_metrics(rewards, prices, env)
            self.save_backtest_results(metrics)
            self.plot_backtest_results(rewards, prices)

            logger.info(f"Backtest completed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error during backtesting: {e}", exc_info=True)
            raise

    def calculate_performance_metrics(self, rewards: list, prices: list, env: TradingEnv) -> dict:
        """
        Calculates common performance metrics.
        """
        try:
            returns = np.diff(rewards) / rewards[:-1] if len(rewards) > 1 else np.array([0])
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) != 0 else 0
            negative_returns = returns[returns < 0]
            sortino_ratio = (np.mean(returns) / np.std(negative_returns)) * np.sqrt(252) if len(negative_returns) > 0 and np.std(negative_returns) != 0 else 0
            max_drawdown = self.calculate_max_drawdown(prices)

            final_step = env.current_step if not env.done else len(self.data) - 1
            final_balance = env.balance + env.shares_held * self.data.loc[final_step, 'close']

            metrics = {
                "total_reward": rewards[-1] if rewards else 0,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "final_balance": final_balance
            }
            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
            raise

    def calculate_max_drawdown(self, prices: list) -> float:
        """
        Calculates the maximum drawdown for the given price series.
        """
        try:
            if not prices:
                return 0
            peak = np.maximum.accumulate(prices)
            drawdowns = (peak - prices) / peak
            max_drawdown = np.max(drawdowns)
            return float(max_drawdown)
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}", exc_info=True)
            raise

    def save_backtest_results(self, results: dict):
        """
        Saves backtest results to a JSON file.
        """
        try:
            results_dir = project_root / 'results'
            results_dir.mkdir(parents=True, exist_ok=True)
            results_file = results_dir / f"{self.ticker}_backtest_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Backtest results saved to {results_file}")
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}", exc_info=True)
            raise

    def plot_backtest_results(self, rewards: list, prices: list):
        """
        Plots backtest results aligned with the dark theme.
        """
        try:
            plt.style.use('dark_background')
            plt.figure(figsize=(14, 7))

            # Plot Cumulative Rewards
            plt.subplot(2, 1, 1)
            plt.plot(rewards, label='Cumulative Reward', color='#116611')
            plt.title('Backtest Cumulative Reward', color='white')
            plt.xlabel('Steps', color='white')
            plt.ylabel('Cumulative Reward', color='white')
            plt.tick_params(colors='white')
            plt.legend()

            # Plot Stock Prices
            plt.subplot(2, 1, 2)
            plt.plot(prices, label='TSLA Price', color='#00CED1')
            plt.title('TSLA Stock Price During Backtest', color='white')
            plt.xlabel('Steps', color='white')
            plt.ylabel('Price ($)', color='white')
            plt.tick_params(colors='white')
            plt.legend()

            plt.tight_layout()
            plot_path = project_root / 'results' / 'backtest_plot.png'
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Backtest results plotted and saved successfully at {plot_path}")
        except Exception as e:
            logger.error(f"Error plotting backtest results: {e}", exc_info=True)
            raise

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        data_store = DataStore(config=config_manager, logger=logger)

        model_trainer = TrainDRLModel(
            ticker="TSLA",
            start_date="2019-01-01",
            end_date="2023-12-31",
            data_store=data_store,
            model_path=str(project_root / 'models' / 'ppo_tsla_trading_model'),
            transaction_cost=0.001,
            config_file=None
        )

        # Ensure the models directory exists
        (project_root / 'models').mkdir(parents=True, exist_ok=True)

        model_trainer.train_model(total_timesteps=100000, learning_rate=0.0001, clip_range=0.2)
        backtest_results = model_trainer.backtest_model()

        print(json.dumps(backtest_results, indent=4))

    except Exception as e:
        logger.critical(f"Failed to run the training and backtesting pipeline: {e}", exc_info=True)
