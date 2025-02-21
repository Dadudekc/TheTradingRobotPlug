# -------------------------------------------------------------------
# File: train_drl_model.py
# Location: C:/Projects/TradingRobotPlug/
# Description: Module to train a DRL model using the PPO algorithm for TSLA trading
#              with standardized column names ('custom_rsi' and 'macd') in the final DataFrame.
# -------------------------------------------------------------------

import os
import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import yfinance as yf
import gymnasium as gym  # ← Using gymnasium
from gymnasium import Env, spaces
from gymnasium.utils import seeding  # For custom seed()
from typing import Optional
from dotenv import load_dotenv
import pandas as pd
import joblib
import sklearn
import logging
from collections import deque
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count

# Try to import shimmy, if not installed, attempt to install:
try:
    import shimmy
except ImportError:
    print("Shimmy not found. Installing shimmy>=2.0...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shimmy>=2.0"])
    import shimmy  # Retry import

# Ensure compatibility with OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# -------------------------------------------------------------------
# Project Setup
# -------------------------------------------------------------------

# Ensure this line sets 'script_dir' to the directory of the script
script_dir = Path(__file__).resolve()
project_root = script_dir.parent  # Adjust if you need a higher-level root

utilities_dir = project_root / 'src' / 'Utilities'
datafetch_dir = project_root / 'src' / 'Data_Fetchers'
tech_indicators_dir = project_root / 'src' / 'Data_Processing' / 'Technical_Indicators'

sys.path.extend([str(utilities_dir), str(datafetch_dir), str(tech_indicators_dir)])

load_dotenv()
CONFIG_PATH = project_root / 'config' / 'config.json'  # Change to 'config.yaml' if using YAML

# -------------------------------------------------------------------
# Attempt to Import or Fallback
# -------------------------------------------------------------------
try:
    from Utilities.model_training.model_training_utils import ModelManager, DataLoader, DataPreprocessor
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
from src.Utilities.data_processing.main_indicators import MainIndicatorsOrchestrator
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
    df.columns = [
        col.lower() if isinstance(col, str) and col.lower() != 'date'
        else 'Date' for col in df.columns
    ]
    return df

def rename_custom_rsi_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames any variant of 'Custom_RSI' to 'custom_rsi'.
    """
    for col in df.columns:
        if isinstance(col, str) and col.lower() == 'custom_rsi' and col != 'custom_rsi':
            df.rename(columns={col: 'custom_rsi'}, inplace=True)
            logger.info(f"Renamed column '{col}' to 'custom_rsi'")
    return df

def rename_macd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames MACD-related columns to standardized names.
    """
    macd_mappings = {
        'macd_line': 'macd_line',
        'macd_signal': 'macd_signal',
        'macd_histogram': 'macd_histogram'
    }
    for original, standardized in macd_mappings.items():
        for col in df.columns:
            if col.lower() == original and col != standardized:
                df.rename(columns={col: standardized}, inplace=True)
                logger.info(f"Renamed column '{col}' to '{standardized}'")
    return df

# -------------------------------------------------------------------
# Custom Trading Environment
# -------------------------------------------------------------------
class TradingEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, transaction_cost: float = 0.001, initial_balance: float = 10000.0):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.transaction_cost = transaction_cost
        self.initial_balance = initial_balance

        # Continuous action space: -1 (Sell) to 1 (Buy)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation space: [balance, shares_held, close, custom_rsi, macd, bollinger_width]
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed: int = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.done = False
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        if self.done:
            terminated = True
            truncated = False
            return self._get_observation(), 0.0, terminated, truncated, {}

        # Map continuous action to discrete actions
        action_val = action[0]
        if action_val <= -0.33:
            action_type = 0  # Hold
        elif -0.33 < action_val < 0.33:
            action_type = 1  # Buy
        else:
            action_type = 2  # Sell

        current_price = self.data.loc[self.current_step, "close"]
        prev_total = self.balance + self.shares_held * current_price

        if action_type == 1:  # Buy
            shares_to_buy = self.balance // (current_price * (1 + self.transaction_cost))
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.shares_held += shares_to_buy

        elif action_type == 2:  # Sell
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.shares_held = 0

        current_total = self.balance + self.shares_held * current_price
        reward = current_total - prev_total
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            self.done = True

        terminated = self.done
        truncated = False  # Set to True if you have truncation conditions

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self) -> np.ndarray:
        current_price = self.data.loc[self.current_step, "close"]
        custom_rsi = self.data.loc[self.current_step, "custom_rsi"]
        macd = self.data.loc[self.current_step, "macd"]
        bollinger_width = self.data.loc[self.current_step, "bollinger_width"]

        return np.array(
            [
                self.balance,
                self.shares_held,
                current_price,
                custom_rsi,
                macd,
                bollinger_width,
            ],
            dtype=np.float32,
        )

    def render(self, mode: str = "human"):
        if mode == "human":
            profit = (
                self.balance
                + self.shares_held * self.data.loc[self.current_step, "close"]
                - self.initial_balance
            )
            print(
                f"Step: {self.current_step}, Balance: {self.balance:.2f}, Close: {self.data.loc[self.current_step, 'close']:.2f}, Profit: {profit:.2f}"
            )

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

        # Load and preprocess data
        self.data = self.fetch_and_preprocess_data()

    def fetch_and_preprocess_data(self) -> pd.DataFrame:
        try:
            # Attempt to load existing data
            df = self.data_store.load_data(self.ticker)
            if df is None or df.empty:
                logger.info(f"Fetching {self.ticker} data from {self.start_date} to {self.end_date} via yfinance.")
                df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
                if "Date" not in df.columns:
                    df.reset_index(inplace=True)
                self.data_store.save_data(df, self.ticker)

            df = standardize_columns_lowercase(df)
            if "date" in df.columns and "Date" not in df.columns:
                df.rename(columns={"date": "Date"}, inplace=True)

            # Basic check
            if "Date" not in df.columns:
                raise KeyError("'Date' column not found in DataFrame.")

            # Set Date index & sort
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)

            # Apply technical indicators
            logger.debug(f"Before apply_all_indicators, df.head():\n{df.head()}")
            df = apply_all_indicators(df, logger=logger, db_handler=None, config=self.config_manager, data_store=self.data_store)

            # Ensure the main columns exist
            required_cols = ["close", "macd", "bollinger_width", "custom_rsi"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                logger.info(f"Attempting manual indicator calculation for missing: {missing}")
                if "macd" in missing:
                    # Minimal MACD
                    df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()

                if "bollinger_width" in missing:
                    df["bollinger_width"] = df["close"].rolling(20).std() * 2

                if "custom_rsi" in missing:
                    delta = df["close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    df["custom_rsi"] = 100 - (100 / (1 + rs))

                # Validate again
                missing2 = [col for col in required_cols if col not in df.columns]
                if missing2:
                    raise ValueError(f"Still missing: {missing2} after manual computation")

            # Dropna on required columns
            df.dropna(subset=required_cols, inplace=True)

            # Save
            self.data_store.save_data(df, self.ticker)
            logger.info(f"Data for {self.ticker} is ready. shape={df.shape}")

            return df.reset_index()

        except Exception as e:
            logger.error(f"Error in fetch_and_preprocess_data: {e}", exc_info=True)
            raise

    def train_model(self, total_timesteps=100000, learning_rate=0.0001, clip_range=0.2):
        """Train the PPO model using the environment."""
        try:
            logger.info(f"Training PPO for {self.ticker} ...")
            env = make_vec_env(lambda: TradingEnv(self.data, self.transaction_cost), n_envs=1)

            # Updated policy_kwargs as per SB3 v1.8.0
            policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
            lr_schedule = lambda progress: learning_rate * (1 - progress)

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=lr_schedule,
                clip_range=clip_range,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(project_root / "logs" / "ppo_logs"),
            )
            model.learn(total_timesteps=total_timesteps)
            model.save(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            self.model = model

        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise

    def backtest_model(self):
        """Perform backtesting."""
        try:
            logger.info(f"Starting backtest for {self.ticker}")
            env = TradingEnv(self.data, self.transaction_cost)
            obs, _ = env.reset()
            model = PPO.load(self.model_path, env=env)

            done = False
            total_reward = 0
            rewards = []
            prices = []
            step_count = 0
            max_steps = len(self.data)

            while not done and step_count < max_steps:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                rewards.append(total_reward)
                prices.append(info.get("price", 0))
                step_count += 1

            if step_count >= max_steps:
                logger.warning("Reached maximum steps in backtest environment.")

            metrics = self.calculate_performance_metrics(rewards, prices, env)
            self.save_backtest_results(metrics)
            self.plot_backtest_results(rewards, prices)
            logger.info(f"Backtest completed: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error in backtest_model: {e}", exc_info=True)
            raise

    def calculate_performance_metrics(self, rewards, prices, env):
        """Compute Sharpe, Sortino, etc."""
        if len(rewards) < 2:
            logger.warning("Insufficient reward data to compute returns.")
            returns = [0]
        else:
            returns = np.diff(rewards) / np.array(rewards[:-1])

        sharpe_ratio = 0.0
        if np.std(returns) != 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

        neg_returns = returns[returns < 0]
        sortino_ratio = 0.0
        if len(neg_returns) > 0 and np.std(neg_returns) != 0:
            sortino_ratio = (np.mean(returns) / np.std(neg_returns)) * np.sqrt(252)

        max_dd = self.calculate_max_drawdown(prices)
        final_step = env.current_step if not env.done else len(self.data) - 1
        final_balance = env.balance + env.shares_held * self.data.loc[final_step, "close"]

        return {
            "total_reward": rewards[-1] if rewards else 0.0,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_dd,
            "final_balance": final_balance,
        }

    def calculate_max_drawdown(self, prices: list) -> float:
        """Max drawdown from the price history."""
        if not prices:
            return 0.0
        peaks = np.maximum.accumulate(prices)
        drawdowns = (peaks - prices) / peaks
        return float(np.max(drawdowns))

    def save_backtest_results(self, results: dict):
        """Saves results to JSON."""
        out_dir = project_root / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{self.ticker}_backtest_results.json"
        with open(fname, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Backtest results saved to {fname}")

    def plot_backtest_results(self, rewards, prices):
        """Plots cumulative rewards and price."""
        plt.style.use("dark_background")
        plt.figure(figsize=(12, 6))

        # Rewards
        plt.subplot(2, 1, 1)
        plt.plot(rewards, color="cyan")
        plt.title("Cumulative Rewards")
        plt.xlabel("Steps")
        plt.ylabel("Rewards")

        # Price
        plt.subplot(2, 1, 2)
        plt.plot(prices, color="orange")
        plt.title("Asset Price During Backtest")
        plt.xlabel("Steps")
        plt.ylabel("Price")

        plt.tight_layout()
        out_path = project_root / "results" / "backtest_plot.png"
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Backtest plot saved to {out_path}")

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        trainer = TrainDRLModel(
            ticker="TSLA",
            start_date="2019-01-01",
            end_date="2023-12-31",
            data_store=data_store,
            model_path=str(project_root / "models" / "ppo_tsla_trading_model"),
            transaction_cost=0.001,
            config_file=None,
        )
        (project_root / "models").mkdir(parents=True, exist_ok=True)

        trainer.train_model(total_timesteps=100_000, learning_rate=0.0001, clip_range=0.2)
        backtest_results = trainer.backtest_model()
        print(json.dumps(backtest_results, indent=4))

    except Exception as ex:
        logger.critical(f"Failed to run DRL training/backtesting pipeline: {ex}", exc_info=True)


##
# Optional validation for 'date' or required columns:
# Modify below if you want to forcibly check for 'macd_line', 'rsi', etc.
# 
# required_cols_check = ['macd_line', 'macd_signal', 'macd_histogram', 'rsi', 'bollinger_width']
# missing_cols_check = [col for col in required_cols_check if col not in df.columns]
# if missing_cols_check:
#     raise ValueError(f"Missing required columns: {missing_cols_check}")
