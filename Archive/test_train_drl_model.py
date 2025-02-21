# -------------------------------------------------------------------
# File: test_train_drl_model.py
# Location: src/tests/
# Description: Test suite for train_drl_model.py
# -------------------------------------------------------------------

import unittest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure that the project root is in sys.path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Now, import the modules after adding project_root to sys.path
from train_drl_model import (
    TrainDRLModel,
    TradingEnv,
    standardize_columns_lowercase,
    rename_custom_rsi_column,
    rename_macd_column
)

# -------------------------------------------------------------------
# Test Suite for TrainDRLModel
# -------------------------------------------------------------------
class TestTrainDRLModel(unittest.TestCase):
    @patch('train_drl_model.yf.download')
    def setUp(self, mock_yf_download):
        """
        Setup for the TestTrainDRLModel test cases.
        """
        # Mock data to be returned by yf.download
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2022-01-01', periods=30, freq='D'),  # Increased periods for indicator calculation
            'Close': [700 + i for i in range(30)],  # Simulate a trend for MACD
            'Volume': [1000 + i * 10 for i in range(30)],
            'High': [705 + i for i in range(30)],
            'Low': [695 + i for i in range(30)]
        })
        
        # Ensure 'Date' is a column, not an index
        mock_yf_download.return_value = mock_data

        # Initialize MagicMock for data_store
        mock_data_store = MagicMock()
        mock_data_store.load_data.return_value = None  # Simulate data not existing

        # Initialize TrainDRLModel with mocked data_store
        self.train_drl_model = TrainDRLModel(
            ticker="TSLA",
            start_date="2022-01-01",
            end_date="2023-12-31",
            data_store=mock_data_store,
            model_path="models/ppo_tsla_trading_model",
            transaction_cost=0.001,
            config_file=None
        )

    def tearDown(self):
        # Stop all patches
        patch.stopall()

    def test_initialization(self):
        """
        Test that TrainDRLModel initializes correctly.
        """
        self.assertEqual(self.train_drl_model.ticker, "TSLA")
        self.assertEqual(self.train_drl_model.start_date, "2022-01-01")
        self.assertEqual(self.train_drl_model.end_date, "2023-12-31")
        self.assertEqual(self.train_drl_model.transaction_cost, 0.001)
        self.assertEqual(self.train_drl_model.model_path, 'models/ppo_tsla_trading_model')
        self.assertIsNotNone(self.train_drl_model.data)
        # Additional assertions can be added here

    @patch('train_drl_model.yf.download')
    def test_fetch_and_preprocess_data_with_existing_data(self, mock_yf_download):
        """
        Test fetch_and_preprocess_data when data already exists in DataStore.
        """
        # Mock DataStore.load_data to return a DataFrame with required columns
        mock_data_store = self.train_drl_model.data_store
        existing_data = pd.DataFrame({
            'date': pd.date_range(start='2022-01-01', periods=5, freq='D'),
            'close': [700, 710, 720, 730, 740],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'high': [705, 715, 725, 735, 745],
            'low': [695, 705, 715, 725, 735]
        })
        mock_data_store.load_data.return_value = existing_data

        # Invoke fetch_and_preprocess_data
        with patch('src.Data_Processing.main_indicators.apply_all_indicators', return_value=existing_data):
            data = self.train_drl_model.fetch_and_preprocess_data()

            # Assertions
            mock_apply_indicators.assert_called_once()
            pd.testing.assert_frame_equal(data, existing_data)
            mock_data_store.save_data.assert_called_once_with(existing_data, "TSLA")

    @patch('train_drl_model.yf.download')
    def test_fetch_and_preprocess_data_with_no_existing_data(self, mock_yf_download):
        """
        Test fetch_and_preprocess_data when data does not exist in DataStore.
        """
        # Mock yf.download to return predefined data
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2022-01-01', periods=5, freq='D'),
            'Close': [700, 710, 720, 730, 740],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'High': [705, 715, 725, 735, 745],
            'Low': [695, 705, 715, 725, 735]
        })
        mock_yf_download.return_value = mock_data

        # Mock DataStore.load_data to return None
        mock_data_store = self.train_drl_model.data_store
        mock_data_store.load_data.return_value = None

        # Mock apply_indicators to return processed data with required columns
        processed_data = pd.DataFrame({
            'Date': mock_data['Date'],
            'close': mock_data['Close'],
            'macd': [0.1, 0.2, 0.3, 0.4, 0.5],
            'bollinger_width': [0.05, 0.06, 0.07, 0.08, 0.09]
        })
        with patch('train_drl_model.apply_indicators', return_value=processed_data) as mock_apply_indicators:
            data = self.train_drl_model.fetch_and_preprocess_data()

            # Assertions
            mock_apply_indicators.assert_called_once()
            mock_data_store.save_data.assert_called_once_with(processed_data, "TSLA")
            pd.testing.assert_frame_equal(data, processed_data)

    @patch('train_drl_model.make_vec_env')
    @patch('train_drl_model.PPO')
    def test_train_model(self, mock_PPO, mock_make_vec_env):
        """
        Test the train_model method.
        """
        # Mock the environment
        mock_env = MagicMock()
        mock_make_vec_env.return_value = mock_env

        # Mock the PPO model
        mock_model_instance = MagicMock()
        mock_PPO.return_value = mock_model_instance

        # Call train_model
        self.train_drl_model.train_model(total_timesteps=1000, learning_rate=0.0001, clip_range=0.2)

        # Ensure make_vec_env is called correctly with any callable and n_envs=1
        mock_make_vec_env.assert_called_once_with(ANY, n_envs=1)

        # Ensure PPO is instantiated correctly
        mock_PPO.assert_called_once()
        args, kwargs = mock_PPO.call_args
        self.assertEqual(args[0], 'MlpPolicy')
        self.assertEqual(args[1], mock_env)
        self.assertEqual(kwargs['clip_range'], 0.2)
        self.assertEqual(kwargs['policy_kwargs'], {'net_arch': [{'pi': [128, 128], 'vf': [128, 128]}]})
        self.assertEqual(kwargs['verbose'], 1)
        self.assertTrue('tensorboard_log' in kwargs)

        # Ensure learn and save are called
        mock_model_instance.learn.assert_called_once_with(total_timesteps=1000)
        mock_model_instance.save.assert_called_once_with('models/ppo_tsla_trading_model')

    @patch('train_drl_model.TradingEnv')
    @patch('train_drl_model.PPO.load')
    def test_backtest_model(self, mock_PPO_load, mock_TradingEnv):
        """
        Test the backtest_model method.
        """
        # Mock the Trading Environment
        mock_env_instance = MagicMock()
        mock_env_instance.balance = 10000.0
        mock_env_instance.shares_held = 0
        mock_env_instance.reset.return_value = "mocked_obs"
        mock_TradingEnv.return_value = mock_env_instance

        # Mock PPO Model and its predict() method
        mock_model = MagicMock()
        mock_model.predict.return_value = (0, None)  # Ensure it returns a tuple
        mock_PPO_load.return_value = mock_model

        # Invoke backtest_model
        metrics = self.train_drl_model.backtest_model()

        # Assertions
        self.assertIsInstance(metrics, dict)
        # Add more specific assertions based on expected metrics

    def test_trading_env_step_buy(self):
        """
        Test the TradingEnv step function with a Buy action.
        """
        env_df = pd.DataFrame({
            'close': [100, 101],
            'custom_rsi': [50, 51],
            'macd': [0.5, 0.6],
            'bollinger_width': [5, 5.1]
        })

        # Initialize TradingEnv with transaction_cost
        env = TradingEnv(data=env_df, transaction_cost=0.001)
        obs = env.reset()

        # Set balance and shares_held after reset
        env.balance = 10000.0
        env.shares_held = 0

        # Action 1 = Buy
        new_obs, reward, done, info = env.step(1)

        # Calculate expected shares to buy
        shares_to_buy = env.balance // (100 * (1 + env.transaction_cost))  # 10000 // 100.1 = 99 shares
        expected_cost = shares_to_buy * 100 * (1 + env.transaction_cost)  # 99 * 100 * 1.001 = 9909.9
        expected_balance = 10000.0 - expected_cost  # 10000.0 - 9909.9 = 90.1
        expected_shares_held = shares_to_buy
        expected_total = expected_balance + expected_shares_held * 100  # 90.1 + 9900 = 9990.1

        # Debug Statements
        print(f"DEBUG: Expected balance: {expected_balance}, Actual balance: {env.balance}")
        print(f"DEBUG: Expected shares held: {expected_shares_held}, Actual shares held: {env.shares_held}")
        print(f"DEBUG: Expected total value: {expected_total}, Actual total: {env.balance + env.shares_held * 100}")

        # Assertions
        self.assertAlmostEqual(env.balance, expected_balance, places=2)
        self.assertEqual(env.shares_held, expected_shares_held)
        self.assertAlmostEqual(env.balance + env.shares_held * env.data.loc[env.current_step, 'close'], expected_total, places=2)

# -------------------------------------------------------------------
# Test Suite for Helper Functions
# -------------------------------------------------------------------
class TestHelperFunctions(unittest.TestCase):
    def test_standardize_columns_lowercase(self):
        """
        Test the standardize_columns_lowercase helper function.
        """
        df = pd.DataFrame({
            'Date': pd.date_range(start='2022-01-01', periods=3),
            'Close': [100, 101, 102],
            'CUSTOM_RSI': [50, 51, 52],
            'MACD': [0.5, 0.6, 0.7],
            'Volume': [1000, 1100, 1200]
        })

        standardized_df = standardize_columns_lowercase(df)

        expected_columns = ['Date', 'close', 'custom_rsi', 'macd', 'volume']
        self.assertListEqual(list(standardized_df.columns), expected_columns)

    def test_rename_custom_rsi_column(self):
        """
        Test the rename_custom_rsi_column helper function.
        """
        df = pd.DataFrame({
            'custom_RSI': [50, 51, 52],
            'other_column': [1, 2, 3]
        })

        renamed_df = rename_custom_rsi_column(df)

        expected_columns = ['custom_rsi', 'other_column']
        self.assertListEqual(list(renamed_df.columns), expected_columns)

    def test_rename_macd_column(self):
        """
        Test the rename_macd_column helper function.
        """
        df = pd.DataFrame({
            'macd_line': [0.5, 0.6, 0.7],
            'macd_signal': [0.4, 0.5, 0.6],
            'other_column': [1, 2, 3]
        })

        renamed_df = rename_macd_column(df)

        expected_columns = ['macd', 'macd', 'other_column']
        self.assertListEqual(list(renamed_df.columns), expected_columns)

# -------------------------------------------------------------------
# Run the Tests
# -------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
