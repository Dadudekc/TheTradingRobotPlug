# -------------------------------------------------------------------
# File Name: C:\TheTradingRobotPlug\Scripts\model_training\continuous_learning\continuous_learning_main.py
# Location: C:\TheTradingRobotPlug\Scripts\model_training\continuous_learning\continuous_learning_main.py
# Description: Main entry point for the continuous learning, training, and backtesting process in the TradingRobotPlug project.
# -------------------------------------------------------------------

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Any, Optional
from configparser import NoSectionError, NoOptionError

# Adjust paths to include utilities and other modules
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
sys.path.extend([
    str(project_root / 'Scripts' / 'Utilities'),
    str(project_root / 'Scripts' / 'Data_Fetchers'),
    str(project_root / 'Scripts' / 'Data_Processing' / 'Technical_Indicators'),
    str(project_root / 'Scripts' / 'model_training' / 'continuous_learning')
])

# Import necessary modules
from config_handling.config_manager import ConfigManager
from config_handling.logging_setup import setup_logging
from data_store import DataStore
from trading_functions import TradingFunctions

# Setup logging
logger = setup_logging("model_training_main")

def load_configurations(config_path: Path) -> ConfigManager:
    """
    Load configurations using the ConfigManager class.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        ConfigManager: An instance of ConfigManager with loaded configurations.
    """
    try:
        if config_path.exists():
            logger.info(f"Loading configuration file from {config_path}")
            return ConfigManager(config_files=[config_path])
        else:
            logger.error(f"Configuration file not found at {config_path}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        sys.exit(1)

def get_config_value(config_manager: ConfigManager, section: str, key: str, default_value: Optional[Any]=None) -> Any:
    """
    Retrieve a configuration value using ConfigManager's get method.

    Args:
        config_manager (ConfigManager): The configuration manager instance.
        section (str): The section of the config file.
        key (str): The key to retrieve from the config file.
        default_value (Any, optional): Default value if the key is not found.

    Returns:
        Any: The value from the config, or default if not found.
    """
    try:
        value = config_manager.get(key, section=section, fallback=default_value)
        if value is None:
            logger.warning(f"Key '{key}' not found in section '{section}'. Using default value: {default_value}")
            return default_value
        return value
    except (NoSectionError, NoOptionError) as e:
        logger.warning(f"Configuration error: {e}. Using default value for '{key}': {default_value}")
        return default_value
    except Exception as e:
        logger.warning(f"Failed to retrieve {key} from section {section}, using default {default_value}: {e}")
        return default_value

def initialize_data_store(config_manager: ConfigManager) -> DataStore:
    """
    Initialize the DataStore with the provided configuration.

    Args:
        config_manager (ConfigManager): The configuration manager instance.

    Returns:
        DataStore: An instance of DataStore initialized with the config data.
    """
    db_path = get_config_value(config_manager, 'DATA_STORE', 'db_path', str(project_root / 'data' / 'trading_data.db'))
    csv_dir = get_config_value(config_manager, 'DATA_STORE', 'csv_dir', str(project_root / 'data'))

    return DataStore(db_path=db_path, config_manager=config_manager, logger=logger)

def train_and_backtest_model(config_manager: ConfigManager, data_store: DataStore, args: argparse.Namespace):
    """
    Train and backtest the trading model.

    Args:
        config_manager (ConfigManager): The configuration manager instance.
        data_store (DataStore): The data storage and retrieval instance.
        args (argparse.Namespace): Command-line arguments for dynamic configuration.
    """
    try:
        # Override config values with CLI arguments if provided
        ticker = args.ticker or get_config_value(config_manager, 'TRADING', 'ticker', 'AAPL')
        start_date = args.start_date or get_config_value(config_manager, 'TRADING', 'start_date', '2020-01-01')
        end_date = args.end_date or get_config_value(config_manager, 'TRADING', 'end_date', '2020-12-31')
        initial_balance = float(args.initial_balance or get_config_value(config_manager, 'TRADING', 'initial_balance', 10000.0))
        transaction_cost = float(args.transaction_cost or get_config_value(config_manager, 'TRADING', 'transaction_cost', 0.001))

        # Initialize TradingFunctions with parameters
        trading_model = TradingFunctions(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            data_store=data_store,
            config_manager=config_manager
        )

        logger.info(f"Starting model training for {ticker} from {start_date} to {end_date}.")
        trading_model.train_model(total_timesteps=10000)

        logger.info(f"Starting backtesting for {ticker}.")
        total_reward, final_portfolio_value, mfe, mae = trading_model.backtest_model()
        logger.info(f"Backtesting completed with Total Reward: {total_reward}, Final Portfolio Value: {final_portfolio_value}, MFE: {mfe}, MAE: {mae}")

        # Optionally save the trained model
        if args.save_model:
            model_path = Path(args.save_model)
            trading_model.save_model(model_path)
            logger.info(f"Trained model saved to {model_path}")

    except Exception as e:
        logger.error(f"An error occurred during the training and backtesting process: {e}", exc_info=True)
        sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for dynamic configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Continuous Learning and Backtesting for TradingRobotPlug.")
    parser.add_argument('--config', type=str, default=str(project_root / 'config' / 'config.ini'),
                        help='Path to the configuration file.')
    parser.add_argument('--ticker', type=str, help='Ticker symbol to train and backtest.')
    parser.add_argument('--start_date', type=str, help='Start date for training data (YYYY-MM-DD).')
    parser.add_argument('--end_date', type=str, help='End date for training data (YYYY-MM-DD).')
    parser.add_argument('--initial_balance', type=float, help='Initial balance for backtesting.')
    parser.add_argument('--transaction_cost', type=float, help='Transaction cost rate.')
    parser.add_argument('--save_model', type=str, help='Path to save the trained model.')
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level.')
    return parser.parse_args()

def main():
    """
    Main function to initialize configurations and run training and backtesting.
    """
    args = parse_arguments()
    
    # Update logging level if provided
    if args.log_level:
        logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
        logger.info(f"Logging level set to {args.log_level.upper()}")

    try:
        config_path = Path(args.config)
        config_manager = load_configurations(config_path)
        data_store = initialize_data_store(config_manager)
        train_and_backtest_model(config_manager, data_store, args)
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()