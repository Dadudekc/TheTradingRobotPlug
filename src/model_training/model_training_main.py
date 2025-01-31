# -------------------------------------------------------------------
# File Path: C:\Projects\TradingRobotPlug\Scripts\model_training\model_training_main.py
# Description: Unified model training script for various models.
# This script fetches data, applies preprocessing, and trains multiple models based on user input via config files.
# -------------------------------------------------------------------

import sys
from pathlib import Path
import logging
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------
# Dynamically set the project root based on the current file's location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]  # Adjusted for the project structure

# Define directories based on your project structure
config_dir = project_root / 'config'
saved_models_dir = project_root / 'SavedModels'
logs_dir = project_root / 'logs'
data_dir = project_root / 'data' / 'alpha_vantage'  # Assuming data is stored here
scripts_dir = project_root / 'Scripts'

# Paths to specific scripts
data_fetching_dir = scripts_dir / 'Data_Fetchers' / 'Stock_Data'
data_processing_dir = scripts_dir / 'Data_Processing' / 'stock_data_handlers'
technical_indicators_dir = scripts_dir / 'Data_Processing' / 'Technical_Indicators'
model_utils_dir = scripts_dir / 'model_training' / 'utils'
utilities_dir = scripts_dir / 'Utilities'

# Add the necessary directories to the Python path
sys.path.extend([
    str(data_fetching_dir),
    str(data_processing_dir),
    str(technical_indicators_dir),
    str(model_utils_dir),
    str(utilities_dir),
    str(saved_models_dir)
])

# Import utilities and model training modules
try:
    # Data fetching
    from stock_data_fetcher import fetch_data
    # Data preprocessing
    from data_imputation import preprocess_data, preprocess_data_for_lstm
    # Technical indicators
    from main_indicators import apply_all_indicators
    # Model training functions
    from arima_model_trainer import train_arima_model
    from lstm_model_trainer import train_lstm_model
    from random_forest_trainer import train_random_forest_model
    from svm_trainer import train_svm_model
    from decision_tree_trainer import train_decision_tree_model
    from gradient_boosting_trainer import train_gradient_boosting_model
    # Evaluation and saving
    from evaluation_helpers import evaluate_model_performance
    from model_saving import save_model_and_scaler
    # Logging setup
    from config_handling.logging_setup import setup_logging
    print("Modules imported successfully!")
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Define supported models
SUPPORTED_MODELS = ['ARIMA', 'LSTM', 'RANDOM_FOREST', 'DECISION_TREE', 'SVM', 'GRADIENT_BOOSTING']

def load_config(config_path):
    """
    Load configuration from a YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def main():
    """
    Main function to handle unified model training.
    Fetches data, processes it, and trains models based on user input via config files.
    """
    # Load configuration
    config_path = config_dir / 'config.yaml'
    config = load_config(config_path)
    
    # Initialize logging
    logger = setup_logging(script_name="model_training_main", log_file=logs_dir / 'model_training.log')
    logger.setLevel(logging.DEBUG)

    # Set parameters for model training from config
    params = config.get('training_params', {})
    
    model_type = params.get("model_type", "LSTM").upper()
    symbol = params.get("symbol", "AAPL")
    csv_path = params.get("csv_path", data_dir / f'{symbol}_data.csv')

    train_model(model_type, csv_path, params, logger)

def train_model(model_type, data_path, params=None, logger=None):
    """
    Function to train the specified model type using the data at data_path.
    """
    # Set up logger if not provided
    if logger is None:
        logger = setup_logging(script_name="train_model", log_file=logs_dir / 'model_training.log')
        logger.setLevel(logging.DEBUG)

    logger.info(f"Starting model training: {model_type}")
    logger.info(f"Received parameters: {params}")
    
    # 1. Load data from CSV
    try:
        logger.info(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully with shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        return

    # 2. Apply technical indicators and feature engineering
    try:
        logger.info("Applying technical indicators...")
        data = apply_all_indicators(data)
        logger.info(f"Technical indicators applied successfully. Data shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error applying technical indicators: {e}")
        return

    # 3. Preprocess data
    try:
        logger.info("Preprocessing data...")
        preprocessed_data = preprocess_data(data, target_column='close')
        
        if len(preprocessed_data) == 2:
            X, y = preprocessed_data
        elif len(preprocessed_data) == 3:
            X, y, additional_info = preprocessed_data
            logger.info(f"Preprocessing returned additional info: {additional_info}")
        else:
            logger.error("Unexpected output from preprocess_data.")
            return
        
        logger.info(f"Data preprocessing completed. X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return

    # 4. Train the specified model
    logger.info(f"Training model for symbol: {symbol}")

    try:
        if model_type == 'ARIMA':
            train_arima(y, symbol, params, logger)
        elif model_type == 'LSTM':
            train_lstm(X, y, symbol, params, logger)
        elif model_type == 'RANDOM_FOREST':
            train_random_forest(X, y, symbol, params, logger)
        elif model_type == 'DECISION_TREE':
            train_decision_tree(X, y, symbol, params, logger)
        elif model_type == 'SVM':
            train_svm(X, y, symbol, params, logger)
        elif model_type == 'GRADIENT_BOOSTING':
            train_gradient_boosting(X, y, symbol, params, logger)
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return
    except Exception as e:
        logger.error(f"Error during {model_type} model training: {e}")
        return

    logger.info(f"Model training for {model_type} completed successfully.")

# Model-specific training functions
def train_arima(y, symbol, params, logger):
    """Train ARIMA model."""
    logger.info("Starting ARIMA model training...")
    model_path = saved_models_dir / f'arima_model_{symbol}.pkl'
    rmse, forecast, trained_model = train_arima_model(
        y, symbol, model_path, order=params.get('arima_order', (1, 1, 1)),
        seasonal_order=params.get('seasonal_order', (0, 1, 1, 12)), logger=logger
    )
    evaluate_model_performance(rmse, params.get('retrain_threshold', 0.05), logger)

def train_lstm(X, y, symbol, params, logger):
    """Train LSTM model."""
    logger.info("Starting LSTM model training...")
    model_path = saved_models_dir / f'lstm_model_{symbol}.h5'
    scaler_path = saved_models_dir / f'scaler_{symbol}.pkl'
    time_steps = params.get('time_steps', 10)

    X_train, X_test, y_train, y_test, scaler = preprocess_data_for_lstm(
        X, y, time_steps, test_size=0.2, logger=logger
    )

    lstm_model = train_lstm_model(X_train, y_train, X_test, y_test, params, logger)

    # Evaluate performance and save
    y_pred = lstm_model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    logger.info(f"LSTM Model Performance: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    save_model_and_scaler(lstm_model, scaler, model_path, scaler_path, logger)

def train_random_forest(X, y, symbol, params, logger):
    """Train Random Forest model."""
    logger.info("Starting Random Forest model training...")
    model_path = saved_models_dir / f'random_forest_model_{symbol}.pkl'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = train_random_forest_model(X_train, y_train, X_test, y_test, params, logger)
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Random Forest Model Performance: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    save_model_and_scaler(rf_model, None, model_path, None, logger)

def train_decision_tree(X, y, symbol, params, logger):
    """Train Decision Tree model."""
    logger.info("Starting Decision Tree model training...")
    model_path = saved_models_dir / f'decision_tree_model_{symbol}.pkl'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dt_model = train_decision_tree_model(X_train, y_train, X_test, y_test, params, logger)
    y_pred = dt_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Decision Tree Model Performance: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    save_model_and_scaler(dt_model, None, model_path, None, logger)

def train_svm(X, y, symbol, params, logger):
    """Train SVM model."""
    logger.info("Starting SVM model training...")
    model_path = saved_models_dir / f'svm_model_{symbol}.pkl'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = train_svm_model(X_train, y_train, X_test, y_test, params, logger)
    y_pred = svm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"SVM Model Performance: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    save_model_and_scaler(svm_model, None, model_path, None, logger)

def train_gradient_boosting(X, y, symbol, params, logger):
    """Train Gradient Boosting model."""
    logger.info("Starting Gradient Boosting model training...")
    model_path = saved_models_dir / f'gradient_boosting_model_{symbol}.pkl'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gb_model = train_gradient_boosting_model(X_train, y_train, X_test, y_test, params, logger)
    y_pred = gb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Gradient Boosting Model Performance: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    save_model_and_scaler(gb_model, None, model_path, None, logger)

if __name__ == "__main__":
    main()
