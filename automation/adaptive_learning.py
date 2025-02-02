"""
automation/adaptive_learning.py
-------------------------------
Provides functions for adaptive learning logic based on historical performance data.
"""

import pandas as pd
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor

def adaptive_learning_logic(performance_log_path="performance_log.csv"):
    """
    Uses historical performance data to predict optimal training parameters.
    
    Args:
        performance_log_path (str): Path to the CSV file containing performance logs.
    
    Returns:
        dict: Predicted optimal training parameters.
    """
    try:
        performance_data = pd.read_csv(performance_log_path)
        if performance_data.empty:
            logging.warning("Performance log is empty.")
            return None
        
        # Assume the log contains feature columns and an 'optimal_parameters' column in JSON string format.
        features = performance_data.drop(columns=['performance_metric', 'optimal_parameters'])
        targets = performance_data['optimal_parameters'].apply(eval)  # Convert string to dict
        
        # Normalize targets into a DataFrame
        targets_df = pd.json_normalize(targets)
        
        X_train, X_test, y_train, y_test = train_test_split(features, targets_df, test_size=0.2, random_state=42)
        
        model = XGBRegressor(random_state=42)
        param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7]}
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        
        accuracy = grid_search.score(X_test, y_test)
        logging.info(f"Adaptive learning model accuracy: {accuracy:.4f}")
        
        latest_features = features.iloc[-1].values.reshape(1, -1)
        predicted_parameters = grid_search.predict(latest_features)
        # For demonstration, assume the only parameter to update is 'n_estimators'
        predicted_parameters_dict = {"n_estimators": int(predicted_parameters[0])}
        return predicted_parameters_dict
    except Exception as e:
        logging.error("Error in adaptive_learning_logic: " + str(e))
        raise

def update_training_parameters(new_parameters, config_path="training_config.json"):
    """
    Updates a JSON configuration file with new training parameters.
    
    Args:
        new_parameters (dict): New training parameters.
        config_path (str): Path to the configuration file.
    """
    import json, os
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}
        config.update(new_parameters)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logging.info("Training configuration updated with new parameters.")
    except Exception as e:
        logging.error("Error updating training parameters: " + str(e))
        raise
