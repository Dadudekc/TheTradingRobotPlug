"""
model_training/random_forest.py
--------------------------------
Trains a Random Forest regressor using randomized hyperparameter search.
"""

import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def train_random_forest_with_auto_optimization(X_train, y_train, X_val, y_val, random_state=42):
    """
    Trains a Random Forest regressor with auto-optimization via RandomizedSearchCV.

    Parameters:
        X_train (np.array): Training features.
        y_train (np.array): Training targets.
        X_val (np.array): Validation features.
        y_val (np.array): Validation targets.
        random_state (int): Seed for reproducibility.
    
    Returns:
        best_rf_model: The optimized RandomForestRegressor model.
        metrics (tuple): (mse, rmse, r2) computed on the validation set.
    """
    try:
        param_grid = {
            'n_estimators': np.linspace(50, 300, num=20, dtype=int),
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestRegressor(random_state=random_state)
        rf_random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_grid,
            n_iter=50,
            cv=3,
            verbose=1,
            random_state=random_state,
            n_jobs=-1
        )
        rf_random_search.fit(X_train, y_train)
        best_rf_model = rf_random_search.best_estimator_
        y_pred_val = best_rf_model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred_val)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred_val)
        logging.info(f"Random Forest -- MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return best_rf_model, (mse, rmse, r2)
    except Exception as e:
        logging.error("Error in train_random_forest_with_auto_optimization: " + str(e))
        raise
