"""
model_training/linear_regression.py
-----------------------------------
Trains a Ridge regression model using randomized hyperparameter search.
"""

import logging
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression_with_auto_optimization(X_train, y_train, X_val, y_val):
    """
    Trains a Ridge regression model with auto-optimization via RandomizedSearchCV.

    Parameters:
        X_train (np.array): Training features.
        y_train (np.array): Training targets.
        X_val (np.array): Validation features.
        y_val (np.array): Validation targets.
    
    Returns:
        best_model: The optimized Ridge regression model.
        metrics (tuple): (mse, rmse, r2) computed on the validation set.
    """
    try:
        param_grid = {'alpha': np.logspace(-4, 0, 50)}
        ridge = Ridge()
        randomized_search = RandomizedSearchCV(
            estimator=ridge,
            param_distributions=param_grid,
            n_iter=10,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        randomized_search.fit(X_train, y_train)
        best_model = randomized_search.best_estimator_
        y_pred_val = best_model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred_val)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred_val)
        logging.info(f"Linear Regression -- MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return best_model, (mse, rmse, r2)
    except Exception as e:
        logging.error("Error in train_linear_regression_with_auto_optimization: " + str(e))
        raise
