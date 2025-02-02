"""
evaluation/metrics.py
---------------------
Provides functions to calculate model evaluation metrics.
"""

import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             explained_variance_score, max_error, mean_absolute_percentage_error)

def calculate_model_accuracy(model, X_test, y_test):
    """
    Calculates accuracy for models that support the score() method.
    For regression, returns the R² score as a percentage.
    
    Args:
        model: Trained model.
        X_test (array-like): Test features.
        y_test (array-like): True target values.
    
    Returns:
        float: Accuracy percentage.
    """
    try:
        if hasattr(model, 'score'):
            return model.score(X_test, y_test) * 100.0
        return 0.0
    except Exception as e:
        raise RuntimeError("Error calculating model accuracy: " + str(e))

def calculate_model_metrics(y_true, y_pred):
    """
    Calculates a set of regression metrics.
    
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.
    
    Returns:
        dict: Dictionary containing MAE, MSE, RMSE, R², Explained Variance, Max Error, and MAPE.
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred)
        max_err = max_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'Explained Variance': explained_var,
            'Max Error': max_err,
            'MAPE': mape
        }
    except Exception as e:
        raise RuntimeError("Error calculating model metrics: " + str(e))

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculates key regression metrics (RMSE, MAE, R²).
    
    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    
    Returns:
        dict: Dictionary with RMSE, MAE, and R².
    """
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    except Exception as e:
        raise RuntimeError("Error calculating regression metrics: " + str(e))
