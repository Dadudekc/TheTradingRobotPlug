# -------------------------------------------------------------------
# File Path: utils/lstm_helpers.py
# Description: Helper module for creating, training, and saving LSTM models.
# -------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def create_lstm_model(input_shape, lstm_units=100, dropout_rate=0.3, learning_rate=0.001):
    """Create and compile an LSTM model."""
    try:
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(lstm_units),
            Dropout(dropout_rate),
            Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model
    except Exception as e:
        raise RuntimeError(f"Error creating LSTM model: {str(e)}")

def evaluate_lstm_model(model, X_test, y_test, scaler):
    """Evaluate LSTM model performance and return metrics."""
    predictions = model.predict(X_test)
    num_features = scaler.n_features_in_
    predictions_reshaped = np.zeros((predictions.shape[0], num_features))
    y_test_reshaped = np.zeros((y_test.shape[0], num_features))
    predictions_reshaped[:, 0] = predictions.flatten()
    y_test_reshaped[:, 0] = y_test.flatten()

    predictions_scaled_back = scaler.inverse_transform(predictions_reshaped)[:, 0]
    y_test_scaled_back = scaler.inverse_transform(y_test_reshaped)[:, 0]

    mse = mean_squared_error(y_test_scaled_back, predictions_scaled_back)
    mae = mean_absolute_error(y_test_scaled_back, predictions_scaled_back)
    r2 = r2_score(y_test_scaled_back, predictions_scaled_back)

    return mse, mae, r2

def calculate_metrics(y_true, y_pred, logger=None):
    """
    Calculate evaluation metrics for regression model performance.
    
    Args:
        y_true (np.array): The ground truth target values.
        y_pred (np.array): The predicted target values from the model.
        logger (logging.Logger, optional): Logger for logging information. Defaults to None.
    
    Returns:
        metrics (dict): Dictionary containing the calculated metrics:
                        - 'mse': Mean Squared Error
                        - 'mae': Mean Absolute Error
                        - 'r2': R-squared (coefficient of determination)
    """
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate R-squared (R²)
    r2 = r2_score(y_true, y_pred)
    
    if logger:
        logger.info(f"Metrics calculated: MSE = {mse}, MAE = {mae}, R² = {r2}")
    
    # Return all metrics in a dictionary
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics
