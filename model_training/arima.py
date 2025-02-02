"""
model_training/arima.py
-----------------------
Provides functions to train an ARIMA model for time series forecasting.
"""

import logging
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import threading

def train_arima_model(close_prices, order=(5, 1, 0)):
    """
    Trains an ARIMA model on the provided time series data.

    Parameters:
        close_prices (array-like): Time series data (e.g., closing prices).
        order (tuple): ARIMA order (p, d, q). Default is (5, 1, 0).
    
    Returns:
        model_fit: The fitted ARIMA model.
        predictions (list): Forecasted values for the test period.
        mse (float): Mean squared error computed on the test set.
    """
    try:
        close_prices = np.array(close_prices)
        n = len(close_prices)
        train_size = int(n * 0.8)
        train, test = close_prices[:train_size], close_prices[train_size:]
        history = list(train)
        predictions = []

        for t in range(len(test)):
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])

        mse = mean_squared_error(test, predictions)
        logging.info(f"ARIMA Model Test MSE: {mse:.4f}")
        return model_fit, predictions, mse
    except Exception as e:
        logging.error("Error in train_arima_model: " + str(e))
        raise

def train_arima_model_in_background(close_prices, order=(5, 1, 0), callback=None):
    """
    Trains an ARIMA model in a background thread.

    Parameters:
        close_prices (array-like): Time series data.
        order (tuple): ARIMA order.
        callback (function): Optional function to call with (model_fit, predictions, mse) when done.
    """
    def background_train():
        try:
            model_fit, predictions, mse = train_arima_model(close_prices, order=order)
            if callback:
                callback(model_fit, predictions, mse)
        except Exception as e:
            logging.error("Background ARIMA training failed: " + str(e))
    
    thread = threading.Thread(target=background_train, daemon=True)
    thread.start()
