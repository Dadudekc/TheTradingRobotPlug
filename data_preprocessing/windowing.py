"""
data_preprocessing/windowing.py
-------------------------------
Provides a function to create windowed data suitable for sequence models such as LSTM.
"""

import numpy as np

def create_windowed_data(X, y, n_steps):
    """
    Transforms the feature matrix and target vector into a windowed dataset.

    Parameters:
        X (array-like): 2D array of features.
        y (array-like): 1D array of target values.
        n_steps (int): Number of time steps per sample.
    
    Returns:
        X_new (np.array): 3D array of shape (samples, n_steps, features).
        y_new (np.array): 1D array of target values corresponding to each window.
    """
    try:
        X, y = np.array(X), np.array(y)
        X_new, y_new = [], []
        for i in range(len(X) - n_steps):
            X_new.append(X[i:i + n_steps])
            y_new.append(y[i + n_steps])
        return np.array(X_new), np.array(y_new)
    except Exception as e:
        raise ValueError("Error in create_windowed_data: " + str(e))
