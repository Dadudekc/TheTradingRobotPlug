# -------------------------------------------------------------------
# File Path: C:\Projects\TradingRobotPlug\Scripts\model_training\utils\optuna_helpers.py
# Description: Helper module for Optuna-based hyperparameter optimization
#              for Linear, Ridge, Lasso, and LSTM models.
# -------------------------------------------------------------------

import sys
from pathlib import Path
import optuna
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Adjust the Python path dynamically if necessary
# (This is generally not recommended; better to structure your packages properly)
# script_dir = Path(__file__).resolve().parent
# project_root = Path('C:/Projects/TradingRobotPlug')
# utilities_dir = project_root / "Scripts" / "Utilities"
# model_training_dir = project_root / "Scripts" / "model_training"
# sys.path.append(str(utilities_dir))
# sys.path.append(str(model_training_dir))

# Import LSTM creation function
# Ensure that lstm_helpers.py exists and contains create_lstm_model
# from lstm_helpers import create_lstm_model  # Adjust the import as needed

# For demonstration, let's assume create_lstm_model is defined here
# If it's in a different file, adjust the import accordingly
def create_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
    """
    Create a simple LSTM model for demonstration purposes.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        lstm_units (int, optional): Number of LSTM units. Defaults to 50.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
        learning_rate (float, optional): Learning rate. Defaults to 0.001.

    Returns:
        tensorflow.keras.Model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    return model


# Define the objective function for Optuna to optimize regression models
def objective_regression(trial, X_train, y_train):
    """
    Optuna objective function for hyperparameter tuning of regression models.

    Args:
        trial (optuna.Trial): Optuna trial object to suggest hyperparameters.
        X_train (numpy.ndarray): Training feature set.
        y_train (numpy.ndarray): Training target set.

    Returns:
        float: Cross-validated mean squared error for the trial's model.
    """
    
    # Choose between different model types (Ridge, Lasso)
    model_type = trial.suggest_categorical('model_type', ['ridge', 'lasso'])
    
    # Suggest the hyperparameter 'alpha' (regularization strength)
    alpha = trial.suggest_float('alpha', 1e-5, 10.0, log=True)
    
    # Define the regression model
    if model_type == 'ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        model = Lasso(alpha=alpha)
    
    # Perform cross-validation to evaluate model performance
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    # We minimize the negative mean squared error, so we return the positive MSE
    return -1 * scores.mean()


# Define the objective function for LSTM models
def objective_lstm(trial, X_train, y_train, X_val, y_val, input_shape):
    """
    Optuna objective function for hyperparameter tuning of LSTM models.

    Args:
        trial (optuna.Trial): Optuna trial object to suggest hyperparameters.
        X_train (numpy.ndarray): Training feature set.
        y_train (numpy.ndarray): Training target set.
        X_val (numpy.ndarray): Validation feature set.
        y_val (numpy.ndarray): Validation target set.
        input_shape (tuple): Shape of the input data.

    Returns:
        float: Validation loss or training loss if no validation data is provided.
    """
    # Suggest hyperparameters
    lstm_units = trial.suggest_int('lstm_units', 50, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    # Create the LSTM model using the suggested hyperparameters
    model = create_lstm_model(input_shape, lstm_units=lstm_units, 
                              dropout_rate=dropout_rate, learning_rate=learning_rate)

    # Train the model on X_train, y_train
    if X_val is not None and y_val is not None:
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)
        loss = history.history['val_loss'][-1]  # Minimize validation loss
    else:
        history = model.fit(X_train, y_train, epochs=10, verbose=0)
        loss = history.history['loss'][-1]  # Minimize training loss if no validation data

    return loss


def tune_hyperparameters(X_train, y_train, X_val=None, y_val=None, input_shape=None, n_trials=20, model_type='lstm'):
    """
    Tune hyperparameters for different models using Optuna. Supports LSTM, Ridge, and Lasso models.

    Args:
        X_train (numpy.ndarray): Training feature set.
        y_train (numpy.ndarray): Training target set.
        X_val (numpy.ndarray, optional): Validation feature set. Default is None.
        y_val (numpy.ndarray, optional): Validation target set. Default is None.
        input_shape (tuple, optional): Input shape for the LSTM model. Default is None.
        n_trials (int): Number of Optuna trials.
        model_type (str): The type of model to optimize. Can be 'lstm', 'ridge', or 'lasso'. Default is 'lstm'.

    Returns:
        dict: Best hyperparameters found by Optuna.
    """
    if model_type == 'lstm':
        if input_shape is None:
            raise ValueError("input_shape must be provided for LSTM hyperparameter tuning.")
        # For LSTM models
        def objective(trial):
            return objective_lstm(trial, X_train, y_train, X_val, y_val, input_shape)

    elif model_type in ['ridge', 'lasso']:
        # For linear regression models (Ridge, Lasso)
        def objective(trial):
            return objective_regression(trial, X_train, y_train)

    else:
        raise ValueError("Unsupported model_type. Choose from 'lstm', 'ridge', 'lasso'.")

    # Create an Optuna study object to minimize the objective
    study = optuna.create_study(direction='minimize')
    
    # Run the optimization process for the defined number of trials
    study.optimize(objective, n_trials=n_trials)
    
    # Return the best hyperparameters
    return study.best_params
