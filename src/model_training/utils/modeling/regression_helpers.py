# -------------------------------------------------------------------
# File Path: C:/TheTradingRobotPlug/Scripts/model_training/utils/regression_helpers.py
# Description: Comprehensive helper module for training, tuning, and 
#              saving regression models, including Linear, Ridge, 
#              and Lasso regression. Supports cross-validation, 
#              hyperparameter tuning, and model persistence.
# -------------------------------------------------------------------

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import logging


# -------------------------------------------------------------------
# Function: select_regression_model
# Description: Dynamically selects and returns the regression model 
#              based on the model type (linear, ridge, lasso).
# -------------------------------------------------------------------
def select_regression_model(model_type='linear', alpha=1.0):
    """
    Select the appropriate regression model based on the model type.

    Args:
        model_type (str): Type of the regression model ('linear', 'ridge', 'lasso').
        alpha (float): Regularization strength for Ridge and Lasso.

    Returns:
        model (sklearn model): Instantiated regression model.
    """
    if model_type == 'ridge':
        return Ridge(alpha=alpha)
    elif model_type == 'lasso':
        return Lasso(alpha=alpha)
    else:
        return LinearRegression()


# -------------------------------------------------------------------
# Function: train_regression_model
# Description: Trains a regression model with optional cross-validation 
#              and hyperparameter tuning.
# -------------------------------------------------------------------
def train_regression_model(X_train, y_train, model_type='linear', alpha=1.0, 
                           use_cross_validation=False, cv_folds=5, tune_hyperparams=False):
    """
    Train a regression model with optional cross-validation and hyperparameter tuning.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target.
        model_type (str): Type of regression model ('linear', 'ridge', 'lasso').
        alpha (float): Regularization strength for Ridge and Lasso.
        use_cross_validation (bool): Whether to perform cross-validation.
        cv_folds (int): Number of folds for cross-validation.
        tune_hyperparams (bool): Whether to perform hyperparameter tuning.

    Returns:
        model (sklearn model): Trained regression model.
    """
    model = select_regression_model(model_type, alpha)

    if tune_hyperparams:
        # Set hyperparameter tuning grids for Ridge and Lasso
        param_grid = {
            'ridge': {'alpha': [0.1, 0.5, 1.0, 10]},
            'lasso': {'alpha': [0.1, 0.5, 1.0, 10]}
        }.get(model_type, {})

        if param_grid:
            model = GridSearchCV(model, param_grid, cv=cv_folds, scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            logging.info(f"Best parameters for {model_type}: {model.best_params_}")
        else:
            logging.warning(f"No hyperparameter grid available for {model_type}.")

    if use_cross_validation:
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
        logging.info(f"{model_type} Cross-validation MSE scores: {cv_scores}")
        logging.info(f"Average CV MSE: {cv_scores.mean()}")

    # Train the model
    model.fit(X_train, y_train)

    return model


# -------------------------------------------------------------------
# Function: save_trained_model
# Description: Saves the trained model to a file for later use.
# -------------------------------------------------------------------
def save_trained_model(model, model_save_path, model_type):
    """
    Save the trained model to a file.

    Args:
        model: Trained model object.
        model_save_path (str): Path to save the model.
        model_type (str): Type of the model (e.g., 'linear', 'ridge', 'lasso').

    Returns:
        None
    """
    try:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        save_path = os.path.join(model_save_path, f"{model_type}_model.pkl")
        joblib.dump(model, save_path)
        logging.info(f"Model saved at {save_path}")
    except Exception as e:
        logging.error(f"Error saving the model: {str(e)}")


# -------------------------------------------------------------------
# Function: load_saved_model
# Description: Loads a previously saved model from a file.
# -------------------------------------------------------------------
def load_saved_model(model_save_path, model_type):
    """
    Load a saved model from a file.

    Args:
        model_save_path (str): Path where the model is saved.
        model_type (str): Type of the model (e.g., 'linear', 'ridge', 'lasso').

    Returns:
        model: Loaded model object.
    """
    try:
        model_file = os.path.join(model_save_path, f"{model_type}_model.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        model = joblib.load(model_file)
        logging.info(f"Model loaded from {model_file}")
        return model
    except Exception as e:
        logging.error(f"Error loading the model: {str(e)}")
        return None


# -------------------------------------------------------------------
# Function: evaluate_regression_model
# Description: Evaluates the regression model using MSE and R².
# -------------------------------------------------------------------
def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate a regression model by calculating MSE and R² scores.

    Args:
        model: Trained regression model.
        X_test (numpy.ndarray): Test feature set.
        y_test (numpy.ndarray): Test target set.

    Returns:
        mse (float): Mean Squared Error.
        r2 (float): R² score.
    """
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Regression Model Performance: MSE: {mse}, R²: {r2}")
        return mse, r2
    except Exception as e:
        logging.error(f"Error evaluating the model: {str(e)}")
        return None, None
