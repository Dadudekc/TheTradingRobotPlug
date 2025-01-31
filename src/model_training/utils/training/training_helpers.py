# -------------------------------------------------------------------
# File Path: C:\Projects\TradingRobotPlug\Scripts\model_training\utils\Training\training_helpers.py
# Description: Defines functions to train various machine learning models,
#              including Gradient Boosting, ARIMA, LSTM, Linear Regression,
#              Random Forest, Decision Tree, SVM, and Transfer Learning.
# -------------------------------------------------------------------

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    LSTM,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import (
    VGG16,
    VGG19,
    ResNet50,
    ResNet50V2,
    ResNet101,
    ResNet101V2,
    InceptionV3,
    Xception,
    InceptionResNetV2,
    MobileNet,
    MobileNetV2,
    DenseNet121,
    DenseNet169,
    NASNetMobile,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7
)

# -------------------------------------------------------------------
# Logging Configuration
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Function: train_gradient_boosting_model
# Description: Trains a Gradient Boosting Classifier and evaluates its accuracy.
# -------------------------------------------------------------------
def train_gradient_boosting_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    logger: Optional[logging.Logger] = None
) -> Tuple[GradientBoostingClassifier, float]:
    """
    Train a Gradient Boosting Classifier using the provided features and labels.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        test_size (float, optional): Fraction of data to be used for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        n_estimators (int, optional): Number of boosting stages. Defaults to 100.
        learning_rate (float, optional): Learning rate for gradient boosting. Defaults to 0.1.
        logger (logging.Logger, optional): Logger instance. Defaults to None.

    Returns:
        Tuple[GradientBoostingClassifier, float]: Trained Gradient Boosting model and accuracy on the test set.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting Gradient Boosting Classifier training.")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize the Gradient Boosting model
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )

    # Train the model
    model.fit(X_train, y_train)
    logger.info("Gradient Boosting Classifier training completed.")

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Gradient Boosting Classifier Accuracy: {accuracy:.4f}")

    return model, accuracy

# -------------------------------------------------------------------
# Function: train_arima_model
# Description: Trains an ARIMA/SARIMAX model and saves the model along with
#              performance metrics such as RMSE.
# -------------------------------------------------------------------
def train_arima_model(
    data: np.ndarray,
    symbol: str,
    model_path: str,
    order: Tuple[int, int, int],
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[float, np.ndarray]:
    """
    Train an ARIMA or SARIMAX model and evaluate its performance.

    Args:
        data (np.ndarray): Time series data.
        symbol (str): Symbol or identifier for the dataset.
        model_path (str): Path to save the trained model.
        order (Tuple[int, int, int]): The (p, d, q) order of the model.
        seasonal_order (Optional[Tuple[int, int, int, int]], optional): The seasonal order of the model. Defaults to None.
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.

    Returns:
        Tuple[float, np.ndarray]: Root Mean Squared Error (RMSE) and forecasted values.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(
        f"Training ARIMA model for {symbol} with order: {order} "
        f"and seasonal_order: {seasonal_order}"
    )

    try:
        # Choose model based on presence of seasonal order
        if seasonal_order:
            model = SARIMAX(
                data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            model = ARIMA(data, order=order)

        # Fit the model
        model_fit = model.fit()
        forecast_steps = 10
        forecast = model_fit.forecast(steps=forecast_steps)

        # Calculate RMSE using the last 'forecast_steps' data points as a simple example
        if len(data) < forecast_steps:
            raise ValueError("Data length is shorter than the number of forecast steps.")
        rmse = np.sqrt(mean_squared_error(data[-forecast_steps:], forecast))
        logger.info(f"ARIMA Model RMSE: {rmse:.4f}")

        # Save the trained model
        model_fit.save(model_path)
        logger.info(f"ARIMA model saved at {model_path}")

        return rmse, forecast

    except Exception as e:
        logger.error(f"Error training ARIMA model for {symbol}: {e}")
        raise

# -------------------------------------------------------------------
# Function: build_lstm_model
# Description: Builds an LSTM model with configurable layers, dropout,
#              and learning rate for time series prediction.
# -------------------------------------------------------------------
def build_lstm_model(
    input_shape: Tuple[int, int],
    lstm_units: int,
    dropout_rate: float,
    learning_rate: float,
    logger: Optional[logging.Logger] = None
) -> tf.keras.Model:
    """
    Build an LSTM model for regression tasks.

    Args:
        input_shape (Tuple[int, int]): Shape of the input data (timesteps, features).
        lstm_units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate between layers.
        learning_rate (float): Learning rate for the optimizer.
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        BatchNormalization(),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1)  # Output layer for regression
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error'
    )
    logger.info(f"LSTM model compiled with learning_rate={learning_rate}.")

    return model

# -------------------------------------------------------------------
# Function: train_lstm_model
# Description: Trains an LSTM model with early stopping and learning
#              rate reduction on plateau.
# -------------------------------------------------------------------
def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lstm_units: int,
    dropout_rate: float,
    learning_rate: float,
    logger: Optional[logging.Logger] = None,
    epochs: int = 50,
    batch_size: int = 32
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train an LSTM model with early stopping and learning rate reduction.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        lstm_units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate.
        learning_rate (float): Learning rate for the optimizer.
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        batch_size (int, optional): Batch size for training. Defaults to 32.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: Trained LSTM model and training history.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        input_shape = (X_train.shape[1], X_train.shape[2])
        logger.info(
            f"Training LSTM model with input shape: {input_shape}, "
            f"LSTM Units: {lstm_units}, Dropout: {dropout_rate}, "
            f"Learning Rate: {learning_rate}"
        )

        model = build_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate, logger)

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        return model, history

    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        raise

# -------------------------------------------------------------------
# Function: train_linear_regression_model
# Description: Trains a linear regression (or Ridge/Lasso) model and
#              saves the model along with performance evaluations.
# -------------------------------------------------------------------
def train_linear_regression_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = 'linear',
    alpha: float = 1.0,
    logger: Optional[logging.Logger] = None
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a linear, Ridge, or Lasso regression model and evaluate its performance.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        model_type (str, optional): Type of regression ('linear', 'ridge', 'lasso'). Defaults to 'linear'.
        alpha (float, optional): Regularization strength for Ridge/Lasso. Defaults to 1.0.
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.

    Returns:
        Tuple[Any, Dict[str, float]]: Trained regression model and a dictionary of performance metrics.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        model_type_lower = model_type.lower()
        if model_type_lower == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type_lower == 'lasso':
            model = Lasso(alpha=alpha)
        elif model_type_lower == 'linear':
            model = LinearRegression()
        else:
            raise ValueError("Invalid model_type. Choose from 'linear', 'ridge', or 'lasso'.")

        model.fit(X_train, y_train)
        logger.info(f"{model_type.capitalize()} Regression training completed.")

        y_pred_val = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred_val)
        r2 = r2_score(y_val, y_pred_val)

        logger.info(
            f"{model_type.capitalize()} Regression - MSE: {mse:.4f}, R²: {r2:.4f}"
        )

        return model, {'mse': mse, 'r2': r2}

    except Exception as e:
        logger.error(f"Error training {model_type} regression model: {e}")
        raise

# -------------------------------------------------------------------
# Function: save_trained_regression_model
# Description: Saves a trained linear regression model to disk.
# -------------------------------------------------------------------
def save_trained_regression_model(
    model: Any,
    model_save_path: str,
    model_type: str,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Save a trained regression model to the specified path.

    Args:
        model (Any): Trained regression model.
        model_save_path (str): Directory path to save the model.
        model_type (str): Type of regression model ('linear', 'ridge', 'lasso').
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.

    Returns:
        str: Path to the saved model file.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        save_dir = Path(model_save_path).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        model_file = save_dir / f"{model_type.lower()}_regression_model.pkl"
        joblib.dump(model, model_file)

        logger.info(f"{model_type.capitalize()} regression model saved at {model_file}")
        return str(model_file)

    except Exception as e:
        logger.error(f"Error saving {model_type} regression model: {e}")
        raise

# -------------------------------------------------------------------
# Function: train_random_forest_model
# Description: Trains a Random Forest model and logs performance.
# -------------------------------------------------------------------
def train_random_forest_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor and evaluate its performance.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.
        params (Dict[str, Any]): Model parameters (e.g., 'n_estimators', 'max_depth').
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.

    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting Random Forest Regressor training.")

    try:
        rf_model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            random_state=params.get('random_state', 42)
        )

        rf_model.fit(X_train, y_train)
        logger.info("Random Forest Regressor training completed.")

        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Random Forest Regressor MSE: {mse:.4f}")

        return rf_model

    except Exception as e:
        logger.error(f"Error training Random Forest Regressor: {e}")
        raise

# -------------------------------------------------------------------
# Function: train_decision_tree_model
# Description: Trains a Decision Tree model and logs performance.
# -------------------------------------------------------------------
def train_decision_tree_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> DecisionTreeRegressor:
    """
    Train a Decision Tree Regressor and evaluate its performance.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.
        params (Dict[str, Any]): Model parameters (e.g., 'max_depth').
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.

    Returns:
        DecisionTreeRegressor: Trained Decision Tree model.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting Decision Tree Regressor training.")

    try:
        dt_model = DecisionTreeRegressor(
            max_depth=params.get('max_depth', None),
            random_state=params.get('random_state', 42)
        )

        dt_model.fit(X_train, y_train)
        logger.info("Decision Tree Regressor training completed.")

        y_pred = dt_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Decision Tree Regressor MSE: {mse:.4f}")

        return dt_model

    except Exception as e:
        logger.error(f"Error training Decision Tree Regressor: {e}")
        raise

# -------------------------------------------------------------------
# Function: train_svm_model
# Description: Trains an SVM model with the provided training data.
# -------------------------------------------------------------------
def train_svm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    logger: Optional[logging.Logger] = None,
    kernel: str = 'rbf',
    C: float = 1e3,
    gamma: float = 0.1
) -> SVR:
    """
    Train a Support Vector Machine (SVM) Regressor.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.
        kernel (str, optional): Kernel type. Defaults to 'rbf'.
        C (float, optional): Regularization parameter. Defaults to 1e3.
        gamma (float, optional): Kernel coefficient. Defaults to 0.1.

    Returns:
        SVR: Trained SVM model.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting SVM Regressor training.")

    try:
        model = SVR(kernel=kernel, C=C, gamma=gamma)
        model.fit(X_train, y_train)
        logger.info("SVM Regressor training completed successfully.")
        return model

    except Exception as e:
        logger.error(f"Error training SVM Regressor: {e}")
        raise

# -------------------------------------------------------------------
# Function: build_transfer_learning_model
# Description: Builds a transfer learning model by leveraging a pre-trained base.
# -------------------------------------------------------------------
def build_transfer_learning_model(
    base_model: tf.keras.Model,
    num_classes: int,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
    logger: Optional[logging.Logger] = None
) -> tf.keras.Model:
    """
    Build a transfer learning model by adding custom layers on top of a pre-trained base model.

    Args:
        base_model (tf.keras.Model): Pre-trained model without the top classification layer.
        num_classes (int): Number of output classes.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.5.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.

    Returns:
        tf.keras.Model: Compiled transfer learning model.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    base_model.trainable = False  # Freeze the base model layers
    logger.info("Base model layers frozen.")

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    logger.info("Transfer Learning model compiled.")

    return model

# -------------------------------------------------------------------
# Function: train_transfer_learning_model
# Description: Trains a Transfer Learning model with given data.
# -------------------------------------------------------------------
def train_transfer_learning_model(
    model: tf.keras.Model,
    train_generator: Any,
    validation_generator: Any,
    epochs: int = 10,
    callbacks: Optional[List[Any]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train a Transfer Learning model using data generators.

    Args:
        model (tf.keras.Model): Transfer Learning model to train.
        train_generator (Any): Training data generator.
        validation_generator (Any): Validation data generator.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        callbacks (Optional[List[Any]], optional): List of Keras callbacks. Defaults to None.
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: Trained Transfer Learning model and training history.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting Transfer Learning model training.")

    try:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        logger.info("Transfer Learning model training completed.")
        return model, history

    except Exception as e:
        logger.error(f"Error during Transfer Learning training: {e}")
        raise

# -------------------------------------------------------------------
# Function: save_trained_transfer_learning_model
# Description: Saves a trained Transfer Learning model to disk.
# -------------------------------------------------------------------
def save_trained_transfer_learning_model(
    model: tf.keras.Model,
    model_save_path: str,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Save a trained Transfer Learning model to the specified path.

    Args:
        model (tf.keras.Model): Trained Transfer Learning model.
        model_save_path (str): File path to save the model (e.g., 'models/transfer_learning_model.h5').
        logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.

    Returns:
        str: Path to the saved model file.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        save_path = Path(model_save_path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model.save(save_path)
        logger.info(f"Transfer Learning model saved at {save_path}")
        return str(save_path)

    except Exception as e:
        logger.error(f"Error saving Transfer Learning model: {e}")
        raise

# -------------------------------------------------------------------
# ARIMA Model Trainer Class
# Description: Manages the training process of ARIMA models, including tracking metrics and generating summaries.
# -------------------------------------------------------------------
class ARIMAModelTrainer:
    def __init__(self, data: np.ndarray, symbol: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the ARIMAModelTrainer.

        Args:
            data (np.ndarray): Time series data.
            symbol (str): Symbol or identifier for the dataset.
            logger (Optional[logging.Logger], optional): Logger instance. Defaults to None.
        """
        self.data = data
        self.symbol = symbol
        self.logger = logger or logging.getLogger(__name__)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.rmse_scores: List[float] = []
        self.best_rmse: float = float('inf')
        self.best_model_path: Optional[str] = None
        self.training_stopped: bool = False

    def train(
        self,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        model_save_dir: str = 'models/arima'
    ) -> Tuple[float, np.ndarray]:
        """
        Train the ARIMA/SARIMAX model and track performance.

        Args:
            order (Tuple[int, int, int]): The (p, d, q) order of the model.
            seasonal_order (Optional[Tuple[int, int, int, int]], optional): The seasonal order of the model. Defaults to None.
            model_save_dir (str, optional): Directory to save the trained model. Defaults to 'models/arima'.

        Returns:
            Tuple[float, np.ndarray]: Root Mean Squared Error (RMSE) and forecasted values.
        """
        self.start_time = time.time()
        self.logger.info(f"Starting ARIMA training for {self.symbol}")

        try:
            model_save_dir_path = Path(model_save_dir).resolve()
            model_save_dir_path.mkdir(parents=True, exist_ok=True)
            model_path = model_save_dir_path / f'arima_model_{self.symbol}.pkl'

            rmse, forecast = train_arima_model(
                data=self.data,
                symbol=self.symbol,
                model_path=str(model_path),
                order=order,
                seasonal_order=seasonal_order,
                logger=self.logger
            )
            self.rmse_scores.append(rmse)
            self.logger.info(f"ARIMA Model RMSE: {rmse:.4f}")

            if rmse < self.best_rmse:
                self.best_rmse = rmse
                self.best_model_path = str(model_path)
                self.logger.info(
                    f"New best model with RMSE: {rmse:.4f} saved at {model_path}"
                )

            return rmse, forecast

        except Exception as e:
            self.logger.error(f"Error during ARIMA training: {e}")
            raise

        finally:
            self.end_time = time.time()
            self.generate_summary()

    def stop_training(self):
        """
        Stop the ARIMA training process and generate a summary.
        """
        self.training_stopped = True
        self.end_time = time.time()
        self.logger.info("ARIMA training stopped by user.")
        self.generate_summary()

    def generate_summary(self):
        """
        Generate and log a summary of the ARIMA training session.
        """
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
        else:
            duration = 0.0
        average_rmse = (
            sum(self.rmse_scores) / len(self.rmse_scores)
            if self.rmse_scores else float('inf')
        )
        summary = (
            f"Training Summary for {self.symbol}:\n"
            f"Total Training Time: {duration:.2f} seconds\n"
            f"Number of Models Trained: {len(self.rmse_scores)}\n"
            f"Average RMSE: {average_rmse:.4f}\n"
            f"Best RMSE: {self.best_rmse:.4f}\n"
            f"Best Model Save Location: {self.best_model_path if self.best_model_path else 'Not saved'}"
        )
        self.logger.info(summary)
        print(summary)  # Or integrate with your GUI to display the summary

# -------------------------------------------------------------------
# Function: load_pretrained_model
# Description: Loads a pretrained model with customizable top layers and configurations.
# -------------------------------------------------------------------
def load_pretrained_model(
    model_name: str,
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 1000,
    weights: str = 'imagenet',
    include_top: bool = False,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-4,
    base_trainable: bool = False,
    logger: Optional[logging.Logger] = None
) -> tf.keras.Model:
    """
    Load a pretrained model from Keras Applications with the option to customize the top layers.

    Args:
        model_name (str): Name of the pre-trained model to load.
        input_shape (tuple): Input shape of the images.
        num_classes (int): Number of classes for the output layer.
        weights (str): Pre-trained weights to load.
        include_top (bool): Whether to include the top (fully connected) layers.
        dropout_rate (float): Dropout rate for regularization in custom layers.
        learning_rate (float): Learning rate for the optimizer.
        base_trainable (bool): If False, freeze the base layers. If True, make them trainable.
        logger (Optional[logging.Logger], optional): Logger instance for logging progress. Defaults to None.

    Returns:
        tf.keras.Model: The customized pre-trained model ready for training.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Mapping model names to Keras applications
    model_dict = {
        'vgg16': VGG16,
        'vgg19': VGG19,
        'resnet50': ResNet50,
        'resnet50v2': ResNet50V2,
        'resnet101': ResNet101,
        'resnet101v2': ResNet101V2,
        'inceptionv3': InceptionV3,
        'xception': Xception,
        'inceptionresnetv2': InceptionResNetV2,
        'mobilenet': MobileNet,
        'mobilenetv2': MobileNetV2,
        'densenet121': DenseNet121,
        'densenet169': DenseNet169,
        'nasnetmobile': NASNetMobile,
        'efficientnetb0': EfficientNetB0,
        'efficientnetb1': EfficientNetB1,
        'efficientnetb2': EfficientNetB2,
        'efficientnetb3': EfficientNetB3,
        'efficientnetb4': EfficientNetB4,
        'efficientnetb5': EfficientNetB5,
        'efficientnetb6': EfficientNetB6,
        'efficientnetb7': EfficientNetB7,
    }

    # Check if the model name is valid
    if model_name.lower() not in model_dict:
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {list(model_dict.keys())}")

    # Load the base model
    base_model_class = model_dict[model_name.lower()]
    base_model = base_model_class(input_shape=input_shape, weights=weights, include_top=include_top)

    # Set the base model layers' trainability
    base_model.trainable = base_trainable
    logger.info(f"Loaded {model_name} with trainable base: {base_trainable}")

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)  # Optional dropout layer for regularization
    activation = 'softmax' if num_classes > 1 else 'sigmoid'
    predictions = Dense(num_classes, activation=activation)(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    loss = 'categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    logger.info(f"{model_name} compiled with learning rate={learning_rate}, dropout_rate={dropout_rate}")

    return model

def preprocess_data_for_lstm(
    data: pd.DataFrame,
    target_column: str = 'close',
    time_steps: int = 10,
    features: Optional[list] = None,
    test_size: float = 0.2,
    scaler_type: str = 'minmax',
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, any, any]:
    """
    Preprocess data for LSTM model training.

    Args:
        data (pd.DataFrame): Input data as a pandas DataFrame.
        target_column (str): Name of the target column in the DataFrame.
        time_steps (int): Number of time steps to consider for each sequence.
        features (list, optional): List of feature column names to use. If None, all columns except target are used.
        test_size (float): Proportion of data to use as the validation set.
        scaler_type (str): Type of scaler to use ('minmax' or 'standard').
        logger (Optional[logging.Logger], optional): Logger for logging progress. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, any, any]:
            - X_train: Training feature sequences.
            - X_val: Validation feature sequences.
            - y_train: Training target sequences.
            - y_val: Validation target sequences.
            - feature_scaler: Scaler used for feature scaling.
            - target_scaler: Scaler used for target scaling.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Starting data preprocessing for LSTM model.")
    
    # Set features if not provided
    if features is None:
        features = [col for col in data.columns if col != target_column]
    logger.info(f"Features selected for preprocessing: {features}")

    # Initialize scalers
    scaler_class = MinMaxScaler if scaler_type == 'minmax' else StandardScaler
    feature_scaler = scaler_class()
    target_scaler = scaler_class()

    # Scale features and target
    scaled_features = feature_scaler.fit_transform(data[features])
    scaled_target = target_scaler.fit_transform(data[[target_column]])

    # Generate sequences for features and target
    def create_sequences(data, time_steps):
        sequences = []
        for i in range(len(data) - time_steps):
            sequences.append(data[i:i + time_steps])
        return np.array(sequences)

    X = create_sequences(scaled_features, time_steps)
    y = create_sequences(scaled_target, time_steps)

    # Split into training and validation sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info("Data preprocessing completed successfully.")
    return X_train, X_val, y_train, y_val, feature_scaler, target_scaler