# -------------------------------------------------------------------
# File Path: C:\TheTradingRobotPlug\Scripts\model_training\utils\training_helpers.py
# Description: Defines functions to train various machine learning models,
#              including Gradient Boosting, ARIMA, LSTM, Linear Regression,
#              Random Forest, Decision Tree, SVM, and Transfer Learning.
# -------------------------------------------------------------------

import os
import logging

import joblib
import numpy as np
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import (accuracy_score, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, LSTM, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from sklearn.svm import SVR

# Import Transfer Learning specific modules
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam


# -------------------------------------------------------------------
# Function: train_gradient_boosting_model
# Description: Trains a Gradient Boosting Classifier and evaluates its accuracy.
# -------------------------------------------------------------------
def train_gradient_boosting_model(X, y, test_size=0.2, random_state=42,
                                  n_estimators=100, learning_rate=0.1):
    """
    Train a Gradient Boosting Classifier using the provided features and labels.

    Args:
        X (numpy.ndarray or pandas.DataFrame): Feature matrix.
        y (numpy.ndarray or pandas.Series): Target labels.
        test_size (float, optional): Fraction of data to be used for testing. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        n_estimators (int, optional): Number of boosting stages. Defaults to 100.
        learning_rate (float, optional): Learning rate for gradient boosting. Defaults to 0.1.

    Returns:
        tuple: Trained Gradient Boosting model and accuracy on the test set.
    """
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

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

# -------------------------------------------------------------------
# Function: train_arima_model
# Description: Trains an ARIMA/SARIMAX model and saves the model along with
#              performance metrics such as RMSE.
# -------------------------------------------------------------------
def train_arima_model(data, symbol, model_path, order,
                      seasonal_order=None, logger=None):
    """
    Train an ARIMA or SARIMAX model and evaluate its performance.

    Args:
        data (array-like): Time series data.
        symbol (str): Symbol or identifier for the dataset.
        model_path (str): Path to save the trained model.
        order (tuple): The (p, d, q) order of the model.
        seasonal_order (tuple, optional): The seasonal order of the model. Defaults to None.
        logger (logging.Logger, optional): Logger instance. Defaults to None.

    Returns:
        tuple: Root Mean Squared Error (RMSE) and forecasted values.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Training ARIMA model for {symbol} with order: {order} "
                f"and seasonal_order: {seasonal_order}")

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

        # Fit the model without 'disp' (removed for compatibility)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)

        # Calculate RMSE using the last 10 data points as a simple example
        rmse = np.sqrt(mean_squared_error(data[-10:], forecast))

        # Save the trained model
        model_fit.save(model_path)
        logger.info(f"Model saved at {model_path}")

        return rmse, forecast

    except Exception as e:
        logger.error(f"Error training ARIMA model for {symbol}: {e}")
        raise

# -------------------------------------------------------------------
# Function: build_lstm_model
# Description: Builds an LSTM model with configurable layers, dropout,
#              and learning rate for time series prediction.
# -------------------------------------------------------------------
def build_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate):
    """
    Build an LSTM model for regression tasks.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        lstm_units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate between layers.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tensorflow.keras.Model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        BatchNormalization(),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1)  # Output layer for regression
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    return model

# -------------------------------------------------------------------
# Function: train_lstm_model
# Description: Trains an LSTM model with early stopping and learning
#              rate reduction on plateau.
# -------------------------------------------------------------------
def train_lstm_model(X_train, y_train, X_val, y_val,
                     lstm_units, dropout_rate, learning_rate,
                     logger, epochs=50):
    """
    Train an LSTM model with early stopping and learning rate reduction.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training targets.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation targets.
        lstm_units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate.
        learning_rate (float): Learning rate for the optimizer.
        logger (logging.Logger): Logger instance.
        epochs (int, optional): Number of training epochs. Defaults to 50.

    Returns:
        tuple: Trained LSTM model and training history.
    """
    try:
        input_shape = (X_train.shape[1], X_train.shape[2])
        logger.info(
            f"Training LSTM model with input shape: {input_shape}, "
            f"LSTM Units: {lstm_units}, Dropout: {dropout_rate}, "
            f"Learning Rate: {learning_rate}"
        )

        model = build_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate)

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
            batch_size=32,
            callbacks=[early_stopping, reduce_lr]
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
def train_linear_regression_model(X_train, y_train, X_val, y_val,
                                  model_type='linear', alpha=1.0, logger=None):
    """
    Train a linear, Ridge, or Lasso regression model and evaluate its performance.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training targets.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation targets.
        model_type (str, optional): Type of regression ('linear', 'ridge', 'lasso'). Defaults to 'linear'.
        alpha (float, optional): Regularization strength for Ridge/Lasso. Defaults to 1.0.
        logger (logging.Logger, optional): Logger instance. Defaults to None.

    Returns:
        tuple: Trained regression model and a dictionary of performance metrics.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        if model_type == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)

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
def save_trained_regression_model(model, model_save_path, model_type, logger=None):
    """
    Save a trained regression model to the specified path.

    Args:
        model (sklearn.base.BaseEstimator): Trained regression model.
        model_save_path (str): Directory path to save the model.
        model_type (str): Type of regression model ('linear', 'ridge', 'lasso').
        logger (logging.Logger, optional): Logger instance. Defaults to None.

    Returns:
        str: Path to the saved model file.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Ensure absolute path and create the directory if it doesn't exist
    model_save_path = os.path.abspath(model_save_path)
    os.makedirs(model_save_path, exist_ok=True)

    try:
        model_file = os.path.join(model_save_path, f"{model_type}_regression_model.pkl")
        joblib.dump(model, model_file)

        logger.info(f"{model_type.capitalize()} regression model saved at {model_file}")

        return model_file

    except Exception as e:
        logger.error(f"Error saving {model_type} regression model: {e}")
        raise

# -------------------------------------------------------------------
# Function: train_random_forest_model
# Description: Trains a Random Forest model and logs performance.
# -------------------------------------------------------------------
def train_random_forest_model(X_train, y_train, X_test, y_test, params, logger):
    """
    Train a Random Forest Regressor and evaluate its performance.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training targets.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test targets.
        params (dict): Model parameters (e.g., 'n_estimators', 'max_depth').
        logger (logging.Logger): Logger instance.

    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """
    logger.info("Training Random Forest model...")

    try:
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', None)

        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=params.get('random_state', 42)
        )

        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Random Forest Model MSE: {mse:.4f}")

        return rf_model

    except Exception as e:
        logger.error(f"Error training Random Forest model: {e}")
        raise

# -------------------------------------------------------------------
# Function: train_decision_tree_model
# Description: Trains a Decision Tree model and logs performance.
# -------------------------------------------------------------------
def train_decision_tree_model(X_train, y_train, X_test, y_test, params, logger):
    """
    Train a Decision Tree Regressor and evaluate its performance.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training targets.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test targets.
        params (dict): Model parameters (e.g., 'max_depth').
        logger (logging.Logger): Logger instance.

    Returns:
        DecisionTreeRegressor: Trained Decision Tree model.
    """
    logger.info("Training Decision Tree model...")

    try:
        max_depth = params.get('max_depth', None)

        dt_model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=params.get('random_state', 42)
        )

        dt_model.fit(X_train, y_train)

        y_pred = dt_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Decision Tree Model MSE: {mse:.4f}")

        return dt_model

    except Exception as e:
        logger.error(f"Error training Decision Tree model: {e}")
        raise

# -------------------------------------------------------------------
# Function: train_svm_model
# Description: Trains an SVM model with the provided training data.
# -------------------------------------------------------------------
def train_svm_model(X_train, y_train, logger, kernel='rbf', C=1e3, gamma=0.1):
    """
    Train an Support Vector Machine (SVM) Regressor.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training targets.
        logger (logging.Logger): Logger instance.
        kernel (str, optional): Kernel type. Defaults to 'rbf'.
        C (float, optional): Regularization parameter. Defaults to 1e3.
        gamma (float, optional): Kernel coefficient. Defaults to 0.1.

    Returns:
        SVR: Trained SVM model.
    """
    logger.info("Training the SVM model")

    try:
        model = SVR(kernel=kernel, C=C, gamma=gamma)  # Hyperparameters can be tuned
        model.fit(X_train, y_train)
        logger.info("SVM model training completed successfully.")
        return model

    except Exception as e:
        logger.error(f"Error training SVM model: {e}")
        raise

# -------------------------------------------------------------------
# Transfer Learning Helpers
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Function: build_transfer_learning_model
# Description: Builds a Transfer Learning model using a pre-trained base.
# -------------------------------------------------------------------
def build_transfer_learning_model(base_model_name='VGG16', input_shape=(224, 224, 3),
                                  num_classes=10, trainable_layers=4):
    """
    Build a Transfer Learning model using a pre-trained base model.

    Args:
        base_model_name (str, optional): Name of the pre-trained model ('VGG16', 'ResNet50', 'MobileNetV2'). Defaults to 'VGG16'.
        input_shape (tuple, optional): Shape of the input images. Defaults to (224, 224, 3).
        num_classes (int, optional): Number of output classes. Defaults to 10.
        trainable_layers (int, optional): Number of layers to make trainable from the top of the base model. Defaults to 4.

    Returns:
        tensorflow.keras.Model: Compiled Transfer Learning model.
    """
    # Select the base model
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported base_model_name. Choose from 'VGG16', 'ResNet50', 'MobileNetV2'.")

    # Freeze the base model
    base_model.trainable = False

    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Construct the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Optionally, make some layers trainable
    if trainable_layers > 0:
        # Unfreeze the top 'trainable_layers' layers
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# -------------------------------------------------------------------
# Function: train_transfer_learning_model
# Description: Trains a Transfer Learning model with given data.
# -------------------------------------------------------------------
def train_transfer_learning_model(model, train_generator, validation_generator,
                                  epochs=10, callbacks=None):
    """
    Train a Transfer Learning model using data generators.

    Args:
        model (tensorflow.keras.Model): Transfer Learning model to train.
        train_generator (tensorflow.keras.preprocessing.image.ImageDataGenerator.flow or similar): Training data generator.
        validation_generator (tensorflow.keras.preprocessing.image.ImageDataGenerator.flow or similar): Validation data generator.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        callbacks (list, optional): List of Keras callbacks. Defaults to None.

    Returns:
        tensorflow.keras.Model: Trained Transfer Learning model.
        tensorflow.keras.callbacks.History: Training history.
    """
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    return model, history

# -------------------------------------------------------------------
# Function: save_trained_transfer_learning_model
# Description: Saves a trained Transfer Learning model to disk.
# -------------------------------------------------------------------
def save_trained_transfer_learning_model(model, model_save_path, logger=None):
    """
    Save a trained Transfer Learning model to the specified path.

    Args:
        model (tensorflow.keras.Model): Trained Transfer Learning model.
        model_save_path (str): File path to save the model (e.g., 'models/transfer_learning_model.h5').
        logger (logging.Logger, optional): Logger instance. Defaults to None.

    Returns:
        str: Path to the saved model file.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Ensure the directory exists
    directory = os.path.dirname(model_save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    try:
        model.save(model_save_path)
        logger.info(f"Transfer Learning model saved at {model_save_path}")
        return model_save_path
    except Exception as e:
        logger.error(f"Error saving Transfer Learning model: {e}")
        raise
