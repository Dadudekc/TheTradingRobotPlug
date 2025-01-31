# -------------------------------------------------------------------
# File Path: C:\Projects\TradingRobotPlug\Scripts\model_training\utils\Training\example_training.py
# Description: Demonstrates example usage of functions in training_helpers.py
#              and optuna_helpers.py, including model training, transfer learning,
#              and hyperparameter tuning.
# -------------------------------------------------------------------

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression

# Import the functions from the training_helpers.py and optuna_helpers.py
from training_helpers import (train_gradient_boosting_model, train_arima_model,
                              train_lstm_model, train_linear_regression_model,
                              save_trained_regression_model, train_random_forest_model,
                              train_decision_tree_model, train_svm_model,
                              build_transfer_learning_model, train_transfer_learning_model,
                              save_trained_transfer_learning_model)
from optuna_helpers import tune_hyperparameters

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("example_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Example 1: Gradient Boosting Classifier
# -------------------------------------------------------------------
def example_gradient_boosting():
    # Create a sample classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    logger.info("Training Gradient Boosting Classifier...")
    model, accuracy = train_gradient_boosting_model(X, y, test_size=0.3, n_estimators=200, learning_rate=0.05)
    logger.info(f"Gradient Boosting Model Accuracy: {accuracy:.4f}")


# -------------------------------------------------------------------
# Example 2: ARIMA Model
# -------------------------------------------------------------------
def example_arima():
    # Create a sample time series data
    data = np.sin(np.arange(100)) + np.random.normal(0, 0.1, 100)

    logger.info("Training ARIMA model...")
    order = (1, 1, 1)  # ARIMA(p,d,q) order
    rmse, forecast = train_arima_model(data, symbol="example_symbol", model_path="arima_model.pkl", order=order, logger=logger)
    logger.info(f"ARIMA Model RMSE: {rmse:.4f}")


# -------------------------------------------------------------------
# Example 3: LSTM Model
# -------------------------------------------------------------------
def example_lstm():
    # Create synthetic time series data for LSTM
    X_train = np.random.randn(100, 10, 1)
    y_train = np.random.randn(100)
    X_val = np.random.randn(20, 10, 1)
    y_val = np.random.randn(20)

    logger.info("Training LSTM model...")
    lstm_units = 64
    dropout_rate = 0.2
    learning_rate = 0.001

    model, history = train_lstm_model(X_train, y_train, X_val, y_val, lstm_units, dropout_rate, learning_rate, logger, epochs=50)
    logger.info("LSTM model training completed.")


# -------------------------------------------------------------------
# Example 4: Linear Regression (or Ridge/Lasso)
# -------------------------------------------------------------------
def example_linear_regression():
    # Create a sample regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Training Linear Regression model...")
    model, metrics = train_linear_regression_model(X_train, y_train, X_val, y_val, model_type='ridge', alpha=0.5, logger=logger)
    logger.info(f"Ridge Regression - MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")

    # Save the trained model
    save_trained_regression_model(model, model_save_path="models/", model_type="ridge", logger=logger)


# -------------------------------------------------------------------
# Example 5: Random Forest Regressor
# -------------------------------------------------------------------
def example_random_forest():
    # Create a sample regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {'n_estimators': 100, 'max_depth': 10}
    logger.info("Training Random Forest model...")
    model = train_random_forest_model(X_train, y_train, X_test, y_test, params=params, logger=logger)


# -------------------------------------------------------------------
# Example 6: Decision Tree Regressor
# -------------------------------------------------------------------
def example_decision_tree():
    # Create a sample regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {'max_depth': 5}
    logger.info("Training Decision Tree model...")
    model = train_decision_tree_model(X_train, y_train, X_test, y_test, params=params, logger=logger)


# -------------------------------------------------------------------
# Example 7: SVM Regressor
# -------------------------------------------------------------------
def example_svm():
    # Create a sample regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Training SVM model...")
    model = train_svm_model(X_train, y_train, logger=logger)


# -------------------------------------------------------------------
# Example 8: Transfer Learning
# -------------------------------------------------------------------
def example_transfer_learning():
    """
    Example of training a Transfer Learning model using a pre-trained base model.
    This example uses synthetic data for demonstration purposes.
    In practice, you should use a real image dataset.
    """

    # Choose a pre-trained base model without the top layer
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

    logger.info("Setting up Transfer Learning example...")

    # Define number of classes
    num_classes = 10  # Adjust based on your dataset

    # Build and train the transfer learning model
    model = build_transfer_learning_model(base_model=base_model, num_classes=num_classes, dropout_rate=0.5, learning_rate=0.001)
    model.summary()

    # Create ImageDataGenerators for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Use 20% of training data for validation
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Path to your training data directory
    train_directory = 'data/train/'  # Replace with your actual data directory

    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = validation_datagen.flow_from_directory(
        train_directory,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
    callbacks = [early_stopping, reduce_lr]

    logger.info("Starting training of Transfer Learning model...")
    # Train the Transfer Learning model
    model, history = train_transfer_learning_model(model, train_generator, validation_generator,
                                                  epochs=20, callbacks=callbacks)
    logger.info("Transfer Learning model training completed.")

    # Save the trained Transfer Learning model
    save_trained_transfer_learning_model(model, model_save_path="models/transfer_learning_model.h5", logger=logger)


# -------------------------------------------------------------------
# Example 9: Hyperparameter Tuning with Optuna
# -------------------------------------------------------------------
def example_hyperparameter_tuning():
    logger.info("Starting hyperparameter tuning example...")

    # Example 1: Hyperparameter Tuning for Ridge Regression
    logger.info("Tuning hyperparameters for Ridge Regression...")

    # Create a sample regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tune hyperparameters for Ridge Regression
    best_params_ridge = tune_hyperparameters(
        X_train, y_train,
        n_trials=20,
        model_type='ridge'
    )
    logger.info(f"Best hyperparameters for Ridge Regression: {best_params_ridge}")

    # Train Ridge Regression with best hyperparameters
    model_ridge, metrics_ridge = train_linear_regression_model(
        X_train, y_train, X_val, y_val,
        model_type='ridge',
        alpha=best_params_ridge['alpha'],
        logger=logger
    )
    logger.info(f"Ridge Regression after tuning - MSE: {metrics_ridge['mse']:.4f}, R²: {metrics_ridge['r2']:.4f}")

    # Save the tuned Ridge Regression model
    save_trained_regression_model(model_ridge, model_save_path="models/", model_type="ridge_tuned", logger=logger)

    # Example 2: Hyperparameter Tuning for LSTM
    logger.info("Tuning hyperparameters for LSTM...")

    # Create synthetic time series data for LSTM
    X_train_lstm = np.random.randn(100, 10, 1)
    y_train_lstm = np.random.randn(100)
    X_val_lstm = np.random.randn(20, 10, 1)
    y_val_lstm = np.random.randn(20)

    input_shape = (10, 1)  # timesteps, features

    # Tune hyperparameters for LSTM
    best_params_lstm = tune_hyperparameters(
        X_train_lstm, y_train_lstm,
        X_val=X_val_lstm, y_val=y_val_lstm,
        input_shape=input_shape,
        n_trials=20,
        model_type='lstm'
    )
    logger.info(f"Best hyperparameters for LSTM: {best_params_lstm}")

    # Train LSTM with best hyperparameters
    model_lstm, history_lstm = train_lstm_model(
        X_train_lstm, y_train_lstm,
        X_val_lstm, y_val_lstm,
        lstm_units=best_params_lstm['lstm_units'],
        dropout_rate=best_params_lstm['dropout_rate'],
        learning_rate=best_params_lstm['learning_rate'],
        logger=logger,
        epochs=50
    )
    logger.info("LSTM after tuning training completed.")

    # Note: For saving LSTM models, you might want to implement a save function similar to others
    # For demonstration, we'll save it using Keras' built-in save method
    lstm_model_save_path = "models/lstm_tuned_model.h5"
    try:
        model_lstm.save(lstm_model_save_path)
        logger.info(f"Tuned LSTM model saved at {lstm_model_save_path}")
    except Exception as e:
        logger.error(f"Error saving tuned LSTM model: {e}")


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Run each example
        example_gradient_boosting()
        example_arima()
        example_lstm()
        example_linear_regression()
        example_random_forest()
        example_decision_tree()
        example_svm()
        example_transfer_learning()
        example_hyperparameter_tuning()

        logger.info("All model training and tuning examples completed.")

    except Exception as e:
        logger.error(f"An error occurred during the training process: {e}")
