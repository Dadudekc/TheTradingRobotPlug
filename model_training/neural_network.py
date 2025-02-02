"""
model_training/neural_network.py
--------------------------------
Provides functions to train either a standard neural network or an LSTM model with regularization and optional transfer learning.
"""

import logging
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

def train_neural_network_or_lstm(X_train, y_train, X_val, y_val,
                                 model_type="neural_network", epochs=50,
                                 pretrained_model_path=None):
    """
    Trains a neural network or LSTM model with regularization and (optionally) transfer learning.

    Parameters:
        X_train (np.array): Training features.
        y_train (np.array): Training targets.
        X_val (np.array): Validation features.
        y_val (np.array): Validation targets.
        model_type (str): Either "neural_network" or "LSTM".
        epochs (int): Number of training epochs.
        pretrained_model_path (str or None): Path to a pre-trained model to load and fine-tune.
        
    Returns:
        model: The trained Keras model.
        metrics (tuple): (mse, rmse, r2) computed on the validation set.
    """
    try:
        if pretrained_model_path:
            model = load_model(pretrained_model_path)
            # Freeze initial layers except for the last two layers.
            for layer in model.layers[:-2]:
                layer.trainable = False
        else:
            model = Sequential()

        if model_type == "neural_network":
            # For standard neural network: expect X_train shape (samples, features)
            input_shape = (X_train.shape[1],)
            model.add(Dense(128, activation='relu', input_shape=input_shape,
                            kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            model.add(Dense(1))
            X_train_processed = X_train
            X_val_processed = X_val

        elif model_type == "LSTM":
            # For LSTM: reshape input to (samples, timesteps, features)
            input_shape = (X_train.shape[1], 1)
            model.add(LSTM(50, return_sequences=False, input_shape=input_shape,
                           kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            model.add(Dense(1))
            # Reshape 2D input to 3D: (samples, features, 1)
            X_train_processed = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val_processed = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        else:
            raise ValueError("Unsupported model type. Choose 'neural_network' or 'LSTM'.")

        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(X_train_processed, y_train,
                            validation_data=(X_val_processed, y_val),
                            epochs=epochs, batch_size=32, callbacks=[early_stopping],
                            verbose=1)

        y_pred_val = model.predict(X_val_processed).flatten()
        mse = mean_squared_error(y_val, y_pred_val)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred_val)

        logging.info(f"Training complete ({model_type}). MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return model, (mse, rmse, r2)
    except Exception as e:
        logging.error("Error in train_neural_network_or_lstm: " + str(e))
        logging.error(traceback.format_exc())
        raise
