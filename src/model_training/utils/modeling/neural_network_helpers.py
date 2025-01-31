# -------------------------------------------------------------------
# File Path: utils/neural_network_helpers.py
# Description: Helper module for building, training, and saving neural networks.
# -------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, ReduceLROnPlateau

def build_neural_network(model_config, input_shape, pretrained_model_path=None):
    """Build the neural network model based on the configuration."""
    layer_mapping = {
        'dense': Dense,
        'batch_norm': BatchNormalization,
        'dropout': Dropout,
        'lstm': LSTM,
        'gru': GRU
    }

    if pretrained_model_path:
        model = load_model(str(pretrained_model_path))
        for layer in model.layers[:-3]:
            layer.trainable = False
    else:
        model = Sequential()
        model.add(Input(shape=input_shape))  # Input shape (timesteps, features)

        for layer in model_config['layers']:
            layer_class = layer_mapping.get(layer['type'].lower())
            if not layer_class:
                raise ValueError(f"Layer type '{layer['type']}' is not recognized.")
            
            if 'units' in layer:
                layer_params = {'units': layer['units'], 'activation': layer.get('activation', 'relu')}
                if layer['type'].lower() in ['lstm', 'gru']:
                    layer_params['return_sequences'] = layer.get('return_sequences', False)
                model.add(layer_class(**layer_params))
            elif 'rate' in layer:
                model.add(layer_class(rate=layer['rate']))

    optimizer_params = model_config.get('optimizer', {})
    learning_rate = optimizer_params.get('learning_rate', 0.001)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)  # Clip gradients

    loss = model_config.get('loss', 'mean_squared_error')

    model.compile(optimizer=optimizer, loss=loss, metrics=['mse', 'mae'])
    
    return model

def train_neural_network(model, X_train, y_train, X_val, y_val, epochs, ticker, log_dir, scheduler_fn):
    """Train the neural network model with callbacks."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint(f"best_model_{ticker}.keras", save_best_only=True, monitor='val_loss', mode='min'),
        TensorBoard(log_dir=str(log_dir / ticker)),
        LearningRateScheduler(scheduler_fn),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, callbacks=callbacks, batch_size=64, verbose=1
    )
    
    return model, history
