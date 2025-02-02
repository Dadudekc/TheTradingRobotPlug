"""
model_training/ensemble.py
--------------------------
Provides functions for creating ensemble models and for model quantization.
"""

import logging
import tensorflow as tf
from sklearn.ensemble import VotingRegressor, VotingClassifier

def create_ensemble_model(base_models, X_train, y_train, method='voting', weights=None):
    """
    Creates an ensemble model using the specified method and fits it on the provided training data.

    Parameters:
        base_models (list): List of (name, model) tuples.
        X_train (np.array): Training features.
        y_train (np.array): Training targets.
        method (str): Ensemble method ('voting' supported; 'stacking' not implemented).
        weights (list or None): Optional weights for voting.
    
    Returns:
        ensemble_model: The fitted ensemble model.
    """
    try:
        if method == 'voting':
            # Determine problem type by checking for predict_proba
            first_model = base_models[0][1]
            if hasattr(first_model, "predict_proba"):
                ensemble_model = VotingClassifier(estimators=base_models, voting='soft', weights=weights)
            else:
                ensemble_model = VotingRegressor(estimators=base_models, weights=weights)
            ensemble_model.fit(X_train, y_train)
        elif method == 'stacking':
            raise NotImplementedError("Stacking ensemble is not implemented in this version.")
        else:
            raise ValueError("Unsupported ensemble method.")
        return ensemble_model
    except Exception as e:
        logging.error("Error in create_ensemble_model: " + str(e))
        raise

def quantize_model(model, quantization_method='weight', representative_data=None):
    """
    Quantizes a Keras model using the TensorFlow Lite converter.

    Parameters:
        model: The trained Keras model.
        quantization_method (str): 'weight' (default) or 'activation'.
        representative_data (function or None): A generator function for representative samples (required for activation quantization).
    
    Returns:
        quantized_model (bytes): The TFLite model as a binary blob.
    """
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantization_method == 'activation':
            if representative_data is None:
                raise ValueError("Representative data is required for activation quantization.")
            converter.representative_dataset = representative_data
        quantized_model = converter.convert()
        return quantized_model
    except Exception as e:
        logging.error("Error in quantize_model: " + str(e))
        raise
