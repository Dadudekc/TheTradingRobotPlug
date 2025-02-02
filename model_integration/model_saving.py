"""
model_integration/model_saving.py
---------------------------------
Provides functions to save and load models, scalers, and associated metadata.
"""

import os
import json
import joblib
import logging
from datetime import datetime

def save_trained_model(model, model_type, scaler=None, file_path=None):
    """
    Saves the trained model, optional scaler, and metadata.
    
    Args:
        model: Trained model object.
        model_type (str): Type of the model (e.g., 'linear_regression', 'neural_network').
        scaler: Optional scaler object.
        file_path (str, optional): File path for saving. Auto-generated if None.
    """
    try:
        if file_path is None:
            file_path = auto_generate_save_path(model_type)
        # Save model
        save_model_by_type(model, model_type, file_path)
        logging.info(f"Model saved to {file_path}")
        
        if scaler:
            scaler_file_path = file_path.replace(".joblib", "_scaler.joblib")
            joblib.dump(scaler, scaler_file_path)
            logging.info(f"Scaler saved to {scaler_file_path}")
        
        metadata = construct_metadata(model, model_type, scaler)
        metadata_file_path = file_path.replace(".joblib", "_metadata.json")
        with open(metadata_file_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"Metadata saved to {metadata_file_path}")
    except Exception as e:
        logging.error("Error in save_trained_model: " + str(e))
        raise

def construct_metadata(model, model_type, scaler=None):
    """
    Constructs metadata for the trained model.
    
    Args:
        model: Trained model object.
        model_type (str): Type of the model.
        scaler: Optional scaler object.
    
    Returns:
        dict: Metadata dictionary.
    """
    metadata = {
        "model_type": model_type,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if hasattr(model, "get_params"):
        metadata["model_parameters"] = {k: str(v) for k, v in model.get_params().items()}
    if scaler and hasattr(scaler, "get_params"):
        metadata["scaler"] = {k: str(v) for k, v in scaler.get_params().items()}
    return metadata

def save_model_by_type(model, model_type, file_path):
    """
    Saves the model to disk based on its type.
    
    Args:
        model: The model to save.
        model_type (str): Model type.
        file_path (str): File path for saving.
    """
    try:
        model_type_lower = model_type.lower()
        if model_type_lower in ['linear_regression', 'random_forest']:
            joblib.dump(model, file_path)
        elif model_type_lower in ['neural_network', 'lstm']:
            # Save Keras models as H5 files
            keras_path = file_path.replace(".joblib", ".h5")
            model.save(keras_path)
        elif model_type_lower == 'arima':
            import pickle
            pkl_path = file_path.replace(".joblib", ".pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        logging.error("Error in save_model_by_type: " + str(e))
        raise

def load_model(model_file_path, scaler_file_path=None, metadata_file_path=None):
    """
    Loads a model, optional scaler, and metadata from disk.
    
    Args:
        model_file_path (str): Path to the saved model file.
        scaler_file_path (str, optional): Path to the scaler file.
        metadata_file_path (str, optional): Path to the metadata JSON file.
    
    Returns:
        tuple: (model, scaler, metadata)
    """
    try:
        if model_file_path.endswith(".h5"):
            from tensorflow.keras.models import load_model as keras_load_model
            model = keras_load_model(model_file_path)
        elif model_file_path.endswith(".joblib"):
            model = joblib.load(model_file_path)
        elif model_file_path.endswith(".pkl"):
            import pickle
            with open(model_file_path, "rb") as f:
                model = pickle.load(f)
        else:
            raise ValueError("Unsupported model file extension.")
        
        scaler = None
        if scaler_file_path and os.path.exists(scaler_file_path):
            scaler = joblib.load(scaler_file_path)
        
        metadata = None
        if metadata_file_path and os.path.exists(metadata_file_path):
            with open(metadata_file_path, "r") as f:
                metadata = json.load(f)
        return model, scaler, metadata
    except Exception as e:
        logging.error("Error in load_model: " + str(e))
        raise

def auto_generate_save_path(model_type, base_dir="saved_models"):
    """
    Generates a file path for saving a model.
    
    Args:
        model_type (str): Type of the model.
        base_dir (str): Directory to save the model.
    
    Returns:
        str: Generated file path.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_dir}/{model_type}_{timestamp}.joblib"
