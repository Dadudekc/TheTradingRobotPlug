# C:\TheTradingRobotPlug\Scripts\model_training\utils\model_saving.py

import joblib
import os

def save_model_and_scaler(model, scaler, model_path, scaler_path, logger=None):
    """
    Save the trained model and the associated scaler to disk.

    Parameters:
    - model: The trained model to be saved.
    - scaler: The scaler (e.g., MinMaxScaler or StandardScaler) used during data preprocessing.
    - model_path: File path where the model will be saved.
    - scaler_path: File path where the scaler will be saved.
    - logger: Logger instance for logging purposes (optional).
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

        # Save the model
        joblib.dump(model, model_path)
        if logger:
            logger.info(f"Model saved successfully at {model_path}")

        # Save the scaler
        joblib.dump(scaler, scaler_path)
        if logger:
            logger.info(f"Scaler saved successfully at {scaler_path}")

    except Exception as e:
        if logger:
            logger.error(f"Error saving model or scaler: {e}")
        raise
