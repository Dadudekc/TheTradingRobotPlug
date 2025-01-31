# model_training/utils/model_persistence.py

import joblib
from pathlib import Path
import os

class ModelPersistence:
    """
    Handles saving and loading machine learning models.
    """

    def __init__(self, model_dir: str, logger: Any):
        """
        Initialize the ModelPersistence.

        Args:
            model_dir (str): Directory to save models.
            logger (logging.Logger): Logger instance.
        """
        self.model_dir = Path(model_dir)
        self.logger = logger
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_name: str):
        """
        Save the model to disk.

        Args:
            model: Trained model object.
            model_name (str): Name of the model file.
        """
        model_path = self.model_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        self.logger.info(f"Model saved at {model_path}")

    def load_model(self, model_path: str):
        """
        Load a model from disk.

        Args:
            model_path (str): Path to the model file.

        Returns:
            Loaded model object.
        """
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return None
        model = joblib.load(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        return model
