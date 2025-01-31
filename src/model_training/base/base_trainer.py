# model_training/base/base_trainer.py

from abc import ABC, abstractmethod
from typing import Any

class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.

    Attributes:
        config (dict): Configuration parameters.
        logger (logging.Logger): Logger instance.
    """

    def __init__(self, config: dict, logger: Any):
        """
        Initialize the BaseTrainer.

        Args:
            config (dict): Configuration parameters.
            logger (logging.Logger): Logger instance.
        """
        self.config = config
        self.logger = logger

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model using the training data.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
        """
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using the test data.

        Args:
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing labels.
        """
        pass

    @abstractmethod
    def save_model(self, path: str):
        """
        Save the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        pass

    @abstractmethod
    def load_model(self, path: str):
        """
        Load a model from a file.

        Args:
            path (str): Path to the saved model.
        """
        pass
