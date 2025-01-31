# model_training/trainers/svm_trainer.py

from sklearn.svm import SVR
from base.base_trainer import BaseTrainer
from utils.evaluation import evaluate_regression
from utils.model_persistence import ModelPersistence
import joblib

class SVMTrainer(BaseTrainer):
    """
    Trainer for Support Vector Machine regression models.
    """

    def __init__(self, config: dict, logger: Any):
        """
        Initialize the SVMTrainer.

        Args:
            config (dict): Configuration parameters.
            logger (logging.Logger): Logger instance.
        """
        super().__init__(config, logger)
        self.model = None
        self.persistence = ModelPersistence(config['model_save_dir'], logger)

    def train(self, X_train, y_train):
        """
        Train the SVM model.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
        """
        self.logger.info("Training SVM model.")
        svr_params = self.config.get('svm_params', {'kernel': 'rbf', 'C': 1000, 'gamma': 0.1})
        self.model = SVR(**svr_params)
        self.model.fit(X_train, y_train)
        self.logger.info("SVM model training completed.")

    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained SVM model.

        Args:
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing labels.

        Returns:
            dict: Evaluation metrics.
        """
        self.logger.info("Evaluating SVM model.")
        predictions = self.model.predict(X_test)
        metrics = evaluate_regression(y_test, predictions)
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def save_model(self, model_name: str):
        """
        Save the trained SVM model.

        Args:
            model_name (str): Name of the model file.
        """
        self.persistence.save_model(self.model, model_name)

    def load_model(self, model_path: str):
        """
        Load a saved SVM model.

        Args:
            model_path (str): Path to the model file.
        """
        self.model = self.persistence.load_model(model_path)
        self.logger.info(f"SVM model loaded from {model_path}.")
