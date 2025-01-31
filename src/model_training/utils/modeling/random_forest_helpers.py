# -------------------------------------------------------------------
# File Path: C:/TheTradingRobotPlug/Scripts/Utilities/random_forest_helpers.py
# Description: Helper functions for training and evaluating Random Forest models.
# -------------------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
import logging

class RandomForestModel:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def train(self, X_train, y_train, feature_names, random_state=None):
        """
        Train the Random Forest model with the provided training data.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training target data.
            feature_names (list): List of feature names.
            random_state (int, optional): Random state for reproducibility. Default is None.

        Returns:
            model: Trained Random Forest model.
            best_params: Best parameters for the Random Forest model (if applicable).
        """
        try:
            # Initialize the Random Forest Regressor with random_state if provided
            self.logger.info("Training Random Forest model with random_state={}".format(random_state))
            model = RandomForestRegressor(random_state=random_state)

            # Fit the model to the training data
            model.fit(X_train, y_train)

            self.logger.info("Random Forest model trained successfully.")

            # Return the trained model and any best parameters (if you implement hyperparameter tuning)
            return model, None
        except Exception as e:
            self.logger.error(f"Error during Random Forest training: {e}")
            return None, None

    def save_model(self, model, file_path):
        """
        Save the trained Random Forest model to a file.

        Args:
            model: Trained Random Forest model.
            file_path (str): Path to save the model file.
        """
        try:
            import joblib
            self.logger.info(f"Saving Random Forest model to {file_path}...")
            joblib.dump(model, file_path)
            self.logger.info("Model saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving Random Forest model: {e}")

    def load_model(self, file_path):
        """
        Load a saved Random Forest model from a file.

        Args:
            file_path (str): Path to the saved model file.

        Returns:
            model: Loaded Random Forest model.
        """
        try:
            import joblib
            self.logger.info(f"Loading Random Forest model from {file_path}...")
            model = joblib.load(file_path)
            self.logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            self.logger.error(f"Error loading Random Forest model: {e}")
            return None

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the trained Random Forest model on test data.

        Args:
            model: Trained Random Forest model.
            X_test (np.ndarray): Test feature data.
            y_test (np.ndarray): Test target data.

        Returns:
            score: Model evaluation score.
        """
        try:
            self.logger.info("Evaluating Random Forest model...")
            score = model.score(X_test, y_test)
            self.logger.info(f"Model evaluation score: {score}")
            return score
        except Exception as e:
            self.logger.error(f"Error evaluating Random Forest model: {e}")
            return None
