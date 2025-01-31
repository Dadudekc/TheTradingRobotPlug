# -------------------------------------------------------------------
# File Path: model_training/utils/deploy.py
# Description: Deploys the trained model for predictions with support 
#              for batch processing, error handling, logging, and notifications.
# -------------------------------------------------------------------

import os
import logging
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import yaml
from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel, ValidationError, validator
from logging.handlers import RotatingFileHandler
import argparse
import smtplib
from email.mime.text import MIMEText

# Load environment variables from .env file if present
from dotenv import load_dotenv
load_dotenv()

# Configuration Models using Pydantic for validation
class ModelConfig(BaseModel):
    path: str
    version: str

class DataConfig(BaseModel):
    input_path: str
    output_dir: str
    batch_size: int = 32

class ScalerConfig(BaseModel):
    path: str

class LoggingConfig(BaseModel):
    file: str
    level: str
    max_bytes: int
    backup_count: int

class EmailConfig(BaseModel):
    enabled: bool
    smtp_server: Optional[str]
    smtp_port: Optional[int]
    sender: Optional[str]
    recipients: Optional[list]
    username: Optional[str]
    password: Optional[str]

class NotificationsConfig(BaseModel):
    email: EmailConfig

class Config(BaseModel):
    model: ModelConfig
    data: DataConfig
    scaler: ScalerConfig
    logging: LoggingConfig
    notifications: NotificationsConfig

    @validator('logging.level')
    def validate_logging_level(cls, v):
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in levels:
            raise ValueError(f'Invalid logging level: {v}')
        return v.upper()

# Function to load configuration from YAML file
def load_config(config_path: str = "config.yaml") -> Config:
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = Config(**config_dict)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except ValidationError as e:
        raise ValueError(f"Configuration validation error: {e}")

# Notification Handler
class Notifier:
    def __init__(self, email_config: EmailConfig):
        self.enabled = email_config.enabled
        if self.enabled:
            required_fields = ['smtp_server', 'smtp_port', 'sender', 'recipients', 'username', 'password']
            for field in required_fields:
                if getattr(email_config, field) is None:
                    raise ValueError(f"Email notifications enabled but '{field}' is not set in configuration.")
            self.smtp_server = email_config.smtp_server
            self.smtp_port = email_config.smtp_port
            self.sender = email_config.sender
            self.recipients = email_config.recipients
            self.username = email_config.username
            self.password = email_config.password

    def send_email(self, subject: str, message: str):
        if not self.enabled:
            return
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.sender
            msg['To'] = ', '.join(self.recipients)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.sender, self.recipients, msg.as_string())
            logging.info("Notification email sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send notification email: {e}")

# Deployment Class
class ModelDeployer:
    def __init__(self, config: Config, notifier: Optional[Notifier] = None):
        self.config = config
        self.notifier = notifier
        self.setup_logging()

    def setup_logging(self):
        """Set up advanced logging with rotating file handler."""
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.config.logging.level))
        handler = RotatingFileHandler(
            self.config.logging.file,
            maxBytes=self.config.logging.max_bytes,
            backupCount=self.config.logging.backup_count
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        self.logger = logger

    def load_model(self) -> tf.keras.Model:
        """Loads the trained TensorFlow/Keras model."""
        try:
            model = tf.keras.models.load_model(self.config.model.path)
            self.logger.info(f"Model loaded successfully from {self.config.model.path} (Version: {self.config.model.version})")
            return model
        except FileNotFoundError:
            self.logger.error(f"Model file not found at {self.config.model.path}.")
            if self.notifier:
                self.notifier.send_email(
                    subject="Model Deployment Failure",
                    message=f"Model file not found at {self.config.model.path}."
                )
            raise
        except Exception as e:
            self.logger.error(f"Error loading model from {self.config.model.path}: {str(e)}")
            if self.notifier:
                self.notifier.send_email(
                    subject="Model Deployment Failure",
                    message=f"Error loading model: {str(e)}"
                )
            raise

    def load_scaler(self) -> Optional[StandardScaler]:
        """Loads the pre-fitted scaler for data preprocessing."""
        try:
            if os.path.exists(self.config.scaler.path):
                scaler = joblib.load(self.config.scaler.path)
                self.logger.info(f"Scaler loaded successfully from {self.config.scaler.path}")
                return scaler
            else:
                self.logger.warning(f"Scaler file not found at {self.config.scaler.path}. Proceeding with default scaling.")
                return None
        except Exception as e:
            self.logger.error(f"Error loading scaler from {self.config.scaler.path}: {str(e)}")
            if self.notifier:
                self.notifier.send_email(
                    subject="Model Deployment Warning",
                    message=f"Error loading scaler: {str(e)}. Proceeding with default scaling."
                )
            return None

    def preprocess_data(self, scaler: Optional[StandardScaler]) -> np.ndarray:
        """Loads and preprocesses input data for predictions."""
        try:
            df = pd.read_csv(self.config.data.input_path)
            if df.empty:
                raise ValueError("Input data is empty.")

            # Data Cleaning
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)

            # Feature Selection: Ensure only numeric features are selected
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_features:
                raise ValueError("No numeric features found in input data.")

            data = df[numeric_features].values

            # Scaling
            if scaler:
                data = scaler.transform(data)
            else:
                scaler = StandardScaler()
                data = scaler.fit_transform(data)
                self.logger.warning("No pre-fitted scaler provided. Data was scaled using a new scaler instance.")

            self.logger.info(f"Input data preprocessed successfully with {data.shape[0]} records and {data.shape[1]} features.")
            return data
        except FileNotFoundError:
            self.logger.error(f"Input data file not found at {self.config.data.input_path}.")
            if self.notifier:
                self.notifier.send_email(
                    subject="Model Deployment Failure",
                    message=f"Input data file not found at {self.config.data.input_path}."
                )
            raise
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {str(e)}")
            if self.notifier:
                self.notifier.send_email(
                    subject="Model Deployment Failure",
                    message=f"Error during data preprocessing: {str(e)}"
                )
            raise

    def make_predictions(self, model: tf.keras.Model, input_data: np.ndarray) -> np.ndarray:
        """Uses the loaded model to make predictions on the input data."""
        try:
            predictions = model.predict(input_data, batch_size=self.config.data.batch_size)
            self.logger.info(f"Predictions made successfully on {input_data.shape[0]} records with batch size {self.config.data.batch_size}.")
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            if self.notifier:
                self.notifier.send_email(
                    subject="Model Deployment Failure",
                    message=f"Error during prediction: {str(e)}"
                )
            raise

    def save_predictions(self, predictions: np.ndarray) -> str:
        """Saves predictions to a timestamped CSV file."""
        try:
            if not os.path.exists(self.config.data.output_dir):
                os.makedirs(self.config.data.output_dir)
                self.logger.info(f"Created output directory at {self.config.data.output_dir}")

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"predictions_{timestamp}.csv"
            output_path = os.path.join(self.config.data.output_dir, output_filename)
            pd.DataFrame(predictions, columns=['Predicted']).to_csv(output_path, index=False)
            self.logger.info(f"Predictions saved to {output_path}.")
            return output_path
        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
            if self.notifier:
                self.notifier.send_email(
                    subject="Model Deployment Failure",
                    message=f"Error saving predictions: {str(e)}"
                )
            raise

    def run(self):
        """Executes the full deployment pipeline."""
        try:
            self.logger.info("Starting model deployment process.")
            
            # Load model
            model = self.load_model()
            
            # Load scaler
            scaler = self.load_scaler()
            
            # Preprocess data
            input_data = self.preprocess_data(scaler)
            
            # Make predictions
            predictions = self.make_predictions(model, input_data)
            
            # Save predictions
            output_file = self.save_predictions(predictions)
            
            self.logger.info(f"Model deployment completed successfully. Predictions saved at {output_file}.")
            if self.notifier:
                self.notifier.send_email(
                    subject="Model Deployment Success",
                    message=f"Model deployment completed successfully. Predictions saved at {output_file}."
                )
        except Exception as e:
            self.logger.critical(f"Model deployment failed: {str(e)}")
            if self.notifier:
                self.notifier.send_email(
                    subject="Model Deployment Critical Failure",
                    message=f"Model deployment failed: {str(e)}"
                )
            raise

# Command-Line Interface
def parse_arguments() -> Dict[str, Any]:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Deploy trained model for predictions.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file.')
    return vars(parser.parse_args())

def main():
    """Main entry point for the deployment script."""
    args = parse_arguments()
    try:
        config = load_config(args['config'])
    except Exception as e:
        logging.critical(f"Failed to load configuration: {str(e)}")
        raise

    # Initialize notifier if enabled
    notifier = None
    if config.notifications.email.enabled:
        try:
            notifier = Notifier(config.notifications.email)
        except Exception as e:
            logging.error(f"Failed to initialize notifier: {str(e)}")

    # Initialize and run deployer
    deployer = ModelDeployer(config=config, notifier=notifier)
    deployer.run()

if __name__ == "__main__":
    main()