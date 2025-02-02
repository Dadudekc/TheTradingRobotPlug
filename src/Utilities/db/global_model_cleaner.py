"""
File: global_model_cleaner.py
Path: C:/Projects/TradingRobotPlug/src/Utilities/global_model_cleaner.py

Description:
    Monitors the PostgreSQL global database to ensure only the top 50 models 
    (based on MSE score) are retained. Periodically checks the database and 
    deletes models that exceed this limit.
"""

import time
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
utilities_dir = project_root / 'src' / 'Utilities'
model_dir = project_root / 'SavedModels'
data_processing_dir = project_root / 'src' / 'Data_Processing'

sys.path.extend([
    str(utilities_dir),
    str(model_dir),
    str(data_processing_dir)
])

# -------------------------------------------------------------------
# Attempt Imports (ConfigManager, Logging)
# -------------------------------------------------------------------
try:
    from Utilities.config_manager import ConfigManager
    from Utilities.shared_utils import setup_logging
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Class Definition
# -------------------------------------------------------------------
class GlobalModelCleaner:
    """
    A class-based agent that monitors the global PostgreSQL database 
    and ensures only the top 50 models (by MSE) are retained.
    """

    def __init__(self):
        """
        Initialize the GlobalModelCleaner with configuration, environment, and logging.
        """
        # Load environment variables
        env_path = project_root / '.env'
        load_dotenv(dotenv_path=env_path)

        # Setup logging
        log_dir = project_root / 'logs' / 'Utilities'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(str(log_dir / 'global_model_cleaner'))
        self.logger.info("Logging initialized for GlobalModelCleaner")

        # Initialize configuration
        self.config_manager = ConfigManager()
        self.logger.info("ConfigManager initialized successfully.")

        # Read environment / config values
        self.postgres_user = self.config_manager.get('POSTGRES_USER')
        self.postgres_password = self.config_manager.get('POSTGRES_PASSWORD')
        self.postgres_host = self.config_manager.get('POSTGRES_HOST', 'localhost')
        self.postgres_port = self.config_manager.get('POSTGRES_PORT', '5432')
        self.postgres_dbname = self.config_manager.get('POSTGRES_DBNAME', 'trading_robot_plug')
        self.check_interval_seconds = int(self.config_manager.get("CHECK_INTERVAL_SECONDS", 600))

        # Create the PostgreSQL connection URL
        self.database_url = (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_dbname}"
        )

        # SQLAlchemy engine (created on init)
        self.engine = self._connect_to_global_db()

    def _connect_to_global_db(self) -> Optional[create_engine]:
        """
        Connect to the PostgreSQL database that stores global models.

        Returns:
            A SQLAlchemy Engine object if connection is successful, otherwise None.
        """
        try:
            engine = create_engine(self.database_url)
            self.logger.info("Successfully connected to the PostgreSQL database.")
            return engine
        except SQLAlchemyError as e:
            self.logger.error(f"Error connecting to PostgreSQL database: {e}")
            return None

    def _fetch_all_models(self) -> List[tuple]:
        """
        Fetch all models from the global database, sorted by MSE score in ascending order.

        Returns:
            A list of tuples with model data: (id, model_name, mse_score, model_blob).
        """
        if not self.engine:
            self.logger.error("No database engine available to fetch models.")
            return []

        try:
            with self.engine.connect() as connection:
                result = connection.execute(
                    text("SELECT id, model_name, mse_score, model_blob "
                         "FROM models "
                         "ORDER BY mse_score ASC")
                )
                models = result.fetchall()
            self.logger.info(f"Fetched {len(models)} models from the database.")
            return models
        except SQLAlchemyError as e:
            self.logger.error(f"Error fetching models: {e}")
            return []

    def _delete_model_by_id(self, model_id: int):
        """
        Delete a model from the global database by its ID.

        Args:
            model_id (int): The ID of the model to be deleted.
        """
        if not self.engine:
            self.logger.error("No database engine available to delete models.")
            return

        try:
            with self.engine.connect() as connection:
                connection.execute(text("DELETE FROM models WHERE id = :id"), {"id": model_id})
            self.logger.info(f"Deleted model with ID: {model_id}")
        except SQLAlchemyError as e:
            self.logger.error(f"Error deleting model with ID {model_id}: {e}")

    def clean_global_database(self):
        """
        Retain only the top 50 models (by ascending MSE). Delete any models beyond the top 50.
        """
        self.logger.info("Starting database cleanup process.")
        models = self._fetch_all_models()
        if len(models) > 50:
            self.logger.info(f"Found {len(models)} models; retaining top 50, deleting the rest.")
            for model in models[50:]:
                model_id, model_name, mse_score, _ = model
                self.logger.info(f"Deleting model '{model_name}', MSE={mse_score} (ID={model_id})")
                self._delete_model_by_id(model_id)
            self.logger.info(
                f"Cleanup completed. Retained top 50 models and deleted {len(models) - 50} models."
            )
        else:
            self.logger.info(f"Database has {len(models)} models. No cleanup required.")

    def run(self):
        """
        Continuously run the database cleaning agent, sleeping between intervals.
        """
        if not self.engine:
            self.logger.error("Could not start agent due to database connection issues.")
            print("Could not start agent. Check logs for details.")
            return

        self.logger.info(
            f"Starting the global model cleaning agent. "
            f"Will check the database every {self.check_interval_seconds / 60:.1f} minutes."
        )
        print(
            f"Starting the global model cleaning agent. "
            f"Checks every {self.check_interval_seconds / 60:.1f} minutes..."
        )

        try:
            while True:
                self.clean_global_database()
                self.logger.info(f"Sleeping for {self.check_interval_seconds} seconds...")
                time.sleep(self.check_interval_seconds)
        except KeyboardInterrupt:
            self.logger.info("Agent stopped by user (KeyboardInterrupt).")
            print("Agent stopped by user.")
        except Exception as e:
            self.logger.error(f"Unexpected error in agent loop: {e}", exc_info=True)
            print(f"Unexpected error in agent loop: {e}")

# -------------------------------------------------------------------
# Future Improvements
# -------------------------------------------------------------------
"""
1. **Connection Pooling**: Use SQLAlchemy's session/pool management for multiple concurrent 
   connections or better performance.
2. **Enhanced Error Handling**: Implement retry logic for transient DB issues; handle specific 
   SQLAlchemy exceptions individually.
3. **Notification System**: Integrate Slack/email alerts for critical cleanup events or errors.
4. **Docker & Orchestration**: Containerize the agent for simpler deployment in cloud environments.
5. **Testing**: Write unit tests and integration tests to ensure robust cleanup operations.
6. **Dynamic Cleanup Threshold**: Make "50" models a configurable parameter.
7. **Graceful Shutdown**: Implement advanced signals or callbacks for a clean shutdown process.
8. **Metrics**: Capture metrics on cleanup frequency, # of deletions, etc.
"""

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    cleaner = GlobalModelCleaner()
    cleaner.run()
