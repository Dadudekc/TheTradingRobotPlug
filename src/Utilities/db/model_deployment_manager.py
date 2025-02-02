# -------------------------------------------------------------------
# File: model_deployment_manager.py
# Location: C:/Projects/TradingRobotPlug/src/Utilities/db
# Description:
#   A class-based manager for model deployments, including table creation,
#   fetching top-performing models, saving new models, and more.
#   Extended with long-term goals in mind, so you can easily expand
#   to new functionalities like advanced analytics or multi-region deployments.
# -------------------------------------------------------------------

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Optional, List
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Integer, TIMESTAMP, Text, JSON, LargeBinary
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base

from logging.handlers import RotatingFileHandler

# -------------------------------------------------------------------
# Local Utilities from Your Project
# -------------------------------------------------------------------
from Utilities.config_manager import ConfigManager
from Utilities.db.db_handler import DBHandler
from Utilities.db.db_connection import DBConnection
from Utilities.db.models import Models
from Utilities.db.database_restructure import DatabaseRestructure
from Utilities.db.db_inspect_and_transfer import DBInspectAndTransfer
from Utilities.db.db_inspect_and_update import DBInspectAndUpdate
from Utilities.db.global_model_cleaner import GlobalModelCleaner
from Utilities.db.inspect_db_data import InspectDBData

# -------------------------------------------------------------------
# SQLAlchemy Declarations
# -------------------------------------------------------------------
Base = declarative_base()

class ModelDeployment(Base):
    """
    SQLAlchemy ORM for the 'model_deployment' table, storing model metadata and performance metrics.
    """
    __tablename__ = 'model_deployment'

    model_id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(255), nullable=False)
    deployment_date = Column(TIMESTAMP, nullable=False)
    description = Column(Text)
    model_version = Column(String(50))
    model_type = Column(String(50))
    performance_metrics = Column(JSON)
    accuracy = Column(Float)
    precision = Column(Float)
    sharpe_ratio = Column(Float)
    serialized_model = Column(LargeBinary, nullable=False)
    training_data_reference = Column(String(255))
    model_status = Column(String(50))

# -------------------------------------------------------------------
# Logger Utility
# -------------------------------------------------------------------
def create_logger(project_root: Path) -> logging.Logger:
    """
    Creates a logger with rotating file handlers for the model_deployment_manager.
    """
    log_name = "model_deployment_manager"
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}.log"

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3
        )
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

# -------------------------------------------------------------------
# Class-Based Manager
# -------------------------------------------------------------------
class ModelDeploymentManager:
    """
    A class to handle model deployments, including:
    - Table creation
    - Fetching top models
    - Saving new model records
    - Future expansions (analytics, multi-region, advanced logs)
    """

    def __init__(self, logger: logging.Logger):
        """
        Initializes the manager, sets up environment variables, and loads DB session.
        """
        self.logger = logger
        self.session_factory = None

        # Attempt to connect to DB via environment
        # The actual connection is done in create_postgres_session
        self.logger.debug("[ModelDeploymentManager] Initialized. Call 'connect_to_db' to finalize DB connection.")

    def connect_to_db(self) -> bool:
        """
        Creates a PostgreSQL session factory using environment variables.
        Returns True if successful, False otherwise.
        """
        try:
            POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
            POSTGRES_DBNAME = os.getenv('POSTGRES_DBNAME', 'trading_robot_plug')
            POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
            POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
            POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')

            database_url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DBNAME}"
            self.logger.debug(f"[ModelDeploymentManager] Connecting to {database_url}")

            engine = create_engine(database_url, pool_size=10, max_overflow=20)
            Base.metadata.create_all(engine)  # Ensure 'model_deployment' table
            self.logger.info("[ModelDeploymentManager] DB connection established, tables verified.")

            self.session_factory = sessionmaker(bind=engine)
            return True
        except SQLAlchemyError as e:
            self.logger.error(f"[ModelDeploymentManager] Error connecting to PostgreSQL: {e}")
            return False

    def create_table_if_not_exists(self) -> None:
        """
        Ensures the model_deployment table exists in the database.
        Useful if the table might be missing. Typically called automatically in `connect_to_db`.
        """
        try:
            # This is largely redundant if Base.metadata.create_all was called,
            # but you can add custom DDL checks here if needed.
            self.logger.debug("[ModelDeploymentManager] create_table_if_not_exists is a placeholder. 'Base.metadata.create_all' is used.")
        except Exception as e:
            self.logger.error(f"[ModelDeploymentManager] Error ensuring table creation: {e}")

    def fetch_top_models(self, limit: int = 5) -> Optional[List[Dict]]:
        """
        Retrieves top-performing models from 'model_deployment' table, sorted by accuracy, sharpe_ratio, precision.

        Returns:
            A list of dictionaries describing each model, or None on error.
        """
        if not self.session_factory:
            self.logger.error("[ModelDeploymentManager] Session factory is None; call 'connect_to_db' first.")
            return None

        session = self.session_factory()
        try:
            self.logger.debug("[ModelDeploymentManager] Fetching top models from DB.")
            results = session.query(
                ModelDeployment.model_id,
                ModelDeployment.model_name,
                ModelDeployment.accuracy,
                ModelDeployment.precision,
                ModelDeployment.sharpe_ratio,
                ModelDeployment.deployment_date
            ).order_by(
                ModelDeployment.accuracy.desc(),
                ModelDeployment.sharpe_ratio.desc(),
                ModelDeployment.precision.desc()
            ).limit(limit).all()

            self.logger.info(f"[ModelDeploymentManager] Retrieved {len(results)} top models.")
            return [row._asdict() for row in results]
        except SQLAlchemyError as e:
            self.logger.error(f"[ModelDeploymentManager] Error fetching top models: {e}")
            return None
        finally:
            session.close()

    def save_model(self, model_data: Dict) -> bool:
        """
        Saves a new model record into 'model_deployment' table.

        Args:
            model_data (Dict): Must contain at least:
                - 'model_name' (str)
                - 'deployment_date' (datetime)
                - 'serialized_model' (bytes)
                - 'accuracy', 'precision', 'sharpe_ratio' (floats)
                - any other relevant fields.
        Returns:
            bool: True if saved successfully, False otherwise.
        """
        if not self.session_factory:
            self.logger.error("[ModelDeploymentManager] Session factory is None; call 'connect_to_db' first.")
            return False

        session = self.session_factory()
        try:
            self.logger.debug(f"[ModelDeploymentManager] Saving model: {model_data.get('model_name', 'Unknown')}")
            record = ModelDeployment(
                model_name=model_data['model_name'],
                deployment_date=model_data['deployment_date'],
                description=model_data.get('description'),
                model_version=model_data.get('model_version'),
                model_type=model_data.get('model_type'),
                performance_metrics=model_data.get('performance_metrics'),
                accuracy=model_data.get('accuracy'),
                precision=model_data.get('precision'),
                sharpe_ratio=model_data.get('sharpe_ratio'),
                serialized_model=model_data['serialized_model'],
                training_data_reference=model_data.get('training_data_reference'),
                model_status=model_data.get('model_status')
            )
            session.add(record)
            session.commit()
            self.logger.info(f"[ModelDeploymentManager] Model '{model_data.get('model_name', 'Unknown')}' saved successfully.")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"[ModelDeploymentManager] Error saving model: {e}")
            return False
        finally:
            session.close()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Future expansions for long-term goals:
    # 1. Remote replication across multiple regions or DB servers
    # 2. Advanced analytics: e.g., combine multiple performance metrics
    # 3. Automatic model archiving after certain conditions
    # 4. Subscription-based notifications on new top models
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
def main():
    """
    Demonstrates usage of ModelDeploymentManager: connecting to DB, 
    fetching top models, and saving a new model example.
    Extended with placeholders for long-term expansions.
    """
    # Setup environment
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print("[model_deployment_manager] No .env found; using environment variables as-is.")

    # Create logger
    logger = create_logger(project_root)
    logger.info("[model_deployment_manager] Starting main demonstration of ModelDeploymentManager.")

    # Instantiate manager
    manager = ModelDeploymentManager(logger=logger)

    # Connect to DB
    if not manager.connect_to_db():
        logger.error("[model_deployment_manager] Could not establish DB connection. Exiting.")
        sys.exit(1)

    # (Optional) explicitly ensure table
    manager.create_table_if_not_exists()

    # Fetch top models
    top_models = manager.fetch_top_models(limit=5)
    if top_models is not None:
        logger.info("[model_deployment_manager] Top 5 Models:")
        for m in top_models:
            logger.info(m)
    else:
        logger.warning("[model_deployment_manager] Could not fetch top models.")

    # Demonstrate saving a new model
    example_model_data = {
        'model_name': 'Awesome Model v2',
        'deployment_date': datetime.now(),
        'description': 'Demo model with new architecture for alpha signals.',
        'model_version': '2.0',
        'model_type': 'neural_network',
        'performance_metrics': {
            'accuracy': 0.91,
            'precision': 0.86,
            'sharpe_ratio': 1.8
        },
        'accuracy': 0.91,
        'precision': 0.86,
        'sharpe_ratio': 1.80,
        'serialized_model': b'\x80\x04\x95\x11\x00\x00\x00\x00\x00\x00\x00C\rNewBinaryData\x94.',
        'training_data_reference': 'v2_dataset_ref',
        'model_status': 'active'
    }
    if manager.save_model(example_model_data):
        logger.info(f"[model_deployment_manager] '{example_model_data['model_name']}' saved successfully.")
    else:
        logger.warning(f"[model_deployment_manager] Failed to save '{example_model_data['model_name']}'.")

if __name__ == "__main__":
    main()
