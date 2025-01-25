# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/src/Utilities/global_model_cleaner.py
# Description: 
#     This script monitors the PostgreSQL global database and ensures 
#     that only the top 50 models (based on MSE score) are retained. 
#     It periodically checks the database and deletes models exceeding 
#     this limit. Logging is set up to track the cleanup process.
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import time
import logging
import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List
# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------
# Dynamically set the project root based on the current file's location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]  # Adjusted for the project structure
utilities_dir = project_root / 'src' / 'Utilities'
model_dir = project_root / 'SavedModels'
model_utils = project_root / 'src' / 'model_training' / 'utils'
data_processing_dir = project_root / 'src' / 'Data_Processing'

# Add the necessary directories to the Python path
sys.path.extend([
    str(utilities_dir),
    str(model_dir),
    str(model_utils),
    str(data_processing_dir)
])

# -------------------------------------------------------------------
# Import ConfigManager and Logging Setup
# -------------------------------------------------------------------
try:
    from config_handling.config_manager import ConfigManager
    from config_handling.logging_setup import setup_logging
except ModuleNotFoundError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------
# Load environment variables from .env file
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------
log_dir = project_root / 'logs' / 'Utilities'
log_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logging(str(log_dir / 'global_model_cleaner'))
logger.info("Logging initialized for global_model_cleaner.py")

# -------------------------------------------------------------------
# Configuration Manager Initialization
# -------------------------------------------------------------------
config_manager = ConfigManager()
logger.info("ConfigManager initialized successfully.")

# -------------------------------------------------------------------
# Global Configuration Variables
# -------------------------------------------------------------------
POSTGRES_USER = config_manager.get('POSTGRES_USER')
POSTGRES_PASSWORD = config_manager.get('POSTGRES_PASSWORD')
POSTGRES_HOST = config_manager.get('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = config_manager.get('POSTGRES_PORT', '5432')
POSTGRES_DBNAME = config_manager.get('POSTGRES_DBNAME', 'trading_robot_plug')
CHECK_INTERVAL_SECONDS = int(config_manager.get("CHECK_INTERVAL_SECONDS", 600))  # Check interval (default: 10 minutes)

# Create the PostgreSQL connection URL
DATABASE_URL = f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DBNAME}'

# -------------------------------------------------------------------
# Function Definitions
# -------------------------------------------------------------------

def connect_to_global_db() -> Optional[create_engine]:
    """
    Connect to the PostgreSQL database for storing top models.
    
    Returns:
        SQLAlchemy Engine object if connection is successful, None otherwise.
    """
    try:
        engine = create_engine(DATABASE_URL)
        logger.info("Successfully connected to PostgreSQL database.")
        return engine
    except SQLAlchemyError as e:
        logger.error(f"Error connecting to PostgreSQL database: {e}")
        return None

def fetch_all_models(engine) -> List[tuple]:
    """
    Fetch all models from the global database sorted by MSE score in ascending order.
    
    Args:
        engine: SQLAlchemy Engine object for the database connection.
    
    Returns:
        List of tuples containing model data (id, model_name, mse_score, model_blob).
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text('SELECT id, model_name, mse_score, model_blob FROM models ORDER BY mse_score ASC'))
            models = result.fetchall()
        logger.info(f"Fetched {len(models)} models from the database.")
        return models
    except SQLAlchemyError as e:
        logger.error(f"Error fetching models: {e}")
        return []

def delete_model_by_id(engine, model_id: int):
    """
    Delete a model from the global database by its ID.
    
    Args:
        engine: SQLAlchemy Engine object for the database connection.
        model_id (int): ID of the model to be deleted.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text('DELETE FROM models WHERE id = :id'), {'id': model_id})
            logger.info(f"Deleted model with ID: {model_id}")
    except SQLAlchemyError as e:
        logger.error(f"Error deleting model with ID {model_id}: {e}")

def clean_global_database(engine):
    """
    Clean the global database, ensuring only the top 50 models (by MSE) are retained.
    Models beyond the top 50 are deleted.
    
    Args:
        engine: SQLAlchemy Engine object for the database connection.
    """
    logger.info("Starting database cleanup process.")
    
    # Fetch all models from the global database sorted by MSE
    models = fetch_all_models(engine)
    
    # Check if there are more than 50 models
    if len(models) > 50:
        logger.info(f"Found {len(models)} models, initiating cleanup to retain top 50 models.")
        
        # Loop through models beyond the top 50 and delete them
        for model in models[50:]:
            model_id, model_name, mse_score, _ = model
            logger.info(f"Deleting model: {model_name} with MSE: {mse_score} (ID: {model_id})")
            delete_model_by_id(engine, model_id)
        
        logger.info(f"Cleanup completed. Retained top 50 models and deleted {len(models) - 50} models.")
    else:
        logger.info(f"Database contains {len(models)} models. No cleanup required.")

def start_agent():
    """
    Start the agent that monitors the global database and cleans it periodically.
    This agent runs as long as the script is active, checking the database every CHECK_INTERVAL_SECONDS.
    """
    logger.info(f"Starting database cleaning agent. Monitoring every {CHECK_INTERVAL_SECONDS / 60:.1f} minutes...")
    print(f"Starting database cleaning agent. Monitoring every {CHECK_INTERVAL_SECONDS / 60:.1f} minutes...")
    
    engine = connect_to_global_db()
    if not engine:
        logger.error("Failed to start agent due to database connection issues.")
        print("Failed to start agent due to database connection issues.")
        return
    
    try:
        while True:
            # Clean the database
            clean_global_database(engine)
            
            # Sleep for the interval before checking again
            logger.info(f"Sleeping for {CHECK_INTERVAL_SECONDS} seconds before next check.")
            time.sleep(CHECK_INTERVAL_SECONDS)
    
    except KeyboardInterrupt:
        logger.info("Agent stopped by user.")
        print("Agent stopped by user.")
    except Exception as e:
        logger.error(f"Unexpected error in the agent: {e}")
        print(f"Unexpected error in the agent: {e}")

# -------------------------------------------------------------------
# Future Improvements
# -------------------------------------------------------------------
"""
Future Improvements:
1. **Implement Connection Pooling**:
    - Use SQLAlchemy's connection pooling features to handle multiple concurrent connections efficiently.
    
2. **Enhanced Error Handling**:
    - Implement retry mechanisms for transient database connection errors.
    - Categorize and handle different types of SQLAlchemy errors more granularly.
    
3. **Logging Enhancements**:
    - Add file-based logging with log rotation to manage log file sizes.
    - Include more contextual information in logs, such as timestamps and function names.
    
4. **Notification System**:
    - Integrate a notification system (e.g., email alerts, Slack notifications) to inform about critical events or errors.
    
5. **Configuration Validation**:
    - Add validation for environment variables to ensure all necessary configurations are set before starting the agent.
    
6. **Dockerization**:
    - Containerize the agent using Docker for easier deployment and scalability.
    
7. **Metrics Collection**:
    - Collect and monitor metrics related to the cleanup process, such as number of deletions per run, time taken, etc.
    
8. **Unit and Integration Testing**:
    - Develop comprehensive tests to ensure each function behaves as expected, enhancing reliability.
    
9. **Dynamic Cleanup Threshold**:
    - Make the cleanup threshold (e.g., top 50 models) configurable via environment variables or a configuration file.
    
10. **Graceful Shutdown Handling**:
    - Implement mechanisms to gracefully shut down the agent, ensuring that ongoing operations are completed or safely terminated.
"""

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
# To run the global_model_cleaner agent, execute this script:
#     python global_model_cleaner.py
# The agent will continuously monitor the global database and clean it 
# by retaining only the top 50 models based on MSE score at intervals 
# defined by CHECK_INTERVAL_SECONDS.

if __name__ == "__main__":
    start_agent()
