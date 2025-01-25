# -------------------------------------------------------------------
# File Path: C:/Projects/TradingRobotPlug/src/Utilities/db_handler.py
# Description: Handles database connections and basic queries using SQLAlchemy for PostgreSQL or SQLite.
# -------------------------------------------------------------------

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from dotenv import load_dotenv
from pathlib import Path
import logging

# -------------------------------------------------------------------
# Environment Setup
# -------------------------------------------------------------------
env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------
# SQLAlchemy Base
# -------------------------------------------------------------------
Base = declarative_base()

# -------------------------------------------------------------------
# DatabaseHandler Class
# -------------------------------------------------------------------
class DatabaseHandler:
    def __init__(self, use_sqlite: bool = False, logger: logging.Logger = None):
        """
        Initialize the DatabaseHandler class.

        :param use_sqlite: Flag to determine if SQLite should be used instead of PostgreSQL.
        :param logger: Logger instance for logging, defaults to None for internal setup.
        """
        self.use_sqlite = use_sqlite
        self.logger = logger or logging.getLogger('db_handler')
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        self.engine = self._create_engine()
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        self.logger.info("DatabaseHandler initialized.")

    def _create_engine(self):
        """Create a SQLAlchemy engine based on the environment."""
        try:
            if self.use_sqlite:
                db_path = os.getenv('SQLITE_DB_PATH', 'data/databases/trading_data.db')
                engine = create_engine(f'sqlite:///{db_path}', echo=False)
                self.logger.info(f"Successfully created SQLite engine at {db_path}.")
            else:
                user = os.getenv('POSTGRES_USER', 'postgres')
                password = os.getenv('POSTGRES_PASSWORD', 'password')
                host = os.getenv('POSTGRES_HOST', 'localhost')
                port = os.getenv('POSTGRES_PORT', '5432')
                dbname = os.getenv('POSTGRES_DBNAME', 'trading_robot_plug')
                DATABASE_URL = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}'
                engine = create_engine(DATABASE_URL, echo=False)
                self.logger.info("Successfully created PostgreSQL engine.")
            return engine
        except Exception as e:
            self.logger.error(f"Error creating engine: {e}")
            raise

    def get_session(self):
        """Provide a new SQLAlchemy session."""
        return self.Session()

    def create_tables(self):
        """Create all tables in the database based on the Base metadata."""
        try:
            Base.metadata.create_all(self.engine)
            self.logger.info("All tables created successfully.")
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    def close(self):
        """Close the SQLAlchemy engine and session."""
        try:
            self.Session.remove()
            self.engine.dispose()
            self.logger.info("Database connection closed.")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")

# -------------------------------------------------------------------
# Future Improvements
# -------------------------------------------------------------------
# 1. Implement connection pooling configurations for enhanced performance.
# 2. Add methods for common CRUD operations to simplify database interactions.
# 3. Integrate Alembic for database migrations.
