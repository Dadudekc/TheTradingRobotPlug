# -------------------------------------------------------------------
# File Path: src/Utilities/db/db_handler.py
# Description: Handles database connections and queries using SQLAlchemy.
#              Supports both PostgreSQL and SQLite with session management.
# -------------------------------------------------------------------

import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from dotenv import load_dotenv
from pathlib import Path

# -------------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------------
env_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------
# SQLAlchemy Base
# -------------------------------------------------------------------
Base = declarative_base()

# -------------------------------------------------------------------
# DBHandler Class
# -------------------------------------------------------------------
class DBHandler:
    """Handles database connections and basic queries using SQLAlchemy."""

    def __init__(self, use_sqlite: bool = False, logger: logging.Logger = None):
        """
        Initialize the DBHandler.

        Args:
            use_sqlite (bool): Whether to use SQLite instead of PostgreSQL.
            logger (logging.Logger): Logger instance for logging.
        """
        self.use_sqlite = use_sqlite
        self.logger = logger or self._setup_logger()

        self.engine = self._create_engine()
        self.Session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
        self.logger.info("DBHandler initialized successfully.")

    def _setup_logger(self):
        """Sets up logging for the database handler."""
        logger = logging.getLogger('DBHandler')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _create_engine(self):
        """Creates a SQLAlchemy engine based on the chosen database (PostgreSQL or SQLite)."""
        try:
            if self.use_sqlite:
                db_path = os.getenv('SQLITE_DB_PATH', 'data/databases/trading_data.db')
                engine = create_engine(f'sqlite:///{db_path}', echo=False)
                self.logger.info(f"SQLite engine created at {db_path}.")
            else:
                user = os.getenv('POSTGRES_USER', 'postgres')
                password = os.getenv('POSTGRES_PASSWORD', 'password')
                host = os.getenv('POSTGRES_HOST', 'localhost')
                port = os.getenv('POSTGRES_PORT', '5432')
                dbname = os.getenv('POSTGRES_DBNAME', 'trading_robot_plug')
                DATABASE_URL = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}'
                engine = create_engine(DATABASE_URL, echo=False)
                self.logger.info("PostgreSQL engine created successfully.")
            return engine
        except Exception as e:
            self.logger.error(f"Error creating database engine: {e}")
            raise

    def get_session(self):
        """Returns a new SQLAlchemy session."""
        return self.Session()

    def create_tables(self):
        """Creates all tables in the database."""
        try:
            Base.metadata.create_all(self.engine)
            self.logger.info("Database tables created successfully.")
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    def close(self):
        """Closes the database connection and session."""
        try:
            self.Session.remove()
            self.engine.dispose()
            self.logger.info("Database connection closed.")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    """Example usage for both PostgreSQL and SQLite."""
    
    # PostgreSQL Example
    db = DBHandler(use_sqlite=False)
    db.create_tables()

    try:
        with db.get_session() as session:
            print("Database session ready for use.")
    finally:
        db.close()

    # SQLite Example
    sqlite_db = DBHandler(use_sqlite=True)
    sqlite_db.create_tables()

    try:
        with sqlite_db.get_session() as session:
            print("SQLite session ready for use.")
    finally:
        sqlite_db.close()

# -------------------------------------------------------------------
# Future Improvements
# - Implement connection pooling for optimized performance.
# - Add CRUD operations for easy database interactions.
# - Integrate Alembic for handling database migrations.
# -------------------------------------------------------------------
