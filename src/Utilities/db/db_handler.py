# -------------------------------------------------------------------
# File Path: src/Utilities/db/db_handler.py
# Description: Handles database connections and queries using SQLAlchemy.
#              Supports both PostgreSQL and SQLite with session management.
# -------------------------------------------------------------------

import os
import logging
from sqlalchemy import create_engine, exc
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
    """
    Handles database connections and basic queries using SQLAlchemy.
    Supports both PostgreSQL and SQLite with robust session management.
    Also provides placeholder CRUD operations for ease of use.
    """

    def __init__(self, use_sqlite: bool = False, logger: logging.Logger = None,
                 pool_size: int = 10, max_overflow: int = 20):
        """
        Initialize the DBHandler.

        Args:
            use_sqlite (bool): Whether to use SQLite instead of PostgreSQL.
            logger (logging.Logger): Logger instance for logging.
            pool_size (int): Size of the connection pool (PostgreSQL only).
            max_overflow (int): Maximum number of connections beyond the pool_size.
        """
        self.use_sqlite = use_sqlite
        self.logger = logger or self._setup_logger()
        self.pool_size = pool_size
        self.max_overflow = max_overflow

        self.engine = self._create_engine()
        self.Session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
        self.logger.info("DBHandler initialized successfully.")

    def _setup_logger(self) -> logging.Logger:
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
                engine = create_engine(
                    DATABASE_URL,
                    echo=False,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow
                )
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
    # CRUD Operation Placeholders
    # -------------------------------------------------------------------
    def create_record(self, record: Base) -> None:
        """
        Inserts a new record into the database.

        Args:
            record (Base): An instance of a SQLAlchemy model.
        """
        session = self.get_session()
        try:
            session.add(record)
            session.commit()
            self.logger.info("Record created successfully.")
        except exc.SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error creating record: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def read_records(self, model: Base, filters: dict = None) -> list:
        """
        Reads records from the database based on optional filters.

        Args:
            model (Base): A SQLAlchemy model class.
            filters (dict): A dictionary of filters to apply (optional).

        Returns:
            list: A list of model instances.
        """
        session = self.get_session()
        try:
            query = session.query(model)
            if filters:
                query = query.filter_by(**filters)
            records = query.all()
            self.logger.info(f"Read {len(records)} records from {model.__tablename__}.")
            return records
        except exc.SQLAlchemyError as e:
            self.logger.error(f"Error reading records: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def update_record(self, model: Base, filters: dict, update_data: dict) -> int:
        """
        Updates records in the database.

        Args:
            model (Base): A SQLAlchemy model class.
            filters (dict): Filters to identify the records.
            update_data (dict): Data to update.

        Returns:
            int: Number of rows updated.
        """
        session = self.get_session()
        try:
            query = session.query(model).filter_by(**filters)
            rows_updated = query.update(update_data)
            session.commit()
            self.logger.info(f"Updated {rows_updated} records in {model.__tablename__}.")
            return rows_updated
        except exc.SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error updating records: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def delete_record(self, model: Base, filters: dict) -> int:
        """
        Deletes records from the database.

        Args:
            model (Base): A SQLAlchemy model class.
            filters (dict): Filters to identify the records to delete.

        Returns:
            int: Number of rows deleted.
        """
        session = self.get_session()
        try:
            rows_deleted = session.query(model).filter_by(**filters).delete()
            session.commit()
            self.logger.info(f"Deleted {rows_deleted} records from {model.__tablename__}.")
            return rows_deleted
        except exc.SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error deleting records: {e}", exc_info=True)
            raise
        finally:
            session.close()

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
            print("PostgreSQL session ready for use.")
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
# Future Improvements:
# - Integrate Alembic for handling database migrations.
# - Expand CRUD methods to support bulk operations.
# - Implement more advanced connection pooling metrics and health checks.
# -------------------------------------------------------------------
