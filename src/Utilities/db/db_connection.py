# -------------------------------------------------------------------
# File: src/Utilities/db/db_connection.py
# Description: Establishes a connection to a PostgreSQL database 
#              using SQLAlchemy. Supports both synchronous and 
#              asynchronous operations with proper session handling.
# -------------------------------------------------------------------

import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager, asynccontextmanager
from dotenv import load_dotenv
import asyncio
import os

# -------------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------------
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_DBNAME = os.getenv("POSTGRES_DBNAME", "trading_robot_plug")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5434")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DBNAME}"
ASYNC_DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://')

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------
logger = logging.getLogger("DBConnection")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# -------------------------------------------------------------------
# SQLAlchemy Base
# -------------------------------------------------------------------
Base = declarative_base()

# -------------------------------------------------------------------
# DBConnection Class
# -------------------------------------------------------------------
class DBConnection:
    """Handles PostgreSQL connections and session management."""

    def __init__(self):
        """Initialize database connection, engines, and session makers."""
        logger.info("Initializing DBConnection...")

        # Create Synchronous Engine
        self.engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create Asynchronous Engine
        self.async_engine = create_async_engine(ASYNC_DATABASE_URL, pool_size=10, max_overflow=20)
        self.AsyncSessionLocal = sessionmaker(self.async_engine, class_=AsyncSession, expire_on_commit=False)

        logger.info("DBConnection initialized successfully.")

    # ------------------------------
    # Synchronous Database Session
    # ------------------------------
    @contextmanager
    def get_session(self) -> Session:
        """
        Provides a synchronous SQLAlchemy session.
        Usage:
            with db.get_session() as session:
                result = session.execute(text("SELECT * FROM table_name"))
        """
        session = None
        try:
            session = self.SessionLocal()
            logger.info("Synchronous PostgreSQL session created.")
            yield session
            session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            if session:
                session.rollback()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if session:
                session.rollback()
        finally:
            if session:
                session.close()
                logger.info("Synchronous PostgreSQL session closed.")

    # ------------------------------
    # Asynchronous Database Session
    # ------------------------------
    @asynccontextmanager
    async def get_async_session(self) -> AsyncSession:
        """
        Provides an asynchronous SQLAlchemy session.
        Usage:
            async with db.get_async_session() as session:
                result = await session.execute(text("SELECT * FROM table_name"))
        """
        async_session = None
        try:
            async_session = self.AsyncSessionLocal()
            logger.info("Asynchronous PostgreSQL session created.")
            yield async_session
            await async_session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Async database error: {e}")
            if async_session:
                await async_session.rollback()
        except Exception as e:
            logger.error(f"Unexpected async error: {e}")
            if async_session:
                await async_session.rollback()
        finally:
            if async_session:
                await async_session.close()
                logger.info("Asynchronous PostgreSQL session closed.")

    # ------------------------------
    # Create Tables
    # ------------------------------
    def create_tables(self):
        """Creates database tables using SQLAlchemy ORM."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully.")
        except SQLAlchemyError as e:
            logger.error(f"Error creating tables: {e}")

    async def create_tables_async(self):
        """Creates database tables asynchronously using SQLAlchemy ORM."""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully (async).")
        except SQLAlchemyError as e:
            logger.error(f"Error creating tables asynchronously: {e}")

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    """Example Usage for synchronous and asynchronous database operations."""
    db = DBConnection()

    # Sync Example
    try:
        with db.get_session() as session:
            logger.info("Running synchronous query...")
            result = session.execute(text("SELECT * FROM stock_trades LIMIT 10;"))
            for row in result.fetchall():
                print(row)
            logger.info("Synchronous query completed.")
    except Exception as e:
        logger.error(f"Sync example error: {e}")

    # Async Example
    async def async_example():
        try:
            async with db.get_async_session() as session:
                logger.info("Running asynchronous query...")
                result = await session.execute(text("SELECT * FROM stock_trades LIMIT 10;"))
                for row in result.fetchall():
                    print(row)
                logger.info("Asynchronous query completed.")
        except Exception as e:
            logger.error(f"Async example error: {e}")

    asyncio.run(async_example())

# -------------------------------------------------------------------
# Future Improvements:
# - Implement connection pool monitoring.
# - Extend compatibility with more database types.
# - Optimize async transaction handling for batch inserts.
# - Provide database health-checking methods.
# -------------------------------------------------------------------
