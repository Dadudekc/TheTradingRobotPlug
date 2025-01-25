# -------------------------------------------------------------------
# File: C:/Projects/TradingRobotPlug/src/Utilities/db/db_connection.py
# Description: Establishes a connection to a PostgreSQL database using SQLAlchemy.
#              Supports both synchronous and asynchronous operations.
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
import logging
from sqlalchemy import create_engine, text, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, scoped_session, Session, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager, asynccontextmanager
from pathlib import Path
import sys
from dotenv import load_dotenv
import asyncio
from datetime import datetime

# -------------------------------------------------------------------
# Project Path Setup
# -------------------------------------------------------------------
def get_project_root() -> Path:
    """Determines the project root directory based on the current file's location."""
    return Path(__file__).resolve().parents[3]

# Set up project paths and add necessary directories to sys.path
project_root = get_project_root()
utilities_dir = project_root / 'src' / 'Utilities'
sys.path.append(str(utilities_dir))

# Load environment variables
env_path = project_root / '.env'
if not env_path.exists():
    logging.warning(f"Environment file {env_path} does not exist. Please create it with the necessary configurations.")
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------
# Logger Configuration
# -------------------------------------------------------------------
logger = logging.getLogger("db_connection")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info("db_connection module loaded successfully.")
logger.info(f"Environment variables loaded from {env_path}")

# -------------------------------------------------------------------
# ConfigManager Initialization
# -------------------------------------------------------------------
try:
    from Utilities.config_manager import ConfigManager
    config_manager = ConfigManager()
    logger.info("ConfigManager initialized successfully.")
except ModuleNotFoundError as e:
    logger.error(f"Failed to import ConfigManager: {e}")
    raise

# -------------------------------------------------------------------
# Database Configuration
# -------------------------------------------------------------------
POSTGRES_HOST = config_manager.get('POSTGRES_HOST', 'localhost')
POSTGRES_DBNAME = config_manager.get('POSTGRES_DBNAME', 'trading_robot_plug')
POSTGRES_USER = config_manager.get('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = config_manager.get('POSTGRES_PASSWORD', 'password')
POSTGRES_PORT = config_manager.get('POSTGRES_PORT', '5434')

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DBNAME}"
ASYNC_DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://')
logger.info(f"Database URL: {DATABASE_URL}")
logger.info(f"Async Database URL: {ASYNC_DATABASE_URL}")

# Declarative Base
Base = declarative_base()

# -------------------------------------------------------------------
# SQLAlchemy Engine and Session Setup
# -------------------------------------------------------------------

# Synchronous Engine and Session
def create_engine_from_url(db_url: str):
    """Creates a SQLAlchemy engine using the provided database URL."""
    try:
        engine = create_engine(db_url, pool_size=10, max_overflow=20)
        logger.info("SQLAlchemy engine created successfully.")
        return engine
    except Exception as e:
        logger.error(f"Failed to create SQLAlchemy engine: {e}", exc_info=True)
        raise

engine = create_engine_from_url(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
logger.info("SQLAlchemy session maker initialized successfully.")

# Context Manager for Synchronous Sessions
@contextmanager
def create_postgres_session() -> Session:
    """Creates a PostgreSQL session using a context manager."""
    session = None
    try:
        session = SessionLocal()
        logger.info("PostgreSQL session created successfully.")
        yield session
        session.commit()
        logger.info("Session committed successfully.")
    except SQLAlchemyError as e:
        logger.error(f"Database error during session: {e}", exc_info=True)
        if session:
            session.rollback()
            logger.info("Session rolled back due to an error.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while creating session: {e}", exc_info=True)
        if session:
            session.rollback()
            logger.info("Session rolled back due to an unexpected error.")
        raise
    finally:
        if session:
            session.close()
            logger.info("PostgreSQL session closed.")

# -------------------------------------------------------------------
# Async SQLAlchemy Engine and Session Setup
# -------------------------------------------------------------------

# Asynchronous Engine and Session
async def create_async_engine_from_url(db_url: str):
    """Creates an asynchronous SQLAlchemy engine using the provided database URL."""
    try:
        async_engine = create_async_engine(db_url, pool_size=10, max_overflow=20)
        logger.info("Asynchronous SQLAlchemy engine created successfully.")
        return async_engine
    except Exception as e:
        logger.error(f"Failed to create asynchronous SQLAlchemy engine: {e}", exc_info=True)
        raise

async_engine = asyncio.run(create_async_engine_from_url(ASYNC_DATABASE_URL))
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# Async Context Manager for Sessions
@asynccontextmanager
async def create_async_postgres_session() -> AsyncSession:
    """Creates an asynchronous PostgreSQL session using a context manager."""
    async_session = None
    try:
        async_session = AsyncSessionLocal()
        logger.info("Asynchronous PostgreSQL session created successfully.")
        yield async_session
        await async_session.commit()
        logger.info("Asynchronous session committed successfully.")
    except SQLAlchemyError as e:
        logger.error(f"Async database error during session: {e}", exc_info=True)
        if async_session:
            await async_session.rollback()
            logger.info("Asynchronous session rolled back due to an error.")
        raise
    except Exception as e:
        logger.error(f"Unexpected async error during session: {e}", exc_info=True)
        if async_session:
            await async_session.rollback()
            logger.info("Asynchronous session rolled back due to an unexpected error.")
        raise
    finally:
        if async_session:
            await async_session.close()
            logger.info("Asynchronous PostgreSQL session closed.")

# -------------------------------------------------------------------
# Define get_db_session Function
# -------------------------------------------------------------------
def get_db_session():
    """
    Provides a new SQLAlchemy session.
    
    Usage:
        with get_db_session() as session:
            # Your database operations
    """
    return create_postgres_session()

# -------------------------------------------------------------------
# NEW: Define the StockTrade Model
# -------------------------------------------------------------------
class StockTrade(Base):
    __tablename__ = 'stock_trades'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    action = Column(String, nullable=False)  # 'buy' or 'sell'
    quantity = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

# -------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    """Example Usage for synchronous and asynchronous database operations."""
    # Sync Example
    try:
        with create_postgres_session() as session:
            logger.info("Example sync session started.")
            result = session.execute(text("SELECT * FROM stock_data LIMIT 10;"))
            data = result.fetchall()
            for row in data:
                print(row)
            logger.info("Example sync session completed.")
    except Exception as e:
        logger.error(f"Error during sync example usage: {e}")

    # Async Example
    async def async_example():
        try:
            async with create_async_postgres_session() as session:
                logger.info("Example async session started.")
                result = await session.execute(text("SELECT * FROM stock_data LIMIT 10;"))
                data = result.fetchall()
                for row in data:
                    print(row)
                logger.info("Example async session completed.")
        except Exception as e:
            logger.error(f"Error during async example usage: {e}")

    asyncio.run(async_example())

# -------------------------------------------------------------------
# Future Improvements
# -------------------------------------------------------------------
"""
Future Improvements:
1. **Enhance Async Capabilities**:
    - Support for async transactions and batch operations.

2. **Extend Configuration Flexibility**:
    - Allow database configurations to be loaded from different sources.

3. **Integrate with FastAPI**:
    - Optimize the module for real-time processing with async frameworks like FastAPI.

4. **Connection Pool Monitoring**:
    - Implement monitoring for connection pools to track performance.

5. **Automated Testing**:
    - Set up unit tests for both synchronous and asynchronous sessions.
"""
