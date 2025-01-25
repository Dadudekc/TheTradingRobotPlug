# -------------------------------------------------------------------
# File: C:/Projects/TradingRobotPlug/src/Utilities/db/models.py
# Description: Contains SQLAlchemy ORM models for database tables.
#              Defines the NewsArticle model and other necessary models.
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.orm import declarative_base
from datetime import datetime

# -------------------------------------------------------------------
# Base Declaration
# -------------------------------------------------------------------
Base = declarative_base()

# -------------------------------------------------------------------
# NewsArticle Model
# -------------------------------------------------------------------
class NewsArticle(Base):
    """
    ORM Model for the 'news_data' table that stores news articles.
    """
    __tablename__ = 'news_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    title = Column(String, nullable=True)
    content = Column(String, nullable=True)
    published_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    source = Column(String, nullable=True)
    url = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)

    def __repr__(self):
        return f"<NewsArticle(id={self.id}, symbol={self.symbol}, published_at={self.published_at}, sentiment_score={self.sentiment_score})>"

# -------------------------------------------------------------------
# Example Initialization (Optional)
# -------------------------------------------------------------------
if __name__ == "__main__":
    """
    This section is for testing purposes. It ensures that the model is set up correctly.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    DATABASE_URL = "postgresql+psycopg2://postgres:password@localhost:5434/trading_robot_plug"
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    print("NewsArticle table created successfully.")
