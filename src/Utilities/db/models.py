# -------------------------------------------------------------------
# File: C:/Projects/TradingRobotPlug/src/Utilities/db/models.py
# Description:
#   Contains SQLAlchemy ORM models for database tables.
#   Includes the NewsArticle model and the Models table with
#   extended fields to accommodate long-term lifecycle management.
# -------------------------------------------------------------------

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text,
    create_engine, func
)
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
    ORM Model for the 'news_data' table that stores news articles
    fetched from various sources.
    """
    __tablename__ = 'news_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    title = Column(String, nullable=True)
    content = Column(Text, nullable=True)  # Store larger text in TEXT column
    published_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    source = Column(String, nullable=True)
    url = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)

    def __repr__(self):
        return (
            f"<NewsArticle("
            f"id={self.id}, symbol={self.symbol}, title={self.title}, "
            f"published_at={self.published_at}, sentiment_score={self.sentiment_score}"
            f")>"
        )

# -------------------------------------------------------------------
# Models Table (Extended for Long-Term Goals)
# -------------------------------------------------------------------
class Models(Base):
    """
    ORM Model for the 'models' table that stores information on various ML or analytics models.
    
    Extended Fields (long-term growth):
    1. lifecycle_stage: Helps track the model's progress (e.g., 'development', 'production', 'retired').
    2. owner: Person or team responsible for this model.
    3. roi_estimate: Potential ROI or performance metric for business justification.
    """
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    description = Column(Text, nullable=True)

    # ---- Extended Fields for Long-Term Goals ----
    lifecycle_stage = Column(String, default="development", nullable=False)
    owner = Column(String, nullable=True)
    roi_estimate = Column(Float, nullable=True)

    def __repr__(self):
        return (
            f"<Models("
            f"id={self.id}, model_name={self.model_name}, model_type={self.model_type}, "
            f"lifecycle_stage={self.lifecycle_stage}, owner={self.owner}, "
            f"created_at={self.created_at}"
            f")>"
        )

# -------------------------------------------------------------------
# Example Initialization (Optional)
# -------------------------------------------------------------------
if __name__ == "__main__":
    """
    This section is for quick testing or initialization.
    It ensures that the NewsArticle and Models tables are set up properly.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # Example: Adjust this PostgreSQL URL for your environment
    DATABASE_URL = "postgresql+psycopg2://postgres:password@localhost:5434/trading_robot_plug"
    
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    print("news_data table created (or verified) successfully.")
    print("models table created (or verified) successfully.")

    # (Optional) Quick test to insert data
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    try:
        # Insert a sample model with extended fields
        new_model = Models(
            model_name="LongTermPredictionModel",
            model_type="neural_network",
            description="A long-horizon forecast model for strategic planning.",
            lifecycle_stage="production",
            owner="ResearchTeam",
            roi_estimate=1.85
        )
        session.add(new_model)
        session.commit()
        print("Inserted sample model with extended fields successfully!")
    except Exception as e:
        session.rollback()
        print(f"Error inserting sample model: {e}")
    finally:
        session.close()
