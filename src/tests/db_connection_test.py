from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

db_name = os.getenv("MYSQL_DB_NAME")
db_user = os.getenv("MYSQL_DB_USER")
db_password = os.getenv("MYSQL_DB_PASSWORD")
db_host = os.getenv("MYSQL_DB_HOST")

# Connect to PlanetScale database
connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
engine = create_engine(connection_string)

# Test connection
with engine.connect() as connection:
    result = connection.execute("SELECT 1")
    print("Connection test result:", [row for row in result])
