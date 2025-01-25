# -------------------------------------------------------------------
# Description: Connects to a PostgreSQL database using SQLAlchemy,
#              retrieves stock data for 'AAPL' from the 'stock_data' table,
#              and fetches the first 10 records sorted by date in ascending order.
# -------------------------------------------------------------------

from sqlalchemy import create_engine, text
import pandas as pd

# Define the PostgreSQL connection details
DATABASE_URL = "postgresql://postgres:password@localhost:5434/trading_robot_plug"

# Create the SQLAlchemy engine for connecting to the database
engine = create_engine(DATABASE_URL)

# Define the query to retrieve data for AAPL from the stock_data table
query = """
SELECT * FROM stock_data
WHERE symbol = 'AAPL'
ORDER BY "Date" ASC
LIMIT 10;  -- Fetching only the first 10 records for brevity
"""

# Execute the query and fetch the results into a DataFrame
with engine.connect() as connection:
    df_aapl = pd.read_sql_query(text(query), con=connection)

# Display the fetched data
print(df_aapl)

# -------------------------------------------------------------------
# Example Usage:
#     Simply run the script to connect to the database and fetch the data:
#         python fetch_aapl_data.py
#     The output will display the first 10 records for 'AAPL' from the 'stock_data' table.
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Future Improvements:
#     - Parameterize the symbol and limit values to make the script more flexible.
#     - Add error handling for database connection issues and query execution.
#     - Implement logging for better tracking of query execution and results.
#     - Allow date range as input to filter data for specific periods.
# -------------------------------------------------------------------
