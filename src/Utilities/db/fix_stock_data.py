import psycopg2
from psycopg2 import sql
import logging

# Setup logging
logging.basicConfig(
    filename="D:/TradingRobotPlug2/logs/fix_stock_data.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Database connection settings
DB_SETTINGS = {
    "dbname": "trading_robot_plug",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5434",
}

def connect_db():
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(**DB_SETTINGS)
        logging.info("✅ Connected to database!")
        return conn
    except Exception as e:
        logging.error(f"❌ Database connection failed: {e}")
        return None

def execute_query(conn, query, params=None, fetch=False):
    """Executes a query and fetches results if needed."""
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
            if fetch:
                return cursor.fetchall()
            conn.commit()
    except Exception as e:
        logging.error(f"❌ Query failed: {e}")

def resolve_duplicates(conn):
    """Removes duplicate (symbol, Date) entries."""
    logging.info("🛠 Resolving duplicate (symbol, Date) entries...")
    execute_query(conn, '''
        CREATE TABLE IF NOT EXISTS stock_data_deduplicated AS
        SELECT DISTINCT ON (symbol, "Date") * 
        FROM stock_data 
        ORDER BY symbol, "Date", id DESC;
    ''')
    execute_query(conn, 'DROP TABLE stock_data;')
    execute_query(conn, 'ALTER TABLE stock_data_deduplicated RENAME TO stock_data;')
    logging.info("✅ Duplicates removed successfully.")

def update_missing_dates(conn):
    """Fills missing Date values using timestamp or assigns new ones."""
    logging.info("🛠 Updating missing dates using timestamp...")

    # Step 1: Update from timestamp
    execute_query(conn, '''
        UPDATE stock_data
        SET "Date" = DATE("timestamp")
        WHERE "Date" IS NULL AND "timestamp" IS NOT NULL;
    ''')
    logging.info("✅ Date values updated from timestamp!")

    # Step 2: Check for remaining NULLs
    remaining_nulls = execute_query(conn, 'SELECT COUNT(*) FROM stock_data WHERE "Date" IS NULL;', fetch=True)[0][0]

    if remaining_nulls > 0:
        logging.warning(f"⚠ {remaining_nulls} records still missing dates. Assigning sequential dates...")
        execute_query(conn, '''
            WITH ranked AS (
                SELECT id, symbol, 
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY id) AS rn
                FROM stock_data
                WHERE "Date" IS NULL
            )
            UPDATE stock_data
            SET "Date" = TO_DATE('2024-01-01', 'YYYY-MM-DD') + (ranked.rn - 1) * INTERVAL '1 day'
            FROM ranked
            WHERE stock_data.id = ranked.id AND stock_data."Date" IS NULL;
        ''')
        logging.info("✅ Remaining NULL dates assigned sequentially.")

def restore_constraints(conn):
    """Re-applies unique constraints and NOT NULL rules."""
    logging.info("🔒 Restoring constraints on (symbol, Date)...")

    # Restore unique constraint
    execute_query(conn, 'ALTER TABLE stock_data ADD CONSTRAINT uix_symbol_date UNIQUE (symbol, "Date");')
    logging.info("✅ Unique constraint restored!")

    # Enforce NOT NULL constraint
    execute_query(conn, 'ALTER TABLE stock_data ALTER COLUMN "Date" SET NOT NULL;')
    logging.info("✅ Date column is now NOT NULL!")

def clean_stock_data():
    """Main function to clean stock data in the database."""
    conn = connect_db()
    if not conn:
        return

    try:
        resolve_duplicates(conn)
        update_missing_dates(conn)
        restore_constraints(conn)

        final_null_count = execute_query(conn, 'SELECT COUNT(*) FROM stock_data WHERE "Date" IS NULL;', fetch=True)[0][0]
        if final_null_count == 0:
            logging.info("🚀 All Date values successfully fixed!")
        else:
            logging.warning(f"⚠ Warning: {final_null_count} records still have NULL dates.")

    except Exception as e:
        logging.error(f"❌ Error in fixing stock data: {e}")
    finally:
        conn.close()
        logging.info("🔌 Database connection closed.")

if __name__ == "__main__":
    clean_stock_data()
