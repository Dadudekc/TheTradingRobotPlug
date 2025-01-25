# setup_database.sh

#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Define the database credentials and script paths
DB_USER="postgres"
DB_HOST="localhost"
DB_PORT="5434"
DB_NAME="trading_robot_plug"
DB_SCRIPTS_DIR="C:/Projects/TradingRobotPlug/db_scripts"

# Check if the SQL files exist before proceeding
REQUIRED_FILES=("create_tables.sql" "create_indexes.sql" "insert_sample_data.sql" "grant_permissions.sql")
for FILE in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${DB_SCRIPTS_DIR}/${FILE}" ]; then
        echo "Error: ${DB_SCRIPTS_DIR}/${FILE} does not exist."
        exit 1
    fi
done

# Execute SQL scripts in the correct order
psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -f "${DB_SCRIPTS_DIR}/create_tables.sql"
psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -f "${DB_SCRIPTS_DIR}/create_indexes.sql"
psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -f "${DB_SCRIPTS_DIR}/insert_sample_data.sql"
psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -f "${DB_SCRIPTS_DIR}/grant_permissions.sql"

# Optional: Run other scripts if needed
# psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -f "${DB_SCRIPTS_DIR}/clean_duplicates.sql"
# psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -f "${DB_SCRIPTS_DIR}/alter_stock_data_table.sql"

echo "Database setup complete!"
