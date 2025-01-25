# db_scripts/restore_database.sql

# Restore command to run from the terminal
psql -U postgres -h localhost -p 5434 trading_robot_plug < trading_robot_plug_backup.sql
