# db_scripts/backup_database.sql

# Backup command to run from the terminal
pg_dump -U postgres -h localhost -p 5434 trading_robot_plug > trading_robot_plug_backup.sql
