-- db_scripts/grant_permissions.sql

-- Replace 'your_user' with the actual PostgreSQL username
GRANT ALL PRIVILEGES ON DATABASE trading_robot_plug TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
