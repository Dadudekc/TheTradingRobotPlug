-- db_scripts/clean_duplicates.sql

-- Remove duplicate entries based on symbol and timestamp, keeping the latest entry
DELETE FROM stock_data a
USING stock_data b
WHERE a.id < b.id
AND a.symbol = b.symbol
AND a.timestamp = b.timestamp;
