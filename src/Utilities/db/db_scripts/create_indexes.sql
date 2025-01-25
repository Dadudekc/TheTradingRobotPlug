-- db_scripts/create_indexes.sql

-- Index on the symbol column for faster lookups by stock symbol
CREATE INDEX IF NOT EXISTS idx_symbol ON stock_data(symbol);

-- Index on the timestamp column for faster range queries
CREATE INDEX IF NOT EXISTS idx_timestamp ON stock_data(timestamp);

-- Index on model_name for quick model lookups
CREATE INDEX IF NOT EXISTS idx_model_name ON model_data(model_name);
