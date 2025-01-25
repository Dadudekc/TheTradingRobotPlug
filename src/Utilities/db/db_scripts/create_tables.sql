-- db_scripts/create_tables.sql

-- Create stock_data table for storing historical stock data
CREATE TABLE IF NOT EXISTS stock_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume INTEGER NOT NULL,
    stochastic DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    williams_r DOUBLE PRECISION,
    rate_of_change DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    bollinger_bands DOUBLE PRECISION,
    proprietary_prediction DOUBLE PRECISION
);

-- Create model_data table for storing model metadata
CREATE TABLE IF NOT EXISTS model_data (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    training_date TIMESTAMP NOT NULL,
    hyperparameters JSONB,
    metrics JSONB,
    description TEXT
);

-- Add additional tables for other data if necessary
-- create_tables.sql

CREATE TABLE IF NOT EXISTS alpha_vantage_daily (
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    PRIMARY KEY (symbol, date)
);
