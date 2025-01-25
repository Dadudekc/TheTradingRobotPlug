-- db_scripts/alter_stock_data_table.sql

BEGIN;

-- Convert empty strings to NULL for numeric fields
UPDATE stock_data SET open = NULL WHERE open = '';
UPDATE stock_data SET high = NULL WHERE high = '';
UPDATE stock_data SET low = NULL WHERE low = '';
UPDATE stock_data SET close = NULL WHERE close = '';
UPDATE stock_data SET volume = NULL WHERE volume = '';

-- Alter columns with proper casting and error handling
ALTER TABLE stock_data
    ALTER COLUMN id TYPE INTEGER USING id::integer,
    ALTER COLUMN symbol TYPE VARCHAR(10),
    ALTER COLUMN timestamp TYPE TIMESTAMP USING timestamp::timestamp,
    ALTER COLUMN open TYPE DOUBLE PRECISION USING open::double precision,
    ALTER COLUMN high TYPE DOUBLE PRECISION USING high::double precision,
    ALTER COLUMN low TYPE DOUBLE PRECISION USING low::double precision,
    ALTER COLUMN close TYPE DOUBLE PRECISION USING close::double precision,
    ALTER COLUMN volume TYPE INTEGER USING volume::integer,
    ALTER COLUMN stochastic TYPE DOUBLE PRECISION USING stochastic::double precision,
    ALTER COLUMN rsi TYPE DOUBLE PRECISION USING rsi::double precision,
    ALTER COLUMN williams_r TYPE DOUBLE PRECISION USING williams_r::double precision,
    ALTER COLUMN rate_of_change TYPE DOUBLE PRECISION USING rate_of_change::double precision,
    ALTER COLUMN macd TYPE DOUBLE PRECISION USING macd::double precision,
    ALTER COLUMN bollinger_bands TYPE DOUBLE PRECISION USING bollinger_bands::double precision,
    ALTER COLUMN proprietary_prediction TYPE DOUBLE PRECISION USING proprietary_prediction::double precision;

COMMIT;
