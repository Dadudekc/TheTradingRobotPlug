ALTER TABLE stock_data
ALTER COLUMN id TYPE integer USING id::integer;

ALTER TABLE stock_data
ALTER COLUMN id SET NOT NULL;

-- Create a sequence for auto-incrementing IDs
CREATE SEQUENCE stock_data_id_seq;

-- Set `id` to use the sequence for auto-increment
ALTER TABLE stock_data
ALTER COLUMN id SET DEFAULT nextval('stock_data_id_seq');

-- Populate `id` for rows where it is NULL
UPDATE stock_data
SET id = nextval('stock_data_id_seq')
WHERE id IS NULL;

-- Set `id` as the primary key
ALTER TABLE stock_data
ADD PRIMARY KEY (id);
ALTER TABLE stock_data
ALTER COLUMN symbol SET NOT NULL,
ALTER COLUMN timestamp SET NOT NULL,
ALTER COLUMN open SET NOT NULL,
ALTER COLUMN high SET NOT NULL,
ALTER COLUMN low SET NOT NULL,
ALTER COLUMN close SET NOT NULL,
ALTER COLUMN volume SET NOT NULL;
ALTER TABLE stock_data
ALTER COLUMN stochastic TYPE double precision USING NULLIF(stochastic, '')::double precision,
ALTER COLUMN rsi TYPE double precision USING NULLIF(rsi, '')::double precision,
ALTER COLUMN williams_r TYPE double precision USING NULLIF(williams_r, '')::double precision,
ALTER COLUMN rate_of_change TYPE double precision USING NULLIF(rate_of_change, '')::double precision,
ALTER COLUMN macd TYPE double precision USING NULLIF(macd, '')::double precision,
ALTER COLUMN bollinger_bands TYPE double precision USING NULLIF(bollinger_bands, '')::double precision,
ALTER COLUMN proprietary_prediction TYPE double precision USING NULLIF(proprietary_prediction, '')::double precision;
