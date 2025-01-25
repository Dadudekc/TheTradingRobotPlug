-- db_scripts/insert_sample_data.sql

-- Insert sample stock data for AAPL
INSERT INTO stock_data (symbol, timestamp, open, high, low, close, volume)
VALUES
('AAPL', '2024-01-01 09:30:00', 150.00, 155.00, 149.00, 152.00, 1000000),
('AAPL', '2024-01-02 09:30:00', 152.00, 158.00, 151.00, 157.00, 1200000);

-- Insert a sample model record
INSERT INTO model_data (model_name, model_type, training_date, hyperparameters, metrics, description)
VALUES
('GradientBoosting_AAPL', 'GradientBoosting', NOW(), '{"n_estimators": 100, "learning_rate": 0.1}', '{"rmse": 0.05}', 'Model trained on AAPL data for price prediction.');
