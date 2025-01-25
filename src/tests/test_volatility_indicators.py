# -------------------------------------------------------------------
# File: test_volatility_indicators.py
# Location: tests
# Description: Test script for volatility_indicators.py
# -------------------------------------------------------------------

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to the Python path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
utilities_dir = project_root / "src" / "Utilities"
sys.path.append(str(utilities_dir))

# Import the VolatilityIndicators class
from volatility_indicators import VolatilityIndicators

class TestVolatilityIndicators(unittest.TestCase):
    def setUp(self):
        """
        Setup a sample DataFrame for testing.
        """
        self.indicators = VolatilityIndicators()
        self.sample_data = {
            'date': pd.date_range(start='2022-01-01', periods=100),
            'high': pd.Series(np.random.uniform(100, 200, 100)),
            'low': pd.Series(np.random.uniform(50, 100, 100)),
            'close': pd.Series(np.random.uniform(75, 175, 100)),
            'volume': pd.Series(np.random.randint(1000, 10000, 100))
        }
        self.df = pd.DataFrame(self.sample_data)

    def test_downcast_dataframe(self):
        """
        Test downcasting of DataFrame to optimize memory.
        """
        original_dtypes = self.df.dtypes.copy()
        downcasted_df = self.indicators.downcast_dataframe(self.df.copy())
        expected_dtypes = {
            'date': 'datetime64[ns]',
            'high': 'float32',
            'low': 'float32',
            'close': 'float32',
            'volume': 'int32'
        }
        for column, dtype in expected_dtypes.items():
            self.assertEqual(downcasted_df[column].dtype, dtype, f"{column} should be {dtype}")

    def test_add_bollinger_bands(self):
        """
        Test adding Bollinger Bands.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        self.assertIn('Bollinger_Low', df.columns)
        self.assertIn('Bollinger_Mid', df.columns)
        self.assertIn('Bollinger_High', df.columns)
        # Check for NaN in the first (window_size -1) rows
        self.assertTrue(df['Bollinger_Low'].isna().sum() >= 19)

    def test_add_standard_deviation(self):
        """
        Test adding Standard Deviation.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        self.assertIn('Standard_Deviation', df.columns)
        self.assertTrue(df['Standard_Deviation'].isna().sum() >= 19)

    def test_add_historical_volatility(self):
        """
        Test adding Historical Volatility.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        self.assertIn('Historical_Volatility', df.columns)
        # Historical Volatility should have reasonable values after the first window_size rows
        self.assertFalse(df['Historical_Volatility'].isna().all())
        # Check that the first window_size-1 rows may have lower or NaN values
        self.assertTrue(df['Historical_Volatility'].isna().sum() <= 19)

    def test_add_chandelier_exit(self):
        """
        Test adding Chandelier Exit.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        self.assertIn('Chandelier_Exit_Long', df.columns)
        # No NaN expected as min_periods=1
        self.assertFalse(df['Chandelier_Exit_Long'].isna().all())

    def test_add_keltner_channel(self):
        """
        Test adding Keltner Channel.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        self.assertIn('Keltner_Channel_Low', df.columns)
        self.assertIn('Keltner_Channel_Basis', df.columns)
        self.assertIn('Keltner_Channel_High', df.columns)
        # Check for NaN in the first (window_size -1) rows
        self.assertTrue(df['Keltner_Channel_Low'].isna().sum() >= 19)

    def test_add_moving_average_envelope(self):
        """
        Test adding Moving Average Envelope.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        self.assertIn('MAE_Upper', df.columns)
        self.assertIn('MAE_Lower', df.columns)
        # Check for NaN in the first (window_size -1) rows
        self.assertTrue(df['MAE_Upper'].isna().sum() >= 9)

    def test_apply_indicators(self):
        """
        Test applying all indicators together.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        expected_columns = [
            'Bollinger_Low', 'Bollinger_Mid', 'Bollinger_High',
            'Standard_Deviation', 'Historical_Volatility',
            'Chandelier_Exit_Long',
            'Keltner_Channel_Low', 'Keltner_Channel_Basis', 'Keltner_Channel_High',
            'MAE_Upper', 'MAE_Lower'
        ]
        for column in expected_columns:
            self.assertIn(column, df.columns)

    def test_process_large_dataset(self):
        """
        Test processing a large dataset by chunking.
        """
        # Create a large synthetic CSV file
        large_data = {
            'date': pd.date_range(start='2022-01-01', periods=50000),
            'high': pd.Series(np.random.uniform(100, 200, 50000)),
            'low': pd.Series(np.random.uniform(50, 100, 50000)),
            'close': pd.Series(np.random.uniform(75, 175, 50000)),
            'volume': pd.Series(np.random.randint(1000, 10000, 50000))
        }
        large_df = pd.DataFrame(large_data)
        csv_path = Path(project_root / 'tests' / 'large_test_data.csv')
        large_df.to_csv(csv_path, index=False)

        # Process the large dataset
        self.indicators.process_large_dataset(str(csv_path), chunksize=50000)

        # Check if the processed file exists
        processed_csv = csv_path.with_name('large_test_data_processed.csv')
        self.assertTrue(processed_csv.exists())

        # Optionally, read a few rows to verify
        processed_df = pd.read_csv(processed_csv, nrows=5)
        expected_columns = [
            'Bollinger_Low', 'Bollinger_Mid', 'Bollinger_High',
            'Standard_Deviation', 'Historical_Volatility',
            'Chandelier_Exit_Long',
            'Keltner_Channel_Low', 'Keltner_Channel_Basis', 'Keltner_Channel_High',
            'MAE_Upper', 'MAE_Lower'
        ]
        for column in expected_columns:
            self.assertIn(column, processed_df.columns)

        # Clean up
        processed_csv.unlink()
        csv_path.unlink()

    def test_process_streaming_data(self):
        """
        Test processing streaming data.
        """
        # Define a simple data stream generator
        def data_stream_generator(n):
            for i in range(n):
                yield {
                    'close': np.random.uniform(75, 175),
                    'high': np.random.uniform(100, 200),
                    'low': np.random.uniform(50, 100),
                    'volume': np.random.randint(1000, 10000)
                }

        # Capture logs using a custom logger
        import io
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.INFO)
        self.indicators.logger.addHandler(ch)

        # Process streaming data
        self.indicators.process_streaming_data(data_stream_generator(25), window_size=20)

        # Retrieve log contents
        log_contents = log_capture_string.getvalue()
        self.assertIn("Processed latest data point", log_contents)

        # Remove the handler
        self.indicators.logger.removeHandler(ch)

if __name__ == '__main__':
    unittest.main()
