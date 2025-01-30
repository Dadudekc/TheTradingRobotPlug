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
import logging
import io

# Add project root to the Python path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[1]
data_processing_dir = project_root / "src" / "Data_Processing"
sys.path.append(str(data_processing_dir))

# Import the VolatilityIndicators class
from src.Data_Processing.Technical_Indicators.volatility_indicators import VolatilityIndicators

class TestVolatilityIndicators(unittest.TestCase):
    def setUp(self):
        """
        Setup a sample DataFrame for testing.
        """
        self.indicators = VolatilityIndicators()
        self.sample_data = {
            'date': pd.date_range(start='2022-01-01', periods=100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(50, 100, 100),
            'close': np.random.uniform(75, 175, 100),
            'volume': np.random.randint(1000, 10000, 100)
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
            with self.subTest(column=column):
                self.assertIn(column, downcasted_df.columns, f"Column '{column}' should be present after downcasting.")
                self.assertEqual(downcasted_df[column].dtype, dtype, f"Column '{column}' should be of type {dtype}.")

    def test_add_bollinger_bands(self):
        """
        Test adding Bollinger Bands.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        expected_columns = ['Bollinger_Low', 'Bollinger_Mid', 'Bollinger_High']
        for column in expected_columns:
            with self.subTest(column=column):
                self.assertIn(column, df.columns, f"Column '{column}' should be added by Bollinger Bands.")

        # Check for NaN in the first (window_size -1) rows
        self.assertTrue(df['Bollinger_Low'].isna().sum() >= 19, "Expected NaN values for initial rows in Bollinger_Low.")

    def test_add_standard_deviation(self):
        """
        Test adding Standard Deviation.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        self.assertIn('Standard_Deviation', df.columns, "Column 'Standard_Deviation' should be added.")
        self.assertTrue(df['Standard_Deviation'].isna().sum() >= 19, "Expected NaN values for initial rows in Standard_Deviation.")

    def test_add_historical_volatility(self):
        """
        Test adding Historical Volatility.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        self.assertIn('Historical_Volatility', df.columns, "Column 'Historical_Volatility' should be added.")
        # Historical Volatility should have reasonable values after the first window_size rows
        self.assertFalse(df['Historical_Volatility'].isna().all(), "Historical Volatility should have non-NaN values after initial rows.")
        # Check that the first window_size-1 rows may have lower or NaN values
        self.assertTrue(df['Historical_Volatility'].isna().sum() <= 19, "Too many NaN values in Historical_Volatility.")

    def test_add_chandelier_exit(self):
        """
        Test adding Chandelier Exit.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        self.assertIn('Chandelier_Exit_Long', df.columns, "Column 'Chandelier_Exit_Long' should be added.")
        # No NaN expected as min_periods=1
        self.assertFalse(df['Chandelier_Exit_Long'].isna().all(), "Chandelier_Exit_Long should not have all NaN values.")

    def test_add_keltner_channel(self):
        """
        Test adding Keltner Channel.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        expected_columns = ['Keltner_Channel_Low', 'Keltner_Channel_Basis', 'Keltner_Channel_High']
        for column in expected_columns:
            with self.subTest(column=column):
                self.assertIn(column, df.columns, f"Column '{column}' should be added by Keltner Channel.")

        # Check for NaN in the first (window_size -1) rows
        self.assertTrue(df['Keltner_Channel_Low'].isna().sum() >= 19, "Expected NaN values for initial rows in Keltner_Channel_Low.")

    def test_add_moving_average_envelope(self):
        """
        Test adding Moving Average Envelope.
        """
        df = self.indicators.apply_indicators(self.df.copy())
        expected_columns = ['MAE_Upper', 'MAE_Lower']
        for column in expected_columns:
            with self.subTest(column=column):
                self.assertIn(column, df.columns, f"Column '{column}' should be added by Moving Average Envelope.")

        # Check for NaN in the first (window_size -1) rows
        self.assertTrue(df['MAE_Upper'].isna().sum() >= 9, "Expected NaN values for initial rows in MAE_Upper.")

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
            with self.subTest(column=column):
                self.assertIn(column, df.columns, f"Column '{column}' should be present after applying all indicators.")

    def test_process_large_dataset(self):
        """
        Test processing a large dataset by chunking.
        """
        # Create a large synthetic CSV file
        large_data = {
            'date': pd.date_range(start='2022-01-01', periods=50000),
            'high': np.random.uniform(100, 200, 50000),
            'low': np.random.uniform(50, 100, 50000),
            'close': np.random.uniform(75, 175, 50000),
            'volume': np.random.randint(1000, 10000, 50000)
        }
        large_df = pd.DataFrame(large_data)
        csv_path = Path(project_root / 'tests' / 'large_test_data.csv')
        
        # Ensure the directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        large_df.to_csv(csv_path, index=False)

        try:
            # Process the large dataset
            self.indicators.process_large_dataset(str(csv_path), chunksize=50000)
        
            # Check if the processed file exists
            processed_csv = csv_path.with_name('large_test_data_processed.csv')
            self.assertTrue(processed_csv.exists(), "Processed CSV file should exist.")

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
                with self.subTest(column=column):
                    self.assertIn(column, processed_df.columns, f"Column '{column}' should be present in the processed CSV.")

        finally:
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
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.INFO)
        
        # Get the specific logger used in volatility_indicators.py
        logger = logging.getLogger('src.Data_Processing.Technical_Indicators.volatility_indicators')
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)

        try:
            # Process streaming data
            self.indicators.process_streaming_data(data_stream_generator(25), window_size=20)

            # Flush and retrieve log contents
            ch.flush()
            log_contents = log_capture_string.getvalue()
            # Uncomment the next line to debug log contents
            # print("Captured Log Contents:\n", log_contents)
        
            # Ensure correct assertion
            self.assertIn("Processed streaming data point", log_contents, "Expected log message not found.")
        finally:
            # Remove handler to clean up
            logger.removeHandler(ch)

if __name__ == '__main__':
    unittest.main()
