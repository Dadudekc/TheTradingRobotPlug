# dtype_handler.py
import pandas as pd
import numpy as np
from Utilities.shared_utils import setup_logging

# Initialize module-level logger
logger = setup_logging(script_name="dtype_handler")

def enforce_dtype(df: pd.DataFrame, dtype_map: dict = None) -> pd.DataFrame:
    """
    Ensures all columns in a DataFrame conform to expected dtypes.
    Uses float32 for efficiency but ensures float64 for high-precision values.
    
    :param df: DataFrame to process
    :param dtype_map: Optional dictionary mapping column names to expected dtypes
    :return: Processed DataFrame with correct dtypes
    """
    if dtype_map is None:
        dtype_map = {
            'mfi': np.float32,
            'mfi_new': np.float32,
            'obv': np.float32,
            'vwap': np.float64,  # VWAP may require higher precision
            'adl': np.float64,  # ADL may require higher precision
            'cmf': np.float32,
            'volume_oscillator': np.float32,
            'volume': np.int64,
            'open': np.float32,
            'high': np.float32,
            'low': np.float32,
            'close': np.float32
        }

    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
                logger.info(f"✅ Converted '{col}' to {dtype}")  # Using module-level logger
            except ValueError:
                logger.warning(f"Could not convert {col} to {dtype}, using safe conversion.")
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)

    return df
