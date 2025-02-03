# -------------------------------------------------------------------
# File Path: src/Utilities/data_processing/preprocessing/dtype_validator.py
# Description: Class-based DataFrame dtype validation and correction.
#              Uses project-wide logging from shared_utils.py.
# -------------------------------------------------------------------

import pandas as pd
import numpy as np
from Utilities.shared_utils import setup_logging


class DtypeValidator:
    """
    Class for validating and correcting DataFrame dtypes.

    Attributes:
        logger (logging.Logger): Project-wide logger.
        expected_dtypes (dict): Expected column dtypes for validation.
    """

    DEFAULT_DTYPES = {
        "mfi": "float32",
        "obv": "float32",
        "vwap": "float32",
        "adl": "float32",
        "cmf": "float32",
        "volume_oscillator": "float32"
    }

    def __init__(self, expected_dtypes=None):
        """
        Initializes the DtypeValidator with expected dtypes.

        Args:
            expected_dtypes (dict, optional): Custom expected dtypes.
        """
        self.logger = setup_logging(script_name="dtype_validator")
        self.expected_dtypes = expected_dtypes if expected_dtypes else self.DEFAULT_DTYPES

    def validate_dtypes(self, df: pd.DataFrame) -> dict:
        """
        Checks for dtype mismatches in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to validate.

        Returns:
            dict: Columns with dtype mismatches.
        """
        dtype_issues = {}
        for col, expected_dtype in self.expected_dtypes.items():
            if col in df.columns and df[col].dtype.name != expected_dtype:
                dtype_issues[col] = (df[col].dtype.name, expected_dtype)

        if dtype_issues:
            self.logger.warning("⚠️ Dtype Issues Detected:")
            for col, (actual, expected) in dtype_issues.items():
                self.logger.warning(f"  - Column '{col}' has dtype {actual}, expected {expected}.")
        else:
            self.logger.info("✅ No dtype issues found.")

        return dtype_issues

    def fix_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts incorrect dtypes to the expected ones.

        Args:
            df (pd.DataFrame): DataFrame with potential dtype mismatches.

        Returns:
            pd.DataFrame: Updated DataFrame with corrected dtypes.
        """
        dtype_issues = self.validate_dtypes(df)

        if dtype_issues:
            self.logger.info("🔄 Converting problematic columns to expected dtypes...")
            for col, (_, expected_dtype) in dtype_issues.items():
                try:
                    df[col] = df[col].astype(expected_dtype)
                    self.logger.info(f"✅ Converted '{col}' to {expected_dtype}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to convert '{col}' to {expected_dtype}: {e}", exc_info=True)

            # Recheck after conversion
            final_issues = self.validate_dtypes(df)
            if not final_issues:
                self.logger.info("✅ All dtype issues resolved.")
            else:
                self.logger.error("❌ Some columns could not be converted:")
                for col, (actual, expected) in final_issues.items():
                    self.logger.error(f"  - Column '{col}' is still {actual}, expected {expected}.")

        return df

    def validate_and_fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs both validation and automatic fixing of dtypes.

        Args:
            df (pd.DataFrame): DataFrame to process.

        Returns:
            pd.DataFrame: Cleaned DataFrame with corrected dtypes.
        """
        self.logger.info("🔍 Running dtype validation and correction...")
        return self.fix_dtypes(df)
