# File: feature_engineering.py
# Location: C:\TheTradingRobotPlug\Scripts\model_training\hyper_parameter\feature_engineering.py
# Description: This script provides a class for automated feature engineering using Featuretools, including validation, logging, and saving/loading of the feature matrix.

import featuretools as ft
import pandas as pd
import logging
from typing import List, Tuple, Optional

class FeatureEngineering:
    def __init__(self, df: pd.DataFrame, target_column: str, index_column: str = 'index'):
        self.df = df
        self.target_column = target_column
        self.index_column = index_column
        self.setup_logging()
        self.validate_data()

    def setup_logging(self) -> None:
        """
        Set up logging for the class.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Logging is set up.")

    def validate_data(self) -> None:
        """
        Validate the input DataFrame, checking for the presence and correct types of index and target columns.
        """
        logging.info("Validating input data...")
        if self.index_column not in self.df.columns:
            logging.error(f"Index column '{self.index_column}' not found in dataframe.")
            raise ValueError(f"Index column '{self.index_column}' not found in dataframe.")
        if not pd.api.types.is_integer_dtype(self.df[self.index_column]):
            logging.error(f"Index column '{self.index_column}' must be of integer type.")
            raise ValueError(f"Index column '{self.index_column}' must be of integer type.")
        if self.target_column not in self.df.columns:
            logging.error(f"Target column '{self.target_column}' not found in dataframe.")
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe.")
        logging.info("Data validation successful.")

    def automated_feature_engineering(
        self, 
        agg_primitives: Optional[List[str]] = None, 
        trans_primitives: Optional[List[str]] = None, 
        max_depth: int = 2
    ) -> Tuple[pd.DataFrame, List[ft.Feature]]:
        """
        Perform automated feature engineering using Featuretools.

        Args:
            agg_primitives (Optional[List[str]]): Aggregation primitives to apply.
            trans_primitives (Optional[List[str]]): Transformation primitives to apply.
            max_depth (int): Maximum depth of the feature engineering graph.

        Returns:
            Tuple[pd.DataFrame, List[ft.Feature]]: The generated feature matrix and the list of feature definitions.
        """
        if agg_primitives is None:
            agg_primitives = ['mean', 'sum', 'count', 'max', 'min']
        if trans_primitives is None:
            trans_primitives = ['day', 'year', 'month', 'weekday', 'cum_sum', 'cum_mean']

        try:
            logging.info("Starting automated feature engineering...")
            es = ft.EntitySet(id='data')
            es = es.add_dataframe(dataframe_name='main', dataframe=self.df, index=self.index_column)
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name='main',
                agg_primitives=agg_primitives,
                trans_primitives=trans_primitives,
                max_depth=max_depth
            )

            logging.info("Feature engineering completed successfully.")
            return feature_matrix, feature_defs

        except Exception as e:
            logging.error(f"Feature engineering failed: {e}")
            raise

    def save_feature_matrix(self, feature_matrix: pd.DataFrame, filename: str) -> None:
        """
        Save the generated feature matrix to a CSV file.

        Args:
            feature_matrix (pd.DataFrame): The feature matrix to save.
            filename (str): The file path to save the feature matrix to.
        """
        try:
            feature_matrix.to_csv(filename, index=False)
            logging.info(f"Feature matrix saved to {filename}")
        except IOError as e:
            logging.error(f"Failed to save feature matrix: {e}")
            raise

    def load_feature_matrix(self, filename: str) -> pd.DataFrame:
        """
        Load a feature matrix from a CSV file.

        Args:
            filename (str): The file path to load the feature matrix from.

        Returns:
            pd.DataFrame: The loaded feature matrix.
        """
        try:
            feature_matrix = pd.read_csv(filename)
            logging.info(f"Feature matrix loaded from {filename}")
            return feature_matrix
        except IOError as e:
            logging.error(f"Failed to load feature matrix: {e}")
            raise
