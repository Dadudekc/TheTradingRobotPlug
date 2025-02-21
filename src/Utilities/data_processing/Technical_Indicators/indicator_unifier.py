# -*- coding: utf-8 -*-
"""
File: indicator_unifier.py
Location: src/Utilities/data_processing/Technical_Indicators/indicator_unifier.py

Description:
    The AllIndicatorsUnifier class aggregates all technical indicators
    (Volume, Volatility, Trend, Momentum, Machine Learning, Custom)
    into a single pipeline for a DataFrame. It offers extensive configuration
    to handle partial usage (only some indicators), parallel processing (optional),
    robust error handling, and logging.

    Key Features:
      - Fine-grained control over which indicator sets to apply.
      - Optional parallel indicator computation for large datasets.
      - Robust logging of missing columns or potential conflicts.
      - Integration with ConfigManager and DataStore for data persistence.
      - Hooks for custom pre-processing or post-processing steps.
"""

import logging
import pandas as pd
from typing import Optional, Dict, List, Callable, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from Utilities.config_manager import ConfigManager
from Utilities.data.data_store import DataStore
from Utilities.shared_utils import setup_logging

# Use the shared setup to get a logger for this script
logger = setup_logging(script_name="Arima_main")


# -------------------------------------------------------------------
# Lazy Import Helpers (to avoid circular imports)
# -------------------------------------------------------------------
def get_volume_indicators():
    from Utilities.data_processing.Technical_Indicators.volume_indicators import VolumeIndicators
    return VolumeIndicators

def get_volatility_indicators():
    from Utilities.data_processing.Technical_Indicators.volatility_indicators import VolatilityIndicators
    return VolatilityIndicators

def get_trend_indicators():
    from Utilities.data_processing.Technical_Indicators.trend_indicators import TrendIndicators
    return TrendIndicators

def get_momentum_indicators():
    from Utilities.data_processing.Technical_Indicators.momentum_indicators import MomentumIndicators
    return MomentumIndicators

def get_machine_learning_indicators():
    from Utilities.data_processing.Technical_Indicators.machine_learning_indicators import MachineLearningIndicators
    return MachineLearningIndicators

def get_custom_indicators():
    from Utilities.data_processing.Technical_Indicators.custom_indicators import CustomIndicators
    return CustomIndicators


class AllIndicatorsUnifier:
    """
    Applies a comprehensive set of technical indicators to a DataFrame,
    either sequentially or in parallel. Provides hooks for custom pre-/post-processing,
    and supports partial usage of indicator groups.
    """

    DEFAULT_OPTIONS: Dict[str, bool] = {
        "volume": True,
        "volatility": True,
        "trend": True,
        "momentum": True,
        "ml": True,
        "custom": True
    }

    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Optional[logging.Logger] = None,
        use_csv: bool = False,
        parallel: bool = False,
        max_workers: int = 4
    ) -> None:
        """
        Initialize the AllIndicatorsUnifier.

        Args:
            config_manager (ConfigManager): Loads environment variables and config.
            logger (Optional[logging.Logger]): Logger instance for logging. Defaults to a new logger.
            use_csv (bool): If True, enable CSV-based data saving in DataStore.
            parallel (bool): If True, process indicator groups in parallel.
            max_workers (int): Number of threads for parallel computation if parallel=True.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing AllIndicatorsUnifier...")

        self.config_manager = config_manager
        self.use_csv = use_csv
        self.parallel = parallel
        self.max_workers = max_workers

        # Data storage / retrieval
        self.data_store = DataStore(
            config=config_manager,
            logger=self.logger,
            use_csv=use_csv
        )

        # Lazy-load each indicator module to avoid circular imports.
        self.volume_indicators = get_volume_indicators()(logger=self.logger, data_store=self.data_store)
        self.volatility_indicators = get_volatility_indicators()(logger=self.logger, data_store=self.data_store)
        self.trend_indicators = get_trend_indicators()(logger=self.logger, data_store=self.data_store)
        self.momentum_indicators = get_momentum_indicators()(logger=self.logger, data_store=self.data_store)
        self.ml_indicators = get_machine_learning_indicators()(logger=self.logger, data_store=self.data_store)
        self.custom_indicators = get_custom_indicators()(
            logger=self.logger,
            data_store=self.data_store,
            config_manager=self.config_manager
        )

        self.logger.info("AllIndicatorsUnifier initialization complete.")

    def apply_all_indicators(
        self,
        df: pd.DataFrame,
        options: Optional[Dict[str, bool]] = None,
        pre_process_hook: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        post_process_hook: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Applies the requested indicator sets to the DataFrame.

        Args:
            df (pd.DataFrame): Input price data.
            options (Optional[Dict[str, bool]]): Flags for each indicator group.
            pre_process_hook (Optional[Callable[[pd.DataFrame], pd.DataFrame]]): Function to call before applying indicators.
            post_process_hook (Optional[Callable[[pd.DataFrame], pd.DataFrame]]): Function to call after applying indicators.

        Returns:
            pd.DataFrame: DataFrame augmented with technical indicators.
        """
        opts = options or self.DEFAULT_OPTIONS
        self.logger.debug(f"Indicator options: {opts}")
        df_result = df.copy()

        if pre_process_hook:
            self.logger.info("Running pre-process hook...")
            df_result = pre_process_hook(df_result)

        # Check for essential columns.
        required_cols = {"Close", "High", "Low", "Volume", "Open"}
        missing_cols = required_cols - set(df_result.columns)
        if missing_cols:
            self.logger.warning(f"Missing columns {missing_cols}. Some indicators may fail.")

        # Build tasks for each enabled indicator set.
        tasks: List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = []
        if opts.get("volume", True):
            tasks.append(("Volume Indicators", self.volume_indicators.apply_indicators))
        if opts.get("volatility", True):
            tasks.append(("Volatility Indicators", self.volatility_indicators.apply_indicators))
        if opts.get("trend", True):
            tasks.append(("Trend Indicators", self.trend_indicators.apply_indicators))
        if opts.get("momentum", True):
            tasks.append(("Momentum Indicators", self.momentum_indicators.apply_indicators))
        if opts.get("ml", True):
            tasks.append(("Machine Learning Indicators", self.ml_indicators.apply_indicators))
        if opts.get("custom", True):
            tasks.append(("Custom Indicators", self.custom_indicators.apply_indicators))

        self.logger.info(f"Applying indicators (parallel={self.parallel}, tasks={len(tasks)})...")
        if self.parallel and len(tasks) > 1:
            df_result = self._apply_indicators_parallel(df_result, tasks)
        else:
            df_result = self._apply_indicators_sequential(df_result, tasks)

        if post_process_hook:
            self.logger.info("Running post-process hook...")
            df_result = post_process_hook(df_result)

        self.logger.info("All requested indicators applied successfully.")
        return df_result

    def _apply_indicators_sequential(
        self,
        df: pd.DataFrame,
        tasks: List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]
    ) -> pd.DataFrame:
        """
        Sequentially applies each indicator set.

        Args:
            df (pd.DataFrame): Input DataFrame.
            tasks (List[Tuple[str, Callable]]): List of indicator tasks.

        Returns:
            pd.DataFrame: DataFrame after applying all tasks.
        """
        df_result = df
        for task_name, task_func in tasks:
            self.logger.info(f"Applying {task_name} (sequential)...")
            try:
                df_result = task_func(df_result)
            except Exception as e:
                self.logger.error(f"Error applying {task_name}: {e}", exc_info=True)
        return df_result

    def _apply_indicators_parallel(
        self,
        df: pd.DataFrame,
        tasks: List[Tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]]
    ) -> pd.DataFrame:
        """
        Applies indicator tasks in parallel using ThreadPoolExecutor.

        Args:
            df (pd.DataFrame): Input DataFrame.
            tasks (List[Tuple[str, Callable]]): List of indicator tasks.

        Returns:
            pd.DataFrame: Merged DataFrame with all partial results.
        """
        self.logger.info(f"Starting parallel indicator application with {self.max_workers} workers.")
        partial_results: Dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._run_indicator_task, task_name, task_func, df.copy()): task_name
                for task_name, task_func in tasks
            }

            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    partial_results[task_name] = future.result()
                    self.logger.info(f"Completed parallel task: {task_name}")
                except Exception as e:
                    self.logger.error(f"Error in parallel task {task_name}: {e}", exc_info=True)

        return self._merge_partial_results(df, partial_results)

    def _merge_partial_results(
        self,
        base_df: pd.DataFrame,
        partial_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merges partial DataFrame results from parallel tasks.

        Args:
            base_df (pd.DataFrame): The original DataFrame.
            partial_results (Dict[str, pd.DataFrame]): Results from each task.

        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        merged_df = base_df.copy()
        self.logger.info("Merging partial parallel results...")
        for task_name, task_df in partial_results.items():
            self.logger.debug(f"Merging results from: {task_name}")
            # Here, if duplicate columns exist, the task's result overwrites the original.
            for col in task_df.columns:
                merged_df[col] = task_df[col]
        return merged_df

    def _run_indicator_task(
        self,
        task_name: str,
        task_func: Callable[[pd.DataFrame], pd.DataFrame],
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Runs a single indicator task in a separate thread.

        Args:
            task_name (str): Name of the task.
            task_func (Callable): The function to apply.
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame result from the task.
        """
        self.logger.debug(f"Running task: {task_name} in parallel.")
        return task_func(df)

    def apply_select_indicators(
        self,
        df: pd.DataFrame,
        indicator_list: List[str]
    ) -> pd.DataFrame:
        """
        Applies only the selected indicator sets.

        Args:
            df (pd.DataFrame): The input DataFrame.
            indicator_list (List[str]): List of indicator groups to apply.
                Valid names: ["volume", "volatility", "trend", "momentum", "ml", "custom"].

        Returns:
            pd.DataFrame: DataFrame with only the selected indicators.
        """
        valid_keys = {"volume", "volatility", "trend", "momentum", "ml", "custom"}
        selected = {k: (k in indicator_list) for k in valid_keys}
        invalid = set(indicator_list) - valid_keys
        if invalid:
            self.logger.warning(f"Ignoring invalid indicators: {invalid}")

        return self.apply_all_indicators(df, options=selected)
