# ==================== Description ====================
# Model Evaluation Utilities
#
# Description:
# This module provides a unified, extensible framework for evaluating machine learning models within the TradingRobotPlug project. It supports:
#
# - **Regression Models**: Evaluates Mean Squared Error (MSE), R² Score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
# - **Classification Models**: Evaluates Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
# - **Time Series Models**: Evaluates LSTM and ARIMA models with specialized metrics.
# - **Trading Models**: Includes specialized financial metrics like Sharpe Ratio, Maximum Drawdown, Sortino Ratio, Calmar Ratio, and more.
# - **Model Comparison and Validation**: Compares new models against baseline models to assess improvements, with configurable thresholds.
# - **Logging and Error Handling**: Robust logging with customizable handlers, and comprehensive error handling for production reliability.
# - **Configuration Management**: Centralized configuration for metrics thresholds and logging levels.
#
# Key Features:
# 1. **Enterprise-Level Scalability**: Modular, scalable design allowing integration into large-scale systems.
# 2. **Advanced Financial Metrics**: Provides extensive financial metrics essential for evaluating trading strategies.
# 3. **Robust Logging and Error Handling**: Uses Python's logging module with customizable handlers and levels, and includes detailed exception handling.
# 4. **Type Annotations and Documentation**: Fully type-annotated code with comprehensive docstrings for maintainability and clarity.
# 5. **Configuration and Extensibility**: Centralized configuration management and easy extensibility for new metrics and model types.
#
# Dependencies:
# - numpy
# - scikit-learn
# - pandas
# - typing
# - logging
# - dataclasses (Python 3.7+)
#
# Usage:
# To use this module, import the `ModelEvaluator` class, instantiate it, and call the relevant methods.
#
# Example:
# from model_evaluation import ModelEvaluator, get_logger
#
# # Initialize logger
# logger = get_logger('evaluation_log', log_file='evaluation.log', level=logging.INFO)
#
# # Initialize evaluator
# evaluator = ModelEvaluator(logger=logger)
#
# # Evaluate regression model
# regression_metrics = evaluator.evaluate_regression(y_true, y_pred)
#
# # Validate new model performance
# validation_passed = evaluator.validate_new_model_performance(
#     new_model, old_model, X_val, y_val, model_type='regression', threshold=0.05
# )
#
# Notes:
# - The module is designed to be integrated into larger systems, with thread-safe operations and minimal side-effects.
# - Custom metrics can be added by extending the `ModelEvaluator` class.

import logging
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass, field

def get_logger(
    name: str = 'model_evaluation',
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

class EvaluationError(Exception):
    """Custom exception for evaluation errors."""
    pass

@dataclass
class EvaluationConfig:
    regression_threshold: float = 0.05
    classification_threshold: float = 0.01
    risk_free_rate: float = 0.03
    logger: logging.Logger = field(default_factory=lambda: get_logger())

class ModelEvaluator:
    """
    Provides methods to evaluate machine learning models for regression, classification, time series, and trading.

    Attributes:
        config (EvaluationConfig): Configuration for evaluation thresholds and settings.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.logger = self.config.logger

    def evaluate_regression(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Evaluate regression model performance using various metrics.

        Args:
            y_true (np.ndarray or pd.Series): True target values.
            y_pred (np.ndarray or pd.Series): Predicted target values.

        Returns:
            Dict[str, float]: Dictionary containing regression metrics.
        """
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2_score': r2, 'mape': mape}
            self.logger.info(f"Regression metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error evaluating regression model: {e}")
            raise EvaluationError("Failed to evaluate regression model.") from e

    def evaluate_classification(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Evaluate classification model performance using various metrics.

        Args:
            y_true (np.ndarray or pd.Series): True labels.
            y_pred (np.ndarray or pd.Series): Predicted labels.

        Returns:
            Dict[str, Any]: Dictionary containing classification metrics.
        """
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist()
            }
            self.logger.info(f"Classification metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error evaluating classification model: {e}")
            raise EvaluationError("Failed to evaluate classification model.") from e

    def evaluate_trading_metrics(
        self,
        predictions: Union[np.ndarray, pd.Series],
        y_true: Union[np.ndarray, pd.Series],
        scaler: Optional[Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate financial trading metrics for model predictions.

        Args:
            predictions (np.ndarray or pd.Series): Model predictions.
            y_true (np.ndarray or pd.Series): True target values.
            scaler (Any, optional): Scaler object for inverse scaling.

        Returns:
            Dict[str, float]: Dictionary containing financial metrics.
        """
        try:
            if scaler:
                predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

            returns = np.diff(predictions) / predictions[:-1]
            avg_return = np.mean(returns)
            std_return = np.std(returns)

            sharpe_ratio = (
                (avg_return - self.config.risk_free_rate / 252) / std_return * np.sqrt(252)
                if std_return != 0 else 0
            )

            negative_returns = returns[returns < 0]
            std_negative = np.std(negative_returns) if len(negative_returns) > 0 else 0
            sortino_ratio = (
                (avg_return - self.config.risk_free_rate / 252) / std_negative * np.sqrt(252)
                if std_negative != 0 else 0
            )

            cumulative_returns = np.cumprod(1 + returns) - 1
            drawdown = cumulative_returns - np.maximum.accumulate(cumulative_returns)
            max_drawdown = np.min(drawdown)

            calmar_ratio = (
                (avg_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
            )

            metrics = {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio
            }
            self.logger.info(f"Trading metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error evaluating trading metrics: {e}")
            raise EvaluationError("Failed to evaluate trading metrics.") from e

    def evaluate_lstm_model(
        self,
        model: Any,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series]
    ) -> Dict[str, float]:
        """
        Evaluate an LSTM model's performance.

        Args:
            model (Any): Trained LSTM model.
            X_test (np.ndarray or pd.DataFrame): Test feature set.
            y_test (np.ndarray or pd.Series): Test target set.

        Returns:
            Dict[str, float]: Dictionary containing regression metrics.
        """
        try:
            y_pred = model.predict(X_test).flatten()
            metrics = self.evaluate_regression(y_test, y_pred)
            return metrics
        except Exception as e:
            self.logger.error(f"Error evaluating LSTM model: {e}")
            raise EvaluationError("Failed to evaluate LSTM model.") from e

    def evaluate_arima_model(
        self,
        rmse: float,
        retrain_threshold: float
    ) -> bool:
        """
        Evaluate ARIMA model performance and determine if retraining is needed.

        Args:
            rmse (float): Root mean squared error.
            retrain_threshold (float): RMSE threshold for retraining.

        Returns:
            bool: True if retraining is required, otherwise False.
        """
        if rmse > retrain_threshold:
            self.logger.warning(f"RMSE {rmse} exceeds threshold of {retrain_threshold}. Retraining triggered.")
            return True  # Indicates retraining is required
        else:
            self.logger.info(f"RMSE {rmse} is within acceptable limits.")
            return False

    def validate_new_model_performance(
        self,
        new_model: Any,
        previous_model: Optional[Any],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        model_type: str = 'regression',
        threshold: Optional[float] = None
    ) -> bool:
        """
        Validate new model performance against a previous model.

        Args:
            new_model (Any): Newly trained model.
            previous_model (Any, optional): Previously trained model.
            X_val (np.ndarray or pd.DataFrame): Validation features.
            y_val (np.ndarray or pd.Series): Validation targets.
            model_type (str): Type of model ('regression' or 'classification').
            threshold (float, optional): Improvement threshold.

        Returns:
            bool: True if new model performance is acceptable, False otherwise.
        """
        try:
            y_pred_new = new_model.predict(X_val)
            if model_type == 'regression':
                metrics_new = self.evaluate_regression(y_val, y_pred_new)
                threshold = threshold or self.config.regression_threshold
            elif model_type == 'classification':
                metrics_new = self.evaluate_classification(y_val, y_pred_new)
                threshold = threshold or self.config.classification_threshold
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            if previous_model is None:
                self.logger.info("No previous model to compare with; accepting new model.")
                return True

            y_pred_prev = previous_model.predict(X_val)
            if model_type == 'regression':
                metrics_prev = self.evaluate_regression(y_val, y_pred_prev)
                improvement = (metrics_prev['mse'] - metrics_new['mse']) / metrics_prev['mse']
            else:
                metrics_prev = self.evaluate_classification(y_val, y_pred_prev)
                improvement = metrics_new['accuracy'] - metrics_prev['accuracy']

            self.logger.info(f"Model improvement: {improvement:.4f}")

            if improvement >= threshold:
                self.logger.info("New model performance meets the threshold; accepting new model.")
                return True
            else:
                self.logger.warning("New model performance does not meet the threshold; rejecting new model.")
                return False

        except Exception as e:
            self.logger.error(f"Error validating model performance: {e}")
            raise EvaluationError("Failed to validate model performance.") from e

    def evaluate_model(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        model_type: str = 'regression'
    ) -> Dict[str, Any]:
        """
        Evaluate a model based on its type.

        Args:
            y_true (np.ndarray or pd.Series): True target values.
            y_pred (np.ndarray or pd.Series): Predicted target values.
            model_type (str): Type of model ('regression', 'classification').

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics.
        """
        if model_type == 'regression':
            return self.evaluate_regression(y_true, y_pred)
        elif model_type == 'classification':
            return self.evaluate_classification(y_true, y_pred)
        else:
            self.logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

    def evaluate_trades(
        self,
        trades_df: pd.DataFrame,
        equity_curve: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate trades and calculate trading metrics.

        Args:
            trades_df (pd.DataFrame): DataFrame containing trade details.
            equity_curve (pd.Series): Series representing the equity curve over time.

        Returns:
            Dict[str, Any]: Dictionary containing trade metrics.
        """
        evaluator = TradingModelEvaluator(
            trades_df=trades_df,
            equity_curve=equity_curve,
            risk_free_rate=self.config.risk_free_rate,
            logger=self.logger
        )
        metrics = evaluator.evaluate_metrics()
        self.logger.info("Trade Evaluation Complete.")
        return metrics.to_dict()

@dataclass
class TradeMetrics:
    total_net_profit: float
    profit_factor: float
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    max_drawdown: Optional[float]
    win_ratio: float
    avg_win_loss_ratio: float
    risk_reward_ratio: float
    ulcer_index: Optional[float]
    car_mdd_ratio: Optional[float]
    rar_mdd_ratio: Optional[float]
    calmar_ratio: Optional[float]
    information_ratio: Optional[float] = None
    omega_ratio: Optional[float] = None

    def to_dict(self):
        """Convert TradeMetrics to a dictionary for serialization."""
        return self.__dict__

class TradingModelEvaluator:
    def __init__(
        self,
        trades_df: pd.DataFrame,
        equity_curve: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize evaluator with trade data and equity curve.
        Supports multi-asset portfolios and calculates advanced metrics.
        """
        self.trades_df = trades_df.copy()
        self.equity_curve = equity_curve.copy() if equity_curve is not None else None
        self.benchmark_returns = benchmark_returns.copy() if benchmark_returns is not None else None
        self.risk_free_rate = risk_free_rate
        self.logger = logger or get_logger()

        # Set up daily returns if equity curve exists
        if self.equity_curve is not None:
            self.daily_returns = self.equity_curve.pct_change().dropna()
            self.logger.info("Initialized TradingModelEvaluator with equity curve.")
        else:
            self.daily_returns = None
            self.logger.warning("Equity curve not provided. Limited metrics available.")

    def calculate_returns(self):
        """Calculate returns and profit/loss for each trade."""
        required_cols = {'exit_price', 'entry_price', 'position_size'}
        if missing := required_cols - set(self.trades_df.columns):
            self.logger.error(f"Missing columns in trade data: {missing}. Cannot calculate returns.")
            raise KeyError(f"Trade DataFrame must contain {required_cols} columns.")

        self.trades_df['return'] = (self.trades_df['exit_price'] - self.trades_df['entry_price']) / self.trades_df['entry_price']
        self.trades_df['profit_loss'] = self.trades_df['return'] * self.trades_df['position_size']
        self.logger.info("Calculated trade returns and profit/loss.")

    def evaluate_metrics(self) -> TradeMetrics:
        """Evaluate a comprehensive set of trading metrics."""
        self.calculate_returns()
        metrics = TradeMetrics(
            total_net_profit=self.total_net_profit(),
            profit_factor=self.profit_factor(),
            sharpe_ratio=self.sharpe_ratio() if self.equity_curve is not None else None,
            sortino_ratio=self.sortino_ratio() if self.equity_curve is not None else None,
            max_drawdown=self.max_drawdown() if self.equity_curve is not None else None,
            win_ratio=self.win_ratio(),
            avg_win_loss_ratio=self.average_win_loss_ratio(),
            risk_reward_ratio=self.risk_reward_ratio(),
            ulcer_index=self.ulcer_index() if self.equity_curve is not None else None,
            car_mdd_ratio=self.car_mdd_ratio() if self.equity_curve is not None else None,
            rar_mdd_ratio=self.rar_mdd_ratio() if self.equity_curve is not None else None,
            calmar_ratio=self.calmar_ratio() if self.equity_curve is not None else None,
            information_ratio=self.information_ratio() if self.benchmark_returns is not None else None,
            omega_ratio=self.omega_ratio() if self.equity_curve is not None else None
        )
        self.logger.info("Evaluated trade metrics.")
        return metrics

    # Metrics Calculation Methods

    def total_net_profit(self) -> float:
        """Calculate total net profit for the strategy."""
        total_profit = self.trades_df['profit_loss'].sum()
        self.logger.debug(f"Total net profit: {total_profit}")
        return total_profit

    def profit_factor(self) -> float:
        """Calculate profit factor."""
        gross_profit = self.trades_df[self.trades_df['profit_loss'] > 0]['profit_loss'].sum()
        gross_loss = self.trades_df[self.trades_df['profit_loss'] < 0]['profit_loss'].abs().sum()
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
        self.logger.debug(f"Profit factor: {profit_factor}")
        return profit_factor

    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = self.daily_returns - (self.risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        self.logger.debug(f"Sharpe ratio: {sharpe_ratio}")
        return sharpe_ratio

    def sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        excess_returns = self.daily_returns - (self.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        self.logger.debug(f"Sortino ratio: {sortino_ratio}")
        return sortino_ratio

    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        peak = self.equity_curve.cummax()
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        self.logger.debug(f"Maximum drawdown: {max_drawdown}")
        return max_drawdown

    def win_ratio(self) -> float:
        """Calculate win ratio."""
        wins = (self.trades_df['profit_loss'] > 0).sum()
        total_trades = len(self.trades_df)
        win_ratio = wins / total_trades if total_trades > 0 else 0.0
        self.logger.debug(f"Win ratio: {win_ratio}")
        return win_ratio

    def average_win_loss_ratio(self) -> float:
        """Calculate average win/loss ratio."""
        avg_win = self.trades_df[self.trades_df['profit_loss'] > 0]['profit_loss'].mean()
        avg_loss = self.trades_df[self.trades_df['profit_loss'] < 0]['profit_loss'].abs().mean()
        win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else np.inf
        self.logger.debug(f"Average win/loss ratio: {win_loss_ratio}")
        return win_loss_ratio

    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio based on win/loss ratio."""
        risk_reward_ratio = self.average_win_loss_ratio()
        self.logger.debug(f"Risk/Reward ratio: {risk_reward_ratio}")
        return risk_reward_ratio

    def ulcer_index(self) -> float:
        """Calculate Ulcer Index."""
        drawdown = 1 - self.equity_curve / self.equity_curve.cummax()
        ulcer_index = np.sqrt((drawdown**2).mean())
        self.logger.debug(f"Ulcer index: {ulcer_index}")
        return ulcer_index

    def rar_mdd_ratio(self) -> float:
        """Calculate RAR/MDD ratio (Risk-Adjusted Return to Maximum Drawdown)."""
        mdd = abs(self.max_drawdown())
        sharpe = self.sharpe_ratio()
        rar_mdd_ratio = sharpe / mdd if mdd != 0 else np.inf
        self.logger.debug(f"RAR/MDD ratio: {rar_mdd_ratio}")
        return rar_mdd_ratio

    def car_mdd_ratio(self) -> Optional[float]:
        """Calculate CAR/MDD ratio."""
        try:
            if self.equity_curve is None or self.equity_curve.empty:
                return None

            if not isinstance(self.equity_curve.index, pd.DatetimeIndex):
                self.equity_curve.index = pd.to_datetime(self.equity_curve.index, errors='coerce')
                self.equity_curve = self.equity_curve.dropna()
                if self.equity_curve.empty or self.equity_curve.index.isnull().all():
                    return None

            time_span = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
            end_value = self.equity_curve.iloc[-1]
            start_value = self.equity_curve.iloc[0]

            car = ((end_value / start_value) ** (1 / time_span)) - 1 if start_value != 0 else None

            mdd = self.max_drawdown()
            return car / mdd if car is not None and mdd != 0 else None

        except Exception as e:
            self.logger.error(f"Error calculating CAR/MDD ratio: {e}")
            return None

    def calmar_ratio(self) -> Optional[float]:
        """Calculate the Calmar Ratio."""
        calmar_ratio = self.car_mdd_ratio()
        self.logger.debug(f"Calmar ratio: {calmar_ratio}")
        return calmar_ratio

    def omega_ratio(self, threshold: float = 0) -> Optional[float]:
        """Calculate the Omega Ratio."""
        if self.daily_returns is None:
            self.logger.warning("Daily returns not available; cannot calculate Omega ratio.")
            return None

        return_threshold = self.daily_returns - threshold
        positive_returns = return_threshold[return_threshold > 0].sum()
        negative_returns = abs(return_threshold[return_threshold < 0].sum())

        omega_ratio = positive_returns / negative_returns if negative_returns != 0 else np.inf
        self.logger.debug(f"Omega ratio: {omega_ratio}")
        return omega_ratio

# Example usage
if __name__ == "__main__":
    # Initialize logger
    logger = get_logger('evaluation_log', log_file='evaluation.log', level=logging.DEBUG)
    config = EvaluationConfig(
        regression_threshold=0.05,
        classification_threshold=0.01,
        risk_free_rate=0.03,
        logger=logger
    )
    evaluator = ModelEvaluator(config=config)

    # Regression Example
    y_true_reg = np.array([3.0, 4.5, 6.7, 8.0])
    y_pred_reg = np.array([2.9, 4.6, 6.5, 8.1])
    regression_metrics = evaluator.evaluate_regression(y_true_reg, y_pred_reg)
    print("Regression Metrics:", regression_metrics)

    # Classification Example
    y_true_clf = np.array([0, 1, 1, 0])
    y_pred_clf = np.array([0, 1, 0, 0])
    classification_metrics = evaluator.evaluate_classification(y_true_clf, y_pred_clf)
    print("Classification Metrics:", classification_metrics)

    # Trading Metrics Example
    predictions = np.array([100, 102, 101, 103, 104])
    y_true_trading = np.array([100, 101, 102, 103, 104])
    trading_metrics = evaluator.evaluate_trading_metrics(predictions, y_true_trading)
    print("Trading Metrics:", trading_metrics)

    # Trade Evaluation Example
    trades_data = {
        'entry_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
        'exit_date': pd.to_datetime(['2023-01-10', '2023-02-10', '2023-03-10']),
        'entry_price': [150, 600, 3200],
        'exit_price': [160, 610, 3150],
        'position_size': [100, 50, 10]
    }
    trades_df = pd.DataFrame(trades_data)
    equity_curve = pd.Series(
        [100000, 102000, 101000, 103000],
        index=pd.date_range('2023-01-01', periods=4)
    )
    trade_metrics = evaluator.evaluate_trades(trades_df, equity_curve)
    print("Trade Metrics:", trade_metrics)

    # Model Validation Example
    class DummyModel:
        def predict(self, X):
            return X * 0.9  # Dummy prediction

    new_model = DummyModel()
    old_model = DummyModel()
    X_val = np.array([10, 20, 30, 40])
    y_val = np.array([12, 22, 32, 42])
    validation_passed = evaluator.validate_new_model_performance(
        new_model=new_model,
        previous_model=old_model,
        X_val=X_val,
        y_val=y_val,
        model_type='regression'
    )
    print("Validation Passed:", validation_passed)
