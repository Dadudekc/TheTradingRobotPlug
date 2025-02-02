"""
evaluation/reports.py
---------------------
Provides functions for generating text-based performance reports.
"""

import logging
from sklearn.metrics import classification_report

def generate_model_reports(model, X_test, y_test, y_pred):
    """
    Generates a comprehensive performance report for the model.
    
    Args:
        model: Trained model.
        X_test (array-like): Test features.
        y_test (array-like): True values.
        y_pred (array-like): Predicted values.
    
    Returns:
        str: A formatted report.
    """
    try:
        report = ""
        # For classification models
        if hasattr(model, "predict_proba") or (hasattr(model, "predict") and isinstance(y_test[0], (int, str))):
            report += "Classification Report:\n"
            report += classification_report(y_test, y_pred)
        else:
            # For regression, use evaluation metrics
            from evaluation.metrics import calculate_model_metrics
            metrics = calculate_model_metrics(y_test, y_pred)
            report += "Regression Metrics:\n"
            for key, value in metrics.items():
                report += f"{key}: {value:.4f}\n"
        logging.info("Model report generated successfully.")
        return report
    except Exception as e:
        logging.error("Error generating model report: " + str(e))
        raise

def generate_custom_model_report(model, X_test, y_test, y_pred):
    """
    Generates a custom report including model parameters and performance.
    
    Args:
        model: Trained model.
        X_test (array-like): Test features.
        y_test (array-like): True values.
        y_pred (array-like): Predicted values.
    
    Returns:
        str: A detailed custom report.
    """
    try:
        report = "Custom Model Report:\n"
        report += "Model Parameters:\n"
        if hasattr(model, 'get_params'):
            for key, value in model.get_params().items():
                report += f"{key}: {value}\n"
        else:
            report += "No parameter information available.\n"
        report += "\nPerformance Metrics:\n"
        from evaluation.metrics import calculate_model_metrics
        metrics = calculate_model_metrics(y_test, y_pred)
        for key, value in metrics.items():
            report += f"{key}: {value:.4f}\n"
        return report
    except Exception as e:
        logging.error("Error generating custom model report: " + str(e))
        raise
