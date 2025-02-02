"""
evaluation/visualization.py
---------------------------
Provides functions to visualize evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(y_true=None, y_pred=None, conf_matrix=None, 
                          class_names=None, save_path="confusion_matrix.png", 
                          show_plot=True):
    """
    Plots and saves a confusion matrix. Either a precomputed confusion matrix or
    y_true and y_pred must be provided.
    
    Args:
        y_true (array-like, optional): True labels.
        y_pred (array-like, optional): Predicted labels.
        conf_matrix (array-like, optional): Precomputed confusion matrix.
        class_names (list, optional): Labels for axes.
        save_path (str): File path for saving the plot.
        show_plot (bool): Whether to display the plot.
    """
    from sklearn.metrics import confusion_matrix
    if conf_matrix is None:
        if y_true is None or y_pred is None:
            raise ValueError("Provide either conf_matrix or both y_true and y_pred.")
        conf_matrix = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = list(range(conf_matrix.shape[0]))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close()

def visualize_training_results(y_test, y_pred):
    """
    Creates a scatter plot comparing actual and predicted values.
    
    Args:
        y_test (array-like): True values.
        y_pred (array-like): Predicted values.
    """
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.5})
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_model_performance(y_true, y_pred):
    """
    Creates scatter and residual plots to evaluate model performance.
    
    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
    """
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Model Performance: Actual vs. Predicted")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.tight_layout()
    plt.show()
    
    # Residual plot
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.show()

def generate_regression_visualizations(y_test, y_pred):
    """
    Generates both training results and performance visualizations.
    
    Args:
        y_test (array-like): True values.
        y_pred (array-like): Predicted values.
    """
    visualize_training_results(y_test, y_pred)
    visualize_model_performance(y_test, y_pred)
