"""
utils/ml_robot_utils.py
-----------------------
Contains utility functions for logging, file path generation, and configuration management.
"""

import logging
import os
from datetime import datetime

class MLRobotUtils:
    """
    Utility class for common operations such as logging messages and generating save paths.
    """
    def log_message(self, message, log_widget=None, is_debug=False, level="INFO"):
        """
        Logs a message to the console and optionally to a provided Tkinter widget.
        
        Args:
            message (str): Message to log.
            log_widget (tkinter.Text, optional): Text widget for GUI logging.
            is_debug (bool): If True, logs at DEBUG level.
            level (str): Log level.
        """
        if level.upper() == "DEBUG":
            logging.debug(message)
        elif level.upper() == "ERROR":
            logging.error(message)
        else:
            logging.info(message)
        
        if log_widget is not None:
            from tkinter import END
            log_widget.config(state="normal")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_widget.insert(END, f"[{timestamp} - {level}] {message}\n")
            log_widget.config(state="disabled")
    
    def auto_generate_save_path(self, model_type, base_dir="saved_models"):
        """
        Generates a file path for saving a model.
        
        Args:
            model_type (str): The type of model.
            base_dir (str): Base directory for saving.
        
        Returns:
            str: A file path string.
        """
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_{timestamp}.joblib"
        return os.path.join(base_dir, filename)
