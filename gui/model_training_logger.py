"""
gui/model_training_logger.py
----------------------------
Defines ModelTrainingLogger to log messages both to the console and to a Tkinter Text widget.
"""

import logging
from datetime import datetime

class ModelTrainingLogger:
    def __init__(self, log_text_widget):
        self.log_text_widget = log_text_widget
        
        # Configure the Python logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)
    
    def log(self, message):
        """
        Logs a message to both the console and the associated Tkinter text widget.
        
        Args:
            message (str): The message to log.
        """
        self.logger.info(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {message}\n"
        
        # Update the GUI text widget
        self.log_text_widget.config(state="normal")
        self.log_text_widget.insert("end", log_entry)
        self.log_text_widget.config(state="disabled")
        self.log_text_widget.see("end")
