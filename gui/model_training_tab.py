"""
gui/model_training_tab.py
-------------------------
Defines the ModelTrainingTab class which builds the GUI,
handles user interactions, and orchestrates training (here via dummy code).
"""

import tkinter as tk
from tkinter import ttk, filedialog
import os
import queue
import threading
from datetime import datetime
import time

import pandas as pd

from gui.model_training_logger import ModelTrainingLogger
from gui.notifications import send_notification

class ModelTrainingTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.queue = queue.Queue()
        self.is_debug_mode = False
        self.training_history = []
        
        # Set up the user interface
        self.setup_ui()
        
        # Begin periodic processing of queued log messages
        self.after(100, self.process_queue)
    
    def setup_ui(self):
        # Title label
        self.title_label = tk.Label(self, text="Model Training", font=("Helvetica", 16))
        self.title_label.pack(pady=10)
        
        # Data file selection
        self.data_file_label = tk.Label(self, text="Data File Path:")
        self.data_file_label.pack()
        self.data_file_entry = tk.Entry(self, width=50)
        self.data_file_entry.pack(pady=5)
        self.browse_button = ttk.Button(self, text="Browse", command=self.browse_data_file)
        self.browse_button.pack(pady=5)
        
        # Model type dropdown
        self.model_type_label = tk.Label(self, text="Select Model Type:")
        self.model_type_label.pack()
        self.model_type_var = tk.StringVar()
        self.model_type_dropdown = ttk.Combobox(
            self, textvariable=self.model_type_var,
            values=["linear_regression", "random_forest", "neural_network", "LSTM", "ARIMA"]
        )
        self.model_type_dropdown.pack(pady=5)
        
        # Start training button
        self.start_training_button = ttk.Button(self, text="Start Training", command=self.start_training)
        self.start_training_button.pack(pady=10)
        
        # Log text widget to display messages
        self.log_text = tk.Text(self, height=10, state="disabled")
        self.log_text.pack(pady=5, fill="both", expand=True)
        
        # Instantiate our logger with the log_text widget
        self.logger = ModelTrainingLogger(self.log_text)
    
    def browse_data_file(self):
        """Opens a file dialog for the user to select a CSV file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            self.data_file_entry.delete(0, tk.END)
            self.data_file_entry.insert(0, file_path)
            self.display_message(f"Selected file: {file_path}")
    
    def start_training(self):
        """Validates inputs and starts the training process in a background thread."""
        data_file_path = self.data_file_entry.get()
        model_type = self.model_type_var.get()
        
        if not data_file_path or not os.path.exists(data_file_path):
            self.display_message("Invalid data file path.", level="ERROR")
            return
        if not model_type:
            self.display_message("Please select a model type.", level="ERROR")
            return
        
        self.disable_training_button()
        self.display_message("Training started...", level="INFO")
        
        # Here you would call your actual training function.
        # For demonstration, we simulate training with a dummy process.
        threading.Thread(target=self.dummy_train_model, daemon=True).start()
    
    def dummy_train_model(self):
        """Simulates a training process by putting messages on the queue."""
        for epoch in range(1, 6):
            time.sleep(1)  # Simulate work (each epoch takes 1 second)
            self.queue.put(f"Epoch {epoch}/5 completed.")
        self.queue.put("Training completed successfully.")
        self.enable_training_button()
    
    def process_queue(self):
        """Processes and displays messages queued by background tasks."""
        try:
            while not self.queue.empty():
                message = self.queue.get_nowait()
                self.logger.log(message)
        except Exception as e:
            self.logger.log(f"Error processing queue: {str(e)}")
        finally:
            self.after(100, self.process_queue)
    
    def display_message(self, message, level="INFO"):
        """
        Formats and displays a message in the log text widget.
        
        Args:
            message (str): The message to display.
            level (str): The log level (e.g., "INFO", "ERROR").
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp} - {level}] {message}\n"
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)
    
    def disable_training_button(self):
        """Disables the training button to prevent duplicate clicks."""
        self.start_training_button.config(state="disabled")
    
    def enable_training_button(self):
        """Enables the training button after training completes."""
        self.start_training_button.config(state="normal")
