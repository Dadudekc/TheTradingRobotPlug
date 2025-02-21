"""
ARIMA Model Trainer with GUI and Data Fetching Integration
----------------------------------------------------------

This script provides a GUI for configuring and running an ARIMA model trainer.
It integrates with `DataFetchUtils` to fetch data from various APIs, prioritizes
API data fetching while allowing optional CSV usage, and includes functionality
for stopping the training process gracefully.

Features:
- GUI for model configuration and training initiation.
- Data fetching from multiple sources via `DataFetchUtils`.
- Optional CSV data loading if API data fetching fails.
- ARIMA model training with parallel processing.
- Real-time status updates and logging.
- Graceful stopping of the training process.
- Integrated `ModelManager` for advanced model handling.
- Modular code with clear sections for ease of debugging.

Author: Victor Dixon
Date: 11/10/2024
"""

# -------------------------------------------------------------------
# Section 1: Imports and Logging Setup
# -------------------------------------------------------------------
import os
import sys
import threading
import asyncio
import random
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time
from PyQt5 import QtWidgets, QtCore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------------------------------------------------
# Section 2: Project Path Setup
# -------------------------------------------------------------------
# Dynamically set the project root based on the current file's location
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]
print("PROJECT_ROOT(arima_model_trainer.py):", project_root)
# Define all relevant directories based on the new structure
directories = {
    'config': project_root / 'config',
    'data': project_root / 'data',
    'database': project_root / 'database',
    'deployment': project_root / 'deployment',
    'docs': project_root / 'docs',
    'reports': project_root / 'reports',
    'scripts': project_root / 'Scripts',
    'model_training': project_root / 'Scripts' / 'model_training',
    'data_fetching': project_root / 'Scripts' / 'data_fetching',
    'data_processing': project_root / 'Scripts' / 'data_processing',
    'backtesting': project_root / 'Scripts' / 'backtesting',
    'trading_robot': project_root / 'Scripts' / 'trading_robot',
    'scheduler': project_root / 'Scripts' / 'scheduler',
    'utilities': project_root / 'Scripts' / 'Utilities',
    'utilities_ai': project_root / 'Scripts' / 'Utilities' / 'ai',
    'utilities_analysis': project_root / 'Scripts' / 'Utilities' / 'analysis',
    'utilities_config_handling': project_root / 'Scripts' / 'Utilities' / 'config_handling',
    'utilities_gui': project_root / 'Scripts' / 'Utilities' / 'gui',
    'utilities_utils': project_root / 'Scripts' / 'Utilities' / 'utils',
    'utilities_tests': project_root / 'Scripts' / 'Utilities' / 'tests',
    'side_projects': project_root / 'side_projects',
    'logs': project_root / 'logs',
    'results': project_root / 'results',
    'SavedModels': project_root / 'SavedModels',
}

# Add all directories to sys.path for module imports
for name, path in directories.items():
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    sys.path.append(str(path.resolve()))

# Shortcuts for frequently accessed paths
config_dir = directories['config']
logs_dir = directories['logs']
results_dir = directories['results']
SavedModels_dir = directories['SavedModels']
utilities_gui = directories['utilities_gui'] / 'jsonfiles'  # Path to gui_config.json

# -------------------------------------------------------------------
# Section 3: Importing Utilities
# -------------------------------------------------------------------
try:
    from Utilities.data.data_fetch_utils import DataFetchUtils
    from Utilities.data.data_store import DataStore
    from Utilities.config_manager import ConfigManager, setup_logging
    from utils.training.model_manager import ModelManager  # Adjusted import path
    print("Modules imported successfully!")
except ImportError as e:
    logging.error(f"Error importing modules: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Section 4: GUI Utilities Class Definition
# -------------------------------------------------------------------
class GUIUtils(QtWidgets.QWidget):
    # Define custom signals
    params_collected = QtCore.pyqtSignal(dict)
    stop_training_signal = QtCore.pyqtSignal()
    update_button_state_signal = QtCore.pyqtSignal(bool)

    def __init__(self, config_dir=None):
        super().__init__()
        self.gui_params = {}
        if config_dir is None:
            config_dir = utilities_gui  # Path to gui_config.json
        self.model_configurations, self.general_settings = self.load_configurations(config_dir)
        self.init_ui()

    def load_configurations(self, config_dir, debug=False):
        """
        Load model configurations and general settings from a JSON file in the specified directory.
        Supports both model-specific fields and general settings.
        """
        config_file_path = config_dir / "gui_config.json"

        if not config_file_path.exists():
            logging.error(f"Configuration file not found: {config_file_path}")
            QtWidgets.QMessageBox.critical(self, "Configuration Error", f"Configuration file not found: {config_file_path}")
            return {}, {}

        try:
            with open(config_file_path, 'r') as file:
                config = json.load(file)

            if not isinstance(config, dict):
                logging.error("Invalid JSON configuration format: Root element should be a dictionary.")
                QtWidgets.QMessageBox.critical(self, "Configuration Error", "Invalid JSON configuration format.")
                return {}, {}

            model_configurations, general_settings = self.parse_configurations(config)

            if debug:
                print(f"Loaded configuration from {config_file_path}")

            return model_configurations, general_settings

        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Error loading configuration: {e}")
            QtWidgets.QMessageBox.critical(self, "Configuration Error", f"Error loading configuration: {e}")
            return {}, {}

    def parse_configurations(self, config):
        """
        Helper function to parse configurations into model-specific and general settings.
        """
        model_configurations = {}
        general_settings = {}

        for key, settings in config.items():
            if key == "General_Settings":
                general_settings.update(settings)
            elif isinstance(settings, dict) and 'fields' in settings:
                model_configurations[key] = settings
            else:
                general_settings[key] = settings

        return model_configurations, general_settings

    def init_ui(self):
        """
        Initialize the main UI layout.
        """
        self.setWindowTitle("ARIMA Model Training Setup")
        self.setGeometry(100, 100, 800, 1000)  # Adjusted size for more widgets

        layout = QtWidgets.QVBoxLayout()

        # Model Type dropdown
        model_type_label = QtWidgets.QLabel("Model Type:")
        self.model_type_combo = QtWidgets.QComboBox(self)

        # Ensure model_configurations is a dictionary before adding items
        if isinstance(self.model_configurations, dict) and self.model_configurations:
            self.model_type_combo.addItems(self.model_configurations.keys())
        else:
            self.model_type_combo.addItem("ARIMA_Model")  # Default model if none loaded

        layout.addWidget(model_type_label)
        layout.addWidget(self.model_type_combo)

        # Field layout (dynamic based on model selection)
        self.field_layout = QtWidgets.QFormLayout()
        self.update_fields()
        self.model_type_combo.currentTextChanged.connect(self.update_fields)
        layout.addLayout(self.field_layout)

        # Log File input
        log_file_label = QtWidgets.QLabel("Log File:")
        self.log_file_input = QtWidgets.QLineEdit(self)
        self.log_file_input.setText(str(logs_dir / 'training.log'))
        layout.addWidget(log_file_label)
        layout.addWidget(self.log_file_input)

        # Log Level input
        log_level_label = QtWidgets.QLabel("Log Level:")
        self.log_level_input = QtWidgets.QLineEdit(self)
        self.log_level_input.setText("DEBUG")
        layout.addWidget(log_level_label)
        layout.addWidget(self.log_level_input)

        # Mode selection
        mode_label = QtWidgets.QLabel("Select Mode:")
        self.mode_combo = QtWidgets.QComboBox(self)
        self.mode_combo.addItems(["Aftermarket Mode", "Overnight Mode"])
        layout.addWidget(mode_label)
        layout.addWidget(self.mode_combo)

        # Data Source selection
        data_source_label = QtWidgets.QLabel("Select Data Source:")
        self.data_source_combo = QtWidgets.QComboBox(self)
        self.data_source_combo.addItems(["Yahoo Finance", "Alpaca", "Alpha Vantage", "Polygon", "Finnhub", "NewsAPI", "CSV"])
        self.data_source_combo.currentTextChanged.connect(self.toggle_csv_input)
        layout.addWidget(data_source_label)
        layout.addWidget(self.data_source_combo)

        # CSV File input (only visible if "CSV" is selected as the data source)
        self.csv_file_label = QtWidgets.QLabel("CSV File Path:")
        self.csv_file_input = QtWidgets.QLineEdit(self)
        self.csv_file_button = QtWidgets.QPushButton("Select CSV")
        self.csv_file_button.clicked.connect(self.select_csv)
        self.csv_file_label.setVisible(False)  # Hidden by default
        self.csv_file_input.setVisible(False)
        self.csv_file_button.setVisible(False)
        layout.addWidget(self.csv_file_label)
        layout.addWidget(self.csv_file_input)
        layout.addWidget(self.csv_file_button)

        # Start Date input
        start_date_label = QtWidgets.QLabel("Start Date (YYYY-MM-DD):")
        self.start_date_input = QtWidgets.QLineEdit(self)
        self.start_date_input.setText("2022-01-01")
        layout.addWidget(start_date_label)
        layout.addWidget(self.start_date_input)

        # End Date input
        end_date_label = QtWidgets.QLabel("End Date (YYYY-MM-DD) or leave blank for today:")
        self.end_date_input = QtWidgets.QLineEdit(self)
        self.end_date_input.setText("")
        layout.addWidget(end_date_label)
        layout.addWidget(self.end_date_input)

        # Interval input
        interval_label = QtWidgets.QLabel("Data Interval (e.g., '1d'):")
        self.interval_input = QtWidgets.QLineEdit(self)
        self.interval_input.setText("1d")
        layout.addWidget(interval_label)
        layout.addWidget(self.interval_input)

        # Symbol input
        symbol_label = QtWidgets.QLabel("Stock Symbol (e.g., 'AAPL'):")
        self.symbol_input = QtWidgets.QLineEdit(self)
        self.symbol_input.setText("AAPL")
        layout.addWidget(symbol_label)
        layout.addWidget(self.symbol_input)

        # Status messages
        status_label = QtWidgets.QLabel("Status:")
        self.status_text = QtWidgets.QTextEdit(self)
        self.status_text.setReadOnly(True)
        layout.addWidget(status_label)
        layout.addWidget(self.status_text)

        # Submit and Stop buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.submit_button = QtWidgets.QPushButton("Start Training")
        self.submit_button.clicked.connect(self.submit_inputs)
        button_layout.addWidget(self.submit_button)

        self.stop_button = QtWidgets.QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect the update_button_state_signal to the slot
        self.update_button_state_signal.connect(self.update_button_state)

    def update_fields(self):
        """
        Update dynamic input fields based on the selected model type.
        """
        # Clear existing fields
        for i in reversed(range(self.field_layout.count())):
            widget = self.field_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        selected_model = self.model_type_combo.currentText()

        if selected_model and selected_model in self.model_configurations:
            model_fields = self.model_configurations[selected_model].get('fields', {})

            self.field_vars = {}

            for field_name, field_info in model_fields.items():
                label = QtWidgets.QLabel(field_info.get('label', field_name))
                if field_info.get('type') == 'file':
                    button = QtWidgets.QPushButton("Select File")
                    button.clicked.connect(partial(self.select_csv_field, field_name))
                    self.field_layout.addRow(label, button)
                    # Store the path in a QLineEdit
                    self.field_vars[field_name] = QtWidgets.QLineEdit(self)
                    self.field_layout.addRow("", self.field_vars[field_name])
                else:
                    input_field = QtWidgets.QLineEdit(self)
                    input_field.setText(str(field_info.get('default', '')))
                    self.field_layout.addRow(label, input_field)
                    self.field_vars[field_name] = input_field

    def toggle_csv_input(self):
        """
        Show or hide the CSV input fields based on data source selection.
        """
        if self.data_source_combo.currentText() == "CSV":
            # Show CSV input if CSV is selected as the data source
            self.csv_file_label.setVisible(True)
            self.csv_file_input.setVisible(True)
            self.csv_file_button.setVisible(True)
        else:
            # Hide CSV input for non-CSV sources
            self.csv_file_label.setVisible(False)
            self.csv_file_input.setVisible(False)
            self.csv_file_button.setVisible(False)
            # Clear any path if switching from CSV to another source
            self.csv_file_input.clear()

    def select_csv(self):
        """
        Open file dialog to select a CSV file.
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if filename:
            self.csv_file_input.setText(filename)

    def select_csv_field(self, field_name):
        """
        Open file dialog to select a CSV file for a specific field.
        """
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*.*)")
        if filename and field_name in self.field_vars:
            self.field_vars[field_name].setText(filename)

    def submit_inputs(self):
        """
        Collect and validate inputs, then emit them via a signal.
        """
        try:
            selected_model = self.model_type_combo.currentText()
            model_fields = self.model_configurations[selected_model].get('fields', {})

            for field_name, field_info in model_fields.items():
                if field_name not in self.field_vars:
                    QtWidgets.QMessageBox.warning(self, "Invalid Input", f"Field {field_name} is missing.")
                    return

                value = self.field_vars[field_name].text().strip()

                if not value:
                    QtWidgets.QMessageBox.warning(self, "Invalid Input", f"{field_info.get('label', field_name)} cannot be empty.")
                    return

                if field_info.get('type') == 'int':
                    try:
                        int_value = int(value)
                        if int_value <= 0:
                            raise ValueError(f"{field_info.get('label', field_name)} must be greater than 0.")
                        self.gui_params[field_name] = int_value
                    except ValueError as ve:
                        QtWidgets.QMessageBox.warning(self, "Invalid Input", f"Invalid input for {field_info.get('label', field_name)}: {ve}")
                        return
                elif field_info.get('type') == 'float':
                    try:
                        float_value = float(value)
                        self.gui_params[field_name] = float_value
                    except ValueError:
                        QtWidgets.QMessageBox.warning(self, "Invalid Input", f"{field_info.get('label', field_name)} must be a float.")
                        return
                else:
                    self.gui_params[field_name] = value

            # Log and mode settings
            self.gui_params['log_file'] = self.log_file_input.text()
            self.gui_params['log_level'] = self.log_level_input.text()
            self.gui_params['mode'] = self.mode_combo.currentText()
            self.gui_params['data_source'] = self.data_source_combo.currentText()
            self.gui_params['start_date'] = self.start_date_input.text()
            self.gui_params['end_date'] = self.end_date_input.text() or None
            self.gui_params['interval'] = self.interval_input.text()
            self.gui_params['symbol'] = self.symbol_input.text()

            # Include CSV path if the CSV data source is selected
            if self.data_source_combo.currentText() == "CSV":
                csv_path = self.csv_file_input.text().strip()
                if not csv_path:
                    QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please select a CSV file.")
                    return
                self.gui_params['csv_path'] = csv_path

            # Emit the collected parameters
            self.params_collected.emit(self.gui_params)

            QtWidgets.QMessageBox.information(self, "Success", "Training started!")
            # Disable the submit button and enable the stop button
            self.submit_button.setEnabled(False)
            self.stop_button.setEnabled(True)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    def stop_training(self):
        """Emit a signal to stop the training."""
        self.stop_training_signal.emit()
        self.log_status("Training stop requested.")
        # Re-enable the submit button and disable the stop button
        self.submit_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def log_status(self, message):
        """Append a status message to the status_text widget."""
        self.status_text.append(message)

    @QtCore.pyqtSlot(bool)
    def update_button_state(self, is_enabled):
        """Enable or disable the submit and stop buttons."""
        self.submit_button.setEnabled(is_enabled)
        self.stop_button.setEnabled(not is_enabled)

# -------------------------------------------------------------------
# Section 5: ARIMA Model Trainer Class Definition
# -------------------------------------------------------------------

class ARIMAModelTrainer(QtCore.QObject):
    # Define signals
    status_update = QtCore.pyqtSignal(str)
    button_state_update = QtCore.pyqtSignal(bool)

    def __init__(self, params, logger, data_store, data_fetcher, parent=None):
        super().__init__(parent)
        """
        Initialize ARIMA Model Trainer.

        Args:
            params (dict): Parameters from GUI input or config files.
            logger (Logger): Logger for logging events.
            data_store (DataStore): DataStore instance for data management.
            data_fetcher (DataFetchUtils): DataFetchUtils instance for data fetching.
        """
        self.params = params
        self.symbol = params.get('symbol', 'AAPL')
        self.csv_path = params.get('csv_path', None)
        self.retrain_threshold = params.get('retrain_threshold', 0.05)
        self.logger = logger
        self.data_store = data_store
        self.data_fetcher = data_fetcher
        self.model_path = f'arima_model_{self.symbol}.pkl'
        self.data = None
        self.best_models = []
        self.order = tuple(map(int, params.get('arima_order', '1,1,1').split(',')))
        self.seasonal_order = tuple(map(int, params.get('seasonal_order', '0,1,1,12').split(',')))
        self.max_iterations = params.get('max_arima_iterations', 50)
        self.stop_requested = False

        try:
            self.model_manager = ModelManager()
            self.logger.info("ModelManager initialized successfully.")
            self.update_gui_status("ModelManager initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize ModelManager: {e}")
            self.update_gui_status(f"Failed to initialize ModelManager: {e}")
            raise e

    def update_gui_status(self, message):
        """Update the GUI status via signal."""
        self.status_update.emit(message)

    async def fetch_data_async(self):
        """Fetch data asynchronously with `DataFetchUtils`, including fallback handling."""
        self.update_gui_status("Attempting to fetch data via DataFetchUtils...")

        source_params = {
            'ticker': self.symbol,
            'start_date': self.params.get('start_date', '2022-01-01'),
            'end_date': self.params.get('end_date', None),
            'interval': self.params.get('interval', '1d'),
            'data_source': self.params.get('data_source', 'Yahoo Finance'),
            'csv_path': self.params.get('csv_path', None),
        }

        try:
            data = await self.data_fetcher.fetch_data(source_params)

            if not data.empty and 'close' in data.columns:
                self.data = data
                self.update_gui_status("Data fetched successfully from selected source.")
                return data

            self.update_gui_status("Data fetching failed or returned empty data.")
            return pd.DataFrame()

        except Exception as e:
            self.update_gui_status(f"Error fetching data: {e}")
            self.logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def apply_derived_features(self):
        """Add derived features like moving averages, RSI, or Bollinger Bands for future scalability."""
        features = self.params.get('features', [])

        if 'moving_average' in features:
            self.data['MovingAverage'] = self.data['close'].rolling(window=5).mean()
            self.update_gui_status("Added Moving Average feature.")
        if 'rsi' in features:
            delta = self.data['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            self.update_gui_status("Added RSI feature.")
        if 'bollinger_bands' in features:
            sma = self.data['close'].rolling(window=20).mean()
            std = self.data['close'].rolling(window=20).std()
            self.data['BollingerUpper'] = sma + (std * 2)
            self.data['BollingerLower'] = sma - (std * 2)
            self.update_gui_status("Added Bollinger Bands feature.")

    def start_training(self, top_n_models=5):
        """Start training mode, continuously train models until user stops."""
        trainer_thread = threading.Thread(target=self._training_process, args=(top_n_models,), daemon=True)
        trainer_thread.start()

    def _training_process(self, top_n_models):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(self.fetch_data_async())
        if data.empty:
            self.update_gui_status("No data loaded. Exiting training.")
            self.submit_button_state(True)
            return

        self.apply_derived_features()

        while not self.stop_requested:
            self.train_models_in_parallel(top_n_models)
            if self.should_early_stop():
                self.update_gui_status("Early stopping triggered. Training halted.")
                break
            time.sleep(1)

        self.logger.info("Training stopped.")
        self.update_gui_status("Training stopped.")
        self.save_best_models()
        self.generate_summary()
        self.submit_button_state(True)

    def submit_button_state(self, is_enabled):
        """Emit signal to update the submit and stop buttons in the GUI."""
        self.button_state_update.emit(is_enabled)

    def train_models_in_parallel(self, top_n_models=5):
        """Train ARIMA models in parallel using multithreading."""
        self.update_gui_status("Starting parallel model training...")

        if len(self.data) < 30:
            self.update_gui_status("Insufficient data points for ARIMA model training.")
            self.logger.error("Insufficient data points for ARIMA model training.")
            return

        model_configurations = [
            (random.choice([(1, 1, 0), (1, 1, 1), (2, 1, 2)]),
             random.choice([(0, 0, 0, 0), (0, 1, 1, 12)]))
            for _ in range(self.max_iterations)
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self.train_arima_model, order, seasonal_order)
                for order, seasonal_order in model_configurations
            ]

            for future in as_completed(futures):
                if self.stop_requested:
                    self.update_gui_status("Stop requested. Terminating training loops.")
                    break
                try:
                    rmse, forecast, trained_model, order, seasonal_order = future.result()
                    if self.stop_requested:
                        self.update_gui_status("Stop requested. Skipping remaining models.")
                        break

                    if rmse == float('inf'):
                        continue

                    if len(forecast) == 0:
                        self.update_gui_status(f"Forecast is empty. Skipping model with RMSE {rmse:.4f}.")
                        continue

                    actual_values = self.data['close'][-len(forecast):]
                    mae = mean_absolute_error(actual_values, forecast)
                    r2 = r2_score(actual_values, forecast)

                    self.best_models.append({
                        'rmse': rmse, 'mae': mae, 'r2': r2,
                        'order': order, 'seasonal_order': seasonal_order,
                        'model': trained_model
                    })
                    self.best_models = sorted(self.best_models, key=lambda x: x['rmse'])[:top_n_models]
                    self.update_iteration_status(len(self.best_models), rmse)

                    self.save_intermediate_results(len(self.best_models))

                except Exception as e:
                    self.logger.error(f"Model training failed: {e}")
                    self.update_gui_status(f"Model training failed: {e}")

    def should_retrain(self, current_rmse, iteration, window=5):
        """Check if retraining is needed based on rolling average of RMSE."""
        if iteration >= window:
            recent_rmse = [model['rmse'] for model in self.best_models[-window:]]
            avg_rmse = sum(recent_rmse) / len(recent_rmse)
            return avg_rmse > self.retrain_threshold
        return False

    def stop_training(self):
        """Stop the training process."""
        self.stop_requested = True
        self.update_gui_status("Stop signal received. Terminating training...")
        self.logger.info("Stop signal received. Terminating training...")

    def save_best_models(self):
        """Save only the best model using ModelManager."""
        if self.best_models:
            best_model_info = self.best_models[0]  # Assuming best model is the one with lowest RMSE
            model = best_model_info['model']
            scaler = None  # Replace with actual scaler if applicable

            metadata = {
                "model_name": f"arima_model_{self.symbol}_best",
                "symbol": self.symbol,
                "rmse": best_model_info['rmse'],
                "mae": best_model_info['mae'],
                "r2": best_model_info['r2'],
                "order": best_model_info['order'],
                "seasonal_order": best_model_info['seasonal_order'],
                "timestamp": pd.Timestamp.now().isoformat()
            }

            try:
                model_name = f"arima_model_{self.symbol}_best"
                model_path, scaler_path, metadata_path = self.model_manager.save_model_and_scaler(
                    model=model,
                    scaler=scaler,
                    model_name=model_name,
                    metadata=metadata,
                    validate=True
                )

                self.logger.info(f"Best model saved: {model_path}")
                if metadata_path:
                    self.logger.info(f"Metadata saved: {metadata_path}")
                self.update_gui_status(f"Best model saved successfully: {model_path}")

            except Exception as e:
                self.logger.error(f"Error saving best model: {e}")
                self.update_gui_status(f"Error saving best model: {e}")

    def save_intermediate_results(self, iteration):
        """Cache intermediate results during training."""
        best_model_info = self.best_models[0]
        model = best_model_info['model']
        scaler = None  # Replace with actual scaler if applicable

        metadata = {
            "model_name": f"arima_model_{self.symbol}_intermediate_v{iteration}",
            "symbol": self.symbol,
            "rmse": best_model_info['rmse'],
            "mae": best_model_info['mae'],
            "r2": best_model_info['r2'],
            "order": best_model_info['order'],
            "seasonal_order": best_model_info['seasonal_order'],
            "iteration": iteration,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        try:
            model_name = f"arima_model_{self.symbol}_intermediate"
            model_path, scaler_path, metadata_path = self.model_manager.save_model_and_scaler(
                model=model,
                scaler=scaler,
                model_name=model_name,
                metadata=metadata,
                validate=False
            )

            self.logger.info(f"Intermediate model saved: {model_path}")
            self.update_gui_status(f"Intermediate model saved: {model_path}")

        except Exception as e:
            self.logger.error(f"Error saving intermediate model at iteration {iteration}: {e}")
            self.update_gui_status(f"Error saving intermediate model at iteration {iteration}: {e}")

    def generate_summary(self):
        """Generate and log a summary of the best model."""
        if self.best_models:
            best_model = self.best_models[0]
            summary = f"Best Model: Order={best_model['order']}, Seasonal Order={best_model['seasonal_order']}, " \
                      f"RMSE={best_model['rmse']:.4f}, MAE={best_model['mae']:.4f}, R2={best_model['r2']:.4f}"
            self.logger.info("Training Summary:")
            self.logger.info(summary)
            self.update_gui_status("Training Summary:")
            self.update_gui_status(summary)

    def visualize_training_progress(self):
        """Visualize the RMSE progression over iterations."""
        if not self.best_models:
            self.update_gui_status("No models to visualize.")
            return

        rmse_values = [model['rmse'] for model in self.best_models]
        iterations = range(1, len(rmse_values) + 1)

        plt.plot(iterations, rmse_values, label='RMSE over Iterations', marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title(f'Training Progress for {self.symbol}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def select_best_model(self):
        """Select the best model based on a weighted score of RMSE, MAE, and R2."""
        if not self.best_models:
            self.update_gui_status("No models trained to select from.")
            self.logger.info("No models trained to select from.")
            return None

        weighted_models = []
        for model in self.best_models:
            weighted_score = (0.5 * model['rmse']) + (0.3 * model['mae']) + (0.2 * (1 - model['r2']))
            weighted_models.append((model, weighted_score))

        weighted_models.sort(key=lambda x: x[1])
        best_model = weighted_models[0][0] if weighted_models else None
        if best_model:
            summary = f"Best model selected: Order={best_model['order']}, Seasonal Order={best_model['seasonal_order']}, RMSE={best_model['rmse']:.4f}"
            self.logger.info(summary)
            self.update_gui_status(summary)
        return best_model

    def should_early_stop(self, patience=10):
        """Check if early stopping is needed based on RMSE stabilization."""
        if len(self.best_models) >= patience:
            recent_rmse = [model['rmse'] for model in self.best_models[-patience:]]
            if max(recent_rmse) - min(recent_rmse) < 0.001:  # RMSE has stabilized
                self.logger.info("Early stopping triggered. RMSE has stabilized.")
                self.update_gui_status("Early stopping triggered. RMSE has stabilized.")
                return True
        return False

    def update_iteration_status(self, iteration, rmse):
        """Update the GUI with real-time progress."""
        message = f"Iteration {iteration}: RMSE = {rmse:.4f}"
        self.update_gui_status(message)

    def train_arima_model(self, order, seasonal_order):
        """Train an ARIMA model based on the given order and seasonal order."""
        self.logger.info(f"Training ARIMA model: order={order}, seasonal_order={seasonal_order}")
        self.update_gui_status(f"Training ARIMA model: order={order}, seasonal_order={seasonal_order}")
        try:
            if seasonal_order != (0, 0, 0, 0):
                model = SARIMAX(self.data['close'], order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(self.data['close'], order=order)

            model_fit = model.fit()

            steps = min(10, len(self.data) - max(order[1], seasonal_order[1] if seasonal_order else 0))

            if steps <= 0:
                self.logger.warning("Not enough data points for forecasting.")
                return float('inf'), [], None, order, seasonal_order

            forecast = model_fit.forecast(steps=steps)

            if len(forecast) == 0:
                self.logger.warning("Forecast is empty.")
                return float('inf'), [], None, order, seasonal_order

            actual_values = self.data['close'][-len(forecast):]
            rmse = mean_squared_error(actual_values, forecast, squared=False)

            return rmse, forecast, model_fit, order, seasonal_order

        except Exception as e:
            self.logger.error(f"Error training ARIMA model: {e}")
            self.update_gui_status(f"Error training ARIMA model: {e}")
            return float('inf'), [], None, order, seasonal_order

# -------------------------------------------------------------------
# Section 6: Running the Model Trainer with GUI
# -------------------------------------------------------------------
def run_model_trainer(params, logger, data_store, data_fetcher, gui_utils):
    """Function to run the ARIMA model trainer."""
    trainer = ARIMAModelTrainer(params=params, logger=logger, data_store=data_store, data_fetcher=data_fetcher)

    # Connect the trainer's signals to the GUI's slots
    trainer.status_update.connect(gui_utils.log_status)
    trainer.button_state_update.connect(gui_utils.update_button_state)
    gui_utils.stop_training_signal.connect(trainer.stop_training)

    # Start training
    trainer.start_training()

# -------------------------------------------------------------------
# Section 7: Main Execution and GUI Launch
# -------------------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui_utils = GUIUtils()

    # Define a slot to handle collected parameters
    def handle_params(params):
        # Set up logging
        log_file_path = Path(params.get('log_file', logs_dir / 'training.log'))
        log_dir = log_file_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists
        max_log_size = 5 * 1024 * 1024  # 5 MB
        backup_count = 3

        # Initialize the logger using setup_logging
        logger = setup_logging(
            script_name="arima_model_trainer",
            log_dir=log_dir,
            max_log_size=max_log_size,
            backup_count=backup_count,
            console_log_level=params.get('log_level', 'DEBUG').upper(),
            file_log_level=params.get('log_level', 'DEBUG').upper()
        )
        logger.setLevel(params.get('log_level', 'DEBUG').upper())

        # Initialize ConfigManager for DataStore
        config = ConfigManager()

        # Initialize DataStore
        data_store = DataStore(config=config, logger=logger, use_csv=False)

        # Initialize DataFetchUtils with the configured logger
        data_fetcher = DataFetchUtils(logger=logger)

        # Start the model trainer in a separate thread
        trainer_thread = threading.Thread(
            target=run_model_trainer,
            args=(params, logger, data_store, data_fetcher, gui_utils),
            daemon=True  # Daemonize thread to exit with the main program
        )
        trainer_thread.start()

    # Connect the GUI's params_collected signal to the handle_params slot
    gui_utils.params_collected.connect(handle_params)

    gui_utils.show()
    sys.exit(app.exec_())
