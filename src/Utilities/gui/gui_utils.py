from PyQt5 import QtWidgets, QtCore
import json
from pathlib import Path

class GUIUtils(QtWidgets.QWidget):
    params_collected = QtCore.pyqtSignal(dict)
    stop_training_signal = QtCore.pyqtSignal()

    def __init__(self, config_path=None):
        super().__init__()
        self.config_path = Path(config_path or "config/gui_config.json")
        self.model_configurations, self.general_settings = self.load_configurations()
        self.init_ui()

    def load_configurations(self):
        if not self.config_path.exists():
            print(f"Configuration file not found: {self.config_path}")
            return {}, {}
        try:
            with open(self.config_path, 'r') as file:
                config = json.load(file)
            return config.get("models", {}), config.get("general_settings", {})
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {}, {}

    def init_ui(self):
        self.setWindowTitle("ARIMA Model Training - Overnight Mode")
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        # Model Type selection
        self.model_type_combo = QtWidgets.QComboBox(self)
        if self.model_configurations:
            self.model_type_combo.addItems(list(self.model_configurations.keys()))
        else:
            self.model_type_combo.addItems(["ARIMA_Model"])
        layout.addWidget(QtWidgets.QLabel("Model Type:"))
        layout.addWidget(self.model_type_combo)

        # Data Source selection
        self.data_source_combo = QtWidgets.QComboBox(self)
        self.data_source_combo.addItems(["Yahoo", "Alpaca", "CSV"])
        layout.addWidget(QtWidgets.QLabel("Data Source:"))
        layout.addWidget(self.data_source_combo)

        # Stock Symbol input field
        self.symbol_input = QtWidgets.QLineEdit(self)
        self.symbol_input.setPlaceholderText("Enter stock symbol, e.g., TSLA")
        self.symbol_input.setText("TSLA")  # Default value is TSLA.
        layout.addWidget(QtWidgets.QLabel("Stock Symbol:"))
        layout.addWidget(self.symbol_input)

        # Start Training Button
        self.start_button = QtWidgets.QPushButton("Start Overnight Training")
        self.start_button.clicked.connect(self.submit)
        layout.addWidget(self.start_button)

        # Stop Training Button
        self.stop_button = QtWidgets.QPushButton("Stop Training")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: self.stop_training_signal.emit())
        layout.addWidget(self.stop_button)

        # Log messages text area
        self.log_text = QtWidgets.QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Training log messages will appear here...")
        layout.addWidget(self.log_text)

    def submit(self):
        """
        Collect user inputs and emit them for the ARIMA worker.
        """
        params = {
            "model_type": self.model_type_combo.currentText(),
            "data_source": self.data_source_combo.currentText(),
            "symbol": self.symbol_input.text().strip(),
            "mode": "Overnight"  # Single mode
        }
        self.params_collected.emit(params)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.show_message("Parameters submitted. Starting training...")

    def show_message(self, message: str):
        """
        Display a message to the user. This appends the message to the log text area
        and shows a temporary tooltip at the center of the widget.
        Args:
            message (str): The message to display.
        """
        self.log_text.append(message)
        QtWidgets.QToolTip.showText(self.mapToGlobal(self.rect().center()), message, self)
