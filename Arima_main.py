import sys
from PyQt5 import QtWidgets, QtCore

import threading
import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, Tuple

import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

from Utilities.gui.gui_utils import GUIUtils  # GUI integration
from Utilities.model_training.model_training_utils import ModelManager  # ModelManager integration
from Utilities.shared_utils import setup_logging

logger = setup_logging(script_name="ARIMA_Trainer")

from Utilities.data_fetchers.main_data_fetcher import DataOrchestrator


class ARIMADataFetcher:
    """
    Wraps the DataOrchestrator to fetch stock data for ARIMA training.
    Expects a params dictionary with keys: 'symbol', 'start_date', 'end_date', 'interval', and 'data_source'.
    
    If no symbol is provided, attempts to use the environment variable 'DEFAULT_SYMBOL';
    otherwise, raises a ValueError.
    """
    def __init__(self) -> None:
        self.orchestrator = DataOrchestrator()

    def flatten_columns(self, columns) -> list:
        # Ensure that each column is converted to a string.
        return [('_'.join(map(str, col)) if isinstance(col, tuple) else str(col)).lower().replace(" ", "_")
                for col in columns]

    async def fetch_data(self, params: Dict[str, Any]) -> pd.DataFrame:
        symbol = params.get("symbol")
        if not symbol:
            backup_symbol = os.getenv("DEFAULT_SYMBOL")
            if backup_symbol:
                logger.info("No symbol provided in parameters; using backup symbol from environment.")
                symbol = backup_symbol
            else:
                raise ValueError("No symbol provided in parameters and no backup symbol available.")
        start_date = params.get("start_date", "2023-01-01")
        end_date = params.get("end_date")
        interval = params.get("interval", "1d")
        data_source = params.get("data_source", "yahoo").lower()
        df = pd.DataFrame()
        if data_source == "alpaca":
            try:
                df = await self.orchestrator.fetch_alpaca_data_async(symbol, start_date, end_date, interval)
                if df.empty:
                    logger.info(f"Alpaca data fetch returned empty for symbol {symbol}.")
            except Exception as e:
                logger.error(f"Alpaca data fetch exception for symbol {symbol}: {e}", exc_info=True)
                df = pd.DataFrame()
        if df.empty:
            logger.info(f"Falling back to Yahoo Finance for symbol {symbol}.")
            try:
                df = await self.orchestrator.fetch_stock_data_async(symbol, start_date, end_date, interval)
            except Exception as e:
                logger.error(f"Yahoo Finance fetch exception for symbol {symbol}: {e}", exc_info=True)
                df = pd.DataFrame()
        if df.empty:
            logger.error(f"Both Alpaca and Yahoo Finance fetches failed for symbol {symbol}.")
        else:
            try:
                df.columns = self.flatten_columns(df.columns)
            except Exception as e:
                logger.error(f"Error flattening columns: {e}", exc_info=True)
        return df


class ARIMAModelTrainer:
    def __init__(self, params: Dict[str, Any], logger: Any, data_fetcher: ARIMADataFetcher, gui: GUIUtils, config_manager: Any = None) -> None:
        """
        Enhanced ARIMA trainer that continuously trains models until an external stop signal is received.
        Integrates GUI updates and model saving via ModelManager.
        
        Args:
            params (dict): Training parameters (e.g. model type, symbol, etc.).
            logger (logging.Logger): Logger instance.
            data_fetcher (ARIMADataFetcher): Instance for fetching training data.
            gui (GUIUtils): Main GUI instance (created on the main thread).
            config_manager (optional): Configuration manager instance.
        """
        self.params = params
        self.logger = logger
        self.data_fetcher = data_fetcher
        self.config_manager = config_manager
        self.gui = gui
        self.stop_event = threading.Event()
        self.best_models: list = []
        self.loop_interval_seconds = 300
        self.model_manager = ModelManager(logger, config_manager)

    def update_gui(self, message: str) -> None:
        """Safely update the GUI from a background thread."""
        QtCore.QTimer.singleShot(0, lambda: self.gui.show_message(message))

    async def fetch_data_async(self) -> pd.DataFrame:
        """Fetch data asynchronously via the data_fetcher."""
        try:
            data = await self.data_fetcher.fetch_data(self.params)
            if data.empty:
                self.logger.error("Fetched data is empty.")
                self.update_gui("Fetched data is empty. Training aborted.")
                return pd.DataFrame()
            return data
        except Exception as e:
            self.logger.error(f"Data fetch error: {e}", exc_info=True)
            self.update_gui(f"Data fetch error: {e}")
            return pd.DataFrame()

    def start_training(self) -> None:
        """Start the training loop in a background thread."""
        self.update_gui("Starting ARIMA training...")
        trainer_thread = threading.Thread(target=self._training_process, daemon=True)
        trainer_thread.start()

    def _training_process(self) -> None:
        """Main training loop that repeatedly trains ARIMA models."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(self.fetch_data_async())
        if data.empty:
            self.logger.info("No data available; training aborted.")
            return

        iteration = 0
        while not self.stop_event.is_set():
            iteration += 1
            self.logger.info(f"ARIMA overnight iteration #{iteration}")
            self.update_gui(f"Starting iteration #{iteration}")

            model_configs = [((random.randint(0, 2), 1, random.randint(0, 2)),
                              (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), 12))
                             for _ in range(5)]
            futures_to_config: Dict[Any, Tuple[tuple, tuple]] = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                for order, seas in model_configs:
                    future = executor.submit(self._train_arima_model, data, order, seas)
                    futures_to_config[future] = (order, seas)
                futures = list(futures_to_config.keys())
                while futures and not self.stop_event.is_set():
                    done, not_done = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        order, seas = futures_to_config[future]
                        try:
                            rmse, model_fit = future.result()
                            if rmse < float('inf'):
                                self.best_models.append((rmse, model_fit, order, seas))
                                self.best_models.sort(key=lambda x: x[0])
                                self.best_models = self.best_models[:5]
                                msg = f"ARIMA{order}x{seas} => RMSE={rmse:.4f}"
                                self.logger.info(msg)
                                self.update_gui(msg)
                        except Exception as e:
                            self.logger.error(f"Train error for {order}x{seas}: {e}", exc_info=True)
                    futures = list(not_done)
                if self.stop_event.is_set():
                    self.logger.info("Stop signal received. Cancelling pending training tasks.")
                    self.update_gui("Stop signal received. Cancelling pending training tasks.")
                    for future in futures:
                        future.cancel()
            self._save_best_model()
            wait_time = 0
            while wait_time < self.loop_interval_seconds and not self.stop_event.is_set():
                time.sleep(1)
                wait_time += 1
        self.logger.info("Overnight training loop stopped by user.")
        self.update_gui("Overnight training loop stopped.")

    def _train_arima_model(self, df: pd.DataFrame, order: tuple, seas_order: tuple) -> Tuple[float, Any]:
        """Train a single ARIMA or SARIMAX model."""
        try:
            if seas_order != (0, 0, 0, 0):
                model = SARIMAX(df['close'], order=order, seasonal_order=seas_order)
            else:
                model = ARIMA(df['close'], order=order)
            result = model.fit()
            forecast_steps = 5
            forecast = result.forecast(forecast_steps)
            actual = df['close'].tail(forecast_steps)
            rmse = mean_squared_error(actual, forecast, squared=False)
            return rmse, result
        except Exception as e:
            self.logger.error(f"ARIMA fit error {order}x{seas_order}: {e}", exc_info=True)
            return float('inf'), None

    def _save_best_model(self) -> None:
        """Save the best-performing model using ModelManager."""
        if not self.best_models:
            self.logger.info("No best model to save.")
            self.update_gui("No best model to save.")
            return
        best_rmse, best_fit, best_order, best_seas = self.best_models[0]
        msg = f"Saving best model: ARIMA{best_order}x{best_seas}, RMSE={best_rmse:.4f}"
        self.logger.info(msg)
        self.update_gui(msg)
        symbol = self.params.get("symbol") or os.getenv("DEFAULT_SYMBOL")
        hyperparameters = {"order": best_order, "seasonal_order": best_seas}
        metrics = {"rmse": best_rmse}
        try:
            saved_files = self.model_manager.save_model(
                model=best_fit,
                symbol=symbol,
                model_type="arima",
                hyperparameters=hyperparameters,
                metrics=metrics,
                scaler=None
            )
            self.logger.info(f"Model saved successfully: {saved_files}")
            self.update_gui(f"Best model saved: ARIMA{best_order}x{best_seas}, RMSE={best_rmse:.4f}")
        except Exception as e:
            error_msg = f"Error saving best model: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.update_gui(error_msg)

    def stop_training(self) -> None:
        """Signal the training loop to exit and cancel pending tasks."""
        self.stop_event.set()
        self.logger.info("Stop event set. Exiting overnight loop soon.")
        self.update_gui("Stop event set. Exiting training soon.")


def handle_params(params: Dict[str, Any], gui: GUIUtils) -> None:
    """
    Handle parameters collected from the GUI, create and start the ARIMA trainer.
    
    Expects that the GUI provides a 'symbol' field (default TSLA).
    """
    if not params.get("symbol"):
        params["symbol"] = "TSLA"
    data_fetcher = ARIMADataFetcher()
    trainer = ARIMAModelTrainer(params, logger, data_fetcher, gui)
    handle_params.trainer_ref = trainer
    trainer.start_training()


def stop_training(gui: GUIUtils) -> None:
    """Stop training if a trainer instance exists."""
    trainer_ref = getattr(handle_params, 'trainer_ref', None)
    if trainer_ref:
        trainer_ref.stop_training()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    gui = GUIUtils(config_path="config/gui_config.json")
    gui.params_collected.connect(lambda params: handle_params(params, gui))
    gui.stop_training_signal.connect(lambda: stop_training(gui))
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
