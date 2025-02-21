import threading
import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

class ARIMAModelTrainer:
    def __init__(self, params, logger, data_fetcher):
        """
        Enhanced ARIMA trainer that continuously trains models until an external stop signal is received.
        """
        self.params = params
        self.logger = logger
        self.data_fetcher = data_fetcher
        self.stop_event = threading.Event()  # Use an Event for safe stopping
        self.best_models = []
        self.loop_interval_seconds = 300  # 5 minutes check loop

    async def fetch_data_async(self) -> pd.DataFrame:
        """
        Fetch data asynchronously via data_fetcher.
        """
        try:
            data = await self.data_fetcher.fetch_data(self.params)
            if data.empty:
                self.logger.error("Fetched data is empty.")
                return pd.DataFrame()
            return data
        except Exception as e:
            self.logger.error(f"Data fetch error: {e}")
            return pd.DataFrame()

    def start_training(self):
        """
        Start the training loop in a background thread.
        """
        trainer_thread = threading.Thread(target=self._training_process, daemon=True)
        trainer_thread.start()

    def _training_process(self):
        """
        Main training loop that repeatedly trains ARIMA models.
        It uses granular waiting intervals and actively cancels pending tasks if the stop signal is set.
        """
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

            # Generate random model configurations
            model_configs = [
                ((random.randint(0, 2), 1, random.randint(0, 2)),
                 (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1), 12))
                for _ in range(5)
            ]

            # Submit training tasks to the ThreadPoolExecutor
            futures_to_config = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                for order, seas in model_configs:
                    future = executor.submit(self._train_arima_model, data, order, seas)
                    futures_to_config[future] = (order, seas)

                # Continuously check for task completion with a short timeout
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
                                self.logger.info(f"ARIMA{order}x{seas} => RMSE={rmse:.4f}")
                        except Exception as e:
                            self.logger.error(f"Train error for {order}x{seas}: {e}")
                    futures = list(not_done)

                # If a stop signal was received, cancel any pending training tasks.
                if self.stop_event.is_set():
                    self.logger.info("Stop signal received. Cancelling pending model training tasks.")
                    for future in futures:
                        future.cancel()

            # Save the best model from this iteration
            self._save_best_model()

            # Wait for the next iteration with a granular sleep that checks for the stop signal every second.
            wait_time = 0
            while wait_time < self.loop_interval_seconds and not self.stop_event.is_set():
                time.sleep(1)
                wait_time += 1

        self.logger.info("Overnight training loop stopped by user.")

    def _train_arima_model(self, df: pd.DataFrame, order, seas_order):
        """
        Train a single ARIMA or SARIMAX model. Returns (rmse, fitted_model).
        """
        try:
            if seas_order != (0, 0, 0, 0):
                model = SARIMAX(df['Close'], order=order, seasonal_order=seas_order)
            else:
                model = ARIMA(df['Close'], order=order)
            result = model.fit()
            forecast_steps = 5
            forecast = result.forecast(forecast_steps)
            actual = df['Close'].tail(forecast_steps)
            rmse = mean_squared_error(actual, forecast, squared=False)
            return rmse, result
        except Exception as e:
            self.logger.error(f"ARIMA fit error {order}x{seas_order}: {e}")
            return float('inf'), None

    def _save_best_model(self):
        """
        Save the top-performing model. (Integrate your model saving logic here.)
        """
        if not self.best_models:
            self.logger.info("No best model to save.")
            return
        best_rmse, best_fit, best_order, best_seas = self.best_models[0]
        self.logger.info(
            f"Saving best model: ARIMA{best_order}x{best_seas}, RMSE={best_rmse:.4f}"
        )
        # TODO: Implement model saving (e.g., using pickle or your ModelManager)

    def stop_training(self):
        """
        Signal the training loop to exit. This method now cancels pending tasks so that
        the "stock signal" (stop signal) immediately stops any ongoing training.
        """
        self.stop_event.set()
        self.logger.info("Stop event set. Exiting overnight loop soon.")
