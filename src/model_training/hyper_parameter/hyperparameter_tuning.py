# -------------------------------------------------------------------
# File Path: C:\TheTradingRobotPlug\Scripts\ModelTraining\hyper_parameter\hyperparameter_tuning.py
# Description: Hyperparameter tuning script with integrated data storage using DataStore.
# -------------------------------------------------------------------

import os
import sys
import logging
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from optuna.visualization import plot_optimization_history, plot_param_importances

# Adjust the Python path dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir))
sys.path.append(project_root)

from Scripts.Utilities.data_store import DataStore
from Scripts.Utilities.config_handling.config_manager import ConfigManager
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Configure logging
log_path = os.path.join(project_root, 'logs', 'hyperparameter_tuning.log')
logging.basicConfig(level=logging.INFO, filename=log_path, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HyperparameterTuning:
    def __init__(self, model, param_grid, X_train, y_train, scoring='neg_mean_squared_error', cv=5, n_trials=100, n_jobs=-1, early_stopping=True):
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.scoring = scoring
        self.cv = cv
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.early_stopping = early_stopping

    def objective(self, trial):
        try:
            params = {
                key: trial.suggest_categorical(key, value) if isinstance(value[0], str) else (
                    trial.suggest_int(key, value[0], value[1]) if isinstance(value[0], int) else
                    trial.suggest_float(key, value[0], value[1])
                )
                for key, value in self.param_grid.items()
            }
            self.model.set_params(**params)

            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
            self.model.fit(X_train_split, y_train_split)

            y_pred = self.model.predict(X_val_split)
            score = mean_squared_error(y_val_split, y_pred)

            mlflow.log_params(params)
            mlflow.log_metric('mean_squared_error', score)

            return score
        except Exception as e:
            logger.error(f"Error in trial: {e}")
            return float('inf')

    def perform_hyperparameter_tuning(self):
        logger.info("Starting hyperparameter tuning with Optuna and MLflow...")

        mlflow.start_run()

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best score achieved: {best_score}")

        mlflow.log_params(best_params)
        mlflow.log_metric('best_score', best_score)

        self.model.set_params(**best_params)
        self.model.fit(self.X_train, self.y_train)

        # Save results and plots
        results_dir = os.path.join(project_root, 'results', 'hyperparameter_tuning')
        os.makedirs(results_dir, exist_ok=True)

        fig_optimization = plot_optimization_history(study)
        fig_importance = plot_param_importances(study)
        fig_optimization.write_html(os.path.join(results_dir, "optimization_history.html"))
        fig_importance.write_html(os.path.join(results_dir, "param_importances.html"))

        explainer = shap.Explainer(self.model, self.X_train)
        shap_values = explainer(self.X_train)
        shap.summary_plot(shap_values, self.X_train, show=False)
        plt.savefig(os.path.join(results_dir, "shap_summary_plot.png"))

        mlflow.end_run()
        return self.model

# Main script
if __name__ == "__main__":
    # Setup configuration manager and logging
    config_files = [Path('C:/TheTradingRobotPlug/config/config.ini')]
    config_manager = ConfigManager(config_files=config_files)

    # Initialize DataStore (use CSV or SQL based on your configuration)
    use_csv = config_manager.get('use_csv', section='DataStore', fallback=False)
    data_store = DataStore(use_csv=use_csv, config_manager=config_manager, logger=logger)

    # Load dataset from SQL or CSV via DataStore
    symbol = 'TSLA'
    data = data_store.load_data(symbol=symbol)

    if data is not None:
        print("Initial data preview:")
        print(data.head())

        # Preprocess the data
        X_train, X_val, y_train, y_val = train_test_split(
            data.drop(columns=['close', 'timestamp']), data['close'], test_size=0.2, random_state=42
        )

        # Define your model and parameter grid
        model = RandomForestRegressor()
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],  # Avoiding None
        }

        # Hyperparameter Tuning
        ht = HyperparameterTuning(model, param_grid, X_train, y_train)

        # Perform hyperparameter tuning
        best_model = ht.perform_hyperparameter_tuning()

        # Print the best model parameters
        print(f"Best Model Parameters: {best_model.get_params()}")
    else:
        logger.error(f"No data found for {symbol}. Hyperparameter tuning aborted.")
