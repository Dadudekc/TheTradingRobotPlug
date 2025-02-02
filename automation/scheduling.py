"""
automation/scheduling.py
------------------------
Provides functions to schedule automated training and monitor training progress.
"""

import schedule
import time
import threading
import logging

def run_schedule():
    """
    Runs pending scheduled tasks in an infinite loop.
    """
    while True:
        schedule.run_pending()
        time.sleep(1)

def start_automated_training(interval, training_function):
    """
    Schedules the provided training_function to run automatically.
    
    Args:
        interval (str): One of "daily", "weekly", or "monthly".
        training_function (callable): Function to be scheduled.
    """
    if interval.lower() == "daily":
        schedule.every().day.at("10:00").do(training_function)
        logging.info("Scheduled training daily at 10:00.")
    elif interval.lower() == "weekly":
        schedule.every().week.do(training_function)
        logging.info("Scheduled training weekly.")
    elif interval.lower() == "monthly":
        schedule.every(30).days.do(training_function)
        logging.info("Scheduled training every 30 days.")
    else:
        raise ValueError("Invalid interval for automated training.")
    
    thread = threading.Thread(target=run_schedule, daemon=True)
    thread.start()

def monitor_training_progress(get_progress_function, update_ui_function, is_training_complete_function, target_epochs=10):
    """
    Continuously monitors training progress and updates the UI until training completes.
    
    Args:
        get_progress_function (callable): Returns a dict with training progress details.
        update_ui_function (callable): Updates the UI with progress data.
        is_training_complete_function (callable): Returns True if training is complete.
        target_epochs (int): Epoch threshold to define completion.
    """
    while True:
        progress_data = get_progress_function()
        update_ui_function(progress_data)
        if is_training_complete_function(progress_data, target_epochs):
            logging.info("Training completed.")
            break
        time.sleep(1)
