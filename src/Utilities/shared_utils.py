import logging
from pathlib import Path  # Add this import

def setup_logging(
    script_name: str,
    log_dir: Path = None,
    max_log_size: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    console_log_level: int = logging.INFO,
    file_log_level: int = logging.DEBUG,
    feedback_loop_enabled: bool = False,
    log_level: int = logging.INFO  # Add this parameter
) -> logging.Logger:
    """
    Sets up a logger with console and file handlers, including optional feedback loop integration.

    Args:
        script_name (str): Name of the script (used as the logger name and file prefix).
        log_dir (Path, optional): Directory to store log files. Defaults to 'logs/Utilities'.
        max_log_size (int): Maximum size of the log file in bytes. Defaults to 5 MB.
        backup_count (int): Number of backup log files to retain. Defaults to 3.
        console_log_level (int): Logging level for the console handler. Defaults to logging.INFO.
        file_log_level (int): Logging level for the file handler. Defaults to logging.DEBUG.
        feedback_loop_enabled (bool): Placeholder for integrating feedback mechanisms. Defaults to False.
        log_level (int): Overall logging level for the logger. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(script_name)
    logger.setLevel(log_level)  # Use the log_level parameter

    # Avoid duplicate handlers
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Ensure the log directory exists
        if log_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            log_dir = project_root / 'logs' / 'Utilities'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Define log file
        log_file = log_dir / f"{script_name}.log"

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_log_level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # Optionally enable a feedback loop (future extension)
    if feedback_loop_enabled:
        logger.debug("Feedback loop is enabled.")

    return logger

