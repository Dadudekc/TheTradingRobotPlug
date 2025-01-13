import logging
from pathlib import Path

def setup_logging(
    log_name: str,
    log_dir: Path = None,
    max_log_size: int = 5 * 1024 * 1024,  # Default 5 MB
    backup_count: int = 3
) -> logging.Logger:
    """
    Sets up a logger with both console and file handlers.

    Args:
        log_name (str): The name of the logger.
        log_dir (Path, optional): Directory to store log files. Defaults to 'logs/Utilities' in the project root.
        max_log_size (int): Maximum size of the log file in bytes. Defaults to 5 MB.
        backup_count (int): Number of backup log files to retain. Defaults to 3.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(log_name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        # Set default log directory if not provided
        if log_dir is None:
            project_root = Path(__file__).resolve().parents[2]
            log_dir = project_root / 'logs' / 'Utilities'
        
        # Ensure the log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)

        # File handler
        log_file = log_dir / f"{log_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
