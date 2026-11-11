import logging
import os
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """Creates and returns a configured logger instance with both console and file handlers.

    Args:
        name (str): Identifier for the logger (usually __name__ or class name).
        log_dir (str): Directory where log files will be persisted.
        log_level (str): Logging threshold ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        logging.Logger: Configured logger ready to use across modules.
    """
    # Ensure logs directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Daily rotating log file name: e.g., logs/2026-11-11.log
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(log_dir, f"{today_str}.log")

    logger = logging.getLogger(name)

    # Convert string level (from yaml) to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Critical: Prevent duplicate handlers if get_logger is called multiple times
    if logger.hasHandlers():
        return logger

    # Standard production log format: [Time] [Level] [Module] - Message
    log_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. File Handler (Writes to disk)
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(log_format)

    # 2. Console Handler (Streams to terminal stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(log_format)

    # Attach both channels
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == "__main__":
    # Self-test block to verify logger independently
    test_logger = get_logger("TestModule", log_dir="logs", log_level="DEBUG")
    test_logger.info("Logger test: Everything initialized successfully.")
    test_logger.warning("Logger test: Sample warning message.")