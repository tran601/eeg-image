import logging
from pathlib import Path
from typing import Tuple


def setup_logger(name: str, config: dict, run_name: str) -> Tuple[logging.Logger, Path]:
    """
    Create a logger using the configuration block.

    Returns the logger and the directory where log files are stored.
    """
    log_dir = Path(config.get("dir", "logs")) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.get("level", "INFO").upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(config.get("format", "%(asctime)s - %(levelname)s - %(message)s"))

    # Clear existing handlers to prevent duplicate logs when re-instantiating.
    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if config.get("console", True):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger, log_dir
