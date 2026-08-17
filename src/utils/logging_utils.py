# =============================================================================
# Logging Utilities
#
# Provides consistent logging across all TAHIMIK modules. Each module
# gets a named logger so debug output can be filtered per-component
# during the thesis defense.
# =============================================================================

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "tahimik",
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger that writes to both console and an optional file.

    Args:
        name: Logger name (e.g., "tahimik.training", "tahimik.evaluation").
        log_file: Path to a log file. If None, logs only to console.
        level: Logging level (DEBUG for thesis defense debugging).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers when the function is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
