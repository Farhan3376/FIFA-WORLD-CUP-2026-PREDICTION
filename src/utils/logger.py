"""Project-wide logging configuration.

Provides a single :func:`get_logger` helper that returns a configured
:class:`logging.Logger` writing both to the console and to a timestamped file
under ``logs/``. Configuration is idempotent: repeated calls for the same
logger name will not attach duplicate handlers.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.config import LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per log file
_BACKUP_COUNT = 3


def get_logger(
    name: str,
    level: str = "INFO",
    log_filename: str = "pipeline.log",
    console: bool = True,
) -> logging.Logger:
    """Return a configured logger that writes to console and a rotating file.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level as a string (e.g. ``"INFO"``, ``"DEBUG"``).
        log_filename: File (under ``logs/``) to append log records to.
        console: Whether to also emit records to stderr.

    Returns:
        A configured :class:`logging.Logger`. Handlers are only added once.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_resolve_level(level))

    # Avoid duplicate handlers if get_logger is called multiple times.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_path: Path = LOGS_DIR / log_filename
    file_handler = RotatingFileHandler(
        file_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Prevent records from propagating to the root logger (avoids duplicates).
    logger.propagate = False
    return logger


def _resolve_level(level: Optional[str]) -> int:
    """Translate a level name to its numeric value, defaulting to INFO."""
    if not level:
        return logging.INFO
    return getattr(logging, level.upper(), logging.INFO)
