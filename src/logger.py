from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = LOG_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _configure_root_logger() -> logging.Logger:
    """Configure the project root logger once and return it."""
    root_logger = logging.getLogger("grade_prediction")
    if root_logger.handlers:
        return root_logger

    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT, "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.propagate = False

    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger.

    Child loggers inherit handlers from the project root logger to keep output consistent.
    """
    root_logger = _configure_root_logger()
    if not name:
        return root_logger
    return root_logger.getChild(name)


__all__ = ["get_logger", "LOG_FILE_PATH"]
