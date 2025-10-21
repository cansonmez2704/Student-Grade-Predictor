"""
Utility helpers shared across the project.

Currently only exposes `save_object`, which serializes Python objects to disk.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any

from exception import CustomException
from logger import get_logger


logger = get_logger(__name__)


def save_object(file_path: str | os.PathLike[str], obj: Any) -> None:
    """
    Persist a Python object to ``file_path`` using ``pickle``.

    Args:
        file_path: Destination path where the serialized object will be stored.
        obj: The Python object to serialize.

    Raises:
        CustomException: If any issue occurs while creating directories or writing the file.
    """
    try:
        file_path = Path(file_path)
        if file_path.parent:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info("Serialized object saved to %s", file_path)
    except Exception as exc:
        logger.error("Failed to serialize object to %s", file_path, exc_info=True)
        raise CustomException(exc, sys)
