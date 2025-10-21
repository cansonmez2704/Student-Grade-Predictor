"""
Utility helpers shared across the project.

Provides helpers for persisting Python objects and evaluating ML models.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import r2_score

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


def get_evaluation_of_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Any],
) -> Tuple[Dict[str, float], str, Any]:
    """
    Fit and evaluate a collection of regression models.

    Returns a tuple containing:
        * report mapping model name -> R^2 score on the test set
        * name of the best-performing model
        * the fitted best model instance
    """
    report: Dict[str, float] = {}
    best_model_name = ""
    best_model_score = float("-inf")
    best_model = None

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            report[model_name] = score
            logger.info("Model %s achieved R2 score %.4f", model_name, score)

            if score > best_model_score:
                best_model_score = score
                best_model_name = model_name
                best_model = model
        except Exception as exc:
            logger.error("Model %s failed during evaluation", model_name, exc_info=True)
            report[model_name] = float("-inf")

    if not report:
        raise CustomException(ValueError("No models were evaluated successfully."), sys)

    return report, best_model_name, best_model
