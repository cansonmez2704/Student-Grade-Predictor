import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from exception import CustomException
from logger import get_logger
from utils import get_evaluation_of_models, save_object
logger = get_logger(__name__)
try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore[assignment]
    logger.warning("xgboost not available; skipping XGBRegressor")
else:  # pragma: no cover - environment guard
    logger.warning(
        "Skipping XGBRegressor to avoid OpenMP shared-memory issues in the sandbox environment"
    )
    XGBRegressor = None  # type: ignore[assignment]
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    minimum_r2_score: float = 0.6
class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    def _get_models(self) -> Dict[str, object]:
        models: Dict[str, object] = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.001),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=200, random_state=42, n_jobs=1
            ),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        }
        if XGBRegressor is not None:
            models["XGBRegressor"] = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                max_depth=4,
                objective="reg:squarederror",
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                n_jobs=1,
            )
        return models
    def initiate_model_trainer(
        self, train_array: np.ndarray, test_array: np.ndarray
    ) -> Tuple[str, float]:
        try:
            logger.info("Starting model training phase")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            models = self._get_models()
            evaluation_report, best_model_name, best_model = get_evaluation_of_models(
                X_train, y_train, X_test, y_test, models
            )
            logger.info("Model evaluation report: %s", evaluation_report)
            if not best_model_name or best_model is None:
                raise CustomException(
                    ValueError("No valid model found during evaluation."), sys
                )
            best_score = evaluation_report[best_model_name]
            if best_score < self.model_trainer_config.minimum_r2_score:
                raise CustomException(
                    ValueError(
                        f"Best model R2 score {best_score:.4f} "
                        f"is below acceptable threshold "
                        f"{self.model_trainer_config.minimum_r2_score:.2f}"
                    ),
                    sys,
                )
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logger.info(
                "Best model: %s | R2: %.4f | RMSE: %.4f | MAE: %.4f",
                best_model_name,
                r2,
                rmse,
                mae,
            )
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logger.info(
                "Persisted best model to %s",
                self.model_trainer_config.trained_model_file_path,
            )
            return best_model_name, best_score
        except Exception as error:
            raise CustomException(error, sys) from error



