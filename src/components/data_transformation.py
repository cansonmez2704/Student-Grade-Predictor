import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from exception import CustomException
from logger import get_logger
from utils import save_object


logger = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self) -> ColumnTransformer:
        """
        Build the preprocessing pipeline for both numerical and categorical features.
        """
        try:
            num_features = ["writing score", "reading score"]
            cat_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logger.info("Configured preprocessing pipelines")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features),
                ]
            )

            return preprocessor

        except Exception as error:
            raise CustomException(error, sys) from error

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test datasets")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = "math score"

            input_feature_train = train_df.drop(columns=[target_column_name])
            target_feature_train = train_df[target_column_name]
            input_feature_test = test_df.drop(columns=[target_column_name])
            target_feature_test = test_df[target_column_name]

            logger.info("Applying preprocessing pipeline to training data")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train)

            logger.info("Transforming test data")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessing_obj,
            )
            logger.info(
                "Saved preprocessor object to %s",
                self.data_transformation_config.preprocessor_file_path,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path,
            )

        except Exception as error:
            raise CustomException(error, sys) from error
