import os
import sys
from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from exception import CustomException
from logger import get_logger


logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        logger.info("Entered the data ingestion component")
        try:
            source_path = "notebook/StudentsPerformance.csv"
            df = pd.read_csv(source_path)
            logger.info("Dataset loaded from %s", source_path)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info("Persisted raw dataset to %s", self.ingestion_config.raw_data_path)

            logger.info("Starting train/test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logger.info(
                "Ingestion completed",
                extra={
                    "train_path": self.ingestion_config.train_data_path,
                    "test_path": self.ingestion_config.test_data_path,
                },
            )

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as error:
            raise CustomException(error, sys) from error


if __name__ == "__main__":
    train_data, test_data = DataIngestion().initiate_data_ingestion()
    print(f"Train data saved to: {train_data}")
    print(f"Test data saved to: {test_data}")
