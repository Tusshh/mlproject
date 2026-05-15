import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

def evaluate_models(x_train, y_train, x_test, y_test, models):

    report = {}

    for i in range(len(models)):
        model = list(models.values())[i]

        model.fit(x_train, y_train)

        y_test_pred = model.predict(x_test)

        test_model_score = r2_score(y_test, y_test_pred)

        report[list(models.keys())[i]] = test_model_score

    return report

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method")

        try:
            # Dataset path
            data_path = os.path.join('notebook', 'data', 'stud.csv')

            # Read dataset
            df = pd.read_csv(data_path)

            logging.info("Dataset read successfully")

            # Create artifacts folder
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # Save raw data
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Train test split initiated")

            # Split dataset
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save train dataset
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Save test dataset
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    logging.info("Starting training pipeline")

    try:
        # Data Ingestion
        obj = DataIngestion()

        train_data, test_data = obj.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()

        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data,
            test_data
        )

        # Model Training
        model_trainer = ModelTrainer()

        model_score = model_trainer.initiate_model_trainer(
            train_arr,
            test_arr,
            preprocessor_path
        )

        print("Model Training Completed")
        print("Model Score:", model_score)

    except Exception as e:
        raise CustomException(e, sys)