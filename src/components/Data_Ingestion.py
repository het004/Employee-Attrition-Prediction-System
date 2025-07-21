import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, source_path):
        logging.info("Starting data ingestion process")

        try:
            # Load dataset
            df = pd.read_csv(source_path)
            logging.info(f"Dataset loaded from {source_path}, shape: {df.shape}")

            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Attrition'])
            logging.info(f"Train shape: {train_set.shape}, Test shape: {test_set.shape}")

            # Save split datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error during data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        source_path = os.path.join("Data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion(source_path)
        print(f"Data Ingestion Complete.\nTrain path: {train_path}\nTest path: {test_path}")
    except Exception as e:
        print(f"Error during ingestion: {e}")
