import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Preparing preprocessing pipeline")

            numerical_columns = [
                'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                'YearsWithCurrManager'
            ]

            categorical_columns = [
                'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole',
                'MaritalStatus', 'OverTime'
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded train and test data")

            preprocessor = self.get_data_transformer_object()

            target_column = "Attrition"
            label_map = {"Yes": 1, "No": 0}

            input_features_train = train_df.drop(columns=[target_column], axis=1)
            target_feature_train = train_df[target_column].map(label_map)

            input_features_test = test_df.drop(columns=[target_column], axis=1)
            target_feature_test = test_df[target_column].map(label_map)

            input_features_train_transformed = preprocessor.fit_transform(input_features_train)
            input_features_test_transformed = preprocessor.transform(input_features_test)

            smote = SMOTE(random_state=42)
            input_features_train_transformed, target_feature_train = smote.fit_resample(
                input_features_train_transformed, target_feature_train
            )

            train_arr = np.c_[input_features_train_transformed, target_feature_train.values]
            test_arr = np.c_[input_features_test_transformed, target_feature_test.values]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("Data transformation complete")
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.Data_Ingestion import DataIngestion

    source_path = os.path.join("Data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion(source_path)

    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)

    print("Transformation complete.")
    print("Train array shape:", train_arr.shape)
    print("Test array shape:", test_arr.shape)
    print("Preprocessor saved to:", preprocessor_path)