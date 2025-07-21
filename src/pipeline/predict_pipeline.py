import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)
            probas = model.predict_proba(data_scaled)

            return prediction, probas[:, 1]  # class label, probability of positive class

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction,
                 HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome,
                 MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating,
                 RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears,
                 TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany,
                 YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager,
                 BusinessTravel, Department, EducationField, Gender, JobRole,
                 MaritalStatus, OverTime):

        self.data = {
            "Age": Age,
            "DailyRate": DailyRate,
            "DistanceFromHome": DistanceFromHome,
            "Education": Education,
            "EnvironmentSatisfaction": EnvironmentSatisfaction,
            "HourlyRate": HourlyRate,
            "JobInvolvement": JobInvolvement,
            "JobLevel": JobLevel,
            "JobSatisfaction": JobSatisfaction,
            "MonthlyIncome": MonthlyIncome,
            "MonthlyRate": MonthlyRate,
            "NumCompaniesWorked": NumCompaniesWorked,
            "PercentSalaryHike": PercentSalaryHike,
            "PerformanceRating": PerformanceRating,
            "RelationshipSatisfaction": RelationshipSatisfaction,
            "StockOptionLevel": StockOptionLevel,
            "TotalWorkingYears": TotalWorkingYears,
            "TrainingTimesLastYear": TrainingTimesLastYear,
            "WorkLifeBalance": WorkLifeBalance,
            "YearsAtCompany": YearsAtCompany,
            "YearsInCurrentRole": YearsInCurrentRole,
            "YearsSinceLastPromotion": YearsSinceLastPromotion,
            "YearsWithCurrManager": YearsWithCurrManager,
            "BusinessTravel": BusinessTravel,
            "Department": Department,
            "EducationField": EducationField,
            "Gender": Gender,
            "JobRole": JobRole,
            "MaritalStatus": MaritalStatus,
            "OverTime": OverTime
        }

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame([self.data])
        except Exception as e:
            raise CustomException(e, sys)
