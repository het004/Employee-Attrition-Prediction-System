import streamlit as st
import pandas as pd
import numpy as np
import os
from src.pipeline.predict_pipeline import PredictPipeline,CustomData

# Page settings
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("🧑‍💼 Employee Attrition Prediction App")
st.write("""
This application predicts the likelihood of an employee leaving the company using a machine learning model trained on HR data.
""")

# Sidebar form
with st.form("attrition_form"):
    st.subheader("Enter Employee Details")

    # Numerical features
    col1, col2 = st.columns(2)
    with col1:
        Age = st.slider("Age", 18, 60, 30)
        DailyRate = st.slider("Daily Rate", 100, 1500, 800)
        DistanceFromHome = st.slider("Distance From Home", 1, 30, 5)
        Education = st.selectbox("Education Level (1=Below College, 5=Doctor)", [1, 2, 3, 4, 5])
        EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        HourlyRate = st.slider("Hourly Rate", 30, 150, 80)
        JobInvolvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        JobSatisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        MonthlyIncome = st.slider("Monthly Income", 1000, 20000, 5000)
        MonthlyRate = st.slider("Monthly Rate", 1000, 30000, 10000)
        NumCompaniesWorked = st.slider("Number of Companies Worked", 0, 10, 1)
        PercentSalaryHike = st.slider("% Salary Hike", 10, 30, 15)

    with col2:
        PerformanceRating = st.selectbox("Performance Rating", [1, 2, 3, 4])
        RelationshipSatisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
        StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        TotalWorkingYears = st.slider("Total Working Years", 0, 40, 8)
        TrainingTimesLastYear = st.slider("Trainings Last Year", 0, 10, 3)
        WorkLifeBalance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
        YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
        YearsInCurrentRole = st.slider("Years in Current Role", 0, 20, 4)
        YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        YearsWithCurrManager = st.slider("Years with Current Manager", 0, 20, 4)

        # Categorical
        BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        Department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        EducationField = st.selectbox("Education Field", ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        JobRole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        OverTime = st.selectbox("OverTime", ["Yes", "No"])

    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    try:
        data = CustomData(
            Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction,
            HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome,
            MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating,
            RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears,
            TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
            YearsSinceLastPromotion, YearsWithCurrManager, BusinessTravel, Department,
            EducationField, Gender, JobRole, MaritalStatus, OverTime
        )

        df = data.get_data_as_data_frame()
        pipeline = PredictPipeline()
        prediction, proba = pipeline.predict(df)

        result = "Yes" if prediction[0] == 1 else "No"
        confidence = round(proba[0] * 100, 2)

        st.markdown("---")
        st.success(f"Attrition Prediction: **{result}**")
        st.info(f"Probability of Attrition: **{confidence}%**")

        if result == "Yes":
            st.warning("⚠️ High risk of attrition. Consider reviewing workload, compensation, or team satisfaction.")
        else:
            st.balloons()
            st.success("✅ Low attrition risk. Keep up the good employee engagement!")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

