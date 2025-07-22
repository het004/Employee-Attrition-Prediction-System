
# 🧑‍💼 Employee Attrition Prediction System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

> **A comprehensive machine learning solution that predicts employee attrition using advanced analytics and provides an interactive web interface for HR professionals.**

## 🚀 Project Overview

This project leverages machine learning algorithms to predict employee turnover probability, helping organizations proactively identify at-risk employees and implement retention strategies. The system features a user-friendly Streamlit web application that provides real-time predictions with confidence scores.

### 🎯 Key Features

- **🔮 Predictive Analytics**: Advanced ML models predict attrition probability with high accuracy
- **📊 Interactive Dashboard**: Streamlit-powered web interface for easy data input and visualization
- **⚡ Real-time Predictions**: Instant results with confidence percentage
- **🐳 Docker Support**: Containerized deployment for seamless scalability
- **🔧 Modular Architecture**: Clean, maintainable code structure following software engineering best practices
- **📈 Multiple ML Algorithms**: Support for XGBoost, CatBoost, LightGBM, and scikit-learn models

## 🏗️ Architecture & Technical Stack

### **Core Technologies**
- **Backend**: Python 3.10+
- **ML Framework**: Scikit-learn, XGBoost, CatBoost, LightGBM
- **Web Interface**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Model Serialization**: Dill
- **Containerization**: Docker

### **Project Structure**
```
Employee-Attrition-Prediction-System/
├── 📁 src/                          # Source code modules
│   ├── 📁 components/               # ML components (data ingestion, transformation, training)
│   ├── 📁 pipeline/                 # Prediction pipeline
│   │   └── predict_pipeline.py     # Main prediction logic
│   ├── exception.py                 # Custom exception handling
│   ├── logger.py                    # Logging configuration
│   └── utils.py                     # Utility functions
├── 📁 artifacts/                    # Model artifacts (saved models, preprocessors)
├── 📁 Data/                         # Training datasets
├── 📄 main.py                       # Streamlit web application
├── 📄 Dockerfile                    # Docker configuration
├── 📄 requirements.txt              # Python dependencies
└── 📄 setup.py                      # Package setup configuration
```

## 💡 Machine Learning Pipeline

### **Data Processing**
- **Feature Engineering**: 30+ employee attributes including demographics, job satisfaction, and performance metrics
- **Data Preprocessing**: Automated scaling, encoding, and transformation pipeline
- **Model Training**: Multiple algorithms with hyperparameter optimization

### **Key Features Analyzed**
- **Demographic**: Age, Gender, Marital Status, Distance from Home
- **Job-Related**: Job Role, Department, Job Level, Job Satisfaction
- **Compensation**: Monthly Income, Hourly Rate, Stock Options, Salary Hike %
- **Work Environment**: Work-Life Balance, Environment Satisfaction, Overtime
- **Career Development**: Training Hours, Years at Company, Promotion History

### **Model Performance**
The system utilizes ensemble methods and advanced algorithms to achieve high prediction accuracy with probability confidence scoring.

## 🚀 Quick Start

### **Local Installation**

1. **Clone the Repository**
```bash
git clone https://github.com/het004/Employee-Attrition-Prediction-System.git
cd Employee-Attrition-Prediction-System
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Application**
```bash
streamlit run main.py
```

4. **Access the Web Interface**
Navigate to `http://localhost:8501` in your browser

### **Docker Deployment**

1. **Build Docker Image**
```bash
docker build -t employee-attrition-predictor .
```

2. **Run Container**
```bash
docker run -p 8501:8501 employee-attrition-predictor
```

## 📊 Web Application Features

### **Interactive Input Form**
- **Slider Controls**: Age, Income, Working Years, Distance from Home
- **Dropdown Selections**: Job Role, Department, Education Level, Satisfaction Ratings
- **Real-time Validation**: Input validation and error handling

### **Prediction Results**
- **Binary Classification**: Yes/No attrition prediction
- **Confidence Score**: Probability percentage of attrition risk
- **Visual Indicators**: Color-coded results with actionable recommendations
- **Success Animations**: Engaging user feedback with Streamlit balloons

### **Risk Assessment**
- **High Risk Alert**: Warning messages for employees likely to leave
- **Retention Recommendations**: Actionable insights for HR teams
- **Low Risk Confirmation**: Positive reinforcement for stable employees

## 🛠️ Technical Implementation

### **Custom Data Class**
```python
class CustomData:
    # Handles 30+ employee features
    # Converts input to DataFrame format
    # Ensures data type consistency
```

### **Prediction Pipeline**
```python
class PredictPipeline:
    # Loads trained model and preprocessor
    # Performs data transformation
    # Returns prediction with probability
```

### **Error Handling & Logging**
- **Custom Exception Class**: Detailed error tracking with file names and line numbers
- **Comprehensive Logging**: Timestamped logs for debugging and monitoring
- **Graceful Failure**: User-friendly error messages in the web interface

## 📈 Business Impact

### **For HR Professionals**
- **Proactive Retention**: Identify at-risk employees before they leave
- **Data-Driven Decisions**: Evidence-based retention strategies
- **Cost Reduction**: Reduce recruitment and training costs
- **Team Stability**: Maintain productivity and morale

### **For Organizations**
- **Talent Retention**: Preserve institutional knowledge and experience
- **ROI Optimization**: Maximize investment in employee development
- **Strategic Planning**: Workforce planning and succession management

## 🔄 Model Deployment & Maintenance

### **Artifact Management**
- **Model Serialization**: Efficient model storage using Dill
- **Version Control**: Systematic model versioning and rollback capability
- **Performance Monitoring**: Continuous model performance tracking

### **Scalability Features**
- **Docker Containerization**: Easy deployment across different environments
- **Modular Design**: Simple integration with existing HR systems
- **API-Ready Architecture**: Extensible for REST API implementation

## 👨‍💻 Developer Information

**Author**: Het Shah  
**Email**: hetshah1718@gmail.com  
**GitHub**: [@het004](https://github.com/het004)

### **Development Highlights**
- **Clean Code Principles**: Follows PEP 8 standards and best practices
- **Object-Oriented Design**: Modular and maintainable code architecture
- **Error Handling**: Robust exception management and logging
- **Documentation**: Comprehensive code documentation and type hints

## 📝 Dependencies

### **Core Libraries**
```
scikit-learn==1.6.1    # Machine learning algorithms
pandas                 # Data manipulation
streamlit             # Web application framework
xgboost               # Gradient boosting
catboost              # Categorical boosting
lightgbm              # Light gradient boosting
dill                  # Model serialization
imblearn              # Imbalanced dataset handling
```

## 🎯 Future Enhancements

- **Real-time Data Integration**: Connect with HR databases
- **Advanced Analytics**: Detailed employee analytics dashboard
- **Mobile Application**: Cross-platform mobile interface
- **API Development**: RESTful API for enterprise integration
- **A/B Testing**: Model performance comparison tools

## 📜 License

This project is open-source and available for educational and commercial use.

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

**🤝 Connect with me for collaboration opportunities and technical discussions.**
