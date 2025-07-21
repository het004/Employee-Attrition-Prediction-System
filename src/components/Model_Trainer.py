import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        try:
            report = {}
            best_model = None
            best_score = 0

            for name, model in models.items():
                logging.info(f"Tuning hyperparameters for {name}")
                clf = GridSearchCV(model, params.get(name, {}), cv=3)
                clf.fit(X_train, y_train)
                best_model_candidate = clf.best_estimator_

                y_pred = best_model_candidate.predict(X_test)
                score = f1_score(y_test, y_pred)
                report[name] = score
                logging.info(f"{name} F1-Score: {score:.4f}")

                if score > best_score:
                    best_model = best_model_candidate
                    best_score = score

            return best_model, report

        except Exception as e:
            raise CustomException(e, sys)

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,4))
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks([0,1], ["No", "Yes"])
        plt.yticks([0,1], ["No", "Yes"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig("artifacts/confusion_matrix.png")
        plt.close()

    def plot_roc_curve(self, y_true, y_probs):
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend()
        plt.savefig("artifacts/roc_curve.png")
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_probs):
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.savefig("artifacts/precision_recall_curve.png")
        plt.close()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting data into features and labels")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
                "RidgeClassifier": RidgeClassifier(class_weight="balanced"),
                "RandomForest": RandomForestClassifier(class_weight="balanced"),
                "GradientBoosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "SVC": SVC(probability=True, class_weight="balanced"),
                "KNeighbors": KNeighborsClassifier(),
                "GaussianNB": GaussianNB(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "LightGBM": LGBMClassifier(),
                "CatBoost": CatBoostClassifier(verbose=0)
            }

            params = {
                "LogisticRegression": {"C": [0.1, 1, 10]},
                "RidgeClassifier": {"alpha": [0.1, 1.0, 10.0]},
                "RandomForest": {"n_estimators": [50, 100]},
                "GradientBoosting": {"learning_rate": [0.05, 0.1], "n_estimators": [50, 100]},
                "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
                "SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
                "KNeighbors": {"n_neighbors": [3, 5, 7]},
                "GaussianNB": {},
                "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
                "LightGBM": {"n_estimators": [50, 100]},
                "CatBoost": {"iterations": [100, 200], "learning_rate": [0.05, 0.1]}
            }

            best_model, model_report = self.evaluate_models(X_train, y_train, X_test, y_test, models, params)

            if best_model is None:
                raise CustomException("No suitable model found with acceptable F1-score.", sys)

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Best model saved: {type(best_model).__name__}")

            y_pred = best_model.predict(X_test)
            y_probs = best_model.predict_proba(X_test)[:, 1]

            print("\n===== Classification Report for Best Model =====")
            print(classification_report(y_test, y_pred))

            self.plot_confusion_matrix(y_test, y_pred)
            self.plot_roc_curve(y_test, y_probs)
            self.plot_precision_recall_curve(y_test, y_probs)

            return model_report

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.Data_Ingestion import DataIngestion
    from src.components.Data_Transformation import DataTransformation

    ingestion = DataIngestion()
    source_path = os.path.join("Data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    train_path, test_path = ingestion.initiate_data_ingestion(source_path)

    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

    trainer = ModelTrainer()
    report = trainer.initiate_model_trainer(train_arr, test_arr)

    print("\nTraining complete. F1-scores:")
    print(report)
