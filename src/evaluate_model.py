
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.config import *


def evaluate_model():

    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    pipeline = joblib.load(
        MODELS_DIR / "meilleur_modele.joblib"
    )

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print("Accuracy :", accuracy)
    print("Precision :", precision)
    print("Recall :", recall)
    print("F1-score :", f1)

    print("Classification Report :")
    print(classification_report(
        y_test,
        y_pred,
        target_names=RISK_LABELS
    ))

    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
