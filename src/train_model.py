
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.config import *
from src.preprocess_data import preprocess_data


def train_model():

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()

    modeles = {
        "L.Regression": LogisticRegression(max_iter=2000),
        "R.N.C": RandomForestClassifier(random_state=RANDOM_STATE)
    }

    resultats = {}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for nom, modele in modeles.items():

        pipeline = Pipeline([
            ("pretraitement", preprocess_data()),
            ("modele", modele)
        ])

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring="accuracy"
        )

        pipeline.fit(X_train, y_train)

        resultats[nom] = scores.mean()

        joblib.dump(
            pipeline,
            MODELS_DIR / f"{nom}.joblib"
        )

        print(nom, "Accuracy CV :", scores.mean())

    meilleur_modele = max(resultats, key=resultats.get)

    meilleur_pipeline = joblib.load(
        MODELS_DIR / f"{meilleur_modele}.joblib"
    )

    joblib.dump(
        meilleur_pipeline,
        MODELS_DIR / "meilleur_modele.joblib"
    )

    print("Meilleur modele :", meilleur_modele)
