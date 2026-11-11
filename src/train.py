import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from xgboost import XGBClassifier

from src.config import CV_FOLDS, MODELS_DIR, RANDOM_STATE
from src.preprocessing import build_pipeline, load_data, split_data






def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4,
                                 eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1),
    }


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
    "roc_auc": "roc_auc_ovr",
}


def compare_models(X_train, y_train):

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    lignes, matrices, pipelines = [], {}, {}

    for nom, modele in get_models().items():
        pipe = build_pipeline(modele)         

        
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
        ligne = {"modele": nom}
        for cle in SCORING:
            ligne[cle] = round(scores[f"test_{cle}"].mean(), 3)
        lignes.append(ligne)

        y_pred = cross_val_predict(pipe, X_train, y_train, cv=cv)
        matrices[nom] = confusion_matrix(y_train, y_pred, labels=sorted(y_train.unique()))

        pipelines[nom] = pipe.fit(X_train, y_train)   
    tableau = pd.DataFrame(lignes).set_index("modele").sort_values("f1", ascending=False)
    return tableau, matrices, pipelines


def save_results(tableau, pipelines):
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(tableau.to_dict(orient="index"), f, indent=2)
    for nom, pipe in pipelines.items():
        joblib.dump(pipe, MODELS_DIR / f"{nom}.joblib")


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    tableau, matrices, pipelines = compare_models(X_train, y_train)
    save_results(tableau, pipelines)
    print(tableau)


if __name__ == "__main__":
    main()