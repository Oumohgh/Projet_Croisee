import json

import joblib
import matplotlib
matplotlib.use("Agg")                     
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from src.config import CV_FOLDS, MODELS_DIR, RANDOM_STATE, RISK_LABELS
from src.preprocessing import build_pipeline, load_data, split_data


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),}


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


def save_confusion_matrices(matrices):
    
    MODELS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(matrices), 2, figsize=(11, 4.5 * len(matrices)))

    for ligne, (nom, matrice) in enumerate(matrices.items()):
        pourcentages = matrice / matrice.sum(axis=1, keepdims=True)

        ConfusionMatrixDisplay(matrice, display_labels=RISK_LABELS).plot(
            ax=axes[ligne, 0], cmap="Blues", colorbar=False)
        axes[ligne, 0].set_title(f"{nom} : effectifs")

        ConfusionMatrixDisplay(pourcentages, display_labels=RISK_LABELS).plot(
            ax=axes[ligne, 1], cmap="Blues", colorbar=False, values_format=".0%")
        axes[ligne, 1].set_title(f"{nom} : % par classe réelle")

    for ax in axes.ravel():
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()


def save_results(tableau, matrices, pipelines):
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(tableau.to_dict(orient="index"), f, indent=2)
    for nom, pipe in pipelines.items():
        joblib.dump(pipe, MODELS_DIR / f"{nom}.joblib")
    save_confusion_matrices(matrices)


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    tableau, matrices, pipelines = compare_models(X_train, y_train)
    save_results(tableau, matrices, pipelines)

    print(tableau)
    for nom, matrice in matrices.items():
        print(f"\nMatrice de confusion : {nom}")
        print(pd.DataFrame(matrice, index=RISK_LABELS, columns=RISK_LABELS))


if __name__ == "__main__":
    main()