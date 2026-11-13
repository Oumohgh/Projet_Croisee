
from sklearn.experimental import enable_iterative_imputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from src.config import *
from src.transformers import IQRClipper


def preprocess_data():

    colonnesIteratives = [
        "GPA", "Social_Media_Hours", "Screen_Time_Hours",
        "Study_Hours_Per_Week", "Year_of_Study"
    ]

    colonnesMediane = [
        "Sleep_Hours", "Financial_Stress", "Family_Support",
        "Attendance_Rate", "Age"
    ]

    imputation = ColumnTransformer(
        transformers=[
            ("iteratif", IterativeImputer(max_iter=10, random_state=42), colonnesIteratives),
            ("mediane", SimpleImputer(strategy="median"), colonnesMediane)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    pipelineNominal = Pipeline([
        ("imputation", SimpleImputer(strategy="most_frequent")),
        ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pipelineOrdinal = Pipeline([
        ("imputation", SimpleImputer(strategy="most_frequent")),
        ("encodage", OrdinalEncoder(
            categories=list(ORDINAL_COLS.values()),
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    pipelineContinues = Pipeline([
        ("clipping", IQRClipper(factor=1.5)),
        ("standardisation", StandardScaler())
    ])

    transformation = ColumnTransformer(
        transformers=[
            ("continues", pipelineContinues, QUANT_CONTINUES),
            ("discretes", StandardScaler(), QUANT_DISCRETES),
            ("nominal", pipelineNominal, NOMINAL_COLS),
            ("ordinal", pipelineOrdinal, list(ORDINAL_COLS.keys()))
        ],
        remainder="drop"
    )

    pretraitement = Pipeline([
        ("imputation", imputation),
        ("transformation", transformation)
    ]).set_output(transform="pandas")

    return pretraitement
