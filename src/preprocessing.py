import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (ID_COL, NOMINAL_COLS, ORDINAL_COLS, PROCESSED_DIR, QUANT_CONTINUES,
                        QUANT_DISCRETES, RANDOM_STATE, RAW_DATA_PATH, RISK_ORDER, TARGET,
                        TEST_SIZE)
from src.transformers import IQRClipper




def load_data():
    return pd.read_csv(RAW_DATA_PATH)


def split_data(df):
    X = df.drop(columns=[ID_COL, TARGET])
    y = df[TARGET].map(RISK_ORDER)              
    if y.isna().any():
        raise ValueError(f"Valeurs inconnues dans {TARGET} : {sorted(df.loc[y.isna(), TARGET].unique())}")
    return train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)


def save_splits(X_train, X_test, y_train, y_test):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(PROCESSED_DIR / "train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(PROCESSED_DIR / "test.csv", index=False)


COLS_ITERATIVE = ["GPA", "Social_Media_Hours", "Screen_Time_Hours",
                  "Study_Hours_Per_Week", "Year_of_Study"]
COLS_MEDIAN = [c for c in QUANT_CONTINUES + QUANT_DISCRETES if c not in COLS_ITERATIVE]


def build_imputation():
    return ColumnTransformer(
        [("iteratif", IterativeImputer(max_iter=10, random_state=42), COLS_ITERATIVE),
         ("mediane", SimpleImputer(strategy="median"), COLS_MEDIAN)],remainder="passthrough",verbose_feature_names_out=False,                  
    )



def build_transformation():
    continues = Pipeline([("clipping", IQRClipper(factor=1.5)),         
                          ("standardisation", StandardScaler())])        
    nominal = Pipeline([("imputation", SimpleImputer(strategy="most_frequent")),
                        ("encodage", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ordinal = Pipeline([("imputation", SimpleImputer(strategy="most_frequent")),
                        ("encodage", OrdinalEncoder(categories=list(ORDINAL_COLS.values()),
                                                    handle_unknown="use_encoded_value", unknown_value=-1))])
    return ColumnTransformer([
        ("continues", continues, QUANT_CONTINUES),
        ("discretes", StandardScaler(), QUANT_DISCRETES),
        ("nominal", nominal, NOMINAL_COLS),                            
        ("ordinal", ordinal, list(ORDINAL_COLS)),
    ], remainder="drop")

def build_preprocessor():
    return Pipeline([("imputation", build_imputation()),
                     ("transformation", build_transformation())]).set_output(transform="pandas")

def build_pipeline(model):
  
    return Pipeline([("prep", build_preprocessor()), ("model", model)])

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)              
    save_splits(X_train, X_test, y_train, y_test)
    print("Train :", X_train.shape, " Test :", X_test.shape) 

    prep = build_preprocessor()
    X_train_prep = prep.fit_transform(X_train)                       
    X_test_prep = prep.transform(X_test)                            
    print("Train pret :", X_train_prep.shape, "manquants :", int(X_train_prep.isna().sum().sum()))
    print("Test pret  :", X_test_prep.shape, "manquants :", int(X_test_prep.isna().sum().sum()))


if __name__ == "__main__":
    main()