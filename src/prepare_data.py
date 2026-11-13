
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import *


def prepare_data():

    data = pd.read_csv(RAW_DATA_PATH)

    X = data.drop(columns=[ID_COL, TARGET])
    y = data[TARGET].map(RISK_ORDER)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("Train :", X_train.shape)
    print("Test :", X_test.shape)
