"""
Train an XGBoost credit risk model on synthetic data.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/credit_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def main():
    df = load_data()
    feature_cols = ["age", "income", "loan_amount", "credit_history_length"]
    X = df[feature_cols].values
    y = (df["default"] == "Y").astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=120,
    )

    y_prob = model.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    model_path = MODEL_DIR / "xgb_model.json"
    scaler_path = MODEL_DIR / "scaler.joblib"

    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Saved XGBoost model to {model_path}")
    print(f"✅ Saved scaler to {scaler_path}")


if __name__ == "__main__":
    main()
