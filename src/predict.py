import sys
from pathlib import Path
from typing import Any, Dict, Union

import pickle
import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
for directory in [PROJECT_ROOT, SRC_DIR]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

try:
    from src.preprocess import add_features, clean_data
except ModuleNotFoundError:
    from preprocess import add_features, clean_data

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DEFAULT_THRESHOLD = 0.5


def load_model(model_path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Pipeline:
    path = Path(model_path)
    if not path.exists():
        fallback = path.with_suffix(".joblib")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at: {path.resolve()}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def get_risk_level(probability: float) -> str:
    if probability < 0.35:
        return "Low"
    elif probability < 0.65:
        return "Medium"
    return "High"


def predict_single(
    customer_data: Dict[str, Any],
    model: Pipeline | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    if model is None:
        model = load_model()

    df = pd.DataFrame([customer_data])
    processed_df = add_features(clean_data(df, drop_duplicates=False))

    proba = float(model.predict_proba(processed_df)[:, 1][0])
    prediction = int(proba >= threshold)

    return {
        "churn_prediction": prediction,
        "churn_label": "Churn" if prediction == 1 else "Stay",
        "churn_probability": round(proba, 4),
        "risk_level": get_risk_level(proba),
        "threshold_used": threshold,
    }


def predict_batch(
    df: pd.DataFrame,
    model: Pipeline | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    if model is None:
        model = load_model()

    original_df = df.copy()
    processed_df = add_features(clean_data(original_df, drop_duplicates=False))

    probabilities = model.predict_proba(processed_df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results_df = original_df.copy()
    results_df["churn_probability"] = [round(float(p), 4) for p in probabilities]
    results_df["churn_prediction"] = predictions
    results_df["churn_label"] = ["Churn" if p == 1 else "Stay" for p in predictions]
    results_df["risk_level"] = [get_risk_level(p) for p in probabilities]

    return results_df


if __name__ == "__main__":
    print("Testing inference module...")
    pipeline = load_model()

    # Sample 1: High-risk customer (new customer, low tenure, high complaints, unhappy)
    sample_high_risk = {
        "Tenure": 1,
        "WarehouseToHome": 15.0,
        "NumberOfDeviceRegistered": 4,
        "PreferedOrderCat": "Mobile",
        "SatisfactionScore": 1,
        "MaritalStatus": "Single",
        "NumberOfAddress": 3,
        "Complain": 1,
        "DaySinceLastOrder": 2.0,
        "CashbackAmount": 130.0,
    }

    # Sample 2: Low-risk customer (loyal customer, long tenure, no complaints, happy)
    sample_low_risk = {
        "Tenure": 24,
        "WarehouseToHome": 6.0,
        "NumberOfDeviceRegistered": 3,
        "PreferedOrderCat": "Laptop & Accessory",
        "SatisfactionScore": 5,
        "MaritalStatus": "Married",
        "NumberOfAddress": 2,
        "Complain": 0,
        "DaySinceLastOrder": 7.0,
        "CashbackAmount": 210.0,
    }

    res1 = predict_single(sample_high_risk, model=pipeline)
    res2 = predict_single(sample_low_risk, model=pipeline)
    
    # Print the predictions for both samples
    print("\nHigh Risk Customer Prediction")
    for k, v in res1.items():
        print(f"{k:20s}: {v}")

    print("\nLow Risk Customer Prediction")
    for k, v in res2.items():
        print(f"{k:20s}: {v}")
