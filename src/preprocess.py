import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/raw/customer_churn.csv"
OUTPUT_PATH = Path("data/processed")

TARGET = "Churn"

BASE_NUMERIC = [
    "Tenure", "WarehouseToHome", "NumberOfDeviceRegistered", "SatisfactionScore",
    "NumberOfAddress", "Complain", "DaySinceLastOrder", "CashbackAmount",
]

ENGINEERED_NUMERIC = ["cashback_per_month", "days_since_order_per_month", "complain_and_unhappy"]

CATEGORICAL = ["PreferedOrderCat", "MaritalStatus", "tenure_band"]


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    df = df.copy()

    for col in ["PreferedOrderCat", "MaritalStatus"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    if "PreferedOrderCat" in df.columns:
        df["PreferedOrderCat"] = df["PreferedOrderCat"].replace({"Mobile Phone": "Mobile"})

    if TARGET in df.columns:
        if df[TARGET].dtype == object or str(df[TARGET].dtype) == "string":
            df[TARGET] = (
                df[TARGET]
                .astype("string")
                .str.strip()
                .str.lower()
                .map({"yes": 1, "no": 0, "1": 1, "0": 0})
            )
        df = df.dropna(subset=[TARGET])
        df[TARGET] = df[TARGET].astype(int)

    if drop_duplicates:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    months = df["Tenure"] + 1

    df["cashback_per_month"] = df["CashbackAmount"] / months
    df["days_since_order_per_month"] = df["DaySinceLastOrder"] / months

    df["complain_and_unhappy"] = ((df["Complain"] == 1) & (df["SatisfactionScore"] <= 2)).astype(int)

    bands = pd.cut(
        df["Tenure"],
        bins=[-1, 3, 6, 12, 24, np.inf],
        labels=["0-3m", "4-6m", "7-12m", "13-24m", "25m+"],
    )
    df["tenure_band"] = bands.astype("string").fillna("Unknown")
    return df


def split_data(df, train_size=0.70, val_size=0.15, test_size=0.15, seed=42):
    if abs(train_size + val_size + test_size - 1.0) > 1e-9:
        raise ValueError("sizes must sum to 1.0")

    X, y = df.drop(columns=[TARGET]), df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=seed)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_size / (val_size + test_size),
        stratify=y_temp, random_state=seed)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), BASE_NUMERIC + ENGINEERED_NUMERIC),
            ("cat", Pipeline(cat_steps), CATEGORICAL),
        ],
        remainder="drop",
    )
    pre.set_output(transform="pandas")
    return pre


if __name__ == "__main__":

    # Load raw data, clean it, add features, and split into train/val/test sets
    raw = pd.read_csv(DATA_PATH)
    print(f"raw: {raw.shape}, duplicates: {raw.duplicated().sum()}")

    # Clean and preprocess the data
    df = add_features(clean_data(raw))
    print(f"clean: {df.shape}")
    df.to_csv(OUTPUT_PATH / "clean.csv", index=False)

    # Split the data into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Print the shapes of the splits and the churn rates
    pre = build_preprocessor(scale=False)
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)
    print("train:", X_train_t.shape, "| val:", X_val_t.shape, "| test:", X_test_t.shape)
    
    # Print the number of samples and churn rates for each split
    for name, y_part in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"{name}: n={len(y_part)}, churn rate={y_part.mean():.4f}")
