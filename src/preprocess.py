import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler

DATA_PATH = "data/raw/customer_churn.csv"
OUTPUT_PATH = Path("data/processed")

TARGET = "Churn"

CONTINUOUS_WITH_NA = ["Tenure", "WarehouseToHome", "DaySinceLastOrder"]
SKEWED_NUMERIC = ["CashbackAmount", "cashback_per_month", "days_since_order_per_month"]
OTHER_NUMERIC = ["NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress"]
BINARY_COLS = ["Complain", "complain_and_unhappy"]

CATEGORICAL = ["PreferedOrderCat", "MaritalStatus", "tenure_band"]


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    df = df.copy()

    for col in ["PreferedOrderCat", "MaritalStatus"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().astype(object)

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


def cap_anomalies(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "WarehouseToHome" in X.columns:
        X["WarehouseToHome"] = X["WarehouseToHome"].clip(upper=100)
    if "NumberOfAddress" in X.columns:
        X["NumberOfAddress"] = X["NumberOfAddress"].clip(upper=15)
    if "Tenure" in X.columns:
        X["Tenure"] = X["Tenure"].clip(upper=60)
    return X


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    months = df["Tenure"] + 1

    df["cashback_per_month"] = df["CashbackAmount"] / months
    df["days_since_order_per_month"] = df["DaySinceLastOrder"] / months

    df["complain_and_unhappy"] = (
        (df["Complain"] == 1) & (df["SatisfactionScore"] <= 2)
    ).astype(int)

    bands = pd.cut(
        df["Tenure"],
        bins=[-1, 3, 6, 12, 24, np.inf],
        labels=["0-3m", "4-6m", "7-12m", "13-24m", "25m+"],
    )

    df["tenure_band"] = bands.astype("string").fillna("Unknown").astype(object)
    return df


def split_data(df, train_size=0.70, val_size=0.15, test_size=0.15, seed=42):
    if abs(train_size + val_size + test_size - 1.0) > 1e-9:
        raise ValueError("sizes must sum to 1.0")

    X, y = df.drop(columns=[TARGET]), df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_size / (val_size + test_size),
        stratify=y_temp, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    na_num_steps = [
        ("cap", FunctionTransformer(cap_anomalies, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    
    skewed_steps = [
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ]

    other_num_steps = [
        ("cap", FunctionTransformer(cap_anomalies, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
    ]

    if scale:
        na_num_steps.append(("scale", RobustScaler()))
        skewed_steps.append(("scale", RobustScaler()))
        other_num_steps.append(("scale", RobustScaler()))

    binary_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
    ]

    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    pre = ColumnTransformer(
        transformers=[
            ("num_na", Pipeline(na_num_steps), CONTINUOUS_WITH_NA),
            ("num_skew", Pipeline(skewed_steps), SKEWED_NUMERIC),
            ("num_other", Pipeline(other_num_steps), OTHER_NUMERIC),
            ("bin", Pipeline(binary_steps), BINARY_COLS),
            ("cat", Pipeline(cat_steps), CATEGORICAL),
        ],
        remainder="drop",
    )
    pre.set_output(transform="pandas")
    return pre


if __name__ == "__main__":

    raw = pd.read_csv(DATA_PATH)
    print(f"raw: {raw.shape}, duplicates: {raw.duplicated().sum()}")

    df = add_features(clean_data(raw))
    print(f"clean: {df.shape}")
    df.to_csv(OUTPUT_PATH / "clean.csv", index=False)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    pre = build_preprocessor(scale=False)
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)
    print("train:", X_train_t.shape, "| val:", X_val_t.shape, "| test:", X_test_t.shape)

    for name, y_part in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"{name}: n={len(y_part)}, churn rate={y_part.mean():.4f}")