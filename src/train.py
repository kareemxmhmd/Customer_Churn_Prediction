import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import (
        DATA_PATH,
        add_features,
        build_preprocessor,
        clean_data,
        split_data,
)

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = PROJECT_ROOT / "models"
MODEL_FILENAME = "best_model.pkl"


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()
    return float(negatives / positives)


def build_pipeline(scale_pos_weight: float, seed: int = SEED) -> Pipeline:
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("pre", build_preprocessor(scale=False)),
        ("model", model),
    ])
    return pipeline


def evaluate_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "pr_auc": float(average_precision_score(y, proba)),
    }


def save_model(pipe: Pipeline, output_dir: Path = MODELS_PATH, filename: str = MODEL_FILENAME) -> Path:
    destination = output_dir / filename
    with open(destination, "wb") as f:
        pickle.dump(pipe, f)
    return destination


if __name__ == "__main__":

    # 1) Load raw data
    raw = pd.read_csv(DATA_PATH)
    print(f"Loaded raw data: {raw.shape}")

    # 2) Clean and engineer features
    df = add_features(clean_data(raw))
    print(f"Processed clean data: {df.shape}")

    # 3) Stratified 70/15/15 split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed=SEED)
    print(f"Splits -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 4) Calculate class imbalance ratio and build model pipeline
    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")
    pipeline = build_pipeline(scale_pos_weight=scale_pos_weight, seed=SEED)

    # 5) Train pipeline (preprocessing parameters fitted on train only)
    print("Training XGBoost pipeline...")
    pipeline.fit(X_train, y_train)

    # 6) Evaluate on validation and test sets
    val_metrics = evaluate_model(pipeline, X_val, y_val)
    print("\nValidation Set Metrics")
    for metric_name, score in val_metrics.items():
        print(f"{metric_name:12s}: {score:.4f}")

    print("\nTest Set Classification Report")
    test_pred = pipeline.predict(X_test)
    print(classification_report(y_test, test_pred, digits=3))

    # 7) Save the trained pipeline as .pkl
    saved_path = save_model(pipeline)
    print(f"Model artifact saved successfully to: {saved_path}")
