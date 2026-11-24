import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "parkinsons.data"
MODELS_DIR = BASE_DIR / "models"
MODEL_ARTIFACT_PATH = MODELS_DIR / "baseline_pipeline.joblib"
METRICS_OUTPUT_PATH = MODELS_DIR / "baseline_metrics.json"


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(path)
    subject_ids = df["name"].apply(lambda x: x.split("_")[2])
    y = df["status"].astype(int)
    X = df.drop(columns=["name", "status"])
    return X, y, subject_ids


def train_and_evaluate() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups = load_data(RAW_DATA_PATH)

    # Participant-level stratified split to prevent speaker leakage
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_subjects = set(groups.iloc[train_idx])
    test_subjects = set(groups.iloc[test_idx])

    assert train_subjects.isdisjoint(test_subjects), "Data leakage detected across subject IDs!"

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "model_stage": "stage_1_baseline",
        "algorithm": "LogisticRegression (class_weight=balanced)",
        "dataset": "UCI Parkinson's (parkinsons.data)",
        "split_strategy": "StratifiedGroupKFold (participant-level, seed=42)",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_participants": len(train_subjects),
        "test_participants": len(test_subjects),
        "feature_names": list(X.columns),
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall_sensitivity": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "specificity": round(specificity, 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    artifact = {
        "pipeline": pipeline,
        "feature_names": list(X.columns),
        "metadata": {
            "model_version": "0.1.0-baseline",
            "algorithm": "LogisticRegression",
            "random_state": 42,
        },
    }

    joblib.dump(artifact, MODEL_ARTIFACT_PATH)
    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model artifact to: {MODEL_ARTIFACT_PATH}")
    print(f"Saved metrics report to: {METRICS_OUTPUT_PATH}\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_and_evaluate()