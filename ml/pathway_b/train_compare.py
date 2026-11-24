"""
ml/pathway_b/train_compare.py
5-fold cross-validated model comparison on acoustic features.

Usage:
    python ml/pathway_b/train_compare.py \
        --features ml/pathway_b/data/processed/features.csv \
        --output   ml/pathway_b/models/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — XGBoost skipped")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES_CSV = BASE_DIR / "data" / "processed" / "features.csv"
DEFAULT_OUTPUT_DIR   = BASE_DIR / "models"

# Non-feature columns
META_COLS = {"filename", "status", "speaker_id"}


def load_features(features_csv: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(features_csv)
    # Drop rows where the target is missing
    df = df.dropna(subset=["status"])
    y = df["status"].astype(int)
    groups = df["speaker_id"].astype(str)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    # Fill NaN with column median
    X = X.fillna(X.median())
    return X, y, groups


def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 4),
        "precision":   round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":      round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(spec, 4),
        "f1":          round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":     round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def build_models(n_pd: int, n_control: int) -> dict[str, Pipeline]:
    """Build pipelines for each classifier."""
    scale_pos_weight = n_control / max(n_pd, 1)  # for XGBoost

    models: dict[str, Pipeline] = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                       max_iter=1000, random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           n_jobs=-1, random_state=42)),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced",
                        probability=True, random_state=42)),
        ]),
    }

    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )),
        ])

    return models


def cross_validate_model(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> dict:
    """Run n-fold cross-validation, return aggregated metrics."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics: list[dict] = []

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Leakage check
        train_groups = set(groups.iloc[train_idx])
        test_groups  = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups), f"[{name}] Fold {fold_i}: speaker leakage!"

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fold_metrics.append(evaluate_fold(y_test.values, y_pred, y_prob))

    # Aggregate across folds
    metric_keys = fold_metrics[0].keys()
    aggregated = {
        k: {
            "mean": round(float(np.mean([f[k] for f in fold_metrics])), 4),
            "std":  round(float(np.std([f[k] for f in fold_metrics])), 4),
        }
        for k in metric_keys
    }

    print(f"  [{name}] Recall={aggregated['recall']['mean']:.4f}±{aggregated['recall']['std']:.4f} "
          f"| F1={aggregated['f1']['mean']:.4f} | ROC-AUC={aggregated['roc_auc']['mean']:.4f}")

    return {"model": name, "cv_folds": n_splits, "metrics": aggregated}


def train_and_compare(features_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {features_csv}")
    X, y, groups = load_features(features_csv)
    n_pd      = int((y == 1).sum())
    n_control = int((y == 0).sum())
    print(f"Dataset: {len(X)} recordings | PD={n_pd} | Control={n_control} | "
          f"Speakers={groups.nunique()} | Features={X.shape[1]}\n")

    models = build_models(n_pd, n_control)
    all_results: list[dict] = []

    for name, pipeline in models.items():
        print(f"Training {name} ...")
        result = cross_validate_model(name, pipeline, X, y, groups)
        all_results.append(result)

    # Rank by recall (primary metric)
    all_results.sort(key=lambda r: r["metrics"]["recall"]["mean"], reverse=True)
    best_name = all_results[0]["model"]
    print(f"\n🏆 Best model (by Recall): {best_name}")

    # Retrain best on full data, save artifact
    best_pipeline = models[best_name]
    best_pipeline.fit(X, y)
    artifact = {
        "pipeline": best_pipeline,
        "feature_names": list(X.columns),
        "metadata": {
            "model_version": "0.2.0-stage2",
            "algorithm": best_name,
            "dataset": "pathway_b acoustic features",
            "cv_folds": 5,
            "random_state": 42,
        },
    }
    model_path = output_dir / "best_pipeline_stage2.joblib"
    joblib.dump(artifact, model_path)

    report = {
        "model_stage": "stage_2_comparison",
        "dataset": str(features_csv),
        "n_recordings": len(X),
        "n_pd": n_pd,
        "n_control": n_control,
        "n_speakers": int(groups.nunique()),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "best_model": best_name,
        "results": all_results,
    }
    report_path = output_dir / "comparison_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved best model to : {model_path}")
    print(f"Saved report to     : {report_path}")
    print("\n--- Full Results ---")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV model comparison on acoustic features.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--output",   type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    train_and_compare(args.features, args.output)


if __name__ == "__main__":
    main()
