"""
ml/pathway_a/experiments/experiment_pitch_free_13feat.py
Phase 2.2A: Systematic Benchmark of Pitch-Free (13-Feature) Acoustic Models.

Evaluates acoustic model candidates on strictly the 13 non-pitch Praat features
under participant-level Leave-One-Group-Out (LOGO) and Repeated StratifiedGroupKFold
cross-validation to eliminate pitch-induced demographic/sex confounding.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Callable

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "parkinsons.data"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "phase2_pitch_free"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PITCH_FREE_13_FEATURES: list[str] = [
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
]

SEEDS: list[int] = [42, 0, 1, 2, 3]  # 5 repeats for SGKF
DECISION_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Candidate Model Definitions
# ---------------------------------------------------------------------------
def get_candidate_models() -> list[dict[str, Any]]:
    """Return dictionary of candidate models with isolated pipeline factories."""
    return [
        {
            "id": "logistic_regression_std",
            "name": "Logistic Regression (StandardScaler, Balanced)",
            "factory": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
            ]),
        },
        {
            "id": "logistic_regression_robust",
            "name": "Logistic Regression (RobustScaler, Balanced)",
            "factory": lambda: Pipeline([
                ("scaler", RobustScaler()),
                ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
            ]),
        },
        {
            "id": "svc_rbf",
            "name": "Support Vector Classifier (RBF Kernel, C=1.0)",
            "factory": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42)),
            ]),
        },
        {
            "id": "svc_linear",
            "name": "Support Vector Classifier (Linear Kernel, C=1.0)",
            "factory": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", C=1.0, probability=True, class_weight="balanced", random_state=42)),
            ]),
        },
        {
            "id": "random_forest",
            "name": "Random Forest (d=4, leaf=3, Balanced)",
            "factory": lambda: Pipeline([
                ("clf", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)),
            ]),
        },
        {
            "id": "extratrees_default",
            "name": "ExtraTrees (d=4, leaf=3, Balanced)",
            "factory": lambda: Pipeline([
                ("clf", ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)),
            ]),
        },
        {
            "id": "extratrees_d3",
            "name": "ExtraTrees (d=3, leaf=3, Balanced - Lower Variance)",
            "factory": lambda: Pipeline([
                ("clf", ExtraTreesClassifier(n_estimators=100, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)),
            ]),
        },
        {
            "id": "gradient_boosting",
            "name": "Gradient Boosting (n=100, d=3, lr=0.05)",
            "factory": lambda: Pipeline([
                ("clf", GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, min_samples_leaf=3, random_state=42)),
            ]),
        },
        {
            "id": "hist_gradient_boosting",
            "name": "HistGradientBoosting (d=3, leaf=3, Balanced)",
            "factory": lambda: Pipeline([
                ("clf", HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=42)),
            ]),
        },
    ]


# ---------------------------------------------------------------------------
# Metric Helper
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute standard classification evaluation metrics for binary classification."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (sens + spec) / 2.0
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": bal_acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier_score_loss(y_true, y_prob),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Validation Engines
# ---------------------------------------------------------------------------
def run_participant_logo(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    candidates: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Execute Leave-One-Group-Out cross-validation across all candidates.
    Returns (summary_df, participant_predictions_df, raw_results).
    """
    logo = LeaveOneGroupOut()
    summary_rows = []
    all_subject_preds = []
    raw_results = {}

    for cand in candidates:
        cand_id = cand["id"]
        cand_name = cand["name"]
        logger.info(f"Running LOGO for {cand_name}...")
        
        recording_preds = []
        recording_probs = []
        subject_results = []
        
        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            sub_id = groups.iloc[test_idx].iloc[0]
            true_label = int(y_test.iloc[0])
            
            model = cand["factory"]()
            model.fit(X_train, y_train)
            
            y_pred_rec = model.predict(X_test)
            y_prob_rec = model.predict_proba(X_test)[:, 1]
            
            recording_preds.extend(y_pred_rec)
            recording_probs.extend(y_prob_rec)
            
            mean_prob = float(np.mean(y_prob_rec))
            sub_pred = int(mean_prob >= DECISION_THRESHOLD)
            
            subject_results.append({
                "candidate_id": cand_id,
                "subject_id": sub_id,
                "true_status": true_label,
                "mean_pd_prob": round(mean_prob, 4),
                "sub_pred": sub_pred,
                "is_correct": int(sub_pred == true_label),
                "recording_accuracy": round(float(np.mean(y_pred_rec == y_test.values)), 4),
            })
            
        sub_df = pd.DataFrame(subject_results)
        all_subject_preds.extend(subject_results)
        
        # Participant-level evaluation
        sub_metrics = compute_metrics(
            sub_df["true_status"].to_numpy(),
            sub_df["sub_pred"].to_numpy(),
            sub_df["mean_pd_prob"].to_numpy(),
        )
        
        # Recording-level evaluation
        rec_metrics = compute_metrics(
            y.to_numpy(),
            np.array(recording_preds),
            np.array(recording_probs),
        )
        
        summary_rows.append({
            "candidate_id": cand_id,
            "candidate_name": cand_name,
            # Subject-level primary metrics
            "sub_balanced_accuracy": round(sub_metrics["balanced_accuracy"], 4),
            "sub_sensitivity": round(sub_metrics["sensitivity"], 4),
            "sub_specificity": round(sub_metrics["specificity"], 4),
            "sub_roc_auc": round(sub_metrics["roc_auc"], 4),
            "sub_f1": round(sub_metrics["f1"], 4),
            "sub_precision": round(sub_metrics["precision"], 4),
            "sub_tp": sub_metrics["tp"],
            "sub_tn": sub_metrics["tn"],
            "sub_fp": sub_metrics["fp"],
            "sub_fn": sub_metrics["fn"],
            # Recording-level secondary metrics
            "rec_balanced_accuracy": round(rec_metrics["balanced_accuracy"], 4),
            "rec_sensitivity": round(rec_metrics["sensitivity"], 4),
            "rec_specificity": round(rec_metrics["specificity"], 4),
            "rec_roc_auc": round(rec_metrics["roc_auc"], 4),
        })
        
        raw_results[cand_id] = {
            "sub_metrics": sub_metrics,
            "rec_metrics": rec_metrics,
            "subject_records": subject_results,
        }

    summary_df = pd.DataFrame(summary_rows).sort_values(by="sub_balanced_accuracy", ascending=False).reset_index(drop=True)
    preds_df = pd.DataFrame(all_subject_preds)
    return summary_df, preds_df, raw_results


def run_repeated_stratified_group_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    candidates: list[dict[str, Any]],
    n_splits: int = 5,
    seeds: list[int] = SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute Repeated StratifiedGroupKFold cross-validation across all candidates.
    Returns (summary_df, fold_metrics_df).
    """
    fold_metric_rows = []
    
    for cand in candidates:
        cand_id = cand["id"]
        cand_name = cand["name"]
        logger.info(f"Running Repeated SGKF for {cand_name} ({len(seeds)} repeats x {n_splits} folds)...")
        
        for repeat_idx, seed in enumerate(seeds):
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            
            for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                test_groups = groups.iloc[test_idx]
                
                model = cand["factory"]()
                model.fit(X_train, y_train)
                
                y_pred_rec = model.predict(X_test)
                y_prob_rec = model.predict_proba(X_test)[:, 1]
                
                # Compute participant-level predictions in test fold
                fold_sub_df = pd.DataFrame({
                    "sub_id": test_groups.values,
                    "y_true": y_test.values,
                    "y_prob": y_prob_rec,
                }).groupby("sub_id").agg({
                    "y_true": "first",
                    "y_prob": "mean",
                })
                fold_sub_df["sub_pred"] = (fold_sub_df["y_prob"] >= DECISION_THRESHOLD).astype(int)
                
                sub_metrics = compute_metrics(
                    fold_sub_df["y_true"].to_numpy(),
                    fold_sub_df["sub_pred"].to_numpy(),
                    fold_sub_df["y_prob"].to_numpy(),
                )
                
                fold_metric_rows.append({
                    "candidate_id": cand_id,
                    "candidate_name": cand_name,
                    "repeat": repeat_idx,
                    "seed": seed,
                    "fold": fold_idx,
                    "sub_balanced_accuracy": sub_metrics["balanced_accuracy"],
                    "sub_sensitivity": sub_metrics["sensitivity"],
                    "sub_specificity": sub_metrics["specificity"],
                    "sub_roc_auc": sub_metrics["roc_auc"],
                    "sub_f1": sub_metrics["f1"],
                })

    fold_df = pd.DataFrame(fold_metric_rows)
    
    # Aggregate mean and std
    summary_rows = []
    for cand_id, g in fold_df.groupby("candidate_id"):
        cand_name = g["candidate_name"].iloc[0]
        summary_rows.append({
            "candidate_id": cand_id,
            "candidate_name": cand_name,
            "sgkf_bal_acc_mean": round(g["sub_balanced_accuracy"].mean(), 4),
            "sgkf_bal_acc_std": round(g["sub_balanced_accuracy"].std(), 4),
            "sgkf_sensitivity_mean": round(g["sub_sensitivity"].mean(), 4),
            "sgkf_sensitivity_std": round(g["sub_sensitivity"].std(), 4),
            "sgkf_specificity_mean": round(g["sub_specificity"].mean(), 4),
            "sgkf_specificity_std": round(g["sub_specificity"].std(), 4),
            "sgkf_roc_auc_mean": round(g["sub_roc_auc"].mean(), 4),
            "sgkf_roc_auc_std": round(g["sub_roc_auc"].std(), 4),
            "sgkf_f1_mean": round(g["sub_f1"].mean(), 4),
            "sgkf_f1_std": round(g["sub_f1"].std(), 4),
        })
        
    summary_df = pd.DataFrame(summary_rows).sort_values(by="sgkf_bal_acc_mean", ascending=False).reset_index(drop=True)
    return summary_df, fold_df


# ---------------------------------------------------------------------------
# Threshold Analysis
# ---------------------------------------------------------------------------
def run_threshold_analysis(
    preds_df: pd.DataFrame,
    thresholds: list[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
) -> pd.DataFrame:
    """Analyze threshold effects across participant LOGO out-of-fold probabilities."""
    rows = []
    for cand_id, g in preds_df.groupby("candidate_id"):
        for th in thresholds:
            sub_preds = (g["mean_pd_prob"] >= th).astype(int)
            m = compute_metrics(g["true_status"].to_numpy(), sub_preds.to_numpy(), g["mean_pd_prob"].to_numpy())
            rows.append({
                "candidate_id": cand_id,
                "threshold": th,
                "balanced_accuracy": round(m["balanced_accuracy"], 4),
                "sensitivity": round(m["sensitivity"], 4),
                "specificity": round(m["specificity"], 4),
                "f1": round(m["f1"], 4),
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    logger.info("Starting Phase 2.2A: Systematic Pitch-Free (13-Feature) Acoustic Model Benchmark")
    
    # 1. Dataset audit
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded raw dataset: {len(df)} rows from {RAW_DATA_PATH}")
    
    # Extract subject grouping
    df["subject_id"] = df["name"].apply(lambda x: x.split("_")[2] if "_" in x else x)
    unique_subjects = df["subject_id"].unique()
    pd_subjects = df[df["status"] == 1]["subject_id"].unique()
    h_subjects = df[df["status"] == 0]["subject_id"].unique()
    
    logger.info(f"Participants: {len(unique_subjects)} total ({len(pd_subjects)} PD, {len(h_subjects)} Healthy)")
    assert len(unique_subjects) == 32, f"Expected 32 participants, got {len(unique_subjects)}"
    assert len(pd_subjects) == 24, f"Expected 24 PD participants, got {len(pd_subjects)}"
    assert len(h_subjects) == 8, f"Expected 8 Healthy participants, got {len(h_subjects)}"
    
    # Verify 13 features
    assert all(f in df.columns for f in PITCH_FREE_13_FEATURES), "Missing pitch-free features"
    
    X = df[PITCH_FREE_13_FEATURES].copy()
    y = df["status"].copy()
    groups = df["subject_id"].copy()
    
    candidates = get_candidate_models()
    logger.info(f"Evaluating {len(candidates)} model candidates...")
    
    # 2. LOGO Cross-Validation
    logo_summary, logo_preds, raw_logo = run_participant_logo(X, y, groups, candidates)
    logo_summary_path = RESULTS_DIR / "model_comparison_logo.csv"
    logo_preds_path = RESULTS_DIR / "participant_predictions_logo.csv"
    logo_summary.to_csv(logo_summary_path, index=False)
    logo_preds.to_csv(logo_preds_path, index=False)
    logger.info(f"Saved LOGO summary to {logo_summary_path}")
    logger.info(f"Saved LOGO predictions to {logo_preds_path}")
    
    # 3. Repeated SGKF Cross-Validation
    sgkf_summary, sgkf_folds = run_repeated_stratified_group_kfold(X, y, groups, candidates)
    sgkf_summary_path = RESULTS_DIR / "model_comparison_sgkf.csv"
    sgkf_folds_path = RESULTS_DIR / "fold_metrics_sgkf.csv"
    sgkf_summary.to_csv(sgkf_summary_path, index=False)
    sgkf_folds.to_csv(sgkf_folds_path, index=False)
    logger.info(f"Saved SGKF summary to {sgkf_summary_path}")
    
    # 4. Threshold Analysis
    thresh_df = run_threshold_analysis(logo_preds)
    thresh_path = RESULTS_DIR / "threshold_analysis.csv"
    thresh_df.to_csv(thresh_path, index=False)
    logger.info(f"Saved threshold analysis to {thresh_path}")
    
    # 5. Combined Benchmark Summary JSON
    benchmark_json = {
        "metadata": {
            "experiment": "Phase 2.2A: Systematic Pitch-Free Acoustic Model Benchmark",
            "features_count": len(PITCH_FREE_13_FEATURES),
            "features": PITCH_FREE_13_FEATURES,
            "excluded_features": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"],
            "dataset_rows": len(df),
            "dataset_subjects": len(unique_subjects),
            "pd_subjects": len(pd_subjects),
            "healthy_subjects": len(h_subjects),
            "validation": "Participant-level Leave-One-Group-Out & 5-Repeat 5-Fold StratifiedGroupKFold",
        },
        "logo_results": logo_summary.to_dict(orient="records"),
        "sgkf_results": sgkf_summary.to_dict(orient="records"),
    }
    json_path = RESULTS_DIR / "benchmark_summary.json"
    with open(json_path, "w") as f:
        json.dump(benchmark_json, f, indent=2)
    logger.info(f"Saved benchmark summary JSON to {json_path}")
    
    print("\n" + "=" * 80)
    print("PHASE 2.2A BENCHMARK RESULTS (PARTICIPANT-LEVEL LOGO)")
    print("=" * 80)
    print(logo_summary[["candidate_id", "sub_balanced_accuracy", "sub_sensitivity", "sub_specificity", "sub_roc_auc", "sub_f1"]].to_string())
    
    print("\n" + "=" * 80)
    print("PHASE 2.2A BENCHMARK RESULTS (REPEATED STRATIFIED GROUP K-FOLD)")
    print("=" * 80)
    print(sgkf_summary[["candidate_id", "sgkf_bal_acc_mean", "sgkf_bal_acc_std", "sgkf_sensitivity_mean", "sgkf_specificity_mean", "sgkf_roc_auc_mean"]].to_string())


if __name__ == "__main__":
    main()
