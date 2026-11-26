"""
ml/pathway_a/experiments/experiment_phase2_relative_pitch.py
Phase 2.2B: Relative Pitch & Normalized Feature Investigation.

Evaluates whether relative/normalized pitch dynamics (semitone range, ratios, pitch position)
provide useful diagnostic information without reintroducing pitch-driven demographic/sex bias.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Callable

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "phase2_relative_pitch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The 13 pitch-free baseline features verified in Phase 2.2A
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

DECISION_THRESHOLD: float = 0.50
SEEDS: list[int] = [42, 0, 1, 2, 3]  # 5 repeats for SGKF


# ---------------------------------------------------------------------------
# Feature Engineering Helpers
# ---------------------------------------------------------------------------
def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mathematically valid relative and normalized pitch features.
    
    Rejected candidate: Pitch Coefficient of Variation (CV_F0 = sigma_F0 / mu_F0)
    Reason: parkinsons.data contains only scalar summary stats (Fo, Fhi, Flo).
    Pitch standard deviation cannot be derived without the raw audio pitch track.
    """
    df = df.copy()
    fo = df["MDVP:Fo(Hz)"]
    fhi = df["MDVP:Fhi(Hz)"]
    flo = df["MDVP:Flo(Hz)"]

    # 1. Semitone pitch range: Delta F0 = 12 * log2(Fhi / Flo)
    df["semitone_range"] = 12.0 * np.log2(fhi / flo)

    # 2. Relative pitch ratios
    df["ratio_fhi_fo"] = fhi / fo
    df["ratio_fo_flo"] = fo / flo
    df["ratio_fhi_flo"] = fhi / flo

    # 3. Logarithmic pitch ratios
    df["log_ratio_fhi_fo"] = np.log(fhi / fo)
    df["log_ratio_fo_flo"] = np.log(fo / flo)

    # 4. Normalized relative position of mean pitch within span: (Fo - Flo) / (Fhi - Flo)
    # Epsilon prevents zero-division in hypothetical degenerate cases
    df["pitch_pos_ratio"] = (fo - flo) / np.maximum(fhi - flo, 1e-6)

    return df


def get_feature_sets() -> dict[str, list[str]]:
    """Define experimental feature groupings for B0, B1, B2, and B3."""
    return {
        # Experiment B0: Verified 13-feature baseline (pitch-free)
        "B0_pitch_free_13": PITCH_FREE_13_FEATURES,

        # Experiment B1: Relative pitch features ONLY
        "B1_semitone_only": ["semitone_range"],
        "B1_all_relative_pitch": ["semitone_range", "ratio_fhi_fo", "ratio_fo_flo", "pitch_pos_ratio"],

        # Experiment B2: Combined (13 pitch-free + all relative pitch)
        "B2_combined_17": PITCH_FREE_13_FEATURES + ["semitone_range", "ratio_fhi_fo", "ratio_fo_flo", "pitch_pos_ratio"],

        # Experiment B3: Systematic Ablations on Pitch-Free Baseline
        "B3_plus_semitone_range": PITCH_FREE_13_FEATURES + ["semitone_range"],
        "B3_plus_linear_ratios": PITCH_FREE_13_FEATURES + ["ratio_fhi_fo", "ratio_fo_flo"],
        "B3_plus_log_ratios": PITCH_FREE_13_FEATURES + ["log_ratio_fhi_fo", "log_ratio_fo_flo"],
        "B3_plus_pitch_pos_ratio": PITCH_FREE_13_FEATURES + ["pitch_pos_ratio"],
    }


# ---------------------------------------------------------------------------
# Metric Calculation
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute binary classification metrics with zero-division handling."""
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
def evaluate_logo_for_feature_sets(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    model_factory: Callable[[], Any],
    model_label: str = "ExtraTrees (d=3, leaf=3)",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run participant-level LOGO cross-validation across all feature set configurations.
    Returns (summary_df, participant_predictions_df).
    """
    logo = LeaveOneGroupOut()
    y = df["status"].copy()
    groups = df["subject_id"].copy()

    summary_rows = []
    all_subject_preds = []

    for set_name, feature_list in feature_sets.items():
        logger.info(f"Evaluating LOGO on {set_name} ({len(feature_list)} features)...")
        X = df[feature_list].copy()

        rec_preds = []
        rec_probs = []
        subject_results = []

        for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            sub_id = groups.iloc[test_idx].iloc[0]
            true_status = int(y_test.iloc[0])

            clf = model_factory()
            clf.fit(X_train, y_train)

            y_pred_rec = clf.predict(X_test)
            y_prob_rec = clf.predict_proba(X_test)[:, 1]

            rec_preds.extend(y_pred_rec)
            rec_probs.extend(y_prob_rec)

            mean_prob = float(np.mean(y_prob_rec))
            sub_pred = int(mean_prob >= DECISION_THRESHOLD)

            subject_results.append({
                "feature_set": set_name,
                "model": model_label,
                "subject_id": sub_id,
                "true_status": true_status,
                "mean_pd_prob": round(mean_prob, 4),
                "sub_pred": sub_pred,
                "is_correct": int(sub_pred == true_status),
                "recording_accuracy": round(float(np.mean(y_pred_rec == y_test.values)), 4),
            })

        sub_df = pd.DataFrame(subject_results)
        all_subject_preds.extend(subject_results)

        sub_metrics = compute_metrics(
            sub_df["true_status"].to_numpy(),
            sub_df["sub_pred"].to_numpy(),
            sub_df["mean_pd_prob"].to_numpy(),
        )

        rec_metrics = compute_metrics(
            y.to_numpy(),
            np.array(rec_preds),
            np.array(rec_probs),
        )

        summary_rows.append({
            "feature_set": set_name,
            "feature_count": len(feature_list),
            "model": model_label,
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
            "rec_balanced_accuracy": round(rec_metrics["balanced_accuracy"], 4),
            "rec_sensitivity": round(rec_metrics["sensitivity"], 4),
            "rec_specificity": round(rec_metrics["specificity"], 4),
            "rec_roc_auc": round(rec_metrics["roc_auc"], 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    preds_df = pd.DataFrame(all_subject_preds)
    return summary_df, preds_df


def evaluate_repeated_sgkf_for_feature_sets(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    model_factory: Callable[[], Any],
    n_splits: int = 5,
    seeds: list[int] = SEEDS,
) -> pd.DataFrame:
    """Run Repeated StratifiedGroupKFold across all feature set configurations."""
    y = df["status"].copy()
    groups = df["subject_id"].copy()

    summary_rows = []

    for set_name, feature_list in feature_sets.items():
        logger.info(f"Evaluating Repeated SGKF on {set_name} ({len(feature_list)} features)...")
        X = df[feature_list].copy()

        fold_bal_accs = []
        fold_sens = []
        fold_specs = []
        fold_aucs = []
        fold_f1s = []

        for repeat_idx, seed in enumerate(seeds):
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

            for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                test_groups = groups.iloc[test_idx]

                clf = model_factory()
                clf.fit(X_train, y_train)

                y_prob_rec = clf.predict_proba(X_test)[:, 1]

                fold_sub_df = pd.DataFrame({
                    "sub_id": test_groups.values,
                    "y_true": y_test.values,
                    "y_prob": y_prob_rec,
                }).groupby("sub_id").agg({
                    "y_true": "first",
                    "y_prob": "mean",
                })
                fold_sub_df["sub_pred"] = (fold_sub_df["y_prob"] >= DECISION_THRESHOLD).astype(int)

                sub_m = compute_metrics(
                    fold_sub_df["y_true"].to_numpy(),
                    fold_sub_df["sub_pred"].to_numpy(),
                    fold_sub_df["y_prob"].to_numpy(),
                )

                fold_bal_accs.append(sub_m["balanced_accuracy"])
                fold_sens.append(sub_m["sensitivity"])
                fold_specs.append(sub_m["specificity"])
                fold_aucs.append(sub_m["roc_auc"])
                fold_f1s.append(sub_m["f1"])

        summary_rows.append({
            "feature_set": set_name,
            "feature_count": len(feature_list),
            "sgkf_bal_acc_mean": round(float(np.mean(fold_bal_accs)), 4),
            "sgkf_bal_acc_std": round(float(np.std(fold_bal_accs)), 4),
            "sgkf_sensitivity_mean": round(float(np.mean(fold_sens)), 4),
            "sgkf_sensitivity_std": round(float(np.std(fold_sens)), 4),
            "sgkf_specificity_mean": round(float(np.mean(fold_specs)), 4),
            "sgkf_specificity_std": round(float(np.std(fold_specs)), 4),
            "sgkf_roc_auc_mean": round(float(np.nanmean(fold_aucs)), 4),
            "sgkf_roc_auc_std": round(float(np.nanstd(fold_aucs)), 4),
            "sgkf_f1_mean": round(float(np.mean(fold_f1s)), 4),
            "sgkf_f1_std": round(float(np.std(fold_f1s)), 4),
        })

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Confounding and Bias Analysis
# ---------------------------------------------------------------------------
def perform_confounding_analysis(df: pd.DataFrame, preds_df: pd.DataFrame) -> dict[str, Any]:
    """
    Examine whether adding relative pitch features reintroduces bias against
    low-pitch healthy males (S13, S43, S49) or harms high-pitch PD (S08, S26, S34, S35).
    """
    fo = df["MDVP:Fo(Hz)"]
    status = df["status"]

    engineered_cols = [
        "semitone_range", "ratio_fhi_fo", "ratio_fo_flo", "ratio_fhi_flo",
        "log_ratio_fhi_fo", "log_ratio_fo_flo", "pitch_pos_ratio"
    ]

    corrs = {}
    for col in engineered_cols:
        corrs[col] = {
            "corr_with_Fo": round(float(df[col].corr(fo)), 4),
            "corr_with_status": round(float(df[col].corr(status)), 4),
        }

    # Track predictions for critical subject subsets across feature sets
    critical_male_healthy = ["S13", "S43", "S49"]
    critical_female_healthy = ["S07", "S10", "S17", "S42", "S50"]
    critical_high_pitch_pd = ["S08", "S26", "S34", "S35"]

    subject_tracking = {}
    for sub in critical_male_healthy + critical_high_pitch_pd + ["S50"]:
        sub_rows = preds_df[preds_df["subject_id"] == sub]
        subject_tracking[sub] = {
            row["feature_set"]: {
                "mean_pd_prob": row["mean_pd_prob"],
                "sub_pred": row["sub_pred"],
                "is_correct": row["is_correct"],
            }
            for _, row in sub_rows.iterrows()
        }

    return {
        "correlations": corrs,
        "critical_subject_tracking": subject_tracking,
    }


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    logger.info("Starting Phase 2.2B: Relative Pitch & Normalized Feature Investigation")

    # 1. Load dataset & parse subject IDs
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_raw["subject_id"] = df_raw["name"].apply(lambda x: x.split("_")[2] if "_" in x else x)
    logger.info(f"Dataset loaded: {len(df_raw)} rows, {df_raw['subject_id'].nunique()} unique participants.")

    # 2. Engineer relative pitch features
    df = compute_engineered_features(df_raw)
    feature_sets = get_feature_sets()

    # 3. Model factory: Use the validated Phase 2.2A architecture (ExtraTrees d=3, leaf=3, Balanced)
    def et_factory():
        return Pipeline([
            ("clf", ExtraTreesClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])

    # 4. Primary Validation: Participant-level LOGO
    logo_summary, logo_preds = evaluate_logo_for_feature_sets(
        df=df,
        feature_sets=feature_sets,
        model_factory=et_factory,
        model_label="ExtraTrees (d=3, leaf=3)",
    )
    logo_summary_path = RESULTS_DIR / "logo_benchmark_summary.csv"
    logo_preds_path = RESULTS_DIR / "participant_predictions_logo.csv"
    logo_summary.to_csv(logo_summary_path, index=False)
    logo_preds.to_csv(logo_preds_path, index=False)
    logger.info(f"Saved LOGO summary to {logo_summary_path}")

    # 5. Secondary Validation: Repeated StratifiedGroupKFold
    sgkf_summary = evaluate_repeated_sgkf_for_feature_sets(
        df=df,
        feature_sets=feature_sets,
        model_factory=et_factory,
        n_splits=5,
        seeds=SEEDS,
    )
    sgkf_summary_path = RESULTS_DIR / "repeated_sgkf_summary.csv"
    sgkf_summary.to_csv(sgkf_summary_path, index=False)
    logger.info(f"Saved SGKF summary to {sgkf_summary_path}")

    # 6. Confounding and Bias Analysis
    confounding_dict = perform_confounding_analysis(df, logo_preds)
    conf_json_path = RESULTS_DIR / "confounding_analysis.json"
    with open(conf_json_path, "w") as f:
        json.dump(confounding_dict, f, indent=2)
    logger.info(f"Saved confounding analysis to {conf_json_path}")

    # 7. Threshold Analysis on B0, B2, and B3_plus_semitone_range
    thresh_rows = []
    for target_set in ["B0_pitch_free_13", "B2_combined_17", "B3_plus_semitone_range"]:
        set_preds = logo_preds[logo_preds["feature_set"] == target_set]
        for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            sub_preds = (set_preds["mean_pd_prob"] >= th).astype(int)
            m = compute_metrics(set_preds["true_status"].to_numpy(), sub_preds.to_numpy(), set_preds["mean_pd_prob"].to_numpy())
            thresh_rows.append({
                "feature_set": target_set,
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
    thresh_df = pd.DataFrame(thresh_rows)
    thresh_path = RESULTS_DIR / "threshold_analysis.csv"
    thresh_df.to_csv(thresh_path, index=False)
    logger.info(f"Saved threshold analysis to {thresh_path}")

    # Print summary tables to stdout
    print("\n" + "=" * 90)
    print("PHASE 2.2B EXPERIMENT RESULTS — PARTICIPANT-LEVEL LOGO (PRIMARY)")
    print("=" * 90)
    print(logo_summary[["feature_set", "feature_count", "sub_balanced_accuracy", "sub_sensitivity", "sub_specificity", "sub_roc_auc", "sub_f1"]].to_string())

    print("\n" + "=" * 90)
    print("PHASE 2.2B EXPERIMENT RESULTS — REPEATED STRATIFIED GROUP K-FOLD (SECONDARY)")
    print("=" * 90)
    print(sgkf_summary[["feature_set", "feature_count", "sgkf_bal_acc_mean", "sgkf_bal_acc_std", "sgkf_sensitivity_mean", "sgkf_specificity_mean", "sgkf_roc_auc_mean"]].to_string())


if __name__ == "__main__":
    main()
