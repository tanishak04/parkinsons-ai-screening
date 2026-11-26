"""
ml/pathway_a/experiments/experiment_calibration.py
Phase 2.2C: Probability Calibration Benchmark for Parkinson's AI Screening.

Follows strict subject-grouped cross-validation to prevent patient identity leakage.
Evaluates:
  - Model A: Formal Stage 1 Baseline (22 features, StandardScaler + LogisticRegression(class_weight='balanced'))
  - Model B: Phase 2 Research Baseline (13 pitch-free features, ExtraTreesClassifier(d=3, leaf=3, class_weight='balanced'))

Calibration Methods:
  1. Uncalibrated (raw model probabilities)
  2. Platt / Sigmoid Scaling (Parametric logistic calibration)
  3. Isotonic Calibration (Exploratory non-parametric step calibration)
  4. Prevalence / Intercept Adjustment (Exploratory logit correction for class_weight='balanced')

Cross-Validation Architecture:
  - Outer: 32-Fold Leave-One-Group-Out (LOGO) & 5-Repeat 5-Fold StratifiedGroupKFold
  - Inner: 5-Fold StratifiedGroupKFold on training participants exclusively
  - Audits inner-vs-outer score distribution mismatch
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
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
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "parkinsons.data"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "phase2_calibration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_22: list[str] = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
    "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

PITCH_FREE_13_FEATURES: list[str] = [
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ",
    "Shimmer:DDA", "NHR", "HNR"
]

SEEDS_SGKF: list[int] = [42, 123, 456, 789, 2024]
N_SPLITS_SGKF: int = 5
INNER_SPLITS: int = 5
DECISION_THRESHOLD: float = 0.50

CRITICAL_SUBJECTS = ["S13", "S43", "S49", "S50", "S08", "S26", "S34", "S35"]


# ---------------------------------------------------------------------------
# Platt / Sigmoid Calibrator Class (Strictly Group-Safe & Parametric)
# ---------------------------------------------------------------------------
class PlattCalibrator:
    """
    Parametric Platt / Sigmoid calibrator using Niculescu-Mizil & Caruana (2005)
    Bayesian target smoothing to prevent infinite weights on separable inner scores:
      y_smooth = (y + 1) / (N_class + 2)
    Fits P(y=1 | s) = 1 / (1 + exp(A * s + B)).
    """
    def __init__(self):
        self.a_ = 1.0
        self.b_ = 0.0
        self.fitted_ = False

    def fit(self, scores: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        scores = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)

        # Bayesian target smoothing (Platt 1999)
        t = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))

        def loss(params):
            a, b = params
            p = expit(a * scores + b)
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            # Binary cross-entropy with smoothed targets
            return -np.sum(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))

        # Initial guess: standard logistic regression slope & intercept
        res = minimize(loss, [0.0, 0.0], method="L-BFGS-B")
        if res.success:
            self.a_, self.b_ = res.x
        else:
            self.a_, self.b_ = 1.0, 0.0
        self.fitted_ = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).ravel()
        p1 = expit(self.a_ * scores + self.b_)
        return np.clip(p1, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Calibration Metrics Helper Functions
# ---------------------------------------------------------------------------
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5, strategy: str = "uniform") -> tuple[float, float, list[dict[str, Any]]]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    Strategies: 'uniform' (equal width) or 'quantile' (equal frequency).
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, []

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bin_edges[1:-1])  # 0 to n_bins - 1
    elif strategy == "quantile":
        # Quantile binning with unique fallbacks
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        # If duplicated edges occur due to tied probabilities, jitter slightly
        for i in range(1, len(bin_edges)):
            if bin_edges[i] <= bin_edges[i - 1]:
                bin_edges[i] = bin_edges[i - 1] + 1e-6
        bin_ids = np.digitize(y_prob, bin_edges[1:-1])
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    ece = 0.0
    mce = 0.0
    bin_details = []

    for b in range(n_bins):
        mask = (bin_ids == b)
        count = np.sum(mask)
        if count > 0:
            conf = np.mean(y_prob[mask])
            acc = np.mean(y_true[mask])
            abs_err = abs(acc - conf)
            ece += (count / n) * abs_err
            mce = max(mce, abs_err)
            bin_details.append({
                "bin_idx": b,
                "count": int(count),
                "conf": float(conf),
                "acc": float(acc),
                "abs_err": float(abs_err)
            })
        else:
            bin_details.append({
                "bin_idx": b,
                "count": 0,
                "conf": 0.0,
                "acc": 0.0,
                "abs_err": 0.0
            })

    return float(ece), float(mce), bin_details


def compute_comprehensive_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict[str, Any]:
    """Computes discrimination, calibration, and classification metrics."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    brier = float(brier_score_loss(y_true, y_prob))
    ece_uni, mce_uni, _ = compute_ece(y_true, y_prob, n_bins=5, strategy="uniform")
    ece_qnt, mce_qnt, _ = compute_ece(y_true, y_prob, n_bins=5, strategy="quantile")

    # Safe ROC-AUC & PR-AUC
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    sens = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    spec = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "brier_score": round(brier, 4),
        "ece_uniform_5": round(ece_uni, 4),
        "ece_quantile_5": round(ece_qnt, 4),
        "mce_uniform_5": round(mce_uni, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


# ---------------------------------------------------------------------------
# Data Loading & Preparation
# ---------------------------------------------------------------------------
def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    # Subject ID extracted from name (e.g. phon_R01_S01_1 -> S01)
    df["subject_id"] = df["name"].apply(lambda x: x.split("_")[2])

    y = df["status"].astype(int)
    groups = df["subject_id"]
    names = df["name"]
    return df, y, groups, names


# ---------------------------------------------------------------------------
# Model Factories
# ---------------------------------------------------------------------------
def build_baseline_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])


def build_research_et_pipeline() -> Pipeline:
    return Pipeline([
        ("classifier", ExtraTreesClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=42
        ))
    ])


# ---------------------------------------------------------------------------
# Nested Group Calibration Engine
# ---------------------------------------------------------------------------
def run_nested_calibration_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    names: pd.Series,
    model_type: str,  # 'baseline_lr_22' or 'research_et_13'
    scheme: str = "LOGO",  # 'LOGO' or 'SGKF'
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """
    Executes nested cross-validation with subject grouping in both inner and outer loops.
    Returns:
      - oof_recordings_df: recording-level predictions for all calibration methods
      - oof_participants_df: participant-level aggregated predictions
      - fold_metrics_list: per-fold performance metrics
      - mismatch_stats: audit of inner vs outer score distribution mismatch
    """
    if model_type == "baseline_lr_22":
        pipeline_builder = build_baseline_lr_pipeline
        feature_cols = FEATURES_22
        supports_prevalence = True
    elif model_type == "research_et_13":
        pipeline_builder = build_research_et_pipeline
        feature_cols = PITCH_FREE_13_FEATURES
        supports_prevalence = False
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    X_sub = X[feature_cols].copy()

    # Configure outer splitter
    if scheme == "LOGO":
        outer_splitter = LeaveOneGroupOut()
        splits = list(outer_splitter.split(X_sub, y, groups))
    elif scheme == "SGKF":
        outer_splitter = StratifiedGroupKFold(n_splits=N_SPLITS_SGKF, shuffle=True, random_state=seed)
        splits = list(outer_splitter.split(X_sub, y, groups))
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    methods = ["uncalibrated", "sigmoid", "isotonic"]
    if supports_prevalence:
        methods.append("prevalence_adjusted")

    recording_records = []
    fold_metrics = []

    inner_scores_all = []
    outer_scores_all = []
    ks_pvalues = []

    logger.info(f"Starting Nested Calibration: Model={model_type}, Scheme={scheme}, Folds={len(splits)}")

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_subjects = set(groups.iloc[train_idx])
        test_subjects = set(groups.iloc[test_idx])
        assert train_subjects.isdisjoint(test_subjects), (
            f"Outer leakage detected in fold {fold_idx}: train and test subjects overlap!"
        )

        X_train, y_train = X_sub.iloc[train_idx], y.iloc[train_idx]
        groups_train = groups.iloc[train_idx]
        X_test, y_test = X_sub.iloc[test_idx], y.iloc[test_idx]

        # -------------------------------------------------------------------
        # Inner Group Cross-Validation on Outer Training Data Exclusively
        # -------------------------------------------------------------------
        inner_splitter = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=42)
        inner_splits = list(inner_splitter.split(X_train, y_train, groups_train))

        inner_oof_probs = np.full(len(X_train), np.nan, dtype=float)
        inner_oof_scores = np.full(len(X_train), np.nan, dtype=float)

        for in_train_idx, in_val_idx in inner_splits:
            in_train_subj = set(groups_train.iloc[in_train_idx])
            in_val_subj = set(groups_train.iloc[in_val_idx])
            assert in_train_subj.isdisjoint(in_val_subj), "Inner fold subject leakage!"

            in_pipeline = pipeline_builder()
            in_pipeline.fit(X_train.iloc[in_train_idx], y_train.iloc[in_train_idx])

            # Extract raw probabilities
            p_val = in_pipeline.predict_proba(X_train.iloc[in_val_idx])[:, 1]
            inner_oof_probs[in_val_idx] = p_val

            # Extract score for sigmoid fitting (log-odds / decision_function)
            if hasattr(in_pipeline.named_steps["classifier"], "decision_function"):
                s_val = in_pipeline.decision_function(X_train.iloc[in_val_idx])
            else:
                s_val = logit(np.clip(p_val, 1e-6, 1.0 - 1e-6))
            inner_oof_scores[in_val_idx] = s_val

        assert not np.isnan(inner_oof_probs).any(), "NaN found in inner OOF probabilities!"

        # -------------------------------------------------------------------
        # Fit Calibrators on Inner Out-of-Fold Validation Data
        # -------------------------------------------------------------------
        # 1. Sigmoid / Platt
        plat_cal = PlattCalibrator()
        plat_cal.fit(inner_oof_scores, y_train.to_numpy())

        # 2. Isotonic (clipped strictly to [0, 1])
        iso_cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_cal.fit(inner_oof_probs, y_train.to_numpy())

        # -------------------------------------------------------------------
        # Refit Base Estimator on Complete Outer Training Set
        # -------------------------------------------------------------------
        outer_pipeline = pipeline_builder()
        outer_pipeline.fit(X_train, y_train)

        # Audit Score Distribution Mismatch
        if hasattr(outer_pipeline.named_steps["classifier"], "decision_function"):
            outer_train_scores = outer_pipeline.decision_function(X_train)
            outer_test_scores = outer_pipeline.decision_function(X_test)
        else:
            p_train_refit = outer_pipeline.predict_proba(X_train)[:, 1]
            outer_train_scores = logit(np.clip(p_train_refit, 1e-6, 1.0 - 1e-6))
            p_test_raw_temp = outer_pipeline.predict_proba(X_test)[:, 1]
            outer_test_scores = logit(np.clip(p_test_raw_temp, 1e-6, 1.0 - 1e-6))

        # KS test to check score distribution shift between inner OOF and outer refit
        ks_stat, ks_pval = ks_2samp(inner_oof_scores, outer_train_scores)
        ks_pvalues.append(ks_pval)
        inner_scores_all.extend(inner_oof_scores.tolist())
        outer_scores_all.extend(outer_train_scores.tolist())

        # -------------------------------------------------------------------
        # Generate Predictions on Completely Untouched Outer Test Set
        # -------------------------------------------------------------------
        raw_test_prob = outer_pipeline.predict_proba(X_test)[:, 1]
        sig_test_prob = plat_cal.predict_proba(outer_test_scores)
        iso_test_prob = iso_cal.predict(raw_test_prob)

        pred_dict = {
            "uncalibrated": raw_test_prob,
            "sigmoid": sig_test_prob,
            "isotonic": iso_test_prob,
        }

        # 4. Prevalence adjustment for Logistic Regression
        if supports_prevalence:
            # Empirical class count in outer training fold
            n_pos_tr = np.sum(y_train == 1)
            n_neg_tr = np.sum(y_train == 0)
            # class_weight='balanced' offsets the logit by ln(n_neg_tr / n_pos_tr)
            logit_shift = math.log(n_neg_tr / n_pos_tr)
            prev_adj_test_prob = expit(outer_test_scores - logit_shift)
            pred_dict["prevalence_adjusted"] = prev_adj_test_prob

        for i, global_idx in enumerate(test_idx):
            row_dict = {
                "model_type": model_type,
                "scheme": scheme,
                "seed": seed,
                "fold": fold_idx,
                "row_index": int(global_idx),
                "name": str(names.iloc[global_idx]),
                "subject_id": str(groups.iloc[global_idx]),
                "y_true": int(y.iloc[global_idx]),
            }
            for m in methods:
                row_dict[f"prob_{m}"] = round(float(pred_dict[m][i]), 6)
            recording_records.append(row_dict)

        # Fold-level recording metrics
        fold_row = {
            "model_type": model_type,
            "scheme": scheme,
            "seed": seed,
            "fold": fold_idx,
            "test_subjects": ",".join(sorted(list(test_subjects))),
            "n_test_recordings": len(test_idx),
            "ks_pvalue_mismatch": round(float(ks_pval), 4),
        }
        for m in methods:
            m_metrics = compute_comprehensive_metrics(y_test.to_numpy(), pred_dict[m])
            for k_met, v_met in m_metrics.items():
                fold_row[f"{m}_{k_met}"] = v_met
        fold_metrics.append(fold_row)

    oof_recordings_df = pd.DataFrame(recording_records)
    fold_metrics_df = pd.DataFrame(fold_metrics)

    # -----------------------------------------------------------------------
    # Participant-Level Aggregation (Mean probability across recordings)
    # -----------------------------------------------------------------------
    participant_rows = []
    for (subj, grp) in oof_recordings_df.groupby(["subject_id"]):
        y_true_sub = int(grp["y_true"].iloc[0])
        n_recs = len(grp)
        part_row = {
            "model_type": model_type,
            "scheme": scheme,
            "seed": seed,
            "subject_id": subj[0] if isinstance(subj, tuple) else subj,
            "true_status": y_true_sub,
            "n_recordings": n_recs,
        }
        for m in methods:
            mean_p = float(grp[f"prob_{m}"].mean())
            part_row[f"prob_{m}"] = round(mean_p, 6)
            part_row[f"pred_{m}"] = int(mean_p >= DECISION_THRESHOLD)
            part_row[f"correct_{m}"] = int(part_row[f"pred_{m}"] == y_true_sub)
        participant_rows.append(part_row)

    oof_participants_df = pd.DataFrame(participant_rows)

    mismatch_stats = {
        "inner_scores_mean": float(np.mean(inner_scores_all)),
        "inner_scores_std": float(np.std(inner_scores_all)),
        "outer_scores_mean": float(np.mean(outer_scores_all)),
        "outer_scores_std": float(np.std(outer_scores_all)),
        "mean_ks_pvalue": float(np.mean(ks_pvalues)),
        "min_ks_pvalue": float(np.min(ks_pvalues)),
    }

    return oof_recordings_df, oof_participants_df, fold_metrics_df, mismatch_stats


# ---------------------------------------------------------------------------
# Reliability Diagram Plotting
# ---------------------------------------------------------------------------
def plot_reliability_diagrams(
    oof_rec_df: pd.DataFrame,
    oof_part_df: pd.DataFrame,
    model_type: str,
    scheme: str,
    save_path: Path
) -> None:
    """Generates 2x2 multi-panel calibration curves & probability histograms."""
    methods = [c.replace("prob_", "") for c in oof_rec_df.columns if c.startswith("prob_")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {
        "uncalibrated": "#1f77b4",
        "sigmoid": "#2ca02c",
        "isotonic": "#d62728",
        "prevalence_adjusted": "#9467bd"
    }

    # Top-Left: Recording-level calibration curves
    ax_rec = axes[0, 0]
    ax_rec.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    for m in methods:
        prob_true, prob_pred = calibration_curve(
            oof_rec_df["y_true"], oof_rec_df[f"prob_{m}"], n_bins=5, strategy="uniform"
        )
        brier = brier_score_loss(oof_rec_df["y_true"], oof_rec_df[f"prob_{m}"])
        ece, _, _ = compute_ece(oof_rec_df["y_true"], oof_rec_df[f"prob_{m}"], n_bins=5, strategy="uniform")
        ax_rec.plot(prob_pred, prob_true, marker="o", lw=2, color=colors.get(m, "blue"),
                    label=f"{m} (Brier={brier:.3f}, ECE={ece:.3f})")
    ax_rec.set_title(f"Recording Level: Calibration Curves ({model_type}, {scheme})")
    ax_rec.set_xlabel("Mean Predicted Probability")
    ax_rec.set_ylabel("Fraction of Positives")
    ax_rec.set_xlim([0.0, 1.0])
    ax_rec.set_ylim([0.0, 1.0])
    ax_rec.legend(loc="lower right")
    ax_rec.grid(True, alpha=0.3)

    # Top-Right: Recording-level probability histograms
    ax_hist = axes[0, 1]
    for m in methods:
        ax_hist.hist(oof_rec_df[f"prob_{m}"], bins=15, range=(0, 1), histtype="step", lw=2,
                     color=colors.get(m, "blue"), label=m, density=True)
    ax_hist.set_title("Recording Level: Probability Distribution")
    ax_hist.set_xlabel("Predicted Probability")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc="upper center")
    ax_hist.grid(True, alpha=0.3)

    # Bottom-Left: Participant-level calibration curves
    ax_part = axes[1, 0]
    ax_part.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    for m in methods:
        prob_true, prob_pred = calibration_curve(
            oof_part_df["true_status"], oof_part_df[f"prob_{m}"], n_bins=5, strategy="uniform"
        )
        brier = brier_score_loss(oof_part_df["true_status"], oof_part_df[f"prob_{m}"])
        ece, _, _ = compute_ece(oof_part_df["true_status"], oof_part_df[f"prob_{m}"], n_bins=5, strategy="uniform")
        ax_part.plot(prob_pred, prob_true, marker="s", lw=2, color=colors.get(m, "blue"),
                     label=f"{m} (Brier={brier:.3f}, ECE={ece:.3f})")
    ax_part.set_title(f"Participant Level: Calibration Curves ({model_type}, {scheme})")
    ax_part.set_xlabel("Mean Subject Predicted Probability")
    ax_part.set_ylabel("Fraction of Positives")
    ax_part.set_xlim([0.0, 1.0])
    ax_part.set_ylim([0.0, 1.0])
    ax_part.legend(loc="lower right")
    ax_part.grid(True, alpha=0.3)

    # Bottom-Right: Participant-level probability histograms
    ax_part_hist = axes[1, 1]
    for m in methods:
        ax_part_hist.hist(oof_part_df[f"prob_{m}"], bins=15, range=(0, 1), histtype="step", lw=2,
                          color=colors.get(m, "blue"), label=m, density=True)
    ax_part_hist.set_title("Participant Level: Subject Probability Distribution")
    ax_part_hist.set_xlabel("Mean Subject Predicted Probability")
    ax_part_hist.set_ylabel("Density")
    ax_part_hist.legend(loc="upper center")
    ax_part_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Reliability diagram saved to {save_path}")


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    X, y, groups, names = load_dataset()
    logger.info(f"Loaded dataset: {len(X)} recordings across {groups.nunique()} participants.")

    all_summary_rows = []
    all_oof_recs = []
    all_oof_parts = []
    all_fold_metrics = []
    mismatch_dict = {}

    # Outer configurations
    configurations = [
        ("baseline_lr_22", "LOGO", [42]),
        ("baseline_lr_22", "SGKF", SEEDS_SGKF),
        ("research_et_13", "LOGO", [42]),
        ("research_et_13", "SGKF", SEEDS_SGKF),
    ]

    for model_type, scheme, seeds in configurations:
        for seed in seeds:
            config_name = f"{model_type}_{scheme}_seed{seed}" if scheme == "SGKF" else f"{model_type}_{scheme}"
            logger.info(f"=== Running Configuration: {config_name} ===")

            oof_recs, oof_parts, fold_mets, mismatch_stats = run_nested_calibration_cv(
                X=X, y=y, groups=groups, names=names, model_type=model_type, scheme=scheme, seed=seed
            )

            oof_recs["config"] = config_name
            oof_parts["config"] = config_name
            fold_mets["config"] = config_name

            all_oof_recs.append(oof_recs)
            all_oof_parts.append(oof_parts)
            all_fold_metrics.append(fold_mets)
            mismatch_dict[config_name] = mismatch_stats

            # Generate plots for each LOGO run
            if scheme == "LOGO":
                plot_path = RESULTS_DIR / f"reliability_diagram_{model_type}.png"
                plot_reliability_diagrams(oof_recs, oof_parts, model_type, scheme, plot_path)

            # Compute summary metrics for each calibration method
            methods = [c.replace("prob_", "") for c in oof_recs.columns if c.startswith("prob_")]
            for m in methods:
                # Recording level
                rec_m = compute_comprehensive_metrics(oof_recs["y_true"], oof_recs[f"prob_{m}"])
                row_rec = {
                    "config": config_name,
                    "model_type": model_type,
                    "scheme": scheme,
                    "seed": seed,
                    "evaluation_level": "recording",
                    "calibration_method": m,
                    **rec_m
                }
                all_summary_rows.append(row_rec)

                # Participant level
                part_m = compute_comprehensive_metrics(oof_parts["true_status"], oof_parts[f"prob_{m}"])
                row_part = {
                    "config": config_name,
                    "model_type": model_type,
                    "scheme": scheme,
                    "seed": seed,
                    "evaluation_level": "participant",
                    "calibration_method": m,
                    **part_m
                }
                all_summary_rows.append(row_part)

    # Combine dataframes
    summary_df = pd.DataFrame(all_summary_rows)
    master_oof_recs_df = pd.concat(all_oof_recs, ignore_index=True)
    master_oof_parts_df = pd.concat(all_oof_parts, ignore_index=True)
    master_fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)

    # Save to disk
    summary_path = RESULTS_DIR / "calibration_metrics_summary.csv"
    oof_rec_path = RESULTS_DIR / "oof_calibrated_predictions_recordings.csv"
    oof_part_path = RESULTS_DIR / "participant_calibrated_predictions.csv"
    fold_met_path = RESULTS_DIR / "fold_level_metrics.csv"
    mismatch_path = RESULTS_DIR / "score_mismatch_audit.json"

    summary_df.to_csv(summary_path, index=False)
    master_oof_recs_df.to_csv(oof_rec_path, index=False)
    master_oof_parts_df.to_csv(oof_part_path, index=False)
    master_fold_metrics_df.to_csv(fold_met_path, index=False)

    with open(mismatch_path, "w") as f:
        json.dump(mismatch_dict, f, indent=2)

    logger.info("Successfully executed Phase 2.2C calibration experiment.")
    logger.info(f"Artifacts saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
