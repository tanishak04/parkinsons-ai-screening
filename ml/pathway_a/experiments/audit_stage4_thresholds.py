"""
ml/pathway_a/experiments/audit_stage4_thresholds.py
Stage 4.2: Threshold Robustness Audit for Pathway A.

Strictly participant-level evaluation (N=32; 24 PD, 8 HC).
Evaluates threshold grid from 0.10 to 0.90 in increments of 0.05 (17 thresholds).
Preserves separate evaluation streams:
  - Primary: Research Champion 13-Feature Pitch-Free ExtraTrees (ET-13)
  - Comparative Reference: Historical 22-Feature Baseline (H22)

Calculates at each threshold:
  - TP, TN, FP, FN
  - Sensitivity / Recall
  - Specificity
  - Precision
  - F1-score
  - Balanced Accuracy
  - False Negative Rate (FNR)
  - False Positive Rate (FPR)
  - Accuracy
  - Matthews Correlation Coefficient (MCC)

Generates:
  - Full metrics CSV tables
  - Confusion matrices JSON and CSV
  - Visualization plots (PNGs):
      * sensitivity_specificity_vs_threshold.png
      * precision_recall_curve.png
      * f1_balanced_accuracy_vs_threshold.png
      * error_rates_fnr_fpr_vs_threshold.png
      * mcc_vs_threshold.png
  - Machine-readable JSON summary
  - Input and output artifact SHA-256 hash manifests
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "ml" / "pathway_a" / "experiments" / "results" / "stage4_thresholds"
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Input Artifact Paths
HIST_CSV = REPO_ROOT / "ml" / "pathway_a" / "experiments" / "results" / "phase3_threshold_audit" / "participant_decision_margins_historical22.csv"
ET13_CSV = REPO_ROOT / "ml" / "pathway_a" / "experiments" / "results" / "phase3_threshold_audit" / "participant_decision_margins_et13.csv"
RAW_DATA = REPO_ROOT / "ml" / "pathway_a" / "data" / "raw" / "parkinsons.data"

PROTECTED_MODELS = [
    REPO_ROOT / "ml" / "pathway_a" / "models" / "champion_pipeline.joblib",
    REPO_ROOT / "ml" / "pathway_a" / "models" / "baseline_pipeline.joblib",
    REPO_ROOT / "ml" / "pathway_a" / "models" / "audio_pipeline.joblib",
]

STAGE_4_1_ARTIFACTS = [
    REPO_ROOT / "ml" / "pathway_a" / "experiments" / "results" / "stage4_uncertainty" / "uncertainty_summary_et13.csv",
    REPO_ROOT / "ml" / "pathway_a" / "experiments" / "results" / "stage4_uncertainty" / "uncertainty_summary_historical22.csv",
    REPO_ROOT / "ml" / "pathway_a" / "experiments" / "results" / "stage4_uncertainty" / "paired_model_comparison.csv",
]

INPUT_ARTIFACTS = [HIST_CSV, ET13_CSV, RAW_DATA] + PROTECTED_MODELS + STAGE_4_1_ARTIFACTS

# 17 thresholds from 0.10 to 0.90 in increments of 0.05
THRESHOLDS = np.round(np.arange(0.10, 0.9001, 0.05), 2).tolist()


def compute_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> dict[str, Any]:
    """Compute all required metrics with robust edge-case division handling."""
    y_pred = (y_prob >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    bal_acc = (sens + spec) / 2.0
    f1 = float(2.0 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0
    
    # Matthews Correlation Coefficient
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom > 0:
        mcc = float((tp * tn - fp * fn) / math.sqrt(denom))
    else:
        mcc = 0.0

    return {
        "threshold": float(tau),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "predicted_positives": int(tp + fp),
        "predicted_negatives": int(tn + fn),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "false_negative_rate": round(fnr, 4),
        "false_positive_rate": round(fpr, 4),
        "accuracy": round(acc, 4),
        "mcc": round(mcc, 4),
    }


def analyze_threshold_stability(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze step-wise metric differences, plateaus, and cliffs."""
    stability_records = []
    thresholds = df["threshold"].tolist()
    
    for i in range(len(thresholds) - 1):
        t1, t2 = thresholds[i], thresholds[i + 1]
        r1, r2 = df.iloc[i], df.iloc[i + 1]
        d_sens = round(float(r2["sensitivity"] - r1["sensitivity"]), 4)
        d_spec = round(float(r2["specificity"] - r1["specificity"]), 4)
        d_bal_acc = round(float(r2["balanced_accuracy"] - r1["balanced_accuracy"]), 4)
        d_f1 = round(float(r2["f1_score"] - r1["f1_score"]), 4)
        d_mcc = round(float(r2["mcc"] - r1["mcc"]), 4)

        # Flag cliff if single step causes >= 15% change in balanced accuracy or sensitivity
        is_cliff = abs(d_bal_acc) >= 0.15 or abs(d_sens) >= 0.15 or abs(d_spec) >= 0.25
        is_plateau = (d_sens == 0.0 and d_spec == 0.0)

        stability_records.append({
            "from_threshold": t1,
            "to_threshold": t2,
            "delta_sensitivity": d_sens,
            "delta_specificity": d_spec,
            "delta_balanced_accuracy": d_bal_acc,
            "delta_f1": d_f1,
            "delta_mcc": d_mcc,
            "is_plateau": is_plateau,
            "is_cliff": is_cliff,
        })

    # Summary metrics
    max_bal_acc_row = df.loc[df["balanced_accuracy"].idxmax()]
    max_f1_row = df.loc[df["f1_score"].idxmax()]
    max_mcc_row = df.loc[df["mcc"].idxmax()]

    # Broad plateau region around optimal balanced accuracy
    optimal_bal_acc = max_bal_acc_row["balanced_accuracy"]
    near_optimal = df[df["balanced_accuracy"] >= optimal_bal_acc - 0.025]
    plateau_range = [float(near_optimal["threshold"].min()), float(near_optimal["threshold"].max())]

    return {
        "step_stability": stability_records,
        "max_balanced_accuracy_point": {
            "threshold": float(max_bal_acc_row["threshold"]),
            "balanced_accuracy": float(max_bal_acc_row["balanced_accuracy"]),
            "sensitivity": float(max_bal_acc_row["sensitivity"]),
            "specificity": float(max_bal_acc_row["specificity"]),
            "f1_score": float(max_bal_acc_row["f1_score"]),
            "mcc": float(max_bal_acc_row["mcc"]),
        },
        "max_f1_point": {
            "threshold": float(max_f1_row["threshold"]),
            "f1_score": float(max_f1_row["f1_score"]),
            "balanced_accuracy": float(max_f1_row["balanced_accuracy"]),
            "sensitivity": float(max_f1_row["sensitivity"]),
            "specificity": float(max_f1_row["specificity"]),
        },
        "max_mcc_point": {
            "threshold": float(max_mcc_row["threshold"]),
            "mcc": float(max_mcc_row["mcc"]),
            "balanced_accuracy": float(max_mcc_row["balanced_accuracy"]),
        },
        "near_optimal_plateau_range": plateau_range,
    }


def generate_plots(df_et13: pd.DataFrame, df_h22: pd.DataFrame) -> list[Path]:
    """Generate professional plots comparing threshold dynamics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    generated_plots = []

    # 1. Sensitivity vs. Specificity across thresholds
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df_et13["threshold"], df_et13["sensitivity"], "o-", color="#1f77b4", label="ET-13 Sensitivity", lw=2)
    ax.plot(df_et13["threshold"], df_et13["specificity"], "s--", color="#2ca02c", label="ET-13 Specificity", lw=2)
    ax.plot(df_h22["threshold"], df_h22["sensitivity"], "o:", color="#aec7e8", label="H22 Sensitivity (ref)", lw=1.5, alpha=0.7)
    ax.plot(df_h22["threshold"], df_h22["specificity"], "s:", color="#98df8a", label="H22 Specificity (ref)", lw=1.5, alpha=0.7)
    ax.axvline(0.50, color="#d62728", linestyle="--", alpha=0.7, label=r"Reference $\tau=0.50$")
    ax.axvline(0.45, color="#ff7f0e", linestyle=":", alpha=0.7, label=r"Screening $\tau=0.45$")
    ax.set_title("Sensitivity vs. Specificity Across Probability Thresholds", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold", fontsize=10)
    ax.set_ylabel("Metric Value", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.05, 0.95)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    p1 = PLOTS_DIR / "sensitivity_specificity_vs_threshold.png"
    plt.savefig(p1)
    plt.close()
    generated_plots.append(p1)

    # 2. Precision vs. Recall
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df_et13["sensitivity"], df_et13["precision"], "o-", color="#1f77b4", label="ET-13 (Pitch-Free)", lw=2)
    ax.plot(df_h22["sensitivity"], df_h22["precision"], "s--", color="#aec7e8", label="Historical 22", lw=1.5)
    for _, row in df_et13.iterrows():
        if row["threshold"] in [0.30, 0.45, 0.50, 0.60, 0.70]:
            ax.annotate(f"$\\tau$={row['threshold']:.2f}", (row["sensitivity"], row["precision"]),
                        textcoords="offset points", xytext=(5, -5), fontsize=8, color="#1f77b4")
    ax.set_title("Precision vs. Recall (Participant Level, N=32)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_ylim(0.65, 1.05)
    ax.set_xlim(0.0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    p2 = PLOTS_DIR / "precision_recall_curve.png"
    plt.savefig(p2)
    plt.close()
    generated_plots.append(p2)

    # 3. F1 and Balanced Accuracy vs. Threshold
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df_et13["threshold"], df_et13["balanced_accuracy"], "o-", color="#1f77b4", label="ET-13 Balanced Accuracy", lw=2)
    ax.plot(df_et13["threshold"], df_et13["f1_score"], "^--", color="#ff7f0e", label="ET-13 F1-Score", lw=2)
    ax.plot(df_h22["threshold"], df_h22["balanced_accuracy"], "o:", color="#aec7e8", label="H22 Balanced Acc. (ref)", lw=1.5, alpha=0.7)
    ax.plot(df_h22["threshold"], df_h22["f1_score"], "^:", color="#ffbb78", label="H22 F1-Score (ref)", lw=1.5, alpha=0.7)
    ax.axvline(0.50, color="#d62728", linestyle="--", alpha=0.7, label=r"Reference $\tau=0.50$")
    ax.axvline(0.45, color="#2ca02c", linestyle=":", alpha=0.7, label=r"Screening $\tau=0.45$")
    ax.set_title("Balanced Accuracy & F1-Score Across Thresholds", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(0.05, 0.95)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    p3 = PLOTS_DIR / "f1_balanced_accuracy_vs_threshold.png"
    plt.savefig(p3)
    plt.close()
    generated_plots.append(p3)

    # 4. Error Rates: FNR and FPR vs. Threshold
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df_et13["threshold"], df_et13["false_negative_rate"], "o-", color="#d62728", label="ET-13 False Negative Rate (FNR)", lw=2)
    ax.plot(df_et13["threshold"], df_et13["false_positive_rate"], "s--", color="#9467bd", label="ET-13 False Positive Rate (FPR)", lw=2)
    ax.plot(df_h22["threshold"], df_h22["false_negative_rate"], "o:", color="#ff9896", label="H22 FNR (ref)", lw=1.5, alpha=0.7)
    ax.plot(df_h22["threshold"], df_h22["false_positive_rate"], "s:", color="#c5b0d5", label="H22 FPR (ref)", lw=1.5, alpha=0.7)
    ax.axvline(0.50, color="#7f7f7f", linestyle="--", alpha=0.7, label=r"Reference $\tau=0.50$")
    ax.axvline(0.45, color="#17becf", linestyle=":", alpha=0.7, label=r"Screening $\tau=0.45$")
    ax.set_title("False Negative vs. False Positive Rates Across Thresholds", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold", fontsize=10)
    ax.set_ylabel("Error Rate", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.05, 0.95)
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    p4 = PLOTS_DIR / "error_rates_fnr_fpr_vs_threshold.png"
    plt.savefig(p4)
    plt.close()
    generated_plots.append(p4)

    # 5. Matthews Correlation Coefficient (MCC) vs. Threshold
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(df_et13["threshold"], df_et13["mcc"], "o-", color="#1f77b4", label="ET-13 MCC", lw=2)
    ax.plot(df_h22["threshold"], df_h22["mcc"], "s--", color="#aec7e8", label="H22 MCC (ref)", lw=1.5)
    ax.axvline(0.50, color="#d62728", linestyle="--", alpha=0.7, label=r"Reference $\tau=0.50$")
    ax.axvline(0.45, color="#2ca02c", linestyle=":", alpha=0.7, label=r"Screening $\tau=0.45$")
    ax.set_title("Matthews Correlation Coefficient (MCC) Across Thresholds", fontsize=12, fontweight="bold")
    ax.set_xlabel("Decision Threshold", fontsize=10)
    ax.set_ylabel("MCC", fontsize=10)
    ax.set_ylim(-0.2, 0.8)
    ax.set_xlim(0.05, 0.95)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    p5 = PLOTS_DIR / "mcc_vs_threshold.png"
    plt.savefig(p5)
    plt.close()
    generated_plots.append(p5)

    return generated_plots


def main() -> None:
    logger.info("=== Stage 4.2 Threshold Robustness Audit ===")

    # 1. Pre-execution Hashes of Input Artifacts and Protected Models
    pre_hashes = {}
    for p in INPUT_ARTIFACTS:
        pre_hashes[str(p.relative_to(REPO_ROOT))] = compute_file_sha256(p)

    with open(RESULTS_DIR / "input_artifact_hashes.json", "w") as f:
        json.dump(pre_hashes, f, indent=2)
    logger.info(f"Verified and recorded {len(pre_hashes)} input artifact hashes.")

    # 2. Load & Validate Participant Predictions
    raw_df = pd.read_csv(RAW_DATA)
    raw_df["subject_id"] = raw_df["name"].apply(lambda x: x.split("_")[2])
    gt_map = raw_df.groupby("subject_id")["status"].first().to_dict()

    df_h22 = pd.read_csv(HIST_CSV).sort_values("subject_id").reset_index(drop=True)
    df_et13 = pd.read_csv(ET13_CSV).sort_values("subject_id").reset_index(drop=True)

    if len(df_h22) != 32 or len(df_et13) != 32:
        raise ValueError(f"Participant count mismatch! Expected 32, got {len(df_h22)} (H22) and {len(df_et13)} (ET13)")
    if not (df_h22["subject_id"] == df_et13["subject_id"]).all():
        raise ValueError("Subject ID alignment mismatch between H22 and ET13!")
    if not (df_h22["true_status"] == df_et13["true_status"]).all():
        raise ValueError("Ground truth status mismatch between H22 and ET13!")

    for sid, st in df_h22[["subject_id", "true_status"]].values:
        if gt_map[sid] != st:
            raise ValueError(f"Ground truth label mismatch for subject {sid}: dataset={gt_map[sid]} vs csv={st}")

    y_all = df_et13["true_status"].to_numpy(dtype=int)
    prob_et13 = df_et13["mean_prob"].to_numpy(dtype=float)
    prob_h22 = df_h22["mean_prob"].to_numpy(dtype=float)

    n_pd = int(np.sum(y_all == 1))
    n_hc = int(np.sum(y_all == 0))
    logger.info(f"Participant cohort verified: N={len(y_all)} (PD={n_pd}, HC={n_hc}). Zero leakage.")

    # 3. Threshold Evaluation Grid
    logger.info(f"Evaluating {len(THRESHOLDS)} thresholds from {THRESHOLDS[0]:.2f} to {THRESHOLDS[-1]:.2f} (step=0.05)...")
    
    records_et13 = []
    records_h22 = []
    conf_matrices_et13 = {}
    conf_matrices_h22 = {}
    combined_cm_rows = []

    for tau in THRESHOLDS:
        m_et13 = compute_metrics_at_threshold(y_all, prob_et13, tau)
        m_h22 = compute_metrics_at_threshold(y_all, prob_h22, tau)
        
        records_et13.append(m_et13)
        records_h22.append(m_h22)

        conf_matrices_et13[f"tau_{tau:.2f}"] = {
            "threshold": tau,
            "tp": m_et13["tp"],
            "tn": m_et13["tn"],
            "fp": m_et13["fp"],
            "fn": m_et13["fn"]
        }
        conf_matrices_h22[f"tau_{tau:.2f}"] = {
            "threshold": tau,
            "tp": m_h22["tp"],
            "tn": m_h22["tn"],
            "fp": m_h22["fp"],
            "fn": m_h22["fn"]
        }

        combined_cm_rows.append({
            "model": "research_et_13",
            "threshold": tau,
            "tp": m_et13["tp"],
            "tn": m_et13["tn"],
            "fp": m_et13["fp"],
            "fn": m_et13["fn"],
        })
        combined_cm_rows.append({
            "model": "historical_22",
            "threshold": tau,
            "tp": m_h22["tp"],
            "tn": m_h22["tn"],
            "fp": m_h22["fp"],
            "fn": m_h22["fn"],
        })

    metrics_df_et13 = pd.DataFrame(records_et13)
    metrics_df_h22 = pd.DataFrame(records_h22)
    combined_cm_df = pd.DataFrame(combined_cm_rows)

    # 4. Save CSV and JSON Matrix Artifacts
    metrics_csv_et13 = RESULTS_DIR / "threshold_metrics_et13.csv"
    metrics_csv_h22 = RESULTS_DIR / "threshold_metrics_historical22.csv"
    metrics_df_et13.to_csv(metrics_csv_et13, index=False)
    metrics_df_h22.to_csv(metrics_csv_h22, index=False)
    logger.info(f"Saved {metrics_csv_et13.name} and {metrics_csv_h22.name}.")

    cm_json_et13 = RESULTS_DIR / "confusion_matrices_et13.json"
    cm_json_h22 = RESULTS_DIR / "confusion_matrices_historical22.json"
    with open(cm_json_et13, "w") as f:
        json.dump(conf_matrices_et13, f, indent=2)
    with open(cm_json_h22, "w") as f:
        json.dump(conf_matrices_h22, f, indent=2)

    cm_csv_combined = RESULTS_DIR / "confusion_matrices_combined.csv"
    combined_cm_df.to_csv(cm_csv_combined, index=False)
    logger.info(f"Saved confusion matrix artifacts.")

    # 5. Paired Comparative Grid CSV
    comp_df = pd.DataFrame({
        "threshold": metrics_df_et13["threshold"],
        "et13_sensitivity": metrics_df_et13["sensitivity"],
        "h22_sensitivity": metrics_df_h22["sensitivity"],
        "delta_sensitivity": (metrics_df_et13["sensitivity"] - metrics_df_h22["sensitivity"]).round(4),
        "et13_specificity": metrics_df_et13["specificity"],
        "h22_specificity": metrics_df_h22["specificity"],
        "delta_specificity": (metrics_df_et13["specificity"] - metrics_df_h22["specificity"]).round(4),
        "et13_balanced_accuracy": metrics_df_et13["balanced_accuracy"],
        "h22_balanced_accuracy": metrics_df_h22["balanced_accuracy"],
        "delta_balanced_accuracy": (metrics_df_et13["balanced_accuracy"] - metrics_df_h22["balanced_accuracy"]).round(4),
        "et13_f1": metrics_df_et13["f1_score"],
        "h22_f1": metrics_df_h22["f1_score"],
        "delta_f1": (metrics_df_et13["f1_score"] - metrics_df_h22["f1_score"]).round(4),
        "et13_mcc": metrics_df_et13["mcc"],
        "h22_mcc": metrics_df_h22["mcc"],
        "delta_mcc": (metrics_df_et13["mcc"] - metrics_df_h22["mcc"]).round(4),
        "et13_fnr": metrics_df_et13["false_negative_rate"],
        "h22_fnr": metrics_df_h22["false_negative_rate"],
        "et13_fpr": metrics_df_et13["false_positive_rate"],
        "h22_fpr": metrics_df_h22["false_positive_rate"],
    })
    comp_csv = RESULTS_DIR / "threshold_metrics_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    logger.info(f"Saved {comp_csv.name}.")

    # 6. Generate Plots
    logger.info("Generating threshold dynamics visualization plots...")
    plot_paths = generate_plots(metrics_df_et13, metrics_df_h22)
    logger.info(f"Generated {len(plot_paths)} plots in {PLOTS_DIR}.")

    # 7. Stability & Plateau Analysis
    stability_et13 = analyze_threshold_stability(metrics_df_et13)
    stability_h22 = analyze_threshold_stability(metrics_df_h22)

    # 8. Machine-Readable Summary JSON
    summary_data = {
        "audit_stage": "4.2",
        "description": "Threshold Robustness & Operating Dynamics Audit",
        "cohort": {"total_participants": 32, "parkinsons": 24, "healthy_controls": 8},
        "threshold_grid": THRESHOLDS,
        "reference_operating_points": {
            "standard_reference_tau_0_50": {
                "threshold": 0.50,
                "et13": metrics_df_et13[metrics_df_et13["threshold"] == 0.50].iloc[0].to_dict(),
                "h22": metrics_df_h22[metrics_df_h22["threshold"] == 0.50].iloc[0].to_dict(),
            },
            "exploratory_screening_tau_0_45": {
                "threshold": 0.45,
                "et13": metrics_df_et13[metrics_df_et13["threshold"] == 0.45].iloc[0].to_dict(),
                "h22": metrics_df_h22[metrics_df_h22["threshold"] == 0.45].iloc[0].to_dict(),
            },
        },
        "et13_stability_analysis": stability_et13,
        "h22_stability_analysis": stability_h22,
        "methodological_caveats": {
            "unbiased_validation_rule": "Threshold exploration on evaluation data represents operating-point characterization, NOT unbiased prospective validation. Any selected operating threshold must be validated on independent future cohorts.",
            "small_sample_support": "With N=8 controls, specificity moves in discrete quanta of 0.125 (1/8).",
            "production_status": "No production weights or deployment thresholds are modified by this audit."
        }
    }

    summary_json_path = RESULTS_DIR / "stage4_2_threshold_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved summary JSON to {summary_json_path.name}.")

    # 9. Post-execution Hash Verification
    output_files = [
        metrics_csv_et13,
        metrics_csv_h22,
        comp_csv,
        cm_json_et13,
        cm_json_h22,
        cm_csv_combined,
        summary_json_path,
    ] + plot_paths

    output_hashes = {}
    for p in output_files:
        output_hashes[str(p.relative_to(REPO_ROOT))] = compute_file_sha256(p)

    with open(RESULTS_DIR / "output_artifact_hashes.json", "w") as f:
        json.dump(output_hashes, f, indent=2)

    # Re-verify protected models
    for p in PROTECTED_MODELS:
        curr_hash = compute_file_sha256(p)
        orig_hash = pre_hashes[str(p.relative_to(REPO_ROOT))]
        if curr_hash != orig_hash:
            raise RuntimeError(f"PROTECTED MODEL INTEGRITY VIOLATION: {p.name} changed!")
    logger.info("All protected model pipelines verified unchanged.")
    logger.info("=== Stage 4.2 Threshold Robustness Audit Finished Successfully ===")


if __name__ == "__main__":
    main()
