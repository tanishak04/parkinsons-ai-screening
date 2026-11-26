"""
ml/pathway_a/experiments/audit_stage4_uncertainty.py
Stage 4.1: Statistical Uncertainty Audit for Pathway A.

Strictly participant-level evaluation (N=32; 24 PD, 8 HC).
Evaluates:
  - Stream A: Historical 22-Feature ExtraTrees Model
  - Stream B: Research Champion 13-Feature Pitch-Free ExtraTrees Model
  - Operating thresholds: tau = 0.50 (reference) and tau = 0.45 (exploratory screening)
  - Exact Clopper-Pearson & Wilson Score 95% CIs for proportions
  - Stratified Participant Bootstrap (B=2,000, seed=42)
  - Paired Model Comparisons (Delta ET13 - H22, Exact McNemar test, DeLong AUC test)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, norm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "ml" / "pathway_a" / "experiments" / "results" / "stage4_uncertainty"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Input Artifact Paths
HIST_CSV = REPO_ROOT / "ml" / "pathway_a" / "experiments" / "results" / "phase3_threshold_audit" / "participant_decision_margins_historical22.csv"
ET13_CSV = REPO_ROOT / "ml" / "pathway_a" / "experiments" / "results" / "phase3_threshold_audit" / "participant_decision_margins_et13.csv"
RAW_DATA = REPO_ROOT / "ml" / "pathway_a" / "data" / "raw" / "parkinsons.data"

PROTECTED_MODELS = [
    REPO_ROOT / "ml" / "pathway_a" / "models" / "champion_pipeline.joblib",
    REPO_ROOT / "ml" / "pathway_a" / "models" / "baseline_pipeline.joblib",
    REPO_ROOT / "ml" / "pathway_a" / "models" / "audio_pipeline.joblib",
]

INPUT_ARTIFACTS = [HIST_CSV, ET13_CSV, RAW_DATA] + PROTECTED_MODELS

N_BOOTSTRAP = 2000
RANDOM_SEED = 42
THRESHOLDS = [0.50, 0.45]


# ---------------------------------------------------------------------------
# Exact Binomial Intervals (Clopper-Pearson & Wilson Score)
# ---------------------------------------------------------------------------
def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson 1-alpha confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def wilson_score_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p_hat = k / n
    z = float(norm.ppf(1.0 - alpha / 2.0))
    denominator = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denominator
    spread = (z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * n**2))) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


# ---------------------------------------------------------------------------
# DeLong Test for Comparing Two Correlated ROC Curves
# ---------------------------------------------------------------------------
def compute_delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict[str, Any]:
    """
    Computes DeLong test for paired ROC curves on matched subjects.
    y_true: binary labels (0, 1)
    prob_a: predicted probabilities model A
    prob_b: predicted probabilities model B
    """
    y_true = np.asarray(y_true, dtype=int)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m = len(pos_idx)
    n = len(neg_idx)

    if m == 0 or n == 0:
        return {"error": "Degenerate class distribution"}

    def structural_components(prob):
        V10 = np.zeros(m)
        V01 = np.zeros(n)
        for i in range(m):
            V10[i] = np.mean((prob[pos_idx[i]] > prob[neg_idx]) + 0.5 * (prob[pos_idx[i]] == prob[neg_idx]))
        for j in range(n):
            V01[j] = np.mean((prob[pos_idx] > prob[neg_idx[j]]) + 0.5 * (prob[pos_idx] == prob[neg_idx[j]]))
        return V10, V01

    v10_a, v01_a = structural_components(prob_a)
    v10_b, v01_b = structural_components(prob_b)

    auc_a = float(np.mean(v10_a))
    auc_b = float(np.mean(v10_b))

    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))

    s = s10 / m + s01 / n
    var_diff = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])

    if var_diff <= 0:
        z_stat = 0.0
        p_val = 1.0
        ci_lower = auc_b - auc_a
        ci_upper = auc_b - auc_a
    else:
        se_diff = math.sqrt(var_diff)
        diff = auc_b - auc_a
        z_stat = float(diff / se_diff)
        p_val = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
        ci_lower = float(diff - 1.95996 * se_diff)
        ci_upper = float(diff + 1.95996 * se_diff)

    return {
        "auc_model_a": round(auc_a, 4),
        "auc_model_b": round(auc_b, 4),
        "auc_difference": round(auc_b - auc_a, 4),
        "variance_difference": round(var_diff, 6),
        "se_difference": round(math.sqrt(max(0.0, var_diff)), 4),
        "z_statistic": round(z_stat, 4),
        "p_value_delong": round(p_val, 4),
        "ci95_delong_lower": round(ci_lower, 4),
        "ci95_delong_upper": round(ci_upper, 4),
        "caveat": "DeLong assumes independent identically distributed sample vectors; cross-validation OOF predictions exhibit training-overlap covariance that can slightly underestimate variance."
    }


# ---------------------------------------------------------------------------
# Metric Calculation Helper
# ---------------------------------------------------------------------------
def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> dict[str, float]:
    y_pred = (y_prob >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    bal_acc = (sens + spec) / 2.0
    f1 = float(2.0 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    
    # Safe AUC
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = float("nan")

    return {
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "f1_score": f1,
        "roc_auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Stage 4.1 Statistical Uncertainty Audit ===")

    # 1. Input Hash Verification
    pre_hashes = {}
    for p in INPUT_ARTIFACTS:
        with open(p, "rb") as f:
            pre_hashes[str(p.relative_to(REPO_ROOT))] = hashlib.sha256(f.read()).hexdigest()

    with open(RESULTS_DIR / "input_artifact_hashes.json", "w") as f:
        json.dump(pre_hashes, f, indent=2)

    # 2. Load & Verify Data Integrity
    raw_df = pd.read_csv(RAW_DATA)
    raw_df["subject_id"] = raw_df["name"].apply(lambda x: x.split("_")[2])
    gt_map = raw_df.groupby("subject_id")["status"].first().to_dict()

    df_h22 = pd.read_csv(HIST_CSV).sort_values("subject_id").reset_index(drop=True)
    df_et13 = pd.read_csv(ET13_CSV).sort_values("subject_id").reset_index(drop=True)

    assert len(df_h22) == 32 and len(df_et13) == 32, "Participant count mismatch!"
    assert (df_h22["subject_id"] == df_et13["subject_id"]).all(), "Subject ID mismatch!"
    assert (df_h22["true_status"] == df_et13["true_status"]).all(), "Status mismatch!"

    for sid, st in df_h22[["subject_id", "true_status"]].values:
        assert gt_map[sid] == st, f"Ground truth mismatch for {sid}"

    y_all = df_h22["true_status"].to_numpy(dtype=int)
    prob_h22 = df_h22["mean_prob"].to_numpy(dtype=float)
    prob_et13 = df_et13["mean_prob"].to_numpy(dtype=float)

    n_total = len(y_all)
    pos_idx = np.where(y_all == 1)[0]  # 24 PD
    neg_idx = np.where(y_all == 0)[0]  # 8 HC
    assert len(pos_idx) == 24 and len(neg_idx) == 8

    # 3. Stratified Bootstrap Engine
    rng = np.random.RandomState(RANDOM_SEED)
    bootstrap_rows = []

    logger.info(f"Running Stratified Participant Bootstrap (B={N_BOOTSTRAP}, seed={RANDOM_SEED})...")

    metric_keys = ["sensitivity", "specificity", "balanced_accuracy", "precision", "f1_score", "roc_auc"]

    # Storage for replicates
    reps_et13 = {tau: {k: [] for k in metric_keys} for tau in THRESHOLDS}
    reps_h22 = {tau: {k: [] for k in metric_keys} for tau in THRESHOLDS}
    reps_diff = {tau: {k: [] for k in metric_keys} for tau in THRESHOLDS}

    for b in range(N_BOOTSTRAP):
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])

        y_b = y_all[sample_idx]
        p_et13_b = prob_et13[sample_idx]
        p_h22_b = prob_h22[sample_idx]

        b_record = {"replicate": b}

        for tau in THRESHOLDS:
            m_et13 = calc_metrics(y_b, p_et13_b, tau)
            m_h22 = calc_metrics(y_b, p_h22_b, tau)

            for k in metric_keys:
                reps_et13[tau][k].append(m_et13[k])
                reps_h22[tau][k].append(m_h22[k])
                diff_val = m_et13[k] - m_h22[k]
                reps_diff[tau][k].append(diff_val)
                b_record[f"et13_{k}_tau_{str(tau).replace('.', '')}"] = m_et13[k]
                b_record[f"h22_{k}_tau_{str(tau).replace('.', '')}"] = m_h22[k]
                b_record[f"diff_{k}_tau_{str(tau).replace('.', '')}"] = diff_val

        bootstrap_rows.append(b_record)

    df_boot = pd.DataFrame(bootstrap_rows)
    df_boot.to_csv(RESULTS_DIR / "bootstrap_replicates.csv", index=False)

    # 4. Compute Point Estimates and Intervals for Stream B (ET-13)
    logger.info("Computing Point Estimates & Confidence Intervals for ET-13...")
    rows_et13 = []
    for tau in THRESHOLDS:
        point = calc_metrics(y_all, prob_et13, tau)
        for k in metric_keys:
            dist = np.array(reps_et13[tau][k])
            p_low = float(np.percentile(dist, 2.5))
            p_high = float(np.percentile(dist, 97.5))
            mean_boot = float(np.mean(dist))
            std_boot = float(np.std(dist))

            res = {
                "model": "research_et_13",
                "threshold": tau,
                "metric": k,
                "point_estimate": round(point[k], 4),
                "bootstrap_mean": round(mean_boot, 4),
                "bootstrap_std": round(std_boot, 4),
                "ci95_percentile_lower": round(p_low, 4),
                "ci95_percentile_upper": round(p_high, 4),
            }

            # Exact binomial for Sens & Spec
            if k == "sensitivity":
                cp_l, cp_u = clopper_pearson_ci(point["tp"], point["tp"] + point["fn"])
                w_l, w_u = wilson_score_ci(point["tp"], point["tp"] + point["fn"])
                res["ci95_clopper_pearson_lower"] = round(cp_l, 4)
                res["ci95_clopper_pearson_upper"] = round(cp_u, 4)
                res["ci95_wilson_lower"] = round(w_l, 4)
                res["ci95_wilson_upper"] = round(w_u, 4)
            elif k == "specificity":
                cp_l, cp_u = clopper_pearson_ci(point["tn"], point["tn"] + point["fp"])
                w_l, w_u = wilson_score_ci(point["tn"], point["tn"] + point["fp"])
                res["ci95_clopper_pearson_lower"] = round(cp_l, 4)
                res["ci95_clopper_pearson_upper"] = round(cp_u, 4)
                res["ci95_wilson_lower"] = round(w_l, 4)
                res["ci95_wilson_upper"] = round(w_u, 4)
            else:
                res["ci95_clopper_pearson_lower"] = None
                res["ci95_clopper_pearson_upper"] = None
                res["ci95_wilson_lower"] = None
                res["ci95_wilson_upper"] = None

            rows_et13.append(res)

    pd.DataFrame(rows_et13).to_csv(RESULTS_DIR / "uncertainty_summary_et13.csv", index=False)

    # 5. Compute Point Estimates and Intervals for Stream A (Historical 22)
    logger.info("Computing Point Estimates & Confidence Intervals for Historical 22...")
    rows_h22 = []
    for tau in THRESHOLDS:
        point = calc_metrics(y_all, prob_h22, tau)
        for k in metric_keys:
            dist = np.array(reps_h22[tau][k])
            p_low = float(np.percentile(dist, 2.5))
            p_high = float(np.percentile(dist, 97.5))
            mean_boot = float(np.mean(dist))
            std_boot = float(np.std(dist))

            res = {
                "model": "historical_22",
                "threshold": tau,
                "metric": k,
                "point_estimate": round(point[k], 4),
                "bootstrap_mean": round(mean_boot, 4),
                "bootstrap_std": round(std_boot, 4),
                "ci95_percentile_lower": round(p_low, 4),
                "ci95_percentile_upper": round(p_high, 4),
            }

            if k == "sensitivity":
                cp_l, cp_u = clopper_pearson_ci(point["tp"], point["tp"] + point["fn"])
                w_l, w_u = wilson_score_ci(point["tp"], point["tp"] + point["fn"])
                res["ci95_clopper_pearson_lower"] = round(cp_l, 4)
                res["ci95_clopper_pearson_upper"] = round(cp_u, 4)
                res["ci95_wilson_lower"] = round(w_l, 4)
                res["ci95_wilson_upper"] = round(w_u, 4)
            elif k == "specificity":
                cp_l, cp_u = clopper_pearson_ci(point["tn"], point["tn"] + point["fp"])
                w_l, w_u = wilson_score_ci(point["tn"], point["tn"] + point["fp"])
                res["ci95_clopper_pearson_lower"] = round(cp_l, 4)
                res["ci95_clopper_pearson_upper"] = round(cp_u, 4)
                res["ci95_wilson_lower"] = round(w_l, 4)
                res["ci95_wilson_upper"] = round(w_u, 4)
            else:
                res["ci95_clopper_pearson_lower"] = None
                res["ci95_clopper_pearson_upper"] = None
                res["ci95_wilson_lower"] = None
                res["ci95_wilson_upper"] = None

            rows_h22.append(res)

    pd.DataFrame(rows_h22).to_csv(RESULTS_DIR / "uncertainty_summary_historical22.csv", index=False)

    # 6. Paired Differences Analysis (ET13 minus H22)
    logger.info("Computing Paired Bootstrap Differences...")
    diff_rows = []
    for tau in THRESHOLDS:
        pt_et13 = calc_metrics(y_all, prob_et13, tau)
        pt_h22 = calc_metrics(y_all, prob_h22, tau)
        for k in metric_keys:
            dist = np.array(reps_diff[tau][k])
            p_low = float(np.percentile(dist, 2.5))
            p_high = float(np.percentile(dist, 97.5))
            pt_diff = pt_et13[k] - pt_h22[k]

            diff_rows.append({
                "threshold": tau,
                "metric": k,
                "point_estimate_h22": round(pt_h22[k], 4),
                "point_estimate_et13": round(pt_et13[k], 4),
                "point_difference": round(pt_diff, 4),
                "bootstrap_mean_diff": round(float(np.mean(dist)), 4),
                "bootstrap_std_diff": round(float(np.std(dist)), 4),
                "ci95_diff_lower": round(p_low, 4),
                "ci95_diff_upper": round(p_high, 4),
                "zero_in_ci": bool(p_low <= 0.0 <= p_high),
            })

    pd.DataFrame(diff_rows).to_csv(RESULTS_DIR / "paired_model_comparison.csv", index=False)

    # 7. Exact McNemar Test on Discordant Pairs
    logger.info("Computing Exact McNemar Test on Discordant Pairs...")
    mcnemar_results = {}
    for tau in THRESHOLDS:
        pred_et13 = (prob_et13 >= tau).astype(int)
        pred_h22 = (prob_h22 >= tau).astype(int)

        corr_et13 = (pred_et13 == y_all).astype(int)
        corr_h22 = (pred_h22 == y_all).astype(int)

        n11 = int(np.sum((corr_h22 == 1) & (corr_et13 == 1)))
        n00 = int(np.sum((corr_h22 == 0) & (corr_et13 == 0)))
        b = int(np.sum((corr_h22 == 1) & (corr_et13 == 0)))  # H22 correct, ET13 wrong
        c = int(np.sum((corr_h22 == 0) & (corr_et13 == 1)))  # ET13 correct, H22 wrong

        # Exact binomial test on discordant pairs
        total_discordant = b + c
        if total_discordant > 0:
            res_mcnemar = binomtest(b, total_discordant, p=0.5, alternative="two-sided")
            p_val = float(res_mcnemar.pvalue)
        else:
            p_val = 1.0

        # Detailed subject IDs of discordant cases
        subjs_h22_better = df_h22.iloc[np.where((corr_h22 == 1) & (corr_et13 == 0))[0]]["subject_id"].tolist()
        subjs_et13_better = df_h22.iloc[np.where((corr_h22 == 0) & (corr_et13 == 1))[0]]["subject_id"].tolist()

        mcnemar_results[f"tau_{tau}"] = {
            "threshold": tau,
            "both_correct_n11": n11,
            "both_incorrect_n00": n00,
            "h22_correct_et13_incorrect_b": b,
            "et13_correct_h22_incorrect_c": c,
            "total_discordant": total_discordant,
            "exact_mcnemar_p_value": round(p_val, 4),
            "subjects_where_h22_correct": subjs_h22_better,
            "subjects_where_et13_correct": subjs_et13_better,
            "interpretation": "Nonsignificant overall accuracy difference does not imply equivalence; errors fundamentally shifted from healthy controls (specificity) to mild PD cases (sensitivity)."
        }

    with open(RESULTS_DIR / "mcnemar_contingency.json", "w") as f:
        json.dump(mcnemar_results, f, indent=2)

    # 8. DeLong AUC Test
    logger.info("Computing Supplementary DeLong AUC Test...")
    delong_res = compute_delong_test(y_true=y_all, prob_a=prob_h22, prob_b=prob_et13)
    with open(RESULTS_DIR / "delong_auc_test.json", "w") as f:
        json.dump(delong_res, f, indent=2)

    # 9. Execution Configuration Record
    exec_config = {
        "stage": "stage_4_1_statistical_uncertainty_audit",
        "sample_size": n_total,
        "pd_count": len(pos_idx),
        "hc_count": len(neg_idx),
        "bootstrap_replicates": N_BOOTSTRAP,
        "bootstrap_seed": RANDOM_SEED,
        "stratification": "class_stratified_participant_level",
        "confidence_level": 0.95,
        "thresholds_evaluated": THRESHOLDS,
        "runtime_libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": "scipy-available"
        }
    }
    with open(RESULTS_DIR / "execution_config.json", "w") as f:
        json.dump(exec_config, f, indent=2)

    logger.info("Stage 4.1 audit computation successfully completed.")


if __name__ == "__main__":
    main()
