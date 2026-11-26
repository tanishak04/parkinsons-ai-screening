"""
ml/pathway_a/experiments/run_experiments.py
Isolated Pathway A baseline benchmarking harness.

Evaluates the exact existing Stage-1 baseline pipeline
(StandardScaler + LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
without modifying protected files, overwriting baseline artifacts, or tuning hyperparameters.

Evaluation schemes:
  1. Repeated 4-fold StratifiedGroupKFold across 10 random seeds (40 folds total)
  2. 32-fold LeaveOneGroupOut (LOGO) across all participants

Usage:
    python ml/pathway_a/experiments/run_experiments.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
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

# ---------------------------------------------------------------------------
# Paths & Contract Constants (read-only references to Pathway A assets)
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PATHWAY_A_DIR = EXPERIMENTS_DIR.parent
RAW_DATA_PATH = PATHWAY_A_DIR / "data" / "raw" / "parkinsons.data"
LEGACY_METRICS_PATH = PATHWAY_A_DIR / "models" / "baseline_metrics.json"

RESULTS_DIR = EXPERIMENTS_DIR / "results"
MODELS_DIR = EXPERIMENTS_DIR / "models"

EXPECTED_FEATURES: list[str] = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
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
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]

SEEDS: list[int] = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
N_SPLITS: int = 4
DECISION_THRESHOLD: float = 0.50

METRIC_NAMES: list[str] = [
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "brier_score",
]


# ---------------------------------------------------------------------------
# Dataset Loading & Validation
# ---------------------------------------------------------------------------
def load_and_validate_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    """Load parkinsons.data and validate schema, order, integrity, and participant grouping."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    dataset_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    df = pd.read_csv(path)

    required_cols = {"name", "status", *EXPECTED_FEATURES}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    feature_cols_in_file = [c for c in df.columns if c not in ("name", "status")]
    if feature_cols_in_file != EXPECTED_FEATURES:
        raise ValueError(
            "Feature columns or order mismatch.\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Found:    {feature_cols_in_file}"
        )

    X = df[EXPECTED_FEATURES].copy()
    for col in EXPECTED_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="raise").astype(float)

    if X.isna().any().any():
        raise ValueError("Dataset contains NaN values in feature columns.")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Dataset contains non-finite values in feature columns.")

    y = df["status"].astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected target values: {sorted(y.unique())}")

    names = df["name"].astype(str)
    groups = names.str.split("_").str[2]
    if groups.isna().any() or (groups == "").any():
        raise ValueError("Failed to parse participant IDs via df['name'].str.split('_').str[2].")

    mixed_status_subjects = df.groupby(groups)["status"].nunique()
    if (mixed_status_subjects > 1).any():
        bad = mixed_status_subjects[mixed_status_subjects > 1].index.tolist()
        raise ValueError(f"Mixed target status detected for participant(s): {bad}")

    subject_labels = df.groupby(groups)["status"].first()
    dataset_info = {
        "path": str(path),
        "sha256": dataset_sha256,
        "n_recordings": int(len(df)),
        "n_features": int(len(EXPECTED_FEATURES)),
        "n_participants": int(groups.nunique()),
        "pd_recordings": int((y == 1).sum()),
        "healthy_recordings": int((y == 0).sum()),
        "pd_participants": int((subject_labels == 1).sum()),
        "healthy_participants": int((subject_labels == 0).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
    }
    return X, y, groups, names, dataset_info


# ---------------------------------------------------------------------------
# Model Factory (Exact Stage-1 Baseline Pipeline)
# ---------------------------------------------------------------------------
def build_baseline_pipeline() -> Pipeline:
    """Instantiate a fresh copy of the existing Stage-1 baseline pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Metrics & Aggregation Helpers
# ---------------------------------------------------------------------------
def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Compute binary classification metrics; returns NaN for metrics undefined on single-class slices."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    has_pos = (tp + fn) > 0
    has_neg = (tn + fp) > 0

    accuracy = float(accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, zero_division=0)) if (tp + fp) > 0 or has_pos else float("nan")
    recall = float(recall_score(y_true, y_pred, zero_division=0)) if has_pos else float("nan")
    specificity = float(tn / (tn + fp)) if has_neg else float("nan")
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if has_pos else float("nan")
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan")
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both_classes else float("nan")
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if has_both_classes else float("nan")

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6) if np.isfinite(precision) else float("nan"),
        "recall_sensitivity": round(recall, 6) if np.isfinite(recall) else float("nan"),
        "specificity": round(specificity, 6) if np.isfinite(specificity) else float("nan"),
        "f1": round(f1, 6) if np.isfinite(f1) else float("nan"),
        "roc_auc": round(roc_auc, 6) if np.isfinite(roc_auc) else float("nan"),
        "pr_auc": round(pr_auc, 6) if np.isfinite(pr_auc) else float("nan"),
        "balanced_accuracy": round(bal_acc, 6) if np.isfinite(bal_acc) else float("nan"),
        "brier_score": round(brier, 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def aggregate_participant_predictions(
    subject_ids: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    """Aggregate recording-level predicted probabilities per participant using mean probability."""
    frame = pd.DataFrame(
        {
            "subject_id": np.asarray(subject_ids),
            "y_true": np.asarray(y_true, dtype=int),
            "y_prob": np.asarray(y_prob, dtype=float),
        }
    )
    grouped = (
        frame.groupby("subject_id", sort=True, as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_prob=("y_prob", "mean"),
            n_recordings=("y_prob", "size"),
        )
    )
    grouped["y_pred"] = (grouped["y_prob"] >= DECISION_THRESHOLD).astype(int)
    return grouped


def summarize_metric_rows(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute mean, std, min, and max across rows for every metric."""
    summary: dict[str, dict[str, float]] = {}
    for col in METRIC_NAMES:
        vals = df[col].dropna().to_numpy(dtype=float)
        summary[col] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals, ddof=0)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }
    return summary


# ---------------------------------------------------------------------------
# Evaluation Scheme 1: Repeated 4-Fold StratifiedGroupKFold (10 Seeds)
# ---------------------------------------------------------------------------
def run_repeated_sgkf(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    names: pd.Series,
    pipeline_fn: Any = build_baseline_pipeline,
    scheme_name: str = "repeated_sgkf_4x10",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Run 10-seed x 4-fold StratifiedGroupKFold and return fold metrics, OOF rows, and seed OOF metrics."""
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    seed_oof_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_oof_prob = np.full(len(X), np.nan, dtype=float)
        seed_oof_fold = np.full(len(X), -1, dtype=int)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            train_subjects = set(groups.iloc[train_idx])
            test_subjects = set(groups.iloc[test_idx])
            assert train_subjects.isdisjoint(test_subjects), (
                f"Participant leakage detected in SGKF seed={seed}, fold={fold_idx}"
            )

            pipeline = pipeline_fn()
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_prob = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
            y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

            seed_oof_prob[test_idx] = y_prob
            seed_oof_fold[test_idx] = fold_idx

            for pos, row_idx in enumerate(test_idx):
                oof_rows.append(
                    {
                        "evaluation_scheme": scheme_name,
                        "seed": int(seed),
                        "fold": int(fold_idx),
                        "row_index": int(row_idx),
                        "name": str(names.iloc[row_idx]),
                        "subject_id": str(groups.iloc[row_idx]),
                        "y_true": int(y.iloc[row_idx]),
                        "y_prob": round(float(y_prob[pos]), 6),
                        "y_pred": int(y_pred[pos]),
                    }
                )

            part_df = aggregate_participant_predictions(
                groups.iloc[test_idx], y.iloc[test_idx], y_prob
            )
            test_pd_subj = int((part_df["y_true"] == 1).sum())
            test_h_subj = int((part_df["y_true"] == 0).sum())

            rec_metrics = compute_binary_metrics(y.iloc[test_idx].to_numpy(), y_prob)
            part_metrics = compute_binary_metrics(
                part_df["y_true"].to_numpy(), part_df["y_prob"].to_numpy()
            )

            common_meta = {
                "seed": int(seed),
                "fold": int(fold_idx),
                "train_samples": int(len(train_idx)),
                "test_samples": int(len(test_idx)),
                "train_participants": int(len(train_subjects)),
                "test_participants": int(len(test_subjects)),
                "test_pd_participants": test_pd_subj,
                "test_healthy_participants": test_h_subj,
            }
            fold_rows.append({**common_meta, "level": "recording", **rec_metrics})
            fold_rows.append({**common_meta, "level": "participant", **part_metrics})

        assert not np.isnan(seed_oof_prob).any(), f"Incomplete OOF coverage for seed={seed}"

        seed_rec = compute_binary_metrics(y.to_numpy(), seed_oof_prob)
        seed_part_df = aggregate_participant_predictions(groups, y, seed_oof_prob)
        seed_part = compute_binary_metrics(
            seed_part_df["y_true"].to_numpy(), seed_part_df["y_prob"].to_numpy()
        )
        seed_oof_rows.append({"seed": int(seed), "level": "recording", **seed_rec})
        seed_oof_rows.append({"seed": int(seed), "level": "participant", **seed_part})

    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows), seed_oof_rows


# ---------------------------------------------------------------------------
# Evaluation Scheme 2: 32-Fold LeaveOneGroupOut (LOGO)
# ---------------------------------------------------------------------------
def run_logo(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    names: pd.Series,
    pipeline_fn: Any = build_baseline_pipeline,
    scheme_name: str = "logo_32",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    """Run 32-fold LeaveOneGroupOut evaluation and return fold metrics, OOF rows, and pooled OOF metrics."""
    logo = LeaveOneGroupOut()
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    oof_prob = np.full(len(X), np.nan, dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        train_subjects = set(groups.iloc[train_idx])
        test_subjects = set(groups.iloc[test_idx])
        assert train_subjects.isdisjoint(test_subjects), (
            f"Participant leakage detected in LOGO fold={fold_idx}"
        )
        assert len(test_subjects) == 1, f"Expected 1 test participant in LOGO fold={fold_idx}"

        subject_id = next(iter(test_subjects))
        true_status = int(y.iloc[test_idx].iloc[0])

        pipeline = pipeline_fn()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

        oof_prob[test_idx] = y_prob

        for pos, row_idx in enumerate(test_idx):
            oof_rows.append(
                {
                    "evaluation_scheme": scheme_name,
                    "seed": "none",
                    "fold": int(fold_idx),
                    "row_index": int(row_idx),
                    "name": str(names.iloc[row_idx]),
                    "subject_id": str(subject_id),
                    "y_true": int(y.iloc[row_idx]),
                    "y_prob": round(float(y_prob[pos]), 6),
                    "y_pred": int(y_pred[pos]),
                }
            )

        rec_metrics = compute_binary_metrics(y.iloc[test_idx].to_numpy(), y_prob)
        mean_prob = float(np.mean(y_prob))
        part_pred = int(mean_prob >= DECISION_THRESHOLD)

        fold_rows.append(
            {
                "fold": int(fold_idx),
                "subject_id": subject_id,
                "true_status": true_status,
                "train_samples": int(len(train_idx)),
                "n_recordings": int(len(test_idx)),
                "mean_pred_prob": round(mean_prob, 6),
                "participant_pred": part_pred,
                "participant_correct": int(part_pred == true_status),
                **rec_metrics,
            }
        )

    assert not np.isnan(oof_prob).any(), "Incomplete LOGO OOF coverage."

    pooled_rec = compute_binary_metrics(y.to_numpy(), oof_prob)
    pooled_part_df = aggregate_participant_predictions(groups, y, oof_prob)
    pooled_part = compute_binary_metrics(
        pooled_part_df["y_true"].to_numpy(), pooled_part_df["y_prob"].to_numpy()
    )
    pooled_summary = {
        "recording": pooled_rec,
        "participant": pooled_part,
    }
    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows), pooled_summary


# ---------------------------------------------------------------------------
# Output Generation
# ---------------------------------------------------------------------------
def load_legacy_fold0_metrics(path: Path) -> dict[str, Any] | None:
    """Read legacy single-fold metrics JSON without modifying it."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_summary_table(
    legacy_metrics: dict[str, Any] | None,
    sgkf_per_fold_summary: dict[str, dict[str, dict[str, float]]],
    sgkf_per_seed_summary: dict[str, dict[str, dict[str, float]]],
    logo_pooled_summary: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a flat human-readable comparison table for benchmark_summary.csv."""
    rows: list[dict[str, Any]] = []

    if legacy_metrics is not None:
        lm = legacy_metrics["metrics"]
        rows.append(
            {
                "benchmark": "legacy_single_fold0_unreliable",
                "level": "recording",
                "aggregation": "single_fold_n37",
                "accuracy": lm.get("accuracy"),
                "precision": lm.get("precision"),
                "recall_sensitivity": lm.get("recall_sensitivity"),
                "specificity": lm.get("specificity"),
                "f1": lm.get("f1_score"),
                "roc_auc": lm.get("roc_auc"),
                "pr_auc": None,
                "balanced_accuracy": None,
                "brier_score": None,
            }
        )

    for level in ("recording", "participant"):
        rows.append(
            {
                "benchmark": "repeated_sgkf_4x10_per_fold_mean",
                "level": level,
                "aggregation": "mean_over_40_folds",
                **{m: sgkf_per_fold_summary[level][m]["mean"] for m in METRIC_NAMES},
            }
        )
        rows.append(
            {
                "benchmark": "repeated_sgkf_4x10_per_fold_std",
                "level": level,
                "aggregation": "std_over_40_folds",
                **{m: sgkf_per_fold_summary[level][m]["std"] for m in METRIC_NAMES},
            }
        )
        rows.append(
            {
                "benchmark": "repeated_sgkf_4x10_seed_oof_mean",
                "level": level,
                "aggregation": "mean_over_10_seed_oofs",
                **{m: sgkf_per_seed_summary[level][m]["mean"] for m in METRIC_NAMES},
            }
        )
        rows.append(
            {
                "benchmark": "repeated_sgkf_4x10_seed_oof_std",
                "level": level,
                "aggregation": "std_over_10_seed_oofs",
                **{m: sgkf_per_seed_summary[level][m]["std"] for m in METRIC_NAMES},
            }
        )
        rows.append(
            {
                "benchmark": "logo_32_pooled_oof",
                "level": level,
                "aggregation": "pooled_oof_all_32_subjects",
                **{m: round(float(logo_pooled_summary[level][m]), 4) for m in METRIC_NAMES},
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups, names, dataset_info = load_and_validate_dataset(RAW_DATA_PATH)
    legacy_metrics = load_legacy_fold0_metrics(LEGACY_METRICS_PATH)

    sgkf_fold_df, sgkf_oof_df, sgkf_seed_rows = run_repeated_sgkf(X, y, groups, names)
    logo_fold_df, logo_oof_df, logo_pooled_summary = run_logo(X, y, groups, names)

    sgkf_seed_df = pd.DataFrame(sgkf_seed_rows)

    sgkf_per_fold_summary = {
        level: summarize_metric_rows(sgkf_fold_df[sgkf_fold_df["level"] == level])
        for level in ("recording", "participant")
    }
    sgkf_per_seed_summary = {
        level: summarize_metric_rows(sgkf_seed_df[sgkf_seed_df["level"] == level])
        for level in ("recording", "participant")
    }

    oof_all_df = pd.concat([sgkf_oof_df, logo_oof_df], ignore_index=True)

    summary_csv_df = build_summary_table(
        legacy_metrics=legacy_metrics,
        sgkf_per_fold_summary=sgkf_per_fold_summary,
        sgkf_per_seed_summary=sgkf_per_seed_summary,
        logo_pooled_summary=logo_pooled_summary,
    )

    summary_payload: dict[str, Any] = {
        "benchmark_stage": "pathway_a_baseline_verification",
        "configuration": {
            "model_pipeline": "StandardScaler -> LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)",
            "decision_threshold": DECISION_THRESHOLD,
            "repeated_sgkf": {
                "n_splits": N_SPLITS,
                "n_repeats": len(SEEDS),
                "total_folds": N_SPLITS * len(SEEDS),
                "seeds": SEEDS,
            },
            "logo": {
                "n_folds": int(groups.nunique()),
            },
            "group_leakage_checked": True,
            "runtime_versions": {
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "joblib": joblib.__version__,
            },
        },
        "dataset": dataset_info,
        "legacy_single_fold0_benchmark": legacy_metrics,
        "repeated_sgkf_4x10": {
            "per_fold_summary": sgkf_per_fold_summary,
            "per_seed_oof_summary": sgkf_per_seed_summary,
            "per_seed_oof_raw": sgkf_seed_rows,
        },
        "logo_32": {
            "pooled_oof": logo_pooled_summary,
        },
    }

    sgkf_fold_path = RESULTS_DIR / "repeated_sgkf_fold_metrics.csv"
    logo_fold_path = RESULTS_DIR / "logo_fold_metrics.csv"
    oof_path = RESULTS_DIR / "oof_predictions.csv"
    summary_json_path = RESULTS_DIR / "benchmark_summary.json"
    summary_csv_path = RESULTS_DIR / "benchmark_summary.csv"

    sgkf_fold_df.to_csv(sgkf_fold_path, index=False)
    logo_fold_df.to_csv(logo_fold_path, index=False)
    oof_all_df.to_csv(oof_path, index=False)
    summary_csv_df.to_csv(summary_csv_path, index=False)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("=" * 72)
    print("  Pathway A — Isolated Baseline Benchmark Complete")
    print("=" * 72)
    print(f"Dataset: {dataset_info['n_recordings']} recordings, "
          f"{dataset_info['n_participants']} participants "
          f"({dataset_info['pd_participants']} PD, {dataset_info['healthy_participants']} Healthy)")
    print(f"Repeated SGKF folds evaluated : {len(sgkf_fold_df) // 2} (10 seeds x 4 folds)")
    print(f"LOGO folds evaluated          : {len(logo_fold_df)}")
    print("Group leakage assertions      : PASSED (0 overlapping participants)\n")
    print(summary_csv_df.to_string(index=False))
    print(f"\nSaved outputs to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
