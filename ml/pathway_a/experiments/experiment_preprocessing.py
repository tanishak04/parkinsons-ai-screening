"""
ml/pathway_a/experiments/experiment_preprocessing.py
Pathway A — Stage 2A: Preprocessing & Regularization Experiments.

Evaluates whether RobustScaler or PowerTransformer(method="yeo-johnson")
improves generalization performance compared to the StandardScaler reference,
across L2, L1, and ElasticNet regularizations for Logistic Regression.

Evaluation methodology:
  - Reuses the verified harness from ml.pathway_a.experiments.run_experiments
  - Repeated 4-fold StratifiedGroupKFold across 10 random seeds (40 folds per variant)
  - 32-fold LeaveOneGroupOut (LOGO) across all participants
  - Recording-level and participant-level evaluations
  - Strict speaker disjointness assertions per fold

Outputs saved to:
  ml/pathway_a/experiments/results/preprocessing/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from ml.pathway_a.experiments.run_experiments import (
    DECISION_THRESHOLD,
    EXPECTED_FEATURES,
    METRIC_NAMES,
    RAW_DATA_PATH,
    SEEDS,
    load_and_validate_dataset,
    run_logo,
    run_repeated_sgkf,
    summarize_metric_rows,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PREPROCESSING_RESULTS_DIR = EXPERIMENTS_DIR / "results" / "preprocessing"


# ---------------------------------------------------------------------------
# Predefined Preprocessing & Regularization Model Configurations
# ---------------------------------------------------------------------------
def make_pipeline_factory(
    scaler_type: str,
    penalty_type: str,
    random_state: int = 42,
) -> Callable[[], Pipeline]:
    """Return a fresh pipeline factory function for a specified scaler and penalty."""

    def _factory() -> Pipeline:
        # 1. Scaler selection
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "power":
            scaler = PowerTransformer(method="yeo-johnson")
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")

        # 2. Classifier & regularization selection
        if penalty_type == "l2":
            clf = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            )
        elif penalty_type == "l1":
            clf = LogisticRegression(
                C=1.0,
                penalty="l1",
                solver="saga",
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
            )
        elif penalty_type == "elasticnet":
            clf = LogisticRegression(
                C=1.0,
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown penalty_type: {penalty_type}")

        return Pipeline(steps=[("scaler", scaler), ("classifier", clf)])

    return _factory


EXPERIMENT_CONFIGS: list[dict[str, Any]] = [
    # Reference Baseline
    {
        "config_id": "std_l2_reference",
        "scaler": "standard",
        "penalty": "l2",
        "solver": "lbfgs",
        "description": "StandardScaler + LogisticRegression(L2, balanced, lbfgs) [Baseline Reference]",
    },
    # Regularization variants on StandardScaler
    {
        "config_id": "std_l1",
        "scaler": "standard",
        "penalty": "l1",
        "solver": "saga",
        "description": "StandardScaler + LogisticRegression(L1, balanced, saga)",
    },
    {
        "config_id": "std_elasticnet",
        "scaler": "standard",
        "penalty": "elasticnet",
        "solver": "saga",
        "description": "StandardScaler + LogisticRegression(ElasticNet l1_ratio=0.5, balanced, saga)",
    },
    # RobustScaler variants
    {
        "config_id": "robust_l2",
        "scaler": "robust",
        "penalty": "l2",
        "solver": "lbfgs",
        "description": "RobustScaler + LogisticRegression(L2, balanced, lbfgs)",
    },
    {
        "config_id": "robust_l1",
        "scaler": "robust",
        "penalty": "l1",
        "solver": "saga",
        "description": "RobustScaler + LogisticRegression(L1, balanced, saga)",
    },
    {
        "config_id": "robust_elasticnet",
        "scaler": "robust",
        "penalty": "elasticnet",
        "solver": "saga",
        "description": "RobustScaler + LogisticRegression(ElasticNet l1_ratio=0.5, balanced, saga)",
    },
    # PowerTransformer (Yeo-Johnson) variants
    {
        "config_id": "power_l2",
        "scaler": "power",
        "penalty": "l2",
        "solver": "lbfgs",
        "description": "PowerTransformer(yeo-johnson) + LogisticRegression(L2, balanced, lbfgs)",
    },
    {
        "config_id": "power_l1",
        "scaler": "power",
        "penalty": "l1",
        "solver": "saga",
        "description": "PowerTransformer(yeo-johnson) + LogisticRegression(L1, balanced, saga)",
    },
    {
        "config_id": "power_elasticnet",
        "scaler": "power",
        "penalty": "elasticnet",
        "solver": "saga",
        "description": "PowerTransformer(yeo-johnson) + LogisticRegression(ElasticNet l1_ratio=0.5, balanced, saga)",
    },
]


def run_preprocessing_experiments() -> None:
    PREPROCESSING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  Pathway A — Stage 2A: Preprocessing & Regularization Experiments")
    print("=" * 78)

    X, y, groups, names, dataset_info = load_and_validate_dataset(RAW_DATA_PATH)
    print(f"Dataset verified: {len(X)} recordings across {groups.nunique()} participants.")
    print(f"Configurations to test: {len(EXPERIMENT_CONFIGS)}")
    print(f"Evaluation: Repeated 4-fold SGKF (10 seeds, 40 folds) + 32-fold LOGO per config\n")

    all_fold_rows: list[dict[str, Any]] = []
    all_oof_rows: list[dict[str, Any]] = []
    comparison_summary_rows: list[dict[str, Any]] = []
    structured_results: dict[str, Any] = {
        "stage": "stage_2a_preprocessing",
        "dataset": dataset_info,
        "configurations": EXPERIMENT_CONFIGS,
        "results": {},
    }

    for idx, cfg in enumerate(EXPERIMENT_CONFIGS, 1):
        cid = cfg["config_id"]
        print(f"[{idx}/{len(EXPERIMENT_CONFIGS)}] Running {cid} ...")

        factory = make_pipeline_factory(cfg["scaler"], cfg["penalty"])

        # 1. Run Repeated 4-Fold SGKF
        sgkf_folds, sgkf_oofs, sgkf_seed_oofs = run_repeated_sgkf(
            X=X,
            y=y,
            groups=groups,
            names=names,
            pipeline_fn=factory,
            scheme_name=f"repeated_sgkf_4x10_{cid}",
        )
        sgkf_folds["config_id"] = cid
        sgkf_oofs["config_id"] = cid

        # 2. Run 32-Fold LOGO
        logo_folds, logo_oofs, logo_pooled = run_logo(
            X=X,
            y=y,
            groups=groups,
            names=names,
            pipeline_fn=factory,
            scheme_name=f"logo_32_{cid}",
        )
        logo_folds["config_id"] = cid
        logo_oofs["config_id"] = cid

        all_fold_rows.append(sgkf_folds)
        all_fold_rows.append(logo_folds)
        all_oof_rows.append(sgkf_oofs)
        all_oof_rows.append(logo_oofs)

        # Summaries
        sgkf_seed_df = pd.DataFrame(sgkf_seed_oofs)
        sgkf_fold_summary = {
            lvl: summarize_metric_rows(sgkf_folds[sgkf_folds["level"] == lvl])
            for lvl in ("recording", "participant")
        }
        sgkf_seed_summary = {
            lvl: summarize_metric_rows(sgkf_seed_df[sgkf_seed_df["level"] == lvl])
            for lvl in ("recording", "participant")
        }

        structured_results["results"][cid] = {
            "config": cfg,
            "repeated_sgkf_4x10": {
                "per_fold": sgkf_fold_summary,
                "per_seed_oof": sgkf_seed_summary,
            },
            "logo_32": logo_pooled,
        }

        # Flat rows for aggregate_comparison.csv
        for lvl in ("recording", "participant"):
            comparison_summary_rows.append(
                {
                    "config_id": cid,
                    "scaler": cfg["scaler"],
                    "penalty": cfg["penalty"],
                    "evaluation": "repeated_sgkf_4x10_fold_mean",
                    "level": lvl,
                    **{m: sgkf_fold_summary[lvl][m]["mean"] for m in METRIC_NAMES},
                }
            )
            comparison_summary_rows.append(
                {
                    "config_id": cid,
                    "scaler": cfg["scaler"],
                    "penalty": cfg["penalty"],
                    "evaluation": "repeated_sgkf_4x10_fold_std",
                    "level": lvl,
                    **{m: sgkf_fold_summary[lvl][m]["std"] for m in METRIC_NAMES},
                }
            )
            comparison_summary_rows.append(
                {
                    "config_id": cid,
                    "scaler": cfg["scaler"],
                    "penalty": cfg["penalty"],
                    "evaluation": "repeated_sgkf_4x10_seed_oof_mean",
                    "level": lvl,
                    **{m: sgkf_seed_summary[lvl][m]["mean"] for m in METRIC_NAMES},
                }
            )
            comparison_summary_rows.append(
                {
                    "config_id": cid,
                    "scaler": cfg["scaler"],
                    "penalty": cfg["penalty"],
                    "evaluation": "repeated_sgkf_4x10_seed_oof_std",
                    "level": lvl,
                    **{m: sgkf_seed_summary[lvl][m]["std"] for m in METRIC_NAMES},
                }
            )
            comparison_summary_rows.append(
                {
                    "config_id": cid,
                    "scaler": cfg["scaler"],
                    "penalty": cfg["penalty"],
                    "evaluation": "logo_32_pooled_oof",
                    "level": lvl,
                    **{m: round(float(logo_pooled[lvl][m]), 4) for m in METRIC_NAMES},
                }
            )

        print(
            f"   SGKF 10-seed OOF (Rec): BalAcc={sgkf_seed_summary['recording']['balanced_accuracy']['mean']:.4f} "
            f"| Recall={sgkf_seed_summary['recording']['recall_sensitivity']['mean']:.4f} "
            f"| Spec={sgkf_seed_summary['recording']['specificity']['mean']:.4f} "
            f"| ROC-AUC={sgkf_seed_summary['recording']['roc_auc']['mean']:.4f} "
            f"| Brier={sgkf_seed_summary['recording']['brier_score']['mean']:.4f}"
        )
        print(
            f"   LOGO Pooled OOF (Subj): BalAcc={logo_pooled['participant']['balanced_accuracy']:.4f} "
            f"| Recall={logo_pooled['participant']['recall_sensitivity']:.4f} "
            f"| Spec={logo_pooled['participant']['specificity']:.4f} "
            f"| ROC-AUC={logo_pooled['participant']['roc_auc']:.4f} "
            f"| Brier={logo_pooled['participant']['brier_score']:.4f}\n"
        )

    # Consolidate and export files
    combined_folds_df = pd.concat(all_fold_rows, ignore_index=True)
    combined_oofs_df = pd.concat(all_oof_rows, ignore_index=True)
    comparison_df = pd.DataFrame(comparison_summary_rows)

    folds_csv_path = PREPROCESSING_RESULTS_DIR / "per_fold_metrics.csv"
    oof_csv_path = PREPROCESSING_RESULTS_DIR / "pooled_oof_predictions.csv"
    comparison_csv_path = PREPROCESSING_RESULTS_DIR / "aggregate_comparison.csv"
    config_json_path = PREPROCESSING_RESULTS_DIR / "experiment_config.json"

    combined_folds_df.to_csv(folds_csv_path, index=False)
    combined_oofs_df.to_csv(oof_csv_path, index=False)
    comparison_df.to_csv(comparison_csv_path, index=False)
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_results, f, indent=2)

    print("=" * 78)
    print("  Stage 2A Preprocessing Experiments Complete")
    print("=" * 78)
    print(f"Per-fold metrics CSV   : {folds_csv_path} (shape={combined_folds_df.shape})")
    print(f"Pooled OOF CSV         : {oof_csv_path} (shape={combined_oofs_df.shape})")
    print(f"Comparison CSV         : {comparison_csv_path} (shape={comparison_df.shape})")
    print(f"Config & Summary JSON  : {config_json_path}")


if __name__ == "__main__":
    run_preprocessing_experiments()
