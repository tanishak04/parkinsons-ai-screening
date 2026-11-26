"""
ml/pathway_a/experiments/model_comparison.py
Pathway A — Stage 2B: Nonlinear Model Family Evaluation.

Evaluates whether nonlinear machine learning models (SVM RBF, Random Forest,
Extra Trees, Gradient Boosting, HistGradientBoosting, and XGBoost) improve Parkinson's
classification generalization over the verified Logistic Regression baseline on the UCI dataset.

Evaluation methodology:
  - Reuses the verified grouped evaluation framework from run_experiments.py
  - Repeated 4-fold StratifiedGroupKFold across 10 random seeds (40 folds per model)
  - 32-fold LeaveOneGroupOut (LOGO) cross-validation (32 folds per model)
  - Recording-level and participant-level evaluations
  - Automated GPU/CUDA environment detection and execution logging

Outputs saved to:
  ml/pathway_a/experiments/results/model_comparison/
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

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
STAGE2B_RESULTS_DIR = EXPERIMENTS_DIR / "results" / "model_comparison"


# ---------------------------------------------------------------------------
# GPU / CUDA Hardware Detection
# ---------------------------------------------------------------------------
def detect_gpu_environment() -> dict[str, Any]:
    """Detect available CUDA hardware and XGBoost GPU support."""
    gpu_info: dict[str, Any] = {
        "cuda_available": False,
        "gpu_name": None,
        "vram_total_mb": None,
        "driver_version": None,
        "cuda_driver_version": None,
        "xgboost_cuda_supported": False,
        "notes": [],
    }

    # 1. Probe nvidia-smi
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 3:
                gpu_info["cuda_available"] = True
                gpu_info["gpu_name"] = parts[0]
                gpu_info["vram_total_mb"] = float(parts[1])
                gpu_info["driver_version"] = parts[2]
    except Exception as e:
        gpu_info["notes"].append(f"nvidia-smi probe failed: {e}")

    # 2. Probe XGBoost CUDA capability
    if HAS_XGBOOST:
        try:
            dmat = xgb.DMatrix(np.zeros((4, 2)), label=np.zeros(4))
            bst = xgb.train({"device": "cuda", "tree_method": "hist"}, dmat, num_boost_round=1)
            gpu_info["xgboost_cuda_supported"] = True
        except Exception as e:
            gpu_info["xgboost_cuda_supported"] = False
            gpu_info["notes"].append(f"XGBoost CUDA check failed: {e}")
    else:
        gpu_info["notes"].append("XGBoost is not installed.")

    gpu_info["scikit_learn_models_device"] = "CPU (scikit-learn is CPU-based)"
    return gpu_info


# ---------------------------------------------------------------------------
# Candidate Model Configurations
# ---------------------------------------------------------------------------
def get_model_definitions(gpu_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the predefined candidate model suite."""
    models: list[dict[str, Any]] = [
        # 1. Verified Baseline Reference
        {
            "model_id": "lr_baseline_reference",
            "model_family": "Logistic Regression (Linear Reference)",
            "device": "cpu",
            "preprocessing": "StandardScaler()",
            "factory": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
            ]),
            "description": "StandardScaler + LogisticRegression(C=1.0, class_weight='balanced', lbfgs)",
        },
        # 2. Support Vector Machine (RBF Kernel)
        {
            "model_id": "svm_rbf",
            "model_family": "Support Vector Machine (RBF Kernel)",
            "device": "cpu",
            "preprocessing": "StandardScaler()",
            "factory": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", CalibratedClassifierCV(
                    SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42),
                    ensemble=False,
                )),
            ]),
            "description": "StandardScaler + CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, balanced), ensemble=False)",
        },
        # 3. Random Forest Classifier
        {
            "model_id": "random_forest",
            "model_family": "Random Forest",
            "device": "cpu",
            "preprocessing": "None (Tree invariant)",
            "factory": lambda: RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "description": "RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=3, balanced)",
        },
        # 4. Extra Trees Classifier
        {
            "model_id": "extra_trees",
            "model_family": "Extra Trees",
            "device": "cpu",
            "preprocessing": "None (Tree invariant)",
            "factory": lambda: ExtraTreesClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "description": "ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=3, balanced)",
        },
        # 5. Gradient Boosting Classifier
        {
            "model_id": "gradient_boosting",
            "model_family": "Gradient Boosting",
            "device": "cpu",
            "preprocessing": "None (Tree invariant)",
            "factory": lambda: GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            ),
            "description": "GradientBoostingClassifier(n_estimators=100, max_depth=3, lr=0.05, subsample=0.8)",
        },
        # 6. HistGradientBoosting Classifier
        {
            "model_id": "hist_gradient_boosting",
            "model_family": "Histogram-based Gradient Boosting",
            "device": "cpu",
            "preprocessing": "None (Tree invariant)",
            "factory": lambda: HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=3,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=42,
            ),
            "description": "HistGradientBoostingClassifier(max_iter=100, max_depth=3, lr=0.05, balanced)",
        },
        # 7. XGBoost (CPU Execution)
        {
            "model_id": "xgboost_cpu",
            "model_family": "XGBoost",
            "device": "cpu",
            "preprocessing": "None (Tree invariant)",
            "factory": lambda: xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                scale_pos_weight=(48.0 / 147.0),
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device="cpu",
                random_state=42,
            ),
            "description": "XGBClassifier(n_estimators=100, max_depth=3, lr=0.05, scale_pos_weight=0.326, device='cpu')",
        },
    ]

    # 8. XGBoost (CUDA GPU Execution - if available)
    if gpu_info.get("xgboost_cuda_supported"):
        models.append({
            "model_id": "xgboost_cuda",
            "model_family": "XGBoost",
            "device": "cuda",
            "preprocessing": "None (Tree invariant)",
            "factory": lambda: xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                scale_pos_weight=(48.0 / 147.0),
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device="cuda",
                random_state=42,
            ),
            "description": f"XGBClassifier(n_estimators=100, max_depth=3, lr=0.05, device='cuda') on {gpu_info.get('gpu_name')}",
        })

    return models


# ---------------------------------------------------------------------------
# Execution Runner
# ---------------------------------------------------------------------------
def run_model_comparisons() -> None:
    STAGE2B_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  Pathway A — Stage 2B: Nonlinear Model Family Evaluation")
    print("=" * 80)

    # 1. GPU Detection
    gpu_info = detect_gpu_environment()
    print("Hardware & Compute Environment:")
    print(f"  CUDA GPU Available : {gpu_info['cuda_available']}")
    if gpu_info["cuda_available"]:
        print(f"  GPU Name           : {gpu_info['gpu_name']} ({gpu_info['vram_total_mb']} MB VRAM)")
        print(f"  Driver Version     : {gpu_info['driver_version']}")
    print(f"  XGBoost CUDA Ready : {gpu_info['xgboost_cuda_supported']}")
    print(f"  Scikit-Learn Device: {gpu_info['scikit_learn_models_device']}")
    print("-" * 80)

    # 2. Dataset Loading & Validation
    X, y, groups, names, dataset_info = load_and_validate_dataset(RAW_DATA_PATH)
    print(f"Dataset verified: {len(X)} recordings across {groups.nunique()} participants.")

    models = get_model_definitions(gpu_info)
    print(f"Models to evaluate: {len(models)}")
    print(f"Evaluation: Repeated 4-fold SGKF (10 seeds, 40 folds) + 32-fold LOGO per model\n")

    all_fold_rows: list[dict[str, Any]] = []
    all_oof_rows: list[dict[str, Any]] = []
    comparison_summary_rows: list[dict[str, Any]] = []
    structured_results: dict[str, Any] = {
        "stage": "stage_2b_model_comparison",
        "dataset": dataset_info,
        "hardware_environment": gpu_info,
        "runtime_versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "xgboost": getattr(xgb, "__version__", None),
        },
        "models": [],
        "results": {},
    }

    for idx, m_meta in enumerate(models, 1):
        mid = m_meta["model_id"]
        mname = m_meta["model_family"]
        dev = m_meta["device"]
        print(f"[{idx}/{len(models)}] Benchmarking {mid} ({mname}) on [{dev.upper()}] ...")

        structured_results["models"].append({
            "model_id": mid,
            "model_family": mname,
            "device": dev,
            "preprocessing": m_meta["preprocessing"],
            "description": m_meta["description"],
        })

        t0 = time.time()
        # 1. Repeated 4-Fold SGKF
        sgkf_folds, sgkf_oofs, sgkf_seed_oofs = run_repeated_sgkf(
            X=X,
            y=y,
            groups=groups,
            names=names,
            pipeline_fn=m_meta["factory"],
            scheme_name=f"repeated_sgkf_4x10_{mid}",
        )
        sgkf_folds["model_id"] = mid
        sgkf_folds["device"] = dev
        sgkf_oofs["model_id"] = mid
        sgkf_oofs["device"] = dev

        # 2. 32-Fold LOGO
        logo_folds, logo_oofs, logo_pooled = run_logo(
            X=X,
            y=y,
            groups=groups,
            names=names,
            pipeline_fn=m_meta["factory"],
            scheme_name=f"logo_32_{mid}",
        )
        logo_folds["model_id"] = mid
        logo_folds["device"] = dev
        logo_folds["level"] = "recording"  # explicit level annotation
        logo_oofs["model_id"] = mid
        logo_oofs["device"] = dev
        elapsed = time.time() - t0

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

        structured_results["results"][mid] = {
            "elapsed_seconds": round(elapsed, 3),
            "repeated_sgkf_4x10": {
                "per_fold": sgkf_fold_summary,
                "per_seed_oof": sgkf_seed_summary,
            },
            "logo_32": logo_pooled,
        }

        # Flat rows for aggregate_comparison.csv
        for lvl in ("recording", "participant"):
            comparison_summary_rows.append({
                "model_id": mid,
                "model_family": mname,
                "device": dev,
                "evaluation": "repeated_sgkf_4x10_fold_mean",
                "level": lvl,
                **{m: sgkf_fold_summary[lvl][m]["mean"] for m in METRIC_NAMES},
            })
            comparison_summary_rows.append({
                "model_id": mid,
                "model_family": mname,
                "device": dev,
                "evaluation": "repeated_sgkf_4x10_fold_std",
                "level": lvl,
                **{m: sgkf_fold_summary[lvl][m]["std"] for m in METRIC_NAMES},
            })
            comparison_summary_rows.append({
                "model_id": mid,
                "model_family": mname,
                "device": dev,
                "evaluation": "repeated_sgkf_4x10_seed_oof_mean",
                "level": lvl,
                **{m: sgkf_seed_summary[lvl][m]["mean"] for m in METRIC_NAMES},
            })
            comparison_summary_rows.append({
                "model_id": mid,
                "model_family": mname,
                "device": dev,
                "evaluation": "repeated_sgkf_4x10_seed_oof_std",
                "level": lvl,
                **{m: sgkf_seed_summary[lvl][m]["std"] for m in METRIC_NAMES},
            })
            comparison_summary_rows.append({
                "model_id": mid,
                "model_family": mname,
                "device": dev,
                "evaluation": "logo_32_pooled_oof",
                "level": lvl,
                **{m: round(float(logo_pooled[lvl][m]), 4) for m in METRIC_NAMES},
            })

        print(f"   Elapsed: {elapsed:.2f}s")
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

    # Export outputs
    combined_folds_df = pd.concat(all_fold_rows, ignore_index=True)
    combined_oofs_df = pd.concat(all_oof_rows, ignore_index=True)
    comparison_df = pd.DataFrame(comparison_summary_rows)

    folds_csv_path = STAGE2B_RESULTS_DIR / "fold_metrics.csv"
    oof_csv_path = STAGE2B_RESULTS_DIR / "pooled_oof_predictions.csv"
    comparison_csv_path = STAGE2B_RESULTS_DIR / "aggregate_comparison.csv"
    config_json_path = STAGE2B_RESULTS_DIR / "experiment_config.json"

    combined_folds_df.to_csv(folds_csv_path, index=False)
    combined_oofs_df.to_csv(oof_csv_path, index=False)
    comparison_df.to_csv(comparison_csv_path, index=False)
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_results, f, indent=2)

    print("=" * 80)
    print("  Stage 2B Nonlinear Model Family Evaluation Complete")
    print("=" * 80)
    print(f"Fold metrics CSV       : {folds_csv_path} (shape={combined_folds_df.shape})")
    print(f"Pooled OOF CSV         : {oof_csv_path} (shape={combined_oofs_df.shape})")
    print(f"Comparison CSV         : {comparison_csv_path} (shape={comparison_df.shape})")
    print(f"Config & Summary JSON  : {config_json_path}")


if __name__ == "__main__":
    run_model_comparisons()
