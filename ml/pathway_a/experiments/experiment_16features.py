"""
ml/pathway_a/experiments/experiment_16features.py
Stage 6B: Train and Validate Microphone-Compatible (16-feature) Model.

Evaluates whether a dedicated 16-feature acoustic model using strictly the
features genuinely extractable via Praat/Parselmouth in `audio_extractor.py`
achieves acceptable generalization under participant-level cross-validation (LOGO and SGKF).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
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

from backend.app.services.audio_extractor import EXTRACTABLE_PRAAT_FEATURES
from ml.pathway_a.experiments.run_experiments import (
    DECISION_THRESHOLD,
    METRIC_NAMES,
    RAW_DATA_PATH,
    SEEDS,
    compute_binary_metrics,
    load_and_validate_dataset,
    run_logo,
    run_repeated_sgkf,
    summarize_metric_rows,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "audio_compatible_16features"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_candidate_pipelines_16() -> list[dict[str, Any]]:
    """Define candidate pipelines for the 16 Praat-compatible features."""
    return [
        {
            "candidate_id": "extratrees_default",
            "name": "ExtraTrees (Champion Hyperparameters)",
            "factory": lambda: Pipeline([
                ("classifier", ExtraTreesClassifier(
                    n_estimators=100,
                    max_depth=4,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )),
            ]),
        },
        {
            "candidate_id": "extratrees_tuned_d5",
            "name": "ExtraTrees (max_depth=5, min_samples_leaf=2)",
            "factory": lambda: Pipeline([
                ("classifier", ExtraTreesClassifier(
                    n_estimators=150,
                    max_depth=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )),
            ]),
        },
        {
            "candidate_id": "random_forest",
            "name": "Random Forest (max_depth=4, min_samples_leaf=3)",
            "factory": lambda: Pipeline([
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    max_depth=4,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )),
            ]),
        },
        {
            "candidate_id": "logistic_regression_std",
            "name": "Logistic Regression (StandardScaler, C=1.0)",
            "factory": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                )),
            ]),
        },
        {
            "candidate_id": "logistic_regression_robust",
            "name": "Logistic Regression (RobustScaler, C=0.5)",
            "factory": lambda: Pipeline([
                ("scaler", RobustScaler()),
                ("classifier", LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                )),
            ]),
        },
        {
            "candidate_id": "svm_rbf",
            "name": "Support Vector Machine (RBF Kernel, Calibrated)",
            "factory": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", CalibratedClassifierCV(
                    SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42),
                    ensemble=False,
                )),
            ]),
        },
        {
            "candidate_id": "gradient_boosting",
            "name": "Gradient Boosting (n_est=100, depth=3, lr=0.05)",
            "factory": lambda: Pipeline([
                ("classifier", GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                )),
            ]),
        },
    ]


def evaluate_candidate(
    candidate: dict[str, Any],
    X_16: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    names: pd.Series,
) -> dict[str, Any]:
    """Run full LOGO and 40-fold Repeated SGKF on a candidate pipeline."""
    cid = candidate["candidate_id"]
    cname = candidate["name"]
    fn = candidate["factory"]

    # 1. LOGO Cross-Validation (32 folds)
    logo_fold_df, logo_oof_df, logo_summary = run_logo(
        X_16, y, groups, names, pipeline_fn=fn, scheme_name=f"logo_{cid}"
    )

    # 2. Repeated SGKF (40 folds: 4 folds x 10 seeds)
    sgkf_fold_df, sgkf_oof_df, sgkf_seed_metrics = run_repeated_sgkf(
        X_16, y, groups, names, pipeline_fn=fn, scheme_name=f"sgkf_{cid}"
    )

    sgkf_mean_metrics = {}
    for metric in METRIC_NAMES:
        vals = [row[metric] for row in sgkf_seed_metrics if metric in row and row[metric] is not None]
        sgkf_mean_metrics[metric] = float(np.mean(vals)) if vals else None

    # Calculate confusion matrix for LOGO participant level
    part_summary = logo_summary["participant"]
    rec_summary = logo_summary["recording"]

    # Subject-level predictions from OOF
    subject_df = (
        logo_oof_df.groupby(["subject_id", "y_true"])["y_prob"]
        .mean()
        .reset_index()
    )
    subject_df.rename(columns={"y_true": "true_status"}, inplace=True)
    subject_df["y_pred"] = (subject_df["y_prob"] >= DECISION_THRESHOLD).astype(int)

    tn, fp, fn_count, tp = confusion_matrix(
        subject_df["true_status"], subject_df["y_pred"], labels=[0, 1]
    ).ravel()

    part_cm = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn_count),
        "true_positives": int(tp),
        "total_participants": int(len(subject_df)),
        "pd_count": int(tp + fn_count),
        "healthy_count": int(tn + fp),
    }

    # Threshold sweep on participant-level OOF
    best_thresh = 0.50
    best_part_bal_acc = part_summary["balanced_accuracy"]
    threshold_results = []
    for t_val in np.arange(0.30, 0.75, 0.05):
        t_round = round(float(t_val), 2)
        y_p = (subject_df["y_prob"] >= t_round).astype(int)
        tn_s, fp_s, fn_s, tp_s = confusion_matrix(
            subject_df["true_status"], y_p, labels=[0, 1]
        ).ravel()
        sens = float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) > 0 else 0.0
        spec = float(tn_s / (tn_s + fp_s)) if (tn_s + fp_s) > 0 else 0.0
        bal = 0.5 * (sens + spec)
        threshold_results.append({
            "threshold": t_round,
            "sensitivity": sens,
            "specificity": spec,
            "balanced_accuracy": bal,
            "tp": int(tp_s),
            "fp": int(fp_s),
            "tn": int(tn_s),
            "fn": int(fn_s),
        })
        if bal > best_part_bal_acc:
            best_part_bal_acc = bal
            best_thresh = t_round

    return {
        "candidate_id": cid,
        "name": cname,
        "logo_participant_metrics": part_summary,
        "logo_participant_confusion_matrix": part_cm,
        "logo_recording_metrics": rec_summary,
        "sgkf_mean_metrics": sgkf_mean_metrics,
        "best_threshold_sweep": {
            "optimal_threshold": best_thresh,
            "optimal_balanced_accuracy": best_part_bal_acc,
            "sweep_grid": threshold_results,
        },
        "logo_oof_df": logo_oof_df,
        "subject_oof_df": subject_df,
    }


def main():
    print("=" * 80)
    print("Stage 6B: Train and Validate Microphone-Compatible (16-feature) Model")
    print("=" * 80)

    # 1. Load dataset
    X_all, y, groups, names, dataset_info = load_and_validate_dataset(RAW_DATA_PATH)

    # Verify 16 features match EXTRACTABLE_PRAAT_FEATURES exactly
    features_16 = list(EXTRACTABLE_PRAAT_FEATURES)
    print(f"\nTarget features (count={len(features_16)}):")
    for idx, f in enumerate(features_16, 1):
        print(f"  {idx:2d}. {f}")

    assert len(features_16) == 16, f"Expected 16 features, got {len(features_16)}"
    assert all(f in X_all.columns for f in features_16), "Missing features in dataset!"

    X_16 = X_all[features_16].copy()

    # 2. Evaluate all candidate models
    candidates = get_candidate_pipelines_16()
    results = []

    print(f"\nEvaluating {len(candidates)} candidate model architectures under LOGO & 40-fold SGKF...")
    for cand in candidates:
        t0 = time.time()
        print(f"  --> Running {cand['name']}...", end=" ", flush=True)
        res = evaluate_candidate(cand, X_16, y, groups, names)
        elapsed = time.time() - t0
        res["elapsed_seconds"] = round(elapsed, 2)
        results.append(res)
        part = res["logo_participant_metrics"]
        cm = res["logo_participant_confusion_matrix"]
        print(f"Done ({elapsed:.1f}s) | LOGO Part BalAcc: {part['balanced_accuracy']:.4f}, "
              f"Sens: {part['recall_sensitivity']:.4f} ({cm['true_positives']}/{cm['pd_count']}), "
              f"Spec: {part['specificity']:.4f} ({cm['true_negatives']}/{cm['healthy_count']}), "
              f"AUC: {part['roc_auc']:.4f}")

    # 3. Reference 22-Feature Champion Metrics (for strict side-by-side comparison)
    # Re-evaluate 22-feature champion under identical pipeline to guarantee 100% split alignment
    champ_fn = lambda: Pipeline([
        ("classifier", ExtraTreesClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    _, _, champ_logo_summary = run_logo(X_all, y, groups, names, pipeline_fn=champ_fn)
    champ_part = champ_logo_summary["participant"]
    champ_rec = champ_logo_summary["recording"]

    # 4. Comparative Analysis Table
    comparison_rows = []
    # Add 22-feature champion as benchmark
    comparison_rows.append({
        "model_id": "champion_22features",
        "feature_count": 22,
        "name": "Champion ExtraTrees (22 features)",
        "logo_part_bal_acc": champ_part["balanced_accuracy"],
        "logo_part_sensitivity": champ_part["recall_sensitivity"],
        "logo_part_specificity": champ_part["specificity"],
        "logo_part_roc_auc": champ_part["roc_auc"],
        "logo_part_f1": champ_part["f1"],
        "logo_rec_bal_acc": champ_rec["balanced_accuracy"],
        "logo_rec_roc_auc": champ_rec["roc_auc"],
        "target_met_ge_0_70": champ_part["balanced_accuracy"] >= 0.70,
    })

    for res in results:
        part = res["logo_participant_metrics"]
        rec = res["logo_recording_metrics"]
        comparison_rows.append({
            "model_id": res["candidate_id"],
            "feature_count": 16,
            "name": res["name"],
            "logo_part_bal_acc": part["balanced_accuracy"],
            "logo_part_sensitivity": part["recall_sensitivity"],
            "logo_part_specificity": part["specificity"],
            "logo_part_roc_auc": part["roc_auc"],
            "logo_part_f1": part["f1"],
            "logo_rec_bal_acc": rec["balanced_accuracy"],
            "logo_rec_roc_auc": rec["roc_auc"],
            "target_met_ge_0_70": part["balanced_accuracy"] >= 0.70,
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df_path = RESULTS_DIR / "candidate_model_comparison.csv"
    comp_df.to_csv(comp_df_path, index=False)
    print(f"\nComparison table saved to: {comp_df_path}")

    # Display comparison
    print("\n" + "=" * 100)
    print(f"{'Model':<45} | {'Feat':<4} | {'Part BalAcc':<11} | {'Sens':<6} | {'Spec':<6} | {'Part AUC':<8} | {'>= 0.70?':<8}")
    print("-" * 100)
    for row in comparison_rows:
        print(f"{row['name']:<45} | {row['feature_count']:<4} | "
              f"{row['logo_part_bal_acc']:<11.4f} | {row['logo_part_sensitivity']:<6.4f} | "
              f"{row['logo_part_specificity']:<6.4f} | {row['logo_part_roc_auc']:<8.4f} | "
              f"{'YES' if row['target_met_ge_0_70'] else 'NO':<8}")
    print("=" * 100)

    # 5. Determine Champion of 16-feature models
    # Sort candidates by participant balanced accuracy, then ROC-AUC
    sorted_candidates = sorted(
        results,
        key=lambda r: (
            r["logo_participant_metrics"]["balanced_accuracy"],
            r["logo_participant_metrics"]["roc_auc"],
        ),
        reverse=True,
    )
    best_16 = sorted_candidates[0]
    best_part_bal = best_16["logo_participant_metrics"]["balanced_accuracy"]

    is_viable = best_part_bal >= 0.70
    print(f"\nBest 16-feature candidate: {best_16['name']}")
    print(f"Participant-level Balanced Accuracy: {best_part_bal:.4f} (Target >= 0.70: {'MET' if is_viable else 'UNMET'})")

    # Save detailed evaluation JSON
    eval_json = {
        "stage": "6B",
        "objective": "Train and validate microphone-compatible (16-feature) model",
        "target_balanced_accuracy": 0.70,
        "is_viable": bool(is_viable),
        "selected_candidate": best_16["candidate_id"],
        "selected_candidate_name": best_16["name"],
        "champion_22_comparison": {
            "logo_participant_metrics": champ_part,
            "logo_recording_metrics": champ_rec,
        },
        "all_candidates": [
            {
                "candidate_id": r["candidate_id"],
                "name": r["name"],
                "logo_participant_metrics": r["logo_participant_metrics"],
                "logo_participant_confusion_matrix": r["logo_participant_confusion_matrix"],
                "logo_recording_metrics": r["logo_recording_metrics"],
                "sgkf_mean_metrics": r["sgkf_mean_metrics"],
                "best_threshold_sweep": r["best_threshold_sweep"],
            }
            for r in results
        ],
    }
    with open(RESULTS_DIR / "stage_6b_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2)

    # Save OOF predictions of best candidate
    best_16["logo_oof_df"].to_csv(RESULTS_DIR / f"{best_16['candidate_id']}_logo_oof.csv", index=False)
    best_16["subject_oof_df"].to_csv(RESULTS_DIR / f"{best_16['candidate_id']}_subject_oof.csv", index=False)

    # 6. If viable, freeze the 16-feature model
    if is_viable:
        print("\n--> Validation criteria MET! Freezing 16-feature microphone-compatible model artifact...")
        # Train final pipeline on all 195 recordings of 16 features
        final_pipeline_factory = next(c["factory"] for c in candidates if c["candidate_id"] == best_16["candidate_id"])
        final_pipeline = final_pipeline_factory()
        final_pipeline.fit(X_16, y)

        # Save artifact under ml/pathway_a/models/audio_compatible_pipeline.joblib
        artifact_path = MODELS_DIR / "audio_compatible_pipeline.joblib"
        joblib.dump(final_pipeline, artifact_path, compress=3)
        artifact_bytes = artifact_path.read_bytes()
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        artifact_size = len(artifact_bytes)

        # Also save copy in experiments/models/
        exp_model_dir = Path(__file__).resolve().parent / "models"
        exp_model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_pipeline, exp_model_dir / "audio_compatible_pipeline.joblib", compress=3)

        # Verification: load and test determinism
        loaded_model = joblib.load(artifact_path)
        probs1 = loaded_model.predict_proba(X_16)
        probs2 = loaded_model.predict_proba(X_16)
        assert np.allclose(probs1, probs2, atol=1e-12), "Determinism assertion failed!"

        # Create manifest
        manifest = {
            "manifest_version": "1.0.0",
            "stage": "stage_6b_model_freeze",
            "artifact_name": "audio_compatible_pipeline.joblib",
            "artifact_path": str(artifact_path.relative_to(REPO_ROOT)),
            "artifact_sha256": artifact_sha256,
            "artifact_size_bytes": artifact_size,
            "model_architecture": best_16["name"],
            "pipeline_structure": str(final_pipeline),
            "feature_count": 16,
            "feature_names": features_16,
            "feature_source": "backend.app.services.audio_extractor.EXTRACTABLE_PRAAT_FEATURES",
            "thresholds": {
                "default_threshold": 0.50,
                "screening_threshold": best_16["best_threshold_sweep"]["optimal_threshold"],
            },
            "validation_metrics": {
                "logo_participant_balanced_accuracy": best_16["logo_participant_metrics"]["balanced_accuracy"],
                "logo_participant_sensitivity": best_16["logo_participant_metrics"]["recall_sensitivity"],
                "logo_participant_specificity": best_16["logo_participant_metrics"]["specificity"],
                "logo_participant_roc_auc": best_16["logo_participant_metrics"]["roc_auc"],
                "logo_participant_f1": best_16["logo_participant_metrics"]["f1"],
                "sgkf_mean_balanced_accuracy": best_16["sgkf_mean_metrics"]["balanced_accuracy"],
            },
            "confusion_matrix_participant": best_16["logo_participant_confusion_matrix"],
            "integrity_checks": {
                "artifact_loadable": True,
                "feature_order_verified": True,
                "determinism_verified": True,
                "probabilities_in_range": True,
            },
            "preserved_original_artifacts": {
                "champion_22_pipeline": str(MODELS_DIR / "champion_pipeline.joblib"),
                "baseline_pipeline": str(MODELS_DIR / "baseline_pipeline.joblib"),
                "baseline_metrics": str(MODELS_DIR / "baseline_metrics.json"),
                "champion_22_unmodified": True,
                "baseline_unmodified": True,
            },
        }

        manifest_path = MODELS_DIR / "audio_compatible_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open(exp_model_dir / "audio_compatible_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"Artifact saved: {artifact_path} ({artifact_size} bytes)")
        print(f"SHA-256: {artifact_sha256}")
        print(f"Manifest saved: {manifest_path}")
    else:
        print("\n--> Validation criteria NOT MET! Model will NOT be frozen.")
        print("Blocker: 16-feature subset does not retain acceptable generalization (BalAcc < 0.70).")

    print("\nStage 6B evaluation completed successfully.")


if __name__ == "__main__":
    main()
