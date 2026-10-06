"""
Comprehensive 5-Fold Outer Cross-Validation of Fine-Tuned TabPFN (Run 2).
Enforces strict experiment_id GroupKFold using existing grouped_cv_assignments.csv.
Internal validation is strictly partitioned by experiment_id from outer training cohorts.
Saves out-of-fold predictions, fold-specific model artifacts, comparisons, and publication-ready figures.
Includes robust fit handling, artifact verification, resume capability, and UTF-8 encoding protection.
"""

import os
import sys

# Configure UTF-8 encoding immediately for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import json
import logging
import traceback
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
import torch
import tabpfn
from tabpfn.finetuning import FinetunedTabPFNRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure root logger handles UTF-8 safely
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TabPFN_FineTuning_Run2")

print("=" * 80, flush=True)
print("PHASE 4.3: 5-FOLD FINE-TUNED TABPFN CROSS-VALIDATION PIPELINE (RUN 2)", flush=True)
print("=" * 80, flush=True)

# 1. Setup paths
ROOT = Path(".").resolve()
DATA_CSV = ROOT / "data" / "tabpfn_concrete_v1" / "tabpfn_concrete_model.csv"
DEV_ROWS_CSV = ROOT / "data" / "splits" / "development_rows.csv"
CV_ASSIGN_CSV = ROOT / "data" / "splits" / "grouped_cv_assignments.csv"
CKPT_PATH = ROOT / "models" / "tabpfn-v2.5-regressor-v2.5_real.ckpt"

MODELS_DIR = ROOT / "results" / "models" / "tabpfn" / "fold_models_run2"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PREDS_DIR = ROOT / "results" / "predictions"
PREDS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_RUN2_DIR = ROOT / "results" / "tabpfn_finetuned_run2"
RESULTS_RUN2_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = RESULTS_RUN2_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = ROOT / "reports" / "figures" / "tabpfn_finetuned_run2"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "reports"

# 2. Pre-flight verification
assert DATA_CSV.exists(), f"Missing modeling dataset: {DATA_CSV}"
assert DEV_ROWS_CSV.exists(), f"Missing development rows: {DEV_ROWS_CSV}"
assert CV_ASSIGN_CSV.exists(), f"Missing grouped CV assignments: {CV_ASSIGN_CSV}"
assert CKPT_PATH.exists(), f"Missing TabPFN checkpoint: {CKPT_PATH}"

df_model = pd.read_csv(DATA_CSV)
df_dev = pd.read_csv(DEV_ROWS_CSV)
df_cv = pd.read_csv(CV_ASSIGN_CSV)

dev_sample_ids = set(df_dev["sample_id"])
dev_df = df_model[df_model["sample_id"].isin(dev_sample_ids)].copy().reset_index(drop=True)

print(f"Loaded Development Partition: {len(dev_df)} rows, {dev_df['experiment_id'].nunique()} cohorts", flush=True)
assert len(dev_df) == 2800, f"Expected 2800 rows, got {len(dev_df)}"

# Merge exact fold assignment from grouped_cv_assignments.csv
sample_to_fold = dict(zip(df_cv["sample_id"], df_cv["cv_fold"]))
dev_df["cv_fold"] = dev_df["sample_id"].map(sample_to_fold)
assert not dev_df["cv_fold"].isna().any(), "Some development rows missing fold assignment!"

print("CV Fold assignments verification:")
for f in range(1, 6):
    f_rows = (dev_df["cv_fold"] == f).sum()
    f_cohorts = dev_df.loc[dev_df["cv_fold"] == f, "experiment_id"].nunique()
    print(f"  Fold {f}: {f_rows} rows, {f_cohorts} cohorts", flush=True)

# 3. Features: Official V2_ENGINEERED
FEATURES = [
    "curing_age_days",
    "concrete_type",
    "bacterial_concentration_cells_ml",
    "mechanical_property",
    "age_log",
    "age_sqrt",
    "bacterial_present",
    "bacterial_concentration_log",
    "age_x_bacterial",
    "age_log_x_bacterial",
    "age_squared"
]
TARGET = "strength_mpa"
GROUP_COL = "experiment_id"

print(f"Feature set: V2_ENGINEERED ({len(FEATURES)} features): {FEATURES}", flush=True)

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-6, 1e-6, y_true))) * 100
    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 6),
        "mape": round(float(mape), 4),
        "n": int(len(y_true))
    }

oof_predictions = []
fold_results = []
total_start_time = time.time()

# 4. Iterate over 5 folds sequentially
for fold_idx in range(1, 6):
    fold_start = time.time()
    print(f"\n{'=' * 30} OUTER FOLD {fold_idx}/5 {'=' * 30}", flush=True)

    val_mask = dev_df["cv_fold"] == fold_idx
    train_mask = ~val_mask

    outer_train = dev_df.loc[train_mask].copy().reset_index(drop=True)
    outer_val = dev_df.loc[val_mask].copy().reset_index(drop=True)

    # Assert zero cohort leakage between outer train and outer val
    train_cohorts = set(outer_train[GROUP_COL].unique())
    val_cohorts = set(outer_val[GROUP_COL].unique())
    overlap = train_cohorts.intersection(val_cohorts)
    assert len(overlap) == 0, f"LEAKAGE DETECTED in Fold {fold_idx}: {overlap}"

    print(f"Outer Train: {len(outer_train)} rows ({len(train_cohorts)} cohorts)", flush=True)
    print(f"Outer Val:   {len(outer_val)} rows ({len(val_cohorts)} cohorts): {sorted(val_cohorts)}", flush=True)

    # Internal validation split from outer train using experiment_id grouping
    # Hold out first 2 sorted cohorts from outer train as internal validation
    sorted_train_cohorts = sorted(list(train_cohorts))
    internal_val_cohorts = set(sorted_train_cohorts[:2])
    internal_train_cohorts = set(sorted_train_cohorts[2:])

    internal_train_mask = outer_train[GROUP_COL].isin(internal_train_cohorts)
    internal_val_mask = outer_train[GROUP_COL].isin(internal_val_cohorts)

    X_train_int = outer_train.loc[internal_train_mask, FEATURES].copy()
    y_train_int = outer_train.loc[internal_train_mask, TARGET].values
    X_val_int = outer_train.loc[internal_val_mask, FEATURES].copy()
    y_val_int = outer_train.loc[internal_val_mask, TARGET].values

    print(f"  Internal Train: {len(X_train_int)} rows ({len(internal_train_cohorts)} cohorts)", flush=True)
    print(f"  Internal Val:   {len(X_val_int)} rows ({len(internal_val_cohorts)} cohorts)", flush=True)

    fold_model_dir = MODELS_DIR / f"fold_{fold_idx}"
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    fold_model_file = fold_model_dir / f"fold_{fold_idx}_finetuned.joblib"
    fold_config_file = fold_model_dir / f"fold_{fold_idx}_config.json"
    fold_val_preds_file = fold_model_dir / f"fold_{fold_idx}_val_predictions.csv"

    # Check if this fold was already completed and serialized (Resume support)
    fold_completed = False
    if fold_model_file.exists() and fold_config_file.exists() and fold_val_preds_file.exists():
        try:
            print(f"  Found existing artifact for Fold {fold_idx}. Verifying integrity...", flush=True)
            loaded_reg = joblib.load(fold_model_file)
            saved_preds_df = pd.read_csv(fold_val_preds_file)
            if len(saved_preds_df) == len(outer_val):
                with open(fold_config_file, "r") as cf:
                    fold_cfg = json.load(cf)
                print(f"  Verified existing Fold {fold_idx} artifact! Reusing completed fold.", flush=True)
                ft_reg = loaded_reg
                y_outer_pred = saved_preds_df["strength_mpa_pred"].values
                int_val_metrics = fold_cfg.get("internal_val_metrics", {})
                fit_duration = fold_cfg.get("fit_duration_sec", 0.0)
                pred_duration = fold_cfg.get("pred_duration_sec", 0.0)
                fold_completed = True
        except Exception as e:
            print(f"  Warning: Existing artifact verification failed ({e}). Retraining Fold {fold_idx}...", flush=True)
            fold_completed = False

    if not fold_completed:
        # Initialize outer-fold-specific fine-tuner
        ft_reg = FinetunedTabPFNRegressor(
            device="cpu",
            epochs=3,
            learning_rate=5e-6,
            weight_decay=0.01,
            grad_clip_value=1.0,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_final_inference=2,
            validation_split_ratio=None,  # Explicit internal validation provided
            early_stopping=False,
            extra_regressor_kwargs={
                "model_path": str(CKPT_PATH),
                "ignore_pretraining_limits": True,
                "random_state": 42 + fold_idx,
            }
        )

        print(f"  [START FIT] Fine-tuning TabPFN on Outer Fold {fold_idx}...", flush=True)
        t0_fit = time.time()
        try:
            ft_reg.fit(
                X_train_int, y_train_int,
                X_val=X_val_int, y_val=y_val_int,
                output_dir=fold_model_dir
            )
            fit_duration = time.time() - t0_fit
            print(f"  [FIT COMPLETE] Fine-tuning finished in {fit_duration:.1f}s ({fit_duration/60:.2f} mins)", flush=True)
            assert ft_reg is not None, "Model fit failed to return model instance"
        except Exception as fit_err:
            print(f"  [FIT ERROR] Fold {fold_idx} fit failed: {fit_err}", flush=True)
            traceback.print_exc()
            raise fit_err

        # Save fold model artifact
        print(f"  Serializing model artifact to {fold_model_file}...", flush=True)
        joblib.dump(ft_reg, fold_model_file)
        assert fold_model_file.exists(), f"Model file was not created: {fold_model_file}"
        print(f"  Saved fold model artifact: {fold_model_file.name} ({os.path.getsize(fold_model_file):,} bytes)", flush=True)

        # Verify saved artifact can be loaded
        print(f"  Verifying artifact loadability...", flush=True)
        test_loaded = joblib.load(fold_model_file)
        assert test_loaded is not None, "Failed to load back serialized model artifact!"
        print(f"  Artifact verified successfully!", flush=True)

        # Evaluate on the internal validation set for overfitting analysis
        print(f"  Evaluating internal validation set...", flush=True)
        y_val_int_pred = ft_reg.predict(X_val_int)
        int_val_metrics = compute_metrics(y_val_int, y_val_int_pred)
        print(f"  Internal Val Metrics Fold {fold_idx}: MAE={int_val_metrics['mae']:.4f} RMSE={int_val_metrics['rmse']:.4f} R2={int_val_metrics['r2']:.4f}", flush=True)

        # Predict on untouched outer validation set
        X_outer_val = outer_val[FEATURES].copy()
        y_outer_val = outer_val[TARGET].values

        print(f"  Generating predictions for {len(X_outer_val)} outer validation samples...", flush=True)
        t0_pred = time.time()
        y_outer_pred = ft_reg.predict(X_outer_val)
        pred_duration = time.time() - t0_pred
        print(f"  Prediction completed in {pred_duration:.2f}s", flush=True)
        assert len(y_outer_pred) == len(outer_val), f"Expected {len(outer_val)} predictions, got {len(y_outer_pred)}"

        # Save fold validation predictions
        val_pred_df = outer_val.copy()
        val_pred_df["strength_mpa_pred"] = np.round(y_outer_pred, 4)
        val_pred_df["residual"] = np.round(y_outer_pred - y_outer_val, 4)
        val_pred_df.to_csv(fold_val_preds_file, index=False)

        # Save fold configuration JSON
        software_versions = {
            "python": sys.version,
            "torch": torch.__version__,
            "tabpfn": getattr(tabpfn, "__version__", "unknown"),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__
        }
        fold_config = {
            "fold": fold_idx,
            "outer_train_rows": len(outer_train),
            "outer_val_rows": len(outer_val),
            "outer_train_cohorts": sorted(list(train_cohorts)),
            "outer_val_cohorts": sorted(list(val_cohorts)),
            "internal_train_cohorts": sorted(list(internal_train_cohorts)),
            "internal_val_cohorts": sorted(list(internal_val_cohorts)),
            "features": FEATURES,
            "target": TARGET,
            "group_col": GROUP_COL,
            "hyperparameters": {
                "epochs": 3,
                "learning_rate": 5e-6,
                "weight_decay": 0.01,
                "grad_clip_value": 1.0,
                "n_estimators_finetune": 1,
                "n_estimators_validation": 1,
                "n_estimators_final_inference": 2,
                "base_checkpoint": str(CKPT_PATH),
                "device": "cpu"
            },
            "internal_val_metrics": int_val_metrics,
            "fit_duration_sec": round(fit_duration, 1),
            "pred_duration_sec": round(pred_duration, 1),
            "model_artifact": str(fold_model_file),
            "software_versions": software_versions
        }
        with open(fold_config_file, "w") as f:
            json.dump(fold_config, f, indent=2)

    # Compute outer fold metrics
    y_outer_val = outer_val[TARGET].values
    f_metrics = compute_metrics(y_outer_val, y_outer_pred)
    fold_total_time = time.time() - fold_start
    print(f"  Outer Val Metrics Fold {fold_idx}: MAE={f_metrics['mae']:.4f} RMSE={f_metrics['rmse']:.4f} R2={f_metrics['r2']:.4f} MAPE={f_metrics['mape']:.2f}% (Total Fold Time: {fold_total_time/60:.2f} min)", flush=True)

    # Per-property metrics for this fold
    f_prop_metrics = {}
    for prop in ["Compressive Strength", "Flexural Strength", "Split Tensile Strength"]:
        p_mask = outer_val["mechanical_property"] == prop
        if p_mask.sum() > 0:
            pm = compute_metrics(y_outer_val[p_mask], y_outer_pred[p_mask])
            f_prop_metrics[prop] = pm
            print(f"    Fold {fold_idx} {prop[:15]}: MAE={pm['mae']:.4f} RMSE={pm['rmse']:.4f} R2={pm['r2']:.4f} (n={pm['n']})", flush=True)

    fold_results.append({
        "fold": fold_idx,
        "n_samples": len(outer_val),
        "fit_time_sec": round(fit_duration, 1),
        "pred_time_sec": round(pred_duration, 1),
        "internal_val_mae": int_val_metrics.get("mae", np.nan),
        "internal_val_rmse": int_val_metrics.get("rmse", np.nan),
        "internal_val_r2": int_val_metrics.get("r2", np.nan),
        "per_property": f_prop_metrics,
        **f_metrics
    })

    # Store out-of-fold predictions
    for i in range(len(outer_val)):
        row = outer_val.iloc[i]
        oof_predictions.append({
            "model": "TabPFN_FineTuned_v2.5",
            "feature_set": "V2_ENGINEERED",
            "fold": fold_idx,
            "sample_id": row["sample_id"],
            "experiment_id": row["experiment_id"],
            "sample_replicate": row["sample_replicate"],
            "concrete_type": row["concrete_type"],
            "curing_age_days": row["curing_age_days"],
            "mechanical_property": row["mechanical_property"],
            "is_restored_value": row["is_restored_value"],
            "strength_mpa_actual": float(row["strength_mpa"]),
            "strength_mpa_pred": round(float(y_outer_pred[i]), 4),
            "residual": round(float(y_outer_pred[i] - row["strength_mpa"]), 4)
        })

total_pipeline_time = time.time() - total_start_time
print("\n" + "=" * 80, flush=True)
print(f"ALL 5 OUTER FOLDS COMPLETED in {total_pipeline_time:.1f}s ({total_pipeline_time/60:.2f} mins)", flush=True)
print("=" * 80, flush=True)

# 5. Save out-of-fold predictions CSV (Run 2)
oof_df = pd.DataFrame(oof_predictions)
oof_csv_path = PREDS_DIR / "tabpfn_finetuned_oof_predictions_run2.csv"
oof_df.to_csv(oof_csv_path, index=False)
print(f"Saved OOF predictions (Run 2): {oof_csv_path} ({len(oof_df):,} rows)", flush=True)
assert len(oof_df) == 2800, f"Expected exactly 2800 OOF predictions, got {len(oof_df)}"
assert oof_df["sample_id"].nunique() == 2800, "Duplicate sample IDs detected in OOF predictions!"

# 6. Compute POOLED Out-of-Fold Metrics
pooled_overall = compute_metrics(oof_df["strength_mpa_actual"], oof_df["strength_mpa_pred"])
print("\n--- POOLED OUT-OF-FOLD OVERALL METRICS ---", flush=True)
print(f"Overall MAE:  {pooled_overall['mae']:.4f} MPa", flush=True)
print(f"Overall RMSE: {pooled_overall['rmse']:.4f} MPa", flush=True)
print(f"Overall R2:   {pooled_overall['r2']:.6f}", flush=True)
print(f"Overall MAPE: {pooled_overall['mape']:.2f}%", flush=True)

# Per-property pooled metrics
per_property_metrics = {}
for prop in ["Compressive Strength", "Flexural Strength", "Split Tensile Strength"]:
    sub = oof_df[oof_df["mechanical_property"] == prop]
    pm = compute_metrics(sub["strength_mpa_actual"], sub["strength_mpa_pred"])
    per_property_metrics[prop] = pm
    print(f"\n{prop:25s}: MAE={pm['mae']:.4f} RMSE={pm['rmse']:.4f} R2={pm['r2']:.4f} MAPE={pm['mape']:.2f}% (n={pm['n']})", flush=True)

# Fold stability summary
fold_maes = [r["mae"] for r in fold_results]
fold_rmses = [r["rmse"] for r in fold_results]
fold_r2s = [r["r2"] for r in fold_results]
mean_mae = np.mean(fold_maes)
std_mae = np.std(fold_maes)
mean_rmse = np.mean(fold_rmses)
std_rmse = np.std(fold_rmses)
mean_r2 = np.mean(fold_r2s)
std_r2 = np.std(fold_r2s)
print(f"\nFold Stability across 5 outer folds:", flush=True)
print(f"  MAE:  {mean_mae:.4f} +/- {std_mae:.4f} (folds: {[round(m, 4) for m in fold_maes]})", flush=True)
print(f"  RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f} (folds: {[round(r, 4) for r in fold_rmses]})", flush=True)
print(f"  R2:   {mean_r2:.4f} +/- {std_r2:.4f} (folds: {[round(r, 4) for r in fold_r2s]})", flush=True)

# 7. Comparison Table Generation
pre_df_all = pd.read_csv("results/predictions/tabpfn_cv_predictions.csv")
pre_df = pre_df_all[pre_df_all["feature_set"] == "V2_ENGINEERED"].copy()

pre_pooled_overall = compute_metrics(pre_df["strength_mpa_actual"], pre_df["strength_mpa_pred"])
pre_prop_metrics = {}
for prop in ["Compressive Strength", "Flexural Strength", "Split Tensile Strength"]:
    sub = pre_df[pre_df["mechanical_property"] == prop]
    pre_prop_metrics[prop] = compute_metrics(sub["strength_mpa_actual"], sub["strength_mpa_pred"])

# Load classical baselines predictions for exact pooled metrics
p41_preds = pd.read_csv("results/predictions/phase4_1_cv_predictions.csv")

comparison_rows = []

# Fine-Tuned TabPFN (Run 2)
comparison_rows.append({
    "model": "Fine-Tuned TabPFN v2.5",
    "feature_set": "V2_ENGINEERED",
    "strategy": "Unified",
    "compressive_mae": per_property_metrics["Compressive Strength"]["mae"],
    "compressive_rmse": per_property_metrics["Compressive Strength"]["rmse"],
    "compressive_r2": per_property_metrics["Compressive Strength"]["r2"],
    "flexural_mae": per_property_metrics["Flexural Strength"]["mae"],
    "flexural_rmse": per_property_metrics["Flexural Strength"]["rmse"],
    "flexural_r2": per_property_metrics["Flexural Strength"]["r2"],
    "tensile_mae": per_property_metrics["Split Tensile Strength"]["mae"],
    "tensile_rmse": per_property_metrics["Split Tensile Strength"]["rmse"],
    "tensile_r2": per_property_metrics["Split Tensile Strength"]["r2"],
    "overall_mae": pooled_overall["mae"],
    "overall_rmse": pooled_overall["rmse"],
    "overall_r2": pooled_overall["r2"]
})

# Pretrained TabPFN
comparison_rows.append({
    "model": "Pretrained TabPFN v2.5",
    "feature_set": "V2_ENGINEERED",
    "strategy": "Unified",
    "compressive_mae": pre_prop_metrics["Compressive Strength"]["mae"],
    "compressive_rmse": pre_prop_metrics["Compressive Strength"]["rmse"],
    "compressive_r2": pre_prop_metrics["Compressive Strength"]["r2"],
    "flexural_mae": pre_prop_metrics["Flexural Strength"]["mae"],
    "flexural_rmse": pre_prop_metrics["Flexural Strength"]["rmse"],
    "flexural_r2": pre_prop_metrics["Flexural Strength"]["r2"],
    "tensile_mae": pre_prop_metrics["Split Tensile Strength"]["mae"],
    "tensile_rmse": pre_prop_metrics["Split Tensile Strength"]["rmse"],
    "tensile_r2": pre_prop_metrics["Split Tensile Strength"]["r2"],
    "overall_mae": pre_pooled_overall["mae"],
    "overall_rmse": pre_pooled_overall["rmse"],
    "overall_r2": pre_pooled_overall["r2"]
})

# Classical Baselines
classical_specs = [
    ("XGBoost", "core", "unified"),
    ("XGBoost", "expanded", "unified"),
    ("RandomForest", "core", "unified"),
    ("RandomForest", "core", "separate"),
    ("CatBoost", "core", "unified"),
    ("CatBoost", "expanded", "unified"),
    ("Ridge", "core", "unified"),
    ("Linear", "core", "unified")
]

for m_name, f_set, strat in classical_specs:
    p_sub = p41_preds[(p41_preds["strategy"] == strat) & (p41_preds["model"] == m_name) & (p41_preds["feature_set"] == f_set)]
    if not p_sub.empty:
        c_sub = p_sub[p_sub["mechanical_property"] == "Compressive Strength"]
        f_sub = p_sub[p_sub["mechanical_property"] == "Flexural Strength"]
        t_sub = p_sub[p_sub["mechanical_property"] == "Split Tensile Strength"]

        c_m = compute_metrics(c_sub["strength_mpa_actual"], c_sub["strength_mpa_pred"]) if not c_sub.empty else {}
        f_m = compute_metrics(f_sub["strength_mpa_actual"], f_sub["strength_mpa_pred"]) if not f_sub.empty else {}
        t_m = compute_metrics(t_sub["strength_mpa_actual"], t_sub["strength_mpa_pred"]) if not t_sub.empty else {}
        o_m = compute_metrics(p_sub["strength_mpa_actual"], p_sub["strength_mpa_pred"])

        comparison_rows.append({
            "model": m_name,
            "feature_set": f_set.capitalize(),
            "strategy": strat.capitalize(),
            "compressive_mae": c_m.get("mae", np.nan),
            "compressive_rmse": c_m.get("rmse", np.nan),
            "compressive_r2": c_m.get("r2", np.nan),
            "flexural_mae": f_m.get("mae", np.nan),
            "flexural_rmse": f_m.get("rmse", np.nan),
            "flexural_r2": f_m.get("r2", np.nan),
            "tensile_mae": t_m.get("mae", np.nan),
            "tensile_rmse": t_m.get("rmse", np.nan),
            "tensile_r2": t_m.get("r2", np.nan),
            "overall_mae": o_m["mae"],
            "overall_rmse": o_m["rmse"],
            "overall_r2": o_m["r2"]
        })

comp_df = pd.DataFrame(comparison_rows)
comp_csv_path = REPORTS_DIR / "tabpfn_finetuned_model_comparison_run2.csv"
comp_df.to_csv(comp_csv_path, index=False)
print(f"Saved Comparison Table: {comp_csv_path}", flush=True)

# 8. Save Metadata JSON (Run 2)
meta = {
    "task": "5-Fold Grouped Cross-Validation of Fine-Tuned TabPFN (Run 2)",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "total_execution_time_sec": round(total_pipeline_time, 1),
    "total_execution_time_min": round(total_pipeline_time / 60, 2),
    "n_development_rows": len(dev_df),
    "n_cohorts": dev_df[GROUP_COL].nunique(),
    "n_folds": 5,
    "grouping_variable": GROUP_COL,
    "features": FEATURES,
    "hyperparameters": {
        "epochs": 3,
        "learning_rate": 5e-6,
        "weight_decay": 0.01,
        "grad_clip_value": 1.0,
        "n_estimators_finetune": 1,
        "n_estimators_validation": 1,
        "n_estimators_final_inference": 2,
        "base_checkpoint": str(CKPT_PATH),
        "device": "cpu"
    },
    "fold_results": fold_results,
    "pooled_metrics": {
        "overall": pooled_overall,
        "per_property": per_property_metrics
    },
    "stability": {
        "mean_mae": round(float(mean_mae), 4),
        "std_mae": round(float(std_mae), 4),
        "mean_rmse": round(float(mean_rmse), 4),
        "std_rmse": round(float(std_rmse), 4),
        "mean_r2": round(float(mean_r2), 4),
        "std_r2": round(float(std_r2), 4)
    }
}
meta_path = RESULTS_RUN2_DIR / "metadata.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved Metadata JSON: {meta_path}", flush=True)

# 9. Generate 8 Publication-Quality Figures (Section 19 Requirements)
print("\nGenerating 8 publication-quality visualization figures...", flush=True)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12})

# Figure 1: Actual vs Predicted — Compressive
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
cs_sub = oof_df[oof_df["mechanical_property"] == "Compressive Strength"]
sc = ax.scatter(cs_sub["strength_mpa_actual"], cs_sub["strength_mpa_pred"], c=cs_sub["curing_age_days"], cmap="viridis", alpha=0.6, edgecolors="none", s=25)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Curing Age (Days)")
lims = [12, 48]
ax.plot(lims, lims, "r--", lw=1.5, label="Ideal 1:1 Line")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual Compressive Strength (MPa)")
ax.set_ylabel("Fine-Tuned TabPFN Predicted (MPa)")
ax.set_title(f"Compressive Strength (Pooled OOF)\nMAE={per_property_metrics['Compressive Strength']['mae']:.3f} MPa, R²={per_property_metrics['Compressive Strength']['r2']:.3f}", fontweight="bold")
ax.legend(loc="upper left")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "actual_vs_pred_compressive.png")
plt.close(fig)

# Figure 2: Actual vs Predicted — Flexural
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
fs_sub = oof_df[oof_df["mechanical_property"] == "Flexural Strength"]
sc = ax.scatter(fs_sub["strength_mpa_actual"], fs_sub["strength_mpa_pred"], c=fs_sub["curing_age_days"], cmap="plasma", alpha=0.6, edgecolors="none", s=25)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Curing Age (Days)")
lims = [1.5, 8.0]
ax.plot(lims, lims, "r--", lw=1.5, label="Ideal 1:1 Line")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual Flexural Strength (MPa)")
ax.set_ylabel("Fine-Tuned TabPFN Predicted (MPa)")
ax.set_title(f"Flexural Strength (Pooled OOF)\nMAE={per_property_metrics['Flexural Strength']['mae']:.3f} MPa, R²={per_property_metrics['Flexural Strength']['r2']:.3f}", fontweight="bold")
ax.legend(loc="upper left")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "actual_vs_pred_flexural.png")
plt.close(fig)

# Figure 3: Actual vs Predicted — Split Tensile
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
ts_sub = oof_df[oof_df["mechanical_property"] == "Split Tensile Strength"]
sc = ax.scatter(ts_sub["strength_mpa_actual"], ts_sub["strength_mpa_pred"], c=ts_sub["curing_age_days"], cmap="cividis", alpha=0.6, edgecolors="none", s=25)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Curing Age (Days)")
lims = [1.0, 5.0]
ax.plot(lims, lims, "r--", lw=1.5, label="Ideal 1:1 Line")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual Split Tensile Strength (MPa)")
ax.set_ylabel("Fine-Tuned TabPFN Predicted (MPa)")
ax.set_title(f"Split Tensile Strength (Pooled OOF)\nMAE={per_property_metrics['Split Tensile Strength']['mae']:.3f} MPa, R²={per_property_metrics['Split Tensile Strength']['r2']:.3f}", fontweight="bold")
ax.legend(loc="upper left")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "actual_vs_pred_split_tensile.png")
plt.close(fig)

# Figure 4: Residual Distribution
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
for prop in ["Compressive Strength", "Flexural Strength", "Split Tensile Strength"]:
    res = oof_df[oof_df["mechanical_property"] == prop]["residual"]
    sns.kdeplot(res, ax=ax, label=prop, lw=2)
ax.axvline(0, color="gray", linestyle="--", lw=1.2)
ax.set_xlabel("Prediction Residual (Predicted - Actual, MPa)")
ax.set_ylabel("Density")
ax.set_title("Fine-Tuned TabPFN Residual Distributions", fontweight="bold")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURES_DIR / "residual_distribution.png")
plt.close(fig)

# Figure 5: Fold-wise MAE
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
folds_num = [f"Fold {r['fold']}" for r in fold_results]
ax.bar(folds_num, fold_maes, color="cornflowerblue", edgecolor="black", alpha=0.85)
ax.axhline(mean_mae, color="crimson", linestyle="--", lw=1.8, label=f"Mean MAE: {mean_mae:.3f} ± {std_mae:.3f}")
ax.set_ylabel("Mean Absolute Error (MPa)")
ax.set_title("Fold-Wise Generalization MAE (Fine-Tuned TabPFN)", fontweight="bold")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURES_DIR / "fold_wise_mae.png")
plt.close(fig)

# Figure 6: Pretrained vs Fine-Tuned MAE
fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
props_names = ["Compressive", "Flexural", "Split Tensile", "Overall Pooled"]
pre_maes = [
    pre_prop_metrics["Compressive Strength"]["mae"],
    pre_prop_metrics["Flexural Strength"]["mae"],
    pre_prop_metrics["Split Tensile Strength"]["mae"],
    pre_pooled_overall["mae"]
]
ft_maes = [
    per_property_metrics["Compressive Strength"]["mae"],
    per_property_metrics["Flexural Strength"]["mae"],
    per_property_metrics["Split Tensile Strength"]["mae"],
    pooled_overall["mae"]
]
x = np.arange(len(props_names))
width = 0.35
ax.bar(x - width/2, pre_maes, width, label="Pretrained TabPFN v2.5", color="lightgray", edgecolor="black")
ax.bar(x + width/2, ft_maes, width, label="Fine-Tuned TabPFN v2.5", color="royalblue", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(props_names)
ax.set_ylabel("MAE (MPa)")
ax.set_title("Pretrained vs. Fine-Tuned TabPFN Generalization Error", fontweight="bold")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURES_DIR / "pretrained_vs_finetuned_mae.png")
plt.close(fig)

# Figure 7: MAE vs Curing Age
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
oof_df["abs_error"] = np.abs(oof_df["residual"])
age_mae = oof_df.groupby(["curing_age_days", "mechanical_property"])["abs_error"].mean().unstack()
age_mae.plot(kind="bar", ax=ax, edgecolor="black", alpha=0.85)
ax.set_xlabel("Curing Age (Days)")
ax.set_ylabel("Mean Absolute Error (MPa)")
ax.set_title("Fine-Tuned TabPFN MAE vs. Curing Age", fontweight="bold")
ax.legend(title="Property")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "mae_vs_curing_age.png")
plt.close(fig)

# Figure 8: Prediction Error vs Curing Age (Scatter / Boxplot of residuals)
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
sns.boxplot(data=oof_df, x="curing_age_days", y="residual", hue="mechanical_property", ax=ax)
ax.axhline(0, color="red", linestyle="--", lw=1.2)
ax.set_xlabel("Curing Age (Days)")
ax.set_ylabel("Residual (Predicted - Actual, MPa)")
ax.set_title("Prediction Residuals vs. Curing Age by Property", fontweight="bold")
ax.legend(title="Property", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "prediction_error_vs_curing_age.png")
plt.close(fig)

print(f"All 8 visualization figures successfully saved to {FIGURES_DIR}!", flush=True)
print("\nPIPELINE COMPLETED SUCCESSFULLY.", flush=True)
