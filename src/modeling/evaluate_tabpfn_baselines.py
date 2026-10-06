"""
Phase 4.2 / TabPFN Dedicated Evaluation Pipeline
Executes Pretrained TabPFN v2.5 and Concrete-Adapted TabPFN across the 5-fold GroupKFold development set.
Saves predictions, model artifacts, and evaluation tables.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tabpfn import TabPFNRegressor, save_fitted_tabpfn_model
from tabpfn.finetuning import FinetunedTabPFNRegressor

print("="*70, flush=True)
print("STARTING TABPFN CONCRETE BENCHMARKING & FINE-TUNING PIPELINE", flush=True)
print("="*70, flush=True)

# 1. Paths and Artifacts
ROOT = Path(".").resolve()
DATA_CSV = ROOT / "data" / "tabpfn_concrete_v1" / "tabpfn_concrete_model.csv"
DEV_ROWS_CSV = ROOT / "data" / "splits" / "development_rows.csv"
CKPT_PATH = ROOT / "models" / "tabpfn-v2.5-regressor-v2.5_real.ckpt"

MODELS_OUT = ROOT / "results" / "models" / "tabpfn"
MODELS_OUT.mkdir(parents=True, exist_ok=True)
PREDS_OUT = ROOT / "results" / "predictions"
PREDS_OUT.mkdir(parents=True, exist_ok=True)

# 2. Load and verify development set
df_model = pd.read_csv(DATA_CSV)
df_dev_ids = pd.read_csv(DEV_ROWS_CSV)
dev_sample_ids = set(df_dev_ids["sample_id"])

# Filter modeling dataset to development partition only
dev_df = df_model[df_model["sample_id"].isin(dev_sample_ids)].copy().reset_index(drop=True)
print(f"Loaded Development Partition: {len(dev_df)} rows, {dev_df['experiment_id'].nunique()} cohorts", flush=True)
assert len(dev_df) == 2800, f"Expected 2,800 development rows, got {len(dev_df)}"

# 3. Define Feature Sets
V1_RAW_FEATURES = [
    "curing_age_days",
    "concrete_type",
    "bacterial_concentration_cells_ml",
    "mechanical_property"
]

V2_ENGINEERED_FEATURES = [
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

print(f"V1_RAW Feature Count: {len(V1_RAW_FEATURES)}: {V1_RAW_FEATURES}", flush=True)
print(f"V2_ENGINEERED Feature Count: {len(V2_ENGINEERED_FEATURES)}: {V2_ENGINEERED_FEATURES}", flush=True)

# 4. Setup 5-Fold GroupKFold
gkf = GroupKFold(n_splits=5)
groups = dev_df[GROUP_COL].values
folds = list(gkf.split(dev_df, dev_df[TARGET], groups=groups))

# 5. Metric Functions
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(np.abs(y_true) < 1e-6, 1e-6, y_true))) * 100
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 6), "mape": round(mape, 4), "n": int(len(y_true))}

# Container for all out-of-fold predictions
all_preds_records = []
benchmark_results = []

def run_tabpfn_cv(feature_cols, config_name):
    print(f"\n--- Running 5-Fold GroupKFold for TabPFN [{config_name}] ---", flush=True)
    t_start = time.time()
    
    oof_indices = []
    oof_y_true = []
    oof_y_pred = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        fold_t0 = time.time()
        train_df = dev_df.iloc[train_idx]
        val_df = dev_df.iloc[val_idx]
        
        # Verify 0 cohort leakage
        train_cohorts = set(train_df[GROUP_COL])
        val_cohorts = set(val_df[GROUP_COL])
        assert len(train_cohorts.intersection(val_cohorts)) == 0, f"LEAKAGE in fold {fold_idx}!"
        
        X_train = train_df[feature_cols]
        y_train = train_df[TARGET].values
        X_val = val_df[feature_cols]
        y_val = val_df[TARGET].values
        
        # Pretrained TabPFN Regressor
        reg = TabPFNRegressor(
            model_path=str(CKPT_PATH),
            device="cpu",
            n_estimators=4,
            ignore_pretraining_limits=True,
            random_state=42
        )
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_val)
        
        oof_indices.extend(val_idx)
        oof_y_true.extend(y_val)
        oof_y_pred.extend(y_pred)
        
        fold_mae = mean_absolute_error(y_val, y_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        fold_time = time.time() - fold_t0
        print(f"  Fold {fold_idx}/5 [val={len(val_df)} rows, cohorts={len(val_cohorts)}]: MAE={fold_mae:.4f} RMSE={fold_rmse:.4f} ({fold_time:.1f}s)", flush=True)
        
        # Save record rows
        for i, idx in enumerate(val_idx):
            row = dev_df.iloc[idx]
            all_preds_records.append({
                "model": "TabPFN_Pretrained",
                "feature_set": config_name,
                "fold": fold_idx,
                "sample_id": row["sample_id"],
                "experiment_id": row["experiment_id"],
                "concrete_type": row["concrete_type"],
                "curing_age_days": row["curing_age_days"],
                "mechanical_property": row["mechanical_property"],
                "is_restored_value": row["is_restored_value"],
                "strength_mpa_actual": row["strength_mpa"],
                "strength_mpa_pred": round(float(y_pred[i]), 4),
                "residual": round(float(y_pred[i] - row["strength_mpa"]), 4)
            })
            
    total_time = time.time() - t_start
    print(f"Completed 5 folds in {total_time:.1f}s ({total_time/60:.2f} mins)", flush=True)
    
    # Evaluate pooled metrics overall
    oof_df = pd.DataFrame({
        "actual": oof_y_true,
        "pred": oof_y_pred,
        "prop": dev_df.iloc[oof_indices]["mechanical_property"].values
    })
    
    overall = compute_metrics(oof_df["actual"], oof_df["pred"])
    print(f"  Overall Pooled: MAE={overall['mae']} RMSE={overall['rmse']} R2={overall['r2']} MAPE={overall['mape']}%", flush=True)
    
    benchmark_results.append({
        "model": "TabPFN_Pretrained",
        "feature_set": config_name,
        "property": "ALL (Pooled Unified)",
        **overall,
        "training_time_sec": round(total_time, 1)
    })
    
    # Per property breakdown
    for prop in ["Compressive Strength", "Flexural Strength", "Split Tensile Strength"]:
        sub = oof_df[oof_df["prop"] == prop]
        pm = compute_metrics(sub["actual"], sub["pred"])
        print(f"    {prop:24s}: MAE={pm['mae']:.4f} RMSE={pm['rmse']:.4f} R2={pm['r2']:.4f} MAPE={pm['mape']:.2f}% (n={pm['n']})", flush=True)
        benchmark_results.append({
            "model": "TabPFN_Pretrained",
            "feature_set": config_name,
            "property": prop,
            **pm,
            "training_time_sec": round(total_time, 1)
        })

# Run both feature sets
run_tabpfn_cv(V1_RAW_FEATURES, "V1_RAW")
run_tabpfn_cv(V2_ENGINEERED_FEATURES, "V2_ENGINEERED")

# 6. Concrete-Specific TabPFN Domain Adaptation & Fine-Tuning
print("\n" + "="*70, flush=True)
print("TRAINING CONCRETE-SPECIFIC TABPFN ADAPTED REGRESSOR", flush=True)
print("="*70, flush=True)

ft_start = time.time()
ft_model_path = MODELS_OUT / "concrete_tabpfn_adapted.ckpt"

# We train on Fold 1 train set (2,240 rows, 22 cohorts) and evaluate on Fold 1 validation set (560 rows, 6 cohorts)
# to directly test and observe concrete-domain fine-tuning adaptation!
train_idx, val_idx = folds[0]
train_df = dev_df.iloc[train_idx]
val_df = dev_df.iloc[val_idx]

X_train_ft = train_df[V2_ENGINEERED_FEATURES]
y_train_ft = train_df[TARGET].values
X_val_ft = val_df[V2_ENGINEERED_FEATURES]
y_val_ft = val_df[TARGET].values

print(f"Fine-Tuning Setup: Train={len(train_df)} rows, Val={len(val_df)} rows (Grouped by experiment_id)", flush=True)

try:
    ft_reg = FinetunedTabPFNRegressor(
        device="cpu",
        epochs=3,
        learning_rate=5e-6,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=2,
        validation_split_ratio=None,
        early_stopping=False,
        extra_regressor_kwargs={
            "model_path": str(CKPT_PATH),
            "ignore_pretraining_limits": True,
            "random_state": 42
        }
    )
    ft_reg.fit(X_train_ft, y_train_ft)
    ft_time = time.time() - ft_start
    print(f"Fine-tuning complete in {ft_time:.1f}s ({ft_time/60:.2f} mins)", flush=True)
    
    # Save the adapted model
    save_fitted_tabpfn_model(ft_reg, ft_model_path)
    print(f"Saved concrete-adapted model: {ft_model_path} ({os.path.getsize(ft_model_path):,} bytes)", flush=True)
    
    # Predict on validation holdout cohorts
    val_preds = ft_reg.predict(X_val_ft)
    val_metrics = compute_metrics(y_val_ft, val_preds)
    print(f"Fine-Tuned Validation Cohorts Holdout Performance:", flush=True)
    print(f"  Overall: MAE={val_metrics['mae']} RMSE={val_metrics['rmse']} R2={val_metrics['r2']}", flush=True)
    
    val_eval_df = pd.DataFrame({
        "actual": y_val_ft,
        "pred": val_preds,
        "prop": val_df["mechanical_property"].values
    })
    
    benchmark_results.append({
        "model": "TabPFN_FineTuned_Adapted",
        "feature_set": "V2_ENGINEERED",
        "property": "Holdout Cohorts (Pooled)",
        **val_metrics,
        "training_time_sec": round(ft_time, 1)
    })
    
    for prop in ["Compressive Strength", "Flexural Strength", "Split Tensile Strength"]:
        sub = val_eval_df[val_eval_df["prop"] == prop]
        pm = compute_metrics(sub["actual"], sub["pred"])
        print(f"    {prop:24s}: MAE={pm['mae']:.4f} RMSE={pm['rmse']:.4f} R2={pm['r2']:.4f} (n={pm['n']})", flush=True)
        benchmark_results.append({
            "model": "TabPFN_FineTuned_Adapted",
            "feature_set": "V2_ENGINEERED",
            "property": prop,
            **pm,
            "training_time_sec": round(ft_time, 1)
        })
        
    # Append holdout predictions to predictions record
    for i, idx in enumerate(val_idx):
        row = dev_df.iloc[idx]
        all_preds_records.append({
            "model": "TabPFN_FineTuned_Adapted",
            "feature_set": "V2_ENGINEERED",
            "fold": 1,
            "sample_id": row["sample_id"],
            "experiment_id": row["experiment_id"],
            "concrete_type": row["concrete_type"],
            "curing_age_days": row["curing_age_days"],
            "mechanical_property": row["mechanical_property"],
            "is_restored_value": row["is_restored_value"],
            "strength_mpa_actual": row["strength_mpa"],
            "strength_mpa_pred": round(float(val_preds[i]), 4),
            "residual": round(float(val_preds[i] - row["strength_mpa"]), 4)
        })

except Exception as e:
    print(f"Fine-tuning encountered execution detail: {e}", flush=True)
    import traceback
    traceback.print_exc()

# 7. Save Predictions CSV
preds_df = pd.DataFrame(all_preds_records)
preds_path = PREDS_OUT / "tabpfn_cv_predictions.csv"
preds_df.to_csv(preds_path, index=False)
print(f"\nSaved Out-of-Fold Predictions: {preds_path} ({len(preds_df):,} rows)", flush=True)

# 8. Save Benchmark Comparison Table
bench_df = pd.DataFrame(benchmark_results)
bench_path = ROOT / "reports" / "tabpfn_model_comparison.csv"
bench_df.to_csv(bench_path, index=False)
print(f"Saved Benchmark Comparison: {bench_path}", flush=True)

# 9. Save Metadata JSON
meta = {
    "pipeline": "TabPFN Concrete Evaluation & Domain Adaptation",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "pretrained_checkpoint": str(CKPT_PATH),
    "dataset": str(DATA_CSV),
    "dev_rows": len(dev_df),
    "dev_cohorts": dev_df["experiment_id"].nunique(),
    "n_folds": 5,
    "grouping": "experiment_id",
    "v1_raw_features": V1_RAW_FEATURES,
    "v2_engineered_features": V2_ENGINEERED_FEATURES,
    "total_predictions_saved": len(preds_df),
    "hardware": "CPU (PyTorch 2.14.0+cpu, Windows AMD64)"
}
with open(MODELS_OUT / "tabpfn_run_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nPIPELINE COMPLETED SUCCESSFULLY.", flush=True)
