"""
Train and Serialize Classical Models (XGBoost, CatBoost, Random Forest)
======================================================================
Trains the authoritative classical baseline pipelines on the full 2,800 development set
using the exact feature sets and hyperparameters established in Phase 4.1.

Serialized models saved to:
    models/classical/

Output includes complete sklearn Pipelines (preprocessing + estimator)
so they can be loaded directly for inference on new concrete mix formulations:
    pipeline = joblib.load("models/classical/xgboost_expanded_unified.joblib")
    predictions = pipeline.predict(df_new)
"""

import hashlib
import json
import os
import pathlib
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb
from catboost import CatBoostRegressor

# Paths
ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "master_dataset.parquet"
DEV_ROWS_PATH = ROOT / "data" / "splits" / "development_rows.csv"
OUTPUT_DIR = ROOT / "models" / "classical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

FEATURE_SETS = {
    "core": {
        "numeric": ["curing_age_days", "bacterial_concentration_cells_ml"],
        "categorical": ["concrete_type", "bacterial_status", "cement_type", "bacterial_species"],
    },
    "expanded": {
        "numeric": ["curing_age_days", "bacterial_concentration_cells_ml"],
        "categorical": ["concrete_type", "bacterial_status", "cement_type", "bacterial_species", "specimen_geometry"],
    },
}

def get_base_models():
    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=6,
            random_seed=RANDOM_STATE, verbose=0,
            loss_function="RMSE",
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("TRAINING AND SERIALIZING CLASSICAL MODELS (XGBoost, CatBoost, RF)")
    print("=" * 70)

    # 1. Load data
    full_df = pd.read_parquet(DATA_PATH)
    dev_ids = set(pd.read_csv(DEV_ROWS_PATH)["sample_id"])
    dev_df = full_df[full_df["sample_id"].isin(dev_ids)].copy().reset_index(drop=True)
    assert len(dev_df) == 2800, f"Expected 2800 dev rows, got {len(dev_df)}"
    print(f"Loaded development dataset: {len(dev_df)} specimens across {dev_df['experiment_id'].nunique()} cohorts.")

    manifest = {
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_dataset": "data/splits/development_rows.csv (N=2,800)",
        "random_state": RANDOM_STATE,
        "models": []
    }

    # 2. Train UNIFIED models (All mechanical properties combined)
    print("\n--- Training Unified Models (Strategy B) ---")
    for fs_name in ["core", "expanded"]:
        fs_def = FEATURE_SETS[fs_name]
        numeric_cols = list(fs_def["numeric"])
        categorical_cols = list(fs_def["categorical"]) + ["mechanical_property"]
        feature_cols = numeric_cols + categorical_cols

        X_train = dev_df[feature_cols]
        y_train = dev_df["strength_mpa"].values

        base_models = get_base_models()
        for model_name, model_inst in base_models.items():
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
                ],
                remainder="drop"
            )
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model_inst)
            ])

            t0 = time.time()
            pipe.fit(X_train, y_train)
            elapsed = time.time() - t0

            filename = f"{model_name.lower()}_{fs_name}_unified.joblib"
            filepath = OUTPUT_DIR / filename
            joblib.dump(pipe, filepath, compress=3)

            sz = filepath.stat().st_size
            h = sha256_file(filepath)
            print(f"  [SAVED] {filename:35s} | {sz/1024/1024:6.2f} MB | {elapsed:4.1f}s | SHA: {h[:16]}...")

            manifest["models"].append({
                "filename": filename,
                "model_name": model_name,
                "strategy": "Unified",
                "feature_set": fs_name,
                "target": "ALL (Unified with mechanical_property indicator)",
                "n_train_samples": len(dev_df),
                "features": feature_cols,
                "file_size_bytes": sz,
                "sha256": h,
                "training_time_sec": round(elapsed, 2)
            })

    # 3. Train SEPARATE models for top configurations (Strategy A)
    print("\n--- Training Separate Property Models (Strategy A: Expanded Feature Set) ---")
    props = ["Compressive Strength", "Flexural Strength", "Split Tensile Strength"]
    prop_slugs = {
        "Compressive Strength": "compressive",
        "Flexural Strength": "flexural",
        "Split Tensile Strength": "split_tensile"
    }

    fs_name = "expanded"
    fs_def = FEATURE_SETS[fs_name]
    numeric_cols = list(fs_def["numeric"])
    categorical_cols = list(fs_def["categorical"])
    feature_cols = numeric_cols + categorical_cols

    for prop in props:
        prop_slug = prop_slugs[prop]
        sub_df = dev_df[dev_df["mechanical_property"] == prop].copy().reset_index(drop=True)
        X_sub = sub_df[feature_cols]
        y_sub = sub_df["strength_mpa"].values

        for model_name in ["XGBoost", "CatBoost"]:
            base_models = get_base_models()
            model_inst = base_models[model_name]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
                ],
                remainder="drop"
            )
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model_inst)
            ])

            t0 = time.time()
            pipe.fit(X_sub, y_sub)
            elapsed = time.time() - t0

            filename = f"{model_name.lower()}_{fs_name}_{prop_slug}.joblib"
            filepath = OUTPUT_DIR / filename
            joblib.dump(pipe, filepath, compress=3)

            sz = filepath.stat().st_size
            h = sha256_file(filepath)
            print(f"  [SAVED] {filename:35s} | {sz/1024/1024:6.2f} MB | {elapsed:4.1f}s | SHA: {h[:16]}...")

            manifest["models"].append({
                "filename": filename,
                "model_name": model_name,
                "strategy": "Separate",
                "feature_set": fs_name,
                "target": prop,
                "n_train_samples": len(sub_df),
                "features": feature_cols,
                "file_size_bytes": sz,
                "sha256": h,
                "training_time_sec": round(elapsed, 2)
            })

    # Save manifest
    manifest_path = OUTPUT_DIR / "classical_models_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest to {manifest_path} ({len(manifest['models'])} models registered).")
    print("=" * 70)
    print("ALL REQUESTED CLASSICAL MODELS SERIALIZED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
