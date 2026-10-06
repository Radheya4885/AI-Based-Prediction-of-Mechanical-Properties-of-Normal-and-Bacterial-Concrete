"""
Split Generator Module
Generates reproducible, leak-free train/test splits for all three evaluation scenarios:
Scenario A: Row-Level Random Baseline (Diagnostic only)
Scenario B: Cohort-Based Grouped Split (Strict experiment_id isolation)
Scenario C: Leave-Age-Out Generalization (6 temporal folds)
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


class SplitGenerator:
    """Generates and persists reproducible train/test validation splits."""

    def __init__(self, data: pd.DataFrame, output_dir: str = "data/splits"):
        self.df = data
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def from_csv(cls, path: str = "data/master_dataset.csv", output_dir: str = "data/splits") -> "SplitGenerator":
        df = pd.read_csv(path)
        return cls(df, output_dir)

    def generate_scenario_a_random(self, seed: int = 42, test_size: float = 0.20) -> Dict[str, Any]:
        """Scenario A: Row-level random baseline split.
        
        Diagnostic comparison only. Repeated specimens from the same mix/batch 
        appear in both train and test.
        """
        rng = np.random.RandomState(seed)
        total_rows = len(self.df)
        indices = np.arange(total_rows)
        rng.shuffle(indices)
        
        split_idx = int(total_rows * (1.0 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_sample_ids = self.df.iloc[train_indices]["sample_id"].tolist()
        test_sample_ids = self.df.iloc[test_indices]["sample_id"].tolist()
        
        result = {
            "scenario": "Scenario A — Random Baseline",
            "split_type": "row_level_random",
            "seed": seed,
            "test_size": test_size,
            "train_row_count": len(train_sample_ids),
            "test_row_count": len(test_sample_ids),
            "train_cohort_count": int(self.df.iloc[train_indices]["experiment_id"].nunique()),
            "test_cohort_count": int(self.df.iloc[test_indices]["experiment_id"].nunique()),
            "train_sample_ids": train_sample_ids,
            "test_sample_ids": test_sample_ids,
            "methodology_note": (
                "Row-level 80/20 random split. Provided strictly for diagnostic baseline comparison. "
                "WARNING: Repeated specimens sharing identical mix proportions and casting batches "
                "occur in both train and test sets, enabling specimen-level memorization."
            ),
        }
        return result

    def generate_scenario_b_cohort(self, seed: int = 42) -> Dict[str, Any]:
        """Scenario B: Cohort-based grouped split.
        
        Strictly groups by experiment_id. No experiment_id appears in both train and test.
        Stratified across mechanical properties to ensure 28 train cohorts (2,800 rows)
        and 8 test cohorts (800 rows), preserving test evaluation data across all properties.
        """
        cohorts = (
            self.df[["experiment_id", "mechanical_property", "concrete_type", "curing_age_days"]]
            .drop_duplicates()
            .sort_values("experiment_id")
            .reset_index(drop=True)
        )
        rng = np.random.RandomState(seed)
        
        # Stratify cohort selection across mechanical properties and concrete types
        cs_nc = sorted(cohorts[(cohorts["mechanical_property"] == "Compressive Strength") & (cohorts["concrete_type"] == "Normal Concrete")]["experiment_id"].tolist())
        cs_bc = sorted(cohorts[(cohorts["mechanical_property"] == "Compressive Strength") & (cohorts["concrete_type"] == "Bacterial Concrete")]["experiment_id"].tolist())
        
        fs_nc = sorted(cohorts[(cohorts["mechanical_property"] == "Flexural Strength") & (cohorts["concrete_type"] == "Normal Concrete")]["experiment_id"].tolist())
        fs_bc = sorted(cohorts[(cohorts["mechanical_property"] == "Flexural Strength") & (cohorts["concrete_type"] == "Bacterial Concrete")]["experiment_id"].tolist())
        
        ts_nc = sorted(cohorts[(cohorts["mechanical_property"] == "Split Tensile Strength") & (cohorts["concrete_type"] == "Normal Concrete")]["experiment_id"].tolist())
        ts_bc = sorted(cohorts[(cohorts["mechanical_property"] == "Split Tensile Strength") & (cohorts["concrete_type"] == "Bacterial Concrete")]["experiment_id"].tolist())
        
        test_cs_nc = rng.choice(cs_nc, size=1, replace=False).tolist()
        test_cs_bc = rng.choice(cs_bc, size=2, replace=False).tolist()
        
        test_fs_nc = rng.choice(fs_nc, size=1, replace=False).tolist()
        test_fs_bc = rng.choice(fs_bc, size=2, replace=False).tolist()
        
        test_ts_nc = rng.choice(ts_nc, size=1, replace=False).tolist()
        test_ts_bc = rng.choice(ts_bc, size=1, replace=False).tolist()
        
        test_cohorts = sorted(test_cs_nc + test_cs_bc + test_fs_nc + test_fs_bc + test_ts_nc + test_ts_bc)
        all_cohorts = sorted(cohorts["experiment_id"].tolist())
        train_cohorts = sorted([c for c in all_cohorts if c not in test_cohorts])
        
        train_df = self.df[self.df["experiment_id"].isin(train_cohorts)]
        test_df = self.df[self.df["experiment_id"].isin(test_cohorts)]
        
        train_sample_ids = train_df["sample_id"].tolist()
        test_sample_ids = test_df["sample_id"].tolist()
        
        # Details of cohorts
        test_cohort_details = (
            self.df[self.df["experiment_id"].isin(test_cohorts)][
                ["experiment_id", "mechanical_property", "concrete_type", "curing_age_days"]
            ]
            .drop_duplicates()
            .sort_values(["mechanical_property", "concrete_type", "curing_age_days"])
            .to_dict(orient="records")
        )
        
        train_cohort_details = (
            self.df[self.df["experiment_id"].isin(train_cohorts)][
                ["experiment_id", "mechanical_property", "concrete_type", "curing_age_days"]
            ]
            .drop_duplicates()
            .sort_values(["mechanical_property", "concrete_type", "curing_age_days"])
            .to_dict(orient="records")
        )
        
        # Count restored values in train and test
        train_restored = int(train_df["is_restored_value"].sum())
        test_restored = int(test_df["is_restored_value"].sum())
        
        result = {
            "scenario": "Scenario B — Cohort-Based Grouped Split",
            "split_type": "cohort_grouped",
            "group_variable": "experiment_id",
            "seed": seed,
            "train_cohort_count": len(train_cohorts),
            "test_cohort_count": len(test_cohorts),
            "train_row_count": len(train_sample_ids),
            "test_row_count": len(test_sample_ids),
            "train_cohorts": train_cohorts,
            "test_cohorts": test_cohorts,
            "train_cohort_details": train_cohort_details,
            "test_cohort_details": test_cohort_details,
            "train_restored_count": train_restored,
            "test_restored_count": test_restored,
            "train_sample_ids": train_sample_ids,
            "test_sample_ids": test_sample_ids,
            "methodology_note": (
                "Cohort-level grouped split (28 train cohorts / 8 test cohorts; 77.8% train / 22.2% test). "
                "Stratified across mechanical properties to guarantee test instances for all 3 targets. "
                "Zero experiment_id overlap ensures models are evaluated on unseen experimental conditions."
            ),
        }
        return result

    def generate_scenario_c_leave_age_out(self) -> Dict[str, Any]:
        """Scenario C: Leave-Age-Out generalization experiment.
        
        For each of the 6 curing ages (7, 14, 21, 28, 56, 90 days), hold that entire
        age out as the test set (600 rows, 6 cohorts) and train on the remaining 5 ages
        (3,000 rows, 30 cohorts).
        """
        all_ages = sorted(self.df["curing_age_days"].unique().tolist())
        folds = {}
        
        for fold_idx, held_out_age in enumerate(all_ages, start=1):
            test_df = self.df[self.df["curing_age_days"] == held_out_age]
            train_df = self.df[self.df["curing_age_days"] != held_out_age]
            
            test_cohorts = sorted(test_df["experiment_id"].unique().tolist())
            train_cohorts = sorted(train_df["experiment_id"].unique().tolist())
            
            test_sample_ids = test_df["sample_id"].tolist()
            train_sample_ids = train_df["sample_id"].tolist()
            
            # Characterize interpolation vs extrapolation
            if held_out_age == min(all_ages):
                gen_type = "Extrapolation (Early hydration kinetics; 7d holdout)"
            elif held_out_age == max(all_ages):
                gen_type = "Extrapolation (Long-term asymptotic maturity; 90d holdout)"
            elif held_out_age == 28:
                gen_type = "Interpolation (Standard 28-day compliance boundary)"
            else:
                gen_type = f"Interpolation ({held_out_age}d holdout between bounding ages)"
                
            folds[f"fold_{fold_idx}_age_{held_out_age}d"] = {
                "fold_number": fold_idx,
                "held_out_age_days": held_out_age,
                "generalization_type": gen_type,
                "train_row_count": len(train_sample_ids),
                "test_row_count": len(test_sample_ids),
                "train_cohort_count": len(train_cohorts),
                "test_cohort_count": len(test_cohorts),
                "test_cohorts": test_cohorts,
                "train_cohorts": train_cohorts,
                "train_restored_count": int(train_df["is_restored_value"].sum()),
                "test_restored_count": int(test_df["is_restored_value"].sum()),
                "train_sample_ids": train_sample_ids,
                "test_sample_ids": test_sample_ids,
            }
            
        result = {
            "scenario": "Scenario C — Leave-Age-Out Generalization",
            "split_type": "leave_one_group_out",
            "group_variable": "curing_age_days",
            "total_folds": len(all_ages),
            "curing_ages": all_ages,
            "folds": folds,
            "methodology_note": (
                "6-fold Leave-One-Age-Out temporal generalization. In each fold, exactly one curing age "
                "is held out (600 rows, 6 cohorts: 3 properties x 2 concrete types), while the model is trained "
                "on the remaining 5 ages (3,000 rows, 30 cohorts). Evaluates whether models interpolate hydration "
                "kinetics (14d, 21d, 28d, 56d) or extrapolate beyond known age boundaries (7d, 90d)."
            ),
        }
        return result

    def generate_all_splits(self, seed: int = 42) -> Dict[str, Any]:
        """Generates and writes all splits to disk."""
        split_a = self.generate_scenario_a_random(seed=seed)
        split_b = self.generate_scenario_b_cohort(seed=seed)
        split_c = self.generate_scenario_c_leave_age_out()
        
        # Save JSON files
        with open(os.path.join(self.output_dir, "scenario_a_random_baseline.json"), "w", encoding="utf-8") as f:
            json.dump(split_a, f, indent=2)
            
        with open(os.path.join(self.output_dir, "scenario_b_cohort_grouped.json"), "w", encoding="utf-8") as f:
            json.dump(split_b, f, indent=2)
            
        with open(os.path.join(self.output_dir, "scenario_c_leave_age_out.json"), "w", encoding="utf-8") as f:
            json.dump(split_c, f, indent=2)
            
        # Save CSV split assignments (mapping sample_id -> split assignment)
        # Scenario A CSV
        df_a = self.df[["sample_id", "experiment_id", "mechanical_property", "concrete_type", "curing_age_days"]].copy()
        df_a["split"] = np.where(df_a["sample_id"].isin(split_a["train_sample_ids"]), "train", "test")
        df_a.to_csv(os.path.join(self.output_dir, "scenario_a_random_baseline.csv"), index=False)
        
        # Scenario B CSV
        df_b = self.df[["sample_id", "experiment_id", "mechanical_property", "concrete_type", "curing_age_days"]].copy()
        df_b["split"] = np.where(df_b["sample_id"].isin(split_b["train_sample_ids"]), "train", "test")
        df_b.to_csv(os.path.join(self.output_dir, "scenario_b_cohort_grouped.csv"), index=False)
        
        # Scenario C CSV (includes fold assignments)
        df_c = self.df[["sample_id", "experiment_id", "mechanical_property", "concrete_type", "curing_age_days"]].copy()
        for fold_key, fold_info in split_c["folds"].items():
            age = fold_info["held_out_age_days"]
            df_c[f"split_fold_held_{age}d"] = np.where(df_c["curing_age_days"] == age, "test", "train")
        df_c.to_csv(os.path.join(self.output_dir, "scenario_c_leave_age_out.csv"), index=False)
        
        # Summary Manifest
        manifest = {
            "source_dataset": "data/master_dataset.csv",
            "total_rows": len(self.df),
            "random_seed": seed,
            "scenarios": {
                "scenario_a": {
                    "file_json": "data/splits/scenario_a_random_baseline.json",
                    "file_csv": "data/splits/scenario_a_random_baseline.csv",
                    "train_rows": split_a["train_row_count"],
                    "test_rows": split_a["test_row_count"],
                },
                "scenario_b": {
                    "file_json": "data/splits/scenario_b_cohort_grouped.json",
                    "file_csv": "data/splits/scenario_b_cohort_grouped.csv",
                    "train_rows": split_b["train_row_count"],
                    "test_rows": split_b["test_row_count"],
                    "train_cohorts": split_b["train_cohort_count"],
                    "test_cohorts": split_b["test_cohort_count"],
                    "test_cohort_list": split_b["test_cohorts"],
                },
                "scenario_c": {
                    "file_json": "data/splits/scenario_c_leave_age_out.json",
                    "file_csv": "data/splits/scenario_c_leave_age_out.csv",
                    "total_folds": split_c["total_folds"],
                    "held_out_ages": split_c["curing_ages"],
                    "rows_per_fold": {"train": 3000, "test": 600},
                },
            },
        }
        with open(os.path.join(self.output_dir, "splits_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            
        return {
            "scenario_a": split_a,
            "scenario_b": split_b,
            "scenario_c": split_c,
            "manifest": manifest,
        }
