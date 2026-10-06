"""
Automated Unit Tests for Phase 3.1 Validation Framework
Runs using Python standard library unittest.
"""

import unittest
import os
import sys
import hashlib
import pandas as pd

# Add workspace root to sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.validation.robust_grouped_cv import (
    RobustValidationPipeline,
    calculate_sha256,
    EXPECTED_CSV_SHA256,
    EXPECTED_PARQUET_SHA256,
)
from src.validation.feature_policy import FORBIDDEN_FEATURES, CANDIDATE_FEATURES


class TestValidationV31(unittest.TestCase):
    """Test suite verifying all Phase 3.1 requirements."""

    def test_01_checksums_match_exact(self):
        csv_hash = calculate_sha256("data/master_dataset.csv")
        parquet_hash = calculate_sha256("data/master_dataset.parquet")
        self.assertEqual(csv_hash, EXPECTED_CSV_SHA256)
        self.assertEqual(parquet_hash, EXPECTED_PARQUET_SHA256)

    def test_02_dataset_dimensions(self):
        df = pd.read_csv("data/master_dataset.csv")
        self.assertEqual(len(df), 3600)
        self.assertEqual(df.shape[1], 16)
        self.assertEqual(df["sample_id"].nunique(), 3600)

    def test_03_cohort_structure_invariants(self):
        pipeline = RobustValidationPipeline()
        res = pipeline.verify_cohort_structure()
        self.assertTrue(res["passed"])
        self.assertEqual(res["total_cohorts"], 36)
        self.assertTrue(res["all_cohorts_100_rows"])
        self.assertTrue(res["no_multiple_properties"])
        self.assertTrue(res["no_multiple_types"])
        self.assertTrue(res["no_multiple_ages"])

    def test_04_locked_final_test_and_dev_splits(self):
        test_cohorts_df = pd.read_csv("data/splits/final_test_cohorts.csv")
        dev_cohorts_df = pd.read_csv("data/splits/development_cohorts.csv")
        test_rows_df = pd.read_csv("data/splits/final_test_rows.csv")
        dev_rows_df = pd.read_csv("data/splits/development_rows.csv")

        self.assertEqual(len(test_cohorts_df), 8)
        self.assertEqual(len(dev_cohorts_df), 28)
        self.assertEqual(len(test_rows_df), 800)
        self.assertEqual(len(dev_rows_df), 2800)

        # Zero cohort overlap
        test_cohorts = set(test_cohorts_df["experiment_id"])
        dev_cohorts = set(dev_cohorts_df["experiment_id"])
        self.assertEqual(len(test_cohorts.intersection(dev_cohorts)), 0)

        # Zero sample overlap
        test_sids = set(test_rows_df["sample_id"])
        dev_sids = set(dev_rows_df["sample_id"])
        self.assertEqual(len(test_sids.intersection(dev_sids)), 0)

    def test_05_grouped_cross_validation_folds(self):
        cv_df = pd.read_csv("data/splits/grouped_cv_assignments.csv")
        self.assertEqual(len(cv_df), 2800)
        self.assertEqual(sorted(cv_df["cv_fold"].unique().tolist()), [1, 2, 3, 4, 5])

        for fold in range(1, 6):
            train_sub = cv_df[cv_df["cv_fold"] != fold]
            val_sub = cv_df[cv_df["cv_fold"] == fold]

            train_cohorts = set(train_sub["experiment_id"])
            val_cohorts = set(val_sub["experiment_id"])
            train_sids = set(train_sub["sample_id"])
            val_sids = set(val_sub["sample_id"])

            self.assertEqual(len(train_cohorts.intersection(val_cohorts)), 0)
            self.assertEqual(len(train_sids.intersection(val_sids)), 0)

    def test_06_leave_age_out_folds(self):
        lao_df = pd.read_csv("data/splits/leave_age_out_assignments.csv")
        self.assertEqual(len(lao_df), 3600)

        ages = [7, 14, 21, 28, 56, 90]
        for age in ages:
            col = f"held_out_{age}d_split"
            self.assertIn(col, lao_df.columns)
            test_mask = (lao_df[col] == "test")
            train_mask = (lao_df[col] == "train")

            self.assertEqual(test_mask.sum(), 600)
            self.assertEqual(train_mask.sum(), 3000)
            self.assertTrue((lao_df.loc[test_mask, "curing_age_days"] == age).all())
            self.assertTrue((lao_df.loc[train_mask, "curing_age_days"] != age).all())

    def test_07_feature_policy_no_forbidden_in_candidate(self):
        self.assertEqual(len(CANDIDATE_FEATURES.intersection(FORBIDDEN_FEATURES)), 0)

    def test_08_all_8_leakage_checks_pass(self):
        pipeline = RobustValidationPipeline()
        dev_test = pipeline.create_locked_final_test_and_dev_splits(seed=42)
        cv = pipeline.create_5fold_grouped_cv(dev_test["dev_df"])
        lao = pipeline.create_leave_one_curing_age_out()
        res = pipeline.run_all_8_leakage_checks(dev_test, cv, lao)
        self.assertTrue(res["all_passed"])
        for chk_name, chk_val in res["checks"].items():
            self.assertTrue(chk_val["passed"], f"Check failed: {chk_name}")


if __name__ == "__main__":
    unittest.main()
