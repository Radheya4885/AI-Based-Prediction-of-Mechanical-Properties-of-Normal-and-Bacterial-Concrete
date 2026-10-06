"""
Automated Unit Tests for Validation Package
Runs comprehensive tests using the Python standard library unittest.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd

# Add workspace root to sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from src.validation.cohort_analyzer import CohortAnalyzer
from src.validation.feature_policy import FeaturePolicy, FORBIDDEN_FEATURES
from src.validation.split_generator import SplitGenerator
from src.validation.evaluator import ModelEvaluator
from src.validation.leakage_checker import DataLeakageChecker


class TestValidationFramework(unittest.TestCase):
    """Test suite for the validation framework components."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv("data/master_dataset.csv")

    def test_cohort_analyzer(self):
        analyzer = CohortAnalyzer(self.df)
        invariants = analyzer.verify_cohort_invariants()
        self.assertTrue(invariants["passed"])
        self.assertEqual(invariants["total_cohorts"], 36)
        self.assertTrue(invariants["all_cohorts_have_100_rows"])
        self.assertEqual(invariants["property_cohort_counts"]["Compressive Strength"], 12)
        self.assertEqual(invariants["property_cohort_counts"]["Flexural Strength"], 12)
        self.assertEqual(invariants["property_cohort_counts"]["Split Tensile Strength"], 12)

    def test_feature_policy(self):
        # Canonical features should pass validation
        canonical_features = FeaturePolicy.get_feature_set_1_canonical(is_unified=True)
        try:
            FeaturePolicy.validate_features(canonical_features)
        except ValueError:
            self.fail("validate_features raised ValueError on valid features")

        # Passing any forbidden feature should raise ValueError
        for forbidden in FORBIDDEN_FEATURES:
            with self.assertRaises(ValueError):
                FeaturePolicy.validate_features(["curing_age_days", forbidden])

    def test_split_generator(self):
        gen = SplitGenerator(self.df, output_dir="data/splits")
        splits = gen.generate_all_splits(seed=42)
        
        split_a = splits["scenario_a"]
        split_b = splits["scenario_b"]
        split_c = splits["scenario_c"]

        # Scenario A assertions
        self.assertEqual(split_a["train_row_count"], 2880)
        self.assertEqual(split_a["test_row_count"], 720)
        self.assertEqual(len(set(split_a["train_sample_ids"]).intersection(set(split_a["test_sample_ids"]))), 0)

        # Scenario B assertions
        self.assertEqual(split_b["train_row_count"], 2800)
        self.assertEqual(split_b["test_row_count"], 800)
        self.assertEqual(split_b["train_cohort_count"], 28)
        self.assertEqual(split_b["test_cohort_count"], 8)
        self.assertEqual(len(set(split_b["train_cohorts"]).intersection(set(split_b["test_cohorts"]))), 0)
        self.assertEqual(len(set(split_b["train_sample_ids"]).intersection(set(split_b["test_sample_ids"]))), 0)

        # Scenario C assertions
        self.assertEqual(split_c["total_folds"], 6)
        for fold_key, fold in split_c["folds"].items():
            age = fold["held_out_age_days"]
            self.assertEqual(fold["train_row_count"], 3000)
            self.assertEqual(fold["test_row_count"], 600)
            train_df = self.df[self.df["sample_id"].isin(fold["train_sample_ids"])]
            test_df = self.df[self.df["sample_id"].isin(fold["test_sample_ids"])]
            self.assertEqual((train_df["curing_age_days"] == age).sum(), 0)
            self.assertEqual((test_df["curing_age_days"] != age).sum(), 0)

    def test_model_evaluator_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 32.0])
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(metrics["mae"], 1.3333, places=3)
        self.assertTrue(metrics["rmse"] > 0)
        self.assertTrue(metrics["r2"] > 0.9)
        self.assertEqual(metrics["count"], 3)

    def test_model_evaluator_restored_isolation(self):
        # Create a mock test dataframe with both normal and restored rows
        mock_test = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3", "S4"],
            "mechanical_property": ["Flexural Strength"] * 4,
            "concrete_type": ["Bacterial Concrete"] * 4,
            "curing_age_days": [56, 56, 90, 90],
            "is_restored_value": [False, False, True, True],
            "strength_mpa": [5.0, 5.2, 5.8, 6.0],
        })
        mock_pred = np.array([5.1, 5.3, 5.9, 6.1])
        
        eval_res = ModelEvaluator.evaluate_predictions(mock_test, mock_pred)
        rb = eval_res["restored_value_breakdown"]
        
        self.assertEqual(rb["restored_count_in_test"], 2)
        self.assertEqual(rb["all_observations"]["count"], 4)
        self.assertEqual(rb["excluding_restored"]["count"], 2)
        self.assertEqual(rb["restored_only"]["count"], 2)

    def test_leakage_checker_enforcement(self):
        checker = DataLeakageChecker(self.df)
        
        # Test intentional violation detection
        bad_split_a = {
            "train_sample_ids": ["CS_NC_07d_001", "CS_NC_07d_002"],
            "test_sample_ids": ["CS_NC_07d_002", "CS_NC_07d_003"],
        }
        res_a = checker.verify_scenario_a(bad_split_a)
        self.assertFalse(res_a["passed"])
        self.assertEqual(res_a["sample_id_overlap_count"], 1)

        # Test intentional feature blacklist violation
        res_feat = checker.verify_feature_policy(["curing_age_days", "sample_id"])
        self.assertFalse(res_feat["passed"])
        self.assertIn("sample_id", res_feat["forbidden_features_detected"])


if __name__ == "__main__":
    unittest.main()
