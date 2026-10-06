"""
Cohort Analyzer Module
Analyzes and verifies the experimental cohorts in the Master Dataset.
Primary cohort identifier: experiment_id.
"""

import pandas as pd
from typing import Dict, Any, List


class CohortAnalyzer:
    """Analyzes and validates the experimental cohorts in the concrete testing dataset."""

    def __init__(self, data: pd.DataFrame):
        self.df = data
        self.cohort_summary = self._analyze_cohorts()

    @classmethod
    def from_csv(cls, path: str = "data/master_dataset.csv") -> "CohortAnalyzer":
        df = pd.read_csv(path)
        return cls(df)

    def _analyze_cohorts(self) -> pd.DataFrame:
        """Groups data by experiment_id and computes verification statistics."""
        summary = (
            self.df.groupby("experiment_id")
            .agg(
                row_count=("sample_id", "count"),
                mechanical_property=("mechanical_property", "first"),
                property_uniques=("mechanical_property", "nunique"),
                concrete_type=("concrete_type", "first"),
                type_uniques=("concrete_type", "nunique"),
                curing_age_days=("curing_age_days", "first"),
                age_uniques=("curing_age_days", "nunique"),
                bacterial_concentration=("bacterial_concentration_cells_ml", "first"),
                specimen_geometry=("specimen_geometry", "first"),
                restored_count=("is_restored_value", "sum"),
                min_strength=("strength_mpa", "min"),
                max_strength=("strength_mpa", "max"),
                mean_strength=("strength_mpa", "mean"),
                std_strength=("strength_mpa", "std"),
            )
            .reset_index()
        )
        return summary

    def verify_cohort_invariants(self) -> Dict[str, Any]:
        """Verifies strict cohort integrity requirements."""
        total_cohorts = len(self.cohort_summary)
        expected_cohorts = 36
        
        # Verify 100 rows per cohort
        rows_per_cohort = self.cohort_summary["row_count"].to_dict()
        all_100_rows = (self.cohort_summary["row_count"] == 100).all()
        
        # Verify single property, type, and age per cohort
        single_prop = (self.cohort_summary["property_uniques"] == 1).all()
        single_type = (self.cohort_summary["type_uniques"] == 1).all()
        single_age = (self.cohort_summary["age_uniques"] == 1).all()
        
        # Verify property distribution (12 cohorts each)
        property_counts = self.cohort_summary["mechanical_property"].value_counts().to_dict()
        type_counts = self.cohort_summary["concrete_type"].value_counts().to_dict()
        age_counts = self.cohort_summary["curing_age_days"].value_counts().to_dict()
        
        passed = (
            total_cohorts == expected_cohorts
            and all_100_rows
            and single_prop
            and single_type
            and single_age
        )
        
        return {
            "passed": passed,
            "total_cohorts": total_cohorts,
            "all_cohorts_have_100_rows": bool(all_100_rows),
            "single_property_per_cohort": bool(single_prop),
            "single_type_per_cohort": bool(single_type),
            "single_age_per_cohort": bool(single_age),
            "property_cohort_counts": property_counts,
            "type_cohort_counts": type_counts,
            "age_cohort_counts": age_counts,
            "restored_cohorts": self.cohort_summary[self.cohort_summary["restored_count"] > 0][
                ["experiment_id", "mechanical_property", "concrete_type", "curing_age_days", "restored_count"]
            ].to_dict(orient="records"),
        }

    def get_cohort_table(self) -> pd.DataFrame:
        """Returns ordered cohort summary table."""
        return self.cohort_summary.sort_values(
            ["mechanical_property", "concrete_type", "curing_age_days"]
        ).reset_index(drop=True)
