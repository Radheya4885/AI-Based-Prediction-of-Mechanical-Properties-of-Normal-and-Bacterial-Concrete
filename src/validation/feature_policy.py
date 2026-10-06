"""
Feature Policy Module
Governs the classification, selection, and strict leakage prevention for input features.
"""

from typing import List, Dict, Set, Any


# Strict blacklist of columns that MUST NEVER be fed as predictive inputs to any ML model
FORBIDDEN_FEATURES: Set[str] = {
    "sample_id",           # Specimen primary key identifier
    "experiment_id",       # Cohort grouping variable (causes severe memorization)
    "sample_replicate",    # Mold index / replicate number (1-100)
    "source_file",         # Lineage metadata
    "source_sheet",        # Lineage metadata
    "source_row_index",    # Lineage metadata (correlated with curing age blocks)
    "is_restored_value",   # Direct target leakage indicator (flags 56d/90d bacterial flexure)
}

# Constant or zero-variance columns that provide no predictive entropy
CONSTANT_FEATURES: Set[str] = {
    "cement_type",         # 100% constant 'OPC 53 (UltraTech)'
}

# Redundant columns that duplicate information from other canonical features
REDUNDANT_FEATURES: Set[str] = {
    "bacterial_species",   # 100% collinear with concrete_type ('None' vs 'Bacillus subtilis')
    "bacterial_status",    # 100% collinear with concrete_type ('Control' vs 'Inoculated')
    "specimen_geometry",   # 100% collinear with mechanical_property
}

# Candidate legitimate predictive features
CANDIDATE_FEATURES: Set[str] = {
    "curing_age_days",
    "concrete_type",
    "bacterial_concentration_cells_ml",
    "mechanical_property",  # Used for Unified model
}


class FeaturePolicy:
    """Manages feature sets for modeling configurations and enforces leakage-free policies."""

    @staticmethod
    def get_feature_set_1_canonical(is_unified: bool = False) -> List[str]:
        """Feature Set 1: Minimal Canonical Representation (Curing Age + Categorical Treatment)."""
        base = ["curing_age_days", "concrete_type"]
        if is_unified:
            base.append("mechanical_property")
        return base

    @staticmethod
    def get_feature_set_2_dosage(is_unified: bool = False) -> List[str]:
        """Feature Set 2: Continuous Dosage Representation (Curing Age + Numerical Concentration)."""
        base = ["curing_age_days", "bacterial_concentration_cells_ml"]
        if is_unified:
            base.append("mechanical_property")
        return base

    @staticmethod
    def get_feature_set_3_extended(is_unified: bool = False) -> List[str]:
        """Feature Set 3: Extended / Ablation Representation (Includes collinear geometry and concentration)."""
        base = [
            "curing_age_days",
            "concrete_type",
            "bacterial_concentration_cells_ml",
        ]
        if is_unified:
            base.extend(["mechanical_property", "specimen_geometry"])
        return base

    @staticmethod
    def validate_features(features: List[str]) -> None:
        """Validates that a proposed feature list does not contain any forbidden features.
        
        Raises:
            ValueError: If any forbidden feature is detected.
        """
        forbidden_present = set(features).intersection(FORBIDDEN_FEATURES)
        if forbidden_present:
            raise ValueError(
                f"DATA LEAKAGE VIOLATION: Forbidden features detected in model inputs: {sorted(forbidden_present)}. "
                "These attributes are reserved exclusively for cohort grouping, lineage tracking, or audit flags."
            )

    @staticmethod
    def describe_policy() -> Dict[str, Any]:
        """Returns structured metadata of the feature governance policy."""
        return {
            "forbidden_features": sorted(FORBIDDEN_FEATURES),
            "constant_features": sorted(CONSTANT_FEATURES),
            "redundant_features": sorted(REDUNDANT_FEATURES),
            "candidate_features": sorted(CANDIDATE_FEATURES),
            "supported_feature_sets": {
                "set_1_canonical": {
                    "unified": FeaturePolicy.get_feature_set_1_canonical(is_unified=True),
                    "separate": FeaturePolicy.get_feature_set_1_canonical(is_unified=False),
                    "rationale": "Minimal parsimonious features without collinear redundancies.",
                },
                "set_2_dosage": {
                    "unified": FeaturePolicy.get_feature_set_2_dosage(is_unified=True),
                    "separate": FeaturePolicy.get_feature_set_2_dosage(is_unified=False),
                    "rationale": "Numerical concentration dosage representation (0 vs 1,000,000 cells/mL).",
                },
                "set_3_extended": {
                    "unified": FeaturePolicy.get_feature_set_3_extended(is_unified=True),
                    "separate": FeaturePolicy.get_feature_set_3_extended(is_unified=False),
                    "rationale": "Ablation set to evaluate model robustness to collinear attributes.",
                },
            },
        }
