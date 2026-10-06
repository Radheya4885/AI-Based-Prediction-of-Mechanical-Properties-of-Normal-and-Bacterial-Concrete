"""
Concrete Strength Prediction - Validation Framework Package
Provides leak-free, reproducible validation splits, feature policies,
automated leakage checks, and comprehensive multi-dimensional metric evaluators.
"""

from .cohort_analyzer import CohortAnalyzer
from .feature_policy import FeaturePolicy, FORBIDDEN_FEATURES, CANDIDATE_FEATURES, REDUNDANT_FEATURES
from .split_generator import SplitGenerator
from .evaluator import ModelEvaluator
from .leakage_checker import DataLeakageChecker

__all__ = [
    "CohortAnalyzer",
    "FeaturePolicy",
    "FORBIDDEN_FEATURES",
    "CANDIDATE_FEATURES",
    "REDUNDANT_FEATURES",
    "SplitGenerator",
    "ModelEvaluator",
    "DataLeakageChecker",
]
