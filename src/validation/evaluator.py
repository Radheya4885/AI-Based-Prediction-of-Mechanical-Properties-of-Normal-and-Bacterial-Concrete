"""
Model Evaluator Module
Provides standard, multi-dimensional metric computation across mechanical properties,
concrete types, curing ages, and tracks restored values separately.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """Computes regression metrics with multi-dimensional slicing and restored value isolation."""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates primary (MAE), secondary (RMSE, R2), and guarded MAPE metrics."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        n_samples = len(y_true)
        if n_samples == 0:
            return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "mape": np.nan, "count": 0}
            
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        # R2 score requires at least 2 distinct values and variance > 0
        if n_samples > 1 and np.var(y_true) > 1e-9:
            r2 = float(r2_score(y_true, y_pred))
        else:
            r2 = np.nan
            
        # Guarded MAPE to avoid division by zero
        epsilon = 1e-6
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100.0)
        
        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4) if not np.isnan(r2) else None,
            "mape": round(mape, 4),
            "count": int(n_samples),
        }

    @classmethod
    def evaluate_predictions(
        cls,
        test_df: pd.DataFrame,
        y_pred: np.ndarray,
        target_col: str = "strength_mpa",
    ) -> Dict[str, Any]:
        """Performs comprehensive multi-dimensional evaluation.
        
        Args:
            test_df: DataFrame containing test observations with metadata columns:
                     ['mechanical_property', 'concrete_type', 'curing_age_days', 'is_restored_value'].
            y_pred: Array of predicted values corresponding to test_df rows.
            target_col: Target column name in test_df.
            
        Returns:
            Dictionary with overall metrics, restored-value breakdowns, and sub-group metrics.
        """
        df_eval = test_df.copy()
        df_eval["y_true"] = df_eval[target_col].values
        df_eval["y_pred"] = y_pred
        
        # 1. Overall Performance
        overall_all = cls.calculate_metrics(df_eval["y_true"], df_eval["y_pred"])
        
        # 2. Restored Value Handling (Section 5 requirement)
        mask_non_restored = ~df_eval["is_restored_value"].astype(bool)
        mask_restored = df_eval["is_restored_value"].astype(bool)
        
        metrics_excluding_restored = cls.calculate_metrics(
            df_eval.loc[mask_non_restored, "y_true"],
            df_eval.loc[mask_non_restored, "y_pred"],
        )
        
        has_restored = mask_restored.sum() > 0
        metrics_restored_only = (
            cls.calculate_metrics(
                df_eval.loc[mask_restored, "y_true"],
                df_eval.loc[mask_restored, "y_pred"],
            )
            if has_restored
            else None
        )
        
        restored_breakdown = {
            "all_observations": overall_all,
            "excluding_restored": metrics_excluding_restored,
            "restored_only": metrics_restored_only,
            "restored_count_in_test": int(mask_restored.sum()),
        }
        
        # 3. By Mechanical Property
        by_property = {}
        for prop in sorted(df_eval["mechanical_property"].unique()):
            sub = df_eval[df_eval["mechanical_property"] == prop]
            by_property[prop] = cls.calculate_metrics(sub["y_true"], sub["y_pred"])
            
        # 4. By Concrete Type
        by_concrete_type = {}
        for ctype in sorted(df_eval["concrete_type"].unique()):
            sub = df_eval[df_eval["concrete_type"] == ctype]
            by_concrete_type[ctype] = cls.calculate_metrics(sub["y_true"], sub["y_pred"])
            
        # 5. By Curing Age
        by_curing_age = {}
        for age in sorted(df_eval["curing_age_days"].unique()):
            sub = df_eval[df_eval["curing_age_days"] == age]
            by_curing_age[f"{age}_days"] = cls.calculate_metrics(sub["y_true"], sub["y_pred"])
            
        return {
            "overall": overall_all,
            "restored_value_breakdown": restored_breakdown,
            "by_mechanical_property": by_property,
            "by_concrete_type": by_concrete_type,
            "by_curing_age_days": by_curing_age,
        }
