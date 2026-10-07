# Audit of Missing vs. Not-Required Model Artifacts (Run 1)

**Project:** AI-Based Prediction of Mechanical Properties of Normal and Bacterial Concrete  
**Audit Date:** 2026-10-07  
**Scope:** Rigorous examination of all potential model binaries across Phases 4.1, 4.2, 4.3, and 5 to distinguish missing artifacts from artifacts intentionally not serialized by workflow design.

---

## 1. Executive Summary

A critical principle of reproducible machine learning audits is that **a prediction CSV is not a model binary**, but **the absence of a serialized model binary does not necessarily indicate a missing research artifact**.

- **Classical Baselines (Phase 4.1):** Zero model binaries exist on disk. This is **by design**: `src/modeling/baseline_training.py` was structured to compute cross-validation folds, output predictions, log metrics, and terminate without dumping `.joblib` estimators.
- **Pretrained TabPFN (Phase 4.2 & Phase 5):** The foundational model binary `tabpfn-v2.5-regressor-v2.5_real.ckpt` (40.8 MB) is **present, verified, and intact**. TabPFN is an in-context learning transformer; fitted instances do not require separate serialization because inference on any input vector is executed on-demand in fractions of a second.
- **Fine-Tuned TabPFN (Phase 4.3):** Folds 4 and 5 `.joblib` binaries are **present and verified** (84.1 MB each). Folds 1, 2, and 3 `.joblib` binaries were missing from the initial laptop transfer. However, their validation predictions ($N=600$ per fold) and configurations are completely preserved, and the combined 2,800-row out-of-fold prediction dataset is intact.

---

## 2. Model Artifact Audit Table

| Expected Artifact | Found? | Filesystem Path | Why It Matters | Required for Thesis? | Recovery Possible? | Action Recommended |
| :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **Dummy Baseline Binaries** (4 configs) | **NO** | *None* | Baseline reference | **NO** | YES (Trivial) | **NO SERIALIZED MODEL REQUIRED BY ORIGINAL PHASE 4.1 WORKFLOW.** Predictions preserved. |
| **Linear Regression Binaries** (4 configs) | **NO** | *None* | Parametric reference | **NO** | YES (Trivial) | **NO SERIALIZED MODEL REQUIRED BY ORIGINAL PHASE 4.1 WORKFLOW.** Predictions preserved. |
| **Ridge Regression Binaries** (4 configs) | **NO** | *None* | Regularized linear reference | **NO** | YES (Trivial) | **NO SERIALIZED MODEL REQUIRED BY ORIGINAL PHASE 4.1 WORKFLOW.** Predictions preserved. |
| **Random Forest Binaries** (4 configs) | **NO** | *None* | Non-linear bagging baseline | **NO** | YES (<2 min) | **NO SERIALIZED MODEL REQUIRED BY ORIGINAL PHASE 4.1 WORKFLOW.** Predictions preserved. |
| **XGBoost Binaries** (4 configs) | **NO** | *None* | Top classical baseline (MAE 1.0729) | **NO** | YES (<3 min) | **NO SERIALIZED MODEL REQUIRED BY ORIGINAL PHASE 4.1 WORKFLOW.** Predictions preserved. |
| **CatBoost Binaries** (4 configs) | **NO** | *None* | Boosting reference | **NO** | YES (<3 min) | **NO SERIALIZED MODEL REQUIRED BY ORIGINAL PHASE 4.1 WORKFLOW.** Predictions preserved. |
| **Base TabPFN v2.5 Checkpoint** | **YES** | `models/tabpfn-v2.5-regressor-v2.5_real.ckpt` | Foundation model weights (40.8 MB) | **YES** | *Present* | Verified (SHA256: `8ab42d2d0abe...`). Do not modify. |
| **Fine-Tuned TabPFN Fold 1 Binary** | **NO** | `results/models/tabpfn/fold_models_run2/fold_1/` | Serialized estimator for Fold 1 | **NO** | YES (~15 min) | `fold_1_val_predictions.csv` & config exist. Re-training optional; not needed for thesis text. |
| **Fine-Tuned TabPFN Fold 2 Binary** | **NO** | `results/models/tabpfn/fold_models_run2/fold_2/` | Serialized estimator for Fold 2 | **NO** | YES (~15 min) | `fold_2_val_predictions.csv` & config exist. Re-training optional; not needed for thesis text. |
| **Fine-Tuned TabPFN Fold 3 Binary** | **NO** | `results/models/tabpfn/fold_models_run2/fold_3/` | Serialized estimator for Fold 3 | **NO** | YES (~15 min) | `fold_3_val_predictions.csv` & config exist. Re-training optional; not needed for thesis text. |
| **Fine-Tuned TabPFN Fold 4 Binary** | **YES** | `results/models/tabpfn/fold_models_run2/fold_4/fold_4_finetuned.joblib` | Serialized estimator for Fold 4 | **YES** | *Present* | Verified (84,093,937 bytes). Preserved. |
| **Fine-Tuned TabPFN Fold 5 Binary** | **YES** | `results/models/tabpfn/fold_models_run2/fold_5/fold_5_finetuned.joblib` | Serialized estimator for Fold 5 | **YES** | *Present* | Verified (84,093,633 bytes). Preserved. |
| **Final Phase 5 TabPFN Model Binary** | **NO** (By Design)| *In-memory transformer context* | Final locked test model instance | **NO** | YES (0.2s) | Re-instantiated on-demand in memory from base checkpoint + dev set in 0.2s. |

---

## 3. Scientific Implications for Thesis and Publication

1. **Thesis Completeness:**
   - Academic manuscripts and engineering theses report **empirical predictions, statistical metrics, validation methodology, and comparison charts**. All of these are 100% preserved in the 4 canonical prediction CSVs (`phase4_1_cv_predictions.csv`, `tabpfn_cv_predictions.csv`, `tabpfn_finetuned_oof_predictions_run2.csv`, `phase5_final_test_predictions.csv`).
2. **Reviewer Reproduction Requirements:**
   - If an external reviewer requires generating predictions for *arbitrary brand-new unobserved concrete mix designs*, the base TabPFN foundation model checkpoint and training scripts are completely present.
3. **No Retraining Required:**
   - Retraining Folds 1–3 of Fine-Tuned TabPFN is purely optional. Since Fine-Tuned TabPFN underperformed Pretrained TabPFN overall (MAE 2.2571 vs 1.2664 MPa), Fine-Tuned TabPFN serves as an exploratory comparative study rather than the primary selected model.
