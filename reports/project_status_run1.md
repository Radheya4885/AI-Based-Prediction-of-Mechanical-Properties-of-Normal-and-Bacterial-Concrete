# End-to-End Project Status Map (Phases 1–5) (Run 1)

**Project:** AI-Based Prediction of Mechanical Properties of Normal and Bacterial Concrete  
**Audit Date:** 2026-10-07  
**Status:** ALL COMPUTATIONAL RESEARCH AND LOCKED EVALUATIONS OFFICIALLY COMPLETE.

---

## Phase-by-Phase Comprehensive Status Summary

```
========================================================================================
PHASE 1: Raw Excel Dataset Audit                        --> STATUS: COMPLETE & SEALED
PHASE 2: Canonical Master Dataset Construction          --> STATUS: COMPLETE & SEALED
PHASE 3: Leakage-Safe Validation Strategy & Splits      --> STATUS: COMPLETE & SEALED
PHASE 4.1: Classical Baseline Benchmarking (24 Configs)  --> STATUS: COMPLETE & VERIFIED
PHASE 4.2: Pretrained TabPFN v2.5 Benchmarking          --> STATUS: COMPLETE & VERIFIED
PHASE 4.3: Fine-Tuned TabPFN 5-Fold Cross-Validation    --> STATUS: COMPLETE & VERIFIED
PHASE 5: Final Locked Test Set Evaluation               --> STATUS: COMPLETE & SEALED
========================================================================================
```

---

### PHASE 1: Raw Excel Dataset Audit
- **Status:** **COMPLETE & IMMUTABLE**
- **Major Artifacts:**
  - `data/C Strenth Reading 1000000.xlsx` (Compressive Strength, 12 sheets)
  - `data/F Strength Reading 1000000.xlsx` (Flexural Strength, 12 sheets)
  - `data/T Strenth Reading 1000000.xlsx` (Split Tensile Strength, 12 sheets)
- **Reports:** [`reports/dataset_audit.md`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/dataset_audit.md), [`reports/eda_report.md`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/eda_report.md)
- **Known Limitations:**
  - Historical laboratory Sheet 2 restoration in Flexural testing (`EXP_FS_BC_56d` and `EXP_FS_BC_90d`) identified and isolated.

---

### PHASE 2: Canonical Master Dataset Construction
- **Status:** **COMPLETE & CRYPTOGRAPHICALLY LOCKED**
- **Major Artifacts:**
  - `data/master_dataset.csv` (3,600 rows, 16 columns, SHA256 verified)
  - `data/master_dataset.parquet` (3,600 rows, 16 columns, SHA256 verified)
  - `data/tabpfn_concrete_v1/tabpfn_concrete_model.csv` (11 features partition)
- **Reports:** [`reports/master_dataset_design.md`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/master_dataset_design.md), [`reports/master_dataset_validation.md`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/master_dataset_validation.md)
- **Known Limitations:**
  - Single experimental campaign domain (OPC 53 cement, *Bacillus subtilis* at $10^6$ cells/mL).

---

### PHASE 3: Leakage-Safe Validation Strategy & Splits
- **Status:** **COMPLETE & STRICTLY VERIFIED (0 LEAKAGE)**
- **Major Artifacts:**
  - `data/splits/final_test_rows.csv` (800 rows, 8 cohorts permanently sealed)
  - `data/splits/development_rows.csv` (2,800 rows, 28 cohorts)
  - `data/splits/grouped_cv_assignments.csv` (5-fold GroupKFold cohort CV)
- **Reports:** [`reports/validation_strategy_v3_1.md`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/validation_strategy_v3_1.md)
- **Known Limitations:**
  - Discrete cohort granularity (uneven fold sizes: three 6-cohort folds, two 5-cohort folds).

---

### PHASE 4.1: Classical Baseline Benchmarking
- **Status:** **COMPLETE & VERIFIED**
- **Major Artifacts:**
  - Predictions: `results/predictions/phase4_1_cv_predictions.csv` (67,200 predictions across 24 configurations)
  - Metrics: `reports/baseline_model_comparison.csv` (72 evaluated slices)
  - Metadata: `results/phase4_1_run_metadata.json`
- **Important Model Binaries:**
  - None (workflow designed to output predictions and metrics directly; no serialized model required).
- **Reports:** [`reports/baseline_modeling_report.md`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/baseline_modeling_report.md)
- **Key Findings:**
  - XGBoost (Expanded, Unified) achieved best classical overall pooled MAE = **1.0729 MPa**.
  - XGBoost (Core, Unified) achieved best classical compressive MAE = **2.3800 MPa**.

---

### PHASE 4.2: Pretrained TabPFN v2.5 Benchmarking
- **Status:** **COMPLETE & VERIFIED**
- **Major Artifacts:**
  - Model Binary: `models/tabpfn-v2.5-regressor-v2.5_real.ckpt` (40.8 MB foundation model weights)
  - Predictions: `results/predictions/tabpfn_cv_predictions.csv` (5,600 predictions)
  - Metrics: `reports/tabpfn_model_comparison.csv`
- **Reports:** [`reports/tabpfn_training_report.md`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/tabpfn_training_report.md)
- **Key Findings:**
  - Pretrained TabPFN v2.5 (V2_ENGINEERED, Unified) achieved Compressive MAE = **1.6168 MPa** (32.1% lower error than XGBoost Core).
  - Overall pooled OOF MAE = **1.2664 MPa**.

---

### PHASE 4.3: Fine-Tuned TabPFN 5-Fold Cross-Validation
- **Status:** **COMPLETE & ASSEMBLED**
- **Major Artifacts:**
  - Predictions: `results/predictions/tabpfn_finetuned_oof_predictions_run2.csv` (2,800 rows complete OOF dataset)
  - Individual Fold Predictions: `fold_1` to `fold_5` val predictions (600, 600, 600, 500, 500 rows)
  - Model Binaries: `fold_4_finetuned.joblib` (84.1 MB), `fold_5_finetuned.joblib` (84.1 MB)
  - Metrics: `reports/tabpfn_finetuned_model_comparison_run2.csv`
- **Reports:** [`reports/tabpfn_finetuned_evaluation_report_run2.md`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/tabpfn_finetuned_evaluation_report_run2.md)
- **Key Findings:**
  - Fine-tuning did not improve overall performance over pretrained foundation weights (Overall MAE worsened from 1.2664 to 2.2571 MPa). Pretrained TabPFN confirmed as the superior architecture.

---

### PHASE 5: Final Locked Test Set Evaluation
- **Status:** **COMPLETE, SEALED, AND OFFICIALLY CLOSED**
- **Major Artifacts:**
  - Predictions: `results/predictions/phase5_final_test_predictions.csv` (800 rows, 8 unseen test cohorts)
  - Metadata: `results/phase5/phase5_metadata.json`
  - Figures: 5 publication plots in `reports/figures/phase5/`
- **Reports:**
  - [`reports/phase5_final_evaluation_report.md`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/phase5_final_evaluation_report.md)
  - [`reports/final_phase5_scientific_audit.md`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/final_phase5_scientific_audit.md)
- **Key Findings:**
  - Compressive Strength generalized solidly (MAE = 2.1052 MPa, $R^2 = 0.8257$, MAPE = 9.90%).
  - Normal Concrete generalized with high precision across all properties (MAE = 1.0740 MPa, $R^2 = 0.9489$).
  - Overall test MAE = 2.9343 MPa ($R^2 = 0.7993$); excluding the known restored cohort (`EXP_FS_BC_90d`), MAE drops to 1.5972 MPa ($R^2 = 0.9607$).
  - Documentation rationale corrected and reconciled with Phase 4.1 classical table.
