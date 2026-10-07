# Phase 4.3 Evaluation Report: 5-Fold Outer Cross-Validation of Fine-Tuned TabPFN (Run 2)

**Project:** AI-Based Prediction of Mechanical Properties of Normal and Bacterial Concrete  
**Execution Timestamp:** 2026-10-07T22:40:41Z  
**Hardware Platform:** AMD Ryzen 7 7445HS (6 cores, 12 logical processors), 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)  
**Execution Environment:** Python 3.14.7 64-bit, PyTorch 2.14.1 (CPU build utilized for deterministic stability), TabPFN 9.1.0, Scikit-Learn 1.9.1, Pandas 3.0.6, NumPy 2.5.3  

---

## 1. Executive Summary

Phase 4.3 executes the complete 5-fold outer cross-validation of **Fine-Tuned TabPFN v2.5** across the 2,800 development specimens using strict `experiment_id` grouping (GroupKFold). 
- Folds 1, 2, and 3 validation predictions and configurations (600 rows each) were verified for integrity and preserved per the continuation protocol.
- Folds 4 and 5 were trained fresh on the new machine with identical hyperparameters, features (`V2_ENGINEERED`), base checkpoint (`tabpfn-v2.5-regressor-v2.5_real.ckpt`, 40,831,868 bytes, SHA256 verified), and grouping architecture.
- Exactly **2,800 out-of-fold (OOF) predictions** were generated with zero duplicates, zero missing development specimens, and zero contamination from the permanently sealed 800-specimen final test set.

---

## 2. Hardware and Software Environment

- **Host Machine:** New PC migration completed.
- **CPU:** AMD Ryzen 7 7445HS with Radeon 740M Graphics (6 physical cores, 12 logical processors)
- **Host Memory:** 16.0 GB RAM
- **Discrete GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM), Driver 595.79
- **PyTorch & TabPFN Build:** PyTorch 2.14.1+cpu with TabPFN 9.1.0. TabPFN in-context fine-tuning completed in 86.8 seconds for Fold 4 and 81.5 seconds for Fold 5 on CPU.

---

## 3. Data Integrity & Verification

1. **Master Parquet Dataset (`data/master_dataset.parquet`)**:
   - SHA256: `0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a` (Exact bit-for-bit match, 3,600 rows × 16 columns).
2. **Master CSV Dataset (`data/master_dataset.csv`)**:
   - Canonical CRLF SHA256: `0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a` (Verified 100% identical data, 3,600 rows, 36 cohorts).
3. **Raw Laboratory Workbooks**:
   - C Workbook (`data/C Strenth Reading 1000000.xlsx`): `5605a0e0913de760752dd0aab707f1a66cbc5af130f9811485e958f624460f19` (PASS)
   - F Workbook (`data/F Strength Reading 1000000.xlsx`): `c28972bd289b860f051602a1877cce31c819b55066ea9afc55937357ff37b44a` (PASS)
   - T Workbook (`data/T Strenth Reading 1000000.xlsx`): `d8283fd9f8c4e75c57ac3696aa36f08895b4d63a997a86530fcd0ff5813ba645` (PASS)
4. **TabPFN Base Checkpoint (`models/tabpfn-v2.5-regressor-v2.5_real.ckpt`)**:
   - Size: Exactly 40,831,868 bytes.
   - SHA256: `8ab42d2d0abe3886a2c54a336b2abf251658a23a9b3f6559c4fdf0c024f48e63` (Exact match).
5. **Permanently Sealed Final Test Dataset (`data/splits/final_test_rows.csv`)**:
   - 800 rows across 8 cohorts. Completely untouched and unaccessed throughout Phase 4.3.

---

## 4. Fold-by-Fold Execution & Performance

| Fold | Outer Train Rows (Cohorts) | Outer Val Rows (Cohorts) | Fit Time | Pred Time | Outer Val MAE (MPa) | Outer Val RMSE (MPa) | Outer Val R² | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 2,200 (22) | 600 (6) | 520.3s | 87.6s | 5.1636 | 7.1085 | 0.7502 | Verified & Reused |
| **Fold 2** | 2,200 (22) | 600 (6) | 607.8s | 84.3s | 2.2350 | 4.6196 | 0.8500 | Verified & Reused |
| **Fold 3** | 2,200 (22) | 600 (6) | 599.1s | 76.4s | 1.9167 | 3.9787 | 0.8611 | Verified & Reused |
| **Fold 4** | 2,300 (23) | 500 (5) | 86.8s | 13.7s | 0.4961 | 0.9678 | 0.9861 | Trained Fresh & Verified |
| **Fold 5** | 2,300 (23) | 500 (5) | 81.5s | 14.1s | 0.9655 | 1.2807 | 0.9944 | Trained Fresh & Verified |

*Note: In `grouped_cv_assignments.csv`, the 28 development cohorts are distributed as 6, 6, 6, 5, 5 cohorts (600, 600, 600, 500, 500 rows), totaling exactly 2,800 rows.*

---

## 5. Pooled Out-of-Fold (OOF) Metrics (N = 2,800)

### Overall Pooled Performance
- **Pooled MAE:** **2.2571 MPa**
- **Pooled RMSE:** **4.3879 MPa**
- **Pooled R²:** **0.8872**
- **Pooled MAPE:** **33.74%**

### Breakdown by Mechanical Property

| Mechanical Property | Specimen Count (N) | MAE (MPa) | RMSE (MPa) | R² | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Compressive Strength** | 900 | 3.4449 | 5.1574 | 0.1080 | 12.01% |
| **Flexural Strength** | 900 | 2.1343 | 4.8612 | -38.1896 | 44.54% |
| **Split Tensile Strength** | 1,000 | 1.2987 | 2.9499 | -30.4857 | 43.59% |
| **Overall (Pooled Unified)** | **2,800** | **2.2571** | **4.3879** | **0.8872** | **33.74%** |

---

## 6. Comprehensive Model Comparison

Evaluating fine-tuned TabPFN against Pretrained TabPFN v2.5 and Classical Baselines (from Phase 4.1):

| Model | Feature Set | Strategy | Compressive MAE | Flexural MAE | Split Tensile MAE | Overall MAE | Overall RMSE | Overall R² |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fine-Tuned TabPFN v2.5** | **V2_ENGINEERED** | **Unified** | 3.4449 | 2.1343 | 1.2987 | 2.2571 | 4.3879 | 0.8872 |
| **Pretrained TabPFN v2.5** | **V2_ENGINEERED** | **Unified** | **1.6168** | 1.8116 | 0.4603 | 1.2664 | 2.3070 | 0.9688 |
| **XGBoost (Overall-Best)** | Expanded | Unified | 2.5192 | 0.4689 | **0.3148** | **1.0729** | **1.6862** | **0.9833** |
| **XGBoost (Comp-Best)** | Core | Unified | 2.3800 | 0.6863 | 0.7084 | 1.2386 | 1.7945 | 0.9811 |
| **RandomForest** | Core | Separate | 3.3204 | **0.4554** | 0.3311 | 1.3319 | 2.2439 | 0.9705 |
| **CatBoost** | Core | Unified | 2.4369 | 1.1307 | 0.8584 | 1.4533 | 2.1873 | 0.9720 |
| **Linear Regression** | Core | Unified | 3.3132 | 1.9738 | 1.8016 | 2.3428 | 3.2323 | 0.9388 |
| **Ridge Regression** | Core | Unified | 3.3107 | 1.9764 | 1.8061 | 2.3445 | 3.2373 | 0.9386 |

---

## 7. Scientific Insights & Analysis

1. **Pretrained vs. Fine-Tuned TabPFN**:
   - Pretrained TabPFN v2.5 achieved an overall MAE of **1.2664 MPa**, whereas Fine-Tuned TabPFN achieved **2.2571 MPa**.
   - The results are consistent with possible scale/regime effects in unified fine-tuning (compressive strength spans 15–45 MPa, whereas flexural and tensile strength span 1–8 MPa), but the present experiment evaluates only the unified strategy and does not independently establish causality.
   - Fold 4 and Fold 5 showed excellent generalization (MAE = 0.4961 MPa and 0.9655 MPa, R² > 0.98), whereas Fold 1 had higher error (MAE = 5.1636 MPa) due to out-of-distribution cohort age assignments.
2. **Best Model per Mechanical Property**:
   - **Compressive Strength:** Pretrained TabPFN v2.5 is the clear winner (**MAE = 1.6168 MPa**), outperforming the best classical compressive baseline (**XGBoost Core Unified: 2.3800 MPa**) by **32.07% (32.1%)**, and outperforming the overall-best classical model (**XGBoost Expanded Unified: 2.5192 MPa**) by **35.82% (35.8%)**.
   - **Flexural Strength:** Random Forest (Separate strategy, Core features) achieved the best performance (**MAE = 0.4554 MPa**).
   - **Split Tensile Strength:** XGBoost (Unified strategy, Expanded features) achieved the best performance (**MAE = 0.3148 MPa**).
3. **Audit Reconciliation Note (XGBoost Configurations)**:
   - XGBoost (Expanded, Unified) achieved the lowest overall MAE across all 24 classical configurations (1.0729 MPa; Compressive MAE = 2.5192 MPa).
   - XGBoost (Core, Unified) achieved the lowest compressive MAE among all 24 classical configurations (Compressive MAE = 2.3800 MPa; Overall MAE = 1.2386 MPa). Both numbers originate directly from `results/predictions/phase4_1_cv_predictions.csv` and are fully validated.
4. **Fold Stability**:
   - Fold MAE mean: 2.1554 ± 1.6298 MPa
   - Fold RMSE mean: 3.5911 ± 2.2716 MPa
   - Fold R² mean: 0.8884 ± 0.0918

---

## 8. Artifacts Generated

1. `results/models/tabpfn/fold_models_run2/fold_4/fold_4_finetuned.joblib` (84.1 MB)
2. `results/models/tabpfn/fold_models_run2/fold_4/fold_4_config.json`
3. `results/models/tabpfn/fold_models_run2/fold_4/fold_4_val_predictions.csv` (500 rows)
4. `results/models/tabpfn/fold_models_run2/fold_5/fold_5_finetuned.joblib` (84.1 MB)
5. `results/models/tabpfn/fold_models_run2/fold_5/fold_5_config.json`
6. `results/models/tabpfn/fold_models_run2/fold_5/fold_5_val_predictions.csv` (500 rows)
7. `results/predictions/tabpfn_finetuned_oof_predictions_run2.csv` (2,800 rows)
8. `reports/tabpfn_finetuned_model_comparison_run2.csv`
9. `results/tabpfn_finetuned_run2/metadata.json`
10. `reports/figures/tabpfn_finetuned_run2/*.png` (8 publication-quality figures)
