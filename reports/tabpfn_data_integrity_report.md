# TabPFN Stage Data Integrity & Safety Audit Report

**Audit Mode:** Post-Execution Verification  
**Audit Timestamp:** 2026-10-01  
**Scope:** Phase 4.2 TabPFN Dataset Copy, Feature Engineering, and Model Training Pipeline  

---

## 1. Cryptographic Checksum Assertions

All primary research files were hashed using SHA-256 before and after the TabPFN stage. Zero bytes were modified in any canonical or locked file.

| File Description | Path | Canonical SHA256 | Observed SHA256 Post-Run | Integrity Status |
| :--- | :--- | :--- | :--- | :---: |
| **Canonical Master CSV** | `data/master_dataset.csv` | `0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a` | `0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a` | ✅ **PASS (UNTOUCHED)** |
| **Canonical Master Parquet** | `data/master_dataset.parquet` | `0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a` | `0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a` | ✅ **PASS (UNTOUCHED)** |
| **Raw Compressive Workbook** | `data/C Strenth Reading 1000000.xlsx` | `5605a0e0913de760752dd0aab707f1a66cbc5af130f9811485e958f624460f19` | `5605a0e0913de760752dd0aab707f1a66cbc5af130f9811485e958f624460f19` | ✅ **PASS (UNTOUCHED)** |
| **Raw Flexural Workbook** | `data/F Strength Reading 1000000.xlsx` | `c28972bd289b860f051602a1877cce31c819b55066ea9afc55937357ff37b44a` | `c28972bd289b860f051602a1877cce31c819b55066ea9afc55937357ff37b44a` | ✅ **PASS (UNTOUCHED)** |
| **Raw Tensile Workbook** | `data/T Strenth Reading 1000000.xlsx` | `d8283fd9f8c4e75c57ac3696aa36f08895b4d63a997a86530fcd0ff5813ba645` | `d8283fd9f8c4e75c57ac3696aa36f08895b4d63a997a86530fcd0ff5813ba645` | ✅ **PASS (UNTOUCHED)** |

---

## 2. Partition & Split Invariant Assertions

| Invariant Item | Expected Count | Observed Count | Verification Status |
| :--- | :--- | :--- | :---: |
| **Locked Final Test Rows** | Exactly `800` rows | 800 rows (`data/splits/final_test_rows.csv`) | ✅ **PASS (SEALED)** |
| **Locked Final Test Cohorts** | Exactly `8` cohorts | 8 cohorts (`data/splits/final_test_cohorts.csv`) | ✅ **PASS (SEALED)** |
| **Development Rows** | Exactly `2,800` rows | 2,800 rows (`data/splits/development_rows.csv`) | ✅ **PASS** |
| **Development Cohorts** | Exactly `28` cohorts | 28 cohorts (`data/splits/development_cohorts.csv`) | ✅ **PASS** |

---

## 3. Strict Compliance Checks

1. **Zero Numerical Modifications to Real Measurements**:
   - `strength_mpa` values in both `master_dataset.csv` and `tabpfn_concrete_model.csv` were verified element-by-element against raw measurements. Zero discrepancies found.
2. **Zero Synthetic Observations**:
   - No rows were fabricated, added, or removed. Row counts in both datasets remain strictly 3,600.
3. **Zero Test Set Leakage**:
   - The 800 final test rows were **never loaded** by the training pipeline. All 5 cross-validation folds, feature evaluations, and fine-tuning steps operated exclusively on the 2,800 development rows.
4. **Zero Cohort Leakage between CV Folds**:
   - In every fold, `set(train_cohorts) ∩ set(val_cohorts) == ∅`. The partition key was strictly `experiment_id`.
5. **Feature Blacklist Enforced**:
   - Identifiers (`sample_id`), partition keys (`experiment_id`), replicate indices (`sample_replicate`), and Excel metadata (`source_row_index`, `is_restored_value`) were strictly isolated from all model inputs.

---

## 4. Final Statement of Verification

All integrity criteria specified in the project directives have been satisfied with zero violations. The TabPFN modeling stage was completed under full scientific reproducibility and total preservation of source data.
