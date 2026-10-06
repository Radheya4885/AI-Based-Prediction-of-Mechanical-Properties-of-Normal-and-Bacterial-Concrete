# Phase 3.1: Robust Grouped Cross-Validation + Locked Final Test Strategy

> **Research & Governance Invariants**:
> - **Zero Machine Learning Models Trained**: No XGBoost, CatBoost, Random Forest, Linear Regression, or TabPFN models fit.
> - **Master Dataset Immutability**: `data/master_dataset.csv` and `data/master_dataset.parquet` SHA256 hashes verified and unchanged.
> - **Primary Grouping Variable**: `experiment_id` (36 cohorts; 100 rows each).
> - **Tripartite Separation**: Strict separation between Model Development (28 cohorts / 2,800 rows), Validation (5-Fold GroupKFold), and Final Locked Evaluation (8 cohorts / 800 rows).
> - **Automated Leakage Tests**: All 8 verification checks PASSED.

---

## 1. Dataset Integrity Verification

Before executing any split generation, the research source artifacts were audited to verify bit-for-bit cryptographic authenticity:

- **Master Dataset CSV**: [`data/master_dataset.csv`](file:///G:/Projects/Concrete testing/data/master_dataset.csv)
- **Master Dataset Parquet**: [`data/master_dataset.parquet`](file:///G:/Projects/Concrete testing/data/master_dataset.parquet)
- **Total Specimen Rows**: `3,600`
- **Total Schema Attributes**: `16`
- **Missing Values**: `0` (100% complete across all 3,600 observations)

---

## 2. Dataset Checksum Verification

| Artifact | Computed SHA256 Checksum | Expected Checksum | Status |
| :--- | :--- | :--- | :---: |
| `master_dataset.csv` | `0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a` | `0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a` | **MATCH (VERIFIED)** |
| `master_dataset.parquet` | `0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a` | `0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a` | **MATCH (VERIFIED)** |

> [!NOTE]
> Cryptographic integrity guarantees that no downstream code has altered, rounded, normalized, or smoothed any raw physical measurement.

---

## 3. Experimental Cohort Structure

The primary experimental grouping variable is **`experiment_id`**.

### Invariant Verification Results:
- **Unique Cohort Count**: `36` distinct experimental conditions.
- **Specimens per Cohort**: Exactly `100` rows per cohort (100% uniform; verified: `True`).
- **Single Curing Age Invariant**: Verified (`True`). Zero cohorts span multiple curing ages.
- **Single Concrete Type Invariant**: Verified (`True`). Zero cohorts mix control and bacterial formulations.
- **Single Mechanical Property Invariant**: Verified (`True`). Zero cohorts mix test geometries or failure modes.
- **Prohibition of `sample_replicate`**: `sample_replicate` is an arbitrary laboratory mold indexing number (1–100) and is **strictly prohibited** from serving as the primary grouping variable.

---

## 4. GroupKFold Methodology (Development Dataset)

To eliminate optimistic performance estimates caused by specimen-level memorization, cross-validation is performed exclusively at the cohort level on the **Development Dataset** (28 cohorts, 2,800 rows):

- **Algorithm**: `sklearn.model_selection.GroupKFold(n_splits=5)`.
- **Grouping Variable**: `groups = dev_df['experiment_id']`.
- **Cohort Integrity Rule**: Every single `experiment_id` belongs entirely to either the training fold or the validation fold ($Train \cap Val = \emptyset$).
- **Purpose**: The 5-fold cross-validation scheme is reserved strictly for **model development, model comparison, feature selection, and hyperparameter tuning**.
- **Lockout Rule**: The locked final test set is never accessed or evaluated during cross-validation.

---

## 5. Exact Fold Distributions

Because the Development Dataset contains exactly 28 cohorts, `GroupKFold(n_splits=5)` partitions the cohorts into three 6-cohort folds (600 validation rows) and two 5-cohort folds (500 validation rows).

> [!IMPORTANT]
> **Balance Transparency**: Due to the discrete 28-cohort structure, mathematical symmetry across properties, concrete types, and curing ages cannot be perfectly equal across every fold. In accordance with strict scientific reporting standards, the exact distributions are documented below without artificial manipulation.

| Fold # | Train Rows | Val Rows | Train Cohorts | Val Cohorts | Val Mechanical Properties | Val Concrete Types | Val Curing Ages | Restored in Val |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :---: |
| **Fold 1** | 2,200 | 600 | 22 | 6 | Comp: 200, Flex: 200, Spli: 200 | Norm: 300, Bact: 300 | 56d: 400, 90d: 200 | 100 |
| **Fold 2** | 2,200 | 600 | 22 | 6 | Comp: 200, Flex: 200, Spli: 200 | Bact: 300, Norm: 300 | 21d: 100, 28d: 400, 56d: 100 | 0 |
| **Fold 3** | 2,200 | 600 | 22 | 6 | Comp: 200, Flex: 200, Spli: 200 | Bact: 300, Norm: 300 | 14d: 100, 21d: 400, 28d: 100 | 0 |
| **Fold 4** | 2,300 | 500 | 23 | 5 | Flex: 200, Spli: 200, Comp: 100 | Norm: 300, Bact: 200 | 14d: 500 | 0 |
| **Fold 5** | 2,300 | 500 | 23 | 5 | Comp: 200, Spli: 200, Flex: 100 | Norm: 300, Bact: 200 | 7d: 300, 90d: 200 | 0 |

#### Validation Cohorts Assigned to Each Fold:
- **Fold 1 Validation Cohorts** (`6` cohorts): `EXP_CS_BC_56d`, `EXP_CS_NC_56d`, `EXP_FS_BC_56d`, `EXP_FS_NC_90d`, `EXP_TS_BC_56d`, `EXP_TS_NC_90d`
- **Fold 2 Validation Cohorts** (`6` cohorts): `EXP_CS_BC_21d`, `EXP_CS_NC_28d`, `EXP_FS_BC_28d`, `EXP_FS_NC_28d`, `EXP_TS_BC_28d`, `EXP_TS_NC_56d`
- **Fold 3 Validation Cohorts** (`6` cohorts): `EXP_CS_BC_14d`, `EXP_CS_NC_21d`, `EXP_FS_BC_21d`, `EXP_FS_NC_21d`, `EXP_TS_BC_21d`, `EXP_TS_NC_28d`
- **Fold 4 Validation Cohorts** (`5` cohorts): `EXP_CS_NC_14d`, `EXP_FS_BC_14d`, `EXP_FS_NC_14d`, `EXP_TS_BC_14d`, `EXP_TS_NC_14d`
- **Fold 5 Validation Cohorts** (`5` cohorts): `EXP_CS_BC_90d`, `EXP_CS_NC_90d`, `EXP_FS_NC_07d`, `EXP_TS_BC_07d`, `EXP_TS_NC_07d`

---

## 6. Final Locked Test Set Methodology

The Final Test Set represents an **unbiased holdout benchmark** for the final model:

- **Cohort Allocation**: Exactly `8` of the 36 cohorts (`800` rows; `22.22%` of the dataset).
- **Grouping Enforcement**: Grouped strictly at the `experiment_id` level. Zero cohort overlap with the development set.
- **Reproducibility**: Pinned to deterministic random seed `42`.
- **Lockout Contract**: Once generated, this test set is permanently **LOCKED**. No model selection, hyperparameter tuning, feature pruning, or preprocessing decisions may peek at or evaluate against this set.

---

## 7. Exact Final Test Cohorts

The 8 locked final test cohorts represent all three mechanical properties and both concrete types:

| # | Cohort ID (`experiment_id`) | Mechanical Property | Concrete Type | Curing Age | Geometry | Bacterial Dosage | Restored Rows |
| :---: | :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| 1 | `EXP_CS_BC_07d` | Compressive Strength | Bacterial Concrete | 7d | 150mm Cube | 1,000,000 | 0 |
| 2 | `EXP_CS_BC_28d` | Compressive Strength | Bacterial Concrete | 28d | 150mm Cube | 1,000,000 | 0 |
| 3 | `EXP_CS_NC_07d` | Compressive Strength | Normal Concrete | 7d | 150mm Cube | 0 | 0 |
| 4 | `EXP_FS_BC_07d` | Flexural Strength | Bacterial Concrete | 7d | 100x100x500mm Prism | 1,000,000 | 0 |
| 5 | `EXP_FS_BC_90d` | Flexural Strength | Bacterial Concrete | 90d | 100x100x500mm Prism | 1,000,000 | 100 |
| 6 | `EXP_FS_NC_56d` | Flexural Strength | Normal Concrete | 56d | 100x100x500mm Prism | 0 | 0 |
| 7 | `EXP_TS_BC_90d` | Split Tensile Strength | Bacterial Concrete | 90d | 150x300mm Cylinder | 1,000,000 | 0 |
| 8 | `EXP_TS_NC_21d` | Split Tensile Strength | Normal Concrete | 21d | 150x300mm Cylinder | 0 | 0 |

- **Files**: [`data/splits/final_test_cohorts.csv`](file:///{os.path.abspath('data/splits/final_test_cohorts.csv').replace(chr(92), '/')}) and [`data/splits/final_test_rows.csv`](file:///{os.path.abspath('data/splits/final_test_rows.csv').replace(chr(92), '/')}).

---

## 8. Exact Development Cohorts

The 28 development cohorts (`2,800` rows; `77.78%` of the dataset) form the closed universe for all subsequent model development and 5-fold cross-validation:

| # | Cohort ID (`experiment_id`) | Mechanical Property | Concrete Type | Curing Age | Restored Rows |
| :---: | :--- | :--- | :--- | :---: | :---: |
| 1 | `EXP_CS_BC_14d` | Compressive Strength | Bacterial Concrete | 14d | 0 |
| 2 | `EXP_CS_BC_21d` | Compressive Strength | Bacterial Concrete | 21d | 0 |
| 3 | `EXP_CS_BC_56d` | Compressive Strength | Bacterial Concrete | 56d | 0 |
| 4 | `EXP_CS_BC_90d` | Compressive Strength | Bacterial Concrete | 90d | 0 |
| 5 | `EXP_CS_NC_14d` | Compressive Strength | Normal Concrete | 14d | 0 |
| 6 | `EXP_CS_NC_21d` | Compressive Strength | Normal Concrete | 21d | 0 |
| 7 | `EXP_CS_NC_28d` | Compressive Strength | Normal Concrete | 28d | 0 |
| 8 | `EXP_CS_NC_56d` | Compressive Strength | Normal Concrete | 56d | 0 |
| 9 | `EXP_CS_NC_90d` | Compressive Strength | Normal Concrete | 90d | 0 |
| 10 | `EXP_FS_BC_14d` | Flexural Strength | Bacterial Concrete | 14d | 0 |
| 11 | `EXP_FS_BC_21d` | Flexural Strength | Bacterial Concrete | 21d | 0 |
| 12 | `EXP_FS_BC_28d` | Flexural Strength | Bacterial Concrete | 28d | 0 |
| 13 | `EXP_FS_BC_56d` | Flexural Strength | Bacterial Concrete | 56d | 100 |
| 14 | `EXP_FS_NC_07d` | Flexural Strength | Normal Concrete | 7d | 0 |
| 15 | `EXP_FS_NC_14d` | Flexural Strength | Normal Concrete | 14d | 0 |
| 16 | `EXP_FS_NC_21d` | Flexural Strength | Normal Concrete | 21d | 0 |
| 17 | `EXP_FS_NC_28d` | Flexural Strength | Normal Concrete | 28d | 0 |
| 18 | `EXP_FS_NC_90d` | Flexural Strength | Normal Concrete | 90d | 0 |
| 19 | `EXP_TS_BC_07d` | Split Tensile Strength | Bacterial Concrete | 7d | 0 |
| 20 | `EXP_TS_BC_14d` | Split Tensile Strength | Bacterial Concrete | 14d | 0 |
| 21 | `EXP_TS_BC_21d` | Split Tensile Strength | Bacterial Concrete | 21d | 0 |
| 22 | `EXP_TS_BC_28d` | Split Tensile Strength | Bacterial Concrete | 28d | 0 |
| 23 | `EXP_TS_BC_56d` | Split Tensile Strength | Bacterial Concrete | 56d | 0 |
| 24 | `EXP_TS_NC_07d` | Split Tensile Strength | Normal Concrete | 7d | 0 |
| 25 | `EXP_TS_NC_14d` | Split Tensile Strength | Normal Concrete | 14d | 0 |
| 26 | `EXP_TS_NC_28d` | Split Tensile Strength | Normal Concrete | 28d | 0 |
| 27 | `EXP_TS_NC_56d` | Split Tensile Strength | Normal Concrete | 56d | 0 |
| 28 | `EXP_TS_NC_90d` | Split Tensile Strength | Normal Concrete | 90d | 0 |

- **Files**: [`data/splits/development_cohorts.csv`](file:///{os.path.abspath('data/splits/development_cohorts.csv').replace(chr(92), '/')}) and [`data/splits/development_rows.csv`](file:///{os.path.abspath('data/splits/development_rows.csv').replace(chr(92), '/')}).

---

## 9. Leave-One-Curing-Age-Out Generalization Methodology

To evaluate how models handle physical hydration maturation curves, the **Leave-One-Curing-Age-Out Generalization** framework tests all six curing ages (7, 14, 21, 28, 56, and 90 days):

- **Protocol**: For each curing age $A$, all `600` specimen observations across all 6 cohorts (3 properties × 2 concrete types) are held out entirely as the test set. The model is trained on the remaining `3,000` observations across the other 5 curing ages.
- **Scientific Scope**: Evaluates whether algorithms learn physical hydration dynamics rather than simply memorizing discrete age timestamps.
- **File**: [`data/splits/leave_age_out_assignments.csv`](file:///G:/Projects/Concrete testing/data/splits/leave_age_out_assignments.csv).

---

## 10. Dynamic Interpolation vs. Extrapolation Classification

For each held-out curing age, the boundary limits of the training set are dynamically determined:

```python
if held_out_age < minimum_training_age:
    classification = 'extrapolation'
elif held_out_age > maximum_training_age:
    classification = 'extrapolation'
else:
    classification = 'interpolation'
```

### Automated Classification Table:

| Fold # | Held-Out Age | Min Train Age | Max Train Age | Generalization Category | Restored in Train | Restored in Test | Scientific Interpretation |
| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **Fold 1** | **7 Days** | 14 Days | 90 Days | **Extrapolation (Early hydration kinetics)** | 200 | 0 | Early-hydration boundary extrapolation |
| **Fold 2** | **14 Days** | 7 Days | 90 Days | **Interpolation (Internal hydration curve between 7d and 90d)** | 200 | 0 | Internal hydration curve interpolation |
| **Fold 3** | **21 Days** | 7 Days | 90 Days | **Interpolation (Internal hydration curve between 7d and 90d)** | 200 | 0 | Internal hydration curve interpolation |
| **Fold 4** | **28 Days** | 7 Days | 90 Days | **Interpolation (Internal hydration curve between 7d and 90d)** | 200 | 0 | Internal hydration curve interpolation |
| **Fold 5** | **56 Days** | 7 Days | 90 Days | **Interpolation (Internal hydration curve between 7d and 90d)** | 100 | 100 | Internal hydration curve interpolation |
| **Fold 6** | **90 Days** | 7 Days | 56 Days | **Extrapolation (Long-term asymptotic maturity)** | 100 | 100 | Asymptotic maturity extrapolation |

---

## 11. Feature Governance Policy

Input features are categorized under strict anti-leakage rules:

| Category | Attribute Names | Status | Rule / Justification |
| :--- | :--- | :---: | :--- |
| **Candidate Model Features** | `curing_age_days`, `concrete_type`, `bacterial_status`, `bacterial_concentration_cells_ml`, `mechanical_property`, `specimen_geometry` | **APPROVED** | Physical parameters known before specimen destruction. |
| **Strictly Forbidden Predictors** | `sample_id`, `experiment_id`, `sample_replicate`, `source_file`, `source_sheet`, `source_row_index`, `is_restored_value` | **FORBIDDEN** | Primary keys, cohort groupings, lab mold indices, workbook lineage, and audit flags. Excluded from feature matrices. |
| **Zero-Variance Constant** | `cement_type` | **EXCLUDED** | 100% constant `'OPC 53 (UltraTech)'`. Zero predictive entropy. |

> [!IMPORTANT]
> Forbidden attributes remain safely inside `data/master_dataset.csv` for auditability and grouping, but are programmatically barred from entering any model training pipeline.

---

## 12. Numerical Value Preservation Policy

The master dataset is an immutable scientific ground truth:
- **Zero In-Place Preprocessing**: The files `data/master_dataset.csv` and `data/master_dataset.parquet` must **NEVER** be normalized, standardized, scaled, rounded, smoothed, clipped, or modified.
- **Target Values**: `strength_mpa` preserves exact 2-to-3 decimal experimental failure measurements.
- **Feature Values**: `curing_age_days`, `bacterial_concentration_cells_ml`, `sample_replicate`, and `source_row_index` remain uncompressed and unrounded.
- **Pipelines**: Any standard scaling, min-max normalization, or one-hot encoding required by algorithms must be encapsulated inside a `Pipeline` or `ColumnTransformer` fitted strictly on training subsets.

---

## 13. Restored Values Evaluation Protocol

The 200 restored observations (`EXP_FS_BC_56d` and `EXP_FS_BC_90d`) are valid experimental specimens recovered from the original laboratory records. To guarantee transparent evaluation, future reports must partition test performance across:
1. **All Test Observations (`all`)**: Standard metric across the complete holdout set.
2. **Excluding Restored Observations (`excluding_restored`)**: Performance strictly on the unmodified baseline observations (`is_restored_value == False`).
3. **Restored Observations Only (`restored_only`)**: Performance isolated to restored observations (`is_restored_value == True`).

In the Locked Final Test Set, `EXP_FS_BC_90d` (`100` rows) is present in test, while `EXP_FS_BC_56d` (`100` rows) resides in the development set.

---

## 14. Programmatic Data Leakage Verification

All 8 automated integrity and anti-leakage checks executed with **100% SUCCESS**:

| Check ID | Verification Rule | Assertion Tested | Status | Diagnostic Summary |
| :---: | :--- | :--- | :---: | :--- |
| **CHECK 1** | No sample_id overlap between train and validation | `Verified across all 5 GroupKFold cross-validation folds.` | **PASS** | Verified. Zero leakage detected. |
| **CHECK 2** | No experiment_id overlap between train and validation | `Verified across all 5 GroupKFold cross-validation folds.` | **PASS** | Verified. Zero leakage detected. |
| **CHECK 3** | No experiment_id overlap between development and final test | `Verified zero overlap between 28 dev cohorts and 8 test cohorts.` | **PASS** | Verified. Zero leakage detected. |
| **CHECK 4** | Leave-one-age-out held-out age does not appear in training | `Verified across all 6 curing ages (7, 14, 21, 28, 56, 90 days).` | **PASS** | Verified. Zero leakage detected. |
| **CHECK 5** | Forbidden identifier/provenance columns are not used as model features | `Verified that sample_id, experiment_id, sample_replicate, source_file, source_sheet, source_row_index, is_restored_value are strictly blacklisted.` | **PASS** | Verified. Zero leakage detected. |
| **CHECK 6** | Master dataset checksum remains unchanged | `Verified data/master_dataset.csv exact SHA256 integrity.` | **PASS** | Verified. Zero leakage detected. |
| **CHECK 7** | Number of rows remains exactly 3,600 | `Verified master dataset row count is 3600.` | **PASS** | Verified. Zero leakage detected. |
| **CHECK 8** | Number of columns remains exactly 16 | `Verified master dataset column count is 16.` | **PASS** | Verified. Zero leakage detected. |

---

## 15. Reproducibility & Generated Split Files

All partition files have been deterministically generated using `random_state = 42` and saved under [`data/splits/`](file:///{os.path.abspath('data/splits').replace(chr(92), '/')}) :

1. **`final_test_cohorts.csv`**: [`data/splits/final_test_cohorts.csv`](file:///G:/Projects/Concrete testing/data/splits/final_test_cohorts.csv) — 8 locked test cohorts metadata.
2. **`development_cohorts.csv`**: [`data/splits/development_cohorts.csv`](file:///G:/Projects/Concrete testing/data/splits/development_cohorts.csv) — 28 development cohorts metadata.
3. **`final_test_rows.csv`**: [`data/splits/final_test_rows.csv`](file:///G:/Projects/Concrete testing/data/splits/final_test_rows.csv) — 800 locked test specimen rows.
4. **`development_rows.csv`**: [`data/splits/development_rows.csv`](file:///G:/Projects/Concrete testing/data/splits/development_rows.csv) — 2,800 development specimen rows.
5. **`grouped_cv_assignments.csv`**: [`data/splits/grouped_cv_assignments.csv`](file:///G:/Projects/Concrete testing/data/splits/grouped_cv_assignments.csv) — 5-fold GroupKFold assignments on development data.
6. **`leave_age_out_assignments.csv`**: [`data/splits/leave_age_out_assignments.csv`](file:///G:/Projects/Concrete testing/data/splits/leave_age_out_assignments.csv) — 6-fold leave-one-curing-age-out assignments.

---

## 16. Methodological Limitations

1. **Discrete Cohort Granularity**: With only 28 development cohorts, 5-fold cross-validation results in uneven fold sizes (three 6-cohort folds, two 5-cohort folds).
2. **Batch Synchronicity**: Specimen replicates within each cohort were cast synchronously in laboratory batches. GroupKFold isolates cohorts, but models must be evaluated on independent future batches to verify cross-laboratory generalization.
3. **Fixed Experimental Domain**: The dataset explores a single cement brand (`OPC 53`), single microbial species (`Bacillus subtilis`), and single bacterial dosage ($10^6$ cells/mL). Cross-validation estimates performance within this physical domain.

---
*End of Phase 3.1 Validation Strategy Report*