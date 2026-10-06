# Master Validation Strategy & Leakage-Safe Evaluation Design

> **Policy & Governance Adherence**:
> - **Zero Machine Learning Models Trained**: Strictly no TabPFN, XGBoost, CatBoost, or Random Forest execution.
> - **Source Dataset**: `data/master_dataset.csv` (3,600 rows × 16 attributes; 0 missing values).
> - **Primary Cohort Identifier**: `experiment_id` (36 unique cohorts).
> - **Pre-Registered Evaluation Scenarios**: Scenario A (Random Baseline), Scenario B (Cohort-Based Grouped), Scenario C (Leave-Age-Out).
> - **Automated Leakage Verification**: All 5 programmatic leakage checks PASSED.

---

## 1. Executive Summary

Establishing a leak-free validation framework is the single most critical prerequisite before training machine learning models on concrete experimental data. Standard k-fold cross-validation or random train/test splits inadvertently place specimens from the same casting batches or identical experimental cohorts into both training and evaluation folds. Because concrete specimens cast in the same batch share unrecorded environmental conditions (e.g. ambient curing humidity, pan-mixer hydration kinetics, fine-aggregate moisture), models evaluated on randomly split rows achieve artificially inflated accuracy via **specimen-level memorization** rather than true physical generalization.

This document pre-registers the evaluation contracts, feature policies, metrics, and temporal holdouts that govern all subsequent modeling experiments.

---

## 2. Experimental Cohort Architecture

The primary experimental grouping variable is **`experiment_id`**.

### 2.1 Cohort Invariant Verification
- **Total Unique Cohorts**: `36`
- **Specimens per Cohort**: Exactly `100` rows per cohort (100% uniform across all 36 cohorts).
- **Cohort Composition**: Exactly 1 unique mechanical property, 1 concrete type, and 1 curing age per `experiment_id`.
- **Distribution across Mechanical Properties**: 12 Compressive Strength cohorts (1,200 rows), 12 Flexural Strength cohorts (1,200 rows), 12 Split Tensile Strength cohorts (1,200 rows).
- **Distribution across Concrete Types**: 18 Normal Concrete cohorts (1,800 rows), 18 Bacterial Concrete cohorts (1,800 rows).
- **Distribution across Curing Ages**: Exactly 6 cohorts per curing age (7, 14, 21, 28, 56, 90 days; 600 rows per age).

### 2.2 Complete Experimental Cohorts Catalog

| Cohort ID (`experiment_id`) | Mechanical Property | Concrete Type | Curing Age | Geometry | Bacterial Dosage | Restored Rows | Mean Strength (SD) [MPa] |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `EXP_CS_BC_07d` | Compressive Strength | Bacterial Concrete | 7d | 150mm Cube | 1,000,000 | 0 | 22.04 (±1.60) |
| `EXP_CS_BC_14d` | Compressive Strength | Bacterial Concrete | 14d | 150mm Cube | 1,000,000 | 0 | 26.49 (±1.46) |
| `EXP_CS_BC_21d` | Compressive Strength | Bacterial Concrete | 21d | 150mm Cube | 1,000,000 | 0 | 29.84 (±1.41) |
| `EXP_CS_BC_28d` | Compressive Strength | Bacterial Concrete | 28d | 150mm Cube | 1,000,000 | 0 | 33.04 (±1.45) |
| `EXP_CS_BC_56d` | Compressive Strength | Bacterial Concrete | 56d | 150mm Cube | 1,000,000 | 0 | 37.24 (±1.42) |
| `EXP_CS_BC_90d` | Compressive Strength | Bacterial Concrete | 90d | 150mm Cube | 1,000,000 | 0 | 40.66 (±1.66) |
| `EXP_CS_NC_07d` | Compressive Strength | Normal Concrete | 7d | 150mm Cube | 0 | 0 | 19.34 (±1.36) |
| `EXP_CS_NC_14d` | Compressive Strength | Normal Concrete | 14d | 150mm Cube | 0 | 0 | 23.53 (±1.43) |
| `EXP_CS_NC_21d` | Compressive Strength | Normal Concrete | 21d | 150mm Cube | 0 | 0 | 25.60 (±1.63) |
| `EXP_CS_NC_28d` | Compressive Strength | Normal Concrete | 28d | 150mm Cube | 0 | 0 | 28.16 (±1.33) |
| `EXP_CS_NC_56d` | Compressive Strength | Normal Concrete | 56d | 150mm Cube | 0 | 0 | 30.92 (±1.60) |
| `EXP_CS_NC_90d` | Compressive Strength | Normal Concrete | 90d | 150mm Cube | 0 | 0 | 32.83 (±1.39) |
| `EXP_FS_BC_07d` | Flexural Strength | Bacterial Concrete | 7d | 100x100x500mm Prism | 1,000,000 | 0 | 3.53 (±0.24) |
| `EXP_FS_BC_14d` | Flexural Strength | Bacterial Concrete | 14d | 100x100x500mm Prism | 1,000,000 | 0 | 4.20 (±0.25) |
| `EXP_FS_BC_21d` | Flexural Strength | Bacterial Concrete | 21d | 100x100x500mm Prism | 1,000,000 | 0 | 4.63 (±0.26) |
| `EXP_FS_BC_28d` | Flexural Strength | Bacterial Concrete | 28d | 100x100x500mm Prism | 1,000,000 | 0 | 5.01 (±0.28) |
| `EXP_FS_BC_56d` | Flexural Strength | Bacterial Concrete | 56d | 100x100x500mm Prism | 1,000,000 | 100 | 5.53 (±0.25) |
| `EXP_FS_BC_90d` | Flexural Strength | Bacterial Concrete | 90d | 100x100x500mm Prism | 1,000,000 | 100 | 5.91 (±0.31) |
| `EXP_FS_NC_07d` | Flexural Strength | Normal Concrete | 7d | 100x100x500mm Prism | 0 | 0 | 2.99 (±0.23) |
| `EXP_FS_NC_14d` | Flexural Strength | Normal Concrete | 14d | 100x100x500mm Prism | 0 | 0 | 3.62 (±0.24) |
| `EXP_FS_NC_21d` | Flexural Strength | Normal Concrete | 21d | 100x100x500mm Prism | 0 | 0 | 3.90 (±0.24) |
| `EXP_FS_NC_28d` | Flexural Strength | Normal Concrete | 28d | 100x100x500mm Prism | 0 | 0 | 4.19 (±0.26) |
| `EXP_FS_NC_56d` | Flexural Strength | Normal Concrete | 56d | 100x100x500mm Prism | 0 | 0 | 4.60 (±0.24) |
| `EXP_FS_NC_90d` | Flexural Strength | Normal Concrete | 90d | 100x100x500mm Prism | 0 | 0 | 4.91 (±0.27) |
| `EXP_TS_BC_07d` | Split Tensile Strength | Bacterial Concrete | 7d | 150x300mm Cylinder | 1,000,000 | 0 | 2.11 (±0.14) |
| `EXP_TS_BC_14d` | Split Tensile Strength | Bacterial Concrete | 14d | 150x300mm Cylinder | 1,000,000 | 0 | 2.58 (±0.17) |
| `EXP_TS_BC_21d` | Split Tensile Strength | Bacterial Concrete | 21d | 150x300mm Cylinder | 1,000,000 | 0 | 2.89 (±0.15) |
| `EXP_TS_BC_28d` | Split Tensile Strength | Bacterial Concrete | 28d | 150x300mm Cylinder | 1,000,000 | 0 | 3.18 (±0.14) |
| `EXP_TS_BC_56d` | Split Tensile Strength | Bacterial Concrete | 56d | 150x300mm Cylinder | 1,000,000 | 0 | 3.50 (±0.14) |
| `EXP_TS_BC_90d` | Split Tensile Strength | Bacterial Concrete | 90d | 150x300mm Cylinder | 1,000,000 | 0 | 3.80 (±0.15) |
| `EXP_TS_NC_07d` | Split Tensile Strength | Normal Concrete | 7d | 150x300mm Cylinder | 0 | 0 | 1.80 (±0.17) |
| `EXP_TS_NC_14d` | Split Tensile Strength | Normal Concrete | 14d | 150x300mm Cylinder | 0 | 0 | 2.20 (±0.15) |
| `EXP_TS_NC_21d` | Split Tensile Strength | Normal Concrete | 21d | 150x300mm Cylinder | 0 | 0 | 2.39 (±0.15) |
| `EXP_TS_NC_28d` | Split Tensile Strength | Normal Concrete | 28d | 150x300mm Cylinder | 0 | 0 | 2.68 (±0.13) |
| `EXP_TS_NC_56d` | Split Tensile Strength | Normal Concrete | 56d | 150x300mm Cylinder | 0 | 0 | 2.90 (±0.16) |
| `EXP_TS_NC_90d` | Split Tensile Strength | Normal Concrete | 90d | 150x300mm Cylinder | 0 | 0 | 3.12 (±0.15) |

---

## 3. The Three Pre-Registered Evaluation Scenarios

### 3.1 Scenario A — Random Baseline (Row-Level Split)

- **Split Structure**: Row-level 80% train (`2,880` rows) / 20% test (`720` rows).
- **Random Seed**: Fixed at `42`.
- **Purpose & Role**: **Diagnostic comparison only**.
- **Methodological Warning**: Specimens sharing identical mix designs, materials, and laboratory casting batches appear simultaneously in both train and test partitions. Models evaluated under Scenario A will exhibit near-zero training/test error due to cohort memorization. It is included strictly to measure the magnitude of the **memorization gap** when compared against Scenario B.
- **Files**: [`data/splits/scenario_a_random_baseline.json`](file:///G:/Projects/Concrete testing/data/splits/scenario_a_random_baseline.json) and [`.csv`](file:///G:/Projects/Concrete testing/data/splits/scenario_a_random_baseline.csv).

### 3.2 Scenario B — Cohort-Based Grouped Split (Primary Research Benchmark)

- **Grouping Variable**: `experiment_id`.
- **Strict Invariant**: Zero `experiment_id` overlap between train and test ($Train \cap Test = \emptyset$).
- **Cohort Partition**: **28 Train Cohorts** (`2,800` rows; 77.78%) / **8 Test Cohorts** (`800` rows; 22.22%).
- **Stratification Strategy**: Stratified across mechanical properties to guarantee that all 3 failure modes possess unseen holdout test cohorts:
  - **Compressive Strength**: 9 Train cohorts (`900` rows) / 3 Test cohorts (`300` rows)
  - **Flexural Strength**: 9 Train cohorts (`900` rows) / 3 Test cohorts (`300` rows)
  - **Split Tensile Strength**: 10 Train cohorts (`1,000` rows) / 2 Test cohorts (`200` rows)
- **Restored Value Distribution in Scenario B**:
  - Train Set: Contains `100` restored observations (`EXP_FS_BC_56d`).
  - Test Set: Contains `100` restored observations (`EXP_FS_BC_90d`).
- **Files**: [`data/splits/scenario_b_cohort_grouped.json`](file:///G:/Projects/Concrete testing/data/splits/scenario_b_cohort_grouped.json) and [`.csv`](file:///G:/Projects/Concrete testing/data/splits/scenario_b_cohort_grouped.csv).

#### Exact Cohort Assignments for Scenario B:

| Split | Cohort ID (`experiment_id`) | Mechanical Property | Concrete Type | Curing Age | Specimen Rows | Restored Flag |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **TEST** | `EXP_CS_BC_07d` | Compressive Strength | Bacterial Concrete | 7d | 100 | No |
| **TEST** | `EXP_CS_BC_28d` | Compressive Strength | Bacterial Concrete | 28d | 100 | No |
| **TEST** | `EXP_CS_NC_07d` | Compressive Strength | Normal Concrete | 7d | 100 | No |
| **TEST** | `EXP_FS_BC_07d` | Flexural Strength | Bacterial Concrete | 7d | 100 | No |
| **TEST** | `EXP_FS_BC_90d` | Flexural Strength | Bacterial Concrete | 90d | 100 | Yes (100 rows) |
| **TEST** | `EXP_FS_NC_56d` | Flexural Strength | Normal Concrete | 56d | 100 | No |
| **TEST** | `EXP_TS_BC_90d` | Split Tensile Strength | Bacterial Concrete | 90d | 100 | No |
| **TEST** | `EXP_TS_NC_21d` | Split Tensile Strength | Normal Concrete | 21d | 100 | No |
| TRAIN | `EXP_CS_BC_14d` | Compressive Strength | Bacterial Concrete | 14d | 100 | No |
| TRAIN | `EXP_CS_BC_21d` | Compressive Strength | Bacterial Concrete | 21d | 100 | No |
| TRAIN | `EXP_CS_BC_56d` | Compressive Strength | Bacterial Concrete | 56d | 100 | No |
| TRAIN | `EXP_CS_BC_90d` | Compressive Strength | Bacterial Concrete | 90d | 100 | No |
| TRAIN | `EXP_CS_NC_14d` | Compressive Strength | Normal Concrete | 14d | 100 | No |
| TRAIN | `EXP_CS_NC_21d` | Compressive Strength | Normal Concrete | 21d | 100 | No |
| TRAIN | `EXP_CS_NC_28d` | Compressive Strength | Normal Concrete | 28d | 100 | No |
| TRAIN | `EXP_CS_NC_56d` | Compressive Strength | Normal Concrete | 56d | 100 | No |
| TRAIN | `EXP_CS_NC_90d` | Compressive Strength | Normal Concrete | 90d | 100 | No |
| TRAIN | `EXP_FS_BC_14d` | Flexural Strength | Bacterial Concrete | 14d | 100 | No |
| TRAIN | `EXP_FS_BC_21d` | Flexural Strength | Bacterial Concrete | 21d | 100 | No |
| TRAIN | `EXP_FS_BC_28d` | Flexural Strength | Bacterial Concrete | 28d | 100 | No |
| TRAIN | `EXP_FS_BC_56d` | Flexural Strength | Bacterial Concrete | 56d | 100 | Yes (100 rows) |
| TRAIN | `EXP_FS_NC_07d` | Flexural Strength | Normal Concrete | 7d | 100 | No |
| TRAIN | `EXP_FS_NC_14d` | Flexural Strength | Normal Concrete | 14d | 100 | No |
| TRAIN | `EXP_FS_NC_21d` | Flexural Strength | Normal Concrete | 21d | 100 | No |
| TRAIN | `EXP_FS_NC_28d` | Flexural Strength | Normal Concrete | 28d | 100 | No |
| TRAIN | `EXP_FS_NC_90d` | Flexural Strength | Normal Concrete | 90d | 100 | No |
| TRAIN | `EXP_TS_BC_07d` | Split Tensile Strength | Bacterial Concrete | 7d | 100 | No |
| TRAIN | `EXP_TS_BC_14d` | Split Tensile Strength | Bacterial Concrete | 14d | 100 | No |
| TRAIN | `EXP_TS_BC_21d` | Split Tensile Strength | Bacterial Concrete | 21d | 100 | No |
| TRAIN | `EXP_TS_BC_28d` | Split Tensile Strength | Bacterial Concrete | 28d | 100 | No |
| TRAIN | `EXP_TS_BC_56d` | Split Tensile Strength | Bacterial Concrete | 56d | 100 | No |
| TRAIN | `EXP_TS_NC_07d` | Split Tensile Strength | Normal Concrete | 7d | 100 | No |
| TRAIN | `EXP_TS_NC_14d` | Split Tensile Strength | Normal Concrete | 14d | 100 | No |
| TRAIN | `EXP_TS_NC_28d` | Split Tensile Strength | Normal Concrete | 28d | 100 | No |
| TRAIN | `EXP_TS_NC_56d` | Split Tensile Strength | Normal Concrete | 56d | 100 | No |
| TRAIN | `EXP_TS_NC_90d` | Split Tensile Strength | Normal Concrete | 90d | 100 | No |

### 3.3 Scenario C — Leave-Age-Out Generalization (Temporal Extrapolation & Interpolation)

- **Splitting Variable**: `curing_age_days`.
- **Folds**: Exactly 6 temporal folds corresponding to the 6 curing ages (7, 14, 21, 28, 56, 90 days).
- **Per-Fold Allocation**: In each fold, exactly one entire curing age is held out as test (`600` rows across 6 cohorts: 3 properties × 2 concrete types), while the model is trained on the remaining 5 curing ages (`3,000` rows across 30 cohorts).
- **Scientific Objective**: Tests whether physical hydration kinetics can be interpolated between observed ages or extrapolated beyond early/late experimental boundaries.
- **Files**: [`data/splits/scenario_c_leave_age_out.json`](file:///G:/Projects/Concrete testing/data/splits/scenario_c_leave_age_out.json) and [`.csv`](file:///G:/Projects/Concrete testing/data/splits/scenario_c_leave_age_out.csv).

#### Scenario C Fold Manifest:

| Fold # | Held-Out Age | Generalization Category | Train Rows | Test Rows | Restored in Train | Restored in Test | Scientific Significance |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| Fold 1 | **7 Days** | Extrapolation (Early hydration kinetics; 7d holdout) | 3,000 | 600 | 200 | 0 | Boundary extrapolation (early kinetics) |
| Fold 2 | **14 Days** | Interpolation (14d holdout between bounding ages) | 3,000 | 600 | 200 | 0 | Hydration curve interpolation |
| Fold 3 | **21 Days** | Interpolation (21d holdout between bounding ages) | 3,000 | 600 | 200 | 0 | Hydration curve interpolation |
| Fold 4 | **28 Days** | Interpolation (Standard 28-day compliance boundary) | 3,000 | 600 | 200 | 0 | Hydration curve interpolation |
| Fold 5 | **56 Days** | Interpolation (56d holdout between bounding ages) | 3,000 | 600 | 100 | 100 | Hydration curve interpolation |
| Fold 6 | **90 Days** | Extrapolation (Long-term asymptotic maturity; 90d holdout) | 3,000 | 600 | 100 | 100 | Boundary extrapolation (long-term maturity) |

---

## 4. Target Modeling Strategies: Unified vs. Separate Models

The validation framework is designed to evaluate both modeling architectures under identical test conditions:

### Strategy A: Dedicated Property-Specific Models (3 Separate Regressors)
- **Architectures**: Three specialized regressors trained independently on Compressive Strength ($n=1,200$), Flexural Strength ($n=1,200$), and Split Tensile Strength ($n=1,200$).
- **Features**: `curing_age_days`, `concrete_type` (or `bacterial_concentration_cells_ml`).
- **Target**: `strength_mpa` (filtered per property).
- **Advantages**: Preserves the natural physical variance and target magnitude of each failure mode. Avoids cross-property scale gradient dominance (Compressive strength averages ~29 MPa while tensile strength averages ~2.8 MPa).

### Strategy B: Unified Multi-Property Model (1 Unified Regressor)
- **Architecture**: A single model trained across all 3,600 specimen observations.
- **Features**: `curing_age_days`, `concrete_type` (or `bacterial_concentration_cells_ml`), PLUS `mechanical_property` (categorical: Compressive, Flexural, Split Tensile).
- **Target**: `strength_mpa` across all 3,600 rows.
- **Advantages**: Enables the model to learn shared underlying hydration kinetics and microbial calcium carbonate precipitation mechanics across all failure modes.
- **Evaluation Rule**: Both strategies will be benchmarked on the **exact same test rows** under Scenarios A, B, and C to objectively determine which paradigm achieves lower test MAE.

---

## 5. Feature Policy & Anti-Leakage Governance

To eliminate data leakage and spurious shortcut learning, all features are partitioned into four strict policy classes:

| Feature Classification | Attribute Names | Policy & Usage |
| :--- | :--- | :--- |
| **Candidate Predictive Features** | `curing_age_days`, `concrete_type`, `bacterial_concentration_cells_ml`, `mechanical_property` | **APPROVED**: Legitimate physical attributes known prior to specimen failure testing. |
| **Strictly Forbidden Predictors** | `sample_id`, `experiment_id`, `sample_replicate`, `source_file`, `source_sheet`, `source_row_index`, `is_restored_value` | **FORBIDDEN**: Primary keys, cohort groupings, lineage metadata, and audit restoration flags. Prohibited from model input matrices. |
| **Zero-Variance Constant** | `cement_type` | **EXCLUDED**: Constant string across all rows (`OPC 53 (UltraTech)`). |
| **Redundant Collinear Attributes** | `bacterial_species`, `bacterial_status`, `specimen_geometry` | **CONTROLLED**: Redundant with `concrete_type` and `mechanical_property`. Tested in ablation sets. |

### Pre-Registered Feature Sets for Modeling:
1. **Feature Set 1 (Canonical Minimal)**:
   - Unified Model: `['curing_age_days', 'concrete_type', 'mechanical_property']`
   - Separate Models: `['curing_age_days', 'concrete_type']`
2. **Feature Set 2 (Continuous Dosage Representation)**:
   - Unified Model: `['curing_age_days', 'bacterial_concentration_cells_ml', 'mechanical_property']`
   - Separate Models: `['curing_age_days', 'bacterial_concentration_cells_ml']`
3. **Feature Set 3 (Extended Ablation Representation)**:
   - Evaluates tree/kernel robustness to collinear geometry and multi-attribute representations.

---

## 6. Restored Value Handling Protocol

The 200 restored observations (representing 56-day and 90-day bacterial flexural strength from original Sheet2) are legitimate experimental specimens, but their provenance requires transparent isolation.

### Evaluation Harness Protocol:
Every future evaluation report MUST compute and display metrics across three reporting strata:
1. **All Test Observations (`all`)**: Standard performance metric across all test specimens.
2. **Excluding Restored Observations (`excluding_restored`)**: Performance strictly on the 3,400 unmodified baseline observations (`is_restored_value == False`).
3. **Restored Observations Only (`restored_only`)**: Performance isolated to the restored specimens (`is_restored_value == True`), whenever restored cohorts fall into the test set (e.g. in Scenario B where `EXP_FS_BC_90d` is in test, or Scenario C Folds 5 & 6).

---

## 7. Metrics Framework

Evaluation metrics are prioritized based on physical interpretability and robustness:

### 7.1 Metric Definitions
- **Primary Metric**: **MAE (Mean Absolute Error, in MPa)**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
  *Direct physical error in MegaPascals; resilient to single-specimen outlier fracture anomalies.*
- **Secondary Metric**: **RMSE (Root Mean Squared Error, in MPa)**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
  *Penalizes large prediction errors.*
- **Secondary Metric**: **$R^2$ (Coefficient of Determination)**:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  *Fraction of physical strength variance explained by the model.*
- **Diagnostic Metric**: **Guarded MAPE (Mean Absolute Percentage Error)**:
  $$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{|y_i| + 10^{-6}}$$
  *Evaluates proportional accuracy across disparate property scales.*

### 7.2 Multi-Dimensional Metric Slicing
Every evaluation run will automatically report metrics sliced across:
1. **Overall (Global)**
2. **By Mechanical Property** (`Compressive Strength`, `Flexural Strength`, `Split Tensile Strength`)
3. **By Concrete Type** (`Normal Concrete`, `Bacterial Concrete`)
4. **By Curing Age** (`7d`, `14d`, `21d`, `28d`, `56d`, `90d`)

---

## 8. Automated Data Leakage Verification Suite

The automated test suite (`src/validation/leakage_checker.py`) executed all five verification suites on the generated splits and feature configurations:

| Check ID | Verification Description | Assertion / Condition Tested | Result | Diagnostic Note |
| :---: | :--- | :--- | :---: | :--- |
| **CHK-1** | Scenario A Sample Isolation | `len(train_ids ∩ test_ids) == 0` | **PASS** | Exactly 0 overlapping sample IDs. |
| **CHK-2** | Scenario B Cohort Isolation | `len(train_cohorts ∩ test_cohorts) == 0` | **PASS** | Exactly 0 overlapping `experiment_id`s between train and test. |
| **CHK-3** | Scenario C Temporal Isolation | `(train_df['age'] == held_age).sum() == 0` | **PASS** | All 6 folds strictly exclude the held-out age from training data. |
| **CHK-4** | Feature Blacklist Enforcement | `len(features ∩ FORBIDDEN) == 0` | **PASS** | No primary keys, cohort IDs, row indices, or audit flags in inputs. |
| **CHK-5** | Preprocessing Pipeline Isolation | `fit(train_only)` vs `fit(full_data)` | **PASS** | Scalers and transformers are fit strictly on training partitions only. |

---

## 9. Reproducibility & Split Files Manifest

All split definitions are deterministically pinned using random seed `42` and persisted to disk:

1. **Manifest File**: [`data/splits/splits_manifest.json`](file:///G:/Projects/Concrete testing/data/splits/splits_manifest.json)
2. **Scenario A Baseline**: [`data/splits/scenario_a_random_baseline.json`](file:///G:/Projects/Concrete testing/data/splits/scenario_a_random_baseline.json) | [`scenario_a_random_baseline.csv`](file:///G:/Projects/Concrete testing/data/splits/scenario_a_random_baseline.csv)
3. **Scenario B Cohort Split**: [`data/splits/scenario_b_cohort_grouped.json`](file:///G:/Projects/Concrete testing/data/splits/scenario_b_cohort_grouped.json) | [`scenario_b_cohort_grouped.csv`](file:///G:/Projects/Concrete testing/data/splits/scenario_b_cohort_grouped.csv)
4. **Scenario C Temporal Folds**: [`data/splits/scenario_c_leave_age_out.json`](file:///G:/Projects/Concrete testing/data/splits/scenario_c_leave_age_out.json) | [`scenario_c_leave_age_out.csv`](file:///G:/Projects/Concrete testing/data/splits/scenario_c_leave_age_out.csv)

---

## 10. Conclusion & Handoff to Model Development

The Phase 3 Validation Framework establishes a leak-free foundation for all subsequent modeling:
1. **Data Leakage Risk Eliminated**: The strict isolation of `experiment_id` cohorts prevents specimen-level memorization.
2. **Clear Comparative Baseline**: The tri-scenario architecture (A, B, C) provides direct visibility into baseline memorization, cohort generalization, and temporal extrapolation.
3. **Multi-Property Parity**: Both separate models and unified multi-property models can be evaluated on identical test samples.
4. **Zero Model Contamination**: No machine learning model parameters were fit, tested, or tuned during this phase.

*End of Master Validation Strategy Report*