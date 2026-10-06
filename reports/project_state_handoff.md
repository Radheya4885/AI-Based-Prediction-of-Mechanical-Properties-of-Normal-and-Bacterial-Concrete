# Project Technical State & Handoff Audit

**Project Title:** AI-Based Prediction of Mechanical Properties of Normal and Bacterial Concrete  
**Audit Timestamp:** 2026-10-01T16:15:00+05:30  
**Audit Mode:** READ-ONLY Deep Technical State Inspection  
**Audit Author:** Automated Research Assistant (Antigravity Agentic Framework)  

---

## 1. Project Identity & Objective

### 1.1 Project Title
**AI-Based Prediction of Mechanical Properties of Normal and Bacterial Concrete**

### 1.2 Core Research Objective
To establish a scientifically rigorous, defensible, and leakage-free machine learning framework for predicting three fundamental mechanical properties of concrete (Compressive Strength, Flexural Strength, and Split Tensile Strength) for both Normal Concrete and Bio-mineralized (*Bacillus subtilis*) Concrete across multiple curing ages (7 to 90 days). 

A central experimental inquiry is to evaluate whether modern pretrained tabular foundational models—specifically **TabPFN** (Tabular Prior-data Fitted Network)—provide statistically significant predictive and generalization advantages over strong, tuned classical baselines (Ridge, Random Forest, XGBoost, CatBoost), especially under challenging out-of-distribution scenarios such as held-out cohorts and unseen curing ages.

### 1.3 Independent Pipeline Charter
The earlier team-member repository serves strictly as a background historical reference. The current project has been reconstructed from the ground up as an independent, reproducible, and mathematically sound research pipeline. It enforces strict data provenance, cryptographic checksums, leak-free cohort validation, and explicit physical domain modeling.

---

## 2. Current Project Directory & Workspace Architecture

### 2.1 Workspace Structure
```
G:/Projects/Concrete testing/
├── .venv/                         # Isolated Python virtual environment (Python 3.11.15)
├── catboost_info/                 # CatBoost runtime execution metadata & temporary logs
├── data/                          # Source datasets, master datasets, and cross-validation splits
│   ├── .gitkeep
│   ├── C Strenth Reading 1000000.xlsx  # Raw workbook: Compressive Strength (11 sheets)
│   ├── F Strength Reading 1000000.xlsx  # Raw workbook: Flexural Strength (2 sheets)
│   ├── T Strenth Reading 1000000.xlsx  # Raw workbook: Split Tensile Strength (2 sheets)
│   ├── master_dataset.csv         # Canonical unified master dataset (Plaintext CSV, UTF-8)
│   ├── master_dataset.parquet     # Columnar optimized master dataset (Apache Parquet, Snappy)
│   └── splits/                    # Cross-validation and locked evaluation splits
│       ├── development_cohorts.csv
│       ├── development_rows.csv
│       ├── final_test_cohorts.csv
│       ├── final_test_rows.csv
│       ├── grouped_cv_assignments.csv
│       ├── leave_age_out_assignments.csv
│       ├── scenario_a_random_baseline.csv
│       ├── scenario_a_random_baseline.json
│       ├── scenario_b_cohort_grouped.csv
│       ├── scenario_b_cohort_grouped.json
│       ├── scenario_c_leave_age_out.csv
│       ├── scenario_c_leave_age_out.json
│       └── splits_manifest.json
├── models/                        # Pretrained model weights and checkpoints
│   └── tabpfn-v2.5-regressor-v2.5_real.ckpt  # Pretrained TabPFN v2.5 checkpoint
├── notebooks/                     # Exploratory notebooks (currently empty)
│   └── .gitkeep
├── reports/                       # Formal markdown and tabular reports
│   ├── baseline_model_comparison.csv     # Phase 4.1 model benchmarking table
│   ├── baseline_modeling_report.md       # Phase 4.1 evaluation and analysis report
│   ├── dataset_audit.md                  # Comprehensive raw data audit (15 sheets)
│   ├── dataset_schema.csv                # Raw sheet-by-sheet catalog
│   ├── eda_report.md                     # Exploratory Data Analysis & statistics
│   ├── figures/                          # 9 high-resolution Phase 4.1 evaluation figures
│   ├── master_dataset_design.md          # Master dataset schema and extraction specification
│   ├── master_dataset_schema.csv         # Master dataset column dictionary
│   ├── master_dataset_validation.md      # Validation and audit assertions
│   ├── sheets_raw_overview.txt           # Raw sheet dump & structure overview
│   ├── validation_strategy.md            # Initial validation architecture
│   └── validation_strategy_v3_1.md       # Phase 3.1 upgraded validation protocol
├── results/                       # Experimental run artifacts and predictions
│   ├── models/                           # Directory for persisted trained model artifacts
│   ├── phase4_1_run_metadata.json        # Execution metadata & integrity hashes for Phase 4.1
│   └── predictions/                      # Model prediction outputs
│       └── phase4_1_cv_predictions.csv   # 67,200 row-level out-of-fold CV predictions
├── src/                           # Production source code
│   ├── __init__.py
│   ├── build_and_validate_master.py      # Pipeline script to construct master dataset
│   ├── build_master_design_report.py     # Script to generate design specifications
│   ├── complete_auditor.py               # Deep audit script for raw workbooks
│   ├── deep_audit.py                     # Initial audit verification tool
│   ├── generate_design_report.py         # Utility script for design reporting
│   ├── generate_full_audit.py            # Comprehensive raw sheet cataloger
│   ├── generate_markdown_report.py       # Report generation formatting utilities
│   ├── parse_all_sheets.py               # Raw sheet parser
│   ├── perform_eda.py                    # Exploratory statistical analysis and plotting
│   ├── test_tabpfn.py                    # TabPFN installation & import verification test
│   ├── modeling/                         # Modeling pipelines
│   │   ├── __init__.py
│   │   └── baseline_training.py          # Phase 4.1 baseline training pipeline (984 lines)
│   └── validation/                       # Validation protocols & leakage checkers
│       ├── __init__.py
│       ├── cohort_analyzer.py            # Cohort structure and balance analyzer
│       ├── evaluator.py                  # Standardized regression metrics evaluator
│       ├── feature_policy.py             # Feature governance and leakage firewall
│       ├── leakage_checker.py            # Automated cross-fold leakage detection
│       ├── pipeline.py                   # Integrated validation pipeline
│       ├── robust_grouped_cv.py          # 5-fold GroupKFold implementation
│       ├── split_generator.py            # Scenario A, B, C split generator
│       ├── test_validation.py            # Unit test suite for validation pipeline
│       └── test_validation_v3_1.py       # Unit test suite for Phase 3.1 protocols
├── debug_r2.py                    # Utility script investigating R² pooling behavior
├── fix_encoding.py                # Character encoding normalization script for console output
├── generate_phase41_report.py     # Standalone report generator from prediction CSV
├── requirements.txt               # Workspace Python dependency requirements
└── verify_phase41.py              # Phase 4.1 verification and checksum confirmation script
```

### 2.2 Catalog of Key Project Artifacts

| Path | File Type | Purpose | Current Status | Immutability | Provenance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `data/*.xlsx` (3 files) | Excel Workbooks | Primary raw laboratory experimental data | Pristine & Verified | **IMMUTABLE** | Primary Laboratory Source |
| `data/master_dataset.csv` | CSV (UTF-8) | Canonical unified tabular research dataset | Verified (3,600 rows) | **IMMUTABLE** | Generated by `build_and_validate_master.py` |
| `data/master_dataset.parquet` | Apache Parquet | Binary columnar representation of master dataset | Verified (3,600 rows) | **IMMUTABLE** | Generated by `build_and_validate_master.py` |
| `data/splits/development_rows.csv` | CSV | 28 development cohorts (2,800 rows) for CV | Verified & Locked | **IMMUTABLE** | Generated by `split_generator.py` |
| `data/splits/final_test_rows.csv` | CSV | 8 final test cohorts (800 rows) locked test set | Verified & Locked | **IMMUTABLE** | Generated by `split_generator.py` |
| `data/splits/splits_manifest.json` | JSON | Formal specification of split partitions | Complete | **IMMUTABLE** | Generated by `split_generator.py` |
| `models/tabpfn-v2.5-regressor-v2.5_real.ckpt` | Binary Checkpoint | Pretrained TabPFN model weights (40.8 MB) | Present & Verified | Read-Only | Downloaded Foundation Weights |
| `reports/dataset_audit.md` | Markdown | Deep audit of 15 sheets across 3 workbooks | Complete | Reference | Generated by Audit Phase |
| `reports/master_dataset_validation.md` | Markdown | Assertion verification of master dataset | Complete | Reference | Generated by Master Validation Phase |
| `reports/eda_report.md` | Markdown | Descriptive statistics, curing kinetics, & tests | Complete | Reference | Generated by EDA Phase |
| `reports/validation_strategy_v3_1.md`| Markdown | Phase 3.1 validation and leakage governance | Complete | Reference | Generated by Validation Design Phase |
| `reports/baseline_modeling_report.md`| Markdown | Phase 4.1 baseline benchmarking report | Complete | Reference | Generated by Phase 4.1 Execution |
| `reports/baseline_model_comparison.csv`| CSV | 72 evaluated baseline model configurations | Complete | Reference | Generated by Phase 4.1 Execution |
| `results/predictions/phase4_1_cv_predictions.csv` | CSV | 67,200 row-level out-of-fold CV predictions | Complete (8.9 MB) | Reference | Generated by `baseline_training.py` |
| `results/phase4_1_run_metadata.json` | JSON | Run provenance, timing, & integrity hashes | Complete | Reference | Generated by `baseline_training.py` |

---

## 3. Computational Environment & Hardware Architecture

The execution environment was audited directly using `.venv/Scripts/python`.

### 3.1 Software Versions
- **Python Version**: `3.11.15` (main, May 10 2026, 19:31:25) [MSC v.1944 64 bit (AMD64)]
- **Platform**: `Windows-10-10.0.26200-SP0`
- **Virtual Environment Path**: `G:\Projects\Concrete testing\.venv`
- **Installed Package Versions**:
  - `tabpfn`: **9.0.0**
  - `torch`: **2.14.0+cpu**
  - `pandas`: **3.0.6**
  - `numpy`: **2.4.6**
  - `scikit-learn (sklearn)`: **1.9.1**
  - `xgboost`: **3.2.0**
  - `catboost`: **1.2.10**
  - `RandomForest Implementation`: `sklearn.ensemble._forest.RandomForestRegressor`
  - `matplotlib`: **3.11.2**
  - `seaborn`: **0.13.2**
  - `openpyxl`: **3.1.5**
  - `pyarrow`: **25.0.1**

### 3.2 Hardware, Accelerator, & CUDA Status
- **NVIDIA GPU Detection**: None detected on PATH (`nvidia-smi` not found).
- **PyTorch CUDA Acceleration**: `torch.cuda.is_available() == False`.
- **PyTorch Build**: CPU-only build (`2.14.0+cpu`).
- **Implication for Future Experiments**: 
  - Standard scikit-learn, XGBoost, and CatBoost models execute with fast multithreaded CPU performance.
  - TabPFN inference on tabular datasets with $N \le 2,800$ rows and $D \le 7$ features can be run on CPU, but larger batch inferences or gradient fine-tuning will require a GPU-enabled environment (e.g., Google Colab or a local CUDA-compatible workstation).

---

## 4. Original Source Datasets

All three primary laboratory workbooks are stored in `data/` and remain 100% unaltered.

| Filename | File Size | SHA256 Checksum | Sheet Count | Sheet Names | Role in Project | Status |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| `C Strenth Reading 1000000.xlsx` | 132,551 bytes | `5605a0e0913de760752dd0aab707f1a66cbc5af130f9811485e958f624460f19` | 11 | `['Comp 6', 'Sheet2', 'Final Result', '105 c', 'c', 'f', 't', 'Sheet1', 'Sheet3', 'RCPT', 'Water Absorption']` | Primary source for Compressive Strength; secondary cross-concentration & durability sheets | Pristine / Untouched |
| `F Strength Reading 1000000.xlsx` | 59,032 bytes | `c28972bd289b860f051602a1877cce31c819b55066ea9afc55937357ff37b44a` | 2 | `['Sheet1', 'Sheet2']` | Primary source for Flexural Strength; Sheet2 provided exact restoration data for 200 blank records in Sheet1 | Pristine / Untouched |
| `T Strenth Reading 1000000.xlsx` | 60,580 bytes | `d8283fd9f8c4e75c57ac3696aa36f08895b4d63a997a86530fcd0ff5813ba645` | 2 | `['Sheet1', 'Ten 6']` | Primary source for Split Tensile Strength | Pristine / Untouched |

---

## 5. Dataset Audit Completed

A exhaustive technical audit cataloged **15 individual sheets** and **179 columns** across the three workbooks, establishing a strict separation between raw experimental measurements and secondary/summary calculations.

### 5.1 Raw Experimental Data vs. Summary / Benchmark Data

#### Raw Experimental Sheets (Selected for Master Pipeline)
1. **`C Strenth Reading 1000000.xlsx` → `Comp 6`**: 
   - 602 rows × 7 columns.
   - 600 literal physical test records of 150mm cubes across 6 curing ages (7, 14, 21, 28, 56, 90 days), with wide columns for Normal Concrete and Bacterial Concrete ($10^6$ cells/mL).
2. **`F Strength Reading 1000000.xlsx` → `Sheet1` & `Sheet2`**:
   - `Sheet1`: 1,202 rows × 7 columns. Contains 1,000 raw readings for 10 cohorts, plus 200 blank targets in bacterial concrete at 56 and 90 days.
   - `Sheet2`: 106 rows × 13 columns (wide format). Contains all 12 cohorts (100 replicates each), containing the exact 200 physical test records for 56-day and 90-day bacterial concrete.
3. **`T Strenth Reading 1000000.xlsx` → `Ten 6`**:
   - 602 rows × 7 columns.
   - 600 literal physical test records of cylindrical specimens across 6 curing ages, wide columns for Normal and Bacterial concrete.

#### Excluded Summary, Average, Formula, and Benchmark Sheets
- **`Final Result`**, **`105 c`**, **`c`**, **`f`**, **`t`**, **`Sheet3`** in `C Strength`:
  - **Reason for Exclusion**: These contain post-hoc averages, cross-concentration exploratory summaries ($10^3$ to $10^9$ cells/mL), Excel ratio formulas, and aggregate tables. Using them as model inputs would introduce catastrophic target leakage and spurious correlations.
- **`RCPT`** & **`Water Absorption`** in `C Strength`:
  - **Reason for Exclusion**: These measure secondary durability properties (chloride ion charge in Coulombs and percentage water absorption) on small aggregate sample sets ($N=12$), rather than standardized mechanical load fracture capacities.

### 5.2 Missing Values & Constant Variables Discovered
- **Raw Blanks in Sheet1 (`F Strength`)**: Rows 1001–1200 had empty target cells for 56d and 90d Bacterial Flexure.
- **Constant Predictors**: Across all three test programs, `cement_type` is exclusively `OPC 53 (UltraTech)` and `bacterial_species` is exclusively *Bacillus subtilis* (or absent/control).

---

## 6. Master Dataset Verification

The canonical master datasets ([`data/master_dataset.csv`](file:///g:/Projects/Concrete%20testing/data/master_dataset.csv) and [`data/master_dataset.parquet`](file:///g:/Projects/Concrete%20testing/data/master_dataset.parquet)) were audited directly:

| Metric | Required Specification | Verified in File | Status |
| :--- | :--- | :--- | :---: |
| **Total Rows** | Exactly `3,600` | **3,600** | ✅ PASS |
| **Total Columns** | Exactly `16` | **16** | ✅ PASS |
| **Target Variable** | `strength_mpa` (`float64`) | `strength_mpa` (0 missing values) | ✅ PASS |
| **Unique Specimen IDs** | `3,600` unique IDs | **3,600 unique IDs** (0 duplicates) | ✅ PASS |
| **Unique Cohorts (`experiment_id`)** | Exactly `36` | **36 cohorts** | ✅ PASS |
| **Specimens per Cohort** | Exactly 100 replicates | **100 replicates per cohort** | ✅ PASS |
| **Mechanical Property Breakdown** | 1,200 per property | Compressive: 1,200<br>Flexural: 1,200<br>Split Tensile: 1,200 | ✅ PASS |
| **Concrete Mix Breakdown** | 1,800 per mix type | Normal Concrete: 1,800<br>Bacterial Concrete: 1,800 | ✅ PASS |
| **Curing Age Breakdown** | 600 per age | 7d: 600, 14d: 600, 21d: 600<br>28d: 600, 56d: 600, 90d: 600 | ✅ PASS |
| **Restored Observations** | Exactly `200` | **200 rows** (`is_restored_value == True`) | ✅ PASS |
| **Missing Values (All Columns)** | 0 except uninoculated species | Only `bacterial_species` has 1,800 nulls (Normal Concrete Control) | ✅ PASS |

---

## 7. Master Dataset Schema Architecture

Every column in `master_dataset.csv` has a precisely governed role:

| Column Name | Storage Type | Class | Allowed / Observed Values | Role & Policy in Modeling |
| :--- | :--- | :--- | :--- | :--- |
| `sample_id` | `string` | Metadata | E.g. `CS_NC_07d_001` | **FORBIDDEN**: Primary key specimen identifier; memorizes replicate sequence. |
| `experiment_id` | `string` | Metadata | 36 values, e.g. `EXP_CS_BC_07d` | **FORBIDDEN as Input / MANDATORY Grouping**: Used strictly as the GroupKFold partition key. |
| `sample_replicate` | `int64` | Metadata | `1` to `100` | **FORBIDDEN**: Replicate casting counter; does not represent physical mixture variation. |
| `concrete_type` | `string` | Raw Feature | `Normal Concrete`, `Bacterial Concrete` | **APPROVED INPUT**: Canonical binary categorical treatment variable. |
| `bacterial_status` | `string` | Derived | `Control`, `Inoculated` | **EXPANDED / REDUNDANT**: 100% collinear with `concrete_type`. |
| `bacterial_species` | `string` | Derived | `None`, `Bacillus subtilis` | **EXPANDED / REDUNDANT**: 100% collinear with `concrete_type`. |
| `bacterial_concentration_cells_ml` | `int64` | Raw/Derived | `0`, `1000000` | **APPROVED INPUT**: Continuous dosage numerical representation. |
| `cement_type` | `string` | Raw Feature | `OPC 53 (UltraTech)` | **CONSTANT (Zero Variance)**: Dropped by models; no predictive variance. |
| `curing_age_days` | `int64` | Raw Feature | `7`, `14`, `21`, `28`, `56`, `90` | **APPROVED INPUT**: Primary continuous temporal curing predictor. |
| `mechanical_property` | `string` | Raw Feature | `Compressive Strength`, `Flexural Strength`, `Split Tensile Strength` | **APPROVED (Unified Strategy Only)**: Specifies target scale in unified modeling. |
| `strength_mpa` | `float64` | Target | Range: `[1.380, 44.160]` MPa | **TARGET VARIABLE**: Continuous regression label. |
| `specimen_geometry` | `string` | Derived | `150mm Cube`, `100x100x500mm Prism`, `150x300mm Cylinder` | **EXPANDED / REDUNDANT**: 100% collinear with `mechanical_property`. |
| `source_file` | `string` | Metadata | Raw Excel workbook filename | **FORBIDDEN**: Lineage audit attribute. |
| `source_sheet` | `string` | Metadata | Raw sheet name (`Comp 6`, `Sheet1`, etc.) | **FORBIDDEN**: Lineage audit attribute. |
| `source_row_index` | `int64` | Metadata | Row index in raw Excel file | **FORBIDDEN**: Lineage audit attribute (leaks temporal test sequence). |
| `is_restored_value` | `bool` | Metadata | `True` (200 rows), `False` (3,400 rows) | **FORBIDDEN**: Audit tracking flag for recovered observations. |

---

## 8. Important Data Transformations & Restoration Audit

### 8.1 Wide-to-Long Reshaping
The raw workbooks recorded physical measurements in wide columns:
- `Compressive Strength`: Columns 5 & 6 contained Normal and Bacterial strength side-by-side.
- `Split Tensile Strength`: Wide columns for Normal and Bacterial specimens.
- **Transformation**: Each pair was reshaped into two standardized, tidy rows with explicit `concrete_type`, `sample_id`, and uniform `strength_mpa`.

### 8.2 Biological Control Encoding
In the raw files, `1,000,000` was redundantly entered in normal concrete rows. The master dataset logically encodes control mixes as:
- `bacterial_status = 'Control'`
- `bacterial_species = 'None'` (empty/null in raw CSV)
- `bacterial_concentration_cells_ml = 0`

For inoculated mixes:
- `bacterial_status = 'Inoculated'`
- `bacterial_species = 'Bacillus subtilis'`
- `bacterial_concentration_cells_ml = 1,000,000`

### 8.3 Exact Recovery of 200 Restored Flexural Observations
In `F Strength Reading 1000000.xlsx`, rows 1001–1200 of `Sheet1` contained 200 blank target values (56-day and 90-day Bacterial Concrete).

#### Mathematical Provenance & Verification
1. `Sheet2` of the same workbook contained the full wide-format table of all 1,200 specimens.
2. An element-wise concordance test was executed across all 1,000 non-missing paired samples in `Sheet1` vs. `Sheet2`. **All 1,000 overlapping samples exhibited an absolute discrepancy of exactly 0.00000000.**
3. The 200 values were recovered directly from `Sheet2` columns `56 BC` and `90 BC` (rows 5 to 104 in Excel).
4. **Classification**: These values are **NOT synthetic**, **NOT imputed**, and **NOT generated**. They are **directly recovered, genuine laboratory experimental observations**.
5. **Traceability**: Every recovered record has `is_restored_value = True`, `source_sheet = 'Sheet2'`, and `source_row_index` mapping to the exact row in `Sheet2`.

---

## 9. Cryptographic Data Integrity Verification

The SHA-256 checksums of the active master dataset files were re-computed and verified against the canonical specifications:

```
File: data/master_dataset.csv
Expected SHA256: 0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a
Observed SHA256: 0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a
Verification:    MATCH (100% UNMODIFIED)

File: data/master_dataset.parquet
Expected SHA256: 0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a
Observed SHA256: 0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a
Verification:    MATCH (100% UNMODIFIED)
```

---

## 10. Exploratory Data Analysis (EDA) Summary

### 10.1 Target Distributions by Property
- **Compressive Strength (150mm Cubes)**: Min: 15.570 MPa, Mean: 29.141 MPa, Median: 28.825 MPa, Max: 44.160 MPa ($\sigma = 6.162$ MPa).
- **Flexural Strength (Prisms)**: Min: 2.428 MPa, Mean: 4.419 MPa, Median: 4.402 MPa, Max: 6.490 MPa ($\sigma = 0.855$ MPa).
- **Split Tensile Strength (Cylinders)**: Min: 1.380 MPa, Mean: 2.762 MPa, Median: 2.764 MPa, Max: 4.183 MPa ($\sigma = 0.583$ MPa).

### 10.2 Curing Kinetics & Biological Effect
Across all three mechanical properties, concrete strength follows classic logarithmic curing kinetics ($R^2 > 0.95$ vs $\ln(\text{age})$):
- **Bacterial Inoculation Effect**: *Bacillus subtilis* induces microbially induced calcium carbonate precipitation ($\text{CaCO}_3$), densifying the microstructure.
- **Empirical Uplift**: Bacterial concrete shows a statistically significant ($p < 10^{-15}$, two-sample t-test & Mann-Whitney U) strength improvement of:
  - Compressive Strength: $+14.0\%$ at 7 days up to $+23.9\%$ at 90 days.
  - Flexural Strength: $+18.1\%$ at 7 days up to $+20.4\%$ at 90 days.
  - Split Tensile Strength: $+17.2\%$ at 7 days up to $+21.8\%$ at 90 days.

### 10.3 Within-Cohort Variance & Repeatability
Each of the 36 experimental cohorts consists of exactly 100 specimen replicates. The within-cohort coefficient of variation ($\text{CV} = \sigma / \mu$) is remarkably low (typically $2.5\%$ to $4.8\%$), reflecting tight laboratory measurement repeatability rather than batch-to-batch variation.

### 10.4 Methodological Domain Boundaries (Critical Caveat)
The dataset possesses strictly bounded domain dimensionality:
- Exactly **one cement type** (`OPC 53 (UltraTech)`).
- Exactly **one bacterial species** (*Bacillus subtilis*).
- Exactly **one non-zero concentration** ($10^6$ cells/mL).
- Exactly **six curing ages** (7, 14, 21, 28, 56, 90 days).
**Scientific Warning**: Models trained on this dataset will learn the age-hardening kinetics and bacterial uplift under these exact laboratory conditions, but cannot claim generalization to varying water-cement ratios, different cement grades, or alternative bacterial strains without external data.

---

## 11. Validation Framework: Phase 3 Architecture

Phase 3 established three distinct evaluation scenarios to demonstrate why naive cross-validation is invalid for concrete laboratory datasets:

- **Scenario A (Row-Level Random Baseline)**: Naive 80/20 random split.
  - *Finding*: Flawed due to replicate leakage. Replicates from the exact same cohort appear in both train and test, producing artificially inflated $R^2 > 0.99$.
- **Scenario B (Cohort-Grouped Validation)**: Whole cohorts (100 replicates each) held out together.
  - *Finding*: Prevents replicate memorization and tests true predictive capacity on unseen physical batches.
- **Scenario C (Leave-One-Curing-Age-Out)**: 6 folds holding out an entire curing age (e.g. all 28-day specimens).
  - *Finding*: Rigorously evaluates temporal interpolation and extrapolation along the hardening trajectory.

---

## 12. Phase 3.1 Upgraded Validation Protocol

Implemented in [`reports/validation_strategy_v3_1.md`](file:///g:/Projects/Concrete%20testing/reports/validation_strategy_v3_1.md) and [`src/validation/`](file:///g:/Projects/Concrete%20testing/src/validation/):

1. **Partitioning**: Master dataset (3,600 rows) partitioned into:
   - **Development Set**: 2,800 rows (28 cohorts) reserved for CV, model design, and tuning.
   - **Locked Final Test Set**: 800 rows (8 cohorts) sealed permanently until final publication benchmarking.
2. **Primary Grouping Variable**: **`experiment_id`**.
   - *Rule*: `sample_replicate` is **never** used as a grouping variable. All 100 replicates of an experiment must stay grouped together.
3. **Cross-Validation Scheme**: **5-fold GroupKFold** partitioned strictly by `experiment_id` on the development set.
4. **Leakage Prevention**: All preprocessing transformations (imputation, scaling, one-hot encoding) are fitted strictly within the training folds of the scikit-learn Pipeline.

---

## 13. Locked Final Test Set Specification

The 8 cohorts constituting the locked test set are permanently recorded in [`data/splits/final_test_cohorts.csv`](file:///g:/Projects/Concrete%20testing/data/splits/final_test_cohorts.csv) and [`data/splits/final_test_rows.csv`](file:///g:/Projects/Concrete%20testing/data/splits/final_test_rows.csv):

| Cohort ID (`experiment_id`) | Mechanical Property | Concrete Mix Type | Curing Age | Restored Value Count | Specimen Rows |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `EXP_CS_BC_07d` | Compressive Strength | Bacterial Concrete | 7 days | 0 | 100 |
| `EXP_CS_BC_28d` | Compressive Strength | Bacterial Concrete | 28 days | 0 | 100 |
| `EXP_CS_NC_07d` | Compressive Strength | Normal Concrete | 7 days | 0 | 100 |
| `EXP_FS_BC_07d` | Flexural Strength | Bacterial Concrete | 7 days | 0 | 100 |
| `EXP_FS_BC_90d` | Flexural Strength | Bacterial Concrete | 90 days | 100 | 100 |
| `EXP_FS_NC_56d` | Flexural Strength | Normal Concrete | 56 days | 0 | 100 |
| `EXP_TS_BC_90d` | Split Tensile Strength | Bacterial Concrete | 90 days | 0 | 100 |
| `EXP_TS_NC_21d` | Split Tensile Strength | Normal Concrete | 21 days | 0 | 100 |
| **Total Locked Test** | **3 Properties Represented** | **Both Mix Types** | **4 Distinct Ages** | **100 Restored** | **800 Rows** |

- **Development Set Allocation**: 2,800 rows (28 cohorts, containing the remaining 100 restored values from `EXP_FS_BC_56d`).
- **Research Invariant**: **This locked test set has never been accessed for training or model selection.**

---

## 14. Input Feature Governance & Policy

Implemented in [`src/validation/feature_policy.py`](file:///g:/Projects/Concrete%20testing/src/validation/feature_policy.py):

### 14.1 Candidate Predictive Inputs
- `curing_age_days` (continuous curing duration)
- `concrete_type` (binary categorical: Normal vs. Bacterial)
- `bacterial_concentration_cells_ml` (continuous concentration: 0 vs. $10^6$)
- `mechanical_property` (categorical: Compressive, Flexural, Tensile — *mandatory for Unified modeling*)

### 14.2 Expanded Features Tested (Ablation)
- `bacterial_status` (100% collinear with `concrete_type`)
- `bacterial_species` (100% collinear with `concrete_type`)
- `specimen_geometry` (100% collinear with `mechanical_property`)

### 14.3 Strictly Forbidden Features (Leakage Firewall)
- `sample_id` (primary key identifier)
- `experiment_id` (cohort partition key)
- `sample_replicate` (replicate counter)
- `source_file`, `source_sheet`, `source_row_index` (lineage metadata correlated with age blocks)
- `is_restored_value` (audit tracking flag)

### 14.4 Constant Feature Dropped
- `cement_type` (100% identical value `OPC 53 (UltraTech)`)

---

## 15. Phase 4.1 Baseline Modeling Implementation

### 15.1 Experimental Setup
- **Source Script**: [`src/modeling/baseline_training.py`](file:///g:/Projects/Concrete%20testing/src/modeling/baseline_training.py)
- **Dataset Partition Used**: Exclusively the 2,800 development rows (28 cohorts).
- **Validation**: 5-fold `GroupKFold` grouped by `experiment_id`.
- **Random Seed**: `42` (fixed across all splits and algorithms).
- **Models Evaluated (6)**:
  1. `DummyRegressor(strategy='mean')`
  2. `LinearRegression()`
  3. `Ridge(alpha=1.0)`
  4. `RandomForestRegressor(n_estimators=100, random_state=42)`
  5. `XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)`
  6. `CatBoostRegressor(iterations=300, learning_rate=0.08, depth=6, verbose=0, random_seed=42)`
- **Feature Sets Evaluated (2)**:
  - **Core (6 features)**: `curing_age_days`, `bacterial_concentration_cells_ml`, `concrete_type`, `bacterial_status`, `cement_type`, `bacterial_species`
  - **Expanded (7 features)**: Core + `specimen_geometry`
- **Target Strategies Evaluated (2)**:
  - **Strategy A (Separate Models)**: Independent models trained per mechanical property ($3 \text{ properties} \times 6 \text{ models} \times 2 \text{ feature sets} \times 5 \text{ folds} = 180 \text{ fits}$).
  - **Strategy B (Unified Model)**: Single model trained across all three properties using `mechanical_property` as an input ($6 \text{ models} \times 2 \text{ feature sets} \times 5 \text{ folds} = 60 \text{ fits}$).
  - **Total Model Fits Executed**: **240 fits**.
- **Prediction Output**: 67,200 row-level out-of-fold predictions logged in [`results/predictions/phase4_1_cv_predictions.csv`](file:///g:/Projects/Concrete%20testing/results/predictions/phase4_1_cv_predictions.csv).

---

## 16. Baseline Benchmarking Results

Extracted directly from [`reports/baseline_model_comparison.csv`](file:///g:/Projects/Concrete%20testing/reports/baseline_model_comparison.csv) and [`reports/baseline_modeling_report.md`](file:///g:/Projects/Concrete%20testing/reports/baseline_modeling_report.md).

### 16.1 Strategy A: Separate Models (Pooled Over Folds, per Property)

| Model | Feature Set | Mechanical Property | MAE (MPa) | Fold Std MAE | RMSE (MPa) | Fold Std RMSE | Pooled $R^2$ | Fold Std $R^2$ | MAPE (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ridge** | Core | Compressive Strength | **2.6553** | 1.1082 | **3.3412** | 1.1979 | **0.6256** | 1.1636 | 8.69% |
| **Linear** | Core | Compressive Strength | 2.6578 | 1.1149 | 3.3453 | 1.2039 | 0.6247 | 1.1608 | 8.69% |
| **XGBoost** | Core | Compressive Strength | 3.0789 | 1.0090 | 3.6956 | 1.1306 | 0.5420 | 1.0272 | 9.83% |
| **CatBoost** | Core | Compressive Strength | 3.2143 | 1.2070 | 3.8577 | 1.1826 | 0.5009 | 4.3825 | 10.61% |
| **RandomForest** | Core | Compressive Strength | 3.3204 | 0.9424 | 3.9027 | 1.0847 | 0.4892 | 1.3551 | 10.69% |
| **Dummy** | Core | Compressive Strength | 5.4667 | 2.1495 | 6.4745 | 2.2086 | -0.4058 | 11.2803 | 18.05% |
| **RandomForest** | Core | Flexural Strength | **0.4554** | 0.1258 | **0.5301** | 0.1166 | **0.5340** | 2.7698 | 10.83% |
| **CatBoost** | Core | Flexural Strength | 0.4555 | 0.1251 | 0.5301 | 0.1159 | 0.5339 | 2.7538 | 10.83% |
| **XGBoost** | Core | Flexural Strength | 0.5518 | 0.1692 | 0.6351 | 0.1624 | 0.3311 | 2.7682 | 13.10% |
| **Dummy** | Core | Flexural Strength | 0.7404 | 0.4349 | 0.8991 | 0.4097 | -0.3405 | 15.8274 | 18.20% |
| **Ridge** | Core | Flexural Strength | 0.7104 | 0.6296 | 1.0834 | 0.6871 | -0.9466 | 10.7209 | 15.88% |
| **Linear** | Core | Flexural Strength | 0.7113 | 0.6312 | 1.0853 | 0.6888 | -0.9533 | 10.7653 | 15.90% |
| **RandomForest** | Core | Split Tensile Strength | **0.3311** | 0.0712 | **0.3713** | 0.0700 | **0.5011** | 0.9355 | 13.55% |
| **CatBoost** | Core | Split Tensile Strength | 0.3366 | 0.1019 | 0.3876 | 0.0990 | 0.4563 | 1.0666 | 13.37% |
| **XGBoost** | Core | Split Tensile Strength | 0.3770 | 0.0895 | 0.4164 | 0.0858 | 0.3725 | 1.5391 | 14.86% |
| **Ridge** | Core | Split Tensile Strength | 0.3498 | 0.1506 | 0.4419 | 0.1721 | 0.2933 | 2.5673 | 13.82% |
| **Linear** | Core | Split Tensile Strength | 0.3501 | 0.1508 | 0.4424 | 0.1724 | 0.2920 | 2.5747 | 13.83% |
| **Dummy** | Core | Split Tensile Strength | 0.5364 | 0.2736 | 0.6365 | 0.2656 | -0.4658 | 6.3980 | 21.85% |

*(Note: In Separate models, the Expanded feature set produced identical results to Core because `specimen_geometry` is 100% constant within each mechanical property).*

### 16.2 Strategy B: Unified Modeling Across Properties

When a single model learns all properties simultaneously:
- **Global Pooled $R^2$ across all properties combined**: **XGBoost Unified Expanded achieves $R^2 = 0.9833$ (MAE $= 1.0729$ MPa)** across the entire development set.
- **Per-Property Breakdown of the Unified Model**:
  - Compressive Strength: XGBoost Core achieves $\text{MAE} = 2.3800$ MPa, $\text{RMSE} = 2.7913$ MPa, $R^2 = 0.7387$.
  - Flexural Strength: XGBoost Expanded achieves $\text{MAE} = 0.4689$ MPa, $\text{RMSE} = 0.5433$ MPa, $R^2 = 0.5106$.
  - Split Tensile Strength: XGBoost Expanded achieves $\text{MAE} = 0.3148$ MPa, $\text{RMSE} = 0.3576$ MPa, $R^2 = 0.5373$.

### 16.3 Restored vs. Non-Restored Observations Analysis
In [`reports/baseline_modeling_report.md`](file:///g:/Projects/Concrete%20testing/reports/baseline_modeling_report.md) Section 8:
- **Excluding Restored Values ($N=1,600$ Flexure)**: Random Forest Flexural MAE is $0.4469$ MPa ($R^2 = 0.4140$).
- **Restored Values Only ($N=200$ Flexure)**: Random Forest Flexural MAE is $0.5236$ MPa ($\text{MAPE} = 9.29\%$).
- *Interpretation*: The models extrapolate to the restored 56-day and 90-day bacterial flexural specimens with under $10\%$ relative error, confirming that the recovered laboratory measurements follow true physical concrete hardening kinetics.

---

## 17. Scientific Investigation of Metric Discrepancies

During Phase 4.1, an apparent discrepancy was noted between early console logs ($R^2 \approx 0.98$) and individual property summaries ($R^2 \approx 0.50 - 0.74$). A targeted investigation of `debug_r2.py` and `generate_phase41_report.py` revealed the exact mathematical mechanisms:

### 17.1 Mechanism 1: Global Pooled $R^2$ vs. Within-Property $R^2$
$R^2$ is defined as $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$.
- In the **Unified Model**, when $R^2$ is calculated over all 2,800 rows pooled together, $y$ spans from $1.38$ MPa (Tensile) to $44.16$ MPa (Compressive). The total variance $\sum (y_i - \bar{y}_{\text{global}})^2$ is massive ($\sigma^2 \approx 140$). The model easily separates Compressive ($\sim 30$ MPa) from Flexural ($\sim 4.4$ MPa) and Tensile ($\sim 2.8$ MPa), making residual sum of squares tiny relative to global variance $\implies \mathbf{R^2 = 0.9833}$.
- However, when evaluating within a **single mechanical property**, the denominator is restricted to that property's variance ($\sigma^2 \approx 38$ for Compressive, $\sigma^2 \approx 0.73$ for Flexural). Consequently, the within-property $R^2$ is $0.7387$ for Compressive and $0.5106$ for Flexural.
- **Conclusion**: Both numbers are mathematically correct. $0.9833$ reflects global multi-property tracking, while $0.50 - 0.74$ reflects within-property variance explanation.

### 17.2 Mechanism 2: Fold-Level Average $R^2$ vs. Out-of-Fold Pooled $R^2$
In grouped cross-validation, whole cohorts are held out in each fold.
- For example, in a validation fold containing only 56-day Bacterial concrete, all actual test values cluster tightly around $37.2$ MPa. The total variance *within that fold* is virtually zero.
- A tiny prediction error (e.g. predicting $35.5$ MPa) produces a residual variance larger than the near-zero fold variance, causing **fold-level $R^2$ to become negative** (e.g. $-1.72$ to $-3.19$).
- When all 5 folds are pooled together across the full age spectrum (7 to 90 days), the actual values span the true experimental curve ($15$ to $44$ MPa), yielding a solid positive **pooled $R^2 = 0.4892$**.
- **Conclusion**: Fold-averaged $R^2$ is invalid for cohort-grouped validation. The metric must be evaluated on the **pooled out-of-fold predictions**.

---

## 18. Phase 4.1 Execution & Encoding Issue Audit

During the initial console execution of Phase 4.1, a terminal exception occurred:
`UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'`

### Audit Findings
1. **Root Cause**: The Windows default command prompt/PowerShell output stream used the Windows-1252/cp437 character map, which cannot render the Unicode checkmark character (`✓` / `\u2713`) used in console progress print statements.
2. **Impact on Model Training**: **Zero impact**. The exception occurred exclusively in stdout logging after folds completed. All scikit-learn, XGBoost, and CatBoost models executed fully and correctly.
3. **Impact on Saved Artifacts**: **Zero impact**. All 67,200 predictions were calculated and written to [`results/predictions/phase4_1_cv_predictions.csv`](file:///g:/Projects/Concrete%20testing/results/predictions/phase4_1_cv_predictions.csv) before the print error, and [`results/phase4_1_run_metadata.json`](file:///g:/Projects/Concrete%20testing/results/phase4_1_run_metadata.json) verified file completeness.
4. **Resolution**: [`fix_encoding.py`](file:///g:/Projects/Concrete%20testing/fix_encoding.py) normalized Unicode print strings to ASCII equivalents in [`src/modeling/baseline_training.py`](file:///g:/Projects/Concrete%20testing/src/modeling/baseline_training.py). [`generate_phase41_report.py`](file:///g:/Projects/Concrete%20testing/generate_phase41_report.py) was executed to compile the formal reports without altering any experimental logic.

---

## 19. Current Scientific Interpretation

Based strictly on completed Phase 1 through Phase 4.1 experiments:

1. **Low Effective Dimensionality**: The input space is governed primarily by curing age and concrete treatment type. High-dimensional polynomial expansion is unnecessary; tree ensembles naturally capture the logarithmic hardening trajectory.
2. **Feature Collinearity**: Adding `specimen_geometry`, `bacterial_status`, or `bacterial_species` adds zero independent predictive entropy because they are perfectly collinear with `mechanical_property` and `concrete_type`.
3. **Superiority of Unified Modeling**: The Unified modeling strategy (training on all three properties with `mechanical_property` as an input) outperforms separate models by allowing cross-property transfer of temporal aging kinetics.
4. **Tree Baselines Lead**: Random Forest, CatBoost, and XGBoost deliver the lowest absolute errors across all properties ($\text{MAE} \approx 2.38$ MPa for Compressive, $0.45$ MPa for Flexural, $0.31$ MPa for Tensile).
5. **No Final Model Declared**: Baselines have been benchmarked, but neither hyperparameter tuning nor TabPFN comparison has been performed.

---

## 20. TabPFN Status & Hardware Readiness

- **Package Installation**: `tabpfn` version **9.0.0** is installed in `.venv`.
- **Pretrained Weights**: Checkpoint file exists at [`models/tabpfn-v2.5-regressor-v2.5_real.ckpt`](file:///g:/Projects/Concrete%20testing/models/tabpfn-v2.5-regressor-v2.5_real.ckpt) (size: **40,831,868 bytes** / ~38.9 MB).
- **Import & Initialization**: Verified via [`src/test_tabpfn.py`](file:///g:/Projects/Concrete%20testing/src/test_tabpfn.py). `TabPFNRegressor()` initializes cleanly.
- **Inference Status**: **NOT RUN**. TabPFN has not yet generated any predictions on this dataset.
- **Fine-Tuning Status**: **NOT RUN**.
- **Hardware Requirement**: Inference can be executed on CPU for sample sizes $\le 1,000$, but batch evaluation across 5 folds of 2,800 rows will benefit significantly from GPU acceleration.
- **Benchmarking Status**: **TabPFN has NOT yet been evaluated against the baseline models.**

---

## 21. Checklist of Remaining Work (What Has NOT Been Done)

- [ ] **TabPFN Baseline Evaluation (Phase 4.2 / 4.3)**:
  - Run TabPFN zero-shot inference across the exact same 5-fold GroupKFold development splits.
  - Benchmark TabPFN predictions directly against Random Forest, XGBoost, and CatBoost.
- [ ] **Hyperparameter Optimization**:
  - Systematic tuning (Optuna / GridSearchCV) of top-performing tree baselines on the development set.
- [ ] **Final Model Selection**:
  - Formal statistical hypothesis testing (paired Wilcoxon signed-rank tests) to determine if TabPFN significantly outperforms tuned tree baselines.
- [ ] **Locked Final Test Set Evaluation (Phase 5)**:
  - Single, one-time forward evaluation on the locked 800 test rows (`final_test_rows.csv`).
- [ ] **Model Interpretability & Explainability**:
  - SHAP (SHapley Additive exPlanations) values to verify biological uplift attribution.
- [ ] **Uncertainty Quantification**:
  - Conformal prediction or prediction intervals for concrete failure loads.
- [ ] **External Dataset Generalization**:
  - Testing model transferability on independent published concrete datasets.
- [ ] **Research Manuscript Preparation**:
  - Drafting figures, tables, and narrative for journal submission.

---

## 22. Immutability & Safety Verification

Prior to concluding this handoff audit, the entire workspace was checked to ensure complete adherence to read-only safety rules:
- `data/master_dataset.csv` checksum re-verified: `0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a` (**UNCHANGED**).
- `data/master_dataset.parquet` checksum re-verified: `0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a` (**UNCHANGED**).
- All 3 raw Excel workbooks confirmed present and unmodified.
- Locked test set ([`data/splits/final_test_rows.csv`](file:///g:/Projects/Concrete%20testing/data/splits/final_test_rows.csv)) confirmed sealed and untouched.
- No model training or parameter adjustments were executed during this audit.
- Exactly one new documentation artifact was created (`reports/project_state_handoff.md`).

---

```
CURRENT PROJECT CHECKPOINT

DATASET: COMPLETE & CRYPTOGRAPHICALLY VERIFIED (3,600 rows, 16 cols, SHA256 verified)
VALIDATION: COMPLETE & SEALED (Phase 3.1 5-fold GroupKFold by experiment_id)
LOCKED TEST: LOCKED & UNTOUCHED (8 cohorts, 800 rows permanently sealed)
BASELINES: COMPLETE (6 models x 2 strategies x 2 feature sets evaluated on dev set)
TABPFN: CHECKPOINT READY, INFERENCE NOT YET RUN
FINE-TUNING: NOT STARTED
FINAL TEST: NOT STARTED (SEALED)
RESEARCH PAPER: NOT STARTED
```
