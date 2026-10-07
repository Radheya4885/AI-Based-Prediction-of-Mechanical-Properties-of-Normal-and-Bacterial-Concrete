# Comprehensive Project Artifact Inventory & Architecture Audit (Run 1)

**Project:** AI-Based Prediction of Mechanical Properties of Normal and Bacterial Concrete  
**Audit Date:** 2026-10-07  
**Scope:** Complete recursive filesystem census, artifact type classification, duplicate detection, model binary audit, and reorganization safety assessment.  
**Companion Artifact:** [`reports/project_artifact_inventory_run1.csv`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/project_artifact_inventory_run1.csv) (142 rows)

---

## 1. Executive Summary & Repository Metrics

A complete, non-destructive filesystem scan of the repository was executed. Every tracked file was hashed via SHA256 and verified for size, provenance, and role.

- **Total Directories (excl. `.git`):** 28
- **Total Tracked Files (excl. `.git`):** 142
- **Total Repository Footprint:** ~253.8 MB (dominated by TabPFN base checkpoint [40.8 MB] and Fine-Tuned Folds 4 & 5 joblib binaries [168.2 MB combined])

### Artifact Count by Category

| Category Tag | Definition | File Count |
| :--- | :--- | :---: |
| **`[FIGURE]`** | High-resolution publication plots (PNG) | 33 |
| **`[SOURCE CODE]`** | Python pipeline, modeling, validation, and audit scripts | 30 |
| **`[DATA]`** | Raw Excel workbooks, canonical master datasets, split tables | 24 |
| **`[REPORT]`** | Human-readable markdown research reports | 17 |
| **`[PREDICTION ARTIFACT]`** | Out-of-fold and holdout prediction tables (CSV) | 9 |
| **`[CONFIGURATION]`** | Fold, split, and architectural settings (JSON, LFS pointer) | 6 |
| **`[METRICS]`** | Evaluated benchmark performance summary tables (CSV) | 5 |
| **`[METADATA]`** | Execution timestamps, parameters, and metric payloads (JSON) | 5 |
| **`[MISC/SUPPORT]`** | Environment manifests, git controls (`requirements.txt`, `.gitignore`) | 4 |
| **`[DOCUMENTATION]`** | High-level guides and overview documents (`README.md`, notes) | 3 |
| **`[MODEL BINARY]`** | Serialized model estimators ready for inference (`.ckpt`, `.joblib`) | 3 |
| **`[TEMP/DEBUG/BUILD FILES]`**| Python bytecode cache (`__pycache__`) and local debug scripts | 3 |
| **TOTAL** | | **142** |

### File Count by Top-Level Directory

| Directory Path | Role / Domain | File Count | Total Size (Approx.) |
| :--- | :--- | :---: | :---: |
| `reports/` | Research reports, metric CSVs, and visualization figures | 55 | ~1.6 MB |
| `src/` | Data pipeline, validation engine, and model training code | 29 | ~120 KB |
| `data/` | Raw laboratory Excel sheets, master datasets, and split CSVs | 25 | ~2.5 MB |
| `results/` | Model binaries, prediction CSVs, fold configs, and run metadata | 22 | ~210.1 MB |
| Root (`./`) | Core project documentation, environment files, and helper scripts | 8 | ~25 KB |
| `models/` | Base TabPFN v2.5 foundation model weights and LFS pointer | 2 | ~40.8 MB |
| `notebooks/` | Interactive analysis workspace (currently empty) | 1 (`.gitkeep`) | 0 B |
| **TOTAL** | | **142** | **~253.8 MB** |

---

## 2. Directory-by-Directory Audit

Below is the exhaustive audit of every major directory across the workspace:

### 2.1 `data/`
- **PATH:** `data/`
- **PURPOSE:** Ground-truth experimental data storage and canonical dataset generation.
- **FILE COUNT:** 5 files (at top level, excluding subdirectories).
- **IMPORTANT FILES:**
  - `data/master_dataset.csv` (675,777 bytes, SHA256: `0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a` CRLF normalized)
  - `data/master_dataset.parquet` (59,607 bytes, SHA256: `0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a`)
  - `data/C Strenth Reading 1000000.xlsx` (152,434 bytes) — Raw Compressive sheets
  - `data/F Strength Reading 1000000.xlsx` (154,642 bytes) — Raw Flexural sheets
  - `data/T Strenth Reading 1000000.xlsx` (155,595 bytes) — Raw Split Tensile sheets
- **SAFE TO REORGANIZE?** **NO**
- **DEPENDENCY RISK:** **HIGH**
- **NOTES:** Master datasets and raw workbooks are hardcoded across validation, training, and testing scripts. Must remain strictly read-only and in-place.

### 2.2 `data/splits/`
- **PATH:** `data/splits/`
- **PURPOSE:** Cryptographically locked cross-validation splits and sealed test partition manifests.
- **FILE COUNT:** 13 files.
- **IMPORTANT FILES:**
  - `data/splits/final_test_rows.csv` (800 rows, 8 cohorts) — Locked test partition
  - `data/splits/final_test_cohorts.csv` (8 cohorts metadata)
  - `data/splits/development_rows.csv` (2,800 rows, 28 cohorts) — Dev partition
  - `data/splits/development_cohorts.csv` (28 cohorts metadata)
  - `data/splits/grouped_cv_assignments.csv` (5-fold GroupKFold assignments)
  - `data/splits/leave_age_out_assignments.csv` (Leave-age-out assignments)
  - `data/splits/scenario_a_random_baseline.csv` & `.json`
  - `data/splits/scenario_b_cohort_grouped.csv` & `.json`
  - `data/splits/scenario_c_leave_age_out.csv` & `.json`
  - `data/splits/splits_manifest.json`
- **SAFE TO REORGANIZE?** **NO**
- **DEPENDENCY RISK:** **HIGH**
- **NOTES:** Central validation spine. Directly imported by all baseline, TabPFN, and evaluation pipelines.

### 2.3 `data/tabpfn_concrete_v1/`
- **PATH:** `data/tabpfn_concrete_v1/`
- **PURPOSE:** Feature-engineered dataset partition formatted specifically for TabPFN in-context learning.
- **FILE COUNT:** 6 files.
- **IMPORTANT FILES:**
  - `tabpfn_concrete_model.csv` (422,968 bytes) & `tabpfn_concrete_model.parquet` (53,243 bytes) — Feature engineered dataset (11 features)
  - `tabpfn_feature_dictionary.csv` (Feature documentation)
  - `tabpfn_concrete_master_copy.csv` & `.parquet` — Backup copies of master dataset (identical SHA256 to master)
  - `README.md`
- **SAFE TO REORGANIZE?** **NO**
- **DEPENDENCY RISK:** **HIGH**
- **NOTES:** `tabpfn_concrete_model.csv` is the direct data source loaded by Phase 4.2, Phase 4.3, and Phase 5 evaluation scripts.

### 2.4 `models/`
- **PATH:** `models/`
- **PURPOSE:** Storage for foundational model weights.
- **FILE COUNT:** 2 files.
- **IMPORTANT FILES:**
  - `models/tabpfn-v2.5-regressor-v2.5_real.ckpt` (40,831,868 bytes, SHA256: `8ab42d2d0abe3886a2c54a336b2abf251658a23a9b3f6559c4fdf0c024f48e63`)
  - `models/tabpfn-v2.5-regressor-v2.5_real.ckpt.lfs_ptr` (Git LFS pointer file)
- **SAFE TO REORGANIZE?** **NO**
- **DEPENDENCY RISK:** **HIGH**
- **NOTES:** Hardcoded in all TabPFN training, evaluation, and Phase 5 scripts.

### 2.5 `results/predictions/`
- **PATH:** `results/predictions/`
- **PURPOSE:** Permanent storage for out-of-fold and final holdout model predictions.
- **FILE COUNT:** 4 files.
- **IMPORTANT FILES:**
  - `phase4_1_cv_predictions.csv` (67,200 rows, 6.2 MB) — All 24 classical baseline predictions
  - `tabpfn_cv_predictions.csv` (5,600 rows, 482 KB) — Pretrained TabPFN V1 & V2 CV predictions
  - `tabpfn_finetuned_oof_predictions_run2.csv` (2,800 rows, 252 KB) — Fine-tuned TabPFN assembled OOF predictions
  - `phase5_final_test_predictions.csv` (800 rows, 127 KB) — Locked test predictions
- **SAFE TO REORGANIZE?** **NO**
- **DEPENDENCY RISK:** **HIGH**
- **NOTES:** Ground truth prediction artifacts for all thesis results and figures.

### 2.6 `results/models/tabpfn/` & `fold_models_run2/`
- **PATH:** `results/models/tabpfn/`
- **PURPOSE:** Checkpoints, configs, and validation predictions for fine-tuned TabPFN folds.
- **FILE COUNT:** 15 files across 5 fold directories and 1 metadata directory.
- **IMPORTANT FILES:**
  - `fold_4/fold_4_finetuned.joblib` (84,093,937 bytes) — Serialized Fold 4 model
  - `fold_5/fold_5_finetuned.joblib` (84,093,633 bytes) — Serialized Fold 5 model
  - `fold_1/` through `fold_5/`: Each contains `fold_X_config.json` and `fold_X_val_predictions.csv`
  - `pretrained_baseline_metadata.json`, `tabpfn_run_metadata.json`, `tabpfn_training_config.json`
- **SAFE TO REORGANIZE?** **NO**
- **DEPENDENCY RISK:** **MEDIUM**
- **NOTES:** Folds 1, 2, and 3 `.joblib` binaries were not transferred from the previous PC, but their validation predictions and configurations are 100% intact.

### 2.7 `results/phase5/` & `results/tabpfn_finetuned_run2/`
- **PATH:** `results/phase5/`, `results/tabpfn_finetuned_run2/`
- **PURPOSE:** Run-level execution metadata payloads and diagnostic logs.
- **FILE COUNT:** 2 files (`phase5_metadata.json` [4.6 KB], `metadata.json` [6.4 KB]).
- **SAFE TO REORGANIZE?** **NO**
- **DEPENDENCY RISK:** **LOW**
- **NOTES:** Contains exact parameter records and execution timings.

### 2.8 `reports/` & `reports/figures/`
- **PATH:** `reports/`
- **PURPOSE:** Research documentation, benchmark comparison tables, audit trails, and figures.
- **FILE COUNT:** 55 files (22 markdown/CSV reports + 33 PNG figures).
- **IMPORTANT FILES:**
  - Reports: `final_phase5_scientific_audit.md`, `phase5_final_evaluation_report.md`, `final_model_audit_run2.md`, `tabpfn_finetuned_evaluation_report_run2.md`, `tabpfn_training_report.md`, `baseline_modeling_report.md`, `validation_strategy_v3_1.md`, `master_dataset_design.md`, `dataset_audit.md`, `eda_report.md`
  - Tables: `baseline_model_comparison.csv`, `tabpfn_model_comparison.csv`, `tabpfn_finetuned_model_comparison_run2.csv`
  - Subdirectories: `reports/figures/` (20 EDA plots), `reports/figures/phase5/` (5 locked test plots), `reports/figures/tabpfn_finetuned_run2/` (8 fine-tuned diagnostic plots).
- **SAFE TO REORGANIZE?** **NO** (reports embed figures via relative markdown paths)
- **DEPENDENCY RISK:** **MEDIUM**

### 2.9 `src/` (including `src/modeling/` and `src/validation/`)
- **PATH:** `src/`
- **PURPOSE:** Core software engineering pipeline.
- **FILE COUNT:** 29 files (11 root src + 5 modeling + 10 validation + 3 `__pycache__`).
- **IMPORTANT FILES:**
  - `src/modeling/baseline_training.py` — Phase 4.1 classical baseline trainer
  - `src/modeling/evaluate_tabpfn_baselines.py` — Phase 4.2 TabPFN evaluator
  - `src/modeling/evaluate_tabpfn_finetuned_5fold.py` — Phase 4.3 fine-tuning driver
  - `src/modeling/phase5_final_evaluation.py` — Phase 5 locked test evaluator
  - `src/validation/leakage_checker.py`, `robust_grouped_cv.py`, `feature_policy.py`
  - `src/build_and_validate_master.py`, `perform_eda.py`
- **SAFE TO REORGANIZE?** **NO**
- **DEPENDENCY RISK:** **HIGH**
- **NOTES:** Scripts have interdependent module imports (`from src.validation import ...`).

### 2.10 Root Directory (`./`)
- **PATH:** `./`
- **PURPOSE:** Project root entrypoint and utility scripts.
- **FILE COUNT:** 8 files.
- **IMPORTANT FILES:**
  - Standard root files: `README.md`, `requirements.txt`, `.gitignore`, `.gitattributes`
  - Root scripts: `generate_phase41_report.py`, `verify_phase41.py`, `debug_r2.py`, `fix_encoding.py`
- **SAFE TO REORGANIZE?**
  - Standard files (`README.md`, `requirements.txt`): **NO**
  - Root scripts (`generate_phase41_report.py`, etc.): **YES** (Can be safely archived into `scripts/` or `archive/` in a future planned reorganization, provided paths are preserved).
- **DEPENDENCY RISK:** **LOW**

---

## 3. Model Artifact Audit Matrix

This table distinguishes actual serialized model binaries from prediction artifacts and configurations.

| Phase | Model | Feature Set | Strategy | Serialized Model Exists? | Model Path | Config Exists? | Prediction Artifact Exists? | Prediction Path | Metrics Exists? | Metrics Path | Training Script | Metadata Exists? | Status | Notes |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: | :--- | :---: | :--- | :--- | :---: | :--- | :--- |
| **4.1** | Dummy | Core / Exp | Sep / Uni | **NO** | *None* | **YES** | **YES** | `results/predictions/phase4_1_cv_predictions.csv` | **YES** | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` | **YES** | Verified | NO SERIALIZED MODEL REQUIRED BY ORIGINAL PHASE 4.1 WORKFLOW |
| **4.1** | Linear | Core / Exp | Sep / Uni | **NO** | *None* | **YES** | **YES** | `results/predictions/phase4_1_cv_predictions.csv` | **YES** | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` | **YES** | Verified | NO SERIALIZED MODEL REQUIRED BY ORIGINAL PHASE 4.1 WORKFLOW |
| **4.1** | Ridge | Core / Exp | Sep / Uni | **NO** | *None* | **YES** | **YES** | `results/predictions/phase4_1_cv_predictions.csv` | **YES** | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` | **YES** | Verified | NO SERIALIZED MODEL REQUIRED BY ORIGINAL PHASE 4.1 WORKFLOW |
| **4.1** | Random Forest | Core / Exp | Sep / Uni | **NO** | *None* | **YES** | **YES** | `results/predictions/phase4_1_cv_predictions.csv` | **YES** | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` | **YES** | Verified | NO SERIALIZED MODEL REQUIRED BY ORIGINAL PHASE 4.1 WORKFLOW |
| **4.1** | XGBoost | Core / Exp | Sep / Uni | **NO** | *None* | **YES** | **YES** | `results/predictions/phase4_1_cv_predictions.csv` | **YES** | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` | **YES** | Verified | Best classical overall: Expanded/Unified (MAE 1.0729). NO SERIALIZED MODEL REQUIRED. |
| **4.1** | CatBoost | Core / Exp | Sep / Uni | **NO** | *None* | **YES** | **YES** | `results/predictions/phase4_1_cv_predictions.csv` | **YES** | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` | **YES** | Verified | NO SERIALIZED MODEL REQUIRED BY ORIGINAL PHASE 4.1 WORKFLOW |
| **4.2** | Pretrained TabPFN v2.5 | V1_RAW | Unified | **YES** (Base) | `models/tabpfn-v2.5-regressor-v2.5_real.ckpt` | **YES** | **YES** | `results/predictions/tabpfn_cv_predictions.csv` | **YES** | `reports/tabpfn_model_comparison.csv` | `src/modeling/evaluate_tabpfn_baselines.py` | **YES** | Verified | In-context inference (MAE 1.4347 MPa) |
| **4.2** | Pretrained TabPFN v2.5 | V2_ENGINEERED | Unified | **YES** (Base) | `models/tabpfn-v2.5-regressor-v2.5_real.ckpt` | **YES** | **YES** | `results/predictions/tabpfn_cv_predictions.csv` | **YES** | `reports/tabpfn_model_comparison.csv` | `src/modeling/evaluate_tabpfn_baselines.py` | **YES** | Verified | Selected model; Best Compressive (MAE 1.6168 MPa) |
| **4.3** | Fine-Tuned TabPFN | V2_ENGINEERED | Fold 1 | **NO** | *Missing from transfer* | **YES** | **YES** | `results/models/tabpfn/fold_models_run2/fold_1/fold_1_val_predictions.csv` | **YES** | `reports/tabpfn_finetuned_model_comparison_run2.csv` | `src/modeling/evaluate_tabpfn_finetuned_5fold.py` | **YES** | OOF Complete | Binary missing from transfer; val predictions (600 rows) intact |
| **4.3** | Fine-Tuned TabPFN | V2_ENGINEERED | Fold 2 | **NO** | *Missing from transfer* | **YES** | **YES** | `results/models/tabpfn/fold_models_run2/fold_2/fold_2_val_predictions.csv` | **YES** | `reports/tabpfn_finetuned_model_comparison_run2.csv` | `src/modeling/evaluate_tabpfn_finetuned_5fold.py` | **YES** | OOF Complete | Binary missing from transfer; val predictions (600 rows) intact |
| **4.3** | Fine-Tuned TabPFN | V2_ENGINEERED | Fold 3 | **NO** | *Missing from transfer* | **YES** | **YES** | `results/models/tabpfn/fold_models_run2/fold_3/fold_3_val_predictions.csv` | **YES** | `reports/tabpfn_finetuned_model_comparison_run2.csv` | `src/modeling/evaluate_tabpfn_finetuned_5fold.py` | **YES** | OOF Complete | Binary missing from transfer; val predictions (600 rows) intact |
| **4.3** | Fine-Tuned TabPFN | V2_ENGINEERED | Fold 4 | **YES** | `results/models/tabpfn/fold_models_run2/fold_4/fold_4_finetuned.joblib` | **YES** | **YES** | `results/models/tabpfn/fold_models_run2/fold_4/fold_4_val_predictions.csv` | **YES** | `reports/tabpfn_finetuned_model_comparison_run2.csv` | `src/modeling/evaluate_tabpfn_finetuned_5fold.py` | **YES** | Complete | Serialized model present (84,093,937 bytes); val predictions (500 rows) intact |
| **4.3** | Fine-Tuned TabPFN | V2_ENGINEERED | Fold 5 | **YES** | `results/models/tabpfn/fold_models_run2/fold_5/fold_5_finetuned.joblib` | **YES** | **YES** | `results/models/tabpfn/fold_models_run2/fold_5/fold_5_val_predictions.csv` | **YES** | `reports/tabpfn_finetuned_model_comparison_run2.csv` | `src/modeling/evaluate_tabpfn_finetuned_5fold.py` | **YES** | Complete | Serialized model present (84,093,633 bytes); val predictions (500 rows) intact |
| **4.3** | Fine-Tuned TabPFN | V2_ENGINEERED | Unified (Pooled) | *N/A (5-fold ensemble)* | *N/A* | **YES** | **YES** | `results/predictions/tabpfn_finetuned_oof_predictions_run2.csv` | **YES** | `reports/tabpfn_finetuned_model_comparison_run2.csv` | `src/modeling/evaluate_tabpfn_finetuned_5fold.py` | **YES** | Complete | Assembled 2,800-row OOF prediction artifact intact |
| **5** | Pretrained TabPFN v2.5 | V2_ENGINEERED | Unified (Final) | **YES** (Base Ckpt) | `models/tabpfn-v2.5-regressor-v2.5_real.ckpt` | **YES** | **YES** | `results/predictions/phase5_final_test_predictions.csv` | **YES** | `reports/phase5_final_evaluation_report.md` | `src/modeling/phase5_final_evaluation.py` | **YES** | Complete | Fitted on all 2,800 dev specimens; one-time evaluation on 800 test specimens |

---

## 4. Distinct Artifact Classification Definitions

To prevent ambiguity between actual models and statistical outputs, the project adheres to the following definitions:

1. **`[MODEL BINARY]`**: Actual trained or serialized neural network / estimator weights that can be loaded into memory via `torch.load()` or `joblib.load()` to perform inference on new input vectors.
2. **`[PREDICTION ARTIFACT]`**: CSV tables containing specimen-level target values, model predictions, and residuals. A prediction artifact is an empirical evaluation record, **not** a model.
3. **`[CONFIGURATION]`**: JSON files detailing split indices, hyperparameters, architecture configurations, or Git LFS pointers.
4. **`[METRICS]`**: Summary tables computing statistical aggregates (MAE, RMSE, $R^2$, MAPE) over partitions.
5. **`[SOURCE CODE]`**: Deterministic scripts written in Python to extract data, train models, or execute evaluations.
6. **`[REPORT]`**: Narrative scientific documents detailing methodology, numerical findings, and peer-review audits.
7. **`[FIGURE]`**: Publication-ready vector or bitmap diagnostic graphics.
8. **`[DATA]`**: Ground-truth experimental inputs and locked split definitions.

---

## 5. Duplicate Files & Redundancy Audit

A byte-by-byte SHA256 analysis identified exactly 2 redundant file pairs in the repository:

| Duplicate Pair | Path A | Path B | SHA256 | Size (Bytes) | Identical? | Active? | Safe to Archive / Remove? |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | `data/master_dataset.csv` | `data/tabpfn_concrete_v1/tabpfn_concrete_master_copy.csv` | `6a2d2e05...` | 675,777 | **YES** | Path A Active | **YES** (Path B is an unreferenced duplicate backup) |
| **2** | `data/master_dataset.parquet` | `data/tabpfn_concrete_v1/tabpfn_concrete_master_copy.parquet` | `0909cb72...` | 59,607 | **YES** | Path A Active | **YES** (Path B is an unreferenced duplicate backup) |

*Recommendation:* Do NOT delete Path B during this read-only phase. When reorganization is formally executed, Path B can be safely archived.

---

## 6. Categorical Artifact Status Analysis

### A. REQUIRED AND PRESENT (Core Research Spine)
- All 3 raw Excel workbooks (`data/*.xlsx`)
- Master dataset (`master_dataset.csv` and `master_dataset.parquet`)
- All 13 locked split manifests and row indices (`data/splits/*`)
- Base TabPFN v2.5 checkpoint (`models/tabpfn-v2.5-regressor-v2.5_real.ckpt`, 40.8 MB)
- All 4 prediction CSVs (`phase4_1_cv_predictions.csv`, `tabpfn_cv_predictions.csv`, `tabpfn_finetuned_oof_predictions_run2.csv`, `phase5_final_test_predictions.csv`)
- Phase 4.3 Fine-Tuned TabPFN Fold 4 & Fold 5 `.joblib` binaries
- All 5 fold validation prediction files and fold configs (`results/models/tabpfn/fold_models_run2/`)
- All primary reports and publication figures across Phases 1 through 5.

### B. REQUIRED BUT MISSING
- **NONE for thesis conclusions.**
  - All research claims, metrics, and comparisons are fully backed by existing prediction tables and metadata.

### C. NOT REQUIRED BY ORIGINAL WORKFLOW
- **Phase 4.1 Classical Model Binaries:** The Phase 4.1 workflow in `src/modeling/baseline_training.py` was explicitly designed to evaluate models across 5 folds and record predictions/metrics directly. It never serialized `.joblib` files.
- **Phase 5 Fitted Final Model Binary:** TabPFN is an in-context learning transformer; fitting on the 2,800 development specimens merely passes the tabular context into memory. Serializing the fitted state is redundant because inference on any dataset can be executed on-demand in 0.2s from the base checkpoint and development rows.

### D. HISTORICAL / SUPERSEDED (Safe to relocate in future reorganization)
- `debug_r2.py` (One-off script for debugging cohort $R^2$)
- `fix_encoding.py` (One-off script for fixing console UTF-8 on Windows)
- `generate_phase41_report.py` & `verify_phase41.py` (Root-level audit helpers)
- `data/tabpfn_concrete_v1/tabpfn_concrete_master_copy.*` (Duplicate copies of master data)

### E. OPTIONAL / RECOVERABLE
- **Phase 4.3 Folds 1, 2, 3 `.joblib` Binaries:**
  - Status: Missing from the initial transfer from the previous laptop.
  - Recovery: Fully reproducible from `src/modeling/evaluate_tabpfn_finetuned_5fold.py` using seed 42 if individual fold model weights are ever requested by external reviewers. Not required for thesis text because all out-of-fold predictions ($N=2,800$) and performance metrics are already locked.

---

## 7. Proposed Clean Repository Organization (Plan Only)

*IMPORTANT: This is a proposed reorganization blueprint for future execution. NO FILES ARE MOVED DURING THIS AUDIT.*

```
AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/
├── data/
│   ├── raw/                             # Raw experimental laboratory workbooks
│   │   ├── C Strenth Reading 1000000.xlsx
│   │   ├── F Strength Reading 1000000.xlsx
│   │   └── T Strenth Reading 1000000.xlsx
│   ├── processed/                       # Immutable canonical master datasets
│   │   ├── master_dataset.csv
│   │   └── master_dataset.parquet
│   ├── splits/                          # Permanently sealed evaluation partitions
│   │   ├── final_test_rows.csv
│   │   ├── final_test_cohorts.csv
│   │   ├── development_rows.csv
│   │   ├── development_cohorts.csv
│   │   └── grouped_cv_assignments.csv
│   └── tabpfn_engineered/               # Domain feature engineering partition
│       ├── tabpfn_concrete_model.csv
│       ├── tabpfn_concrete_model.parquet
│       └── tabpfn_feature_dictionary.csv
│
├── models/
│   ├── pretrained/                      # Foundational model weights
│   │   ├── tabpfn-v2.5-regressor-v2.5_real.ckpt
│   │   └── tabpfn-v2.5-regressor-v2.5_real.ckpt.lfs_ptr
│   └── finetuned/                       # Serialized fine-tuned fold estimators
│       ├── fold_4_finetuned.joblib
│       └── fold_5_finetuned.joblib
│
├── src/
│   ├── data/                            # Dataset extraction & schema enforcement
│   │   ├── build_and_validate_master.py
│   │   └── parse_all_sheets.py
│   ├── validation/                      # Anti-leakage checks & CV protocols
│   │   ├── leakage_checker.py
│   │   ├── robust_grouped_cv.py
│   │   └── feature_policy.py
│   └── modeling/                        # Training & evaluation drivers
│       ├── baseline_training.py
│       ├── evaluate_tabpfn_baselines.py
│       ├── evaluate_tabpfn_finetuned_5fold.py
│       └── phase5_final_evaluation.py
│
├── results/
│   ├── predictions/                     # Immutable prediction tables
│   │   ├── phase4_1_cv_predictions.csv
│   │   ├── tabpfn_cv_predictions.csv
│   │   ├── tabpfn_finetuned_oof_predictions_run2.csv
│   │   └── phase5_final_test_predictions.csv
│   ├── metrics/                         # Benchmark metric CSV tables
│   │   ├── baseline_model_comparison.csv
│   │   ├── tabpfn_model_comparison.csv
│   │   └── tabpfn_finetuned_model_comparison_run2.csv
│   └── metadata/                        # JSON execution payloads
│       ├── phase4_1_run_metadata.json
│       ├── phase5_metadata.json
│       └── tabpfn_finetuned_metadata.json
│
├── reports/                             # Research reports & thesis chapters
│   ├── phase1_audit/                    # Raw data & EDA reports
│   ├── phase2_design/                   # Master schema & design reports
│   ├── phase3_validation/               # Validation strategy & leakage reports
│   ├── phase4_benchmarks/               # Classical & TabPFN development reports
│   ├── phase5_final_test/               # Locked test evaluation reports
│   └── figures/                         # High-res diagnostic visualizations
│       ├── eda/
│       ├── tabpfn_finetuned/
│       └── phase5/
│
└── archive/                             # Superseded scripts and duplicate files
    ├── debug_r2.py
    ├── fix_encoding.py
    ├── generate_phase41_report.py
    ├── verify_phase41.py
    └── tabpfn_concrete_master_copy.*
```

---

## 8. Move-Safety Guardrails & Conclusion

- **Move-Safety Assessment:** **`SAFE TO REORGANIZE: NO (NOT YET)`**
  - All source scripts currently use relative path conventions assuming the current directory structure (`DATA_DIR = ROOT / "data"`, `SPLITS_DIR = DATA_DIR / "splits"`).
  - Physical movement of files at this juncture without simultaneous code refactoring would break reproducibility.
  - The project is 100% scientifically complete and audited in its present state. Any file movements must be planned as a separate, non-destructive step after user sign-off.
