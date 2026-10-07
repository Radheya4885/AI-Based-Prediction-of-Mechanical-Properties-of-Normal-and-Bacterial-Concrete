# Stage A: Staged Project Reorganization Plan (Run 1)

**Project:** AI-Based Prediction of Mechanical Properties of Normal and Bacterial Concrete  
**Plan Date:** 2026-10-07  
**Execution Phase:** STAGE A ONLY (READ-ONLY ARCHITECTURE PLANNING)  
**Safety Classification:** **`SAFE TO EXECUTE STAGE B: NO (READ-ONLY PLANNING COMPLETE; REQUIRES USER APPROVAL BEFORE MOVES)`**  
**Companion Artifacts:**
- Move Map: [`reports/project_reorganization_move_map_run1.csv`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/project_reorganization_move_map_run1.csv) (142 rows)
- Dependency Map: [`reports/project_dependency_map_run1.csv`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/project_dependency_map_run1.csv) (245 path dependencies)
- Risk Matrix: [`reports/project_reorganization_risks_run1.md`](file:///C:/Users/MAYURSHRESH/OneDrive/Desktop/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/reports/project_reorganization_risks_run1.md)

---

## 1. Current Repository Structure

The current repository layout comprises 142 tracked files across 28 directories:

```
AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/
├── .gitattributes                           # Git LFS tracking rules (*.ckpt, *.joblib)
├── .gitignore                               # Standard Python gitignore rules
├── README.md                                # Project overview and installation guide
├── requirements.txt                         # Python package dependencies
├── debug_r2.py                              # Root utility: R² debug helper
├── fix_encoding.py                          # Root utility: Windows encoding helper
├── generate_phase41_report.py               # Root utility: Phase 4.1 report generator
├── verify_phase41.py                        # Root utility: Phase 4.1 verification helper
├── data/
│   ├── C Strenth Reading 1000000.xlsx       # Raw Compressive laboratory workbook
│   ├── F Strength Reading 1000000.xlsx       # Raw Flexural laboratory workbook
│   ├── T Strenth Reading 1000000.xlsx       # Raw Split Tensile laboratory workbook
│   ├── master_dataset.csv                   # Immutable canonical master dataset (CSV)
│   ├── master_dataset.parquet               # Immutable canonical master dataset (Parquet)
│   ├── splits/                              # 13 locked split manifests and row tables
│   │   ├── development_cohorts.csv
│   │   ├── development_rows.csv
│   │   ├── final_test_cohorts.csv
│   │   ├── final_test_rows.csv              # Permanently sealed test set (800 rows)
│   │   ├── grouped_cv_assignments.csv
│   │   ├── leave_age_out_assignments.csv
│   │   ├── scenario_a_random_baseline.*
│   │   ├── scenario_b_cohort_grouped.*
│   │   ├── scenario_c_leave_age_out.*
│   │   └── splits_manifest.json
│   └── tabpfn_concrete_v1/                  # TabPFN engineered dataset partition
│       ├── tabpfn_concrete_model.csv        # 11-feature model input
│       ├── tabpfn_concrete_model.parquet
│       ├── tabpfn_feature_dictionary.csv
│       ├── tabpfn_concrete_master_copy.csv  # Exact duplicate of master_dataset.csv
│       ├── tabpfn_concrete_master_copy.parquet
│       └── README.md
├── models/
│   ├── tabpfn-v2.5-regressor-v2.5_real.ckpt # Base TabPFN v2.5 foundation weights (40.8 MB)
│   └── tabpfn-v2.5-regressor-v2.5_real.ckpt.lfs_ptr # Git LFS pointer file
├── notebooks/                               # Interactive notebooks (currently empty)
├── reports/
│   ├── figures/                             # 20 EDA / Phase 2-3 figures
│   ├── figures/phase5/                      # 5 Phase 5 final test figures
│   ├── figures/tabpfn_finetuned_run2/       # 8 Phase 4.3 fine-tuned TabPFN figures
│   ├── baseline_model_comparison.csv        # Phase 4.1 metrics table (72 rows)
│   ├── baseline_modeling_report.md
│   ├── dataset_audit.md
│   ├── eda_report.md
│   ├── final_model_audit_run2.md
│   ├── final_phase5_scientific_audit.md
│   ├── master_dataset_design.md
│   ├── master_dataset_validation.md
│   ├── phase5_final_evaluation_report.md
│   ├── project_state_handoff.md
│   ├── tabpfn_finetuned_evaluation_report_run2.md
│   ├── tabpfn_training_report.md
│   ├── validation_strategy_v3_1.md
│   └── ... (additional schemas and handoffs)
├── results/
│   ├── phase4_1_run_metadata.json
│   ├── phase5/                              # Phase 5 execution metadata
│   │   └── phase5_metadata.json
│   ├── predictions/                         # Canonical prediction artifacts
│   │   ├── phase4_1_cv_predictions.csv      # 67,200 rows (24 classical configurations)
│   │   ├── tabpfn_cv_predictions.csv        # 5,600 rows (Pretrained TabPFN V1 & V2)
│   │   ├── tabpfn_finetuned_oof_predictions_run2.csv # 2,800 rows (Fine-Tuned TabPFN OOF)
│   │   └── phase5_final_test_predictions.csv# 800 rows (Final locked test set)
│   ├── models/tabpfn/                       # Fine-Tuned fold checkpoints & configs
│   │   ├── fold_models_run2/
│   │   │   ├── fold_1/ (config + val predictions; binary missing)
│   │   │   ├── fold_2/ (config + val predictions; binary missing)
│   │   │   ├── fold_3/ (config + val predictions; binary missing)
│   │   │   ├── fold_4/ (config + val predictions + fold_4_finetuned.joblib [84 MB])
│   │   │   └── fold_5/ (config + val predictions + fold_5_finetuned.joblib [84 MB])
│   │   └── ... (metadata and training configs)
│   └── tabpfn_finetuned_run2/metadata.json
└── src/
    ├── build_and_validate_master.py
    ├── perform_eda.py
    ├── parse_all_sheets.py
    ├── modeling/                            # Active training & evaluation drivers
    │   ├── baseline_training.py
    │   ├── evaluate_tabpfn_baselines.py
    │   ├── evaluate_tabpfn_finetuned_5fold.py
    │   └── phase5_final_evaluation.py
    └── validation/                          # Leakage checkers & CV generators
        ├── leakage_checker.py
        ├── robust_grouped_cv.py
        ├── feature_policy.py
        └── split_generator.py
```

---

## 2. Proposed Target Organization

The target structure balances aesthetic cleanliness with **zero risk to canonical research assets**:

```
AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete/
├── .gitattributes                           # [UNCHANGED]
├── .gitignore                               # [UNCHANGED]
├── README.md                                # [UNCHANGED]
├── requirements.txt                         # [UNCHANGED]
│
├── data/                                    # [CANONICAL ASSETS KEPT IN PLACE]
│   ├── C Strenth Reading 1000000.xlsx
│   ├── F Strength Reading 1000000.xlsx
│   ├── T Strenth Reading 1000000.xlsx
│   ├── master_dataset.csv                   # Immutable ground truth
│   ├── master_dataset.parquet               # Immutable ground truth
│   ├── splits/                              # Immutable locked partitions
│   │   ├── development_rows.csv
│   │   ├── final_test_rows.csv
│   │   └── grouped_cv_assignments.csv
│   └── tabpfn_concrete_v1/                  # TabPFN engineered inputs
│       ├── tabpfn_concrete_model.csv
│       ├── tabpfn_concrete_model.parquet
│       └── tabpfn_feature_dictionary.csv
│
├── models/
│   ├── pretrained/                          # Relocate base checkpoint here in Stage B
│   │   ├── tabpfn-v2.5-regressor-v2.5_real.ckpt
│   │   └── tabpfn-v2.5-regressor-v2.5_real.ckpt.lfs_ptr
│   └── finetuned/                           # Keep fine-tuned fold models cleanly grouped
│       ├── fold_4_finetuned.joblib
│       └── fold_5_finetuned.joblib
│
├── results/
│   ├── predictions/                         # [CANONICAL PREDICTIONS KEPT IN PLACE]
│   │   ├── phase4_1_cv_predictions.csv
│   │   ├── tabpfn_cv_predictions.csv
│   │   ├── tabpfn_finetuned_oof_predictions_run2.csv
│   │   └── phase5_final_test_predictions.csv
│   ├── metrics/                             # Consolidate benchmark CSV tables here
│   │   ├── baseline_model_comparison.csv
│   │   ├── tabpfn_model_comparison.csv
│   │   └── tabpfn_finetuned_model_comparison_run2.csv
│   ├── metadata/                            # Consolidate execution payloads here
│   │   ├── phase4_1_run_metadata.json
│   │   ├── phase5_metadata.json
│   │   └── tabpfn_finetuned_metadata.json
│   └── fold_artifacts/                      # Individual fold configs & val CSVs
│       └── fold_1/ through fold_5/
│
├── src/                                     # [ACTIVE PYTHON CODE KEPT IN PLACE]
│   ├── modeling/
│   ├── validation/
│   ├── build_and_validate_master.py
│   └── perform_eda.py
│
├── reports/                                 # [REPORTS ORGANIZED LOGICALLY BY PHASE]
│   ├── phase1_audit/
│   ├── phase2_design/
│   ├── phase3_validation/
│   ├── phase4_benchmarks/
│   ├── phase5_final_test/
│   └── figures/                             # Keep figure assets organized
│       ├── phase1_eda/
│       ├── phase4_tabpfn/
│       └── phase5_final_test/
│
└── archive/                                 # [NON-BREAKING ARCHIVE STAGING]
    ├── superseded_scripts/                  # One-off audit scripts moved here
    │   ├── debug_r2.py
    │   ├── fix_encoding.py
    │   ├── generate_phase41_report.py
    │   └── verify_phase41.py
    └── duplicate_data/                      # Redundant duplicate backups
        ├── tabpfn_concrete_master_copy.csv
        └── tabpfn_concrete_master_copy.parquet
```

---

## 3. High-Level Move Map Summary

Of the 142 files in the repository:
- **Files to Keep Strictly in Place:** **136 files** (95.8% of repository).
  - All canonical data files, raw sheets, locked splits, prediction CSVs, active source modules, and active figures.
- **Files to Archive (Safe, Non-Breaking):** **6 files** (4.2% of repository).
  - 4 root utility scripts (`debug_r2.py`, `fix_encoding.py`, `generate_phase41_report.py`, `verify_phase41.py`).
  - 2 redundant duplicate dataset copies (`tabpfn_concrete_master_copy.csv`, `tabpfn_concrete_master_copy.parquet`).
- **Files to Move in Stage B:** **0 files until explicit approval and path redirection.**

---

## 4. Code Reference Analysis

The repository scan discovered **245 direct path references across 23 Python scripts**:
- `src/modeling/baseline_training.py` contains 47 path references to `data/`, `data/splits/`, `results/predictions/`, and `reports/`.
- `src/modeling/phase5_final_evaluation.py` contains 29 path references to `data/`, `data/splits/final_test_rows.csv`, `models/`, and `results/predictions/`.
- `src/validation/robust_grouped_cv.py` contains 46 path references to `data/master_dataset.*` and `data/splits/`.
- `src/perform_eda.py` contains 44 path references to `data/` and `reports/figures/`.

**Strategic Decision:** To guarantee 100% scientific reproducibility and prevent path breakage, **none of the paths accessed by these scripts should be physically moved**. Keeping canonical data, splits, predictions, and model weights at their current locations eliminates 100% of runtime path failure risks.

---

## 5. Git / LFS Considerations

- Current `.gitattributes`:
  ```
  *.ckpt filter=lfs diff=lfs merge=lfs -text
  *.joblib filter=lfs diff=lfs merge=lfs -text
  ```
- Because `.gitattributes` uses pattern matching (`*.ckpt`, `*.joblib`), moving `tabpfn-v2.5-regressor-v2.5_real.ckpt` or `fold_4_finetuned.joblib` to any subdirectory would **not** break Git LFS tracking rules.
- However, moving large files triggers full Git file renames / re-staging. To avoid repository churn, keeping them at their present relative locations is safest.

---

## 6. Duplicate Data Handling

The two duplicate pairs:
1. `data/master_dataset.csv` == `data/tabpfn_concrete_v1/tabpfn_concrete_master_copy.csv`
2. `data/master_dataset.parquet` == `data/tabpfn_concrete_v1/tabpfn_concrete_master_copy.parquet`

- **Analysis:**
  - `data/master_dataset.*` is imported by 18 scripts.
  - `data/tabpfn_concrete_v1/tabpfn_concrete_master_copy.*` is imported by **0 scripts**.
- **Action for Stage B:**
  - Move `tabpfn_concrete_master_copy.*` into `archive/duplicate_data/`.
  - Maintain `data/master_dataset.*` as the sole canonical source.

---

## 7. Obsolete Utility Scripts Handling

The four root scripts:
- `debug_r2.py`
- `fix_encoding.py`
- `generate_phase41_report.py`
- `verify_phase41.py`

- **Analysis:**
  - None of these are imported by any `src/` modules.
  - They were created as one-off diagnostic helpers during previous audits.
- **Action for Stage B:**
  - Safely relocate them into `archive/superseded_scripts/`.
  - Update any documentation links referencing them.

---

## 8. Model Artifact Handling

1. **Phase 4.1 Classical Models:**
   - No model binaries exist because `src/modeling/baseline_training.py` did not serialize them.
   - Authoritative artifacts (`results/predictions/phase4_1_cv_predictions.csv` and `reports/baseline_model_comparison.csv`) remain in place.
2. **Phase 4.2 & Phase 5 Pretrained TabPFN:**
   - `models/tabpfn-v2.5-regressor-v2.5_real.ckpt` remains in `models/`.
3. **Phase 4.3 Fine-Tuned TabPFN:**
   - Fold 4 and Fold 5 `.joblib` binaries remain in `results/models/tabpfn/fold_models_run2/`.
   - Folds 1–3 configurations and validation predictions remain alongside them.

---

## 9. Rollback Strategy

If any file move in Stage B results in a missing path error or test failure:
1. **Automated Manifest Check:** Compare before/after state against `reports/project_reorganization_move_map_run1.csv`.
2. **Deterministic Reversion:** Every move is a 1-to-1 path translation; reverse the move operation immediately using a pre-generated rollback script.
3. **Integrity Validation:** Re-compute SHA256 checksums on all reversed files to guarantee zero corruption.

---

## 10. Post-Migration Validation Plan (For Future Stage B)

Before declaring Stage B complete:
1. Run `python -c "import hashlib; assert hashlib.sha256(open('data/master_dataset.parquet','rb').read()).hexdigest() == '0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a'"`
2. Verify all 4 prediction CSV row counts ($67,200$, $5,600$, $2,800$, $800$).
3. Run `python -m py_compile src/modeling/*.py src/validation/*.py` to confirm all imports resolve.
4. Execute validation suite: `python src/validation/test_validation_v3_1.py`.
