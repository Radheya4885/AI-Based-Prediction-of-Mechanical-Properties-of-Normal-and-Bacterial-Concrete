# Stage A: Reorganization Risk Assessment & Mitigation Matrix (Run 1)

**Project:** AI-Based Prediction of Mechanical Properties of Normal and Bacterial Concrete  
**Date:** 2026-10-07  
**Scope:** Identification of potential failure modes during proposed repository reorganization and explicit mitigation strategies.

---

## 1. Risk Evaluation Summary

Reorganizing a completed data-science / machine-learning research repository carries significant risks of silent breakage, broken relative imports, unresolvable markdown links, and invalidating verified cryptographic hashes.

Below is the comprehensive risk register across 7 critical failure categories:

---

## 2. Risk Matrix

| Risk ID | Domain | Risk Description | Severity | Likelihood | Impact | Mitigation Strategy |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **R-01** | **Canonical Datasets** | Moving `data/master_dataset.*` or raw Excel sheets breaks hardcoded paths in 18 source scripts. | **CRITICAL** | HIGH | Complete pipeline failure | **STRICT NON-MOVE POLICY:** Keep master dataset and raw workbooks in current canonical location. |
| **R-02** | **Locked Test Set** | Moving `data/splits/final_test_rows.csv` invalidates immutability verification in Phase 5 scripts. | **CRITICAL** | HIGH | Phase 5 verification failure | **STRICT NON-MOVE POLICY:** `data/splits/*` remains completely untouched. |
| **R-03** | **Prediction Files** | Moving `results/predictions/*` breaks report generation, comparison tables, and diagnostic figure scripts. | **HIGH** | MEDIUM | Broken figure generation | **KEEP IN PLACE:** Maintain predictions in `results/predictions/`. |
| **R-04** | **Python Module Imports** | Moving or nesting scripts inside `src/` breaks `from src.validation import ...` and `from src.modeling import ...`. | **HIGH** | HIGH | `ModuleNotFoundError` across repository | **KEEP `src/` TREE INTACT:** Maintain existing Python module hierarchy. |
| **R-05** | **Git LFS Pointers** | Misconfiguring `.gitattributes` when relocating `.ckpt` or `.joblib` files leads to raw binary commits or pointer decoupling. | **MEDIUM** | LOW | Repository bloat / LFS desync | **EXTENSION-BASED RULES:** `.gitattributes` tracks `*.ckpt` and `*.joblib` globally; no path-specific rules needed. |
| **R-06** | **Markdown Image Links** | Moving figures out of `reports/figures/` breaks relative image tags in 17 research reports. | **MEDIUM** | HIGH | Broken images in reports and thesis | **KEEP FIGURE PATHS INTACT:** Retain `reports/figures/` relative paths. |
| **R-07** | **Script Redirection** | Moving root scripts (`generate_phase41_report.py`, etc.) breaks user terminal workflows or documentation instructions. | **LOW** | LOW | Minor terminal inconvenience | **ARCHIVE STAGING:** Only move to `archive/superseded_scripts/` after creating redirection notes. |

---

## 3. High-Risk vs. Low-Risk Operations

### High-Risk Operations (DISCOURAGED / PROHIBITED IN STAGE B):
1. **DO NOT move `data/master_dataset.csv` or `data/master_dataset.parquet`.**
2. **DO NOT move any files in `data/splits/`.**
3. **DO NOT move `results/predictions/*.csv`.**
4. **DO NOT move `models/tabpfn-v2.5-regressor-v2.5_real.ckpt`.**
5. **DO NOT restructure active Python packages in `src/modeling/` or `src/validation/`.**

### Low-Risk Operations (PERMISSIBLE IN STAGE B WITH CARE):
1. Moving unreferenced duplicate master copies (`data/tabpfn_concrete_v1/tabpfn_concrete_master_copy.*`) to `archive/duplicate_data/`.
2. Moving 4 one-off root diagnostic scripts (`debug_r2.py`, `fix_encoding.py`, `generate_phase41_report.py`, `verify_phase41.py`) to `archive/superseded_scripts/`.
3. Cleaning compiled Python bytecode caches (`__pycache__`).

---

## 4. Stage B Execution Safeguards

Before any physical file movement is executed in Stage B:
1. **Pre-Move Checkpoint:** Record exact filesystem state and SHA256 hashes of all 142 files.
2. **Dry-Run Validation:** Test path resolution in a temporary scratch script before moving.
3. **Single-Operation Verification:** Move one group at a time (e.g. archive root scripts first), verify syntax and execution, then proceed.
4. **Final Hash Matching:** Confirm that master datasets, splits, predictions, and model weights have zero byte-level discrepancies post-move.
