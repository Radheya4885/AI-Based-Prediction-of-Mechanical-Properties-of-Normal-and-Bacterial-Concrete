# Phase 4 Final Model & Scientific Audit Report (Run 2)

**Project:** AI-Based Prediction of Mechanical Properties of Normal and Bacterial Concrete  
**Audit Date:** 2026-10-07  
**Auditor:** Pair Programming Assistant (Antigravity)  
**Audit Scope:** Verification of Phase 4.1 (Classical Baselines), Phase 4.2 (Pretrained TabPFN), and Phase 4.3 (Fine-Tuned TabPFN) Artifacts, Data Integrity, Fold Geometry, and Scientific Reproducibility prior to Phase 5.

---

## 1. Phase 4.1 Artifact Verification (Classical Baselines)

All 6 baseline model families from Phase 4.1 were audited across their 24 experimental configurations (6 models × 2 feature sets × 2 target strategies):

| Model | Feature Set | Strategy | Prediction Artifact | Metrics Artifact | Training Config & Script | Target / Property Handling | Validation Protocol | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Dummy** | Core / Expanded | Separate & Unified | `results/predictions/phase4_1_cv_predictions.csv` | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` (`strategy="mean"`) | Mean target baseline per fold / property | 5-Fold GroupKFold (`experiment_id`) | **VERIFIED** |
| **Linear** | Core / Expanded | Separate & Unified | `results/predictions/phase4_1_cv_predictions.csv` | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` (`LinearRegression()`) | OLS fitting on standardized numericals + one-hot cats | 5-Fold GroupKFold (`experiment_id`) | **VERIFIED** |
| **Ridge** | Core / Expanded | Separate & Unified | `results/predictions/phase4_1_cv_predictions.csv` | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` (`Ridge(alpha=1.0, seed=42)`) | L2 regularized regression | 5-Fold GroupKFold (`experiment_id`) | **VERIFIED** |
| **RandomForest** | Core / Expanded | Separate & Unified | `results/predictions/phase4_1_cv_predictions.csv` | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` (`n_estimators=200, seed=42`) | Ensemble tree regressor with bagging | 5-Fold GroupKFold (`experiment_id`) | **VERIFIED** |
| **XGBoost** | Core / Expanded | Separate & Unified | `results/predictions/phase4_1_cv_predictions.csv` | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` (`n_estimators=300, lr=0.05, max_depth=6, seed=42`) | Gradient boosted decision trees | 5-Fold GroupKFold (`experiment_id`) | **VERIFIED** |
| **CatBoost** | Core / Expanded | Separate & Unified | `results/predictions/phase4_1_cv_predictions.csv` | `reports/baseline_model_comparison.csv` | `src/modeling/baseline_training.py` (`iterations=300, lr=0.05, depth=6, seed=42`) | Symmetric trees with native categorical handling | 5-Fold GroupKFold (`experiment_id`) | **VERIFIED** |

- **Execution Metadata:** Recorded in `results/phase4_1_run_metadata.json` (Timestamp: 2026-09-26T23:57:09Z).
- **CV Prediction Count:** Exactly **67,200 rows** (24 configurations × 2,800 development specimens).
- **Target Immutability:** Ground-truth master hashes verified before/after execution (`csv_sha256`: `0d18...000a`, `parq_sha256`: `0909...678a`).

---

## 2. Phase 4.2 Artifact Verification (Pretrained TabPFN v2.5)

- **Model Checkpoint:** `models/tabpfn-v2.5-regressor-v2.5_real.ckpt` (Exactly 40,831,868 bytes, SHA256: `8ab42d2d0abe3886a2c54a336b2abf251658a23a9b3f6559c4fdf0c024f48e63`).
- **Feature Sets Evaluated:** `V1_RAW` (4 features) and `V2_ENGINEERED` (11 features).
- **Prediction Artifact:** `results/predictions/tabpfn_cv_predictions.csv` (Exactly 5,600 rows: 2 feature sets × 2,800 development specimens).
- **Metadata Artifact:** `results/models/tabpfn/pretrained_baseline_metadata.json` and `results/models/tabpfn/tabpfn_run_metadata.json`.
- **Reproduced Metrics (`V2_ENGINEERED`, Unified):**
  - Compressive MAE: **1.6168 MPa** (RMSE: 2.0385, R²: 0.8606)
  - Flexural MAE: **1.8116 MPa** (RMSE: 3.4360, R²: -18.5785)
  - Split Tensile MAE: **0.4603 MPa** (RMSE: 0.7331, R²: -0.9444)
  - Overall MAE: **1.2664 MPa** (RMSE: 2.3070, R²: 0.9688)
- **Status:** **VERIFIED** (100% bit-for-bit reproduction from saved predictions).

---

## 3. Phase 4.3 Artifact Verification (Fine-Tuned TabPFN v2.5)

- **Fold Models & Artifacts:**
  - `results/models/tabpfn/fold_models_run2/fold_1/`: `fold_1_config.json`, `fold_1_val_predictions.csv` (600 rows). Model binary omitted in transfer, OOF predictions preserved per policy.
  - `results/models/tabpfn/fold_models_run2/fold_2/`: `fold_2_config.json`, `fold_2_val_predictions.csv` (600 rows). Preserved per policy.
  - `results/models/tabpfn/fold_models_run2/fold_3/`: `fold_3_config.json`, `fold_3_val_predictions.csv` (600 rows). Preserved per policy.
  - `results/models/tabpfn/fold_models_run2/fold_4/`: `fold_4_finetuned.joblib` (84,093,937 bytes), `fold_4_config.json`, `fold_4_val_predictions.csv` (500 rows).
  - `results/models/tabpfn/fold_models_run2/fold_5/`: `fold_5_finetuned.joblib` (84,093,633 bytes), `fold_5_config.json`, `fold_5_val_predictions.csv` (500 rows).
- **OOF Prediction Artifact:** `results/predictions/tabpfn_finetuned_oof_predictions_run2.csv` (Exactly 2,800 rows).
- **Metadata Artifact:** `results/tabpfn_finetuned_run2/metadata.json` (Timestamp: 2026-10-07T22:40:41Z).
- **Reproduced Metrics (`V2_ENGINEERED`, Unified):**
  - Compressive MAE: **3.4449 MPa** (RMSE: 5.1574, R²: 0.1080)
  - Flexural MAE: **2.1343 MPa** (RMSE: 4.8612, R²: -38.1896)
  - Split Tensile MAE: **1.2987 MPa** (RMSE: 2.9499, R²: -30.4857)
  - Overall MAE: **2.2571 MPa** (RMSE: 4.3879, R²: 0.8872)
- **Publication Figures:** 8/8 figures present in `reports/figures/tabpfn_finetuned_run2/`.
- **Status:** **VERIFIED**.

---

## 4. Fold-Size and Cohort Geometry Verification

### Investigation of Fold Sizes (600, 600, 600, 500, 500)
- **Data Reality:**
  - Total Master Dataset: 3,600 specimens across 36 cohorts (exactly 100 specimens per cohort).
  - Development Partition: Exactly **2,800 specimens** across **28 cohorts**.
  - Final Test Partition: Exactly **800 specimens** across **8 cohorts**.
- **GroupKFold Mathematical Constraints:**
  - The cross-validation protocol mandates strict grouping by `experiment_id` to prevent specimen leakage across curing age / concrete type / bacterial concentration replicates.
  - Partitioning 28 cohorts into 5 non-overlapping folds:
    $$\frac{28}{5} = 5 \text{ cohorts remainder } 3$$
  - Thus, exactly 3 folds receive 6 cohorts ($3 \times 600 = 1,800$ rows) and 2 folds receive 5 cohorts ($2 \times 500 = 1,000$ rows).
  - Total rows: $1,800 + 1,000 = \mathbf{2,800 \text{ rows}}$.
- **Findings:**
  - Fold 1: 6 cohorts (600 specimens)
  - Fold 2: 6 cohorts (600 specimens)
  - Fold 3: 6 cohorts (600 specimens)
  - Fold 4: 5 cohorts (500 specimens)
  - Fold 5: 5 cohorts (500 specimens)
- **Traceability:**
  - This 6/6/6/5/5 cohort distribution is identical to the canonical assignments defined in `data/splits/grouped_cv_assignments.csv` since Phase 3.1.
  - Folds 1–3 and Folds 4–5 follow the exact same methodology, assignment file, and sample IDs.
- **Status:** **100% VALID & ARCHITECTURALLY INTENDED**.

---

## 5. Out-of-Fold (OOF) Integrity Verification

Audit of `results/predictions/tabpfn_finetuned_oof_predictions_run2.csv`:
- **Total Row Count:** Exactly 2,800.
- **Unique `sample_id` Count:** Exactly 2,800.
- **Duplicate Specimens:** **0**.
- **Missing Development Specimens:** **0** (100% bijective match with `data/splits/development_rows.csv`).
- **Final Test Contamination:** **0** (Set intersection with `final_test_rows.csv` has length 0).
- **Target Value Fidelity:** `strength_mpa_actual` matches original laboratory readings bit-for-bit with max difference = 0.0000.
- **Status:** **PASSED ALL INTEGRITY ASSERTIONS**.

---

## 6. Final Test Set Verification

Audit of `data/splits/final_test_rows.csv`:
- **Row Count:** Exactly 800 rows across 8 cohorts:
  `['EXP_CS_BC_07d', 'EXP_CS_BC_28d', 'EXP_CS_NC_07d', 'EXP_FS_BC_07d', 'EXP_FS_BC_90d', 'EXP_FS_NC_56d', 'EXP_TS_BC_90d', 'EXP_TS_NC_21d']`
- **Zero Usage Verification:**
  - Tested set intersection against `phase4_1_cv_predictions.csv`: **0 overlapping IDs**.
  - Tested set intersection against `tabpfn_cv_predictions.csv`: **0 overlapping IDs**.
  - Tested set intersection against `tabpfn_finetuned_oof_predictions_run2.csv`: **0 overlapping IDs**.
- **Status:** **PERMANENTLY SEALED & COMPLETELY UNTOUCHED**.

---

## 7. Model Comparison Verification

Direct metric reproduction from saved prediction files:

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

- **Random Forest (Core, Separate):** Confirmed Compressive MAE = 3.3204, Flexural MAE = 0.4554, Tensile MAE = 0.3311, Overall MAE = 1.3319.
- **XGBoost (Expanded, Unified):** Lowest overall MAE across all 24 classical configurations. Compressive MAE = 2.5192, Flexural MAE = 0.4689, Tensile MAE = 0.3148, Overall MAE = 1.0729.
- **XGBoost (Core, Unified):** Lowest compressive MAE among all 24 classical configurations. Compressive MAE = 2.3800, Flexural MAE = 0.6863, Tensile MAE = 0.7084, Overall MAE = 1.2386.
- **CatBoost (Core, Unified):** Confirmed Compressive MAE = 2.4369, Flexural MAE = 1.1307, Tensile MAE = 0.8584, Overall MAE = 1.4533.

---

## 8. Supported Scientific Conclusions

| Conclusion | Audit Finding | Supporting Evidence |
| :--- | :---: | :--- |
| **A. Fine-tuning TabPFN did NOT improve over pretrained TabPFN overall.** | **SUPPORTED** | Overall MAE worsened from 1.2664 MPa (pretrained) to 2.2571 MPa (fine-tuned); RMSE worsened from 2.3070 to 4.3879 MPa. |
| **B. Pretrained TabPFN remains superior overall to fine-tuned TabPFN.** | **SUPPORTED** | Pretrained outperforms fine-tuned across overall MAE, RMSE, and R², and on all 3 individual property targets. |
| **C. XGBoost has the best overall MAE among classical baselines.** | **SUPPORTED** | XGBoost (Expanded, Unified) achieved overall MAE of 1.0729 MPa, the lowest of all 24 evaluated classical configurations. |
| **D. Random Forest is best for flexural strength.** | **SUPPORTED** | Random Forest (Core, Separate) achieved flexural MAE of 0.4554 MPa (R² = 0.5340), outperforming XGBoost (0.4689 MPa) and TabPFN (1.8116 MPa). |
| **E. XGBoost is best for split tensile strength.** | **SUPPORTED** | XGBoost (Expanded, Unified) achieved tensile MAE of 0.3148 MPa (R² = 0.5373), the best in the benchmark. |
| **F. Pretrained TabPFN is best for compressive strength.** | **SUPPORTED** | Pretrained TabPFN v2.5 achieved compressive MAE of 1.6168 MPa (R² = 0.8606), beating the best classical compressive configuration (**XGBoost Core, Unified** at 2.3800 MPa) by **32.07% (32.1%)**, and beating overall-best classical **XGBoost (Expanded, Unified** at 2.5192 MPa) by **35.82% (35.8%)**. |
| **G. Fine-tuned TabPFN Folds 4 and 5 performed substantially better than Folds 1–3.** | **SUPPORTED** | Folds 4 and 5 achieved MAE of 0.4961 MPa and 0.9655 MPa (R² > 0.98), whereas Folds 1–3 had MAEs of 5.1636, 2.2350, and 1.9167 MPa. |

---

## 9. Discrepancies and Clarifications

1. **Fold Size Notation:**
   - Previous notes informally referred to "600 rows per fold", which was an approximation ($2800 / 5 = 560$).
   - The exact mathematical partition enforced by 28 discrete cohorts of 100 rows each is **600, 600, 600, 500, 500 rows**, which exactly matches `grouped_cv_assignments.csv`. Zero rows were lost or duplicated.
2. **Mechanistic Interpretation of TabPFN Fine-Tuning Performance:**
   - The earlier report stated that unified fine-tuning *caused* scale-regime distortion between properties.
   - Per audit correction: The experiment empirically evaluated only unified fine-tuning without a separate-property fine-tuning ablation. Therefore, the scientific statement is refined to:
     > *"The results are consistent with possible scale/regime effects in unified fine-tuning, but the present experiment does not independently establish causality."*
3. **Numerical Reconciliation of XGBoost Compressive MAE (2.3800 vs 2.5192 MPa):**
   - In the model comparison table, the XGBoost row originally listed **XGBoost (Expanded, Unified)** because it attained the lowest overall MAE across all 24 classical configurations (1.0729 MPa, with compressive MAE = 2.5192 MPa).
   - In Conclusion F and compressive benchmark comparisons, the cited classical champion is **XGBoost (Core, Unified)**, which is the single best classical model specifically for compressive strength (MAE = 2.3800 MPa, RMSE = 2.7913, R² = 0.7387).
   - Both figures are directly traced to `results/predictions/phase4_1_cv_predictions.csv` and reproduce with zero error. Pretrained TabPFN (MAE = 1.6168 MPa) improves over **both** XGBoost configurations:
     - Improvement over XGBoost (Core, Unified: 2.3800 MPa): **32.07%** (rounded to 32.1%).
     - Improvement over XGBoost (Expanded, Unified: 2.5192 MPa): **35.82%** (rounded to 35.8%).


---

## 10. Recommendation for Phase 5

- **Phase 4.1:** **VERIFIED** (All 24 configurations and 67,200 predictions intact and reproducible).
- **Phase 4.2:** **VERIFIED** (Pretrained TabPFN metrics and 5,600 predictions reproducible).
- **Phase 4.3:** **VERIFIED** (All 5 folds complete, 2,800 OOF predictions assembled, verified against ground truth).
- **Final Test Set:** **VERIFIED** (800 rows sealed, 0 contamination).
- **Data Files:** **VERIFIED** (Master CSV, Master Parquet, and Excel hashes identical).

### **Final Verdict:**
**ALL MODELING PHASES 4.1–4.3 VERIFIED. READY FOR PHASE 5.**
