# TabPFN Feature Engineering & Representation Report

**Dataset Version:** TabPFN Dedicated Concrete Dataset V1  
**Generated:** 2026-10-01  
**Target Variable:** `strength_mpa` (Strictly Unmodified)  
**Source Dataset:** [`data/tabpfn_concrete_v1/tabpfn_concrete_model.csv`](file:///g:/Projects/Concrete%20testing/data/tabpfn_concrete_v1/tabpfn_concrete_model.csv)  

---

## 1. Feature Engineering Rationale & Philosophy

The objective of feature engineering for TabPFN is **not** to synthesize artificial variance or generate synthetic specimens. Rather, it is to provide explicit mathematical representations of the underlying non-linear physics and microbial chemical kinetics that govern concrete strength development over time.

While TabPFN's pretrained transformer attention mechanism excels at in-context tabular inference, providing explicit physical representations (such as logarithmic hydration curves and treatment interactions) reduces the epistemic burden on the in-context attention heads when predicting across distinct experimental cohorts.

---

## 2. Feature Configurations

### 2.1 Configuration `V1_RAW` (Canonical Representation)
Only the original, scientifically essential input features without manual expansion:

| Feature Name | Type | Physical / Domain Role |
| :--- | :--- | :--- |
| `curing_age_days` | Continuous | Elapsed hydration curing time in days (7, 14, 21, 28, 56, 90) |
| `concrete_type` | Categorical | Concrete mix designation ('Normal Concrete' vs 'Bacterial Concrete') |
| `bacterial_concentration_cells_ml` | Continuous | Microorganism cell dosage (0 for Control, 1,000,000 for Inoculated) |
| `mechanical_property` | Categorical | Target property type ('Compressive Strength', 'Flexural Strength', 'Split Tensile Strength') |

### 2.2 Configuration `V2_ENGINEERED` (Physical Non-Linear Representation)
Includes the original features plus selected deterministic non-linear transformations:

| Feature Name | Category | Mathematical Formulation | Physical Significance |
| :--- | :--- | :--- | :--- |
| `curing_age_days` | Original | $t$ | Base elapsed hydration duration |
| `concrete_type` | Original | Categorical string | Mix identification |
| `bacterial_concentration_cells_ml` | Original | Dosage | Cellular concentration |
| `mechanical_property` | Original | Categorical string | Task identification (Unified modeling) |
| `age_log` | Derived | $\ln(1 + t)$ | Classic logarithmic hydration kinetics of Portland cement |
| `age_sqrt` | Derived | $\sqrt{t}$ | Parabolic diffusion-limited hydration and carbonation kinetics |
| `bacterial_present` | Derived | $\mathbb{I}(\text{Bacterial Concrete})$ | Clean binary biological treatment indicator ($0$ or $1$) |
| `bacterial_concentration_log` | Derived | $\ln(1 + \text{cells/mL})$ | Logarithmic biological dosage scale ($0$ vs $13.8155$) |
| `age_x_bacterial` | Derived | $t \times \text{bacterial\_present}$ | Time-dependent progressive microbial $\text{CaCO}_3$ precipitation gain |
| `age_log_x_bacterial` | Derived | $\ln(1 + t) \times \text{bacterial\_present}$ | Log-time bio-mineralization interaction term |
| `age_squared` | Derived | $t^2$ | Controlled second-order curvature for maturity modeling |

*(Note: `age_cubed` was computed in the dataset for ablation analysis, but excluded from `V2_ENGINEERED` to prevent polynomial runaway at the 90-day boundary).*

---

## 3. Strict Integrity & Traceability Rules
1. **Target Immutability**: `strength_mpa` is identical to the canonical master dataset across all 3,600 rows. Zero rows were altered, smoothed, or imputed.
2. **Metadata Isolation**: Specimen IDs (`sample_id`), cohort IDs (`experiment_id`), replicate counters (`sample_replicate`), and Excel source indices are retained in the dataset purely for evaluation routing and **are strictly blacklisted from model inputs**.
3. **Canonical Source Preservation**: `data/master_dataset.csv` and `data/master_dataset.parquet` remain untouched.
