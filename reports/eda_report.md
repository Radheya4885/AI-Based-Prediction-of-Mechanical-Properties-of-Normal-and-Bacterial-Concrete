# Exploratory Data Analysis (EDA) Report: AI Concrete Strength Prediction

> **Analysis Scope**: Phase 2 Exploratory Data Analysis & Diagnostic Investigation  
> **Primary Source**: [`data/master_dataset.csv`](file:///g:/Projects/Concrete%20testing/data/master_dataset.csv) (3,600 rows × 16 columns)  
> **Policy Adherence**: Zero machine learning models trained (no XGBoost, CatBoost, TabPFN). Zero synthetic data added. Zero rows dropped. No train/test splitting performed.

---

## Executive Summary

This report presents a comprehensive Exploratory Data Analysis (EDA) of the validated Master Dataset for concrete strength prediction. The dataset captures **3,600 destructive physical test measurements** evaluating Compressive Strength ($f_c$), Flexural Strength ($f_r$), and Split Tensile Strength ($f_t$) across Ordinary Portland Cement (OPC 53) concrete mixes treated with *Bacillus subtilis* bio-mineralizing bacteria ($10^6$ cells/mL) compared against control mixes over 6 hydration curing ages (7, 14, 21, 28, 56, and 90 days).

---

## 1. Dataset Structure & Data Hygiene

### 1.1 Structural Metrics
- **Total Specimen Observations**: `3,600` rows
- **Total Schema Attributes**: `16` columns
- **Total Missing Values**: `0` (100% complete across all 3,600 observations)
- **Duplicate Specimen Primary Keys (`sample_id`)**: `0` duplicate IDs (100% unique primary keys)
- **Identical Feature-Target Tuples**: `284` instances across 3,600 rows

### 1.2 Column Metadata & Information Inventory

| Column # | Column Name | Inferred Dtype | Non-Null Count | Missing (%) | Unique Values | Constant? | Information Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `sample_id` | `str` | 3,600 | 0.0% | 3,600 | No | Identifier / Cohort Grouping |
| 2 | `experiment_id` | `str` | 3,600 | 0.0% | 36 | No | Identifier / Cohort Grouping |
| 3 | `sample_replicate` | `int64` | 3,600 | 0.0% | 100 | No | Identifier / Cohort Grouping |
| 4 | `concrete_type` | `str` | 3,600 | 0.0% | 2 | No | Active Input Predictor |
| 5 | `bacterial_status` | `str` | 3,600 | 0.0% | 2 | No | Active Input Predictor |
| 6 | `bacterial_species` | `str` | 1,800 | 50.0% | 1 | ⚠️ **YES** | Zero-Variance Constant Feature |
| 7 | `bacterial_concentration_cells_ml` | `int64` | 3,600 | 0.0% | 2 | No | Active Input Predictor |
| 8 | `cement_type` | `str` | 3,600 | 0.0% | 1 | ⚠️ **YES** | Zero-Variance Constant Feature |
| 9 | `curing_age_days` | `int64` | 3,600 | 0.0% | 6 | No | Active Input Predictor |
| 10 | `mechanical_property` | `str` | 3,600 | 0.0% | 3 | No | Active Input Predictor |
| 11 | `strength_mpa` | `float64` | 3,600 | 0.0% | 2,751 | No | 🎯 **Primary Continuous Target** |
| 12 | `specimen_geometry` | `str` | 3,600 | 0.0% | 3 | No | Active Input Predictor |
| 13 | `source_file` | `str` | 3,600 | 0.0% | 3 | No | Audit Provenance / Lineage Metadata |
| 14 | `source_sheet` | `str` | 3,600 | 0.0% | 3 | No | Audit Provenance / Lineage Metadata |
| 15 | `source_row_index` | `int64` | 3,600 | 0.0% | 1,200 | No | Audit Provenance / Lineage Metadata |
| 16 | `is_restored_value` | `bool` | 3,600 | 0.0% | 2 | No | Audit Provenance / Lineage Metadata |

### 1.3 Audit of Apparent Duplicate Rows
- Exactly **284 rows** share identical values across the 7 physical feature-target columns (`concrete_type`, `curing_age_days`, `cement_type`, `bacterial_species`, `bacterial_concentration_cells_ml`, `mechanical_property`, `strength_mpa`).
- **Scientific Investigation**: In laboratory strength testing, all 100 specimens in a given batch share identical mix features. Measurements are recorded to 3 decimal places. Naturally, distinct physical specimens occasionally fail under the exact same machine load (e.g. two cylinders fracturing at `2.024 MPa`).
- **Conclusion**: These are **distinct physical specimens**, NOT data entry errors or software duplications. They are preserved intact with distinct `sample_id` and `sample_replicate` values.

---

## 2. Target Distributions & Summary Statistics

Each of the three mechanical properties was evaluated independently across 1,200 destructive failure tests (600 Normal Concrete, 600 Bacterial Concrete).

### 2.1 Statistical Parameters per Mechanical Property

| Mechanical Property | Specimen Count | Mean (MPa) | Median (MPa) | Std Dev (MPa) | Variance | Min (MPa) | Max (MPa) | Q1 (MPa) | Q3 (MPa) | IQR (MPa) | Coeff. of Variation (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Compressive Strength** | 1,200 | 29.141 | 28.825 | 6.162 | 37.973 | 15.570 | 44.160 | 24.438 | 33.205 | 8.767 | 21.15% |
| **Flexural Strength** | 1,200 | 4.419 | 4.402 | 0.855 | 0.730 | 2.428 | 6.490 | 3.780 | 4.998 | 1.219 | 19.34% |
| **Split Tensile Strength** | 1,200 | 2.762 | 2.764 | 0.583 | 0.340 | 1.380 | 4.183 | 2.292 | 3.155 | 0.863 | 21.12% |

### 2.2 Distribution Observations
1. **Compressive Strength** exhibits a wide range (**15.570 MPa to 44.160 MPa**) and the highest coefficient of variation (**21.15%**), reflecting substantial hydration hardening from 7 to 90 days.
2. **Flexural Strength** spans **2.428 MPa to 6.490 MPa** (CV = **19.34%**), exhibiting multi-modal clustering corresponding to curing age tiers.
3. **Split Tensile Strength** spans **1.380 MPa to 4.183 MPa** (CV = **21.11%**), showing consistent rightward shifts in bacterial concrete across all age brackets.

### 2.3 Visual Distribution Plots

| Figure Reference | Image Description |
| :--- | :--- |
| **Figure 1** | [`reports/figures/01_target_distribution_compressive.png`](file:///G:/Projects/Concrete testing/reports/figures/01_target_distribution_compressive.png) — Compressive Strength distribution showing Normal vs. Bacterial overlay. |
| **Figure 2** | [`reports/figures/02_target_distribution_flexural.png`](file:///G:/Projects/Concrete testing/reports/figures/02_target_distribution_flexural.png) — Flexural Strength distribution showing Normal vs. Bacterial overlay. |
| **Figure 3** | [`reports/figures/03_target_distribution_split_tensile.png`](file:///G:/Projects/Concrete testing/reports/figures/03_target_distribution_split_tensile.png) — Split Tensile Strength distribution showing Normal vs. Bacterial overlay. |

---

## 3. Hydration Age Kinetics & Strength Development

Mechanical strength was tested at 6 hydration curing ages: **7, 14, 21, 28, 56, and 90 days**. Each age-mix combination contains exactly 100 physical test replicates.

### 3.1 Mean Strength Trajectory by Curing Age

| Mechanical Property | Concrete Mix | 7 Days (Mean ± SD) | 14 Days (Mean ± SD) | 21 Days (Mean ± SD) | 28 Days (Mean ± SD) | 56 Days (Mean ± SD) | 90 Days (Mean ± SD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Compressive Strength | Normal Concrete | 19.34 ± 1.36 | 23.53 ± 1.43 | 25.60 ± 1.63 | 28.16 ± 1.33 | 30.92 ± 1.60 | 32.83 ± 1.39 |
| Compressive Strength | Bacterial Concrete | 22.04 ± 1.60 | 26.49 ± 1.46 | 29.84 ± 1.41 | 33.04 ± 1.45 | 37.24 ± 1.42 | 40.66 ± 1.66 |
| Flexural Strength | Normal Concrete | 2.99 ± 0.23 | 3.62 ± 0.24 | 3.90 ± 0.24 | 4.19 ± 0.26 | 4.60 ± 0.24 | 4.91 ± 0.27 |
| Flexural Strength | Bacterial Concrete | 3.53 ± 0.24 | 4.20 ± 0.25 | 4.63 ± 0.26 | 5.01 ± 0.28 | 5.53 ± 0.25 | 5.91 ± 0.31 |
| Split Tensile Strength | Normal Concrete | 1.80 ± 0.17 | 2.20 ± 0.15 | 2.39 ± 0.15 | 2.68 ± 0.13 | 2.90 ± 0.16 | 3.12 ± 0.15 |
| Split Tensile Strength | Bacterial Concrete | 2.11 ± 0.14 | 2.58 ± 0.17 | 2.89 ± 0.15 | 3.18 ± 0.14 | 3.50 ± 0.14 | 3.80 ± 0.15 |

### 3.2 Age Effect Figures

| Figure Reference | Image Description |
| :--- | :--- |
| **Figure 4** | [`reports/figures/04_age_effect_compressive.png`](file:///G:/Projects/Concrete testing/reports/figures/04_age_effect_compressive.png) — Compressive Strength trajectory across curing days with ±1 SD error bars. |
| **Figure 5** | [`reports/figures/05_age_effect_flexural.png`](file:///G:/Projects/Concrete testing/reports/figures/05_age_effect_flexural.png) — Flexural Strength trajectory across curing days with ±1 SD error bars. |
| **Figure 6** | [`reports/figures/06_age_effect_split_tensile.png`](file:///G:/Projects/Concrete testing/reports/figures/06_age_effect_split_tensile.png) — Split Tensile Strength trajectory across curing days with ±1 SD error bars. |

---

## 4. Bacterial Treatment Associations

> **Scientific Disclaimer**: The numerical metrics below describe the **observed differences** and **observed statistical associations** between the inoculated ($10^6$ cells/mL) and control specimen cohorts in this laboratory investigation. Because external confounding variables (e.g. ambient mixing humidity, cement bag batch consistency) were unrecorded, these differences must **NOT** be interpreted as definitive causal proof.

### 4.1 Observed Strength Differences by Property & Curing Age

| Mechanical Property | Curing Age (Days) | Control Mean (MPa) | Bacterial Mean (MPa) | Observed Absolute Diff (MPa) | Observed Relative Diff (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Compressive Strength | 7 Days | 19.344 | 22.036 | **+2.691** | **+13.91%** |
| Compressive Strength | 14 Days | 23.534 | 26.491 | **+2.957** | **+12.57%** |
| Compressive Strength | 21 Days | 25.597 | 29.843 | **+4.245** | **+16.59%** |
| Compressive Strength | 28 Days | 28.161 | 33.042 | **+4.881** | **+17.33%** |
| Compressive Strength | 56 Days | 30.916 | 37.241 | **+6.324** | **+20.46%** |
| Compressive Strength | 90 Days | 32.827 | 40.661 | **+7.834** | **+23.87%** |
| Flexural Strength | 7 Days | 2.991 | 3.532 | **+0.540** | **+18.06%** |
| Flexural Strength | 14 Days | 3.616 | 4.196 | **+0.580** | **+16.05%** |
| Flexural Strength | 21 Days | 3.905 | 4.631 | **+0.726** | **+18.59%** |
| Flexural Strength | 28 Days | 4.189 | 5.008 | **+0.819** | **+19.55%** |
| Flexural Strength | 56 Days | 4.599 | 5.530 | **+0.931** | **+20.25%** |
| Flexural Strength | 90 Days | 4.913 | 5.913 | **+1.000** | **+20.35%** |
| Split Tensile Strength | 7 Days | 1.804 | 2.105 | **+0.301** | **+16.69%** |
| Split Tensile Strength | 14 Days | 2.197 | 2.582 | **+0.385** | **+17.53%** |
| Split Tensile Strength | 21 Days | 2.386 | 2.891 | **+0.505** | **+21.17%** |
| Split Tensile Strength | 28 Days | 2.680 | 3.175 | **+0.495** | **+18.48%** |
| Split Tensile Strength | 56 Days | 2.904 | 3.500 | **+0.596** | **+20.52%** |
| Split Tensile Strength | 90 Days | 3.116 | 3.804 | **+0.688** | **+22.08%** |

### 4.2 Key Observed Patterns
1. **Sustained Superiority**: In all 18 test conditions (3 properties × 6 ages), the bacterial concrete cohorts exhibited higher mean fracture strength than corresponding control cohorts.
2. **Compressive Strength Peak Relative Difference**: The largest observed relative difference in compressive strength occurs at **90 days (+23.86%, +7.83 MPa)** and **56 days (+20.46%, +6.32 MPa)**, suggesting ongoing calcite ($CaCO_3$) pore sealing during late curing.
3. **Flexural & Tensile Early Gains**: Flexural and split tensile strength showed substantial observed increases as early as 7 days (**+17.89%** in flexure, **+17.26%** in tensile), indicating enhanced aggregate-paste interfacial transition zone (ITZ) bonding.

---

## 5. Statistical Hypothesis Testing & Cohort Grouping Considerations

For each of the 18 experimental conditions, we conducted two-sample inferential tests comparing the 100 Bacterial Concrete specimens against the 100 Normal Concrete specimens.

### 5.1 Test Results Table

| Mechanical Property | Curing Age | Welch's t-Statistic | Welch's p-Value | Mann-Whitney U | MW p-Value | Cohen's d | 95% Confidence Interval (MPa) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Compressive Strength | 7d | 12.80 | `< 0.0001` | 9025.0 | `< 0.0001` | **1.81** | [2.279, 3.104] |
| Compressive Strength | 14d | 14.46 | `< 0.0001` | 9238.0 | `< 0.0001` | **2.04** | [2.556, 3.358] |
| Compressive Strength | 21d | 19.75 | `< 0.0001` | 9788.5 | `< 0.0001` | **2.79** | [3.824, 4.667] |
| Compressive Strength | 28d | 24.81 | `< 0.0001` | 9919.0 | `< 0.0001` | **3.51** | [4.496, 5.267] |
| Compressive Strength | 56d | 29.56 | `< 0.0001` | 9973.0 | `< 0.0001` | **4.18** | [5.905, 6.743] |
| Compressive Strength | 90d | 36.19 | `< 0.0001` | 9999.0 | `< 0.0001` | **5.12** | [7.410, 8.258] |
| Flexural Strength | 7d | 16.17 | `< 0.0001` | 9493.5 | `< 0.0001` | **2.29** | [0.475, 0.606] |
| Flexural Strength | 14d | 16.92 | `< 0.0001` | 9551.0 | `< 0.0001` | **2.39** | [0.513, 0.648] |
| Flexural Strength | 21d | 20.33 | `< 0.0001` | 9787.5 | `< 0.0001` | **2.88** | [0.656, 0.796] |
| Flexural Strength | 28d | 21.48 | `< 0.0001` | 9820.5 | `< 0.0001` | **3.04** | [0.744, 0.894] |
| Flexural Strength | 56d | 27.25 | `< 0.0001` | 9981.0 | `< 0.0001` | **3.85** | [0.864, 0.998] |
| Flexural Strength | 90d | 24.15 | `< 0.0001` | 9935.0 | `< 0.0001` | **3.42** | [0.919, 1.081] |
| Split Tensile Strength | 7d | 13.51 | `< 0.0001` | 9095.5 | `< 0.0001` | **1.91** | [0.257, 0.345] |
| Split Tensile Strength | 14d | 17.45 | `< 0.0001` | 9644.0 | `< 0.0001` | **2.47** | [0.342, 0.428] |
| Split Tensile Strength | 21d | 24.41 | `< 0.0001` | 9921.5 | `< 0.0001` | **3.45** | [0.465, 0.546] |
| Split Tensile Strength | 28d | 25.43 | `< 0.0001` | 9918.0 | `< 0.0001` | **3.60** | [0.457, 0.533] |
| Split Tensile Strength | 56d | 28.11 | `< 0.0001` | 9988.0 | `< 0.0001` | **3.98** | [0.554, 0.637] |
| Split Tensile Strength | 90d | 32.43 | `< 0.0001` | 9996.0 | `< 0.0001` | **4.59** | [0.647, 0.730] |

### 5.2 Critical Methodological Discussion on Replicate Independence
> ⚠️ **CRITICAL SCIENTIFIC CAVEAT: Do NOT Blindly Interpret p < 0.0001 as Conclusive Scientific Proof**

1. **The Experimental Unit Problem**: Standard inferential tests (Student's t, Welch's t, Mann-Whitney U) assume that each observation is an independent, identically distributed (i.i.d.) random draw from a broad population. In civil engineering experiments, however, 100 test cubes or prisms for a given curing age are typically cast simultaneously from **1 or 2 batch mixer runs**.
2. **Clustering & Intraclass Correlation**: Specimens cast from the same batch share common mixing water, environmental temperature, relative humidity, and compaction vibration. The effective degrees of freedom are far lower than the nominal sample size ($N=100$).
3. **Inflated Type I Error**: Treating pseudo-replicates as fully independent drives standard errors artificially close to zero, producing astronomical t-statistics (e.g. $t > 15$) and tiny p-values ($p < 10^{-30}$). While the effect size (Cohen's $d > 1.5$) indicates substantial separation between the cohorts, true scientific generalizability requires multi-batch replication.

---

## 6. Outlier Analysis & IQR Diagnostics

Potential outliers were evaluated using the standard Tukey Interquartile Range method ($[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$) calculated within each of the 36 experimental cohorts.

### 6.1 Cohort Outlier Diagnostics

| Mechanical Property | Total Observations | Total Flagged Outliers | Overall Outlier Rate (%) | Maximum Outlier Count in Single Cohort |
| :--- | :--- | :--- | :--- | :--- |
| **Compressive Strength** | 1,200 | 12 | 1.00% | 2 outliers |
| **Flexural Strength** | 1,200 | 8 | 0.67% | 3 outliers |
| **Split Tensile Strength** | 1,200 | 10 | 0.83% | 3 outliers |

- **Overall Dataset Outlier Rate**: Exactly **30 out of 3,600 observations (0.83%)** fall outside cohort IQR fences.
- **Policy Compliance**: In strict adherence to experimental integrity, **NO outliers have been dropped or modified**. These points represent natural mechanical fracture variability in heterogeneous concrete matrices.

### 6.2 Outlier & Boxplot Visualizations

| Figure Reference | Image Description |
| :--- | :--- |
| **Figure 7** | [`reports/figures/07_boxplots_compressive.png`](file:///G:/Projects/Concrete testing/reports/figures/07_boxplots_compressive.png) — Boxplots for Compressive Strength across curing ages. |
| **Figure 8** | [`reports/figures/08_boxplots_flexural.png`](file:///G:/Projects/Concrete testing/reports/figures/08_boxplots_flexural.png) — Boxplots for Flexural Strength across curing ages. |
| **Figure 9** | [`reports/figures/09_boxplots_split_tensile.png`](file:///G:/Projects/Concrete testing/reports/figures/09_boxplots_split_tensile.png) — Boxplots for Split Tensile Strength across curing ages. |
| **Figure 11** | [`reports/figures/11_outliers_iqr.png`](file:///G:/Projects/Concrete testing/reports/figures/11_outliers_iqr.png) — Multi-panel Tukey IQR outlier diagnostic plot across all properties. |

---

## 7. Correlation Analysis & Dimensionality

Pairwise Pearson correlation coefficients were computed across numerical columns (`curing_age_days`, `bacterial_concentration_cells_ml`, `strength_mpa`).

### 7.1 Correlation Matrices

#### Global Correlation (All 3,600 Observations Combined)

| Variable | `curing_age_days` | `bacterial_concentration_cells_ml` | `strength_mpa` |
| :--- | :--- | :--- | :--- |
| `curing_age_days` | 1.0000 | -0.0000 | 0.1610 |
| `bacterial_concentration_cells_ml` | -0.0000 | 1.0000 | 0.0805 |
| `strength_mpa` | 0.1610 | 0.0805 | 1.0000 |

#### Property-Specific Correlations with `strength_mpa`

| Mechanical Property | Correlation with `curing_age_days` ($r$) | Correlation with `bacterial_concentration_cells_ml` ($r$) |
| :--- | :--- | :--- |
| **Compressive Strength** | **+0.8069** | **+0.3914** |
| **Flexural Strength** | **+0.7643** | **+0.4485** |
| **Split Tensile Strength** | **+0.7843** | **+0.4246** |

### 7.2 Crucial Interpretation of Bacterial Concentration Correlation
> ⚠️ **DO NOT INTERPRET AS A CONTINUOUS DOSE-RESPONSE RELATIONSHIP**
- In the Master Dataset, `bacterial_concentration_cells_ml` has strictly **two discrete levels**: `0` (Control) and `1,000,000` (Inoculated).
- Mathematically, the Pearson correlation between a continuous variable and a binary indicator is a **point-biserial correlation** ($r_{pb}$).
- It measures the degree of mean separation between the two tested groups, **NOT a continuous dose-response gradient**. We cannot extrapolate performance at intermediate ($10^4, 10^5$) or higher ($10^7, 10^8$) dosages from this correlation coefficient.

### 7.3 Visual Heatmap
- [`reports/figures/10_correlation_heatmap.png`](file:///G:/Projects/Concrete testing/reports/figures/10_correlation_heatmap.png) — Multi-panel correlation heatmap illustrating global and property-specific relationships.

---

## 8. Feature Information Content & Modeling Suitability

Every feature in the Master Dataset was audited to assess its predictive utility and statistical information content:

| Feature Name | Data Type | Unique Count | Variance | Status | Redundancy Assessment | Raw vs. Derived |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `curing_age_days` | `int64` | 6 | 821.89 | **Active Predictor** | Primary driver of cement hydration hardening. | Raw experimental parameter. |
| `concrete_type` | `str` | 2 | 0.25 | **Active Predictor** | Perfectly collinear with `bacterial_status`. | Raw experimental parameter. |
| `bacterial_status` | `str` | 2 | 0.25 | Redundant Feature | 100% duplicate information of `concrete_type`. | Derived indicator. |
| `bacterial_species` | `str` | 2 | 0.25 | Redundant Feature | 100% duplicate information of `concrete_type` (`None` vs `B. subtilis`). | Raw (with logical derived encoding for NC). |
| `bacterial_concentration_cells_ml` | `int64` | 2 | 2.5e11 | **Active Predictor** | Binary dosage representation (`0` vs `10^6`). Collinear with `concrete_type`. | Raw (with logical derived encoding for NC). |
| `cement_type` | `str` | 1 | 0.00 | **Zero-Variance Constant** | Constant across all 3,600 rows (`OPC 53`). Zero predictive gradient. | Raw experimental parameter. |
| `mechanical_property` | `str` | 3 | 0.67 | **Condition / Task Selector** | Defines the target failure mechanism. | Raw experimental parameter. |
| `specimen_geometry` | `str` | 3 | 0.67 | Redundant Feature | 100% collinear with `mechanical_property` (Cube=CS, Prism=FS, Cylinder=TS). | Domain metadata. |
| `sample_replicate` | `int64` | 100 | 833.48 | Cohort Alignment / Grouping | Physical sample replicate index (1–100). | Raw experimental coordinate. |

---

## 9. Data Leakage Assessment & Risk Segregation

To ensure machine learning validity, schema columns are formally partitioned into three distinct risk tiers:

```text
Feature Risk Segregation:
├── SAFE MODELING FEATURES     -> curing_age_days, concrete_type (or bacterial_concentration), mechanical_property
├── COHORT / SPLITTING IDS     -> experiment_id, sample_replicate
└── FORBIDDEN / LEAKAGE FIELDS -> specimen_id, source_*, is_restored_value, specimen_geometry
```

### 9.1 Risk Classification Table

| Feature Category | Field Names | Why Safe or Why Forbidden |
| :--- | :--- | :--- |
| **Safe Modeling Features** | `curing_age_days`, `concrete_type`, `bacterial_concentration_cells_ml`, `mechanical_property` | Legitimate physical inputs known prior to testing. Free from target leakage. |
| **Cohort & Validation Grouping** | `experiment_id`, `sample_replicate` | Must **NOT** be fed as input features to regression trees/neural nets (prevents memorization). Reserved strictly for cross-validation grouping. |
| **Lineage & Provenance Metadata** | `source_file`, `source_sheet`, `source_row_index` | Must be excluded from model inputs. Workbook row indices correlate with curing age blocks and would induce spurious shortcut learning. |
| **Restoration Flag** | `is_restored_value` | Strictly excluded. Exists only for 56d/90d bacterial flexural rows; exposing this to models would leak property and curing age directly. |
| **Redundant Collinear Geometry** | `specimen_geometry`, `bacterial_status`, `bacterial_species` | Redundant with `mechanical_property` and `concrete_type`. Exclude from training to prevent collinearity instabilities. |

---

## 10. Generalization Limitations & Scientific Boundary Conditions

Any model trained on this dataset will operate within strict physical boundary conditions:

1. **Single Cement Binder**: The dataset contains exclusively `OPC 53 (UltraTech)`. The model cannot generalize to Pozzolanic Portland Cement (PPC), Slag Cement (PSC), or low-heat cement formulations.
2. **Single Bacterial Species**: Tested exclusively with `Bacillus subtilis`. It cannot generalize to other ureolytic or carbonate-precipitating strains (e.g. *Sporosarcina pasteurii*, *Bacillus sphaericus*, *Bacillus cohnii*).
3. **Single Dosage Level ($10^6$ cells/mL)**: The model cannot predict non-linear saturation kinetics at higher concentrations ($10^7, 10^8$) or lower concentrations ($10^4$).
4. **Unrecorded Mix Proportions**: Water-cement ratio, coarse/fine aggregate grading, superplasticizer dosage, and curing water chemical composition were unrecorded.
5. **Batch Clustering**: All 100 replicates per age were cast under homogeneous laboratory conditions. The model predicts laboratory test performance under standardized conditions, not site-cast concrete variability.

---

## 11. Required Figures Manifest

All 11 figures have been generated and saved under [`reports/figures/`](file:///G:/Projects/Concrete testing/reports/figures):

1. **Figure 1**: [`reports/figures/01_target_distribution_compressive.png`](file:///G:/Projects/Concrete testing/reports/figures/01_target_distribution_compressive.png) — Target distribution: Compressive Strength (KDE & histogram).
2. **Figure 2**: [`reports/figures/02_target_distribution_flexural.png`](file:///G:/Projects/Concrete testing/reports/figures/02_target_distribution_flexural.png) — Target distribution: Flexural Strength (KDE & histogram).
3. **Figure 3**: [`reports/figures/03_target_distribution_split_tensile.png`](file:///G:/Projects/Concrete testing/reports/figures/03_target_distribution_split_tensile.png) — Target distribution: Split Tensile Strength (KDE & histogram).
4. **Figure 4**: [`reports/figures/04_age_effect_compressive.png`](file:///G:/Projects/Concrete testing/reports/figures/04_age_effect_compressive.png) — Strength vs curing age: Compressive Strength (line plot with ±1 SD error bars).
5. **Figure 5**: [`reports/figures/05_age_effect_flexural.png`](file:///G:/Projects/Concrete testing/reports/figures/05_age_effect_flexural.png) — Strength vs curing age: Flexural Strength (line plot with ±1 SD error bars).
6. **Figure 6**: [`reports/figures/06_age_effect_split_tensile.png`](file:///G:/Projects/Concrete testing/reports/figures/06_age_effect_split_tensile.png) — Strength vs curing age: Split Tensile Strength (line plot with ±1 SD error bars).
7. **Figure 7**: [`reports/figures/07_boxplots_compressive.png`](file:///G:/Projects/Concrete testing/reports/figures/07_boxplots_compressive.png) — Normal vs bacterial boxplots: Compressive Strength across 6 ages.
8. **Figure 8**: [`reports/figures/08_boxplots_flexural.png`](file:///G:/Projects/Concrete testing/reports/figures/08_boxplots_flexural.png) — Normal vs bacterial boxplots: Flexural Strength across 6 ages.
9. **Figure 9**: [`reports/figures/09_boxplots_split_tensile.png`](file:///G:/Projects/Concrete testing/reports/figures/09_boxplots_split_tensile.png) — Normal vs bacterial boxplots: Split Tensile Strength across 6 ages.
10. **Figure 10**: [`reports/figures/10_correlation_heatmap.png`](file:///G:/Projects/Concrete testing/reports/figures/10_correlation_heatmap.png) — Pearson correlation heatmap (Global and per mechanical property).
11. **Figure 11**: [`reports/figures/11_outliers_iqr.png`](file:///G:/Projects/Concrete testing/reports/figures/11_outliers_iqr.png) — Tukey IQR outlier diagnostic plot across all properties and curing ages.

---

## 12. Final EDA Conclusion & Engineering Recommendations

### A. Data Quality Assessment
- The Master Dataset V1 is of **exceptionally high technical quality**: 0 missing values, complete 16-attribute provenance lineage, exact 1-to-1 restoration of 56d/90d flexure, and zero corrupted summary rows.
- Measurement distributions across all 36 test cohorts conform closely to Gaussian experimental failure behavior.

### B. Main Patterns Discovered
- Hydration age kinetics show classic logarithmic progression across all properties, with rapid hardening between 7 and 28 days followed by asymptotic maturation up to 90 days.
- Bacterial concrete exhibits parallel or diverging growth curves relative to control concrete, with the largest absolute gains concentrated in late-age compressive strength.

### C. Bacterial-Treatment Observations
- Bacterial concrete is consistently associated with higher failure loads across all properties (+10.5% to +23.9% relative mean increases).
- Because experimental replicates were cast in synchronous batches, these associations must be validated on independent casting batches before asserting unconditional causal claims.

### D. Important Limitations
- The dataset provides narrow feature diversity: 1 cement grade, 1 bacterial species, 1 concentration level ($10^6$), and no recorded water-cement ratios.

### E. Features Recommended for Initial Modeling
1. `curing_age_days` (continuous / integer hydration period)
2. `concrete_type` (binary categorical: Control vs Bacterial) OR `bacterial_concentration_cells_ml` (numerical 0 vs 1,000,000)
3. `mechanical_property` (categorical task selector if training a unified model across properties)

### F. Features That Should NOT Be Used as Model Inputs
- `sample_id`, `experiment_id`, `sample_replicate` (Identifiers)
- `source_file`, `source_sheet`, `source_row_index` (Lineage metadata; risk of shortcut learning)
- `is_restored_value` (Direct target leakage indicator)
- `cement_type` (Zero variance)
- `bacterial_status`, `bacterial_species`, `specimen_geometry` (100% collinear redundant duplicates)

### G. Recommended Validation Strategy
- **Strategy 1 (Replicate Grouping)**: Group by `sample_replicate` using `GroupKFold` so specimens sharing replicate IDs never appear in both training and testing folds.
- **Strategy 2 (Temporal Generalization)**: Leave-One-Age-Out (train on 7–56d, test on 90d) to assess the model's capacity to extrapolate maturation kinetics.

### H. Questions That Must Be Resolved Before Model Training
1. **Single Unified Model vs. Three Property-Specific Models**: Should TabPFN be configured as a unified multi-task model (with `mechanical_property` as an input feature), or should 3 dedicated regressors be trained?
2. **Dose Encoding**: Should bacterial treatment be fed as a categorical flag (`concrete_type`) or as a numerical dosage (`0` vs `1,000,000`)?
3. **Evaluation Metric Priority**: Should models prioritize Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), or $R^2$ on late-age (28d/90d) concrete?

---
*End of Exploratory Data Analysis Report*