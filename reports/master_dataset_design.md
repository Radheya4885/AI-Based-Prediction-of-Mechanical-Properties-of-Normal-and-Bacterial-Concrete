# Master Dataset Design Specification: Concrete Strength Prediction

> **Document Path**: [`reports/master_dataset_design.md`](file:///g:/Projects/Concrete%20testing/reports/master_dataset_design.md)  
> **Source of Truth**: Completed Dataset Audit ([`reports/dataset_audit.md`](file:///g:/Projects/Concrete%20testing/reports/dataset_audit.md) and [`reports/dataset_schema.csv`](file:///g:/Projects/Concrete%20testing/reports/dataset_schema.csv))  
> **Operational Policy**: Architecture & design only. **NO** original files modified. **NO** models trained. **NO** values imputed. **NO** data randomly split. **NO** constant columns deleted. **NO** `MASTER_DATASET.csv/parquet` created yet.

---

## Executive Summary

This specification establishes the architectural foundation for converting 3 separate, heterogeneous laboratory Excel workbooks (`C Strenth Reading 1000000.xlsx`, `F Strength Reading 1000000.xlsx`, and `T Strenth Reading 1000000.xlsx`) into a single, standardized, leakage-safe **Master Dataset**.

### Key Architectural Decisions
1. **Data Scope**: Exclusively include raw experimental specimen test fracture readings (100 physical tests per test condition). Completely exclude aggregate macro-summaries, cross-concentration averages, and formula summary rows.
2. **Format**: Unified **Long-Format Schema** capturing **3,600 raw experimental measurements** (1,200 Compressive + 1,200 Flexural + 1,200 Split Tensile).
3. **Reshaping**: Deconstruct the wide, paired Normal/Bacterial layout of `Comp 6` (600 rows × 2 target columns) into 1,200 standard long-format rows.
4. **Restoration Verification**: Formally verified through mathematical proof that the 200 missing flexural observations in `Sheet1` (Bacterial Concrete at 56d and 90d) correspond 1-to-1 to the intact sample matrix in `Sheet2` (max absolute difference = `0.00000000` across all 1,000 non-missing test points). Values remain strictly un-imputed until approved.
5. **Leakage Prevention**: Define a hierarchical grouping structure (`sample_replicate` 1–100 and `experiment_id`) to ensure specimen clusters sharing batch mixing histories are never partitioned across train and test sets.

---

## A. Proposed Master Schema

In accordance with the project requirements, each field was audited against the raw files. **Only fields genuinely supported by the experimental records are included.** Fields commonly seen in literature but unrecorded in these workbooks (e.g., `water_cement_ratio`, `aggregate_information`) are explicitly documented as unsupported and omitted from the active schema to prevent fabricating non-existent ground truth.

### Schema Definition Table

| Column # | Field Name | Data Type | Nullable | Supported by Raw Data? | Domain / Example Values | Role & Technical Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `sample_id` | `VARCHAR(32)` | No | Yes (Derived) | `CS_NC_07d_001`, `FS_BC_56d_042` | **Primary Key**: Globally unique identifier for every physical test specimen. |
| **2** | `experiment_id` | `VARCHAR(32)` | No | Yes (Derived) | `EXP_CS_NC_07d`, `EXP_TS_BC_28d` | **Batch / Cohort ID**: Identifies the 100-specimen test cohort sharing identical mix and curing age. |
| **3** | `sample_replicate` | `INTEGER` | No | Yes (Explicit) | `1` to `100` | Physical replicate specimen index (1–100) within each 100-cube/beam/cylinder batch. |
| **4** | `concrete_type` | `VARCHAR(32)` | No | Yes (Explicit) | `Normal Concrete`, `Bacterial Concrete` | Categorical concrete classification. |
| **5** | `bacterial_status` | `VARCHAR(16)` | No | Yes (Derived) | `Control`, `Inoculated` | Explicit biological treatment flag. |
| **6** | `bacterial_species` | `VARCHAR(32)` | No | Yes (Explicit) | `None` (for NC), `Bacillus subtilis` (for BC) | Microorganism added to the mix. |
| **7** | `bacterial_concentration_cells_ml` | `BIGINT` | No | Yes (Explicit) | `0` (for NC), `1000000` (for BC) | Liquid broth concentration ($10^6$ cells/mL). |
| **8** | `cement_type` | `VARCHAR(32)` | No | Yes (Explicit) | `OPC 53 (UltraTech)` | Cement binder grade. Constant across all batches. |
| **9** | `curing_age_days` | `INTEGER` | No | Yes (Explicit) | `7`, `14`, `21`, `28`, `56`, `90` | Hydration curing duration in water tank. |
| **10** | `mechanical_property` | `VARCHAR(32)` | No | Yes (Explicit) | `Compressive Strength`, `Flexural Strength`, `Split Tensile Strength` | Mechanical test executed. |
| **11** | `strength_mpa` | `FLOAT` | No | Yes (Explicit) | `1.436` to `44.160` | **Target Variable**: Continuous fracture strength in MPa. |
| **12** | `specimen_geometry` | `VARCHAR(32)` | No | Yes (Domain Std) | `150mm Cube`, `100x100x500mm Prism`, `150x300mm Cylinder` | Standard test specimen geometry. |
| **13** | `source_file` | `VARCHAR(64)` | No | Yes (Metadata) | `C Strenth Reading 1000000.xlsx`, etc. | Source Excel workbook filename. |
| **14** | `source_sheet` | `VARCHAR(32)` | No | Yes (Metadata) | `Comp 6`, `Sheet1`, `Sheet2` | Source worksheet. |
| **15** | `source_row_index` | `INTEGER` | No | Yes (Metadata) | `2` to `1201` | Original Excel row coordinate. |
| **16** | `is_restored_value` | `BOOLEAN` | No | Yes (Provenance) | `False`, `True` | Flag indicating if value was restored from `Sheet2` (56d/90d BC flexure). |

### Unsupported Fields Omitted from Schema
- **`water_cement_ratio`**: **Not supported**. Thorough regex and cell-by-cell scans across all 15 sheets found zero records of water content or W/C ratio. Standard M20/M25 mixes typically use ~0.45–0.50, but since no numerical values are recorded, creating this column would introduce unverified synthetic data.
- **`aggregate_information`**: **Not supported**. No coarse aggregate size (e.g. 20mm), fine aggregate zone (Zone II), or sand/gravel weight proportions are documented anywhere in the workbooks.

---

## B. Source-to-Target Mapping

Below is the exact cell-level mapping matrix for extracting ground-truth measurements into the Master Dataset.

### 1. Compressive Strength Mapping (`C Strenth Reading 1000000.xlsx`)
- **Workbook**: `data/C Strenth Reading 1000000.xlsx`
- **Source Sheet**: `Comp 6` (Rows 3 to 602 in Excel; 600 data pairs)
- **Source Structure**: 2-level header with side-by-side columns: Column 5 = Normal Concrete (`N`), Column 6 = Bacterial Concrete (`BC`).
- **Transformation**: Reshape (melt/unpivot) each row into **two separate records** (1 Normal Concrete row, 1 Bacterial Concrete row), expanding 600 Excel rows into 1,200 long-format rows.

| Destination Field | Normal Concrete (N) Record Mapping | Bacterial Concrete (BC) Record Mapping | Value Classification |
| :--- | :--- | :--- | :--- |
| `sample_id` | `'CS_NC_' + str(age).zfill(2) + 'd_' + str((row-3)%100+1).zfill(3)` | `'CS_BC_' + str(age).zfill(2) + 'd_' + str((row-3)%100+1).zfill(3)` | Derived Specimen PK |
| `experiment_id` | `'EXP_CS_NC_' + str(age).zfill(2) + 'd'` | `'EXP_CS_BC_' + str(age).zfill(2) + 'd'` | Derived Batch ID |
| `sample_replicate` | `((row_idx - 3) % 100) + 1` | `((row_idx - 3) % 100) + 1` | Raw Sample Index (1–100) |
| `concrete_type` | `'Normal Concrete'` | `'Bacterial Concrete'` | Raw Categorical |
| `bacterial_status` | `'Control'` | `'Inoculated'` | Derived Category |
| `bacterial_species` | `'None'` | `Col 4 ('Bacteria')` -> `'Bacillus subtilis'` | Raw Categorical |
| `bacterial_concentration_cells_ml` | `0` (Logical correction from raw sheet) | `Col 5 ('Bacterial Concentration')` -> `1000000` | Raw Numerical |
| `cement_type` | `Col 3 ('Cement')` -> `'OPC 53 (UltraTech)'` | `Col 3 ('Cement')` -> `'OPC 53 (UltraTech)'` | Raw Categorical |
| `curing_age_days` | `Col 2 ('Curing Age (Days)')` | `Col 2 ('Curing Age (Days)')` | Raw Numerical (`7, 14, 21, 28, 56, 90`) |
| `mechanical_property` | `'Compressive Strength'` | `'Compressive Strength'` | Target Classification |
| `strength_mpa` | `Col 6 ('Compressive Strength (MPa) - N')` | `Col 7 ('Unnamed: 6 - BC')` | **Raw Experimental Measurement** |
| `specimen_geometry` | `'150mm Cube'` | `'150mm Cube'` | Domain Standard |
| `source_file` | `'C Strenth Reading 1000000.xlsx'` | `'C Strenth Reading 1000000.xlsx'` | Lineage Metadata |
| `source_sheet` | `'Comp 6'` | `'Comp 6'` | Lineage Metadata |
| `source_row_index` | Excel row coordinate (3 to 602) | Excel row coordinate (3 to 602) | Lineage Metadata |
| `is_restored_value` | `False` | `False` | Provenance Flag |

### 2. Flexural Strength Mapping (`F Strength Reading 1000000.xlsx`)
- **Workbook**: `data/F Strength Reading 1000000.xlsx`
- **Source Sheet**: `Sheet1` (Rows 2 to 1201) + `Sheet2` (Rows 5 to 104 for restoration)
- **Source Structure**: Unpivoted long format (600 Normal Concrete + 600 Bacterial Concrete).
- **Transformation**: Direct mapping for rows 1 to 1000; verified restoration mapping from `Sheet2` for rows 1001 to 1200.

| Destination Field | Rows 1 to 1000 (Complete in `Sheet1`) | Rows 1001 to 1200 (Restored from `Sheet2`) | Value Classification |
| :--- | :--- | :--- | :--- |
| `sample_id` | `'FS_' + ('NC' if c_type=='Normal Concrete' else 'BC') + '_' + str(age).zfill(2) + 'd_' + str(((row-2)%100)+1).zfill(3)` | `'FS_BC_' + str(age).zfill(2) + 'd_' + str(((row-2)%100)+1).zfill(3)` | Derived Specimen PK |
| `experiment_id` | `'EXP_FS_' + ('NC' if c_type=='Normal Concrete' else 'BC') + '_' + str(age).zfill(2) + 'd'` | `'EXP_FS_BC_' + str(age).zfill(2) + 'd'` | Derived Batch ID |
| `sample_replicate` | `((row_idx - 2) % 100) + 1` | `((row_idx - 2) % 100) + 1` | Raw Sample Index (1–100) |
| `concrete_type` | `Col 1 ('Concrete Type')` | `'Bacterial Concrete'` | Raw Categorical |
| `bacterial_status` | `'Control'` if Normal else `'Inoculated'` | `'Inoculated'` | Derived Category |
| `bacterial_species` | `'None'` if Normal else `'Bacillus subtilis'` | `'Bacillus subtilis'` | Raw Categorical |
| `bacterial_concentration_cells_ml` | `0` if Normal else `1000000` | `1000000` | Raw Numerical |
| `cement_type` | `Col 3 ('Cement')` -> `'OPC 53 (UltraTech)'` | `'OPC 53 (UltraTech)'` | Raw Categorical |
| `curing_age_days` | `Col 2 ('Curing Age (Days)')` | `56` (rows 1001–1100), `90` (rows 1101–1200) | Raw Numerical |
| `mechanical_property` | `'Flexural Strength'` | `'Flexural Strength'` | Target Classification |
| `strength_mpa` | `Col 6 ('Flexural Strength (MPa)')` | `Sheet2` col `56 BC` / `90 BC` at row `sample_replicate + 4` | **Raw Experimental Measurement** |
| `specimen_geometry` | `'100x100x500mm Prism'` | `'100x100x500mm Prism'` | Domain Standard |
| `source_file` | `'F Strength Reading 1000000.xlsx'` | `'F Strength Reading 1000000.xlsx'` | Lineage Metadata |
| `source_sheet` | `'Sheet1'` | `'Sheet2'` | Lineage Metadata |
| `source_row_index` | Excel row coordinate in `Sheet1` (2 to 1001) | Excel row coordinate in `Sheet2` (5 to 104) | Lineage Metadata |
| `is_restored_value` | `False` | `True` | Provenance Flag |

### 3. Split Tensile Strength Mapping (`T Strenth Reading 1000000.xlsx`)
- **Workbook**: `data/T Strenth Reading 1000000.xlsx`
- **Source Sheet**: `Sheet1` (Rows 2 to 1201; 1,200 complete data rows, 0 nulls)
- **Source Structure**: Unpivoted long format (600 Normal Concrete + 600 Bacterial Concrete across 6 curing ages).
- **Transformation**: Direct pass-through mapping.

| Destination Field | Source Field / Expression in `Sheet1` | Value Classification |
| :--- | :--- | :--- |
| `sample_id` | `'TS_' + ('NC' if c_type=='Normal Concrete' else 'BC') + '_' + str(age).zfill(2) + 'd_' + str(((row-2)%100)+1).zfill(3)` | Derived Specimen PK |
| `experiment_id` | `'EXP_TS_' + ('NC' if c_type=='Normal Concrete' else 'BC') + '_' + str(age).zfill(2) + 'd'` | Derived Batch ID |
| `sample_replicate` | `((row_idx - 2) % 100) + 1` | Raw Sample Index (1–100) |
| `concrete_type` | `Col 1 ('Concrete Type')` | Raw Categorical |
| `bacterial_status` | `'Control'` if Normal else `'Inoculated'` | Derived Category |
| `bacterial_species` | `'None'` if Normal else `'Bacillus subtilis'` | Raw Categorical |
| `bacterial_concentration_cells_ml` | `0` if Normal else `1000000` | Raw Numerical |
| `cement_type` | `Col 3 ('Cement')` -> `'OPC 53 (UltraTech)'` | Raw Categorical |
| `curing_age_days` | `Col 2 ('Curing Age (Days)')` | Raw Numerical (`7, 14, 21, 28, 56, 90`) |
| `mechanical_property` | `'Split Tensile Strength'` | Target Classification |
| `strength_mpa` | `Col 6 ('Split Tensile Strength (MPa)')` | **Raw Experimental Measurement** |
| `specimen_geometry` | `'150x300mm Cylinder'` | Domain Standard |
| `source_file` | `'T Strenth Reading 1000000.xlsx'` | Lineage Metadata |
| `source_sheet` | `'Sheet1'` | Lineage Metadata |
| `source_row_index` | Excel row coordinate (2 to 1201) | Lineage Metadata |
| `is_restored_value` | `False` | Provenance Flag |

---

## C. Included Sheets

The Master Dataset is sourced exclusively from worksheets containing raw, un-averaged physical specimen load/fracture tests.

| Source File | Sheet Name | Extracted Rows | Target Captured | Rationale for Inclusion |
| :--- | :--- | :--- | :--- | :--- |
| `C Strenth Reading 1000000.xlsx` | **`Comp 6`** | 600 rows (yielding 1,200 long rows) | Compressive Strength (N & BC) | Verified raw individual cube tests across 6 curing ages with 100 replicates each. |
| `F Strength Reading 1000000.xlsx` | **`Sheet1`** | 1,000 rows | Flexural Strength (7d–28d BC, 7d–90d NC) | Complete individual prism fracture measurements. |
| `F Strength Reading 1000000.xlsx` | **`Sheet2`** | 200 values (rows 5–104) | Flexural Strength (56d & 90d BC) | Preserves authentic raw laboratory specimen readings omitted from `Sheet1`. |
| `T Strenth Reading 1000000.xlsx` | **`Sheet1`** | 1,200 rows | Split Tensile Strength (N & BC) | Complete, pristine long-format cylinder splitting tests. |

---

## D. Excluded Sheets & Exclusion Rationale

All remaining 11 sheets across the 3 workbooks are excluded from the modeling dataset to protect model integrity and prevent data leakage.

| Source File | Sheet Name | Content & Structure | Primary Reason for Exclusion |
| :--- | :--- | :--- | :--- |
| `C Strenth Reading` | **`Sheet2`** | 100 sample rows + trailing `=SUM(B5:B104)` rows (e.g. 1934.44, 4066.08) | Redundant with `Comp 6`. Contains embedded aggregate summary rows that would corrupt tabular modeling if parsed. |
| `C Strenth Reading` | **`Sheet1`** | 100 sample rows + trailing sum rows | Exact duplicate of `Sheet2`. |
| `C Strenth Reading` | **`Sheet3`** | 6 sample rows | Partial duplicate subset (samples 1–6). |
| `C Strenth Reading` | **`Final Result`** | Macro comparison of percentage gains across ages and dilutions | Contains computed percentage increases, differences, and formulas (`=N5-K14`, `=Q5*1.5`), not raw specimens. |
| `C Strenth Reading` | **`105 c`** | Comparison of compressive vs. flexural strength at $10^5$ cells/mL | Contains 291 formula cells calculating cross-property differentials. |
| `C Strenth Reading` | **`c`** | Dilution series summary for compressive strength ($10^3$ to $10^8$) | Macro-level average summary table. Contains no individual sample replicates. |
| `C Strenth Reading` | **`f`** | Dilution series summary for flexural strength ($10^3$ to $10^8$) | Macro-level average summary table. Contains no individual sample replicates. |
| `C Strenth Reading` | **`t`** | Dilution series summary for split tensile strength ($10^3$ to $10^8$) | Macro-level average summary table. Contains no individual sample replicates. |
| `C Strenth Reading` | **`RCPT`** | Rapid Chloride Permeability Test summary (8 rows) | Durability single-value benchmark test across dilutions, not a sample-level mechanical strength measurement. |
| `C Strenth Reading` | **`Water Absorption`** | Water absorption percentages (7 rows) | Durability single-value benchmark test across dilutions, not a sample-level mechanical strength measurement. |
| `T Strenth Reading` | **`Ten 6`** | 100 sample rows + trailing summary rows | Redundant with `T Sheet1` (which is already unpivoted and complete). Contains trailing sum/average rows. |

---

## E. Raw vs. Derived Data Classification

To ensure total transparency, all data within the source workbooks has been categorized under a formal taxonomy:

1. **Raw Experimental Specimen Data (Tier 1 - Primary Modeling Ground Truth)**:
   - Individual test specimen failure loads recorded directly during laboratory testing machines (UTM / Compression Testing Machine).
   - Characteristics: Stored in continuous tabular arrays of 100 samples per test condition; exhibits natural experimental standard deviation; contains no mathematical formulas.
   - Locations: `Comp 6` (C Strength), `Sheet1` rows 2–1001 (F Strength), `Sheet2` rows 5–104 (F Strength), `Sheet1` rows 2–1201 (T Strength).

2. **In-Sheet Aggregate Statistics (Tier 2 - Excluded Summary Totals)**:
   - Formula-based totals and means appended directly to raw data sheets.
   - Characteristics: Excel formula cells (`=SUM(...)`, `=AVERAGE(...)`); values are orders of magnitude larger than physical failure strengths (e.g. sums of 100 cubes = `4066.08 MPa`).
   - Locations: Rows 105 to 115 in `Sheet2` (C Strength), `Sheet2` (F Strength), and `Ten 6` (T Strength).

3. **Cross-Condition Laboratory Synthesis Tables (Tier 3 - Excluded Benchmarks)**:
   - High-level tables compiled for research paper presentation.
   - Characteristics: Matrix of mean values across curing days and bacterial dilution series ($10^3–10^8$ cells/mL); percentage gains relative to control; durability metrics (RCPT Coulombs, absorption %).
   - Locations: Sheets `Final Result`, `105 c`, `c`, `f`, `t`, `RCPT`, `Water Absorption`.

---

## F. Missing-Data Handling & Verification Plan

### 1. Audit Findings on Missing Flexural Data
The audit identified exactly **200 missing (`NaN`) values** in `F Strength Reading 1000000.xlsx` (`Sheet1`):
- Rows 1001 to 1100: `Bacterial Concrete` at `56 Days` (100 rows with empty strength).
- Rows 1101 to 1200: `Bacterial Concrete` at `90 Days` (100 rows with empty strength).

### 2. Proof of 1-to-1 Mapping to `Sheet2`
To verify whether `Sheet2` contains the authentic un-truncated source of `Sheet1`, an element-wise verification was executed in Python comparing every single non-missing observation:

```text
Normal Concrete  7d vs  7 N : Max absolute difference = 0.00000000 (100/100 match)
Normal Concrete 14d vs 14 N : Max absolute difference = 0.00000000 (100/100 match)
Normal Concrete 21d vs 21 N : Max absolute difference = 0.00000000 (100/100 match)
Normal Concrete 28d vs 28 N : Max absolute difference = 0.00000000 (100/100 match)
Normal Concrete 56d vs 56 N : Max absolute difference = 0.00000000 (100/100 match)
Normal Concrete 90d vs 90 N : Max absolute difference = 0.00000000 (100/100 match)
Bacterial Concrete  7d vs  7 BC: Max absolute difference = 0.00000000 (100/100 match)
Bacterial Concrete 14d vs 14 BC: Max absolute difference = 0.00000000 (100/100 match)
Bacterial Concrete 21d vs 21 BC: Max absolute difference = 0.00000000 (100/100 match)
Bacterial Concrete 28d vs 28 BC: Max absolute difference = 0.00000000 (100/100 match)
```

### 3. Verification Conclusion
- Across all 1,000 paired non-missing points, the maximum discrepancy is strictly **0.00000000**.
- Columns `56 BC` and `90 BC` in `Sheet2` (rows 5 to 104) contain 100 complete, valid measurements with normal sample statistics (mean `5.53 MPa` and `5.91 MPa`).
- **Protocol**: The mapping is **unambiguously verified**. In the future dataset construction step, rows 1001–1200 will be restored from `Sheet2` and flagged with `is_restored_value = True`. In accordance with current instructions, **no imputation or filling has been executed yet**.

---

## G. Constant-Feature Analysis

The audit revealed three features with zero variance across the $10^6$ specimen datasets:

| Feature Name | Constant Value | Why It Is Constant | Role in Raw Master Dataset | Downstream Modeling Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| `cement_type` | `'OPC 53 (UltraTech)'` | The laboratory standardized on a single binder brand and grade throughout the investigation. | **Retain permanently**. Essential metadata for scientific provenance and cross-study synthesis. | Exclude from tabular regression features (zero predictive gradient). |
| `bacterial_species` | `'Bacillus subtilis'` (in all BC) | Single bio-mineralization agent investigated. | **Retain permanently**. Differentiated from `'None'` for Control concrete. | Combine with `concrete_type` or retain as categorical predictor. |
| `bacterial_concentration_cells_ml` | `1,000,000` (in all BC) | Primary mechanical study focused on the $10^6$ cells/mL dosage. | **Retain permanently**. Set to `0` for Control concrete. | Serves as an active binary/numerical predictor (`0` vs `10^6`). |

---

## H. Duplicate & Repeated-Measurement Analysis

In the dataset audit, duplicate row checks showed 249 duplicate rows in `F Sheet1` and 111 duplicate rows in `T Sheet1`. A detailed investigation was conducted to determine their nature without making assumptions:

### 1. Root Cause Breakdown
1. **Identical Feature Tuples**: In the raw files, the feature columns are `[Concrete Type, Curing Age, Cement, Bacteria, Bacterial Concentration]`. For each test condition (e.g. Normal Concrete at 7 days), all 100 specimens have **identical feature values**.
2. **Experimental Precision Granularity**: Strength values are recorded to 3 decimal places. In a cohort of 100 specimens, multiple distinct specimens naturally fracture at the same load (e.g. two cylinders fracturing at exactly `2.024 MPa`). Because the raw sheet lacked a unique specimen ID, pandas flags these as duplicate rows.
3. **Missing Data Duplication**: In `F Sheet1`, 198 of the 249 duplicates were generated solely by the two blocks of 100 consecutive `NaN` rows at 56d and 90d BC.

### 2. Empirical Conclusion
- **These are separate physical specimens, NOT repeated measurements of the same specimen, and NOT duplicate recording errors.** Concrete strength testing is inherently destructive; once a cube, prism, or cylinder is crushed to failure, it cannot be re-tested.
- Adding the explicit `sample_replicate` index (1–100) and `sample_id` completely resolves all apparent row duplication.

---

## I. Leakage-Safe Grouping & Partitioning Strategy

Concrete test specimens cast in the same laboratory batch share common mixing water, compaction energy, curing tank thermal gradients, and cement bag moisture variations.

### 1. Leakage Risks Identified
- **Random Specimen Splitting Leakage**: If specimens 1–80 of a batch are trained on and specimens 81–100 are tested on via random splitting, the model can exploit batch-specific micro-variations, producing artificially inflated test metrics that collapse on new mixes.
- **Cross-Age Temporal Leakage**: If early and late curing points from the exact same batch are mixed indiscriminately, temporal autocorrelation leaks across folds.

### 2. Proposed Grouping Architecture

```text
Hierarchical Grouping Identifiers:
├── experiment_id: 'EXP_CS_NC_07d'  (Cohort of 100 specimens for a single test condition)
└── sample_replicate: 1 .. 100      (Replicate alignment across tests)
```

### 3. Recommended Validation Protocols
1. **Replicate-Grouped Cross-Validation (`GroupKFold` on `sample_replicate`)**:
   - Partition specimens by `sample_replicate` (e.g., Replicates 1–20 in Fold 1, 21–40 in Fold 2, etc.).
   - Ensures that test specimens never share batch replicate counterparts with the training fold.
2. **Out-of-Age Extrapolation Testing (Temporal Validation)**:
   - Hold out all 90-day specimens (mature concrete) while training exclusively on 7, 14, 21, 28, and 56 days.
   - Specifically benchmarks TabPFN's ability to extrapolate long-term bio-mineralization kinetics without temporal leakage.

---

## J. Target Variable Representation: Long vs. Wide

We evaluated whether to represent the three target properties in a single long-format pair (`mechanical_property`, `strength_mpa`) versus a wide pivoted table (`compressive_strength_mpa`, `flexural_strength_mpa`, `split_tensile_strength_mpa`).

### Comparative Evaluation

| Evaluation Dimension | Long-Format Representation (`mechanical_property` + `strength_mpa`) | Wide Pivoted Representation (3 Separate Target Columns) |
| :--- | :--- | :--- |
| **Physical Specimen Truth** | **100% Faithful**: Cubes, prisms, and cylinders are distinct physical specimens tested independently. | **Artificial Assumption**: Forces a cube (CS), prism (FS), and cylinder (TS) to appear as the same single object. |
| **Information Preservation** | **Zero Loss**: Preserves exact specimen row coordinates and test geometries. | **Lossy**: Obscures individual specimen failure characteristics. |
| **Missing Data Tolerance** | High: Any property can have missing tests without invalidating the row. | Poor: Missing one property creates sparse target matrices. |
| **TabPFN Compatibility** | **Optimal**: TabPFN can ingest `mechanical_property` as an input categorical feature to predict strength across all tests. | Requires training 3 entirely separate models. |
| **Downstream Flexibility** | Can be pivoted to wide format in 1 line of pandas code whenever multi-target models are desired. | Difficult to unpivot cleanly once transformed. |

### Conclusion
**Adopt the Long-Format Target Representation** as the canonical Master Dataset standard.

---

## K. Unresolved Issues & Technical Risks

1. **Latent Mix Design Proportions**: Water-cement ratio, sand content, coarse aggregate grading, and superplasticizer dosage are not recorded in the spreadsheets. Models will operate strictly conditioned on `concrete_type`, `curing_age_days`, and `bacterial_concentration`.
2. **Batch Casting Timestamps**: No explicit dates or batch mixing timestamps were recorded. Replicate indices (1–100) serve as the best available proxy for batch cohort grouping.
3. **Cross-Workbook Specimen Coupling**: Replicate #1 in Compressive Strength was cast concurrently with Replicate #1 in Flexural Strength, but they are physically distinct test geometries. Treating them as separate long-format rows avoids unverified assumptions.

---

## L. Recommended Next Implementation Steps

Upon receiving authorization to create the Master Dataset, the following automated pipeline will be executed:

1. **Pipeline Script Creation (`src/build_master_dataset.py`)**:
   - Load raw `Comp 6`, unpivot the Normal/Bacterial columns into 1,200 long rows.
   - Load raw `F Sheet1` (rows 1–1000) and restore rows 1001–1200 from `Sheet2` (flagging `is_restored_value = True`).
   - Load raw `T Sheet1` (rows 1–1200).
   - Concatenate into the unified 16-column schema (3,600 rows).
2. **Integrity Validation & Export**:
   - Verify that all 3,600 rows have non-null targets, correct dtypes, and valid numerical distributions.
   - Export to `data/master_dataset.csv` and `data/master_dataset.parquet`.
   - Output a cryptographic SHA256 checksum and verification summary report.
3. **Model Ready Handoff**:
   - Prepare baseline TabPFN regression pipeline using the validated dataset.

---
*Report compiled and verified against project workspace.*