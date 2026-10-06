# Master Dataset V1 Validation & Audit Report

> **Artifact References**:
> - Master CSV: [`data/master_dataset.csv`](file:///G:/Projects/Concrete testing/data/master_dataset.csv)
> - Master Parquet: [`data/master_dataset.parquet`](file:///G:/Projects/Concrete testing/data/master_dataset.parquet)
> - Schema Documentation: [`reports/master_dataset_schema.csv`](file:///G:/Projects/Concrete testing/reports/master_dataset_schema.csv)

---

## 1. Executive Validation Summary

| Verification Item | Requirement / Expected | Observed in Master Dataset | Validation Status |
| :--- | :--- | :--- | :--- |
| **Total Row Count** | Exactly `3,600` rows | `3,600` rows | ✅ **PASS** |
| **Total Column Count** | Exactly `16` columns | `16` columns | ✅ **PASS** |
| **Target Non-Null Count** | `3,600` non-null values | `3,600` non-null values | ✅ **PASS** |
| **Restored Values Count** | Exactly `200` values | `200` values | ✅ **PASS** |
| **Unique Specimen IDs** | `3,600` unique IDs | `3,600` unique IDs (0 duplicates) | ✅ **PASS** |
| **Concrete Mix Counts** | 1,800 Normal / 1,800 Bacterial | 1,800 Normal Concrete / 1,800 Bacterial Concrete | ✅ **PASS** |
| **Mechanical Properties** | 1,200 Comp / 1,200 Flex / 1,200 Tens | 1,200 Compressive / 1,200 Flexural / 1,200 Split Tensile | ✅ **PASS** |
| **Curing Ages** | 6 ages (7, 14, 21, 28, 56, 90) | 6 ages (600 rows per age) | ✅ **PASS** |
| **Original Raw Excel Files** | 100% Unaltered | Preserved in `data/` untouched | ✅ **PASS** |
| **Machine Learning Models** | No models trained | None initiated | ✅ **PASS** |

---

## 2. Cryptographic Checksums & File Manifest

| File Name | File Format | File Size | SHA256 Checksum |
| :--- | :--- | :--- | :--- |
| `master_dataset.csv` | Plaintext CSV (UTF-8) | 679,378 bytes | `0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a` |
| `master_dataset.parquet` | Apache Parquet (Snappy) | 59,607 bytes | `0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a` |

---

## 3. Detailed Target Distribution by Mechanical Property

Each mechanical property represents 1,200 distinct destructive physical tests conducted across concrete specimens:

| Mechanical Property | Specimen Count | Min Strength (MPa) | Mean Strength (MPa) | Median (MPa) | Max Strength (MPa) | Std Dev (MPa) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Compressive Strength** | 1,200 | 15.570 | 29.141 | 28.825 | 44.160 | 6.162 |
| **Flexural Strength** | 1,200 | 2.428 | 4.419 | 4.402 | 6.490 | 0.855 |
| **Split Tensile Strength** | 1,200 | 1.380 | 2.762 | 2.764 | 4.183 | 0.583 |

### Distribution Breakdown by Concrete Type & Curing Age

| Mechanical Property | Concrete Type | 7 Days | 14 Days | 21 Days | 28 Days | 56 Days | 90 Days | Total Rows |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Compressive Strength | Normal Concrete | 19.34 | 23.53 | 25.60 | 28.16 | 30.92 | 32.83 | 600 |
| Compressive Strength | Bacterial Concrete | 22.04 | 26.49 | 29.84 | 33.04 | 37.24 | 40.66 | 600 |
| Flexural Strength | Normal Concrete | 2.99 | 3.62 | 3.90 | 4.19 | 4.60 | 4.91 | 600 |
| Flexural Strength | Bacterial Concrete | 3.53 | 4.20 | 4.63 | 5.01 | 5.53 | 5.91 | 600 |
| Split Tensile Strength | Normal Concrete | 1.80 | 2.20 | 2.39 | 2.68 | 2.90 | 3.12 | 600 |
| Split Tensile Strength | Bacterial Concrete | 2.11 | 2.58 | 2.89 | 3.18 | 3.50 | 3.80 | 600 |

---

## 4. Audit of Restored Observations

In `F Strength Reading 1000000.xlsx`, rows 1001–1200 of `Sheet1` contained 200 blank target values:
- **Condition 1**: `Bacterial Concrete` at `56 Days` (100 rows)
- **Condition 2**: `Bacterial Concrete` at `90 Days` (100 rows)

### Mathematical Provenance & Mapping Protocol
Prior to restoration, all 10 non-missing test conditions in `Sheet1` were matched element-wise against `Sheet2`. All 1,000 paired sample points exhibited an absolute discrepancy of strictly **0.00000000**.

- Exactly **200 values were restored** from columns `56 BC` and `90 BC` of `Sheet2` (rows 5 to 104 in Excel).
- For complete provenance, each restored record has:
  - `is_restored_value = True`
  - `source_sheet = 'Sheet2'`
  - `source_row_index = sample_replicate + 4` (referencing the exact Excel row in `Sheet2`)
- All other 3,400 records maintain `is_restored_value = False`.

### Summary Statistics of Restored Specimens

| Restored Condition | Number of Specimens | Min (MPa) | Mean (MPa) | Median (MPa) | Max (MPa) | Std Dev (MPa) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **56-Day Bacterial Flexure** | 100 | 4.918 | 5.530 | 5.523 | 6.172 | 0.246 |
| **90-Day Bacterial Flexure** | 100 | 5.191 | 5.913 | 5.926 | 6.490 | 0.312 |

---

## 5. Duplicate Record Investigation & Specimen Identity

- **Unique Specimen IDs**: Exactly **3,600 unique `sample_id` values** exist (0 duplicates).
- **Duplicate Physical Feature-Target Tuples**: 162 rows across the dataset share identical numerical values with another row in the same batch (e.g., two specimens fracturing at `2.024 MPa`).
  - **Explanation**: These represent **distinct physical test specimens** tested on the machine that naturally fractured at identical loads within measurement resolution (3 decimal places).
  - **Handling**: These are **NOT** duplicate recordings or software artifacts; they are legitimate experimental measurements and have been preserved intact.

---

## 6. Experimental Nature & Grouping Semantics

- **Not 3,600 Independent Mixes**: The 3,600 rows do **NOT** represent 3,600 separate concrete casting experiments. They represent individual destructive physical tests conducted on specimens prepared under **36 distinct experimental conditions** (3 properties × 2 mix types × 6 curing ages = 36 cohorts of 100 replicates).
- **`experiment_id`**: Serves as the cohort identifier for the 100-specimen test group (e.g. `EXP_CS_NC_07d`). Note: True laboratory batch mixing timestamps are unavailable in the source files.
- **`sample_replicate`**: Preserves the replicate index (1–100). As specified in the design, `sample_replicate` should **NOT** be used as the sole validation grouping variable; future validation schemes must combine cohort grouping with stratified out-of-age testing.

---

## 7. Derived Logical Representations

- For **Normal Concrete** (Control), the biological fields are encoded as:
  - `bacterial_status = 'Control'`
  - `bacterial_species = 'None'` (derived logical representation)
  - `bacterial_concentration_cells_ml = 0` (derived logical representation)
- In the raw spreadsheets, `1000000` was redundantly entered in normal concrete rows. The derived encoding provides logical consistency while preserving full traceability via `source_row_index`.

---

## 8. Final Compliance Statement

- [x] Expected row count = 3,600 verified.
- [x] Target non-null count = 3,600 verified.
- [x] Target ranges verified.
- [x] 6 curing ages verified (7, 14, 21, 28, 56, 90 days).
- [x] Normal/Bacterial counts verified (1,800 Normal, 1,800 Bacterial).
- [x] Three mechanical properties verified (1,200 Compressive, 1,200 Flexural, 1,200 Tensile).
- [x] Restored-value count = 200 verified.
- [x] Duplicate specimen IDs = 0 verified.
- [x] Duplicate physical records explained and preserved.
- [x] Full source traceability preserved.
- [x] Cryptographic SHA256 checksums recorded.
- [x] **No machine learning models trained.**