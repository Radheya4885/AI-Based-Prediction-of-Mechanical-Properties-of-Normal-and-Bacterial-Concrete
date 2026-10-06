# TabPFN Dedicated Concrete Dataset V1

## 1. Overview & Purpose
This directory (`data/tabpfn_concrete_v1/`) contains the dedicated, isolated dataset copy and feature representations for TabPFN (Tabular Prior-data Fitted Network) foundational modeling and fine-tuning.

To ensure total scientific integrity, the canonical master dataset files ([`data/master_dataset.csv`](file:///g:/Projects/Concrete%20testing/data/master_dataset.csv) and [`data/master_dataset.parquet`](file:///g:/Projects/Concrete%20testing/data/master_dataset.parquet)) remain **permanently immutable and untouched**. All TabPFN feature engineering, transformations, and representations are applied strictly to the files in this directory.

---

## 2. Checksum Verification & Lineage

| Dataset File | File Format | File Size | SHA256 Checksum | Provenance Status |
| :--- | :--- | :--- | :--- | :--- |
| `tabpfn_concrete_master_copy.csv` | CSV (UTF-8) | 679,378 bytes | `0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a` | Exact byte-level replica of master CSV |
| `tabpfn_concrete_master_copy.parquet` | Parquet | 59,607 bytes | `0909cb72fcf61bcff7a1e9f98f2f0363953f7d4a346b0dd41da7da0efa63678a` | Exact byte-level replica of master Parquet |
| `tabpfn_concrete_model.csv` | CSV (UTF-8) | Derived / Extended | *Updated upon feature engineering* | Official modeling representation |
| `tabpfn_concrete_model.parquet` | Parquet | Derived / Extended | *Updated upon feature engineering* | Official modeling representation |

### Verification Record
- **Source CSV**: `data/master_dataset.csv` (SHA256: `0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a`)
- **Copied CSV**: `data/tabpfn_concrete_v1/tabpfn_concrete_master_copy.csv` (SHA256: `0d18690dac46ce06ccb3bca1c8b7eb1dd752b7cbcc9d95821473730afb70000a`)
- **Element-by-Element Check**: Verified 3,600 rows × 16 columns with 0 mismatches across all columns before feature modification.

---

## 3. Immutable Source Rule
1. Under NO circumstances may files in `data/*.xlsx`, `data/master_dataset.*`, or `data/splits/` be altered, overwritten, or modified.
2. The locked final test set (`final_test_rows.csv`, 800 rows across 8 cohorts) remains sealed and must never be used for model selection, feature engineering tuning, or TabPFN training.
3. Target values (`strength_mpa`), curing ages, specimen IDs, and laboratory measurements remain strictly unmodified.

---

## 4. Modeling Configurations
This modeling dataset supports two standard representations:
1. **`V1_RAW`**: Only canonical original features (`curing_age_days`, `concrete_type`, `bacterial_concentration_cells_ml`, `mechanical_property`).
2. **`V2_ENGINEERED`**: Original features plus selected deterministic nonlinear physical representations (e.g. logarithmic curing kinetics, biological presence indicators, interaction terms).
