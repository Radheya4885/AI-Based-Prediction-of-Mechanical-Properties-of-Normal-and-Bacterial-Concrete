# Concrete Strength Prediction Project - Dataset Audit Report

> **Status**: Dataset Audit Complete (Read-Only Mode)
> **Policy Adherence**: Original datasets have NOT been modified, overwritten, merged, or trimmed. No models have been trained.

---

## Executive Summary

| Metric | Value | Details |
| :--- | :--- | :--- |
| **Total Files Audited** | **3 Excel Workbooks** | `C Strenth Reading 1000000.xlsx`, `F Strength Reading 1000000.xlsx`, `T Strenth Reading 1000000.xlsx` |
| **Total Sheets Audited** | **15 Sheets** | 11 in C Strength, 2 in F Strength, 2 in T Strength |
| **Total Columns Cataloged** | **179 Columns** | Documented in `reports/dataset_schema.csv` |
| **Primary Targets** | **3 Mechanical Properties** | Compressive Strength (MPa), Flexural Strength (MPa), Split Tensile Strength (MPa) |
| **Secondary Durability Tests** | **2 Properties** | Rapid Chloride Permeability Test (RCPT in Coulombs), Water Absorption (%) |
| **Curing Ages** | **6 Ages** | 7, 14, 21, 28, 56, and 90 Days |
| **Bacterial Species** | **1 Species** | *Bacillus subtilis* |
| **Bacterial Concentrations** | **7 Concentrations** | Primary: $10^6$ (1,000,000 cells/mL); Summary sheets also include Control ($0$), $10^3, 10^4, 10^5, 10^7, 10^8, 10^9$ cells/mL |

---

## 1. Detailed Dataset Audits

### 1.1 Dataset File: `C Strenth Reading 1000000.xlsx`
- **File Path**: [`data/C Strenth Reading 1000000.xlsx`](file:///G:/Projects/Concrete testing/data/C Strenth Reading 1000000.xlsx)
- **File Size**: 132,551 bytes
- **Primary Target Variable**: **Compressive Strength (MPa)**
- **Description**: Compressive Strength primary workbook with durability and cross-concentration comparisons

#### Sheet Overview (11 sheets)

| # | Sheet Name | Total Rows | Total Cols | Nature of Content | Derived/Formulas | Duplicate Rows |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **`Comp 6`** | 602 | 7 | **Raw Experimental Measurements** | None (hardcoded values) | 0 |
| 2 | **`Sheet2`** | 115 | 13 | Raw Sample Matrix + Summary Stats | Yes (24 formula cells) | 4 |
| 3 | **`Final Result`** | 70 | 22 | Calculated / Summary | Yes (48 formula cells) | 19 |
| 4 | **`105 c`** | 71 | 22 | Calculated / Summary | Yes (291 formula cells) | 11 |
| 5 | **`c`** | 11 | 16 | Calculated / Summary | None (hardcoded values) | 3 |
| 6 | **`f`** | 11 | 17 | Calculated / Summary | None (hardcoded values) | 3 |
| 7 | **`t`** | 12 | 18 | Calculated / Summary | None (hardcoded values) | 4 |
| 8 | **`Sheet1`** | 115 | 13 | **Raw Experimental Measurements** | Yes (24 formula cells) | 4 |
| 9 | **`Sheet3`** | 10 | 14 | Calculated / Summary | None (hardcoded values) | 2 |
| 10 | **`RCPT`** | 12 | 4 | Durability Benchmark Readings | None (hardcoded values) | 2 |
| 11 | **`Water Absorption`** | 12 | 4 | Durability Benchmark Readings | None (hardcoded values) | 3 |

#### Comprehensive Column & Statistical Audit for `C Strenth Reading 1000000.xlsx`

##### Sheet: `Comp 6`
- **Dimensions**: 602 rows x 7 columns cataloged
- **Detected Excel Formulas**: None (all cell values are stored as literals)

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Concrete Type` | `str` | 600/600 | 0.0% | 1 | ⚠️ **YES** | - | - | - | Constant Feature |
| 2 | `Curing Age (Days)` | `int64` | 600/600 | 0.0% | 6 | No | 7.0 | 36.0 | 90.0 | Input Feature (Curing Age) |
| 3 | `Cement` | `str` | 600/600 | 0.0% | 1 | ⚠️ **YES** | - | - | - | Constant Feature |
| 4 | `Bacteria` | `str` | 600/600 | 0.0% | 1 | ⚠️ **YES** | - | - | - | Constant Feature |
| 5 | `Bacterial Concentration (cells/mL)` | `int64` | 600/600 | 0.0% | 1 | ⚠️ **YES** | 1000000.0 | 1000000.0 | 1000000.0 | Constant Feature |
| 6 | `Compressive Strength - Normal Concrete (MPa)` | `float64` | 600/600 | 0.0% | 520 | No | 15.57 | 26.7298 | 36.41 | 🎯 **Compressive Strength** |
| 7 | `Compressive Strength - Bacterial Concrete (MPa)` | `float64` | 600/600 | 0.0% | 525 | No | 17.95 | 31.552 | 44.16 | 🎯 **Compressive Strength** |

##### Sheet: `Sheet2`
- **Dimensions**: 115 rows x 13 columns cataloged
- **Detected Excel Formulas**: `B105` = `=SUM(B5:B104)`, `C105` = `=SUM(C5:C104)`, `D105` = `=SUM(D5:D104)`, `E105` = `=SUM(E5:E104)`

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Sample No` | `object` | 107/111 | 3.6% | 101 | No | 1.0 | 49.6792 | 100.0 | Identifier / Index |
| 2 | `7 N` | `object` | 109/111 | 1.8% | 98 | No | 15.57 | 37.4871 | 1934.44 | 🎯 **Compressive Strength** |
| 3 | `7 BC` | `object` | 109/111 | 1.8% | 100 | No | 17.95 | 42.7641 | 2203.59 | 🎯 **Compressive Strength** |
| 4 | `14 N` | `float64` | 102/111 | 8.11% | 94 | No | 20.62 | 46.3748 | 2353.35 | 🎯 **Compressive Strength** |
| 5 | `14 BC` | `float64` | 102/111 | 8.11% | 90 | No | 23.38 | 52.2019 | 2649.05 | 🎯 **Compressive Strength** |
| 6 | `21 N` | `float64` | 102/111 | 8.11% | 94 | No | 20.64 | 50.4413 | 2559.71 | 🎯 **Compressive Strength** |
| 7 | `21 BC` | `float64` | 102/111 | 8.11% | 91 | No | 27.23 | 58.8075 | 2984.26 | 🎯 **Compressive Strength** |
| 8 | `28 N` | `float64` | 102/111 | 8.11% | 93 | No | 24.81 | 55.4931 | 2816.07 | 🎯 **Compressive Strength** |
| 9 | `28 BC` | `float64` | 102/111 | 8.11% | 91 | No | 29.36 | 65.112 | 3304.19 | 🎯 **Compressive Strength** |
| 10 | `56 N` | `float64` | 102/111 | 8.11% | 98 | No | 27.55 | 60.9235 | 3091.64 | 🎯 **Compressive Strength** |
| 11 | `56 BC` | `float64` | 102/111 | 8.11% | 91 | No | 32.73 | 73.3859 | 3724.06 | 🎯 **Compressive Strength** |
| 12 | `90 N` | `float64` | 102/111 | 8.11% | 91 | No | 29.29 | 64.6879 | 3282.67 | 🎯 **Compressive Strength** |
| 13 | `90 BC` | `float64` | 102/111 | 8.11% | 94 | No | 36.16 | 80.1257 | 4066.08 | 🎯 **Compressive Strength** |

##### Sheet: `Final Result`
- **Dimensions**: 70 rows x 22 columns cataloged
- **Detected Excel Formulas**: `Q5` = `=N5-K14`, `R5` = `=Q5*1.5`, `S5` = `=R5+K14`, `U5` = `=S5-T5`

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `0` | `object` | 7/70 | 90.0% | 7 | No | 3.0 | 6.0 | 9.0 | Feature / Reading |
| 2 | `1` | `object` | 49/70 | 30.0% | 7 | No | 7.0 | 36.0 | 90.0 | Feature / Reading |
| 3 | `2` | `object` | 51/70 | 27.14% | 34 | No | 15.056 | 28.0958 | 39.425 | Feature / Reading |
| 4 | `3` | `object` | 49/70 | 30.0% | 45 | No | 17.158 | 34.1503 | 54.92 | Feature / Reading |
| 5 | `4` | `float64` | 0/70 | 100.0% | 0 | No | - | - | - | Feature / Reading |
| 6 | `5` | `object` | 49/70 | 30.0% | 7 | No | 7.0 | 36.0 | 90.0 | Feature / Reading |
| 7 | `6` | `object` | 51/70 | 27.14% | 34 | No | 2.212 | 4.4285 | 6.39 | Feature / Reading |
| 8 | `7` | `object` | 49/70 | 30.0% | 44 | No | 2.5 | 5.239 | 8.2395 | Feature / Reading |
| 9 | `8` | `float64` | 0/70 | 100.0% | 0 | No | - | - | - | Feature / Reading |
| 10 | `9` | `object` | 49/70 | 30.0% | 7 | No | 7.0 | 36.0 | 90.0 | Feature / Reading |
| 11 | `10` | `object` | 51/70 | 27.14% | 30 | No | 1.804 | 2.9061 | 4.719 | Feature / Reading |
| 12 | `11` | `object` | 49/70 | 30.0% | 44 | No | 2.1052 | 3.5843 | 5.422 | Feature / Reading |
| 13 | `12` | `float64` | 0/70 | 100.0% | 0 | No | - | - | - | Feature / Reading |
| 14 | `13` | `object` | 20/70 | 71.43% | 13 | No | 2.8451 | 25.2635 | 90.0 | Feature / Reading |
| 15 | `14` | `object` | 20/70 | 71.43% | 19 | No | 3.0 | 4.3544 | 5.865 | Feature / Reading |
| 16 | `15` | `object` | 14/70 | 80.0% | 13 | No | 3.4366 | 4.7386 | 6.3538 | Feature / Reading |
| 17 | `16` | `float64` | 6/70 | 91.43% | 6 | No | 0.5451 | 0.9029 | 1.1038 | Feature / Reading |
| 18 | `17` | `float64` | 6/70 | 91.43% | 6 | No | 0.8177 | 1.3543 | 1.6557 | Feature / Reading |
| 19 | `18` | `float64` | 12/70 | 82.86% | 12 | No | 3.2641 | 4.4707 | 5.865 | Feature / Reading |
| 20 | `19` | `float64` | 12/70 | 82.86% | 12 | No | 0.45 | 2.3498 | 5.1 | Feature / Reading |
| 21 | `20` | `float64` | 6/70 | 91.43% | 6 | No | -0.2326 | 0.1551 | 0.4616 | Feature / Reading |
| 22 | `21` | `float64` | 12/70 | 82.86% | 12 | No | 3.4366 | 4.7386 | 6.3538 | Feature / Reading |

##### Sheet: `105 c`
- **Dimensions**: 71 rows x 21 columns cataloged
- **Detected Excel Formulas**: `F5` = `=((D5-C5)/(C5))*100`, `N5` = `=L5-K5`, `U5` = `=S5-R5`, `F6` = `=((D6-C6)/(C6))*100`

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `0` | `object` | 7/71 | 90.14% | 7 | No | 3.0 | 6.0 | 9.0 | Feature / Reading |
| 2 | `1` | `object` | 51/71 | 28.17% | 8 | No | 7.0 | 36.0 | 90.0 | Feature / Reading |
| 3 | `2` | `object` | 51/71 | 28.17% | 46 | No | 15.056 | 24.4722 | 31.9948 | Feature / Reading |
| 4 | `3` | `object` | 49/71 | 30.99% | 46 | No | 17.158 | 27.6224 | 36.6611 | Feature / Reading |
| 5 | `4` | `float64` | 0/71 | 100.0% | 0 | No | - | - | - | Feature / Reading |
| 6 | `5` | `float64` | 50/71 | 29.58% | 49 | No | 0.5432 | 6.011 | 23.1477 | Feature / Reading |
| 7 | `6` | `float64` | 0/71 | 100.0% | 0 | No | - | - | - | Feature / Reading |
| 8 | `7` | `float64` | 0/71 | 100.0% | 0 | No | - | - | - | Feature / Reading |
| 9 | `8` | `float64` | 1/71 | 98.59% | 1 | ⚠️ **YES** | 4.6663 | 4.6663 | 4.6663 | Constant Feature |
| 10 | `9` | `object` | 51/71 | 28.17% | 8 | No | 7.0 | 36.0 | 90.0 | Feature / Reading |
| 11 | `10` | `object` | 51/71 | 28.17% | 40 | No | 1.2 | 2.9095 | 4.6201 | Feature / Reading |
| 12 | `11` | `object` | 49/71 | 30.99% | 38 | No | 1.128 | 3.3569 | 5.6033 | Feature / Reading |
| 13 | `12` | `float64` | 0/71 | 100.0% | 0 | No | - | - | - | Feature / Reading |
| 14 | `13` | `float64` | 49/71 | 30.99% | 42 | No | -0.082 | 0.7669 | 4.12 | Feature / Reading |
| 15 | `14` | `float64` | 0/71 | 100.0% | 0 | No | - | - | - | Feature / Reading |
| 16 | `15` | `float64` | 0/71 | 100.0% | 0 | No | - | - | - | Feature / Reading |
| 17 | `16` | `object` | 51/71 | 28.17% | 8 | No | 7.0 | 36.0 | 90.0 | Feature / Reading |
| 18 | `17` | `object` | 51/71 | 28.17% | 46 | No | 1.2 | 3.6965 | 5.865 | Feature / Reading |
| 19 | `18` | `object` | 49/71 | 30.99% | 45 | No | 1.2666 | 3.9707 | 6.3538 | Feature / Reading |
| 20 | `19` | `float64` | 0/71 | 100.0% | 0 | No | - | - | - | Feature / Reading |
| 21 | `20` | `float64` | 49/71 | 30.99% | 38 | No | -0.7396 | 0.47 | 2.9197 | Feature / Reading |

##### Sheet: `c`
- **Dimensions**: 11 rows x 15 columns cataloged
- **Detected Excel Formulas**: None (all cell values are stored as literals)

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Days` | `int64` | 6/6 | 0.0% | 6 | No | 7.0 | 36.0 | 90.0 | Input Feature (Curing Age) |
| 2 | `NC` | `float64` | 6/6 | 0.0% | 6 | No | 15.056 | 22.9087 | 28.596 | Feature / Reading |
| 3 | `BC3` | `float64` | 6/6 | 0.0% | 6 | No | 17.158 | 23.9208 | 29.385 | Feature / Reading |
| 4 | `NC.1` | `float64` | 6/6 | 0.0% | 6 | No | 18.4589 | 25.47 | 30.8464 | Feature / Reading |
| 5 | `BC4` | `float64` | 6/6 | 0.0% | 6 | No | 19.9781 | 28.1392 | 34.9099 | Feature / Reading |
| 6 | `NC.2` | `float64` | 6/6 | 0.0% | 6 | No | 18.9 | 26.0814 | 31.9948 | Feature / Reading |
| 7 | `BC5` | `float64` | 6/6 | 0.0% | 6 | No | 20.9974 | 29.5693 | 36.6611 | Feature / Reading |
| 8 | `NC.3` | `float64` | 6/6 | 0.0% | 6 | No | 17.98 | 25.1614 | 31.0748 | Feature / Reading |
| 9 | `BC6` | `float64` | 6/6 | 0.0% | 6 | No | 20.3474 | 28.9193 | 36.0111 | Feature / Reading |
| 10 | `NC.4` | `float64` | 6/6 | 0.0% | 6 | No | 17.38 | 24.5614 | 30.4748 | Feature / Reading |
| 11 | `BC7` | `float64` | 6/6 | 0.0% | 6 | No | 19.6474 | 28.2193 | 35.3111 | Feature / Reading |
| 12 | `NC.5` | `float64` | 6/6 | 0.0% | 6 | No | 16.68 | 23.8614 | 29.7748 | Feature / Reading |
| 13 | `BC8` | `float64` | 6/6 | 0.0% | 6 | No | 19.1474 | 27.7193 | 34.8111 | Feature / Reading |
| 14 | `NC.6` | `float64` | 6/6 | 0.0% | 6 | No | 16.08 | 23.2614 | 29.1748 | Feature / Reading |
| 15 | `BC9` | `float64` | 6/6 | 0.0% | 6 | No | 18.2974 | 26.8693 | 33.9611 | Feature / Reading |

##### Sheet: `f`
- **Dimensions**: 11 rows x 15 columns cataloged
- **Detected Excel Formulas**: None (all cell values are stored as literals)

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Days` | `int64` | 6/6 | 0.0% | 6 | No | 7.0 | 36.0 | 90.0 | Input Feature (Curing Age) |
| 2 | `NC` | `float64` | 6/6 | 0.0% | 6 | No | 2.212 | 2.702 | 3.1 | Feature / Reading |
| 3 | `BC3` | `float64` | 6/6 | 0.0% | 5 | No | 2.5 | 3.0 | 3.3 | Feature / Reading |
| 4 | `NC.1` | `float64` | 6/6 | 0.0% | 6 | No | 2.7906 | 3.7966 | 4.6201 | Feature / Reading |
| 5 | `BC4` | `float64` | 6/6 | 0.0% | 6 | No | 3.1984 | 4.4833 | 5.6033 | Feature / Reading |
| 6 | `NC.2` | `float64` | 6/6 | 0.0% | 6 | No | 2.7906 | 3.7966 | 4.6201 | Feature / Reading |
| 7 | `BC5` | `float64` | 6/6 | 0.0% | 6 | No | 3.1984 | 4.4833 | 5.6033 | Feature / Reading |
| 8 | `NC.3` | `float64` | 6/6 | 0.0% | 6 | No | 2.2906 | 3.2966 | 4.1201 | Feature / Reading |
| 9 | `BC6` | `float64` | 6/6 | 0.0% | 6 | No | 2.5784 | 3.8633 | 4.9833 | Feature / Reading |
| 10 | `NC.4` | `float64` | 6/6 | 0.0% | 6 | No | 1.6906 | 2.6966 | 3.5201 | Feature / Reading |
| 11 | `BC7` | `float64` | 6/6 | 0.0% | 6 | No | 1.8784 | 3.1633 | 4.2833 | Feature / Reading |
| 12 | `NC.5` | `float64` | 6/6 | 0.0% | 6 | No | 1.2906 | 2.2966 | 3.1201 | Feature / Reading |
| 13 | `BC8` | `float64` | 6/6 | 0.0% | 6 | No | 1.2784 | 2.5633 | 3.6833 | Feature / Reading |
| 14 | `NC.6` | `float64` | 6/6 | 0.0% | 6 | No | 0.6906 | 1.6966 | 2.5201 | Feature / Reading |
| 15 | `BC9` | `float64` | 6/6 | 0.0% | 6 | No | 0.5284 | 1.8133 | 2.9333 | Feature / Reading |

##### Sheet: `t`
- **Dimensions**: 12 rows x 15 columns cataloged
- **Detected Excel Formulas**: None (all cell values are stored as literals)

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Days` | `int64` | 6/6 | 0.0% | 6 | No | 7.0 | 36.0 | 90.0 | Input Feature (Curing Age) |
| 2 | `NC` | `float64` | 6/6 | 0.0% | 6 | No | 2.8451 | 3.7904 | 4.719 | Feature / Reading |
| 3 | `BC3` | `float64` | 6/6 | 0.0% | 6 | No | 3.1725 | 4.277 | 5.422 | Feature / Reading |
| 4 | `NC.1` | `float64` | 6/6 | 0.0% | 6 | No | 3.0 | 4.0867 | 5.1 | Feature / Reading |
| 5 | `BC4` | `float64` | 6/6 | 0.0% | 6 | No | 3.4366 | 4.4321 | 5.5888 | Feature / Reading |
| 6 | `NC.2` | `float64` | 6/6 | 0.0% | 6 | No | 3.45 | 4.6997 | 5.865 | Feature / Reading |
| 7 | `BC5` | `float64` | 6/6 | 0.0% | 6 | No | 3.8866 | 5.0451 | 6.3538 | Feature / Reading |
| 8 | `NC.3` | `float64` | 6/6 | 0.0% | 6 | No | 2.95 | 4.1997 | 5.365 | Feature / Reading |
| 9 | `BC6` | `float64` | 6/6 | 0.0% | 6 | No | 3.3866 | 4.5451 | 5.8538 | Feature / Reading |
| 10 | `NC.4` | `float64` | 6/6 | 0.0% | 6 | No | 2.35 | 3.5997 | 4.765 | Feature / Reading |
| 11 | `BC7` | `float64` | 6/6 | 0.0% | 6 | No | 2.6866 | 3.8451 | 5.1538 | Feature / Reading |
| 12 | `NC.5` | `float64` | 6/6 | 0.0% | 6 | No | 1.8 | 3.0497 | 4.215 | Feature / Reading |
| 13 | `BC8` | `float64` | 6/6 | 0.0% | 6 | No | 2.0666 | 3.2251 | 4.5338 | Feature / Reading |
| 14 | `NC.6` | `float64` | 6/6 | 0.0% | 6 | No | 1.2 | 2.4497 | 3.615 | Feature / Reading |
| 15 | `BC9` | `float64` | 6/6 | 0.0% | 6 | No | 1.2666 | 2.4251 | 3.7338 | Feature / Reading |

##### Sheet: `Sheet1`
- **Dimensions**: 115 rows x 13 columns cataloged
- **Detected Excel Formulas**: `B105` = `=SUM(B5:B104)`, `C105` = `=SUM(C5:C104)`, `D105` = `=SUM(D5:D104)`, `E105` = `=SUM(E5:E104)`

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Sample No` | `object` | 107/111 | 3.6% | 101 | No | 1.0 | 49.6792 | 100.0 | Identifier / Index |
| 2 | `7 N` | `object` | 109/111 | 1.8% | 98 | No | 15.57 | 37.4871 | 1934.44 | 🎯 **Compressive Strength** |
| 3 | `7 BC` | `object` | 109/111 | 1.8% | 100 | No | 17.95 | 42.7641 | 2203.59 | 🎯 **Compressive Strength** |
| 4 | `14 N` | `float64` | 102/111 | 8.11% | 94 | No | 20.62 | 46.3748 | 2353.35 | 🎯 **Compressive Strength** |
| 5 | `14 BC` | `float64` | 102/111 | 8.11% | 90 | No | 23.38 | 52.2019 | 2649.05 | 🎯 **Compressive Strength** |
| 6 | `21 N` | `float64` | 102/111 | 8.11% | 94 | No | 20.64 | 50.4413 | 2559.71 | 🎯 **Compressive Strength** |
| 7 | `21 BC` | `float64` | 102/111 | 8.11% | 91 | No | 27.23 | 58.8075 | 2984.26 | 🎯 **Compressive Strength** |
| 8 | `28 N` | `float64` | 102/111 | 8.11% | 93 | No | 24.81 | 55.4931 | 2816.07 | 🎯 **Compressive Strength** |
| 9 | `28 BC` | `float64` | 102/111 | 8.11% | 91 | No | 29.36 | 65.112 | 3304.19 | 🎯 **Compressive Strength** |
| 10 | `56 N` | `float64` | 102/111 | 8.11% | 98 | No | 27.55 | 60.9235 | 3091.64 | 🎯 **Compressive Strength** |
| 11 | `56 BC` | `float64` | 102/111 | 8.11% | 91 | No | 32.73 | 73.3859 | 3724.06 | 🎯 **Compressive Strength** |
| 12 | `90 N` | `float64` | 102/111 | 8.11% | 91 | No | 29.29 | 64.6879 | 3282.67 | 🎯 **Compressive Strength** |
| 13 | `90 BC` | `float64` | 102/111 | 8.11% | 94 | No | 36.16 | 80.1257 | 4066.08 | 🎯 **Compressive Strength** |

##### Sheet: `Sheet3`
- **Dimensions**: 10 rows x 14 columns cataloged
- **Detected Excel Formulas**: None (all cell values are stored as literals)

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `0` | `float64` | 0/10 | 100.0% | 0 | No | - | - | - | Feature / Reading |
| 2 | `1` | `object` | 7/10 | 30.0% | 7 | No | 1.0 | 3.5 | 6.0 | Feature / Reading |
| 3 | `2` | `object` | 7/10 | 30.0% | 6 | No | 19.15 | 20.015 | 21.78 | Feature / Reading |
| 4 | `3` | `object` | 7/10 | 30.0% | 7 | No | 20.62 | 23.0883 | 24.82 | Feature / Reading |
| 5 | `4` | `object` | 7/10 | 30.0% | 7 | No | 21.38 | 22.8183 | 24.11 | Feature / Reading |
| 6 | `5` | `object` | 7/10 | 30.0% | 7 | No | 24.17 | 25.8667 | 28.07 | Feature / Reading |
| 7 | `6` | `object` | 7/10 | 30.0% | 7 | No | 23.43 | 25.6833 | 27.12 | Feature / Reading |
| 8 | `7` | `object` | 7/10 | 30.0% | 7 | No | 28.73 | 29.33 | 30.91 | Feature / Reading |
| 9 | `8` | `object` | 7/10 | 30.0% | 7 | No | 26.76 | 28.0183 | 29.12 | Feature / Reading |
| 10 | `9` | `object` | 7/10 | 30.0% | 7 | No | 32.41 | 33.38 | 34.92 | Feature / Reading |
| 11 | `10` | `object` | 7/10 | 30.0% | 7 | No | 28.61 | 30.5067 | 31.93 | Feature / Reading |
| 12 | `11` | `object` | 7/10 | 30.0% | 7 | No | 36.03 | 37.7083 | 39.1 | Feature / Reading |
| 13 | `12` | `object` | 7/10 | 30.0% | 7 | No | 30.9 | 33.2133 | 35.86 | Feature / Reading |
| 14 | `13` | `object` | 7/10 | 30.0% | 7 | No | 36.16 | 40.53 | 43.63 | Feature / Reading |

##### Sheet: `RCPT`
- **Dimensions**: 12 rows x 3 columns cataloged
- **Detected Excel Formulas**: None (all cell values are stored as literals)

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Sr. No` | `int64` | 8/8 | 0.0% | 8 | No | 1.0 | 4.5 | 8.0 | Identifier / Index |
| 2 | `Bacterial Dose (cells/ml)` | `str` | 8/8 | 0.0% | 8 | No | - | - | - | Input Feature (Bacterial Concentration) |
| 3 | `RCPT (Coulombs)` | `int64` | 8/8 | 0.0% | 8 | No | 1900.0 | 2613.75 | 3100.0 | 🎯 **Rapid Chloride Permeability** |

##### Sheet: `Water Absorption`
- **Dimensions**: 12 rows x 3 columns cataloged
- **Detected Excel Formulas**: None (all cell values are stored as literals)

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Sr. No` | `int64` | 7/7 | 0.0% | 7 | No | 1.0 | 4.0 | 7.0 | Identifier / Index |
| 2 | `Bacterial Dose (cells/ml)` | `str` | 7/7 | 0.0% | 7 | No | - | - | - | Input Feature (Bacterial Concentration) |
| 3 | `Water Absorption (%)` | `float64` | 7/7 | 0.0% | 7 | No | 4.3 | 4.9286 | 5.8 | 🎯 **Water Absorption** |

---

### 1.2 Dataset File: `F Strength Reading 1000000.xlsx`
- **File Path**: [`data/F Strength Reading 1000000.xlsx`](file:///G:/Projects/Concrete testing/data/F Strength Reading 1000000.xlsx)
- **File Size**: 59,032 bytes
- **Primary Target Variable**: **Flexural Strength (MPa)**
- **Description**: Flexural Strength workbook containing long-format and sample matrix sheets

#### Sheet Overview (2 sheets)

| # | Sheet Name | Total Rows | Total Cols | Nature of Content | Derived/Formulas | Duplicate Rows |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **`Sheet1`** | 1201 | 6 | **Raw Experimental Measurements** | None (hardcoded values) | 249 |
| 2 | **`Sheet2`** | 115 | 13 | Raw Sample Matrix + Summary Stats | Yes (24 formula cells) | 4 |

#### Comprehensive Column & Statistical Audit for `F Strength Reading 1000000.xlsx`

##### Sheet: `Sheet1`
- **Dimensions**: 1201 rows x 6 columns cataloged
- **Detected Excel Formulas**: None (all cell values are stored as literals)

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Concrete Type` | `str` | 1200/1200 | 0.0% | 2 | No | - | - | - | Input Feature (Concrete Type) |
| 2 | `Curing Age (Days)` | `int64` | 1200/1200 | 0.0% | 6 | No | 7.0 | 36.0 | 90.0 | Input Feature (Curing Age) |
| 3 | `Cement` | `str` | 1200/1200 | 0.0% | 1 | ⚠️ **YES** | - | - | - | Constant Feature |
| 4 | `Bacteria` | `str` | 1200/1200 | 0.0% | 1 | ⚠️ **YES** | - | - | - | Constant Feature |
| 5 | `Bacterial Concentration (cells/mL)` | `int64` | 1200/1200 | 0.0% | 1 | ⚠️ **YES** | 1000000.0 | 1000000.0 | 1000000.0 | Constant Feature |
| 6 | `Flexural Strength (MPa)` | `float64` | 1000/1200 | 16.67% | 842 | No | 2.428 | 4.1581 | 5.698 | 🎯 **Flexural Strength** |

##### Sheet: `Sheet2`
- **Dimensions**: 115 rows x 13 columns cataloged
- **Detected Excel Formulas**: `B105` = `=SUM(B5:B104)`, `C105` = `=SUM(C5:C104)`, `D105` = `=SUM(D5:D104)`, `E105` = `=SUM(E5:E104)`

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Sample No` | `object` | 107/111 | 3.6% | 101 | No | 1.0 | 49.6792 | 100.0 | Identifier / Index |
| 2 | `7 N` | `object` | 109/111 | 1.8% | 103 | No | 2.428 | 5.7914 | 299.135 | 🎯 **Flexural Strength** |
| 3 | `7 BC` | `object` | 109/111 | 1.8% | 103 | No | 2.687 | 6.8396 | 353.169 | 🎯 **Flexural Strength** |
| 4 | `14 N` | `float64` | 102/111 | 8.11% | 100 | No | 3.11 | 7.1257 | 361.603 | 🎯 **Flexural Strength** |
| 5 | `14 BC` | `float64` | 102/111 | 8.11% | 99 | No | 3.555 | 8.2696 | 419.65 | 🎯 **Flexural Strength** |
| 6 | `21 N` | `float64` | 102/111 | 8.11% | 100 | No | 3.315 | 7.6942 | 390.453 | 🎯 **Flexural Strength** |
| 7 | `21 BC` | `float64` | 102/111 | 8.11% | 97 | No | 3.898 | 9.1249 | 463.056 | 🎯 **Flexural Strength** |
| 8 | `28 N` | `float64` | 102/111 | 8.11% | 94 | No | 3.558 | 8.2553 | 418.927 | 🎯 **Flexural Strength** |
| 9 | `28 BC` | `float64` | 102/111 | 8.11% | 96 | No | 4.376 | 9.8693 | 500.831 | 🎯 **Flexural Strength** |
| 10 | `56 N` | `float64` | 102/111 | 8.11% | 93 | No | 3.938 | 9.0628 | 459.902 | 🎯 **Flexural Strength** |
| 11 | `56 BC` | `float64` | 102/111 | 8.11% | 98 | No | 4.918 | 10.898 | 553.031 | 🎯 **Flexural Strength** |
| 12 | `90 N` | `float64` | 102/111 | 8.11% | 96 | No | 4.269 | 9.6824 | 491.344 | 🎯 **Flexural Strength** |
| 13 | `90 BC` | `float64` | 102/111 | 8.11% | 95 | No | 5.191 | 11.6526 | 591.327 | 🎯 **Flexural Strength** |

---

### 1.3 Dataset File: `T Strenth Reading 1000000.xlsx`
- **File Path**: [`data/T Strenth Reading 1000000.xlsx`](file:///G:/Projects/Concrete testing/data/T Strenth Reading 1000000.xlsx)
- **File Size**: 60,580 bytes
- **Primary Target Variable**: **Split Tensile Strength (MPa)**
- **Description**: Split Tensile Strength workbook containing long-format and sample matrix sheets

#### Sheet Overview (2 sheets)

| # | Sheet Name | Total Rows | Total Cols | Nature of Content | Derived/Formulas | Duplicate Rows |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **`Sheet1`** | 1201 | 6 | **Raw Experimental Measurements** | None (hardcoded values) | 111 |
| 2 | **`Ten 6`** | 116 | 13 | Raw Sample Matrix + Summary Stats | Yes (24 formula cells) | 4 |

#### Comprehensive Column & Statistical Audit for `T Strenth Reading 1000000.xlsx`

##### Sheet: `Sheet1`
- **Dimensions**: 1201 rows x 6 columns cataloged
- **Detected Excel Formulas**: None (all cell values are stored as literals)

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Concrete Type` | `str` | 1200/1200 | 0.0% | 2 | No | - | - | - | Input Feature (Concrete Type) |
| 2 | `Curing Age (Days)` | `int64` | 1200/1200 | 0.0% | 6 | No | 7.0 | 36.0 | 90.0 | Input Feature (Curing Age) |
| 3 | `Cement` | `str` | 1200/1200 | 0.0% | 1 | ⚠️ **YES** | - | - | - | Constant Feature |
| 4 | `Bacteria` | `str` | 1200/1200 | 0.0% | 1 | ⚠️ **YES** | - | - | - | Constant Feature |
| 5 | `Bacterial Concentration (cells/mL)` | `int64` | 1200/1200 | 0.0% | 1 | ⚠️ **YES** | 1000000.0 | 1000000.0 | 1000000.0 | Constant Feature |
| 6 | `Split Tensile Strength (MPa)` | `float64` | 1200/1200 | 0.0% | 921 | No | 1.38 | 2.7621 | 4.183 | 🎯 **Tensile Strength** |

##### Sheet: `Ten 6`
- **Dimensions**: 116 rows x 13 columns cataloged
- **Detected Excel Formulas**: `B104` = `=SUM(B4:B103)`, `C104` = `=SUM(C4:C103)`, `D104` = `=SUM(D4:D103)`, `E104` = `=SUM(E4:E103)`

| Col # | Column Name | Data Type | Non-Null / Total | Missing (%) | Unique | Constant? | Min | Mean | Max | Nature / Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Sample No` | `float64` | 100/113 | 11.5% | 100 | No | 1.0 | 50.5 | 100.0 | Identifier / Index |
| 2 | `7 N` | `object` | 109/113 | 3.54% | 102 | No | 1.38 | 5.3575 | 180.405 | 🎯 **Tensile Strength** |
| 3 | `7 BC` | `object` | 109/113 | 3.54% | 103 | No | 1.772 | 4.0577 | 210.518 | 🎯 **Tensile Strength** |
| 4 | `14 N` | `object` | 109/113 | 3.54% | 97 | No | 1.715 | 4.2562 | 219.706 | 🎯 **Tensile Strength** |
| 5 | `14 BC` | `float64` | 102/113 | 9.73% | 94 | No | 2.181 | 5.0884 | 258.217 | 🎯 **Tensile Strength** |
| 6 | `21 N` | `float64` | 102/113 | 9.73% | 85 | No | 2.077 | 4.7013 | 238.571 | 🎯 **Tensile Strength** |
| 7 | `21 BC` | `float64` | 102/113 | 9.73% | 95 | No | 2.599 | 5.6967 | 289.088 | 🎯 **Tensile Strength** |
| 8 | `28 N` | `float64` | 102/113 | 9.73% | 96 | No | 2.362 | 5.2811 | 267.997 | 🎯 **Tensile Strength** |
| 9 | `28 BC` | `float64` | 102/113 | 9.73% | 88 | No | 2.725 | 6.2571 | 317.524 | 🎯 **Tensile Strength** |
| 10 | `56 N` | `float64` | 102/113 | 9.73% | 94 | No | 2.505 | 5.723 | 290.42 | 🎯 **Tensile Strength** |
| 11 | `56 BC` | `float64` | 102/113 | 9.73% | 96 | No | 3.146 | 6.8973 | 350.014 | 🎯 **Tensile Strength** |
| 12 | `90 N` | `float64` | 102/113 | 9.73% | 96 | No | 2.682 | 6.1407 | 311.618 | 🎯 **Tensile Strength** |
| 13 | `90 BC` | `float64` | 102/113 | 9.73% | 88 | No | 3.383 | 7.4967 | 380.428 | 🎯 **Tensile Strength** |

---

## 2. Cross-Dataset Metadata & Feature Identification

### 2.1 Concrete Type, Curing Age, Bacteria & Concentrations Identified

| Attribute | Observed Values | Present in Sheets | Significance |
| :--- | :--- | :--- | :--- |
| **Concrete Type** | `Normal Concrete` (NC), `Bacterial Concrete` (BC) | `Comp 6`, `Sheet1` (F & T), `Sheet2`, `Ten 6`, `c`, `f`, `t` | Core categorical input feature. |
| **Curing Age (Days)** | `7`, `14`, `21`, `28`, `56`, `90` | All sheets | Primary temporal numerical feature governing cement hydration and biomineralization. |
| **Cement Type** | `OPC 53 (UltraTech)` | `Comp 6`, `Sheet1` (F & T) | Ordinary Portland Cement 53 grade. Currently **constant across all rows**. |
| **Bacterial Species** | `Bacillus subtilis` | `Comp 6`, `Sheet1` (F & T) | Ureolytic / calcite-precipitating endospore-forming bacteria. **Constant across all rows**. |
| **Bacterial Concentration** | `1,000,000 cells/mL` ($10^6$) | Main reading sheets | In raw sample sheets, concentration is fixed at $10^6$. In summary sheets (`c`, `f`, `t`, `RCPT`, `Water Absorption`), concentrations span `Control (0)`, $10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9$ cells/mL. |
| **Durability: RCPT** | `1900` to `3100` Coulombs | `RCPT` in C Strength | Rapid Chloride Permeability Test: shows lowest permeability at $10^5$ cells/mL ($1900$ C) vs Control ($3100$ C). |
| **Durability: Water Absorption** | `4.3%` to `5.8%` | `Water Absorption` in C Strength | Minimum water absorption observed at $10^5$ cells/mL ($4.3%$) vs $10^3$ ($5.8%$). |

### 2.2 Target Variables Identified

1. **Compressive Strength (MPa)**:
   - Primary target in `C Strenth Reading 1000000.xlsx`.
   - Raw individual sample readings located in `Comp 6` (600 rows paired) and `Sheet2` (100 samples across 12 conditions).
   - Range: Normal concrete 7d min **15.57 MPa** to Bacterial concrete 90d max **44.16 MPa**.

2. **Flexural Strength (MPa)**:
   - Primary target in `F Strength Reading 1000000.xlsx`.
   - Raw individual sample readings in `Sheet1` (1200 rows) and `Sheet2` (100 samples across 12 conditions).
   - Range: Normal concrete 7d min **2.21 MPa** to Bacterial concrete 90d max **6.39 MPa** (summary) / **6.19 MPa** (samples).

3. **Split Tensile Strength (MPa)**:
   - Primary target in `T Strenth Reading 1000000.xlsx`.
   - Raw individual sample readings in `Sheet1` (1200 rows) and `Ten 6` (100 samples across 12 conditions).
   - Range: Normal concrete 7d min **1.44 MPa** to Bacterial concrete 90d max **4.25 MPa**.

### 2.3 Identification of Raw vs. Derived/Calculated Tables

- **Raw Experimental Measurements**:
  - `C Strenth Reading 1000000.xlsx` -> `Comp 6`: 600 side-by-side rows representing 100 concrete cylinder/cube tests per curing age for both Control and Bacterial mix.
  - `F Strength Reading 1000000.xlsx` -> `Sheet1`: 1200 rows (unpivoted long format).
  - `F Strength Reading 1000000.xlsx` -> `Sheet2` (rows 5 to 104): 100 physical beam flexural fracture tests per condition.
  - `T Strenth Reading 1000000.xlsx` -> `Sheet1`: 1200 rows (unpivoted long format).
  - `T Strenth Reading 1000000.xlsx` -> `Ten 6` (rows 4 to 103): 100 physical Brazilian cylinder split tensile tests per condition.
- **Derived / Summary / Benchmark Tables**:
  - `Sheet2` / `Ten 6` (rows 105 onwards): Contain `SUM` formulas and mean calculations (`=AVERAGE(...)` / hardcoded sums up to 4066.08).
  - `Final Result`, `105 c`, `c`, `f`, `t`: Macro-level laboratory summary tables comparing relative gain (%) and average strengths across bacteria dilution series ($10^3$ to $10^8$).
  - `RCPT` and `Water Absorption`: Durability single-value measurements across doses.

---

## 3. Data Quality Issues, Anomalies & Potential Risks

### ⚠️ Critical Issue 1: Missing Data in `F Strength Reading 1000000.xlsx` (`Sheet1`)
- In `Sheet1` of `F Strength Reading 1000000.xlsx`, rows 1001 to 1200 have **200 completely empty (`NaN`) target values**:
  - `Bacterial Concrete` at **56 Days**: 100 missing rows (rows 1001–1100).
  - `Bacterial Concrete` at **90 Days**: 100 missing rows (rows 1101–1200).
- **Root Cause Discovered in Audit**: The data transcriber unpivoted `Sheet2` into `Sheet1` but stopped at 28 days for Bacterial Concrete.
- **Remedy Available**: The original measurements for `56 BC` and `90 BC` **DO EXIST** in `Sheet2` of the same workbook (100 complete readings for each condition). When training/cleaning is permitted, these values can be mapped from `Sheet2` without data loss.

### ⚠️ Critical Issue 2: Summary Rows Embedded in Matrix Sheets (`Sheet2` and `Ten 6`)
- `Sheet2` in `C Strenth Reading 1000000.xlsx`, `Sheet2` in `F Strength Reading 1000000.xlsx`, and `Ten 6` in `T Strenth Reading 1000000.xlsx` contain **100 data rows (samples 1 to 100)** followed immediately by **10–12 summary rows** (column sums, averages, curing age markers).
- Example: In `C Strength Sheet2`, row 105 contains column sums (`1934.44` for 7 N, `4066.08` for 90 BC).
- **Risk**: If these sheets are loaded directly into machine learning pipelines without row slicing, summary sums will be treated as physical sample strengths, resulting in catastrophic model distortion.

### ⚠️ Critical Issue 3: Zero-Variance (Constant) Feature Columns
- Across all primary raw sheets (`Comp 6`, `F Sheet1`, `T Sheet1`):
  - `Cement` is strictly `'OPC 53 (UltraTech)'` (variance = 0).
  - `Bacteria` is strictly `'Bacillus subtilis'` (variance = 0).
  - `Bacterial Concentration (cells/mL)` is strictly `1,000,000` (variance = 0).
- **Implication**: These features cannot provide predictive gradient within the $10^6$ dataset alone. If multi-concentration data (from sheets `c`, `f`, `t`) is later incorporated, concentration will become an active numerical feature.

### ⚠️ Critical Issue 4: Schema Discrepancy Across Target Workbooks
- `T Strength` (`Sheet1`) and `F Strength` (`Sheet1`) use an **unpivoted long format** (1200 rows with a `'Concrete Type'` column separating Normal and Bacterial).
- `C Strength` (`Comp 6`) uses a **wide side-by-side format** (600 rows where Column 5 is Normal Concrete Compressive Strength and Column 6 is Bacterial Concrete Compressive Strength, with a 2-level header).
- **Harmonization Required**: For unified multi-task or tabular modeling, `Comp 6` must be reshaped to match the long schema of F and T.

### ⚠️ Critical Issue 5: Duplicate Feature Tuples & Measurement Repetition
- Because input features (`Concrete Type`, `Curing Age`, `Cement`, `Bacteria`, `Concentration`) are identical for batches of 100 test samples, rows sharing identical strength measurements appear as duplicate rows (e.g. 111 duplicate rows in `T Sheet1`, 51 duplicate non-null rows in `F Sheet1`).
- These are legitimate experimental measurement repetitions, not artificial duplicate recordings, but must be treated carefully to avoid cross-fold data contamination during train/test splitting.

### ⚠️ Critical Issue 6: Data Leakage Risks
1. **Leakage from Summary Sheets**: Sheets `Final Result`, `105 c`, `c`, `f`, `t` contain average values calculated from test specimens. If summary sheet rows are merged with individual sample rows, aggregate target statistics will leak directly into training.
2. **Leakage from Paired Batch Splitting**: Test specimens were produced in concurrent laboratory casting batches. If individual specimens are shuffled randomly with K-fold CV rather than stratified or grouped, test sets will share batch variance with training sets.

---

## 4. Verification of Operational Constraints

- [x] **Constraint 15**: **No datasets merged**. Each file and sheet was audited strictly in isolation.
- [x] **Constraint 16**: **No model trained**. No machine learning, TabPFN fitting, or statistical modeling was initiated.
- [x] **Constraint 17**: **No rows or columns removed**. All original files in `data/` remain 100% unaltered and intact.
- [x] **Constraint 18**: **Artifacts generated**:
  - Audit report: [`reports/dataset_audit.md`](file:///g:/Projects/Concrete%20testing/reports/dataset_audit.md)
  - Schema metadata CSV: [`reports/dataset_schema.csv`](file:///g:/Projects/Concrete%20testing/reports/dataset_schema.csv)

---

## 5. Concise Synthesis & Next Steps

| Item | Findings / Summary |
| :--- | :--- |
| **Total Files** | 3 Excel workbooks (`C Strenth Reading 1000000.xlsx`, `F Strength Reading 1000000.xlsx`, `T Strenth Reading 1000000.xlsx`) |
| **Total Sheets** | 15 total sheets (11 in C, 2 in F, 2 in T) |
| **Total Rows per Dataset** | `C Comp 6`: 600 data rows (paired N & BC); `F Sheet1`: 1200 rows; `T Sheet1`: 1200 rows. (Sample matrix sheets `Sheet2` & `Ten 6`: 100 test samples each) |
| **Available Mechanical Properties** | Compressive Strength (15.57 – 44.16 MPa), Flexural Strength (2.21 – 6.19 MPa), Split Tensile Strength (1.44 – 4.25 MPa) |
| **Available Bacterial Concentrations** | $10^6$ cells/mL across all raw sample datasets. Concentrations $10^3, 10^4, 10^5, 10^7, 10^8$ available in summary sheets `c`, `f`, `t`. |
| **Missing / Constant Variables** | **Missing**: 200 values in `F Sheet1` (56d BC & 90d BC). **Constants**: `Cement` ('OPC 53'), `Bacteria` ('Bacillus subtilis'), `Bacterial Concentration` ($10^6$) |
| **Potential Data-Quality Issues** | Transcriber truncation in `F Sheet1`, trailing summary sum rows in matrix sheets, schema divergence (paired columns in C vs long rows in F and T) |
| **Possible Data Leakage Risks** | Merging summary sheets with raw specimen sheets; random train/test splitting across identical sample batches |