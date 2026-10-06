# AI-Based Prediction of Mechanical Properties of Normal and Bacterial Concrete

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![Model: TabPFN](https://img.shields.io/badge/Model-TabPFN%20v2.5-brightgreen.svg)](https://github.com/prior-labs/tabpfn)

A machine learning framework designed for predicting the mechanical properties (**Compressive Strength**, **Flexural Strength**, and **Split Tensile Strength**) of normal and bio-mineralized (*Bacillus subtilis*) bacterial concrete across curing ages ranging from 7 to 90 days.

---

## 📌 Project Overview

This repository implements an end-to-end data processing, validation, and predictive modeling pipeline. It evaluates classical machine learning algorithms alongside modern tabular foundation models (**TabPFN v2.5**) under rigorous, leakage-free validation regimes:
- **Scenario A:** Standard random baseline split
- **Scenario B:** Cohort-grouped cross-validation (GroupKFold across mix designs)
- **Scenario C:** Leave-Age-Out evaluation (Out-of-Distribution curing age generalization)

---

## 📁 Repository Structure

```
├── data/                                # Datasets and CV split allocations
│   ├── C Strenth Reading 1000000.xlsx   # Compressive strength raw sheets
│   ├── F Strength Reading 1000000.xlsx  # Flexural strength raw sheets
│   ├── T Strenth Reading 1000000.xlsx   # Split tensile strength raw sheets
│   ├── master_dataset.csv               # Unified master dataset (CSV)
│   ├── master_dataset.parquet           # Optimized master dataset (Parquet)
│   └── splits/                          # Locked evaluation splits & manifests
│       ├── development_cohorts.csv
│       ├── development_rows.csv
│       ├── final_test_cohorts.csv
│       ├── final_test_rows.csv
│       ├── grouped_cv_assignments.csv
│       ├── leave_age_out_assignments.csv
│       ├── scenario_a_random_baseline.csv
│       ├── scenario_b_cohort_grouped.csv
│       └── scenario_c_leave_age_out.csv
├── models/                              # Pretrained weights & model checkpoints
│   └── tabpfn-v2.5-regressor-v2.5_real.ckpt # TabPFN v2.5 checkpoint
├── notebooks/                           # Jupyter notebooks for experimentation
├── reports/                             # Markdown reports, schemas & EDA
│   ├── figures/                         # High-res diagnostic & benchmark plots
│   ├── baseline_model_comparison.csv    # Benchmark metrics table
│   ├── baseline_modeling_report.md      # Comprehensive modeling report
│   ├── dataset_audit.md                 # Raw data audit report
│   ├── eda_report.md                    # Exploratory data analysis
│   ├── master_dataset_design.md         # Schema & extraction specification
│   ├── tabpfn_training_report.md        # TabPFN training & fine-tuning report
│   └── validation_strategy_v3_1.md      # Leakage-free validation strategy
├── results/                             # Evaluation results & trained fold models
│   ├── models/                          # Exported fold models & metadata
│   └── predictions/                     # Out-of-fold and test predictions
├── src/                                 # Source code
│   ├── modeling/                        # Training pipelines & baselines
│   │   └── baseline_training.py
│   ├── validation/                      # Validation protocols & leakage checkers
│   │   ├── cohort_analyzer.py
│   │   ├── evaluator.py
│   │   ├── feature_policy.py
│   │   ├── leakage_checker.py
│   │   ├── robust_grouped_cv.py
│   │   └── split_generator.py
│   ├── build_and_validate_master.py     # Master dataset generator
│   └── perform_eda.py                   # EDA & visualization generator
├── requirements.txt                     # Project dependencies
└── .gitignore                           # Git ignore rules
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Radheya4885/AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete.git
   cd AI-Based-Prediction-of-Mechanical-Properties-of-Normal-and-Bacterial-Concrete
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage & Workflow

### 1. Master Dataset Construction & Validation
Reconstruct and validate the canonical master dataset from raw experimental Excel workbooks:
```bash
python src/build_and_validate_master.py
```

### 2. Exploratory Data Analysis & Integrity Checks
Run full statistical analysis, distributions, and correlation maps:
```bash
python src/perform_eda.py
```

### 3. Baseline & Foundation Model Training
Train and benchmark regression baselines (Ridge, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, TabPFN):
```bash
python src/modeling/baseline_training.py
```

### 4. Run Validation Suite
Run unit tests for leakage detection and cross-validation assertions:
```bash
python src/validation/test_validation_v3_1.py
```

---

## 📊 Evaluation & Benchmarks

The project benchmarks performance across:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

Full tabular summaries and figures are available in [`reports/baseline_modeling_report.md`](reports/baseline_modeling_report.md) and [`reports/figures/`](reports/figures/).

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
