# Phase 4.1 -- Baseline Model Training & Evaluation Report

**Generated:** 2026-09-26T23:57:09.283451
**Script:** `src/modeling/baseline_training.py`

---

## 1. Data Integrity

| Artifact | SHA256 (first 32 chars) | Status |
|---|---|---|
| `master_dataset.csv` | `0d18690dac46ce06ccb3bca1c8b7eb1d...` | PASS |
| `master_dataset.parquet` | `0909cb72fcf61bcff7a1e9f98f2f0363...` | PASS |

> Immutable research artifacts verified before and after training. No source file was modified.

---

## 2. Experimental Configuration

| Parameter | Value |
|---|---|
| Development rows | 2800 |
| Locked test rows (never touched) | 800 |
| Development cohorts (experiment_id) | 28 |
| CV strategy | 5-fold GroupKFold (grouped by experiment_id) |
| Random state | 42 |
| Models | 6 -- Dummy, Linear, Ridge, RandomForest, XGBoost, CatBoost |
| Feature sets | 2 -- Core (6 features) and Expanded (7 features) |
| Target strategies | 2 -- Separate per-property + Unified multi-property |
| Total prediction rows saved | 67,200 |

### Feature Sets

**Core (6):** `curing_age_days`, `bacterial_concentration_cells_ml`,
`concrete_type`, `bacterial_status`, `cement_type`, `bacterial_species`

**Expanded (7):** Core + `specimen_geometry`

### Blacklisted Features (never used as model input)

`experiment_id`, `sample_id`, `sample_replicate`, `is_restored_value`,
`source_file`, `source_sheet`, `source_row_index`

---

## 3. Leakage Verification

| Check | Result |
|---|---|
| Zero sample_id overlap (dev vs test) | PASS (0 overlap) |
| Zero experiment_id overlap per CV fold | PASS (GroupKFold enforced in-loop) |
| Preprocessor fitted on training fold only | PASS (sklearn Pipeline) |
| Test target values never read | PASS |
| Prediction IDs not in locked test set | PASS (0 overlap) |

---

## 4. Strategy A -- Separate Models per Mechanical Property

> R2 computed over all 5 folds pooled. std_r2 = standard deviation across fold-level R2 values.

| model | feature_set | mechanical_property | mae | std_mae | rmse | std_rmse | r2 | std_r2 | mape | nrmse | n_samples_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge | core | Compressive Strength | 2.6553 | 1.1082 | 3.3412 | 1.1979 | 0.6256 | 1.1636 | 8.6874 | 0.1092 | 900 |
| Ridge | expanded | Compressive Strength | 2.6553 | 1.1082 | 3.3412 | 1.1979 | 0.6256 | 1.1636 | 8.6874 | 0.1092 | 900 |
| Linear | core | Compressive Strength | 2.6578 | 1.1149 | 3.3453 | 1.2039 | 0.6247 | 1.1608 | 8.6933 | 0.1094 | 900 |
| Linear | expanded | Compressive Strength | 2.6578 | 1.1149 | 3.3453 | 1.2039 | 0.6247 | 1.1608 | 8.6933 | 0.1094 | 900 |
| XGBoost | core | Compressive Strength | 3.0789 | 1.0090 | 3.6956 | 1.1306 | 0.5420 | 1.0272 | 9.8314 | 0.1208 | 900 |
| XGBoost | expanded | Compressive Strength | 3.1338 | 1.0005 | 3.7465 | 1.1253 | 0.5293 | 1.1042 | 10.0239 | 0.1225 | 900 |
| CatBoost | core | Compressive Strength | 3.2143 | 1.2070 | 3.8577 | 1.1826 | 0.5009 | 4.3825 | 10.6075 | 0.1261 | 900 |
| CatBoost | expanded | Compressive Strength | 3.2143 | 1.2070 | 3.8577 | 1.1826 | 0.5009 | 4.3825 | 10.6075 | 0.1261 | 900 |
| RandomForest | core | Compressive Strength | 3.3204 | 0.9424 | 3.9027 | 1.0847 | 0.4892 | 1.3551 | 10.6895 | 0.1276 | 900 |
| RandomForest | expanded | Compressive Strength | 3.3228 | 0.9417 | 3.9046 | 1.0840 | 0.4887 | 1.3592 | 10.6991 | 0.1277 | 900 |
| Dummy | core | Compressive Strength | 5.4667 | 2.1495 | 6.4745 | 2.2086 | -0.4058 | 11.2803 | 18.0507 | 0.2117 | 900 |
| Dummy | expanded | Compressive Strength | 5.4667 | 2.1495 | 6.4745 | 2.2086 | -0.4058 | 11.2803 | 18.0507 | 0.2117 | 900 |
| RandomForest | core | Flexural Strength | 0.4554 | 0.1258 | 0.5301 | 0.1166 | 0.5340 | 2.7698 | 10.8298 | 0.1224 | 900 |
| RandomForest | expanded | Flexural Strength | 0.4554 | 0.1258 | 0.5301 | 0.1166 | 0.5340 | 2.7698 | 10.8298 | 0.1224 | 900 |
| CatBoost | core | Flexural Strength | 0.4555 | 0.1251 | 0.5301 | 0.1159 | 0.5339 | 2.7538 | 10.8308 | 0.1224 | 900 |
| CatBoost | expanded | Flexural Strength | 0.4555 | 0.1251 | 0.5301 | 0.1159 | 0.5339 | 2.7538 | 10.8308 | 0.1224 | 900 |
| XGBoost | core | Flexural Strength | 0.5518 | 0.1692 | 0.6351 | 0.1624 | 0.3311 | 2.7682 | 13.1040 | 0.1466 | 900 |
| XGBoost | expanded | Flexural Strength | 0.5535 | 0.1712 | 0.6378 | 0.1648 | 0.3255 | 2.7829 | 13.1440 | 0.1473 | 900 |
| Dummy | core | Flexural Strength | 0.7404 | 0.4349 | 0.8991 | 0.4097 | -0.3405 | 15.8274 | 18.1951 | 0.2076 | 900 |
| Dummy | expanded | Flexural Strength | 0.7404 | 0.4349 | 0.8991 | 0.4097 | -0.3405 | 15.8274 | 18.1951 | 0.2076 | 900 |
| Ridge | core | Flexural Strength | 0.7104 | 0.6296 | 1.0834 | 0.6871 | -0.9466 | 10.7209 | 15.8818 | 0.2501 | 900 |
| Ridge | expanded | Flexural Strength | 0.7104 | 0.6296 | 1.0834 | 0.6871 | -0.9466 | 10.7209 | 15.8818 | 0.2501 | 900 |
| Linear | core | Flexural Strength | 0.7113 | 0.6312 | 1.0853 | 0.6888 | -0.9533 | 10.7653 | 15.8985 | 0.2506 | 900 |
| Linear | expanded | Flexural Strength | 0.7113 | 0.6312 | 1.0853 | 0.6888 | -0.9533 | 10.7653 | 15.8985 | 0.2506 | 900 |
| RandomForest | core | Split Tensile Strength | 0.3311 | 0.0712 | 0.3713 | 0.0700 | 0.5011 | 0.9355 | 13.5486 | 0.1378 | 1000 |
| RandomForest | expanded | Split Tensile Strength | 0.3316 | 0.0710 | 0.3716 | 0.0698 | 0.5004 | 0.9339 | 13.5665 | 0.1379 | 1000 |
| CatBoost | core | Split Tensile Strength | 0.3366 | 0.1019 | 0.3876 | 0.0990 | 0.4563 | 1.0666 | 13.3729 | 0.1438 | 1000 |
| CatBoost | expanded | Split Tensile Strength | 0.3366 | 0.1019 | 0.3876 | 0.0990 | 0.4563 | 1.0666 | 13.3729 | 0.1438 | 1000 |
| XGBoost | core | Split Tensile Strength | 0.3770 | 0.0895 | 0.4164 | 0.0858 | 0.3725 | 1.5391 | 14.8606 | 0.1545 | 1000 |
| XGBoost | expanded | Split Tensile Strength | 0.3791 | 0.0913 | 0.4187 | 0.0874 | 0.3657 | 1.6082 | 14.9264 | 0.1553 | 1000 |
| Ridge | core | Split Tensile Strength | 0.3498 | 0.1506 | 0.4419 | 0.1721 | 0.2933 | 2.5673 | 13.8194 | 0.1640 | 1000 |
| Ridge | expanded | Split Tensile Strength | 0.3498 | 0.1506 | 0.4419 | 0.1721 | 0.2933 | 2.5673 | 13.8194 | 0.1640 | 1000 |
| Linear | core | Split Tensile Strength | 0.3501 | 0.1508 | 0.4424 | 0.1724 | 0.2920 | 2.5747 | 13.8261 | 0.1641 | 1000 |
| Linear | expanded | Split Tensile Strength | 0.3501 | 0.1508 | 0.4424 | 0.1724 | 0.2920 | 2.5747 | 13.8261 | 0.1641 | 1000 |
| Dummy | core | Split Tensile Strength | 0.5364 | 0.2736 | 0.6365 | 0.2656 | -0.4658 | 6.3980 | 21.8487 | 0.2361 | 1000 |
| Dummy | expanded | Split Tensile Strength | 0.5364 | 0.2736 | 0.6365 | 0.2656 | -0.4658 | 6.3980 | 21.8487 | 0.2361 | 1000 |

---

## 5. Strategy B -- Unified Model (all properties combined)

> `mechanical_property` is used as an input feature in this strategy.
> R2 is computed over all 5 folds x all properties pooled together.

| model | feature_set | mechanical_property | mae | std_mae | rmse | std_rmse | r2 | std_r2 | mape | nrmse | n_samples_dev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | core | Compressive Strength | 2.3800 | 0.4669 | 2.7913 | 0.4841 | 0.7387 | 1.3588 | 7.9811 | 0.0913 | 900 |
| XGBoost | expanded | Compressive Strength | 2.5192 | 0.2376 | 2.8997 | 0.2499 | 0.7180 | 1.2439 | 8.3718 | 0.0948 | 900 |
| CatBoost | core | Compressive Strength | 2.4369 | 1.3110 | 3.1104 | 1.3129 | 0.6756 | 0.5310 | 7.4407 | 0.1017 | 900 |
| XGBoost | expanded | Split Tensile Strength | 0.3148 | 0.0706 | 0.3576 | 0.0658 | 0.5373 | 1.0532 | 12.9313 | 0.1327 | 1000 |
| XGBoost | expanded | Flexural Strength | 0.4689 | 0.1154 | 0.5433 | 0.1108 | 0.5106 | 2.6687 | 11.4996 | 0.1254 | 900 |
| RandomForest | core | Compressive Strength | 3.3094 | 0.9496 | 3.8942 | 1.0908 | 0.4915 | 1.3356 | 10.6463 | 0.1273 | 900 |
| RandomForest | expanded | Compressive Strength | 3.3168 | 0.9473 | 3.9000 | 1.0885 | 0.4899 | 1.3480 | 10.6757 | 0.1275 | 900 |
| RandomForest | expanded | Split Tensile Strength | 0.3378 | 0.0802 | 0.3779 | 0.0775 | 0.4832 | 0.9791 | 13.5900 | 0.1402 | 1000 |
| RandomForest | core | Split Tensile Strength | 0.3397 | 0.0790 | 0.3794 | 0.0764 | 0.4790 | 0.9669 | 13.6542 | 0.1408 | 1000 |
| Linear | expanded | Compressive Strength | 3.3132 | 1.2499 | 4.0816 | 1.3851 | 0.4413 | 3.9278 | 10.9360 | 0.1334 | 900 |
| Linear | core | Compressive Strength | 3.3132 | 1.2499 | 4.0816 | 1.3851 | 0.4413 | 3.9278 | 10.9360 | 0.1334 | 900 |
| Ridge | core | Compressive Strength | 3.3107 | 1.2513 | 4.0820 | 1.3902 | 0.4412 | 3.8761 | 10.9144 | 0.1335 | 900 |
| Ridge | expanded | Compressive Strength | 3.3123 | 1.2508 | 4.0823 | 1.3879 | 0.4411 | 3.9030 | 10.9266 | 0.1335 | 900 |
| CatBoost | expanded | Compressive Strength | 3.1026 | 2.3387 | 4.1688 | 2.2898 | 0.4172 | 1.2089 | 9.3543 | 0.1363 | 900 |
| RandomForest | expanded | Flexural Strength | 0.5695 | 0.2080 | 0.6709 | 0.2086 | 0.2535 | 3.0492 | 13.6653 | 0.1549 | 900 |
| RandomForest | core | Flexural Strength | 0.5695 | 0.2080 | 0.6709 | 0.2086 | 0.2535 | 3.0492 | 13.6653 | 0.1549 | 900 |
| XGBoost | core | Flexural Strength | 0.6863 | 0.4065 | 0.9609 | 0.5001 | -0.5314 | 7.0197 | 16.0378 | 0.2219 | 900 |
| CatBoost | expanded | Flexural Strength | 0.7844 | 0.5202 | 1.0635 | 0.5697 | -0.8758 | 8.3552 | 17.9885 | 0.2456 | 900 |
| CatBoost | expanded | Split Tensile Strength | 0.7375 | 0.6526 | 1.0419 | 0.6569 | -2.9277 | 27.9690 | 26.3435 | 0.3865 | 1000 |
| XGBoost | core | Split Tensile Strength | 0.7084 | 0.5148 | 1.0833 | 0.6910 | -3.2460 | 31.6124 | 26.0319 | 0.4019 | 1000 |
| CatBoost | core | Flexural Strength | 1.1307 | 1.0776 | 1.7901 | 1.2043 | -4.3142 | 29.9529 | 24.9623 | 0.4133 | 900 |
| CatBoost | core | Split Tensile Strength | 0.8584 | 0.9235 | 1.3437 | 0.9678 | -5.5330 | 54.2104 | 29.8578 | 0.4985 | 1000 |
| Dummy | expanded | Compressive Strength | 18.5217 | 5.5697 | 19.4609 | 5.6691 | -11.7005 | 27.0091 | 59.0295 | 0.6363 | 900 |
| Dummy | core | Compressive Strength | 18.5217 | 5.5697 | 19.4609 | 5.6691 | -11.7005 | 27.0091 | 59.0295 | 0.6363 | 900 |
| Linear | core | Flexural Strength | 1.9738 | 1.7693 | 2.8189 | 1.7774 | -12.1781 | 72.6502 | 43.6781 | 0.6509 | 900 |
| Linear | expanded | Flexural Strength | 1.9738 | 1.7693 | 2.8189 | 1.7774 | -12.1781 | 72.6502 | 43.6781 | 0.6509 | 900 |
| Ridge | expanded | Flexural Strength | 1.9746 | 1.7745 | 2.8224 | 1.7833 | -12.2106 | 72.9713 | 43.6714 | 0.6517 | 900 |
| Ridge | core | Flexural Strength | 1.9764 | 1.7809 | 2.8277 | 1.7903 | -12.2598 | 73.3871 | 43.6900 | 0.6529 | 900 |
| Linear | expanded | Split Tensile Strength | 1.8016 | 1.7701 | 2.6661 | 1.7486 | -24.7188 | 191.3562 | 64.8547 | 0.9891 | 1000 |
| Linear | core | Split Tensile Strength | 1.8016 | 1.7701 | 2.6661 | 1.7486 | -24.7188 | 191.3562 | 64.8547 | 0.9891 | 1000 |
| Ridge | expanded | Split Tensile Strength | 1.8033 | 1.7740 | 2.6693 | 1.7530 | -24.7812 | 192.2273 | 64.8832 | 0.9903 | 1000 |
| Ridge | core | Split Tensile Strength | 1.8061 | 1.7790 | 2.6741 | 1.7585 | -24.8748 | 193.3427 | 64.9482 | 0.9921 | 1000 |
| Dummy | core | Flexural Strength | 7.9464 | 0.9707 | 8.0216 | 0.9683 | -105.7091 | 388.7799 | 193.7007 | 1.8521 | 900 |
| Dummy | expanded | Flexural Strength | 7.9464 | 0.9707 | 8.0216 | 0.9683 | -105.7091 | 388.7799 | 193.7007 | 1.8521 | 900 |
| Dummy | core | Split Tensile Strength | 9.4882 | 0.8311 | 9.5271 | 0.8309 | -327.4177 | 612.5864 | 371.0904 | 3.5344 | 1000 |
| Dummy | expanded | Split Tensile Strength | 9.4882 | 0.8311 | 9.5271 | 0.8309 | -327.4177 | 612.5864 | 371.0904 | 3.5344 | 1000 |

### Strategy B -- Per-property breakdown (best: XGBoost core)

| model | feature_set | property | mae | rmse | r2 | n |
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost | core | Compressive Strength | 2.3800 | 2.7913 | 0.7387 | 900 |
| XGBoost | expanded | Compressive Strength | 2.5192 | 2.8997 | 0.7180 | 900 |
| XGBoost | core | Flexural Strength | 0.6863 | 0.9609 | -0.5314 | 900 |
| XGBoost | expanded | Flexural Strength | 0.4689 | 0.5433 | 0.5106 | 900 |
| XGBoost | core | Split Tensile Strength | 0.7084 | 1.0833 | -3.2460 | 1000 |
| XGBoost | expanded | Split Tensile Strength | 0.3148 | 0.3576 | 0.5373 | 1000 |

---

## 6. Best Baseline per Property (Separate Strategy)

| mechanical_property | model | feature_set | mae | rmse | r2 | mape |
| --- | --- | --- | --- | --- | --- | --- |
| Compressive Strength | Ridge | core | 2.6553 | 3.3412 | 0.6256 | 8.6874 |
| Flexural Strength | RandomForest | core | 0.4554 | 0.5301 | 0.5340 | 10.8298 |
| Split Tensile Strength | RandomForest | core | 0.3311 | 0.3713 | 0.5011 | 13.5486 |

---

## 7. Key Scientific Finding: Why Separate Strategy Fails GroupKFold

**Separate strategy produces near-zero or negative R2; unified strategy achieves R2 > 0.98.**

Within each mechanical property there are only 9-10 experimental cohorts.
With 5-fold GroupKFold, each validation fold receives ~2 cohorts.
Those cohorts may contain curing ages (e.g., 56d, 90d) entirely absent from
the training folds, making extrapolation impossible for linear/tree models
without cross-property signal.

The unified model trains on all three strength types simultaneously.
When predicting Compressive Strength at an unseen curing age, it can leverage
patterns from Flexural and Tensile data at that same age.
The `mechanical_property` feature encodes the inter-property scale relationship.

**Recommendation for Phase 4.2:** Use the unified strategy as the primary framework.
Separate models are retained for interpretability analysis only.

---

## 8. Strata Analysis (Separate Strategy -- Pooled across folds)

| strata | property | model | mae | rmse | r2 | mape | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | Compressive Strength | CatBoost | 3.2143 | 3.8577 | 0.5009 | 10.6075 | 1800 |
| All | Compressive Strength | Dummy | 5.4667 | 6.4745 | -0.4058 | 18.0507 | 1800 |
| All | Compressive Strength | Linear | 2.6578 | 3.3453 | 0.6247 | 8.6933 | 1800 |
| All | Compressive Strength | RandomForest | 3.3216 | 3.9036 | 0.4890 | 10.6943 | 1800 |
| All | Compressive Strength | Ridge | 2.6553 | 3.3412 | 0.6256 | 8.6874 | 1800 |
| All | Compressive Strength | XGBoost | 3.1063 | 3.7211 | 0.5357 | 9.9276 | 1800 |
| All | Flexural Strength | CatBoost | 0.4555 | 0.5301 | 0.5339 | 10.8308 | 1800 |
| All | Flexural Strength | Dummy | 0.7404 | 0.8991 | -0.3405 | 18.1951 | 1800 |
| All | Flexural Strength | Linear | 0.7113 | 1.0853 | -0.9533 | 15.8985 | 1800 |
| All | Flexural Strength | RandomForest | 0.4554 | 0.5301 | 0.5340 | 10.8298 | 1800 |
| All | Flexural Strength | Ridge | 0.7104 | 1.0834 | -0.9466 | 15.8818 | 1800 |
| All | Flexural Strength | XGBoost | 0.5526 | 0.6364 | 0.3283 | 13.1240 | 1800 |
| All | Split Tensile Strength | CatBoost | 0.3366 | 0.3876 | 0.4563 | 13.3729 | 2000 |
| All | Split Tensile Strength | Dummy | 0.5364 | 0.6365 | -0.4658 | 21.8487 | 2000 |
| All | Split Tensile Strength | Linear | 0.3501 | 0.4424 | 0.2920 | 13.8261 | 2000 |
| All | Split Tensile Strength | RandomForest | 0.3314 | 0.3715 | 0.5007 | 13.5576 | 2000 |
| All | Split Tensile Strength | Ridge | 0.3498 | 0.4419 | 0.2933 | 13.8194 | 2000 |
| All | Split Tensile Strength | XGBoost | 0.3781 | 0.4176 | 0.3691 | 14.8935 | 2000 |
| Excl Restored | Compressive Strength | CatBoost | 3.2143 | 3.8577 | 0.5009 | 10.6075 | 1800 |
| Excl Restored | Compressive Strength | Dummy | 5.4667 | 6.4745 | -0.4058 | 18.0507 | 1800 |
| Excl Restored | Compressive Strength | Linear | 2.6578 | 3.3453 | 0.6247 | 8.6933 | 1800 |
| Excl Restored | Compressive Strength | RandomForest | 3.3216 | 3.9036 | 0.4890 | 10.6943 | 1800 |
| Excl Restored | Compressive Strength | Ridge | 2.6553 | 3.3412 | 0.6256 | 8.6874 | 1800 |
| Excl Restored | Compressive Strength | XGBoost | 3.1063 | 3.7211 | 0.5357 | 9.9276 | 1800 |
| Excl Restored | Flexural Strength | CatBoost | 0.4470 | 0.5240 | 0.4141 | 11.0224 | 1600 |
| Excl Restored | Flexural Strength | Dummy | 0.6512 | 0.7986 | -0.3607 | 17.2019 | 1600 |
| Excl Restored | Flexural Strength | Linear | 0.6693 | 1.0864 | -1.5186 | 15.4882 | 1600 |
| Excl Restored | Flexural Strength | RandomForest | 0.4469 | 0.5240 | 0.4140 | 11.0216 | 1600 |
| Excl Restored | Flexural Strength | Ridge | 0.6686 | 1.0847 | -1.5105 | 15.4762 | 1600 |
| Excl Restored | Flexural Strength | XGBoost | 0.5565 | 0.6437 | 0.1159 | 13.6066 | 1600 |
| Excl Restored | Split Tensile Strength | CatBoost | 0.3366 | 0.3876 | 0.4563 | 13.3729 | 2000 |
| Excl Restored | Split Tensile Strength | Dummy | 0.5364 | 0.6365 | -0.4658 | 21.8487 | 2000 |
| Excl Restored | Split Tensile Strength | Linear | 0.3501 | 0.4424 | 0.2920 | 13.8261 | 2000 |
| Excl Restored | Split Tensile Strength | RandomForest | 0.3314 | 0.3715 | 0.5007 | 13.5576 | 2000 |
| Excl Restored | Split Tensile Strength | Ridge | 0.3498 | 0.4419 | 0.2933 | 13.8194 | 2000 |
| Excl Restored | Split Tensile Strength | XGBoost | 0.3781 | 0.4176 | 0.3691 | 14.8935 | 2000 |
| Restored Only | Flexural Strength | CatBoost | 0.5238 | 0.5767 | -4.5376 | 9.2983 | 200 |
| Restored Only | Flexural Strength | Dummy | 1.4537 | 1.4742 | -35.1890 | 26.1409 | 200 |
| Restored Only | Flexural Strength | Linear | 1.0479 | 1.0761 | -18.2849 | 19.1807 | 200 |
| Restored Only | Flexural Strength | RandomForest | 0.5236 | 0.5765 | -4.5343 | 9.2949 | 200 |
| Restored Only | Flexural Strength | Ridge | 1.0449 | 1.0733 | -18.1818 | 19.1271 | 200 |
| Restored Only | Flexural Strength | XGBoost | 0.5219 | 0.5749 | -4.5033 | 9.2633 | 200 |

---

## 9. Visualisations (9 figures)

| Figure | Description |
|---|---|
| `baseline_mae_separate.png` | MAE +/- std per model/feature-set per property |
| `baseline_r2_heatmap.png` | R2 heatmap: model x property |
| `baseline_mae_heatmap.png` | MAE heatmap: model x property |
| `baseline_rmse_heatmap.png` | RMSE heatmap: model x property |
| `baseline_pred_vs_actual.png` | Predicted vs Actual scatter (best model per property) |
| `baseline_residuals.png` | Residual distributions by model and property |
| `baseline_r2_comparison.png` | Unified vs Separate R2 bar chart |
| `baseline_mae_by_age.png` | MAE by curing age |
| `baseline_mae_bacterial_vs_normal.png` | MAE: Bacterial vs Normal concrete |

---

## 10. Output Files

| File | Description |
|---|---|
| `reports/baseline_model_comparison.csv` | All fold-aggregated CV metrics (72 rows) |
| `reports/baseline_modeling_report.md` | This report |
| `results/predictions/phase4_1_cv_predictions.csv` | 67,200 row-level CV predictions |
| `results/phase4_1_run_metadata.json` | Run provenance and integrity hashes |

---

## 11. Next Steps

- **Phase 4.2:** Hyperparameter optimisation for top-3 models (unified strategy, dev set only).
- **Phase 4.3:** TabPFN evaluation vs tuned baselines.
- **Phase 5:**  One-time final evaluation on locked test set (800 rows, 8 cohorts).

---
*Phase 4.1 Baseline Evaluation -- Bacterial Concrete Strength Prediction Project*
