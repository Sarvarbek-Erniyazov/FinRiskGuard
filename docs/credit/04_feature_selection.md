# 🟠 Credit Scoring — Feature Selection

**Source:** `src/features/home_credit/feature_selector.py`  
**Pipeline:** `src/pipelines/credit_pipeline.py` — Stage 5

[← Feature Engineering](03_feature_engineering.md) | [← Back to README](../../README.md) | [→ Leakage Audit](05_leakage.md)

---

## Overview

195 columns after feature engineering → **105 final features** for training.

Two-step process:
```
Step 1: Correlation filter (threshold=0.95)  → 195 → 165 cols
Step 2: MI + XGB union (top_k=70)            → 105 final
```

> First run: top_k=60 → 93 features. Second run: top_k raised to 70 → **105 final** (target encoding + enhanced aggregations added more high-quality features).

Runtime: ~3.5 minutes (MI: ~3 min · XGB: ~2 sec · on 246K rows)

---

## Step 1 — Correlation Filter

**Problem (from EDA):** Building feature groups (`_AVG`, `_MODE`, `_MEDI`) measure the same property in three scales — near-perfect correlation with each other. Keeping all three adds noise without signal.

**Solution:** Remove one of each pair with correlation > 0.95:

```
Correlation filter: threshold = 0.95
Dropped 30 correlated features
Sample dropped: ['AMT_GOODS_PRICE', 'REGION_RATING_CLIENT_W_CITY',
                 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
                 'YEARS_BEGINEXPLUATATION_MODE']...
Remaining features: 165
```

---

## Step 2 — Dual Selection: MI + XGB Union

**Why union (not rank)?**

Credit dataset has 122 raw features + 78 FE = 200 total (after corr filter: 165). MI and XGB capture different aspects:
- **MI** captures statistical dependency — finds features like `FLAG_CONT_MOBILE`, `FE_cc_count` that have non-linear relationships with default
- **XGB** captures model-relevant importance — finds `FE_ext_mean`, `FE_ext23_prod` that drive tree splits

Union ensures **neither method's unique findings are lost**. With only 165 features post-filter (vs 391 in fraud), union is feasible without excessive noise.

### Mutual Information (top_k=70)

```
MI selection: top_k = 70
Top 5 MI features: ['FLAG_CONT_MOBILE', 'FE_cc_count',
                    'FE_bureau_C_ratio', 'FE_cc_dpd_mean', 'FLAG_OWN_REALTY']
Selected: 70 features
```

### XGBoost Importance (top_k=70)

```python
XGBClassifier(
    n_estimators=200,
    scale_pos_weight=11,   # class imbalance ratio from EDA
    tree_method='hist',
)

Top 5 XGB features: ['FE_ext_mean', 'FE_ext23_prod',
                     'FE_ext_min', 'NAME_EDUCATION_TYPE', 'CODE_GENDER']
Selected: 70 features
```

### Union Combination

```
MI selected    : 70
XGB selected   : 70
Final selected : 105  (union)
```

### Top 20 Features by Average Rank

| Feature | MI score | XGB importance | MI rank | XGB rank | Avg rank |
|---|---|---|---|---|---|
| FLAG_DOCUMENT_3 | 0.0451 | 0.0118 | 9 | 10 | 9.5 |
| **EXT_SOURCE_1_isnan** | 0.0391 | 0.0135 | 14 | 8 | 11.0 |
| **FE_cc_utilization** | 0.0399 | 0.0118 | 13 | 11 | 12.0 |
| NAME_EDUCATION_TYPE | 0.0315 | 0.0177 | 20 | 4 | 12.0 |
| **FE_ext_mean** | 0.0236 | 0.0614 | 23 | **1** | 12.0 |
| FE_ext_min | 0.0220 | 0.0200 | 26 | 3 | 14.5 |
| **FE_ext23_prod** | 0.0209 | 0.0388 | 29 | 2 | 15.5 |
| CODE_GENDER | 0.0172 | 0.0172 | 37 | 5 | 21.0 |
| FE_ext_max | 0.0169 | 0.0170 | 38 | 6 | 22.0 |
| **FE_NAME_FAMILY_STATUS_freq** | 0.0340 | 0.0078 | 18 | 27 | 22.5 |
| **FE_ext13_prod** | 0.0185 | 0.0104 | 34 | 14 | 24.0 |
| **FE_annuity_credit_ratio** | 0.0193 | 0.0097 | 33 | 16 | 24.5 |
| **FE_NAME_EDUCATION_TYPE_freq** | 0.0317 | 0.0071 | 19 | 32 | 25.5 |
| REGION_RATING_CLIENT | 0.0216 | 0.0083 | 27 | 24 | 25.5 |
| FLAG_OWN_CAR | 0.0159 | 0.0109 | 39 | 13 | 26.0 |
| OWN_CAR_AGE | 0.0230 | 0.0073 | 25 | 30 | 27.5 |
| **FE_cc_count** | 0.0464 | 0.0057 | 2 | 61 | 31.5 |
| **FE_credit_goods_ratio** | 0.0063 | 0.0104 | 59 | 15 | 37.0 |
| **FE_bureau_C_ratio** | 0.0460 | 0.0053 | 3 | 75 | 39.0 |
| NAME_INCOME_TYPE | 0.0235 | 0.0058 | 24 | 55 | 39.5 |

**Bold** = engineered features. Note `EXT_SOURCE_1_isnan` (NaN flag) ranks #2 overall — confirming EDA finding that missing bureau score is itself a strong signal.

---

## Final Selection — 105 Features

| Split | Shape |
|---|---|
| Train | 246,008 × 107 (105 features + TARGET + SK_ID_CURR) |
| Val | 61,503 × 107 |
| Test | 48,744 × 106 |

Saved to `data/features/credit/`:
- `train_credit_features.parquet`
- `val_credit_features.parquet`
- `test_credit_features.parquet`

---

## Why top_k Was Raised from 60 → 70

First run (top_k=60): 93 features. Second run raised to 70 for two reasons:
1. Target encoding added 2 new high-quality features (`FE_ORGANIZATION_TYPE_target_enc`, `FE_OCCUPATION_TYPE_target_enc`)
2. Enhanced bureau aggregation added `FE_bureau_C_ratio`, `FE_bureau_overdue_12m`, `FE_bureau_overdue_3m` — new signals not in first run

Union with top_k=70 → **105 final** — all meaningful new features captured.

---

[← Feature Engineering](03_feature_engineering.md) | [← Back to README](../../README.md) | [→ Leakage Audit](05_leakage.md)