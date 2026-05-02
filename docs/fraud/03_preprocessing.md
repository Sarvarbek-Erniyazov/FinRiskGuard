# 🔴 Fraud Detection — Preprocessing

**Source:** `src/data/ieee_cis/preprocessor.py`  
**Pipeline:** `src/pipelines/fraud_pipeline.py` — Stage 4 (runs AFTER feature engineering)

[← Feature Engineering](02_feature_engineering.md) | [← Back to README](../../README.md) | [→ Feature Selection](04_feature_selection.md)

---

## Pipeline Overview

Preprocessing runs **after** feature engineering. All steps fitted on train only — val and test apply saved artifacts.

```
Input (post-FE): 483 columns
      ↓
Step 1: Drop high-missing columns (>90%)     → -12 cols
Step 2: Drop redundant columns               →  -8 cols
Step 3: Fix FE dtype mismatch                →   0 cols (dtype fix only)
Step 4: NaN flags                            → +15 cols
Step 5: Encode M columns                     →   0 cols (in-place)
Step 6: D column imputation (card1 group)    →   0 cols (in-place)
Step 7: Numerical imputation (median)        →   0 cols (in-place)
Step 8: Categorical imputation (mode)        →   0 cols (in-place)
Step 9: OrdinalEncoder                       →   0 cols (in-place)
      ↓
Output: 478 columns
```

| | Train | Val | Test |
|---|---|---|---|
| Input | 472,432 × 483 | 118,108 × 483 | 506,691 × 481 |
| Output | 472,432 × **478** | 118,108 × **478** | 506,691 × **499** |

> Test has more columns (499) because 13 identity columns absent in Kaggle test CSV are added as placeholders before OrdinalEncoder.

---

## Step 1 — Drop High-Missing Columns

**Problem (from EDA):** 12 columns exceed 90% missing — pure noise, no predictive signal.

**Decision:** Drop list derived from EDA, saved in `prep_artifacts.pkl`, applied identically to val/test.

```
Dropped 12 columns (>90% missing):
['dist2', 'D7', 'id_07', 'id_08', 'id_18', 'id_21',
 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27']
```

![Missing Values](../../outputs/figures/fraud/eda/02_missing/missing_overview.png)

---

## Step 2 — Drop Redundant Columns

**Problem:** Several C and V columns have near-perfect correlation to other columns — identical information, double computation.

```
Dropped 8 redundant columns:
['C1', 'C4', 'C8', 'C12', 'V242', 'V244', 'V49', 'V90']
```

---

## Step 3 — FE Dtype Fix

**Problem:** Card aggregation features created as `object` dtype due to pandas groupby behavior — silently breaks downstream numerical operations.

**Fix:**
```python
# 6 columns converted to float64:
['FE_card1_amt_mean', 'FE_card1_amt_std', 'FE_card1_amt_count',
 'FE_card1a1_amt_mean', 'FE_card1a1_amt_std', 'FE_card1a1_amt_count']
```

---

## Step 4 — NaN Flags

**Problem (from EDA):** D column missingness is non-random — NaN in D columns often signals "no prior transaction history," which correlates with fraud. Simply imputing destroys this signal.

**Solution:** Create binary flag **before** imputation:
```python
# 15 NaN flag columns added:
['dist1_isnan', 'D1_isnan', 'D2_isnan', 'D3_isnan', 'D4_isnan',
 'D5_isnan', 'D6_isnan', 'D8_isnan', 'D9_isnan', 'D10_isnan',
 'D11_isnan', 'D12_isnan', 'D13_isnan', 'D14_isnan', 'D15_isnan']
```

SHAP analysis confirmed multiple NaN flags (`D6_isnan`, `D2_isnan`, `D3_isnan`, `D5_isnan`) appear in the top 30 features by average rank.

---

## Step 5 — M Column Encoding

**Problem:** M columns contain `'T'`, `'F'`, and `NaN` — three distinct states. Standard boolean conversion loses the NaN signal.

**Solution:** Three-value encoding preserving NaN as information:
```python
M columns : T=1, F=0, NaN=-1

# M4 special case — ordinal (3 levels from EDA):
M4 : M0=0, M1=1, M2=2
```

---

## Step 6 — D Column Imputation

**Problem:** D columns have 50–85% missing. Global median imputation is wrong — D values vary significantly by card (different cards have different transaction histories).

**Solution:** Impute by `card1` group median, with global median fallback for unseen cards:
```python
# For each D column:
# 1. Compute median per card1 group → save to d_medians dict
# 2. Fill NaN with card1 group median
# 3. Remaining NaN → global median fallback

[TRAIN] Imputed 14 D columns by card1 group median
[TEST]  Applied card1 group imputation to 14 D columns
```

---

## Step 7 — Numerical Imputation

**Problem:** 370 numerical columns have scattered missing values after D columns are handled.

**Solution:** Median imputation — train medians saved, applied to val/test:
```
[TRAIN] Imputed 370 numerical columns with median
[TEST]  Applied median imputation to 370 numerical columns
```

---

## Step 8 — Categorical Imputation

**Problem:** 20 categorical columns have missing values.

**Solution:** Mode imputation — train modes saved, applied to val/test:
```
[TRAIN] Imputed 20 categorical columns with mode
[TEST]  Applied mode imputation to 20 categorical columns
```

---

## Step 9 — OrdinalEncoder

**Problem:** Gradient boosting models require numeric input. 24 categorical columns remain after imputation.

**Solution:**
```python
OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1,        # unseen categories in test → -1
    encoded_missing_value=-2
)

# Fitted on 24 columns:
['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
 'id_12', 'id_15', 'id_16', 'id_28', 'id_29', 'id_30', 'id_31',
 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
 'DeviceType', 'DeviceInfo', 'card1_addr1', 'card_full',
 'FE_uid', 'FE_uid_ext']
```

`unknown_value=-1` ensures test set categories not seen during training are handled gracefully without errors.

**Test set edge case:** 13 identity columns absent in Kaggle test CSV added as `'missing'` placeholder before encoding:
```
[TEST] Added 13 missing columns with placeholder:
['id_12', 'id_15', 'id_16', 'id_28', 'id_29', 'id_30', 'id_31',
 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']
```

---

## Saved Artifacts

All artifacts saved to `outputs/models/fraud/prep_artifacts.pkl`:

| Artifact | Contents |
|---|---|
| `drop_cols` | 12 high-missing column names |
| `nan_flag_cols` | 15 NaN flag column names |
| `num_fills` | 370 column → train median dict |
| `cat_fills` | 20 column → train mode dict |
| `d_medians` | per card1 group medians for 14 D cols |
| `encoder` | fitted OrdinalEncoder (24 columns) |

---

[← Feature Engineering](02_feature_engineering.md) | [← Back to README](../../README.md) | [→ Feature Selection](04_feature_selection.md)