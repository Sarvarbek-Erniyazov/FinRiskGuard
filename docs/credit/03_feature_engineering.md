# 🟠 Credit Scoring — Feature Engineering

**Source:** `src/features/home_credit/feature_engineer.py`  
**Pipeline:** `src/pipelines/credit_pipeline.py` — Stage 4 (runs AFTER preprocessing)

[← Preprocessing](02_preprocessing.md) | [← Back to README](../../README.md) | [→ Feature Selection](04_feature_selection.md)

---

## Results

| | Train | Val | Test |
|---|---|---|---|
| Input shape | 246,008 × 117 | 61,503 × 117 | 48,744 × 116 |
| Output shape | 246,008 × **195** | 61,503 × **195** | 48,744 × **194** |
| **FE features created** | **78** | 78 applied | 78 applied |

> First run created 67 features. Second run added target encoding + enhanced bureau/previous app aggregations → **78 final features**.

---

## 1. EXT_SOURCE Combination Features

**Problem (from EDA):** EXT_SOURCE_1/2/3 are the three strongest individual predictors. But they interact non-linearly — a borrower with two high scores and one low score behaves differently from one with three medium scores.

**Solution:** 8 combination features capturing different aspects of the three scores:

```python
FE_ext_mean    = mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)
FE_ext_min     = min(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)
FE_ext_max     = max(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)
FE_ext_std     = std(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)
FE_ext_sum     = sum(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)
FE_ext12_prod  = EXT_SOURCE_1 × EXT_SOURCE_2
FE_ext23_prod  = EXT_SOURCE_2 × EXT_SOURCE_3
FE_ext13_prod  = EXT_SOURCE_1 × EXT_SOURCE_3
```

`FE_ext_mean` became the **#1 most important feature** in the final model — confirmed by both XGBoost importance and SHAP.

---

## 2. Age & Employment Features

**Problem (from EDA):** `DAYS_BIRTH` is raw negative days — models cannot learn "age groups" from raw values. Younger borrowers (20-30) default at significantly higher rates.

**Solution:**
```python
FE_age_years           = abs(DAYS_BIRTH) / 365
FE_age_group           = pd.cut(age_years, bins=[0,25,35,45,55,100])
                         # 0=young, 1=25-35, 2=35-45, 3=45-55, 4=senior

FE_employment_ratio    = abs(DAYS_EMPLOYED) / (abs(DAYS_BIRTH) + 1)
                         # fraction of life spent employed

FE_registration_age_ratio = abs(DAYS_REGISTRATION) / (abs(DAYS_BIRTH) + 1)
                            # how early in life the account was registered
```

---

## 3. Credit & Income Ratio Features

**Problem (from EDA):** Absolute amounts (`AMT_CREDIT`, `AMT_ANNUITY`) are less informative than ratios — a $50K loan means something very different for someone earning $20K vs $200K.

**Solution:** 6 ratio features:
```python
FE_income_log           = log1p(AMT_INCOME_TOTAL)
FE_credit_income_ratio  = AMT_CREDIT  / (AMT_INCOME_TOTAL + 1)
FE_annuity_income_ratio = AMT_ANNUITY / (AMT_INCOME_TOTAL + 1)
FE_annuity_credit_ratio = AMT_ANNUITY / (AMT_CREDIT + 1)
FE_credit_goods_diff    = AMT_CREDIT  - AMT_GOODS_PRICE
FE_credit_goods_ratio   = AMT_CREDIT  / (AMT_GOODS_PRICE + 1)
```

`FE_annuity_credit_ratio` appeared in SHAP top features — monthly payment burden relative to total loan.

---

## 4. Frequency Encoding

**Problem:** 6 high-cardinality categoricals have 10–60+ unique values. OrdinalEncoder assigns arbitrary integers. Frequency encoding replaces each category with its prevalence in train.

**Solution:**
```python
# 6 columns frequency-encoded → 6 FE_*_freq features:
FE_ORGANIZATION_TYPE_freq
FE_OCCUPATION_TYPE_freq
FE_NAME_INCOME_TYPE_freq
FE_NAME_EDUCATION_TYPE_freq
FE_NAME_HOUSING_TYPE_freq
FE_NAME_FAMILY_STATUS_freq

[TRAIN] Frequency encoded 6 columns
[TEST]  Applied frequency encoding to 6 columns
```

---

## 5. Target Encoding (Leakage-Free)

**Problem:** `ORGANIZATION_TYPE` (58 unique values) and `OCCUPATION_TYPE` (18 unique) have high cardinality. Frequency encoding only captures category size — target encoding captures default rate per category, a stronger signal.

**Leakage prevention:** Cross-validated target encoding (CV=5) with smoothing — target is never computed from the full train set to avoid leakage:

```python
# CV=5, smoothing=20.0 — reduces overfitting on rare categories
FE_ORGANIZATION_TYPE_target_enc
FE_OCCUPATION_TYPE_target_enc

[TRAIN] Target encoded 2 columns (CV=5, smoothing=20.0)
[TEST]  Applied target encoding to 2 columns
```

Smoothing formula: `encoded = (count × category_mean + smoothing × global_mean) / (count + smoothing)` — rare categories are pulled toward the global mean.

---

## 6. Bureau Aggregation (Enhanced)

**Problem (from EDA):** 83% of applicants have bureau records. Bureau DPD history, overdue amounts, and credit utilization are strong default signals — but they exist in a separate 1.7M row table.

**Solution:** Aggregate per `SK_ID_CURR`:

```python
# Bureau + bureau_balance joined → 17 features:
FE_bureau_count           # number of past credits
FE_bureau_active_count    # currently active credits
FE_bureau_closed_count    # closed credits
FE_bureau_credit_sum      # total credit amount
FE_bureau_debt_sum        # total debt
FE_bureau_overdue_sum     # total overdue amount
FE_bureau_overdue_count   # number of overdue credits
FE_bureau_dpd_mean        # mean days past due
FE_bureau_dpd_max         # max days past due
FE_bureau_overdue_12m     # overdue count in last 12 months
FE_bureau_overdue_3m      # overdue count in last 3 months
FE_bureau_C_ratio         # ratio of closed months
FE_bureau_X_ratio         # ratio of unknown months
FE_bureau_max_streak      # max consecutive overdue months
FE_bureau_prolong_sum     # total prolongations

Bureau agg shape: (305,811 × 18)
```

---

## 7. Previous Application Aggregation (Enhanced)

**Problem (from EDA):** A borrower refused 4 times before being approved signals very different risk than one approved on first application.

**Solution:**
```python
FE_prev_count             # total past applications
FE_prev_approved_count    # approved count
FE_prev_refused_count     # refused count
FE_prev_approved_rate     # approved / total
FE_prev_refused_rate      # refused / total
FE_prev_credit_mean       # mean credit amount applied for
FE_prev_credit_sum        # total credit applied for
FE_prev_credit_trend      # last / first credit amount (escalating risk?)
FE_prev_last_decision_abs # days since last application
FE_prev_annuity_mean      # mean annuity of past apps
FE_prev_approved_last3    # approved count in last 3 applications

Previous app agg shape: (338,857 × 14)
```

`FE_prev_credit_sum` → top SHAP contributor in default case explanation.

---

## 8. POS CASH Aggregation

**Problem:** POS and cash loan monthly snapshots — DPD history signals repayment behavior.

```python
FE_pos_count              # number of POS/cash contracts
FE_pos_months_balance_mean
FE_pos_sk_dpd_mean        # mean DPD
FE_pos_sk_dpd_max         # max DPD
FE_pos_sk_dpd_def_mean    # mean defined DPD
FE_pos_completed_count    # completed contracts count

POS CASH agg shape: (337,252 × 7)
```

---

## 9. Installments Aggregation

**Problem (from EDA):** Late payment rate from installment history is a direct predictor of future default.

```python
FE_inst_count             # total installments
FE_inst_payment_diff_mean # mean underpayment (AMT_INSTALMENT - AMT_PAYMENT)
FE_inst_payment_diff_sum  # total underpayment
FE_inst_days_diff_mean    # mean days late
FE_inst_days_diff_max     # max days late
FE_inst_late_count        # number of late payments
FE_inst_late_rate         # fraction of late payments

Installments agg shape: (339,587 × 8)
```

---

## 10. Credit Card Aggregation

**Problem:** Credit card utilization — balance relative to limit — is a standard credit risk signal.

```python
FE_cc_count               # number of credit card records
FE_cc_balance_mean        # mean balance
FE_cc_balance_max         # max balance
FE_cc_credit_limit_mean   # mean credit limit
FE_cc_drawings_mean       # mean monthly drawings
FE_cc_drawings_sum        # total drawings
FE_cc_dpd_mean            # mean DPD
FE_cc_dpd_max             # max DPD
FE_cc_utilization         # balance_mean / (limit_mean + 1)

Credit card agg shape: (103,558 × 10)
```

`FE_cc_utilization` — confirmed important by SHAP analysis.

---

## All 78 Engineered Features

| Group | Count | Key features |
|---|---|---|
| EXT_SOURCE combinations | 8 | `FE_ext_mean` → #1 model feature |
| Age & employment | 4 | `FE_age_years`, `FE_employment_ratio` |
| Credit & income ratios | 6 | `FE_annuity_credit_ratio`, `FE_credit_income_ratio` |
| Frequency encoding | 6 | `FE_*_freq` for 6 categoricals |
| Target encoding | 2 | `FE_ORGANIZATION_TYPE_target_enc` |
| Bureau aggregation | 17 | `FE_bureau_active_count`, `FE_bureau_dpd_mean` |
| Previous app aggregation | 11 | `FE_prev_credit_sum`, `FE_prev_approved_rate` |
| POS CASH aggregation | 6 | `FE_pos_sk_dpd_mean` |
| Installments aggregation | 7 | `FE_inst_late_rate` |
| Credit card aggregation | 9 | `FE_cc_utilization` |
| **TOTAL** | **78** | |

---

[← Preprocessing](02_preprocessing.md) | [← Back to README](../../README.md) | [→ Feature Selection](04_feature_selection.md)