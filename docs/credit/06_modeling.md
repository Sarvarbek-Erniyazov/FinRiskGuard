# 🟠 Credit Scoring — Modeling

**Source:** `src/models/credit/credit_scorer.py`  
**Run:** 2026-05-01 20:48 → 2026-05-01 23:25 (~2 hours 37 minutes)

[← Leakage Audit](05_leakage.md) | [← Back to README](../../README.md) | [→ Model Analysis](07_model_analysis.md)

---

## Training Input

```
Train : 246,008 × 105  | Default rate: 8.07%
Val   : 61,503  × 105  | Default rate: 8.07%
```

---

## Step 1 — Baseline Models

Three model families trained with default hyperparameters and imbalance handling.

**Imbalance strategy:**
```python
XGBoost  → scale_pos_weight = 11
LightGBM → is_unbalance = True
CatBoost → auto_class_weights = 'Balanced'
```

| Model | AUC-ROC | AUC-PR | F1 | Recall | Precision |
|---|---|---|---|---|---|
| baseline_xgb | 0.7814 | 0.2783 | 0.3010 | 0.6528 | 0.1956 |
| baseline_lgb | 0.7816 | 0.2786 | 0.2944 | 0.6858 | 0.1874 |
| baseline_cat | 0.7822 | 0.2820 | 0.2916 | 0.6971 | 0.1843 |

All three baselines tightly clustered — 0.781–0.782 AUC-ROC range. CatBoost leads baseline by a narrow margin.

---

## Step 2 — Optuna Hyperparameter Tuning

**Problem:** Baseline hyperparameters are suboptimal. Credit dataset has different characteristics than fraud — lower imbalance, more structured features.

**Solution:** Optuna TPE sampler, 100 trials per model, AUC-ROC as objective:

```
XGBoost  tuning: 20:48 → 21:54  (~66 min)  | Best AUC-ROC: 0.7846
LightGBM tuning: 21:54 → 22:17  (~23 min)  | Best AUC-ROC: 0.7840
CatBoost tuning: 22:17 → 23:15  (~58 min)  | Best AUC-ROC: 0.7848
```

**Tuned results:**

| Model | AUC-ROC | AUC-PR | F1 | Recall | Precision |
|---|---|---|---|---|---|
| tuned_xgb | 0.7846 | 0.2857 | 0.2987 | 0.6860 | 0.1909 |
| tuned_lgb | 0.7840 | 0.2837 | 0.2926 | 0.6918 | 0.1856 |
| tuned_cat | **0.7848** | 0.2852 | 0.2984 | 0.6838 | 0.1908 |

**Tuning gains over baseline:**
```
XGBoost  : +0.0032 AUC-ROC | +0.0074 AUC-PR
LightGBM : +0.0024 AUC-ROC | +0.0051 AUC-PR
CatBoost : +0.0026 AUC-ROC | +0.0032 AUC-PR
```

Saved: `outputs/models/credit/params_xgb.json`, `params_lgb.json`, `params_cat.json`

---

## Step 3 — Stacking Ensemble

**Problem:** All three tuned models are very close in AUC-ROC (0.7840–0.7848). Stacking can extract complementary signal from their diverse predictions.

**Architecture:**
```python
CV method    : StratifiedKFold(n_splits=5)
              # credit applicants are independent — no temporal ordering
Meta-learner : StandardScaler + LogisticRegression

# 5-fold OOF predictions:
Fold 1/5 | OOF range: [0.003, 0.978]
Fold 2/5 | OOF range: [0.008, 0.977]
Fold 3/5 | OOF range: [0.003, 0.983]
Fold 4/5 | OOF range: [0.005, 0.974]
Fold 5/5 | OOF range: [0.003, 0.980]
```

**OOF prediction stats:**
```
XGB: mean=0.3797 | std=0.2195 | min=0.003 | max=0.983
LGB: mean=0.3865 | std=0.2202 | min=0.003 | max=0.977
CAT: mean=0.3821 | std=0.2169 | min=0.003 | max=0.981
```

**Stacking result:**
```
AUC-ROC  : 0.7849  ← best overall
AUC-PR   : 0.2854
F1       : 0.2896
Recall   : 0.7098
Precision: 0.1819
```

Stacking edged out individual tuned models — +0.0001 to +0.0009 AUC-ROC. Small but consistent gain across both runs.

---

## Step 4 — Final Leaderboard

| Rank | Model | AUC-ROC | AUC-PR | F1 |
|---|---|---|---|---|
| 🥇 | **stacking** | **0.7849** | **0.2854** | 0.2896 |
| 🥈 | tuned\_cat | 0.7848 | 0.2852 | 0.2984 |
| 🥉 | tuned\_xgb | 0.7846 | 0.2857 | 0.2987 |
| 4 | tuned\_lgb | 0.7840 | 0.2837 | 0.2926 |
| 5 | baseline\_cat | 0.7822 | 0.2820 | 0.2916 |
| 6 | baseline\_lgb | 0.7816 | 0.2786 | 0.2944 |
| 7 | baseline\_xgb | 0.7814 | 0.2783 | 0.3010 |

**Best model: `stacking`**

---

## Step 5 — Threshold Optimization

**Problem:** Default threshold 0.50 maximizes accuracy — not business value. Approving a loan that defaults costs far more than rejecting a good borrower.

**Business requirement:** Recall ≥ 0.70 (catch at least 70% of all defaults), then maximize F1.

```
Optimizing threshold (Recall >= 0.70, max F1)...
Best threshold : 0.50
F1 at 0.50     : 0.2896
```

Note: For credit scoring, the optimal threshold happened to be 0.50 — the default. The threshold search confirmed this is the best point satisfying Recall ≥ 0.70.

**Final model at threshold 0.50:**

```
AUC-ROC  : 0.7849
AUC-PR   : 0.2854
Recall   : 0.7098   ← 71.0% of all defaults caught
Precision: 0.1819
F1       : 0.2896
Threshold: 0.500

Confusion Matrix:
              Predicted 0    Predicted 1
Actual 0       40,689         15,849   (false alarms)
Actual 1        1,441          3,524   (defaults caught)

Defaults caught (TP): 3,524  / 4,965  → 71.0%
Defaults missed (FN): 1,441  / 4,965  → 29.0%
False alarm     (FP): 15,849 / 56,538 → 28.0%
```

---

## Saved Artifacts

```
outputs/models/credit/
├── baseline_xgb.pkl
├── baseline_lgb.pkl
├── baseline_cat.pkl
├── tuned_xgb.pkl
├── tuned_lgb.pkl
├── tuned_cat.pkl
├── stacking_meta.pkl      ← Pipeline: StandardScaler + LogisticRegression
├── params_xgb.json
├── params_lgb.json
├── params_cat.json
├── credit_model.pkl       ← BEST MODEL (stacking)
└── credit_model_metadata.json
```

---

[← Leakage Audit](05_leakage.md) | [← Back to README](../../README.md) | [→ Model Analysis](07_model_analysis.md)