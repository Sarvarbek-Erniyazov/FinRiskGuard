# 🔴 Fraud Detection — Modeling

**Source:** `src/models/fraud/fraud_detector.py`  
**Run:** 2026-04-30 21:03 → 2026-05-01 01:21 (~4 hours 18 minutes)

[← Leakage Audit](05_leakage.md) | [← Back to README](../../README.md) | [→ Model Analysis](07_model_analysis.md)

---

## Training Input

```
Train : 472,432 × 204  | Fraud rate: 3.51%
Val   : 118,108 × 204  | Fraud rate: 3.44%

Feature breakdown:
  FE_ engineered  : 33  (D_normalized: 14)
  _isnan flags    : 9
  Raw features    : 162
  TOTAL           : 204
```

---

## Step 1 — Baseline Models

Three model families trained with default hyperparameters and imbalance handling.

**Imbalance strategy:**
```python
XGBoost  → scale_pos_weight = 28
LightGBM → is_unbalance = True
CatBoost → auto_class_weights = 'Balanced'
```

| Model | AUC-ROC | AUC-PR | F1 | Recall | Precision |
|---|---|---|---|---|---|
| baseline_xgb | 0.9169 | 0.5326 | 0.4348 | 0.6673 | 0.3224 |
| baseline_lgb | 0.9140 | 0.5124 | 0.3641 | 0.7168 | 0.2440 |
| baseline_cat | 0.9072 | 0.4803 | 0.3178 | 0.7416 | 0.2023 |

**XGBoost baseline** leads on both AUC-ROC and AUC-PR. All three baselines use threshold=0.50.

---

## Step 2 — Optuna Hyperparameter Tuning

**Problem:** Baseline hyperparameters are suboptimal. Grid search is too slow on 472K rows.

**Solution:** Optuna TPE sampler, 100 trials per model, AUC-ROC as objective:

```
XGBoost  tuning: 21:03 → 22:55  (~112 min) | Best AUC-ROC: 0.9258
LightGBM tuning: 22:55 → 23:44  (~49 min)  | Best AUC-ROC: 0.9231
CatBoost tuning: 23:44 → 01:12  (~88 min)  | Best AUC-ROC: 0.9180
```

**Tuned results:**

| Model | AUC-ROC | AUC-PR | F1 | Recall | Precision |
|---|---|---|---|---|---|
| tuned_xgb | **0.9258** | **0.5637** | 0.4728 | 0.6661 | 0.3665 |
| tuned_lgb | 0.9231 | 0.5567 | 0.4734 | 0.6722 | 0.3653 |
| tuned_cat | 0.9180 | 0.5244 | 0.3983 | 0.6966 | 0.2789 |

**Tuning gains over baseline:**
```
XGBoost  : +0.0089 AUC-ROC | +0.0311 AUC-PR
LightGBM : +0.0091 AUC-ROC | +0.0443 AUC-PR
CatBoost : +0.0108 AUC-ROC | +0.0441 AUC-PR
```

Saved: `outputs/models/fraud/params_xgb.json`, `params_lgb.json`, `params_cat.json`

---

## Step 3 — Stacking Ensemble

**Problem:** Single models may overfit to specific feature subsets. Stacking combines diverse predictions.

**Architecture:**
```python
CV method   : TimeSeriesSplit(n_splits=5)  # temporal data — no shuffle
Calibration : CalibratedClassifierCV(isotonic) per fold
Meta-learner: LogisticRegression

# 5-fold OOF predictions:
Fold 1/5 | OOF: 78,738 | Train:  78,742
Fold 2/5 | OOF: 78,738 | Train: 157,480
Fold 3/5 | OOF: 78,738 | Train: 236,218
Fold 4/5 | OOF: 78,738 | Train: 314,956
Fold 5/5 | OOF: 78,738 | Train: 393,694
```

**OOF prediction stats:**
```
XGB: mean=0.0307 | std=0.1090
LGB: mean=0.0307 | std=0.1069
CAT: mean=0.0307 | std=0.1054
```

**Stacking result:**
```
AUC-ROC  : 0.8957
AUC-PR   : 0.4791
F1       : 0.4264
Recall   : 0.2955
Precision: 0.7655
```

Stacking trades recall for very high precision (76.6%) — useful for a conservative alert system but loses too much recall for our business requirement (≥70%). `tuned_xgb` remains best overall.

---

## Step 4 — Final Leaderboard

| Rank | Model | AUC-ROC | AUC-PR | F1 |
|---|---|---|---|---|
| 🥇 | **tuned_xgb** | **0.9258** | **0.5637** | 0.4728 |
| 🥈 | tuned_lgb | 0.9231 | 0.5567 | 0.4734 |
| 🥉 | tuned_cat | 0.9180 | 0.5244 | 0.3983 |
| 4 | baseline_xgb | 0.9169 | 0.5326 | 0.4348 |
| 5 | baseline_lgb | 0.9140 | 0.5124 | 0.3641 |
| 6 | baseline_cat | 0.9072 | 0.4803 | 0.3178 |
| 7 | stacking | 0.8957 | 0.4791 | 0.4264 |

**Best model: `tuned_xgb`**

---

## Step 5 — Threshold Optimization

**Problem:** Default threshold 0.50 maximizes accuracy — not business value. Missing a fraud costs far more than a false alarm.

**Business requirement:** Recall ≥ 0.70 (catch at least 70% of all fraud), then maximize F1.

```
Optimizing threshold (Recall >= 0.70, max F1)...
Best threshold : 0.44
F1 at 0.44    : 0.4389
```

**Final model at threshold 0.44:**

```
AUC-ROC  : 0.9258
AUC-PR   : 0.5637
Recall   : 0.7050   ← 70.5% of all fraud caught
Precision: 0.3187
F1       : 0.4389
Threshold: 0.440

Confusion Matrix:
              Predicted 0   Predicted 1
Actual 0      107,919       6,125    (false alarms)
Actual 1        1,199       2,865    (fraud caught)

Fraud caught  (TP): 2,865  / 4,064  → 70.5%
Fraud missed  (FN): 1,199  / 4,064  → 29.5%
False alarm   (FP): 6,125  / 114,044 → 5.4%
```

---

## Saved Artifacts

```
outputs/models/fraud/
├── baseline_xgb.pkl
├── baseline_lgb.pkl
├── baseline_cat.pkl
├── tuned_xgb.pkl          ← BEST MODEL
├── tuned_lgb.pkl
├── tuned_cat.pkl
├── stacking_meta.pkl
├── params_xgb.json
├── params_lgb.json
├── params_cat.json
├── fraud_model.pkl
└── fraud_model_metadata.json
```

---

[← Leakage Audit](05_leakage.md) | [← Back to README](../../README.md) | [→ Model Analysis](07_model_analysis.md)