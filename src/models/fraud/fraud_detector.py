import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("fraud_detector")

ROOT_DIR     = Path(__file__).resolve().parents[3]
MODELS_DIR   = ROOT_DIR / "outputs" / "models" / "fraud"
REPORTS_DIR  = ROOT_DIR / "outputs" / "reports" / "fraud"
FEATURES_DIR = ROOT_DIR / "data" / "features" / "fraud"

for d in [MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def evaluate(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> dict:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score,
        confusion_matrix, classification_report,
    )
    y_pred  = (y_prob >= threshold).astype(int)
    metrics = {
        "auc_roc"  : roc_auc_score(y_true, y_prob),
        "auc_pr"   : average_precision_score(y_true, y_prob),
        "f1"       : f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }
    logger.info(f"\n{'='*50}")
    logger.info(f"EVALUATION — {label}")
    logger.info(f"  AUC-ROC   : {metrics['auc_roc']:.4f}  ← PRIMARY")
    logger.info(f"  AUC-PR    : {metrics['auc_pr']:.4f}  ← SECONDARY")
    logger.info(f"  F1        : {metrics['f1']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  Threshold : {metrics['threshold']:.3f}")
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\n  Confusion Matrix:\n{cm}")
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"  Fraud caught  (TP): {tp:,}")
    logger.info(f"  Fraud missed  (FN): {fn:,}")
    logger.info(f"  False alarm   (FP): {fp:,}")
    logger.info(f"  Correct clear (TN): {tn:,}")
    logger.info(
        f"\n{classification_report(y_true, y_pred, digits=4, zero_division=0)}"
    )
    return metrics


def find_best_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    min_recall: float = 0.70,
) -> Tuple[float, float]:
    from sklearn.metrics import f1_score, recall_score

    thresholds = np.arange(0.05, 0.95, 0.01)
    best_thr   = 0.5
    best_f1    = 0.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        rec    = recall_score(y_true, y_pred, zero_division=0)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        if rec >= min_recall and f1 > best_f1:
            best_f1  = f1
            best_thr = thr

    if best_f1 == 0.0:
        logger.info(
            f"  WARNING: No threshold satisfies recall>={min_recall} "
            f"— falling back to F1-max"
        )
        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            f1     = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_thr = thr

    logger.info(
        f"  Best threshold (recall>={min_recall}): "
        f"{best_thr:.2f} | F1: {best_f1:.4f}"
    )
    return float(best_thr), float(best_f1)


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    logger.info("=" * 60)
    logger.info("STEP 1: BASELINE MODELS")
    logger.info("=" * 60)

    baseline_results = {}

    logger.info("\nTraining XGBoost baseline...")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=28,
        objective="binary:logistic", tree_method="hist",
        eval_metric="auc", early_stopping_rounds=50,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    baseline_results["xgb"] = evaluate(
        y_val, xgb.predict_proba(X_val)[:, 1], label="XGBoost Baseline"
    )
    joblib.dump(xgb, MODELS_DIR / "baseline_xgb.pkl")

    logger.info("\nTraining LightGBM baseline...")
    lgb = LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True, objective="binary", metric="auc",
        early_stopping_rounds=50,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    baseline_results["lgb"] = evaluate(
        y_val, lgb.predict_proba(X_val)[:, 1], label="LightGBM Baseline"
    )
    joblib.dump(lgb, MODELS_DIR / "baseline_lgb.pkl")

    logger.info("\nTraining CatBoost baseline...")
    cat = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        auto_class_weights="Balanced", eval_metric="AUC",
        early_stopping_rounds=50, random_seed=42, verbose=0,
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    baseline_results["cat"] = evaluate(
        y_val, cat.predict_proba(X_val)[:, 1], label="CatBoost Baseline"
    )
    joblib.dump(cat, MODELS_DIR / "baseline_cat.pkl")

    logger.info("\nBASELINE SUMMARY (by AUC-ROC):")
    for name, m in sorted(
        baseline_results.items(),
        key=lambda x: x[1]["auc_roc"], reverse=True
    ):
        logger.info(
            f"  {name:5s} AUC-ROC={m['auc_roc']:.4f} | "
            f"AUC-PR={m['auc_pr']:.4f}"
        )
    return baseline_results


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
) -> dict:
    import optuna
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("\nOptuna tuning: XGBoost...")

    def objective(trial):
        params = {
            "n_estimators"         : trial.suggest_int("n_estimators", 300, 2000),
            "max_depth"            : trial.suggest_int("max_depth", 3, 6),
            "learning_rate"        : trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample"            : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree"     : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight"     : trial.suggest_int("min_child_weight", 5, 20),
            "reg_alpha"            : trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
            "reg_lambda"           : trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "scale_pos_weight"     : 28,
            "objective"            : "binary:logistic",
            "tree_method"          : "hist",
            "eval_metric"          : "auc",
            "early_stopping_rounds": 50,
            "random_state"         : 42,
            "n_jobs"               : -1,
            "verbosity"            : 0,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({
        "scale_pos_weight": 28, "objective": "binary:logistic",
        "tree_method": "hist", "eval_metric": "auc",
        "early_stopping_rounds": 50, "random_state": 42,
        "n_jobs": -1, "verbosity": 0,
    })
    with open(MODELS_DIR / "params_xgb.json", "w") as f:
        json.dump(best, f, indent=2)
    logger.info(f"  XGB best AUC-ROC: {study.best_value:.4f}")
    return best


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
) -> dict:
    import optuna
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("\nOptuna tuning: LightGBM...")

    def objective(trial):
        params = {
            "n_estimators"         : trial.suggest_int("n_estimators", 300, 2000),
            "max_depth"            : trial.suggest_int("max_depth", 3, 6),
            "learning_rate"        : trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample"            : trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree"     : trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples"    : trial.suggest_int("min_child_samples", 20, 100),
            "num_leaves"           : trial.suggest_int("num_leaves", 20, 150),
            "reg_alpha"            : trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
            "reg_lambda"           : trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "is_unbalance"         : True,
            "objective"            : "binary",
            "metric"               : "auc",
            "early_stopping_rounds": 50,
            "random_state"         : 42,
            "n_jobs"               : -1,
            "verbose"              : -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({
        "is_unbalance": True, "objective": "binary", "metric": "auc",
        "early_stopping_rounds": 50, "random_state": 42,
        "n_jobs": -1, "verbose": -1,
    })
    with open(MODELS_DIR / "params_lgb.json", "w") as f:
        json.dump(best, f, indent=2)
    logger.info(f"  LGB best AUC-ROC: {study.best_value:.4f}")
    return best


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 100,
) -> dict:
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("\nOptuna tuning: CatBoost...")

    def objective(trial):
        params = {
            "iterations"           : trial.suggest_int("iterations", 300, 2000),
            "depth"                : trial.suggest_int("depth", 3, 6),
            "learning_rate"        : trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "l2_leaf_reg"          : trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "bagging_temperature"  : trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength"      : trial.suggest_float("random_strength", 0.5, 10.0, log=True),
            "auto_class_weights"   : "Balanced",
            "eval_metric"          : "AUC",
            "early_stopping_rounds": 50,
            "random_seed"          : 42,
            "verbose"              : 0,
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective, n_trials=n_trials,
        show_progress_bar=True, catch=(Exception,)
    )
    best = study.best_params
    best.update({
        "auto_class_weights": "Balanced", "eval_metric": "AUC",
        "early_stopping_rounds": 50, "random_seed": 42, "verbose": 0,
    })
    with open(MODELS_DIR / "params_cat.json", "w") as f:
        json.dump(best, f, indent=2)
    logger.info(f"  CAT best AUC-ROC: {study.best_value:.4f}")
    return best


def train_tuned_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
) -> Tuple[dict, dict]:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    logger.info("=" * 60)
    logger.info("STEP 2: TUNED MODELS")
    logger.info("=" * 60)

    tuned_results = {}
    tuned_models  = {}

    xgb = XGBClassifier(**params["xgb"])
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    tuned_results["xgb"] = evaluate(
        y_val, xgb.predict_proba(X_val)[:, 1], label="XGBoost Tuned"
    )
    tuned_models["xgb"] = xgb
    joblib.dump(xgb, MODELS_DIR / "tuned_xgb.pkl")

    lgb = LGBMClassifier(**params["lgb"])
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    tuned_results["lgb"] = evaluate(
        y_val, lgb.predict_proba(X_val)[:, 1], label="LightGBM Tuned"
    )
    tuned_models["lgb"] = lgb
    joblib.dump(lgb, MODELS_DIR / "tuned_lgb.pkl")

    cat = CatBoostClassifier(**params["cat"])
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    tuned_results["cat"] = evaluate(
        y_val, cat.predict_proba(X_val)[:, 1], label="CatBoost Tuned"
    )
    tuned_models["cat"] = cat
    joblib.dump(cat, MODELS_DIR / "tuned_cat.pkl")

    logger.info("\nTUNED SUMMARY (by AUC-ROC):")
    for name, m in sorted(
        tuned_results.items(),
        key=lambda x: x[1]["auc_roc"], reverse=True
    ):
        logger.info(
            f"  {name:5s} AUC-ROC={m['auc_roc']:.4f} | "
            f"AUC-PR={m['auc_pr']:.4f}"
        )
    return tuned_results, tuned_models


def train_stacking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    tuned_models: dict,
) -> Tuple[dict, object, np.ndarray]:
    """
    Stacking with calibrated base models and LogisticRegression meta-learner.
    CalibratedClassifierCV(method='isotonic', cv='prefit') applied per fold
    to fix uncalibrated probability issue causing stacking underperformance.
    """
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    logger.info("=" * 60)
    logger.info("STEP 3: STACKING ENSEMBLE")
    logger.info("  CV method    : TimeSeriesSplit(n_splits=5)")
    logger.info("  Calibration  : isotonic (cv=prefit) per fold")
    logger.info("  Meta learner : LogisticRegression")
    logger.info("=" * 60)

    tscv      = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros((len(X_train), 3))
    val_preds = np.zeros((len(X_val), 3))

    for fold, (tr_idx, oof_idx) in enumerate(tscv.split(X_train)):
        X_tr  = X_train.iloc[tr_idx]
        X_oof = X_train.iloc[oof_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_oof = y_train.iloc[oof_idx]

        for i, name in enumerate(["xgb", "lgb", "cat"]):
            if name == "xgb":
                m = XGBClassifier(**params["xgb"])
                m.fit(X_tr, y_tr, eval_set=[(X_oof, y_oof)], verbose=False)
            elif name == "lgb":
                m = LGBMClassifier(**params["lgb"])
                m.fit(X_tr, y_tr, eval_set=[(X_oof, y_oof)], callbacks=[])
            else:
                m = CatBoostClassifier(**params["cat"])
                m.fit(X_tr, y_tr, eval_set=(X_oof, y_oof), verbose=False)

            cal = CalibratedClassifierCV(m, method="isotonic", cv="prefit")
            cal.fit(X_oof, y_oof)

            oof_preds[oof_idx, i] = cal.predict_proba(X_oof)[:, 1]
            val_preds[:, i]       += cal.predict_proba(X_val)[:, 1] / 5

        logger.info(
            f"  Fold {fold+1}/5 complete | "
            f"OOF: {len(oof_idx):,} | Train: {len(tr_idx):,}"
        )

    logger.info("\n  OOF prediction stats:")
    for i, name in enumerate(["XGB", "LGB", "CAT"]):
        logger.info(
            f"    {name}: mean={oof_preds[:,i].mean():.4f} | "
            f"std={oof_preds[:,i].std():.4f}"
        )

    meta = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    meta.fit(oof_preds, y_train)

    stacked_prob   = meta.predict_proba(val_preds)[:, 1]
    stacked_result = evaluate(y_val, stacked_prob, label="Stacking Ensemble")

    joblib.dump(meta, MODELS_DIR / "stacking_meta.pkl")
    logger.info("  Saved: stacking_meta.pkl")
    return stacked_result, meta, stacked_prob


def select_best_model(
    baseline_results: dict,
    tuned_results: dict,
    stacked_result: dict,
    y_val: pd.Series,
    val_probs: dict,
    stacked_prob: np.ndarray,
) -> None:
    logger.info("=" * 60)
    logger.info("STEP 4: FINAL COMPARISON & BEST MODEL SELECTION")
    logger.info("=" * 60)

    all_results = {}
    for name, m in baseline_results.items():
        all_results[f"baseline_{name}"] = m
    for name, m in tuned_results.items():
        all_results[f"tuned_{name}"] = m
    all_results["stacking"] = stacked_result

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["auc_roc"],
        reverse=True,
    )

    logger.info("\nFINAL LEADERBOARD (by AUC-ROC):")
    for rank, (name, m) in enumerate(sorted_results, 1):
        logger.info(
            f"  #{rank} {name:<20} "
            f"AUC-ROC={m['auc_roc']:.4f} | "
            f"AUC-PR={m['auc_pr']:.4f} | "
            f"F1={m['f1']:.4f}"
        )

    best_name, _ = sorted_results[0]
    logger.info(f"\nBEST MODEL: {best_name}")

    if "stacking" in best_name:
        best_prob  = stacked_prob
        best_model = joblib.load(MODELS_DIR / "stacking_meta.pkl")
    else:
        parts      = best_name.split("_")
        best_prob  = val_probs[best_name]
        best_model = joblib.load(
            MODELS_DIR / f"{parts[0]}_{parts[1]}.pkl"
        )

    logger.info("\nOptimizing threshold (Recall >= 0.70, max F1)...")
    best_threshold, _ = find_best_threshold(y_val, best_prob, min_recall=0.70)

    final_metrics = evaluate(
        y_val, best_prob,
        threshold=best_threshold,
        label=f"BEST MODEL ({best_name}) — Optimized Threshold",
    )

    joblib.dump(best_model, MODELS_DIR / "fraud_model.pkl")

    metadata = {
        "model_name"      : best_name,
        "primary_metric"  : "auc_roc",
        "secondary_metric": "auc_pr",
        "auc_roc"         : final_metrics["auc_roc"],
        "auc_pr"          : final_metrics["auc_pr"],
        "f1"              : final_metrics["f1"],
        "precision"       : final_metrics["precision"],
        "recall"          : final_metrics["recall"],
        "threshold"       : best_threshold,
        "cv_method"       : "TimeSeriesSplit(n_splits=5)",
        "calibration"     : "isotonic (cv=prefit per fold)",
        "meta_learner"    : "LogisticRegression",
        "class_imbalance" : "27.6:1",
        "scale_pos_weight": 28,
        "features_used"   : 200,
        "leaderboard"     : {k: v for k, v in all_results.items()},
    }
    with open(MODELS_DIR / "fraud_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\nSaved:")
    logger.info("  outputs/models/fraud/fraud_model.pkl")
    logger.info("  outputs/models/fraud/fraud_model_metadata.json")
    logger.info(f"\nFinal AUC-ROC : {final_metrics['auc_roc']:.4f}")
    logger.info(f"Final AUC-PR  : {final_metrics['auc_pr']:.4f}")


def main(n_trials: int = 100, stage: str = "all") -> None:
    logger.info("=" * 60)
    logger.info("FINRISKGUARD — FRAUD DETECTOR")
    logger.info(f"Stage: {stage} | n_trials: {n_trials}")
    logger.info("Primary metric  : AUC-ROC")
    logger.info("Secondary metric: AUC-PR")
    logger.info("=" * 60)

    train = pd.read_parquet(FEATURES_DIR / "train_fraud_features.parquet")
    val   = pd.read_parquet(FEATURES_DIR / "val_fraud_features.parquet")

    X_train = train.drop(columns=["isFraud"])
    y_train = train["isFraud"]
    X_val   = val.drop(columns=["isFraud"])
    y_val   = val["isFraud"]

    logger.info(f"Train: {X_train.shape} | Fraud rate: {y_train.mean()*100:.2f}%")
    logger.info(f"Val  : {X_val.shape}   | Fraud rate: {y_val.mean()*100:.2f}%")

    fe_count  = sum(1 for c in X_train.columns if c.startswith("FE_"))
    nan_count = sum(1 for c in X_train.columns if c.endswith("_isnan"))
    d_norm    = sum(1 for c in X_train.columns if "normalized" in c)
    logger.info(f"  FE_ engineered  : {fe_count}")
    logger.info(f"  └─ D_normalized : {d_norm}")
    logger.info(f"  _isnan flags    : {nan_count}")

    val_probs        = {}
    baseline_results = {}

    if stage == "all":
        baseline_results = train_baseline(X_train, y_train, X_val, y_val)
        for name in ["xgb", "lgb", "cat"]:
            m = joblib.load(MODELS_DIR / f"baseline_{name}.pkl")
            val_probs[f"baseline_{name}"] = m.predict_proba(X_val)[:, 1]

        xgb_params = tune_xgboost(X_train, y_train, X_val, y_val, n_trials)
        lgb_params = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials)
        cat_params = tune_catboost(X_train, y_train, X_val, y_val, n_trials)

    elif stage == "cat_only":
        logger.info("Loading saved baseline models and XGB/LGB params...")
        for name in ["xgb", "lgb", "cat"]:
            m    = joblib.load(MODELS_DIR / f"baseline_{name}.pkl")
            prob = m.predict_proba(X_val)[:, 1]
            val_probs[f"baseline_{name}"] = prob
            baseline_results[name] = evaluate(
                y_val, prob, label=f"{name.upper()} Baseline (loaded)"
            )
        with open(MODELS_DIR / "params_xgb.json") as f:
            xgb_params = json.load(f)
        with open(MODELS_DIR / "params_lgb.json") as f:
            lgb_params = json.load(f)
        cat_params = tune_catboost(X_train, y_train, X_val, y_val, n_trials)

    else:
        raise ValueError(f"Unknown stage: {stage}")

    params = {"xgb": xgb_params, "lgb": lgb_params, "cat": cat_params}

    tuned_results, tuned_models = train_tuned_models(
        X_train, y_train, X_val, y_val, params
    )
    for name in ["xgb", "lgb", "cat"]:
        val_probs[f"tuned_{name}"] = (
            tuned_models[name].predict_proba(X_val)[:, 1]
        )

    stacked_result, meta, stacked_prob = train_stacking(
        X_train, y_train, X_val, y_val, params, tuned_models
    )
    val_probs["stacking"] = stacked_prob

    select_best_model(
        baseline_results, tuned_results, stacked_result,
        y_val, val_probs, stacked_prob,
    )

    logger.info("=" * 60)
    logger.info("FRAUD DETECTOR COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--stage", choices=["all", "cat_only"], default="all")
    args = parser.parse_args()
    main(n_trials=args.n_trials, stage=args.stage)