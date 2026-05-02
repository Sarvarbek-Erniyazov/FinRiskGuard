import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import sys

# LightGBM uchun yangi importlar
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("credit_scorer")

ROOT_DIR     = Path(__file__).resolve().parents[3]
MODELS_DIR   = ROOT_DIR / "outputs" / "models" / "credit"
REPORTS_DIR  = ROOT_DIR / "outputs" / "reports" / "credit"
FEATURES_DIR = ROOT_DIR / "data" / "features" / "credit"
LOGS_DIR     = ROOT_DIR / "outputs" / "logs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "TARGET"
ID_COL     = "SK_ID_CURR"


def setup_file_logger():
    from datetime import datetime
    log = logging.getLogger("credit_scorer")
    if any(isinstance(h, logging.FileHandler) for h in log.handlers):
        return
    log_path = LOGS_DIR / f"credit_scorer_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(file_handler)
    logger.info(f"Log file: {log_path}")


def evaluate(y_true, y_prob, threshold: float = 0.5, label: str = "") -> dict:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score,
        confusion_matrix, classification_report,
    )
    y_pred = (y_prob >= threshold).astype(int)
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
    logger.info(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    logger.info(f"  AUC-PR    : {metrics['auc_pr']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  Threshold : {metrics['threshold']:.3f}")
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\n  Confusion Matrix:\n{cm}")
    logger.info(f"\n{classification_report(y_true, y_pred, digits=4, zero_division=0)}")
    return metrics


def find_best_threshold_recall(y_true, y_prob, min_recall: float = 0.70) -> float:
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
    logger.info(f"  Best threshold (recall>={min_recall}): {best_thr:.2f} | F1: {best_f1:.4f}")
    return float(best_thr)


def train_baseline(X_train, y_train, X_val, y_val) -> dict:
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    logger.info("=" * 60)
    logger.info("STEP 1: BASELINE MODELS")
    logger.info("=" * 60)

    baseline_results = {}

    logger.info("\nTraining XGBoost baseline...")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11,
        objective="binary:logistic", tree_method="hist",
        eval_metric="auc", early_stopping_rounds=50,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    baseline_results["xgb"] = evaluate(y_val, xgb.predict_proba(X_val)[:, 1], label="XGBoost Baseline")
    joblib.dump(xgb, MODELS_DIR / "baseline_xgb.pkl")

    logger.info("\nTraining LightGBM baseline...")
    lgb = LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, is_unbalance=True,
        objective="binary", metric="auc",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
            callbacks=[early_stopping(50), log_evaluation(-1)])
    baseline_results["lgb"] = evaluate(y_val, lgb.predict_proba(X_val)[:, 1], label="LightGBM Baseline")
    joblib.dump(lgb, MODELS_DIR / "baseline_lgb.pkl")

    logger.info("\nTraining CatBoost baseline...")
    cat = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        auto_class_weights="Balanced", eval_metric="AUC",
        early_stopping_rounds=50, random_seed=42, verbose=0,
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    baseline_results["cat"] = evaluate(y_val, cat.predict_proba(X_val)[:, 1], label="CatBoost Baseline")
    joblib.dump(cat, MODELS_DIR / "baseline_cat.pkl")

    logger.info("\nBASELINE SUMMARY:")
    for name, m in baseline_results.items():
        logger.info(f"  {name:5s} AUC-ROC={m['auc_roc']:.4f} | AUC-PR={m['auc_pr']:.4f}")
    return baseline_results


def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 100) -> dict:
    import optuna
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("\nOptuna tuning: XGBoost...")

    def objective(trial):
        params = {
            "n_estimators"         : trial.suggest_int("n_estimators", 100, 2000),
            "max_depth"            : trial.suggest_int("max_depth", 3, 6),
            "learning_rate"        : trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample"            : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree"     : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight"     : trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha"            : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"           : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight"     : 11,
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
    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": 11, "objective": "binary:logistic",
        "tree_method": "hist", "eval_metric": "auc",
        "early_stopping_rounds": 50, "random_state": 42,
        "n_jobs": -1, "verbosity": 0,
    })
    with open(MODELS_DIR / "params_xgb.json", "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"  XGB best AUC-ROC: {study.best_value:.4f}")
    logger.info(f"  Saved: params_xgb.json")
    return best_params


def tune_lightgbm(X_train, y_train, X_val, y_val, n_trials: int = 100) -> dict:
    import optuna
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("\nOptuna tuning: LightGBM...")

    def objective(trial):
        params = {
            "n_estimators"         : trial.suggest_int("n_estimators", 100, 2000),
            "max_depth"            : trial.suggest_int("max_depth", 3, 6),
            "learning_rate"        : trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample"            : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree"     : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples"    : trial.suggest_int("min_child_samples", 5, 100),
            "num_leaves"           : trial.suggest_int("num_leaves", 20, 200),
            "reg_alpha"            : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"           : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "is_unbalance"         : True,
            "objective"            : "binary",
            "metric"               : "auc",
            "random_state"         : 42,
            "n_jobs"               : -1,
            "verbose"              : -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  callbacks=[early_stopping(50), log_evaluation(-1)])
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    best_params.update({
        "is_unbalance": True, "objective": "binary", "metric": "auc",
        "random_state": 42, "n_jobs": -1, "verbose": -1,
    })
    with open(MODELS_DIR / "params_lgb.json", "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"  LGB best AUC-ROC: {study.best_value:.4f}")
    logger.info(f"  Saved: params_lgb.json")
    return best_params


def tune_catboost(X_train, y_train, X_val, y_val, n_trials: int = 100) -> dict:
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("\nOptuna tuning: CatBoost...")

    def objective(trial):
        params = {
            "iterations"           : trial.suggest_int("iterations", 100, 2000),
            "depth"                : trial.suggest_int("depth", 3, 6),
            "learning_rate"        : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg"          : trial.suggest_float("l2_leaf_reg", 3.0, 10.0, log=True),
            "bagging_temperature"  : trial.suggest_float("bagging_temperature", 0.0, 0.8),
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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, catch=(Exception,))
    best_params = study.best_params
    best_params.update({
        "auto_class_weights"   : "Balanced",
        "eval_metric"          : "AUC",
        "early_stopping_rounds": 50,
        "random_seed"          : 42,
        "verbose"              : 0,
    })
    with open(MODELS_DIR / "params_cat.json", "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"  CAT best AUC-ROC: {study.best_value:.4f}")
    logger.info(f"  Saved: params_cat.json")
    return best_params


def train_tuned_models(X_train, y_train, X_val, y_val, params: dict) -> Tuple[dict, dict]:
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    logger.info("=" * 60)
    logger.info("STEP 2b: TUNED MODELS (final fit on full train)")
    logger.info("=" * 60)

    tuned_results = {}
    tuned_models  = {}

    xgb = XGBClassifier(**params["xgb"])
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    tuned_results["xgb"] = evaluate(y_val, xgb.predict_proba(X_val)[:, 1], label="XGBoost Tuned")
    tuned_models["xgb"]  = xgb
    joblib.dump(xgb, MODELS_DIR / "tuned_xgb.pkl")

    lgb = LGBMClassifier(**params["lgb"])
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
            callbacks=[early_stopping(50), log_evaluation(-1)])
    tuned_results["lgb"] = evaluate(y_val, lgb.predict_proba(X_val)[:, 1], label="LightGBM Tuned")
    tuned_models["lgb"]  = lgb
    joblib.dump(lgb, MODELS_DIR / "tuned_lgb.pkl")

    cat = CatBoostClassifier(**params["cat"])
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    tuned_results["cat"] = evaluate(y_val, cat.predict_proba(X_val)[:, 1], label="CatBoost Tuned")
    tuned_models["cat"]  = cat
    joblib.dump(cat, MODELS_DIR / "tuned_cat.pkl")

    logger.info("\nTUNED SUMMARY:")
    for name, m in tuned_results.items():
        logger.info(f"  {name:5s} AUC-ROC={m['auc_roc']:.4f} | AUC-PR={m['auc_pr']:.4f}")
    return tuned_results, tuned_models


def train_stacking(X_train, y_train, X_val, y_val, params: dict) -> Tuple[dict, object, np.ndarray]:
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    logger.info("=" * 60)
    logger.info("STEP 3: STACKING ENSEMBLE")
    logger.info("  Base models : Optuna best params — fresh per fold")
    logger.info("  Meta learner: LogisticRegression + StandardScaler")
    logger.info("=" * 60)

    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X_train), 3))
    val_preds = np.zeros((len(X_val), 3))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = X_train.iloc[tr_idx]
        X_vf = X_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]
        y_vf = y_train.iloc[val_idx]

        xgb = XGBClassifier(**params["xgb"])
        xgb.fit(X_tr, y_tr, eval_set=[(X_vf, y_vf)], verbose=False)
        oof_preds[val_idx, 0] = xgb.predict_proba(X_vf)[:, 1]
        val_preds[:, 0]       += xgb.predict_proba(X_val)[:, 1] / 5

        lgb = LGBMClassifier(**params["lgb"])
        lgb.fit(X_tr, y_tr, eval_set=[(X_vf, y_vf)], 
                callbacks=[early_stopping(50), log_evaluation(-1)])
        oof_preds[val_idx, 1] = lgb.predict_proba(X_vf)[:, 1]
        val_preds[:, 1]       += lgb.predict_proba(X_val)[:, 1] / 5

        cat = CatBoostClassifier(**params["cat"])
        cat.fit(X_tr, y_tr, eval_set=(X_vf, y_vf), verbose=False)
        oof_preds[val_idx, 2] = cat.predict_proba(X_vf)[:, 1]
        val_preds[:, 2]       += cat.predict_proba(X_val)[:, 1] / 5

        logger.info(f"  Fold {fold+1}/5 complete | "
                    f"OOF range: [{oof_preds[val_idx].min():.3f}, "
                    f"{oof_preds[val_idx].max():.3f}]")

    logger.info(f"\n  OOF predictions stats:")
    for i, name in enumerate(["XGB", "LGB", "CAT"]):
        logger.info(
            f"    {name}: mean={oof_preds[:, i].mean():.4f} | "
            f"std={oof_preds[:, i].std():.4f} | "
            f"min={oof_preds[:, i].min():.4f} | "
            f"max={oof_preds[:, i].max():.4f}"
        )

    scaler     = StandardScaler()
    oof_scaled = scaler.fit_transform(oof_preds)
    

    meta = LogisticRegression(
        C=1.0, class_weight="balanced",
        random_state=42, max_iter=1000,
    )
    meta.fit(oof_scaled, y_train)

    meta_pipeline  = Pipeline([("scaler", scaler), ("meta", meta)])
    stacked_prob   = meta_pipeline.predict_proba(val_preds)[:, 1]
    stacked_result = evaluate(y_val, stacked_prob, label="Stacking Ensemble")

    joblib.dump(meta_pipeline, MODELS_DIR / "stacking_meta.pkl")
    logger.info("  Saved: stacking_meta.pkl (Pipeline: scaler + LR)")
    return stacked_result, meta_pipeline, stacked_prob


def select_best_model(
    baseline_results, tuned_results, stacked_result,
    y_val, val_probs,
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

    logger.info("\nFINAL LEADERBOARD (by AUC-ROC):")
    sorted_results = sorted(
        all_results.items(), key=lambda x: x[1]["auc_roc"], reverse=True
    )
    for rank, (name, m) in enumerate(sorted_results, 1):
        logger.info(
            f"  #{rank} {name:20s} "
            f"AUC-ROC={m['auc_roc']:.4f} | "
            f"AUC-PR={m['auc_pr']:.4f} | "
            f"F1={m['f1']:.4f}"
        )

    best_name, _ = sorted_results[0]
    logger.info(f"\nBEST MODEL: {best_name}")

    if "stacking" in best_name:
        best_prob  = val_probs["stacking"]
        best_model = joblib.load(MODELS_DIR / "stacking_meta.pkl")
    else:
        parts      = best_name.split("_")
        best_prob  = val_probs[best_name]
        best_model = joblib.load(MODELS_DIR / f"{parts[0]}_{parts[1]}.pkl")

    logger.info("\nOptimizing threshold (Recall >= 0.70, max F1)...")
    best_threshold = find_best_threshold_recall(y_val, best_prob, min_recall=0.70)

    final_metrics = evaluate(
        y_val, best_prob,
        threshold=best_threshold,
        label=f"BEST MODEL ({best_name}) — Optimized Threshold",
    )

    joblib.dump(best_model, MODELS_DIR / "credit_model.pkl")

    metadata = {
        "model_name" : best_name,
        "auc_roc"    : final_metrics["auc_roc"],
        "auc_pr"     : final_metrics["auc_pr"],
        "f1"         : final_metrics["f1"],
        "precision"  : final_metrics["precision"],
        "recall"     : final_metrics["recall"],
        "threshold"  : best_threshold,
        "leaderboard": {k: v for k, v in all_results.items()},
    }
    with open(MODELS_DIR / "credit_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("\nSaved:")
    logger.info("  outputs/models/credit/credit_model.pkl")
    logger.info("  outputs/models/credit/credit_model_metadata.json")


def main(n_trials: int = 100):
    setup_file_logger()

    logger.info("=" * 60)
    logger.info("FINRISKGUARD — CREDIT SCORER")
    logger.info(f"n_trials: {n_trials}")
    logger.info("=" * 60)

    train = pd.read_parquet(FEATURES_DIR / "train_credit_features.parquet")
    val   = pd.read_parquet(FEATURES_DIR / "val_credit_features.parquet")

    drop_cols = [c for c in [TARGET_COL, ID_COL] if c in train.columns]

    X_train = train.drop(columns=drop_cols)
    y_train = train[TARGET_COL]
    X_val   = val.drop(columns=[c for c in drop_cols if c in val.columns])
    y_val   = val[TARGET_COL]

    logger.info(f"Train: {X_train.shape} | Default rate: {y_train.mean()*100:.2f}%")
    logger.info(f"Val   : {X_val.shape}   | Default rate: {y_val.mean()*100:.2f}%")

    val_probs = {}

    baseline_results = train_baseline(X_train, y_train, X_val, y_val)
    for name in ["xgb", "lgb", "cat"]:
        m = joblib.load(MODELS_DIR / f"baseline_{name}.pkl")
        val_probs[f"baseline_{name}"] = m.predict_proba(X_val)[:, 1]

    xgb_params = tune_xgboost(X_train, y_train, X_val, y_val, n_trials)
    lgb_params = tune_lightgbm(X_train, y_train, X_val, y_val, n_trials)
    cat_params = tune_catboost(X_train, y_train, X_val, y_val, n_trials)

    params = {"xgb": xgb_params, "lgb": lgb_params, "cat": cat_params}

    tuned_results, tuned_models = train_tuned_models(
        X_train, y_train, X_val, y_val, params
    )
    for name in ["xgb", "lgb", "cat"]:
        val_probs[f"tuned_{name}"] = tuned_models[name].predict_proba(X_val)[:, 1]

    stacked_result, meta_pipeline, stacked_prob = train_stacking(
        X_train, y_train, X_val, y_val, params
    )
    val_probs["stacking"] = stacked_prob

    select_best_model(
        baseline_results, tuned_results, stacked_result,
        y_val, val_probs,
    )

    logger.info("=" * 60)
    logger.info("CREDIT SCORER COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100)
    args = parser.parse_args()
    main(n_trials=args.n_trials)