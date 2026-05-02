import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("home_credit.feature_selector")

TARGET_COL = "TARGET"
ID_COL     = "SK_ID_CURR"

# Home Credit: 122 original + ~60 FE = ~180 features
# top_k = 60 → ~33% (IEEE-CIS 34.6% bilan mos)
TOP_K_DEFAULT = 70


def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    exclude_cols: list = None,
) -> Tuple[pd.DataFrame, list]:

    logger.info(f"Correlation filter: threshold={threshold}")

    if exclude_cols is None:
        exclude_cols = []

    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]

    df = df.drop(columns=drop_cols)
    logger.info(f"  Dropped {len(drop_cols)} correlated features: {drop_cols[:5]}...")
    logger.info(f"  Remaining features: {df.shape[1]}")
    return df, drop_cols


def mutual_information_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = TOP_K_DEFAULT,
) -> Tuple[pd.Series, list]:

    from sklearn.feature_selection import mutual_info_classif

    logger.info(f"Mutual Information selection: top_k={top_k}")
    mi_scores = mutual_info_classif(X.fillna(-999), y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    selected  = mi_series.head(top_k).index.tolist()
    logger.info(f"  Top 5 MI features: {selected[:5]}")
    logger.info(f"  Selected {len(selected)} features")
    return mi_series, selected


def xgboost_importance_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = TOP_K_DEFAULT,
    threshold: float = 0.0001,
) -> Tuple[pd.Series, list]:

    from xgboost import XGBClassifier

    logger.info(f"XGBoost importance selection: top_k={top_k}")

    # EDA: imbalance 11.4:1
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=11,
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X.fillna(-999), y)

    importance = pd.Series(
        xgb.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    selected = importance[importance > threshold].head(top_k).index.tolist()
    logger.info(f"  Top 5 XGB features: {selected[:5]}")
    logger.info(f"  Selected {len(selected)} features")
    return importance, selected


def combined_selection(
    mi_selected: list,
    xgb_selected: list,
    mode: str = "union",
) -> list:

    logger.info(f"Combined selection mode: {mode}")
    if mode == "union":
        final = list(set(mi_selected) | set(xgb_selected))
    elif mode == "intersection":
        final = list(set(mi_selected) & set(xgb_selected))
    else:
        final = mi_selected

    logger.info(f"  MI selected    : {len(mi_selected)}")
    logger.info(f"  XGB selected   : {len(xgb_selected)}")
    logger.info(f"  Final selected : {len(final)}")
    return final


def selection_report(
    mi_series: pd.Series,
    xgb_series: pd.Series,
    final_features: list,
    top_n: int = 20,
) -> pd.DataFrame:

    report = pd.DataFrame({
        "MI_score":       mi_series,
        "XGB_importance": xgb_series,
    }).fillna(0)

    report["MI_rank"]  = report["MI_score"].rank(ascending=False)
    report["XGB_rank"] = report["XGB_importance"].rank(ascending=False)
    report["avg_rank"] = (report["MI_rank"] + report["XGB_rank"]) / 2
    report["selected"] = report.index.isin(final_features)
    report = report.sort_values("avg_rank")

    logger.info(f"\nTop {top_n} features by average rank:")
    logger.info(f"\n{report.head(top_n).to_string()}")
    return report


def select_features(
    df_train: pd.DataFrame,
    target_col: str = TARGET_COL,
    corr_threshold: float = 0.95,
    top_k: int = TOP_K_DEFAULT,
    mode: str = "union",
) -> Tuple[list, pd.DataFrame, dict]:

    logger.info("=" * 50)
    logger.info(f"FEATURE SELECTION — target={target_col}")
    logger.info(f"Input shape: {df_train.shape}")

    exclude_cols = [TARGET_COL, ID_COL]
    if target_col not in exclude_cols:
        exclude_cols.append(target_col)

    df_filtered, dropped_corr = remove_correlated_features(
        df_train, threshold=corr_threshold, exclude_cols=exclude_cols
    )

    feature_cols = [c for c in df_filtered.columns if c not in exclude_cols]
    X = df_filtered[feature_cols]
    y = df_filtered[target_col]

    mi_series,  mi_selected  = mutual_information_selection(X, y, top_k=top_k)
    xgb_series, xgb_selected = xgboost_importance_selection(X, y, top_k=top_k)
    final_features = combined_selection(mi_selected, xgb_selected, mode=mode)
    report         = selection_report(mi_series, xgb_series, final_features)

    artifacts = {
        "dropped_corr":   dropped_corr,
        "final_features": final_features,
        "report":         report,
    }

    logger.info(f"Final selected features: {len(final_features)}")
    logger.info("FEATURE SELECTION COMPLETE")
    logger.info("=" * 50)
    return final_features, report, artifacts