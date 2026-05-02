import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("ieee_cis.feature_selector")

TARGET_COL = "isFraud"
TOP_K      = 200

EXCLUDE_COLS = [
    "isFraud", "TransactionID", "TransactionDT",
    "card1_addr1", "card_full", "FE_uid", "FE_uid_ext",
]


def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    dropped_corr_fitted: list = None,
) -> Tuple[pd.DataFrame, list]:
    """
    Remove highly correlated features.
    D_normalized cols may be dropped here — force-included later.
    Train: compute drop list. Test: apply fitted list.
    """
    if dropped_corr_fitted is not None:
        cols = [c for c in dropped_corr_fitted if c in df.columns]
        df   = df.drop(columns=cols)
        logger.info(f"[TEST] Dropped {len(cols)} correlated features")
        return df, dropped_corr_fitted

    logger.info(f"Correlation filter: threshold={threshold}")
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in EXCLUDE_COLS
    ]

    corr_matrix = df[feature_cols].corr().abs()
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    drop_cols = [
        col for col in upper.columns
        if any(upper[col] > threshold)
    ]
    df = df.drop(columns=drop_cols)

    logger.info(
        f"  Dropped {len(drop_cols)} correlated features "
        f"(threshold={threshold})"
    )
    logger.info(f"  Sample dropped  : {drop_cols[:5]}")
    logger.info(f"  Remaining cols  : {df.shape[1]}")
    return df, drop_cols


def mutual_information_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = TOP_K,
) -> Tuple[pd.Series, list]:
    """
    Select top_k features by Mutual Information score.
    NaN filled with -999 sentinel before MI computation.
    """
    from sklearn.feature_selection import mutual_info_classif

    logger.info(f"Mutual Information selection: top_k={top_k}")
    mi_scores = mutual_info_classif(
        X.fillna(-999), y, random_state=42
    )
    mi_series = pd.Series(
        mi_scores, index=X.columns
    ).sort_values(ascending=False)

    selected = mi_series.head(top_k).index.tolist()

    logger.info(f"  Top 5 MI features : {selected[:5]}")
    logger.info(
        f"  MI score range    : "
        f"{mi_series.iloc[0]:.4f} → {mi_series.iloc[top_k-1]:.4f}"
    )
    logger.info(f"  Selected          : {len(selected)} features")
    return mi_series, selected


def xgboost_importance_selection(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = TOP_K,
) -> Tuple[pd.Series, list]:
    """
    Select top_k features by XGBoost feature importance.
    scale_pos_weight=28 — class imbalance ratio (27.6:1).
    """
    from xgboost import XGBClassifier

    logger.info(f"XGBoost importance selection: top_k={top_k}")

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=28,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(X.fillna(-999), y)

    importance = pd.Series(
        xgb.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    selected = (
        importance[importance > 0.0001]
        .head(top_k)
        .index.tolist()
    )

    logger.info(f"  Top 5 XGB features : {selected[:5]}")
    logger.info(
        f"  Importance range   : "
        f"{importance.iloc[0]:.4f} → {importance.iloc[len(selected)-1]:.6f}"
    )
    logger.info(f"  Selected           : {len(selected)} features")
    return importance, selected


def combined_selection(
    mi_selected: list,
    xgb_selected: list,
    mi_series: pd.Series,
    xgb_series: pd.Series,
    top_k: int = TOP_K,
) -> Tuple[list, list, list]:
    """
    Combine MI and XGB selections via rank-based strategy.
    Top top_k features by average MI+XGB rank selected.
    """
    union_set        = set(mi_selected) | set(xgb_selected)
    intersection_set = set(mi_selected) & set(xgb_selected)

    all_features = list(union_set)
    report       = pd.DataFrame({
        "MI_score"       : mi_series.reindex(all_features).fillna(0),
        "XGB_importance" : xgb_series.reindex(all_features).fillna(0),
    })
    report["MI_rank"]  = report["MI_score"].rank(ascending=False)
    report["XGB_rank"] = report["XGB_importance"].rank(ascending=False)
    report["avg_rank"] = (report["MI_rank"] + report["XGB_rank"]) / 2

    union_features        = list(union_set)
    intersection_features = list(intersection_set)
    rank_features         = (
        report.sort_values("avg_rank")
        .head(top_k)
        .index.tolist()
    )

    logger.info("Combined selection summary:")
    logger.info(f"  MI selected           : {len(mi_selected)}")
    logger.info(f"  XGB selected          : {len(xgb_selected)}")
    logger.info(f"  Union                 : {len(union_features)}")
    logger.info(f"  Intersection          : {len(intersection_features)}")
    logger.info(f"  Rank-based (top-{top_k}) : {len(rank_features)}")
    logger.info(
        f"  MI∩XGB overlap        : "
        f"{len(intersection_features)/len(mi_selected)*100:.1f}% of MI selected"
    )
    return union_features, intersection_features, rank_features


def selection_report(
    mi_series: pd.Series,
    xgb_series: pd.Series,
    final_features: list,
    top_n: int = 30,
) -> pd.DataFrame:
    """Build and log feature selection report."""
    all_features = list(set(mi_series.index) | set(xgb_series.index))
    report       = pd.DataFrame({
        "MI_score"       : mi_series.reindex(all_features).fillna(0),
        "XGB_importance" : xgb_series.reindex(all_features).fillna(0),
    })
    report["MI_rank"]  = report["MI_score"].rank(ascending=False)
    report["XGB_rank"] = report["XGB_importance"].rank(ascending=False)
    report["avg_rank"] = (report["MI_rank"] + report["XGB_rank"]) / 2
    report["selected"] = report.index.isin(final_features)
    report             = report.sort_values("avg_rank")

    logger.info(f"\nTop {top_n} features by average rank:")
    logger.info(f"\n{report.head(top_n).to_string()}")

    fe_count     = sum(1 for c in final_features if c.startswith("FE_"))
    nan_count    = sum(1 for c in final_features if c.endswith("_isnan"))
    d_norm_count = sum(1 for c in final_features if "normalized" in c)
    raw_count    = len(final_features) - fe_count - nan_count

    logger.info("\nSelected feature breakdown:")
    logger.info(f"  FE_ engineered     : {fe_count}")
    logger.info(f"  └─ D_normalized    : {d_norm_count}")
    logger.info(f"  _isnan flags       : {nan_count}")
    logger.info(f"  Raw features       : {raw_count}")
    logger.info(f"  TOTAL              : {len(final_features)}")
    return report


def select_features(
    df_train: pd.DataFrame,
    target_col: str = TARGET_COL,
    corr_threshold: float = 0.95,
    top_k: int = TOP_K,
    mode: str = "rank",
) -> Tuple[list, pd.DataFrame, dict]:
    """
    Full feature selection pipeline.
    Mode: 'rank' — top 200 by average MI+XGB rank.

    D_normalized features are force-included after rank selection
    because they are dropped by the correlation filter (corr > 0.95
    with original D columns) despite carrying critical client-stable
    temporal signal absent from raw D columns.
    """
    logger.info("=" * 50)
    logger.info(
        f"FEATURE SELECTION — target={target_col} | "
        f"mode={mode} | top_k={top_k}"
    )
    logger.info(f"Input shape: {df_train.shape}")

    # Step 1: correlation filter
    df_filtered, dropped_corr = remove_correlated_features(
        df_train, threshold=corr_threshold
    )

    # Step 2: X, y
    feature_cols = [
        c for c in df_filtered.columns
        if c not in EXCLUDE_COLS
    ]
    X = df_filtered[feature_cols]
    y = df_filtered[target_col]
    logger.info(f"Features after correlation filter: {X.shape[1]}")

    # Step 3: MI selection
    mi_series, mi_selected = mutual_information_selection(
        X, y, top_k=top_k
    )

    # Step 4: XGB selection
    xgb_series, xgb_selected = xgboost_importance_selection(
        X, y, top_k=top_k
    )

    # Step 5: combined
    union_f, intersection_f, rank_f = combined_selection(
        mi_selected, xgb_selected,
        mi_series, xgb_series,
        top_k=top_k,
    )

    # Step 6: final selection
    if mode == "union":
        final_features = union_f
    elif mode == "intersection":
        final_features = intersection_f
    else:
        final_features = rank_f

    # Step 7: force-include D_normalized
    # These are dropped by correlation filter (corr > 0.95 with raw D cols)
    # but contain client-stable temporal signal not in raw D columns.
    d_norm_cols   = [
        c for c in df_train.columns
        if c.startswith("FE_D") and c.endswith("_normalized")
    ]
    force_include = [c for c in d_norm_cols if c not in final_features]
    if force_include:
        final_features = final_features + force_include
        logger.info(
            f"Force-included {len(force_include)} D_normalized features "
            f"(dropped by correlation filter): {force_include}"
        )

    # Step 8: report
    report = selection_report(mi_series, xgb_series, final_features)

    logger.info(f"\nFinal selected features ({mode}): {len(final_features)}")
    logger.info("FEATURE SELECTION COMPLETE")
    logger.info("=" * 50)

    artifacts = {
        "dropped_corr"         : dropped_corr,
        "final_features"       : final_features,
        "union_features"       : union_f,
        "intersection_features": intersection_f,
        "rank_features"        : rank_f,
        "report"               : report,
        "mode"                 : mode,
        "top_k"                : top_k,
    }
    return final_features, report, artifacts