import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("home_credit.feature_engineer")

TARGET_COL            = "TARGET"
ID_COL                = "SK_ID_CURR"
DAYS_EMPLOYED_ANOMALY = 365243


def fe_ext_source(df: pd.DataFrame) -> pd.DataFrame:
    """Create EXT_SOURCE combination features."""
    logger.info("FE: EXT_SOURCE combination features")
    new_cols = {}
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
                if c in df.columns]

    if len(ext_cols) >= 2:
        ext_df = df[ext_cols]
        new_cols["FE_ext_mean"] = ext_df.mean(axis=1)
        new_cols["FE_ext_min"]  = ext_df.min(axis=1)
        new_cols["FE_ext_max"]  = ext_df.max(axis=1)
        new_cols["FE_ext_std"]  = ext_df.std(axis=1).fillna(0)
        new_cols["FE_ext_sum"]  = ext_df.sum(axis=1)

    if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_2" in df.columns:
        new_cols["FE_ext12_prod"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
    if "EXT_SOURCE_2" in df.columns and "EXT_SOURCE_3" in df.columns:
        new_cols["FE_ext23_prod"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_3" in df.columns:
        new_cols["FE_ext13_prod"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info(f"  Created: {list(new_cols.keys())}")
    return df


def fe_age_employment(df: pd.DataFrame) -> pd.DataFrame:
    """Create age and employment ratio features."""
    logger.info("FE: Age & Employment features")
    new_cols = {}

    if "DAYS_BIRTH" in df.columns:
        age_years = abs(df["DAYS_BIRTH"]) / 365
        new_cols["FE_age_years"] = age_years
        new_cols["FE_age_group"] = pd.cut(
            age_years,
            bins=[0, 25, 35, 45, 55, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        new_cols["FE_employment_ratio"] = (
            abs(df["DAYS_EMPLOYED"]) / (abs(df["DAYS_BIRTH"]) + 1)
        )

    if "DAYS_REGISTRATION" in df.columns and "DAYS_BIRTH" in df.columns:
        new_cols["FE_registration_age_ratio"] = (
            abs(df["DAYS_REGISTRATION"]) / (abs(df["DAYS_BIRTH"]) + 1)
        )

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info(f"  Created: {list(new_cols.keys())}")
    return df


def fe_credit_income(df: pd.DataFrame) -> pd.DataFrame:
    """Create credit-to-income and annuity ratio features."""
    logger.info("FE: Credit & Income ratio features")
    new_cols = {}

    if "AMT_INCOME_TOTAL" in df.columns:
        new_cols["FE_income_log"] = np.log1p(df["AMT_INCOME_TOTAL"])

    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        new_cols["FE_credit_income_ratio"] = (
            df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
        )

    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        new_cols["FE_annuity_income_ratio"] = (
            df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
        )

    if "AMT_ANNUITY" in df.columns and "AMT_CREDIT" in df.columns:
        new_cols["FE_annuity_credit_ratio"] = (
            df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
        )

    if "AMT_GOODS_PRICE" in df.columns and "AMT_CREDIT" in df.columns:
        new_cols["FE_credit_goods_diff"]  = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
        new_cols["FE_credit_goods_ratio"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info(f"  Created: {list(new_cols.keys())}")
    return df


def aggregate_bureau(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate bureau + bureau_balance with enhanced overdue features."""
    logger.info("FE: Bureau aggregation (enhanced)")

    # -- Bureau balance aggregations -------------------------------------------
    bb = bureau_balance.copy()

    bb["is_overdue"]  = bb["STATUS"].isin(["1","2","3","4","5"]).astype(np.int8)
    bb["is_closed"]   = (bb["STATUS"] == "C").astype(np.int8)
    bb["is_unknown"]  = (bb["STATUS"] == "X").astype(np.int8)

    # Last 12 months overdue count
    bb_12m = bb[bb["MONTHS_BALANCE"] >= -12]
    bb_12m_agg = bb_12m.groupby("SK_ID_BUREAU").agg(
        bb_overdue_12m=("is_overdue", "sum"),
    ).reset_index()

    # Last 3 months — recency signal
    bb_3m = bb[bb["MONTHS_BALANCE"] >= -3]
    bb_3m_agg = bb_3m.groupby("SK_ID_BUREAU").agg(
        bb_overdue_3m=("is_overdue", "sum"),
    ).reset_index()

    # Full history aggregation
    bb_full = bb.groupby("SK_ID_BUREAU").agg(
        bb_dpd_mean    =("is_overdue", "mean"),
        bb_count       =("STATUS",     "count"),
        bb_C_ratio     =("is_closed",  "mean"),
        bb_X_ratio     =("is_unknown", "mean"),
    ).reset_index()

    # Max consecutive overdue streak
    def max_streak(statuses):
        streak = max_s = 0
        for s in statuses:
            if s in ["1","2","3","4","5"]:
                streak += 1
                max_s = max(max_s, streak)
            else:
                streak = 0
        return max_s

    # Fixed: Added group_keys=False to avoid FutureWarnings in Pandas 2.2+
    streak_df = bureau_balance.sort_values(
        ["SK_ID_BUREAU", "MONTHS_BALANCE"]
    ).groupby("SK_ID_BUREAU", group_keys=False)["STATUS"].apply(max_streak).reset_index()
    streak_df.columns = ["SK_ID_BUREAU", "bb_max_streak"]

    # Merge all bb aggregations
    bb_agg = bb_full \
        .merge(bb_12m_agg, on="SK_ID_BUREAU", how="left") \
        .merge(bb_3m_agg,  on="SK_ID_BUREAU", how="left") \
        .merge(streak_df,  on="SK_ID_BUREAU", how="left")
    bb_agg = bb_agg.fillna(0)

    # -- Merge bb into bureau --------------------------------------------------
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # -- Bureau-level aggregation ----------------------------------------------
    agg = bureau.groupby("SK_ID_CURR").agg(
        bureau_count          =("SK_ID_BUREAU",         "count"),
        bureau_active_count   =("CREDIT_ACTIVE",         lambda x: (x == "Active").sum()),
        bureau_closed_count   =("CREDIT_ACTIVE",         lambda x: (x == "Closed").sum()),
        bureau_credit_sum     =("AMT_CREDIT_SUM",        "sum"),
        bureau_credit_mean     =("AMT_CREDIT_SUM",        "mean"),
        bureau_credit_max      =("AMT_CREDIT_SUM",        "max"),
        bureau_debt_sum        =("AMT_CREDIT_SUM_DEBT",  "sum"),
        bureau_overdue_sum     =("AMT_CREDIT_SUM_OVERDUE","sum"),
        bureau_overdue_count   =("AMT_CREDIT_SUM_OVERDUE",lambda x: (x > 0).sum()),
        bureau_prolong_sum     =("CNT_CREDIT_PROLONG",   "sum"),
        bureau_dpd_mean       =("bb_dpd_mean",          "mean"),
        bureau_dpd_max        =("bb_dpd_mean",          "max"),
        bureau_overdue_12m     =("bb_overdue_12m",       "sum"),
        bureau_overdue_3m      =("bb_overdue_3m",        "sum"),
        bureau_C_ratio         =("bb_C_ratio",           "mean"),
        bureau_X_ratio         =("bb_X_ratio",           "mean"),
        bureau_max_streak      =("bb_max_streak",        "max"),
    ).reset_index()

    agg.columns = [
        "SK_ID_CURR" if c == "SK_ID_CURR" else f"FE_{c}"
        for c in agg.columns
    ]
    logger.info(f"  Bureau agg shape: {agg.shape}")
    return agg


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous_application with trend features."""
    logger.info("FE: Previous Application aggregation (enhanced)")

    prev = prev.copy()
    prev = prev.sort_values(["SK_ID_CURR", "DAYS_DECISION"])

    # -- Base aggregation ------------------------------------------------------
    agg = prev.groupby("SK_ID_CURR").agg(
        prev_count            =("SK_ID_PREV",             "count"),
        prev_approved_count   =("NAME_CONTRACT_STATUS",   lambda x: (x == "Approved").sum()),
        prev_refused_count    =("NAME_CONTRACT_STATUS",   lambda x: (x == "Refused").sum()),
        prev_credit_mean      =("AMT_CREDIT",             "mean"),
        prev_credit_sum       =("AMT_CREDIT",             "sum"),
        prev_annuity_mean     =("AMT_ANNUITY",            "mean"),
        prev_down_payment_mean=("AMT_DOWN_PAYMENT",       "mean"),
        prev_days_decision_mean=("DAYS_DECISION",          "mean"),
        prev_last_decision    =("DAYS_DECISION",          "max"),
        prev_credit_first     =("AMT_CREDIT",             "first"),
        prev_credit_last      =("AMT_CREDIT",             "last"),
    ).reset_index()

    # Derived features
    agg["prev_approved_rate"]  = (
        agg["prev_approved_count"] / (agg["prev_count"] + 1)
    )
    agg["prev_refused_rate"]   = (
        agg["prev_refused_count"] / (agg["prev_count"] + 1)
    )
    # Credit trend: last vs first (Ratio > 1 means customer is requesting more credit)
    agg["prev_credit_trend"]   = (
        agg["prev_credit_last"] / (agg["prev_credit_first"] + 1)
    )
    agg["prev_last_decision_abs"] = abs(agg["prev_last_decision"])

    # Drop helper cols
    agg = agg.drop(columns=["prev_credit_first", "prev_credit_last",
                             "prev_last_decision"])

    # Last 3 applications — approved count
    last3 = prev.groupby("SK_ID_CURR").tail(3)
    last3_agg = last3.groupby("SK_ID_CURR").agg(
        prev_approved_last3=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
    ).reset_index()
    agg = agg.merge(last3_agg, on="SK_ID_CURR", how="left")

    # Rename
    agg.columns = [
        "SK_ID_CURR" if c == "SK_ID_CURR"
        else c if c.startswith("FE_")
        else f"FE_{c}"
        for c in agg.columns
    ]
    logger.info(f"  Previous app agg shape: {agg.shape}")
    return agg


def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance into per-applicant features."""
    logger.info("FE: POS CASH aggregation")

    agg = pos.groupby("SK_ID_CURR").agg(
        pos_count             =("SK_ID_PREV",            "count"),
        pos_months_balance_mean=("MONTHS_BALANCE",       "mean"),
        pos_sk_dpd_mean       =("SK_DPD",                "mean"),
        pos_sk_dpd_max        =("SK_DPD",                "max"),
        pos_sk_dpd_def_mean   =("SK_DPD_DEF",            "mean"),
        pos_completed_count   =("NAME_CONTRACT_STATUS",  lambda x: (x == "Completed").sum()),
    ).reset_index()

    agg.columns = [
        "SK_ID_CURR" if c == "SK_ID_CURR" else f"FE_{c}"
        for c in agg.columns
    ]
    logger.info(f"  POS CASH agg shape: {agg.shape}")
    return agg


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments."""
    logger.info("FE: Installments aggregation")

    inst = inst.copy()
    inst["FE_inst_payment_diff"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["FE_inst_days_diff"]    = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["FE_inst_late"]         = (inst["FE_inst_days_diff"] > 0).astype(int)

    agg = inst.groupby("SK_ID_CURR").agg(
        inst_count            =("SK_ID_PREV",            "count"),
        inst_payment_diff_mean=("FE_inst_payment_diff",  "mean"),
        inst_payment_diff_sum =("FE_inst_payment_diff",  "sum"),
        inst_days_diff_mean   =("FE_inst_days_diff",     "mean"),
        inst_days_diff_max    =("FE_inst_days_diff",     "max"),
        inst_late_count       =("FE_inst_late",          "sum"),
        inst_late_rate        =("FE_inst_late",          "mean"),
    ).reset_index()

    agg.columns = [
        "SK_ID_CURR" if c == "SK_ID_CURR" else f"FE_{c}"
        for c in agg.columns
    ]
    logger.info(f"  Installments agg shape: {agg.shape}")
    return agg


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance including utilization."""
    logger.info("FE: Credit Card aggregation")

    agg = cc.groupby("SK_ID_CURR").agg(
        cc_count          =("SK_ID_PREV",              "count"),
        cc_balance_mean   =("AMT_BALANCE",             "mean"),
        cc_balance_max    =("AMT_BALANCE",             "max"),
        cc_credit_limit_mean=("AMT_CREDIT_LIMIT_ACTUAL","mean"),
        cc_drawings_mean  =("AMT_DRAWINGS_CURRENT",    "mean"),
        cc_drawings_sum   =("AMT_DRAWINGS_CURRENT",    "sum"),
        cc_dpd_mean       =("SK_DPD",                  "mean"),
        cc_dpd_max        =("SK_DPD",                  "max"),
    ).reset_index()

    cc_util = cc.groupby("SK_ID_CURR").agg(
        balance_mean=("AMT_BALANCE",              "mean"),
        limit_mean  =("AMT_CREDIT_LIMIT_ACTUAL",  "mean"),
    ).reset_index()
    cc_util["FE_cc_utilization"] = (
        cc_util["balance_mean"] / (cc_util["limit_mean"] + 1)
    )

    agg = agg.merge(
        cc_util[["SK_ID_CURR", "FE_cc_utilization"]],
        on="SK_ID_CURR", how="left"
    )

    agg.columns = [
        "SK_ID_CURR" if c == "SK_ID_CURR"
        else c if c.startswith("FE_")
        else f"FE_{c}"
        for c in agg.columns
    ]
    logger.info(f"  Credit card agg shape: {agg.shape}")
    return agg


def fe_frequency_encoding(
    df: pd.DataFrame,
    freq_maps: dict = None,
) -> Tuple[pd.DataFrame, dict]:
    """Frequency encoding for high-cardinality categorical columns."""
    logger.info("FE: Frequency encoding")

    encode_cols = [
        "ORGANIZATION_TYPE", "OCCUPATION_TYPE",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_HOUSING_TYPE", "NAME_FAMILY_STATUS",
    ]
    encode_cols = [c for c in encode_cols if c in df.columns]

    if freq_maps is not None:
        new_cols = {
            f"FE_{col}_freq": df[col].map(freq_map).fillna(0)
            for col, freq_map in freq_maps.items()
            if col in df.columns
        }
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        logger.info(f"  [TEST] Applied frequency encoding to {len(freq_maps)} columns")
        return df, freq_maps

    freq_maps = {}
    new_cols  = {}
    for col in encode_cols:
        freq_map       = df[col].value_counts(normalize=True).to_dict()
        freq_maps[col] = freq_map
        new_cols[f"FE_{col}_freq"] = df[col].map(freq_map).fillna(0)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info(f"  [TRAIN] Frequency encoded {len(encode_cols)} columns")
    return df, freq_maps


def fe_target_encoding(
    df: pd.DataFrame,
    target_maps: dict = None,
    target_col: str = TARGET_COL,
    n_folds: int = 5,
    smoothing: float = 20.0,
) -> Tuple[pd.DataFrame, dict]:
    """Target encoding with CV to prevent leakage."""
    logger.info("FE: Target encoding")

    encode_cols = [
        "ORGANIZATION_TYPE",
        "OCCUPATION_TYPE",
    ]
    encode_cols = [c for c in encode_cols if c in df.columns]

    if target_maps is not None:
        new_cols = {}
        for col, tmap in target_maps.items():
            if col not in df.columns:
                continue
            global_mean = tmap["__global__"]
            col_map     = {k: v for k, v in tmap.items() if k != "__global__"}
            new_cols[f"FE_{col}_target_enc"] = (
                df[col].map(col_map).fillna(global_mean)
            )
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        logger.info(f"  [TEST] Applied target encoding to {len(target_maps)} columns")
        return df, target_maps

    if target_col not in df.columns:
        logger.info("  TARGET not found — skipping target encoding")
        return df, {}

    from sklearn.model_selection import KFold

    global_mean = df[target_col].mean()
    kf          = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    target_maps = {}
    new_cols_df = pd.DataFrame(index=df.index)

    for col in encode_cols:
        encoded = pd.Series(np.nan, index=df.index)

        for tr_idx, val_idx in kf.split(df):
            tr_fold  = df.iloc[tr_idx]
            val_fold = df.iloc[val_idx]

            stats = tr_fold.groupby(col)[target_col].agg(["mean", "count"])
            smooth = (
                (stats["count"] * stats["mean"] + smoothing * global_mean)
                / (stats["count"] + smoothing)
            )
            encoded.iloc[val_idx] = val_fold[col].map(smooth).fillna(global_mean)

        new_cols_df[f"FE_{col}_target_enc"] = encoded

        stats_full = df.groupby(col)[target_col].agg(["mean", "count"])
        smooth_full = (
            (stats_full["count"] * stats_full["mean"] + smoothing * global_mean)
            / (stats_full["count"] + smoothing)
        )
        tmap = smooth_full.to_dict()
        tmap["__global__"] = global_mean
        target_maps[col]   = tmap

    df = pd.concat([df, new_cols_df], axis=1)
    logger.info(f"  [TRAIN] Target encoded {len(encode_cols)} columns "
                f"(CV={n_folds}, smoothing={smoothing})")
    return df, target_maps


def _build_aggregations(tables: dict) -> dict:
    """Compute all supplementary aggregations once."""
    return {
        "bureau_agg" : aggregate_bureau(tables["bureau"], tables["bureau_balance"]),
        "prev_agg"   : aggregate_previous_application(tables["previous_app"]),
        "pos_agg"    : aggregate_pos_cash(tables["pos_cash"]),
        "inst_agg"   : aggregate_installments(tables["installments"]),
        "cc_agg"     : aggregate_credit_card(tables["credit_card"]),
    }


def _merge_aggregations(app: pd.DataFrame, aggs: dict) -> pd.DataFrame:
    """Left-merge all cached aggregations onto the application DataFrame."""
    for agg_df in aggs.values():
        app = app.merge(agg_df, on=ID_COL, how="left")
    return app


def feature_engineer_train(
    app: pd.DataFrame,
    tables: dict,
) -> Tuple[pd.DataFrame, dict]:
    """Build all FE features for train set. Returns fitted artifacts."""
    logger.info("=" * 50)
    logger.info("FEATURE ENGINEERING TRAIN DATA")
    logger.info(f"Input shape: {app.shape}")

    app = fe_ext_source(app)
    app = fe_age_employment(app)
    app = fe_credit_income(app)
    app, freq_maps   = fe_frequency_encoding(app)
    app, target_maps = fe_target_encoding(app)

    aggs = _build_aggregations(tables)
    app  = _merge_aggregations(app, aggs)

    fe_cols = [c for c in app.columns if c.startswith("FE_")]
    logger.info(f"Total FE features created: {len(fe_cols)}")
    logger.info(f"Output shape: {app.shape}")
    logger.info("FEATURE ENGINEERING TRAIN COMPLETE")
    logger.info("=" * 50)

    artifacts = {
        "freq_maps"  : freq_maps,
        "target_maps": target_maps,
        "bureau_agg" : aggs["bureau_agg"],
        "prev_agg"   : aggs["prev_agg"],
        "pos_agg"    : aggs["pos_agg"],
        "inst_agg"   : aggs["inst_agg"],
        "cc_agg"     : aggs["cc_agg"],
    }
    return app, artifacts


def feature_engineer_test(
    app: pd.DataFrame,
    tables: dict,
    artifacts: dict,
) -> pd.DataFrame:
    """Apply fitted FE artifacts to val or test."""
    logger.info("=" * 50)
    logger.info("FEATURE ENGINEERING TEST DATA")
    logger.info(f"Input shape: {app.shape}")

    app = fe_ext_source(app)
    app = fe_age_employment(app)
    app = fe_credit_income(app)
    app, _ = fe_frequency_encoding(app, freq_maps=artifacts["freq_maps"])
    app, _ = fe_target_encoding(app,    target_maps=artifacts["target_maps"])

    aggs = {
        "bureau_agg" : artifacts["bureau_agg"],
        "prev_agg"   : artifacts["prev_agg"],
        "pos_agg"    : artifacts["pos_agg"],
        "inst_agg"   : artifacts["inst_agg"],
        "cc_agg"     : artifacts["cc_agg"],
    }
    app = _merge_aggregations(app, aggs)

    fe_cols = [c for c in app.columns if c.startswith("FE_")]
    logger.info(f"Total FE features applied: {len(fe_cols)}")
    logger.info(f"Output shape: {app.shape}")
    logger.info("FEATURE ENGINEERING TEST COMPLETE")
    logger.info("=" * 50)

    return app