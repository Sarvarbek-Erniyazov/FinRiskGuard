import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("ieee_cis.feature_engineer")

START_DATE = pd.Timestamp("2017-12-01")

PEAK_FRAUD_HOURS = [5, 6, 7, 8, 9]
RISKY_BROWSERS   = ["opera", "android", "samsung", "firefox", "mobile"]
PROTON_DOMAINS   = ["protonmail.com", "pm.me"]
HIGH_RISK_EMAIL_DOMAINS = [
    "mail.com", "outlook.es", "aim.com", "outlook.com",
    "icloud.com", "gmail.com", "hotmail.com",
]

D_COLS_NORMALIZE = [
    "D1", "D2", "D3", "D4", "D5",
    "D6", "D8", "D9", "D10", "D11",
    "D12", "D13", "D14", "D15",
]

AGG_GROUP_COLS = ["card1", "card1_addr1"]
AGG_TARGET_COL = "TransactionAmt"


def fe_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionAmt" not in df.columns:
        return df
    skew_before = df["TransactionAmt"].skew()
    log_vals    = np.log1p(df["TransactionAmt"])
    skew_after  = log_vals.skew()
    df = pd.concat(
        [df, pd.DataFrame({"FE_amt_log": log_vals}, index=df.index)],
        axis=1
    )
    logger.info("FE: log1p(TransactionAmt)")
    logger.info(f"  Skewness: {skew_before:.2f} → {skew_after:.2f}")
    return df


def fe_temporal(df: pd.DataFrame) -> pd.DataFrame:
    if "TransactionDT" not in df.columns:
        return df
    dt = START_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
    new_cols = {
        "FE_hour"              : dt.dt.hour.astype(np.int8),
        "FE_dayofweek"         : dt.dt.dayofweek.astype(np.int8),
        "FE_day"               : dt.dt.day.astype(np.int8),
        "FE_month"             : dt.dt.month.astype(np.int8),
        "FE_is_night"          : dt.dt.hour.isin(range(0, 6)).astype(np.int8),
        "FE_is_weekend"        : dt.dt.dayofweek.isin([5, 6]).astype(np.int8),
        "FE_is_peak_fraud_hour": dt.dt.hour.isin(PEAK_FRAUD_HOURS).astype(np.int8),
        "_TransactionDT_days"  : (df["TransactionDT"] / 86400).astype(np.float32),
    }
    df = pd.concat(
        [df, pd.DataFrame(new_cols, index=df.index)], axis=1
    )
    logger.info("FE: Temporal features from TransactionDT")
    logger.info(f"  START_DATE         : {START_DATE.date()}")
    logger.info(f"  Date range         : {dt.min()} → {dt.max()}")
    logger.info(f"  PEAK_FRAUD_HOURS   : {PEAK_FRAUD_HOURS}")
    logger.info(
        "  Created: FE_hour, FE_dayofweek, FE_day, FE_month, "
        "FE_is_night, FE_is_weekend, FE_is_peak_fraud_hour, "
        "_TransactionDT_days"
    )
    return df


def fe_d_normalization(df: pd.DataFrame) -> pd.DataFrame:
    if "_TransactionDT_days" not in df.columns:
        logger.info("FE: D normalization skipped — _TransactionDT_days missing")
        return df
    dt_days  = df["_TransactionDT_days"]
    new_cols = {}
    created  = []
    for col in D_COLS_NORMALIZE:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            norm_col           = f"FE_{col}_normalized"
            new_cols[norm_col] = (df[col] - dt_days).astype(np.float32)
            created.append(norm_col)
    if new_cols:
        df = pd.concat(
            [df, pd.DataFrame(new_cols, index=df.index)], axis=1
        )
    logger.info("FE: D column normalization")
    logger.info(f"  Formula  : FE_D{{n}}_normalized = D{{n}} - TransactionDT_days")
    logger.info(f"  Created  : {len(created)} normalized D features")
    logger.info(f"  Columns  : {created}")
    return df


def fe_card_combinations(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    if "card1" in df.columns and "addr1" in df.columns:
        new_cols["card1_addr1"] = (
            df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        )
    card_cols = [c for c in ["card1", "card2", "card3", "card5"] if c in df.columns]
    if card_cols:
        new_cols["card_full"] = (
            df[card_cols].fillna("nan").astype(str).agg("_".join, axis=1)
        )
    if new_cols:
        df = pd.concat(
            [df, pd.DataFrame(new_cols, index=df.index)], axis=1
        )
    logger.info("FE: Card combination fingerprints")
    logger.info(f"  Created: {list(new_cols.keys())}")
    return df


def fe_uid(df: pd.DataFrame) -> pd.DataFrame:
    new_cols  = {}
    uid_parts = []
    for col in ["card1", "card2", "P_emaildomain"]:
        if col in df.columns:
            uid_parts.append(df[col].fillna("nan").astype(str))
    if len(uid_parts) >= 2:
        uid_series = uid_parts[0]
        for part in uid_parts[1:]:
            uid_series = uid_series + "_" + part
        new_cols["FE_uid"] = uid_series

    ext_parts = []
    for col in ["card1", "addr1"]:
        if col in df.columns:
            ext_parts.append(df[col].fillna("nan").astype(str))
    if "FE_D1_normalized" in df.columns:
        d1_binned = (
            (df["FE_D1_normalized"] // 30).fillna(-999).astype(int).astype(str)
        )
        ext_parts.append(d1_binned)
    if len(ext_parts) >= 2:
        uid_ext = ext_parts[0]
        for part in ext_parts[1:]:
            uid_ext = uid_ext + "_" + part
        new_cols["FE_uid_ext"] = uid_ext

    if new_cols:
        df = pd.concat(
            [df, pd.DataFrame(new_cols, index=df.index)], axis=1
        )
    logger.info("FE: UID features")
    logger.info(f"  Created: {list(new_cols.keys())}")
    return df


def fe_email_risk(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    if "P_emaildomain" in df.columns:
        new_cols["FE_P_email_is_proton"] = df["P_emaildomain"].isin(PROTON_DOMAINS).astype(np.int8)
        new_cols["FE_P_email_high_risk"] = df["P_emaildomain"].isin(HIGH_RISK_EMAIL_DOMAINS).astype(np.int8)
    if "R_emaildomain" in df.columns:
        new_cols["FE_R_email_is_proton"] = df["R_emaildomain"].isin(PROTON_DOMAINS).astype(np.int8)
        new_cols["FE_R_email_high_risk"] = df["R_emaildomain"].isin(HIGH_RISK_EMAIL_DOMAINS).astype(np.int8)
    if "P_emaildomain" in df.columns and "R_emaildomain" in df.columns:
        new_cols["FE_email_match"] = (
            df["P_emaildomain"] == df["R_emaildomain"]
        ).astype(np.int8)
    if new_cols:
        df = pd.concat(
            [df, pd.DataFrame(new_cols, index=df.index)], axis=1
        )
    logger.info("FE: Email domain risk flags")
    logger.info(f"  Created: {list(new_cols.keys())}")
    return df


def fe_browser_risk(df: pd.DataFrame) -> pd.DataFrame:
    if "id_31" not in df.columns:
        logger.info("FE: Browser risk flag skipped — id_31 not found")
        return df
    browser_lower = df["id_31"].astype(str).str.lower()
    risky_flag    = browser_lower.str.contains(
        "|".join(RISKY_BROWSERS), na=False
    ).astype(np.int8)
    df = pd.concat(
        [df, pd.DataFrame({"FE_browser_is_risky": risky_flag}, index=df.index)],
        axis=1
    )
    risky_count = risky_flag.sum()
    logger.info("FE: Browser risk flag")
    logger.info(f"  RISKY_BROWSERS     : {RISKY_BROWSERS}")
    logger.info(f"  id_31 dtype        : {df['id_31'].dtype}")
    logger.info(
        f"  Risky transactions : {risky_count:,} "
        f"({risky_count/len(df)*100:.2f}%)"
    )
    logger.info("  Created: FE_browser_is_risky")
    return df


def fe_device_type(df: pd.DataFrame) -> pd.DataFrame:
    if "DeviceType" not in df.columns:
        logger.info("FE: Device type skipped — DeviceType not found")
        return df
    mobile_flag = (
        df["DeviceType"].astype(str).str.lower() == "mobile"
    ).astype(np.int8)
    df = pd.concat(
        [df, pd.DataFrame({"FE_device_is_mobile": mobile_flag}, index=df.index)],
        axis=1
    )
    mobile_count = mobile_flag.sum()
    logger.info("FE: Device type flag")
    logger.info(f"  DeviceType dtype    : {df['DeviceType'].dtype}")
    logger.info(
        f"  Mobile transactions : {mobile_count:,} "
        f"({mobile_count/len(df)*100:.2f}%)"
    )
    logger.info("  Created: FE_device_is_mobile")
    return df


def fe_card_aggregations(
    df: pd.DataFrame,
    agg_maps: dict = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Card-level TransactionAmt aggregations (mean, std, count).
    Groups: card1 and card1_addr1.
    __global__ key stores fallback value for unseen groups.
    __group_col__ removed — group column inferred from feature name.
    """
    new_cols = {}

    if "TransactionAmt" not in df.columns:
        logger.info("FE: Card aggregations skipped — TransactionAmt missing")
        return df, agg_maps or {}

    if agg_maps is not None:
        # Test/val mode — infer group_col from feature name
        for feat, agg_map in agg_maps.items():
            global_val = agg_map.get("__global__", 0)
            group_col  = "card1_addr1" if "card1a1" in feat else "card1"
            if group_col in df.columns:
                new_cols[feat] = df[group_col].map(agg_map).fillna(global_val)
        if new_cols:
            df = pd.concat(
                [df, pd.DataFrame(new_cols, index=df.index)], axis=1
            )
        logger.info("FE: Card aggregations")
        logger.info(f"  [TEST] Applied {len(agg_maps)} aggregation maps")
        return df, agg_maps

    # Train mode
    agg_maps = {}
    for group_col in AGG_GROUP_COLS:
        if group_col not in df.columns:
            continue

        grp          = df.groupby(group_col)["TransactionAmt"]
        global_mean  = df["TransactionAmt"].mean()
        global_std   = df["TransactionAmt"].std()
        global_count = df.groupby(group_col).size().mean()

        prefix    = "card1" if group_col == "card1" else "card1a1"
        mean_key  = f"FE_{prefix}_amt_mean"
        std_key   = f"FE_{prefix}_amt_std"
        cnt_key   = f"FE_{prefix}_amt_count"

        mean_map  = grp.mean().to_dict()
        std_map   = grp.std().fillna(0).to_dict()
        count_map = df.groupby(group_col).size().to_dict()

        # Only __global__ stored — no __group_col__ sentinel
        mean_map["__global__"]  = global_mean
        std_map["__global__"]   = global_std
        count_map["__global__"] = global_count

        new_cols[mean_key] = df[group_col].map(mean_map).fillna(global_mean)
        new_cols[std_key]  = df[group_col].map(std_map).fillna(global_std)
        new_cols[cnt_key]  = df[group_col].map(count_map).fillna(global_count)

        agg_maps[mean_key] = mean_map
        agg_maps[std_key]  = std_map
        agg_maps[cnt_key]  = count_map

    if new_cols:
        df = pd.concat(
            [df, pd.DataFrame(new_cols, index=df.index)], axis=1
        )
    logger.info("FE: Card aggregations (TransactionAmt by card1 and card1_addr1)")
    logger.info(f"  [TRAIN] Created: {list(new_cols.keys())}")
    return df, agg_maps


def fe_frequency_encoding(
    df: pd.DataFrame,
    freq_maps: dict = None,
) -> Tuple[pd.DataFrame, dict]:
    base_cols = [
        c for c in [
            "card1", "card2", "addr1",
            "P_emaildomain", "R_emaildomain",
            "card1_addr1", "card_full",
        ]
        if c in df.columns
    ]
    fe_uid_cols = [c for c in ["FE_uid", "FE_uid_ext"] if c in df.columns]
    encode_cols = base_cols + fe_uid_cols

    if freq_maps is not None:
        new_cols = {}
        for col, freq_map in freq_maps.items():
            if col not in df.columns:
                continue
            feat_name = (
                f"{col}_freq" if col.startswith("FE_")
                else f"FE_{col}_freq"
            )
            new_cols[feat_name] = df[col].map(freq_map).fillna(0)
        if new_cols:
            df = pd.concat(
                [df, pd.DataFrame(new_cols, index=df.index)], axis=1
            )
        logger.info("FE: Frequency encoding")
        logger.info(f"  [TEST] Applied {len(freq_maps)} frequency maps")
        return df, freq_maps

    freq_maps = {}
    new_cols  = {}
    for col in encode_cols:
        freq_map       = df[col].value_counts(normalize=True).to_dict()
        freq_maps[col] = freq_map
        feat_name = (
            f"{col}_freq" if col.startswith("FE_")
            else f"FE_{col}_freq"
        )
        new_cols[feat_name] = df[col].map(freq_map).fillna(0)
    df = pd.concat(
        [df, pd.DataFrame(new_cols, index=df.index)], axis=1
    )
    logger.info("FE: Frequency encoding")
    logger.info(f"  [TRAIN] Encoded {len(encode_cols)} columns: {encode_cols}")
    logger.info(f"  Feature names: {list(new_cols.keys())}")
    return df, freq_maps


def fe_card1_addr1_count(
    df: pd.DataFrame,
    count_map: dict = None,
) -> Tuple[pd.DataFrame, dict]:
    if "card1_addr1" not in df.columns:
        return df, count_map or {}
    if count_map is not None:
        df = pd.concat(
            [df, pd.DataFrame(
                {"FE_card1_addr1_count": df["card1_addr1"].map(count_map).fillna(0)},
                index=df.index
            )],
            axis=1
        )
        logger.info("FE: card1_addr1 count — [TEST] Applied")
        return df, count_map
    count_map = df["card1_addr1"].value_counts().to_dict()
    df = pd.concat(
        [df, pd.DataFrame(
            {"FE_card1_addr1_count": df["card1_addr1"].map(count_map).fillna(0)},
            index=df.index
        )],
        axis=1
    )
    logger.info("FE: card1_addr1 count — [TRAIN] Created")
    return df, count_map


def fe_cleanup_temp(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ["_TransactionDT_days"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"FE: Cleaned temp columns: {drop_cols}")
    return df


def feature_engineer_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    logger.info("=" * 50)
    logger.info("FEATURE ENGINEERING — TRAIN")
    logger.info(f"Input shape: {df.shape}")

    df                = fe_log_transform(df)
    df                = fe_temporal(df)
    df                = fe_d_normalization(df)
    df                = fe_card_combinations(df)
    df                = fe_uid(df)
    df                = fe_email_risk(df)
    df                = fe_browser_risk(df)
    df                = fe_device_type(df)
    df, agg_maps      = fe_card_aggregations(df)
    df, freq_maps     = fe_frequency_encoding(df)
    df, count_map     = fe_card1_addr1_count(df)
    df                = fe_cleanup_temp(df)

    fe_cols = [c for c in df.columns if c.startswith("FE_")]
    logger.info(f"Total FE features created : {len(fe_cols)}")
    logger.info(f"FE columns                : {fe_cols}")
    logger.info(f"Output shape              : {df.shape}")
    logger.info("FEATURE ENGINEERING TRAIN COMPLETE")
    logger.info("=" * 50)

    artifacts = {
        "freq_maps": freq_maps,
        "count_map": count_map,
        "agg_maps" : agg_maps,
    }
    return df, artifacts


def feature_engineer_test(
    df: pd.DataFrame,
    artifacts: dict,
) -> pd.DataFrame:
    logger.info("=" * 50)
    logger.info("FEATURE ENGINEERING — TEST/VAL")
    logger.info(f"Input shape: {df.shape}")

    df    = fe_log_transform(df)
    df    = fe_temporal(df)
    df    = fe_d_normalization(df)
    df    = fe_card_combinations(df)
    df    = fe_uid(df)
    df    = fe_email_risk(df)
    df    = fe_browser_risk(df)
    df    = fe_device_type(df)
    df, _ = fe_card_aggregations(df,  agg_maps=artifacts["agg_maps"])
    df, _ = fe_frequency_encoding(df, freq_maps=artifacts["freq_maps"])
    df, _ = fe_card1_addr1_count(df,  count_map=artifacts["count_map"])
    df    = fe_cleanup_temp(df)

    fe_cols = [c for c in df.columns if c.startswith("FE_")]
    logger.info(f"Total FE features applied : {len(fe_cols)}")
    logger.info(f"Output shape              : {df.shape}")
    logger.info("FEATURE ENGINEERING TEST/VAL COMPLETE")
    logger.info("=" * 50)

    return df