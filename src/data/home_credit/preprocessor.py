import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("home_credit.preprocessor")

# ── Constants ────────────────────────────────────────────────────────────────

TARGET_COL             = "TARGET"
ID_COL                 = "SK_ID_CURR"
DAYS_EMPLOYED_ANOMALY  = 365243

# EXT_SOURCE_1 has 65.99% missing — threshold must be > 0.66 to keep it
HIGH_MISSING_THRESHOLD = 0.67

# EDA: important columns for NaN flags
NAN_FLAG_COLS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_GOODS_PRICE",
    "AMT_ANNUITY",
    "OWN_CAR_AGE",
    "DAYS_LAST_PHONE_CHANGE",
]


def get_encoder_cols(encoder: OrdinalEncoder) -> list:
    return encoder.feature_names_in_.tolist()


# ── Step 1: Drop high missing ─────────────────────────────────────────────────

def drop_high_missing(
    df: pd.DataFrame,
    threshold: float = HIGH_MISSING_THRESHOLD,
    drop_cols_fitted: list = None,
) -> Tuple[pd.DataFrame, list]:

    if drop_cols_fitted is not None:
        cols_to_drop = [c for c in drop_cols_fitted if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        logger.info(f"[TEST] Dropped {len(cols_to_drop)} high-missing columns")
        return df, drop_cols_fitted

    exclude     = [c for c in [TARGET_COL, ID_COL] if c in df.columns]
    feature_df  = df.drop(columns=exclude)
    missing_pct = feature_df.isnull().mean()
    drop_cols   = missing_pct[missing_pct > threshold].index.tolist()
    df          = df.drop(columns=drop_cols)

    logger.info(f"[TRAIN] Dropped {len(drop_cols)} columns with >{threshold*100:.0f}% missing")
    logger.info(f"  Sample: {drop_cols[:5]}...")
    return df, drop_cols


# ── Step 2: DAYS_EMPLOYED anomaly fix ────────────────────────────────────────

def fix_days_employed(
    df: pd.DataFrame,
    anomaly_median: float = None,
) -> Tuple[pd.DataFrame, float]:

    if "DAYS_EMPLOYED" not in df.columns:
        return df, None

    if anomaly_median is not None:
        df = df.copy()
        df["DAYS_EMPLOYED_ANOM"] = (
            df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY
        ).astype(np.int8)
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
            DAYS_EMPLOYED_ANOMALY, anomaly_median
        )
        logger.info("[TEST] Applied DAYS_EMPLOYED anomaly fix")
        return df, anomaly_median

    anom_mask     = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY
    normal_median = df.loc[~anom_mask, "DAYS_EMPLOYED"].median()

    new_cols = {
        "DAYS_EMPLOYED_ANOM": anom_mask.astype(np.int8),
        "DAYS_EMPLOYED":      df["DAYS_EMPLOYED"].replace(
            DAYS_EMPLOYED_ANOMALY, normal_median
        ),
    }
    df = df.assign(**new_cols)

    logger.info(
        f"[TRAIN] DAYS_EMPLOYED anomaly: {anom_mask.sum():,} rows fixed "
        f"→ median={normal_median:.0f}"
    )
    return df, normal_median


# ── Step 3: NaN flags ────────────────────────────────────────────────────────

def add_nan_flags(
    df: pd.DataFrame,
    nan_flag_cols_fitted: list = None,
) -> Tuple[pd.DataFrame, list]:

    if nan_flag_cols_fitted is not None:
        # Apply train nan flags to val/test — same columns regardless of missing
        new_cols = {}
        for col in nan_flag_cols_fitted:
            src_col = col.replace("_isnan", "")
            if src_col in df.columns:
                new_cols[col] = df[src_col].isnull().astype(np.int8).values
            else:
                new_cols[col] = np.zeros(len(df), dtype=np.int8)

        if new_cols:
            df = pd.concat(
                [df, pd.DataFrame(new_cols, index=df.index)],
                axis=1,
            )
        logger.info(f"[TEST] Applied {len(new_cols)} NaN flag columns from train")
        return df, nan_flag_cols_fitted

    new_cols          = {}
    nan_flag_cols_out = []
    for col in NAN_FLAG_COLS:
        if col in df.columns and df[col].isnull().any():
            flag_col = f"{col}_isnan"
            new_cols[flag_col] = df[col].isnull().astype(np.int8).values
            nan_flag_cols_out.append(flag_col)

    if new_cols:
        df = pd.concat(
            [df, pd.DataFrame(new_cols, index=df.index)],
            axis=1,
        )
    logger.info(f"[TRAIN] Added {len(new_cols)} NaN flag columns: {nan_flag_cols_out}")
    return df, nan_flag_cols_out


# ── Step 4: Impute numerical ──────────────────────────────────────────────────

def impute_numerical(
    df: pd.DataFrame,
    fill_values: dict = None,
) -> Tuple[pd.DataFrame, dict]:

    exclude  = [c for c in [TARGET_COL, ID_COL] if c in df.columns]
    num_cols = [
        c for c in df.select_dtypes(
            include=["float64", "float32", "int64", "int32"]
        ).columns
        if c not in exclude
    ]

    if fill_values is not None:
        updates = {
            col: df[col].fillna(val)
            for col, val in fill_values.items()
            if col in df.columns
        }
        if updates:
            df = df.assign(**updates)
        logger.info(f"[TEST] Applied numerical imputation to {len(fill_values)} columns")
        return df, fill_values

    fill_values = {}
    updates     = {}
    for col in num_cols:
        if df[col].isnull().any():
            median_val       = df[col].median()
            fill_values[col] = median_val
            updates[col]     = df[col].fillna(median_val)

    if updates:
        df = df.assign(**updates)
    logger.info(f"[TRAIN] Imputed {len(fill_values)} numerical columns with median")
    return df, fill_values


# ── Step 5: Impute categorical ────────────────────────────────────────────────

def impute_categorical(
    df: pd.DataFrame,
    cat_fill_values: dict = None,
) -> Tuple[pd.DataFrame, dict]:

    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    if cat_fill_values is not None:
        updates = {
            col: df[col].fillna(val)
            for col, val in cat_fill_values.items()
            if col in df.columns
        }
        if updates:
            df = df.assign(**updates)
        logger.info(f"[TEST] Applied categorical imputation to {len(cat_fill_values)} columns")
        return df, cat_fill_values

    cat_fill_values = {}
    updates         = {}
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val             = df[col].mode()[0]
            cat_fill_values[col] = mode_val
            updates[col]         = df[col].fillna(mode_val)

    if updates:
        df = df.assign(**updates)
    logger.info(f"[TRAIN] Imputed {len(cat_fill_values)} categorical columns with mode")
    return df, cat_fill_values


# ── Step 6: Encode categoricals ───────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    encoder: OrdinalEncoder = None,
    cat_cols: list = None,
) -> Tuple[pd.DataFrame, OrdinalEncoder, list]:

    if encoder is not None:
        train_cat_cols = get_encoder_cols(encoder)
        missing_cols   = [c for c in train_cat_cols if c not in df.columns]
        if missing_cols:
            missing_df = pd.DataFrame(
                "missing", index=df.index, columns=missing_cols
            )
            df = pd.concat([df, missing_df], axis=1)
            logger.info(f"[TEST] Added {len(missing_cols)} missing columns")

        df = df.copy()
        df[train_cat_cols] = encoder.transform(df[train_cat_cols].astype(str))
        logger.info(f"[TEST] OrdinalEncoder applied to {len(train_cat_cols)} columns")
        return df, encoder, train_cat_cols

    if cat_cols is None:
        cat_cols = [
            c for c in df.select_dtypes(include=["object", "string"]).columns
            if c != ID_COL
        ]

    cat_cols = [c for c in cat_cols if c in df.columns]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
    )
    df = df.copy()
    df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))
    logger.info(f"[TRAIN] OrdinalEncoder fitted on {len(cat_cols)} columns")
    return df, encoder, cat_cols


# ── Main pipelines ────────────────────────────────────────────────────────────

def preprocess_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    logger.info("=" * 50)
    logger.info("PREPROCESSING TRAIN DATA")
    logger.info(f"Input shape: {df.shape}")

    df, drop_cols            = drop_high_missing(df)
    df, anomaly_median       = fix_days_employed(df)
    df, nan_flag_cols        = add_nan_flags(df)
    df, num_fills            = impute_numerical(df)
    df, cat_fills            = impute_categorical(df)
    df, encoder, cat_cols    = encode_categoricals(df)

    artifacts = {
        "drop_cols":      drop_cols,
        "anomaly_median": anomaly_median,
        "nan_flag_cols":  nan_flag_cols,
        "num_fills":      num_fills,
        "cat_fills":      cat_fills,
        "encoder":        encoder,
        "cat_cols":       cat_cols,
    }

    logger.info(f"Output shape: {df.shape}")
    logger.info("PREPROCESSING TRAIN COMPLETE")
    logger.info("=" * 50)
    return df, artifacts


def preprocess_test(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    logger.info("=" * 50)
    logger.info("PREPROCESSING TEST DATA")
    logger.info(f"Input shape: {df.shape}")

    df, _ = drop_high_missing(df,    drop_cols_fitted=artifacts["drop_cols"])
    df, _ = fix_days_employed(df,    anomaly_median=artifacts["anomaly_median"])
    df, _ = add_nan_flags(df,        nan_flag_cols_fitted=artifacts["nan_flag_cols"])
    df, _ = impute_numerical(df,     fill_values=artifacts["num_fills"])
    df, _ = impute_categorical(df,   cat_fill_values=artifacts["cat_fills"])
    df, _, _ = encode_categoricals(df, encoder=artifacts["encoder"])

    logger.info(f"Output shape: {df.shape}")
    logger.info("PREPROCESSING TEST COMPLETE")
    logger.info("=" * 50)
    return df