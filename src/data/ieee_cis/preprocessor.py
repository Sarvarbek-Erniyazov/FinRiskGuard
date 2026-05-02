import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("ieee_cis.preprocessor")

HIGH_MISSING_THRESHOLD = 0.90

DROP_HIGH_MISSING_COLS = [
    "id_24", "id_25", "id_07", "id_08", "id_21",
    "id_26", "id_27", "id_23", "id_22",
    "dist2", "D7", "id_18",
]

REDUNDANT_COLS = ["C1", "C4", "C8", "C12", "V242", "V244", "V49", "V90"]

NAN_FLAG_COLS = [
    "dist1",
    "D1",  "D2",  "D3",  "D4",  "D5",
    "D6",  "D8",  "D9",  "D10", "D11",
    "D12", "D13", "D14", "D15",
]

M_COLS   = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]
M4_MAP   = {"M0": 0, "M1": 1, "M2": 2}
BOOL_MAP = {"T": 1, "F": 0}

# FE numerical features that must NOT be OrdinalEncoded.
# These are float aggregation features created in feature_engineer.py.
# Without this exclusion, pandas may detect them as object dtype
# due to __group_col__ / __global__ sentinel keys in agg_maps dict,
# causing OrdinalEncoder to incorrectly treat them as categorical.
FE_NUMERICAL_COLS = [
    "FE_card1_amt_mean",
    "FE_card1_amt_std",
    "FE_card1_amt_count",
    "FE_card1a1_amt_mean",
    "FE_card1a1_amt_std",
    "FE_card1a1_amt_count",
]

# Columns that are never passed to OrdinalEncoder
ENCODE_EXCLUDE = ["TransactionID"] + FE_NUMERICAL_COLS


def drop_high_missing(
    df: pd.DataFrame,
    threshold: float = HIGH_MISSING_THRESHOLD,
    drop_cols_fitted: list = None,
) -> Tuple[pd.DataFrame, list]:
    """
    Drop columns exceeding missing value threshold.
    Train: compute and fit drop list.
    Test : apply fitted drop list from train.
    """
    if drop_cols_fitted is not None:
        cols = [c for c in drop_cols_fitted if c in df.columns]
        df   = df.drop(columns=cols)
        logger.info(f"[TEST] Dropped {len(cols)} high-missing columns")
        logger.info(f"  Columns: {cols}")
        return df, drop_cols_fitted

    exclude     = ["isFraud", "TransactionID", "TransactionDT"]
    feature_df  = df.drop(
        columns=[c for c in exclude if c in df.columns]
    )
    missing_pct = feature_df.isnull().mean()
    drop_cols   = missing_pct[missing_pct > threshold].index.tolist()
    df          = df.drop(columns=drop_cols)

    logger.info(
        f"[TRAIN] Dropped {len(drop_cols)} columns "
        f"with >{threshold*100:.0f}% missing"
    )
    logger.info(f"  Dropped: {drop_cols}")
    logger.info(f"  EDA reference list   : {DROP_HIGH_MISSING_COLS}")
    return df, drop_cols


def drop_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """Drop manually identified redundant columns (high collinearity)."""
    cols = [c for c in REDUNDANT_COLS if c in df.columns]
    df   = df.drop(columns=cols)
    logger.info(f"Dropped {len(cols)} redundant columns: {cols}")
    return df


def add_nan_flags(
    df: pd.DataFrame,
    nan_flag_cols_fitted: list = None,
) -> Tuple[pd.DataFrame, list]:
    """
    Add binary NaN indicator columns for informative missing features.
    NaN in D columns signals absence of transaction history.
    Train: detect which NAN_FLAG_COLS have missing values.
    Test : apply same flag columns as train.
    """
    if nan_flag_cols_fitted is not None:
        new_cols = {}
        for flag_col in nan_flag_cols_fitted:
            src = flag_col.replace("_isnan", "")
            if src in df.columns:
                new_cols[flag_col] = (
                    df[src].isnull().astype(np.int8).values
                )
            else:
                new_cols[flag_col] = np.zeros(len(df), dtype=np.int8)
                logger.info(
                    f"  WARNING: {src} not found in test — "
                    f"{flag_col} filled with 0"
                )
        if new_cols:
            df = pd.concat(
                [df, pd.DataFrame(new_cols, index=df.index)], axis=1
            )
        logger.info(
            f"[TEST] Applied {len(new_cols)} NaN flag columns from train"
        )
        return df, nan_flag_cols_fitted

    new_cols    = {}
    fitted_cols = []
    for col in NAN_FLAG_COLS:
        if col in df.columns and df[col].isnull().any():
            flag           = f"{col}_isnan"
            new_cols[flag] = df[col].isnull().astype(np.int8).values
            fitted_cols.append(flag)

    if new_cols:
        df = pd.concat(
            [df, pd.DataFrame(new_cols, index=df.index)], axis=1
        )
    logger.info(f"[TRAIN] Added {len(new_cols)} NaN flag columns")
    logger.info(f"  Flags: {fitted_cols}")
    return df, fitted_cols


def encode_m_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode M columns from string T/F to binary 0/1.
    NaN encoded as -1 — preserves NaN signal.
    M4 encoded ordinally: M0=0, M1=1, M2=2.
    """
    for col in M_COLS:
        if col in df.columns:
            df[col] = (
                df[col].map(BOOL_MAP).fillna(-1).astype(np.int8)
            )
    if "M4" in df.columns:
        df["M4"] = (
            df["M4"].map(M4_MAP).fillna(-1).astype(np.int8)
        )
    logger.info(
        "Encoded M columns: T=1, F=0, NaN=-1 | "
        "M4 ordinal: M0=0, M1=1, M2=2"
    )
    return df


def impute_d_columns(
    df: pd.DataFrame,
    d_medians: dict = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Impute D columns using card1 group median.
    D columns represent time deltas relative to card activity.
    Same card1 group shares similar temporal patterns.
    Global median used as fallback for unseen card1 values.
    """
    d_cols = [
        c for c in df.columns
        if c.startswith("D")
        and c[1:].isdigit()
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if d_medians is not None:
        for col, fill_map in d_medians.items():
            if col not in df.columns:
                continue
            global_med = fill_map.get("__global__", 0)
            df[col] = df.groupby("card1")[col].transform(
                lambda x: x.fillna(x.median())
            )
            df[col] = df[col].fillna(global_med)
        logger.info(
            f"[TEST] Applied card1 group imputation "
            f"to {len(d_medians)} D columns"
        )
        return df, d_medians

    d_medians = {}
    for col in d_cols:
        if df[col].isnull().sum() == 0:
            continue
        global_med       = df[col].median()
        d_medians[col]   = {"__global__": global_med}
        df[col] = df.groupby("card1")[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df[col].fillna(global_med)

    logger.info(
        f"[TRAIN] Imputed {len(d_medians)} D columns "
        f"by card1 group median"
    )
    return df, d_medians


def impute_numerical(
    df: pd.DataFrame,
    fill_values: dict = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Impute numerical columns with median.
    Median chosen over mean — robust to outliers.
    """
    exclude  = ["isFraud", "TransactionID", "TransactionDT"]
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
        logger.info(
            f"[TEST] Applied median imputation "
            f"to {len(fill_values)} numerical columns"
        )
        return df, fill_values

    fill_values = {}
    updates     = {}
    for col in num_cols:
        if df[col].isnull().any():
            med              = df[col].median()
            fill_values[col] = med
            updates[col]     = df[col].fillna(med)

    if updates:
        df = df.assign(**updates)
    logger.info(
        f"[TRAIN] Imputed {len(fill_values)} numerical columns "
        f"with median"
    )
    return df, fill_values


def impute_categorical(
    df: pd.DataFrame,
    cat_fill_values: dict = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Impute categorical (object/str) columns with mode.
    Applied before OrdinalEncoder to avoid unknown value issues.
    Excludes FE_NUMERICAL_COLS — those are float, not categorical.
    """
    cat_cols = [
        c for c in df.select_dtypes(include=["object"]).columns
        if c not in ENCODE_EXCLUDE
    ]

    if cat_fill_values is not None:
        updates = {
            col: df[col].fillna(val)
            for col, val in cat_fill_values.items()
            if col in df.columns
        }
        if updates:
            df = df.assign(**updates)
        logger.info(
            f"[TEST] Applied mode imputation "
            f"to {len(cat_fill_values)} categorical columns"
        )
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
    logger.info(
        f"[TRAIN] Imputed {len(cat_fill_values)} categorical columns "
        f"with mode"
    )
    return df, cat_fill_values


def fix_fe_numerical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure FE numerical aggregation columns have correct float64 dtype.
    These columns may be detected as object dtype due to agg_maps
    dict sentinel keys (__group_col__, __global__) in feature_engineer.
    Converting to float64 before encode_categoricals prevents them
    from being incorrectly passed to OrdinalEncoder.
    """
    fixed = []
    for col in FE_NUMERICAL_COLS:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                    np.float64
                )
                fixed.append(col)
            except Exception:
                pass
    if fixed:
        logger.info(
            f"[FIX] Converted {len(fixed)} FE numerical cols to float64: "
            f"{fixed}"
        )
    return df


def encode_categoricals(
    df: pd.DataFrame,
    encoder: OrdinalEncoder = None,
    cat_cols: list = None,
) -> Tuple[pd.DataFrame, OrdinalEncoder, list]:
    """
    Ordinal encode categorical (string) columns.
    handle_unknown='use_encoded_value' with unknown_value=-1
    ensures unseen categories in test do not cause errors.

    FE_NUMERICAL_COLS are explicitly excluded — they are float
    aggregation features, not categorical, and must never be encoded.

    Must run AFTER feature_engineer.py — encoding destroys raw strings
    needed for browser/device/email feature engineering.
    """
    if encoder is not None:
        train_cat_cols = encoder.feature_names_in_.tolist()
        missing_cols   = [
            c for c in train_cat_cols if c not in df.columns
        ]

        if missing_cols:
            missing_df = pd.DataFrame(
                "missing", index=df.index, columns=missing_cols
            )
            df = pd.concat([df, missing_df], axis=1)
            logger.info(
                f"[TEST] Added {len(missing_cols)} missing columns "
                f"with placeholder: {missing_cols}"
            )

        df = df.copy()
        df[train_cat_cols] = encoder.transform(
            df[train_cat_cols].astype(str)
        )
        logger.info(
            f"[TEST] OrdinalEncoder applied "
            f"to {len(train_cat_cols)} columns"
        )
        return df, encoder, train_cat_cols

    # Train mode — exclude FE_NUMERICAL_COLS explicitly
    if cat_cols is None:
        cat_cols = [
            c for c in df.select_dtypes(include=["object"]).columns
            if c not in ENCODE_EXCLUDE
        ]

    cat_cols = [c for c in cat_cols if c in df.columns]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
    )
    df = df.copy()
    df[cat_cols] = encoder.fit_transform(df[cat_cols].astype(str))
    logger.info(
        f"[TRAIN] OrdinalEncoder fitted "
        f"on {len(cat_cols)} columns: {cat_cols}"
    )
    return df, encoder, cat_cols


def preprocess_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Full preprocessing pipeline for training data.
    Order:
      1. Drop high-missing columns  (>90%)
      2. Drop redundant columns
      3. Fix FE numerical dtypes    (prevent OrdinalEncoder bug)
      4. Add NaN flag columns
      5. Encode M columns
      6. Impute D columns           (card1 group median)
      7. Impute numerical           (global median)
      8. Impute categorical         (mode)
      9. Encode categoricals        (OrdinalEncoder)

    Must run AFTER feature_engineer_train().
    """
    logger.info("=" * 50)
    logger.info("PREPROCESSING — TRAIN")
    logger.info(f"Input shape: {df.shape}")

    df, drop_cols         = drop_high_missing(df)
    df                    = drop_redundant(df)
    df                    = fix_fe_numerical_dtypes(df)
    df, nan_flag_cols     = add_nan_flags(df)
    df                    = encode_m_columns(df)
    df, d_medians         = impute_d_columns(df)
    df, num_fills         = impute_numerical(df)
    df, cat_fills         = impute_categorical(df)
    df, encoder, cat_cols = encode_categoricals(df)

    artifacts = {
        "drop_cols"    : drop_cols,
        "nan_flag_cols": nan_flag_cols,
        "d_medians"    : d_medians,
        "num_fills"    : num_fills,
        "cat_fills"    : cat_fills,
        "encoder"      : encoder,
        "cat_cols"     : cat_cols,
    }

    logger.info(f"Output shape: {df.shape}")
    logger.info("PREPROCESSING TRAIN COMPLETE")
    logger.info("=" * 50)
    return df, artifacts


def preprocess_test(
    df: pd.DataFrame,
    artifacts: dict,
) -> pd.DataFrame:
    """
    Apply preprocessing to validation or test data.
    Uses fitted artifacts from training only — no leakage.
    """
    logger.info("=" * 50)
    logger.info("PREPROCESSING — TEST/VAL")
    logger.info(f"Input shape: {df.shape}")

    df, _ = drop_high_missing(
        df, drop_cols_fitted=artifacts["drop_cols"]
    )
    df    = drop_redundant(df)
    df    = fix_fe_numerical_dtypes(df)
    df, _ = add_nan_flags(
        df, nan_flag_cols_fitted=artifacts["nan_flag_cols"]
    )
    df    = encode_m_columns(df)
    df, _ = impute_d_columns(
        df, d_medians=artifacts["d_medians"]
    )
    df, _ = impute_numerical(
        df, fill_values=artifacts["num_fills"]
    )
    df, _ = impute_categorical(
        df, cat_fill_values=artifacts["cat_fills"]
    )
    df, _, _ = encode_categoricals(
        df, encoder=artifacts["encoder"]
    )

    logger.info(f"Output shape: {df.shape}")
    logger.info("PREPROCESSING TEST/VAL COMPLETE")
    logger.info("=" * 50)
    return df