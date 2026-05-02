import argparse
import joblib
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.logger import get_logger
from src.data.ieee_cis.loader import load_and_merge
from src.data.ieee_cis.splitter import temporal_split
from src.data.ieee_cis.preprocessor import preprocess_train, preprocess_test
from src.features.ieee_cis.feature_engineer import (
    feature_engineer_train,
    feature_engineer_test,
)
from src.features.ieee_cis.feature_selector import select_features

logger = get_logger("fraud_pipeline")

ROOT_DIR     = Path(__file__).resolve().parents[2]
MODELS_DIR   = ROOT_DIR / "outputs" / "models" / "fraud"
REPORTS_DIR  = ROOT_DIR / "outputs" / "reports" / "fraud"
FEATURES_DIR = ROOT_DIR / "data" / "features" / "fraud"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def run_load() -> tuple:
    """Load and merge transaction + identity tables for train and test."""
    logger.info("=" * 60)
    logger.info("STAGE 1: Loading IEEE-CIS data")
    logger.info("=" * 60)

    df_train = load_and_merge("train")
    df_test  = load_and_merge("test")

    logger.info(f"Train shape      : {df_train.shape}")
    logger.info(f"Test shape       : {df_test.shape}")
    logger.info(f"Train fraud rate : {df_train['isFraud'].mean()*100:.2f}%")
    return df_train, df_test


def run_split(df_train: pd.DataFrame) -> tuple:
    """Temporal train/val split — last 20% by time goes to val."""
    logger.info("=" * 60)
    logger.info("STAGE 2: Temporal split (val_ratio=0.20)")
    logger.info("=" * 60)

    train, val = temporal_split(df_train)

    logger.info(
        f"Train : {train.shape} | "
        f"Fraud rate: {train['isFraud'].mean()*100:.2f}%"
    )
    logger.info(
        f"Val   : {val.shape}   | "
        f"Fraud rate: {val['isFraud'].mean()*100:.2f}%"
    )
    return train, val


def run_feature_engineering(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple:
    """
    Feature engineering — must run BEFORE preprocessing.
    FE uses raw string columns (id_31, DeviceType, emaildomain)
    that OrdinalEncoder would destroy if preprocessing ran first.
    All maps fitted on train only — val/test use train artifacts.
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Feature Engineering")
    logger.info("  Order  : FE FIRST → Preprocessing SECOND")
    logger.info("  Reason : FE needs raw string columns")
    logger.info("=" * 60)

    train, fe_artifacts = feature_engineer_train(train)
    val                 = feature_engineer_test(val,  fe_artifacts)
    test                = feature_engineer_test(test, fe_artifacts)

    joblib.dump(fe_artifacts, MODELS_DIR / "fe_artifacts.pkl")

    fe_cols = [c for c in train.columns if c.startswith("FE_")]
    d_norm  = [c for c in fe_cols if "normalized" in c]

    logger.info("Saved: fe_artifacts.pkl")
    logger.info(f"Train : {train.shape}")
    logger.info(f"Val   : {val.shape}")
    logger.info(f"Test  : {test.shape}")
    logger.info(f"FE features total      : {len(fe_cols)}")
    logger.info(f"  D_normalized features: {len(d_norm)}")
    logger.info(f"FE columns: {fe_cols}")
    return train, val, test, fe_artifacts


def run_preprocess(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple:
    """
    Preprocessing — runs after feature engineering.
    Steps: drop high-missing → drop redundant → NaN flags →
           encode M cols → impute D cols → impute numerical →
           impute categorical → OrdinalEncoder.
    All imputers/encoders fitted on train only.
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: Preprocessing")
    logger.info("  Order: After Feature Engineering")
    logger.info("=" * 60)

    train, prep_artifacts = preprocess_train(train)
    val                   = preprocess_test(val,  prep_artifacts)
    test                  = preprocess_test(test, prep_artifacts)

    joblib.dump(prep_artifacts, MODELS_DIR / "prep_artifacts.pkl")
    logger.info("Saved: prep_artifacts.pkl")
    logger.info(f"Train : {train.shape}")
    logger.info(f"Val   : {val.shape}")
    logger.info(f"Test  : {test.shape}")
    return train, val, test, prep_artifacts


def run_feature_selection(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple:
    """
    Feature selection — MI + XGB rank-based, top_k=200.

    top_k raised from 150 → 200:
      - 14 new D_normalized features need space
      - Extended UID and card1_addr1 aggregations added
      - Competitive solutions use 250-300+ features
      - rank mode picks best 200 by avg MI+XGB rank

    Selection fitted on train only — no val/test leakage.
    """
    logger.info("=" * 60)
    logger.info("STAGE 5: Feature Selection")
    logger.info("  Method : MI + XGB rank-based (mode='rank')")
    logger.info("  Top_k  : 200 (raised from 150 — D_normalized + extended FE)")
    logger.info("=" * 60)

    final_features, report, fs_artifacts = select_features(
        df_train       = train,
        target_col     = "isFraud",
        corr_threshold = 0.95,
        top_k          = 200,
        mode           = "rank",
    )

    report.to_csv(REPORTS_DIR / "feature_selection_report.csv")
    joblib.dump(fs_artifacts, MODELS_DIR / "fs_artifacts.pkl")
    logger.info("Saved: feature_selection_report.csv, fs_artifacts.pkl")

    fraud_cols      = final_features + ["isFraud"]
    test_cols       = [c for c in final_features if c in test.columns]
    missing_in_test = [c for c in final_features if c not in test.columns]

    if missing_in_test:
        logger.info(
            f"WARNING: {len(missing_in_test)} selected features "
            f"missing in test set:"
        )
        logger.info(f"  {missing_in_test}")

    train_out = train[[c for c in fraud_cols if c in train.columns]]
    val_out   = val[[c for c in fraud_cols   if c in val.columns]]
    test_out  = test[[c for c in test_cols   if c in test.columns]]

    train_out.to_parquet(FEATURES_DIR / "train_fraud_features.parquet")
    val_out.to_parquet(FEATURES_DIR   / "val_fraud_features.parquet")
    test_out.to_parquet(FEATURES_DIR  / "test_fraud_features.parquet")
    logger.info("Saved: train/val/test_fraud_features.parquet")

    logger.info(f"Train : {train_out.shape}")
    logger.info(f"Val   : {val_out.shape}")
    logger.info(f"Test  : {test_out.shape}")

    fe_count   = sum(1 for c in final_features if c.startswith("FE_"))
    nan_count  = sum(1 for c in final_features if c.endswith("_isnan"))
    d_norm_cnt = sum(1 for c in final_features if "normalized" in c)
    raw_count  = len(final_features) - fe_count - nan_count

    logger.info("Selected feature breakdown:")
    logger.info(f"  FE_ engineered   : {fe_count}")
    logger.info(f"  └─ D_normalized  : {d_norm_cnt}")
    logger.info(f"  _isnan flags     : {nan_count}")
    logger.info(f"  Raw features     : {raw_count}")
    logger.info(f"  TOTAL            : {len(final_features)}")

    return train_out, val_out, test_out, final_features


def main(stage: str = "all") -> None:
    """
    Full fraud detection data pipeline.

    Stages:
      load       → load raw CSV files
      split      → temporal train/val split
      fe         → feature engineering (must be first)
      preprocess → preprocessing (after FE)
      fs         → feature selection
      all        → run all stages

    Pipeline order: load → split → FE → preprocess → fs
    """
    logger.info("=" * 60)
    logger.info("FINRISKGUARD — FRAUD PIPELINE (IEEE-CIS)")
    logger.info(f"Stage: {stage}")
    logger.info("=" * 60)

    df_train  = df_test  = None
    train     = val      = test = None
    train_out = val_out  = test_out = None
    final_features = []

    if stage in ("load", "all"):
        df_train, df_test = run_load()

    if stage in ("split", "all"):
        train, val = run_split(df_train)

    if stage in ("fe", "all"):
        train, val, test, fe_artifacts = run_feature_engineering(
            train, val, df_test
        )

    if stage in ("preprocess", "all"):
        train, val, test, prep_artifacts = run_preprocess(
            train, val, test
        )

    if stage in ("fs", "all"):
        train_out, val_out, test_out, final_features = (
            run_feature_selection(train, val, test)
        )

    logger.info("=" * 60)
    logger.info("FRAUD PIPELINE COMPLETE")
    if train_out is not None:
        logger.info(f"Train    : {train_out.shape}")
        logger.info(f"Val      : {val_out.shape}")
        logger.info(f"Test     : {test_out.shape}")
        logger.info(f"Features : {len(final_features)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FinRiskGuard — Fraud Detection Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["load", "split", "fe", "preprocess", "fs", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()
    main(stage=args.stage)