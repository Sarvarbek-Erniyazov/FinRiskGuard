import argparse
import joblib
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.logger import get_logger
from src.data.home_credit.loader import load_application, load_supplementary
from src.data.home_credit.preprocessor import preprocess_train, preprocess_test
from src.data.home_credit.splitter import stratified_split
from src.features.home_credit.feature_engineer import (
    feature_engineer_train, feature_engineer_test
)
from src.features.home_credit.feature_selector import select_features

logger = get_logger("credit_pipeline")

ROOT_DIR     = Path(__file__).resolve().parents[2]
MODELS_DIR   = ROOT_DIR / "outputs" / "models" / "credit"
REPORTS_DIR  = ROOT_DIR / "outputs" / "reports" / "credit"
FEATURES_DIR = ROOT_DIR / "data" / "features" / "credit"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Stage 1: Load ─────────────────────────────────────────────────────────────

def run_load():
    """Load application train/test and all supplementary tables."""
    logger.info("STAGE 1: Loading Home Credit data")

    app_train = load_application("train")
    app_test  = load_application("test")

    logger.info("Loading supplementary tables...")
    tables = load_supplementary()

    return app_train, app_test, tables


# ── Stage 2: Split ────────────────────────────────────────────────────────────

def run_split(app_train: pd.DataFrame):
    logger.info("STAGE 2: Stratified split")
    train, val = stratified_split(app_train, target_col="TARGET")
    return train, val


# ── Stage 3: Preprocess ───────────────────────────────────────────────────────

def run_preprocess(train: pd.DataFrame,
                   val: pd.DataFrame,
                   app_test: pd.DataFrame):
    logger.info("STAGE 3: Preprocessing")

    train, prep_artifacts = preprocess_train(train)
    val   = preprocess_test(val,      prep_artifacts)
    test  = preprocess_test(app_test, prep_artifacts)

    joblib.dump(prep_artifacts, MODELS_DIR / "prep_artifacts.pkl")
    logger.info("  Saved: prep_artifacts.pkl")

    logger.info(f"  Train shape : {train.shape}")
    logger.info(f"  Val shape   : {val.shape}")
    logger.info(f"  Test shape  : {test.shape}")

    return train, val, test, prep_artifacts


# ── Stage 4: Feature Engineering ─────────────────────────────────────────────

def run_feature_engineering(train: pd.DataFrame,
                             val: pd.DataFrame,
                             test: pd.DataFrame,
                             tables: dict):
    logger.info("STAGE 4: Feature Engineering")

    train, fe_artifacts = feature_engineer_train(train, tables)
    val   = feature_engineer_test(val,  tables, fe_artifacts)
    test  = feature_engineer_test(test, tables, fe_artifacts)

    joblib.dump(fe_artifacts, MODELS_DIR / "fe_artifacts.pkl")
    logger.info("  Saved: fe_artifacts.pkl")

    logger.info(f"  Train shape : {train.shape}")
    logger.info(f"  Val shape   : {val.shape}")
    logger.info(f"  Test shape  : {test.shape}")

    return train, val, test, fe_artifacts


# ── Stage 5: Feature Selection ────────────────────────────────────────────────

def run_feature_selection(train: pd.DataFrame,
                           val: pd.DataFrame,
                           test: pd.DataFrame):
    logger.info("STAGE 5: Credit Feature Selection")

    final_features, report, fs_artifacts = select_features(
        df_train=train,
        target_col="TARGET",
        corr_threshold=0.95,
        top_k=70,
        mode="union",
    )

    report.to_csv(REPORTS_DIR / "feature_selection_report.csv")
    joblib.dump(fs_artifacts, MODELS_DIR / "fs_artifacts.pkl")
    logger.info("  Saved: feature_selection_report.csv, fs_artifacts.pkl")

    credit_cols      = final_features + ["TARGET", "SK_ID_CURR"]
    test_credit_cols = [c for c in final_features if c in test.columns]
    test_credit_cols = test_credit_cols + ["SK_ID_CURR"]

    train_credit = train[[c for c in credit_cols if c in train.columns]]
    val_credit   = val[[c for c in credit_cols if c in val.columns]]
    test_credit  = test[[c for c in test_credit_cols if c in test.columns]]

    train_credit.to_parquet(FEATURES_DIR / "train_credit_features.parquet")
    val_credit.to_parquet(FEATURES_DIR   / "val_credit_features.parquet")
    test_credit.to_parquet(FEATURES_DIR  / "test_credit_features.parquet")
    logger.info("  Saved: train/val/test_credit_features.parquet")

    logger.info(f"  Train : {train_credit.shape}")
    logger.info(f"  Val   : {val_credit.shape}")
    logger.info(f"  Test  : {test_credit.shape}")

    return train_credit, val_credit, test_credit, final_features


# ── Main ─────────────────────────────────────────────────────────────────────

def main(stage: str = "all"):
    logger.info("=" * 60)
    logger.info("FINRISKGUARD — CREDIT PIPELINE (Home Credit)")
    logger.info(f"Stage: {stage}")
    logger.info("=" * 60)

    app_train, app_test, tables             = None, None, None
    train, val, test                         = None, None, None
    train_credit, val_credit, test_credit    = None, None, None

    if stage in ("load", "all"):
        app_train, app_test, tables = run_load()

    if stage in ("split", "all"):
        train, val = run_split(app_train)

    if stage in ("preprocess", "all"):
        train, val, test, prep_artifacts = run_preprocess(
            train, val, app_test
        )

    if stage in ("fe", "all"):
        train, val, test, fe_artifacts = run_feature_engineering(
            train, val, test, tables
        )

    if stage in ("fs", "all"):
        train_credit, val_credit, test_credit, features = \
            run_feature_selection(train, val, test)

    logger.info("=" * 60)
    logger.info("CREDIT PIPELINE COMPLETE")
    logger.info(f"Train : {train_credit.shape if train_credit is not None else 'N/A'}")
    logger.info(f"Val   : {val_credit.shape if val_credit is not None else 'N/A'}")
    logger.info(f"Test  : {test_credit.shape if test_credit is not None else 'N/A'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FinRiskGuard Credit Pipeline (Home Credit)"
    )
    parser.add_argument(
        "--stage",
        choices=["load", "split", "preprocess", "fe", "fs", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    args = parser.parse_args()
    main(stage=args.stage)