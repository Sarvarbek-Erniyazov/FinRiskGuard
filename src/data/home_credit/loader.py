import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("home_credit.loader")

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR  = ROOT_DIR / "data" / "raw" / "home_credit"


def load_application(split: str = "train") -> pd.DataFrame:
    """Load application train or test CSV."""
    path = RAW_DIR / f"application_{split}.csv"
    logger.info(f"Loading {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"  Shape        : {df.shape}")
    logger.info(f"  Memory       : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    if "TARGET" in df.columns:
        logger.info(f"  Default rate : {df['TARGET'].mean()*100:.2f}%")
        logger.info(f"  Imbalance    : {(1 - df['TARGET'].mean()) / df['TARGET'].mean():.1f}:1")
    else:
        logger.info("  TARGET not present (test set — expected)")
    return df


def load_bureau() -> pd.DataFrame:
    """Load bureau.csv — external credit history."""
    path = RAW_DIR / "bureau.csv"
    logger.info(f"Loading {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"  bureau shape : {df.shape}")
    return df


def load_bureau_balance() -> pd.DataFrame:
    """Load bureau_balance.csv — monthly bureau status."""
    path = RAW_DIR / "bureau_balance.csv"
    logger.info(f"Loading {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"  bureau_balance shape : {df.shape}")
    return df


def load_previous_application() -> pd.DataFrame:
    """Load previous_application.csv — prior loan applications."""
    path = RAW_DIR / "previous_application.csv"
    logger.info(f"Loading {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"  previous_application shape : {df.shape}")
    return df


def load_pos_cash() -> pd.DataFrame:
    """Load POS_CASH_balance.csv — POS and cash loan monthly snapshots."""
    path = RAW_DIR / "POS_CASH_balance.csv"
    logger.info(f"Loading {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"  POS_CASH_balance shape : {df.shape}")
    return df


def load_credit_card() -> pd.DataFrame:
    """Load credit_card_balance.csv — credit card monthly snapshots."""
    path = RAW_DIR / "credit_card_balance.csv"
    logger.info(f"Loading {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"  credit_card_balance shape : {df.shape}")
    return df


def load_installments() -> pd.DataFrame:
    """Load installments_payments.csv — repayment history."""
    path = RAW_DIR / "installments_payments.csv"
    logger.info(f"Loading {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"  installments_payments shape : {df.shape}")
    return df


def load_supplementary() -> dict:
    """Load all supplementary tables (bureau through installments). 
    Application is loaded separately via load_application()."""
    logger.info("=" * 50)
    logger.info("LOADING SUPPLEMENTARY TABLES")

    tables = {
        "bureau":         load_bureau(),
        "bureau_balance": load_bureau_balance(),
        "previous_app":   load_previous_application(),
        "pos_cash":       load_pos_cash(),
        "credit_card":    load_credit_card(),
        "installments":   load_installments(),
    }

    logger.info("ALL SUPPLEMENTARY TABLES LOADED")
    logger.info("=" * 50)
    return tables