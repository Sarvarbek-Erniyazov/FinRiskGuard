import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("ieee_cis.loader")

ROOT_DIR = Path(__file__).resolve().parents[3]
RAW_DIR  = ROOT_DIR / "data" / "raw" / "ieee_cis"


def load_transaction(split: str = "train") -> pd.DataFrame:
    path = RAW_DIR / f"{split}_transaction.csv"
    logger.info(f"Loading {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"  {split}_transaction shape: {df.shape}")
    return df


def load_identity(split: str = "train") -> pd.DataFrame:
    path = RAW_DIR / f"{split}_identity.csv"
    logger.info(f"Loading {path.name}...")
    df = pd.read_csv(path)
    logger.info(f"  {split}_identity shape: {df.shape}")
    return df


def load_and_merge(split: str = "train") -> pd.DataFrame:
    logger.info("=" * 50)
    logger.info(f"LOADING & MERGING: {split}")

    txn = load_transaction(split)
    idn = load_identity(split)

    df = txn.merge(idn, on="TransactionID", how="left")

    logger.info(f"  Merged shape : {df.shape}")
    logger.info(f"  Memory usage : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    if "isFraud" in df.columns:
        logger.info(f"  Fraud rate   : {df['isFraud'].mean()*100:.2f}%")
    else:
        logger.info(f"  isFraud column not present (test set — expected)")

    logger.info("LOADING COMPLETE")
    logger.info("=" * 50)

    return df