import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("ieee_cis.splitter")

TIME_COL  = "TransactionDT"
VAL_RATIO = 0.20


def temporal_split(
    df: pd.DataFrame,
    val_ratio: float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logger.info("Starting temporal split...")
    logger.info(f"Input shape: {df.shape}")

    df_sorted = df.sort_values(TIME_COL).reset_index(drop=True)
    n         = len(df_sorted)
    val_start = int(n * (1 - val_ratio))

    train = df_sorted.iloc[:val_start].reset_index(drop=True)
    val   = df_sorted.iloc[val_start:].reset_index(drop=True)

    logger.info(f"Train shape : {train.shape} | Date range: {train[TIME_COL].min()} → {train[TIME_COL].max()}")
    logger.info(f"Val shape   : {val.shape}   | Date range: {val[TIME_COL].min()} → {val[TIME_COL].max()}")

    if "isFraud" in train.columns:
        logger.info(f"Train fraud rate: {train['isFraud'].mean()*100:.2f}%")
        logger.info(f"Val fraud rate  : {val['isFraud'].mean()*100:.2f}%")

    return train, val


def get_tscv(n_splits: int = 5) -> TimeSeriesSplit:
    logger.info(f"Creating TimeSeriesSplit with {n_splits} folds")
    return TimeSeriesSplit(n_splits=n_splits)