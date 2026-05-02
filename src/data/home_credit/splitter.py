import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.logger import get_logger

logger = get_logger("home_credit.splitter")

VAL_RATIO  = 0.20
TARGET_COL = "TARGET"


def stratified_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    val_ratio: float = VAL_RATIO,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logger.info("Starting stratified split...")
    logger.info(f"Input shape  : {df.shape}")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=random_state,
    )

    for train_idx, val_idx in sss.split(df, df[target_col]):
        train = df.iloc[train_idx].reset_index(drop=True)
        val   = df.iloc[val_idx].reset_index(drop=True)

    logger.info(f"Train shape       : {train.shape}")
    logger.info(f"Val shape         : {val.shape}")
    logger.info(f"Train default rate: {train[target_col].mean()*100:.2f}%")
    logger.info(f"Val default rate  : {val[target_col].mean()*100:.2f}%")

    return train, val