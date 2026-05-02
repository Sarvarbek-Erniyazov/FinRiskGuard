import logging
import os
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR  = ROOT_DIR / "outputs" / "logs"

TASK_PREFIX_MAP = {
    "ieee_cis.loader"           : ("fraud", "01_loader"),
    "ieee_cis.splitter"         : ("fraud", "02_splitter"),
    "ieee_cis.feature_engineer" : ("fraud", "03_feature_engineer"),
    "ieee_cis.preprocessor"     : ("fraud", "04_preprocessor"),
    "ieee_cis.feature_selector" : ("fraud", "05_feature_selector"),
    "fraud_pipeline"            : ("fraud", "06_pipeline"),
    "fraud_detector"            : ("fraud", "07_detector"),

    "home_credit.loader"           : ("credit", "01_loader"),
    "home_credit.splitter"         : ("credit", "02_splitter"),
    "home_credit.feature_engineer" : ("credit", "03_feature_engineer"),
    "home_credit.preprocessor"     : ("credit", "04_preprocessor"),
    "home_credit.feature_selector" : ("credit", "05_feature_selector"),
    "credit_pipeline"              : ("credit", "06_pipeline"),
    "credit_scorer"                : ("credit", "07_scorer"),
    "shap_explainer"               : ("credit", "08_shap_explainer"),
}


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    date_str = datetime.now().strftime("%Y%m%d")

    if name in TASK_PREFIX_MAP:
        task, step   = TASK_PREFIX_MAP[name]
        task_log_dir = LOG_DIR / task
        task_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = task_log_dir / f"{step}_{date_str}.log"
    else:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"{name}_{date_str}.log"

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger