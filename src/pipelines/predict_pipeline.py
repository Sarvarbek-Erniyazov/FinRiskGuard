import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


class FraudPredictor:
    MODELS_DIR = ROOT_DIR / "outputs" / "models" / "fraud"

    def __init__(self):
        self.fe_artifacts   = joblib.load(self.MODELS_DIR / "fe_artifacts.pkl")
        self.prep_artifacts = joblib.load(self.MODELS_DIR / "prep_artifacts.pkl")
        self.fs_artifacts   = joblib.load(self.MODELS_DIR / "fs_artifacts.pkl")
        self.model          = joblib.load(self.MODELS_DIR / "fraud_model.pkl")

        with open(self.MODELS_DIR / "fraud_model_metadata.json") as f:
            self.metadata = json.load(f)

        self.threshold   = self.metadata["threshold"]
        self.model_name  = self.metadata["model_name"]
        self.final_feats = self.fs_artifacts["final_features"]

    def _apply_fe(self, df):
        from src.features.ieee_cis.feature_engineer import feature_engineer_test
        return feature_engineer_test(df, self.fe_artifacts)

    def _apply_prep(self, df):
        from src.data.ieee_cis.preprocessor import preprocess_test
        return preprocess_test(df, self.prep_artifacts)

    def _apply_fs(self, df):
        missing = [c for c in self.final_feats if c not in df.columns]
        for col in missing:
            df[col] = 0.0
        return df[self.final_feats]

    def _predict_proba(self, X):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        
        is_stacking = (
            isinstance(self.model, LogisticRegression) or
            isinstance(self.model, Pipeline)
        )
        
        if is_stacking:
            xgb = joblib.load(self.MODELS_DIR / "tuned_xgb.pkl")
            lgb = joblib.load(self.MODELS_DIR / "tuned_lgb.pkl")
            cat = joblib.load(self.MODELS_DIR / "tuned_cat.pkl")
            meta_input = np.column_stack([
                xgb.predict_proba(X)[:, 1],
                lgb.predict_proba(X)[:, 1],
                cat.predict_proba(X)[:, 1],
            ])
            return self.model.predict_proba(meta_input)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, raw_df):
        df    = raw_df.copy()
        df    = self._apply_fe(df)
        df    = self._apply_prep(df)
        # Barcha ustunlarni numeric qilib, xatolarni 0.0 bilan to'ldiramiz
        df    = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X     = self._apply_fs(df)
        
        proba = self._predict_proba(X)
        flags = (proba >= self.threshold).tolist()
        proba_list = proba.tolist()

        if len(proba_list) == 1:
            return {
                "fraud_probability": round(proba_list[0], 6),
                "is_fraud":          flags[0],
                "threshold":         self.threshold,
                "model":             self.model_name,
            }
        return {
            "fraud_probability": [round(p, 6) for p in proba_list],
            "is_fraud":          flags,
            "threshold":         self.threshold,
            "model":             self.model_name,
        }

    def predict_single(self, transaction):
        return self.predict(pd.DataFrame([transaction]))

    def get_metadata(self):
        return {
            "model_name"     : self.metadata["model_name"],
            "auc_roc"        : self.metadata["auc_roc"],
            "auc_pr"         : self.metadata["auc_pr"],
            "f1"             : self.metadata["f1"],
            "precision"      : self.metadata["precision"],
            "recall"         : self.metadata["recall"],
            "threshold"      : self.metadata["threshold"],
            "features_used"  : self.metadata["features_used"],
            "class_imbalance": self.metadata["class_imbalance"],
            "cv_method"      : self.metadata["cv_method"],
            "task"           : "fraud_detection",
            "dataset"        : "IEEE-CIS (Vesta Corp)",
        }


class CreditPredictor:
    MODELS_DIR = ROOT_DIR / "outputs" / "models" / "credit"

    def __init__(self):
        self.fe_artifacts   = joblib.load(self.MODELS_DIR / "fe_artifacts.pkl")
        self.prep_artifacts = joblib.load(self.MODELS_DIR / "prep_artifacts.pkl")
        self.fs_artifacts   = joblib.load(self.MODELS_DIR / "fs_artifacts.pkl")
        self.model          = joblib.load(self.MODELS_DIR / "credit_model.pkl")

        with open(self.MODELS_DIR / "credit_model_metadata.json") as f:
            self.metadata = json.load(f)

        self.threshold   = self.metadata["threshold"]
        self.model_name  = self.metadata["model_name"]
        self.final_feats = self.fs_artifacts["final_features"]

    def _apply_fe(self, df):
        from src.features.home_credit.feature_engineer import feature_engineer_test
        empty_tables = {
            "bureau"        : pd.DataFrame(),
            "bureau_balance": pd.DataFrame(),
            "previous_app"  : pd.DataFrame(),
            "pos_cash"      : pd.DataFrame(),
            "installments"  : pd.DataFrame(),
            "credit_card"   : pd.DataFrame(),
        }
        return feature_engineer_test(df, empty_tables, self.fe_artifacts)

    def _apply_prep(self, df):
        from src.data.home_credit.preprocessor import preprocess_test
        return preprocess_test(df, self.prep_artifacts)

    def _apply_fs(self, df):
        missing = [c for c in self.final_feats if c not in df.columns]
        for col in missing:
            df[col] = 0.0
        return df[self.final_feats]

    def _predict_proba(self, X):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        
        is_stacking = (
            isinstance(self.model, LogisticRegression) or
            isinstance(self.model, Pipeline)
        )
        
        if is_stacking:
            xgb = joblib.load(self.MODELS_DIR / "tuned_xgb.pkl")
            lgb = joblib.load(self.MODELS_DIR / "tuned_lgb.pkl")
            cat = joblib.load(self.MODELS_DIR / "tuned_cat.pkl")
            meta_input = np.column_stack([
                xgb.predict_proba(X)[:, 1],
                lgb.predict_proba(X)[:, 1],
                cat.predict_proba(X)[:, 1],
            ])
            return self.model.predict_proba(meta_input)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def predict(self, raw_df):
        df    = raw_df.copy()
        df    = self._apply_fe(df)
        df    = self._apply_prep(df)
        # Barcha ustunlarni numeric qilib, xatolarni 0.0 bilan to'ldiramiz
        df    = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X     = self._apply_fs(df)
        
        proba = self._predict_proba(X)
        flags = (proba >= self.threshold).tolist()
        proba_list = proba.tolist()

        if len(proba_list) == 1:
            return {
                "default_probability": round(proba_list[0], 6),
                "will_default":        flags[0],
                "threshold":           self.threshold,
                "model":               self.model_name,
            }
        return {
            "default_probability": [round(p, 6) for p in proba_list],
            "will_default":        flags,
            "threshold":           self.threshold,
            "model":               self.model_name,
        }

    def predict_single(self, applicant):
        return self.predict(pd.DataFrame([applicant]))

    def get_metadata(self):
        return {
            "model_name"     : self.metadata["model_name"],
            "auc_roc"        : self.metadata["auc_roc"],
            "auc_pr"         : self.metadata["auc_pr"],
            "f1"             : self.metadata["f1"],
            "precision"      : self.metadata["precision"],
            "recall"         : self.metadata["recall"],
            "threshold"      : self.metadata["threshold"],
            "features_used"  : self.metadata["features_used"],
            "class_imbalance": self.metadata["class_imbalance"],
            "cv_method"      : self.metadata["cv_method"],
            "task"           : "credit_default_scoring",
            "dataset"        : "Home Credit Default Risk",
        }


_fraud_predictor  = None
_credit_predictor = None


def get_fraud_predictor():
    global _fraud_predictor
    if _fraud_predictor is None:
        _fraud_predictor = FraudPredictor()
    return _fraud_predictor


def get_credit_predictor():
    global _credit_predictor
    if _credit_predictor is None:
        _credit_predictor = CreditPredictor()
    return _credit_predictor