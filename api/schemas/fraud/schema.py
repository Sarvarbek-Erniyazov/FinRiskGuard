from pydantic import BaseModel, Field
from typing import Optional


class FraudTransactionRequest(BaseModel):
    TransactionID : int   = Field(..., example=2987004)
    TransactionDT : int   = Field(..., example=86400)
    TransactionAmt: float = Field(..., example=117.5)

    model_config = {"extra": "allow"}


class FraudBatchRequest(BaseModel):
    transactions: list[FraudTransactionRequest] = Field(..., min_length=1, max_length=10000)


class FraudPredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud         : bool
    threshold        : float
    model            : str
    decision         : str


class FraudBatchResponse(BaseModel):
    count            : int
    fraud_count      : int
    fraud_probability: list[float]
    is_fraud         : list[bool]
    threshold        : float
    model            : str


class FraudMetadataResponse(BaseModel):
    model_name     : str
    auc_roc        : float
    auc_pr         : float
    f1             : float
    precision      : float
    recall         : float
    threshold      : float
    features_used  : int
    class_imbalance: str
    cv_method      : str
    task           : str
    dataset        : str