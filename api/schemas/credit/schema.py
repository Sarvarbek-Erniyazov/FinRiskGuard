from pydantic import BaseModel, Field
from typing import Optional


class CreditApplicantRequest(BaseModel):
    SK_ID_CURR        : int   = Field(..., example=100002)
    AMT_CREDIT        : float = Field(..., example=406597.5)
    AMT_INCOME_TOTAL  : float = Field(..., example=202500.0)
    AMT_ANNUITY       : float = Field(..., example=24700.5)
    DAYS_BIRTH        : int   = Field(..., example=-9461)
    DAYS_EMPLOYED     : int   = Field(..., example=-637)
    EXT_SOURCE_1      : Optional[float] = Field(None, example=0.502)
    EXT_SOURCE_2      : Optional[float] = Field(None, example=0.651)
    EXT_SOURCE_3      : Optional[float] = Field(None, example=0.493)

    model_config = {"extra": "allow"}


class CreditBatchRequest(BaseModel):
    applicants: list[CreditApplicantRequest] = Field(..., min_length=1, max_length=10000)


class CreditPredictionResponse(BaseModel):
    default_probability: float
    will_default       : bool
    threshold          : float
    model              : str
    risk_label         : str


class CreditBatchResponse(BaseModel):
    count              : int
    high_risk_count    : int
    default_probability: list[float]
    will_default       : list[bool]
    threshold          : float
    model              : str


class CreditMetadataResponse(BaseModel):
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