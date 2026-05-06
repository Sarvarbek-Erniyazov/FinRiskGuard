import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas.credit.schema import (
    CreditApplicantRequest,
    CreditBatchRequest,
    CreditPredictionResponse,
    CreditBatchResponse,
    CreditMetadataResponse,
)
from src.pipelines.predict_pipeline import get_credit_predictor

router = APIRouter()


@router.post("/predict", response_model=CreditPredictionResponse)
def predict_credit(request: CreditApplicantRequest):
    try:
        predictor = get_credit_predictor()
        result    = predictor.predict_single(request.model_dump())
        return CreditPredictionResponse(
            **result,
            risk_label="HIGH_RISK" if result["will_default"] else "LOW_RISK",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=CreditBatchResponse)
def predict_credit_batch(request: CreditBatchRequest):
    try:
        predictor = get_credit_predictor()
        df        = pd.DataFrame([a.model_dump() for a in request.applicants])
        result    = predictor.predict(df)
        return CreditBatchResponse(
            count              = len(request.applicants),
            high_risk_count    = sum(result["will_default"]),
            default_probability= result["default_probability"],
            will_default       = result["will_default"],
            threshold          = result["threshold"],
            model              = result["model"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata", response_model=CreditMetadataResponse)
def credit_metadata():
    try:
        predictor = get_credit_predictor()
        return CreditMetadataResponse(**predictor.get_metadata())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))