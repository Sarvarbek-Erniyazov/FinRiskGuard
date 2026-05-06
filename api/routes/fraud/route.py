import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas.fraud.schema import (
    FraudTransactionRequest,
    FraudBatchRequest,
    FraudPredictionResponse,
    FraudBatchResponse,
    FraudMetadataResponse,
)
from src.pipelines.predict_pipeline import get_fraud_predictor

router = APIRouter()


@router.post("/predict", response_model=FraudPredictionResponse)
def predict_fraud(request: FraudTransactionRequest):
    try:
        predictor = get_fraud_predictor()
        result    = predictor.predict_single(request.model_dump())
        return FraudPredictionResponse(
            **result,
            decision="FRAUD" if result["is_fraud"] else "LEGITIMATE",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=FraudBatchResponse)
def predict_fraud_batch(request: FraudBatchRequest):
    try:
        predictor = get_fraud_predictor()
        df        = pd.DataFrame([t.model_dump() for t in request.transactions])
        result    = predictor.predict(df)
        return FraudBatchResponse(
            count            = len(request.transactions),
            fraud_count      = sum(result["is_fraud"]),
            fraud_probability= result["fraud_probability"],
            is_fraud         = result["is_fraud"],
            threshold        = result["threshold"],
            model            = result["model"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata", response_model=FraudMetadataResponse)
def fraud_metadata():
    try:
        predictor = get_fraud_predictor()
        return FraudMetadataResponse(**predictor.get_metadata())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))