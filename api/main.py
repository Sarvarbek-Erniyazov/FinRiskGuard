from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes.fraud.route import router as fraud_router
from api.routes.credit.route import router as credit_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.pipelines.predict_pipeline import get_fraud_predictor, get_credit_predictor
    print("Loading Fraud model...")
    get_fraud_predictor()
    print("Loading Credit model...")
    get_credit_predictor()
    print("Both models loaded.")
    yield
    print("Shutting down.")


app = FastAPI(
    title       = "FinRiskGuard API",
    description = "Fraud Detection and Credit Default Scoring",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(fraud_router,  prefix="/api/v1/fraud",  tags=["Fraud Detection"])
app.include_router(credit_router, prefix="/api/v1/credit", tags=["Credit Scoring"])


@app.get("/", tags=["Root"])
def root():
    return {
        "service" : "FinRiskGuard",
        "version" : "1.0.0",
        "tasks"   : ["fraud_detection", "credit_default_scoring"],
        "docs"    : "/docs",
        "health"  : "/health",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status" : "ok",
        "service": "FinRiskGuard API",
    }