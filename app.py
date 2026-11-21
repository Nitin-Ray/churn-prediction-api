"""Churn prediction REST API.  Run: uvicorn app:app --reload   Docs: /docs"""
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import engineer  
from fastapi.responses import RedirectResponse

artifact = joblib.load("model.joblib")
PIPE, THRESHOLD = artifact["pipeline"], artifact["threshold"]

app = FastAPI(title="Customer Churn Prediction API", version="1.0",
              description="Send customer details, get churn probability + binary prediction.")

YN = Literal["Yes", "No"]
YN_NPS = Literal["Yes", "No", "No phone service"]
YN_NIS = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YN
    Dependents: YN
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YN
    MultipleLines: YN_NPS
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YN_NIS
    OnlineBackup: YN_NIS
    DeviceProtection: YN_NIS
    TechSupport: YN_NIS
    StreamingTV: YN_NIS
    StreamingMovies: YN_NIS
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YN
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"example": {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.7, "TotalCharges": 190.4}}}


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: Literal["low", "medium", "high"]
    threshold_used: float



@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

@app.get("/health")
def health():
    return {"status": "ok", "model": artifact["model_name"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    proba = float(PIPE.predict_proba(pd.DataFrame([customer.model_dump()]))[0, 1])
    risk = "high" if proba >= 0.6 else "medium" if proba >= THRESHOLD else "low"
    return Prediction(churn_probability=round(proba, 4), churn_prediction=int(proba >= THRESHOLD),
                      risk_level=risk, threshold_used=THRESHOLD)
