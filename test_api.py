from fastapi.testclient import TestClient
from app import app

c = TestClient(app)
risky = app.openapi()["components"]["schemas"]["Customer"]["example"]
loyal = {**risky, "tenure": 60, "Contract": "Two year", "InternetService": "DSL",
         "MonthlyCharges": 50.0, "TotalCharges": 3000.0, "PaymentMethod": "Credit card (automatic)"}

def test_health():
    assert c.get("/health").status_code == 200

def test_risky_customer_flagged():
    r = c.post("/predict", json=risky).json()
    assert r["churn_prediction"] == 1 and r["churn_probability"] > 0.5

def test_loyal_customer_not_flagged():
    assert c.post("/predict", json=loyal).json()["churn_prediction"] == 0

def test_validation_error():
    assert c.post("/predict", json={**risky, "Contract": "Weekly"}).status_code == 422
