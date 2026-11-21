# Customer Churn Prediction Pipeline

End-to-end: clean data → feature engineering → 3 tuned models → business-cost model selection → FastAPI service.

## Run
```bash
pip install -r requirements.txt
python train.py                 # trains, tunes, saves model.joblib + model_comparison.csv
uvicorn app:app --reload        # API at http://127.0.0.1:8000  (Swagger UI: /docs)
pytest -q                       # API tests
```
Dataset: IBM Telco Customer Churn (7,043 rows) in `data/telco_churn.csv`.

## Pipeline
| Step | Detail |
|---|---|
| Cleaning | `TotalCharges` blanks (new customers, tenure=0) → 0; median imputer as safety net |
| Encoding | One-hot (`handle_unknown="ignore"`) on all categoricals |
| New features | `tenure_group`, `avg_monthly_spend`, `charge_ratio`, `num_services`, `has_family` |
| Scaling | `StandardScaler` on numeric columns |
| Models | Logistic Regression, Random Forest, Gradient Boosting |
| Tuning | `GridSearchCV`, 5-fold stratified CV, scoring = ROC-AUC |
| Evaluation | ROC-AUC, PR-AUC, precision, recall on 20% stratified hold-out |

Everything (feature engineering + preprocessing + model) lives in one sklearn `Pipeline`, so the API applies the exact training transformations — no train/serve skew.

## Model selection (business impact)
Accuracy is misleading (~73% of customers don't churn). Each model's decision threshold is chosen to minimise
`cost = 200 × missed churners (FN) + 50 × wasted retention offers (FP)` using **out-of-fold training predictions** (no test leakage). The model with the lowest hold-out cost is deployed.
Cost values are assumptions — edit `COST_FN` / `COST_FP` in `train.py`.

| Model | CV AUC | Test AUC | PR-AUC | Threshold | Precision | Recall | Cost |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.846 | 0.842 | 0.633 | 0.37 | 0.457 | 0.904 | **27,250** |
| Random Forest | 0.848 | 0.844 | 0.655 | 0.39 | 0.469 | 0.864 | 28,500 |
| Gradient Boosting | 0.849 | 0.844 | 0.662 | 0.20 | 0.475 | 0.837 | 29,500 |

AUCs are nearly identical, so the simpler, interpretable, fastest model wins on business cost.

## API
`POST /predict` — body = customer fields (see `/docs` for schema + example)
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
 "gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":2,
 "PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No",
 "OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes",
 "StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes",
 "PaymentMethod":"Electronic check","MonthlyCharges":95.7,"TotalCharges":190.4}'
```
Response: `{"churn_probability": 0.78, "churn_prediction": 1, "risk_level": "high", "threshold_used": 0.37}`

`GET /health` — liveness + deployed model name. Invalid input (e.g. unknown contract) → `422`.

## Files
`features.py` shared feature engineering · `train.py` training/tuning/selection · `app.py` FastAPI · `test_api.py` tests
