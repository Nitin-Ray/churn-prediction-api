"""Shared feature engineering (used by training AND the API so they never drift)."""
import numpy as np
import pandas as pd

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
           "charge_ratio", "num_services"]
CATEGORICAL = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
               "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
               "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group", "has_family"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean + add 5 features: tenure_group, avg_monthly_spend, charge_ratio,
    num_services, has_family."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
   
    df.loc[(df["tenure"] == 0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    df["tenure_group"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                                labels=["0-12m", "13-24m", "25-48m", "49-72m"]).astype(str)
    df["avg_monthly_spend"] = (df["TotalCharges"] / df["tenure"].replace(0, np.nan)) \
        .fillna(df["MonthlyCharges"])
    df["charge_ratio"] = df["MonthlyCharges"] / df["avg_monthly_spend"].replace(0, np.nan)
    df["num_services"] = df[SERVICE_COLS].eq("Yes").sum(axis=1)
    df["has_family"] = np.where((df["Partner"] == "Yes") | (df["Dependents"] == "Yes"), "Yes", "No")
    return df[NUMERIC + CATEGORICAL]
