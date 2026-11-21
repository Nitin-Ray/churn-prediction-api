"""Train, tune, evaluate 3 churn models -> save best pipeline + business threshold."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from features import CATEGORICAL, NUMERIC, engineer

SEED = 42
COST_FN = 200  
COST_FP = 50     

df = pd.read_csv("data/telco_churn.csv").drop(columns="customerID")
y = (df.pop("Churn") == "Yes").astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])

def make_pipe(clf):
    return Pipeline([("fe", FunctionTransformer(engineer)), ("prep", prep), ("clf", clf)])

candidates = {
    "LogisticRegression": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                           {"clf__C": [0.01, 0.1, 1, 10]}),
    "RandomForest": (RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
                     {"clf__n_estimators": [200, 400], "clf__max_depth": [6, 10, None],
                      "clf__min_samples_leaf": [3, 10]}),
    "GradientBoosting": (GradientBoostingClassifier(random_state=SEED),
                         {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                          "clf__max_depth": [2, 3]}),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

def best_threshold(y_true, proba):
    """Threshold that minimises expected business cost."""
    ts = np.linspace(0.05, 0.95, 91)
    costs = [COST_FN * ((proba < t) & (y_true == 1)).sum() + COST_FP * ((proba >= t) & (y_true == 0)).sum() for t in ts]
    return float(ts[int(np.argmin(costs))])

results, fitted = [], {}
for name, (clf, grid) in candidates.items():
    gs = GridSearchCV(make_pipe(clf), grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
    # threshold chosen on out-of-fold TRAIN predictions -> no test leakage
    oof = cross_val_predict(gs.best_estimator_, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = best_threshold(y_tr.values, oof)
    p = gs.predict_proba(X_te)[:, 1]
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    results.append({"model": name, "cv_auc": round(float(gs.best_score_), 4),
                    "test_auc": round(roc_auc_score(y_te, p), 4),
                    "test_pr_auc": round(average_precision_score(y_te, p), 4),
                    "threshold": round(thr, 2), "precision": round(float(tp / (tp + fp)), 3),
                    "recall": round(float(tp / (tp + fn)), 3),
                    "business_cost": int(COST_FN * fn + COST_FP * fp),
                    "best_params": gs.best_params_})
    fitted[name] = (gs.best_estimator_, thr)
    print(f"{name}: {results[-1]}")

res = pd.DataFrame(results).sort_values("business_cost")
best = res.iloc[0]["model"]
print("\n", res.drop(columns="best_params").to_string(index=False))
print(f"\nBEST (lowest business cost): {best}")


pipe, thr = fitted[best]
pipe.fit(df, y)
joblib.dump({"pipeline": pipe, "threshold": thr, "model_name": best}, "model.joblib")
res.to_csv("model_comparison.csv", index=False)
json.dump({"best_model": best, "threshold": thr}, open("model_meta.json", "w"))
