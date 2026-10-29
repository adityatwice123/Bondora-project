"""Train the quick XGBoost, LightGBM and Random Forest pipelines the app ships with.

    python train_quick_model.py --data LoanData_Bondora.csv

For each model the script (1) fits on 80% of the loans and scores the other 20% for artifacts/metrics.json,
then (2) refits on ALL loans and saves the pipeline the app loads. The shipped models therefore have no
held-out score of their own; metrics.json describes the 80% versions. Settings are deliberately quick and
untuned. The tuned champion lives in train_champion.py.
"""
import argparse
import json
import os

import joblib
import lightgbm
import sklearn
import xgboost
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import loan_utils as lu
from training_common import load_modelling_data

SEED = 42


def build_preprocessor():
    numeric = Pipeline([("winsorizer", lu.Winsorizer()), ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return ColumnTransformer([("num", numeric, lu.NUMERIC_ALL), ("cat", OneHotEncoder(handle_unknown="ignore"), lu.CATEGORICAL)])


MODELS = {
    "XGBoost": XGBClassifier(random_state=SEED, n_estimators=100, max_depth=6, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(random_state=SEED, n_estimators=150, max_depth=10, verbose=-1),
    "Random Forest": RandomForestClassifier(random_state=SEED, n_estimators=100, max_depth=15, n_jobs=-1),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="LoanData_Bondora.csv")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()

    X, y, _ = load_modelling_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    os.makedirs(args.out, exist_ok=True)

    metrics = {}
    for name, model in MODELS.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clone(model))]).fit(x_train, y_train)
        probabilities = pipe.predict_proba(x_test)[:, 1]
        flagged = probabilities >= lu.THRESHOLD
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "precision_at_threshold": float(precision_score(y_test, flagged, zero_division=0)),
            "recall_at_threshold": float(recall_score(y_test, flagged, zero_division=0)),
        }
        print(f"{name}: held-out ROC-AUC {metrics[name]['roc_auc']:.3f}")
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"threshold": lu.THRESHOLD, "metrics": metrics,
                   "library_versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__}}, handle, indent=2)

    for name, model in MODELS.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", clone(model))]).fit(X, y)
        path = os.path.join(args.out, f"{name.replace(' ', '_').lower()}_pipeline.joblib")
        joblib.dump(pipe, path, compress=3)
        print(f"Saved {name} to {path}")


if __name__ == "__main__":
    main()
