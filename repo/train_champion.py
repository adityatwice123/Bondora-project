"""Rebuild the notebook's tuned XGBoost champion so the app can use the same model the report describes.

    python train_champion.py --data LoanData_Bondora.csv

Follows notebook sections 1.9 and 2.3 to 2.6: no scaling, rare categories pooled at 1%, a randomised search
(8 draws, 3 folds) on the 80% training split only, a cutoff chosen on out-of-fold training scores, and one
scoring pass on the 20% test split. The pipeline is saved as artifacts/xgboost_tuned_pipeline.joblib and the
app picks it up on the next start. Expect roughly 5 to 10 minutes on the full data.
"""
import argparse
import json
import os

import joblib
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

import loan_utils as lu
from training_common import load_modelling_data

SEED = 42
SPACE = {
    "classifier__n_estimators": randint(200, 700), "classifier__learning_rate": uniform(0.02, 0.08),
    "classifier__max_depth": randint(3, 9), "classifier__min_child_weight": randint(1, 25),
    "classifier__subsample": uniform(0.6, 0.4), "classifier__colsample_bytree": uniform(0.5, 0.5),
    "classifier__reg_lambda": loguniform(0.5, 10),
}


def build_pipeline():
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("clip", lu.Winsorizer(0.005, 0.995)), ("impute", SimpleImputer(strategy="median"))]), lu.NUMERIC_ALL),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False), lu.CATEGORICAL),
    ], verbose_feature_names_out=False)
    classifier = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                               tree_method="hist", n_jobs=-1, random_state=SEED)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def bootstrap_ci(y_true, score, n=1000):
    rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx])) > 1:
            values.append(roc_auc_score(y_true[idx], score[idx]))
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="LoanData_Bondora.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--n-iter", type=int, default=8, help="random search draws (the notebook used 8)")
    args = parser.parse_args()

    X, y, _ = load_modelling_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    os.makedirs(args.out, exist_ok=True)

    print(f"Random search with {args.n_iter} draws...")
    search = RandomizedSearchCV(build_pipeline(), SPACE, n_iter=args.n_iter, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                scoring="roc_auc", random_state=SEED, n_jobs=1, refit=False)
    search.fit(x_train, y_train)
    best = {key.replace("classifier__", ""): (value.item() if hasattr(value, "item") else value) for key, value in search.best_params_.items()}
    print(f"Best 3-fold ROC-AUC {search.best_score_:.4f} with {best}")
    pipe = build_pipeline().set_params(**search.best_params_)

    print("Choosing the cutoff on out-of-fold training scores...")
    oof = cross_val_predict(pipe, x_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    grid = np.linspace(0.20, 0.90, 71)
    threshold = float(grid[np.argmax([balanced_accuracy_score(y_train, oof >= t) for t in grid])])
    print(f"Cutoff {threshold:.2f} (the app uses loan_utils.THRESHOLD = {lu.THRESHOLD})")
    if abs(threshold - lu.THRESHOLD) > 1e-9:
        print("WARNING: the cutoff differs from loan_utils.THRESHOLD. Update that constant if you want the app to follow it.")

    pipe.fit(x_train, y_train)
    scores = pipe.predict_proba(x_test)[:, 1]
    flagged = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, flagged).ravel()
    decile = np.digitize(scores, np.quantile(scores, np.linspace(0.1, 0.9, 9)))
    metrics = {
        "model": "XGBoost (tuned)", "threshold": threshold, "best_params": best, "n_test": int(len(y_test)),
        "roc_auc": float(roc_auc_score(y_test, scores)), "roc_auc_ci": bootstrap_ci(y_test, scores),
        "pr_auc": float(average_precision_score(y_test, scores)), "balanced_accuracy": float(balanced_accuracy_score(y_test, flagged)),
        "precision": float(tp / (tp + fp)), "recall": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y_test, flagged)), "mcc": float(matthews_corrcoef(y_test, flagged)), "brier": float(brier_score_loss(y_test, scores)),
        "decile_default_rate": [float(y_test.values[decile == d].mean()) for d in range(10)],
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "decile_default_rate"}, indent=2))
    joblib.dump(pipe, os.path.join(args.out, "xgboost_tuned_pipeline.joblib"), compress=3)
    with open(os.path.join(args.out, "champion_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print("Saved xgboost_tuned_pipeline.joblib and champion_metrics.json. Restart the app to use the tuned champion.")


if __name__ == "__main__":
    main()
