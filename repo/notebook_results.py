"""Results copied from the research notebook (Bondora_Risk_NeSy_XAI.ipynb).

Everything here is a recorded notebook output, not something the app recomputes.
The notebook's champion is a tuned XGBoost scored on a held-out test set of
22,069 loans. The app shows these numbers on the "Models" and "How it works"
tabs and always labels them as notebook results.
"""

TEST_SET_SIZE = 22_069
POPULATION = {"all_loans": 179_235, "final_outcome": 113_560, "modelling": 110_342, "default_rate": 0.641}

# 5-fold cross-validated scores on the training set (notebook sections 2.2 and 2.3)
CV_COLUMNS = ["ROC-AUC", "ROC-AUC std", "PR-AUC", "F1@0.5", "Bal.Acc@0.5", "MCC@0.5", "Brier"]
CV_TABLE = {
    "XGBoost (tuned)":        [0.7718, 0.0012, 0.8378, 0.8135, 0.6789, 0.4030, 0.1781],
    "LightGBM (tuned)":       [0.7711, 0.0017, 0.8363, 0.8123, 0.6838, 0.4066, 0.1783],
    "XGBoost":                [0.7701, 0.0016, 0.8360, 0.8131, 0.6766, 0.3997, 0.1787],
    "LightGBM":               [0.7692, 0.0013, 0.8351, 0.8137, 0.6777, 0.4019, 0.1791],
    "Random Forest":          [0.7666, 0.0011, 0.8300, 0.8143, 0.6720, 0.3969, 0.1807],
    "Hist Gradient Boosting": [0.7623, 0.0016, 0.8291, 0.8101, 0.6694, 0.3864, 0.1818],
    "Neural Network (MLP)":   [0.7469, 0.0021, 0.8167, 0.8027, 0.6646, 0.3694, 0.1875],
    "Decision Tree":          [0.7277, 0.0017, 0.7949, 0.8008, 0.6457, 0.3426, 0.1925],
    "Logistic Regression":    [0.7221, 0.0023, 0.7960, 0.7964, 0.6450, 0.3341, 0.1951],
}
# ROC-AUC on the held-out test set (notebook section 2.6). LightGBM (tuned) was not scored there.
TEST_AUC = {
    "XGBoost (tuned)": 0.7807, "LightGBM": 0.7781, "XGBoost": 0.7779, "Random Forest": 0.7746,
    "Hist Gradient Boosting": 0.7705, "Neural Network (MLP)": 0.7569, "Decision Tree": 0.7360,
    "Logistic Regression": 0.7300,
}
CHAMPION = "XGBoost (tuned)"

# Champion on the held-out test set at the 0.65 cutoff
TEST_METRICS = {
    "ROC-AUC": 0.7807, "ROC-AUC 95% CI": (0.7745, 0.7869), "PR-AUC": 0.8452, "Accuracy": 0.7173,
    "Balanced accuracy": 0.7089, "Precision": 0.8042, "Recall": 0.7388, "Specificity": 0.6791,
    "F1": 0.7701, "MCC": 0.4074, "Brier": 0.1744, "Train ROC-AUC (in-sample)": 0.8412,
}
CONFUSION_PCT = [[67.91, 32.09], [26.12, 73.88]]  # rows: actual repaid, actual default (share of each true class)
DECILES = [14.0, 34.8, 50.8, 60.7, 67.4, 74.6, 78.7, 82.5, 85.6, 91.7]  # observed default rate (%) by predicted-risk decile

# Robustness (notebook section 2.7)
ROBUSTNESS = [
    ("Random split (reference)", 0.7807),
    ("Time split: train to 2018, test 2019 onward", 0.7154),
    ("Without Interest", 0.7788),
    ("Without Country", 0.7783),
    ("Without the 5 engineered ratios", 0.7810),
]

# SHAP importance of the tuned champion on 4,000 test loans (notebook section 3.1)
NOTEBOOK_SHAP = [
    ("Loan duration", 0.2944), ("EE score", 0.2462), ("Country", 0.2343), ("Education", 0.1592),
    ("Previous repayment ratio", 0.1521), ("Interest rate", 0.1444), ("Home ownership", 0.1286),
    ("Verification type", 0.1015), ("Age", 0.0951), ("Existing liabilities", 0.0926),
]

# Neurosymbolic scorecard: logistic regression on the nine written rules (notebook section 3.3.3)
SCORECARD_BIAS = 0.235
SCORECARD_WEIGHTS = {
    "HighInterestRate": 0.7395, "LongLoanTerm": 0.5339, "HighPaymentBurden": 0.2134,
    "LargeLoanVsIncome": 0.0387, "HighLiabilityLoad": -0.0243, "ShortEmploymentTenure": -0.0325,
    "FirstTimeBorrower": -0.0988, "LowIncome": -0.1366, "ProvenRepayer": -0.5285,
}
SCORECARD_THRESHOLD = 0.646
SCORECARD_TEST_AUC = 0.6473
MINED_RULES = {"rules": 14, "fidelity": 0.787, "always_flag_baseline": 0.589}

# Expert-constraint test: share of test applicants whose score moves the wrong way (notebook section 3.3.5).
# Tuple order: (constraint, champion, knowledge-constrained model)
CONSTRAINT_VIOLATIONS = [
    ("Interest +5 points", 0.183, 0.000),
    ("Loan term +6 months", 0.058, 0.000),
    ("Earlier repayments +500", 0.429, 0.024),
]
MONOTONE_AUC = 0.7763

# Fairness audit (notebook section 3.5). False-positive rate = good borrowers wrongly flagged.
FAIRNESS = {
    "countries": ["Estonia", "Finland", "Spain"],
    "fpr_global": [16.2, 62.7, 74.5],
    "fpr_per_country": [32.5, 33.2, 31.1],
    "flagged_global": [36.2, 78.9, 89.6],
    "recall_global": [54.2, 84.6, 93.4],
    "recall_per_country": [75.0, 57.7, 58.3],
    "fpr_gap": (0.583, 0.020),
    "balanced_accuracy": (0.709, 0.665),
}
COUNTRY_THRESHOLDS = {"FI": 0.748, "ES": 0.82, "EE": 0.523, "SK": 0.65}
AGE_FPR = {"24 or younger": 30.4, "25 to 34": 26.8, "35 to 44": 30.9, "45 to 54": 34.6, "55 and over": 45.4}
