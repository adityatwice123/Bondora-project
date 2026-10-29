import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import explain  # noqa: E402
import loan_utils as lu  # noqa: E402
import reasoning  # noqa: E402


def _predicates(*on):
    return pd.Series({rule["name"]: int(rule["name"] in on) for rule in lu.RULEBOOK})


def test_scorecard_matches_notebook_examples():
    # the three applicants printed in notebook section 3.3.7
    assert round(reasoning.scorecard(_predicates("LongLoanTerm", "HighPaymentBurden", "LargeLoanVsIncome"))[0], 2) == 0.74
    assert round(reasoning.scorecard(_predicates("LowIncome"))[0], 2) == 0.52
    assert round(reasoning.scorecard(_predicates("ProvenRepayer", "LowIncome", "HighLiabilityLoad"))[0], 2) == 0.39


def test_agreement_labels():
    assert reasoning.agreement(True, True) == ("Consensus: default", False)
    assert reasoning.agreement(False, False) == ("Consensus: repay", False)
    assert reasoning.agreement(True, False)[1] and reasoning.agreement(False, True)[1]


def test_rescale_payment_identity_and_direction():
    same = lu.rescale_payment(397.38, 5425, 34.57, 60, 5425, 34.57, 60)
    assert same == pytest.approx(397.38)
    assert lu.rescale_payment(397.38, 5425, 34.57, 60, 10850, 34.57, 60) == pytest.approx(2 * 397.38)  # twice the loan, twice the payment
    assert lu.rescale_payment(397.38, 5425, 34.57, 60, 5425, 34.57, 36) > 397.38  # shorter term, bigger instalment
    assert lu.rescale_payment(0, 5425, 34.57, 60, 9000, 34.57, 60) == 0  # unknown payment stays unknown


def test_friendly_names_are_readable():
    for column in lu.RAW_INPUT_COLUMNS + lu.ENGINEERED:
        label = explain.friendly_name(column)
        assert label == label.strip() and label[0].isupper()
        assert not (label.islower() or label.lower() == label and len(label) > 14), label  # no run-together lowercase names
    assert explain.friendly_name("ExistingLiabilities") == "Existing liabilities (count)"
    assert explain.friendly_name("num__SomeNewField") == "Some new field"
    assert explain.feature_base("cat__Country_EE") == "Country"
    assert explain.feature_base("Education_infrequent_sklearn") == "Education"
    assert explain.feature_base("num__AmountToIncome") == "AmountToIncome"


def test_format_value():
    assert explain.format_value("Interest", 34.57) == "34.6%"
    assert explain.format_value("PaymentToIncome", 0.159) == "16%"
    assert explain.format_value("LoanDuration", 60) == "60 months"
    assert explain.format_value("Amount", 5425) == "€5,425"
    assert explain.format_value("Country", "ES") == "ES"


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(ROOT / "LoanData_Sample.csv", low_memory=False)


def test_balanced_sample_country_mix(sample):
    picked = explain.balanced_sample(sample)
    share = picked["Country"].value_counts(normalize=True)
    assert 0.40 < share["EE"] < 0.65 and share["ES"] > 0.10 and share["FI"] > 0.15


def test_sample_has_only_needed_columns_and_no_personal_ids(sample):
    assert "UserName" not in sample.columns and "DateOfBirth" not in sample.columns
    assert set(lu.RAW_INPUT_COLUMNS) <= set(sample.columns)


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_shipped_models_load_and_score(name, sample):
    model = joblib.load(ROOT / f"artifacts/{name}_pipeline.joblib")
    engineered = lu.engineer_features(sample)
    scores = model.predict_proba(engineered)[:, 1]
    assert np.isfinite(scores).all() and scores.min() >= 0 and scores.max() <= 1
    assert 0.05 < (scores >= lu.THRESHOLD).mean() < 0.60
    options = lu.category_options(model)
    assert "EE" in options["Country"] and "Missing" in options["CreditScoreEeMini"]


def test_constraint_check_shape(sample):
    model = joblib.load(ROOT / "artifacts/xgboost_pipeline.joblib")
    row = sample.iloc[[0]][lu.RAW_INPUT_COLUMNS]
    result = reasoning.constraint_check(model, row)
    assert len(result) == len(reasoning.CONSTRAINTS)
    assert set(result["Result"]) <= {"Broken", "Respected"}
