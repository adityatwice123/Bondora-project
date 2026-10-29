import re

import numpy as np
import pandas as pd
import streamlit as st

# Plain-English names for every model input. Anything missing falls back to a split of the camel-case name.
FRIENDLY_NAMES = {
    "Age": "Age",
    "Interest": "Interest rate",
    "LoanDuration": "Loan duration",
    "IncomeTotal": "Monthly income",
    "ExistingLiabilities": "Existing liabilities (count)",
    "LiabilitiesTotal": "Total liabilities",
    "NoOfPreviousLoansBeforeLoan": "Previous loans (count)",
    "AmountOfPreviousLoansBeforeLoan": "Previous loans amount",
    "PreviousRepaymentsBeforeLoan": "Earlier repayments",
    "PreviousEarlyRepaymentsCountBeforeLoan": "Early repayments (count)",
    "PaymentToIncome": "Payment as share of income",
    "LiabToIncome": "Liabilities as share of income",
    "AmountToIncome": "Loan amount as share of income",
    "GrantedRatio": "Granted amount ratio",
    "PrevRepaymentRatio": "Previous repayment ratio",
    "AppliedAmount": "Amount applied for",
    "Amount": "Amount granted",
    "MonthlyPayment": "Monthly payment",
    "Country": "Country",
    "Education": "Education",
    "VerificationType": "Verification type",
    "HomeOwnershipType": "Home ownership",
    "EmploymentDurationCurrentEmployer": "Employment duration",
    "CreditScoreFiAsiakasTietoRiskGrade": "FI risk grade",
    "CreditScoreEeMini": "EE score",
    "NewCreditCustomer": "New customer",
}

_CURRENCY = {"IncomeTotal", "LiabilitiesTotal", "AmountOfPreviousLoansBeforeLoan", "PreviousRepaymentsBeforeLoan",
             "AppliedAmount", "Amount", "MonthlyPayment"}
_PERCENT = {"PaymentToIncome", "LiabToIncome", "PrevRepaymentRatio", "GrantedRatio"}


def friendly_name(name):
    base = name.split("__", 1)[-1]
    if base in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[base]
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", base.replace("_", " "))
    return spaced.strip().capitalize()


def feature_base(name):
    """Map an encoded column such as 'cat__Country_EE' back to its original feature 'Country'."""
    base = name.split("__", 1)[-1]
    for column in sorted(FRIENDLY_NAMES, key=len, reverse=True):
        if base == column or base.startswith(column + "_"):
            return column
    return base


def format_value(feature, value):
    """Show a model input the way a person would write it."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not available"
    if isinstance(value, (str, np.str_)):
        return str(value)
    value = float(value)
    if feature in _PERCENT:
        return f"{value:.0%}"
    if feature == "AmountToIncome":
        return f"{value:.1f}x income"
    if feature == "Interest":
        return f"{value:.1f}%"
    if feature == "LoanDuration":
        return f"{value:.0f} months"
    if feature in _CURRENCY:
        return f"€{value:,.0f}"
    if feature == "NewCreditCustomer":
        return "Yes" if value >= 0.5 else "No"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def balanced_sample(frame, n=800, seed=42):
    """Draw a sample whose country mix is close to the held-out test set (about 52% EE, 26% FI, 21% ES).

    The sample file is dominated by Estonian loans, so a plain random draw would make
    "what matters overall" mostly an Estonia story.
    """
    shares = {"EE": 0.52, "FI": 0.26, "ES": 0.21, "SK": 0.01}
    parts = []
    for country, share in shares.items():
        rows = frame.loc[frame["Country"].astype(str) == country]
        take = min(len(rows), int(round(n * share)))
        if take:
            parts.append(rows.sample(take, random_state=seed))
    return pd.concat(parts) if parts else frame.sample(min(n, len(frame)), random_state=seed)


@st.cache_resource
def tree_explainer(model_name, _model):
    import shap
    return shap.TreeExplainer(_model.named_steps["classifier"])


def _class_one_values(values):
    if isinstance(values, list):
        return np.asarray(values[1] if len(values) > 1 else values[0])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def _transform(model, engineered):
    transformed = model.named_steps["preprocessor"].transform(engineered)
    return transformed.toarray() if hasattr(transformed, "toarray") else transformed


def local_shap(model_name, model, engineered, top=8):
    """SHAP contributions for one applicant, one-hot columns folded back into their original feature."""
    explainer = tree_explainer(model_name, model)
    values = _class_one_values(explainer.shap_values(_transform(model, engineered)))[0]
    names = model.named_steps["preprocessor"].get_feature_names_out()
    grouped = {}
    for name, value in zip(names, values):
        base = feature_base(name)
        grouped[base] = grouped.get(base, 0.0) + float(value)
    row = engineered.iloc[0]
    rows = [{"feature": base, "label": friendly_name(base), "shap": value,
             "display": format_value(base, row.get(base, ""))} for base, value in grouped.items()]
    frame = pd.DataFrame(rows)
    frame = frame.reindex(frame["shap"].abs().sort_values(ascending=False).index).head(top)
    base_value = float(np.asarray(getattr(explainer, "expected_value", 0)).reshape(-1)[-1])
    return frame.reset_index(drop=True), base_value


@st.cache_data(show_spinner=False)
def global_shap_importance(model_name, _model, sample, top=10):
    """Mean absolute SHAP per original feature over a sample of engineered applicants."""
    values = _class_one_values(tree_explainer(model_name, _model).shap_values(_transform(_model, sample)))
    names = _model.named_steps["preprocessor"].get_feature_names_out()
    grouped = {}
    for name, value in zip(names, np.abs(values).mean(axis=0)):
        base = feature_base(name)
        grouped[base] = grouped.get(base, 0.0) + float(value)
    frame = pd.DataFrame([{"feature": key, "label": friendly_name(key), "importance": value} for key, value in grouped.items()])
    return frame.nlargest(top, "importance")


@st.cache_data(show_spinner=False)
def sample_scores(model_name, _model, sample):
    """Predicted default probability for every loan in the (engineered) sample."""
    return _model.predict_proba(sample)[:, 1]
