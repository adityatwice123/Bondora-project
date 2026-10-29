"""Bondora default-risk platform (Streamlit).

Tabs: Assess (sample or custom applicant), Explanation (why, rules, what-if), Data and preprocessing (funnel, column checks,
learned limits, what the model saw), Models (comparison and results), How it works (pipeline, honest limits, fairness).
"""
import json
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import charts
import components as ui
import explain
import loan_utils as lu
import notebook_results as nr
import preprocessing as pp
import reasoning
from styles import get_css

BASE_DIR = Path(__file__).resolve().parent  # files are found next to app.py, wherever Streamlit is launched from
THRESHOLD = lu.THRESHOLD
COUNTRY_THRESHOLDS = nr.COUNTRY_THRESHOLDS
TUNED = "XGBoost (tuned)"
MODEL_PATHS = {
    TUNED: BASE_DIR / "artifacts" / "xgboost_tuned_pipeline.joblib",
    "XGBoost": BASE_DIR / "artifacts" / "xgboost_pipeline.joblib",
    "LightGBM": BASE_DIR / "artifacts" / "lightgbm_pipeline.joblib",
    "Random Forest": BASE_DIR / "artifacts" / "random_forest_pipeline.joblib",
}
OPTIONAL_MODELS = {TUNED, "Random Forest"}  # shown only when their file exists
RULE_TEXT = {rule["name"]: rule["text"] for rule in lu.RULEBOOK}
RULE_SHORT = {
    "HighPaymentBurden": "Heavy payment burden", "HighLiabilityLoad": "High existing liabilities",
    "LargeLoanVsIncome": "Large loan vs income", "HighInterestRate": "High interest rate", "LongLoanTerm": "Long loan term",
    "FirstTimeBorrower": "First-time borrower", "ShortEmploymentTenure": "Short employment", "ProvenRepayer": "Proven repayer",
    "LowIncome": "Low income",
}

st.set_page_config(page_title="Bondora Default Risk", layout="wide", initial_sidebar_state="collapsed")
st.markdown(get_css(), unsafe_allow_html=True)


# --------------------------------------------------------------------------- data and models
@st.cache_resource
def load_models():
    models, messages = {}, []
    for label, path in MODEL_PATHS.items():
        if not path.exists():
            if label not in OPTIONAL_MODELS:
                messages.append(("warning", f"{label} is unavailable because {path} is missing. Run train_quick_model.py to create it."))
            continue
        try:
            pipeline = joblib.load(path)
            assert {"preprocessor", "classifier"} <= set(pipeline.named_steps), "expected steps 'preprocessor' and 'classifier'"
            models[label] = pipeline
        except Exception as error:  # pickles can fail when library versions differ
            messages.append(("warning", f"{label} could not be loaded ({error}). Run train_quick_model.py with your installed library versions."))
    return models, messages


@st.cache_data
def load_sample_data():
    frame = pd.read_csv(BASE_DIR / "LoanData_Sample.csv", low_memory=False)
    frame["LoanDate"] = pd.to_datetime(frame["LoanDate"], errors="coerce")
    return frame


@st.cache_data
def balanced_engineered():
    """Engineered features for a country-balanced draw from the sample, used for 'overall' views."""
    return lu.engineer_features(explain.balanced_sample(load_sample_data()))


@st.cache_data
def sample_engineered():
    return lu.engineer_features(load_sample_data())


@st.cache_data(show_spinner=False)
def scored_sample(_models, names):
    """Scores from every model for the country-balanced sample, next to each loan's recorded outcome."""
    raw = explain.balanced_sample(load_sample_data())
    engineered = lu.engineer_features(raw)
    frame = pd.DataFrame({"Loan": raw["LoanId"].astype(str).str[:8].values, "Country": raw["Country"].values,
                          "Issued": raw["LoanDate"].dt.year.values, "Amount (EUR)": raw["Amount"].values,
                          "Interest (%)": raw["Interest"].values, "Months": raw["LoanDuration"].values})
    for model_name, model in _models.items():
        frame[f"{model_name} score"] = model.predict_proba(engineered)[:, 1]
    frame["Recorded outcome"] = np.where(raw["DefaultDate"].notna().values, "Defaulted", "Repaid")
    return frame


def default_model_name(models):
    return TUNED if TUNED in models else ("XGBoost" if "XGBoost" in models else next(iter(models)))


def selected_model(models):
    name = st.session_state.get("selected_model", default_model_name(models))
    return name if name in models else default_model_name(models)


def active_threshold(country):
    if st.session_state.get("country_thresholds"):
        return COUNTRY_THRESHOLDS.get(country, THRESHOLD)
    return THRESHOLD


# --------------------------------------------------------------------------- form helpers
def row_number(row, column, default, integer=False):
    value = row.get(column, default)
    if pd.isna(value):
        value = default
    return int(value) if integer else float(value)


def row_choice(row, column, choices):
    value = lu._cat_to_str(pd.Series([row.get(column)]))[0]
    return value if value in choices else ("Missing" if "Missing" in choices else choices[0])


def neutral_row(model):
    options = lu.category_options(model)
    values = {
        "Age": 35, "AppliedAmount": 5000.0, "Amount": 5000.0, "Interest": 15.0, "LoanDuration": 36,
        "MonthlyPayment": 150.0, "IncomeTotal": 2500.0, "ExistingLiabilities": 0,
        "LiabilitiesTotal": 0.0, "NoOfPreviousLoansBeforeLoan": 1, "AmountOfPreviousLoansBeforeLoan": 3000.0,
        "PreviousRepaymentsBeforeLoan": 1500.0, "PreviousEarlyRepaymentsCountBeforeLoan": 0,
        "NewCreditCustomer": 0,
    }
    for column, choices in options.items():
        values[column] = "EE" if column == "Country" and "EE" in choices else ("Missing" if "Missing" in choices else choices[0])
    return pd.Series(values)


def clear_form_values():
    """Forget typed form values (keys such as custom_0_age) but keep the drawn sample applicant."""
    for key in list(st.session_state.keys()):
        if key.startswith(("custom_", "random_")) and key != "random_row":
            st.session_state.pop(key, None)


def choose_random_row(sample_df):
    countries = sample_df["Country"].dropna().astype(str).unique().tolist()
    country = np.random.default_rng().choice(countries)
    return sample_df.loc[sample_df["Country"].astype(str) == country].sample(1).iloc[0]


def draw_random(sample_df):
    st.session_state["random_row"] = choose_random_row(sample_df)
    clear_form_values()
    st.session_state.pop("last_result", None)
    st.session_state["form_revision"] = st.session_state.get("form_revision", 0) + 1


def start_app():
    st.session_state["stage"] = "app"


def start_over():
    st.session_state["stage"] = "landing"
    st.session_state.pop("last_result", None)
    clear_form_values()


def locked_or_open(options, column, country, open_for):
    """Bureau scores only exist for their home market, so every other country is locked to 'Missing'."""
    if country == open_for:
        return options[column], False
    return (["Missing"] if "Missing" in options[column] else [options[column][0]]), True


# --------------------------------------------------------------------------- landing
def render_landing():
    st.markdown('<main class="landing-screen">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Bondora loan data</div>', unsafe_allow_html=True)
    st.title("Estimate how likely a loan is to default, and see why.")
    st.markdown('<p class="landing-copy">Enter an applicant\'s details or pick a real one from the Bondora loan book. The app gives a risk estimate, compares the models, and shows which factors pushed it up or down.</p>', unsafe_allow_html=True)
    st.write("")
    columns = st.columns(3)
    steps = [("1", "Choose an applicant", "Enter details yourself or draw a real example."),
             ("2", "Get a risk estimate", "Trained models score the loan and a written rulebook gives a second opinion."),
             ("3", "See the reasons", "Plain-language drivers, rules that fired, and a what-if explorer.")]
    for column, (number, title, copy) in zip(columns, steps):
        with column:
            st.markdown(f'<div class="step-number">{number}</div><div class="step-title">{title}</div><div class="step-copy">{copy}</div>', unsafe_allow_html=True)
    st.write("")
    st.button("Start an assessment", on_click=start_app, type="primary")
    st.markdown('<div class="landing-footer">Research prototype built on public Bondora data. Not for real lending decisions.</div>', unsafe_allow_html=True)
    st.markdown('</main>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- assess tab
def input_form(row, options, random_mode, namespace):
    key = lambda name: f"{namespace}_{name}"
    container = st.expander("Review or edit details", expanded=True) if random_mode else nullcontext()
    with container:
        ui.label("About the applicant")
        country_col, hint_col = st.columns(2)
        with country_col:
            country = st.selectbox("Country", options["Country"], index=options["Country"].index(row_choice(row, "Country", options["Country"])),
                                   help="Bondora country code.", key=key("country"))
        with hint_col:
            st.caption("Country decides which bureau score can exist: the EE score for Estonia, the FI risk grade for Finland. Other markets have neither, so those fields lock to Missing.")
        with st.form("assessment_form"):
            one, two = st.columns(2)
            with one:
                age = st.number_input("Age", min_value=18, max_value=100, value=min(max(row_number(row, "Age", 35, True), 18), 100), help="Applicant age in years.", key=key("age"))
                income = st.number_input("Monthly income (EUR)", min_value=0.0, value=row_number(row, "IncomeTotal", 2500.0), help="Total monthly income in euros.", key=key("income"))
            with two:
                customer_value = 1 if row.get("NewCreditCustomer") == 1 else 0
                new_customer = st.selectbox("Customer type", [0, 1], index=customer_value,
                                            format_func=lambda value: "New customer" if value == 1 else "Returning customer",
                                            help="Whether this is the applicant's first credit customer record.", key=key("new_customer"))

            ui.label("The loan")
            loan_one, loan_two = st.columns(2)
            with loan_one:
                applied_amount = st.number_input("Amount applied for (EUR)", min_value=0.0, value=row_number(row, "AppliedAmount", 5000.0), help="Requested loan amount.", key=key("applied_amount"))
                amount = st.number_input("Amount granted (EUR)", min_value=0.0, value=row_number(row, "Amount", 5000.0), help="Granted loan amount.", key=key("amount"))
                interest = st.number_input("Interest rate (%)", min_value=0.0, value=row_number(row, "Interest", 15.0), help="Annual interest rate.", key=key("interest"))
            with loan_two:
                duration = st.number_input("Duration (months)", min_value=1, max_value=120, value=min(max(row_number(row, "LoanDuration", 36, True), 1), 120),
                                           help="Bondora offers up to 60 months. Longer terms are treated as 60 by the models.", key=key("duration"))
                monthly_payment = st.number_input("Monthly payment (EUR)", min_value=0.0, value=row_number(row, "MonthlyPayment", 0.0),
                                                  help="Expected monthly payment. 0 means unknown and is treated as missing, as in the notebook.", key=key("monthly_payment"))

            ui.label("Credit history")
            history_one, history_two = st.columns(2)
            with history_one:
                existing_liabilities = st.number_input("Existing liabilities (count)", min_value=0, value=row_number(row, "ExistingLiabilities", 0, True), help="Number of existing liabilities.", key=key("existing_liabilities"))
                liabilities_total = st.number_input("Total liabilities (EUR)", min_value=0.0, value=row_number(row, "LiabilitiesTotal", 0.0), help="Total value of existing liabilities.", key=key("liabilities_total"))
                prev_loans = st.number_input("Previous loans (count)", min_value=0, value=row_number(row, "NoOfPreviousLoansBeforeLoan", 0, True), help="Loans held before this application.", key=key("prev_loans"))
            with history_two:
                prev_loans_amt = st.number_input("Previous loans amount (EUR)", min_value=0.0, value=row_number(row, "AmountOfPreviousLoansBeforeLoan", 0.0), help="Amount borrowed on previous loans.", key=key("prev_loans_amt"))
                prev_repayments = st.number_input("Previous repayments (EUR)", min_value=0.0, value=row_number(row, "PreviousRepaymentsBeforeLoan", 0.0), help="Amount repaid before this loan.", key=key("prev_repayments"))
                prev_early_repay = st.number_input("Previous early repayments", min_value=0, value=row_number(row, "PreviousEarlyRepaymentsCountBeforeLoan", 0, True), help="Count of previous early repayments.", key=key("prev_early_repay"))

            ui.label("Background")
            back_one, back_two = st.columns(2)
            ee_choices, ee_locked = locked_or_open(options, "CreditScoreEeMini", country, "EE")
            fi_choices, fi_locked = locked_or_open(options, "CreditScoreFiAsiakasTietoRiskGrade", country, "FI")
            with back_one:
                verification = st.selectbox("Verification type", options["VerificationType"], index=options["VerificationType"].index(row_choice(row, "VerificationType", options["VerificationType"])), help="Bondora category code.", key=key("verification"))
                education = st.selectbox("Education", options["Education"], index=options["Education"].index(row_choice(row, "Education", options["Education"])), help="Bondora education category code.", key=key("education"))
                home_ownership = st.selectbox("Home ownership", options["HomeOwnershipType"], index=options["HomeOwnershipType"].index(row_choice(row, "HomeOwnershipType", options["HomeOwnershipType"])), help="Bondora home ownership category code.", key=key("home_ownership"))
            with back_two:
                employment = st.selectbox("Employment duration", options["EmploymentDurationCurrentEmployer"], index=options["EmploymentDurationCurrentEmployer"].index(row_choice(row, "EmploymentDurationCurrentEmployer", options["EmploymentDurationCurrentEmployer"])), help="Bondora employment duration code.", key=key("employment"))
                fi_score = st.selectbox("FI risk grade", fi_choices, index=fi_choices.index(row_choice(row, "CreditScoreFiAsiakasTietoRiskGrade", fi_choices)), disabled=fi_locked,
                                        help="Finnish bureau grade. Only exists for Finnish applicants.", key=key(f"fi_score_{country}"))
                ee_score = st.selectbox("EE score", ee_choices, index=ee_choices.index(row_choice(row, "CreditScoreEeMini", ee_choices)), disabled=ee_locked,
                                        help="Estonian bureau score. Only exists for Estonian applicants, and Missing is common there.", key=key(f"ee_score_{country}"))
            submitted = st.form_submit_button("Assess this applicant", type="primary")
    values = {
        "Age": age, "AppliedAmount": applied_amount, "Amount": amount, "Interest": interest, "LoanDuration": duration, "MonthlyPayment": monthly_payment, "IncomeTotal": income,
        "ExistingLiabilities": existing_liabilities, "LiabilitiesTotal": liabilities_total, "NoOfPreviousLoansBeforeLoan": prev_loans, "AmountOfPreviousLoansBeforeLoan": prev_loans_amt,
        "PreviousRepaymentsBeforeLoan": prev_repayments, "PreviousEarlyRepaymentsCountBeforeLoan": prev_early_repay, "NewCreditCustomer": new_customer, "Country": country,
        "VerificationType": verification, "Education": education, "HomeOwnershipType": home_ownership, "EmploymentDurationCurrentEmployer": employment,
        "CreditScoreFiAsiakasTietoRiskGrade": fi_score, "CreditScoreEeMini": ee_score,
    }
    return submitted, pd.DataFrame([values])


def render_assess(models, sample_df):
    ui.label("Step 1")
    st.subheader("Who do you want to assess?")
    choices = ["Custom applicant", "Random from the Bondora sample"]
    mode = st.segmented_control("Applicant source", choices, default=choices[0], key="input_mode", label_visibility="collapsed") or choices[0]
    random_mode = mode == choices[1]
    previous_mode = st.session_state.get("previous_input_mode")
    if previous_mode is not None and previous_mode != mode:
        clear_form_values()
        st.session_state.pop("last_result", None)
        st.session_state["form_revision"] = st.session_state.get("form_revision", 0) + 1
    st.session_state["previous_input_mode"] = mode

    with st.expander("Advanced", expanded=False):
        names = list(models)
        st.selectbox("Model", names, index=names.index(selected_model(models)), key="selected_model",
                     help="The model used for the headline score and the explanation. Every model is always compared below.")
        st.checkbox("Use per-country cutoffs (experimental)", key="country_thresholds",
                    help="Cutoffs were tuned on the notebook's champion, so they are only a rough guide for the other models.")
        for kind, message in load_models()[1]:
            st.info(message) if kind == "info" else st.warning(message)
    model = models[selected_model(models)]
    options = lu.category_options(model)

    if random_mode:
        if "random_row" not in st.session_state:
            st.session_state["random_row"] = choose_random_row(sample_df)
        row = st.session_state["random_row"]
        ui.label("Selected sample applicant")
        ui.outcome_card(row)
        draw_col, _ = st.columns([1, 3])
        with draw_col:
            st.button("Draw another", key="draw_another", on_click=draw_random, args=(sample_df,))
        st.caption("The sample holds real Bondora loans issued 2009 to 2016, mostly Estonian. The draw picks a country first so all three markets appear.")
    else:
        row = neutral_row(model)

    ui.label("Step 2")
    namespace = f"{'random' if random_mode else 'custom'}_{st.session_state.get('form_revision', 0)}"
    submitted, input_data = input_form(row, options, random_mode, namespace)

    if submitted:
        if float(input_data.loc[0, "IncomeTotal"]) <= 0:
            st.error("Monthly income must be greater than zero before the score can be calculated.")
        else:
            if input_data.loc[0, "Amount"] > input_data.loc[0, "AppliedAmount"]:
                st.warning("The granted amount is higher than the applied amount. The score can still be calculated.")
            with st.spinner("Calculating the assessment"):
                engineered = lu.engineer_features(input_data)
                probabilities = {name: float(m.predict_proba(engineered)[0, 1]) for name, m in models.items()}
            st.session_state["result_id"] = st.session_state.get("result_id", 0) + 1
            st.session_state["last_result"] = {"id": st.session_state["result_id"], "input": input_data, "engineered": engineered,
                                               "probabilities": probabilities, "random_mode": random_mode, "source_row": row}
    render_result(models)


def render_result(models):
    result = st.session_state.get("last_result")
    if not result:
        return
    name = selected_model(models)
    model = models[name]
    probability = result["probabilities"][name]
    country = str(result["input"].loc[0, "Country"])
    threshold = active_threshold(country)
    flagged = probability >= threshold

    ui.label("Step 3")
    st.subheader("Assessment")
    left, right = st.columns([1, 1.4])
    with left:
        ui.risk_result(probability, threshold)
        extra = f" Using the experimental cutoff for {country}." if st.session_state.get("country_thresholds") else ""
        st.caption(f"{name} score. Scores of {charts.pct_label(threshold)} or above are flagged. The cutoff balances catching defaulters against wrongly flagging good borrowers.{extra}")
        if result["random_mode"]:
            source = result["source_row"]
            has_outcome = pd.notna(source.get("DefaultDate")) or str(source.get("Status", "")) in {"Repaid", "Defaulted"}
            if has_outcome and st.checkbox("Reveal what actually happened", key="reveal_outcome"):
                defaulted = pd.notna(source.get("DefaultDate"))
                st.caption(f"Recorded outcome: {'defaulted' if defaulted else 'repaid with no default'}. "
                           "This loan was part of the data the models learned from, so the match is flattering.")
    with right:
        ui.show_chart(charts.risk_meter(probability, threshold))
    ui.show_chart(charts.model_bars(result["probabilities"], threshold))
    values = list(result["probabilities"].values())
    if len(values) > 1 and max(values) - min(values) > 0.15:
        st.info("The models disagree by more than 15 points. Treat this case with extra care.")

    # second opinion from the written rules
    predicates = lu.symbolic_predicates(result["engineered"]).iloc[0]
    rules_prob, _ = reasoning.scorecard(predicates)
    verdict, needs_human = reasoning.agreement(flagged, rules_prob >= nr.SCORECARD_THRESHOLD)
    ui.label("Second opinion")
    st.subheader("What the written rules say")
    kind = "warn" if verdict == "Consensus: default" else ("ok" if verdict == "Consensus: repay" else "info")
    rule_left, rule_right = st.columns([1, 2])
    with rule_left:
        ui.stat_cards([(f"{rules_prob:.0%}", "Risk from nine hand-written credit rules")])
    with rule_right:
        st.markdown(ui.chip(verdict, kind), unsafe_allow_html=True)
        if needs_human:
            st.write("The model and the rules would decide differently, so a credit officer should look at this case.")
        else:
            st.write("The model and the rules reach the same decision, so the rules can serve as a plain-language reason.")
        st.caption(f"The rules alone are much cruder (ROC-AUC {nr.SCORECARD_TEST_AUC:.2f} against {nr.TEST_METRICS['ROC-AUC']:.2f} for the notebook champion). They are a sanity check, not a second model to trust.")

    # where this applicant sits in the sample
    scores = explain.sample_scores(name, model, balanced_engineered())
    st.caption(f"{name} scores this applicant higher than {np.mean(scores < probability):.0%} of {len(scores):,} loans in the country-balanced sample.")

    download = result["input"].copy()
    for model_name, value in result["probabilities"].items():
        download[f"{model_name}_probability"] = value
    download["rules_probability"] = rules_prob
    download["threshold"] = threshold
    download["selected_model"] = name
    st.caption("See the Explanation tab to understand why.")
    st.download_button("Download this assessment (CSV)", download.to_csv(index=False), "bondora_assessment.csv", "text/csv")


# --------------------------------------------------------------------------- explanation tab
def render_explanation(models):
    result = st.session_state.get("last_result")
    name = selected_model(models)
    model = models[name]
    st.subheader("What drives a score")
    st.write("The model looks at things like interest rate, loan term, income and credit history. This page shows how much each factor pushed this applicant's risk up or down, which written rules apply, and what would change the score.")

    if not result:
        ui.note("Run an assessment on the Assess tab and the applicant-specific explanation appears here. The overall charts further down do not need one.")
    else:
        country = str(result["input"].loc[0, "Country"])
        st.subheader("Why this applicant got this score")
        try:
            local, base_value = explain.local_shap(name, model, result["engineered"])
            up = local.loc[local["shap"] > 0, "label"].head(2).tolist()
            down = local.loc[local["shap"] < 0, "label"].head(2).tolist()
            st.write(f"The biggest factors raising this applicant's risk are {', '.join(up) or 'none'}; the biggest factors lowering it are {', '.join(down) or 'none'}.")
            ui.show_chart(charts.signed_shap_bar(local))
            with st.expander("Technical detail"):
                st.write(f"SHAP values are in log-odds on transformed features and are grouped back to the original inputs. Base value: {base_value:.3f}. Model: {name}.")
        except Exception as error:
            st.info(f"This applicant's explanation is unavailable: {error}")

        st.subheader("Rules of thumb that apply")
        predicates = lu.symbolic_predicates(result["engineered"]).iloc[0]
        rules_prob, fired = reasoning.scorecard(predicates)
        rows, risk_count, protective_count = [], 0, 0
        for rule in lu.RULEBOOK:
            applies = bool(predicates[rule["name"]])
            risk_count += int(applies and rule["expected"] > 0)
            protective_count += int(applies and rule["expected"] < 0)
            rows.append({"Rule": rule["text"], "Applies": "Applies" if applies else "Does not apply",
                         "Normally": "raises risk" if rule["expected"] > 0 else "lowers risk",
                         "Scorecard weight": f"{nr.SCORECARD_WEIGHTS[rule['name']]:+.2f}"})
        ui.show_table(pd.DataFrame(rows))
        st.caption(f"{risk_count} risk {'signal' if risk_count == 1 else 'signals'} and {protective_count} protective {'signal' if protective_count == 1 else 'signals'} apply. "
                   "These are simple credit rules written by hand, so they can disagree with the model. The scorecard weight is how much each rule counts in the log-odds when they are combined.")
        if len(fired):
            ui.show_chart(charts.rule_weights_bar(fired, RULE_SHORT))

        st.subheader("Does the model follow basic credit logic for this applicant?")
        check = reasoning.constraint_check(model, result["input"])
        shown = check.copy()
        shown["Score moves by"] = shown["Score moves by"].map(lambda v: f"{v * 100:+.1f} pts")
        ui.show_table(shown)
        broken = int((check["Result"] == "Broken").sum())
        st.caption(f"{broken} of {len(check)} checks broken. Tree models have no built-in idea that more debt means more risk, so some breaks are normal. "
                   "In the notebook the champion broke these rules for 6% to 43% of test applicants, which is why a constrained variant was built.")

        render_what_if(model, name, result)

    st.subheader("What matters most overall")
    try:
        importance = explain.global_shap_importance(name, model, balanced_engineered())
        ui.show_chart(charts.global_importance(importance))
        st.caption(f"Average SHAP contribution of {name} over {len(balanced_engineered()):,} sample loans with a country mix close to the test set. "
                   "These loans were in the training data, and the shipped models are untuned, so the ranking can differ from the notebook's tuned champion below.")
    except Exception as error:
        st.info(f"Overall explanation is unavailable: {error}")
    with st.expander("Compare with the notebook's tuned champion"):
        ui.show_table(pd.DataFrame(nr.NOTEBOOK_SHAP, columns=["Feature", "Mean |SHAP| (log-odds)"]).round(3))
        st.caption("Tuned XGBoost on 4,000 held-out test loans. Loan duration, the EE score and Country lead in both views.")

    st.subheader("Does the model rank risk well?")
    ui.show_chart(charts.decile_chart())
    st.caption(f"Notebook result for the tuned champion on its held-out test set of {nr.TEST_SET_SIZE:,} loans. The riskiest tenth defaults more than six times as often as the safest tenth.")


def render_what_if(model, name, result):
    st.subheader("What if?")
    base = result["input"].iloc[0].copy()
    rid = result["id"]
    ui.note("Change a number and the score updates. The monthly payment is re-estimated from the amount, rate and term, and the granted share of the request stays the same, so the ratios stay consistent.")
    one, two = st.columns(2)
    base_interest, base_amount = float(base["Interest"]), float(base["Amount"])
    base_duration, base_income = int(min(max(base["LoanDuration"], 3), lu.MAX_TERM)), float(base["IncomeTotal"])
    with one:
        interest = st.slider("Interest rate (%)", 0.0, max(60.0, base_interest), min(base_interest, max(60.0, base_interest)), key=f"wi_interest_{rid}")
        amount = st.slider("Loan amount (EUR)", 100.0, max(30000.0, 2 * base_amount), max(base_amount, 100.0), step=100.0, key=f"wi_amount_{rid}")
    with two:
        duration = st.slider("Duration (months)", 3, lu.MAX_TERM, base_duration, key=f"wi_duration_{rid}")
        income = st.slider("Monthly income (EUR)", 100.0, max(15000.0, 2 * base_income), max(base_income, 100.0), step=50.0, key=f"wi_income_{rid}")

    def scenario_for(rate):
        item = base.copy()
        item["Interest"], item["Amount"], item["LoanDuration"], item["IncomeTotal"] = rate, amount, duration, income
        item["MonthlyPayment"] = lu.rescale_payment(base["MonthlyPayment"], base["Amount"], base["Interest"], base["LoanDuration"], amount, rate, duration)
        ratio = float(base["Amount"]) / float(base["AppliedAmount"]) if float(base["AppliedAmount"]) > 0 else 1.0
        item["AppliedAmount"] = amount / ratio if ratio > 0 else amount
        return item

    def probability_for(rate):
        return float(model.predict_proba(lu.engineer_features(pd.DataFrame([scenario_for(rate)])))[0, 1])

    current = result["probabilities"][name]
    scenario = probability_for(interest)
    st.metric("Scenario probability", f"{scenario:.1%}", delta=f"{(scenario - current) * 100:+.1f} pts vs assessed", delta_color="inverse")
    rates = np.linspace(max(0.0, interest - 10), min(80.0, interest + 10), 9)
    curve = pd.DataFrame({"interest": rates, "probability": [probability_for(rate) for rate in rates]})
    ui.show_chart(charts.line_what_if(curve, {"interest": interest, "probability": scenario}))


# --------------------------------------------------------------------------- data and preprocessing tab
def render_data(models, sample_df):
    name = selected_model(models)
    model = models[name]
    result = st.session_state.get("last_result")

    st.subheader("From 179,235 loans to a modelling set")
    ui.stat_cards([(f"{nr.POPULATION['all_loans']:,}", f"loans in the Bondora snapshot of {nr.SNAPSHOT_DATE}"), (str(nr.N_COLUMNS), "columns in the raw file"),
                   (f"{nr.POPULATION['modelling']:,}", "loans with a final outcome and 12+ months of history"), (f"{nr.POPULATION['default_rate']:.1%}", "of those defaulted")])
    ui.show_chart(charts.funnel_chart())
    st.write("**Target.** A loan counts as a default when Bondora has recorded a default date, which happens once it is more than 60 days overdue. "
             "Two traps were handled. Loans still running are not good loans, so only final outcomes are kept. Young loans have not had time to default, so only loans with at least 12 months of history stay in.")
    st.caption("Because unfinished loans are left out, the 64% default rate is higher than the platform-wide rate. The model ranks risk among loans that reached a final outcome.")

    st.subheader("Which columns the model may use")
    ui.stat_cards([(str(nr.N_COLUMNS), "original columns"), ("21", "application-time inputs the model sees"), ("91", "columns kept out, each for a stated reason")])
    st.write("")
    ui.show_chart(charts.governance_chart())
    with st.expander("Reason for each group"):
        ui.show_table(pd.DataFrame(nr.GOVERNANCE, columns=["Group", "Columns", "Why it is not a model input"]))
    st.caption("A column is allowed only if it is known at application time. Gender and language are held out and used only for the fairness audit. A coded check in the notebook makes sure every column sits in exactly one group.")

    st.subheader("Cleaning and new features")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Five ratios turn raw amounts into affordability signals**")
        ui.show_table(pd.DataFrame(nr.ENGINEERED_FEATURES, columns=["Feature", "Meaning"]))
    with right:
        zero = nr.ZERO_PAYMENT
        ui.note(f"<b>Data-quality catch.</b> {zero['loans']:,} loans had a monthly payment of exactly 0. {zero['issued_2013_2014']:.1%} were issued in 2013 to 2014 and "
                f"{zero['default_rate']:.1%} defaulted, against {zero['population_default_rate']:.1%} overall. A model could learn \"this is a 2013 loan\" from that zero, "
                "so a payment of 0 is treated as missing.")
    sample_pct = pp.sample_missing(sample_engineered())
    ui.show_chart(charts.missing_chart(sample_pct))
    handling = []
    for feature, share in nr.FEATURE_MISSING_PCT.items():
        categorical = feature in lu.CATEGORICAL
        handling.append({"Feature": explain.friendly_name(feature), "Missing in modelling set": f"{share:.1f}%", "Missing in sample": f"{sample_pct[feature]:.1f}%",
                         "How it is handled": "Blank becomes its own 'Missing' category" if categorical else "Filled with the training median"})
    with st.expander("Missing values for all 26 features"):
        ui.show_table(pd.DataFrame(handling))
    st.caption("Previous repayment ratio is missing for borrowers with no earlier loans, so those applicants get the typical ratio. That is a known simplification. "
               "A blank bureau score is not an error: it means no score exists for that market.")

    st.subheader("What the preprocessing learned")
    limits = pp.learned_limits(model)
    shown = pd.DataFrame({"Feature": limits["label"],
                          "Lower limit": [explain.format_value(f, v) for f, v in zip(limits["feature"], limits["lower"])],
                          "Upper limit": [explain.format_value(f, v) for f, v in zip(limits["feature"], limits["upper"])],
                          "Fill value if missing": [explain.format_value(f, v) for f, v in zip(limits["feature"], limits["median_fill"])]})
    ui.show_table(shown)
    prep = model.named_steps["preprocessor"]
    extra = []
    if len(prep.named_transformers_["num"].steps) > 2:
        extra.append("This model also standardises numbers, which makes no difference to trees.")
    if getattr(prep.named_transformers_["cat"], "min_frequency", None):
        extra.append("Categories seen in under 1% of training loans are pooled into one 'infrequent' column.")
    width = len(prep.get_feature_names_out())
    st.caption(f"Shown for {name}. Limits are the 0.5th and 99.5th percentile of the training loans; anything outside is clipped. 26 features become {width} model columns after one-hot encoding. " + " ".join(extra))

    st.subheader("What the model saw for this applicant")
    if not result:
        ui.note("Run an assessment on the Assess tab and this section follows that applicant through each preprocessing step.")
    else:
        numeric, categories = pp.preprocess_trace(model, result["engineered"], result["input"])
        changed = int((numeric["What happened"] != "Used as entered").sum())
        st.write(f"{changed} of {len(numeric)} numeric inputs were changed before scoring. Everything else was used as entered.")
        ui.show_table(numeric)
        with st.expander("Category fields"):
            ui.show_table(categories)

    st.subheader("The sample you can draw from")
    summary = pp.sample_summary(sample_df)
    ui.stat_cards([(f"{summary['rows']:,}", "real Bondora loans in the file"), (f"{summary['first_year']} to {summary['last_year']}", "issue years"),
                   (f"{summary['resolved']:,}", "with a final outcome"), (f"{summary['default_rate']:.1%}", "defaulted among those")])
    st.write("")
    ui.show_chart(charts.country_default_chart(summary["by_country"]))
    st.caption("The sample is mostly Estonian and only holds older loans, which is why its default rate is below the 64% of the full modelling set. "
               "Run make_sample.py with the full CSV for a sample that matches the country mix.")

    st.subheader("Exploration figures")
    for filename, title, caption in nr.EDA_FIGURES:
        path = BASE_DIR / "figures" / filename
        if path.exists():
            with st.expander(title):
                st.image(str(path), width="stretch")
                st.caption(caption)


# --------------------------------------------------------------------------- models tab
def describe_model(name, pipeline):
    classifier = pipeline.named_steps["classifier"]
    params = classifier.get_params()
    width = len(pipeline.named_steps["preprocessor"].get_feature_names_out())
    tuned = name == TUNED
    return {"Model": name, "Type": type(classifier).__name__, "Trees": params.get("n_estimators"),
            "Max depth": params.get("max_depth"), "Model columns": width,
            "Tuned": "Yes, random search" if tuned else "No, quick settings",
            "Trained on": "Training split (80%)" if tuned else "All resolved loans"}


def render_models(models):
    st.subheader("Models in this app")
    ui.show_table(pd.DataFrame([describe_model(name, model) for name, model in models.items()]))
    if TUNED not in models:
        ui.note("The notebook's tuned champion is not installed. Run <code>python train_champion.py</code> with the full Bondora CSV and the app picks it up on the next start.")
    metrics_path = BASE_DIR / "artifacts" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text()).get("metrics", {})
        frame = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
                                                                      "precision_at_threshold": "Precision at cutoff", "recall_at_threshold": "Recall at cutoff"})
        st.caption("Held-out scores written by train_quick_model.py (trained on 80% of loans, scored on the other 20%).")
        ui.show_table(frame.round(3))

    st.subheader("Live comparison on the sample")
    engineered = balanced_engineered()
    rows, flags = [], {}
    for name, model in models.items():
        scores = explain.sample_scores(name, model, engineered)
        flags[name] = scores >= THRESHOLD
        rows.append({"Model": name, "Average score": f"{scores.mean():.1%}", f"Flagged at {charts.pct_label(THRESHOLD)}": f"{flags[name].mean():.1%}"})
    ui.show_table(pd.DataFrame(rows))
    if len(flags) > 1:
        names = list(flags)
        same = float(np.mean(flags[names[0]] == flags[names[1]]))
        st.caption(f"{names[0]} and {names[1]} reach the same flag/no-flag decision on {same:.0%} of {len(engineered):,} sample loans. "
                   "These loans were in the training data, so use this to compare models with each other, not to judge accuracy.")

    scored = scored_sample(models, tuple(models))
    score_columns = [f"{n} score" for n in models]
    lead = f"{selected_model(models)} score"
    view = scored.sort_values(lead, ascending=False).head(12).copy()
    for column in score_columns:
        view[column] = view[column].map(lambda v: f"{v:.1%}")
    st.markdown(f"**Sample loans with the highest {selected_model(models)} scores, next to what really happened**")
    ui.show_table(view)
    st.download_button("Download all scored sample loans (CSV)", scored.to_csv(index=False), "bondora_scored_sample.csv", "text/csv")

    st.subheader("How the model families compared in the notebook")
    ui.show_chart(charts.cv_chart())
    table = pd.DataFrame(nr.CV_TABLE, index=nr.CV_COLUMNS).T
    table.insert(0, "Test ROC-AUC", [nr.TEST_AUC.get(name) for name in table.index])
    table = table.reset_index().rename(columns={"index": "Model"})
    ui.show_table(table.style.format({column: "{:.4f}" for column in table.columns if column != "Model"}, na_rep="n/a"))
    st.caption("Cross-validated scores come from the training set. The test column was scored once on 22,069 held-out loans; the tuned LightGBM was not scored there. "
               "Boosted trees beat logistic regression by about five ROC-AUC points, so the problem is not linear. Tuned XGBoost won under a rule fixed beforehand, "
               "but its lead over tuned LightGBM (0.0007) is smaller than the fold-to-fold spread, so either would be a fair champion.")

    st.subheader("Champion results on the held-out test set")
    metrics = nr.TEST_METRICS
    ui.stat_cards([(f"{metrics['ROC-AUC']:.3f}", f"ROC-AUC, 95% interval {metrics['ROC-AUC 95% CI'][0]:.3f} to {metrics['ROC-AUC 95% CI'][1]:.3f}"),
                   (f"{metrics['PR-AUC']:.3f}", "PR-AUC"),
                   (f"{metrics['Recall']:.1%}", "Recall: defaulters caught"),
                   (f"{metrics['Precision']:.1%}", "Precision: flagged loans that default")])
    st.write("")
    ui.stat_cards([(f"{metrics['Balanced accuracy']:.3f}", "Balanced accuracy"), (f"{metrics['F1']:.3f}", "F1"),
                   (f"{metrics['MCC']:.3f}", "MCC"), (f"{metrics['Brier']:.3f}", "Brier score (0.230 for always guessing the average)")])
    left, right = st.columns(2)
    with left:
        ui.show_chart(charts.confusion_chart())
    with right:
        robust = pd.DataFrame(nr.ROBUSTNESS, columns=["Setting", "ROC-AUC"])
        robust["Change"] = (robust["ROC-AUC"] - nr.ROBUSTNESS[0][1]).map(lambda v: "n/a" if abs(v) < 1e-9 else f"{v:+.3f}")
        robust["ROC-AUC"] = robust["ROC-AUC"].map(lambda v: f"{v:.3f}")
        st.markdown("**Robustness**")
        ui.show_table(robust)
    st.caption(f"Cutoff {charts.pct_label(THRESHOLD)}, chosen on out-of-fold training scores. In-sample ROC-AUC was {metrics['Train ROC-AUC (in-sample)']:.3f}, so the test figure is the honest one. "
               "The time split (0.715) is the more cautious number to quote for real use.")


# --------------------------------------------------------------------------- how it works tab
def render_how_it_works(models):
    st.subheader("How it works")
    tuned_line = ("The tuned champion is installed." if TUNED in models else "The tuned champion is not installed; this app ships XGBoost and LightGBM retrained on all loans with quick, untuned settings.")
    steps = [
        (1, "Data", "Bondora peer-to-peer loans.", "179,235 loans were reduced to 110,342 with a clear outcome and at least 12 months of history. 64% defaulted. Loans still running are not treated as good loans."),
        (2, "Cleaning and safety checks", "Inputs are reviewed before modelling.", "112 columns were reviewed. 37 post-start columns were removed to avoid leakage, along with 6 discontinued and 10 almost-empty fields. Gender and language are not model inputs."),
        (3, "Features", "The app turns application details into 26 model inputs.", "21 application fields plus five ratios. Numbers are clipped at the 0.5% and 99.5% quantiles and missing values are filled with the median; categories are one-hot encoded. A monthly payment of exactly 0 is treated as missing."),
        (4, "Models", "Several model families were compared before deployment.", f"Seven families were compared with five-fold cross-validation and the top two were tuned. Tuned XGBoost was the notebook champion. {tuned_line}"),
        (5, "Decision", "A probability becomes a practical flag.", "The app uses a 65% cutoff chosen on out-of-fold training scores. Per-country cutoffs are available as an experimental option."),
        (6, "Explanation", "The result includes model drivers and written rules.", "SHAP groups transformed columns back to readable features. Nine hand-written credit rules are combined into a scorecard and compared with the model. A constraint check shows where the model departs from basic credit logic."),
        (7, "Fairness check", "The notebook audited country, gender code and age.", "Country and Age remain model inputs. The fairness figures below are from the notebook and are not recomputed live."),
    ]
    ui.step_flow(steps)

    st.subheader("How good is it?")
    left, right = st.columns(2)
    with left:
        ui.surface("<div class='summary-label'>Held-out test ROC-AUC</div><div class='risk-number'>0.781</div><div class='muted'>95% interval: 0.775 to 0.787</div>")
    with right:
        ui.surface("<div class='summary-label'>Time-based ROC-AUC</div><div class='risk-number'>0.715</div><div class='muted'>Train through 2018, test from 2019</div>")
    st.write("")
    st.write("The second figure is the more cautious estimate for real use, because borrower behaviour changes over time. "
             "Precision is 80.4% and recall is 73.9% at the 65% cutoff. These numbers belong to the notebook's tuned champion; "
             "models retrained on all loans have no held-out score of their own.")

    st.subheader("Limits and fairness")
    st.caption("From the notebook's audit, not recomputed live")
    ui.show_chart(charts.fairness_chart())
    gap_before, gap_after = nr.FAIRNESS["fpr_gap"]
    acc_before, acc_after = nr.FAIRNESS["balanced_accuracy"]
    st.markdown(
        "- Gender is not an input, but disparities appear, mostly through country.\n"
        "- Gender code 2 is almost entirely Spanish applicants, so it is not a gender effect.\n"
        f"- Applicants aged 55 and over are wrongly flagged more often: {nr.AGE_FPR['55 and over']}% versus {nr.AGE_FPR['25 to 34']}% for ages 25 to 34. Removing Age does not close the gap.\n"
        "- Country and Age remain model inputs, and using country in a credit decision may be legally restricted.\n"
        f"- Per-country cutoffs narrowed the false-positive gap across Estonia, Spain and Finland from {gap_before} to {gap_after}, but lowered balanced accuracy from {acc_before} to {acc_after}, "
        f"and recall in Spain fell from {nr.FAIRNESS['recall_global'][2]}% to {nr.FAIRNESS['recall_per_country'][2]}%. This is a policy trade-off.\n"
        "- Slovakia (66 test loans) is too small to audit and is left out of the chart.")
    st.markdown("**Neurosymbolic layer**")
    st.markdown(
        f"- The nine written rules form a scorecard with ROC-AUC {nr.SCORECARD_TEST_AUC:.2f}, far below the model, so they explain rather than decide.\n"
        f"- {nr.MINED_RULES['rules']} IF-THEN rules mined from the champion reproduce {nr.MINED_RULES['fidelity']:.1%} of its decisions (always saying FLAG would give {nr.MINED_RULES['always_flag_baseline']:.1%}).\n"
        f"- Adding expert constraints to the champion removed most wrong-way score changes at a cost of {nr.TEST_METRICS['ROC-AUC'] - nr.MONOTONE_AUC:.4f} ROC-AUC.")
    ui.show_chart(charts.constraint_chart())
    st.warning("Research prototype. Not suitable for real lending decisions.")
    with st.expander("Technical details"):
        st.write("Libraries: Streamlit, pandas, NumPy, scikit-learn, XGBoost, LightGBM, SHAP and Plotly. "
                 "Retraining needs the full Bondora CSV: `python train_quick_model.py` for the quick models, `python train_champion.py` for the tuned champion.")


# --------------------------------------------------------------------------- main
models, model_messages = load_models()
if not models:
    st.error("No model artifacts could be loaded. Run `python train_quick_model.py` with the full Bondora CSV, or fix the messages below.")
    for _, message in model_messages:
        st.warning(message)
    st.stop()
sample_df = load_sample_data()
st.session_state.setdefault("stage", "landing")

if st.session_state["stage"] == "landing":
    render_landing()
else:
    st.markdown('<div class="topbar"><div class="wordmark">Bondora Default Risk</div></div>', unsafe_allow_html=True)
    _, top_right = st.columns([5, 1])
    with top_right:
        st.button("Back to start", on_click=start_over, key="back_to_start")
    assess_tab, explanation_tab, data_tab, models_tab, how_tab = st.tabs(["Assess", "Explanation", "Data and preprocessing", "Models", "How it works"])
    with assess_tab:
        render_assess(models, sample_df)
    with explanation_tab:
        render_explanation(models)
    with data_tab:
        render_data(models, sample_df)
    with models_tab:
        render_models(models)
    with how_tab:
        render_how_it_works(models)
    st.button("Start over", on_click=start_over, key="start_over")
