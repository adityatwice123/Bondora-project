# Bondora Credit Risk Platform

A Streamlit research prototype that scores the default risk of a Bondora peer-to-peer loan and explains the score.
You can enter an applicant yourself or draw a real loan from a sample, compare the models side by side, see which
factors pushed the score up or down, check a hand-written credit rulebook against the model, and read the
notebook's results and fairness audit.

> Research prototype built on public Bondora data. Not for real lending decisions.

## What the app shows

| Tab | What you get |
|---|---|
| **Assess** | Custom applicant or random real loan from the sample. Risk score, cutoff meter, every model side by side, a second opinion from nine written rules, where the score sits in the sample, CSV download. For sample loans you can reveal the recorded outcome. |
| **Explanation** | SHAP drivers for this applicant, which rules of thumb apply and their weights, a check of whether the model follows basic credit logic, a what-if explorer, overall importance, and the notebook's risk-decile result. |
| **Models** | The models installed, a live comparison on the sample, the notebook's comparison of seven model families (cross-validated and test scores), and the champion's held-out results with a confusion matrix and robustness table. |
| **How it works** | The pipeline step by step, how good it is (0.781 held-out, 0.715 on a time split), the fairness audit by country and age, and the neurosymbolic results. |

## Run it

Python 3.11 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. On share.streamlit.io choose **New app**, pick the repo, branch `main`, and set the main file to `app.py`.
3. Under **Advanced settings** choose Python 3.11 or newer, then deploy.

## What is in the repo

```
app.py                 Streamlit app (four tabs)
charts.py              Plotly charts
components.py          Reusable UI pieces
explain.py             SHAP, plain-English labels, country-balanced sample
reasoning.py           Written-rule scorecard and expert-constraint check
loan_utils.py          Feature engineering, rulebook, payment re-estimate
notebook_results.py    Numbers recorded in the notebook (shown on the Models and How it works tabs)
styles.py              Page styling
train_quick_model.py   Trains the quick models the app ships with
train_champion.py      Rebuilds the notebook's tuned XGBoost champion
make_sample.py         Builds a representative sample from the full CSV
training_common.py     Data preparation shared by the training scripts
artifacts/             Shipped model pipelines
LoanData_Sample.csv    5,000 real loans (only the columns the app needs)
Bondora_Risk_NeSy_XAI.ipynb   The research notebook with all outputs
tests/                 Unit tests and end-to-end app tests
```

## Models: read this before quoting a number

- **The shipped models are quick and untuned.** `artifacts/xgboost_pipeline.joblib` (100 trees, depth 6) and
  `artifacts/lightgbm_pipeline.joblib` (150 trees, depth 10) were refit on **all** resolved loans, so they have no held-out score of their own.
- **The headline numbers belong to the notebook's tuned champion**: test ROC-AUC 0.781 (95% interval 0.775 to 0.787),
  precision 80.4% and recall 73.9% at the 0.65 cutoff, and 0.715 ROC-AUC on a time split (train to 2018, test 2019 onward). The app labels them as notebook results.
- **To use the tuned champion in the app**, download the full Bondora CSV, then run:
  ```bash
  python train_champion.py --data LoanData_Bondora.csv
  ```
  This writes `artifacts/xgboost_tuned_pipeline.joblib`. The app finds it on the next start and makes it the default model.
- Sample loans were part of the training data, so scores on them (and "Reveal what actually happened") are optimistic.
- The shipped sample holds loans issued 2009 to 2016 and is mostly Estonian. Run `python make_sample.py --data LoanData_Bondora.csv`
  for a sample stratified by country and spread across all issue years. The overall-importance chart uses a country-balanced draw from whatever sample is present.

## Retrain

```bash
python train_quick_model.py --data LoanData_Bondora.csv   # quick models, writes artifacts/metrics.json
python train_champion.py   --data LoanData_Bondora.csv    # tuned champion, about 5 to 10 minutes
```

The model files are pickles. If you change the pinned scikit-learn, XGBoost or LightGBM versions, rerun `train_quick_model.py`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The tests cover the rulebook scorecard (it reproduces the three applicants printed in the notebook), payment re-estimation,
labels, the sample and the shipped models, and drive the whole app without a browser: landing page, custom and random
applicants, bureau-score locking by country, the what-if explorer, and per-country cutoffs.

## Limits and fairness

- Country and Age are model inputs. Using country in a credit decision may be legally restricted.
- Gender is not an input, yet disparities appear, mostly through country. The notebook's audit found 89.6% of Spanish applicants flagged against 36.2% of Estonians.
- Applicants aged 55 and over are wrongly flagged more often (45.4% against 26.8% for ages 25 to 34). Removing Age did not close the gap.
- Per-country cutoffs (experimental toggle) narrowed the false-positive gap across Estonia, Spain and Finland from 0.583 to 0.020 but cut balanced accuracy from 0.709 to 0.665.
  The cutoffs were tuned on the notebook champion, so they are only a rough guide for the quick models.
- The 64% default rate comes from loans with a final outcome. It is not the platform-wide default rate.

## Put it on GitHub

```bash
git init
git add .
git commit -m "Bondora credit risk platform: assessment, explanations, model comparison, fairness"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
