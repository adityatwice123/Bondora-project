"""Build the 5,000-loan sample the app draws from, using the full Bondora CSV.

    python make_sample.py --data LoanData_Bondora.csv --n 5000

The sample is stratified by country (close to the test set: about 52% EE, 26% FI, 21% ES) and then drawn at random
across all issue years, so it is not just the oldest loans. Only the columns the app needs are kept, which keeps
the file small and leaves out user names and other identifiers.
"""
import argparse

import pandas as pd

import loan_utils as lu

SHARES = {"EE": 0.52, "FI": 0.26, "ES": 0.21, "SK": 0.01}
KEEP = lu.RAW_INPUT_COLUMNS + ["LoanId", "LoanDate", "DefaultDate", "Status", "ReportAsOfEOD"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="LoanData_Bondora.csv")
    parser.add_argument("--out", default="LoanData_Sample.csv")
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args()

    raw = pd.read_csv(args.data, low_memory=False)
    raw["LoanDate"] = pd.to_datetime(raw["LoanDate"])
    report_date = pd.to_datetime(raw["ReportAsOfEOD"].iloc[0])
    resolved = raw["DefaultDate"].notna() | (raw["Status"] == "Repaid")
    seasoned = raw["LoanDate"] <= report_date - pd.DateOffset(months=12)
    pool = raw.loc[resolved & seasoned]
    parts = []
    for country, share in SHARES.items():
        rows = pool.loc[pool["Country"] == country]
        parts.append(rows.sample(min(len(rows), round(args.n * share)), random_state=42))
    sample = pd.concat(parts).sample(frac=1, random_state=42)[KEEP]
    sample.to_csv(args.out, index=False)
    print(f"Wrote {len(sample):,} loans to {args.out}")
    print(sample["Country"].value_counts().to_dict(), "| issue years", sample["LoanDate"].dt.year.min(), "to", sample["LoanDate"].dt.year.max())


if __name__ == "__main__":
    main()
