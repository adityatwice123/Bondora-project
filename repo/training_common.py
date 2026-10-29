"""Data preparation shared by the training scripts. Mirrors notebook sections 1.3 to 1.7."""
import pandas as pd

import loan_utils as lu


def load_modelling_data(path):
    """Return features X, target y (1 = default) and the loan date / country for each modelling loan.

    Keeps loans with a final outcome (defaulted, or repaid with no default) that were issued at least
    12 months before the snapshot, so young loans do not look artificially safe.
    """
    raw = pd.read_csv(path, low_memory=False)
    raw["LoanDate"] = pd.to_datetime(raw["LoanDate"])
    raw["DefaultFlag"] = raw["DefaultDate"].notna().astype(int)
    report_date = pd.to_datetime(raw["ReportAsOfEOD"].iloc[0])
    cutoff = report_date - pd.DateOffset(months=12)
    resolved = (raw["DefaultFlag"] == 1) | (raw["Status"] == "Repaid")
    seasoned = raw["LoanDate"] <= cutoff
    frame = raw.loc[resolved & seasoned].copy()
    print(f"{len(raw):,} loans -> {len(frame):,} with a final outcome and 12+ months of history "
          f"({frame['DefaultFlag'].mean():.1%} defaulted)")
    return lu.engineer_features(frame), frame["DefaultFlag"].astype(int), frame[["LoanDate", "Country"]]
