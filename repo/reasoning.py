"""Neurosymbolic helpers: the written-rule scorecard and the expert-constraint check.

Both mirror notebook section 3.3. The scorecard weights are the ones the notebook fitted on its
training split; the app only applies them, it does not refit anything.
"""
import numpy as np
import pandas as pd

import loan_utils as lu
import notebook_results as nr


def scorecard(predicates):
    """Probability from the nine written rules: sigmoid(bias + sum of weights of rules that fire).

    `predicates` is a Series of 0/1 values indexed by rule name (see lu.symbolic_predicates).
    Returns the probability and a frame of the rules that fired with their log-odds weights.
    """
    fired = [(name, nr.SCORECARD_WEIGHTS[name]) for name, on in predicates.items() if int(on) == 1]
    logit = nr.SCORECARD_BIAS + sum(weight for _, weight in fired)
    prob = float(1.0 / (1.0 + np.exp(-logit)))
    frame = pd.DataFrame(fired, columns=["rule", "weight"]).sort_values("weight", ascending=False, key=abs)
    return prob, frame.reset_index(drop=True)


def agreement(model_flag, rules_flag):
    """Notebook section 3.3.6: do the learned model and the written rules tell the same story?"""
    if model_flag and rules_flag:
        return "Consensus: default", False
    if not model_flag and not rules_flag:
        return "Consensus: repay", False
    if model_flag:
        return "Conflict: the model flags it, the rules would not", True
    return "Conflict: the rules flag it, the model would not", True


CONSTRAINTS = [
    # label, expected direction (+1 means risk should not fall), field, change
    ("Interest rate +5 points", +1, "Interest", lambda v: v + 5.0),
    ("Loan term +6 months (max 60)", +1, "LoanDuration", lambda v: min(v + 6, lu.MAX_TERM)),
    ("Monthly income +20%", -1, "IncomeTotal", lambda v: v * 1.2),
    ("Monthly liabilities +20%", +1, "LiabilitiesTotal", lambda v: v * 1.2),
    ("Earlier repayments +500", -1, "PreviousRepaymentsBeforeLoan", lambda v: v + 500.0),
]


def constraint_check(model, raw_input, eps=0.005):
    """Change one input at a time and see whether the score moves the way an analyst expects.

    A constraint is broken when the score moves the wrong way by more than half a percentage
    point. Ratios are rebuilt from the changed raw fields, as in notebook section 3.3.5.
    """
    base = raw_input.iloc[0]
    base_prob = float(model.predict_proba(lu.engineer_features(raw_input))[0, 1])
    rows = []
    for label, sign, field, change in CONSTRAINTS:
        changed = base.copy()
        changed[field] = change(float(changed[field]))
        prob = float(model.predict_proba(lu.engineer_features(pd.DataFrame([changed])))[0, 1])
        delta = prob - base_prob
        rows.append({
            "Change": label,
            "Analyst expects": "Risk should not fall" if sign > 0 else "Risk should not rise",
            "Score moves by": delta,
            "Result": "Broken" if sign * delta < -eps else "Respected",
        })
    return pd.DataFrame(rows)
