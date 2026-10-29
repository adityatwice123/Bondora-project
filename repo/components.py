import streamlit as st

import charts

CHART_CONFIG = {"displayModeBar": False}


def show_chart(figure):
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)


def show_table(frame, **kwargs):
    st.dataframe(frame, hide_index=True, width="stretch", **kwargs)


def section_title(title, copy=None):
    st.subheader(title)
    if copy:
        st.caption(copy)


def label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def surface(content, class_name="surface-card"):
    st.markdown(f'<div class="{class_name}">{content}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note-card">{text}</div>', unsafe_allow_html=True)


def chip(text, kind="info"):
    return f'<span class="verdict-chip chip-{kind}">{text}</span>'


def stat_cards(items):
    """items: list of (value, label). Rendered as a row of cards."""
    columns = st.columns(len(items))
    for column, (value, text) in zip(columns, items):
        with column:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{value}</div><div class="stat-label">{text}</div></div>', unsafe_allow_html=True)


def step_flow(steps):
    for number, title, copy, details in steps:
        st.markdown(f'<div class="flow-line"><span class="flow-dot">{number}</span><span class="flow-title">{title}</span><p class="flow-copy">{copy}</p></div>', unsafe_allow_html=True)
        with st.expander("Technical details", expanded=False):
            st.write(details)


def risk_result(probability, threshold):
    high = probability >= threshold
    color = "risk-high" if high else "risk-low"
    verdict = "Flagged as higher risk" if high else "Lower risk"
    st.markdown(f'<div class="risk-number {color}">{probability:.1%}</div><div class="risk-verdict {color}">{verdict}</div>', unsafe_allow_html=True)


def outcome_card(row):
    issued = str(row.get("LoanDate", ""))[:4] or "n/a"
    fields = [
        ("Age", f"{int(row.get('Age', 0))}"),
        ("Country", str(row.get("Country", "Unknown"))),
        ("Loan amount", f"€{float(row.get('Amount', 0)):,.0f}"),
        ("Interest", f"{float(row.get('Interest', 0)):.1f}%"),
        ("Duration", f"{int(row.get('LoanDuration', 0))} months"),
        ("Monthly income", f"€{float(row.get('IncomeTotal', 0)):,.0f}"),
        ("Customer", "New" if row.get("NewCreditCustomer") == 1 else "Returning"),
        ("Issued", issued),
    ]
    columns = st.columns(len(fields))
    for column, (text, value) in zip(columns, fields):
        with column:
            st.markdown(f'<div class="summary-label">{text}</div><div class="summary-value">{value}</div>', unsafe_allow_html=True)
