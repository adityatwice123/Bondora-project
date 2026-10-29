"""End-to-end checks that drive the Streamlit app without a browser."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    return at


def _start(at):
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _submit(at):
    [b for b in at.button if b.label == "Assess this applicant"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_landing_then_tabs(app):
    assert not app.exception
    _start(app)
    assert [t.label for t in app.tabs] == ["Assess", "Explanation", "Models", "How it works"]


def test_custom_applicant_end_to_end(app):
    _submit(_start(app))
    result = app.session_state["last_result"]
    assert result["probabilities"] and all(0 <= p <= 1 for p in result["probabilities"].values())
    assert any(m.label == "Scenario probability" for m in app.metric)
    assert not [e for e in app.error]


def test_random_sample_applicant_end_to_end(app):
    _start(app)
    app.session_state["input_mode"] = "Random from the Bondora sample"
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    _submit(app)
    result = app.session_state["last_result"]
    assert result["random_mode"] is True and len(result["probabilities"]) >= 2


def test_bureau_scores_lock_outside_their_country(app):
    _start(app)
    app.selectbox(key="custom_0_country").select("ES").run()
    ee = app.selectbox(key="custom_0_ee_score_ES")
    fi = app.selectbox(key="custom_0_fi_score_ES")
    assert ee.options == ["Missing"] and ee.disabled
    assert fi.options == ["Missing"] and fi.disabled
    app.selectbox(key="custom_0_country").select("EE").run()
    assert not app.selectbox(key="custom_0_ee_score_EE").disabled
    assert app.selectbox(key="custom_0_fi_score_EE").disabled


def test_what_if_at_unchanged_sliders_matches_assessment(app):
    _submit(_start(app))
    result = app.session_state["last_result"]
    name = "XGBoost (tuned)" if "XGBoost (tuned)" in result["probabilities"] else "XGBoost"
    expected = result["probabilities"][name]
    metric = [m for m in app.metric if m.label == "Scenario probability"][0]
    assert metric.value == f"{expected:.1%}"


def test_country_cutoff_label_follows_threshold(app):
    _start(app)
    app.selectbox(key="custom_0_country").select("ES").run()
    app.checkbox(key="country_thresholds").check().run()
    _submit(app)
    text = " ".join(c.value for c in app.caption)
    assert "82% or above" in text and "experimental cutoff for ES" in text


def test_draw_another_keeps_the_new_applicant(app):
    _start(app)
    app.session_state["input_mode"] = "Random from the Bondora sample"
    app.run()
    seen = {app.session_state["random_row"]["LoanId"]}
    for _ in range(6):
        app.button(key="draw_another").click().run()
        assert not app.exception, [e.value for e in app.exception]
        seen.add(app.session_state["random_row"]["LoanId"])
    assert len(seen) > 1  # a fresh applicant was drawn and kept
