import plotly.graph_objects as go

import notebook_results as nr

TEAL = "#0F5C5C"
TEAL_SOFT = "#8FB3B3"
BRICK = "#B4452F"
GREEN = "#2F6B4F"
INK = "#1F2933"
MUTED = "#6B7280"
LINE = "#E7E2DA"
STONE = "#B7B0A6"


def _layout(title=None, height=300, legend=False):
    return dict(title=dict(text=title or "", font=dict(size=15)), height=height, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color=INK),
                margin=dict(l=12, r=12, t=42 if title else 12, b=18), showlegend=legend)


def pct_label(threshold):
    """65%, 74.8% and so on, without trailing zeros."""
    return f"{threshold * 100:g}%"


def risk_meter(probability, threshold):
    figure = go.Figure(go.Bar(x=[probability], y=["Applicant"], orientation="h",
                              marker_color=BRICK if probability >= threshold else GREEN,
                              hovertemplate="Probability: %{x:.1%}<extra></extra>"))
    layout = _layout(height=135)
    layout["margin"] = dict(l=12, r=12, t=34, b=18)
    figure.update_layout(**layout, xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor=LINE, title=None),
                         yaxis=dict(showticklabels=False),
                         shapes=[dict(type="line", x0=threshold, x1=threshold, y0=-0.5, y1=0.5, line=dict(color=TEAL, width=3))],
                         annotations=[dict(x=threshold, y=1.0, xref="x", yref="paper", text=f"{pct_label(threshold)} cutoff",
                                           showarrow=False, xanchor="left", xshift=6, yanchor="bottom", font=dict(color=TEAL, size=12))])
    return figure


def model_bars(probabilities, threshold, title="What each model says"):
    names = list(probabilities)
    values = list(probabilities.values())
    colors = [BRICK if value >= threshold else GREEN for value in values]
    figure = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors,
                              text=[f"{value:.1%}" for value in values], textposition="outside", cliponaxis=False,
                              hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    figure.update_layout(**_layout(title, height=max(170, 55 * len(names) + 50)),
                         xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor=LINE), yaxis=dict(autorange="reversed"),
                         shapes=[dict(type="line", x0=threshold, x1=threshold, y0=-0.5, y1=len(names) - 0.5,
                                      line=dict(color=TEAL, width=2, dash="dot"))])
    return figure


def signed_shap_bar(frame):
    frame = frame.sort_values("shap").copy()
    frame["axis_label"] = frame.apply(lambda row: f"{row['label']} ({row['display']})", axis=1)
    colors = [BRICK if value >= 0 else GREEN for value in frame["shap"]]
    figure = go.Figure(go.Bar(x=frame["shap"], y=frame["axis_label"], orientation="h", marker_color=colors,
                              hovertemplate="%{y}<br>SHAP value: %{x:.3f}<extra></extra>"))
    layout = _layout(height=max(300, 46 * len(frame)))
    layout["margin"] = dict(l=220, r=36, t=12, b=42)
    figure.update_layout(**layout, xaxis=dict(title="Pushes risk down     Pushes risk up", zeroline=True, zerolinecolor=LINE, gridcolor=LINE),
                         yaxis=dict(automargin=True, title=""))
    return figure


def rule_weights_bar(frame, labels):
    frame = frame.sort_values("weight")
    colors = [BRICK if value >= 0 else GREEN for value in frame["weight"]]
    figure = go.Figure(go.Bar(x=frame["weight"], y=[labels[name] for name in frame["rule"]], orientation="h", marker_color=colors,
                              text=[f"{value:+.2f}" for value in frame["weight"]], textposition="outside", cliponaxis=False,
                              hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
    figure.update_layout(**_layout(height=max(170, 50 * len(frame) + 40)),
                         xaxis=dict(title="Pushes risk down     Pushes risk up", zeroline=True, zerolinecolor=LINE, gridcolor=LINE),
                         yaxis=dict(automargin=True))
    return figure


def line_what_if(frame, current):
    figure = go.Figure(go.Scatter(x=frame["interest"], y=frame["probability"], mode="lines+markers",
                                  line=dict(color=TEAL, width=2), marker=dict(color=TEAL), name="Scenario",
                                  hovertemplate="Interest %{x:.1f}%<br>Probability %{y:.1%}<extra></extra>"))
    figure.add_scatter(x=[current["interest"]], y=[current["probability"]], mode="markers", marker=dict(size=11, color=BRICK), name="Current",
                       hovertemplate="Interest %{x:.1f}%<br>Probability %{y:.1%}<extra></extra>")
    figure.update_layout(**_layout("Probability as interest changes", height=270, legend=True), xaxis_title="Interest rate (%)",
                         yaxis_title="Probability", yaxis_tickformat=".0%", yaxis_gridcolor=LINE, xaxis_gridcolor=LINE,
                         legend=dict(orientation="h", y=1.15))
    return figure


def global_importance(frame, title="What matters most overall"):
    frame = frame.sort_values("importance")
    figure = go.Figure(go.Bar(x=frame["importance"], y=frame["label"], orientation="h", marker_color=TEAL,
                              text=[f"{value:.3f}" for value in frame["importance"]], textposition="outside", cliponaxis=False))
    figure.update_layout(**_layout(title, height=max(330, 35 * len(frame) + 50)), xaxis_title="Mean absolute contribution (log-odds)",
                         xaxis_gridcolor=LINE, yaxis=dict(automargin=True))
    return figure


def decile_chart():
    labels = ["Safest 10%", "2", "3", "4", "5", "6", "7", "8", "9", "Riskiest 10%"]
    figure = go.Figure(go.Bar(x=labels, y=nr.DECILES, marker_color=TEAL, text=[f"{rate:.1f}%" for rate in nr.DECILES],
                              textposition="outside", cliponaxis=False))
    figure.update_layout(**_layout("Observed default rate by risk decile", height=340),
                         yaxis=dict(range=[0, 108], ticksuffix="%", gridcolor=LINE), xaxis_title="Risk decile (1 = safest)")
    figure.update_xaxes(tickangle=-30)
    return figure


def cv_chart():
    """Cross-validated ROC-AUC for every model family the notebook compared."""
    names = list(nr.CV_TABLE)
    values = [nr.CV_TABLE[name][0] for name in names]
    colors = [TEAL if name == nr.CHAMPION else (STONE if name == "Logistic Regression" else TEAL_SOFT) for name in names]
    figure = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color=colors, text=[f"{value:.3f}" for value in values],
                              textposition="outside", cliponaxis=False,
                              error_x=dict(type="data", array=[nr.CV_TABLE[name][1] for name in names], color=MUTED, thickness=1.2, width=3),
                              hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    figure.update_layout(**_layout("Cross-validated ROC-AUC (training set, 5 folds)", height=420),
                         xaxis=dict(range=[0.70, 0.79], gridcolor=LINE), yaxis=dict(autorange="reversed", automargin=True))
    return figure


def confusion_chart():
    z = nr.CONFUSION_PCT
    figure = go.Figure(go.Heatmap(z=z, x=["Predicted repay", "Predicted default"], y=["Actually repaid", "Actually defaulted"],
                                  colorscale=[[0, "#F1EDE6"], [1, TEAL]], showscale=False, zmin=0, zmax=100,
                                  text=[[f"{v:.1f}%" for v in row] for row in z], texttemplate="%{text}",
                                  textfont=dict(size=20, color=INK), hoverinfo="skip"))
    figure.update_layout(**_layout("Share of each true outcome", height=280), yaxis=dict(autorange="reversed"))
    return figure


def fairness_chart():
    """False-positive rate by country: one global cutoff against a cutoff per country."""
    countries = nr.FAIRNESS["countries"]
    figure = go.Figure()
    figure.add_bar(name="One cutoff for everyone (65%)", x=nr.FAIRNESS["fpr_global"], y=countries, orientation="h", marker_color=BRICK,
                   text=[f"{v:.1f}%" for v in nr.FAIRNESS["fpr_global"]], textposition="outside", cliponaxis=False)
    figure.add_bar(name="Cutoff per country (experimental)", x=nr.FAIRNESS["fpr_per_country"], y=countries, orientation="h", marker_color=TEAL,
                   text=[f"{v:.1f}%" for v in nr.FAIRNESS["fpr_per_country"]], textposition="outside", cliponaxis=False)
    figure.update_layout(**_layout("Good borrowers wrongly flagged, by country", height=330, legend=True), barmode="group",
                         xaxis=dict(range=[0, 92], ticksuffix="%", gridcolor=LINE), yaxis=dict(autorange="reversed"),
                         legend=dict(orientation="h", y=-0.18))
    return figure


def constraint_chart():
    labels = [row[0] for row in nr.CONSTRAINT_VIOLATIONS]
    figure = go.Figure()
    figure.add_bar(name="Notebook champion", x=labels, y=[row[1] * 100 for row in nr.CONSTRAINT_VIOLATIONS], marker_color=BRICK,
                   text=[f"{row[1] * 100:.1f}%" for row in nr.CONSTRAINT_VIOLATIONS], textposition="outside", cliponaxis=False)
    figure.add_bar(name="With expert constraints", x=labels, y=[row[2] * 100 for row in nr.CONSTRAINT_VIOLATIONS], marker_color=TEAL,
                   text=[f"{row[2] * 100:.1f}%" for row in nr.CONSTRAINT_VIOLATIONS], textposition="outside", cliponaxis=False)
    figure.update_layout(**_layout("Applicants whose score moves the wrong way", height=320, legend=True), barmode="group",
                         yaxis=dict(range=[0, 52], ticksuffix="%", gridcolor=LINE), legend=dict(orientation="h", y=-0.2))
    return figure
