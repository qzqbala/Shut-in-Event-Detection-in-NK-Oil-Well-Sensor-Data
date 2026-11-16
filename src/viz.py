"""Plotly figures (interactive HTML + static PNG for the report)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import FIGURES, FOUNTAIN, SRP

# categorical slots (validated reference palette), assigned by entity, never by rank
TYPE_COLORS = {FOUNTAIN: "#2a78d6", SRP: "#eb6834"}
METHOD_COLORS = {"baseline": "#8a8984", "rule_based": "#2a78d6", "statistical": "#eb6834",
                 "reference": "#1baf7a"}
SHUTIN_FILL = "rgba(227, 73, 72, 0.18)"
TEXT = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e8e7e3"

LAYOUT = dict(template="plotly_white", font=dict(family="Inter, Arial, sans-serif", size=13, color=TEXT),
              margin=dict(l=60, r=30, t=70, b=50), hovermode="closest",
              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))


def _style(fig, title, x_title=None, y_title=None, height=450):
    fig.update_layout(title=dict(text=title, x=0, xanchor="left"), height=height, **LAYOUT)
    fig.update_xaxes(title=x_title, gridcolor=GRID, linecolor=GRID, title_font_color=MUTED)
    fig.update_yaxes(title=y_title, gridcolor=GRID, linecolor=GRID, title_font_color=MUTED,
                     zeroline=False)
    return fig


def save(fig, name: str, png: bool = True) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIGURES / f"{name}.html", include_plotlyjs="cdn")
    if png:
        try:
            fig.write_image(FIGURES / f"{name}.png", width=1100, height=fig.layout.height or 450, scale=2)
        except Exception as err:   # kaleido / Chrome not installed -> HTML only
            print(f"PNG export skipped for {name}: {err}")


def timeline(grid, events, well_id, parameter, start=None, end=None, method="reference"):
    """Interactive time series of one parameter with shaded shut-in events."""
    g = grid[grid["well_id"] == well_id]
    if start is not None:
        g = g[g["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        g = g[g["timestamp"] <= pd.Timestamp(end)]
    color = TYPE_COLORS.get(g["well_type"].iloc[0], "#2a78d6") if len(g) else "#2a78d6"
    fig = go.Figure(go.Scatter(x=g["timestamp"], y=g[parameter], mode="lines", name=parameter,
                               line=dict(width=1.5, color=color), connectgaps=False,
                               hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra></extra>"))
    ev = events[(events["well_id"] == well_id) & (events["method"] == method)]
    if len(g):
        ev = ev[(ev["end_ts"] >= g["timestamp"].min()) & (ev["start_ts"] <= g["timestamp"].max())]
    for _, e in ev.iterrows():
        fig.add_vrect(x0=e["start_ts"], x1=e["end_ts"] + pd.Timedelta(hours=1),
                      fillcolor=SHUTIN_FILL, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"shut-in ({method})",
                             marker=dict(size=12, symbol="square", color=SHUTIN_FILL)))
    _style(fig, f"{well_id}: {parameter} with {len(ev)} shut-in events ({method})",
           None, parameter, height=420)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    return fig


def daily_pattern(profile: pd.DataFrame):
    fig = go.Figure()
    for wt, g in profile.groupby("well_type"):
        fig.add_trace(go.Scatter(x=g["hour"], y=g["mean"], mode="lines+markers", name=wt,
                                 line=dict(width=2, color=TYPE_COLORS[wt]), marker=dict(size=8),
                                 hovertemplate="hour %{x}: %{y:.3f}<extra>" + wt + "</extra>"))
    fig.add_hline(y=1, line=dict(color=MUTED, width=1, dash="dot"))
    _style(fig, "Daily pattern: primary signal by hour of day (1.0 = well's own mean, running hours)",
           "Hour of day", "Relative level")
    fig.update_xaxes(dtick=2)
    return fig


def correlation_heatmap(corr_f: pd.DataFrame, corr_s: pd.DataFrame):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.18,
                        subplot_titles=("Fountain wells", "Sucker-rod pump wells"))
    for col, c in enumerate([corr_f, corr_s], start=1):
        fig.add_trace(go.Heatmap(z=c.values, x=c.columns, y=c.index, zmin=-1, zmax=1,
                                 colorscale=[[0, "#2a78d6"], [0.5, "#f1f0ed"], [1, "#e34948"]],
                                 text=c.values.round(2), texttemplate="%{text}",
                                 showscale=(col == 2), colorbar=dict(title="Spearman ρ")),
                      row=1, col=col)
    _style(fig, "Parameter correlations by well type (Spearman, running hours)", height=480)
    fig.update_yaxes(autorange="reversed")
    return fig


def duration_box(events: pd.DataFrame, method: str = "reference"):
    ev = events[events["method"] == method]
    fig = go.Figure()
    for wt, g in ev.groupby("well_type"):
        fig.add_trace(go.Box(x=g["well_id"], y=g["duration_h"], name=wt, boxpoints="all",
                             jitter=0.4, pointpos=0, marker=dict(size=6, color=TYPE_COLORS[wt]),
                             line=dict(width=1.5, color=TYPE_COLORS[wt]), fillcolor="rgba(0,0,0,0)",
                             hovertemplate="%{x}: %{y} h<extra></extra>"))
    _style(fig, "Shut-in duration per well (reference events, log scale)", "Well", "Duration, hours")
    fig.update_yaxes(type="log")
    return fig


def monthly_frequency(monthly: pd.DataFrame):
    fig = go.Figure()
    for wt, g in monthly.groupby("well_type"):
        fig.add_trace(go.Bar(x=g["month"], y=g["n_events"], name=wt, marker_color=TYPE_COLORS[wt],
                             customdata=g["downtime_h"],
                             hovertemplate="%{x}: %{y} events, %{customdata} h downtime<extra>"
                             + wt + "</extra>"))
    _style(fig, "Shut-in events per month by well type (reference)", "Month", "Number of events")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    return fig


def method_comparison(metrics: pd.DataFrame):
    cols = {"hour_f1": "Hour F1", "event_precision": "Event precision",
            "event_recall": "Event recall", "event_f1": "Event F1"}
    fig = go.Figure()
    for _, r in metrics.iterrows():
        fig.add_trace(go.Bar(x=list(cols.values()), y=[r[c] for c in cols], name=r["method"],
                             marker_color=METHOD_COLORS[r["method"]],
                             text=[f"{r[c]:.2f}" for c in cols], textposition="outside",
                             hovertemplate="%{x}: %{y:.3f}<extra>" + r["method"] + "</extra>"))
    _style(fig, "Detector comparison on the test split (chronological, fitted on train)",
           None, "Score", height=460)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(range=[0, 1.12])
    return fig


def area_shift(grid: pd.DataFrame, well_id: str = "NK-68"):
    g = grid[(grid["well_id"] == well_id) & grid["has_data"]].set_index("timestamp")
    daily = g[["dynamometer_area", "area_norm"]].resample("D").median()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=("Raw dynamometer_area (daily median)",
                                        "Normalised per scale period (area / train median of period)"))
    fig.add_trace(go.Scatter(x=daily.index, y=daily["dynamometer_area"], mode="lines",
                             line=dict(width=2, color="#eb6834"), name="raw"), row=1, col=1)
    fig.add_trace(go.Scatter(x=daily.index, y=daily["area_norm"], mode="lines",
                             line=dict(width=2, color="#2a78d6"), name="normalised"), row=2, col=1)
    _style(fig, f"{well_id}: dynamometer_area scale shift in May 2026 and its correction", height=520)
    return fig


def missing_heatmap(grid: pd.DataFrame):
    g = grid.copy()
    g["day"] = g["timestamp"].dt.floor("D")
    cov = g.groupby(["well_id", "day"])["has_data"].mean().unstack()
    fig = go.Figure(go.Heatmap(z=cov.values * 100, x=cov.columns, y=cov.index,
                               colorscale=[[0, "#e34948"], [0.5, "#f5c48a"], [1, "#f1f0ed"]],
                               zmin=0, zmax=100, colorbar=dict(title="% hours"),
                               hovertemplate="%{y} %{x|%Y-%m-%d}: %{z:.0f}% hours with data<extra></extra>"))
    _style(fig, "Data coverage: share of hours with any record, per well and day", height=420)
    return fig
