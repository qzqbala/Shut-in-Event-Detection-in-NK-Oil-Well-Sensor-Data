"""Streamlit dashboard: pick a well, a parameter and a period; see the signal with
detected shut-ins, the event table and the method comparison.

Run:  streamlit run app.py   (run `python main.py --all` once before)
"""
import pandas as pd
import streamlit as st

from src import viz
from src.config import DATA_PROCESSED, FOUNTAIN_PARAMS, SRP_PARAMS, TABLES

st.set_page_config(page_title="NK wells - shut-in detection", layout="wide")


@st.cache_data
def load():
    grid = pd.read_parquet(DATA_PROCESSED / "hourly_wide.parquet")
    events = pd.read_parquet(DATA_PROCESSED / "shut_in_events.parquet")
    metrics = pd.read_csv(TABLES / "method_metrics_test.csv")
    return grid, events, metrics


try:
    grid, events, metrics = load()
except FileNotFoundError:
    st.error("Processed data not found. Run `python main.py --all` first.")
    st.stop()

st.title("NK oil wells — shut-in event detection")

# ---- filters (one row)
c1, c2, c3, c4 = st.columns([1, 1.3, 1.2, 2])
well = c1.selectbox("Well", sorted(grid["well_id"].unique()))
wtype = grid.loc[grid["well_id"] == well, "well_type"].iloc[0]
params = FOUNTAIN_PARAMS if wtype == "fountain" else SRP_PARAMS + ["area_norm"]
params = [p for p in params if grid.loc[grid["well_id"] == well, p].notna().any()]
param = c2.selectbox("Parameter", params)
method = c3.selectbox("Shut-in method", ["reference", "rule_based", "statistical", "baseline"])
dmin, dmax = grid["timestamp"].min().date(), grid["timestamp"].max().date()
period = c4.date_input("Period", (pd.Timestamp("2026-05-01").date(), pd.Timestamp("2026-06-30").date()),
                       min_value=dmin, max_value=dmax)
start, end = (period if isinstance(period, tuple) and len(period) == 2 else (dmin, dmax))
end_ts = pd.Timestamp(end) + pd.Timedelta(hours=23)

st.caption(f"Well type: **{wtype}** · test split starts 2026-08-08 23:00")

# ---- KPIs for the selection
sel = events[(events["well_id"] == well) & (events["method"] == method)
             & (events["start_ts"] <= end_ts) & (events["end_ts"] >= pd.Timestamp(start))]
g = grid[(grid["well_id"] == well) & grid["timestamp"].between(pd.Timestamp(start), end_ts)]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Shut-in events", len(sel))
k2.metric("Downtime, h", int(sel["shut_in_h"].sum()))
k3.metric("Mean duration, h", f"{sel['duration_h'].mean():.1f}" if len(sel) else "—")
k4.metric("Hours without data", int((~g["has_data"].astype(bool)).sum()))

st.plotly_chart(viz.timeline(grid, events, well, param, start, end_ts, method),
                width="stretch")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Events in the period")
    st.dataframe(sel[["event_id", "start_ts", "end_ts", "duration_h", "shut_in_h", "split"]],
                 hide_index=True, width="stretch")
with right:
    st.subheader("Method comparison (test split)")
    st.plotly_chart(viz.method_comparison(metrics), width="stretch")

with st.expander("Metric table"):
    st.dataframe(metrics.round(3), hide_index=True, width="stretch")
