"""Exploratory analysis by well type: descriptive statistics, daily (hour-of-day) patterns,
parameter correlations and shut-in frequency / duration statistics."""
import pandas as pd

from .config import FOUNTAIN, FOUNTAIN_PARAMS, SRP, SRP_PARAMS
from .detection import primary_signal


def params_for(well_type: str) -> list:
    return FOUNTAIN_PARAMS if well_type == FOUNTAIN else SRP_PARAMS


def describe_by_type(grid: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics of every parameter, per well type (hours with data only).
    Raw dynamometer_area is replaced by the scale-normalised area_norm."""
    frames = []
    for wt, g in grid[grid["has_data"]].groupby("well_type"):
        cols = [c if c != "dynamometer_area" else "area_norm" for c in params_for(wt)]
        d = g[cols].describe(percentiles=[0.05, 0.5, 0.95]).T
        d.insert(0, "well_type", wt)
        frames.append(d)
    out = pd.concat(frames)
    out.index.name = "parameter"
    return out.reset_index().round(3)


def describe_by_well(grid: pd.DataFrame) -> pd.DataFrame:
    g = grid[grid["has_data"]].copy()
    g["primary"] = primary_signal(g)
    return (g.groupby(["well_type", "well_id"])["primary"]
             .agg(["count", "mean", "median", "std", "min", "max"]).round(3).reset_index())


def daily_profile(grid: pd.DataFrame, running_only: bool = True) -> pd.DataFrame:
    """Mean primary signal by hour of day, normalised by each well's overall mean
    (1.0 = the well's average), then averaged over wells of the same type."""
    g = grid[grid["has_data"]].copy()
    g["primary"] = primary_signal(g)
    if running_only:
        g = g[g["primary"] > 0]
    g["rel"] = g["primary"] / g.groupby("well_id")["primary"].transform("mean")
    per_well = g.groupby(["well_type", "well_id", "hour"])["rel"].mean().reset_index()
    return per_well.groupby(["well_type", "hour"])["rel"].agg(["mean", "std"]).reset_index()


def correlations(grid: pd.DataFrame, well_type: str, method: str = "spearman") -> pd.DataFrame:
    """Rank correlation between parameters of one well type (running hours only)."""
    g = grid[grid["has_data"] & (grid["well_type"] == well_type)]
    g = g[primary_signal(g) > 0]
    cols = [c if c != "dynamometer_area" else "area_norm" for c in params_for(well_type)]
    cols = [c for c in cols if g[c].notna().sum() > 100]
    return g[cols].corr(method=method).round(3)


def event_stats(events: pd.DataFrame, grid: pd.DataFrame, method: str = "reference") -> pd.DataFrame:
    """Frequency and duration of shut-ins per well."""
    ev = events[events["method"] == method]
    hours = grid.groupby("well_id")["has_data"].sum().rename("hours_with_data")
    s = ev.groupby(["well_type", "well_id"]).agg(
        n_events=("duration_h", "count"), total_h=("shut_in_h", "sum"), mean_h=("duration_h", "mean"),
        median_h=("duration_h", "median"), max_h=("duration_h", "max")).reset_index()
    s = s.merge(hours.reset_index(), on="well_id", how="right")
    s["well_type"] = s["well_type"].fillna(
        s["well_id"].map(grid.drop_duplicates("well_id").set_index("well_id")["well_type"]))
    s = s.fillna({"n_events": 0, "total_h": 0})
    s["events_per_30d"] = (s["n_events"] / (s["hours_with_data"] / 720)).round(2)
    s["downtime_pct"] = (100 * s["total_h"] / s["hours_with_data"]).round(2)
    return s.sort_values(["well_type", "well_id"]).round(2)


def duration_buckets(events: pd.DataFrame, method: str = "reference") -> pd.DataFrame:
    ev = events[events["method"] == method]
    bins = [0, 1, 3, 12, 24, 72, 10_000]
    labels = ["1 h", "2-3 h", "4-12 h", "13-24 h", "1-3 days", "> 3 days"]
    b = pd.cut(ev["duration_h"], bins=bins, labels=labels)
    return (ev.assign(bucket=b).groupby(["well_type", "bucket"], observed=False).size()
              .rename("n_events").reset_index())


def monthly_events(events: pd.DataFrame, method: str = "reference") -> pd.DataFrame:
    ev = events[events["method"] == method].copy()
    ev["month"] = ev["start_ts"].dt.to_period("M").astype(str)
    return (ev.groupby(["month", "well_type"]).agg(n_events=("event_id", "count"),
                                                  downtime_h=("shut_in_h", "sum"))
              .reset_index())
