"""Data quality audit: missing hours, missing parameters, quality flags,
common outages, duplicates, and checks of the known dataset issues."""
import pandas as pd

from .config import FOUNTAIN, FOUNTAIN_PARAMS, NK7_REGIME_START, NK7_WELL, SRP_PARAMS


def basic_overview(long_df: pd.DataFrame) -> dict:
    return {
        "rows_long": int(len(long_df)),
        "wells": int(long_df["well_id"].nunique()),
        "parameters": int(long_df["parameter"].nunique()),
        "period_start": str(long_df["timestamp"].min()),
        "period_end": str(long_df["timestamp"].max()),
        "rows_train": int((long_df["split"] == "train").sum()),
        "rows_test": int((long_df["split"] == "test").sum()),
        "duplicate_rows": int(long_df.duplicated(["well_id", "timestamp", "parameter"]).sum()),
        "null_values": int(long_df["value"].isna().sum()),
    }


def flag_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    f = long_df[long_df["flag"].astype(str) != ""]
    return (f.groupby(["well_id", "parameter", "flag"], observed=True).size()
             .rename("rows").reset_index().sort_values("rows", ascending=False))


def missing_hours(grid: pd.DataFrame) -> pd.DataFrame:
    """Share of hours without any record, per well."""
    g = grid.groupby("well_id")["has_data"].agg(total_hours="size", hours_with_data="sum")
    g["missing_hours"] = g["total_hours"] - g["hours_with_data"]
    g["missing_pct"] = (100 * g["missing_hours"] / g["total_hours"]).round(2)
    return g.reset_index()


def missing_parameters(grid: pd.DataFrame) -> pd.DataFrame:
    """Within hours that have data: share of each expected parameter missing."""
    rows = []
    present = grid[grid["has_data"]]
    for well, g in present.groupby("well_id"):
        params = FOUNTAIN_PARAMS if g["well_type"].iloc[0] == FOUNTAIN else SRP_PARAMS
        for p in params:
            if g[p].notna().sum() == 0:
                continue  # parameter not measured at this well (e.g. NK-76 casing)
            rows.append({"well_id": well, "parameter": p,
                         "missing_pct": round(100 * g[p].isna().mean(), 2)})
    return pd.DataFrame(rows).pivot(index="well_id", columns="parameter", values="missing_pct")


def gap_runs(grid: pd.DataFrame, min_hours: int = 6) -> pd.DataFrame:
    """Consecutive missing-hour runs (data outages) of at least `min_hours`."""
    rows = []
    for well, g in grid.groupby("well_id"):
        miss = ~g["has_data"].to_numpy()
        ts = g["timestamp"].to_numpy()
        start = None
        for i, m in enumerate(miss):
            if m and start is None:
                start = i
            if (not m or i == len(miss) - 1) and start is not None:
                end = i if m else i - 1
                length = end - start + 1
                if length >= min_hours:
                    rows.append({"well_id": well, "gap_start": ts[start], "gap_end": ts[end],
                                 "hours": length})
                start = None
    return pd.DataFrame(rows)


def common_outages(gaps: pd.DataFrame, min_wells: int = 6) -> pd.DataFrame:
    """Days on which many wells have a data gap at the same time (collection outages)."""
    if gaps.empty:
        return gaps
    gaps = gaps.copy()
    gaps["day"] = pd.to_datetime(gaps["gap_start"]).dt.floor("D")
    agg = gaps.groupby("day").agg(wells=("well_id", "nunique"), max_hours=("hours", "max"))
    return agg[agg["wells"] >= min_wells].reset_index()


def check_nk7_regime(grid: pd.DataFrame) -> pd.DataFrame:
    g = grid[(grid["well_id"] == NK7_WELL) & grid["has_data"]].copy()
    g["period"] = (g["timestamp"] >= pd.Timestamp(NK7_REGIME_START)).map(
        {False: f"before {NK7_REGIME_START}", True: f"from {NK7_REGIME_START}"})
    return g.groupby("period").agg(
        spm_median=("SPM", "median"),
        spm_zero_pct=("SPM", lambda s: round(100 * (s == 0).mean(), 1)),
        area_negative_pct=("dynamometer_area", lambda s: round(100 * (s < 0).mean(), 1)),
        hours=("SPM", "size")).reset_index()


def run_audit(long_df: pd.DataFrame, grid: pd.DataFrame, shifts: pd.DataFrame) -> dict:
    """Run every check; returns a dict of DataFrames/dicts (saved by the pipeline)."""
    gaps = gap_runs(grid)
    return {
        "overview": basic_overview(long_df),
        "flags": flag_summary(long_df),
        "missing_hours": missing_hours(grid),
        "missing_parameters": missing_parameters(grid),
        "gaps": gaps,
        "common_outages": common_outages(gaps),
        "area_shift": shifts,
        "nk7_regime": check_nk7_regime(grid),
    }
