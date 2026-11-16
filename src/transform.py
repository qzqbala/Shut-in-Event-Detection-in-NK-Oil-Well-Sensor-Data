"""Cleaning and reshaping: masking impossible values, long -> wide pivot,
full hourly grid, dynamometer_area scale normalisation, NK-7 regime label."""
import numpy as np
import pandas as pd

from .config import (AREA_SHIFT_FACTOR, FOUNTAIN_PARAMS, MASK_FLAGS, NK7_REGIME_START,
                     NK7_WELL, SRP_PARAMS)


def mask_flagged_values(df: pd.DataFrame) -> pd.DataFrame:
    """Set values with physically impossible flags (16-bit overflow, |area|>1e6) to NaN.
    Rows are kept so the audit can still count them."""
    out = df.copy()
    flags = out["flag"].astype(str)
    bad = np.zeros(len(out), dtype=bool)
    for f in MASK_FLAGS:
        bad |= flags.str.contains(f, regex=False).to_numpy()
    out["value_clean"] = out["value"].where(~bad)
    out["masked"] = bad
    return out


def to_wide(df: pd.DataFrame, value_col: str = "value_clean") -> pd.DataFrame:
    """Pivot long -> wide: one row per (well, hour), one column per parameter."""
    wide = (df.pivot_table(index=["well_id", "well_type", "timestamp", "split"],
                           columns="parameter", values=value_col, aggfunc="first",
                           observed=True)
              .reset_index())
    wide.columns.name = None
    for col in FOUNTAIN_PARAMS + SRP_PARAMS:
        if col not in wide:
            wide[col] = np.nan
    # number of raw records in the hour (max over parameters) - useful to spot partial hours
    nobs = df.groupby(["well_id", "timestamp"], observed=True)["n_obs"].max().rename("n_obs")
    wide = wide.merge(nobs.reset_index(), on=["well_id", "timestamp"], how="left")
    return wide.sort_values(["well_id", "timestamp"], ignore_index=True)


def add_full_hour_grid(wide: pd.DataFrame) -> pd.DataFrame:
    """Reindex each well to a complete hourly grid. Missing hours become rows with
    has_data=False, so "no data" is never confused with "shut-in"."""
    frames = []
    test_start = wide.loc[wide["split"] == "test", "timestamp"].min()
    full_idx = pd.date_range(wide["timestamp"].min(), wide["timestamp"].max(), freq="h")
    for well, g in wide.groupby("well_id", observed=True):
        g = g.set_index("timestamp").reindex(full_idx)
        g.index.name = "timestamp"
        g["has_data"] = g["well_type"].notna().astype(bool)
        g["well_id"] = well
        g["well_type"] = g["well_type"].astype(object).ffill().bfill()
        frames.append(g.reset_index())
    grid = pd.concat(frames, ignore_index=True)
    grid["split"] = np.where(grid["timestamp"] >= test_start, "test", "train")
    return grid


def detect_area_shift(wide: pd.DataFrame) -> pd.DataFrame:
    """Find, per SRP well, the first day whose daily median dynamometer_area exceeds
    AREA_SHIFT_FACTOR x the January-April median (the documented May 2026 scale shift)."""
    rows = []
    srp = wide[wide["dynamometer_area"].notna()]
    for well, g in srp.groupby("well_id", observed=True):
        daily = g.set_index("timestamp")["dynamometer_area"].resample("D").median().dropna()
        base = daily[:"2026-04-30"].median()
        after = daily["2026-04-25":]
        jump = after[after.abs() > AREA_SHIFT_FACTOR * abs(base)]
        shift_day = jump.index.min() if len(jump) else pd.NaT
        post = daily[shift_day:].median() if pd.notna(shift_day) else np.nan
        rows.append({"well_id": well, "pre_median": base, "shift_date": shift_day,
                     "post_median": post,
                     "factor": post / base if pd.notna(shift_day) and base else np.nan})
    return pd.DataFrame(rows)


def normalize_area(wide: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    """Divide dynamometer_area by its median within each scale period (before/after shift)
    and operating regime. Medians come from the train split only (no leakage).
    The NK-7 low-SPM regime (mostly negative areas) is left un-normalised (NaN)."""
    out = wide.copy()
    out["area_period"] = "pre"
    out["area_norm"] = np.nan
    for _, s in shifts.iterrows():
        m = out["well_id"] == s["well_id"]
        if pd.notna(s["shift_date"]):
            out.loc[m & (out["timestamp"] >= s["shift_date"]), "area_period"] = "post"
        for period in ("pre", "post"):
            mp = m & (out["area_period"] == period) & (out["regime"] == "normal")
            running = mp & (out["split"] == "train") & (out["SPM"] > 0)
            med = out.loc[running, "dynamometer_area"].median()
            if pd.notna(med) and med > 0:
                out.loc[mp, "area_norm"] = out.loc[mp, "dynamometer_area"] / med
    return out


def add_regime(wide: pd.DataFrame) -> pd.DataFrame:
    """Operating regime label: NK-7 switches to a low-SPM regime from NK7_REGIME_START."""
    out = wide.copy()
    out["regime"] = "normal"
    m = (out["well_id"] == NK7_WELL) & (out["timestamp"] >= pd.Timestamp(NK7_REGIME_START))
    out.loc[m, "regime"] = "nk7_low_spm"
    return out


def build_hourly_table(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full transformation chain. Returns (hourly wide grid, area shift table)."""
    clean = mask_flagged_values(long_df)
    wide = to_wide(clean)
    shifts = detect_area_shift(wide)
    grid = add_full_hour_grid(wide)
    grid = add_regime(grid)
    grid = normalize_area(grid, shifts)
    grid["hour"] = grid["timestamp"].dt.hour
    grid["month"] = grid["timestamp"].dt.to_period("M").astype(str)
    return grid, shifts
