"""Shut-in detection.

There are no verified shut-in labels in the dataset, so a REFERENCE ("silver standard")
is built from the strictest signal available:
  * sucker-rod pump: SPM hourly mean == 0, i.e. every raw record of the hour shows no strokes.
    Rod weight is NOT required to be 0: a stopped pump still carries the hanging rod string,
    and in the intermittent regime of NK-82 / NK-93 the controller keeps reporting that static
    load (~8,500-9,000 kg) during idle hours. Requiring weight == 0 labelled those real stops
    as "running" and penalised detectors that found them.
  * fountain: tubing pressure < 0.5 atm AND line pressure < 0.5 atm or not reported
    (no flow along the flow path; casing/annulus pressure usually stays high during a
    surface shut-in). Line pressure is missing for long periods (NK-83 from April, NK-75
    from late June), so requiring it left fully depressurised hours unevaluated.

Detectors only see the primary signal (SPM / tubing pressure) plus secondary signals
that are NOT part of the reference (pump fillage, casing pressure):
  1. baseline     - global fixed threshold on the primary signal, hour by hour;
  2. rule_based   - expert rules: primary near zero AND secondary signal confirms
                    (no dynamogram / pump fillage ~0 for SRP), persistence for fountain
                    wells, bridging of short data gaps inside an event;
  3. statistical  - per-well, per-regime 3-component Gaussian mixture on
                    [log1p(primary), log1p(secondary)] fitted on TRAIN, plus
                    rolling-median smoothing and the same gap bridging.
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .config import (BASELINE_PRESSURE, BASELINE_SPM, EVENT_MERGE_GAP_H, FOUNTAIN, PRESSURE_OFF_ATM,
                     PRIMARY_SIGNAL, RANDOM_STATE, RULE_FILLAGE_MAX, RULE_MERGE_GAP_H,
                     RULE_PRIMARY_MAX, SRP, STAT_COMPONENTS, STAT_LOW_MEAN,
                     STAT_SMOOTH_H, ZERO_EPS)


def primary_signal(grid: pd.DataFrame) -> pd.Series:
    """SPM for sucker-rod pump wells, tubing pressure for fountain wells."""
    return pd.Series(np.where(grid["well_type"] == FOUNTAIN,
                              grid[PRIMARY_SIGNAL[FOUNTAIN]], grid[PRIMARY_SIGNAL[SRP]]),
                     index=grid.index, dtype="float64")


def secondary_signal(grid: pd.DataFrame) -> pd.Series:
    """Pump fillage for SRP (missing when no dynamogram is recorded -> 0),
    casing pressure for fountain wells."""
    fill = grid["pump_fillage"].fillna(0)
    return pd.Series(np.where(grid["well_type"] == FOUNTAIN, grid["casing_pressure"], fill),
                     index=grid.index, dtype="float64")


# ---------------------------------------------------------------- reference labels
def make_labels(grid: pd.DataFrame) -> pd.Series:
    """1 = shut-in, 0 = running, NaN = not evaluable (no data / sensors missing)."""
    label = pd.Series(np.nan, index=grid.index, dtype="float64")

    srp = (grid["well_type"] == SRP) & grid["has_data"] & grid["SPM"].notna()
    stopped = grid["SPM"].abs() <= ZERO_EPS
    label[srp] = stopped[srp].astype(float)

    fnt = (grid["well_type"] == FOUNTAIN) & grid["has_data"] & grid["tubing_pressure"].notna()
    line = grid["line_pressure"]
    off = (grid["tubing_pressure"] < PRESSURE_OFF_ATM) & (line.isna() | (line < PRESSURE_OFF_ATM))
    label[fnt] = off[fnt].astype(float)
    return label


# ---------------------------------------------------------------- helpers
def bridge_data_gaps(flags: pd.Series, has_data: pd.Series, grid: pd.DataFrame,
                     max_gap: int = RULE_MERGE_GAP_H) -> pd.Series:
    """Inside each well, mark hours WITHOUT data as shut-in when they form a hole of at most
    `max_gap` hours between two shut-in hours (a missing hour is not a restart)."""
    out = flags.astype(bool).copy()
    for _, idx in grid.groupby("well_id").groups.items():
        f = out.loc[idx].to_numpy().copy()
        hd = has_data.loc[idx].to_numpy()
        n, i = len(f), 0
        while i < n:
            if f[i]:
                j = i + 1
                while j < n and not f[j] and not hd[j]:
                    j += 1
                if j < n and f[j] and 0 < j - i - 1 <= max_gap:
                    f[i + 1:j] = True
                i = max(j, i + 1)
            else:
                i += 1
        out.loc[idx] = f
    return out


# ---------------------------------------------------------------- detectors
def detect_baseline(grid: pd.DataFrame, fitted=None) -> pd.Series:
    """Baseline: SPM < 0.5 strokes/min or tubing pressure < 0.5 atm, hour by hour."""
    sig = primary_signal(grid)
    thr = np.where(grid["well_type"] == FOUNTAIN, BASELINE_PRESSURE, BASELINE_SPM)
    return ((sig < thr) & grid["has_data"]).astype(bool)


def detect_rule_based(grid: pd.DataFrame, fitted=None) -> pd.Series:
    """Expert rules.
    SRP:      SPM < 0.5 AND pump fillage missing or < 5 %  (no pumping card = not pumping)
    fountain: tubing pressure < 0.5 atm on >= 2 of 3 consecutive hours (persistence)
    Then: data gaps of <= 2 h inside an event are bridged."""
    sig = primary_signal(grid)
    is_f = grid["well_type"] == FOUNTAIN
    srp_rule = ((~is_f) & (sig < RULE_PRIMARY_MAX)
                & (grid["pump_fillage"].isna() | (grid["pump_fillage"] < RULE_FILLAGE_MAX)))
    low = (is_f & (sig < RULE_PRIMARY_MAX)).astype(int)
    persist = low.groupby(grid["well_id"]).transform(
        lambda s: s.rolling(3, center=True, min_periods=1).sum()) >= 2
    fnt_rule = is_f & (sig < RULE_PRIMARY_MAX) & persist
    flags = (srp_rule | fnt_rule) & grid["has_data"]
    return bridge_data_gaps(flags, grid["has_data"], grid)


def _features(grid: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"p": np.log1p(primary_signal(grid).clip(lower=0)),
                         "s": np.log1p(secondary_signal(grid).clip(lower=0))}, index=grid.index)


def fit_statistical(train: pd.DataFrame) -> dict:
    """Per (well, regime) Gaussian mixture on [log1p(primary), log1p(secondary)].
    Components whose mean primary signal is ~0 form the 'shut-in' state."""
    feats = _features(train)
    models = {}
    for (well, regime), idx in train.groupby(["well_id", "regime"]).groups.items():
        x = feats.loc[idx].dropna()
        if len(x) < 50:
            continue
        gmm = GaussianMixture(n_components=STAT_COMPONENTS, covariance_type="full",
                              random_state=RANDOM_STATE, reg_covar=1e-2).fit(x.to_numpy())
        # components whose mean primary signal is ~0 represent the shut-in state
        low = np.flatnonzero(gmm.means_[:, 0] < STAT_LOW_MEAN)
        models[(well, regime)] = {"gmm": gmm, "low": low}
    return models


def detect_statistical(grid: pd.DataFrame, fitted: dict, smooth: int = STAT_SMOOTH_H) -> pd.Series:
    feats = _features(grid)
    flags = pd.Series(False, index=grid.index)
    for (well, regime), idx in grid.groupby(["well_id", "regime"]).groups.items():
        model = fitted.get((well, regime))
        x = feats.loc[idx].dropna()
        if model is None or x.empty:
            continue
        if len(model["low"]) == 0:
            # train had (almost) no shut-ins: no mixture component sits at zero
            # -> fall back to "primary signal is exactly zero"
            proba = (x["p"] <= 0).astype(float).to_numpy()
        else:
            proba = model["gmm"].predict_proba(x.to_numpy())[:, model["low"]].sum(axis=1)
        p = pd.Series(np.nan, index=idx)
        p.loc[x.index] = proba
        p_s = p.rolling(smooth, center=True, min_periods=1).median()   # removes 1-h spikes
        flags.loc[idx] = ((p_s > 0.5) & p.notna()).to_numpy()
    return bridge_data_gaps(flags & grid["has_data"], grid["has_data"], grid)


METHODS = {
    "baseline": (None, detect_baseline),
    "rule_based": (None, detect_rule_based),
    "statistical": (fit_statistical, detect_statistical),
}


# ---------------------------------------------------------------- events
def hours_to_events(grid: pd.DataFrame, flags: pd.Series, method: str,
                    merge_gap_h: int = EVENT_MERGE_GAP_H) -> pd.DataFrame:
    """Collapse shut-in hours of a well into events (one row per event).
    Runs separated by at most `merge_gap_h` non-shut-in hours are merged into one event, so a
    brief restart attempt does not split one shut-in into several.
    duration_h = hours from first to last shut-in hour (restart hours included);
    shut_in_h  = shut-in hours inside the event (use this for downtime sums)."""
    rows = []
    d = grid[["well_id", "well_type", "timestamp", "split"]].copy()
    d["f"] = flags.astype(bool).to_numpy()
    for well, g in d.groupby("well_id"):
        f = g["f"].to_numpy()
        ts = g["timestamp"].to_numpy()
        starts = np.flatnonzero(f & ~np.r_[False, f[:-1]])
        ends = np.flatnonzero(f & ~np.r_[f[1:], False])
        runs = []
        for s, e in zip(starts, ends):
            if runs and s - runs[-1][1] - 1 <= merge_gap_h:
                runs[-1][1] = e
            else:
                runs.append([s, e])
        for s, e in runs:
            rows.append({"well_id": well, "well_type": g["well_type"].iloc[s], "method": method,
                         "start_ts": pd.Timestamp(ts[s]), "end_ts": pd.Timestamp(ts[e]),
                         "duration_h": int(e - s + 1), "shut_in_h": int(f[s:e + 1].sum()),
                         "split": g["split"].iloc[s]})
    ev = pd.DataFrame(rows, columns=["well_id", "well_type", "method", "start_ts", "end_ts",
                                     "duration_h", "shut_in_h", "split"])
    ev.insert(0, "event_id", [f"{method[:3].upper()}-{i:05d}" for i in range(len(ev))])
    return ev


def reference_flags(grid: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """Reference shut-in hours for event building (short data gaps bridged)."""
    return bridge_data_gaps(labels.fillna(0).astype(bool), grid["has_data"], grid)
