"""Partial-hour downtime and intermittent (cycling) operation of sucker-rod pump wells.

Every value is the hourly MEAN of ~10-minute records. A pump that stops for 20 minutes
therefore shows SPM ~ 2/3 of its normal speed instead of 0, and the shut-in detectors
(SPM ~ 0) never see it. This module estimates, per SRP hour:

    run_frac    = SPM / nominal speed, clipped to [0, 1]  (share of the hour the pump ran)
    est_down_h  = 1 - run_frac                             (hours stopped within that hour)

Nominal speed = median of running hours (SPM >= 1) in the trailing 72 h (720 h fallback after
long stops). An hour whose level matches both neighbours within 5 % is a new speed set-point
(operator changed the speed), not a partial stop, and keeps run_frac = 1.

A well-day with >= 6 partial hours is flagged `cycling`: the pump runs intermittently
(pump-off / time-cycle control, e.g. NK-7, NK-82, NK-93 from June). Stops inside that regime
are operating behaviour rather than discrete shut-ins, but they still cost production time,
so they are counted in est_down_h.

Limitation: the stop is assumed to happen inside the hour; the hourly mean cannot tell when.
"""
import numpy as np
import pandas as pd

from .config import (CYCLING_MIN_PARTIAL_H, NOMINAL_FALLBACK, NOMINAL_WINDOW, PARTIAL_FRAC, SRP,
                     SPM_RUNNING_MIN, STABLE_TOL, ZERO_EPS)


def _per_well(g: pd.DataFrame) -> pd.DataFrame:
    spm = g.set_index("timestamp")["SPM"]
    running = spm.where(spm >= SPM_RUNNING_MIN)
    nominal = (running.rolling(NOMINAL_WINDOW, closed="left").median()
               .fillna(running.rolling(NOMINAL_FALLBACK, closed="left").median()))
    prev, nxt = spm.shift(1), spm.shift(-1)
    stable = ((spm >= SPM_RUNNING_MIN)
              & ((spm - prev).abs() / spm <= STABLE_TOL)
              & ((spm - nxt).abs() / spm <= STABLE_TOL))
    frac = (spm / nominal).clip(upper=1.0)
    frac = frac.where(~stable & nominal.notna(), 1.0)          # set-point change / no baseline
    frac = frac.where(spm.abs() > ZERO_EPS, 0.0)                # full-hour stop
    frac = frac.where(spm.notna())                              # no data
    out = pd.DataFrame({"nominal_spm": nominal.to_numpy(), "run_frac": frac.to_numpy()}, index=g.index)
    out["partial_stop"] = (out["run_frac"] > 0) & (out["run_frac"] < PARTIAL_FRAC)
    return out


def add_partial_downtime(grid: pd.DataFrame) -> pd.DataFrame:
    """Add nominal_spm, run_frac, partial_stop, est_down_h and cycling to the hourly grid."""
    out = grid.copy()
    out["nominal_spm"] = np.nan
    out["run_frac"] = np.nan
    out["partial_stop"] = False
    srp = out["well_type"] == SRP
    parts = [_per_well(g) for _, g in out[srp].groupby("well_id", sort=False)]
    if parts:
        res = pd.concat(parts)
        out.loc[res.index, ["nominal_spm", "run_frac", "partial_stop"]] = res
    out["partial_stop"] = out["partial_stop"].astype(bool)
    # hour counts as fully down only if the pump really stopped; partial hours count their share
    out["est_down_h"] = np.where(out["run_frac"] < PARTIAL_FRAC, 1 - out["run_frac"],
                                 np.where(out["run_frac"].notna(), 0.0, np.nan))
    day = out["timestamp"].dt.floor("D")
    n_partial = out.groupby([out["well_id"], day])["partial_stop"].transform("sum")
    out["cycling"] = srp & (n_partial >= CYCLING_MIN_PARTIAL_H)
    return out
