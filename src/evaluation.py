"""Comparing detectors: hour-level precision/recall/F1, event-level matching,
start/duration MAE, run time and peak memory. Fit on train, report on test."""
import numpy as np
import pandas as pd

from .detection import METHODS, hours_to_events, make_labels, reference_flags
from .utils import measure


def hour_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    m = y_true.notna()
    t = y_true[m].astype(bool).to_numpy()
    p = y_pred[m].astype(bool).to_numpy()
    tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"hour_precision": prec, "hour_recall": rec, "hour_f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def match_events(true_ev: pd.DataFrame, pred_ev: pd.DataFrame) -> dict:
    """A predicted event matches a true event of the same well if they overlap in time.
    Each true event is paired with the predicted event of largest overlap."""
    matched_true, matched_pred = 0, set()
    start_err, dur_err = [], []
    for well, t_g in true_ev.groupby("well_id"):
        p_g = pred_ev[pred_ev["well_id"] == well]
        for _, t in t_g.iterrows():
            ov = p_g[(p_g["start_ts"] <= t["end_ts"]) & (p_g["end_ts"] >= t["start_ts"])]
            if ov.empty:
                continue
            matched_true += 1
            overlap = (ov["end_ts"].clip(upper=t["end_ts"]) -
                       ov["start_ts"].clip(lower=t["start_ts"])).dt.total_seconds()
            best = ov.loc[overlap.idxmax()]
            matched_pred.update(ov.index)
            start_err.append(abs((best["start_ts"] - t["start_ts"]).total_seconds()) / 3600)
            dur_err.append(abs(best["duration_h"] - t["duration_h"]))
    n_true, n_pred = len(true_ev), len(pred_ev)
    ev_prec = len(matched_pred) / n_pred if n_pred else 0.0
    ev_rec = matched_true / n_true if n_true else 0.0
    return {"events_true": n_true, "events_pred": n_pred,
            "event_precision": ev_prec, "event_recall": ev_rec,
            "event_f1": 2 * ev_prec * ev_rec / (ev_prec + ev_rec) if ev_prec + ev_rec else 0.0,
            "start_mae_h": float(np.mean(start_err)) if start_err else np.nan,
            "duration_mae_h": float(np.mean(dur_err)) if dur_err else np.nan}


def compare_methods(grid: pd.DataFrame, eval_split: str = "test"):
    """Fit every method on train, predict on the full grid, evaluate on `eval_split`.
    Returns (metrics table, events of all methods incl. reference, hourly predictions)."""
    labels = make_labels(grid)
    train = grid[grid["split"] == "train"]
    preds, perf, fitted_all = {}, {}, {}
    for name, (fit_fn, detect_fn) in METHODS.items():
        res = {}
        with measure(res, "fit"):
            fitted = fit_fn(train) if fit_fn else None
        with measure(res, "detect"):
            preds[name] = detect_fn(grid, fitted)
        fitted_all[name] = fitted
        perf[name] = {"fit_time_s": res["fit"]["time_s"], "detect_time_s": res["detect"]["time_s"],
                      "peak_mem_mb": max(res["fit"]["peak_mem_mb"], res["detect"]["peak_mem_mb"])}

    events = [hours_to_events(grid, reference_flags(grid, labels), "reference")]
    events += [hours_to_events(grid, p, n) for n, p in preds.items()]
    events = pd.concat(events, ignore_index=True)

    in_split = grid["split"] == eval_split
    true_ev = events[(events["method"] == "reference") & (events["split"] == eval_split)]
    rows = []
    for name in METHODS:
        pred_ev = events[(events["method"] == name) & (events["split"] == eval_split)]
        row = {"method": name}
        row.update(hour_metrics(labels[in_split], preds[name][in_split]))
        row.update(match_events(true_ev, pred_ev))
        row.update(perf[name])
        rows.append(row)
    metrics = pd.DataFrame(rows)
    hourly = grid[["well_id", "well_type", "timestamp", "split", "regime"]].copy()
    hourly["label"] = labels
    for n, p in preds.items():
        hourly[n] = p.astype(bool)
    return metrics, events, hourly


def metrics_by_well_type(hourly: pd.DataFrame, eval_split: str = "test") -> pd.DataFrame:
    rows = []
    h = hourly[hourly["split"] == eval_split]
    for wt, g in h.groupby("well_type"):
        for name in METHODS:
            r = {"well_type": wt, "method": name}
            r.update({k: v for k, v in hour_metrics(g["label"], g[name]).items()
                      if k.startswith("hour_")})
            rows.append(r)
    return pd.DataFrame(rows)
