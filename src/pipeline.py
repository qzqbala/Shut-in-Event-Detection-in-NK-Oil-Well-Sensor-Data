"""End-to-end pipeline used by both the console menu and the command line."""
import json

import pandas as pd

from . import database, eda, quality, viz
from .config import DATA_PROCESSED, FOUNTAIN, SRP, TABLES
from .data_loader import benchmark_storage, load_dataset, save_parquet
from .evaluation import compare_methods, metrics_by_well_type
from .transform import build_hourly_table
from .utils import df_memory_mb, measure


def _save_table(df: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)


def step_load() -> pd.DataFrame:
    df = load_dataset()
    save_parquet(df)
    print(f"Loaded {len(df):,} long rows, {df_memory_mb(df)} MB in memory -> Parquet saved")
    return df


def step_transform(long_df):
    grid, shifts = build_hourly_table(long_df)
    grid.to_parquet(DATA_PROCESSED / "hourly_wide.parquet", index=False)
    print(f"Wide hourly grid: {len(grid):,} rows x {grid.shape[1]} columns")
    return grid, shifts


def step_audit(long_df, grid, shifts) -> dict:
    audit = quality.run_audit(long_df, grid, shifts)
    for name, obj in audit.items():
        if isinstance(obj, pd.DataFrame):
            _save_table(obj.reset_index() if name == "missing_parameters" else obj, f"quality_{name}")
    (TABLES / "quality_overview.json").write_text(json.dumps(audit["overview"], indent=2))
    return audit


def step_detect(grid):
    metrics, events, hourly = compare_methods(grid, "test")
    metrics_train, _, _ = compare_methods(grid, "train")
    by_type = metrics_by_well_type(hourly, "test")
    events.to_parquet(DATA_PROCESSED / "shut_in_events.parquet", index=False)
    _save_table(metrics.round(4), "method_metrics_test")
    _save_table(metrics_train.round(4), "method_metrics_train")
    _save_table(by_type.round(4), "method_metrics_by_well_type_test")
    _save_table(events, "shut_in_events")
    return metrics, events, hourly


def step_database(long_df, grid, events) -> dict:
    timings = database.build_database(long_df, grid, events)
    results = database.run_all_queries()
    for name, r in results.items():
        _save_table(r["df"], f"sql_{name}")
    bench = database.benchmark_pandas_vs_sql(long_df)
    _save_table(bench, "benchmark_pandas_vs_sql")
    return {"load_timings": timings, "queries": results, "pandas_vs_sql": bench}


def step_eda(grid, events) -> dict:
    out = {
        "describe_by_type": eda.describe_by_type(grid),
        "describe_by_well": eda.describe_by_well(grid),
        "daily_profile": eda.daily_profile(grid),
        "corr_fountain": eda.correlations(grid, FOUNTAIN),
        "corr_srp": eda.correlations(grid, SRP),
        "event_stats": eda.event_stats(events, grid),
        "duration_buckets": eda.duration_buckets(events),
        "monthly_events": eda.monthly_events(events),
    }
    for name, df in out.items():
        _save_table(df.reset_index() if name.startswith("corr") else df, f"eda_{name}")
    return out


def step_figures(grid, events, eda_out, metrics) -> list:
    figs = {
        "01_coverage_heatmap": viz.missing_heatmap(grid),
        "02_area_shift_NK-68": viz.area_shift(grid, "NK-68"),
        "03_daily_pattern": viz.daily_pattern(eda_out["daily_profile"]),
        "04_correlations": viz.correlation_heatmap(eda_out["corr_fountain"], eda_out["corr_srp"]),
        "05_timeline_NK-7_SPM": viz.timeline(grid, events, "NK-7", "SPM", "2026-05-01", "2026-06-15"),
        "06_shutin_durations": viz.duration_box(events),
        "07_monthly_events": viz.monthly_frequency(eda_out["monthly_events"]),
        "08_method_comparison": viz.method_comparison(metrics),
    }
    for name, fig in figs.items():
        viz.save(fig, name)
    return list(figs)


def run_all() -> dict:
    perf = {}
    with measure(perf, "load"):
        long_df = step_load()
    with measure(perf, "transform"):
        grid, shifts = step_transform(long_df)
    with measure(perf, "audit"):
        step_audit(long_df, grid, shifts)
    with measure(perf, "detect"):
        metrics, events, _ = step_detect(grid)
    with measure(perf, "database"):
        db = step_database(long_df, grid, events)
    with measure(perf, "eda"):
        eda_out = step_eda(grid, events)
    with measure(perf, "figures"):
        step_figures(grid, events, eda_out, metrics)
    storage = benchmark_storage()
    _save_table(storage, "benchmark_storage")
    perf_df = pd.DataFrame(perf).T.reset_index(names="step")
    # the detect step measures memory per method internally (nested tracemalloc),
    # so its outer peak is not meaningful -> see peak_mem_mb in method_metrics_test.csv
    perf_df.loc[perf_df["step"] == "detect", "peak_mem_mb"] = float("nan")
    _save_table(perf_df, "pipeline_performance")
    print("\nDetector comparison (test split):")
    print(metrics[["method", "hour_precision", "hour_recall", "hour_f1", "event_precision",
                   "event_recall", "event_f1", "duration_mae_h", "detect_time_s"]].round(3)
          .to_string(index=False))
    return {"metrics": metrics, "events": events, "db": db, "storage": storage, "perf": perf}


if __name__ == "__main__":
    run_all()
