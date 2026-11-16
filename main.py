"""Console menu for the NK shut-in detection project.

Run:  python main.py            (interactive menu)
      python main.py --all      (run the whole pipeline without the menu)
"""
import sys

import pandas as pd

from src import database, pipeline
from src.config import DATA_PROCESSED
from src.data_loader import benchmark_storage
from src.utils import load_state, save_state

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

MENU = """
================  NK Oil Wells - Shut-in Event Detection  ================
 1 - Load data (chunked CSV -> optimised dtypes -> Parquet)
 2 - Data quality audit
 3 - Run shut-in detectors and compare methods (train -> test)
 4 - Build SQLite database and run the SQL queries
 5 - Search shut-in events of a well
 6 - EDA summary by well type
 7 - Build figures (Plotly HTML + PNG)
 8 - Benchmarks: CSV vs Parquet, pandas vs SQL
 9 - Run everything
 0 - Exit
"""


class Session:
    """Keeps loaded data between menu actions."""

    def __init__(self):
        self.long_df = None
        self.grid = None
        self.shifts = None
        self.metrics = None
        self.events = None
        self.state = load_state()

    def need_data(self):
        if self.long_df is None:
            self.long_df = pipeline.step_load()
            self.grid, self.shifts = pipeline.step_transform(self.long_df)

    def need_events(self):
        self.need_data()
        if self.events is None:
            self.metrics, self.events, _ = pipeline.step_detect(self.grid)


def ask_choice(prompt: str, options: list) -> str:
    """Repeat the question until the user enters one of the options."""
    while True:
        value = input(prompt).strip()
        if value in options:
            return value
        print(f"  Please enter one of: {', '.join(options)}")


def action_load(s: Session):
    s.long_df = None
    s.need_data()


def action_audit(s: Session):
    s.need_data()
    audit = pipeline.step_audit(s.long_df, s.grid, s.shifts)
    print("\nOverview:", audit["overview"])
    print("\nMissing hours per well:\n", audit["missing_hours"].to_string(index=False))
    print("\nQuality flags (top 10):\n", audit["flags"].head(10).to_string(index=False))
    print("\nCommon collection outages:\n", audit["common_outages"].to_string(index=False))
    print("\ndynamometer_area scale shift:\n", audit["area_shift"].round({"pre_median": 1, "post_median": 1, "factor": 1}).to_string(index=False))
    print("\nNK-7 regime check:\n", audit["nk7_regime"].round(2).to_string(index=False))


def action_detect(s: Session):
    s.events = None
    s.need_events()
    cols = ["method", "hour_precision", "hour_recall", "hour_f1", "event_precision",
            "event_recall", "event_f1", "start_mae_h", "duration_mae_h",
            "fit_time_s", "detect_time_s", "peak_mem_mb"]
    print("\nTest split metrics:\n", s.metrics[cols].round(3).to_string(index=False))
    best = s.metrics.sort_values("event_f1", ascending=False).iloc[0]["method"]
    print(f"\nBest method by event F1: {best}")
    s.state["best_method"] = best


def action_database(s: Session):
    s.need_events()
    out = pipeline.step_database(s.long_df, s.grid, s.events)
    print("Load timings:", out["load_timings"])
    for name, r in out["queries"].items():
        print(f"\n--- {name}: {r['desc']}  ({r['time_s']} s, {len(r['df'])} rows)")
        print(r["df"].head(8).to_string(index=False))


def action_search(s: Session):
    s.need_events()
    wells = sorted(s.grid["well_id"].unique())
    print("Wells:", ", ".join(wells))
    well = input("Well id (e.g. NK-7): ").strip().upper()
    if well not in wells:
        print(f"Unknown well '{well}'.")
        return
    method = ask_choice("Method [reference/baseline/rule_based/statistical]: ",
                        ["reference", "baseline", "rule_based", "statistical"])
    try:
        min_h = int(input("Minimum duration in hours [1]: ") or 1)
    except ValueError:
        print("Not a number - using 1 hour.")
        min_h = 1
    ev = s.events[(s.events["well_id"] == well) & (s.events["method"] == method)
                  & (s.events["duration_h"] >= min_h)]
    if ev.empty:
        print("No events found.")
    else:
        print(ev[["event_id", "start_ts", "end_ts", "duration_h", "shut_in_h", "split"]].to_string(index=False))
        print(f"\n{len(ev)} events, total downtime {ev['shut_in_h'].sum()} h, "
              f"mean {ev['duration_h'].mean():.1f} h")
        out = DATA_PROCESSED.parent.parent / "outputs" / "tables" / f"search_{well}_{method}.csv"
        ev.to_csv(out, index=False)
        print(f"Saved to {out}")
    s.state["last_search"] = {"well": well, "method": method, "min_hours": min_h}


def action_eda(s: Session):
    s.need_events()
    out = pipeline.step_eda(s.grid, s.events)
    print("\nDescriptive statistics by type:\n",
          out["describe_by_type"][["well_type", "parameter", "count", "mean", "50%", "std"]]
          .to_string(index=False))
    print("\nShut-in statistics per well:\n", out["event_stats"].to_string(index=False))
    print("\nSpearman correlations, sucker-rod pump:\n", out["corr_srp"].to_string())


def action_figures(s: Session):
    s.need_events()
    eda_out = pipeline.step_eda(s.grid, s.events)
    names = pipeline.step_figures(s.grid, s.events, eda_out, s.metrics)
    print("Saved figures:", ", ".join(names), "-> outputs/figures/")


def action_bench(s: Session):
    s.need_data()
    print("\nStorage formats (train.csv):\n", benchmark_storage().to_string(index=False))
    try:
        print("\npandas vs SQL:\n", database.benchmark_pandas_vs_sql(s.long_df).to_string(index=False))
    except Exception as err:   # database not built yet
        print(f"Build the database first (menu 4): {err}")


def action_all(s: Session):
    res = pipeline.run_all()
    s.metrics, s.events = res["metrics"], res["events"]


ACTIONS = {"1": action_load, "2": action_audit, "3": action_detect, "4": action_database,
           "5": action_search, "6": action_eda, "7": action_figures, "8": action_bench,
           "9": action_all}


def main():
    if "--all" in sys.argv:
        pipeline.run_all()
        return
    session = Session()
    if session.state:
        print(f"Previous session: {session.state}")
    while True:
        print(MENU)
        choice = input("Select an option: ").strip()
        if choice == "0":
            save_state(session.state)
            print("Bye!")
            break
        action = ACTIONS.get(choice)
        if action is None:
            print("Unknown option, try again.")
            continue
        try:
            action(session)
            session.state["last_action"] = choice
            save_state(session.state)
        except FileNotFoundError as err:
            print(f"File error: {err}")
        except KeyboardInterrupt:
            print("\nAction interrupted.")


if __name__ == "__main__":
    main()
