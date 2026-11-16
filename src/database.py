"""SQLite layer: create the normalised schema, load data, run the analysis queries,
and benchmark pandas vs SQL for the same aggregation."""
import re
import sqlite3
import time

import pandas as pd

from .config import DB_PATH, SQL_DIR

WELL_TYPE_DESC = {
    "fountain": "Natural-flow (fountain) well; monitored by wellhead pressures",
    "sucker_rod_pump": "Mechanised well with sucker-rod pump; monitored by SPM, loads, dynamogram",
}
METHOD_DESC = {
    "reference": "Multi-sensor consensus (silver standard)",
    "baseline": "Global threshold on primary signal, hour by hour",
    "rule_based": "Expert rules with secondary-signal confirmation and gap bridging",
    "statistical": "Per-well Gaussian mixture fitted on train + smoothing",
}


def connect(path=DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def create_schema(con: sqlite3.Connection) -> None:
    con.executescript((SQL_DIR / "schema.sql").read_text())


def _ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d %H:%M:%S")


def load_reference_tables(con, long_df: pd.DataFrame) -> None:
    pd.DataFrame({"well_type": list(WELL_TYPE_DESC), "description": list(WELL_TYPE_DESC.values())}
                 ).to_sql("well_types", con, if_exists="append", index=False)
    wells = long_df[["well_id", "well_type"]].drop_duplicates().astype(str)
    wells.to_sql("wells", con, if_exists="append", index=False)
    params = long_df[["parameter", "unit", "well_type"]].drop_duplicates("parameter").astype(str)
    params.to_sql("parameters", con, if_exists="append", index=False)
    pd.DataFrame({"method": list(METHOD_DESC), "description": list(METHOD_DESC.values())}
                 ).to_sql("detection_methods", con, if_exists="append", index=False)


def load_measurements(con, long_df: pd.DataFrame, chunksize: int = 50_000) -> None:
    """Insert long-format measurements in chunks and normalise the ';'-separated flags."""
    m = pd.DataFrame({
        "measurement_id": range(1, len(long_df) + 1),
        "well_id": long_df["well_id"].astype(str).to_numpy(),
        "ts": _ts(long_df["timestamp"]).to_numpy(),
        "parameter": long_df["parameter"].astype(str).to_numpy(),
        "value": long_df["value"].astype(float).to_numpy(),
        "n_obs": long_df["n_obs"].astype(int).to_numpy(),
        "split": long_df["split"].astype(str).to_numpy(),
    })
    m.to_sql("measurements", con, if_exists="append", index=False, chunksize=chunksize)

    flags = pd.DataFrame({"measurement_id": m["measurement_id"],
                          "flag": long_df["flag"].astype(str).to_numpy()})
    flags = flags[flags["flag"] != ""]
    flags["flag"] = flags["flag"].str.split(";")
    flags = flags.explode("flag")
    flags.to_sql("measurement_flags", con, if_exists="append", index=False)


def load_hourly_wide(con, grid: pd.DataFrame) -> None:
    w = grid.rename(columns={"timestamp": "ts", "SPM": "spm"})
    cols = ["well_id", "ts", "split", "has_data", "regime", "tubing_pressure", "casing_pressure",
            "line_pressure", "spm", "pump_fillage", "min_rod_weight", "max_rod_weight",
            "dynamometer_area", "area_norm"]
    w = w[cols].copy()
    w["ts"] = _ts(w["ts"])
    w["has_data"] = w["has_data"].astype(int)
    w.to_sql("hourly_wide", con, if_exists="append", index=False, chunksize=20_000)


def load_events(con, events: pd.DataFrame) -> None:
    e = events[["event_id", "well_id", "method", "start_ts", "end_ts", "duration_h", "split"]].copy()
    e["start_ts"] = _ts(e["start_ts"])
    e["end_ts"] = _ts(e["end_ts"])
    con.execute("DELETE FROM shut_in_events")
    e.to_sql("shut_in_events", con, if_exists="append", index=False)
    con.commit()


def build_database(long_df, grid, events) -> dict:
    """Create and fill the whole database. Returns load timings."""
    timings = {}
    con = connect()
    try:
        t0 = time.perf_counter(); create_schema(con)
        load_reference_tables(con, long_df); timings["reference_tables_s"] = time.perf_counter() - t0
        t0 = time.perf_counter(); load_measurements(con, long_df)
        timings["measurements_s"] = time.perf_counter() - t0
        t0 = time.perf_counter(); load_hourly_wide(con, grid)
        timings["hourly_wide_s"] = time.perf_counter() - t0
        load_events(con, events)
        con.commit()
    finally:
        con.close()
    return {k: round(v, 3) for k, v in timings.items()}


def parse_queries() -> list[dict]:
    """Split sql/queries.sql into named queries."""
    text = (SQL_DIR / "queries.sql").read_text()
    blocks = re.split(r"^-- name:\s*", text, flags=re.M)[1:]
    queries = []
    for block in blocks:
        name, rest = block.split("\n", 1)
        desc = re.search(r"^-- desc:\s*(.*)$", rest, flags=re.M)
        sql = "\n".join(l for l in rest.splitlines() if not l.startswith("--")).strip()
        queries.append({"name": name.strip(), "desc": desc.group(1) if desc else "", "sql": sql})
    return queries


def run_query(sql: str) -> pd.DataFrame:
    con = connect()
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


def run_all_queries() -> dict:
    results = {}
    for q in parse_queries():
        t0 = time.perf_counter()
        try:
            df = run_query(q["sql"])
        except sqlite3.Error as err:
            print(f"Query {q['name']} failed: {err}")
            continue
        results[q["name"]] = {"desc": q["desc"], "df": df,
                              "time_s": round(time.perf_counter() - t0, 4)}
    return results


def benchmark_pandas_vs_sql(long_df: pd.DataFrame, repeats: int = 3) -> pd.DataFrame:
    """Same task both ways: monthly mean of every parameter per well."""
    sql = """SELECT well_id, parameter, substr(ts,1,7) AS month, AVG(value) AS mean_value
             FROM measurements GROUP BY well_id, parameter, month"""
    t_sql, t_pd = [], []
    for _ in range(repeats):
        t0 = time.perf_counter(); res_sql = run_query(sql); t_sql.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        res_pd = (long_df.assign(month=long_df["timestamp"].dt.month)
                  .groupby(["well_id", "parameter", "month"], observed=True)["value"].mean())
        t_pd.append(time.perf_counter() - t0)
    same = len(res_sql) == len(res_pd)
    return pd.DataFrame([
        {"approach": "SQL (SQLite, GROUP BY)", "time_s": round(min(t_sql), 4), "groups": len(res_sql)},
        {"approach": "pandas groupby (in memory)", "time_s": round(min(t_pd), 4), "groups": len(res_pd)},
    ]).assign(results_match=same)
