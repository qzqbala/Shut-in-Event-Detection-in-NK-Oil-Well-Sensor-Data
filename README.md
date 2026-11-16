# Shut-in Event Detection — NK Oil Wells

Final project for **Programming for Data Analysis and Database** (AAI, Astana IT University).
Cluster 4 — *Industrial Sensor Data Analytics*. Group AAI-2605.

**Team:** Nikita Trubarov · Dulat Tulebaev · Nurdaulet Tastan

The project finds **shut-ins** (periods when a well is stopped) in hourly sensor data from
10 wells of the NK field. It also measures how often shut-ins happen and how long they last,
and compares a baseline, a rule-based and a statistical detector on the chronological
train/test split that comes with the dataset.

Dataset: [Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2](https://huggingface.co/datasets/Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2)
(Apache-2.0; data provided by Galaz and Company LLP). 258,387 long-format rows,
10 wells (3 fountain, 7 sucker-rod pump), 2026-01-01 – 2026-09-30, hourly.

## Quick start

```bash
git clone https://github.com/qzqbala/Shut-in-Event-Detection-in-NK-Oil-Well-Sensor-Data.git
cd Shut-in-Event-Detection-in-NK-Oil-Well-Sensor-Data
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py --all        # full pipeline (~1 min): data -> audit -> detection -> SQLite -> EDA -> figures
python main.py              # interactive console menu
streamlit run app.py        # dashboard (run the pipeline once first)
```

`train.csv` and `test.csv` are in `data/raw/`. If they are missing, the loader downloads them
from Hugging Face.

**Google Colab:** upload the project folder (or `git clone`), then run
`!pip install -r requirements.txt` and `!python main.py --all`.

Tested on Python 3.13 with pandas 3.0.5 and pandas 2.2.3; both give identical results.

## Console menu

| Key | Action |
|---|---|
| 1 | Load data: chunked CSV → optimised dtypes → Parquet |
| 2 | Data quality audit (missing hours/parameters, flags, outages, known issues) |
| 3 | Run the three detectors, compare metrics (fit on train, evaluate on test) |
| 4 | Build the SQLite database and run the 14 SQL queries |
| 5 | Search shut-in events of a well (by method and minimum duration) → CSV |
| 6 | EDA summary by well type |
| 7 | Build figures (interactive HTML + PNG) |
| 8 | Benchmarks: CSV vs Parquet, pandas vs SQL |
| 9 | Run everything |
| 0 | Exit (program state is saved to `outputs/session_state.json`) |

## Project structure

```
nk_shutin/
├── main.py                 console menu / CLI entry point
├── app.py                  Streamlit dashboard (well, parameter, period, method filters)
├── requirements.txt
├── src/
│   ├── config.py           paths, constants, thresholds
│   ├── data_loader.py      download, chunked CSV reading, dtype optimisation, Parquet, storage benchmark
│   ├── transform.py        masking flagged values, long→wide pivot, hourly grid, area scale fix, NK-7 regime
│   ├── downtime.py         partial-hour downtime of pump wells (SPM / nominal speed), cycling regime
│   ├── quality.py          data quality audit
│   ├── detection.py        reference labels, baseline / rule-based / statistical detectors, event table
│   ├── evaluation.py       hour- and event-level metrics, MAE, time and memory
│   ├── database.py         SQLite schema + loading, query runner, pandas-vs-SQL benchmark
│   ├── eda.py              descriptive stats, daily patterns, correlations, shut-in statistics
│   ├── viz.py              Plotly figures
│   ├── pipeline.py         end-to-end steps
│   └── utils.py            time/memory measurement, JSON state
├── sql/
│   ├── schema.sql          normalised schema (8 tables)
│   └── queries.sql         14 analysis queries (JOIN, GROUP BY/HAVING, subqueries, CTE, LAG, RANK, window frames)
├── docs/er_diagram.png     ER diagram (source: er_diagram.dot)
├── data/raw/               train.csv, test.csv
├── data/processed/         Parquet files + nk_wells.db (generated)
└── outputs/
    ├── tables/             all results as CSV (metrics, audit, SQL results, EDA)
    └── figures/            8 figures (.html interactive, .png static)
```

## Method in one paragraph

The dataset has no verified shut-in labels. We therefore build a **reference (silver
standard)** from the strictest signal available:
- **Sucker-rod pump:** SPM is exactly 0 for the whole hour (every raw record shows no strokes).
  Rod weight is not required to be 0: a stopped pump still carries the hanging rod string,
  and NK-82 / NK-93 keep reporting that static load while idle.
- **Fountain well:** tubing pressure below 0.5 atm, and line pressure below 0.5 atm or not
  reported (line pressure is missing for long periods on NK-83 and NK-75).

The detectors use only the primary signal (SPM or tubing pressure) and one secondary
signal: pump fillage for pump wells, casing pressure for fountain wells. All three run on
the same data:
- **Baseline:** a fixed threshold, applied hour by hour.
- **Rule-based:** expert rules. A pump-well shut-in needs no pumping card. A fountain
  shut-in must persist for at least 2 of 3 hours. Data gaps of up to 2 h inside an event
  are bridged.
- **Statistical:** a Gaussian mixture with 3 components per well and regime, fitted on
  the train split only, followed by smoothing.

For the reference and every detector, shut-in hours separated by at most 3 hours of
operation form **one event** (a short restart attempt is not a new shut-in).
`duration_h` is the event length; `shut_in_h` is the downtime inside it.

Values are hourly means of ~10-minute records, so a pump that stops for part of an hour
shows a reduced SPM, not 0. `downtime.py` estimates the stopped share of each pump-well hour
(1 − SPM / nominal speed) and flags days with intermittent (cycling) operation; query
`q14_partial_downtime` reports availability including this partial-hour downtime.

Evaluation uses the test split (2026-08-08 23:00 onward) and reports:
- hour-level precision, recall and F1;
- event-level precision, recall and F1;
- start-time and duration MAE;
- run time and peak memory.

## Main results (test split)

| Method | Hour F1 | Event precision | Event recall | Event F1 | Duration MAE, h | Detect time, s |
|---|---|---|---|---|---|---|
| Baseline | 0.890 | 0.462 | **1.000** | 0.632 | **0.16** | 0.01 |
| Rule-based | **0.972** | 0.840 | 0.977 | **0.903** | 0.19 | 0.04 |
| Statistical (GMM) | 0.921 | **0.842** | 0.372 | 0.516 | 0.63 | 0.06 |

The rule-based detector gives the best hour- and event-level F1. The baseline also fires
on hours where the pump ran only briefly (SPM < 0.5) and on single noisy hours. The GMM
is as precise as the rules but misses most 1-hour stops: its 3-hour smoothing removes
them, and the test period is dominated by the short stops of the NK-82 / NK-93 cycling
regime.

Partial-hour downtime matters: for NK-7, NK-82 and NK-93 the stopped share of partial
hours (370–560 h per well) is as large as the full shut-in downtime, so availability drops
to 83–90 % (q14) instead of 92–96 % from shut-ins alone (q10).

Full tables are in `outputs/tables/`. The written report and the slide deck have the
discussion and limitations.

## Team contributions

| Member | Role | Main work |
|---|---|---|
| Nikita Trubarov | Data engineering | `data_loader.py` (chunked loading, dtype optimisation, Parquet, storage benchmark), `transform.py` (long → wide pivot, hourly grid, area normalisation, NK-7 regime), `quality.py` (data quality audit); report sections 2–3 |
| Dulat Tulebaev | Methods and integration (team lead) | `detection.py` (reference labels, baseline / rule-based / GMM detectors, event table), `evaluation.py` (metrics, MAE, time and memory), `pipeline.py`, `main.py` console menu, GitHub repository and README; report sections 6–7 |
| Nurdaulet Tastan | Database and visualisation | `sql/schema.sql`, `sql/queries.sql`, `database.py`, ER diagram, pandas-vs-SQL benchmark, `eda.py`, `viz.py` (8 figures), `app.py` Streamlit dashboard, presentation; report sections 4–5 |

Every member reviewed the whole code base and can explain any module during the defense.

## Licence and attribution

Code: MIT. Data: Apache-2.0, © Galaz and Company LLP (ТОО «Галаз и Компания»), published by
A. Tleubayeva on Hugging Face (2026).
