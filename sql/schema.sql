-- NK oil wells: normalised schema (SQLite)
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS shut_in_events;
DROP TABLE IF EXISTS measurement_flags;
DROP TABLE IF EXISTS measurements;
DROP TABLE IF EXISTS hourly_wide;
DROP TABLE IF EXISTS parameters;
DROP TABLE IF EXISTS wells;
DROP TABLE IF EXISTS well_types;
DROP TABLE IF EXISTS detection_methods;

CREATE TABLE well_types (
    well_type     TEXT PRIMARY KEY,           -- fountain / sucker_rod_pump
    description   TEXT
);

CREATE TABLE wells (
    well_id       TEXT PRIMARY KEY,           -- NK-7 ...
    well_type     TEXT NOT NULL REFERENCES well_types(well_type)
);

CREATE TABLE parameters (
    parameter     TEXT PRIMARY KEY,           -- SPM, tubing_pressure ...
    unit          TEXT NOT NULL,
    well_type     TEXT NOT NULL REFERENCES well_types(well_type)
);

-- long format: one row = one parameter of one well in one hour
CREATE TABLE measurements (
    measurement_id INTEGER PRIMARY KEY,
    well_id        TEXT NOT NULL REFERENCES wells(well_id),
    ts             TEXT NOT NULL,             -- 'YYYY-MM-DD HH:MM:SS', start of the hour
    parameter      TEXT NOT NULL REFERENCES parameters(parameter),
    value          REAL,
    n_obs          INTEGER,
    split          TEXT NOT NULL CHECK (split IN ('train', 'test')),
    UNIQUE (well_id, ts, parameter)
);

-- ';'-separated flags normalised into one row per flag (1NF)
CREATE TABLE measurement_flags (
    measurement_id INTEGER NOT NULL REFERENCES measurements(measurement_id),
    flag           TEXT NOT NULL,
    PRIMARY KEY (measurement_id, flag)
);

-- wide format produced by the pandas pivot (analysis table, complete hourly grid)
CREATE TABLE hourly_wide (
    well_id          TEXT NOT NULL REFERENCES wells(well_id),
    ts               TEXT NOT NULL,
    split            TEXT NOT NULL,
    has_data         INTEGER NOT NULL,
    regime           TEXT,
    tubing_pressure  REAL, casing_pressure REAL, line_pressure REAL,
    spm              REAL, pump_fillage REAL, min_rod_weight REAL, max_rod_weight REAL,
    dynamometer_area REAL, area_norm REAL,
    PRIMARY KEY (well_id, ts)
);

CREATE TABLE detection_methods (
    method        TEXT PRIMARY KEY,           -- reference / baseline / rule_based / statistical
    description   TEXT
);

CREATE TABLE shut_in_events (
    event_id      TEXT PRIMARY KEY,
    well_id       TEXT NOT NULL REFERENCES wells(well_id),
    method        TEXT NOT NULL REFERENCES detection_methods(method),
    start_ts      TEXT NOT NULL,
    end_ts        TEXT NOT NULL,
    duration_h    INTEGER NOT NULL CHECK (duration_h >= 1),
    split         TEXT NOT NULL
);

CREATE INDEX idx_meas_well_ts ON measurements(well_id, ts);
CREATE INDEX idx_meas_param   ON measurements(parameter);
CREATE INDEX idx_events_well  ON shut_in_events(well_id, method, start_ts);
