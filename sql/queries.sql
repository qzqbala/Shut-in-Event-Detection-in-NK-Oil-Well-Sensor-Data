-- Analysis queries. Each query starts with "-- name: <id>" and a "-- desc:" line;
-- src/database.py splits the file on these markers and loads every result into pandas.

-- name: q01_coverage_per_well
-- desc: Data coverage per well (JOIN + GROUP BY): rows, hours with data, first/last timestamp
SELECT w.well_id,
       w.well_type,
       COUNT(*)                      AS n_rows,
       COUNT(DISTINCT m.ts)          AS hours_with_data,
       COUNT(DISTINCT m.parameter)   AS n_parameters,
       MIN(m.ts)                     AS first_ts,
       MAX(m.ts)                     AS last_ts
FROM measurements m
JOIN wells w ON w.well_id = m.well_id
GROUP BY w.well_id, w.well_type
ORDER BY w.well_type, w.well_id;

-- name: q02_stats_by_type_parameter
-- desc: Descriptive statistics per well type and parameter (3-table JOIN, flagged rows excluded)
SELECT w.well_type,
       m.parameter,
       p.unit,
       COUNT(*)              AS n,
       ROUND(AVG(m.value),2) AS mean_value,
       ROUND(MIN(m.value),2) AS min_value,
       ROUND(MAX(m.value),2) AS max_value
FROM measurements m
JOIN wells w      ON w.well_id = m.well_id
JOIN parameters p ON p.parameter = m.parameter
WHERE m.measurement_id NOT IN (SELECT measurement_id FROM measurement_flags)
GROUP BY w.well_type, m.parameter, p.unit
ORDER BY w.well_type, m.parameter;

-- name: q03_data_gaps_lag
-- desc: Missing hours found with LAG: gaps between consecutive recorded hours longer than 1 h
WITH hours AS (
    SELECT DISTINCT well_id, ts FROM measurements
),
lagged AS (
    SELECT well_id,
           LAG(ts) OVER (PARTITION BY well_id ORDER BY ts) AS prev_ts,
           ts
    FROM hours
)
SELECT well_id,
       prev_ts                                                      AS last_hour_before_gap,
       ts                                                           AS first_hour_after_gap,
       CAST(ROUND((julianday(ts) - julianday(prev_ts)) * 24) AS INT) - 1 AS missing_hours
FROM lagged
WHERE prev_ts IS NOT NULL
  AND (julianday(ts) - julianday(prev_ts)) * 24 > 1.5
ORDER BY missing_hours DESC;

-- name: q04_flag_share_having
-- desc: Share of flagged records per well and parameter (LEFT JOIN + GROUP BY + HAVING > 1 %)
SELECT m.well_id,
       m.parameter,
       COUNT(*)                                                   AS n_rows,
       COUNT(f.measurement_id)                                    AS n_flagged,
       ROUND(100.0 * COUNT(f.measurement_id) / COUNT(*), 2)       AS flagged_pct,
       GROUP_CONCAT(DISTINCT f.flag)                              AS flags
FROM measurements m
LEFT JOIN measurement_flags f ON f.measurement_id = m.measurement_id
GROUP BY m.well_id, m.parameter
HAVING flagged_pct > 1
ORDER BY flagged_pct DESC;

-- name: q05_monthly_zero_spm_having
-- desc: Monthly share of hours with SPM = 0 per sucker-rod pump well (GROUP BY + HAVING)
SELECT well_id,
       substr(ts, 1, 7)                                      AS month,
       COUNT(*)                                              AS hours,
       SUM(CASE WHEN value = 0 THEN 1 ELSE 0 END)            AS zero_hours,
       ROUND(100.0 * SUM(CASE WHEN value = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS zero_pct
FROM measurements
WHERE parameter = 'SPM'
GROUP BY well_id, month
HAVING zero_pct >= 2
ORDER BY zero_pct DESC;

-- name: q06_monthly_rank_cte
-- desc: CTE + RANK(): ranking of wells by monthly mean of their primary signal within each month and well type
WITH monthly AS (
    SELECT h.well_id,
           w.well_type,
           substr(h.ts, 1, 7) AS month,
           AVG(CASE WHEN w.well_type = 'fountain' THEN h.tubing_pressure ELSE h.spm END) AS primary_mean
    FROM hourly_wide h
    JOIN wells w ON w.well_id = h.well_id
    WHERE h.has_data = 1
    GROUP BY h.well_id, w.well_type, month
)
SELECT month, well_type, well_id,
       ROUND(primary_mean, 2) AS primary_mean,
       RANK() OVER (PARTITION BY month, well_type ORDER BY primary_mean DESC) AS rank_in_month,
       ROUND(primary_mean - LAG(primary_mean) OVER (PARTITION BY well_id ORDER BY month), 2)
           AS change_vs_prev_month
FROM monthly
ORDER BY month, well_type, rank_in_month;

-- name: q07_wells_above_field_avg_subquery
-- desc: Subquery: sucker-rod pump wells whose mean pump fillage is above the field average
SELECT well_id,
       ROUND(AVG(value), 2) AS mean_fillage,
       (SELECT ROUND(AVG(value), 2) FROM measurements WHERE parameter = 'pump_fillage') AS field_mean
FROM measurements
WHERE parameter = 'pump_fillage'
GROUP BY well_id
HAVING AVG(value) > (SELECT AVG(value) FROM measurements WHERE parameter = 'pump_fillage')
ORDER BY mean_fillage DESC;

-- name: q08_events_by_well_month
-- desc: Shut-in frequency and duration per well and month (reference events, GROUP BY)
SELECT e.well_id,
       w.well_type,
       substr(e.start_ts, 1, 7)       AS month,
       COUNT(*)                       AS n_events,
       SUM(e.duration_h)              AS downtime_h,
       ROUND(AVG(e.duration_h), 1)    AS mean_duration_h,
       MAX(e.duration_h)              AS max_duration_h
FROM shut_in_events e
JOIN wells w ON w.well_id = e.well_id
WHERE e.method = 'reference'
GROUP BY e.well_id, w.well_type, month
ORDER BY e.well_id, month;

-- name: q09_time_between_events
-- desc: CTE + LAG(): hours of operation between consecutive shut-ins (mean time between shut-ins per well)
WITH ordered AS (
    SELECT well_id, start_ts, end_ts, duration_h,
           LAG(end_ts) OVER (PARTITION BY well_id ORDER BY start_ts) AS prev_end
    FROM shut_in_events
    WHERE method = 'reference'
)
SELECT well_id,
       COUNT(*)                                                         AS n_events,
       ROUND(AVG((julianday(start_ts) - julianday(prev_end)) * 24), 1)  AS mean_hours_between,
       ROUND(MIN((julianday(start_ts) - julianday(prev_end)) * 24), 1)  AS min_hours_between,
       ROUND(AVG(duration_h), 1)                                        AS mean_duration_h
FROM ordered
GROUP BY well_id
ORDER BY n_events DESC;

-- name: q10_availability
-- desc: Well availability: shut-in hours vs hours with data (two aggregated subqueries joined)
SELECT d.well_id,
       d.hours_with_data,
       COALESCE(e.shut_in_hours, 0)                                         AS shut_in_hours,
       ROUND(100.0 * (d.hours_with_data - COALESCE(e.shut_in_hours, 0)) / d.hours_with_data, 2)
           AS availability_pct
FROM (SELECT well_id, SUM(has_data) AS hours_with_data FROM hourly_wide GROUP BY well_id) d
LEFT JOIN (SELECT well_id, SUM(duration_h) AS shut_in_hours
           FROM shut_in_events WHERE method = 'reference' GROUP BY well_id) e
       ON e.well_id = d.well_id
ORDER BY availability_pct;

-- name: q11_rolling_24h_primary
-- desc: Window frame: 24-hour rolling mean of the primary signal and drops below 25 % of it (daily summary)
WITH base AS (
    SELECT h.well_id, h.ts,
           CASE WHEN w.well_type = 'fountain' THEN h.tubing_pressure ELSE h.spm END AS primary_value
    FROM hourly_wide h JOIN wells w ON w.well_id = h.well_id
    WHERE h.has_data = 1
),
rolled AS (
    SELECT well_id, ts, primary_value,
           AVG(primary_value) OVER (PARTITION BY well_id ORDER BY ts
                                    ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING) AS rolling_24h
    FROM base
)
SELECT well_id,
       COUNT(*)                                                                AS hours,
       SUM(CASE WHEN primary_value < 0.25 * rolling_24h THEN 1 ELSE 0 END)     AS sharp_drops,
       ROUND(AVG(rolling_24h), 2)                                              AS mean_rolling_24h
FROM rolled
GROUP BY well_id
ORDER BY sharp_drops DESC;

-- name: q12_method_comparison
-- desc: Events per detection method and split (JOIN with method catalogue + GROUP BY)
SELECT dm.method,
       e.split,
       COUNT(e.event_id)            AS n_events,
       SUM(e.duration_h)            AS downtime_h,
       ROUND(AVG(e.duration_h), 1)  AS mean_duration_h
FROM detection_methods dm
LEFT JOIN shut_in_events e ON e.method = dm.method
GROUP BY dm.method, e.split
ORDER BY e.split, dm.method;

-- name: q13_shutin_start_hour
-- desc: Hour-of-day distribution of shut-in starts by well type (daily pattern)
SELECT w.well_type,
       CAST(substr(e.start_ts, 12, 2) AS INT) AS start_hour,
       COUNT(*)                               AS n_events
FROM shut_in_events e
JOIN wells w ON w.well_id = e.well_id
WHERE e.method = 'reference'
GROUP BY w.well_type, start_hour
ORDER BY w.well_type, start_hour;
