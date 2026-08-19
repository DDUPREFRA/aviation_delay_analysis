-- ============================================================
-- 0. VERIFY CONNECTION AND TABLE
-- ============================================================

-- Tableau analysis uses complete calendar years only.
-- The analysis includes 2019 and 2022-2025.
-- Data for 2020 and 2021 were not downloaded.
-- Incomplete 2026 is excluded from the dashboard results.

SELECT
    current_database() AS connected_database,
    current_user AS connected_user;

SELECT
    COUNT(*) AS total_rows
FROM flights;


-- ============================================================
-- 1. AIRLINE ANALYSIS
-- ============================================================

-- 1. Airline delay rate.
SELECT
    airline_name,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY airline_name
HAVING COUNT(*) > 1000000
ORDER BY delay_rate_pct DESC;


-- 2. Airline delay volume.
SELECT
    airline_name,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY airline_name
HAVING COUNT(*) > 1000000
ORDER BY delayed_flights DESC;

-- 3. Airline departure delay severity among delayed flights.
SELECT
    airline_name,
    COUNT(*) AS delayed_flights,
    ROUND(CAST(AVG(dep_delay) AS NUMERIC), 2) AS avg_dep_delay_when_delayed,
    ROUND(CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY dep_delay) AS NUMERIC), 2) AS median_dep_delay_when_delayed,
    ROUND(CAST(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY dep_delay) AS NUMERIC), 2) AS p90_dep_delay_when_delayed
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND is_delayed = 1
  AND year < 2026
GROUP BY airline_name
HAVING COUNT(*) > 100000
ORDER BY median_dep_delay_when_delayed DESC;


-- 4. Airline arrival delay severity among delayed flights.
SELECT
    airline_name,
    COUNT(*) AS delayed_flights,
    ROUND(CAST(AVG(arr_delay) AS NUMERIC), 2) AS avg_arr_delay_when_delayed,
    ROUND(CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY arr_delay) AS NUMERIC), 2) AS median_arr_delay_when_delayed,
    ROUND(CAST(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY arr_delay) AS NUMERIC), 2) AS p90_arr_delay_when_delayed
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND is_delayed = 1
  AND year < 2026
GROUP BY airline_name
HAVING COUNT(*) > 100000
ORDER BY median_arr_delay_when_delayed DESC;




-- ============================================================
-- 3. AIRPORT ANALYSIS
-- ============================================================

-- 6. Delay rate by origin airport.
SELECT
    origin,
    origin_city_name,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct,
    ROUND(CAST(AVG(dep_delay) AS NUMERIC), 2) AS avg_dep_delay
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY origin, origin_city_name
HAVING COUNT(*) >= 200000
ORDER BY delay_rate_pct DESC;


-- ============================================================
-- 4. ROUTE ANALYSIS
-- ============================================================

-- 8. Delay rate by route.
SELECT
    origin_city_name,
    dest_city_name,
    route,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY origin_city_name, dest_city_name, route
HAVING COUNT(*) >= 50000
ORDER BY delay_rate_pct DESC;


-- ============================================================
-- 5. TIME ANALYSIS
-- ============================================================

-- 10. Delay rate by month.
SELECT
    month,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY month
ORDER BY month;


-- 11. Delay rate by quarter.
SELECT
    quarter,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY quarter
ORDER BY quarter;


-- 12. Delay rate by day of week.
SELECT
    day_of_week,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY day_of_week
ORDER BY day_of_week;


-- 13. Delay rate by scheduled departure hour.
SELECT
    dep_hour,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY dep_hour
ORDER BY dep_hour;


-- 14. Peak vs off-peak delay rate.
SELECT
    is_peak_hour,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY is_peak_hour
ORDER BY is_peak_hour;


-- 15. Weekday vs weekend delay rate.
SELECT
    is_weekend,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY is_weekend
ORDER BY is_weekend;


-- ============================================================
-- 6. DELAY CAUSE ANALYSIS
-- ============================================================

-- 16. Most common reported delay causes among delayed operated flights.
-- Select the cause with the most positive reported minutes for each flight.
-- If every cause-specific value is zero, label the record separately instead
-- of allowing the first cause column to win an all-zero tie.
WITH labeled_delays AS (
    SELECT
        CASE
            WHEN selected_cause.cause_minutes > 0
                THEN selected_cause.main_delay_cause
            ELSE 'no_reported_cause'
        END AS main_delay_cause
    FROM flights AS f
    CROSS JOIN LATERAL (
        SELECT
            cause_values.main_delay_cause,
            cause_values.cause_minutes
        FROM (
            VALUES
                (1, 'carrier_delay', COALESCE(f.carrier_delay, 0)),
                (2, 'weather_delay', COALESCE(f.weather_delay, 0)),
                (3, 'nas_delay', COALESCE(f.nas_delay, 0)),
                (4, 'security_delay', COALESCE(f.security_delay, 0)),
                (5, 'late_aircraft_delay', COALESCE(f.late_aircraft_delay, 0))
        ) AS cause_values(cause_order, main_delay_cause, cause_minutes)
        ORDER BY cause_values.cause_minutes DESC, cause_values.cause_order
        LIMIT 1
    ) AS selected_cause
    WHERE f.cancelled = 0
      AND f.diverted = 0
      AND f.is_delayed = 1
      AND f.year < 2026
),
cause_counts AS (
    SELECT
        main_delay_cause,
        COUNT(*) AS delayed_flights
    FROM labeled_delays
    GROUP BY main_delay_cause
)
SELECT
    main_delay_cause,
    delayed_flights,
    ROUND(
        CAST(
            delayed_flights * 100.0
            / NULLIF(SUM(delayed_flights) OVER (), 0)
            AS NUMERIC
        ),
        2
    ) AS pct_of_delayed_flights
FROM cause_counts
ORDER BY delayed_flights DESC;


-- 17. Delay severity by distance group among delayed operated flights.
SELECT
    CASE
        WHEN distance < 500 THEN '1. Short (<500 mi)'
        WHEN distance < 1000 THEN '2. Medium (500-999 mi)'
        WHEN distance < 2000 THEN '3. Long (1000-1999 mi)'
        ELSE '4. Very Long (2000+ mi)'
    END AS distance_group,
    COUNT(*) AS delayed_flights,
    ROUND(CAST(AVG(dep_delay) AS NUMERIC), 2) AS avg_dep_delay,
    ROUND(CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY dep_delay) AS NUMERIC), 2) AS median_dep_delay,
    ROUND(CAST(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY dep_delay) AS NUMERIC), 2) AS p90_dep_delay,
    ROUND(CAST(AVG(arr_delay) AS NUMERIC), 2) AS avg_arr_delay,
    ROUND(CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY arr_delay) AS NUMERIC), 2) AS median_arr_delay,
    ROUND(CAST(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY arr_delay) AS NUMERIC), 2) AS p90_arr_delay
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND is_delayed = 1
  AND year < 2026
GROUP BY distance_group
ORDER BY distance_group;


-- 18. Delay rate by distance group.
SELECT
    CASE
        WHEN distance < 500 THEN '1. Short (<500 mi)'
        WHEN distance < 1000 THEN '2. Medium (500-999 mi)'
        WHEN distance < 2000 THEN '3. Long (1000-1999 mi)'
        ELSE '4. Very Long (2000+ mi)'
    END AS distance_group,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY distance_group
ORDER BY distance_group;


-- ============================================================
-- 7. YEAR-OVER-YEAR ANALYSIS
-- ============================================================

-- 19. Overall delay rate by year.
SELECT
    year,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY year
ORDER BY year;


-- 21. Delay rate by airline and year.
SELECT
    year,
    airline_name,
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS delay_rate_pct
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026
GROUP BY year, airline_name
HAVING COUNT(*) >= 50000
ORDER BY airline_name, year;


-- ============================================================
-- 8. DASHBOARD KPI SUMMARY
-- ============================================================

-- 23. Flight volume by complete year.
SELECT
    year,
    COUNT(*) AS scheduled_flights,
    SUM(CASE WHEN cancelled = 0 AND diverted = 0 THEN 1 ELSE 0 END) AS operated_non_diverted_flights,
    SUM(cancelled) AS cancelled_flights,
    SUM(diverted) AS diverted_flights
FROM flights
WHERE year < 2026
GROUP BY year
ORDER BY year;


-- 24. Overview KPI card for operated, non-diverted flights.
SELECT
    COUNT(*) AS total_operated_flights,
    SUM(is_delayed) AS total_delayed_flights,
    ROUND(CAST(SUM(is_delayed) * 100.0 / NULLIF(COUNT(*), 0) AS NUMERIC), 2) AS overall_delay_rate_pct,
    COUNT(DISTINCT airline_name) AS total_airlines,
    COUNT(DISTINCT origin) AS total_origin_airports,
    COUNT(DISTINCT route) AS total_routes,
    ROUND(CAST(AVG(dep_delay) FILTER (WHERE is_delayed = 1) AS NUMERIC), 2) AS avg_dep_delay_when_delayed,
    ROUND(CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY dep_delay) FILTER (WHERE is_delayed = 1) AS NUMERIC), 2) AS median_dep_delay_when_delayed
FROM flights
WHERE cancelled = 0
  AND diverted = 0
  AND year < 2026;
