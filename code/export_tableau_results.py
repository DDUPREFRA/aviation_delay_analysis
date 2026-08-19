"""
export_tableau_results.py

WHAT THIS SCRIPT DOES:
1. Reads the numbered queries from aviation_delay_tableau_queries.sql.
2. Runs the queries against the existing PostgreSQL flights table.
3. Replaces the matching Tableau CSV exports only after each query succeeds.
4. Prints the output path and row count for every refreshed file.
"""


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import csv
import os
import re
from pathlib import Path

import psycopg2


# ============================================================
# 2. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SQL_PATH = PROJECT_ROOT / "sql" / "aviation_delay_tableau_queries.sql"
EXPORT_DIR = PROJECT_ROOT / "sql_exports"

DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "flight_delays")


# ============================================================
# 3. TABLEAU QUERY-TO-FILE MAPPING
# ============================================================

OUTPUT_FILES = {
    1: "03_01_airline_delay_rate.csv",
    2: "04_02_airline_delay_volume.csv",
    3: "05_03_airline_departure_delay_severity_among_delayed_flights.csv",
    4: "06_04_airline_arrival_delay_severity_among_delayed_flights.csv",
    6: "08_06_delay_rate_by_origin_airport.csv",
    8: "09_08_delay_rate_by_route.csv",
    10: "10_10_delay_rate_by_month.csv",
    11: "11_11_delay_rate_by_quarter.csv",
    12: "12_12_delay_rate_by_day_of_week.csv",
    13: "13_13_delay_rate_by_scheduled_departure_hour.csv",
    14: "14_14_peak_vs_off_peak_delay_rate.csv",
    15: "15_15_weekday_vs_weekend_delay_rate.csv",
    16: "16_16_most_common_delay_causes_among_delayed_operated_flights.csv",
    17: "17_17_delay_severity_by_distance_group_among_delayed_operated_flights.csv",
    18: "18_18_delay_rate_by_distance_group.csv",
    19: "19_19_overall_delay_rate_by_year.csv",
    21: "20_21_delay_rate_by_airline_and_year.csv",
    23: "21_23_flight_volume_by_year.csv",
    24: "22_24_overview_kpi_card_for_operated_non_diverted_flights.csv",
}


# ============================================================
# 4. READ THE NUMBERED SQL QUERIES
# ============================================================

sql_text = SQL_PATH.read_text(encoding="utf-8")

query_pattern = re.compile(
    r"^-- (?P<number>\d+)\. [^\n]+\n"
    r"(?P<query>SELECT\b.*?;)",
    flags=re.MULTILINE | re.DOTALL,
)

queries = {
    int(match.group("number")): match.group("query")
    for match in query_pattern.finditer(sql_text)
}

missing_queries = sorted(set(OUTPUT_FILES) - set(queries))

if missing_queries:
    raise RuntimeError(
        f"Missing numbered SQL queries: {missing_queries}"
    )


# ============================================================
# 5. CONNECT TO POSTGRESQL
# ============================================================

print("=" * 60)
print("REFRESH TABLEAU SQL EXPORTS")
print("=" * 60)
print(f"SQL file:    {SQL_PATH}")
print(f"Output folder: {EXPORT_DIR}")

connection = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)


# ============================================================
# 6. MAKE SURE AIRLINE NAMES ARE AVAILABLE
# ============================================================

# The Python database loader replaces the flights table whenever
# the cleaned CSV is reloaded. Restore the readable airline-name
# column if the replacement table does not contain it.
print()
print("Checking airline-name column...", flush=True)

with connection.cursor() as cursor:
    cursor.execute(
        """
        ALTER TABLE flights
        ADD COLUMN IF NOT EXISTS airline_name VARCHAR(100);
        """
    )

    cursor.execute(
        """
        UPDATE flights
        SET airline_name = CASE op_unique_carrier
            WHEN 'AA' THEN 'American Airlines'
            WHEN 'AS' THEN 'Alaska Airlines'
            WHEN 'B6' THEN 'JetBlue Airways'
            WHEN 'DL' THEN 'Delta Air Lines'
            WHEN 'F9' THEN 'Frontier Airlines'
            WHEN 'G4' THEN 'Allegiant Air'
            WHEN 'HA' THEN 'Hawaiian Airlines'
            WHEN 'NK' THEN 'Spirit Airlines'
            WHEN 'OH' THEN 'PSA Airlines'
            WHEN 'OO' THEN 'SkyWest Airlines'
            WHEN 'QX' THEN 'Horizon Air'
            WHEN 'UA' THEN 'United Airlines'
            WHEN 'WN' THEN 'Southwest Airlines'
            WHEN 'YV' THEN 'Mesa Airlines'
            WHEN 'YX' THEN 'Republic Airways'
            WHEN 'MQ' THEN 'Envoy Air'
            WHEN '9E' THEN 'Endeavor Air'
            WHEN 'PT' THEN 'Piedmont Airlines'
            WHEN 'EV' THEN 'ExpressJet Airlines'
            ELSE op_unique_carrier
        END
        WHERE airline_name IS NULL
           OR airline_name = '';
        """
    )

connection.commit()
print("Airline-name column is ready.", flush=True)


# ============================================================
# 7. RUN QUERIES AND REPLACE CSV FILES
# ============================================================

EXPORT_DIR.mkdir(parents=True, exist_ok=True)

try:
    with connection.cursor() as cursor:
        for query_number, output_name in OUTPUT_FILES.items():
            output_path = EXPORT_DIR / output_name
            temporary_path = output_path.with_suffix(".csv.tmp")

            print()
            print(f"Running query {query_number}: {output_name}", flush=True)

            cursor.execute(queries[query_number])
            rows = cursor.fetchall()
            headers = [column.name for column in cursor.description]

            with temporary_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as output_file:
                writer = csv.writer(output_file)
                writer.writerow(headers)
                writer.writerows(rows)

            temporary_path.replace(output_path)

            print(f"Saved rows: {len(rows):,}", flush=True)
            print(f"Saved file: {output_path}", flush=True)

finally:
    connection.close()


# ============================================================
# 8. COMPLETION SUMMARY
# ============================================================

print()
print("=" * 60)
print("TABLEAU EXPORT REFRESH COMPLETE")
print("=" * 60)
print(f"Files refreshed: {len(OUTPUT_FILES)}")
