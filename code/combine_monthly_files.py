"""
combine_monthly_files.py

WHAT THIS SCRIPT DOES:
1. Uses the expected monthly files named like raw_data/2019/2019:01.csv.
2. Keeps only the columns used in the project.
3. Combines all monthly files into one large raw file.
4. Sorts the combined data by flight date.
5. Saves the result as raw_data/flights_all_raw.csv.

This script does NOT clean the data. It only merges the monthly files downloaded from BTS.
"""


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder where the monthly raw CSV files are stored.
RAW_DIR = PROJECT_ROOT / "raw_data"

# Main combined raw file created by this script.
OUTPUT_PATH = RAW_DIR / "flights_all_raw.csv"

# Temporary output file used while saving.
# The script saves here first, then replaces OUTPUT_PATH after saving succeeds.
TEMP_OUTPUT_PATH = RAW_DIR / "flights_all_raw_temp.csv"

# These are the columns we keep from each monthly file.
# TAIL_NUM is kept in the raw/cleaned data, but it is not used in the
# simple classification model.
KEEP_COLS = [
    "YEAR",
    "QUARTER",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "TAIL_NUM",
    "ORIGIN",
    "ORIGIN_CITY_NAME",
    "ORIGIN_STATE_ABR",
    "DEST",
    "DEST_CITY_NAME",
    "DEST_STATE_ABR",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "DEP_DELAY",
    "DEP_DELAY_NEW",
    "DEP_DEL15",
    "ARR_DELAY",
    "ARR_DELAY_NEW",
    "ARR_DEL15",
    "CANCELLED",
    "CANCELLATION_CODE",
    "DIVERTED",
    "AIR_TIME",
    "DISTANCE",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]


def print_section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 3. LIST EXPECTED MONTHLY FILES
# ============================================================

# These are the exact monthly files used in the project.
# They already exist in the raw_data folder.
# This list does not create files; it only tells Python which files to read.
files = [
    "raw_data/2019/2019:01.csv",
    "raw_data/2019/2019:02.csv",
    "raw_data/2019/2019:03.csv",
    "raw_data/2019/2019:04.csv",
    "raw_data/2019/2019:05.csv",
    "raw_data/2019/2019:06.csv",
    "raw_data/2019/2019:07.csv",
    "raw_data/2019/2019:08.csv",
    "raw_data/2019/2019:09.csv",
    "raw_data/2019/2019:10.csv",
    "raw_data/2019/2019:11.csv",
    "raw_data/2019/2019:12.csv",
    "raw_data/2022/2022:01.csv",
    "raw_data/2022/2022:02.csv",
    "raw_data/2022/2022:03.csv",
    "raw_data/2022/2022:04.csv",
    "raw_data/2022/2022:05.csv",
    "raw_data/2022/2022:06.csv",
    "raw_data/2022/2022:07.csv",
    "raw_data/2022/2022:08.csv",
    "raw_data/2022/2022:09.csv",
    "raw_data/2022/2022:10.csv",
    "raw_data/2022/2022:11.csv",
    "raw_data/2022/2022:12.csv",
    "raw_data/2023/2023:01.csv",
    "raw_data/2023/2023:02.csv",
    "raw_data/2023/2023:03.csv",
    "raw_data/2023/2023:04.csv",
    "raw_data/2023/2023:05.csv",
    "raw_data/2023/2023:06.csv",
    "raw_data/2023/2023:07.csv",
    "raw_data/2023/2023:08.csv",
    "raw_data/2023/2023:09.csv",
    "raw_data/2023/2023:10.csv",
    "raw_data/2023/2023:11.csv",
    "raw_data/2023/2023:12.csv",
    "raw_data/2024/2024:01.csv",
    "raw_data/2024/2024:02.csv",
    "raw_data/2024/2024:03.csv",
    "raw_data/2024/2024:04.csv",
    "raw_data/2024/2024:05.csv",
    "raw_data/2024/2024:06.csv",
    "raw_data/2024/2024:07.csv",
    "raw_data/2024/2024:08.csv",
    "raw_data/2024/2024:09.csv",
    "raw_data/2024/2024:10.csv",
    "raw_data/2024/2024:11.csv",
    "raw_data/2024/2024:12.csv",
    "raw_data/2025/2025:01.csv",
    "raw_data/2025/2025:02.csv",
    "raw_data/2025/2025:03.csv",
    "raw_data/2025/2025:04.csv",
    "raw_data/2025/2025:05.csv",
    "raw_data/2025/2025:06.csv",
    "raw_data/2025/2025:07.csv",
    "raw_data/2025/2025:08.csv",
    "raw_data/2025/2025:09.csv",
    "raw_data/2025/2025:10.csv",
    "raw_data/2025/2025:11.csv",
    "raw_data/2025/2025:12.csv",
    "raw_data/2026/2026:01.csv",
    "raw_data/2026/2026:02.csv",
    "raw_data/2026/2026:03.csv",
]

print_section("MERGE RAW MONTHLY FILES")
print(f"Input folder: {RAW_DIR.relative_to(PROJECT_ROOT)}")
print(f"Expected monthly files: {len(files)}")

# Check that all monthly files exist before merging.
missing_files = []

for file_path in files:
    full_path = PROJECT_ROOT / file_path

    if not full_path.exists():
        missing_files.append(file_path)

if missing_files:
    missing_names = "\n".join(missing_files)

    raise FileNotFoundError(
        "Some expected monthly files are missing:\n"
        f"{missing_names}"
    )


# ============================================================
# 4. LOAD EACH MONTHLY FILE
# ============================================================

dfs = []
source_row_count = 0
date_col_candidates = ["FlightDate", "FL_DATE", "fl_date"]

for file_path in files:
    filename = file_path
    full_path = PROJECT_ROOT / file_path

    # Read one month at a time.
    # Keeps the code simple and lets us print progress month by month.
    temp = pd.read_csv(full_path, low_memory=False)

    # Stop early if a file does not contain one of the expected columns.
    missing_cols = []

    for col in KEEP_COLS:
        if col not in temp.columns:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"{filename} is missing columns: {missing_cols}")

    # Keep the selected columns in the same order for every month.
    temp = temp[KEEP_COLS]

    print(f"Loaded {filename}: {len(temp):,} rows")
    source_row_count += len(temp)

    date_column_found = False

    for col in date_col_candidates:
        if col in temp.columns:
            date_column_found = True

    if not date_column_found:
        print(f"  WARNING: no date column found in {filename}")

    dfs.append(temp)


# ============================================================
# 5. COMBINE AND SORT
# ============================================================

print()
print("Combining monthly files...")

# Stack all monthly dataframes into one large dataframe.
# ignore_index=True gives the final combined file a clean row index.
df = pd.concat(dfs, ignore_index=True)

# Find the date column name used in the source files.
date_col = None

for col in date_col_candidates:
    if col in df.columns:
        date_col = col

if date_col is None:
    raise KeyError("No date column found. Expected one of: FlightDate, FL_DATE, fl_date.")

# Sort by date so the combined raw file is in chronological order.
df = df.sort_values(date_col).reset_index(drop=True)

# This check protects against accidental row loss during the merge.
if len(df) != source_row_count:
    raise ValueError(
        f"Row count mismatch: source files had {source_row_count:,} rows, "
        f"but combined data has {len(df):,} rows."
    )


# ============================================================
# 6. SAVE COMBINED RAW FILE
# ============================================================

# Save to a temporary file first. If the script is interrupted, it will not
# replace the good output file with a half-written one.
df.to_csv(TEMP_OUTPUT_PATH, index=False)
TEMP_OUTPUT_PATH.replace(OUTPUT_PATH)

print_section("MERGE SUMMARY")
print(f"Files merged: {len(files)}")
print(f"Rows merged:  {df.shape[0]:,}")
print(f"Columns kept: {df.shape[1]}")
print(f"Output file:  {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
