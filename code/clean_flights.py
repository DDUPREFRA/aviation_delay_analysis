"""
clean_flights.py

WHAT THIS SCRIPT DOES:
1. Loads the merged raw flight file.
2. Standardizes column names to snake_case.
3. Fixes dates and numeric columns.
4. Removes cancelled/diverted flights from the modeling dataset.
5. Checks outliers but does not remove them by default.
6. Creates simple modeling/EDA features.
7. Saves processed/flights_all_cleaned.csv.
"""


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
from pathlib import Path

import pandas as pd


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main file used as input for the cleaning step.
# This file is created by combine_monthly_files.py.
INPUT_PATH = PROJECT_ROOT / "raw_data" / "flights_all_raw.csv"

# Main cleaned file created by this script.
# The EDA, forecasting, classification, SQL, and PostgreSQL scripts use this file.
OUTPUT_PATH = PROJECT_ROOT / "processed" / "flights_all_cleaned.csv"

# Temporary output file used while saving.
# The script saves here first, then replaces OUTPUT_PATH after saving succeeds.
TEMP_OUTPUT_PATH = PROJECT_ROOT / "processed" / "flights_all_cleaned_temp.csv"

# These columns must exist for the project to work.
# If one is missing, the script stops and tells us immediately.
REQUIRED_COLS = [
    "fl_date",
    "cancelled",
    "diverted",
    "dep_delay",
    "arr_delay",
    "crs_dep_time",
    "day_of_week",
    "tail_num",
    "origin",
    "dest",
]

# These columns should be numeric.
# pd.to_numeric later makes sure they behave like numbers, not text.
NUMERIC_COLS = [
    "cancelled",
    "diverted",
    "dep_del15",
    "arr_del15",
    "dep_delay",
    "dep_delay_new",
    "arr_delay",
    "arr_delay_new",
    "air_time",
    "distance",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
    "crs_dep_time",
    "day_of_week",
]

# BTS provides these delay-cause columns.
# They are useful for EDA/SQL, but not used as classification predictors.
DELAY_CAUSE_COLS = [
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]


def print_section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 3. LOAD DATA
# ============================================================

# Check that the merged raw file exists before trying to clean it.
# If it does not exist, stop the script and tell the user what to run first.
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Combined raw file not found: {INPUT_PATH}\n"
        
    )

print_section("CLEAN FLIGHT DATA")
print(f"Input file: {INPUT_PATH}", flush=True)

# Load the merged raw file created by combine_monthly_files.py into pandas dataframe.
df = pd.read_csv(INPUT_PATH, low_memory=False)

# Save the starting number of rows and columns.
starting_rows = len(df)
starting_cols = len(df.columns)


# ============================================================
# 4. STANDARDIZE COLUMN NAMES
# ============================================================

# Convert column names to snake_case.
# Example: FL_DATE becomes fl_date.
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)


# ============================================================
# 5. BASIC INSPECTION
# ============================================================

# Save the biggest missing-value counts before cleaning.
# These are printed at the end in the cleaning summary.
initial_missing = df.isna().sum().sort_values(ascending=False).head(10)


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

# Start with an empty list.
# Any required column that is missing will be added to this list and printed.
missing_required = []

for col in REQUIRED_COLS:
    if col not in df.columns:
        missing_required.append(col)

if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")


# ============================================================
# 7. REMOVE DUPLICATES
# ============================================================

# Count exact duplicate rows before removing them.
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates().copy()


# ============================================================
# 8. FIX DATA TYPES
# ============================================================

# Convert the flight date column into a real datetime column.
df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce", format="mixed")

# Convert numeric columns from text/object into numeric values.
# Bad values become NaN, then the missing-value section handles them.
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 9. FILTER ROWS NOT USED IN ANALYSIS
# ============================================================

# Cancelled and diverted flights are removed because they do not have a normal
# completed-flight delay outcome for this analysis.
# We keep this dataset focused on flights that actually completed normally.
for col in ["cancelled", "diverted"]:
    df[col] = df[col].fillna(0).astype(int)

before = len(df)

completed_flights = df["cancelled"] == 0
not_diverted = df["diverted"] == 0

df = df[completed_flights & not_diverted].copy()
cancelled_diverted_removed = before - len(df)


# ============================================================
# 10. HANDLE MISSING VALUES
# ============================================================

# Fill missing delay flags with 0.
# After cancelled/diverted flights are removed, missing usually means not delayed.
for col in ["dep_del15", "arr_del15"]:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype(int)

# Fill missing delay-cause minutes with 0.
# If no cause is listed, that cause added 0 minutes.
for col in DELAY_CAUSE_COLS:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Fill missing delay-minute columns with 0 for completed flights.
for col in ["dep_delay", "dep_delay_new", "arr_delay", "arr_delay_new", "air_time"]:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Non-cancelled flights do not have a cancellation code.
if "cancellation_code" in df.columns:
    df["cancellation_code"] = df["cancellation_code"].fillna("not_cancelled")

# Keep tail_num, but label missing tail numbers as unknown.
if "tail_num" in df.columns:
    df["tail_num"] = df["tail_num"].fillna("unknown")

# Drop rows with invalid or missing flight dates.
df = df.dropna(subset=["fl_date"])


# ============================================================
# 11. CHECK OUTLIERS
# ============================================================

# Create an empty list to store the outlier count for each column.
outlier_rows = []

# Check outliers for the main numeric columns we care about.
for col in ["arr_delay", "dep_delay", "distance"]:

    # Only run the check if the column exists in the dataframe.
    if col in df.columns:

        # IQR outliers are counted for reporting only.
        # They are not removed because long delays are real flight events.
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        # IQR means interquartile range.
        # It is the distance between the 25th percentile and 75th percentile.
        iqr = q3 - q1

        # These are the standard IQR outlier fences.
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Mark rows that are below or above the fences.
        below_lower_fence = df[col] < lower
        above_upper_fence = df[col] > upper

        # Count how many rows are outside the fences.
        count = (below_lower_fence | above_upper_fence).sum()

        # Save the result so it can be printed in the cleaning summary.
        outlier_rows.append({"column": col, "iqr_outliers": count})

        # Optional: remove outliers
        # df = df[(df[col] >= lower) & (df[col] <= upper)].copy()

        # Optional: cap outliers instead of removing them
        # df[col] = df[col].clip(lower, upper)


# ============================================================
# 12. CREATE FEATURES
# ============================================================

# Main classification target:
# 1 means the flight departed 15+ minutes late.
# 0 means the flight departed less than 15 minutes late.
departure_delay_is_15_plus = df["dep_delay"] >= 15

# Convert True/False into 1/0 so machine learning models can use it.
df["is_delayed"] = departure_delay_is_15_plus.astype(int)

# Create a delay category for EDA and SQL analysis.
# This gives more detail than just delayed vs not delayed.
df["delay_category"] = pd.cut(
    df["dep_delay"],

    # These are the delay-minute ranges.
    # -inf to 0: early or exactly on time
    # 0 to 15: small delay
    # 15 to 60: moderate delay
    # 60 to 120: severe delay
    # 120+: extreme delay
    bins=[-float("inf"), 0, 15, 60, 120, float("inf")],

    # These labels match the ranges above.
    labels=[
        "early_on_time",
        "minor_1_15",
        "moderate_16_60",
        "severe_61_120",
        "extreme_120_plus",
    ],
)

# Create date-based features from the flight date.
# These are useful for EDA and can also be used in models if needed.
df["day_of_year"] = df["fl_date"].dt.dayofyear
df["week_of_year"] = df["fl_date"].dt.isocalendar().week.astype(int)

# Create departure hour from scheduled departure time.
# Example: 1345 becomes 13.
# We use scheduled time instead of actual departure time to avoid data leakage.
df["dep_hour"] = (df["crs_dep_time"] // 100).clip(0, 23).astype(int)

# Create a peak-hour flag.
# Morning peak: 5 AM to 9 AM.
# Evening peak: 4 PM to 7 PM.
morning_peak = df["dep_hour"].between(5, 9)
evening_peak = df["dep_hour"].between(16, 19)

# Convert True/False into 1/0.
df["is_peak_hour"] = (morning_peak | evening_peak).astype(int)

# Create a weekend flag.
# BTS day_of_week uses 1 = Monday and 7 = Sunday.
# So 6 and 7 are weekend days.
weekend_day = df["day_of_week"] >= 6
df["is_weekend"] = weekend_day.astype(int)

# Create a route column by combining origin and destination.
# Example: JFK_LAX.
df["route"] = df["origin"] + "_" + df["dest"]

# Pick the reported delay-cause column with the largest positive value.
# Some departure-delayed flights have no positive cause-specific minutes.
# Label those records separately instead of assigning the first tied column.
reported_cause_minutes = df[DELAY_CAUSE_COLS]
has_reported_cause = reported_cause_minutes.max(axis=1) > 0

df["main_delay_cause"] = "no_reported_cause"
df.loc[has_reported_cause, "main_delay_cause"] = (
    reported_cause_minutes.loc[has_reported_cause].idxmax(axis=1)
)

# Flights departing fewer than 15 minutes late do not belong to the
# departure-delay cause analysis.
not_delayed = df["dep_delay"] < 15
df.loc[not_delayed, "main_delay_cause"] = "no_delay"

# ============================================================
# 13. FINAL VALIDATION
# ============================================================

# Sort one final time so the cleaned data is in date order.
df = df.sort_values("fl_date").reset_index(drop=True)

# Save the biggest remaining missing-value counts for the final summary.
final_missing = df.isna().sum().sort_values(ascending=False).head(10)


# ============================================================
# 14. SAVE CLEANED DATA
# ============================================================

# Save to a temporary file first, then replace the final CSV safely.
df.to_csv(TEMP_OUTPUT_PATH, index=False)
os.replace(TEMP_OUTPUT_PATH, OUTPUT_PATH)

# Print the final cleaning report.
print_section("CLEANING SUMMARY")

# Show how many rows/columns came in and how many are left.
print(f"Input rows:                  {starting_rows:,}")
print(f"Input columns:               {starting_cols}")
print(f"Duplicate rows removed:      {duplicate_count:,}")
print(f"Cancelled/diverted removed:  {cancelled_diverted_removed:,}")
print(f"Final rows:                  {len(df):,}")
print(f"Final columns:               {len(df.columns)}")

# Show the final departure-delay rate in the cleaned dataset.
print(f"Departure delay rate:        {df['is_delayed'].mean() * 100:.2f}%")

# Show where the cleaned CSV was saved.
print(f"Output file:                 {OUTPUT_PATH}")

print()
print("Top initial missing values:")

# Print the top missing-value columns before cleaning.
print(initial_missing.to_string())

print()
print("Top final missing values:")

# Print the top missing-value columns after cleaning.
print(final_missing.to_string())

print()
print("Outlier check:")

# Convert the outlier summary list into a small dataframe for clean printing.
print(pd.DataFrame(outlier_rows).to_string(index=False))
