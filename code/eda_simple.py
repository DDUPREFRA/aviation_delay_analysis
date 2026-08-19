# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main cleaned file used for EDA.
PATH = PROJECT_ROOT / "processed" / "flights_all_cleaned.csv"

# Folder where forecasting EDA plots are saved.
PLOT_DIR = PROJECT_ROOT / "plots"

# Make sure the plot folder exists before saving images.
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Only load the columns needed for this EDA file.
# This keeps the script lighter than loading every column.
cols = [
    "fl_date",
    "year",
    "month",
    "op_unique_carrier",
    "is_delayed",
    "arr_delay",
    "dep_delay",
    "distance",
    "dep_hour",
]

# Read the cleaned dataset created by clean_flights.py.
df = pd.read_csv(PATH, usecols=cols, parse_dates=["fl_date"], low_memory=False)

# All EDA charts exclude incomplete 2026.
# Summary statistics and outlier analysis use the full dataset,
# including 2026, because those records will be used for modeling.
eda_df = df.loc[df["year"] != 2026].copy()

def print_section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 3. BASIC CHECK
# ============================================================

# Full dataset information, including 2026.
full_row_count = len(df)
full_start_date = df["fl_date"].min().date()
full_end_date = df["fl_date"].max().date()
full_delay_rate = df["is_delayed"].mean() * 100
missing_values = df.isna().sum()

# Complete-year EDA information, excluding incomplete 2026.
eda_row_count = len(eda_df)
eda_start_date = eda_df["fl_date"].min().date()
eda_end_date = eda_df["fl_date"].max().date()
eda_delay_rate = eda_df["is_delayed"].mean() * 100

# ============================================================
# 4. DELAY OVERVIEW
# ============================================================

# is_delayed = 0 means the departure delay was under 15 minutes – the flight is not delayed.
# is_delayed = 1 means the departure delay was 15+ minutes – the flight is delayed.

delay_counts = (
    eda_df["is_delayed"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

delay_counts.index = [
    "Under 15 Minutes Late",
    "15+ Minutes Late",
]

# Keep the overall delay share and yearly comparison together.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Delayed vs non-delayed.
delay_counts.plot(
    kind="pie",
    ax=axes[0],
    autopct="%1.1f%%",
    colors=["steelblue", "tomato"],
    startangle=90,
)
axes[0].set_title("Delayed vs On-Time Flights")
axes[0].set_ylabel("")

yearly_delay_rate = (
    eda_df.groupby("year")["is_delayed"]
    .mean()
    .mul(100)
)

# Yearly delay rate.
yearly_delay_rate.plot(
    kind="bar",
    ax=axes[1],
    color="steelblue",
)
axes[1].set_title("Departure Delay Rate by Year")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Flights Departing 15+ Minutes Late (%)")
axes[1].set_ylim(bottom=0)
axes[1].tick_params(axis="x", rotation=0)

fig.tight_layout()
fig.savefig(
    PLOT_DIR / "delay_overview.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)



# ============================================================
# 5. DELAY PATTERNS
# ============================================================

# These charts compare monthly and hourly departure-delay patterns,
# both overall and separately for each complete year.

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

# ------------------------------------------------------------
# Overall departure-delay rate by month
# ------------------------------------------------------------

monthly_delay_rate = (
    eda_df.groupby("month")["is_delayed"]
    .mean()
    .mul(100)
)

monthly_delay_rate.plot(
    marker="o",
    ax=axes[0],
    color="steelblue",
)
axes[0].set_title("Departure Delay Rate by Month")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Flights Departing 15+ Minutes Late (%)")
axes[0].set_xticks(range(1, 13))
axes[0].set_xlim(1, 12)
axes[0].set_ylim(bottom=0)


# ------------------------------------------------------------
# Monthly departure-delay rate by year
# ------------------------------------------------------------

monthly_delay_rate_by_year = (
    eda_df.groupby(["month", "year"])["is_delayed"]
    .mean()
    .mul(100)
    .unstack("year")
)

monthly_delay_rate_by_year.plot(
    ax=axes[1],
    style=["-o", "--s", "-.^", ":D", "-P"],
    linewidth=2,
    markersize=5,
)

axes[1].set_title("Monthly Departure Delay Rate by Year")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Flights Departing 15+ Minutes Late (%)")
axes[1].set_xticks(range(1, 13))
axes[1].set_xlim(1, 12)
axes[1].set_ylim(10, 32)
axes[1].legend(title="Year")


# ------------------------------------------------------------
# Overall delay rate by scheduled departure hour
# ------------------------------------------------------------

hourly_delay_rate = (
    eda_df.groupby("dep_hour")["is_delayed"]
    .mean()
    .mul(100)
)

hourly_delay_rate.plot(
    marker="o",
    ax=axes[2],
    color="steelblue",
)

axes[2].set_title(
    "Departure Delay Rate by Scheduled Departure Hour"
)
axes[2].set_xlabel("Scheduled Departure Hour")
axes[2].set_ylabel("Flights Departing 15+ Minutes Late (%)")
axes[2].set_xticks(range(24))
axes[2].set_xlim(0, 23)
axes[2].set_ylim(bottom=0)

# ------------------------------------------------------------
# Hourly departure-delay rate by year
# ------------------------------------------------------------

hourly_delay_rate_by_year = (
    eda_df.groupby(["dep_hour", "year"])["is_delayed"]
    .mean()
    .mul(100)
    .unstack("year")
)

hourly_delay_rate_by_year.plot(
    ax=axes[3],
    style=["-o", "--s", "-.^", ":D", "-P"],
    linewidth=2,
    markersize=4,
)

axes[3].set_title("Hourly Departure Delay Rate by Year")
axes[3].set_xlabel("Scheduled Departure Hour")
axes[3].set_ylabel("Flights Departing 15+ Minutes Late (%)")
axes[3].set_xticks(range(24))
axes[3].set_xlim(0, 23)
axes[3].set_ylim(5, 35)
axes[3].legend(title="Year")


fig.tight_layout()

fig.savefig(
    PLOT_DIR / "delay_patterns.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 6. FLIGHT VOLUME
# ============================================================

# These charts show how the number of scheduled flights changes
# by year, month, and scheduled departure hour.

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Flight count by year
yearly_flight_counts = eda_df.groupby("year").size()
yearly_flight_counts.plot(
    kind="bar",
    ax=axes[0],
    color="steelblue",
)
axes[0].set_title("Flight Count by Year")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Number of Flights")
axes[0].tick_params(axis="x", rotation=0)

# Monthly flight count for each year
monthly_flight_counts_by_year = (
    eda_df.groupby(["month", "year"])
    .size()
    .unstack("year")
)
monthly_flight_counts_by_year.plot(
    ax=axes[1],
    style=["-o", "--s", "-.^", ":D", "-P"],
    linewidth=2,
    markersize=5,
)
axes[1].set_title("Monthly Flight Count by Year")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Number of Flights")
axes[1].set_xticks(range(1, 13))
axes[1].set_xlim(1, 12)
axes[1].legend(title="Year")

# Flight count by scheduled departure hour
hourly_flight_counts = (
    eda_df.groupby("dep_hour")
    .size()
    .reindex(range(24), fill_value=0)
)
hourly_flight_counts.plot(
    kind="bar",
    ax=axes[2],
    color="steelblue",
)
axes[2].set_title("Flight Count by Scheduled Departure Hour")
axes[2].set_xlabel("Scheduled Departure Hour")
axes[2].set_ylabel("Number of Flights")
axes[2].tick_params(axis="x", rotation=0)

fig.tight_layout()
fig.savefig(
    PLOT_DIR / "flight_volume.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# 7. DELAY DISTRIBUTIONS
# ============================================================

# These histograms show the shapes of arrival delays,
# departure delays, and flight distances.

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Use five-minute delay buckets.
delay_bins = range(-60, 505, 5)

# Clip only the values passed to the chart.
# The original dataframe is not modified.
# Values below -60 appear at the left boundary.
# Values above 500 appear at the right boundary.
eda_df["arr_delay"].clip(-60, 500).hist(
    bins=delay_bins,
    ax=axes[0],
    color="steelblue",
)

axes[0].set_title("Arrival Delay")
axes[0].set_xlabel("Minutes")
axes[0].set_ylabel("Flights")
axes[0].set_xlim(-60, 500)


eda_df["dep_delay"].clip(-60, 500).hist(
    bins=delay_bins,
    ax=axes[1],
    color="steelblue",
)

axes[1].set_title("Departure Delay")
axes[1].set_xlabel("Minutes")
axes[1].set_ylabel("Flights")
axes[1].set_xlim(-60, 500)


# Values above 4,000 miles appear at the right boundary.
# The original distance values are not modified.
eda_df["distance"].clip(0, 4000).hist(
    bins=50,
    ax=axes[2],
    color="steelblue",
)

axes[2].set_title("Flight Distance")
axes[2].set_xlabel("Miles")
axes[2].set_ylabel("Flights")
axes[2].set_xlim(0, 4000)


fig.tight_layout()

fig.savefig(
    PLOT_DIR / "delay_distributions.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 8. OUTLIERS
# ============================================================

# This section includes all available observations, including 2026,
# because the full dataset will be used for predictive modeling.
#
# The records originated in the official BTS dataset. Cancelled,
# diverted, and exact duplicate records were removed during cleaning.
# Extreme values are reported and reviewed, not automatically removed.

outlier_columns = [
    "arr_delay",
    "dep_delay",
    "distance",
]

# Summarize the center, spread, and upper tail of each variable.
outlier_summary = df[outlier_columns].describe(
    percentiles=[
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]
)

# Remove missing values only when calculating extreme-delay
# counts and percentages.
arrival_nonmissing = df["arr_delay"].dropna()
departure_nonmissing = df["dep_delay"].dropna()

# Count delays exceeding 300 minutes.
long_arrival_delays = arrival_nonmissing.gt(300).sum()
long_departure_delays = departure_nonmissing.gt(300).sum()

# Calculate the percentage of nonmissing observations exceeding
# 300 minutes.
long_arrival_delay_pct = (
    arrival_nonmissing.gt(300).mean() * 100
)

long_departure_delay_pct = (
    departure_nonmissing.gt(300).mean() * 100
)

# Keep the extreme departure-delay records for basic automated
# consistency checks.
extreme_departure_records = df.loc[
    df["dep_delay"] > 300,
    [
        "fl_date",
        "op_unique_carrier",
        "dep_delay",
        "arr_delay",
        "distance",
        "is_delayed",
    ],
].copy()

# Every departure delay above 300 minutes should have
# is_delayed equal to 1.
extreme_label_mismatches = (
    extreme_departure_records["is_delayed"] != 1
).sum()

# Count extreme departure records without a corresponding
# arrival-delay value.
extreme_missing_arrival_delays = (
    extreme_departure_records["arr_delay"].isna().sum()
)

# Find the 10 largest departure delays without sorting the
# entire dataframe.
top_departure_delays = (
    extreme_departure_records
    .nlargest(10, "dep_delay")
    .copy()
)

# Compare arrival delay with departure delay.
# A negative value means the flight recovered some delay.
# A positive value means additional delay accumulated.
top_departure_delays["arrival_minus_departure"] = (
    top_departure_delays["arr_delay"]
    - top_departure_delays["dep_delay"]
)


# ============================================================
# 9. PRINT EDA SUMMARY
# ============================================================

print_section("EDA SUMMARY")

print(f"Input file: {PATH}")

# Show full dataset information, including 2026.
print()
print("Full dataset — includes 2026")
print(f"Rows:                 {full_row_count:,}")
print(f"Date range:           {full_start_date} to {full_end_date}")
print(f"Departure delay rate: {full_delay_rate:.2f}%")

# Show complete-year EDA information, excluding 2026.
print()
print("Complete-year EDA dataset — excludes 2026")
print(f"Rows:                 {eda_row_count:,}")
print(f"Date range:           {eda_start_date} to {eda_end_date}")
print(f"Departure delay rate: {eda_delay_rate:.2f}%")

# Show missing values from the full dataset.
print()
print("Missing values — full dataset:")
print(missing_values.to_string())

# Show summary statistics.
print()
print("Outlier summary — full dataset:")
print(outlier_summary.to_string())

# Show extreme-delay counts and percentages.
print()
print(
    f"Arrival delays over 300 minutes: "
    f"{long_arrival_delays:,} "
    f"({long_arrival_delay_pct:.3f}%)"
)

print(
    f"Departure delays over 300 minutes: "
    f"{long_departure_delays:,} "
    f"({long_departure_delay_pct:.3f}%)"
)

# Show the basic automated consistency checks.
print()
print("Extreme-delay consistency checks:")
print(
    f"Departure delays over 300 minutes checked: "
    f"{len(extreme_departure_records):,}"
)
print(
    f"Incorrect is_delayed labels: "
    f"{extreme_label_mismatches:,}"
)
print(
    f"Missing corresponding arrival delays: "
    f"{extreme_missing_arrival_delays:,}"
)

# Show the largest officially reported departure delays.
print()
print("Top 10 largest officially reported departure delays:")
print(top_departure_delays.to_string(index=False))
