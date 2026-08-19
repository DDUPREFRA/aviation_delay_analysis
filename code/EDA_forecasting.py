# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main cleaned flight dataset.
PATH = PROJECT_ROOT / "processed" / "flights_all_cleaned.csv"

# Folder where forecasting EDA plots are saved.
PLOT_DIR = PROJECT_ROOT / "plots"

# Create the plot folder if it does not already exist.
os.makedirs(PLOT_DIR, exist_ok=True)


def print_section(title):
    """Print a clearly separated terminal section."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 3. LOAD DATA
# ============================================================

# Only two columns are required:
# fl_date identifies the date of each flight.
# is_delayed equals 1 when a flight departed 15+ minutes late.
df = pd.read_csv(
    PATH,
    usecols=["fl_date", "is_delayed"],
    parse_dates=["fl_date"],
)


# ============================================================
# 4. CREATE DAILY DEPARTURE-DELAY RATE
# ============================================================

# Convert flight-level observations into one departure-delay
# percentage for each date.
#
# For example, a value of 20 means that 20% of flights scheduled
# on that date departed at least 15 minutes late.
full_daily_delay_rate = (
    df.groupby("fl_date")["is_delayed"]
    .mean()
    .mul(100)
    .sort_index()
)


# ============================================================
# 5. CREATE TRAINING AND TEST SERIES
# ============================================================

# Use 2022–2025 for forecasting EDA and model training.
# The pandemic-affected 2020–2021 period is not included in
# this project.
training_series = full_daily_delay_rate.loc[
    "2022-01-01":"2025-12-31"
].asfreq("D")

# Reserve Q1 2026 for evaluating future forecasts.
# This series must not be used to fit the forecasting model.
test_series = full_daily_delay_rate.loc[
    "2026-01-01":"2026-03-31"
].asfreq("D")


# ============================================================
# 6. CHECK FOR MISSING CALENDAR DAYS
# ============================================================

# Count missing dates after enforcing a continuous daily frequency.
missing_training_days = training_series.isna().sum()
missing_test_days = test_series.isna().sum()

# If only a small number of isolated training dates are missing,
# estimate them from nearby training observations.
if missing_training_days > 0:
    print(
        "Missing training days filled by interpolation: "
        f"{missing_training_days}"
    )

    training_series = training_series.interpolate(
        method="time",
        limit=3,
    )

# Do not silently remove dates because forecasting requires a
# continuous daily time index.
if training_series.isna().any():
    remaining_missing = training_series.isna().sum()

    raise ValueError(
        "The training series still contains "
        f"{remaining_missing} missing calendar days after "
        "interpolation. Inspect these dates before modeling."
    )

# The test period contains the actual values used for evaluation.
# Missing test dates should be investigated instead of estimated.
if test_series.isna().any():
    raise ValueError(
        "The Q1 2026 test series contains "
        f"{missing_test_days} missing calendar days. "
        "Inspect the source data before evaluating forecasts."
    )


# ============================================================
# 7. SAVE SUMMARY VALUES
# ============================================================

training_start_date = training_series.index.min().date()
training_end_date = training_series.index.max().date()
training_day_count = len(training_series)
average_training_delay_rate = training_series.mean()

test_start_date = test_series.index.min().date()
test_end_date = test_series.index.max().date()
test_day_count = len(test_series)


# ============================================================
# 8. PLOT DAILY DEPARTURE-DELAY RATE
# ============================================================

# Plot only the training period so Q1 2026 remains reserved
# for final forecast evaluation.
fig, ax = plt.subplots(figsize=(14, 4))

training_series.plot(
    ax=ax,
    color="steelblue",
    linewidth=0.8,
)

ax.set_title("Daily Departure Delay Rate — Training Period")
ax.set_ylabel("Flights Departing 15+ Minutes Late (%)")
ax.set_xlabel("Date")

fig.tight_layout()

fig.savefig(
    PLOT_DIR / "forecast_01_daily_delay_rate.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 9. AUGMENTED DICKEY-FULLER TEST
# ============================================================

# The ADF test evaluates whether the training series contains
# evidence of a unit root.
#
# Null hypothesis:
# The series has a unit root and is non-stationary.
#
# A p-value below 0.05 provides evidence against the null
# hypothesis and supports treating the series as stationary.
adf_stat, adf_pvalue, *_ = adfuller(
    training_series,
    autolag="AIC",
)

if adf_pvalue < 0.05:
    adf_verdict = "Stationary"
else:
    adf_verdict = "Non-stationary"


# ============================================================
# 10. ACF AND PACF
# ============================================================

# ACF measures correlation between the daily departure-delay
# rate and its previous values.
#
# PACF measures the relationship with each lag after accounting
# for the effects of shorter lags.
#
# Lags are measured in days. Forty lags include several weekly
# cycles while keeping the chart readable.
lags = max(
    1,
    min(40, len(training_series) // 2 - 1),
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Autocorrelation function.
plot_acf(
    training_series,
    lags=lags,
    ax=axes[0],
)

axes[0].set_title("ACF — Daily Departure Delay Rate")
axes[0].set_xlabel("Lag (days)")
axes[0].set_xticks(range(0, lags + 1))
axes[0].set_ylim(-1.1, 1.1)
axes[0].grid(axis="x", alpha=0.3)
axes[0].tick_params(
    axis="x",
    labelsize=8,
    rotation=90,
)

# Partial autocorrelation function.
plot_pacf(
    training_series,
    lags=lags,
    ax=axes[1],
    method="ywm",
)

axes[1].set_title("PACF — Daily Departure Delay Rate")
axes[1].set_xlabel("Lag (days)")
axes[1].set_xticks(range(0, lags + 1))
axes[1].set_ylim(-1.1, 1.1)
axes[1].grid(axis="x", alpha=0.3)
axes[1].tick_params(
    axis="x",
    labelsize=8,
    rotation=90,
)

fig.tight_layout()

fig.savefig(
    PLOT_DIR / "forecast_02_acf_pacf.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 11. PRINT FORECASTING EDA SUMMARY
# ============================================================

print_section("FORECASTING EDA SUMMARY")

print(f"Input file: {PATH}")

print()
print("Training dataset — used for EDA and model fitting")
print(
    f"Date range:               "
    f"{training_start_date} to {training_end_date}"
)
print(f"Calendar days:            {training_day_count:,}")
print(
    f"Average daily delay rate: "
    f"{average_training_delay_rate:.2f}%"
)
print(
    f"Missing days encountered: "
    f"{missing_training_days:,}"
)

print()
print("Test dataset — reserved for forecast evaluation")
print(
    f"Date range:               "
    f"{test_start_date} to {test_end_date}"
)
print(f"Calendar days:            {test_day_count:,}")
print(f"Missing days:             {missing_test_days:,}")

print()
print("Stationarity check — training series only")
print(f"ADF statistic: {adf_stat:.4f}")
print(f"ADF p-value:   {adf_pvalue:.6f}")
print(f"ADF result:    {adf_verdict}")

print()
print("Saved plots:")
print(
    "- Daily departure-delay rate: "
    f"{PLOT_DIR / 'forecast_01_daily_delay_rate.png'}"
)
print(
    "- ACF and PACF: "
    f"{PLOT_DIR / 'forecast_02_acf_pacf.png'}"
)
