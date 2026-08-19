# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

# warnings is used to hide long model warning messages.
import warnings

# Path makes file paths easier to build.
from pathlib import Path

# Main data, modeling, plotting, and testing libraries.
import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet
from scipy.stats import jarque_bera
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Main project folder.
# This makes the rest of the file paths shorter.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Hide SARIMAX convergence warnings so the printed output is easier to read.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Main cleaned flight file used for forecasting.
INPUT_PATH = PROJECT_ROOT / "processed" / "flights_all_cleaned.csv"

# Output files created by this script.
# OUTPUT_PATH saves the detailed metrics table.
# SUMMARY_OUTPUT_PATH saves the readable report.
# PLOT_DIR saves the residual plots.
OUTPUT_PATH = PROJECT_ROOT / "query_results" / "calendar_forecast_metrics_no_defs.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "query_results" / "calendar_forecast_summary_no_defs.txt"
PLOT_DIR = PROJECT_ROOT / "plots" / "forecast_residuals_no_defs"

# Use data starting from this date.
# This avoids mixing the post-COVID recovery years with 2019 in the forecast model.
START_DATE = "2022-01-01"

# Holidays are flagged from 2 days before to 2 days after each holiday.
HOLIDAY_WINDOW = 2

# Create the plot folder if it does not already exist.
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD CLEANED DATA
# ============================================================

print()
print("=" * 60)
print("FORECASTING RUN - NO CUSTOM FUNCTIONS / NO LOOPS")
print("=" * 60)
print(f"Input file: {INPUT_PATH}")

# Only two columns are needed.
# fl_date is the date, and is_delayed is the departure-delay flag.
df = pd.read_csv(
    INPUT_PATH,
    usecols=["fl_date", "is_delayed"],
    parse_dates=["fl_date"],
)


# ============================================================
# 4. BUILD DAILY TIME SERIES
# ============================================================

# The raw/cleaned data is one row per flight.
# Forecasting needs one value per day, so we aggregate flight rows into daily rows.

# Average the flight-level delay flag by day.
# Since is_delayed is 0 or 1, the mean is the daily delay rate.
daily = df.groupby("fl_date")["is_delayed"].mean()

# Sort dates, convert to percent, and keep the modeling period.
daily = daily.sort_index()
daily = daily * 100
daily = daily.loc[START_DATE:]

# Make sure the series has one row per calendar day.
daily = daily.asfreq("D")

# Count missing calendar days before changing anything.
missing_days = daily.isna().sum()

print()
print(f"Missing calendar days found: {missing_days}")

# Fill gaps of no more than three consecutive days.
if missing_days > 0:
    daily = daily.interpolate(
        method="time",
        limit=3,
    )

# Do not silently delete dates that are still missing.
# SARIMA requires a continuous, evenly spaced daily series.
remaining_missing_days = daily.isna().sum()

if remaining_missing_days > 0:
    raise ValueError(
        "The daily series still contains "
        f"{remaining_missing_days} missing calendar days "
        "after interpolation."
    )


# ============================================================
# 5. STATIONARITY CHECK
# ============================================================

# Use only 2022-2025 for the stationarity decision.
# Q1 2026 remains untouched until final model evaluation.
development_daily = daily.loc[
    "2022-01-01":"2025-12-31"
]

# The ADF test evaluates whether the development series
# contains evidence of a unit root.
#
# Null hypothesis:
# The series has a unit root and is non-stationary.
#
# A p-value below 0.05 provides evidence against the
# unit-root null hypothesis.
adf_result = adfuller(
    development_daily,
    autolag="AIC",
)
adf_stat = adf_result[0]
adf_pvalue = adf_result[1]

print()
print("Stationarity check:")
print(f"ADF statistic: {adf_stat:.4f}")
print(f"ADF p-value:   {adf_pvalue:.6f}")


# ============================================================
# 6. HOLIDAY FEATURES
# ============================================================

# Get the years covered by the time series.
# These years are used to pull the matching US holidays.
years = range(daily.index.min().year, daily.index.max().year + 1)

# Get US holiday dates.
holiday_dates = pd.to_datetime(list(holidays.US(years=years).keys()))

# Create holiday-window dates.
# This marks each holiday plus 2 days before and 2 days after.
# Example: if the holiday is July 4, this also flags July 2, 3, 5, and 6.
holiday_window_dates = pd.concat(
    [
        pd.Series(holiday_dates - pd.Timedelta(days=2)),
        pd.Series(holiday_dates - pd.Timedelta(days=1)),
        pd.Series(holiday_dates),
        pd.Series(holiday_dates + pd.Timedelta(days=1)),
        pd.Series(holiday_dates + pd.Timedelta(days=2)),
    ],
    ignore_index=True,
)

# SARIMAX needs a numeric outside variable.
# holiday_window = 1 means the date is close to a holiday.
# holiday_window = 0 means the date is not close to a holiday.
features = pd.DataFrame(index=daily.index)
features["holiday_window"] = features.index.isin(holiday_window_dates).astype(int)

# Prophet needs a holiday dataframe with specific column names.
# ds = holiday date.
# holiday = holiday name.
# lower_window and upper_window tell Prophet to include days around each holiday.
prophet_holidays = pd.DataFrame(
    list(holidays.US(years=years).items()),
    columns=["ds", "holiday"],
)
prophet_holidays["ds"] = pd.to_datetime(prophet_holidays["ds"])
prophet_holidays["lower_window"] = -HOLIDAY_WINDOW
prophet_holidays["upper_window"] = HOLIDAY_WINDOW


# ============================================================
# 7. TRAIN/TEST PERIODS
# ============================================================

# Q1 2026 test period.
# Training data is every day before January 1, 2026 and after January 1, 2022.
# Test data is January 1, 2026 through March 31, 2026.
q1_split_date = pd.Timestamp("2026-01-01")
q1_end_date = pd.Timestamp("2026-03-31")
q1_train_y = daily[daily.index < q1_split_date]
q1_test_y = daily[(daily.index >= q1_split_date) & (daily.index <= q1_end_date)]

# Q3 2025 test period.
# Training data is every day before July 1, 2025 and afer January 1, 2022.
# Test data is July 1, 2025 through September 30, 2025.
q3_split_date = pd.Timestamp("2025-07-01")
q3_end_date = pd.Timestamp("2025-09-30")
q3_train_y = daily[daily.index < q3_split_date]
q3_test_y = daily[(daily.index >= q3_split_date) & (daily.index <= q3_end_date)]

# This list will store one dictionary per model result.
# At the end, rows becomes a dataframe called results.
rows = []


# ============================================================
# 8. Q3 2025 MODEL VALIDATION
# ============================================================

# Q3 2025 is the validation period used to compare methods.
# The selected method is later refitted through 2025 and evaluated
# once on the untouched Q1 2026 test period.
period_name = "Q3 2025"
train_y = q3_train_y
test_y = q3_test_y

print()
print("=" * 60)
print(f"{period_name}: train through {train_y.index[-1].date()}, test through {test_y.index[-1].date()}")
print("=" * 60)

# Q3 2025 is the practice test used to choose the final method.
# Q1 2026 is not examined until one method has been selected.

# ------------------------------
# Q3 Naive baseline
# ------------------------------

# This simple baseline predicts that every future day will have
# the same delay rate as the final day in the training period.
model_name = "Naive Baseline"
predicted = pd.Series(
    train_y.iloc[-1],
    index=test_y.index,
    name="forecast",
)

mae = mean_absolute_error(test_y, predicted)
rmse = np.sqrt(mean_squared_error(test_y, predicted))
mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100

rows.append(
    {
        "period": period_name,
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
)

print(
    f"{model_name}: "
    f"MAE {mae:.2f} pp | "
    f"RMSE {rmse:.2f} pp | "
    f"MAPE {mape:.2f}%"
)


# ------------------------------
# Q3 seven-day seasonal-naive baseline
# ------------------------------

# This baseline repeats the final seven training values.
# It never uses actual values from the test period.
model_name = "7-Day Seasonal Naive"
last_seven_days = train_y.iloc[-7:].to_numpy()
predicted_values = np.resize(
    last_seven_days,
    len(test_y),
)
predicted = pd.Series(
    predicted_values,
    index=test_y.index,
    name="forecast",
)

mae = mean_absolute_error(test_y, predicted)
rmse = np.sqrt(mean_squared_error(test_y, predicted))
mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100

rows.append(
    {
        "period": period_name,
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
)

print(
    f"{model_name}: "
    f"MAE {mae:.2f} pp | "
    f"RMSE {rmse:.2f} pp | "
    f"MAPE {mape:.2f}%"
)


# ------------------------------
# Q3 recent weekday-average baseline
# ------------------------------

# This baseline uses the most recent year of training data.
# It calculates a separate average for Monday, Tuesday,
# and every other day of the week.
model_name = "Recent Weekday Average"
recent_year = train_y.tail(365)

weekday_averages = recent_year.groupby(
    recent_year.index.dayofweek
).mean()

# Monday is represented by 0 and Sunday is represented by 6.
# Each Q3 date receives the recent average for its weekday.
test_weekdays = pd.Series(
    test_y.index.dayofweek,
    index=test_y.index,
)

predicted = test_weekdays.map(weekday_averages)
predicted.name = "forecast"

mae = mean_absolute_error(test_y, predicted)
rmse = np.sqrt(mean_squared_error(test_y, predicted))
mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100

rows.append(
    {
        "period": period_name,
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
)

print(
    f"{model_name}: "
    f"MAE {mae:.2f} pp | "
    f"RMSE {rmse:.2f} pp | "
    f"MAPE {mape:.2f}%"
)


# ------------------------------
# Q3 recent 28-day-mean baseline
# ------------------------------

# This baseline predicts that the test period will remain at
# the average level observed during the final four training weeks.
model_name = "Recent 28-Day Mean"
recent_28_day_mean = train_y.tail(28).mean()

predicted = pd.Series(
    recent_28_day_mean,
    index=test_y.index,
    name="forecast",
)

mae = mean_absolute_error(test_y, predicted)
rmse = np.sqrt(mean_squared_error(test_y, predicted))
mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100

rows.append(
    {
        "period": period_name,
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
)

print(
    f"{model_name}: "
    f"MAE {mae:.2f} pp | "
    f"RMSE {rmse:.2f} pp | "
    f"MAPE {mape:.2f}%"
)


# ------------------------------
# Q3 Manual SARIMA
# ------------------------------

# Manual SARIMA again uses the fixed parameters chosen by us.
model_name = "Manual SARIMA(1,0,0)(1,0,0,7)"
print(f"Fitting {model_name}...")

# Create the manual SARIMA model.
model = SARIMAX(
    train_y,
    order=(1, 0, 0),
    seasonal_order=(1, 0, 0, 7),
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True,
)
# Train the model on the training period.
result = model.fit(disp=False)

# Forecast the same number of days as the test period.
forecast = result.get_forecast(steps=len(test_y))

# Keep only the forecasted values.
predicted = forecast.predicted_mean

# Compare actual test values against forecasted values.
mae = mean_absolute_error(test_y, predicted)
rmse = np.sqrt(mean_squared_error(test_y, predicted))
mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100

# Residuals are the forecast errors.
# Residual = actual value - forecast value.
residuals = (test_y - predicted).dropna()

# Ljung-Box checks whether residuals still have time patterns.
# Lags 7, 14, 21, and 28 represent 1 to 4 weeks.
lb = acorr_ljungbox(residuals, lags=[7, 14, 21, 28], return_df=True)

# Jarque-Bera checks whether residuals are roughly normally distributed.
jb_stat, jb_pvalue = jarque_bera(residuals)

# First residual plot: histogram/distribution.
# It shows whether forecast errors are centered around zero.
residual_distribution_plot = PLOT_DIR / f"{period_name} - {model_name} - residual distribution.png"
fig, ax = plt.subplots(figsize=(8, 5))
residuals.hist(bins=30, ax=ax, color="steelblue")
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero")
ax.axvline(residuals.mean(), color="tomato", linewidth=2, label="Mean")
ax.set_title(f"Residual Distribution - {period_name} - {model_name}")
ax.set_xlabel("Residual: Actual - Forecast")
ax.set_ylabel("Days")
ax.legend()
plt.tight_layout()
plt.savefig(residual_distribution_plot)
plt.close(fig)

# Second residual plot: residuals over time.
# It shows when the model over-forecasted or
# under-forecasted at specific times.
residual_over_time_plot = (
    PLOT_DIR
    / f"{period_name} - {model_name} - residuals over time.png"
)
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(residuals.index, residuals, color="steelblue", s=25, alpha=0.8)
ax.plot(residuals.index, residuals, color="steelblue", linewidth=1, alpha=0.5)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title(f"Residuals Over Time - {period_name} - {model_name}")
ax.set_xlabel("Date")
ax.set_ylabel("Residual: Actual - Forecast")
plt.tight_layout()
plt.savefig(residual_over_time_plot)
plt.close(fig)

# Add this model's results to the final results list.
rows.append(
    {
        "period": period_name,
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "residual_mean": residuals.mean(),
        "residual_median": residuals.median(),
        "residual_std": residuals.std(),
        "residual_variance": residuals.var(),
        "residual_min": residuals.min(),
        "residual_max": residuals.max(),
        "residual_skew": residuals.skew(),
        "residual_kurtosis": residuals.kurtosis(),
        "jarque_bera_pvalue": jb_pvalue,
        "ljung_box_pvalue_lag_7": lb.loc[7, "lb_pvalue"],
        "ljung_box_pvalue_lag_14": lb.loc[14, "lb_pvalue"],
        "ljung_box_pvalue_lag_21": lb.loc[21, "lb_pvalue"],
        "ljung_box_pvalue_lag_28": lb.loc[28, "lb_pvalue"],
        "residual_distribution_plot": str(residual_distribution_plot),
        "residual_over_time_plot": str(residual_over_time_plot),
    }
)
print(
    f"{model_name}: "
    f"MAE {mae:.2f} pp | "
    f"RMSE {rmse:.2f} pp | "
    f"MAPE {mape:.2f}%"
)

# ------------------------------
# Q3 Auto SARIMA
# ------------------------------

# Run Auto ARIMA again because the training period is different.
# A different training period can produce different selected SARIMA parameters.
print("Running auto_arima search...")
auto_model = auto_arima(
    train_y,
    start_p=0,
    max_p=3,
    start_q=0,
    max_q=3,
    d=0,
    start_P=0,
    max_P=3,
    start_Q=0,
    max_Q=3,
    D=0,
    m=7,
    seasonal=True,
    information_criterion="aic",
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)

# Save the regular ARIMA order chosen by auto_arima.
auto_order = auto_model.order

# Save the seasonal SARIMA order chosen by auto_arima.
auto_seasonal_order = auto_model.seasonal_order

# Auto SARIMA uses the parameters selected by Q3 auto_arima.
model_name = f"Auto SARIMA{auto_order}{auto_seasonal_order}"
print(f"Fitting {model_name}...")

# Create the auto-selected SARIMA model.
model = SARIMAX(
    train_y,
    order=auto_order,
    seasonal_order=auto_seasonal_order,
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True,
)

# Train the model on the training period.
result = model.fit(disp=False)

# Forecast the same number of days as the test period.
forecast = result.get_forecast(steps=len(test_y))

# Keep only the forecasted values.
predicted = forecast.predicted_mean

# Compare actual test values against forecasted values.
mae = mean_absolute_error(test_y, predicted)
rmse = np.sqrt(mean_squared_error(test_y, predicted))
mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100

# Residuals are the forecast errors.
# Residual = actual value - forecast value.
residuals = (test_y - predicted).dropna()

# Ljung-Box checks whether residuals still have time patterns.
# Lags 7, 14, 21, and 28 represent 1 to 4 weeks.
lb = acorr_ljungbox(residuals, lags=[7, 14, 21, 28], return_df=True)

# Jarque-Bera checks whether residuals are roughly normally distributed.
jb_stat, jb_pvalue = jarque_bera(residuals)


# First residual plot: histogram/distribution.
# It shows whether forecast errors are centered around zero.
residual_distribution_plot = PLOT_DIR / f"{period_name} - {model_name} - residual distribution.png"
fig, ax = plt.subplots(figsize=(8, 5))
residuals.hist(bins=30, ax=ax, color="steelblue")
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero")
ax.axvline(residuals.mean(), color="tomato", linewidth=2, label="Mean")
ax.set_title(f"Residual Distribution - {period_name} - {model_name}")
ax.set_xlabel("Residual: Actual - Forecast")
ax.set_ylabel("Days")
ax.legend()
plt.tight_layout()
plt.savefig(residual_distribution_plot)
plt.close(fig)

# Second residual plot: residuals over time.
# It shows when the model over-forecasted or under-forecasted at specific times.
residual_over_time_plot = PLOT_DIR / f"{period_name} - {model_name} - residuals over time.png"
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(residuals.index, residuals, color="steelblue", s=25, alpha=0.8)
ax.plot(residuals.index, residuals, color="steelblue", linewidth=1, alpha=0.5)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title(f"Residuals Over Time - {period_name} - {model_name}")
ax.set_xlabel("Date")
ax.set_ylabel("Residual: Actual - Forecast")
plt.tight_layout()
plt.savefig(residual_over_time_plot)
plt.close(fig)


# Add this model's results to the final results list.
rows.append(
    {
        "period": period_name,
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "residual_mean": residuals.mean(),
        "residual_median": residuals.median(),
        "residual_std": residuals.std(),
        "residual_variance": residuals.var(),
        "residual_min": residuals.min(),
        "residual_max": residuals.max(),
        "residual_skew": residuals.skew(),
        "residual_kurtosis": residuals.kurtosis(),
        "jarque_bera_pvalue": jb_pvalue,
        "ljung_box_pvalue_lag_7": lb.loc[7, "lb_pvalue"],
        "ljung_box_pvalue_lag_14": lb.loc[14, "lb_pvalue"],
        "ljung_box_pvalue_lag_21": lb.loc[21, "lb_pvalue"],
        "ljung_box_pvalue_lag_28": lb.loc[28, "lb_pvalue"],
        "residual_distribution_plot": str(residual_distribution_plot),
        "residual_over_time_plot": str(residual_over_time_plot),
    }
)
print(
    f"{model_name}: "
    f"MAE {mae:.2f} pp | "
    f"RMSE {rmse:.2f} pp | "
    f"MAPE {mape:.2f}%"
)

# ------------------------------
# Q3 SARIMAX + holiday
# ------------------------------

# SARIMAX + holiday adds the holiday_window variable to the Auto SARIMA structure.
model_name = f"SARIMAX + holiday {auto_order}{auto_seasonal_order}"
print(f"Fitting {model_name}...")

# Get the holiday feature for the Q3 training and test dates.
train_exog = features.loc[train_y.index, ["holiday_window"]]
test_exog = features.loc[test_y.index, ["holiday_window"]]

# Create the SARIMAX model with the holiday feature.
model = SARIMAX(
    train_y,
    exog=train_exog,
    order=auto_order,
    seasonal_order=auto_seasonal_order,
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True,
)

# Train the model on the training period.
result = model.fit(disp=False)

# Forecast the same number of days as the test period.
forecast = result.get_forecast(steps=len(test_y), exog=test_exog)

# Keep only the forecasted values.
predicted = forecast.predicted_mean

# Compare actual test values against forecasted values.
mae = mean_absolute_error(test_y, predicted)
rmse = np.sqrt(mean_squared_error(test_y, predicted))
mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100

# Residuals are the forecast errors.
# Residual = actual value - forecast value.
residuals = (test_y - predicted).dropna()

# Ljung-Box checks whether residuals still have time patterns.
# Lags 7, 14, 21, and 28 represent 1 to 4 weeks.
lb = acorr_ljungbox(residuals, lags=[7, 14, 21, 28], return_df=True)

# Jarque-Bera checks whether residuals are roughly normally distributed.
jb_stat, jb_pvalue = jarque_bera(residuals)


# First residual plot: histogram/distribution.
# It shows whether forecast errors are centered around zero.
residual_distribution_plot = PLOT_DIR / f"{period_name} - {model_name} - residual distribution.png"
fig, ax = plt.subplots(figsize=(8, 5))
residuals.hist(bins=30, ax=ax, color="steelblue")
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero")
ax.axvline(residuals.mean(), color="tomato", linewidth=2, label="Mean")
ax.set_title(f"Residual Distribution - {period_name} - {model_name}")
ax.set_xlabel("Residual: Actual - Forecast")
ax.set_ylabel("Days")
ax.legend()
plt.tight_layout()
plt.savefig(residual_distribution_plot)
plt.close(fig)

# Second residual plot: residuals over time.
# It shows when the model over-forecasted or under-forecasted at specific times.
residual_over_time_plot = PLOT_DIR / f"{period_name} - {model_name} - residuals over time.png"
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(residuals.index, residuals, color="steelblue", s=25, alpha=0.8)
ax.plot(residuals.index, residuals, color="steelblue", linewidth=1, alpha=0.5)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title(f"Residuals Over Time - {period_name} - {model_name}")
ax.set_xlabel("Date")
ax.set_ylabel("Residual: Actual - Forecast")
plt.tight_layout()
plt.savefig(residual_over_time_plot)
plt.close(fig)


# Add this model's results to the final results list.
rows.append(
    {
        "period": period_name,
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "residual_mean": residuals.mean(),
        "residual_median": residuals.median(),
        "residual_std": residuals.std(),
        "residual_variance": residuals.var(),
        "residual_min": residuals.min(),
        "residual_max": residuals.max(),
        "residual_skew": residuals.skew(),
        "residual_kurtosis": residuals.kurtosis(),
        "jarque_bera_pvalue": jb_pvalue,
        "ljung_box_pvalue_lag_7": lb.loc[7, "lb_pvalue"],
        "ljung_box_pvalue_lag_14": lb.loc[14, "lb_pvalue"],
        "ljung_box_pvalue_lag_21": lb.loc[21, "lb_pvalue"],
        "ljung_box_pvalue_lag_28": lb.loc[28, "lb_pvalue"],
        "residual_distribution_plot": str(residual_distribution_plot),
        "residual_over_time_plot": str(residual_over_time_plot),
    }
)
print(
    f"{model_name}: "
    f"MAE {mae:.2f} pp | "
    f"RMSE {rmse:.2f} pp | "
    f"MAPE {mape:.2f}%"
)

# ------------------------------
# Q3 Prophet + holidays
# ------------------------------

# Prophet uses its own date/target format again.
model_name = "Prophet + holidays"
print(f"Fitting {model_name}...")

# Convert the training series to Prophet format: ds = date, y = value.
train_df = train_y.reset_index()
train_df.columns = ["ds", "y"]

# Create the Prophet model.
# yearly_seasonality captures yearly patterns.
# weekly_seasonality captures weekday/weekend patterns.
# Holidays adds the US holiday table built earlier.
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    holidays=prophet_holidays,
    seasonality_mode="additive",
    changepoint_prior_scale=0.05,
)
# Train Prophet on the training period.
model.fit(train_df)

# Future contains the test dates Prophet should forecast.
future = pd.DataFrame({"ds": test_y.index})

# Predict those dates.
forecast = model.predict(future)

# Prophet stores the forecasted values in yhat.
predicted = forecast.set_index("ds")["yhat"]

# Compare actual test values against forecasted values.
mae = mean_absolute_error(test_y, predicted)
rmse = np.sqrt(mean_squared_error(test_y, predicted))
mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100


# Residuals are the forecast errors.
# Residual = actual value - forecast value.
residuals = (test_y - predicted).dropna()

# Ljung-Box checks whether residuals still have time patterns.
# Lags 7, 14, 21, and 28 represent 1 to 4 weeks.
lb = acorr_ljungbox(residuals, lags=[7, 14, 21, 28], return_df=True)

# Jarque-Bera checks whether residuals are roughly normally distributed.
jb_stat, jb_pvalue = jarque_bera(residuals)


# First residual plot: histogram/distribution.
# It shows whether forecast errors are centered around zero.
residual_distribution_plot = PLOT_DIR / f"{period_name} - {model_name} - residual distribution.png"
fig, ax = plt.subplots(figsize=(8, 5))
residuals.hist(bins=30, ax=ax, color="steelblue")
ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero")
ax.axvline(residuals.mean(), color="tomato", linewidth=2, label="Mean")
ax.set_title(f"Residual Distribution - {period_name} - {model_name}")
ax.set_xlabel("Residual: Actual - Forecast")
ax.set_ylabel("Days")
ax.legend()
plt.tight_layout()
plt.savefig(residual_distribution_plot)
plt.close(fig)

# Second residual plot: residuals over time.
# It shows when the model over-forecasted or under-forecasted at specific times.
residual_over_time_plot = PLOT_DIR / f"{period_name} - {model_name} - residuals over time.png"
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(residuals.index, residuals, color="steelblue", s=25, alpha=0.8)
ax.plot(residuals.index, residuals, color="steelblue", linewidth=1, alpha=0.5)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title(f"Residuals Over Time - {period_name} - {model_name}")
ax.set_xlabel("Date")
ax.set_ylabel("Residual: Actual - Forecast")
plt.tight_layout()
plt.savefig(residual_over_time_plot)
plt.close(fig)


# Add this model's results to the final results list.
rows.append(
    {
        "period": period_name,
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "residual_mean": residuals.mean(),
        "residual_median": residuals.median(),
        "residual_std": residuals.std(),
        "residual_variance": residuals.var(),
        "residual_min": residuals.min(),
        "residual_max": residuals.max(),
        "residual_skew": residuals.skew(),
        "residual_kurtosis": residuals.kurtosis(),
        "jarque_bera_pvalue": jb_pvalue,
        "ljung_box_pvalue_lag_7": lb.loc[7, "lb_pvalue"],
        "ljung_box_pvalue_lag_14": lb.loc[14, "lb_pvalue"],
        "ljung_box_pvalue_lag_21": lb.loc[21, "lb_pvalue"],
        "ljung_box_pvalue_lag_28": lb.loc[28, "lb_pvalue"],
        "residual_distribution_plot": str(residual_distribution_plot),
        "residual_over_time_plot": str(residual_over_time_plot),
    }
)
print(
    f"{model_name}: "
    f"MAE {mae:.2f} pp | "
    f"RMSE {rmse:.2f} pp | "
    f"MAPE {mape:.2f}%"
)



# ============================================================
# 9. SELECT THE BEST Q3 METHOD
# ============================================================

# Build a table containing only the Q3 validation results.
validation_results = pd.DataFrame(rows)

# MAE is the main selection metric.
# RMSE breaks a tie by giving more weight to large errors.
best_validation = (
    validation_results
    .sort_values(["mae", "rmse"])
    .iloc[0]
)

selected_model_name = best_validation["model"]

print()
print("=" * 60)
print("Q3 2025 MODEL SELECTION")
print("=" * 60)
print(f"Selected method: {selected_model_name}")
print(
    "Validation MAE:  "
    f"{best_validation['mae']:.2f} percentage points"
)
print(
    "Validation RMSE: "
    f"{best_validation['rmse']:.2f} percentage points"
)
print(f"Validation MAPE: {best_validation['mape']:.2f}%")


# ============================================================
# 10. FINAL Q1 2026 EVALUATION
# ============================================================

# Q1 2026 is the final exam for the forecasting project.
# The model was already selected using Q3 2025, so this section
# does not compare models or choose a new winner.
#
# The selected method is first refitted using every available
# training day from January 1, 2022 through December 31, 2025.
# It then forecasts the 90 unseen days in Q1 2026.
#
# Keeping model selection and final evaluation separate prevents
# the Q1 results from influencing which method is chosen.


# ############################################################
# 10A. PREPARE THE FINAL TRAINING AND TEST PERIODS
# ############################################################

# q1_train_y contains only information available before 2026.
# q1_test_y contains the actual Q1 2026 rates that the forecast
# will be compared against after all predictions have been made.
period_name = "Q1 2026"
train_y = q1_train_y
test_y = q1_test_y

print()
print("=" * 60)
print(
    f"{period_name}: train through "
    f"{train_y.index[-1].date()}, "
    f"test through {test_y.index[-1].date()}"
)
print("=" * 60)
print(f"Final selected method: {selected_model_name}")


# ############################################################
# 10B. REFIT THE METHOD SELECTED DURING Q3 VALIDATION
# ############################################################

# selected_model_name was determined only from Q3 2025 results.
# The if/elif blocks below find that selected method and rebuild
# it using the larger final training dataset through 2025.
#
# Only one block runs. All other blocks are skipped.


# ------------------------------------------------------------
# Option 1: last-value naive baseline
# ------------------------------------------------------------

if selected_model_name == "Naive Baseline":
    # Predict the final observed training rate for every day
    # in Q1 2026.
    final_model_name = selected_model_name
    predicted = pd.Series(
        train_y.iloc[-1],
        index=test_y.index,
        name="forecast",
    )

# ------------------------------------------------------------
# Option 2: seven-day seasonal-naive baseline
# ------------------------------------------------------------

elif selected_model_name == "7-Day Seasonal Naive":
    # Repeat the final seven training values across Q1 2026.
    # This preserves the most recent weekly pattern without
    # looking at any actual Q1 values.
    final_model_name = selected_model_name
    last_seven_days = train_y.iloc[-7:].to_numpy()
    predicted_values = np.resize(
        last_seven_days,
        len(test_y),
    )
    predicted = pd.Series(
        predicted_values,
        index=test_y.index,
        name="forecast",
    )

# ------------------------------------------------------------
# Option 3: recent weekday-average baseline
# ------------------------------------------------------------

elif selected_model_name == "Recent Weekday Average":
    final_model_name = selected_model_name

    # Recalculate weekday averages using the most recent year
    # available before Q1 2026 begins.
    recent_year = train_y.tail(365)
    weekday_averages = recent_year.groupby(
        recent_year.index.dayofweek
    ).mean()

    test_weekdays = pd.Series(
        test_y.index.dayofweek,
        index=test_y.index,
    )

    predicted = test_weekdays.map(weekday_averages)
    predicted.name = "forecast"

# ------------------------------------------------------------
# Option 4: recent 28-day-mean baseline
# ------------------------------------------------------------

elif selected_model_name == "Recent 28-Day Mean":
    final_model_name = selected_model_name

    # Recalculate the recent level using only the final
    # 28 training days before Q1 2026 begins.
    recent_28_day_mean = train_y.tail(28).mean()

    predicted = pd.Series(
        recent_28_day_mean,
        index=test_y.index,
        name="forecast",
    )

# ------------------------------------------------------------
# Option 5: manually specified SARIMA
# ------------------------------------------------------------

elif selected_model_name.startswith("Manual SARIMA"):
    # Refit the fixed SARIMA structure selected before modeling.
    # trend="c" gives the undifferenced series a nonzero average.
    # The seasonal period of 7 represents weekly seasonality.
    final_model_name = "Manual SARIMA(1,0,0)(1,0,0,7)"
    model = SARIMAX(
        train_y,
        order=(1, 0, 0),
        seasonal_order=(1, 0, 0, 7),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    result = model.fit(disp=False)
    forecast = result.get_forecast(steps=len(test_y))
    predicted = forecast.predicted_mean

# ------------------------------------------------------------
# Option 6: automatically selected SARIMA
# ------------------------------------------------------------

elif selected_model_name.startswith("Auto SARIMA"):
    # Run auto_arima again because the final training dataset now
    # includes all observations through December 31, 2025.
    # The search uses training data only and never sees Q1 values.
    print("Running final auto_arima search using data through 2025...")
    auto_model = auto_arima(
        train_y,
        start_p=0,
        max_p=3,
        start_q=0,
        max_q=3,
        d=0,
        start_P=0,
        max_P=3,
        start_Q=0,
        max_Q=3,
        D=0,
        m=7,
        seasonal=True,
        information_criterion="aic",
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    auto_order = auto_model.order
    auto_seasonal_order = auto_model.seasonal_order
    final_model_name = (
        f"Auto SARIMA{auto_order}{auto_seasonal_order}"
    )
    model = SARIMAX(
        train_y,
        order=auto_order,
        seasonal_order=auto_seasonal_order,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    result = model.fit(disp=False)
    forecast = result.get_forecast(steps=len(test_y))
    predicted = forecast.predicted_mean

# ------------------------------------------------------------
# Option 7: SARIMAX with the holiday-window feature
# ------------------------------------------------------------

elif selected_model_name.startswith("SARIMAX + holiday"):
    # Select the ARIMA orders using the final training data, then
    # add the holiday-window indicator as an outside variable.
    # Future holiday dates are allowed because the calendar is
    # known before the forecast is made.
    print("Running final auto_arima search using data through 2025...")
    auto_model = auto_arima(
        train_y,
        start_p=0,
        max_p=3,
        start_q=0,
        max_q=3,
        d=0,
        start_P=0,
        max_P=3,
        start_Q=0,
        max_Q=3,
        D=0,
        m=7,
        seasonal=True,
        information_criterion="aic",
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    auto_order = auto_model.order
    auto_seasonal_order = auto_model.seasonal_order
    final_model_name = (
        "SARIMAX + holiday "
        f"{auto_order}{auto_seasonal_order}"
    )
    train_exog = features.loc[
        train_y.index,
        ["holiday_window"],
    ]
    test_exog = features.loc[
        test_y.index,
        ["holiday_window"],
    ]
    model = SARIMAX(
        train_y,
        exog=train_exog,
        order=auto_order,
        seasonal_order=auto_seasonal_order,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    result = model.fit(disp=False)
    forecast = result.get_forecast(
        steps=len(test_y),
        exog=test_exog,
    )
    predicted = forecast.predicted_mean

# ------------------------------------------------------------
# Option 8: Prophet with holidays
# ------------------------------------------------------------

elif selected_model_name.startswith("Prophet"):
    # Prophet receives the full training series, weekly and yearly
    # seasonality settings, and the known US holiday calendar.
    final_model_name = "Prophet + holidays"
    train_df = train_y.reset_index()
    train_df.columns = ["ds", "y"]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=prophet_holidays,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )
    model.fit(train_df)

    future = pd.DataFrame({"ds": test_y.index})
    forecast = model.predict(future)
    predicted = forecast.set_index("ds")["yhat"]

# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

else:
    # Stop the script if the selected name does not match any
    # supported forecasting method.
    raise ValueError(
        f"Unknown selected model: {selected_model_name}"
    )


# ############################################################
# 10C. CALCULATE FINAL Q1 FORECAST ACCURACY
# ############################################################

# MAE is the average absolute distance between the actual and
# predicted delay rates. It is the main accuracy measure.
#
# RMSE gives extra weight to especially large forecast errors.
#
# Because the target is already a percentage, MAE and RMSE are
# measured in percentage points. MAPE remains a percentage.

mae = mean_absolute_error(test_y, predicted)
rmse = np.sqrt(mean_squared_error(test_y, predicted))
mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100

# A positive forecast error means the actual delay rate was higher
# than predicted. A negative value means the model predicted a
# higher delay rate than the one that occurred.
forecast_errors = (test_y - predicted).dropna()

# Ljung-Box checks whether the errors still contain weekly-style
# time patterns. Larger p-values are preferred because they provide
# less evidence of remaining autocorrelation.
lb = acorr_ljungbox(
    forecast_errors,
    lags=[7, 14, 21, 28],
    return_df=True,
)

# Jarque-Bera checks whether the forecast-error distribution is
# approximately normal. Normality is more important for reliable
# uncertainty intervals than for point-forecast accuracy itself.
jb_stat, jb_pvalue = jarque_bera(forecast_errors)


# ############################################################
# 10D. SAVE THE FINAL ACTUAL-VERSUS-FORECAST PLOT
# ############################################################

# The blue line contains the actual Q1 daily delay rates.
# The red line contains the predictions created without using
# those actual Q1 values.

forecast_plot = (
    PLOT_DIR
    / f"{period_name} - {final_model_name} - forecast vs actual.png"
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    test_y.index,
    test_y,
    color="steelblue",
    linewidth=2,
    label="Actual",
)
ax.plot(
    predicted.index,
    predicted,
    color="tomato",
    linewidth=2,
    label="Forecast",
)
ax.set_title(
    f"Actual vs Forecast - {period_name} - {final_model_name}"
)
ax.set_xlabel("Date")
ax.set_ylabel("Daily Departure Delay Rate (%)")
ax.legend()
plt.tight_layout()
plt.savefig(
    forecast_plot,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ############################################################
# 10E. SAVE THE FORECAST-ERROR DISTRIBUTION
# ############################################################

# This histogram shows whether errors are concentrated near zero,
# whether the model is systematically too high or too low, and
# whether a small number of days produced unusually large errors.

residual_distribution_plot = (
    PLOT_DIR
    / f"{period_name} - {final_model_name} - forecast error distribution.png"
)

fig, ax = plt.subplots(figsize=(8, 5))
forecast_errors.hist(
    bins=30,
    ax=ax,
    color="steelblue",
)
ax.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=1,
    label="Zero",
)
ax.axvline(
    forecast_errors.mean(),
    color="tomato",
    linewidth=2,
    label="Mean",
)
ax.set_title(
    f"Forecast Error Distribution - {period_name}"
)
ax.set_xlabel("Forecast Error: Actual - Forecast")
ax.set_ylabel("Days")
ax.legend()
plt.tight_layout()
plt.savefig(
    residual_distribution_plot,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ############################################################
# 10F. SAVE FORECAST ERRORS OVER TIME
# ############################################################

# This chart shows when the errors occurred. Long runs above or
# below zero can reveal bias or time patterns the model missed.
residual_over_time_plot = (
    PLOT_DIR
    / f"{period_name} - {final_model_name} - forecast errors over time.png"
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(
    forecast_errors.index,
    forecast_errors,
    color="steelblue",
    s=25,
    alpha=0.8,
)
ax.plot(
    forecast_errors.index,
    forecast_errors,
    color="steelblue",
    linewidth=1,
    alpha=0.5,
)
ax.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1,
)
ax.set_title(
    f"Forecast Errors Over Time - {period_name}"
)
ax.set_xlabel("Date")
ax.set_ylabel("Forecast Error: Actual - Forecast")
plt.tight_layout()
plt.savefig(
    residual_over_time_plot,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ############################################################
# 10G. SAVE ACF AND PACF OF THE FINAL FORECAST ERRORS
# ############################################################

# These charts examine whether Q1 forecast errors still contain
# short-term or weekly time patterns.
#
# Lag 1 represents one day.
# Lag 7 represents one week.
# Lag 14 represents two weeks.
# Lag 28 represents four weeks.
#
# A spike outside a confidence band suggests that the errors at
# that lag are more strongly related than expected from random
# variation alone.

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 4),
)

# ACF shows the total correlation between forecast errors
# separated by different numbers of days.
plot_acf(
    forecast_errors,
    lags=28,
    ax=axes[0],
)

axes[0].set_title("ACF - Q1 2026 Forecast Errors")
axes[0].set_xlabel("Lag (days)")
axes[0].set_ylim(-1.1, 1.1)

# PACF shows the direct relationship at each lag after the
# shorter intervening lags have been taken into account.
plot_pacf(
    forecast_errors,
    lags=28,
    ax=axes[1],
    method="ywm",
)

axes[1].set_title("PACF - Q1 2026 Forecast Errors")
axes[1].set_xlabel("Lag (days)")
axes[1].set_ylim(-1.1, 1.1)

fig.tight_layout()

forecast_error_acf_pacf_plot = (
    PLOT_DIR
    / "Q1 2026 - forecast error ACF PACF.png"
)

fig.savefig(
    forecast_error_acf_pacf_plot,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ############################################################
# 10H. ADD THE FINAL Q1 RESULT TO THE RESULTS TABLE
# ############################################################

# Q3 contributes every candidate's validation result.
# Q1 contributes only this one selected method's final result.
# This preserves the distinction between model selection and the
# final out-of-sample evaluation.
rows.append(
    {
        "period": period_name,
        "model": final_model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "residual_mean": forecast_errors.mean(),
        "residual_median": forecast_errors.median(),
        "residual_std": forecast_errors.std(),
        "residual_variance": forecast_errors.var(),
        "residual_min": forecast_errors.min(),
        "residual_max": forecast_errors.max(),
        "residual_skew": forecast_errors.skew(),
        "residual_kurtosis": forecast_errors.kurtosis(),
        "jarque_bera_pvalue": jb_pvalue,
        "ljung_box_pvalue_lag_7": lb.loc[7, "lb_pvalue"],
        "ljung_box_pvalue_lag_14": lb.loc[14, "lb_pvalue"],
        "ljung_box_pvalue_lag_21": lb.loc[21, "lb_pvalue"],
        "ljung_box_pvalue_lag_28": lb.loc[28, "lb_pvalue"],
        "forecast_plot": str(forecast_plot),
        "residual_distribution_plot": str(
            residual_distribution_plot
        ),
        "residual_over_time_plot": str(
            residual_over_time_plot
        ),
        "forecast_error_acf_pacf_plot": str(
            forecast_error_acf_pacf_plot
        ),
    }
)


# ############################################################
# 10I. PRINT THE FINAL TEST RESULT
# ############################################################

# These are the final Q1 numbers to report in the project.
# They describe performance on data that was not used to select
# the forecasting method.
print()
print("FINAL Q1 2026 TEST")
print(f"Model: {final_model_name}")
print(f"MAE:  {mae:.2f} percentage points")
print(f"RMSE: {rmse:.2f} percentage points")
print(f"MAPE: {mape:.2f}%")


# ============================================================
# 11. MODEL SUMMARY
# ============================================================

# Turn all saved model results into one table.
results = pd.DataFrame(rows)

# Keep only the main columns for the summary.
summary_table = results[["period", "model", "mae", "rmse", "mape"]]

# Build the readable summary report.
forecast_summary = f"""
============================================================
FORECASTING SUMMARY
============================================================

{summary_table.to_string(index=False)}

MAE and RMSE are reported in percentage points.
MAPE is reported as a percentage.
Q3 2025 is the model-selection period.
Q1 2026 contains only the selected method's final test result.

Metrics CSV: {OUTPUT_PATH}
Summary TXT: {SUMMARY_OUTPUT_PATH}
Residual plots folder: {PLOT_DIR}
"""

# ============================================================
# 12. SAVE OUTPUTS
# ============================================================

# Create the output folder if needed.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save the full model metrics table.
results.to_csv(OUTPUT_PATH, index=False)

# Save the readable text summary.
SUMMARY_OUTPUT_PATH.write_text(forecast_summary)

# Print the same summary in the Python output window.
print()
print(forecast_summary)
