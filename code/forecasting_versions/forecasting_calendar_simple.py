"""
forecasting_calendar_simple.py

WHAT THIS SCRIPT DOES:
1. Builds a daily departure-delay-rate time series from the cleaned flight data.
2. Splits the time series into training and test periods.
3. Compares Manual SARIMA, Auto SARIMA, SARIMAX with holidays, and Prophet.
4. Calculates forecast accuracy, residual diagnostics, and saved plots.
5. Saves a detailed metrics CSV and a shorter text summary.

HOW TO READ THIS FILE:
1. Configuration: file paths and forecasting settings.
2. Helper functions: reusable blocks for metrics, plots, data prep, and models.
3. Main pipeline: the actual forecasting workflow. Start there if you feel lost.

MAIN PIPELINE:
1. Print run settings.
2. Build the daily time series.
3. Run the ADF stationarity check.
4. Create holiday features.
5. Choose train/test periods.
6. For each period, train and compare the models.
7. Save the final metrics, summary, and plots.

"""


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from __future__ import annotations

import os
import warnings
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera

# Main project folder.
PROJECT_ROOT = Path("/Users/daviddupre/Documents/PERSONAL PROJECTS PORTFOLIO/aviation_delay_analysis")

# Keep Matplotlib cache files inside this project folder.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

# Save forecasting plots to PNG files instead of opening pop-up windows.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Hide SARIMAX convergence warnings so the report output stays readable.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Main cleaned flight file used for forecasting.
INPUT_PATH = PROJECT_ROOT / "processed" / "flights_all_cleaned.csv"

# CSV file where detailed model metrics are saved.
OUTPUT_PATH = PROJECT_ROOT / "query_results" / "calendar_forecast_metrics.csv"

# Text file where the short forecasting summary is saved.
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "query_results" / "calendar_forecast_summary.txt"

# Folder where forecast and residual plots are saved.
PLOT_DIR = PROJECT_ROOT / "plots" / "forecast_residuals"

# Use data starting from this date.
START_DATE = "2022-01-01"

# Forecast 90 days for each test period.
TEST_DAYS = 90

# Flag US holidays plus/minus 2 days for SARIMAX and Prophet.
HOLIDAY_WINDOW = 2


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

# Helper functions are reusable code blocks.
# They keep the main forecasting pipeline shorter and easier to read.
# The main pipeline starts later in section 4.
#
# If this section feels abstract, skip to section 4 first.
# Section 4 shows the actual workflow in order.


# ------------------------------------------------------------
# 3A. METRICS AND TESTS
# ------------------------------------------------------------

# This group is similar to R's accuracy(), residuals(), acf(), and Box.test().
# It calculates forecast errors and residual diagnostics.

def evaluate(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    # Compare the real values to the forecasted values.
    # Lower MAE, RMSE, and MAPE usually means a better forecast.
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


def run_series_tests(series: pd.Series) -> dict[str, float]:
    # ADF test checks if the daily departure-delay-rate series is roughly stationary.
    # A p-value below 0.05 usually means the series is stationary enough to model.
    adf_stat, adf_pvalue, *_ = adfuller(series)
    return {"adf_stat": adf_stat, "adf_pvalue": adf_pvalue}


def run_residual_tests(residuals: pd.Series) -> dict[str, float]:
    # Residuals are the model errors:
    # residual = actual value - forecast value.

    # Remove missing residual values before running tests.
    # The residual tests cannot use NaN values.
    residuals = residuals.dropna()

    # We use Ljung-Box to check if residuals still have time patterns.
    # Ideally, residuals should look random after forecasting.

    # These are weekly-style lags for daily data:
    # lag 7 = 1 week
    # lag 14 = 2 weeks
    # lag 21 = 3 weeks
    # lag 28 = 4 weeks
    lb_lags = []

    # Only keep a lag if we have enough residual values to test it.
    for lag in [7, 14, 21, 28]:
        if lag < len(residuals):
            lb_lags.append(lag)

    # Run the Ljung-Box test using the valid lags.
    # If no lags are valid, create an empty dataframe instead.
    lb = acorr_ljungbox(residuals, lags=lb_lags, return_df=True) if lb_lags else pd.DataFrame()

    # Jarque-Bera checks whether residuals look normally distributed.
    jb_stat, jb_pvalue = jarque_bera(residuals)

    tests = {
        "residual_mean": residuals.mean(),
        "residual_median": residuals.median(),
        "residual_std": residuals.std(),
        "residual_variance": residuals.var(),
        "residual_min": residuals.min(),
        "residual_max": residuals.max(),
        "residual_skew": residuals.skew(),
        "residual_kurtosis": residuals.kurtosis(),
        "jarque_bera_pvalue": jb_pvalue,
    }

    # Go through each lag and create a dictionary.
    # Of p-values for each lag.
    for lag in lb_lags:
        tests[f"ljung_box_pvalue_lag_{lag}"] = lb.loc[lag, "lb_pvalue"]
    # Send the full dictionary back to the rest of the forecasting pipeline.
    return tests


# ------------------------------------------------------------
# 3B. NAMES AND PRINTED OUTPUT
# ------------------------------------------------------------

# This group only controls readable names and readable printed summaries.
# It does not fit models or change the forecasting results.

def safe_name(value: str) -> str:
    # Turn model names into safe file names for saved plots.
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in cleaned.split("_") if part)


def print_model_results(
    model_name: str,
    metrics: dict[str, float],
    residual_tests: dict[str, float],
) -> None:
    # Print one model's key results without flooding the console.
    lb_values = [
        value
        for key, value in residual_tests.items()
        if key.startswith("ljung_box_pvalue_")
    ]
    min_ljung_box = min(lb_values) if lb_values else np.nan

    print(
        f"{short_model_name(model_name):<18} "
        f"MAE {metrics['mae']:>6.2f}% | "
        f"RMSE {metrics['rmse']:>6.2f}% | "
        f"MAPE {metrics['mape']:>6.2f}% | "
        f"resid mean {residual_tests['residual_mean']:>6.2f} | "
        f"min LB p {min_ljung_box:.3g}"
    )


def short_model_name(model_name: str) -> str:
    # Make model names short enough for the final console table.
    if model_name.startswith("Manual SARIMA"):
        return "Manual SARIMA"
    if model_name.startswith("Auto SARIMA"):
        return "Auto SARIMA"
    if model_name.startswith("SARIMAX + holiday"):
        return "SARIMAX + Holiday"
    if model_name.startswith("Prophet"):
        return "Prophet + Holidays"
    return model_name


def minimum_ljung_box_pvalue(row: pd.Series) -> float:
    # Get the smallest Ljung-Box p-value stored for one model.
    lb_cols = []

    for col in row.index:
        if col.startswith("ljung_box_pvalue_lag_"):
            lb_cols.append(col)

    values = []

    for col in lb_cols:
        if pd.notna(row[col]):
            values.append(row[col])

    return min(values) if values else np.nan


def build_forecast_summary(results: pd.DataFrame) -> str:
    # Create a clean text summary for the final forecasting results.
    lines = [
        "=" * 60,
        "FORECASTING SUMMARY",
        "=" * 60,
        "",
        f"Input file: {INPUT_PATH}",
        f"Start date: {START_DATE}",
        f"Test window: {TEST_DAYS} days",
        "Models compared: SARIMA, Auto SARIMA, SARIMAX + holidays, Prophet",
        f"Holiday window: +/- {HOLIDAY_WINDOW} days",
        "",
    ]

    summary = results.copy()

    simple_model_names = []

    for model_name in summary["model"]:
        simple_model_names.append(short_model_name(model_name))

    summary["model"] = simple_model_names

    # Print a separate summary for each test period.
    for period_name, period_results in summary.groupby("period", sort=False):
        period_results = period_results.sort_values("rmse").reset_index(drop=True)
        best = period_results.iloc[0]

        lines.extend(
            [
                "-" * 60,
                period_name,
                "-" * 60,
                f"Period: {period_name}",
                "Models compared: SARIMA, Auto SARIMA, SARIMAX + holidays, Prophet",
                "",
                f"Best model by RMSE: {best['model']}",
                f"MAE:                {best['mae']:.2f}%",
                f"RMSE:               {best['rmse']:.2f}%",
                f"MAPE:               {best['mape']:.2f}%",
                "",
                "Residual diagnostics:",
                f"Residual mean:      {best['residual_mean']:.2f}",
                f"Residual std:       {best['residual_std']:.2f}",
                f"Ljung-Box min p:    {minimum_ljung_box_pvalue(best):.3g}",
                f"Jarque-Bera p:      {best['jarque_bera_pvalue']:.3g}",
                "",
                "Model comparison:",
            ]
        )

        display = period_results[
            ["model", "mae", "rmse", "mape", "residual_mean", "residual_std"]
        ].round(2)
        lines.append(display.to_string(index=False))
        lines.append("")

    lines.extend(
        [
            "-" * 60,
            "Saved outputs",
            "-" * 60,
            f"Metrics CSV: {OUTPUT_PATH}",
            f"Summary TXT: {SUMMARY_OUTPUT_PATH}",
            f"Plot folder: {PLOT_DIR}",
            "",
            "Saved plots:",
            "- Forecast vs actual",
            "- Residuals over time",
            "- Residual distribution",
            "- Residual ACF/PACF",
            "=" * 60,
        ]
    )

    return "\n".join(lines)


# ------------------------------------------------------------
# 3C. SAVED PLOTS
# ------------------------------------------------------------

# This group saves the visual outputs for each model.
# Each model gets forecast and residual plots.

def save_residual_acf_pacf_plot(
    residuals: pd.Series,
    period_name: str,
    model_name: str,
    plot_dir: Path,
) -> Path:
    # Save ACF/PACF plots of the residuals.
    # Residuals are the model errors.
    # ACF/PACF check whether those errors still have time patterns.
    residuals = residuals.dropna()
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / f"{safe_name(period_name)}_{safe_name(model_name)}_acf_pacf.png"
    lags = max(1, min(28, len(residuals) // 2 - 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(residuals, lags=lags, ax=axes[0])
    axes[0].set_title("Residual ACF")
    plot_pacf(residuals, lags=lags, ax=axes[1])
    axes[1].set_title("Residual PACF")
    fig.suptitle(f"{period_name} -- {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def save_forecast_plot(
    train_y: pd.Series,
    test_y: pd.Series,
    predicted: pd.Series,
    period_name: str,
    model_name: str,
    plot_dir: Path,
) -> Path:
    # Save a line chart comparing recent training data, actual test data, and forecast.
    # This is the main visual for judging the forecast.
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / f"{safe_name(period_name)}_{safe_name(model_name)}_forecast.png"

    recent_train = train_y.tail(90)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent_train.index, recent_train, label="Train", color="gray", linewidth=1)
    ax.plot(test_y.index, test_y, label="Actual", color="steelblue", linewidth=2)
    ax.plot(predicted.index, predicted, label="Forecast", color="tomato", linewidth=2)
    ax.axvline(test_y.index.min(), color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{period_name} -- {model_name}")
    ax.set_ylabel("Daily Departure-Delay Rate (%)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def save_residual_distribution_plot(
    residuals: pd.Series,
    period_name: str,
    model_name: str,
    plot_dir: Path,
) -> Path:
    # Save a histogram of forecast errors.
    # This shows whether forecast errors are mostly near zero or spread out.
    residuals = residuals.dropna()
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / f"{safe_name(period_name)}_{safe_name(model_name)}_residual_distribution.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    residuals.hist(bins=30, ax=ax, color="steelblue")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Zero")
    ax.axvline(residuals.mean(), color="tomato", linewidth=2, label="Mean")
    ax.set_title(f"Residual Distribution -- {period_name} -- {model_name}")
    ax.set_xlabel("Residual: Actual - Forecast")
    ax.set_ylabel("Days")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def save_residual_over_time_plot(
    residuals: pd.Series,
    period_name: str,
    model_name: str,
    plot_dir: Path,
) -> Path:
    # Save residuals over time.
    # This shows whether the model errors change across the test period.
    residuals = residuals.dropna()
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / f"{safe_name(period_name)}_{safe_name(model_name)}_residuals_over_time.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(residuals.index, residuals, color="steelblue", s=25, alpha=0.8)
    ax.plot(residuals.index, residuals, color="steelblue", linewidth=1, alpha=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Residuals Over Time -- {period_name} -- {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual: Actual - Forecast")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


# ------------------------------------------------------------
# 3D. DATA PREPARATION
# ------------------------------------------------------------

# This group prepares the data before modeling.
# It creates the daily time series, holiday features, and train/test periods.

def build_daily_series(path: Path, start_date: str) -> pd.Series:
    # Load cleaned flight data.
    # Turn millions of flight rows into one daily departure-delay-rate time series.
    df = pd.read_csv(path, usecols=["fl_date", "is_delayed"], parse_dates=["fl_date"])

    # Daily departure-delay rate is the percent of flights that departed late each day.
    daily = df.groupby("fl_date")["is_delayed"].mean()
    daily = daily.sort_index()
    daily = daily * 100

    # A daily frequency keeps the time series evenly spaced.
    daily = daily.loc[start_date:]
    daily = daily.asfreq("D")

    # Fill short gaps so the forecasting models receive a continuous series.
    daily = daily.interpolate(method="time", limit=3).dropna()
    if len(daily) < 180:
        raise ValueError(
            f"Need at least 180 daily observations after {start_date}; found {len(daily)}."
        )
    return daily


def build_calendar_features(index: pd.DatetimeIndex, holiday_window: int) -> pd.DataFrame:
    # Create a simple holiday feature for SARIMAX.
    # 1 means the day is near a US holiday, 0 means it is not.
    features = pd.DataFrame(index=index)

    years = range(index.min().year, index.max().year + 1)
    holiday_dates = pd.to_datetime(list(holidays.US(years=years).keys()))

    features["holiday_window"] = 0
    for date in holiday_dates:
        # Mark the holiday itself and nearby days.
        window = pd.date_range(
            date - pd.Timedelta(days=holiday_window),
            date + pd.Timedelta(days=holiday_window),
        )
        features.loc[features.index.isin(window), "holiday_window"] = 1

    return features


def build_prophet_holidays(index: pd.DatetimeIndex, holiday_window: int) -> pd.DataFrame:
    # Create the holiday table in the format Prophet expects.
    years = range(index.min().year, index.max().year + 1)
    holiday_items = list(holidays.US(years=years).items())

    holiday_names = []
    holiday_dates = []

    for date, name in holiday_items:
        holiday_dates.append(date)
        holiday_names.append(name)

    return pd.DataFrame(
        {
            "holiday": holiday_names,
            "ds": pd.to_datetime(holiday_dates),
            "lower_window": -holiday_window,
            "upper_window": holiday_window,
        }
    )


def choose_periods(series: pd.Series, test_days: int) -> dict[str, dict[str, pd.Timestamp]]:
    # Pick the future periods we want to test.
    # The model trains on older days and forecasts these newer days.
    latest_end = series.index.max()

    # The latest test period uses the most recent TEST_DAYS in the dataset.
    latest_split = latest_end - pd.Timedelta(days=test_days - 1)

    if latest_split.month == 1 and latest_split.day == 1 and latest_end.month == 3 and latest_end.day == 31:
        latest_period_name = f"Q1 {latest_end.year}"
    else:
        latest_period_name = f"Latest {test_days} Days"

    periods = {
        latest_period_name: {
            "split_date": latest_split,
            "end_date": latest_end,
        }
    }

    previous_year = latest_end.year - 1

    # Q3 of the previous year gives a second test period for comparison.
    summer_split = pd.Timestamp(f"{previous_year}-07-01")
    summer_end = pd.Timestamp(f"{previous_year}-09-30")
    if series.index.min() < summer_split and series.index.max() >= summer_end:
        periods[f"Q3 {previous_year}"] = {
            "split_date": summer_split,
            "end_date": summer_end,
        }

    return periods


# ------------------------------------------------------------
# 3E. MODEL FITTING
# ------------------------------------------------------------

# This group fits the forecasting models.
# SARIMA and SARIMAX both use statsmodels SARIMAX.
# Prophet uses its own Prophet API.

def fit_sarimax_forecast(
    train_y: pd.Series,
    test_y: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    train_exog: pd.DataFrame | None = None,
    test_exog: pd.DataFrame | None = None,
) -> tuple[pd.Series, object]:
    # Fit a SARIMA/SARIMAX model and forecast the test period.
    # If exog is included, this becomes SARIMAX with outside information like holidays.
    # If exog is None, this behaves like SARIMA.
    model = SARIMAX(
        train_y,
        exog=train_exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    forecast = result.get_forecast(steps=len(test_y), exog=test_exog)
    return forecast.predicted_mean, result


def fit_prophet_forecast(
    train_y: pd.Series,
    test_y: pd.Series,
    prophet_holidays: pd.DataFrame,
) -> pd.Series:
    # Fit a Prophet model and forecast the test period.
    # Prophet expects columns named ds for date and y for the target value.
    from prophet import Prophet

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
    return forecast.set_index("ds")["yhat"]


# ============================================================
# 4. RUN FORECASTING PIPELINE
# ============================================================

# Start here if you want to understand the forecasting story.
# load series -> split train/test -> fit models -> accuracy -> residual checks.

def main() -> None:
    # ========================================================
    # STEP 1: PRINT RUN SETTINGS
    # ========================================================
    # This shows which files the forecasting script will read and write.
    print()
    print("=" * 60)
    print("FORECASTING RUN")
    print("=" * 60)
    print(f"Input file:  {INPUT_PATH}")
    print(f"Metrics CSV: {OUTPUT_PATH}")
    print(f"Summary TXT: {SUMMARY_OUTPUT_PATH}")
    print(f"Plot folder: {PLOT_DIR}")

    # ========================================================
    # STEP 2: BUILD THE TIME SERIES
    # ========================================================
    # This is like creating ts(...) in R.
    # Each point is one day's departure-delay rate.
    daily = build_daily_series(INPUT_PATH, START_DATE)

    # ========================================================
    # STEP 3: CHECK STATIONARITY
    # ========================================================
    # This runs the ADF test before modeling.
    series_tests = run_series_tests(daily)
    print("\nDaily departure-delay-rate stationarity check:")
    print(f"  ADF statistic: {series_tests['adf_stat']:.4f}")
    print(f"  ADF p-value:   {series_tests['adf_pvalue']:.6f}")
    print(f"  p-value < 0.05: {series_tests['adf_pvalue'] < 0.05}")

    # ========================================================
    # STEP 4: CREATE HOLIDAY FEATURES
    # ========================================================
    # SARIMAX and Prophet can use holidays as extra calendar information.
    features = build_calendar_features(daily.index, HOLIDAY_WINDOW)
    prophet_holidays = build_prophet_holidays(daily.index, HOLIDAY_WINDOW)

    # ========================================================
    # STEP 5: CHOOSE TRAIN/TEST PERIODS
    # ========================================================
    # This is like using window(...) in R to create train and test data.
    periods = choose_periods(daily, TEST_DAYS)

    # Store all model results here.
    rows = []

    # ========================================================
    # STEP 6: RUN EACH TEST PERIOD
    # ========================================================
    # Example periods: Q1 2026 and Q3 2025.
    for period_name, dates in periods.items():
        split_date = dates["split_date"]
        end_date = dates["end_date"]

        # Training data is everything before the test period.
        train_y = daily[daily.index < split_date]

        # Test data is the period we want the model to forecast.
        test_y = daily[(daily.index >= split_date) & (daily.index <= end_date)]

        # Skip this period if there is not enough data.
        if len(train_y) < 90 or len(test_y) == 0:
            print(f"Skipping {period_name}: not enough train/test observations.")
            continue

        print(f"\n{'=' * 60}")
        print(f"{period_name}: train through {train_y.index[-1].date()}, test through {test_y.index[-1].date()}")
        print(f"{'=' * 60}")

        # ====================================================
        # STEP 7: FIND AUTO SARIMA ORDER
        # ====================================================
        # This is similar to auto.arima() in R.
        print("\nRunning auto_arima search...")

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
        print(f"Best auto SARIMA order: {auto_order}{auto_seasonal_order}")

        # ====================================================
        # STEP 8: DEFINE MODELS TO COMPARE
        # ====================================================
        # This list controls which models run below.
        model_specs = [
            {
                "name": "Manual SARIMA(1,0,0)(1,0,0,7)",
                "type": "sarimax",
                "order": (1, 0, 0),
                "seasonal_order": (1, 0, 0, 7),
                "cols": [],
            },
            {
                "name": f"Auto SARIMA{auto_order}{auto_seasonal_order}",
                "type": "sarimax",
                "order": auto_order,
                "seasonal_order": auto_seasonal_order,
                "cols": [],
            },
            {
                "name": f"SARIMAX + holiday {auto_order}{auto_seasonal_order}",
                "type": "sarimax",
                "order": auto_order,
                "seasonal_order": auto_seasonal_order,
                "cols": ["holiday_window"],
            },
            {
                "name": "Prophet + holidays",
                "type": "prophet",
            },
        ]

        # ====================================================
        # STEP 9: TRAIN, FORECAST, AND EVALUATE EACH MODEL
        # ====================================================
        for spec in model_specs:
            # Prophet uses its own function because it has a different format from SARIMA.
            if spec["type"] == "prophet":
                predicted = fit_prophet_forecast(train_y, test_y, prophet_holidays)

            # SARIMA and SARIMAX both use the SARIMAX function from statsmodels.
            else:
                cols = spec["cols"]

                # Empty cols means regular SARIMA.
                # holiday_window means SARIMAX uses the holiday flag.
                train_exog = features.loc[train_y.index, cols] if cols else None
                test_exog = features.loc[test_y.index, cols] if cols else None

                predicted, result = fit_sarimax_forecast(
                    train_y,
                    test_y,
                    spec["order"],
                    spec["seasonal_order"],
                    train_exog,
                    test_exog,
                )

            # Compare forecast to actual values.
            # This is similar to accuracy(fit, test) in R.
            metrics = evaluate(test_y, predicted)

            # Residual = actual departure-delay rate - forecast departure-delay rate.
            residuals = test_y - predicted

            # This includes Ljung-Box, similar to Box.test(...) in R.
            residual_tests = run_residual_tests(residuals)

            # Save forecast and residual plots for this model.
            forecast_plot = save_forecast_plot(
                train_y,
                test_y,
                predicted,
                period_name,
                spec["name"],
                PLOT_DIR,
            )
            residual_distribution_plot = save_residual_distribution_plot(
                residuals,
                period_name,
                spec["name"],
                PLOT_DIR,
            )
            residual_over_time_plot = save_residual_over_time_plot(
                residuals,
                period_name,
                spec["name"],
                PLOT_DIR,
            )
            residual_acf_pacf_plot = save_residual_acf_pacf_plot(
                residuals,
                period_name,
                spec["name"],
                PLOT_DIR,
            )

            # Print one short result line in the console.
            print_model_results(
                spec["name"],
                metrics,
                residual_tests,
            )

            # Store this model's metrics, residual tests, and plot paths.
            rows.append(
                {
                    "period": period_name,
                    "model": spec["name"],
                    **metrics,
                    **residual_tests,
                    "forecast_plot": str(forecast_plot),
                    "residual_distribution_plot": str(residual_distribution_plot),
                    "residual_over_time_plot": str(residual_over_time_plot),
                    "residual_acf_pacf_plot": str(residual_acf_pacf_plot),
                }
            )

    # ========================================================
    # STEP 10: SAVE FINAL OUTPUTS
    # ========================================================
    # Save the detailed metrics CSV and the shorter text summary.
    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    forecast_summary = build_forecast_summary(results)
    SUMMARY_OUTPUT_PATH.write_text(forecast_summary)

    print("\n" + forecast_summary)


# Start the forecasting script.
if __name__ == "__main__":
    main()
