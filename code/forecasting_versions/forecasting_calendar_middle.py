"""
forecasting_calendar_middle.py

MIDDLE-POINT VERSION

This version keeps the core analysis visible in the main script:
- MAE/RMSE/MAPE are calculated directly in the model loop.
- residuals are calculated directly in the model loop.
- Ljung-Box and Jarque-Bera are calculated directly in the model loop.

Only plotting is kept in helper functions because plot formatting is long and
repetitive.
"""


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera

# Main project folder.
PROJECT_ROOT = Path("/Users/daviddupre/Documents/PERSONAL PROJECTS PORTFOLIO/aviation_delay_analysis")

# Save plots to files instead of opening pop-up windows.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# ============================================================
# 2. CONFIGURATION
# ============================================================

warnings.filterwarnings("ignore", category=ConvergenceWarning)

INPUT_PATH = PROJECT_ROOT / "processed" / "flights_all_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "query_results" / "calendar_forecast_metrics_middle.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "query_results" / "calendar_forecast_summary_middle.txt"
PLOT_DIR = PROJECT_ROOT / "plots" / "forecast_residuals_middle"

START_DATE = "2022-01-01"
TEST_DAYS = 90
HOLIDAY_WINDOW = 2


# ============================================================
# 3. PLOTTING HELPER FUNCTIONS ONLY
# ============================================================

def clean_file_name(value):
    # Turn names like "SARIMAX + holiday" into file-safe text.
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in cleaned.split("_") if part)


def save_forecast_plot(train_y, test_y, predicted, period_name, model_name):
    # Save train/test/forecast line plot.
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / f"{clean_file_name(period_name)}_{clean_file_name(model_name)}_forecast.png"

    recent_train = train_y.tail(90)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent_train.index, recent_train, label="Train", color="gray", linewidth=1)
    ax.plot(test_y.index, test_y, label="Actual", color="steelblue", linewidth=2)
    ax.plot(predicted.index, predicted, label="Forecast", color="tomato", linewidth=2)
    ax.axvline(test_y.index.min(), color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{period_name} -- {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Departure-Delay Rate (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path


def save_residual_distribution_plot(residuals, period_name, model_name):
    # Save histogram of forecast errors.
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / f"{clean_file_name(period_name)}_{clean_file_name(model_name)}_residual_distribution.png"

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


def save_residual_over_time_plot(residuals, period_name, model_name):
    # Save residuals across the test dates.
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / f"{clean_file_name(period_name)}_{clean_file_name(model_name)}_residuals_over_time.png"

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


def save_residual_acf_pacf_plot(residuals, period_name, model_name):
    # Save residual ACF/PACF plots.
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / f"{clean_file_name(period_name)}_{clean_file_name(model_name)}_acf_pacf.png"

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


# ============================================================
# 4. BUILD DAILY TIME SERIES
# ============================================================

print()
print("=" * 60)
print("FORECASTING RUN - MIDDLE VERSION")
print("=" * 60)
print(f"Input file: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH, usecols=["fl_date", "is_delayed"], parse_dates=["fl_date"])

daily = df.groupby("fl_date")["is_delayed"].mean()
daily = daily.sort_index()
daily = daily * 100
daily = daily.loc[START_DATE:]
daily = daily.asfreq("D")
daily = daily.interpolate(method="time", limit=3).dropna()


# ============================================================
# 5. STATIONARITY CHECK
# ============================================================

adf_result = adfuller(daily)
adf_stat = adf_result[0]
adf_pvalue = adf_result[1]

print()
print("Stationarity check:")
print(f"ADF statistic: {adf_stat:.4f}")
print(f"ADF p-value:   {adf_pvalue:.6f}")


# ============================================================
# 6. HOLIDAY FEATURES
# ============================================================

years = range(daily.index.min().year, daily.index.max().year + 1)
holiday_dates = pd.to_datetime(list(holidays.US(years=years).keys()))

features = pd.DataFrame(index=daily.index)
features["holiday_window"] = 0

for date in holiday_dates:
    window = pd.date_range(
        date - pd.Timedelta(days=HOLIDAY_WINDOW),
        date + pd.Timedelta(days=HOLIDAY_WINDOW),
    )
    features.loc[features.index.isin(window), "holiday_window"] = 1

holiday_items = list(holidays.US(years=years).items())
prophet_holidays = pd.DataFrame(
    {
        "holiday": [name for date, name in holiday_items],
        "ds": pd.to_datetime([date for date, name in holiday_items]),
        "lower_window": -HOLIDAY_WINDOW,
        "upper_window": HOLIDAY_WINDOW,
    }
)


# ============================================================
# 7. TRAIN/TEST PERIODS
# ============================================================

latest_end = daily.index.max()
latest_split = latest_end - pd.Timedelta(days=TEST_DAYS - 1)

if latest_split.month == 1 and latest_split.day == 1 and latest_end.month == 3 and latest_end.day == 31:
    latest_period_name = f"Q1 {latest_end.year}"
else:
    latest_period_name = f"Latest {TEST_DAYS} Days"

periods = {
    latest_period_name: {
        "split_date": latest_split,
        "end_date": latest_end,
    }
}

previous_year = latest_end.year - 1
summer_split = pd.Timestamp(f"{previous_year}-07-01")
summer_end = pd.Timestamp(f"{previous_year}-09-30")

if daily.index.min() < summer_split and daily.index.max() >= summer_end:
    periods[f"Q3 {previous_year}"] = {
        "split_date": summer_split,
        "end_date": summer_end,
    }


# ============================================================
# 8. FORECASTING MODELS
# ============================================================

rows = []

for period_name, dates in periods.items():
    split_date = dates["split_date"]
    end_date = dates["end_date"]

    train_y = daily[daily.index < split_date]
    test_y = daily[(daily.index >= split_date) & (daily.index <= end_date)]

    print()
    print("=" * 60)
    print(f"{period_name}: train through {train_y.index[-1].date()}, test through {test_y.index[-1].date()}")
    print("=" * 60)

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
    auto_order = auto_model.order
    auto_seasonal_order = auto_model.seasonal_order

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

    for spec in model_specs:
        model_name = spec["name"]

        if spec["type"] == "prophet":
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

        else:
            cols = spec["cols"]
            train_exog = features.loc[train_y.index, cols] if cols else None
            test_exog = features.loc[test_y.index, cols] if cols else None

            model = SARIMAX(
                train_y,
                exog=train_exog,
                order=spec["order"],
                seasonal_order=spec["seasonal_order"],
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False)
            forecast = result.get_forecast(steps=len(test_y), exog=test_exog)
            predicted = forecast.predicted_mean

        # These metrics stay visible in the main script.
        mae = mean_absolute_error(test_y, predicted)
        rmse = np.sqrt(mean_squared_error(test_y, predicted))
        mape = np.mean(np.abs((test_y - predicted) / test_y)) * 100

        # Residuals stay visible in the main script.
        residuals = test_y - predicted
        residuals = residuals.dropna()

        lb_lags = []
        for lag in [7, 14, 21, 28]:
            if lag < len(residuals):
                lb_lags.append(lag)

        if lb_lags:
            lb = acorr_ljungbox(residuals, lags=lb_lags, return_df=True)
        else:
            lb = pd.DataFrame()

        jb_stat, jb_pvalue = jarque_bera(residuals)

        forecast_plot = save_forecast_plot(train_y, test_y, predicted, period_name, model_name)
        residual_distribution_plot = save_residual_distribution_plot(residuals, period_name, model_name)
        residual_over_time_plot = save_residual_over_time_plot(residuals, period_name, model_name)
        residual_acf_pacf_plot = save_residual_acf_pacf_plot(residuals, period_name, model_name)

        row = {
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
            "forecast_plot": str(forecast_plot),
            "residual_distribution_plot": str(residual_distribution_plot),
            "residual_over_time_plot": str(residual_over_time_plot),
            "residual_acf_pacf_plot": str(residual_acf_pacf_plot),
        }

        for lag in lb_lags:
            row[f"ljung_box_pvalue_lag_{lag}"] = lb.loc[lag, "lb_pvalue"]

        rows.append(row)

        print(f"{model_name}: MAE {mae:.2f}% | RMSE {rmse:.2f}% | MAPE {mape:.2f}%")


# ============================================================
# 9. SAVE FINAL OUTPUTS
# ============================================================

results = pd.DataFrame(rows)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

summary_lines = [
    "=" * 60,
    "FORECASTING SUMMARY - MIDDLE VERSION",
    "=" * 60,
]

for period_name, period_results in results.groupby("period", sort=False):
    period_results = period_results.sort_values("rmse").reset_index(drop=True)
    best = period_results.iloc[0]

    summary_lines.extend(
        [
            "",
            "-" * 60,
            period_name,
            "-" * 60,
            f"Best model by RMSE: {best['model']}",
            f"MAE:                {best['mae']:.2f}%",
            f"RMSE:               {best['rmse']:.2f}%",
            f"MAPE:               {best['mape']:.2f}%",
        ]
    )

summary_lines.extend(
    [
        "",
        f"Metrics CSV: {OUTPUT_PATH}",
        f"Summary TXT: {SUMMARY_OUTPUT_PATH}",
        f"Plot folder: {PLOT_DIR}",
    ]
)

forecast_summary = "\n".join(summary_lines)
SUMMARY_OUTPUT_PATH.write_text(forecast_summary)
print()
print(forecast_summary)
