# Aviation Delay Analysis — Project Guide

This guide documents the active project workflow. Run commands from the repository root. Each active script resolves the project directory automatically, so no local path changes are required.

## Main Run Order

Run these scripts in this order when rebuilding the project:

1. `code/combine_monthly_files.py`
   - Combines the monthly `YYYY:MM.csv` files into one raw file.
   - Output: `raw_data/flights_all_raw.csv`

2. `code/clean_flights.py`
   - Cleans the merged raw file.
   - Creates simple features used by EDA, forecasting, SQL, and classification.
   - Output: `processed/flights_all_cleaned.csv`

3. `code/eda_simple.py`
   - Creates the main exploratory charts in `plots/`.

4. `code/EDA_forecasting.py`
   - Simple forecasting EDA before modeling.
   - Checks daily delay rate, stationarity, ACF, and PACF.

5. `code/forecasting_calendar_no_defs.py`
   - Uses Q3 2025 to compare forecasting baselines, SARIMA, SARIMAX, and Prophet.
   - Evaluates the selected Prophet model once on Q1 2026.
   - Saves metrics in `query_results/` and diagnostics in `plots/forecast_residuals_no_defs/`.

6. `code/classification_simple.py`
   - Compares classification models.
   - Saves model comparison, feature importance, and SHAP outputs.

7. `code/load_to_postgres.py`
   - Optional.
   - Loads the cleaned CSV into PostgreSQL.

8. `code/export_tableau_results.py`
   - Optional; requires PostgreSQL.
   - Executes the final SQL reporting queries and refreshes `sql_exports/`.

## Setup

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The optional PostgreSQL steps read connection settings from `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, and `DB_NAME`. Do not store real credentials in the repository.

## Active Data Files

- `raw_data/<year>/<year>:<month>.csv`
  - Clean monthly raw files used by the merge script.

- `raw_data/flights_all_raw.csv`
  - Combined raw file created by `combine_monthly_files.py`.

- `processed/flights_all_cleaned.csv`
  - Main cleaned dataset used by the rest of the project.

## Output Folders

- `query_results/`
  - CSV and text outputs from forecasting, classification, and supporting analysis.

- `sql_exports/`
  - Compact Tableau-ready CSV files generated from PostgreSQL.

- `plots/`
  - EDA, forecasting, classification, and Tableau-related visual outputs.

## Archive Folder

The local `archive/` folder contains superseded files, duplicate downloads, experiments, and scratch scripts. It is excluded from Git because it is not required to understand or reproduce the active project.
