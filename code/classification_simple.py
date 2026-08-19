# ============================================================
# 1. IMPORT LIBRARIES
# ==================== ========================================

# os is used to set the matplotlib cache folder.
import os

# Path makes file paths easier to build.
from pathlib import Path

# Main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep matplotlib cache files inside the project folder.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

# Main data, plotting, modeling, and explanation libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# To explain model predictions and create SHAP feature-importance results.
import shap
from scipy import sparse
# Import classification algorithms.
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
# To fill missing values.
from sklearn.impute import SimpleImputer
#Import logistic regression.
from sklearn.linear_model import LogisticRegression
# Import model evaluation scores.
from sklearn.metrics import (
    # Overall % correct.
    accuracy_score,
    # Shows and incorrect predictions.
    confusion_matrix,
    # Balance between precision and recall.
    f1_score,
    # Of flights predicted delayed, how many were actually delayed?
    precision_score,
    # Of flights actually delayed, how many did the model catch?
    recall_score,
    # Measures how well the model separates delayed vs not delayed using probabilities.
    roc_auc_score,
)

# turns text categories into 0/1 columns.
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Imports XGBoost classifier.
from xgboost import XGBClassifier


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Main input file created by the cleaning script.
INPUT_PATH = PROJECT_ROOT / "processed" / "flights_all_cleaned.csv"

# Output files created by this script.
OUTPUT_PATH = PROJECT_ROOT / "query_results" / "simple_classification_metrics.txt"
COMPARISON_OUTPUT = PROJECT_ROOT / "query_results" / "classification_model_comparison.csv"
PLOT_DIR = PROJECT_ROOT / "plots" / "classification"

# NROWS can limit the number of source rows, but the cleaned file is
# arranged by date. A small NROWS value may remove the later validation
# and test periods, so keep this as None for the time-based analysis.
NROWS = None

# SAMPLE_FRAC controls how much of the full dataset is sampled for modeling.
# 0.10 means 10% of the cleaned data.
# For a quicker test that still keeps every year, temporarily use 0.01.
SAMPLE_FRAC = 0.10

# CHUNKSIZE lets pandas read the large CSV in smaller pieces.
CHUNKSIZE = 500_000

# RANDOM_STATE makes the random sampling and train/test split repeatable.
RANDOM_STATE = 42

# SHAP_SAMPLE controls how many rows are explained by SHAP.
# A smaller value can be used for a quick test.
SHAP_SAMPLE = 1000

# Time-based classification split:
# Train on earlier flights, validate on 2025, and test on Q1 2026.
TRAIN_END_DATE = "2024-12-31"
VALIDATION_START_DATE = "2025-01-01"
VALIDATION_END_DATE = "2025-12-31"
TEST_START_DATE = "2026-01-01"
TEST_END_DATE = "2026-03-31"

# This is the classification target:
# 1 means the flight departed 15+ minutes late.
# 0 means the flight did not depart 15+ minutes late.
TARGET = "is_delayed"

# Date column used for time-based validation.
DATE_COLUMN = "fl_date"

# Numeric features are already numbers.
NUMERIC_FEATURES = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "dep_hour",
    "is_peak_hour",
    "is_weekend",
    "distance",
]

# Categorical features are text labels.
# These will be converted into 0/1 dummy columns by OneHotEncoder.
CATEGORICAL_FEATURES = [
    "op_unique_carrier",
    "origin_state_abr",
    "dest_state_abr",
    "route",
]

# Columns needed from the cleaned data.
USE_COLUMNS = [DATE_COLUMN, TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Create output folders if they do not already exist.
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION RUN")
print("=" * 60)
print(f"Input file:     {INPUT_PATH}")
print(f"Report:         {OUTPUT_PATH}")
print(f"Comparison CSV: {COMPARISON_OUTPUT}")
print(f"Plot folder:    {PLOT_DIR}")

# ------------------------------------------------------------
# 3A. READ AND SAMPLE THE DATA IN CHUNKS
# ------------------------------------------------------------

# Reading all 36 million rows before sampling would require a large
# amount of memory. Instead, read one chunk at a time and keep the
# requested percentage from every chunk.
#
# Sampling every chunk also preserves rows from all years. This is
# important because the model uses older data for training, 2025 for
# validation, and Q1 2026 for the final test.
sampled_chunks = []
chunk_number = 0

for chunk in pd.read_csv(
    INPUT_PATH,
    usecols=USE_COLUMNS,
    nrows=NROWS,
    parse_dates=[DATE_COLUMN],
    chunksize=CHUNKSIZE,
):
    chunk_sample = chunk.sample(
        frac=SAMPLE_FRAC,
        random_state=RANDOM_STATE + chunk_number,
    )
    sampled_chunks.append(chunk_sample)
    chunk_number += 1

# Combine the smaller samples after every source chunk has been read.
df = pd.concat(sampled_chunks, ignore_index=True)

# Free the temporary list because the combined dataframe is now ready.
del sampled_chunks

# Remove rows where the target is missing.
df = df.dropna(subset=[TARGET])

print()
print(f"Rows used: {len(df):,}")
print(f"Source chunks read: {chunk_number:,}")


# ============================================================
# 4. SPLIT FEATURES AND TARGET
# ============================================================

# Sort by date so the time-based split is easy to understand.
df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

# y contains the answer the model is trying to predict.
y = df[TARGET].astype(int)

# This is the percent of flights that are delayed in the modeling sample.
delay_base_rate = y.mean() * 100

# Time-based split:
# Train = older flights.
# Validation = 2025 flights used to compare models.
# Test = Q1 2026 flights used as the final future-style holdout.
train_rows = df[DATE_COLUMN] <= TRAIN_END_DATE
validation_rows = (df[DATE_COLUMN] >= VALIDATION_START_DATE) & (df[DATE_COLUMN] <= VALIDATION_END_DATE)
test_rows = (df[DATE_COLUMN] >= TEST_START_DATE) & (df[DATE_COLUMN] <= TEST_END_DATE)

train_df = df[train_rows].copy()
validation_df = df[validation_rows].copy()
test_df = df[test_rows].copy()

# Stop with a clear message if a date period contains no sampled rows.
# This can happen if NROWS is set too low on a date-sorted source file.
if train_df.empty:
    raise ValueError("The training dataset is empty. Check the date range and sampling settings.")

if validation_df.empty:
    raise ValueError(
        "The validation dataset is empty. Keep NROWS=None so 2025 is included."
    )

if test_df.empty:
    raise ValueError(
        "The test dataset is empty. Keep NROWS=None so Q1 2026 is included."
    )

# X contains the model inputs.
X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
X_validation = validation_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

# y contains the answer the model is trying to predict.
y_train = train_df[TARGET].astype(int)
y_validation = validation_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)

print(f"Training rows:   {len(train_df):,} through {TRAIN_END_DATE}")
print(f"Validation rows: {len(validation_df):,} from {VALIDATION_START_DATE} to {VALIDATION_END_DATE}")
print(f"Test rows:       {len(test_df):,} from {TEST_START_DATE} to {TEST_END_DATE}")


# ============================================================
# 5. PREPROCESSING
# ============================================================
# Before modeling, all inputs need to be numeric and clean.
#
# Numeric columns need:
# 1. missing values filled
# 2. scaling
#
# Text/category columns need:
# 1. missing values filled
# 2. conversion into 0/1 dummy columns
#
# Important rule: fit preprocessing on training data only.
# Then apply the same fitted preprocessing to validation and test data.

# Fill missing numeric values using training data medians.
numeric_imputer = SimpleImputer(strategy="median")
numeric_imputer.fit(X_train[NUMERIC_FEATURES])

X_train_numeric = numeric_imputer.transform(X_train[NUMERIC_FEATURES])
X_validation_numeric = numeric_imputer.transform(X_validation[NUMERIC_FEATURES])
X_test_numeric = numeric_imputer.transform(X_test[NUMERIC_FEATURES])

# Scale numeric values using training data only.
numeric_scaler = StandardScaler()
numeric_scaler.fit(X_train_numeric)
# Fill missing numeric values in train, validation, and test.
# The missing-value rule was learned from the training data only.
X_train_numeric = numeric_scaler.transform(X_train_numeric)
X_validation_numeric = numeric_scaler.transform(X_validation_numeric)
X_test_numeric = numeric_scaler.transform(X_test_numeric)

# Fill missing text values using the most common training value.
categorical_imputer = SimpleImputer(strategy="most_frequent")
categorical_imputer.fit(X_train[CATEGORICAL_FEATURES])

# Fill missing text/category values in train, validation, and test.
# The most-common-value rule was learned from the training data only.
X_train_categorical = categorical_imputer.transform(X_train[CATEGORICAL_FEATURES])
X_validation_categorical = categorical_imputer.transform(X_validation[CATEGORICAL_FEATURES])
X_test_categorical = categorical_imputer.transform(X_test[CATEGORICAL_FEATURES])

# Convert text categories into 0/1 dummy columns.
onehot_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=100)

# Learn the category-to-dummy-column mapping from training data only.
onehot_encoder.fit(X_train_categorical)

# Apply the same dummy-column mapping to train, validation, and test.
X_train_categorical = onehot_encoder.transform(X_train_categorical)
X_validation_categorical = onehot_encoder.transform(X_validation_categorical)
X_test_categorical = onehot_encoder.transform(X_test_categorical)

# Combine numeric columns and dummy columns into final model-ready data.
X_train_ready = sparse.hstack([X_train_numeric, X_train_categorical], format="csr")
X_validation_ready = sparse.hstack([X_validation_numeric, X_validation_categorical], format="csr")
X_test_ready = sparse.hstack([X_test_numeric, X_test_categorical], format="csr")

# Save the final feature names for feature-importance plots and SHAP.
numeric_feature_names = np.array(NUMERIC_FEATURES)
categorical_feature_names = onehot_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
feature_names = np.concatenate([numeric_feature_names, categorical_feature_names])

# ============================================================
# 6. CREATE MODELS
# ============================================================
# The data has already been preprocessed manually.
# Each model below is now just the classifier itself.

# Logistic Regression:
logistic_model = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
)

# AdaBoost:
adaboost_model = AdaBoostClassifier(
    n_estimators=80,
    learning_rate=0.5,
    random_state=RANDOM_STATE,
)

# Random Forest:
random_forest_model = RandomForestClassifier(
    n_estimators=80,
    max_depth=14,
    min_samples_leaf=50,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# Extra Trees:
extra_trees_model = ExtraTreesClassifier(
    n_estimators=80,
    max_depth=14,
    min_samples_leaf=50,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# XGBoost:
xgboost_model = XGBClassifier(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# ============================================================
# 7. TRAIN MODELS AND CALCULATE METRICS
# ============================================================

# comparison_rows stores one result row per model.
comparison_rows = []

# fitted_models stores the trained models for later use.
fitted_models = {}

# ------------------------------------------------------------
# 7A. LOGISTIC REGRESSION
# ------------------------------------------------------------

model_name = "Logistic Regression"
print()
print(f"Training {model_name}...")

# Train the model on the preprocessed training data.
logistic_model.fit(X_train_ready, y_train)
fitted_models[model_name] = logistic_model

# Score the model on the 2025 validation data.
validation_predicted = logistic_model.predict(X_validation_ready)
validation_predicted_proba = logistic_model.predict_proba(X_validation_ready)[:, 1]

validation_roc_auc = roc_auc_score(y_validation, validation_predicted_proba)
validation_accuracy = accuracy_score(y_validation, validation_predicted)
validation_precision = precision_score(y_validation, validation_predicted, zero_division=0)
validation_recall = recall_score(y_validation, validation_predicted, zero_division=0)
validation_f1 = f1_score(y_validation, validation_predicted, zero_division=0)

# Save Logistic Regression validation metrics.
# Q1 2026 is not scored yet because it is the final holdout.
comparison_rows.append(
    {
        "model": model_name,
        "validation_roc_auc": validation_roc_auc,
        "validation_accuracy": validation_accuracy,
        "validation_precision_delayed": validation_precision,
        "validation_recall_delayed": validation_recall,
        "validation_f1_delayed": validation_f1,
    }
)

print(
    f"  Validation ROC-AUC {validation_roc_auc * 100:>5.1f}% | "
    f"Validation Recall {validation_recall * 100:>5.1f}% | "
    f"Validation F1 {validation_f1 * 100:>5.1f}%"
)
# Logistic Regression gives each feature a coefficient.
# Bigger absolute coefficient = stronger effect on the prediction.
importance = np.abs(logistic_model.coef_[0])

# Get the positions of the top 20 strongest features.
top_idx = np.argsort(importance)[-20:]

# Get the names of those top 20 features.
top_features = pd.Series(feature_names[top_idx])

# Clean the feature names so the plot is easier to read.
top_features = top_features.str.replace("numeric__", "", regex=False)
top_features = top_features.str.replace("categorical__", "", regex=False)
top_features = top_features.str.replace("_", " ", regex=False)
top_features = top_features.str.title()

# Get the importance values for the same top 20 features.
top_importance = importance[top_idx]

# Save a horizontal bar chart showing the top 20 most important Logistic Regression features.
importance_plot = PLOT_DIR / "logistic_regression_feature_importance.png"
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_features, top_importance, color="steelblue")
ax.set_title("Top 20 Features - Logistic Regression")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(importance_plot)
plt.close(fig)
print(f"  Feature importance plot: {importance_plot}")

# ------------------------------------------------------------
# 7B. ADABOOST
# ------------------------------------------------------------

model_name = "AdaBoost"
print()
print(f"Training {model_name}...")

# Train AdaBoost on the preprocessed training data.
adaboost_model.fit(X_train_ready, y_train)
fitted_models[model_name] = adaboost_model

# Score the model on the 2025 validation data.
validation_predicted = adaboost_model.predict(X_validation_ready)
validation_predicted_proba = adaboost_model.predict_proba(X_validation_ready)[:, 1]

validation_roc_auc = roc_auc_score(y_validation, validation_predicted_proba)
validation_accuracy = accuracy_score(y_validation, validation_predicted)
validation_precision = precision_score(y_validation, validation_predicted, zero_division=0)
validation_recall = recall_score(y_validation, validation_predicted, zero_division=0)
validation_f1 = f1_score(y_validation, validation_predicted, zero_division=0)

# Save AdaBoost validation metrics.
# Q1 2026 is not scored yet because it is the final holdout.
comparison_rows.append(
    {
        "model": model_name,
        "validation_roc_auc": validation_roc_auc,
        "validation_accuracy": validation_accuracy,
        "validation_precision_delayed": validation_precision,
        "validation_recall_delayed": validation_recall,
        "validation_f1_delayed": validation_f1,
    }
)

print(
    f"  Validation ROC-AUC {validation_roc_auc * 100:>5.1f}% | "
    f"Validation Recall {validation_recall * 100:>5.1f}% | "
    f"Validation F1 {validation_f1 * 100:>5.1f}%"
)

# Get the top 20 most important AdaBoost features for the feature-importance plot.
importance = adaboost_model.feature_importances_
top_idx = np.argsort(importance)[-20:]
top_features = pd.Series(feature_names[top_idx])
top_features = top_features.str.replace("numeric__", "", regex=False)
top_features = top_features.str.replace("categorical__", "", regex=False)
top_features = top_features.str.replace("_", " ", regex=False)
top_features = top_features.str.title()
top_importance = importance[top_idx]

# Create and save a horizontal bar chart showing the top 20 most important AdaBoost features.

importance_plot = PLOT_DIR / "adaboost_feature_importance.png"
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_features, top_importance, color="steelblue")
ax.set_title("Top 20 Features - AdaBoost")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(importance_plot)
plt.close(fig)
print(f"  Feature importance plot: {importance_plot}")


# ------------------------------------------------------------
# 7C. RANDOM FOREST
# ------------------------------------------------------------

model_name = "Random Forest"
print()
print(f"Training {model_name}...")

# Train Random Forest on the preprocessed training data.
random_forest_model.fit(X_train_ready, y_train)
fitted_models[model_name] = random_forest_model

# Score the model on the 2025 validation data.
validation_predicted = random_forest_model.predict(X_validation_ready)
validation_predicted_proba = random_forest_model.predict_proba(X_validation_ready)[:, 1]

validation_roc_auc = roc_auc_score(y_validation, validation_predicted_proba)
validation_accuracy = accuracy_score(y_validation, validation_predicted)
validation_precision = precision_score(y_validation, validation_predicted, zero_division=0)
validation_recall = recall_score(y_validation, validation_predicted, zero_division=0)
validation_f1 = f1_score(y_validation, validation_predicted, zero_division=0)

# Save Random Forest validation metrics.
# Q1 2026 is not scored yet because it is the final holdout.
comparison_rows.append(
    {
        "model": model_name,
        "validation_roc_auc": validation_roc_auc,
        "validation_accuracy": validation_accuracy,
        "validation_precision_delayed": validation_precision,
        "validation_recall_delayed": validation_recall,
        "validation_f1_delayed": validation_f1,
    }
)

print(
    f"  Validation ROC-AUC {validation_roc_auc * 100:>5.1f}% | "
    f"Validation Recall {validation_recall * 100:>5.1f}% | "
    f"Validation F1 {validation_f1 * 100:>5.1f}%"
)

# Get the top 20 most important Random Forest features.
importance = random_forest_model.feature_importances_
top_idx = np.argsort(importance)[-20:]
top_features = pd.Series(feature_names[top_idx])
top_features = top_features.str.replace("numeric__", "", regex=False)
top_features = top_features.str.replace("categorical__", "", regex=False)
top_features = top_features.str.replace("_", " ", regex=False)
top_features = top_features.str.title()
top_importance = importance[top_idx]

# Save a horizontal bar chart showing the top 20 Random Forest features.
importance_plot = PLOT_DIR / "random_forest_feature_importance.png"
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_features, top_importance, color="steelblue")
ax.set_title("Top 20 Features - Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(importance_plot)
plt.close(fig)
print(f"  Feature importance plot: {importance_plot}")


# ------------------------------------------------------------
# 7D. EXTRA TREES
# ------------------------------------------------------------

model_name = "Extra Trees"
print()
print(f"Training {model_name}...")

# Train Extra Trees on the preprocessed training data.
extra_trees_model.fit(X_train_ready, y_train)
fitted_models[model_name] = extra_trees_model

# Score the model on the 2025 validation data.
validation_predicted = extra_trees_model.predict(X_validation_ready)
validation_predicted_proba = extra_trees_model.predict_proba(X_validation_ready)[:, 1]

validation_roc_auc = roc_auc_score(y_validation, validation_predicted_proba)
validation_accuracy = accuracy_score(y_validation, validation_predicted)
validation_precision = precision_score(y_validation, validation_predicted, zero_division=0)
validation_recall = recall_score(y_validation, validation_predicted, zero_division=0)
validation_f1 = f1_score(y_validation, validation_predicted, zero_division=0)

# Save Extra Trees validation metrics.
# Q1 2026 is not scored yet because it is the final holdout.
comparison_rows.append(
    {
        "model": model_name,
        "validation_roc_auc": validation_roc_auc,
        "validation_accuracy": validation_accuracy,
        "validation_precision_delayed": validation_precision,
        "validation_recall_delayed": validation_recall,
        "validation_f1_delayed": validation_f1,
    }
)

print(
    f"  Validation ROC-AUC {validation_roc_auc * 100:>5.1f}% | "
    f"Validation Recall {validation_recall * 100:>5.1f}% | "
    f"Validation F1 {validation_f1 * 100:>5.1f}%"
)

# Get the top 20 most important Extra Trees features.
importance = extra_trees_model.feature_importances_
top_idx = np.argsort(importance)[-20:]
top_features = pd.Series(feature_names[top_idx])
top_features = top_features.str.replace("numeric__", "", regex=False)
top_features = top_features.str.replace("categorical__", "", regex=False)
top_features = top_features.str.replace("_", " ", regex=False)
top_features = top_features.str.title()
top_importance = importance[top_idx]

# Save a horizontal bar chart showing the top 20 Extra Trees features.
importance_plot = PLOT_DIR / "extra_trees_feature_importance.png"
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_features, top_importance, color="steelblue")
ax.set_title("Top 20 Features - Extra Trees")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(importance_plot)
plt.close(fig)
print(f"  Feature importance plot: {importance_plot}")


# ------------------------------------------------------------
# 7E. XGBOOST
# ------------------------------------------------------------

model_name = "XGBoost"
print()
print(f"Training {model_name}...")

# Train XGBoost on the preprocessed training data.
xgboost_model.fit(X_train_ready, y_train)
fitted_models[model_name] = xgboost_model

# Score the model on the 2025 validation data.
validation_predicted = xgboost_model.predict(X_validation_ready)
validation_predicted_proba = xgboost_model.predict_proba(X_validation_ready)[:, 1]

validation_roc_auc = roc_auc_score(y_validation, validation_predicted_proba)
validation_accuracy = accuracy_score(y_validation, validation_predicted)
validation_precision = precision_score(y_validation, validation_predicted, zero_division=0)
validation_recall = recall_score(y_validation, validation_predicted, zero_division=0)
validation_f1 = f1_score(y_validation, validation_predicted, zero_division=0)

# Save XGBoost validation metrics.
# Q1 2026 is not scored yet because it is the final holdout.
comparison_rows.append(
    {
        "model": model_name,
        "validation_roc_auc": validation_roc_auc,
        "validation_accuracy": validation_accuracy,
        "validation_precision_delayed": validation_precision,
        "validation_recall_delayed": validation_recall,
        "validation_f1_delayed": validation_f1,
    }
)

print(
    f"  Validation ROC-AUC {validation_roc_auc * 100:>5.1f}% | "
    f"Validation Recall {validation_recall * 100:>5.1f}% | "
    f"Validation F1 {validation_f1 * 100:>5.1f}%"
)

# Get the top 20 most important XGBoost features for the feature-importance plot.
importance = xgboost_model.feature_importances_
top_idx = np.argsort(importance)[-20:]
top_features = pd.Series(feature_names[top_idx])
top_features = top_features.str.replace("numeric__", "", regex=False)
top_features = top_features.str.replace("categorical__", "", regex=False)
top_features = top_features.str.replace("_", " ", regex=False)
top_features = top_features.str.title()
top_importance = importance[top_idx]

# Create and save a horizontal bar chart showing the top 20 most important XGBoost features.
importance_plot = PLOT_DIR / "xgboost_feature_importance.png"
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_features, top_importance, color="steelblue")
ax.set_title("Top 20 Features - XGBoost")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(importance_plot)
plt.close(fig)
print(f"  Feature importance plot: {importance_plot}")


# ============================================================
# 8. MODEL COMPARISON TABLE
# ============================================================

# Turn the saved model rows into one comparison table.
comparison = pd.DataFrame(comparison_rows)

# Sort models by validation ROC-AUC from highest to lowest.
comparison = comparison.sort_values("validation_roc_auc", ascending=False).reset_index(drop=True)

# Save the full comparison table.
comparison.to_csv(COMPARISON_OUTPUT, index=False)


# ============================================================
# 9. SELECT THE MODEL AND RUN THE FINAL Q1 2026 TEST
# ============================================================

# The comparison table already ranks all models from best to worst.
# Select the first row using validation ROC-AUC only.
# The Q1 2026 test data did not influence this choice.
selected_row = comparison.iloc[0]
selected_model_name = selected_row["model"]
selected_model = fitted_models[selected_model_name]

# Make Q1 2026 predictions only after the model has been selected.
predicted = selected_model.predict(X_test_ready)
predicted_proba = selected_model.predict_proba(X_test_ready)[:, 1]

# Calculate the selected model's final Q1 2026 metrics.
test_roc_auc = roc_auc_score(y_test, predicted_proba)
test_accuracy = accuracy_score(y_test, predicted)
test_precision = precision_score(y_test, predicted, zero_division=0)
test_recall = recall_score(y_test, predicted, zero_division=0)
test_f1 = f1_score(y_test, predicted, zero_division=0)

# Confusion matrix values.
cm = confusion_matrix(y_test, predicted)
tn = cm[0, 0]
fp = cm[0, 1]
fn = cm[1, 0]
tp = cm[1, 1]
total = tn + fp + fn + tp

# Print the selected model and final holdout results.
print()
print("=" * 60)
print("FINAL Q1 2026 TEST")
print("=" * 60)
print(f"Selected model: {selected_model_name}")
print(f"Test ROC-AUC:   {test_roc_auc * 100:.1f}%")
print(f"Test accuracy:  {test_accuracy * 100:.1f}%")
print(f"Test precision: {test_precision * 100:.1f}%")
print(f"Test recall:    {test_recall * 100:.1f}%")
print(f"Test F1:        {test_f1 * 100:.1f}%")

# ============================================================
# 10. SHAP EXPLANATION FOR LOGISTIC REGRESSION
# ============================================================

# SHAP is used to explain the Logistic Regression model.
# It helps answer: "Which features influenced the delay predictions the most?"
logistic_model = fitted_models["Logistic Regression"]

# Use a small background sample so SHAP can understand what a normal
# model input looks like. This keeps the explanation step fast.
background = X_train_ready[:200]

# Use a sample of test rows to explain the model's predictions.
# These are the actual rows SHAP will calculate explanations for.
explained = X_test_ready[:SHAP_SAMPLE]

# The preprocessed feature matrix may be sparse because of one-hot encoding.
# SHAP works more easily with dense arrays, so convert if needed.
if hasattr(background, "toarray"):
    background = background.toarray()

if hasattr(explained, "toarray"):
    explained = explained.toarray()

# Clean the transformed feature names so the SHAP chart is readable.
# Example: "categorical__op_unique_carrier_DL" becomes "Op Unique Carrier Dl".
shap_feature_names = pd.Series(feature_names)
shap_feature_names = shap_feature_names.str.replace("numeric__", "", regex=False)
shap_feature_names = shap_feature_names.str.replace("categorical__", "", regex=False)
shap_feature_names = shap_feature_names.str.replace("_", " ", regex=False)
shap_feature_names = shap_feature_names.str.title()
shap_feature_names = shap_feature_names.tolist()

# Build a SHAP explainer for a linear model.
# LinearExplainer is appropriate here because Logistic Regression is linear.
explainer = shap.LinearExplainer(logistic_model, background)

# Calculate SHAP values for the explained test rows.
# Each SHAP value shows how much a feature pushed a prediction up or down.
shap_values = explainer(explained)

# Take the average absolute SHAP value for each feature.
# This ranks features by overall impact size, regardless of direction.
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

# Store the SHAP importance results in a dataframe.
shap_summary = pd.DataFrame(
    {
        "feature": shap_feature_names,
        "mean_abs_shap_value": mean_abs_shap,
    }
)

# Sort features from most influential to least influential.
shap_summary = shap_summary.sort_values(
    "mean_abs_shap_value",
    ascending=False
).reset_index(drop=True)

# Add a rank column so the output is easier to read.
shap_summary["rank"] = shap_summary.index + 1
shap_summary = shap_summary[["rank", "feature", "mean_abs_shap_value"]]

# Save the SHAP feature-importance table as a CSV file.
shap_summary_path = PLOT_DIR / "logistic_regression_shap_summary.csv"
shap_summary.to_csv(shap_summary_path, index=False)

# Save a SHAP bar chart showing the top 20 most influential features.
shap_plot = PLOT_DIR / "logistic_regression_shap_summary.png"
plt.figure()
shap.summary_plot(
    shap_values.values,
    explained,
    feature_names=shap_feature_names,
    plot_type="bar",
    show=False,
    max_display=20,
)
plt.tight_layout()
plt.savefig(shap_plot)
# Close the saved figure so the script continues without waiting
# for the user to close a plot window.
plt.close()

# Keep the top 10 features for the printed model summary.
shap_display = shap_summary.head(10).copy()
shap_display["mean_abs_shap_value"] = shap_display["mean_abs_shap_value"].round(4)

# Save the most important SHAP feature so it can be mentioned later.
top_shap_feature = shap_display.iloc[0]["feature"]


# ============================================================
# 11. BUILD REPORT
# ============================================================

# Make a clean comparison table for the text report.
comparison_display = comparison.copy()
comparison_display["validation_roc_auc"] = (comparison_display["validation_roc_auc"] * 100).round(1)
comparison_display["validation_accuracy"] = (comparison_display["validation_accuracy"] * 100).round(1)
comparison_display["validation_precision_delayed"] = (comparison_display["validation_precision_delayed"] * 100).round(1)
comparison_display["validation_recall_delayed"] = (comparison_display["validation_recall_delayed"] * 100).round(1)
comparison_display["validation_f1_delayed"] = (comparison_display["validation_f1_delayed"] * 100).round(1)

comparison_display = comparison_display.rename(
    columns={
        "model": "Model",
        "validation_roc_auc": "Validation ROC-AUC %",
        "validation_accuracy": "Validation Accuracy %",
        "validation_precision_delayed": "Validation Precision %",
        "validation_recall_delayed": "Validation Recall %",
        "validation_f1_delayed": "Validation F1 %",
    }
)

comparison_display.insert(0, "Rank", range(1, len(comparison_display) + 1))

# Build the readable text report.
output_text = f"""
============================================================
CLASSIFICATION SUMMARY
============================================================

Input file: {INPUT_PATH}
Rows used:  {len(df):,}
Target:     {TARGET}
Departure delay rate: {delay_base_rate:.1f}%

Time-based split:
Train:      through {TRAIN_END_DATE} ({len(train_df):,} rows)
Validation: {VALIDATION_START_DATE} to {VALIDATION_END_DATE} ({len(validation_df):,} rows)
Test:       {TEST_START_DATE} to {TEST_END_DATE} ({len(test_df):,} rows)

Models compared:
Logistic Regression, AdaBoost, Random Forest, Extra Trees, XGBoost

Feature groups:
Numeric: {", ".join(NUMERIC_FEATURES)}
Categorical: {", ".join(CATEGORICAL_FEATURES)}

------------------------------------------------------------
Main results
------------------------------------------------------------
Selected model:      {selected_model_name}
Validation ROC-AUC:  {selected_row['validation_roc_auc'] * 100:.1f}%
Test ROC-AUC:        {test_roc_auc * 100:.1f}%
Test accuracy:       {test_accuracy * 100:.1f}%
Test recall:         {test_recall * 100:.1f}%
Test precision:      {test_precision * 100:.1f}%
Test F1:             {test_f1 * 100:.1f}%
Top SHAP feature:    {top_shap_feature}

------------------------------------------------------------
Validation model leaderboard
------------------------------------------------------------
{comparison_display.to_string(index=False)}

The model leaderboard uses 2025 validation results only.
Q1 2026 was evaluated once using the selected model.

------------------------------------------------------------
Confusion matrix
------------------------------------------------------------
                          PREDICTED
                  On Time        Delayed       Row Total
             +-----------------------------------------------
  ACTUAL     |
  On Time    |   {tn:>9,}    {fp:>9,}     {tn + fp:>9,}
  Delayed    |   {fn:>9,}    {tp:>9,}     {fn + tp:>9,}
             +-----------------------------------------------
  Col Total  |   {tn + fn:>9,}    {fp + tp:>9,}     {total:>9,}

------------------------------------------------------------
Top SHAP features - Logistic Regression
------------------------------------------------------------
{shap_display.to_string(index=False)}

------------------------------------------------------------
Saved outputs
------------------------------------------------------------
Report: {OUTPUT_PATH}
Model comparison CSV: {COMPARISON_OUTPUT}
Plot folder: {PLOT_DIR}
SHAP values: {shap_summary_path}
"""


# ============================================================
# 12. SAVE OUTPUTS
# ============================================================

# Save the readable text report.
OUTPUT_PATH.write_text(output_text)

# Print the same report in the Python output window.
print(output_text)
print(f"Saved report to: {OUTPUT_PATH}")
print(f"Saved model comparison to: {COMPARISON_OUTPUT}")
print(f"Saved plots to: {PLOT_DIR}")
