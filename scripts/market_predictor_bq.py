"""
GCP-Ready: Housing Market Prediction with Walk-Forward Validation (Nonlinear Models)
Author: Kapil T. (2025)
"""

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
from datetime import datetime
import pandas as pd
import numpy as np
import os
import warnings

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from google.cloud import bigquery
from google.oauth2 import service_account

import shap
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
GCP_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_creds.json")
credentials = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS_PATH)

PROJECT_ID = 'housing-bubble-predictor'
DATASET_ID = 'housing_curated'
TABLE_ID = 'table_obt_housing'
OUTPUT_TABLE_ID = 'model_predictions'
METRICS_TABLE_ID = 'model_metrics'

FULL_TABLE_ID = f'{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}'
OUTPUT_FULL_TABLE_ID = f'{PROJECT_ID}.{DATASET_ID}.{OUTPUT_TABLE_ID}'
METRICS_FULL_TABLE_ID = f'{PROJECT_ID}.{DATASET_ID}.{METRICS_TABLE_ID}'

RUN_MODE = 'manual'  # 'manual' or 'monthly'

# ---------------------------------------------------------
# STEP 1: LOAD OBT DATA
# ---------------------------------------------------------
def load_data():
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    query = f"SELECT * FROM `{FULL_TABLE_ID}` ORDER BY quarter_id"
    df = client.query(query).to_dataframe()
    df['quarter_start_date'] = pd.PeriodIndex(df['quarter_id'], freq='Q').to_timestamp()
    return df.sort_values('quarter_start_date')

# ---------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# ---------------------------------------------------------
def feature_engineering(df):
    df['price_to_income'] = df['home_price_index_value'] / df['real_income_value']
    df['permits_per_completion'] = df['permits_issued_value'] / df['housing_completions_value']
    df['unemp_x_mortgage'] = df['unemployment_rate_value'] * df['mortgage_rate_30yr_value']
    df['income_x_sentiment'] = df['real_income_value'] * df['consumer_sentiment_value']
    df['post_gfc_flag'] = (df['quarter_start_date'] >= '2009-01-01').astype(int)
    df['post_covid_flag'] = (df['quarter_start_date'] >= '2020-01-01').astype(int)
    df['mortgage_rate_30yr_lag_1'] = df['mortgage_rate_30yr_value'].shift(1)
    df['unemployment_rate_lag_1'] = df['unemployment_rate_value'].shift(1)
    df['real_income_lag_1'] = df['real_income_value'].shift(1)
    df['rate_hike_cycle_flag'] = ((df['quarter_start_date'] >= '2022-01-01') & (df['quarter_start_date'] <= '2023-06-30')).astype(int)
    df['inflation_surge_flag'] = ((df['quarter_start_date'] >= '2021-04-01') & (df['quarter_start_date'] <= '2022-12-31')).astype(int)
    return df

# ---------------------------------------------------------
# STEP 3: FEATURE SELECTION (RFE)
# ---------------------------------------------------------
def select_features(df):
    df = df.dropna()
    df['log_hpi'] = np.log(df['home_price_index_value'])
    drop_cols = ['quarter_id', 'home_price_index_value', 'quarter_start_date']
    X = df.drop(columns=drop_cols + ['log_hpi'])
    y = df['log_hpi']
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=18)
    X_selected = rfe.fit_transform(X, y)
    selected_cols = X.columns[rfe.support_]
    X_selected_df = pd.DataFrame(X_selected, columns=selected_cols, index=df.index)
    return df, X_selected_df, y

# ---------------------------------------------------------
# STEP 4: WALK-FORWARD PREDICTION + SHAP
# ---------------------------------------------------------
def walk_forward(X, y, df_dates):
    models = {
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(verbosity=0, random_state=42)
    }
    grids = {
        'DecisionTree': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5]},
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]},
        'XGBoost': {'n_estimators': [100], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1]}
    }

    window_size = int(len(X) * 0.25)
    test_size = 1
    run_time = datetime.now()
    all_results = []
    shap_done = False

    for model_name, model in models.items():
        rmse_list, smape_list, adj_r2_list, dates, preds, trues = [], [], [], [], [], []

        for i in range(window_size, len(X)):
            X_train, y_train = X.iloc[:i], y.iloc[:i]
            X_test, y_test = X.iloc[i:i+test_size], y.iloc[i:i+test_size]

            grid = GridSearchCV(model, grids[model_name], cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            y_pred_log = best_model.predict(X_test)

            y_pred = np.exp(y_pred_log)
            y_actual = np.exp(y_test.values)

            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            smape = 100 * np.mean(2 * np.abs(y_actual - y_pred) / (np.abs(y_actual) + np.abs(y_pred)))
            r2 = r2_score(y_actual, y_pred)
            # adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1) if len(y_test) > 1 else np.nan

            all_results.append({
                'quarter_id': df_dates.iloc[i]['quarter_id'],
                'quarter_start_date': df_dates.iloc[i]['quarter_start_date'],
                'model_type': model_name,
                'actual': y_actual[0],
                'predicted': y_pred[0],
                'RMSE': rmse,
                'SMAPE': smape,
                # 'Adjusted_R2': adj_r2,
                'run_timestamp': run_time
            })

            # --------------------------------------------
            # SHAP: Explain XGBoost model (once only)
            # --------------------------------------------
            if model_name == 'XGBoost' and RUN_MODE == 'manual' and not shap_done:
                explainer = shap.Explainer(best_model, X_train)
                shap_values = explainer(X_train)
                plt.figure(figsize=(12, 6))
                shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
                plt.title("SHAP Feature Importance - XGBoost")
                plt.tight_layout()
                plt.savefig("shap_summary_xgb.png", dpi=300)
                plt.close()
                print("📊 SHAP plot saved as shap_summary_xgb.png")
                shap_done = True

    return pd.DataFrame(all_results)

# ---------------------------------------------------------
# STEP 5: SPLIT METRICS AND PREDICTIONS
# ---------------------------------------------------------
def split_outputs(results_df):
    pred_df = results_df[['quarter_id', 'quarter_start_date', 'model_type', 'actual', 'predicted', 'run_timestamp']].copy()
    metrics_df = results_df.groupby('model_type').agg({
        'RMSE': 'mean',
        'SMAPE': 'mean',
        # 'Adjusted_R2': 'mean'
    }).reset_index()
    metrics_df['run_timestamp'] = results_df['run_timestamp'].iloc[0]
    return pred_df, metrics_df

# ---------------------------------------------------------
# STEP 6: UPLOAD TO BIGQUERY
# ---------------------------------------------------------
def save_to_bq(df, table_id):
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    disposition = 'WRITE_TRUNCATE' if RUN_MODE == 'manual' else 'WRITE_APPEND'
    job_config = bigquery.LoadJobConfig(write_disposition=disposition)
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
    print(f"✅ Uploaded to `{table_id}` as {RUN_MODE.upper()} run")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("🔄 Starting Nonlinear Model Forecast Pipeline...")
    df = load_data()
    df = feature_engineering(df)
    df_clean, X_selected_df, y = select_features(df)
    print(f"✅ Using {X_selected_df.shape[1]} features after RFE")

    results_df = walk_forward(X_selected_df, y, df_clean[['quarter_id', 'quarter_start_date']])
    pred_df, metrics_df = split_outputs(results_df)

    save_to_bq(pred_df, OUTPUT_FULL_TABLE_ID)
    save_to_bq(metrics_df, METRICS_FULL_TABLE_ID)
    print("🎯 Forecasting complete. Results saved.")

if __name__ == '__main__':
    main()