"""
GCP-Ready: Housing Market Prediction with Walk-Forward Validation
Author: Kapil T. (2025)
"""

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from google.cloud import bigquery
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from google.oauth2 import service_account
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Set your GCP project and dataset information
GCP_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_creds.json")
credentials = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS_PATH)

PROJECT_ID = 'housing-bubble-predictor'  # GCP Project ID
DATASET_ID = 'housing_curated'  # Dataset name in BigQuery
TABLE_ID = 'table_obt_housing'  # Source table containing OBT data
OUTPUT_TABLE_ID = 'model_predictions'  # Destination table for storing predictions
METRICS_TABLE_ID = 'model_metrics' # Destination table for storing metrics

FULL_TABLE_ID = f'{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}'
OUTPUT_FULL_TABLE_ID = f'{PROJECT_ID}.{DATASET_ID}.{OUTPUT_TABLE_ID}'
METRICS_FULL_TABLE_ID = f'{PROJECT_ID}.{DATASET_ID}.{METRICS_TABLE_ID}'

RUN_MODE = 'manual'  # 'manual' (overwrite) or 'monthly' (append)

# ---------------------------------------------------------
# STEP 1: LOAD OBT DATA FROM BIGQUERY
# ---------------------------------------------------------
def load_data():
    """
    Load One Big Table (OBT) from BigQuery and create a quarter_start_date column
    for modeling, plus a formatted quarter_dash string column for labeling.
    """
    
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    query = f"SELECT * FROM `{FULL_TABLE_ID}` ORDER BY quarter_id"
    df = client.query(query).to_dataframe()

    # Convert '2023Q4' → '2023-10-01' for modeling and '2023-Q4' for labeling
    df['quarter_start_date'] = pd.to_datetime(
    df['quarter_id'].str[:4] + '-' + ((df['quarter_id'].str[-1].astype(int) - 1) * 3 + 1).astype(str).str.zfill(2) + '-01'
)
    df['quarter_start_date'] = pd.PeriodIndex(df['quarter_id'], freq='Q').to_timestamp()
    df['quarter_dash'] = df['quarter_id'].str[:4] + '-' + df['quarter_id'].str[-2:]
    return df.sort_values('quarter_start_date')

# ---------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# ---------------------------------------------------------

def feature_engineering(df):
    """
    Create macroeconomic ratios and regime flags.
    These features improve realism and signal diversity.
    """
    df['price_to_income'] = df['home_price_index_value'] / df['real_income_value']
    df['permits_per_completion'] = df['permits_issued_value'] / df['housing_completions_value']
    df['post_gfc_flag'] = (df['quarter_start_date'] >= '2009-01-01').astype(int)
    df['post_covid_flag'] = (df['quarter_start_date'] >= '2020-01-01').astype(int)
    df['unemp_x_mortgage'] = df['unemployment_rate_value'] * df['mortgage_rate_30yr_value']
    df['income_x_sentiment'] = df['real_income_value'] * df['consumer_sentiment_value']
    return df

...
# ---------------------------------------------------------
# STEP 3: FILTER FEATURES USING CORRELATION + VIF
# ---------------------------------------------------------
def filter_features(df, target_col='home_price_index_value', vif_thresh=5.0, corr_thresh=0.9):
    """
    Feature filtering based on target correlation, inter-feature correlation, and multicollinearity.

    Phase 1: Show full correlation heatmap including the target for interpretability
    Phase 2: Drop inter-correlated features (excluding the target)
    Phase 3: Drop features with high VIF
    """


    # ---------------------------
    # CANDIDATE FEATURE SELECTION
    # ---------------------------
    candidate_cols = [
        'consumer_sentiment_value', 'post_covid_flag',
        'unemployment_rate_value', 'price_to_income',
        'permits_per_completion', 'mortgage_rate_30yr_value',
        'fed_funds_rate_value', 'post_gfc_flag', 'unemp_x_mortgage',
        'income_x_sentiment'
    ]

    # ---------------------------
    # CORRELATION MATRIX WITH TARGET
    # ---------------------------
    corr_target_matrix = df[[target_col] + candidate_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_target_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation with Target')
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # CORRELATION AMONG FEATURES
    # ---------------------------
    X = df[candidate_cols].copy().dropna()
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    if to_drop_corr:
        print(f"\n⚠️ Dropping due to correlation > {corr_thresh}: {to_drop_corr}")
    X_filtered = X.drop(columns=to_drop_corr)

    # ---------------------------
    # VIF COMPUTATION AND FILTERING
    # ---------------------------
    def compute_vif(X):
        return pd.DataFrame({
            'feature': X.columns,
            'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        })

    print("\n🔎 Initial VIF Check:")
    print(compute_vif(X_filtered))

    while True:
        vif_df = compute_vif(X_filtered)
        if vif_df['VIF'].max() > vif_thresh:
            drop_col = vif_df.sort_values('VIF', ascending=False).iloc[0]['feature']
            print(f"⚠️ Dropping {drop_col} due to high VIF")
            X_filtered = X_filtered.drop(columns=[drop_col])
        else:
            break

    print("\n✅ Final VIF:")
    print(compute_vif(X_filtered))

    selected_features = list(X_filtered.columns)
    df_final = df[['quarter_id', 'quarter_start_date', 'quarter_dash', target_col]].join(X_filtered, how='left')
    return df_final.dropna(), selected_features


# ---------------------------------------------------------
# STEP 4: WALK-FORWARD PREDICTION
# ---------------------------------------------------------
def walk_forward(df, feature_cols):
    """
    Perform walk-forward prediction using 3 regression models:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression

    For each quarter, the model trains on all prior data and predicts the next.

    Key Notes:
    • Time identifiers (quarter_id, quarter_dash, quarter_start_date) are dropped from features.
    • Results include actual vs predicted values, model type, and timestamps.

    Returns:
        DataFrame with prediction results for each quarter.
    """

    # Extract target variable
    y = df['home_price_index_value']

    # Construct feature matrix from selected columns
    X = df[feature_cols].copy()

    # 🚫 Remove temporal leakage columns if accidentally included
    for col in ['quarter_id', 'quarter_dash', 'quarter_start_date']:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Initialize scaler and model set
    scaler = StandardScaler()
    window_size = int(len(df) * 0.8)  # 80% training window
    run_time = datetime.now()     # Timestamp for this run

    # Define models to test
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }

    results = []

    # Rolling window: train on prior data, predict next quarter
    for start in range(0, len(X) - window_size):
        end = start + window_size

        # Split train/test
        X_train, X_test = X.iloc[start:end], X.iloc[end:end+1]
        y_train, y_test = y.iloc[start:end], y.iloc[end:end+1]

        # Metadata for the prediction quarter
        qid_test = df.iloc[end]['quarter_id']
        qdash_test = df.iloc[end]['quarter_dash']
        qdate_test = df.iloc[end]['quarter_start_date']

        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit and predict with each model
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)[0]

            # Append result row
            results.append({
                'quarter_id': qid_test,
                'quarter_start_date': qdate_test,
                'quarter_dash': qdash_test,
                'model_type': name,
                'actual': y_test.values[0],
                'predicted': y_pred,
                'run_timestamp': run_time
            })

    return pd.DataFrame(results)

# ---------------------------------------------------------
# STEP 5: COMPUTE EVALUATION METRICS
# ---------------------------------------------------------
def compute_metrics(pred_df):
    metrics = []
    run_time = datetime.now()
    for model in pred_df['model_type'].unique():
        df_model = pred_df[pred_df['model_type'] == model].copy()
        y_true = df_model['actual'].values
        y_pred = df_model['predicted'].values

        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - df_model.shape[1] - 1)
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

        metrics.append({
            'model_type': model,
            'RMSE': rmse,
            'Adjusted_R2': adj_r2,
            'SMAPE': smape,
            'run_timestamp': run_time
        })

    return pd.DataFrame(metrics)

# ---------------------------------------------------------
# STEP 6: UPLOAD TO BIGQUERY
# ---------------------------------------------------------

def save_to_bq(df, table_id):
    """
    Upload predictions to BigQuery:
    - Overwrite if RUN_MODE == 'manual'
    - Append if RUN_MODE == 'monthly'
    """
    client = bigquery.Client()
    disposition = 'WRITE_TRUNCATE' if RUN_MODE == 'manual' else 'WRITE_APPEND'
    job_config = bigquery.LoadJobConfig(write_disposition=disposition)
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
    print(f"✅ Uploaded to `{table_id}` as {RUN_MODE.upper()} run")

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    print("🔄 Starting Housing Market Forecast Pipeline...\n")
    
    print("📥 Step 1: Loading data from BigQuery...")
    df = load_data()

    print("🔧 Step 2: Running feature engineering...")
    df = feature_engineering(df)

    print("🔎 Step 3: Filtering features (VIF + correlation)...")
    df_clean, features = filter_features(df)
    print(f"✅ Selected Features: {features}\n")

    print("📈 Step 4: Performing walk-forward prediction...")
    pred_df = walk_forward(df_clean, features)

    print("📊 Step 5: Computing evaluation metrics...")
    metrics_df = compute_metrics(pred_df)

    print("📤 Step 6: Uploading predictions and metrics to BigQuery...")
    save_to_bq(pred_df, OUTPUT_FULL_TABLE_ID)
    save_to_bq(metrics_df, METRICS_FULL_TABLE_ID)

    print("\n🎉 Forecast Pipeline Finished!")

# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------
if __name__ == '__main__':
    main()