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
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Set your GCP project and dataset information
PROJECT_ID = 'housing-bubble-predictor-data'  # GCP Project ID
DATASET_ID = 'housing_curated'  # Dataset name in BigQuery
TABLE_ID = 'table_obt_housing'  # Source table containing OBT data
OUTPUT_TABLE_ID = 'model_predictions'  # Destination table for storing predictions

FULL_TABLE_ID = f'{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}'
OUTPUT_FULL_TABLE_ID = f'{PROJECT_ID}.{DATASET_ID}.{OUTPUT_TABLE_ID}'

RUN_MODE = 'manual'  # 'manual' (overwrite) or 'monthly' (append)

# ---------------------------------------------------------
# STEP 1: LOAD OBT DATA FROM BIGQUERY
# ---------------------------------------------------------
def load_data():
    """
    Load One Big Table (OBT) from BigQuery.
    """
    client = bigquery.Client()
    query = f"SELECT * FROM `{FULL_TABLE_ID}` ORDER BY quarter_id"
    df = client.query(query).to_dataframe()
    df['date'] = pd.to_datetime(df['quarter_id'].str.replace('Q', ''), format='%Y%q')
    return df.sort_values('date')

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
    df['post_gfc_flag'] = (df['date'] >= '2009-01-01').astype(int)
    df['post_covid_flag'] = (df['date'] >= '2020-01-01').astype(int)
    return df

# ---------------------------------------------------------
# STEP 3: FILTER FEATURES USING CORRELATION + VIF
# ---------------------------------------------------------
def filter_features(df, target_col='home_price_index_value', vif_thresh=5.0, corr_thresh=0.9):
    """
    Keep only uncorrelated and low-VIF features for model stability.
    Also display correlation heatmap and VIF stats for interpretability.
    """
    candidate_cols = [
        'consumer_sentiment_value', 'post_covid_flag',
        'unemployment_rate_value', 'price_to_income',
        'permits_per_completion', 'mortgage_rate_30yr_value',
        'fed_funds_rate_value', 'post_gfc_flag', 'new_home_sales_value'
    ]

    X = df[candidate_cols].copy().dropna()

    # ----------------------
    # Display Correlation Heatmap
    # ----------------------
    print("📊 Correlation Matrix Heatmap:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Correlation Matrix of Candidate Features", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Drop highly correlated columns
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    if to_drop_corr:
        print(f"\n🧹 Dropping highly correlated features (ρ > {corr_thresh}): {to_drop_corr}\n")
    X_filtered = X.drop(columns=to_drop_corr)

    # ----------------------
    # Compute Initial VIFs
    # ----------------------
    def compute_vif(X):
        return pd.DataFrame({
            'feature': X.columns,
            'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        })

    vif_df_initial = compute_vif(X_filtered)
    print("📌 Initial VIF Values (before filtering):\n", vif_df_initial, "\n")

    # ----------------------
    # Drop High VIFs Iteratively
    # ----------------------
    while True:
        vif_df = compute_vif(X_filtered)
        if vif_df['VIF'].max() > vif_thresh:
            drop_col = vif_df.sort_values('VIF', ascending=False).iloc[0]['feature']
            print(f"⚠️ Dropping '{drop_col}' due to high VIF = {vif_df['VIF'].max():.2f}")
            X_filtered = X_filtered.drop(columns=[drop_col])
        else:
            break

    # Final VIF Output
    print("\n✅ Final VIF Values (after filtering):\n", compute_vif(X_filtered), "\n")

    selected_features = list(X_filtered.columns)
    df_final = df[['quarter_id', 'date', target_col]].join(X_filtered, how='left')
    return df_final.dropna(), selected_features

# ---------------------------------------------------------
# STEP 4: WALK-FORWARD PREDICTION
# ---------------------------------------------------------
def walk_forward(df, feature_cols):
    """
    For each quarter, train on prior data and predict next value using:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    """
    X = df[feature_cols]
    y = df['home_price_index_value']
    scaler = StandardScaler()
    window_size = int(len(df) * 0.8)

    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }

    results = []
    run_time = datetime.utcnow()  # Common timestamp for run

    for start in range(0, len(X) - window_size):
        end = start + window_size
        X_train, X_test = X.iloc[start:end], X.iloc[end:end+1]
        y_train, y_test = y.iloc[start:end], y.iloc[end:end+1]
        date_test = df.iloc[end]['quarter_id']

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)[0]

            results.append({
                'quarter_id': date_test,
                'model_type': name,
                'actual': y_test.values[0],
                'predicted': y_pred,
                'run_timestamp': run_time
            })

    return pd.DataFrame(results)

# ---------------------------------------------------------
# STEP 5: UPLOAD TO BIGQUERY
# ---------------------------------------------------------
def save_to_bq(df):
    """
    Upload predictions to BigQuery:
    - Overwrite if RUN_MODE == 'manual'
    - Append if RUN_MODE == 'monthly'
    """
    client = bigquery.Client()
    disposition = 'WRITE_TRUNCATE' if RUN_MODE == 'manual' else 'WRITE_APPEND'

    job_config = bigquery.LoadJobConfig(write_disposition=disposition)
    client.load_table_from_dataframe(df, OUTPUT_FULL_TABLE_ID, job_config=job_config).result()
    print(f"✅ Uploaded predictions to `{OUTPUT_FULL_TABLE_ID}` as {RUN_MODE.upper()} run")

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

    print("📤 Step 5: Uploading results to BigQuery...")
    save_to_bq(pred_df)

    print("\n🎉 Forecast Pipeline Finished!")

# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------
if __name__ == '__main__':
    main()