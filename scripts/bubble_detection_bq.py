"""
bubble_detection_bq.py

This script reads the curated quarterly One Big Table (OBT) from BigQuery, applies rule-based logic to detect potential housing market bubbles, and exports the final bubble signal and intermediate flags back to BigQuery and to a local CSV (optional for dev/debug).

Author: Kapil Tare
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
import os
from google.oauth2 import service_account

# -----------------------------------------
# 1. Authenticate with GCP
# -----------------------------------------
# Make sure to set GOOGLE_APPLICATION_CREDENTIALS env variable before running
# === STEP 1: Load GCP credentials ===
GCP_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_creds.json")
credentials = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS_PATH)

# === STEP 2: Set GCS bucket and BigQuery project/dataset info ===
# bucket_name = "housing-bubble-predictor-data"
# gcs_folder = "cleaned_data"
project_id = "housing-bubble-predictor"
dataset_id = "housing_curated"

# === STEP 3: Create BigQuery client with credentials ===
client = bigquery.Client(credentials=credentials, project=project_id)
client = bigquery.Client()

# -----------------------------------------
# 2. Load Quarterly Housing Data (OBT Table)
# -----------------------------------------
OBT_TABLE = "housing_curated.table_obt_housing"
query = f"""
    SELECT
        quarter_id,
        home_price_index_value,
        mortgage_rate_30yr_value,
        real_income_value
    FROM `{OBT_TABLE}`
    WHERE home_price_index_value IS NOT NULL
      AND mortgage_rate_30yr_value IS NOT NULL
      AND real_income_value IS NOT NULL
    ORDER BY quarter_id
"""
df = client.query(query).to_dataframe()

# -----------------------------------------
# 3. Feature Engineering (Affordability, Growth, Z-Score, etc.)
# -----------------------------------------

# 3.1: Housing Affordability Index (Proxy)
# Real income divided by home price * mortgage rate
# ↓ Lower values = lower affordability
df["HAI_proxy"] = df["real_income_value"] / (df["home_price_index_value"] * df["mortgage_rate_30yr_value"])

# 3.2: Price Growth over 4 quarters
# Used instead of single-quarter jumps to detect sustained overheating
df["price_growth_4q"] = df["home_price_index_value"].pct_change(4)

# 3.3: Growth Acceleration
# Change in growth rate itself (second derivative), smoothed to reduce noise
df["growth_accel"] = df["price_growth_4q"].diff().rolling(2).mean()

# 3.4: Z-Score Deviation from 5-Year Mean (20 quarters)
# Measures how irrationally far prices have deviated from historical average
df["price_zscore"] = (
    (df["home_price_index_value"] - df["home_price_index_value"].rolling(20).mean()) /
    df["home_price_index_value"].rolling(20).std()
)

# 3.5: Short-Term Momentum
# Positive recent quarterly changes over 3 quarters indicate herd behavior
df["momentum_positive"] = df["home_price_index_value"].pct_change(1).rolling(3).mean() > 0

# 3.6: Inverse Correlation Between Price & Rates
# Normally, prices should drop when mortgage rates rise. If this breaks down, it may be speculative.
df["price_rate_corr"] = df["home_price_index_value"].rolling(4).corr(df["mortgage_rate_30yr_value"])

# 3.7: Affordability Crash
# Affordability index dropping sharply signals demand-side collapse risk
df["hai_qoq_change"] = df["HAI_proxy"].pct_change()

# -----------------------------------------
# 4. Scoring Engine (Hybrid Bubble Risk Model)
# -----------------------------------------

scores = []
for i in range(20, len(df)):
    row = df.iloc[i]
    score = 0
    notes = []

    g = row["price_growth_4q"]
    accel = row["growth_accel"]
    z = abs(row["price_zscore"])
    m = row["momentum_positive"]
    c = row["price_rate_corr"]
    h = row["hai_qoq_change"]

    # -----------------------------
    # 📈 Price Growth (Trailing 4Q)
    # -----------------------------
    if g > 0.25:
        score += 30
        notes.append("Growth > 25%")
    elif g > 0.20:
        score += 25
        notes.append("Growth > 20%")
    elif g > 0.15:
        score += 20
        notes.append("Growth > 15%")
    elif g > 0.10:
        score += 10
        notes.append("Growth > 10%")
    elif g > 0.05:
        score += 5
        notes.append("Growth > 5%")

    # ----------------------------------
    # ⏩ Growth Acceleration
    # ----------------------------------
    if accel > 0.03:
        score += 5
        notes.append("Acceleration > 3%")

    # -------------------------------
    # 📊 Z-Score Deviation
    # -------------------------------
    if z > 3:
        score += 25
        notes.append("Z-Score > 3")
    elif z > 2:
        score += 15
        notes.append("Z-Score > 2")
    elif z > 1.5:
        score += 10
        notes.append("Z-Score > 1.5")
    elif z > 1:
        score += 5
        notes.append("Z-Score > 1")

    # -------------------------------
    # 📉 Affordability Crash
    # -------------------------------
    if h < -0.05:
        score += 10
        notes.append("Affordability ↓ > 5% QoQ")

    # -------------------------------
    # 🔁 Positive Momentum
    # -------------------------------
    if m:
        score += 15
        notes.append("Momentum Positive (3Q trend)")

    # -------------------------------
    # 🔀 Rate-Price Inverse Breakdown
    # -------------------------------
    if not np.isnan(c):
        if abs(c) > 0.9:
            score += 20
            notes.append("Price-Rate Corr > 0.9")
        elif abs(c) > 0.8:
            score += 15
            notes.append("Price-Rate Corr > 0.8")
        elif abs(c) > 0.6:
            score += 10
            notes.append("Price-Rate Corr > 0.6")

    # --------------------------------
    # 🧮 Compound Rule
    # --------------------------------
    if g > 0.10 and z > 1.5:
        score += 10
        notes.append("Growth > 10% AND Z > 1.5")

    # -----------------------------------------
    # 🧾 Store Final Scores + Flags
    # -----------------------------------------
    scores.append({
        "quarter_id": row["quarter_id"],
        "home_price_index_value": row["home_price_index_value"],
        "mortgage_rate_30yr_value": row["mortgage_rate_30yr_value"],
        "real_income_value": row["real_income_value"],
        "HAI_proxy": row["HAI_proxy"],
        "price_growth_4q": g,
        "price_zscore": row["price_zscore"],
        "growth_accel": accel,
        "momentum_positive": m,
        "price_rate_corr": c,
        "hai_qoq_change": h,
        "risk_score": score,
        "risk_level": "High" if score >= 60 else "Medium" if score >= 40 else "Low",
        "realistic_bubble_flag": 1 if score >= 40 else 0,
        "notes": "; ".join(notes)
    })

bubble_df = pd.DataFrame(scores)

# -----------------------------------------
# 5. Export Final Table to BigQuery
# -----------------------------------------
TABLE_ID = "housing_curated.hybrid_bubble_flags"
job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")

job = client.load_table_from_dataframe(bubble_df, TABLE_ID, job_config=job_config)
job.result()

print(f"✅ Hybrid Bubble Flags exported to: {TABLE_ID}")