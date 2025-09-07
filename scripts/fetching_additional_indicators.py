"""
Download key macroeconomic indicators from FRED API and save as individual CSV files.

Each CSV will contain:
  - `period_start_date` in YYYY-MM-DD format
  - A single metric column with a clean, renamed label

✅ Includes retry logic with exponential backoff
✅ Handles FRED API errors and failed requests
✅ Designed for monthly indicators
✅ Compatible with GCS upload pipeline and BigQuery schema

Author: Kapil Tare
"""

import os
import time
import requests
import pandas as pd

# 📁 Where the raw CSVs will be saved
OUTPUT_PATH = "data/raw/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 🔐 Load FRED API key securely from env variable
api_key = os.getenv("FRED_API_KEY")
if not api_key:
    raise ValueError("❌ FRED_API_KEY not found. Please set it using: export FRED_API_KEY=your_key")

# 📊 Economic indicators to fetch: {FRED_ID: Clean Column Name}
series_mapping = {
    "MORTGAGE30US": "mortgage_30y",                # 30-Year Fixed Mortgage Rate
    "UNRATE": "unemployment_rate",                 # Civilian Unemployment Rate
    "CPIAUCSL": "cpi_all_urban",                   # CPI - All Urban Consumers
    "FEDFUNDS": "federal_funds_rate",              # Federal Funds Effective Rate
    "CSUSHPISA": "home_price_index",               # Case-Shiller Home Price Index
    "DSPIC96": "real_disposable_income",           # Disposable Personal Income (Real)
    "UMCSENT": "consumer_sentiment",               # University of Michigan Sentiment Index
    #"HOAFFORD": "housing_affordability",           # Housing Affordability Index
    "HSN1FNSA": "new_home_sales",                  # New One-Family Homes Sold
    "PCE": "pce"                                    # Personal Consumption Expenditures
}

# 🔁 Retry settings
MAX_RETRIES = 3
DELAY_BETWEEN_CALLS = 1.5  # seconds

# 📦 FRED API download function
def download_fred_series(series_id: str, renamed_column: str):
    print(f"\n⬇️ Downloading {series_id} → {renamed_column}")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 🌐 GET request to FRED API
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # 📊 Convert to DataFrame
            observations = data.get("observations", [])
            df = pd.DataFrame(observations)[["date", "value"]]
            df = df.rename(columns={"date": "period_start_date", "value": renamed_column})
            df["period_start_date"] = pd.to_datetime(df["period_start_date"])
            df[renamed_column] = pd.to_numeric(df[renamed_column], errors="coerce")

            # 💾 Save to CSV
            out_path = os.path.join(OUTPUT_PATH, f"{series_id}.csv")
            df.to_csv(out_path, index=False)
            print(f"✅ Saved: {out_path}")
            print(f"🕒 Time Range: {df['period_start_date'].min().date()} → {df['period_start_date'].max().date()}")
            break  # Exit retry loop on success

        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed for {series_id}: {e}")
            if attempt == MAX_RETRIES:
                print(f"❌ Failed after {MAX_RETRIES} attempts.")
            else:
                wait_time = 2 ** attempt
                print(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

# 🚀 Loop through each economic indicator
for sid, col_name in series_mapping.items():
    download_fred_series(sid, col_name)
    time.sleep(DELAY_BETWEEN_CALLS)  # avoid rate limiting