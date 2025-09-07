"""
Download raw monthly time series from FRED API and save as CSV files.

Each file will contain:
  - `period_start_date` column in 'YYYY-MM-DD' format
  - A renamed metric column (e.g., total_starts instead of 'value')
Output CSVs are saved in the `data/raw/` directory.

Includes:
✅ Retry logic with exponential backoff (up to 3 retries)
✅ Delay between requests to avoid rate limiting
✅ Error handling for malformed ZIP responses

Author: Kapil Tare
"""

import os
import time
import requests
import pandas as pd
from io import BytesIO
import zipfile

# 📁 Directory to save raw CSVs
OUTPUT_PATH = "data/raw/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 🔐 Load FRED API key from environment variable
api_key = os.getenv("FRED_API_KEY")
if not api_key:
    raise ValueError("❌ FRED_API_KEY is not set. Use: export FRED_API_KEY=your_api_key")

# 🔁 Mapping of FRED series IDs to renamed column names
series_mapping = {
    "HOUST": "total_starts",                          # Total Privately-Owned Housing Starts
    "HOUSTS": "single_family_starts",                 # Single-Family Housing Starts
    "HOUST2F": "two_to_four_unit_starts",             # 2–4 Unit Multifamily Housing Starts
    "HOUST5F": "five_or_more_unit_starts",             # 5+ Unit Multifamily Housing Starts
    "PERMIT": "total_permits",                        # Total Privately-Owned Housing Permits
    "PERMIT1": "single_family_permits",               # Single-Family Housing Permits
    "PERMIT5": "multi_family_permits",                # 5+ Unit Multifamily Housing Permits
    "PERMIT24": "two_to_four_unit_permits",           # 2–4 Unit Multifamily Housing Permits
    "COMPUTSA": "total_completions",                  # Total Privately-Owned Housing Completions
    "COMPU1USA": "single_family_completions",         # Single-Family Housing Completions
    "COMPU5MUSA": "multi_family_completions",         # 5+ Unit Multifamily Housing Completions
    "COMPU24USA": "two_to_four_unit_completions"      # 2–4 Unit Multifamily Housing Completions
}

# ⏳ Retry settings
MAX_RETRIES = 3
DELAY_BETWEEN_CALLS = 1.5  # seconds

# 🔁 Function to download and save a FRED series
def download_fred_series(series_id: str, renamed_column: str):
    print(f"\n📦 Downloading FRED series: {series_id} → {renamed_column}")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=csv"
    print(f"🔗 API URL: {url}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 🛰 Make the GET request
            response = requests.get(url)
            response.raise_for_status()
            print("✅ API call successful.")

            # 📦 Extract ZIP contents
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                csv_filenames = z.namelist()
                print(f"📂 ZIP Contents: {csv_filenames}")

                # ✅ We expect a file named exactly "observations.csv"
                csv_name = next((name for name in csv_filenames if name.lower() == "obs._by_real-time_period.csv"), None)
                if csv_name is None:
                    raise ValueError(f"❌ 'observations.csv' not found in ZIP for {series_id}. Received: {csv_filenames}")

                # 🧾 Read into DataFrame
                with z.open(csv_name) as csv_file:
                    df = pd.read_csv(csv_file)

                # 🔧 Rename columns for standardization
                df = df.rename(columns={
                    "date": "period_start_date",
                    "value": renamed_column
                })

                # 💾 Save to CSV
                output_path = f"{OUTPUT_PATH}/{series_id}.csv"
                df.to_csv(output_path, index=False)
                print(f"✅ Saved: {output_path}")

                # 📅 Print time coverage
                print(f"🕒 Time Range: {df['period_start_date'].min()} → {df['period_start_date'].max()}")
                break  # ✅ Success — break out of retry loop

        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                print(f"❌ Giving up after {MAX_RETRIES} attempts for series {series_id}")
            else:
                wait_time = 2 ** attempt  # exponential backoff
                print(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

# 🚀 Loop through all FRED series
for sid, col_name in series_mapping.items():
    download_fred_series(sid, col_name)
    time.sleep(DELAY_BETWEEN_CALLS)  # ⏸ Delay between API calls