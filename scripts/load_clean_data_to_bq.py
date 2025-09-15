# scripts/load_cleaned_data_to_bq.py

import os
from google.cloud import bigquery
from google.oauth2 import service_account

# ----------------------------------------
# 🔐 STEP 1: Authenticate using service account
# ----------------------------------------
GCP_CREDENTIALS_PATH = "gcp_creds.json"
credentials = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS_PATH)

# ----------------------------------------
# 🧠 STEP 2: Initialize BigQuery client
# ----------------------------------------
bq_client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# ----------------------------------------
# 📂 STEP 3: Define constants
# ----------------------------------------
CLEAN_DATA_DIR = "data/clean"
BQ_DATASET_NAME = "raw_indicators"  # Change if needed
PROJECT_ID = credentials.project_id  # Dynamically pulled from creds

# ----------------------------------------
# 🔄 STEP 4: Loop through all cleaned CSVs and upload to BigQuery
# ----------------------------------------
for filename in os.listdir(CLEAN_DATA_DIR):
    if filename.endswith(".csv"):
        file_path = os.path.join(CLEAN_DATA_DIR, filename)
        table_name = filename.replace(".csv", "")  # Use filename as table name

        table_id = f"{PROJECT_ID}.{BQ_DATASET_NAME}.{table_name}"

        print(f"📤 Uploading {file_path} to BigQuery table {table_id}...")

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )

        with open(file_path, "rb") as f:
            load_job = bq_client.load_table_from_file(f, table_id, job_config=job_config)

        load_job.result()  # ⏳ Waits for the job to complete
        print(f"✅ Successfully loaded {filename} into {table_id}")

# ----------------------------------------
# 🧾 STEP 5: Optional Summary
# ----------------------------------------
print("🎉 All cleaned files successfully uploaded to BigQuery.")