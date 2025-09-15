# ------------------------------------------------------------
# 📦 Load Cleaned GCS Data → BigQuery Staging Tables (GCP)
# ------------------------------------------------------------

from google.cloud import bigquery
import os

# 🔐 STEP 1: Set Google credentials
# This tells the BigQuery client where to find the service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_creds.json"

# ------------------------------------------------------------
# 🔧 STEP 2: Set project-specific constants
# ------------------------------------------------------------

# 🔢 GCP project ID (replace with your actual project ID if different)
PROJECT_ID = "housing-buuble-predictor"  # ❗️REQUIRED

# 🏷️ BigQuery dataset where all staging tables will reside
DATASET_ID = "housing_staging"

# ☁️ GCS bucket and folder where cleaned CSVs are stored
BUCKET_NAME = "housing-bubble-predictor-data"
GCS_FOLDER = "cleaned_data"

# ------------------------------------------------------------
# 📄 STEP 3: Define mapping of file names to BigQuery table names
# ------------------------------------------------------------
# 🔁 For each CSV file, specify which BigQuery staging table it belongs to

TABLE_MAPPING = {
    "COMPLETUSA.csv": "stg_completusa",
    "PERMITUSA.csv": "stg_permitusa",
    "HOUST.csv": "stg_houst",
    "HOUSTNE.csv": "stg_houstne",
    "HOUSTMW.csv": "stg_houstmw",
    "HOUSTS.csv": "stg_housts",
    "HOUSTW.csv": "stg_houstw",
    "MORTGAGE30US.csv": "stg_mortgage30us",
    "UNRATE.csv": "stg_unrate",
    "CPIAUCSL.csv": "stg_cpiaucsl",
    "FEDFUNDS.csv": "stg_fedfunds",
    "CSUSHPISA.csv": "stg_csushpisa",
    "DSPIC96.csv": "stg_dspic96",
    "UMCSENT.csv": "stg_umcsent",
    "HOAFFORD.csv": "stg_hoafford",
    "HSN1FNSA.csv": "stg_hsn1fnsa",
    "PCE.csv": "stg_pce",
    "COMPUTPUSA.csv": "stg_computpusa"
}

# ------------------------------------------------------------
# 🧠 STEP 4: Initialize BigQuery client
# ------------------------------------------------------------

# This client allows us to perform BQ operations using Python
client = bigquery.Client(project=PROJECT_ID)

# ------------------------------------------------------------
# 🚀 STEP 5: Loop through each CSV → Load into corresponding table
# ------------------------------------------------------------

for filename, table_name in TABLE_MAPPING.items():
    # 👇 GCS path to current cleaned file
    uri = f"gs://{BUCKET_NAME}/{GCS_FOLDER}/{filename}"

    # 👇 Full target BQ table name in format: project.dataset.table
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    # 🖨️ Log the file being processed
    print(f"🔁 Loading {filename} into {table_id} from {uri}...")

    # ------------------------------------------------------------
    # 🧾 Define BigQuery load job configuration
    # ------------------------------------------------------------

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,     # 📂 All files are in CSV format
        skip_leading_rows=1,                         # 🧹 Skip the header row
        autodetect=True,                             # 🧠 Let BQ detect column types from content
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE  # 🧼 Overwrite existing table
    )

    # ------------------------------------------------------------
    # ⏳ Trigger the load job from GCS → BigQuery
    # ------------------------------------------------------------

    job = client.load_table_from_uri(
        uri,             # 📍 Source CSV file in GCS
        table_id,        # 🏁 Target BigQuery table
        job_config=job_config
    )

    # ✅ Wait for job to complete before proceeding
    job.result()

    # 🖨️ Confirm successful load
    print(f"✅ Loaded {table_id}")