import os
from google.cloud import bigquery
from google.oauth2 import service_account

# === STEP 1: Load GCP credentials ===
GCP_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_creds.json")
credentials = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS_PATH)

# === STEP 2: Set GCS bucket and BigQuery project/dataset info ===
bucket_name = "housing-bubble-predictor-data"
gcs_folder = "cleaned_data"
project_id = "housing-bubble-predictor"
dataset_id = "housing_staging"

# === STEP 3: Create BigQuery client with credentials ===
client = bigquery.Client(credentials=credentials, project=project_id)

# === STEP 4: Mapping of GCS files to BigQuery staging tables ===
file_to_table_map = {
    "COMPU1USA.csv": "stg_COMPU1USA",
    "COMPU5MUSA.csv": "stg_COMPU5MUSA",
    "COMPU24USA.csv": "stg_COMPU24USA",
    "COMPUTSA.csv": "stg_COMPUTSA",
    "CPIAUCSL.csv": "stg_CPIAUCSL",
    "CSUSHPISA.csv": "stg_CSUSHPISA",
    "DSPIC96.csv": "stg_DSPIC96",
    "FEDFUNDS.csv": "stg_FEDFUNDS",
    "HOUST.csv": "stg_HOUST",
    "HOUST2F.csv": "stg_HOUST2F",
    "HOUST5F.csv": "stg_HOUST5F",
    "HOUSTS.csv": "stg_HOUSTS",
    "HSN1FNSA.csv": "stg_HSN1FNSA",
    "MORTGAGE30US.csv": "stg_MORTGAGE30US",
    "PCE.csv": "stg_PCE",
    "PERMIT.csv": "stg_PERMIT",
    "PERMIT1.csv": "stg_PERMIT1",
    "PERMIT5.csv": "stg_PERMIT5",
    "PERMIT24.csv": "stg_PERMIT24",
    "PRFI.csv": "stg_PRFI",
    "RPI.csv": "stg_RPI",
    "UMCSENT.csv": "stg_UMCSENT",
    "UNRATE.csv": "stg_UNRATE",
}

# === STEP 5: Loop over files and load to BQ ===
for file_name, table_name in file_to_table_map.items():
    gcs_uri = f"gs://{bucket_name}/{gcs_folder}/{file_name}"
    table_id = f"{project_id}.{dataset_id}.{table_name}"

    print(f"Loading {file_name} from GCS → {table_id}...")

    # === STEP 6: Define schema (2 columns: period_start_date + value) ===
    job_config = bigquery.LoadJobConfig(
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        autodetect=False,
        schema=[
            bigquery.SchemaField("period_start_date", "DATE"),
            bigquery.SchemaField("value", "FLOAT64")
        ],
        field_delimiter=",",
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ignore_unknown_values=True
    )

    # === STEP 7: Load the table from GCS URI ===
    load_job = client.load_table_from_uri(
        gcs_uri,
        table_id,
        job_config=job_config,
    )

    load_job.result()  # Wait for job to complete

    print(f"✅ Loaded {table_id} successfully.")

print("🎉 All tables loaded from GCS to BigQuery successfully.")