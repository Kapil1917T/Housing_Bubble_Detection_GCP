import os
from google.cloud import storage
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load environment variables from .env (if present, useful for local testing)
load_dotenv()

# === GCP CREDENTIAL SETUP ===
GCP_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "gcp_creds.json")

# Load credentials from file
credentials = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS_PATH)

# Initialize storage client with explicit credentials
storage_client = storage.Client(credentials=credentials)

# === CONFIGURATION ===
# Folder where cleaned CSVs are stored
LOCAL_CLEAN_FOLDER = "data/clean"

# Name of your target GCS bucket
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "housing-bubble-predictor-data")

# Optional prefix (folder path) inside the GCS bucket
GCS_FOLDER_PREFIX = "cleaned_data"

# === FUNCTION TO UPLOAD TO GCS ===
def upload_to_gcs(storage_client, local_path, gcs_path):
    """Uploads a local file to GCS."""
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"[✓] Uploaded: {local_path} → gs://{GCS_BUCKET_NAME}/{gcs_path}")

# === MAIN FUNCTION ===
def main():
    if not os.path.isdir(LOCAL_CLEAN_FOLDER):
        print(f"[X] Folder not found: {LOCAL_CLEAN_FOLDER}")
        return

    # Walk through all files in data/clean
    for filename in os.listdir(LOCAL_CLEAN_FOLDER):
        if filename.endswith(".csv"):
            local_file = os.path.join(LOCAL_CLEAN_FOLDER, filename)
            gcs_file_path = f"{GCS_FOLDER_PREFIX}/{filename}"
            try:
                upload_to_gcs(storage_client, local_file, gcs_file_path)
            except Exception as e:
                print(f"[!] Failed to upload {filename}: {e}")

# === ENTRY POINT ===
if __name__ == "__main__":
    main()