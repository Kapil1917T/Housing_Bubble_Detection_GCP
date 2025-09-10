import os
import pandas as pd

# Define raw and cleaned folder paths
RAW_FOLDER = 'data/raw'
CLEAN_FOLDER = 'data/clean'

# Create the clean folder if it doesn't exist
os.makedirs(CLEAN_FOLDER, exist_ok=True)

# Loop through each CSV in the raw data folder
for filename in os.listdir(RAW_FOLDER):
    if filename.endswith('.csv'):
        raw_path = os.path.join(RAW_FOLDER, filename)
        clean_path = os.path.join(CLEAN_FOLDER, filename)

        # 🔹 Step 1: Read raw CSV
        df = pd.read_csv(raw_path)

        # 🔹 Step 2: Rename `period_start_date` → `date` (if present)
        if 'period_start_date' in df.columns:
            df.rename(columns={'period_start_date': 'date'}, inplace=True)

        # 🔹 Step 3: Identify and rename the "value" column
        # This is the economic indicator column (e.g., "UNRATE", "COMPUMUSA", etc.)
        # It's usually the second column after the date
        data_columns = [col for col in df.columns if col not in ['date', 'realtime_start_date', 'realtime_end_date']]
        if len(data_columns) == 1:
            df.rename(columns={data_columns[0]: 'value'}, inplace=True)
        elif 'value' not in df.columns:
            print(f"⚠️ Skipped {filename}: could not confidently identify 'value' column.")
            continue  # Skip files where value column can't be reliably renamed

        # Step 4.a: Add missing columns (if needed) and reorder
        if 'realtime_start_date' not in df.columns:
            df['realtime_start_date'] = pd.NA
        if 'realtime_end_date' not in df.columns:
            df['realtime_end_date'] = pd.NA
        
        # 🔹 Step 4.b: Retain only necessary columns (order enforced)
        keep_cols = ['date', 'realtime_start_date', 'realtime_end_date', 'value']
        df = df[[col for col in keep_cols if col in df.columns]]

        # 🔹 Step 5: Drop fully empty rows
        df.dropna(how='all', inplace=True)

        # 🔹 Step 6: Sort chronologically if date exists
        if 'date' in df.columns:
            df.sort_values('date', inplace=True)

        # 🔹 Step 7: Save cleaned version
        df.to_csv(clean_path, index=False)
        print(f"✅ Cleaned and saved: {clean_path}")