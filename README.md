# 🏠 Housing Bubble Detection & Forecasting (End-to-End GCP Pipeline)

Welcome to the Housing Bubble Detection & Forecasting project, built entirely on **Google Cloud Platform (GCP)** with full support for **time series forecasting**, **macroeconomic risk scoring**, and **CI/CD automation**. This project blends real-world economic theory with modern cloud-first data science to tackle a critical question:

> **Can we detect early signs of housing market bubbles before they burst?**

Over the course of several weeks, I developed this project from scratch with careful attention to **reproducibility**, **modularity**, and **economic realism**. The goal was not just to build another ML model, but to simulate what a real analytics pipeline for real estate or financial policy might look like inside a large organization.

🌐 [**Live Streamlit App**](https://housingbubbledetection-es8gr954eezai7dkyzvv8k.streamlit.app/)  
Experience the working dashboard in real time! No setup needed — just click and explore bubble risk trends, macro indicators, and model forecasts.

> 🔁 The app auto-refreshes predictions and risk flags using GitHub Actions and BigQuery-backed pipelines.

---

## 📌 Project Context & Motivation

The 2008 housing crisis showed us how delayed signals and over-optimistic valuations can wreak havoc. Today, with more data than ever, we can do better. This project aims to:

- Quantify **bubble-like behavior** using publicly available macroeconomic indicators  
- Forecast future values of the **Home Price Index (HPI)** using interpretable models  
- Deliver results via a **dashboard** and **refreshable cloud pipeline** for decision-makers  

Unlike most toy projects that stop at modeling, this repo shows how to:

- Ingest economic data programmatically  
- Clean, store, and version it securely in the cloud  
- Create and monitor predictive models  
- Automate everything with GitHub Actions  
- Visualize signals for **strategic storytelling**  

---

## 🌍 Tools, Stack & Infrastructure

- `Python` for all modeling, cleaning, and pipeline scripts  
- `Google Cloud Storage (GCS)` for cleaned CSV uploads  
- `BigQuery` for warehousing, modeling, and joining staging tables  
- `Streamlit` for dashboarding (deployed via Streamlit Community Cloud)  
- `GitHub Actions` to orchestrate periodic refreshes and model retraining  
- `FRED API` as the data source for 23 economic indicators (CPI, HPI, FEDFUNDS, etc.)  

> This is an end-to-end **ML operations workflow**, not just a static notebook.

---

## ⚙️ Pipeline Structure (Fully Modularized)

Each stage of the project is implemented as an **independent and reproducible component**:

### 1. Data Ingestion & Cleaning
- Fetches macroeconomic indicators from **FRED API** into `data/raw/`  
- Cleans and standardizes files using `clean_fred_data.py`  
- Uploads cleaned files to GCS bucket: `housing-bubble-predictor-data/cleaned_data`  

### 2. BigQuery Staging & OBT Creation
- All 23 cleaned CSVs are ingested as `stg_*` tables in BigQuery  
- Final OBT (`table_obt_housing`) is created with proper date formatting (e.g., `1974Q1`), joins, and deduplication  
- View (`view_obt_housing`) created for real-time dashboarding  

### 3. Model Training & Forecasting
- `market_predictor_bq.py` uses walk-forward validation with `XGBoost`, `Random Forest`, and `Decision Tree`  
- Feature engineering includes rolling stats, interaction terms, and policy event flags (rate hikes, inflation spikes)  
- Outputs predictions + metrics (`RMSE`, `SMAPE`, `Adjusted R²`) to BigQuery tables  

### 4. Bubble Risk Detection
- `bubble_detection_bq.py` computes:  
  - HPI acceleration, Affordability proxy  
  - Macro correlation breakdowns, Price z-scores, Sentiment flags  
- Scores each quarter on speculative risk and flags realistic bubbles (e.g., **2005–2008** period)  

### 5. Visualization Dashboard
- `streamlit_dashboard.py` shows:  
  - Predicted vs Actual HPI (toggle between models)  
  - Forecast drift vs macro indicators  
  - Bubble score timeline and signal breakdown  
- Includes explanatory tooltips, KPIs, and **strategic commentary**  

---

## 🗺️ Project Architecture

```
FRED API ──► GitHub Actions (ETL) ──► GCS (Cleaned CSVs)
                      │
                      ▼
              BigQuery Staging Tables
                      ▼
           One Big Table (Quarterly HPI)
                      ▼
    ┌────────────────────┬────────────────────┐
    │                    │                    │
Model Training      Bubble Scoring       Streamlit App
(XGBoost, RF, DT)   (Rule-based Logic)   (3-tab Dashboard)
                      │
                      ▼
     BigQuery Tables: Predictions, Flags, KPIs
```

---

## 🚀 What Makes This Project Unique

- ✅ **End-to-end reproducibility**: From ingestion to risk flag, everything is version-controlled and refreshable  
- 📈 **Economic realism**: All modeling logic is benchmarked against actual **2006–2008 crash** behavior  
- ☁️ **Cloud-native**: No local dependencies — works entirely on GCP with secure credentials  
- 🔁 **Modular & scalable**: Can easily extend to city-level HPI, Zillow/Redfin, or LSTM models  
- ⚙️ **CI/CD included**: Every script is connected to a YAML, refreshable via cron

---

## 🛌 For Recruiters & Reviewers

If you're reviewing this project from a **hiring** or **academic** lens, here’s what I’d like to highlight:

- This project blends **domain understanding (macro econ)** with **ML & infra maturity**
- Built to mirror **real-world deployment** — not just a Kaggle notebook
- Clean structure: `requirements.txt`, `.gitignore`, secrets separation, auto-refresh YAMLs
- Demonstrates **awareness of model interpretability**, policy events, economic indicators
- Easily extendable to **fintech, proptech, policy analytics, or investment planning** use cases

---

## ✅ Next Steps (Planned Enhancements)

- Add SHAP value visualizations to dashboard for interpretability  
- Extend bubble detection to **MSA/ZIP-level granularity** (Zillow/Redfin if possible)  
- Trigger **email/SMS alerts** for high-risk quarters using GCP Pub/Sub  
- Add **Looker Studio** connectors for enterprise reporting layer  
- Try transformer-based models or `Prophet` for multi-horizon forecasting  

---

## 👨‍💼 Author

**Kapil Tare** \
M.S. Applied Data Science, Syracuse University  \
[LinkedIn](https://linkedin.com/in/kapiltare) • [GitHub](https://github.com/Kapil1917T)  

Always happy to chat about forecasting, cloud infra, or macroeconomic modeling.  
Feel free to connect or collaborate!  

---

## ⛔ Disclaimer

> This project is built for **personal portfolio** demonstration.  
All data used is publicly available. Any insights or risk predictions should **not** be construed as **financial advice**. Always consult professional analysts or institutions before making economic decisions based on such models.

---

## 🎓 Acknowledgments

- **FRED API** for access to rich macroeconomic data  
- **Streamlit** for simplifying interactive dashboards  
- **Google Cloud** for robust infra and generous free tier  

---

**Thank you for visiting!**  
If you found this project insightful or want to collaborate — let’s connect 🚀

— *Kapil*

