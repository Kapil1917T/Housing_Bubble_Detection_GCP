import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from google.cloud import bigquery
from google.oauth2 import service_account

# --- SETUP ---
st.set_page_config(page_title="Housing Market Risk Dashboard", layout="wide")
st.title("\U0001F3E1 U.S. Housing Market Bubble & Risk Dashboard")

# --- GCP CREDENTIALS ---
GCP_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_creds.json")
credentials = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS_PATH)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

# --- DATA LOADING ---
@st.cache_data(show_spinner=True)
def load_data():
    obt_df = client.query("SELECT * FROM housing_curated.table_obt_housing").to_dataframe()
    pred_df = client.query("SELECT * FROM housing_curated.model_predictions").to_dataframe()
    bubble_df = client.query("SELECT * FROM housing_curated.hybrid_bubble_flags").to_dataframe()
    metrics_df = client.query("SELECT * FROM housing_curated.model_metrics").to_dataframe()
    return obt_df, pred_df, bubble_df, metrics_df

obt_df, pred_df, bubble_df, metrics_df = load_data()

# --- DATA CLEANING + ENHANCEMENTS ---

# convert quarter_id to datetime for the obt_df
obt_df['quarter_start_date'] = pd.to_datetime(
    obt_df['quarter_id'].str[:4] + '-' + ((obt_df['quarter_id'].str[-1].astype(int) - 1) * 3 + 1).astype(str).str.zfill(2) + '-01'
)
obt_df['quarter_start_date'] = pd.PeriodIndex(obt_df['quarter_id'], freq='Q').to_timestamp()
obt_df['quarter_dash'] = obt_df['quarter_id'].str[:4] + '-' + obt_df['quarter_id'].str[-2:]

# convert quarter_id to datetime for the bubble_df
bubble_df['quarter_start_date'] = pd.to_datetime(
    bubble_df['quarter_id'].str[:4] + '-' + ((bubble_df['quarter_id'].str[-1].astype(int) - 1) * 3 + 1).astype(str).str.zfill(2) + '-01'
)
bubble_df['quarter_start_date'] = pd.PeriodIndex(bubble_df['quarter_id'], freq='Q').to_timestamp()
bubble_df['quarter_dash'] = bubble_df['quarter_id'].str[:4] + '-' + bubble_df['quarter_id'].str[-2:]

pred_df = pred_df.merge(
    obt_df[['quarter_id', 'quarter_start_date', 'mortgage_rate_30yr_value', 'unemployment_rate_value', 'consumer_sentiment_value']],
    on='quarter_id', how='left')

pred_df = pred_df.merge(metrics_df, on='model_type', how='left')

# convert quarter_id to datetime for the pred_df
pred_df['quarter_start_date'] = pd.to_datetime(
    pred_df['quarter_id'].str[:4] + '-' + ((pred_df['quarter_id'].str[-1].astype(int) - 1) * 3 + 1).astype(str).str.zfill(2) + '-01'
)
pred_df['quarter_start_date'] = pd.PeriodIndex(pred_df['quarter_id'], freq='Q').to_timestamp()
pred_df['quarter_dash'] = pred_df['quarter_id'].str[:4] + '-' + pred_df['quarter_id'].str[-2:]

bubble_df = bubble_df.merge(
    obt_df[['quarter_id', 'mortgage_rate_30yr_value', 'unemployment_rate_value', 'consumer_sentiment_value']],
    on='quarter_id', how='left')

# --- TABS ---
tabs = st.tabs(["\U0001F4C8 Market Forecasting", "\U0001F6A8 Bubble Risk Signals", "\U0001F4CA Macroeconomic Indicators"])

# =============================
# TAB 1: MARKET FORECASTING
# =============================
with tabs[0]:
    st.header("Forecast vs Actual: Housing Price Trends")
    selected_model = st.selectbox("Choose model for forecast analysis:", ["Linear", "Ridge", "Lasso"])
    model_df = pred_df[pred_df['model_type'] == selected_model].copy()

    # Chart 1 - Predicted vs Actual
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=model_df['quarter_start_date'], y=model_df['actual'], name='Actual', mode='lines+markers'))
    fig1.add_trace(go.Scatter(x=model_df['quarter_start_date'], y=model_df['predicted'], name='Predicted', mode='lines+markers'))
    fig1.update_layout(title='Predicted vs Actual HPI', xaxis_title='Quarter', yaxis_title='Housing Price Index')
    st.plotly_chart(fig1, use_container_width=True)

    # KPI Metrics
    st.subheader("Model Performance KPIs")
    k1, k2, k3 = st.columns(3)
    k1.metric("📉 RMSE", f"{model_df['RMSE'].iloc[0]:.2f}")
    k2.metric("🎯 Adjusted R²", f"{model_df['Adjusted_R2'].iloc[0]:.2f}")
    k3.metric("📊 SMAPE", f"{model_df['SMAPE'].iloc[0]:.2f}%")

    # Chart 2 - Forecast Misses
    st.subheader("Forecast Misses (Error Over Time)")
    model_df['error'] = model_df['actual'] - model_df['predicted']
    fig2 = px.bar(model_df, x='quarter_start_date', y='error', labels={'error': 'Forecast Error'})
    fig2.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3 - Drift vs Macro
    st.subheader("Forecast Drift vs Mortgage Rate")
    model_df['drift'] = model_df['predicted'].diff()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=model_df['quarter_start_date'], y=model_df['drift'], name='Forecast Drift'))
    fig3.add_trace(go.Scatter(x=model_df['quarter_start_date'], y=model_df['mortgage_rate_30yr_value'], name='30Y Mortgage Rate'))
    fig3.update_layout(title='Model Drift vs Mortgage Rate', xaxis_title='Quarter')
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4 - Mortgage Rate Regimes
    st.subheader("Mortgage Regimes vs Forecast Drift")
    model_df['regime'] = pd.cut(model_df['mortgage_rate_30yr_value'], bins=[0, 3, 6, 10], labels=['Low', 'Moderate', 'High'])
    fig4 = px.box(model_df, x='regime', y='drift', color='regime')
    fig4.update_layout(title='Forecast Drift by Mortgage Rate Regime')
    st.plotly_chart(fig4, use_container_width=True)

# =============================
# TAB 2: BUBBLE RISK SIGNALS
# =============================
with tabs[1]:
    st.header("Bubble Risk Monitoring")

    # Risk Score
    fig1 = px.line(bubble_df, x='quarter_start_date', y='risk_score', title='Risk Score Over Time')
    fig1.add_hline(y=0.5, line_dash="dash", line_color="gray")
    st.plotly_chart(fig1, use_container_width=True)

    # Price vs Sentiment
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bubble_df['quarter_start_date'], y=bubble_df['price_zscore'], name='Price Z-Score'))
    fig2.add_trace(go.Scatter(x=bubble_df['quarter_start_date'], y=bubble_df['consumer_sentiment_value'], name='Sentiment'))
    fig2.update_layout(title='Price Z-Score vs Sentiment')
    st.plotly_chart(fig2, use_container_width=True)

    # Momentum and Growth
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=bubble_df['quarter_start_date'], y=bubble_df['price_growth_4q'], name='4Q Price Growth'))
    fig3.add_trace(go.Scatter(x=bubble_df['quarter_start_date'], y=bubble_df['growth_accel'], name='Growth Acceleration'))
    fig3.update_layout(title='Growth Momentum')
    st.plotly_chart(fig3, use_container_width=True)

    # Affordability Proxy
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=bubble_df['quarter_start_date'], y=bubble_df['HAI_proxy'], name='Affordability Proxy'))
    fig4.add_trace(go.Scatter(x=bubble_df['quarter_start_date'], y=bubble_df['realistic_bubble_flag']*100, name='Bubble Flag (scaled)'))
    fig4.update_layout(title='Affordability vs Bubble Flags')
    st.plotly_chart(fig4, use_container_width=True)

# =============================
# TAB 3: MACRO INDICATORS
# =============================
with tabs[2]:
    st.header("Macroeconomic Indicators Explorer")

    fig1 = px.line(obt_df, x='quarter_start_date', y='unemployment_rate_value', title='Unemployment Rate')
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(obt_df, x='quarter_start_date', y='consumer_sentiment_value', title='Consumer Sentiment')
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(obt_df, x='quarter_start_date', y='real_income_value', title='Real Disposable Income')
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(obt_df, x='quarter_start_date', y='fed_funds_rate_value', title='Federal Funds Rate')
    st.plotly_chart(fig4, use_container_width=True)