import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from google.cloud import bigquery
from datetime import datetime
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


# Streamlit Tab 1 Update: Forecasting Tab with Meta Information

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Assume obt_df, pred_df, bubble_df, metrics_df already loaded and cleaned

# =============================
# TAB 1: MARKET FORECASTING
# =============================
with tabs[0]:
    st.header("Forecast vs Actual: Housing Price Trends")
    selected_model = st.selectbox("Choose model for forecast analysis:", ["XGBoost", "RandomForest", "DecisionTree"])
    model_df = pred_df[pred_df['model_type'] == selected_model].copy()
    model_df = model_df.drop_duplicates(subset=['quarter_id', 'model_type'])
    model_df = model_df.sort_values("quarter_start_date")
    
    
    def add_macro_event_lines(fig):
        """
        Adds macroeconomic event lines (vertical) to a Plotly figure.
        These include financial shocks and policy events like COVID-19, GFC, etc.

        Parameters:
        - fig (plotly.graph_objs.Figure): The input Plotly figure.

        Returns:
        - fig: The updated Plotly figure with vertical event lines.
        """

        # Event color coding by type
        event_colors = {
        "shock": "red",
        "policy": "blue"
    }

        macro_events = [
            {"label": "Dot-Com Bust", "date": "2001-03-01", "type": "shock"},
            {"label": "GFC", "date": "2008-12-01", "type": "shock"},
            {"label": "COVID-19", "date": "2020-03-01", "type": "shock"},
            {"label": "Rate Hike Start", "date": "2022-03-01", "type": "policy"}
        ]

        for event in macro_events:
            try:
                # 👇 Convert to Pandas Timestamp to avoid datetime + int errors
                raw_date = pd.to_datetime(event["date"])
                event_date = raw_date.to_period("Q").start_time

                fig.add_vline(
                    x=event_date,
                    line_dash="dash",
                    line_color=event_colors.get(event["type"], "gray"),
                    line_width=2,
                    opacity=0.7,
                    annotation=dict(
                        text=event["label"],
                        showarrow=False,
                        font=dict(color=event_colors.get(event["type"], "gray")),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=0.5,
                        borderpad=2,
                        x=0,  # left-aligned
                        xanchor="left",
                        y=1.05,
                        yanchor="bottom"
                    )
                )

            except Exception as e:
                st.warning(f"⚠️ Could not add macro event line for '{event['label']}': {str(e)}")

        return fig

    # Explanation Block — Dynamic
    st.markdown("#### ✍️ What does this chart show?")
    if selected_model == "XGBoost":
        st.info("""
        This chart compares the predicted Housing Price Index (HPI) values from the XGBoost model to the actual HPI over time.
        XGBoost, a gradient boosting method, often performs well by capturing complex nonlinear patterns in economic data.
        Use this to evaluate whether the model successfully captured key market behaviors (e.g., 2008 crash, COVID-19 dip).
        """)
    elif selected_model == "RandomForest":
        st.info("""
        This line chart illustrates the prediction capability of the Random Forest model against real housing prices.
        Random Forests are ensemble models that reduce overfitting by averaging across multiple decision trees.
        This helps assess model consistency during volatile periods such as post-2020 inflation.
        """)
    else:
        st.info("""
        This chart shows the Decision Tree model's forecast vs actual housing price index.
        While simpler, Decision Trees can struggle during abrupt market shifts.
        This helps understand where the model deviates from economic reality.
        """)

    # Chart 1 - Predicted vs Actual
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=model_df['quarter_start_date'], y=model_df['actual'], name='Actual', mode='lines+markers', line=dict(width=2)))
    fig1.add_trace(go.Scatter(x=model_df['quarter_start_date'], y=model_df['predicted'], name='Predicted', mode='lines+markers', line=dict(dash='dot', width=2)))
    fig1.update_layout(title='Predicted vs Actual HPI', xaxis_title='Years', yaxis_title='Housing Price Index')
    fig1 = add_macro_event_lines(fig1)
    st.plotly_chart(fig1, use_container_width=True)

    # KPI Metrics
    st.subheader("Model Performance KPIs")
    k1, k2 = st.columns(2)
    with k1:
        st.metric("\U0001F4C9 RMSE", f"{model_df['RMSE'].iloc[0]:.2f}")
        st.caption("""
        Root Mean Squared Error: Measures average magnitude of forecast error. Lower values mean better prediction accuracy.
        Particularly sensitive to large errors (outliers).
        """)
    # with k2:
    #     st.metric("\U0001F4AF Adjusted R²", f"{model_df['Adjusted_R2'].iloc[0]:.2f}")
    #     st.caption("""
    #     Adjusted R²: Indicates the proportion of variance explained by the model while adjusting for number of predictors.
    #     Helps evaluate overall goodness of fit.
    #     """)
    with k2:
        st.metric("\U0001F4CA SMAPE", f"{model_df['SMAPE'].iloc[0]:.2f}%")
        st.caption("""
        Symmetric Mean Absolute Percentage Error: Measures relative forecast error. Less biased toward large values.
        Useful for interpretability in percentage terms.
        """)

    # ========================
    # Chart 2 - Forecast Misses
    # ========================

    st.subheader("Forecast Misses (Error Over Time)")

# Static metadata explanation for all models
    st.info("""
    This bar chart visualizes how far off the model’s predictions were from reality in each quarter.  
    • **Positive bars** indicate the model **underestimated** actual prices.  
    • **Negative bars** indicate the model **overestimated** actual prices.  

    Hence, I'm using this chart to:  
    • Detect quarters where the model failed to track real-world trends  
    • Identify volatility during key periods like economic shocks (e.g., COVID-19, GFC)  
    • Evaluate model stability across different market regimes  
    """, icon="📉")

    # Compute and plot forecast error
    model_df['error'] = model_df['actual'] - model_df['predicted']
    fig2 = px.bar(
        model_df,
        x='quarter_start_date',
        y='error',
        labels={'error': 'Forecast Error', 'quarter_start_date': 'Years'},
        title='Quarterly Forecast Error'
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="black")
    fig2 = add_macro_event_lines(fig2)
    st.plotly_chart(fig2, use_container_width=True)


    # ==============================
    # Chart 3 - Forecast Drift vs Mortgage Rate
    # ==============================

    st.subheader("Forecast Drift vs Mortgage Rate")

    # Explanation box (static)
    st.info("""
    **What is Forecast Drift?**  
    Forecast drift is the **quarter-over-quarter change in predicted housing prices**. It reflects how rapidly the model
    adjusts its predictions in response to new economic inputs.

    **Why is it important?**  
    Comparing forecast drift to mortgage rates helps evaluate the **model’s sensitivity to monetary policy changes** 
    (e.g., rate hikes, QE, or tapering).

    **Insights:**  
    • Large spikes in drift indicate economic shocks (e.g., COVID, 2008)  
    • Gradual drifts show stable market trends  
    • Mortgage rates act as a policy signal — sudden rate changes should ideally be captured by forecast drift  
    """, icon="📈")

    # Compute forecast drift (quarterly change)
    model_df['drift'] = model_df['predicted'].diff()

    # Plot forecast drift alongside mortgage rate
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=model_df['quarter_start_date'], y=model_df['drift'], name='Forecast Drift'))
    fig3.add_trace(go.Scatter(x=model_df['quarter_start_date'], y=model_df['mortgage_rate_30yr_value'], name='30Y Mortgage Rate'))
    fig3.update_layout(title='Model Drift vs Mortgage Rate', xaxis_title='Years')
    fig3 = add_macro_event_lines(fig3)
    st.plotly_chart(fig3, use_container_width=True)


    # ==============================
    # Chart 4 - Forecast Drift by Mortgage Rate Regime
    # ==============================

    st.subheader("Forecast Drift by Mortgage Regime")

    # Explanation box (static)
    st.info("""
    **What is a Mortgage Rate Regime?**  
    Mortgage regimes classify periods based on average **30-year mortgage rates**:  
    • **Low**: 0% to 3%  
    • **Moderate**: 3% to 6%  
    • **High**: 6% to 10%  

    **Why this matters:**  
    Grouping drift values by regime helps assess whether the model performs **worse during high-rate environments**, 
    which are often unstable or recession-prone.

    **Insights:**  
    • Forecast drift is generally more erratic under **High-rate regimes**, showing greater model instability  
    • **Moderate-rate periods** see the most consistent and stable predictions  
    • Under **Low-rate regimes**, the model may overcorrect due to excess optimism in macro indicators  
    """, icon="📊")

    # Categorize mortgage rate into regime buckets
    model_df['regime'] = pd.cut(
        model_df['mortgage_rate_30yr_value'],
        bins=[0, 3, 6, 10],
        labels=['Low', 'Moderate', 'High']
    )

    # Violin + scatter plot to show drift variation by regime
    fig4 = px.violin(
        model_df,
        x='regime',
        y='drift',
        color='regime',
        box=True,
        points="all",  # add scatter dots
        title='Forecast Drift by Mortgage Rate Regime',
        labels={'regime': 'Mortgage Rate Regime', 'drift': 'Forecast Drift'}
    )
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