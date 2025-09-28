import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import json
from google.cloud import bigquery
from datetime import datetime
from google.oauth2 import service_account
from google.cloud import bigquery

# --- SETUP ---
st.set_page_config(page_title="Housing Market Risk Dashboard", layout="wide")
st.title("\U0001F3E1 U.S. Housing Market Bubble & Risk Dashboard")

# --- GCP CREDENTIALS ---
# GCP_CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_creds.json")
# credentials = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS_PATH)
# client = bigquery.Client(credentials=credentials, project=credentials.project_id)

creds_json = os.environ.get("GCP_CREDENTIALS_JSON")
info = json.loads(creds_json)
credentials = service_account.Credentials.from_service_account_info(info)
client = bigquery.Client(credentials=credentials, project=info["project_id"])


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
tabs = st.tabs(["\U0001F4C8 Market Forecasting", "\U0001F6A8 Bubble Risk Signals"])


# Streamlit Tab 1 Update: Forecasting Tab with Meta Information

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
        {"label": "Dot-Com Bubble Burst", "date": "2001-03-01", "type": "shock", "description": "Tech stock crash and economic slowdown"},
        {"label": "2008 Housing Crash", "date": "2008-12-01", "type": "shock", "description": "Global Financial Crisis (GFC) triggered by housing market collapse"},
        {"label": "COVID-19", "date": "2020-03-01", "type": "shock", "description": "Global pandemic causing economic shutdowns"},
        {"label": "Fed QE Start", "date": "2020-03-01", "type": "policy", "description": "Federal Reserve begins quantitative easing by buying assets to support economy"},
        #{"label": "Fed Rate Hike Cycle Begins", "date": "2022-03-01", "type": "policy", "description": "Fed starts increasing interest rates to combat inflation"},
        {"label": "2023 Banking Crisis", "date": "2023-03-01", "type": "shock", "description": "Collapse of major banks like SVB and Signature causing market turmoil"}
        ]


        for event in macro_events:
            try:
                # Safe conversion to Timestamp (prevents +int errors)
                event_date = pd.Period(event["date"], freq="M").to_timestamp()
                color = event_colors.get(event["type"], "gray")

                # Add invisible scatter for hover info
                fig.add_trace(go.Scatter(
                x=[event_date],
                y=[fig.layout.yaxis.range[1] * 0.98 if fig.layout.yaxis.range else None],  # Position near top
                mode="markers",
                marker=dict(size=8, color=event_colors.get(event["type"], "gray"), opacity=0),
                hoverinfo="text",
                hovertext=f"<b>{event['label']}</b><br>{event_date.strftime('%b %Y')}<br>{event['description']}",
                showlegend=False
                ))
                
                # Add vertical line
                fig.add_vline(
                    x=event_date,line_dash="dash",line_color=color,line_width=2,opacity=0.7
                )

                # Add annotation label
                fig.add_annotation(
                    x=event_date, y=1, yref="paper", showarrow=False, text=event["label"], font=dict(color=color),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=0.5, borderpad=2, xanchor="left",
                    hovertext=event['description']
                )
                
            except Exception as e:
                st.warning(f"⚠️ Could not add macro event line for '{event['label']}': {str(e)}")

        return fig
    
    # Chart 1 - Predicted vs Actual HPI
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=model_df['quarter_start_date'], y=model_df['actual'], name='Actual', mode='lines+markers', line=dict(width=2)))
    fig1.add_trace(go.Scatter(x=model_df['quarter_start_date'], y=model_df['predicted'], name='Predicted', mode='lines+markers', line=dict(dash='dot', width=2)))
    fig1.update_layout(title='Predicted vs Actual HPI', xaxis_title='Years', yaxis_title='Housing Price Index')
    fig1 = add_macro_event_lines(fig1)
    st.plotly_chart(fig1, use_container_width=True)
    
    with st.expander("ℹ️  **About this Chart: Forecast vs Actual Housing Price Index (HPI)**", expanded=False):
        st.markdown("""        
        ✍️  **What does this chart show?**
        
        This chart compares the predicted Housing Price Index (HPI) values from the 3 non-linear models (XGBoost, Random Forest, and Decision Tree) to the 
        actual HPI over time. The goal is to evaluate how well each model tracks real-world housing price trends, especially during periods of economic
        stress or rapid change.
        
        🔍 **Why is this chartimportant?**
        
        Evaluating forecast accuracy over time helps assess model reliability and responsiveness to macroeconomic shifts. Housing markets are influenced by
        complex factors like interest rates, income levels, and consumer sentiment. A model that closely tracks actual prices during crises (e.g., 2008 GFC, 2020 COVID)
        demonstrates robustness and practical utility for policymakers, investors, and analysts.
        
        🧮 **How is forecast accuracy being measured?**
        
        - **RMSE (Root Mean Squared Error)**: Captures average magnitude of forecast errors. Sensitive to large deviations.
        - **SMAPE (Symmetric Mean Absolute Percentage Error)**: Measures relative forecast error in percentage terms. Less biased toward large values.
        
        **Metric Interpretation**: Lower RMSE and SMAPE values indicate better predictive performance. However, understanding when and why the model deviates from reality
        during key events is equally important for assessing practical reliability.
        
        **How to interpret this chart?**
        
        - **Close Tracking**: Periods where predicted and actual lines overlap closely indicate strong model performance.
        - **Divergences**: Gaps between predicted and actual lines highlight forecast errors. Analyzing these during macro events reveals model strengths and weaknesses.

        📌 **Key Observations:**

        - **2008 Housing Crash**: The drift sharply increases after the mortgage rate begins falling, indicating the model reacts to recession-era monetary easing 
        but with a slight delay. This highlights the lagged forecast adjustment during crisis onset.
        - **2020 COVID Shock**: A steep negative drift spike occurs as the Fed slashes rates to near-zero. The model reacts sharply, showing high sensitivity to
        sudden rate cuts, though the magnitude reflects model recalibration shock rather than true anticipation.
        - **2022–2023 Rate Hike Cycle**: Multiple drift spikes correspond with aggressive rate hikes by the Fed. However, some lag exists between rate jumps 
        and drift peaks, suggesting the model adjusts reactively, not proactively.
        - **Dot-Com Bubble (2001)**: Drift remains relatively muted despite mild rate changes, suggesting the model does not perceive this as a housing-led 
        macro shift — which aligns with reality (the bubble wasn’t housing-related).
        - **Gradual Rate Environments (2012–2018)**: Periods of policy stability show minimal drift, indicating forecast consistency and economic stability — 
        a desirable trait for long-range forecasting systems.
        """)

    # KPI Metrics
    st.subheader("Model Performance KPIs")
    k1, k2 = st.columns(2)
    with k1:
        st.metric("\U0001F4C9 RMSE", f"{model_df['RMSE'].iloc[0]:.2f}")
        st.caption("""
        Root Mean Squared Error: Measures average magnitude of forecast error. Lower values mean better prediction accuracy.
        Particularly sensitive to large errors (outliers).
        """)
    with k2:
        st.metric("\U0001F4CA SMAPE", f"{model_df['SMAPE'].iloc[0]:.2f}%")
        st.caption("""
        Symmetric Mean Absolute Percentage Error: Measures relative forecast error. Less biased toward large values.
        Useful for interpretability in percentage terms.
        """)

    # Chart 2 - Forecast Misses

    st.subheader("Forecast Misses (Error Over Time)")

    # Compute and plot forecast error
    model_df['error'] = model_df['actual'] - model_df['predicted']
    fig2 = px.bar(
        model_df,
        x='quarter_start_date',
        y='error',
        labels={'error': 'Forecast Error (Index Points)', 'quarter_start_date': 'Years'},
        title='Quarterly Forecast Error'
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="black")
    fig2 = add_macro_event_lines(fig2)
    st.plotly_chart(fig2, use_container_width=True)
    
    with st.expander("ℹ️  **About this Chart: Forecast Misses (Error Over Time)**", expanded=False):
        st.markdown("""
        ✍️  **What does this chart show?**
        
        This chart visualizes the **quarterly forecast errors** of the selected model by plotting the difference between actual and 
        predicted Housing Price Index (HPI) values over time. The goal is to identify periods where the model struggled to track real-world housing price trends, 
        especially during economic shocks or rapid changes.
        
        🔍 **Why is this chart important?**
        
        Evaluating forecast errors over time helps assess model reliability and stability across different market regimes. Large errors during key events (e.g., 2008 GFC, 2020 COVID)
        highlight model weaknesses and areas for improvement. Consistent small errors indicate robustness and practical utility for policymakers, investors, and analysts.
        
        **How to interpret this chart?**
        
        - **Positive Bars**: Indicate the model **underestimated** actual prices (predicted < actual). Suggests the model missed upward price trends.
        - **Negative Bars**: Indicate the model **overestimated** actual prices (predicted > actual). Suggests the model missed downward price trends.
        - **Magnitude of Bars**: Larger bars represent bigger forecast misses. Analyzing these during macro events reveals model strengths and weaknesses.
        """
        )

    # Chart 3 - Forecast Drift vs Mortgage Rate

    st.subheader("Forecast Drift vs Mortgage Rate")

    # Compute forecast drift (quarterly change)
    model_df['drift'] = model_df['predicted'].diff()
    
    # Plot with dual y-axes
    fig3_dual = go.Figure()
    
    # Left Y-Axis: Forecast Drift
    fig3_dual.add_trace(go.Scatter(
        x=model_df['quarter_start_date'],
        y=model_df['drift'],
        name='Forecast Drift',
        mode='lines',
        line=dict(color='blue'),
        yaxis='y1'
    ))
    
    # Right Y-Axis: 30Y Mortgage Rate
    fig3_dual.add_trace(go.Scatter(
        x=model_df['quarter_start_date'],
        y=model_df['mortgage_rate_30yr_value'],
        name='30Y Mortgage Rate',
        mode='lines',
        line=dict(color='orange', dash='dot'),
        yaxis='y2'
    ))
    
    # Update layout with dual y-axes (✅ no titlefont)
    fig3_dual.update_layout(
        title='Forecast Responsiveness vs Mortgage Rate (Dual Y-Axis)',
        xaxis=dict(title='Quarter'),
        yaxis=dict(
            title=dict(text='Forecast Drift (Index Points)', font=dict(color='blue')),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title=dict(text='Mortgage Rate (%)', font=dict(color='orange')),
            tickfont=dict(color='orange'),
            overlaying='y',
            side='right'
        ),
        legend=dict(x=1.01, y=1.15, orientation='v'),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    fig3_dual = add_macro_event_lines(fig3_dual)
    st.plotly_chart(fig3_dual, use_container_width=True)
    
    with st.expander("ℹ️  **About this Chart: Forecast Drift vs Mortgage Rate**", expanded=False):
        st.markdown("""
            **What is Forecast Drift?**  
            
            Forecast drift is the **quarter-over-quarter change in predicted housing prices**. that reflects how rapidly the model
            adjusts its predictions in response to new economic inputs. Basically, It represents the **absolute change in predicted Housing Price Index (HPI)** 
            from one quarter to the next. Larger HPI values indicate that the model is rapidly adjusting forecasts due to new macroeconomic signals.

            **Why is it important?** 
            
            Comparing forecast drift to mortgage rates helps evaluate the **model’s sensitivity to monetary policy changes** (e.g., rate hikes, QE, or tapering).
    
            **SIDE NOTE:** 
            The goal of this visualization is not to compare the absolute magnitudes of the two axes, but to see whether **sudden rate changes 
            (spikes or drops)** correspond with **high forecast drift or not, indicating model responsiveness.**

            **Insights:**
            
            - Large spikes in **forecast drift** around:
                - 2008 (📉 Housing Crash),
                - 2020 (🦠 COVID Shock),
                - 2022–23 (📈 Rate Hike Cycle)  
            - This Indicates our tree-based models also have limitations wherein they **lag** or **underreact** to sudden policy changes or external shocks as 
            they are **not time-aware** and can only recognizes patterns, but lacks understanding of policy dynamics or temporal causality.
        """)

# =============================
# TAB 2: BUBBLE RISK SIGNALS
# =============================
bubble_df = bubble_df.drop_duplicates(subset='quarter_id')
bubble_df = bubble_df.sort_values("quarter_start_date")

with tabs[1]:
    st.header("Bubble Risk Monitoring")
    
    # Risk Score
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=bubble_df["quarter_start_date"],
        y=bubble_df["risk_score"],
        mode="lines+markers",
        name="Risk Score",
        line=dict(color="crimson", width=2),
    ))

    # Macro event lines
    fig_risk = add_macro_event_lines(fig_risk)

    fig_risk.update_layout(
        yaxis_title="Risk Score (0 to 100 pts)",
        xaxis_title="Years",
        title="Risk Score Evolution Across Economic Events",
        margin=dict(t=60, b=40),
        height=400
    )
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    with st.expander("ℹ️  **About this Chart: Bubble Risk Score Over Time**", expanded=False):
        st.markdown("""        
        ✍️  **What does this chart show?**
        
        This chart visualizes the **housing market risk score over time**, based on a hybrid set of rules that quantify signs of speculative behavior, 
        affordability stress, and macroeconomic misalignment in the U.S. housing market. Each point on the line represents the **risk score for 
        that quarter (0–100)**, calculated using real economic indicators.

        🔍 **Why are we measuring risk scores?**  
        
        The goal is to detect early-warning signals of potential housing market bubbles — periods where home prices may be rising irrationally due to speculation, 
        policy distortion, or affordability collapse. This risk score helps policymakers, analysts, and investors track when the 
        market is **overheating** or **reverting to normalcy**, especially across historic economic cycles.
        
        **SIDE NOTE:** My main aim of building this system is to **not predict whether a bubble has been formed or will burst**. Instead, it's tracking the 
        **buildup and intensity of bubble-like conditions** — enabling proactive insights **before** a full-blown crash.

        🧮 **How is the risk score being calculated?**  
        
        My method uses a **rule-based scoring engine** applied to each quarter in the housing market:

        - **Price Growth (4Q trailing)**: Detects sustained surges in home prices (up to 30 pts)
        - **Growth Acceleration**: Flags second-derivative surges in market pace (5 pts)
        - **Z-Score Deviation**: Measures how far prices deviate from historical norms (up to 25 pts)
        - **Affordability Index Drop**: Captures when housing becomes financially out of reach (10 pts)
        - **Momentum Indicators**: Tracks herd-driven buying behavior (15 pts)
        - **Rate/Price Decoupling**: Checks if mortgage rates rise but prices don’t fall (up to 20 pts)
        - **Compound Rule**: Extra points if high growth and high Z-score co-occur (10 pts)
        
        These signals are scored and aggregated into a **0–100 risk score**, with thresholds:
        - 🔴 **High Risk (≥ 60)**  
        - 🟠 **Medium Risk (40–59)**  
        - 🟢 **Low Risk (< 40)**  
        
        📈 **How to interpret this chart?**

        - **Higher spikes** indicate periods of heightened bubble-like behavior.
        - **Lower scores** reflect fundamental alignment or post-correction normalization.
        - **Vertical event markers** help anchor the risk score trends to known real-life economic events (e.g., 2008 Crash, COVID QE, 2023 Banking Crisis).
        - The chart lets us **visually backtest** our bubble signals against history and assess how rational the housing market has been across time.


        🧠 **How well does this match real-life housing bubbles?**
        - **2001**: No false positive — dot-com wasn’t a housing bubble.
        - **2005–2008**: Sharp rise in risk score, sustained >60, followed by crash — aligned with the 2008 bubble burst.
        - **2012–2018**: Low scores during stable, non-speculative recovery.
        - **2020–2021**: Spike during COVID due to ultra-low rates, low affordability, and high momentum.
        - **2022–2023**: Sticky prices + affordability crash = another spike — reflecting growing fragility.

        📌 **Key Observations:**

        - **Sharp spikes** in 2008, 2020, and 2022–23 align with real-world crises.
        - **Bubble flags emerge** during periods of rising prices despite falling affordability.
        - **Low risk periods** (e.g., 2012–2018) reflect strong fundamental alignment and stable policy conditions.
        - **No false positives** during non-housing events like the 2001 tech crash — validating our rule set.
        - The rule-based system balances interpretability with accuracy — no black-box assumptions.
        """)
    
    # Price vs Sentiment (Dual Axis)
    # Z-Score Line
    fig2_pr_sent = go.Figure()
    fig2_pr_sent.add_trace(go.Scatter(
        x=bubble_df['quarter_start_date'],
        y=bubble_df['price_zscore'],
        name="Price Z-Score",
        line=dict(color='blue'),
        yaxis="y1"
    ))

    # Sentiment Line
    fig2_pr_sent.add_trace(go.Scatter(
        x=bubble_df['quarter_start_date'],
        y=bubble_df['consumer_sentiment_value'],
        name="Sentiment",
        line=dict(color='orange', dash='dot'),
        yaxis="y2"
    ))

    # Layout: Dual Axis Setup
    fig2_pr_sent.update_layout(
    title="Price Z-Score vs Consumer Sentiment",
    xaxis=dict(
        title="Years"
    ),
    yaxis=dict(
        title="Z-Score (Housing Price)",
        title_font=dict(color="blue"),
        tickfont=dict(color="blue"),
        side="left",
        range=[-5, 5]
    ),
    yaxis2=dict(
        title="Consumer Sentiment (Index)",
        title_font=dict(color="orange"),
        tickfont=dict(color="orange"),
        overlaying="y",
        side="right",
        range=[50, 115],
        showgrid=False
    ),
    legend=dict(
        x=0.01,
        y=0.99,
        orientation="h"
    ),
    margin=dict(l=60, r=60, t=50, b=50),
    template="plotly_white"
)
    
    # Add macroeconomic event lines
    fig2_pr_sent = add_macro_event_lines(fig2_pr_sent)

    # Streamlit container
    st.plotly_chart(fig2_pr_sent, use_container_width=True)
    
    with st.expander("ℹ️  **About this Chart: Price Z-Score vs Consumer Sentiment**", expanded=False):
        st.markdown("""
            ✍️  **What does this chart show?**  
            This chart compares the **Z-Score of housing prices** (how far prices deviate from historical norms)  
            against **Consumer Sentiment** (a proxy for how confident households feel about the economy).

            🔍  **Why this comparison matters?**  
            - Rising **price Z-scores** can signal that prices are becoming irrational relative to long-term trends.
            - **Consumer sentiment** often reflects whether the public is aware of, or participating in, that irrationality.
            - A **divergence** between these two metrics may indicate early signs of **unsustainable pricing behavior**.

            🧮  **How is Z-score calculated?**  
            - Computed as:  
            `Z = (Current Price - 5Y Rolling Mean) / 5Y Rolling StdDev`
            - A Z-score > 2 is often seen as strong evidence of overvaluation.

            📈  **How to interpret this chart?**  
            - **2005–2007**: Z-score > 2 while consumer sentiment was also high — classic signs of speculative optimism.
            - **2008–2009**: Sentiment collapses *before* Z-score drops — public panic preceded price correction.
            - **2020–2022**: Z-score climbs fast post-COVID while sentiment **plummets** due to inflation, signaling misalignment.
            - **Recent Quarters**: High Z-scores + weak sentiment → public **disbelief** in continued price levels.
            

            📌 **Key Observations:**
            - The gap between sentiment and price valuation has widened in recent years, raising red flags.
            - Ideal scenario: both metrics rise/fall together, showing price movements are economically grounded.
            - Current scenario: prices remain elevated despite low sentiment → **vulnerability to correction**.
            - Overall, When Z-score is high but sentiment drops (e.g., 2005–07, 2022–23), it may signal a disconnect — where prices remain 
            irrationally high despite rising public doubt — a potential bubble warning zone.

            **SIDE NOTE:**  
            A **high Z-score with falling sentiment** often signals the "last leg" of a bubble — when public optimism has faded but prices haven't corrected yet.
    """)

    # # Momentum and Growth
    # fig3 = go.Figure()
    # fig3.add_trace(go.Scatter(x=bubble_df['quarter_start_date'], y=bubble_df['price_growth_4q'], name='4Q Price Growth'))
    # fig3.add_trace(go.Scatter(x=bubble_df['quarter_start_date'], y=bubble_df['growth_accel'], name='Growth Acceleration'))
    # fig3.update_layout(title='Growth Momentum')
    # st.plotly_chart(fig3, use_container_width=True)

    # # Affordability Proxy
    # fig4 = go.Figure()
    # fig4.add_trace(go.Scatter(x=bubble_df['quarter_start_date'], y=bubble_df['HAI_proxy'], name='Affordability Proxy'))
    # fig4.add_trace(go.Scatter(x=bubble_df['quarter_start_date'], y=bubble_df['realistic_bubble_flag']*100, name='Bubble Flag (scaled)'))
    # fig4.update_layout(title='Affordability vs Bubble Flags')
    # st.plotly_chart(fig4, use_container_width=True)

# # =============================
# # TAB 3: MACRO INDICATORS
# # =============================
# with tabs[2]:
#     st.header("Macroeconomic Indicators Explorer")

#     fig1 = px.line(obt_df, x='quarter_start_date', y='unemployment_rate_value', title='Unemployment Rate')
#     st.plotly_chart(fig1, use_container_width=True)

#     fig2 = px.line(obt_df, x='quarter_start_date', y='consumer_sentiment_value', title='Consumer Sentiment')
#     st.plotly_chart(fig2, use_container_width=True)

#     fig3 = px.line(obt_df, x='quarter_start_date', y='real_income_value', title='Real Disposable Income')
#     st.plotly_chart(fig3, use_container_width=True)

#     fig4 = px.line(obt_df, x='quarter_start_date', y='fed_funds_rate_value', title='Federal Funds Rate')
#     st.plotly_chart(fig4, use_container_width=True)