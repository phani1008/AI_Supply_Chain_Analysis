import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from groq import Groq
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Supply Chain AI — Disruption Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #60a5fa; }
    .metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🚚 Supply Chain AI")
    st.markdown("*Disruption Predictor & Auto-Healer*")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Enter your Groq API key..."
    )

    forecast_days = st.slider("Forecast Horizon (days)", 7, 60, 30)
    risk_threshold = st.slider("Risk Threshold (%)", 10, 50, 25)

    scenario = st.selectbox("Inject Scenario", [
        "None",
        "Port Strike (Mumbai)",
        "Festival Surge (Diwali)",
        "Weather Disaster (Cyclone)",
        "Fuel Price Surge"
    ])

    custom_news = st.text_area(
        "News Headlines",
        value="""Port strike disrupts Mumbai shipping
Fuel prices surge 15% after OPEC decision
Cyclone warning issued for Bay of Bengal
Festival season demand surge expected"""
    )

    run_button = st.button("🚀 Run Full Analysis", use_container_width=True)

st.title("🚚 Supply Chain Disruption Predictor & Auto-Healer")
st.markdown("*Predicting supply chain failures using Prophet + LLaMA 3.3 70B*")

# ─────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────
@st.cache_data
def generate_data(scenario="None"):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=730, freq='D')

    data = pd.DataFrame({
        'ds': dates,
        'y': (
            np.random.normal(1000, 150, len(dates))
            + np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 200
            + np.random.choice([0] * 90 + [500] * 10, len(dates))
        ),
        'supplier_delay': np.random.choice([0, 1], len(dates), p=[0.85, 0.15]),
        'warehouse_stock': np.random.normal(5000, 500, len(dates)),
        'fuel_price': np.random.normal(95, 10, len(dates)),
    })

    recent_start = dates[-120]

    if scenario == "Port Strike (Mumbai)":
        mask = data['ds'].between(recent_start, recent_start + timedelta(days=15))
        data.loc[mask, 'y'] *= 0.25
    elif scenario == "Festival Surge (Diwali)":
        mask = data['ds'].between(recent_start, recent_start + timedelta(days=20))
        data.loc[mask, 'y'] *= 2.0
    elif scenario == "Weather Disaster (Cyclone)":
        mask = data['ds'].between(recent_start, recent_start + timedelta(days=10))
        data.loc[mask, 'y'] *= 0.2
    elif scenario == "Fuel Price Surge":
        mask = data['ds'].between(recent_start, recent_start + timedelta(days=30))
        data.loc[mask, 'fuel_price'] *= 1.4
        data.loc[mask, 'y'] *= 0.7

    data['y'] = data['y'].clip(lower=50)
    return data

# ─────────────────────────────────────────
# FORECAST
# ─────────────────────────────────────────
@st.cache_data
def run_forecast(forecast_days, scenario):
    data = generate_data(scenario)
    model = Prophet(
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(data[['ds', 'y']])
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return data, forecast

# ─────────────────────────────────────────
# FIX 1: DISRUPTION DETECTION
# Use FULL forecast quantile baseline, not just future slice
# ─────────────────────────────────────────
def detect_disruptions(future_forecast, full_forecast, threshold):
    df = future_forecast.copy()

    df['uncertainty_range'] = df['yhat_upper'] - df['yhat_lower']
    df['uncertainty_pct'] = (
        df['uncertainty_range'] / df['yhat'].abs()
    ) * 100

    # FIX: Use full forecast for baseline quantile
    low_demand_threshold = full_forecast['yhat'].quantile(0.15)

    # FIX: Only flag if BOTH conditions are extreme
    # uncertainty > threshold AND demand is critically low
    disruptions = df[
        (df['uncertainty_pct'] > threshold) &
        (df['yhat'] < low_demand_threshold)
    ].copy()

    disruptions['risk_level'] = disruptions['uncertainty_pct'].apply(
        lambda x:
        '🔴 CRITICAL' if x > 60
        else '🟡 HIGH' if x > 40
        else '🟢 MEDIUM'
    )

    return disruptions

# ─────────────────────────────────────────
# GROQ FUNCTIONS
# ─────────────────────────────────────────
def analyze_news(api_key, headlines):
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert supply chain analyst."},
                {"role": "user", "content": f"""Analyze these supply chain news headlines and provide:
RISK SCORE: [X/10]
KEY DISRUPTION SIGNALS: [list]
RECOMMENDED ACTIONS: [list]

Headlines:
{headlines}"""}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"News analysis failed: {str(e)}"

def auto_healer(api_key, disruption_info):
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a supply chain optimization AI for Amazon India."},
                {"role": "user", "content": f"""Generate an emergency recovery plan for this supply chain situation:

{disruption_info}

Include:
- Immediate actions (next 24 hours)
- Alternate supplier strategies
- Inventory rebalancing steps
- Cost impact estimate"""}
            ],
            temperature=0.6,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Auto-healer failed: {str(e)}"

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
if not run_button:
    col1, col2, col3, col4 = st.columns(4)
    for col, (value, label) in zip(
        [col1, col2, col3, col4],
        [("5–7","Days Early Warning"),("730","Training Days"),("AI","Risk Engine"),("LLaMA","Powered")]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)
    st.info("👈 Configure settings in the sidebar and click Run Full Analysis.")

else:
    if not groq_key:
        st.error("Please enter your Groq API key.")
        st.stop()

    with st.spinner("Training forecasting model..."):
        data, forecast = run_forecast(forecast_days, scenario)
        time.sleep(1)

    # Future forecast slice
    future_forecast = forecast[forecast['ds'] > data['ds'].max()].copy()

    # FIX 1 applied — pass full forecast for baseline
    disruptions = detect_disruptions(future_forecast, forecast, risk_threshold)

    avg_demand   = int(future_forecast['yhat'].mean())
    peak_demand  = int(future_forecast['yhat'].max())
    num_disruptions = len(disruptions)

    # FIX 2: Health score — cap disruptions to max 10 for scoring
    capped = min(num_disruptions, 10)
    health_score = round(10 - capped, 1)

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    risk_color = "#ef4444" if num_disruptions > 5 else "#f59e0b" if num_disruptions > 2 else "#10b981"
    health_color = "#ef4444" if health_score < 4 else "#f59e0b" if health_score < 7 else "#10b981"

    cards = [
        (f"{avg_demand:,}", "Avg Daily Demand", "#60a5fa"),
        (f"{peak_demand:,}", "Peak Demand", "#60a5fa"),
        (str(num_disruptions), "Risk Periods Detected", risk_color),
        (f"{health_score}/10", "Supply Chain Health", health_color),
    ]
    for col, (value, label, color) in zip([col1,col2,col3,col4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ─────────────────────────────────────────
    # CHARTS
    # ─────────────────────────────────────────
    st.subheader("📊 Analytics Dashboard")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.patch.set_facecolor('#111827')
    for ax in axes.flat:
        ax.set_facecolor('#1f2937')
        ax.tick_params(colors='#9ca3af', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#374151')

    # Chart 1 — Forecast
    last_60 = forecast.tail(60 + forecast_days)
    axes[0,0].plot(last_60['ds'], last_60['yhat'], color='#60a5fa', linewidth=2, label='Forecast')
    axes[0,0].fill_between(last_60['ds'], last_60['yhat_lower'], last_60['yhat_upper'],
                           alpha=0.2, color='#60a5fa', label='Confidence')
    axes[0,0].set_title('📈 Demand Forecast', color='#e5e7eb', fontsize=11, fontweight='bold')
    axes[0,0].legend(facecolor='#374151', labelcolor='#e5e7eb', fontsize=8)
    axes[0,0].tick_params(axis='x', rotation=30)

    # Chart 2 — Risk scores (based on actual uncertainty)
    risk_vals = future_forecast['uncertainty_pct'].values[:30] if len(future_forecast) >= 30 \
        else np.pad(future_forecast['uncertainty_pct'].values, (0, 30-len(future_forecast)), constant_values=15)
    risk_vals = np.clip(risk_vals / 10, 1, 10)
    bar_colors = ['#ef4444' if x > 7 else '#f59e0b' if x > 4 else '#10b981' for x in risk_vals]
    axes[0,1].bar(range(len(risk_vals)), risk_vals, color=bar_colors, alpha=0.85)
    axes[0,1].axhline(y=7, color='#ef4444', linestyle='--', linewidth=1, label='High Risk Line')
    axes[0,1].set_title('🎯 Daily Risk Score (Next 30 Days)', color='#e5e7eb', fontsize=11, fontweight='bold')
    axes[0,1].set_ylabel('Score (1-10)', color='#9ca3af', fontsize=8)
    axes[0,1].legend(facecolor='#374151', labelcolor='#e5e7eb', fontsize=8)

    # Chart 3 — Warehouse stock
    stock_data = data.tail(90)
    axes[1,0].plot(stock_data['ds'], stock_data['warehouse_stock'], color='#34d399', linewidth=2)
    axes[1,0].axhline(y=4000, color='#ef4444', linestyle='--', linewidth=1.5, label='Min Threshold')
    axes[1,0].fill_between(stock_data['ds'], stock_data['warehouse_stock'], 4000,
                           where=(stock_data['warehouse_stock'] < 4000),
                           color='#ef4444', alpha=0.3, label='Danger Zone')
    axes[1,0].set_title('🏭 Warehouse Stock Levels', color='#e5e7eb', fontsize=11, fontweight='bold')
    axes[1,0].legend(facecolor='#374151', labelcolor='#e5e7eb', fontsize=8)
    axes[1,0].tick_params(axis='x', rotation=30)

    # Chart 4 — Supplier delay
    delay_data = data.tail(90)
    delay_rolling = delay_data['supplier_delay'].rolling(7).mean().fillna(0)
    axes[1,1].plot(delay_data['ds'], delay_rolling, color='#f97316', linewidth=2)
    axes[1,1].fill_between(delay_data['ds'], delay_rolling, alpha=0.2, color='#f97316')
    axes[1,1].set_title('⚠️ Supplier Delay Rate (7-day avg)', color='#e5e7eb', fontsize=11, fontweight='bold')
    axes[1,1].set_ylabel('Delay Rate', color='#9ca3af', fontsize=8)
    axes[1,1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ─────────────────────────────────────────
    # DISRUPTION ALERTS
    # ─────────────────────────────────────────
    st.subheader("🚨 Disruption Alerts")

    if disruptions.empty:
        st.success("✅ No major disruptions detected in the forecast window. Supply chain looks healthy!")
    else:
        for _, row in disruptions.head(5).iterrows():
            level = row['risk_level']
            if 'CRITICAL' in level:
                st.error(f"{level} | {row['ds'].strftime('%d %b %Y')} | Predicted Demand: {int(row['yhat']):,} units | Uncertainty: {row['uncertainty_pct']:.1f}%")
            elif 'HIGH' in level:
                st.warning(f"{level} | {row['ds'].strftime('%d %b %Y')} | Predicted Demand: {int(row['yhat']):,} units | Uncertainty: {row['uncertainty_pct']:.1f}%")
            else:
                st.info(f"{level} | {row['ds'].strftime('%d %b %Y')} | Predicted Demand: {int(row['yhat']):,} units | Uncertainty: {row['uncertainty_pct']:.1f}%")

    st.markdown("---")

    # ─────────────────────────────────────────
    # NEWS ANALYSIS
    # ─────────────────────────────────────────
    st.subheader("📰 Live News Risk Analysis")
    with st.spinner("🤖 LLaMA 3.3 70B analyzing news..."):
        analysis = analyze_news(groq_key, custom_news)
        st.text_area("AI News Analysis", analysis, height=250)

    st.markdown("---")

    # ─────────────────────────────────────────
    # AUTO HEALER
    # ─────────────────────────────────────────
    st.subheader("🔧 Auto-Healer Recovery Plan")
    context = f"""
Supply Chain Status:
- Risk periods detected: {num_disruptions}
- Average demand next {forecast_days} days: {avg_demand:,} units
- Peak demand: {peak_demand:,} units
- Health score: {health_score}/10
- Active scenario: {scenario}
- News signals: {custom_news[:300]}
    """
    with st.spinner("🔧 Generating recovery plan..."):
        plan = auto_healer(groq_key, context)
        st.text_area("AI Recovery Plan", plan, height=300)

    st.markdown("---")
    st.caption(f"Analysis completed at {datetime.now().strftime('%d %b %Y %H:%M:%S')}")
