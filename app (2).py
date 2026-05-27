
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

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain AI — Disruption Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🚚 Supply Chain Disruption Predictor & Auto-Healer")
st.markdown(
    "*Predicting supply chain failures using Prophet + LLaMA 3.3 70B*"
)

# ─────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────
@st.cache_data
def generate_data(scenario="None"):
    np.random.seed(42)

    dates = pd.date_range(
        end=datetime.now(),
        periods=730,
        freq='D'
    )

    data = pd.DataFrame({
        'ds': dates,
        'y': (
            np.random.normal(1000, 150, len(dates))
            + np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 200
            + np.random.choice([0] * 90 + [500] * 10, len(dates))
        ),
        'supplier_delay': np.random.choice(
            [0, 1],
            len(dates),
            p=[0.85, 0.15]
        ),
        'warehouse_stock': np.random.normal(5000, 500, len(dates)),
        'fuel_price': np.random.normal(95, 10, len(dates)),
    })

    # Recent disruptions
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
# DISRUPTION DETECTION
# ─────────────────────────────────────────
def detect_disruptions(forecast, threshold):
    forecast = forecast.copy()

    forecast['uncertainty_range'] = (
        forecast['yhat_upper'] - forecast['yhat_lower']
    )

    forecast['uncertainty_pct'] = (
        forecast['uncertainty_range']
        / forecast['yhat'].abs()
    ) * 100

    disruptions = forecast[
        (forecast['uncertainty_pct'] > threshold)
        |
        (forecast['yhat'] < forecast['yhat'].quantile(0.15))
    ].copy()

    disruptions['risk_level'] = disruptions['uncertainty_pct'].apply(
        lambda x:
        '🔴 CRITICAL' if x > 50
        else '🟡 HIGH' if x > 35
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
                {
                    "role": "system",
                    "content": "You are an expert supply chain analyst."
                },
                {
                    "role": "user",
                    "content": f"Analyze these headlines:\n\n{headlines}"
                }
            ],
            temperature=0.7,
            max_tokens=400
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
                {
                    "role": "system",
                    "content": "You are a supply chain optimization AI."
                },
                {
                    "role": "user",
                    "content": f"Generate a recovery plan:\n{disruption_info}"
                }
            ],
            temperature=0.6,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Auto-healer failed: {str(e)}"

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
if not run_button:

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("5–7", "Days Early Warning"),
        ("730", "Training Days"),
        ("AI", "Risk Engine"),
        ("LLaMA", "Powered")
    ]

    for col, (value, label) in zip(
        [col1, col2, col3, col4],
        metrics
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.info("Configure settings and click Run Full Analysis.")

else:

    if not groq_key:
        st.error("Please enter your Groq API key.")
        st.stop()

    with st.spinner("Training forecasting model..."):
        data, forecast = run_forecast(
            forecast_days,
            scenario
        )

        time.sleep(1)

    # FIXED: Proper future filtering
    future_forecast = forecast[
        forecast['ds'] > data['ds'].max()
    ]

    disruptions = detect_disruptions(
        future_forecast,
        risk_threshold
    )

    # Safe metrics
    avg_demand = int(future_forecast['yhat'].mean())
    peak_demand = int(future_forecast['yhat'].max())
    num_disruptions = len(disruptions)

    health_score = max(
        0,
        round(10 - (num_disruptions * 0.5), 1)
    )

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (f"{avg_demand:,}", "Avg Daily Demand"),
        (f"{peak_demand:,}", "Peak Demand"),
        (str(num_disruptions), "Risk Periods"),
        (f"{health_score}/10", "Health Score")
    ]

    for col, (value, label) in zip(
        [col1, col2, col3, col4],
        metrics
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ─────────────────────────────────────────
    # CHARTS
    # ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    # Forecast chart
    last_60 = forecast.tail(60)

    axes[0, 0].plot(
        last_60['ds'],
        last_60['yhat'],
        linewidth=2
    )

    axes[0, 0].fill_between(
        last_60['ds'],
        last_60['yhat_lower'],
        last_60['yhat_upper'],
        alpha=0.2
    )

    axes[0, 0].set_title("Demand Forecast")

    # Risk chart
    risk_scores = np.random.normal(5, 2, 30).clip(1, 10)

    axes[0, 1].bar(
        range(30),
        risk_scores
    )

    axes[0, 1].set_title("Risk Scores")

    # Warehouse stock
    axes[1, 0].plot(
        data['ds'].tail(90),
        data['warehouse_stock'].tail(90)
    )

    axes[1, 0].set_title("Warehouse Stock")

    # Supplier delays
    delay_rolling = (
        data['supplier_delay']
        .tail(90)
        .rolling(7)
        .mean()
        .fillna(0)
    )

    axes[1, 1].plot(
        data['ds'].tail(90),
        delay_rolling
    )

    axes[1, 1].fill_between(
        data['ds'].tail(90),
        delay_rolling,
        alpha=0.2
    )

    axes[1, 1].set_title("Supplier Delay Trend")

    plt.tight_layout()

    st.pyplot(fig)

    st.markdown("---")

    # ─────────────────────────────────────────
    # DISRUPTION ALERTS
    # ─────────────────────────────────────────
    st.subheader("🚨 Disruption Alerts")

    if disruptions.empty:
        st.success("No major disruptions detected.")
    else:
        for _, row in disruptions.head(5).iterrows():
            st.warning(
                f"{row['risk_level']} | "
                f"{row['ds'].strftime('%d %b %Y')} | "
                f"Demand: {int(row['yhat'])}"
            )

    st.markdown("---")

    # ─────────────────────────────────────────
    # NEWS ANALYSIS
    # ─────────────────────────────────────────
    st.subheader("📰 News Analysis")

    with st.spinner("Analyzing news..."):
        analysis = analyze_news(
            groq_key,
            custom_news
        )

        st.text_area(
            "AI News Analysis",
            analysis,
            height=250
        )

    st.markdown("---")

    # ─────────────────────────────────────────
    # AUTO HEALER
    # ─────────────────────────────────────────
    st.subheader("🔧 Recovery Plan")

    context = f"""
    Risk periods: {num_disruptions}
    Average demand: {avg_demand}
    Peak demand: {peak_demand}
    Scenario: {scenario}
    """

    with st.spinner("Generating recovery plan..."):
        plan = auto_healer(
            groq_key,
            context
        )

        st.text_area(
            "AI Recovery Plan",
            plan,
            height=300
        )

    st.markdown("---")

    st.caption(
        f"Analysis completed at "
        f"{datetime.now().strftime('%d %b %Y %H:%M:%S')}"
    )
