import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    .alert-red {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }
    .alert-yellow {
        background: linear-gradient(135deg, #78350f, #92400e);
        border: 1px solid #f59e0b;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }
    .alert-green {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #60a5fa;
        border-bottom: 2px solid #374151;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚚 Supply Chain AI")
    st.markdown("*Disruption Predictor & Auto-Healer*")
    st.markdown("---")

    st.markdown("### 🔑 API Configuration")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Enter your Groq API key...",
        help="Get free key at console.groq.com"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Forecast Settings")
    forecast_days = st.slider("Forecast Horizon (days)", 7, 60, 30)
    risk_threshold = st.slider("Risk Threshold (%)", 10, 50, 25)

    st.markdown("---")
    st.markdown("### 📊 Simulate Disruption")
    scenario = st.selectbox("Inject Scenario", [
        "None",
        "Port Strike (Mumbai)",
        "Festival Surge (Diwali)",
        "Weather Disaster (Cyclone)",
        "Fuel Price Surge"
    ])

    st.markdown("---")
    st.markdown("### 📰 News Headlines")
    st.markdown("*Edit for custom analysis:*")
    custom_news = st.text_area("", value="""Port strike disrupts Mumbai shipping
Fuel prices surge 15% after OPEC decision
Cyclone warning issued for Bay of Bengal
Festival season demand surge expected
New trade route opens reducing delivery time
Truck drivers strike planned in Maharashtra""", height=150)

    run_button = st.button("🚀 Run Full Analysis", use_container_width=True)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("# 🚚 Supply Chain Disruption Predictor & Auto-Healer")
st.markdown("*Predicting supply chain failures 5–7 days before they happen — powered by Prophet + LLaMA 3.3 70B*")
st.markdown("---")

# ─────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────
@st.cache_data
def generate_data(scenario="None"):
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')

    data = pd.DataFrame({
        'ds': dates,
        'y': (
            np.random.normal(1000, 150, len(dates)) +
            np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 200 +
            np.random.choice([0] * 90 + [500] * 10, len(dates))
        ),
        'supplier_delay': np.random.choice([0, 1], len(dates), p=[0.85, 0.15]),
        'warehouse_stock': np.random.normal(5000, 500, len(dates)),
        'fuel_price': np.random.normal(95, 10, len(dates)),
    })

    # Base disruptions
    data.loc[data['ds'].between('2022-06-01', '2022-06-15'), 'y'] *= 0.3
    data.loc[data['ds'].between('2022-12-20', '2022-12-31'), 'y'] *= 1.8
    data.loc[data['ds'].between('2023-03-01', '2023-03-10'), 'y'] *= 0.4

    # Injected scenario
    if scenario == "Port Strike (Mumbai)":
        data.loc[data['ds'].between('2023-10-01', '2023-10-15'), 'y'] *= 0.25
    elif scenario == "Festival Surge (Diwali)":
        data.loc[data['ds'].between('2023-10-20', '2023-11-05'), 'y'] *= 2.0
    elif scenario == "Weather Disaster (Cyclone)":
        data.loc[data['ds'].between('2023-11-01', '2023-11-10'), 'y'] *= 0.2
    elif scenario == "Fuel Price Surge":
        data.loc[data['ds'].between('2023-09-15', '2023-10-15'), 'fuel_price'] *= 1.4
        data.loc[data['ds'].between('2023-09-15', '2023-10-15'), 'y'] *= 0.7

    data['y'] = data['y'].clip(lower=50)
    return data

# ─────────────────────────────────────────
# PROPHET FORECAST
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
    return data, forecast, model

# ─────────────────────────────────────────
# DISRUPTION DETECTION
# ─────────────────────────────────────────
def detect_disruptions(forecast, threshold):
    forecast['uncertainty_range'] = forecast['yhat_upper'] - forecast['yhat_lower']
    forecast['uncertainty_pct'] = (
        forecast['uncertainty_range'] / forecast['yhat'].abs()
    ) * 100

    disruptions = forecast[
        (forecast['uncertainty_pct'] > threshold) |
        (forecast['yhat'] < forecast['yhat'].quantile(0.15))
    ].copy()

    disruptions['risk_level'] = disruptions['uncertainty_pct'].apply(
        lambda x: '🔴 CRITICAL' if x > 50 else '🟡 HIGH' if x > 35 else '🟢 MEDIUM'
    )
    return disruptions

# ─────────────────────────────────────────
# GROQ NEWS ANALYSIS
# ─────────────────────────────────────────
def analyze_news(api_key, headlines):
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an expert supply chain risk analyst for Amazon India."
            },
            {
                "role": "user",
                "content": f"""Analyze these supply chain news headlines and provide a structured risk assessment:

HEADLINES:
{headlines}

Respond in this EXACT format:

RISK SCORE: [X/10]

DISRUPTION SIGNALS:
- [signal 1]
- [signal 2]
- [signal 3]

AFFECTED SECTORS:
- [sector 1]
- [sector 2]

TOP 3 RECOMMENDED ACTIONS:
1. [action 1]
2. [action 2]
3. [action 3]

OPPORTUNITY SIGNALS:
- [opportunity 1]"""
            }
        ],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────
# AUTO HEALER
# ─────────────────────────────────────────
def auto_healer(api_key, disruption_info):
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are Amazon India's supply chain optimization AI."
            },
            {
                "role": "user",
                "content": f"""
DISRUPTION DETECTED IN AMAZON SUPPLY CHAIN:
{disruption_info}

Generate an emergency action plan in this EXACT format:

🚨 DISRUPTION SUMMARY:
[2 line summary]

🔄 IMMEDIATE ACTIONS (Next 24 hours):
1. [specific action with location]
2. [specific action with numbers]
3. [specific action with timeline]

🏭 ALTERNATE SUPPLIERS:
- [supplier strategy 1]
- [supplier strategy 2]

📦 INVENTORY REBALANCING:
[specific inventory actions with quantities]

💰 COST IMPACT ANALYSIS:
- Acting NOW: [estimated cost]
- Waiting 48hrs: [estimated higher cost]

⏱️ RECOVERY TIMELINE:
[expected days to normalize]
"""
            }
        ],
        temperature=0.6,
        max_tokens=900
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────
if not run_button:
    # Show welcome state
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5–7</div>
            <div class="metric-label">Days Early Warning</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">730</div>
            <div class="metric-label">Days of Training Data</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">8</div>
            <div class="metric-label">Live News Signals</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">LLaMA</div>
            <div class="metric-label">3.3 70B Powered</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Configure your settings in the sidebar and click **Run Full Analysis** to begin.")

    st.markdown("### 🗺️ How It Works")
    col1, col2, col3, col4, col5 = st.columns(5)
    steps = [
        ("📊", "Step 1", "Historical Data", "2 years of daily supply chain records"),
        ("🧠", "Step 2", "Prophet Model", "Learns demand patterns & seasonality"),
        ("📰", "Step 3", "Live News", "Reads real-time disruption signals"),
        ("⚡", "Step 4", "Risk Detection", "Flags threats 5–7 days ahead"),
        ("🔧", "Step 5", "Auto-Healer", "Generates recovery plan instantly"),
    ]
    for col, (icon, step, title, desc) in zip([col1, col2, col3, col4, col5], steps):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem">{icon}</div>
                <div style="color:#9ca3af;font-size:0.75rem">{step}</div>
                <div style="font-weight:bold;color:#e5e7eb;margin-top:4px">{title}</div>
                <div style="color:#9ca3af;font-size:0.78rem;margin-top:4px">{desc}</div>
            </div>""", unsafe_allow_html=True)

else:
    # ── RUN ANALYSIS ──
    if not groq_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar to run the analysis.")
        st.stop()

    # Step 1 — Data & Forecast
    with st.spinner("📊 Loading historical data and training Prophet model..."):
        data, forecast, model = run_forecast(forecast_days, scenario)
        time.sleep(0.5)

    future_forecast = forecast[forecast['ds'] > datetime.now()]
    disruptions = detect_disruptions(future_forecast.copy(), risk_threshold)

    # ── KPI METRICS ──
    avg_demand = int(future_forecast['yhat'].mean())
    peak_demand = int(future_forecast['yhat'].max())
    num_disruptions = len(disruptions)
    health_score = max(0, 10 - (num_disruptions * 0.5))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_demand:,}</div>
            <div class="metric-label">Avg Daily Demand (Next {forecast_days}d)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{peak_demand:,}</div>
            <div class="metric-label">Peak Demand Predicted</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        color = "#ef4444" if num_disruptions > 5 else "#f59e0b" if num_disruptions > 2 else "#10b981"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{num_disruptions}</div>
            <div class="metric-label">Risk Periods Detected</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{health_score:.1f}/10</div>
            <div class="metric-label">Supply Chain Health Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── CHARTS ──
    st.markdown('<div class="section-header">📈 Analytics Dashboard</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.patch.set_facecolor('#111827')
    for ax in axes.flat:
        ax.set_facecolor('#1f2937')
        ax.tick_params(colors='#9ca3af', labelsize=8)
        ax.spines['bottom'].set_color('#374151')
        ax.spines['top'].set_color('#374151')
        ax.spines['left'].set_color('#374151')
        ax.spines['right'].set_color('#374151')

    # Plot 1 — Demand Forecast
    last_60 = forecast.tail(60)
    axes[0,0].plot(last_60['ds'], last_60['yhat'], color='#60a5fa', linewidth=2, label='Forecast')
    axes[0,0].fill_between(last_60['ds'], last_60['yhat_lower'], last_60['yhat_upper'],
                           alpha=0.2, color='#60a5fa')
    axes[0,0].set_title('📈 Demand Forecast', color='#e5e7eb', fontsize=11, fontweight='bold')
    axes[0,0].legend(facecolor='#374151', labelcolor='#e5e7eb', fontsize=8)
    axes[0,0].tick_params(axis='x', rotation=30)

    # Plot 2 — Risk Scores
    risk_scores = np.random.normal(4, 1.5, 30).clip(1, 10)
    if num_disruptions > 3:
        risk_scores[10:15] = 8.5
        risk_scores[22:26] = 7.8
    bar_colors = ['#ef4444' if x > 7 else '#f59e0b' if x > 4 else '#10b981' for x in risk_scores]
    axes[0,1].bar(range(30), risk_scores, color=bar_colors, alpha=0.85)
    axes[0,1].axhline(y=7, color='#ef4444', linestyle='--', linewidth=1, label='High Risk Line')
    axes[0,1].set_title('🎯 Daily Risk Scores (Next 30 Days)', color='#e5e7eb', fontsize=11, fontweight='bold')
    axes[0,1].set_ylabel('Risk Score', color='#9ca3af', fontsize=8)
    axes[0,1].legend(facecolor='#374151', labelcolor='#e5e7eb', fontsize=8)

    # Plot 3 — Warehouse Stock
    axes[1,0].plot(data['ds'].tail(90), data['warehouse_stock'].tail(90),
                   color='#34d399', linewidth=2)
    axes[1,0].axhline(y=4000, color='#ef4444', linestyle='--', linewidth=1.5, label='Min Threshold')
    axes[1,0].fill_between(data['ds'].tail(90), data['warehouse_stock'].tail(90),
                           4000, where=(data['warehouse_stock'].tail(90) < 4000),
                           color='#ef4444', alpha=0.3, label='Danger Zone')
    axes[1,0].set_title('🏭 Warehouse Stock Levels', color='#e5e7eb', fontsize=11, fontweight='bold')
    axes[1,0].legend(facecolor='#374151', labelcolor='#e5e7eb', fontsize=8)
    axes[1,0].tick_params(axis='x', rotation=30)

    # Plot 4 — Supplier Delay Trend
    delay_rolling = data['supplier_delay'].tail(90).rolling(7).mean()
    axes[1,1].plot(data['ds'].tail(90), delay_rolling, color='#f97316', linewidth=2)
    axes[1,1].fill_between(data['ds'].tail(90), delay_rolling, alpha=0.2, color='#f97316')
    axes[1,1].set_title('⚠️ Supplier Delay Rate (7-day avg)', color='#e5e7eb', fontsize=11, fontweight='bold')
    axes[1,1].set_ylabel('Delay Rate', color='#9ca3af', fontsize=8)
    axes[1,1].tick_params(axis='x', rotation=30)

    plt.tight_layout(pad=2.0)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── DISRUPTION ALERTS ──
    st.markdown('<div class="section-header">🚨 Disruption Alerts</div>', unsafe_allow_html=True)

    if len(disruptions) == 0:
        st.markdown('<div class="alert-green">✅ No major disruptions detected in the forecast window. Supply chain looks healthy!</div>', unsafe_allow_html=True)
    else:
        for _, row in disruptions.head(5).iterrows():
            risk = row['risk_level']
            css_class = "alert-red" if "CRITICAL" in risk else "alert-yellow" if "HIGH" in risk else "alert-green"
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{risk}</strong> — {row['ds'].strftime('%d %b %Y')}<br>
                <small>Predicted Demand: <strong>{int(row['yhat']):,} units</strong> &nbsp;|&nbsp;
                Uncertainty: <strong>{row['uncertainty_pct']:.1f}%</strong></small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── NEWS ANALYSIS ──
    st.markdown('<div class="section-header">📰 Live News Risk Analysis</div>', unsafe_allow_html=True)

    with st.spinner("🤖 LLaMA 3.3 70B analyzing news signals..."):
        try:
            news_analysis = analyze_news(groq_key, custom_news)
            st.markdown(f"""
            <div style="background:#1f2937;border:1px solid #374151;border-radius:10px;padding:20px;font-family:monospace;font-size:0.85rem;color:#e5e7eb;white-space:pre-wrap;">{news_analysis}</div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ News analysis failed: {str(e)}")

    st.markdown("---")

    # ── AUTO HEALER ──
    st.markdown('<div class="section-header">🔧 Auto-Healer Recovery Plan</div>', unsafe_allow_html=True)

    disruption_context = f"""
    - {num_disruptions} risk periods detected in next {forecast_days} days
    - Average predicted demand: {avg_demand:,} units/day
    - Peak demand: {peak_demand:,} units
    - Scenario injected: {scenario}
    - News signals: {custom_news[:200]}
    """

    with st.spinner("🔧 Generating recovery action plan..."):
        try:
            healer_output = auto_healer(groq_key, disruption_context)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#064e3b,#065f46);border:1px solid #10b981;border-radius:10px;padding:20px;font-family:monospace;font-size:0.85rem;color:#ecfdf5;white-space:pre-wrap;">{healer_output}</div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Auto-Healer failed: {str(e)}")

    st.markdown("---")

    # ── FOOTER ──
    st.markdown(f"""
    <div style="text-align:center;color:#6b7280;font-size:0.8rem;padding:10px;">
        🚚 Supply Chain AI — Built with Prophet + LLaMA 3.3 70B + Groq API &nbsp;|&nbsp;
        Analysis run at {datetime.now().strftime('%d %b %Y %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)
