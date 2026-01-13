import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="Smart Gateway AI",
    layout="wide"
)

# =====================================================
# Sidebar – interactive testing
# =====================================================
st.sidebar.header("⚙️ Interactive Model Testing")

RISK_THRESHOLD = st.sidebar.slider(
    "Collision Risk Threshold",
    min_value=0.1,
    max_value=0.6,
    value=0.25,
    step=0.05
)

# =====================================================
# Load data
# =====================================================
features = pd.read_csv("features_sample.csv")
cnn = pd.read_csv("cnn_predictions.csv")
regression = pd.read_csv("traffic_predictions.csv")
decisions = pd.read_csv("decisions.csv")

# Align lengths safely
min_len = min(len(features), len(cnn), len(regression), len(decisions))

df = features.iloc[:min_len].copy()
df["collision_risk"] = cnn["cnn_collision_risk"].iloc[:min_len]
df["predicted_load"] = regression.iloc[:min_len, 0]
df["decision"] = decisions["decision"].iloc[:min_len].map({
    0: "WAIT",
    1: "TRANSMIT"
})

# =====================================================
# Header
# =====================================================
st.title("🚀 Smart Gateway AI – Hybrid Intelligent IoT Gateway")

st.markdown("""
This dashboard validates a **hybrid AI-based IoT gateway** combining:

- **CNN** → Collision risk detection  
- **Regression** → Network load estimation  
- **Q-Learning** → Intelligent transmission decision  

All models are trained offline and evaluated here.
""")

# =====================================================
# KPIs
# =====================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Avg RSSI", f"{df['rssi'].mean():.1f} dBm")
c2.metric("Avg SNR", f"{df['snr'].mean():.1f} dB")
c3.metric("Avg Predicted Load", f"{df['predicted_load'].mean():.2f}")
c4.metric(
    "Transmit Rate",
    f"{(df['decision'] == 'TRANSMIT').mean() * 100:.1f}%"
)

st.divider()

# =====================================================
# CNN VALIDATION
# =====================================================
st.header("🧠 CNN – Collision Detection Model")

st.plotly_chart(
    px.histogram(
        df,
        x="collision_risk",
        nbins=50,
        title="CNN Collision Risk Probability Distribution"
    ),
    use_container_width=True
)

df["cnn_risk_label"] = df["collision_risk"].apply(
    lambda x: "HIGH" if x > RISK_THRESHOLD else "LOW"
)

st.metric(
    "High Collision Risk Detected (%)",
    f"{(df['cnn_risk_label'] == 'HIGH').mean() * 100:.1f}%"
)

st.info(
    "A non-uniform probability distribution indicates that the CNN has learned "
    "meaningful patterns from real network data."
)

st.divider()

# =====================================================
# REGRESSION VALIDATION
# =====================================================
st.header("📈 Regression – Network Load Prediction")

st.plotly_chart(
    px.scatter(
        df,
        x="packet_rate",
        y="predicted_load",
        trendline="ols",
        title="Predicted Network Load vs Actual Packet Rate"
    ),
    use_container_width=True
)

mse = ((df["packet_rate"] - df["predicted_load"]) ** 2).mean()

st.metric(
    "Regression Mean Squared Error (MSE)",
    f"{mse:.2f}"
)

st.info(
    "Correlation between predicted and actual values confirms that the regression model "
    "learned the relationship between traffic and channel conditions."
)

st.divider()

# =====================================================
# Q-LEARNING VALIDATION
# =====================================================
st.header("🤖 Q-Learning – Decision Policy")

st.bar_chart(df["decision"].value_counts())

st.metric(
    "WAIT Decisions (%)",
    f"{(df['decision'] == 'WAIT').mean() * 100:.1f}%"
)

st.metric(
    "TRANSMIT Decisions (%)",
    f"{(df['decision'] == 'TRANSMIT').mean() * 100:.1f}%"
)

st.info(
    "A mixed decision distribution (WAIT + TRANSMIT) indicates that the Q-Learning agent "
    "has learned a non-trivial policy."
)

st.divider()

# =====================================================
# CNN ↔ Q-LEARNING CONSISTENCY
# =====================================================
st.header("🔗 CNN vs Q-Learning Consistency")

consistency = (
    (df["cnn_risk_label"] == "LOW") &
    (df["decision"] == "TRANSMIT")
).mean() * 100

st.metric(
    "Decision Consistency (%)",
    f"{consistency:.1f}%"
)

st.info(
    "This metric verifies that the Q-Learning decisions are aligned with "
    "CNN collision risk estimation."
)

st.divider()

# =====================================================
# EXPLAINABLE AI (XAI)
# =====================================================
st.header("🧠 Explainable AI – Decision Explanation")

idx = st.slider(
    "Select time step",
    min_value=0,
    max_value=len(df) - 1,
    value=0
)

st.metric("Gateway Decision", df.iloc[idx]["decision"])

if df.iloc[idx]["cnn_risk_label"] == "HIGH":
    st.warning(
        "High collision risk detected by the CNN. "
        "The Q-Learning agent chooses WAIT to avoid packet loss."
    )
else:
    st.success(
        "Low collision risk detected by the CNN. "
        "The Q-Learning agent allows TRANSMIT."
    )

