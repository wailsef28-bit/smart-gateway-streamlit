import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Gateway AI", layout="wide")

# Load data
features = pd.read_csv("features.csv")
cnn = pd.read_csv("cnn_predictions.csv")
decisions = pd.read_csv("decisions.csv")

# Merge
df = features.iloc[:len(decisions)].copy()
df["collision_risk"] = cnn["cnn_collision_risk"]
df["decision"] = decisions["decision"].map({0: "WAIT", 1: "TRANSMIT"})

st.title(" Smart Gateway AI – Intelligent IoT Gateway")

# KPI
c1, c2, c3 = st.columns(3)
c1.metric("Avg RSSI", f"{df['rssi'].mean():.1f} dBm")
c2.metric("Avg SNR", f"{df['snr'].mean():.1f} dB")
c3.metric("Transmit Rate", f"{(df['decision']=='TRANSMIT').mean()*100:.1f}%")

st.divider()

# Plots
st.subheader(" Network Load")
st.plotly_chart(px.line(df, y="packet_rate"), use_container_width=True)

st.subheader(" Collision Risk (CNN)")
st.plotly_chart(px.line(df, y="collision_risk"), use_container_width=True)

st.subheader(" Gateway Decision")
st.plotly_chart(px.line(df, y=(df["decision"]=="TRANSMIT").astype(int)),
                use_container_width=True)
