import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Gateway AI", layout="wide")

st.title("🚀 Smart Gateway AI Dashboard")

# Load data
df = pd.read_csv("features.csv")

st.subheader("📡 Network Load")
st.plotly_chart(px.line(df, y="packet_rate"), use_container_width=True)

st.subheader("📶 Signal Quality")
st.plotly_chart(px.line(df, y=["rssi", "snr"]), use_container_width=True)

st.subheader("⚠️ Interference Indicator")
st.plotly_chart(px.line(df, y="crc_error_rate"), use_container_width=True)
