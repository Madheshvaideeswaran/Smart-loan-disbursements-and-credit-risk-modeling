# app.py - Smart & Optimized Loan Disbursements with Risk Scenario Modeling
# Engineered by Madhesh Vaideeswaran

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fpdf
import os
import warnings
import matplotlib.pyplot as plt
from io import BytesIO
warnings.filterwarnings("ignore")

# Install packages if not available
for pkg in ['catboost', 'statsmodels', 'plotly', 'matplotlib']:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

from statsmodels.tsa.statespace.sarimax import SARIMAX
from catboost import CatBoostRegressor

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Smart Loan Disbursements", layout="wide")

st.title("📊 Smart & Optimized Loan Disbursements with Risk Scenario Modeling")
st.markdown("Forecast future disbursements with **CatBoost + SARIMA Hybrid Model**")

# -------------------------------
# Load Data
# -------------------------------
uploaded_file = st.file_uploader("Upload Loan Disbursement Dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect date + disbursement column
    date_col = [c for c in df.columns if "date" in c][0]
    value_col = [c for c in df.columns if "disb" in c or "amount" in c][0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df.set_index(date_col, inplace=True)

    st.subheader("📌 Raw Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    train = df.iloc[:-12]
    test = df.iloc[-12:]

    # -------------------------------
    # Hybrid Model: CatBoost + SARIMA
    # -------------------------------
    # SARIMA
    sarima_model = SARIMAX(train[value_col], order=(1,1,1), seasonal_order=(1,1,1,12))
    sarima_fit = sarima_model.fit(disp=False)

    # CatBoost
    train["month"] = train.index.month
    train["year"] = train.index.year
    cb = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, verbose=0)
    cb.fit(train[["month","year"]], train[value_col])

    # Forecast horizon
    forecast_index = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS")

    # SARIMA forecast
    sarima_forecast = sarima_fit.get_forecast(steps=12)
    sarima_pred = sarima_forecast.predicted_mean
    baseline_ci = sarima_forecast.conf_int(alpha=0.05)

    # CatBoost forecast
    future_features = pd.DataFrame({"month": forecast_index.month, "year": forecast_index.year})
    cb_pred = cb.predict(future_features)

    # Hybrid forecast (average)
    hybrid_forecast = (sarima_pred.values + cb_pred) / 2

    # -------------------------------
    # Scenario Modeling
    # -------------------------------
    scenario_multiplier = st.slider("⚡ Scenario Stress Multiplier", 0.5, 1.5, 1.0, 0.05)
    scenario_forecast = hybrid_forecast * scenario_multiplier

    # Confidence intervals for scenario
    scenario_ci_upper = baseline_ci.iloc[:,1] * scenario_multiplier
    scenario_ci_lower = baseline_ci.iloc[:,0] * scenario_multiplier

    # -------------------------------
    # Auto-theme Detection
    # -------------------------------
    theme = st.get_option("theme.base")
    if theme == "dark":
        theme_color = "#ff6f61"  # softer red
        grid_color = "gray"
        bg_color = "black"
        font_color = "white"
    else:
        theme_color = "#e53935"  # bright red
        grid_color = "lightgray"
        bg_color = "white"
        font_color = "black"

    # -------------------------------
    # Visualization (Plotly)
    # -------------------------------
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Historical data
    fig.add_trace(go.Scatter(
        x=df.index, y=df[value_col], mode="lines+markers",
        name="Historical", line=dict(color="black")
    ))

    # Baseline forecast
    fig.add_trace(go.Scatter(
        x=forecast_index, y=hybrid_forecast,
        mode="lines+markers", name="Baseline Forecast", line=dict(color="blue", dash="dash")
    ))

    # Scenario forecast
    fig.add_trace(go.Scatter(
        x=forecast_index, y=scenario_forecast,
        mode="lines+markers", name="Scenario Forecast",
        line=dict(color=theme_color, dash="dot")
    ))

    # Confidence Intervals
    fig.add_trace(go.Scatter(
        x=forecast_index.to_list() + forecast_index[::-1].to_list(),
        y=baseline_ci.iloc[:,1].to_numpy().tolist() + baseline_ci.iloc[::-1,0].to_numpy().tolist(),
        fill='toself', fillcolor='#1e88e5', opacity=0.1,
        line=dict(color='rgba(0,0,0,0)'), name='Baseline CI', hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_index.to_list() + forecast_index[::-1].to_list(),
        y=scenario_ci_upper.to_numpy().tolist() + scenario_ci_lower.iloc[::-1].to_numpy().tolist(),
        fill='toself', fillcolor=theme_color, opacity=0.15,
        line=dict(color='rgba(0,0,0,0)'), name='Scenario CI', hoverinfo='skip'
    ))

    # Layout
    fig.update_layout(
        title=dict(
            text="12-Month Disbursement Forecast: Baseline vs Adjusted Scenario",
            font=dict(size=20, color=font_color)
        ),
        xaxis=dict(
            title=dict(text="Date", font=dict(size=16, color=font_color)),
            tickfont=dict(size=13, color=font_color),
            showgrid=True, gridcolor=grid_color, zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Disbursement Volume", font=dict(size=16, color=font_color)),
            tickfont=dict(size=13, color=font_color),
            showgrid=True, gridcolor=grid_color, zeroline=False
        ),
        legend=dict(font=dict(size=12, color=font_color)),
        plot_bgcolor=bg_color
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # Export Options
    # -------------------------------
    forecast_df = pd.DataFrame({
        "Date": forecast_index,
        "Baseline Forecast": hybrid_forecast,
        "Scenario Forecast": scenario_forecast,
        "Baseline Lower CI": baseline_ci.iloc[:,0].values,
        "Baseline Upper CI": baseline_ci.iloc[:,1].values,
        "Scenario Lower CI": scenario_ci_lower.values,
        "Scenario Upper CI": scenario_ci_upper.values
    })

    st.subheader("📥 Export Forecast Results")
    st.dataframe(forecast_df)

    # Export CSV
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📂 Download CSV", csv, "forecast_results.csv", "text/csv")

    # -------------------------------
    # Export PDF with Matplotlib Chart
    # -------------------------------
    def create_pdf(df):
        pdf = fpdf.FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Forecast Results", ln=True, align="C")

        # Table rows
        for i in range(len(df)):
            row = df.iloc[i].astype(str).tolist()
            pdf.cell(200, 10, " | ".join(row), ln=True)

        # Generate matplotlib chart (fallback for PDF)
        plt.figure(figsize=(8,4))
        plt.plot(df["Date"], df["Baseline Forecast"], label="Baseline", linestyle="--", color="blue")
        plt.plot(df["Date"], df["Scenario Forecast"], label="Scenario", linestyle=":", color="red")
        plt.fill_between(df["Date"], df["Baseline Lower CI"], df["Baseline Upper CI"], color="blue", alpha=0.1)
        plt.fill_between(df["Date"], df["Scenario Lower CI"], df["Scenario Upper CI"], color="red", alpha=0.1)
        plt.legend()
        plt.title("12-Month Disbursement Forecast")
        plt.xticks(rotation=45)
        plt.tight_layout()

        img_bytes = BytesIO()
        plt.savefig(img_bytes, format="PNG")
        plt.close()
        img_bytes.seek(0)

        pdf.image(img_bytes, x=10, y=pdf.get_y()+10, w=180)

        output = pdf.output(dest='S').encode('latin-1')
        return output

    pdf_bytes = create_pdf(forecast_df)
    st.download_button("📄 Download PDF", data=pdf_bytes, file_name="forecast_results.pdf", mime="application/pdf")
