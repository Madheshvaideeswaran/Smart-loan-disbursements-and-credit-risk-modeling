# app.py - Smart & Optimized Loan Disbursements with Risk Scenario Modeling
# Engineered by Madhesh Vaideeswaran

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fpdf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Install packages if not available
for pkg in ['catboost', 'statsmodels', 'plotly', 'fpdf2', 'openpyxl', 'matplotlib', 'seaborn']:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import catboost as cb
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =============================
# Streamlit Page Config
# =============================
st.set_page_config(
    page_title="Smart Loan Disbursement Risk Forecaster",
    layout="wide",
    page_icon="🏦"
)

# =============================
# Main App Title + Signature Below — WHITE HEADER, RED NAME, BIGGER
# =============================
st.markdown("""
    <h1 style='text-align: center; font-weight: 700; font-size: 2.5rem; color: #ffffff; margin-bottom: 0.2rem; text-shadow: 1px 1px 3px rgba(0,0,0,0.6);'>
    Smart & Optimized Loan Disbursements with Risk Scenario Modeling
    </h1>
    <p style='text-align: center; font-weight: 600; font-size: 1.3rem; color: #ffffff; margin-top: 0; margin-bottom: 0.2rem;'>
    Engineered by <span style='color: #d62828; font-size: 1.4rem;'>Madhesh Vaideeswaran</span>
    </p>
    <p style='text-align: center; color: #e0e0e0; margin-top: 0; font-size: 1.1rem;'>
    AI-driven risk classification and forecasting for strategic portfolio decisions
    </p>
    <hr style='margin: 1.5rem 0; border-color: #555;'>
""", unsafe_allow_html=True)

# =============================
# Load Data (cached)
# =============================
@st.cache_data
def load_data():
    df = pd.read_csv("sip.csv", header=None, skiprows=1)
    df.columns = ["SNo", "AccountID", "score"] + [f"month_{i+1}" for i in range(6)]
    return df

@st.cache_data
def load_disburse():
    df = pd.read_excel("DB_Skew.xlsx", sheet_name="Sheet1")
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').set_index('ds')
    return df['y']

df = load_data()
disburse_series = load_disburse()

# =============================
# Data Prep (UNCHANGED)
# =============================
score_col = "score"
id_col = "AccountID"
repay_cols = [f"month_{i+1}" for i in range(6)]

df_model = df.dropna(subset=[score_col]).copy()
df_model = df_model[[id_col, score_col] + repay_cols]

for c in repay_cols:
    df_model[c] = df_model[c].astype(str).str.strip()

def risk_bin(s):
    if s < 40:
        return "Low Risk"
    elif s <= 60:
        return "Medium Risk"
    else:
        return "High Risk"

df_model["risk_label"] = df_model[score_col].astype(float).apply(risk_bin)

mapping = {"OTR": 1, "NonOTR": 0, "Default": -1}
df_model[repay_cols] = df_model[repay_cols].replace(mapping)

X = df_model[repay_cols].copy()
for i in range(2, len(repay_cols)+1):
    X[f"rolling_mean_{i}"] = X.iloc[:, :i].mean(axis=1)
    X[f"rolling_std_{i}"] = X.iloc[:, :i].std(axis=1)
    X[f"rolling_sum_{i}"] = X.iloc[:, :i].sum(axis=1)

X["trend"] = X.diff(axis=1).mean(axis=1)
X["has_default"] = (X[repay_cols] == -1).any(axis=1).astype(int)
X["num_otr"] = (X[repay_cols] == 1).sum(axis=1)
X["num_nonotr"] = (X[repay_cols] == 0).sum(axis=1)
X["num_default"] = (X[repay_cols] == -1).sum(axis=1)
X["pct_default"] = X["num_default"] / len(repay_cols)
X["months_recorded"] = X[repay_cols].notna().sum(axis=1)

X = X.fillna(0)
y = df_model["risk_label"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# =============================
# Train CatBoost (cached) — UNCHANGED + "Smart Brain AI" branding
# =============================
@st.cache_resource
def train_catboost(X_train, y_train, X_test, y_test):
    class_counts = y_train.value_counts()
    total = sum(class_counts)
    weights = {cls: total / count for cls, count in class_counts.items()}
    weights["Low Risk"] *= 2.0
    weights["Medium Risk"] *= 1.5

    model = cb.CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        cat_features=[i for i in range(len(repay_cols))],
        class_weights=weights,
        random_seed=42,
        verbose=False
    )
    with st.spinner("🧠 Training Smart Brain AI (CatBoost Classifier)..."):
        model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    return model

model = train_catboost(X_train, y_train, X_test, y_test)
y_pred = model.predict(X_test).flatten()

# =============================
# Tabs for Outputs
# =============================
tab1, tab2, tab3 = st.tabs([
    "Probability of Default Score Classification (CatBoost AI Engine)",
    "Risk Scenario Configuration",
    "SARIMAX Forecast: Disbursement Volume Under Adjusted Risk Conditions"
])

# Dynamic color theming based on scenario (subtle)
def get_theme_color(risk_delta):
    if risk_delta > 0.15: return "#d62828"  # red
    elif risk_delta > 0.05: return "#f77f00" # orange
    elif risk_delta < -0.15: return "#06d6a0" # green
    elif risk_delta < -0.05: return "#8ac926" # light green
    else: return "#1e88e5" # neutral blue

# =============================
# Tab 1: Probability of Default Score Classification (CatBoost AI Engine)
# =============================
with tab1:
    st.header("Probability of Default Score Classification (CatBoost AI Engine)")
    st.write("""
        This module leverages the **CatBoost AI Engine** to classify borrowers into risk tiers based on 6-month repayment behavior. 
        The model is calibrated to prioritize detection of Medium and High Risk accounts to support proactive portfolio management.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Predicted Risk Tier Distribution")
        risk_counts = pd.Series(y_pred).value_counts()
        fig_pie = go.Figure(data=[go.Pie(labels=risk_counts.index, values=risk_counts.values, hole=0.3)])
        fig_pie.update_layout(title="AI-Classified Risk Allocation", title_x=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Model Confusion Matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred, labels=["Low Risk", "Medium Risk", "High Risk"])
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Low", "Medium", "High"], y=["Low", "Medium", "High"],
                                           colorscale='Blues', text=cm, texttemplate="%{text}"))
        fig_cm.update_layout(title="Classification Accuracy by Tier", title_x=0.5)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col3:
        from sklearn.metrics import classification_report, accuracy_score
        report = classification_report(y_test, y_pred, output_dict=True)
        acc = accuracy_score(y_test, y_pred)
        st.metric("Overall Model Accuracy", f"{acc:.2%}")
        st.metric("Macro F1-Score", f"{report['macro avg']['f1-score']:.2%}")
        st.caption("Smart Brain AI Engine: Optimized for early risk signal detection.")

# =============================
# Tab 2: Risk Scenario Configuration — RELATIVE PERCENTAGE CHANGE
# =============================
with tab2:
    st.header("Risk Scenario Configuration")
    st.write("""
        Adjust the proportion of High Risk borrowers using a relative percentage change.
        The system will propagate this change through the disbursement forecast using sensitivity modeling.
    """)

    col1, col2 = st.columns(2)
    with col1:
        direction = st.radio("Adjustment Direction", ["Increase", "Decrease"], horizontal=True)
    with col2:
        pct_points = st.number_input(
            "Relative Change (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            help="Enter the relative percentage change to apply to the High Risk population (e.g., 10 means 10% increase or decrease from baseline)."
        )

    high_risk_pct = (y_pred == "High Risk").mean()
    if direction == "Increase":
        scenario_high_risk_pct = high_risk_pct * (1 + pct_points / 100)
    else:
        scenario_high_risk_pct = high_risk_pct * (1 - pct_points / 100)
    scenario_high_risk_pct = max(0.0, min(1.0, scenario_high_risk_pct))
    delta_risk = scenario_high_risk_pct - high_risk_pct

    theme_color = get_theme_color(delta_risk)

    st.markdown(f"""
    <div style='padding: 1rem; border-radius: 8px; background-color: {theme_color}10; border-left: 4px solid {theme_color}; margin: 1rem 0;'>
        <strong>Scenario Summary:</strong> High Risk population adjusted to <strong>{scenario_high_risk_pct:.1%}</strong> 
        ({direction} of <strong>{pct_points}%</strong> from baseline {high_risk_pct:.1%})
    </div>
    """, unsafe_allow_html=True)

# =============================
# Tab 3: SARIMAX Forecast — PLOTLY FOR UI, MATPLOTLIB FOR PDF
# =============================
with tab3:
    st.header("SARIMAX Forecast: Disbursement Volume Under Adjusted Risk Conditions")
    st.write("""
        This forecast projects 12-month disbursement volume using a **seasonal ARIMA (SARIMAX)** model, 
        adjusted by the configured risk scenario. Sensitivity factor: 0.8 (i.e., 1% change in risk → 0.8% change in disbursement).
    """)

    # SARIMAX fitting (cached) — UNCHANGED
    @st.cache_resource
    def fit_sarimax(series):
        sarimax_model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        with st.spinner("⏳ Generating SARIMAX forecast..."):
            return sarimax_model.fit(disp=False)

    fitted_sarimax = fit_sarimax(disburse_series)
    forecast_steps = 12
    last_date = disburse_series.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
    baseline_forecast = fitted_sarimax.get_forecast(steps=forecast_steps)
    baseline_mean = baseline_forecast.predicted_mean
    baseline_ci = baseline_forecast.conf_int(alpha=0.05)

    # Adjustment factor — UNCHANGED LOGIC
    SENSITIVITY = 0.8
    delta_risk_pct = scenario_high_risk_pct - high_risk_pct
    adjustment_pct = -delta_risk_pct * SENSITIVITY
    adjustment_factor = 1 + adjustment_pct

    scenario_mean = baseline_mean * adjustment_factor
    scenario_ci_lower = baseline_ci.iloc[:,0] * adjustment_factor
    scenario_ci_upper = baseline_ci.iloc[:,1] * adjustment_factor

    # === Interactive Plotly Chart — FOR DISPLAY ONLY ===
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=disburse_series.index,
        y=disburse_series.values,
        mode='lines',
        name='Historical',
        line=dict(color='#424242', width=2),
        hovertemplate='<b>Date</b>: %{x}<br><b>Disbursement</b>: %{y:,.0f}<extra></extra>'
    ))

    # Baseline Forecast
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=baseline_mean,
        mode='lines+markers',
        name='Baseline Forecast',
        line=dict(color='#1e88e5', width=2.5),
        marker=dict(size=6),
        hovertemplate='<b>Date</b>: %{x}<br><b>Baseline</b>: %{y:,.0f}<extra></extra>'
    ))

    # Scenario Forecast
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=scenario_mean,
        mode='lines+markers',
        name=f'Adjusted Forecast ({direction} {pct_points}%)',
        line=dict(color=theme_color, width=2.5, dash='dash'),
        marker=dict(size=6),
        hovertemplate='<b>Date</b>: %{x}<br><b>Scenario</b>: %{y:,.0f}<extra></extra>'
    ))

    # Confidence Intervals
    fig.add_trace(go.Scatter(
        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
        y=baseline_ci.iloc[:,1].tolist() + baseline_ci.iloc[:,0][::-1].tolist(),
        fill='toself',
        fillcolor='#1e88e5',
        opacity=0.1,
        line=dict(color='rgba(0,0,0,0)'),
        name='Baseline CI',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
        y=scenario_ci_upper.tolist() + scenario_ci_lower[::-1].tolist(),
        fill='toself',
        fillcolor=theme_color,
        opacity=0.15,
        line=dict(color='rgba(0,0,0,0)'),
        name='Scenario CI',
        hoverinfo='skip'
    ))

    # ✅ 100% SAFE LAYOUT — NO DEPRECATED PROPERTIES
    fig.update_layout(
        title="12-Month Disbursement Forecast: Baseline vs Adjusted Scenario",
        title_font_size=20,
        title_font_color="black",
        
        xaxis_title="Date",
        xaxis_title_font_size=16,
        xaxis_title_font_color="black",
        xaxis_tickfont_size=13,
        xaxis_tickfont_color="black",
        xaxis_showgrid=True,
        xaxis_gridcolor="lightgray",
        xaxis_zeroline=False,
        
        yaxis_title="Disbursement Volume",
        yaxis_title_font_size=16,
        yaxis_title_font_color="black",
        yaxis_tickfont_size=13,
        yaxis_tickfont_color="black",
        yaxis_showgrid=True,
        yaxis_gridcolor="lightgray",
        yaxis_zeroline=False,
        
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255,255,255,0.8)',
            font_size=13,
            font_color="black"
        ),
        
        hovermode='x unified',
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='black',
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # === Impact Summary ===
    baseline_total = baseline_mean.sum()
    scenario_total = scenario_mean.sum()
    difference = scenario_total - baseline_total
    pct_change = (difference / baseline_total) * 100

    st.subheader("Forecast Impact Summary")
    impact_df = pd.DataFrame({
        "Metric": ["Total Forecasted Disbursement (Year 6)", "Variance vs Baseline", "Percentage Change"],
        "Baseline": [f"{baseline_total:,.0f}", "-", "-"],
        "Adjusted Scenario": [f"{scenario_total:,.0f}", f"{difference:+,.0f}", f"{pct_change:+.2f}%"]
    })
    st.table(impact_df)

    # === Prepare Forecast Data for CSV ===
    forecast_df = pd.DataFrame({
        "Date": forecast_index,
        "Baseline_Forecast": baseline_mean,
        "Scenario_Forecast": scenario_mean,
        "Baseline_Lower_CI": baseline_ci.iloc[:,0],
        "Baseline_Upper_CI": baseline_ci.iloc[:,1],
        "Scenario_Lower_CI": scenario_ci_lower,
        "Scenario_Upper_CI": scenario_ci_upper
    })

    # === PDF Export — USE MATPLOTLIB FOR RELIABLE CHARTS ===
    st.subheader("Export Reports")

    def create_pdf():
        pdf = fpdf.FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", 'B', 16)
        pdf.set_text_color(0, 0, 0)

        # Title
        pdf.cell(0, 10, "Smart Loan Disbursement Risk Forecast", ln=True, align='C')
        pdf.ln(5)

        # Subtitle
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, "Engineered by Madhesh Vaideeswaran", ln=True, align='C')
        pdf.ln(10)

        # Scenario Summary
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Scenario Summary:", ln=True)
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, f"High Risk population adjusted to {scenario_high_risk_pct:.1%} ({direction} of {pct_points}% from baseline {high_risk_pct:.1%})", ln=True)
        pdf.ln(10)

        # === Generate Matplotlib Chart for Forecast ===
        plt.figure(figsize=(10, 5))
        plt.plot(disburse_series.index, disburse_series.values, label='Historical', color='#424242', linewidth=2)
        plt.plot(forecast_index, baseline_mean, label='Baseline Forecast', color='#1e88e5', linewidth=2.5, marker='o')
        plt.plot(forecast_index, scenario_mean, label=f'Adjusted Forecast ({direction} {pct_points}%)', color=theme_color, linestyle='--', linewidth=2.5, marker='s')

        plt.fill_between(forecast_index, baseline_ci.iloc[:,0], baseline_ci.iloc[:,1], color='#1e88e5', alpha=0.1)
        plt.fill_between(forecast_index, scenario_ci_lower, scenario_ci_upper, color=theme_color, alpha=0.15)

        plt.title("12-Month Disbursement Forecast", fontsize=14, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Disbursement Volume", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save and embed
        plt.savefig("temp_forecast.png", dpi=150, bbox_inches='tight')
        plt.close()

        pdf.image("temp_forecast.png", x=10, y=pdf.get_y(), w=180, h=90)
        pdf.ln(10)

        # Check if we need a new page
        if pdf.get_y() > 240:
            pdf.add_page()

        # === Generate Matplotlib Pie Chart ===
        plt.figure(figsize=(6, 4))
        risk_counts = pd.Series(y_pred).value_counts()
        plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90, colors=['#06d6a0', '#8ac926', '#d62828'])
        plt.title("AI-Classified Risk Allocation", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("temp_pie.png", dpi=150, bbox_inches='tight')
        plt.close()

        pdf.image("temp_pie.png", x=10, y=pdf.get_y(), w=180, h=70)
        pdf.ln(10)

        # Check if we need a new page
        if pdf.get_y() > 240:
            pdf.add_page()

        # === Generate Matplotlib Confusion Matrix ===
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred, labels=["Low Risk", "Medium Risk", "High Risk"])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
        plt.title("Classification Accuracy by Tier", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("temp_cm.png", dpi=150, bbox_inches='tight')
        plt.close()

        pdf.image("temp_cm.png", x=10, y=pdf.get_y(), w=180, h=70)
        pdf.ln(10)

        # Check if we need a new page
        if pdf.get_y() > 240:
            pdf.add_page()

        # === Add Forecast Impact Summary Table ===
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Forecast Impact Summary", ln=True)
        pdf.set_font("Arial", '', 10)
        for idx, row in impact_df.iterrows():
            pdf.cell(60, 10, str(row["Metric"]), border=1)
            pdf.cell(60, 10, str(row["Baseline"]), border=1)
            pdf.cell(60, 10, str(row["Adjusted Scenario"]), border=1)
            pdf.ln(10)

        # Cleanup temp files
        for fname in ["temp_forecast.png", "temp_pie.png", "temp_cm.png"]:
            if os.path.exists(fname):
                os.remove(fname)

        # Ensure bytes
        output = pdf.output(dest='S')
        if isinstance(output, bytearray):
            output = bytes(output)
        return output

    pdf_bytes = create_pdf()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Report (PDF)",
            data=pdf_bytes,
            file_name='disbursement_forecast_report.pdf',
            mime='application/pdf',
            use_container_width=True
        )
    with col2:
        csv_bytes = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast Data (CSV)",
            data=csv_bytes,
            file_name='disbursement_forecast.csv',
            mime='text/csv',
            use_container_width=True
        )

# =============================
# Footer with Bold Signature
# =============================
st.divider()
footer_col1, footer_col2 = st.columns([2, 1])
with footer_col1:
    st.caption("Model logic and calculations preserved. Optimized for performance with caching.")
with footer_col2:
    st.markdown("""
        <div style='text-align: right; font-weight: 600; font-size: 1.1rem; color: #2c3e50; padding: 0.5rem 0;'>
        Engineered by <span style='color: #d62828;'>Madhesh Vaideeswaran</span>
        </div>
    """, unsafe_allow_html=True)