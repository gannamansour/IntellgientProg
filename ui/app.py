"""
ui/app.py
Streamlit web app — Heart Disease Detection System.
Combines the rule-based expert system and the Decision Tree model.

Run with:  streamlit run ui/app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib

from rule_based_system.expert_system import assess_patient

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 Heart Disease Detection System")
st.markdown("---")

# ── Sidebar – user inputs ────────────────────────────────────────────────
st.sidebar.header("Patient Data Input")

age      = st.sidebar.slider("Age",            20, 80, 50)
sex      = st.sidebar.selectbox("Sex",         ["Male (1)", "Female (0)"])
cp       = st.sidebar.selectbox("Chest Pain Type (0=none, 1-3=type)", [0,1,2,3])
trestbps = st.sidebar.slider("Resting BP (mmHg)", 80, 200, 130)
chol     = st.sidebar.slider("Cholesterol (mg/dL)", 100, 400, 220)
fbs      = st.sidebar.selectbox("Fasting Blood Sugar >120 mg/dL", [0, 1])
restecg  = st.sidebar.selectbox("Resting ECG (0/1/2)", [0, 1, 2])
thalach  = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exang    = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak  = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope    = st.sidebar.selectbox("Slope of ST segment (0/1/2)", [0, 1, 2])
ca       = st.sidebar.selectbox("# Major Vessels (0-3)", [0, 1, 2, 3])
thal     = st.sidebar.selectbox("Thal (1=normal, 2=fixed, 3=reversable)", [1, 2, 3])

sex_val = 1 if sex.startswith("Male") else 0

# ── Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Data Insights", "📋 Comparison Report"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 – Prediction
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("Risk Assessment")

    col1, col2 = st.columns(2)

    # ── Expert System ──────────────────────────────────────────
    with col1:
        st.subheader("Rule-Based Expert System")
        patient_raw = dict(age=age, chol=chol, trestbps=trestbps,
                           thalach=thalach, oldpeak=oldpeak,
                           cp=cp, exang=exang, fbs=fbs, ca=ca, thal=thal)
        result_es = assess_patient(patient_raw)

        colour = {"LOW RISK":"🟢","MODERATE RISK":"🟡","HIGH RISK":"🔴"}
        st.metric("Verdict", f"{colour[result_es['verdict']]} {result_es['verdict']}")
        st.metric("Rules Fired", result_es["risk_score"])

        if result_es["rules_fired"]:
            st.write("**Triggered rules:**")
            for r in result_es["rules_fired"]:
                st.write(f"  • {r}")
        else:
            st.write("No risk rules triggered.")

    # ── Decision Tree ──────────────────────────────────────────
    with col2:
        st.subheader("Decision Tree Model")
        MODEL_PATH = "ml_model/decision_tree_model.pkl"

        if not os.path.exists(MODEL_PATH):
            st.warning("Model not trained yet. Run `python ml_model/train_model.py` first.")
        else:
            model = joblib.load(MODEL_PATH)

            # Normalise the continuous inputs the same way the cleaner did
            raw_df = pd.read_csv("data/raw_data.csv")
            scaler = MinMaxScaler()
            num_cols = ["age","trestbps","chol","thalach","oldpeak"]
            scaler.fit(raw_df[num_cols])

            patient_norm = {
                "age":      scaler.transform([[age,0,0,0,0]])[0][0],
                "sex":      sex_val,
                "cp":       cp,
                "trestbps": scaler.transform([[0,trestbps,0,0,0]])[0][1],
                "chol":     scaler.transform([[0,0,chol,0,0]])[0][2],
                "fbs":      fbs,
                "restecg":  restecg,
                "thalach":  scaler.transform([[0,0,0,thalach,0]])[0][3],
                "exang":    exang,
                "oldpeak":  scaler.transform([[0,0,0,0,oldpeak]])[0][4],
                "slope":    slope,
                "ca":       ca,
                "thal":     thal,
            }

            df_in = pd.DataFrame([patient_norm])
            pred  = model.predict(df_in)[0]
            prob  = model.predict_proba(df_in)[0][pred]
            label = "🔴 Heart Disease Detected" if pred == 1 else "🟢 No Heart Disease"

            st.metric("Prediction", label)
            st.metric("Model Confidence", f"{prob:.1%}")

            # Mini bar chart for probabilities
            probs = model.predict_proba(df_in)[0]
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.barh(["No Disease", "Disease"], probs,
                    color=["#2ecc71", "#e74c3c"])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)
            plt.close()

# ═══════════════════════════════════════════════════════════════
# TAB 2 – Data Insights
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("Dataset Insights")

    try:
        df = pd.read_csv("data/cleaned_data.csv")
    except FileNotFoundError:
        st.warning("Run `python utils/data_processing.py` to generate cleaned_data.csv first.")
        df = pd.read_csv("data/raw_data.csv")

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().round(3))

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df.corr(), annot=True, fmt=".1f", cmap="coolwarm",
                    ax=ax, linewidths=0.5)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        df["target"].value_counts().plot.bar(
            ax=ax, color=["#2ecc71", "#e74c3c"],
            edgecolor="white"
        )
        ax.set_xticklabels(["No Disease", "Disease"], rotation=0)
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    st.subheader("Feature Distributions")
    feat = st.selectbox("Select feature", df.columns.drop("target"))
    fig, ax = plt.subplots(figsize=(6, 3))
    df[feat].hist(ax=ax, bins=20, color="#3498db", edgecolor="white")
    ax.set_title(feat)
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════
# TAB 3 – Comparison Report
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("Expert System vs Decision Tree — Comparison")

    try:
        with open("reports/accuracy_comparison.md") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.info("Train the model first (`python ml_model/train_model.py`) to generate the report.")
