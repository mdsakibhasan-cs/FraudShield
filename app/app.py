import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "reports" / "fraud_risk_report.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_random_forest.joblib"

st.set_page_config(
    page_title="FraudShield",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_report():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Report not found: {REPORT_PATH}"
        )

    return pd.read_csv(REPORT_PATH)


df = load_report()


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.title("🛡️ FraudShield")
st.sidebar.caption(
    "Financial Fraud Detection & Risk Analytics Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Explainability",
        "Transaction Analysis",
        "Fraud Detection",
        "Anomaly Detection",
        "Risk Analysis",
        "Suspicious Transactions",
    ],
)


# -------------------------------------------------
# Header
# -------------------------------------------------

st.title("🛡️ FraudShield")
st.caption(
    "Financial Fraud Detection & Risk Analytics Platform"
)

st.divider()


# -------------------------------------------------
# Overview
# -------------------------------------------------

if page == "Overview":

    total = len(df)
    actual_fraud = int(df["Class"].sum())
    predicted_fraud = int(df["Predicted Fraud"].sum())
    anomalies = int(df["Is Anomaly"].sum())
    high_risk = int(
        (df["Risk Level"] == "High").sum()
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Test Transactions", f"{total:,}")
    col2.metric("Actual Fraud", f"{actual_fraud:,}")
    col3.metric("Predicted Fraud", f"{predicted_fraud:,}")
    col4.metric("Anomalies", f"{anomalies:,}")
    col5.metric("High Risk", f"{high_risk:,}")

    st.subheader("Risk Distribution")

    risk_counts = (
        df["Risk Level"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High"],
            fill_value=0
        )
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Level",
        "Transactions",
    ]

    fig = px.bar(
        risk_counts,
        x="Risk Level",
        y="Transactions",
        title="Transaction Risk Levels",
        text="Transactions",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Model Performance")

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

    metric_col1.metric("Accuracy", "99.96%")
    metric_col2.metric("Precision", "91.86%")
    metric_col3.metric("Recall", "82.29%")
    metric_col4.metric("F1 Score", "86.81%")
    metric_col5.metric("ROC-AUC", "0.9711")

    st.info(
        "Evaluation uses a duplicate-aware, leakage-free test split "
        "from the ULB Credit Card Fraud Detection dataset."
    )


# -------------------------------------------------
# Transaction Analysis
# -------------------------------------------------

elif page == "Transaction Analysis":

    st.header("📊 Transaction Analysis")

    col1, col2 = st.columns(2)

    with col1:
        amount_fig = px.histogram(
            df,
            x="Amount",
            nbins=50,
            title="Transaction Amount Distribution",
        )

        st.plotly_chart(
            amount_fig,
            use_container_width=True
        )

    with col2:
        fraud_amount = (
            df.groupby("Class")["Amount"]
            .mean()
            .reset_index()
        )

        fraud_amount["Class"] = fraud_amount[
            "Class"
        ].map({
            0: "Normal",
            1: "Fraud",
        })

        avg_fig = px.bar(
            fraud_amount,
            x="Class",
            y="Amount",
            title="Average Transaction Amount",
            text_auto=".2f",
        )

        st.plotly_chart(
            avg_fig,
            use_container_width=True
        )

    st.subheader("Transaction Statistics")

    stats = df["Amount"].describe().to_frame(
        "Value"
    )

    st.dataframe(
        stats,
        width="stretch"
    )


# -------------------------------------------------
# Fraud Detection
# -------------------------------------------------

elif page == "Fraud Detection":

    st.header("🚨 Fraud Detection")

    fraud_probability = df[
        "Fraud Probability"
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted Fraud",
        f"{int(df['Predicted Fraud'].sum()):,}"
    )

    col2.metric(
        "Fraud Probability ≥ 50%",
        f"{int((fraud_probability >= 0.5).sum()):,}"
    )

    col3.metric(
        "Maximum Fraud Probability",
        f"{fraud_probability.max():.2%}"
    )

    fig = px.histogram(
        df,
        x="Fraud Probability",
        nbins=50,
        title="Fraud Probability Distribution",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Highest Fraud Probability Transactions")

    fraud_table = (
        df[
            [
                "Time",
                "Amount",
                "Fraud Probability",
                "Predicted Fraud",
                "Risk Score",
                "Risk Level",
            ]
        ]
        .sort_values(
            "Fraud Probability",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        fraud_table,
        width="stretch"
    )


# -------------------------------------------------
# Anomaly Detection
# -------------------------------------------------

elif page == "Anomaly Detection":

    st.header("🔎 Anomaly Detection")

    anomalies = int(df["Is Anomaly"].sum())
    normal = len(df) - anomalies

    col1, col2 = st.columns(2)

    col1.metric(
        "Detected Anomalies",
        f"{anomalies:,}"
    )

    col2.metric(
        "Normal Transactions",
        f"{normal:,}"
    )

    anomaly_counts = pd.DataFrame({
        "Status": [
            "Normal",
            "Anomaly",
        ],
        "Transactions": [
            normal,
            anomalies,
        ],
    })

    fig = px.pie(
        anomaly_counts,
        names="Status",
        values="Transactions",
        title="Normal vs Anomalous Transactions",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    score_fig = px.histogram(
        df,
        x="Anomaly Score",
        nbins=50,
        title="Anomaly Score Distribution",
    )

    st.plotly_chart(
        score_fig,
        use_container_width=True
    )

    st.subheader("Detected Anomalies")

    anomaly_table = (
        df[df["Is Anomaly"]]
        [
            [
                "Time",
                "Amount",
                "Fraud Probability",
                "Anomaly Score",
                "Risk Score",
                "Risk Level",
            ]
        ]
        .sort_values(
            "Anomaly Score",
            ascending=False
        )
        .head(30)
    )

    st.dataframe(
        anomaly_table,
        width="stretch"
    )


# -------------------------------------------------
# Risk Analysis
# -------------------------------------------------

elif page == "Risk Analysis":

    st.header("⚠️ Risk Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Low Risk",
        f"{int((df['Risk Level'] == 'Low').sum()):,}"
    )

    col2.metric(
        "Medium Risk",
        f"{int((df['Risk Level'] == 'Medium').sum()):,}"
    )

    col3.metric(
        "High Risk",
        f"{int((df['Risk Level'] == 'High').sum()):,}"
    )

    fig = px.histogram(
        df,
        x="Risk Score",
        nbins=50,
        title="Risk Score Distribution",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Highest Risk Transactions")

    risk_table = (
        df[
            [
                "Time",
                "Amount",
                "Fraud Probability",
                "Anomaly Score",
                "Risk Score",
                "Risk Level",
                "Predicted Fraud",
                "Is Anomaly",
            ]
        ]
        .sort_values(
            "Risk Score",
            ascending=False
        )
        .head(30)
    )

    st.dataframe(
        risk_table,
        width="stretch"
    )


# -------------------------------------------------
# Explainability
# -------------------------------------------------

elif page == "Explainability":

    st.header("Model Explainability")

    st.write(
        "This page shows global Random Forest feature importance "
        "for the fraud classification model."
    )

    @st.cache_resource
    def load_model():
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )
        return joblib.load(MODEL_PATH)

    model = load_model()

    feature_names = model.feature_names_in_
    importances = model.named_steps[
        "classifier"
    ].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(
        "Importance",
        ascending=False
    )

    top_features = importance_df.head(15).sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Features Used by the Fraud Classifier",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Feature Importance Table")

    st.dataframe(
        importance_df.head(30),
        width="stretch",
    )

    st.info(
        "Feature importance indicates how strongly features contributed "
        "to the Random Forest model's predictions. It does not establish "
        "causation and is not a transaction-specific explanation."
    )


# -------------------------------------------------
# Suspicious Transactions
# -------------------------------------------------

elif page == "Suspicious Transactions":

    st.header("🚨 Suspicious Transactions")

    suspicious = df[
        (df["Predicted Fraud"] == 1)
        | (df["Is Anomaly"])
    ].copy()

    col1, col2 = st.columns(2)

    col1.metric(
        "Suspicious Transactions",
        f"{len(suspicious):,}"
    )

    col2.metric(
        "High Risk Suspicious",
        f"{int((suspicious['Risk Level'] == 'High').sum()):,}"
    )

    suspicious = suspicious.sort_values(
        "Risk Score",
        ascending=False
    )

    display_columns = [
        "Time",
        "Amount",
        "Fraud Probability",
        "Anomaly Score",
        "Risk Score",
        "Risk Level",
        "Predicted Fraud",
        "Is Anomaly",
    ]

    st.dataframe(
        suspicious[display_columns],
        width="stretch",
        height=600,
    )


