import numpy as np
import pandas as pd


def calculate_risk_scores(fraud_probability, anomaly_score):
    """Combine fraud probability and anomaly score into a 0-100 risk score."""
    fraud_probability = np.asarray(fraud_probability, dtype=float)
    anomaly_score = np.asarray(anomaly_score, dtype=float)

    if len(fraud_probability) != len(anomaly_score):
        raise ValueError("Input arrays must have the same length.")

    if len(fraud_probability) == 0:
        raise ValueError("Input arrays cannot be empty.")

    fraud_probability = np.clip(fraud_probability, 0, 1)

    min_score = anomaly_score.min()
    max_score = anomaly_score.max()

    if max_score == min_score:
        anomaly_component = np.zeros_like(anomaly_score)
    else:
        anomaly_component = (
            (anomaly_score - min_score)
            / (max_score - min_score)
        )

    risk_score = (
        fraud_probability * 70
        + anomaly_component * 30
    )

    return np.clip(risk_score, 0, 100)


def assign_risk_level(risk_scores):
    """Convert numerical risk scores into risk categories."""
    scores = np.asarray(risk_scores, dtype=float)

    levels = np.select(
        [
            scores >= 70,
            scores >= 40
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    return levels


def create_risk_report(df, fraud_probability, anomaly_score):
    """Create a transaction-level risk report."""
    if len(df) != len(fraud_probability):
        raise ValueError("Data and prediction lengths do not match.")

    risk_scores = calculate_risk_scores(
        fraud_probability,
        anomaly_score
    )

    report = df.copy()

    report["Fraud Probability"] = fraud_probability
    report["Anomaly Score"] = anomaly_score
    report["Risk Score"] = risk_scores
    report["Risk Level"] = assign_risk_level(risk_scores)

    return report


if __name__ == "__main__":
    test_fraud_probability = np.array([0.01, 0.50, 0.95])
    test_anomaly_score = np.array([0.10, 0.50, 0.90])

    scores = calculate_risk_scores(
        test_fraud_probability,
        test_anomaly_score
    )

    levels = assign_risk_level(scores)

    test_report = pd.DataFrame({
        "Fraud Probability": test_fraud_probability,
        "Anomaly Score": test_anomaly_score,
        "Risk Score": scores,
        "Risk Level": levels
    })

    print("RISK SCORING SUCCESSFUL")
    print(test_report.to_string(index=False))
