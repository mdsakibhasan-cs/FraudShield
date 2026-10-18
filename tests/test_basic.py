import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import load_data
from preprocessing import validate_dataset, clean_data
from feature_engineering import create_features
from fraud_detection import build_model, prepare_data, split_data
from anomaly_detection import build_anomaly_model, detect_anomalies
from risk_scoring import calculate_risk_scores, assign_risk_level
from evaluation import evaluate_classifier


def test_dataset_loads():
    df = load_data()
    assert len(df) == 284807
    assert "Class" in df.columns
    assert "Amount" in df.columns
    assert "Time" in df.columns


def test_dataset_validation():
    df = load_data()
    assert validate_dataset(df) is True


def test_clean_data():
    df = load_data().head(1000).copy()
    cleaned = clean_data(df)

    assert len(cleaned) == len(df)
    assert cleaned.isna().sum().sum() == 0


def test_feature_engineering():
    df = load_data().head(100).copy()
    featured = create_features(df)

    for column in [
        "Hour",
        "Amount_Log",
        "Amount_Per_Time",
        "Time_Delta",
    ]:
        assert column in featured.columns


def test_classifier_builds():
    model = build_model()
    assert model is not None


def test_train_test_split():
    df = create_features(load_data().head(5000))
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(
        X, y
    )

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_anomaly_model():
    df = create_features(load_data().head(1000))
    X = df.drop(columns=["Class"])

    model = build_anomaly_model(
        contamination=0.01
    )

    model.fit(X)

    is_anomaly, scores = detect_anomalies(
        model,
        X
    )

    assert len(is_anomaly) == len(df)
    assert len(scores) == len(df)


def test_risk_scoring():
    fraud_probability = np.array([
        0.01,
        0.50,
        0.95,
    ])

    anomaly_score = np.array([
        0.10,
        0.50,
        0.90,
    ])

    scores = calculate_risk_scores(
        fraud_probability,
        anomaly_score,
    )

    levels = assign_risk_level(scores)

    assert len(scores) == 3
    assert len(levels) == 3
    assert scores.min() >= 0
    assert scores.max() <= 100


def test_evaluation_metrics():
    y_true = np.array([
        0, 0, 0, 1, 1, 1
    ])

    y_pred = np.array([
        0, 0, 1, 1, 1, 0
    ])

    y_probability = np.array([
        0.05,
        0.10,
        0.60,
        0.80,
        0.90,
        0.40,
    ])

    results = evaluate_classifier(
        y_true,
        y_pred,
        y_probability,
    )

    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1"] <= 1
    assert 0 <= results["roc_auc"] <= 1


def test_report_columns():
    expected_columns = {
        "Class",
        "Fraud Probability",
        "Anomaly Score",
        "Risk Score",
        "Risk Level",
        "Is Anomaly",
        "Predicted Fraud",
    }

    report_path = PROJECT_ROOT / "reports" / "fraud_risk_report.csv"

    assert report_path.exists()

    report = pd.read_csv(report_path)

    assert expected_columns.issubset(
        set(report.columns)
    )

    assert len(report) == 56783
    assert report.isna().sum().sum() == 0
