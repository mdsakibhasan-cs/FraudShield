import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_anomaly_model(contamination=0.002):
    """Build an Isolation Forest anomaly detection pipeline."""
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5.")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "isolation_forest",
            IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
        )
    ])


def prepare_features(df):
    """Prepare numeric transaction features for anomaly detection."""
    return df.drop(columns=["Class"], errors="ignore")


def detect_anomalies(model, X):
    """Return anomaly labels and anomaly scores."""
    predictions = model.predict(X)

    # Isolation Forest: -1 = anomaly, 1 = normal.
    is_anomaly = predictions == -1

    scores = -model.decision_function(X)

    return is_anomaly, scores


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import create_features

    df = load_data()
    df = create_features(df)

    X = prepare_features(df)

    model = build_anomaly_model()
    model.fit(X)

    is_anomaly, scores = detect_anomalies(model, X)

    print("ANOMALY DETECTION SUCCESSFUL")
    print(f"Total transactions: {len(df)}")
    print(f"Detected anomalies: {is_anomaly.sum()}")
    print(f"Detected normal: {(~is_anomaly).sum()}")
