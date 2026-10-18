import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from data_loader import load_data
from feature_engineering import create_features
from fraud_detection import build_model
from anomaly_detection import build_anomaly_model, detect_anomalies
from risk_scoring import create_risk_report
from evaluation import evaluate_classifier


def run_pipeline():
    print("LOADING DATA...")
    df = load_data()

    print("CREATING FEATURES...")
    df = create_features(df)

    # -------------------------------------------------
    # Remove exact duplicate feature rows for evaluation
    # -------------------------------------------------
    feature_columns = df.drop(columns=["Class"]).columns
    feature_keys = (
        df[feature_columns]
        .astype(str)
        .agg("|".join, axis=1)
    )

    unique_indices = feature_keys.drop_duplicates().index
    unique_df = df.loc[unique_indices].copy()

    print(f"Original rows       : {len(df)}")
    print(f"Unique feature rows : {len(unique_df)}")
    print(f"Removed duplicates  : {len(df) - len(unique_df)}")

    # -------------------------------------------------
    # Prepare data
    # -------------------------------------------------
    X = unique_df.drop(columns=["Class"])
    y = unique_df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training rows       : {len(X_train)}")
    print(f"Testing rows        : {len(X_test)}")

    # -------------------------------------------------
    # Fraud classification
    # -------------------------------------------------
    print("\nTRAINING FRAUD CLASSIFIER...")

    classifier = build_model()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_probability = classifier.predict_proba(X_test)[:, 1]

    evaluation = evaluate_classifier(
        y_test,
        y_pred,
        y_probability
    )

    # -------------------------------------------------
    # Anomaly detection
    # -------------------------------------------------
    print("TRAINING ANOMALY DETECTOR...")

    anomaly_model = build_anomaly_model()

    anomaly_model.fit(X_train)

    test_is_anomaly, test_anomaly_score = detect_anomalies(
        anomaly_model,
        X_test
    )

    # -------------------------------------------------
    # Test-set risk report
    # -------------------------------------------------
    test_report = unique_df.loc[X_test.index].copy()

    test_report = create_risk_report(
        test_report,
        y_probability,
        test_anomaly_score
    )

    test_report["Is Anomaly"] = test_is_anomaly

    test_report["Predicted Fraud"] = (
        y_pred
    )

    # -------------------------------------------------
    # Save models
    # -------------------------------------------------
    joblib.dump(
        classifier,
        "models/fraud_random_forest.joblib"
    )

    joblib.dump(
        anomaly_model,
        "models/isolation_forest.joblib"
    )

    # -------------------------------------------------
    # Save leakage-aware test report
    # -------------------------------------------------
    output_path = "reports/fraud_risk_report.csv"

    test_report.to_csv(
        output_path,
        index=False
    )

    # -------------------------------------------------
    # Final output
    # -------------------------------------------------
    print("\n" + "=" * 60)
    print("LEAKAGE-AWARE END-TO-END PIPELINE SUCCESSFUL")
    print("=" * 60)

    print(f"Test transactions : {len(test_report)}")
    print(f"Actual fraud      : {(y_test == 1).sum()}")
    print(f"Predicted fraud   : {int(y_pred.sum())}")
    print(f"Anomalies         : {int(test_is_anomaly.sum())}")

    print("\nRisk distribution:")
    print(
        test_report["Risk Level"]
        .value_counts()
        .to_string()
    )

    print("\nLeakage-free evaluation:")
    print(f"Accuracy : {evaluation['accuracy']:.4f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall   : {evaluation['recall']:.4f}")
    print(f"F1       : {evaluation['f1']:.4f}")
    print(f"ROC-AUC  : {evaluation['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(evaluation["confusion_matrix"])

    print(f"\nReport saved: {output_path}")

    print("Models saved:")
    print(" - models/fraud_random_forest.joblib")
    print(" - models/isolation_forest.joblib")


if __name__ == "__main__":
    run_pipeline()
