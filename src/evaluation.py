from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def evaluate_classifier(y_true, y_pred, y_probability):
    """Calculate standard fraud-classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Normal", "Fraud"],
            digits=4,
            zero_division=0
        )
    }


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import create_features
    from fraud_detection import prepare_data, split_data, build_model

    df = load_data()
    df = create_features(df)

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    results = evaluate_classifier(
        y_test,
        y_pred,
        y_probability
    )

    print("EVALUATION MODULE SUCCESSFUL")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1       : {results['f1']:.4f}")
    print(f"ROC-AUC  : {results['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print("\nClassification Report:")
    print(results["classification_report"])
