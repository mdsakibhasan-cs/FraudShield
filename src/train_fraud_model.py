from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from data_loader import load_data
from feature_engineering import create_features
from fraud_detection import prepare_data, split_data, build_model


df = load_data()
df = create_features(df)

X, y = prepare_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)

model = build_model()

print("TRAINING RANDOM FOREST...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nMODEL EVALUATION")
print("=" * 50)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Fraud"],
    digits=4
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
