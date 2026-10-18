import joblib

from data_loader import load_data
from feature_engineering import create_features
from fraud_detection import prepare_data, split_data, build_model


df = load_data()
df = create_features(df)

X, y = prepare_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)

model = build_model()
model.fit(X_train, y_train)

output_path = "models/fraud_random_forest.joblib"
joblib.dump(model, output_path)

print("MODEL SAVED SUCCESSFULLY")
print(f"Path: {output_path}")
