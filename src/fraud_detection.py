import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def prepare_data(df):
    """Separate features and target."""
    data = df.copy()

    X = data.drop(columns=["Class"])
    y = data["Class"]

    return X, y


def build_model():
    """Build the fraud classification pipeline."""
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ])

    return model


def split_data(X, y, test_size=0.2):
    """Create a stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import create_features

    df = load_data()
    df = create_features(df)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model()

    print("FRAUD DETECTION MODEL READY")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training fraud cases: {(y_train == 1).sum()}")
    print(f"Testing fraud cases: {(y_test == 1).sum()}")
