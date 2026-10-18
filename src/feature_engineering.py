import numpy as np
import pandas as pd


def create_features(df):
    """Create additional transaction-level features."""
    data = df.copy()

    # Time converted to hour-of-day representation.
    data["Hour"] = (data["Time"] / 3600) % 24

    # Log transform reduces the effect of extremely large amounts.
    data["Amount_Log"] = np.log1p(data["Amount"])

    # Avoid division by zero.
    data["Amount_Per_Time"] = data["Amount"] / (data["Time"] + 1)

    # Time gap between consecutive records.
    data["Time_Delta"] = data["Time"].diff().fillna(0)

    return data


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    featured_df = create_features(df)

    new_columns = [
        "Hour",
        "Amount_Log",
        "Amount_Per_Time",
        "Time_Delta"
    ]

    print("FEATURE ENGINEERING SUCCESSFUL")
    print(f"Original columns: {len(df.columns)}")
    print(f"New columns: {len(featured_df.columns)}")
    print("Added features:")
    print(featured_df[new_columns].head().to_string(index=False))
