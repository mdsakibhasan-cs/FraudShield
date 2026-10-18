import pandas as pd


def validate_dataset(df):
    """Validate the basic structure of the fraud dataset."""
    required_columns = {"Time", "Amount", "Class"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["Class"].isna().any():
        raise ValueError("Target column 'Class' contains missing values.")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Target column 'Class' must contain only 0 and 1.")

    return True


def clean_data(df):
    """Perform safe preprocessing without removing duplicate rows."""
    data = df.copy()

    validate_dataset(data)

    numeric_columns = data.select_dtypes(include="number").columns

    data[numeric_columns] = data[numeric_columns].replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    if data[numeric_columns].isna().sum().sum() > 0:
        data[numeric_columns] = data[numeric_columns].fillna(
            data[numeric_columns].median()
        )

    return data


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    cleaned_df = clean_data(df)

    print("PREPROCESSING SUCCESSFUL")
    print(f"Original rows: {len(df)}")
    print(f"Cleaned rows: {len(cleaned_df)}")
    print(f"Missing values after cleaning: {cleaned_df.isna().sum().sum()}")
    print(f"Duplicate rows retained: {cleaned_df.duplicated().sum()}")
