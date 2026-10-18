import pandas as pd


def class_distribution(df):
    """Return transaction counts and percentages by fraud class."""
    counts = df["Class"].value_counts().sort_index()
    percentages = df["Class"].value_counts(normalize=True).sort_index() * 100

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages
    })

    return result


def amount_statistics(df):
    """Return transaction amount statistics by class."""
    return (
        df.groupby("Class")["Amount"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )


def time_statistics(df):
    """Return transaction time statistics by class."""
    return (
        df.groupby("Class")["Time"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()

    print("EDA ANALYSIS SUCCESSFUL")

    print("\nCLASS DISTRIBUTION")
    print(class_distribution(df).to_string())

    print("\nAMOUNT STATISTICS")
    print(amount_statistics(df).to_string(index=False))

    print("\nTIME STATISTICS")
    print(time_statistics(df).to_string(index=False))
