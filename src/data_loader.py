from pathlib import Path
import pandas as pd


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "creditcard_dataset"
    / "creditcard.csv"
)


def load_data(file_path=DEFAULT_DATA_PATH):
    """Load the credit-card fraud dataset."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    required_columns = {"Time", "Amount", "Class"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df


if __name__ == "__main__":
    df = load_data()

    print("DATA LOADED SUCCESSFULLY")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Fraud transactions: {(df['Class'] == 1).sum()}")
    print(f"Normal transactions: {(df['Class'] == 0).sum()}")
