# src/ingest.py

import pandas as pd
import os

def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Loads review dataset with columns:
    author, posted_on, rating, text
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    expected_cols = {"author", "posted_on", "rating", "text"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"CSV missing required columns. Found: {df.columns}, "
            f"Expected at least: {expected_cols}"
        )

    # Basic cleaning
    df["posted_on"] = pd.to_datetime(df["posted_on"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Drop rows with missing text/rating
    df = df.dropna(subset=["text", "rating"])

    return df


# Wrapper function for notebook compatibility
def ingest(csv_path: str) -> pd.DataFrame:
    """Wrapper for load_raw_data to match notebook imports"""
    return load_raw_data(csv_path)


def main():
    input_csv = "Data/comcast_consumeraffairs_complaints.csv"
    output_parquet = "Data/processed/reviews_raw.parquet"

    os.makedirs("Data/processed", exist_ok=True)

    df = load_raw_data(input_csv)
    df.to_parquet(output_parquet, index=False)

    print(f"✅ Saved ingested data to {output_parquet}")
    print(df.head())


if __name__ == "__main__":
    main()


