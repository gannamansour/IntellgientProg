"""
utils/data_processing.py
Handles loading, cleaning, and preprocessing the heart disease dataset.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_clean(raw_path: str, output_path: str) -> pd.DataFrame:
    """Load raw CSV, clean it, normalise numerical columns, and save."""
    df = pd.read_csv(raw_path)

    # 1. Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # 2. Fill missing numeric values with column mean
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # 3. Normalise continuous features (keep target as-is)
    numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # 4. Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"[✓] Cleaned data saved → {output_path}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    df = load_and_clean(
        raw_path="data/raw_data.csv",
        output_path="data/cleaned_data.csv",
    )
    print(df.head())
