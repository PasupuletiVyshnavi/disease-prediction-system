
import pandas as pd
from pathlib import Path

# Locate the downloaded Heart Disease dataset
file_path = Path("data/heart+disease/processed.cleveland.data")

# Check whether the dataset file exists
if not file_path.exists():
    print("Dataset file not found!")
    print("Please check the filename and folder.")
else:
    # Column names for the Cleveland Heart Disease dataset
    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]

    # Load the dataset and identify missing values
    df = pd.read_csv(
        file_path,
        header=None,
        names=columns,
        na_values="?"
    )

    # Convert all columns to numeric values
    df = df.apply(pd.to_numeric, errors="coerce")

    print("Heart Disease Dataset Loaded Successfully!")
    print("Dataset shape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTarget values:")
    print(df["target"].value_counts(dropna=False))
