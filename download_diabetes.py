
from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

print("Downloading Diabetes dataset...")

# Fetch the UCI Early Stage Diabetes Risk Prediction dataset
diabetes = fetch_ucirepo(id=529)

# Get input features and target labels
X = diabetes.data.features
y = diabetes.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Create data folder if needed
Path("data").mkdir(exist_ok=True)

# Save dataset as CSV
df.to_csv("data/diabetes.csv", index=False)

print("\nDiabetes dataset downloaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())
