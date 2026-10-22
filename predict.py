
import joblib
import pandas as pd

# Load the saved model
saved_data = joblib.load("disease_model.joblib")

model = saved_data["model"]
feature_names = saved_data["feature_names"]
target_names = saved_data["target_names"]

# Get one sample from the original dataset
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)

sample = data.data.iloc[[0]]

# Predict using the trained model
prediction = model.predict(sample)[0]

print("Prediction completed successfully!")

print("Predicted result:", target_names[prediction])

print("\nSample input values:")
print(sample)

print("\nNote: This is an educational demonstration, not a medical diagnosis.")
