
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

print("Dataset loaded successfully!")
print("Total records:", len(X))
print("Total features:", len(X.columns))

print("\nFirst five records:")
print(X.head())

# Step 2: Check missing values
print("\nMissing values:", X.isnull().sum().sum())

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 4: Create the machine learning models

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC()
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
}

# Step 5: Train and evaluate models
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print("\nModel:", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=data.target_names
        )
    )

# Step 6: Select the model with highest test accuracy
best_model_name = max(results, key=results.get)
best_trained_model = models[best_model_name]

print("\nBest model by test accuracy:", best_model_name)

# Step 7: Save the trained model and feature information
joblib.dump(
    {
        "model": best_trained_model,
        "feature_names": list(X.columns),
        "target_names": list(data.target_names)
    },
    "disease_model.joblib"
)

print("Best model saved successfully!")

# Step 8: Save accuracy results for the website
accuracy_results = {
    name: round(score * 100, 2)
    for name, score in results.items()
}

with open("model_results.json", "w") as file:
    json.dump(accuracy_results, file, indent=4)

print("Model accuracy results saved successfully!")

# Step 9: Draw the model comparison graph
plt.figure(figsize=(8, 5))

plt.bar(
    results.keys(),
    [score * 100 for score in results.values()]
)

plt.title("Disease Classification Model Comparison")
plt.xlabel("Machine Learning Algorithm")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("model_comparison.png")
plt.show()

print("\nProject completed successfully!")
