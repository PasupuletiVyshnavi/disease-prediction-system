
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# 1. Load the Heart Disease dataset
file_path = Path("data/heart+disease/processed.cleveland.data")

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(
    file_path,
    header=None,
    names=columns,
    na_values="?"
)

df = df.apply(pd.to_numeric, errors="coerce")

# 2. Separate input features and target
df = df.dropna(subset=["target"])

X = df.drop(columns=["target"])
y = (df["target"] > 0).astype(int)

# 3. Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Define four machine learning algorithms
models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),

    "SVM": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC()
    ),

    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=100, random_state=42)
    ),

    "XGBoost": make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )
    )
}

# 5. Train and evaluate the models
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = round(accuracy * 100, 2)

    print("\nAlgorithm:", name)
    print("Test accuracy:", results[name], "%")
    print(classification_report(y_test, predictions))

# 6. Select and save the best model
best_name = max(results, key=results.get)
best_model = models[best_name]

Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "feature_names": list(X.columns),
        "target_names": ["No heart disease", "Heart disease"]
    },
    "models/heart_disease_model.joblib"
)

with open("models/heart_model_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nBest algorithm:", best_name)
print("Heart Disease model saved successfully!")
print("Accuracy results:", results)
