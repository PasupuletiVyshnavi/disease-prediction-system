
import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# 1. Load the dataset
file_path = Path("data/diabetes.csv")
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Clean column names
df.columns = df.columns.str.strip()

# 3. Separate input features and target
# The UCI dataset normally names its target column "class"
target_column = next(
    (col for col in df.columns if col.lower() == "class"),
    None
)

if target_column is None:
    raise ValueError(
        "Target column 'class' not found. Check the printed column names."
    )

X = df.drop(columns=[target_column]).copy()
y = df[target_column].astype(str).str.strip()

# Convert text features into numeric values
X = pd.get_dummies(X, drop_first=False, dtype=int)

# Convert target labels to 0 and 1
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nTarget labels:", list(encoder.classes_))
print("Number of input features:", X.shape[1])

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Define four algorithms
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

# 6. Train and evaluate each model
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = round(accuracy * 100, 2)

    print("\nAlgorithm:", name)
    print("Test accuracy:", results[name], "%")
    print(classification_report(y_test, predictions))

# 7. Save the best model and results
best_name = max(results, key=results.get)
best_model = models[best_name]

Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "feature_names": list(X.columns),
        "target_names": [str(c) for c in encoder.classes_],
        "target_column": target_column
    },
    "models/diabetes_model.joblib"
)

with open("models/diabetes_model_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nBest algorithm:", best_name)
print("Diabetes model saved successfully!")
print("Accuracy results:", results)
