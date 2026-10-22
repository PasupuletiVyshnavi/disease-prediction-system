
from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Load Breast Cancer model and dataset
breast_saved = joblib.load(BASE_DIR / "disease_model.joblib")
breast_model = breast_saved["model"]
breast_names = list(breast_saved["target_names"])
breast_data = load_breast_cancer(as_frame=True).data

# Load Heart Disease model and dataset
heart_saved = joblib.load(
    BASE_DIR / "models" / "heart_disease_model.joblib"
)
heart_model = heart_saved["model"]
heart_names = list(heart_saved["target_names"])

heart_columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

heart_df = pd.read_csv(
    BASE_DIR / "data" / "heart+disease" / "processed.cleveland.data",
    header=None,
    names=heart_columns,
    na_values="?"
)
heart_df = heart_df.apply(pd.to_numeric, errors="coerce")
heart_df = heart_df.dropna(subset=["target"]).reset_index(drop=True)
heart_data = heart_df.drop(columns=["target"])

# Load Diabetes model and dataset
diabetes_saved = joblib.load(
    BASE_DIR / "models" / "diabetes_model.joblib"
)
diabetes_model = diabetes_saved["model"]
diabetes_names = list(diabetes_saved["target_names"])

diabetes_df = pd.read_csv(BASE_DIR / "data" / "diabetes.csv")
diabetes_df.columns = diabetes_df.columns.str.strip()

target_column = next(
    (col for col in diabetes_df.columns
     if col.lower() == "class"),
    None
)

if target_column is None:
    raise ValueError("The diabetes dataset has no 'class' column.")

diabetes_features = diabetes_df.drop(columns=[target_column])
diabetes_features = pd.get_dummies(
    diabetes_features, drop_first=False, dtype=int
)

# Match the exact feature columns used when training the model
diabetes_features = diabetes_features.reindex(
    columns=diabetes_saved["feature_names"],
    fill_value=0
)

# Available sample counts
sample_counts = {
    "breast": len(breast_data),
    "heart": len(heart_data),
    "diabetes": len(diabetes_features)
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    selected_disease = "breast"
    sample_number = 1
    algorithm = None

    if request.method == "POST":
        selected_disease = request.form.get("disease", "breast")

        try:
            sample_number = int(request.form.get("sample_number", "1"))

            if selected_disease not in sample_counts:
                raise ValueError("Please select a valid disease.")

            if not 1 <= sample_number <= sample_counts[selected_disease]:
                raise ValueError(
                    f"Enter a sample number from 1 to "
                    f"{sample_counts[selected_disease]}."
                )

            row_index = sample_number - 1

            if selected_disease == "breast":
                sample = breast_data.iloc[[row_index]]
                result = int(breast_model.predict(sample)[0])
                prediction = breast_names[result]
                algorithm = breast_saved.get(
                    "best_model_name", "Trained Breast Cancer Model"
                )

            elif selected_disease == "heart":
                sample = heart_data.iloc[[row_index]]
                result = int(heart_model.predict(sample)[0])
                prediction = heart_names[result]
                algorithm = heart_saved.get(
                    "best_model_name", "Trained Heart Disease Model"
                )

            else:
                sample = diabetes_features.iloc[[row_index]]
                result = int(diabetes_model.predict(sample)[0])
                prediction = diabetes_names[result]
                algorithm = diabetes_saved.get(
                    "best_model_name", "Trained Diabetes Model"
                )

        except (ValueError, TypeError) as exc:
            error = str(exc) or "Please enter a valid sample number."
        except Exception:
            error = (
                "Prediction could not be completed. "
                "Please check your model and dataset files."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        algorithm=algorithm,
        selected_disease=selected_disease,
        sample_number=sample_number,
        sample_counts=sample_counts
    )

if __name__ == "__main__":
    app.run(debug=True)
