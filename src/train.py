# -------------------------------
# Sanity check (must print)
# -------------------------------
print("TRAINING SCRIPT STARTED")

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data_loader import load_complaints


# -------------------------------
# MLflow setup (DagsHub backend)
# -------------------------------
mlflow.set_tracking_uri(
    "https://Aditya-Raj-Kaushik:4f52a7331637b455d4de1e22c042b32bbd70f59a@dagshub.com/Aditya-Raj-Kaushik/complaint-priority.mlflow"
)


import dagshub

dagshub.init(
    repo_owner="Aditya-Raj-Kaushik",
    repo_name="complaint-priority",
    mlflow=True
)

mlflow.set_experiment("complaint_priority_training")



def main():
    print("Loading dataset...")

    df = load_complaints("data/raw/complaints.csv")

    print("Dataset shape:", df.shape)

    X = df.drop(columns=["priority"])
    y = df["priority"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    # -------------------------------
    # Feature + model pipeline
    # -------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=3000,
                    ngram_range=(1, 2),
                ),
                "complaint_text",
            ),
            ("num", "passthrough", ["customer_tenure"]),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    # -------------------------------
    # MLflow experiment
    # -------------------------------
    with mlflow.start_run():
        print("Training model...")

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # Log parameters
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("max_features", 3000)
        mlflow.log_param("ngram_range", "1-2")

        # Log & register model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="complaint_priority_model",
        )

        print("Model logged to MLflow successfully")


if __name__ == "__main__":
    main()

