print("MULTI-EXPERIMENT TRAINING STARTED")

import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .data_loader import load_complaints



# --------------------------------------------------
# DagsHub + MLflow initialization (AUTH HANDLED)
# --------------------------------------------------
dagshub.init(
    repo_owner="Aditya-Raj-Kaushik",
    repo_name="complaint-priority",
    mlflow=True
)

mlflow.set_experiment("complaint_priority_v3_complex_data")


# --------------------------------------------------
# Single experiment runner
# --------------------------------------------------
def run_experiment(max_features: int, C: float):
    df = load_complaints("data/raw/complaints.csv")

    X = df.drop(columns=["priority"])
    y = df["priority"]

    y_encoded = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,              
                ),
                "complaint_text",
            ),
            ("num", "passthrough", ["customer_tenure"]),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        C=C,
        n_jobs=1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        # --------------------
        # Log to MLflow
        # --------------------
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("tfidf_max_features", max_features)
        mlflow.log_param("tfidf_ngram_range", "1-2")
        mlflow.log_param("C", C)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model"
        )

        print(
            f"[RUN] max_features={max_features}, C={C} "
            f"-> accuracy={acc:.4f}, f1={f1:.4f}"
        )


# --------------------------------------------------
# Experiment grid (SMALL DATASET)
# --------------------------------------------------
def main():
    max_features_list = [500, 1000, 2000]
    C_list = [0.3, 1.0, 3.0]

    for max_features in max_features_list:
        for C in C_list:
            run_experiment(max_features, C)


if __name__ == "__main__":
    main()
