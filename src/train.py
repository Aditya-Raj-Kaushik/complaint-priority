print("EMBEDDING-BASED BINARY URGENCY TRAINING STARTED")

import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# --------------------------------------------------
# Initialize DagsHub + MLflow
# --------------------------------------------------
dagshub.init(
    repo_owner="Aditya-Raj-Kaushik",
    repo_name="complaint-priority",
    mlflow=True
)

mlflow.set_experiment("complaint_priority_v6_embeddings_binary")


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
def load_data():
    return pd.read_csv("data/raw/complaints.csv")


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------
def main():
    df = load_data()

    # -------------------------------
    # Binary urgency reframing
    # -------------------------------
    df["urgency"] = df["priority"].apply(
        lambda x: "urgent" if x in ["medium", "high"] else "non_urgent"
    )

    texts = df["complaint_text"].tolist()
    labels = df["urgency"].tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -------------------------------
    # Sentence embeddings
    # -------------------------------
    embedding_model = SentenceTransformer("all-mpnet-base-v2")


    X_train = embedding_model.encode(
        X_train_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    X_test = embedding_model.encode(
        X_test_texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # -------------------------------
    # Classifier
    # -------------------------------
    classifier = LogisticRegression(
    max_iter=3000,
    C=2.5,
    class_weight="balanced",
    solver="liblinear"
)


    with mlflow.start_run():
        classifier.fit(X_train, y_train)

        preds = classifier.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        # -------------------------------
        # MLflow logging
        # -------------------------------
        mlflow.log_param("task_type", "binary_urgency_classification")
        mlflow.log_param("positive_class", "urgent")
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("embedding_dim", X_train.shape[1])
        mlflow.log_param("classifier", "logistic_regression")
        mlflow.log_param("C", 1.5)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        mlflow.sklearn.log_model(
            classifier,
            artifact_path="model",
            registered_model_name="complaint_priority_model_binary_embeddings"
        )

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Weighted: {f1:.4f}")


if __name__ == "__main__":
    main()
