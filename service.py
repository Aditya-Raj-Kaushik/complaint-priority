import bentoml
from bentoml.io import JSON
from sentence_transformers import SentenceTransformer
import numpy as np

# Load Production MLflow model
model_ref = bentoml.mlflow.import_model(
    name="complaint_priority_model_binary_embeddings",
    model_uri="models:/complaint_priority_model_binary_embeddings/Production",
)

model = model_ref.load_model()

# ⚠️ Lightweight embedding model for inference
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

svc = bentoml.Service(
    name="complaint_priority_service",
    runners=[],
)

@svc.api(input=JSON(), output=JSON())
def predict(input_json):
    text = input_json["text"]

    embedding = embedding_model.encode([text])
    prediction = model.predict(embedding)[0]

    label = "urgent" if prediction == 1 else "non_urgent"

    return {
        "prediction": label
    }
