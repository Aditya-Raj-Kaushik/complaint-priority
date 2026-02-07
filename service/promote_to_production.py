import mlflow
import dagshub

dagshub.init(
    repo_owner="Aditya-Raj-Kaushik",
    repo_name="complaint-priority",
    mlflow=True
)

client = mlflow.tracking.MlflowClient()

MODEL_NAME = "complaint_priority_model_binary_embeddings"

versions = client.search_model_versions(f"name='{MODEL_NAME}'")

if not versions:
    raise RuntimeError(f"No versions found for model {MODEL_NAME}")

latest_version = max(int(v.version) for v in versions)

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=str(latest_version),
    stage="Production",
    archive_existing_versions=True
)

print(
    f"Model '{MODEL_NAME}' version {latest_version} promoted to Production"
)
